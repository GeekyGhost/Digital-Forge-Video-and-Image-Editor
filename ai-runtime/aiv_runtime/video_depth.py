"""Per-frame depth for a clip with Video Depth Anything Small (VFX Phase 4).

Output is a depth SEQUENCE in the mask-cache layout the compositor already reads for SAM 2 mattes:

    <cache>/masks/00000.png ...   8-bit greyscale, one per source frame of the clip, from frame 0
    <cache>/manifest.json         kind "depth", frames, size, fps, and the normalisation range

Values are relative disparity (Video Depth Anything's output: larger is NEARER), normalised once per
clip -- not per frame, or the depth of a still background would pump -- from its 0.5th to 99.5th
percentile to 0..255. So 255 is near and 0 is far, which is what a ``depth`` mask entry expects.
The files are written at the size the model ran at (at most ``--max-res`` on the long edge); the
app stretches them to the clip's size before placing them.

Video Depth Anything Small is Apache-2.0 (the Base and Large checkpoints are not; this worker only
loads Small). Its requirements.txt pins torch 2.1.1 / xformers 0.0.23; neither is needed here:
xformers is optional upstream and the motion runtime uses the project's torch 2.7.1 build.

Run:  python -m aiv_runtime.video_depth --video V --cache-dir D --checkpoint C
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from aiv_runtime import motion_track  # noqa: E402

LICENCE = "Video Depth Anything Small: Apache-2.0 (ByteDance)"
DEFAULT_REPO = Path(__file__).resolve().parents[2] / "vendor" / "Video-Depth-Anything"
MAX_FRAMES = 1500
VITS = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}


class DepthModel(Protocol):
    def infer_video_depth(self, frames: np.ndarray, target_fps: float, input_size: int = 518,
                          device: str = "cuda", fp32: bool = False) -> tuple[np.ndarray, float]: ...


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Video Depth Anything Small for a clip")
    parser.add_argument("--video", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--repo", default=str(DEFAULT_REPO))
    parser.add_argument("--max-res", type=int, default=1280, help="long edge the model runs at")
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--max-frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def working_size(width: int, height: int, max_res: int) -> tuple[int, int]:
    """The clip's size with the long edge at most ``max_res``, even dimensions."""
    scale = min(1.0, max_res / max(width, height))
    w = max(2, int(round(width * scale / 2)) * 2)
    h = max(2, int(round(height * scale / 2)) * 2)
    return w, h


def normalise(depths: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Disparity to bytes over the whole clip: percentiles 0.5 / 99.5 -> 0 / 255, 255 near."""
    finite = depths[np.isfinite(depths)]
    if not finite.size:
        return np.zeros(depths.shape, dtype=np.uint8), 0.0, 0.0
    low, high = (float(value) for value in np.percentile(finite, [0.5, 99.5]))
    span = high - low if high > low else 1.0
    scaled = np.clip((np.nan_to_num(depths, nan=low) - low) / span, 0.0, 1.0)
    return np.round(scaled * 255.0).astype(np.uint8), low, high


def write_sequence(cache: Path, depths: np.ndarray) -> None:
    masks = cache / "masks"
    masks.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(depths):
        Image.fromarray(frame, mode="L").save(masks / f"{index:05d}.png", compress_level=2)


def load_model(checkpoint: Path, repo: Path, device: str) -> DepthModel:
    import torch  # noqa: PLC0415

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))  # upstream imports `utils.util` from its root
    from video_depth_anything.video_depth import VideoDepthAnything  # noqa: PLC0415

    model = VideoDepthAnything(**VITS)
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"), strict=True)
    return model.to(device).eval()


def main(argv: Sequence[str] | None = None, model: DepthModel | None = None) -> int:
    motion_track.force_utf8_io()
    args = build_parser().parse_args(argv)
    started = time.perf_counter()
    cache = Path(args.cache_dir)
    manifest_path = cache / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"reused": True, "cachePath": str(cache)})
        print(json.dumps(manifest))
        return 0
    source = motion_track.probe(args.video)
    width, height = source["width"], source["height"]
    if width <= 0 or height <= 0:
        raise RuntimeError("could not read the clip's size")
    count = min(args.max_frames, source["frames"] or args.max_frames)
    w, h = working_size(width, height, args.max_res)
    frames = motion_track.read_frames(args.video, w, h, 0, count)
    device = "cpu"
    if model is None:
        import torch  # noqa: PLC0415

        device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
        model = load_model(Path(args.checkpoint), Path(args.repo), device)
    depths, _ = model.infer_video_depth(frames, source["fps"], input_size=args.input_size, device=device, fp32=args.fp32 or device == "cpu")
    depths = np.asarray(depths)[: frames.shape[0]]
    if depths.shape[1:] != (h, w):
        raise RuntimeError(f"the model returned {depths.shape[1:]} frames, expected {(h, w)}")
    encoded, low, high = normalise(depths.astype(np.float32))
    cache.mkdir(parents=True, exist_ok=True)
    write_sequence(cache, encoded)
    manifest = {
        "ok": True, "kind": "depth", "source": "video-depth-anything-small", "licence": LICENCE,
        "video": args.video, "frames": int(encoded.shape[0]), "width": w, "height": h,
        "sourceWidth": width, "sourceHeight": height, "fps": source["fps"],
        "disparityLow": low, "disparityHigh": high, "encoding": "8-bit disparity, 255 near, per-clip percentiles 0.5/99.5",
        "device": device, "seconds": round(time.perf_counter() - started, 2), "cachePath": str(cache),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
