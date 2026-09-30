"""Background removal for a timeline clip: a matte sequence in the matte-cache layout.

The timeline applies the matte as the clip's alpha (the compositor's `sequence` mask entry), so the
monitor and the export cut the clip out the same way. Three ways to make it, each in its own step:

* ``sam2``: SAM 2 follows the subject picked with include/exclude points. Points may be given on
  SEVERAL frames (the Clip Editor's pick, plus "track assist": CoTracker3 follows the picked points
  and they are given again every few frames), and propagation runs forward from the first prompted
  frame and backward to the start, so a pick in the middle of a clip covers the whole clip.
  Masks are hard-edged (0/255).
* ``birefnet``: BiRefNet per frame, a soft alpha (hair, motion blur) with no notion of which subject.
  With ``--guide-dir`` (a SAM 2 matte from the step above) the two are combined as the character
  matte does (`character_matte.combine`): BiRefNet's alpha inside a dilated SAM 2 region, and at
  least the eroded SAM 2 interior, so BiRefNet can neither add a second subject nor punch a hole.

SAM 2 and BiRefNet may live in different environments, so each step is its own process. Output:
``<cache>/masks/00000.png ...`` (L, the source's size) and ``manifest.json``; the last stdout line is
the manifest as JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from aiv_runtime import sam2_track  # noqa: E402  (stdlib-only at import time)

MAX_SAM2_FRAMES = sam2_track.MAX_TRACK_FRAMES  # 900: SAM 2 holds per-frame state in memory
MAX_BIREFNET_FRAMES = 5400  # 3:45 at 24 fps; BiRefNet keeps nothing between frames


def parse_prompts(raw: list[str]) -> list[tuple[int, float, float, int]]:
    """``frame,x,y,label``: frame index in the range, x/y normalised to the source, label 1/0."""
    prompts = []
    for item in raw:
        parts = item.split(",")
        if len(parts) != 4:
            raise ValueError(f"a prompt is frame,x,y,label: {item!r}")
        frame, x, y, label = int(parts[0]), float(parts[1]), float(parts[2]), int(parts[3])
        if frame < 0 or not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) or label not in (0, 1):
            raise ValueError(f"a prompt is outside the clip: {item!r}")
        prompts.append((frame, x, y, label))
    return prompts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matte a timeline clip into a matte cache")
    parser.add_argument("--mode", choices=("sam2", "birefnet"), required=True)
    parser.add_argument("--video-path", required=True, help="a video or a still image")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float, default=0.0, help="0 for a still")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-id", required=True)
    parser.add_argument("--model-root", default="", help="SAM 2 pack or models/BiRefNet")
    parser.add_argument("--prompt", action="append", default=[], help="frame,x,y,label (sam2)")
    parser.add_argument("--guide-dir", default="", help="a SAM 2 matte cache to refine (birefnet)")
    parser.add_argument("--smooth", type=float, default=0.0, help="0..0.6 causal smoothing (birefnet)")
    parser.add_argument("--device", default="cuda")
    return parser


def probe_source(path: str) -> dict:
    source = sam2_track.probe(path)
    if not source["width"] or not source["height"]:
        raise ValueError(f"could not read the picture size of {path}")
    return source


def extract(path: str, start: float, end: float, fps: float, limit: int, out: Path, still: bool) -> int:
    """Frames of the range as numbered JPEGs (a still gives one frame)."""
    out.mkdir(parents=True, exist_ok=True)
    count = 1 if still else max(1, min(limit, int(round((end - start) * fps))))
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]
    if not still:
        command += ["-ss", f"{start:.6f}"]
    command += ["-i", path, "-frames:v", str(count), "-q:v", "2", "-start_number", "0", str(out / "%05d.jpg")]
    subprocess.run(command, check=True)
    produced = len(list(out.glob("*.jpg")))
    if not produced:
        raise RuntimeError("ffmpeg produced no frames for the clip")
    return produced


def track_prompts(model_root: str, frames_dir: Path, source: dict, frame_count: int,
                  prompts: list[tuple[int, float, float, int]], device: str) -> dict:
    """SAM 2 with prompts on any frames: forward from the first prompt, then backward to frame 0."""
    import numpy as np
    import torch

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as error:
        raise RuntimeError("SAM 2 is not installed. Install it with: " + sam2_track.INSTALL_HINT) from error
    checkpoint, config_name = sam2_track.find_checkpoint(Path(model_root))
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    by_frame: dict[int, list[tuple[float, float, int]]] = {}
    for frame, x, y, label in prompts:
        by_frame.setdefault(min(frame, frame_count - 1), []).append((x, y, label))
    print(f"[clip-matte] SAM 2 {checkpoint.name} on {device} · {frame_count} frames · "
          f"prompts on {len(by_frame)} frame(s)", file=sys.stderr, flush=True)
    predictor = build_sam2_video_predictor(config_name, str(checkpoint), device=device)
    state = predictor.init_state(video_path=str(frames_dir))
    for frame, points in sorted(by_frame.items()):
        coords = np.array([[x * source["width"], y * source["height"]] for x, y, _ in points], dtype=np.float32)
        labels = np.array([label for _, _, label in points], dtype=np.int32)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=frame, obj_id=1, points=coords, labels=labels)
    masks: dict[int, "np.ndarray"] = {}
    for frame_idx, _ids, logits in predictor.propagate_in_video(state):
        masks[frame_idx] = (logits[0] > 0.0).cpu().numpy().squeeze()
    first = min(by_frame)
    if first > 0:
        for frame_idx, _ids, logits in predictor.propagate_in_video(state, start_frame_idx=first, reverse=True):
            if frame_idx not in masks:
                masks[frame_idx] = (logits[0] > 0.0).cpu().numpy().squeeze()
    if not masks:
        raise RuntimeError("SAM 2 produced no masks; try other points")
    del predictor, state
    if device == "cuda":
        torch.cuda.empty_cache()
    return masks


def write_manifest(cache: Path, cache_id: str, frame_count: int, source: dict, model: str, method: str, extra: dict) -> dict:
    manifest_path = cache / "manifest.json"
    manifest = {
        "cacheId": cache_id, "cachePath": str(cache), "manifestPath": str(manifest_path),
        "frameCount": frame_count, "width": source["width"], "height": source["height"], "fps": source["fps"],
        "model": model, "method": method, "createdAt": datetime.now(timezone.utc).isoformat(), "reused": False,
        **extra,
    }
    pending = manifest_path.with_suffix(".json.pending")
    pending.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pending.replace(manifest_path)
    return manifest


def run_sam2(args, source: dict, still: bool) -> dict:
    import numpy as np
    from PIL import Image

    prompts = parse_prompts(args.prompt)
    if not any(label == 1 for *_, label in prompts):
        raise ValueError("SAM 2 needs at least one include point")
    cache = Path(args.cache_dir)
    (cache / "masks").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aiv-clip-matte-") as scratch:
        frames = Path(scratch) / "frames"
        count = extract(args.video_path, args.start_time, args.end_time, source["fps"], MAX_SAM2_FRAMES, frames, still)
        masks = track_prompts(args.model_root, frames, source, count, prompts, args.device)
    last = np.zeros((source["height"], source["width"]), dtype=np.uint8)
    for index in range(count):
        current = masks.get(index)
        if current is not None:
            last = current.astype(np.uint8) * 255
        Image.fromarray(last, mode="L").resize((source["width"], source["height"]), Image.Resampling.NEAREST) \
            .save(cache / "masks" / f"{index:05d}.png", compress_level=2)
    frames_prompted = len({frame for frame, *_ in prompts})
    return write_manifest(cache, args.cache_id, count, source, "sam2", "sam2", {"promptFrames": frames_prompted})


def run_birefnet(args, source: dict, still: bool) -> dict:
    import numpy as np
    from PIL import Image

    from aiv_runtime.character_matte import combine, stabilise

    root = Path(args.model_root)
    if not (root / "config.json").is_file():
        raise FileNotFoundError(f"BiRefNet is not installed at {root}")
    guide = sorted((Path(args.guide_dir) / "masks").glob("*.png")) if args.guide_dir else []
    from aiv_runtime.image_studio.birefnet import BiRefNetPipeline

    pipeline = BiRefNetPipeline.create(str(root), args.device)
    cache = Path(args.cache_dir)
    (cache / "masks").mkdir(parents=True, exist_ok=True)
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="aiv-clip-matte-") as scratch:
        frames_dir = Path(scratch) / "frames"
        limit = len(guide) if guide else MAX_BIREFNET_FRAMES
        count = extract(args.video_path, args.start_time, args.end_time, source["fps"], limit, frames_dir, still)
        previous = None
        for index, path in enumerate(sorted(frames_dir.glob("*.jpg"))):
            with Image.open(path) as opened:
                colour = opened.convert("RGB")
            if colour.size != (source["width"], source["height"]):
                colour = colour.resize((source["width"], source["height"]), Image.Resampling.BICUBIC)
            soft = np.asarray(pipeline.remove_background(colour).mask.convert("L"))
            hard = None
            if guide:
                with Image.open(guide[min(index, len(guide) - 1)]) as mask:
                    hard = np.asarray(mask.convert("L").resize(colour.size, Image.Resampling.NEAREST))
            alpha = stabilise(previous, combine(hard, soft, colour.height), max(0.0, min(0.6, args.smooth)))
            previous = alpha
            Image.fromarray(alpha, mode="L").save(cache / "masks" / f"{index:05d}.png", compress_level=2)
            if index % 24 == 0:
                print(f"[clip-matte] BiRefNet {index + 1}/{count}", file=sys.stderr, flush=True)
    method = "sam2-birefnet" if guide else "birefnet"
    return write_manifest(cache, args.cache_id, count, source, "sam2" if guide else "birefnet", method,
                          {"seconds": round(time.time() - started, 1)})


def main(argv=None) -> int:
    sam2_track.force_utf8_io()
    args = build_parser().parse_args(argv)
    if not Path(args.video_path).is_file():
        raise FileNotFoundError(args.video_path)
    manifest = Path(args.cache_dir) / "manifest.json"
    if manifest.is_file():
        result = json.loads(manifest.read_text(encoding="utf-8"))
        result["reused"] = True
        print(json.dumps(result))
        return 0
    still = args.end_time <= args.start_time
    source = probe_source(args.video_path)
    result = run_sam2(args, source, still) if args.mode == "sam2" else run_birefnet(args, source, still)
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
