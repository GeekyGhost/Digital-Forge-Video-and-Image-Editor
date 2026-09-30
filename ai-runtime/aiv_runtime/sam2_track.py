"""SAM 2 click-to-track masking for a range of a timeline clip.

You click a subject once in the program monitor; SAM 2 propagates that selection through every
frame of the range and this worker writes out the result. Three products, because they serve
different editorial jobs:

* `matte`  - a black-and-white MP4 of the mask itself, for use as a luma key or track matte.
* `cutout` - the subject on black, for compositing over another layer.
* `removed`- the subject blacked out, i.e. the inverse, for clean-plate style work.

SAM 2 large is only 857 MB, so it fits alongside anything else and needs no offload machinery.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

# The distribution is named "sam-2" even though the import is "sam2"; requesting "sam2" both fails
# pip's name check and collides with an unrelated PyPI project of that name.
INSTALL_HINT = '.venv\\Scripts\\python.exe -m pip install -e ".\\ai-runtime[masking]"'

MAX_TRACK_FRAMES = 900  # ~37 s at 24 fps; the video predictor holds per-frame state in memory.


def force_utf8_io() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SAM 2 subject tracking for a clip range.")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    # Normalised 0..1 so the UI can send monitor-relative coordinates without knowing the
    # source resolution. Repeatable: several positives refine, negatives carve away.
    parser.add_argument("--point", action="append", default=[],
                        help="x,y,label  where label is 1 for include and 0 for exclude")
    parser.add_argument("--output-kind", choices=("matte", "cutout", "removed"), default="matte")
    parser.add_argument("--feather", type=int, default=0, help="edge softening in pixels")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_points(raw: list[str]) -> list[tuple[float, float, int]]:
    points = []
    for entry in raw:
        parts = entry.split(",")
        if len(parts) != 3:
            raise ValueError(f"point must be 'x,y,label'; received {entry!r}")
        x, y, label = float(parts[0]), float(parts[1]), int(parts[2])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"point coordinates are normalised 0..1; received {x},{y}")
        if label not in (0, 1):
            raise ValueError("point label must be 1 (include) or 0 (exclude)")
        points.append((x, y, label))
    return points


def validate(args: argparse.Namespace) -> list[tuple[float, float, int]]:
    for value in (args.model_root, args.video_path):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    if args.start_time < 0 or args.end_time <= args.start_time:
        raise ValueError("tracking requires 0 <= start-time < end-time")
    if Path(args.output_path).suffix.lower() != ".mp4":
        raise ValueError("output must be an MP4 file")
    points = parse_points(args.point)
    if not points:
        raise ValueError("at least one point is required — click the subject to track")
    if not any(label == 1 for _, _, label in points):
        raise ValueError("at least one point must be an include point")
    return points


def probe(path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,avg_frame_rate",
         "-show_entries", "format=duration", "-of", "json", path],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    stream = (payload.get("streams") or [{}])[0]
    numerator, _, denominator = (stream.get("avg_frame_rate") or "24/1").partition("/")
    try:
        fps = float(numerator) / float(denominator or 1)
    except (ValueError, ZeroDivisionError):
        fps = 24.0
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps if fps > 0 else 24.0,
        "duration": float((payload.get("format") or {}).get("duration") or 0.0),
    }


def extract_frames(args: argparse.Namespace, source: dict, scratch: Path) -> int:
    """SAM 2's video predictor consumes a directory of JPEGs."""
    frames_dir = scratch / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    duration = args.end_time - args.start_time
    count = min(int(round(duration * source["fps"])), MAX_TRACK_FRAMES)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{args.start_time:.6f}", "-i", args.video_path,
    ]
    preview_width = int(getattr(args, "preview_width", 0) or 0)
    if preview_width:
        command.extend(["-vf", f"scale={preview_width}:-2"])
    command.extend([
        "-frames:v", str(count), "-q:v", "2", "-start_number", "0",
        str(frames_dir / "%05d.jpg"),
    ])
    subprocess.run(command, check=True)
    produced = len(list(frames_dir.glob("*.jpg")))
    if not produced:
        raise RuntimeError("ffmpeg produced no frames for the selected range")
    return produced


def find_checkpoint(model_root: Path) -> tuple[Path, str]:
    """Locate the checkpoint and the matching Hydra config name the pack ships with."""
    checkpoint = next(iter(sorted(model_root.glob("*.pt"))), None)
    if checkpoint is None:
        raise FileNotFoundError(f"no SAM 2 checkpoint (*.pt) found under {model_root}")
    # sam2.1_hiera_l.yaml -> configs/sam2.1/sam2.1_hiera_l.yaml as registered by the package.
    config = next(iter(sorted(model_root.glob("*.yaml"))), None)
    config_name = f"configs/sam2.1/{config.name}" if config else "configs/sam2.1/sam2.1_hiera_l.yaml"
    return checkpoint, config_name


def track(args: argparse.Namespace, points: list, source: dict, scratch: Path, frame_count: int):
    import numpy as np
    import torch

    try:
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as error:
        raise RuntimeError(
            "SAM 2 is not installed. Install it with: " + INSTALL_HINT
        ) from error

    model_root = Path(args.model_root)
    checkpoint, config_name = find_checkpoint(model_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sam2] {checkpoint.name} on {device} · {frame_count} frames", file=sys.stderr)

    predictor = build_sam2_video_predictor(config_name, str(checkpoint), device=device)
    frames_dir = scratch / "frames"
    state = predictor.init_state(video_path=str(frames_dir))

    # Points arrive normalised; SAM 2 wants pixels in the source frame's own space.
    coords = np.array([[x * source["width"], y * source["height"]] for x, y, _ in points], dtype=np.float32)
    labels = np.array([label for _, _, label in points], dtype=np.int32)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1,
                                    points=coords, labels=labels)

    masks: dict[int, "np.ndarray"] = {}
    for frame_idx, _obj_ids, mask_logits in predictor.propagate_in_video(state):
        masks[frame_idx] = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
    if not masks:
        raise RuntimeError("SAM 2 produced no masks; try a different point")
    del predictor, state
    if device == "cuda":
        torch.cuda.empty_cache()
    return masks


def write_masks(masks: dict, args: argparse.Namespace, source: dict, scratch: Path) -> int:
    from PIL import Image
    import numpy as np

    mask_dir = scratch / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for index in sorted(masks):
        mask = masks[index].astype(np.uint8) * 255
        image = Image.fromarray(mask, mode="L").resize(
            (source["width"], source["height"]), Image.NEAREST
        )
        if args.feather > 0:
            from PIL import ImageFilter

            image = image.filter(ImageFilter.GaussianBlur(radius=args.feather))
        image.save(mask_dir / f"{index:05d}.png")
        written += 1
    return written


def compose(args: argparse.Namespace, source: dict, scratch: Path) -> None:
    """Turn the mask sequence into the requested deliverable with one ffmpeg graph."""
    mask_input = str(scratch / "masks" / "%05d.png")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    fps = source["fps"]
    common = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y"]

    if args.output_kind == "matte":
        command = [*common, "-framerate", f"{fps:.6f}", "-i", mask_input,
                   "-vf", "format=gray,format=yuv420p",
                   "-c:v", "libx264", "-preset", "medium", "-crf", "14", args.output_path]
    else:
        duration = args.end_time - args.start_time
        # alphamerge needs the mask as the alpha of the source, then flatten over black.
        chain = ("[0:v]format=rgba,setsar=1[base];[1:v]format=gray[m];"
                 "[base][m]alphamerge,format=rgba[keyed];"
                 "color=c=black:s={w}x{h}:r={r}[bg];[bg][keyed]overlay=shortest=1,format=yuv420p[out]"
                 ).format(w=source["width"], h=source["height"], r=f"{fps:.6f}")
        if args.output_kind == "removed":
            chain = chain.replace("[1:v]format=gray[m];", "[1:v]format=gray,negate[m];")
        command = [*common,
                   "-ss", f"{args.start_time:.6f}", "-t", f"{duration:.6f}", "-i", args.video_path,
                   "-framerate", f"{fps:.6f}", "-i", mask_input,
                   "-filter_complex", chain, "-map", "[out]",
                   "-c:v", "libx264", "-preset", "medium", "-crf", "16", args.output_path]
    subprocess.run(command, check=True)


def main() -> int:
    force_utf8_io()
    args = build_parser().parse_args()
    points = validate(args)
    source = probe(args.video_path)
    if source["width"] <= 0:
        raise ValueError("the source clip has no decodable video stream")
    if args.dry_run:
        print(json.dumps({"ok": True, "points": len(points), "kind": args.output_kind}))
        return 0

    with tempfile.TemporaryDirectory(prefix="aiv-sam2-") as scratch_dir:
        scratch = Path(scratch_dir)
        frame_count = extract_frames(args, source, scratch)
        masks = track(args, points, source, scratch, frame_count)
        written = write_masks(masks, args, source, scratch)
        compose(args, source, scratch)
    print(json.dumps({"ok": True, "outputPath": args.output_path,
                      "frames": written, "kind": args.output_kind}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
