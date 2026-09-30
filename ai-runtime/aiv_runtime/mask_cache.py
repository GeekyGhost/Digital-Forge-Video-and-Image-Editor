"""Acquire and persist a reusable SAM 2 matte sequence.

Tracking is the expensive, model-dependent operation. Effects, colors, transitions and preview
quality are intentionally absent from this worker so they can change without loading SAM 2 again.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from aiv_runtime import sam2_track


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache a reusable SAM 2 video matte")
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--cache-id", required=True)
    parser.add_argument("--start-time", type=float, required=True)
    parser.add_argument("--end-time", type=float, required=True)
    parser.add_argument("--point", action="append", default=[])
    return parser


def _remove_small_regions(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    """Remove isolated matte islands while retaining the original soft edge when OpenCV exists.

    Same as `mask_effects._remove_small_regions` (VFX Phase 7 moved the bake here).
    """
    if minimum_area <= 0:
        return mask
    try:
        import cv2
    except ImportError:
        return mask
    binary = (mask >= 96).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros_like(binary)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            keep[labels == label] = 1
    keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
    return mask * keep.astype(np.float32)


def stabilize(cache: Path, smoothing: int, cleanup: int) -> dict:
    """Bake the causal matte stabilisation (`mask_effects.stabilize_masks`) into a matte-cache layout.

    VFX Phase 7: the compositor reads any frame of a matte directly, so the smoothing that depends on
    every earlier frame is done once here rather than on every render.
    """
    manifest_path = cache / "manifest.json"
    if not manifest_path.is_file() or not (cache / "masks").is_dir():
        raise FileNotFoundError("the matte cache is missing or incomplete")
    smoothing = max(0, min(5, smoothing))
    cleanup = max(0, min(2000, cleanup))
    target = cache / f"stable-s{smoothing}-c{cleanup}"
    target_manifest = target / "manifest.json"
    if target_manifest.is_file():
        result = json.loads(target_manifest.read_text(encoding="utf-8"))
        result["reused"] = True
        return result
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out_dir = target / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    memory = smoothing / (smoothing + 4.0) if smoothing else 0.0
    previous: np.ndarray | None = None
    paths = sorted((cache / "masks").glob("*.png"))
    for index, path in enumerate(paths):
        with Image.open(path) as frame:
            current = np.asarray(frame.convert("L"), dtype=np.float32)
        current = _remove_small_regions(current, cleanup)
        if previous is not None and memory:
            current = current * (1 - memory) + previous * memory
        previous = current
        Image.fromarray(np.clip(current, 0, 255).astype(np.uint8), mode="L").save(out_dir / path.name, compress_level=2)
        if index % 24 == 0:
            print(f"[mask-cache] stabilised {index + 1}/{len(paths)}", file=sys.stderr, flush=True)
    result = {
        **manifest, "cachePath": str(target), "manifestPath": str(target_manifest),
        "temporalSmooth": smoothing, "cleanup": cleanup, "stabilizedFrom": str(cache), "reused": False,
    }
    pending = target_manifest.with_suffix(".json.pending")
    pending.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pending.replace(target_manifest)
    return result


def main() -> int:
    sam2_track.force_utf8_io()
    if "--stabilize" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description="Bake a cached matte's temporal smoothing and cleanup")
        parser.add_argument("--stabilize", action="store_true")
        parser.add_argument("--cache-dir", required=True)
        parser.add_argument("--temporal-smooth", type=int, default=2)
        parser.add_argument("--cleanup", type=int, default=300)
        options = parser.parse_args()
        print(json.dumps(stabilize(Path(options.cache_dir), options.temporal_smooth, options.cleanup)))
        return 0
    args = build_parser().parse_args()
    cache = Path(args.cache_dir)
    manifest_path = cache / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({"reused": True, "manifestPath": str(manifest_path), "cachePath": str(cache)})
        print(json.dumps(manifest))
        return 0

    if not Path(args.video_path).is_file():
        raise FileNotFoundError(args.video_path)
    if not Path(args.model_root).exists():
        raise FileNotFoundError(args.model_root)
    if args.start_time < 0 or args.end_time <= args.start_time:
        raise ValueError("mask tracking requires 0 <= start-time < end-time")
    points = sam2_track.parse_points(args.point)
    if not points or not any(label == 1 for _, _, label in points):
        raise ValueError("SAM 2 needs at least one include point")

    source = sam2_track.probe(args.video_path)
    expected = int(round((args.end_time - args.start_time) * source["fps"]))
    if expected > sam2_track.MAX_TRACK_FRAMES:
        raise ValueError(
            f"selected range is {expected} frames; SAM 2 supports at most {sam2_track.MAX_TRACK_FRAMES}"
        )
    cache.mkdir(parents=True, exist_ok=True)
    mask_dir = cache / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aiv-mask-cache-") as scratch_name:
        scratch = Path(scratch_name)
        frame_count = sam2_track.extract_frames(args, source, scratch)
        tracked = sam2_track.track(args, points, source, scratch, frame_count)
        last = np.zeros((source["height"], source["width"]), dtype=np.uint8)
        for index in range(frame_count):
            current = tracked.get(index)
            if current is not None:
                last = current.astype(np.uint8) * 255
            Image.fromarray(last, mode="L").resize(
                (source["width"], source["height"]), Image.Resampling.NEAREST,
            ).save(mask_dir / f"{index:05d}.png", compress_level=2)
            if index % 24 == 0:
                print(f"[mask-cache] wrote {index + 1}/{frame_count}", file=sys.stderr, flush=True)

    manifest = {
        "cacheId": args.cache_id,
        "cachePath": str(cache),
        "manifestPath": str(manifest_path),
        "frameCount": frame_count,
        "width": source["width"],
        "height": source["height"],
        "fps": source["fps"],
        "model": "sam2",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "reused": False,
    }
    pending_manifest = manifest_path.with_suffix(".json.pending")
    pending_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pending_manifest.replace(manifest_path)
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
