"""Matte a generated character into an RGBA element (VFX Phase 5).

Two models, each doing what it is good at:
- SAM 2 (a cached video track, `mask_cache.py` layout) decides WHICH pixels are the character, with
  temporal identity: it followed a spinning dancer where a point tracker failed (Phase 4b). Its
  masks are hard-edged.
- BiRefNet gives SOFT edges frame by frame (hair, fingers, motion blur), which the generated plain
  background makes easy, but it has no idea of identity and can pick up a stray object.

The combination (`combine`): BiRefNet's alpha inside a dilated SAM 2 region, and at least the eroded
SAM 2 interior, so BiRefNet can neither add a second subject nor punch a hole in the character.
Without SAM 2 masks it is BiRefNet alone (the driving clip was not the plate, so there was nothing to
seed SAM 2 with); without BiRefNet it is SAM 2 with a light feather.

Runs in the Image Studio environment (BiRefNet lives there). Writes `frame_%04d.png` (straight alpha)
into the element folder, then prints one JSON line of stats.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

_PACKAGE_ROOT = str(Path(__file__).resolve().parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)


def region_radii(height: int) -> tuple[int, int]:
    """(dilate, erode) in pixels for a frame this tall: about 2.5% and 1%, at least 3 and 1."""
    return max(3, round(height * 0.025)), max(1, round(height * 0.01))


def _morph(mask: Image.Image, radius: int, grow: bool) -> Image.Image:
    if radius <= 0:
        return mask
    size = min(2 * radius + 1, 65)
    size += (size + 1) % 2
    return mask.filter(ImageFilter.MaxFilter(size) if grow else ImageFilter.MinFilter(size))


def combine(sam2: np.ndarray | None, soft: np.ndarray | None, height: int) -> np.ndarray:
    """uint8 alpha from a SAM 2 mask (0/255) and a BiRefNet alpha (0..255), either may be None."""
    if sam2 is None and soft is None:
        raise ValueError("no matte: neither SAM 2 masks nor BiRefNet are available")
    if sam2 is None:
        return soft.astype(np.uint8)
    hard = Image.fromarray(sam2.astype(np.uint8), mode="L")
    grow, shrink = region_radii(height)
    if soft is None:
        return np.asarray(hard.filter(ImageFilter.GaussianBlur(radius=max(1.0, shrink * 0.75))), dtype=np.uint8)
    region = np.asarray(_morph(hard, grow, True).filter(ImageFilter.GaussianBlur(radius=grow / 3)), dtype=np.float32) / 255.0
    interior = np.asarray(_morph(hard, shrink, False), dtype=np.float32)
    alpha = np.maximum(soft.astype(np.float32) * region, interior)
    return np.clip(alpha + 0.5, 0, 255).astype(np.uint8)


def stabilise(previous: np.ndarray | None, current: np.ndarray, memory: float) -> np.ndarray:
    """A light causal blend against edge shimmer (the Mask VFX `stabilize_masks` recipe)."""
    if previous is None or memory <= 0:
        return current
    return np.clip(current.astype(np.float32) * (1 - memory) + previous.astype(np.float32) * memory + 0.5, 0, 255).astype(np.uint8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matte a generated character into RGBA frames")
    parser.add_argument("--frames-dir", required=True, help="colour frames 00000.png ...")
    parser.add_argument("--sam2-dir", default="", help="a mask cache folder (masks/00000.png ...), optional")
    parser.add_argument("--birefnet-root", default="", help="models/BiRefNet, optional")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--smooth", type=float, default=0.2, help="0..0.6 causal alpha smoothing")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    frames = sorted(Path(args.frames_dir).glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no frames in {args.frames_dir}")
    masks = sorted((Path(args.sam2_dir) / "masks").glob("*.png")) if args.sam2_dir else []
    if masks and len(masks) < len(frames):
        raise ValueError(f"the SAM 2 track has {len(masks)} masks for {len(frames)} frames")
    pipeline = None
    if args.birefnet_root and Path(args.birefnet_root, "config.json").is_file():
        from aiv_runtime.image_studio.birefnet import BiRefNetPipeline

        pipeline = BiRefNetPipeline.create(args.birefnet_root, args.device)
    if pipeline is None and not masks:
        raise RuntimeError("neither BiRefNet nor a SAM 2 track is available to matte the character")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    previous = None
    coverage = []
    for index, path in enumerate(frames):
        with Image.open(path) as opened:
            colour = opened.convert("RGB")
        sam2 = None
        if masks:
            with Image.open(masks[index]) as mask:
                sam2 = np.asarray(mask.convert("L").resize(colour.size, Image.Resampling.NEAREST))
        soft = np.asarray(pipeline.remove_background(colour).mask.convert("L")) if pipeline else None
        alpha = stabilise(previous, combine(sam2, soft, colour.height), max(0.0, min(0.6, args.smooth)))
        previous = alpha
        coverage.append(float((alpha > 127).mean()))
        rgba = colour.copy()
        rgba.putalpha(Image.fromarray(alpha, mode="L"))
        rgba.save(out / f"frame_{index:04d}.png", compress_level=2)
        if index % 24 == 0:
            print(f"[character-matte] {index + 1}/{len(frames)}", file=sys.stderr, flush=True)
    print(json.dumps({
        "frames": len(frames), "sam2": bool(masks), "birefnet": pipeline is not None,
        "seconds": round(time.time() - started, 1),
        "coverageMin": round(min(coverage), 4), "coverageMax": round(max(coverage), 4),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
