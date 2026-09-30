"""Clean plate for a locked-off shot: hide the actor a generated character replaces (VFX Phase 5).

Wan Animate 2 has no replacement mode (it makes the character on a plain background), so where the
character does not cover the actor -- hair above a helmet, a sleeve past an arm -- the actor shows.
For a camera that does not move, the background behind the actor is in other frames. This worker
builds it and writes a patch element: per plate frame, the clean background with alpha = that
frame's actor matte (dilated, feathered), cropped to the actor's neighbourhood. Laid between the
plate and the character, it paints the actor out wherever the character does not paint over.

How the clean still is made, per pixel of the region:
1. **Range samples** (up to 32 frames of the plate's range, paired with the SAM 2 actor mattes,
   dilated): the median of the samples where the actor is not. Same lighting as the shot.
2. **Context samples** (up to 64 frames within 30 s either side of the range, no mattes), read
   outward from the range and kept until the camera setup changes (two frames in a row whose
   background differs from (1) -- a camera move or a cut ends the side): their plain median is the
   background. Used where (1) has fewer than 3 samples. The actor may stand in one spot in most of
   them, so only samples near a guess (the range's background inpainted across the gap) count:
   brick and asphalt, not the costume. (A stock clip can hold several setups; only the shot's own neighbours count.)
3. What neither covers is filled by OpenCV's Telea inpainting, and reported as `holes`.

Refuses when the range itself does not hold still (the camera moves): `residual` is the mean
absolute difference between each range sample and the clean still on its uncovered pixels.

Runs in the Image Studio environment (numpy, Pillow, OpenCV). ffmpeg on PATH. Prints one JSON line.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

RANGE_SAMPLES = 32
CONTEXT_SAMPLES = 64
#: How far either side of the range context frames are looked for, in seconds.
CONTEXT_WINDOW_SECONDS = 30
#: How far (mean absolute, 0..255) a context sample may be from the inpainted guess and count as background.
PRIOR_DISTANCE = 40.0
MIN_RANGE_VOTES = 3
MIN_CONTEXT_VOTES = 8
#: Mean absolute difference (0..255) above which the range is not a locked-off shot.
MAX_RESIDUAL = 6.0
#: A context frame belongs to the shot when the MEDIAN absolute difference from the range's clean
#: background, over the background around the actor, is at most this (the actor may stand there).
MAX_CONTEXT_DIFFERENCE = 5.0


def spread(first: int, count: int, most: int) -> list[int]:
    """Up to `most` frame indices spread evenly over [first, first + count), ends included."""
    if count <= 0 or most <= 0:
        return []
    if count <= most:
        return list(range(first, first + count))
    step = (count - 1) / (most - 1) if most > 1 else 0
    return sorted({first + round(index * step) for index in range(most)})


def context_sides(total: int, first: int, count: int, window: int, step: int) -> tuple[list[int], list[int]]:
    """Frames of the clip outside [first, first + count) within `window` frames of it, every `step`,
    each side ordered outward from the range (nearest first): the same camera setup is most likely
    right next to the shot, and a side is read only until the setup changes."""
    before = list(range(first - step, max(-1, first - window - 1), -step))
    after = list(range(first + count - 1 + step, min(total, first + count + window), step))
    return before, after


def agreeing_run(differences: list[float], limit: float, patience: int = 2) -> int:
    """How many of a side's frames (nearest first) belong to the shot: up to the first `patience`
    consecutive frames that differ from the range's background by more than `limit` (a camera move or
    a cut), less those failing frames."""
    kept, misses = 0, 0
    for index, difference in enumerate(differences):
        if difference <= limit:
            kept, misses = index + 1, 0
        else:
            misses += 1
            if misses >= patience:
                break
    return kept


def dilation_radius(plate_height: int, fraction: float) -> int:
    return max(2, round(plate_height * fraction))


def region_of(mattes: list[np.ndarray], margin: int, width: int, height: int) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1): the union box of the mattes (> 127) grown by `margin`, inside the frame."""
    x0, y0, x1, y1 = width, height, -1, -1
    for matte in mattes:
        ys, xs = np.nonzero(matte > 127)
        if xs.size:
            x0, x1 = min(x0, int(xs.min())), max(x1, int(xs.max()))
            y0, y1 = min(y0, int(ys.min())), max(y1, int(ys.max()))
    if x1 < 0:
        return None
    return max(0, x0 - margin), max(0, y0 - margin), min(width, x1 + 1 + margin), min(height, y1 + 1 + margin)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    import cv2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel) > 0


def masked_median(frames: np.ndarray, hidden: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median of `frames` (N, H, W, 3 uint8) over the samples where `hidden` (N, H, W) is
    False. Returns (median float32 with NaN where no sample, votes int)."""
    if hidden is None:
        return np.median(frames.astype(np.float32), axis=0), np.full(frames.shape[1:3], len(frames))
    votes = (~hidden).sum(axis=0)
    median = np.full(frames.shape[1:], np.nan, dtype=np.float32)
    # Row bands keep the float copy small (a 1300 x 750 region x 32 samples is ~370 MB at once).
    for top in range(0, frames.shape[1], 64):
        band = frames[:, top:top + 64].astype(np.float32)
        band[hidden[:, top:top + 64]] = np.nan
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                median[top:top + 64] = np.nanmedian(band, axis=0)
    return median, votes


def background_samples(frames: np.ndarray, prior: np.ndarray, need: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The median of the context samples that look like background, where `need` is set.

    Context frames have no matte, and an actor who keeps to one spot is in most of them there, so a
    plain median returns the actor. `prior` (the range's clean background inpainted across the gap)
    says roughly what colour belongs there: a sample counts when it is within `PRIOR_DISTANCE` of it
    (mean over the channels) -- the brick or the asphalt, not the dark costume. Returns the median
    (NaN where no sample counts) and how many counted."""
    distance = np.abs(frames.astype(np.float32) - prior.astype(np.float32)).mean(axis=-1)
    hidden = (distance > PRIOR_DISTANCE) | ~need[None]
    return masked_median(frames, hidden)


def build_clean(range_frames: np.ndarray, range_hidden: np.ndarray, sides: list[np.ndarray],
                outside: np.ndarray) -> tuple[np.ndarray, dict]:
    """The clean still (H, W, 3 uint8) and stats. `sides` are the context frames before and after
    the range, each ordered outward; `outside` marks the region's pixels the actor never covers in
    the range, where a context frame is compared with the range's background."""
    import cv2

    clean, votes = masked_median(range_frames, range_hidden)
    stats: dict = {"rangeSamples": int(len(range_frames)), "contextKept": 0, "contextSamples": int(sum(len(side) for side in sides))}
    residuals = []
    for frame, hidden in zip(range_frames, range_hidden):
        shown = ~hidden & (votes >= MIN_RANGE_VOTES)
        if shown.any():
            residuals.append(float(np.abs(frame.astype(np.float32) - np.nan_to_num(clean))[shown].mean()))
    stats["residual"] = round(float(np.median(residuals)) if residuals else 0.0, 2)
    need = votes < MIN_RANGE_VOTES
    if need.any() and outside.any() and stats["contextSamples"]:
        reference = np.nan_to_num(clean)[outside]
        keep = []
        for side in sides:
            differences = [float(np.median(np.abs(frame[outside].astype(np.float32) - reference))) for frame in side]
            keep.extend(side[:agreeing_run(differences, MAX_CONTEXT_DIFFERENCE)])
        stats["contextKept"] = len(keep)
        if len(keep) >= MIN_CONTEXT_VOTES:
            prior = cv2.inpaint(np.nan_to_num(clean).clip(0, 255).astype(np.uint8), need.astype(np.uint8) * 255, 9, cv2.INPAINT_TELEA)
            context_median, context_votes = background_samples(np.stack(keep), prior, need)
            filled = need & (context_votes >= MIN_CONTEXT_VOTES // 2)
            clean = np.where(filled[..., None], context_median, clean)
            need = need & ~filled
    holes = need | np.isnan(clean).any(axis=-1)
    stats["holes"] = int(holes.sum())
    stats["_holes"] = holes
    still = np.nan_to_num(clean).clip(0, 255).round().astype(np.uint8)
    if holes.any():
        still = cv2.inpaint(still, holes.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA)
    return still, stats


def patch_alpha(matte: np.ndarray, radius: int) -> np.ndarray:
    """The patch's alpha for one frame: opaque over the actor grown by half of `radius`, fading out
    to about `radius` (a blurred dilation), so no hard seam sits on the plate."""
    import cv2

    inner = dilate(matte > 127, max(1, radius // 2)).astype(np.float32)
    soft = cv2.GaussianBlur(dilate(matte > 127, radius).astype(np.float32), (0, 0), max(1.0, radius / 2))
    return (np.maximum(soft, inner) * 255).round().astype(np.uint8)


def read_frames(video: Path, indices: list[int], region: tuple[int, int, int, int]) -> np.ndarray:
    """RGB frames at decode `indices` (sorted, unique), cropped to `region`, via one ffmpeg pass."""
    x0, y0, x1, y1 = region
    width, height = x1 - x0, y1 - y0
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video),
               "-vf", f"select='{expression}',crop={width}:{height}:{x0}:{y0},format=rgb24",
               "-fps_mode", "passthrough", "-frames:v", str(len(indices)), "-f", "rawvideo", "-"]
    raw = subprocess.run(command, check=True, capture_output=True).stdout
    size = width * height * 3
    if len(raw) < size * len(indices):
        raise ValueError(f"ffmpeg gave {len(raw) // size} of {len(indices)} frames")
    return np.frombuffer(raw[:size * len(indices)], dtype=np.uint8).reshape(len(indices), height, width, 3)


#: How far (mean absolute, 0..255) a filled still may stray from `clean.png` outside the holes.
MAX_FILL_DRIFT = 4.0


def check_filled(still: np.ndarray, out: Path) -> None:
    """Refuse a filled still that is not `clean.png` outside `holes.png` (a fill model that returned
    the wrong range, or the wrong picture): only the holes may change."""
    clean_path, holes_path = out / "clean.png", out / "holes.png"
    if not clean_path.is_file():
        return
    clean = np.asarray(Image.open(clean_path).convert("RGB")).astype(np.float32)
    if clean.shape != still.shape:
        raise ValueError(f"the filled still is {still.shape[1]} x {still.shape[0]}, clean.png is {clean.shape[1]} x {clean.shape[0]}")
    keep = np.ones(clean.shape[:2], bool)
    if holes_path.is_file():
        keep = np.asarray(Image.open(holes_path).convert("L")) < 128
    drift = float(np.abs(still.astype(np.float32) - clean)[keep].mean()) if keep.any() else 0.0
    if drift > MAX_FILL_DRIFT:
        raise ValueError(f"the filled still differs from clean.png outside the holes by {drift:.1f} levels "
                         f"(at most {MAX_FILL_DRIFT}); the fill is not used")


def load_matte(matte_dir: Path, index: int, size: tuple[int, int]) -> np.ndarray:
    """The actor matte for track frame `index` (clamped to the track), at plate `size` (w, h)."""
    files = sorted(matte_dir.glob("*.png"))
    if not files:
        raise ValueError(f"no mattes in {matte_dir}")
    path = files[min(len(files) - 1, max(0, index))]
    image = Image.open(path).convert("L")
    if image.size != size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image)


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    width, height = args.plate_width, args.plate_height
    matte_dir = Path(args.matte_dir)
    matte = lambda frame: load_matte(matte_dir, frame - args.matte_first, (width, height))  # noqa: E731
    frames = list(range(args.first, args.first + args.count))
    mattes = [matte(frame) for frame in frames]
    patch_radius = dilation_radius(height, 0.02)
    hide_radius = dilation_radius(height, 0.03)
    region = region_of(mattes, dilation_radius(height, 0.06), width, height)
    if region is None:
        raise ValueError("the actor matte is empty on every frame of the range")
    x0, y0, x1, y1 = region
    crop = lambda array: array[y0:y1, x0:x1]  # noqa: E731

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.still:
        # Patch only, from a still made earlier and filled elsewhere (LaMa over `holes.png`).
        still = np.asarray(Image.open(args.still).convert("RGB"))
        if still.shape[:2] != (y1 - y0, x1 - x0):
            raise ValueError(f"the still is {still.shape[1]} x {still.shape[0]}, the region is {x1 - x0} x {y1 - y0}")
        check_filled(still, out)
        stats = {"still": str(args.still)}
    else:
        range_indices = spread(args.first, args.count, RANGE_SAMPLES)
        window = round(CONTEXT_WINDOW_SECONDS * args.fps)
        before, after = context_sides(args.total_frames, args.first, args.count, window,
                                      max(1, 2 * window // CONTEXT_SAMPLES)) if args.total_frames else ([], [])
        everything = sorted(set(range_indices) | set(before) | set(after))
        decoded = dict(zip(everything, read_frames(Path(args.video), everything, region)))
        range_frames = np.stack([decoded[index] for index in range_indices])
        range_hidden = np.stack([dilate(crop(matte(index)), hide_radius) for index in range_indices])
        outside = ~range_hidden.any(axis=0)
        sides = [np.stack([decoded[index] for index in side]) if side else np.zeros((0, *range_frames.shape[1:]), np.uint8)
                 for side in (before, after)]
        still, stats = build_clean(range_frames, range_hidden, sides, outside)
        if stats["residual"] > MAX_RESIDUAL:
            raise ValueError(f"the shot is not locked off: frames differ from the clean plate by {stats['residual']} "
                             f"levels on the background (at most {MAX_RESIDUAL}); a clean plate needs a camera that does not move")
        Image.fromarray(still).save(out / "clean.png")
        if stats["holes"]:
            Image.fromarray((stats.pop("_holes") * 255).astype(np.uint8)).save(out / "holes.png")
        stats.pop("_holes", None)
    coverage = []
    for index, frame in enumerate(frames):
        alpha = patch_alpha(crop(mattes[index]), patch_radius)
        coverage.append(float((alpha > 127).mean()))
        Image.fromarray(np.dstack([still, alpha])).save(out / f"frame_{index:04d}.png", compress_level=4)
    stats.update({
        "ok": True, "frames": len(frames), "crop": [x0, y0, x1 - x0, y1 - y0], "width": x1 - x0, "height": y1 - y0,
        "coverageMax": round(max(coverage), 4), "seconds": round(time.monotonic() - started, 1),
        "patchRadius": patch_radius, "hideRadius": hide_radius,
    })
    return stats


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--total-frames", type=int, default=0, help="frames in the whole clip (0: range only)")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--plate-width", type=int, required=True)
    parser.add_argument("--plate-height", type=int, required=True)
    parser.add_argument("--matte-dir", required=True, help="the SAM 2 track's masks folder (00000.png ...)")
    parser.add_argument("--matte-first", type=int, default=0, help="the source frame masks/00000.png belongs to")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--still", default="", help="write the patch from this still (region-sized) instead of building one")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        stats = run(parse(argv))
    except Exception as error:  # the Rust side reads the last JSON line
        print(json.dumps({"ok": False, "error": str(error)}), flush=True)
        return 1
    print(json.dumps(stats), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
