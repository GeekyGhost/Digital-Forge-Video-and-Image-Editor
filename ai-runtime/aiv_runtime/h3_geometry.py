"""Geometry rules for MiniMax-H3, kept in one place because three layers need them.

The editor validates a request in Rust, the panel snaps the controls in TypeScript and the adapter
validates again in Python. When those three disagree the user gets a worker that dies after loading
20 GB of weights, which is the most expensive way possible to report a typo. `src/features/
generation/h3.ts` mirrors this file; `tests/test_h3_geometry.py` pins both against the numbers
diffusers actually uses.

The rules come from `diffusers.modular_pipelines.minimax_h3.modular_pipeline`:
`resolve_canvas_size` and `align_num_frames`. They are NOT the same as LTX's -- LTX wants 64-pixel
axes and 8n+1 frames, H3 wants 32-pixel axes and 17n+5.
"""

from __future__ import annotations

import os

# The released checkpoint's constants. `CANVAS_MULTIPLE` is the video VAE's spatial compression and
# `FRAMES_PER_CHUNK` / `LATENTS_PER_CHUNK` are its temporal chunking, so none of these are
# preferences that can be tuned -- a value off the grid is a decode the VAE cannot perform.
CANVAS_MULTIPLE = 32
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
FPS = 24
MIN_ASPECT_RATIO = 1 / 4
MAX_ASPECT_RATIO = 4

# The trained canvas. `canvas_short_edge` is a pipeline config rather than a hard limit, which is
# what lets a 16 GB card run below the trained size; the area budget is the checkpoint's.
DEFAULT_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344

# The 16 GB profile. 544 keeps the short edge on the 32-grid and lands 960x544 -- the geometry an
# early RTX 4090 Laptop report finished in about three minutes. Below 384 the model drifts off the
# resolutions it was trained at and quality falls away faster than the time saved is worth.
LOW_VRAM_SHORT_EDGE = 544
MIN_SHORT_EDGE = 384

# Duration limits, in seconds, that the checkpoint declares. They bound the frame count *after*
# alignment, which is why the frame bounds below are not simply 5*24 and 15*24.
MIN_SECONDS = 5.0
MAX_SECONDS = 15.0


def align_num_frames(num_frames: int) -> int:
    """Snap a frame count up to the next `17n + 5` the video VAE can encode.

    Snapping *up* matches diffusers. A count snapped down could fall under the 5 second floor that
    the pipeline then rejects, so the direction is load-bearing rather than a rounding preference.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    while num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        num_frames += 1
    return num_frames


#: The shortest aligned count that is still at least `MIN_SECONDS` -- 124 frames, 5.167 s.
MIN_FRAMES = align_num_frames(int(MIN_SECONDS * FPS))
#: The longest aligned count that is still at most `MAX_SECONDS` -- 345 frames, 14.375 s. Note that
#: this is NOT `align_num_frames(15 * 24)`: that snaps up to 362, which is 15.083 s and rejected.
MAX_FRAMES = max(
    frames
    for frames in range(MIN_FRAMES, int(MAX_SECONDS * FPS) + 1)
    if frames % FRAMES_PER_CHUNK == LATENTS_PER_CHUNK and frames / FPS <= MAX_SECONDS
)


def clamp_num_frames(num_frames: int) -> int:
    """Align a frame count and hold it inside the checkpoint's duration window."""
    return min(MAX_FRAMES, max(MIN_FRAMES, align_num_frames(num_frames)))


def resolve_canvas(
    aspect_width: float,
    aspect_height: float,
    short_edge: int = DEFAULT_SHORT_EDGE,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    """Resolve a display aspect ratio into an H3 canvas, returning `(width, height)`.

    Mirrors `diffusers…resolve_canvas_size`, which returns `(height, width)` -- the order is
    flipped here to match every other width/height pair in this project. Only the *ratio* of the
    first two arguments matters, so a keyframe's pixel dimensions can be passed directly.
    """
    if aspect_width <= 0 or aspect_height <= 0:
        raise ValueError(f"aspect ratio must be positive, got {aspect_width}:{aspect_height}")
    ratio = aspect_width / aspect_height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiniMax-H3 supports aspect ratios from 1:4 to 4:1, got "
            f"{aspect_width}:{aspect_height} ({ratio:g})"
        )

    if ratio >= 1.0:
        width, height = short_edge * ratio, float(short_edge)
    else:
        width, height = float(short_edge), short_edge / ratio

    area = width * height
    if area > max_pixels:
        scale = (max_pixels / area) ** 0.5
        width, height = width * scale, height * scale

    return (
        max(CANVAS_MULTIPLE, round(width / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
        max(CANVAS_MULTIPLE, round(height / CANVAS_MULTIPLE) * CANVAS_MULTIPLE),
    )


def validate_canvas(width: int, height: int) -> None:
    """Reject a canvas the VAE cannot decode, before 20 GB of weights are loaded to discover it."""
    for name, value in (("width", width), ("height", height)):
        if value <= 0 or value % CANVAS_MULTIPLE:
            raise ValueError(f"{name} must be a positive multiple of {CANVAS_MULTIPLE}, got {value}")
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"canvas is limited to {MAX_PIXELS} pixels ({MAX_PIXELS // 1000}k), got {width}x{height}"
        )
    ratio = width / height
    if not MIN_ASPECT_RATIO <= ratio <= MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio must be between 1:4 and 4:1, got {width}x{height}")


def validate_num_frames(num_frames: int) -> None:
    """Reject a frame count off the `17n + 5` grid or outside the 5-15 second window."""
    if num_frames % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"num-frames must be of the form 17n + 5, got {num_frames}")
    seconds = num_frames / FPS
    allow_short = os.environ.get("AIV_ALLOW_SHORT_FRAMES", "").lower() in ("1", "true", "yes")
    min_sec = 0.5 if allow_short else MIN_SECONDS
    if not min_sec <= seconds <= MAX_SECONDS:
        raise ValueError(
            f"MiniMax-H3 generates {MIN_SECONDS:g}-{MAX_SECONDS:g} seconds; {num_frames} frames is "
            f"{seconds:.3f} s (valid counts run {MIN_FRAMES}-{MAX_FRAMES})"
        )
