"""Geometry rules for LTX 2.3, kept in one place because three adapters and two layers need them.

`h3_geometry.py` states the house rule and the reason for it: the editor validates a request in
Rust, the panel snaps the controls in TypeScript and the adapter validates again in Python, and
when those layers disagree the user gets a worker that dies after loading 20 GB of weights, which
is the most expensive way possible to report a typo.

LTX had no such module, and its three adapters disagreed four ways:

    rule            single_stage      retake_16gb        foley_v2a
    axis multiple   64                32                 64, then halved
    frame grid      8n+1, reject      8n+1, reject       8n+1, floor-snap
    max frames      none              193                none
    pixel cap       768x768 fixed     768x768, env       768x512

`--num-frames 481` was accepted by generation and OOMed after loading 27 GB -- exactly the failure
the H3 module exists to prevent. `832x480` was retakeable but not generatable.

`src/features/generation/ltxGeometry.ts` mirrors this file and
`ai-runtime/tests/test_ltx_geometry.py` pins both, the way the H3 pair is pinned.

Where the numbers come from
---------------------------
`AXIS_MULTIPLE` is **32**, not 64. The pinned Lightricks tree says so in as many words:
`ltx_pipelines/utils/helpers.py::assert_resolution` computes `divisor = 64 if is_two_stage else 32`
and documents it, `ti2vid_one_stage.py:126` calls it with `is_two_stage=False`, and
`retake.py:311` checks `% 32` directly. Every 64 in this project came from a two-stage reference,
and there is no stage 2 here -- see the plan's settled decisions.

`FRAMES_PER_LATENT` is the temporal compression, so `8n + 1` is a decode the VAE can perform rather
than a preference. It is `SpatioTemporalScaleFactors.default().time` in the pinned tree; retake
reads it from there at runtime and this module pins the same value so the panel can snap without
importing the LTX package.
"""

from __future__ import annotations

import os

# The checkpoint's own grid. Neither is tunable: a value off it is a decode the VAE cannot perform.
AXIS_MULTIPLE = 32
FRAMES_PER_LATENT = 8
FRAME_REMAINDER = 1

#: The shortest legal count. One latent frame plus the anchor.
MIN_FRAMES = FRAMES_PER_LATENT + FRAME_REMAINDER

# VRAM budgets for the 16 GB profile. These are OUR limits, not the model's, which is why they are
# overridable and why the adapters print measured peaks: the real cost is tokens -- pixels TIMES
# latent frames -- so a short window at a higher resolution can be cheaper than a long one at a low
# resolution. See `pixel_frames`.
DEFAULT_MAX_PIXELS = 768 * 768

#: The longest count the 16 GB profile has been run at. Generation had no cap at all, which is how
#: `--num-frames 481` reached a 27 GB load before failing.
DEFAULT_MAX_FRAMES = 193

# The token budget: pixels TIMES latent frames, which is what the cost actually follows. Retake's
# own comment has said so since it was written and its guard then tested pixels alone, so a short
# window at a high resolution was refused while a long one at a low resolution -- more expensive --
# was waved through.
#
# The largest completed local LTX 2.3 retake recorded in `.aiv/worker.log` is
# 736x544 over 185 frames = 9,609,216 pixel-frames under CPU streaming, at 7,918 MiB allocated.
# This replaces the old 14,745,600 theoretical corner, which had never been run and therefore was
# not a valid 16-GB qualification. Other offload modes remain explicitly experimental.
#
#   known:  768x512x49  = 2,752,512 pixel-frames peaked at 7,905 MiB of 16,376
#   probe:  768x512x121 = 6,291,456 pixel-frames
#   probe:  768x768x49  = 4,128,768 pixel-frames
#
# Both probes print their peak next to their token count. Set this from the larger one that fits.
DEFAULT_MAX_PIXEL_FRAMES = 9_609_216


def snap_axis(value: int) -> int:
    """Round one axis UP onto the grid.

    Up, not nearest: rounding down silently shrinks the user's canvas, and the pixel budget is
    enforced separately by `snap_canvas`, which steps back down if the rounded pair overflows it.
    """
    if value <= 0:
        raise ValueError(f"axis must be positive, got {value}")
    return max(AXIS_MULTIPLE, -(-value // AXIS_MULTIPLE) * AXIS_MULTIPLE)


def snap_axes(width: int, height: int) -> tuple[int, int]:
    """Round both axes up onto the grid, without applying any pixel budget."""
    return snap_axis(width), snap_axis(height)


def snap_canvas(width: int, height: int, max_pixels: int = DEFAULT_MAX_PIXELS) -> tuple[int, int]:
    """Fit a canvas onto the grid and inside a pixel budget, preserving the aspect ratio.

    Scales to the budget first, rounds up onto the grid, then steps the longer axis back down a
    grid unit at a time if rounding pushed the pair over. That last loop is what makes the result
    always legal rather than usually legal.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"canvas must be positive, got {width}x{height}")
    scale = min(1.0, (max_pixels / float(width * height)) ** 0.5)
    target_width, target_height = snap_axes(round(width * scale), round(height * scale))
    while target_width * target_height > max_pixels:
        if target_width >= target_height and target_width > AXIS_MULTIPLE:
            target_width -= AXIS_MULTIPLE
        elif target_height > AXIS_MULTIPLE:
            target_height -= AXIS_MULTIPLE
        else:
            break
    return target_width, target_height


def snap_frames(num_frames: int, direction: str = "nearest") -> int:
    """Snap a frame count onto `8n + 1`.

    `nearest` is the default and is what the panel and the validators use: refusing a count the
    user cannot reasonably be expected to compute is the behaviour this module exists to end.

    Direction matters where the count describes existing footage rather than a request. Foley used
    to floor unconditionally, which is half of why its mux truncated the picture -- the audio was
    planned shorter than the clip. Snapping to nearest and letting the mux carry the source's own
    duration is the pair of changes that fixes it, so `down` is kept only for callers that really
    must not exceed the source.
    """
    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    lower = (num_frames - FRAME_REMAINDER) // FRAMES_PER_LATENT * FRAMES_PER_LATENT + FRAME_REMAINDER
    if lower < MIN_FRAMES:
        lower = MIN_FRAMES
    if lower == num_frames:
        return num_frames
    upper = lower + FRAMES_PER_LATENT
    if direction == "down":
        return lower
    if direction == "up":
        return upper
    if direction != "nearest":
        raise ValueError(f"direction must be nearest, up or down, got {direction!r}")
    return lower if (num_frames - lower) <= (upper - num_frames) else upper


def latent_frames(num_frames: int) -> int:
    """How many latent frames a count encodes to. The temporal half of the real cost."""
    return (num_frames - 1) // FRAMES_PER_LATENT + 1


def pixel_frames(width: int, height: int, num_frames: int) -> int:
    """Pixels times latent frames: the metric the VRAM cost actually follows.

    `ltx_retake_16gb.py` has said so in a comment since it was written, printed this number, and
    then guarded on pixels alone.
    """
    return width * height * latent_frames(num_frames)


def env_max_pixels(*names: str, default: int = DEFAULT_MAX_PIXELS) -> int:
    """Read a pixel budget override, taking the first name that is set.

    Several names because `AIV_RETAKE_MAX_PIXELS` shipped first and is documented; a later, better
    name keeps it as an alias rather than breaking anyone's notes.
    """
    for name in names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        try:
            value = int(raw)
        except ValueError as error:
            raise ValueError(f"{name} must be an integer number of pixels, got {raw!r}") from error
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value
    return default


def validate_axes(width: int, height: int, max_pixels: int = DEFAULT_MAX_PIXELS) -> None:
    """The last-resort assert, not the user-facing gate.

    Callers snap first; this exists so that a path which forgot to fails here, in argument
    validation, rather than 27 GB into a load.
    """
    for name, value in (("width", width), ("height", height)):
        if value <= 0 or value % AXIS_MULTIPLE:
            raise ValueError(f"{name} must be a positive multiple of {AXIS_MULTIPLE}, got {value}")
    if width * height > max_pixels:
        raise ValueError(
            f"{width}x{height} is {width * height:,} pixels, over the {max_pixels:,} budget for "
            "this profile"
        )


def validate_pixel_frames(
    width: int, height: int, num_frames: int, budget: int = DEFAULT_MAX_PIXEL_FRAMES
) -> None:
    """Guard on the metric the cost follows, rather than on pixels and frames separately.

    The separate caps miss the diagonal: they accept the corner where both are at their maximum,
    and they refuse a short window at a high resolution that is genuinely cheaper than a long one
    at a low resolution.
    """
    cost = pixel_frames(width, height, num_frames)
    if cost > budget:
        raise ValueError(
            f"{width}x{height} over {num_frames} frames is {cost:,} pixel-frames, over the "
            f"{budget:,} budget. Cost scales with pixels TIMES latent frames, so shortening the "
            "window buys resolution and vice versa."
        )


def validate_num_frames(num_frames: int, max_frames: int | None = DEFAULT_MAX_FRAMES) -> None:
    """Reject a count off the `8n + 1` grid, or longer than the profile has been run at."""
    if num_frames < MIN_FRAMES:
        raise ValueError(f"num-frames must be at least {MIN_FRAMES}, got {num_frames}")
    if (num_frames - FRAME_REMAINDER) % FRAMES_PER_LATENT:
        raise ValueError(f"num-frames must be of the form 8n + 1, got {num_frames}")
    if max_frames is not None and num_frames > max_frames:
        raise ValueError(
            f"{num_frames} frames is over the {max_frames} this 16 GB profile has been run at; "
            "shorten the request or raise the profile's limit deliberately"
        )
