/**
 * LTX 2.3 geometry, mirroring `ai-runtime/aiv_runtime/ltx_geometry.py`.
 *
 * The Python module carries the reasoning; this file exists so the panel snaps to the same grid
 * the worker validates against, instead of the user discovering a typo after a 27 GB load.
 * `ltxGeometry.test.ts` and `ai-runtime/tests/test_ltx_geometry.py` pin the pair.
 *
 * The axis multiple is **32**, not 64. `ltx_pipelines/utils/helpers.py::assert_resolution` computes
 * `divisor = 64 if is_two_stage else 32`, `ti2vid_one_stage.py` calls it with `is_two_stage=False`,
 * and `retake.py` checks `% 32` directly. There is no stage 2 in this project, so every 64 here was
 * inherited from a two-stage reference -- which is why 832x480 used to be retakeable but not
 * generatable.
 */

export const LTX_AXIS_MULTIPLE = 32;
export const LTX_FRAMES_PER_LATENT = 8;
export const LTX_FRAME_REMAINDER = 1;
/** One latent frame plus the anchor. */
export const LTX_MIN_FRAMES = LTX_FRAMES_PER_LATENT + LTX_FRAME_REMAINDER;
/** Our 16 GB budget, not the model's. */
export const LTX_MAX_PIXELS = 768 * 768;
/** The longest count this profile has been run at. Generation used to have no cap at all. */
export const LTX_MAX_FRAMES = 193;
/**
 * The token budget: pixels times latent frames, which is what VRAM cost actually follows. Exactly
 * the largest completed local LTX 2.3 retake: 736×544 over 185 frames, 9,609,216 pixel-frames,
 * with 7,918 MiB allocated under CPU streaming. Mirrors `DEFAULT_MAX_PIXEL_FRAMES` in Python.
 */
export const LTX_MAX_PIXEL_FRAMES = 9_609_216;

/** Round one axis UP onto the grid. Down would silently shrink the user's canvas. */
export function snapLtxAxis(value: number): number {
  return Math.max(LTX_AXIS_MULTIPLE, Math.ceil(Math.max(1, value) / LTX_AXIS_MULTIPLE) * LTX_AXIS_MULTIPLE);
}

/**
 * Fit a canvas onto the grid and inside a pixel budget, preserving the aspect ratio.
 *
 * Scales to the budget, rounds up onto the grid, then steps the longer axis back down if rounding
 * pushed the pair over -- so the result is always legal rather than usually legal.
 */
export function snapLtxCanvas(
  width: number,
  height: number,
  maxPixels: number = LTX_MAX_PIXELS,
  minAxis: number = LTX_AXIS_MULTIPLE,
): { width: number; height: number } {
  const sourceWidth = Math.max(1, width);
  const sourceHeight = Math.max(1, height);
  const scale = Math.min(1, Math.sqrt(maxPixels / (sourceWidth * sourceHeight)));
  const floor = Math.max(LTX_AXIS_MULTIPLE, snapLtxAxis(minAxis));
  let fittedWidth = Math.max(floor, snapLtxAxis(Math.round(sourceWidth * scale)));
  let fittedHeight = Math.max(floor, snapLtxAxis(Math.round(sourceHeight * scale)));
  while (fittedWidth * fittedHeight > maxPixels) {
    if (fittedWidth >= fittedHeight && fittedWidth > floor) fittedWidth -= LTX_AXIS_MULTIPLE;
    else if (fittedHeight > floor) fittedHeight -= LTX_AXIS_MULTIPLE;
    else break;
  }
  return { width: fittedWidth, height: fittedHeight };
}

export type LtxSnapDirection = "nearest" | "up" | "down";

/**
 * Snap a frame count onto `8n + 1`.
 *
 * Nearest by default: refusing a count the user cannot reasonably be expected to compute is the
 * behaviour this module exists to end. `down` exists only for callers describing existing footage
 * that must not be exceeded.
 */
export function snapLtxFrames(numFrames: number, direction: LtxSnapDirection = "nearest"): number {
  const requested = Math.max(1, Math.round(numFrames));
  let lower = Math.floor((requested - LTX_FRAME_REMAINDER) / LTX_FRAMES_PER_LATENT)
    * LTX_FRAMES_PER_LATENT + LTX_FRAME_REMAINDER;
  if (lower < LTX_MIN_FRAMES) lower = LTX_MIN_FRAMES;
  if (lower === requested) return requested;
  const upper = lower + LTX_FRAMES_PER_LATENT;
  if (direction === "down") return lower;
  if (direction === "up") return upper;
  return (requested - lower) <= (upper - requested) ? lower : upper;
}

/** How many latent frames a count encodes to. The temporal half of the real cost. */
export function ltxLatentFrames(numFrames: number): number {
  return Math.floor((numFrames - 1) / LTX_FRAMES_PER_LATENT) + 1;
}

/** Pixels times latent frames: the metric VRAM cost actually follows, not pixels alone. */
export function ltxPixelFrames(width: number, height: number, numFrames: number): number {
  return width * height * ltxLatentFrames(numFrames);
}

export type LtxFrameRange = {
  startFrame: number;
  endFrame: number;
  startSeconds: number;
  endSeconds: number;
  frames: number;
};

/** Resolve a half-open time request to the covering half-open source-frame interval. */
export function ltxFrameRange(
  startSeconds: number,
  endSeconds: number,
  fpsNumerator: number,
  fpsDenominator: number,
  totalFrames: number = Number.MAX_SAFE_INTEGER,
): LtxFrameRange {
  const numerator = Math.max(1, Math.round(fpsNumerator));
  const denominator = Math.max(1, Math.round(fpsDenominator));
  const fps = numerator / denominator;
  const first = Math.max(0, Math.min(totalFrames, Math.floor(startSeconds * fps + 1e-9)));
  let last = Math.max(first, Math.min(totalFrames, Math.ceil(endSeconds * fps - 1e-9)));
  if (last <= first && first < totalFrames) last = first + 1;
  return {
    startFrame: first,
    endFrame: last,
    startSeconds: first * denominator / numerator,
    endSeconds: last * denominator / numerator,
    frames: last - first,
  };
}

/** The last-resort check, not the user-facing gate. Callers snap first. */
export function ltxGeometryError(
  width: number,
  height: number,
  numFrames: number,
  maxPixels: number = LTX_MAX_PIXELS,
  maxFrames: number | null = LTX_MAX_FRAMES,
): string | undefined {
  for (const [name, value] of [["Width", width], ["Height", height]] as const) {
    if (value <= 0 || value % LTX_AXIS_MULTIPLE !== 0) {
      return `${name} must be a positive multiple of ${LTX_AXIS_MULTIPLE}; got ${value}.`;
    }
  }
  if (width * height > maxPixels) {
    return `${width}x${height} is ${width * height} pixels, over the ${maxPixels} budget for this profile.`;
  }
  if (numFrames < LTX_MIN_FRAMES) return `Frame count must be at least ${LTX_MIN_FRAMES}; got ${numFrames}.`;
  if ((numFrames - LTX_FRAME_REMAINDER) % LTX_FRAMES_PER_LATENT !== 0) {
    return `Frame count must be 8n + 1; got ${numFrames}.`;
  }
  if (maxFrames !== null && numFrames > maxFrames) {
    return `${numFrames} frames is over the ${maxFrames} this 16 GB profile has been run at.`;
  }
  return undefined;
}

/**
 * Every reason a retake cannot run, mirroring
 * `ai-runtime/aiv_runtime/ltx_retake_16gb.py::plan_retake_window`.
 *
 * The worker no longer loads the source: it cuts a window around the selection, as large as the
 * profile allows, and regenerates inside that. So the clip's LENGTH is no longer a limit — a
 * 20-minute source is fine. What still binds is the canvas, which a window cannot change, and the
 * length of the selection itself, which cannot exceed one window.
 */
export function ltxRetakeProblems(
  width: number,
  height: number,
  sourceFrames: number,
  fps: number,
  selectionStartSeconds: number,
  selectionEndSeconds: number,
  maxPixels: number = LTX_MAX_PIXELS,
  maxPixelFrames: number = LTX_MAX_PIXEL_FRAMES,
): string[] {
  const problems: string[] = [];
  const pixels = width * height;
  const rate = Math.max(fps, 1e-6);

  if (sourceFrames > 0 && sourceFrames < LTX_MIN_FRAMES) {
    problems.push(`The clip is ${sourceFrames} frames; retake needs at least ${LTX_MIN_FRAMES}.`);
  }
  if (pixels > maxPixels) {
    problems.push(
      `${width}×${height} is ${pixels.toLocaleString()} pixels, over the ${maxPixels.toLocaleString()} budget — ${(pixels / maxPixels).toFixed(1)}× too large. A window makes a clip shorter, not smaller: retake needs a downscaled copy, 768×768 or less.`,
    );
  }
  const offGrid = ([["width", width], ["height", height]] as const)
    .filter(([, value]) => value % LTX_AXIS_MULTIPLE !== 0)
    .map(([name, value]) => `${name} ${value}`);
  if (offGrid.length > 0) {
    problems.push(
      `${offGrid.join(" and ")} is off the ${LTX_AXIS_MULTIPLE}-pixel grid the VAE decodes on (the nearest are ${snapLtxAxis(width)}×${snapLtxAxis(height)}). A window cannot fix this either — the canvas is the canvas.`,
    );
  }

  // The longest window this canvas can afford, which is the cap on the selection.
  const affordable = Math.min(
    LTX_MAX_FRAMES,
    Math.max(1, Math.floor(maxPixelFrames / Math.max(1, pixels)) - 1) * LTX_FRAMES_PER_LATENT + 1,
    sourceFrames > 0 ? sourceFrames : LTX_MAX_FRAMES,
  );
  const maxWindow = snapLtxFrames(affordable, "down");
  const selectionFrames = ltxFrameRange(
    selectionStartSeconds, selectionEndSeconds,
    Math.round(rate * 1_001), 1_001, sourceFrames,
  ).frames;
  if (selectionFrames > maxWindow) {
    problems.push(
      `The selected range is ${selectionFrames} frames (${(selectionFrames / rate).toFixed(1)}s), and at ${width}×${height} the most that can be retaken at once is ${maxWindow} (${(maxWindow / rate).toFixed(1)}s). Shorten the selection, or use a smaller canvas.`,
    );
  }
  return problems;
}
