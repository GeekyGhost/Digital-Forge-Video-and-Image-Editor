/**
 * Making a cutout belong to the plate it was dropped onto.
 *
 * This is the group that separates a pasted subject from a composited one, and none of it needs a
 * model. A BiRefNet cutout arrives with a rim of whatever it used to be standing in front of, no
 * relationship to the light in its new home, and nothing under it — three tells, three fixes.
 */

import { blurBuffer } from "./photoEdit";

export type LayerEdge = {
  /** Pixels of contaminated rim to eat and repaint from inside the subject. */
  defringe: number;
  /** How much of the backdrop's light spills around the edge, 0-100. */
  lightWrap: number;
  /** How far in from the edge that spill reaches, in pixels. */
  lightWrapSpread: number;
  /** Opacity of the shadow cast under the subject, 0-100. */
  contactShadow: number;
  contactShadowSoftness: number;
  /** Pushes the shadow down the frame, in fractions of the layer's height. */
  contactShadowOffset: number;
  /** Squashes it, so it reads as lying on a surface rather than standing behind. */
  contactShadowSquash: number;
};

export const NEUTRAL_EDGE: LayerEdge = {
  defringe: 0,
  lightWrap: 0, lightWrapSpread: 12,
  contactShadow: 0, contactShadowSoftness: 40, contactShadowOffset: 0.02, contactShadowSquash: 0.18,
};

export const isNeutralEdge = (edge: LayerEdge | undefined): boolean =>
  !edge || (edge.defringe === 0 && edge.lightWrap === 0 && edge.contactShadow === 0);

/**
 * Eat the contaminated rim and repaint it from the colours just inside.
 *
 * A cutout's edge pixels are a blend of the subject and whatever was behind it, so a subject lifted
 * off grass keeps a green fringe that survives every adjustment downstream and is the loudest tell
 * in a composite. Eroding alone leaves a hard, thin edge; the repaint is what makes it look cut
 * rather than trimmed.
 *
 * The repaint spreads inside colour outward by blurring the RGB weighted by the ERODED alpha, so
 * only pixels that were safely inside contribute. Blurring the raw RGB would smear the very
 * contamination this is removing back over the edge.
 */
export function defringeLayer(
  data: Uint8ClampedArray, width: number, height: number, pixels: number,
): void {
  const amount = Math.round(pixels);
  if (amount < 1) return;
  const count = width * height;

  const inside = new Uint8Array(count);
  for (let index = 0, at = 3; index < count; index++, at += 4) inside[index] = data[at];
  erode(inside, width, height, amount);

  // Premultiply by the eroded alpha, blur both, divide back out: colour spread from the interior
  // only, with no contribution from the rim being replaced.
  const spread = Math.max(2, amount * 2);
  const weighted = new Uint8ClampedArray(data.length);
  for (let index = 0, at = 0; index < count; index++, at += 4) {
    const weight = inside[index] / 255;
    weighted[at] = data[at] * weight;
    weighted[at + 1] = data[at + 1] * weight;
    weighted[at + 2] = data[at + 2] * weight;
    weighted[at + 3] = inside[index];
  }
  blurBuffer(weighted, width, height, spread);
  const carrier = new Uint8ClampedArray(count * 4);
  for (let index = 0, at = 0; index < count; index++, at += 4) {
    carrier[at + 3] = inside[index];
  }
  blurBuffer(carrier, width, height, spread);

  for (let index = 0, at = 0; index < count; index++, at += 4) {
    const original = data[at + 3];
    if (original === 0) continue;
    const kept = inside[index];
    if (kept >= original) continue;
    const weight = carrier[at + 3];
    if (weight > 4) {
      // How much of this pixel was rim decides how far it moves toward the interior colour.
      const pull = 1 - kept / original;
      const scale = 255 / weight;
      data[at] += (weighted[at] * scale - data[at]) * pull;
      data[at + 1] += (weighted[at + 1] * scale - data[at + 1]) * pull;
      data[at + 2] += (weighted[at + 2] * scale - data[at + 2]) * pull;
    }
    data[at + 3] = kept;
  }
}

/**
 * Let the backdrop's light spill around the inside of the subject's edge.
 *
 * Real subjects catch light from behind them. Ten lines of blurred backdrop added inside the rim
 * does more for "does this belong here" than any blend mode, which is why it sits next to defringe
 * rather than in the effects list.
 */
export function applyLightWrap(
  data: Uint8ClampedArray, backdrop: Uint8ClampedArray,
  width: number, height: number, amount: number, spreadPixels: number,
): void {
  const strength = Math.min(1, Math.max(0, amount / 100));
  if (strength <= 0) return;
  const spread = Math.max(1, Math.round(spreadPixels));
  const count = width * height;

  const soft = new Uint8ClampedArray(backdrop);
  blurBuffer(soft, width, height, spread);

  // The band is what the alpha loses when eroded: the inside of the edge, and nothing else.
  const band = new Uint8Array(count);
  for (let index = 0, at = 3; index < count; index++, at += 4) band[index] = data[at];
  const eroded = band.slice();
  erode(eroded, width, height, spread);

  for (let index = 0, at = 0; index < count; index++, at += 4) {
    const rim = band[index] - eroded[index];
    if (rim <= 0) continue;
    const weight = (rim / 255) * strength;
    // Screen, so the wrap adds light and never darkens the subject it is meant to lift.
    for (let channel = 0; channel < 3; channel++) {
      const base = data[at + channel] / 255;
      const light = soft[at + channel] / 255;
      const screened = base + light - base * light;
      data[at + channel] = (base + (screened - base) * weight) * 255;
    }
  }
}

/**
 * A shadow for the subject to stand on, derived from its own alpha.
 *
 * Returns a separate buffer rather than drawing into anything: the shadow belongs UNDER the layer,
 * so the compositor has to place it before the subject, and handing back a buffer keeps that
 * ordering the caller's decision rather than a hidden side effect.
 */
export function buildContactShadow(
  alphaSource: Uint8ClampedArray, width: number, height: number, edge: LayerEdge,
): Uint8ClampedArray | undefined {
  const opacity = Math.min(1, Math.max(0, edge.contactShadow / 100));
  if (opacity <= 0) return undefined;
  const shadow = new Uint8ClampedArray(width * height * 4);
  const squash = Math.min(0.95, Math.max(0.02, edge.contactShadowSquash));
  const offset = edge.contactShadowOffset * height;

  // Find where the subject sits, so the shadow is anchored to its feet rather than to the frame.
  let lowest = 0;
  for (let index = 0, at = 3; index < width * height; index++, at += 4) {
    if (alphaSource[at] > 8) lowest = Math.max(lowest, (index / width) | 0);
  }
  const anchor = lowest + offset;

  for (let y = 0; y < height; y++) {
    // Project each output row back up into the subject: rows near the anchor read from far up it,
    // which is what flattens the silhouette into something lying on the ground.
    const sourceY = Math.round(anchor + (y - anchor) / squash);
    if (sourceY < 0 || sourceY >= height) continue;
    for (let x = 0; x < width; x++) {
      const alpha = alphaSource[(sourceY * width + x) * 4 + 3];
      if (alpha === 0) continue;
      shadow[(y * width + x) * 4 + 3] = alpha;
    }
  }
  blurBuffer(shadow, width, height, Math.max(1, edge.contactShadowSoftness / 100 * 40));
  for (let at = 3; at < shadow.length; at += 4) shadow[at] *= opacity;
  return shadow;
}

/** Minimum filter over an 8-bit buffer, which is what shrinks a matte by `radius` pixels. */
export function erode(alpha: Uint8Array, width: number, height: number, radius: number): void {
  if (radius < 1) return;
  const scratch = new Uint8Array(alpha.length);
  minimumPass(alpha, scratch, width, height, radius);
  minimumPass(scratch, alpha, height, width, radius);
}

/** One axis, written transposed; twice returns to the original orientation. */
function minimumPass(
  source: Uint8Array, target: Uint8Array, width: number, height: number, radius: number,
): void {
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let lowest = 255;
      const from = Math.max(0, x - radius), to = Math.min(width - 1, x + radius);
      for (let at = from; at <= to; at++) {
        const value = source[row + at];
        if (value < lowest) lowest = value;
      }
      target[x * height + y] = lowest;
    }
  }
}

/**
 * The grade that would make `layer` sit in the same tonal range as `backdrop`.
 *
 * Means and spreads per channel, solved into the exposure, temperature, tint and contrast the
 * pipeline already has. This is the one that most directly replaces a round trip through Qwen: it
 * is a closed-form solve, not a generation, and it lands as ordinary slider values the user can then
 * disagree with.
 */
export function matchColour(
  layer: Uint8ClampedArray, backdrop: Uint8ClampedArray, coverage?: Uint8Array,
): { exposure: number; temperature: number; tint: number; contrast: number } | undefined {
  const front = statistics(layer, coverage, true);
  const back = statistics(backdrop, coverage, false);
  if (front.count < 32 || back.count < 32) return undefined;

  const exposure = Math.log2(Math.max(0.05, Math.min(20, back.luma / Math.max(1, front.luma))));
  // The pipeline warms by scaling red up and blue down, so the imbalance between the two ratios is
  // what temperature has to undo.
  const redRatio = back.r / Math.max(1, front.r);
  const blueRatio = back.b / Math.max(1, front.b);
  const greenRatio = back.g / Math.max(1, front.g);
  const temperature = clamp((redRatio - blueRatio) / 0.32 * 100, -100, 100);
  const tint = clamp(-(greenRatio - (redRatio + blueRatio) / 2) / 0.18 * 100, -100, 100);
  const contrast = clamp((back.spread / Math.max(1, front.spread) - 1) * 100, -80, 80);
  return {
    exposure: Number(clamp(exposure, -3, 3).toFixed(2)),
    temperature: Math.round(temperature),
    tint: Math.round(tint),
    contrast: Math.round(contrast),
  };
}

function statistics(data: Uint8ClampedArray, coverage: Uint8Array | undefined, insideOnly: boolean) {
  let r = 0, g = 0, b = 0, luma = 0, lumaSquared = 0, count = 0;
  for (let pixel = 0, at = 0; at < data.length; at += 4, pixel++) {
    // The subject is measured only where it exists; the plate is measured only where the subject
    // will land. Comparing a cutout's transparent margin against a full frame would skew both.
    if (insideOnly && data[at + 3] < 128) continue;
    if (coverage && coverage[pixel] < 128) continue;
    const value = data[at] * 0.2126 + data[at + 1] * 0.7152 + data[at + 2] * 0.0722;
    r += data[at]; g += data[at + 1]; b += data[at + 2];
    luma += value; lumaSquared += value * value;
    count++;
  }
  if (!count) return { r: 0, g: 0, b: 0, luma: 0, spread: 0, count: 0 };
  const mean = luma / count;
  return {
    r: r / count, g: g / count, b: b / count, luma: mean,
    spread: Math.sqrt(Math.max(0, lumaSquared / count - mean * mean)),
    count,
  };
}

const clamp = (value: number, low: number, high: number) => value < low ? low : value > high ? high : value;
