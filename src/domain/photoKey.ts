/**
 * Keys for third-party stock elements shot on black (VFX Phase 2): unmultiply and a luma key.
 *
 * These are the CPU references the GPU shaders (UNMULTIPLY_FS, LUMA_KEY_FS in src/render/shaders.ts)
 * are held to, so the arithmetic and the points where it rounds are written out rather than left to
 * a library. Both work on gamma-ENCODED values, in place, on straight-alpha RGBA bytes.
 *
 * Wan Alpha elements carry real alpha and need neither.
 */

import type { LumaKeyNode } from "./photoDocument";

/**
 * Unmultiply: alpha becomes max(r, g, b) (times any alpha the pixel already had), and the colour is
 * divided by that maximum so the brightest channel reaches 255.
 *
 * Encoded rather than linear, because the thing it has to agree with is this app's Screen blend,
 * which is defined on encoded values (photoBlend.ts): composited normally, c/m at alpha m gives
 * c + B - m*B over a backdrop B, which is exactly Screen in the brightest channel and close in the
 * others. Unmultiplying in linear would match additive light instead, and look unlike the Screen
 * people already use to drop fire on black over a plate. Pure black becomes fully transparent.
 */
export function applyUnmultiply(pixels: Uint8ClampedArray): void {
  for (let at = 0; at < pixels.length; at += 4) {
    const m = Math.max(pixels[at], pixels[at + 1], pixels[at + 2]);
    if (m === 0) {
      pixels[at] = pixels[at + 1] = pixels[at + 2] = pixels[at + 3] = 0;
      continue;
    }
    // (c * 255) / m, in that order: the product is an exact integer, so the quotient is correctly
    // rounded in float32 and float64 alike and the GPU lands on the same byte (ties included).
    pixels[at] = pixels[at] * 255 / m;
    pixels[at + 1] = pixels[at + 1] * 255 / m;
    pixels[at + 2] = pixels[at + 2] * 255 / m;
    pixels[at + 3] = pixels[at + 3] * m / 255;
  }
}

/** The key's multiplier for one luma value, 0..1. Shared by the node and its tests. */
export function lumaKeyAmount(luma: number, node: Pick<LumaKeyNode, "low" | "high" | "softness" | "invert">): number {
  const low = node.low / 100 * 255, high = node.high / 100 * 255;
  const soft = Math.max(0, node.softness) / 100 * 255;
  let keep: number;
  if (soft <= 0) keep = luma >= low && luma <= high ? 1 : 0;
  else {
    const rise = Math.min(1, Math.max(0, (luma - (low - soft)) / soft));
    const fall = Math.min(1, Math.max(0, ((high + soft) - luma) / soft));
    keep = Math.min(rise, fall);
  }
  return node.invert ? 1 - keep : keep;
}

/** Luma key: alpha times how far the pixel's encoded Rec. 709 luma sits inside [low, high]. */
export function applyLumaKey(pixels: Uint8ClampedArray, node: Pick<LumaKeyNode, "low" | "high" | "softness" | "invert">): void {
  for (let at = 0; at < pixels.length; at += 4) {
    const luma = 0.2126 * pixels[at] + 0.7152 * pixels[at + 1] + 0.0722 * pixels[at + 2];
    pixels[at + 3] = pixels[at + 3] * lumaKeyAmount(luma, node);
  }
}
