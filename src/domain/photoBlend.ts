/**
 * Blend modes.
 *
 * Applied to gamma-encoded values, not to the linear light the rest of the pipeline works in. That
 * is a deliberate exception to this project's own rule and the one place it should be broken: the
 * W3C compositing spec and every tool anyone has used define these on encoded values, so blending
 * in linear would give a Multiply that is technically defensible and visibly unlike Photoshop's.
 * Matching what people expect matters more here than internal consistency.
 *
 * Only the separable modes are here. Hue, Saturation, Colour and Luminosity need all three channels
 * at once and are a different shape of function; they can come later without disturbing these.
 */

export type BlendMode =
  | "normal" | "multiply" | "screen" | "overlay" | "darken" | "lighten"
  | "colorDodge" | "colorBurn" | "hardLight" | "softLight" | "difference" | "exclusion";

export const BLEND_MODES: Array<{ id: BlendMode; name: string }> = [
  { id: "normal", name: "Normal" },
  { id: "multiply", name: "Multiply" },
  { id: "screen", name: "Screen" },
  { id: "overlay", name: "Overlay" },
  { id: "darken", name: "Darken" },
  { id: "lighten", name: "Lighten" },
  { id: "colorDodge", name: "Colour Dodge" },
  { id: "colorBurn", name: "Colour Burn" },
  { id: "hardLight", name: "Hard Light" },
  { id: "softLight", name: "Soft Light" },
  { id: "difference", name: "Difference" },
  { id: "exclusion", name: "Exclusion" },
];

/**
 * One channel of one pixel. `backdrop` and `source` are 0..1.
 *
 * Straight from the compositing spec, including its edge cases: dodge and burn have to special-case
 * a saturated backdrop or a black source, or they divide by zero and produce the white and black
 * speckle that gives away a hand-rolled implementation.
 */
export function blendChannel(mode: BlendMode, backdrop: number, source: number): number {
  switch (mode) {
    case "multiply": return backdrop * source;
    case "screen": return backdrop + source - backdrop * source;
    case "overlay": return blendChannel("hardLight", source, backdrop);
    case "darken": return Math.min(backdrop, source);
    case "lighten": return Math.max(backdrop, source);
    case "colorDodge":
      if (backdrop <= 0) return 0;
      if (source >= 1) return 1;
      return Math.min(1, backdrop / (1 - source));
    case "colorBurn":
      if (backdrop >= 1) return 1;
      if (source <= 0) return 0;
      return 1 - Math.min(1, (1 - backdrop) / source);
    case "hardLight":
      return source <= 0.5
        ? backdrop * (2 * source)
        : backdrop + (2 * source - 1) - backdrop * (2 * source - 1);
    case "softLight": {
      const d = backdrop <= 0.25
        ? ((16 * backdrop - 12) * backdrop + 4) * backdrop
        : Math.sqrt(backdrop);
      return source <= 0.5
        ? backdrop - (1 - 2 * source) * backdrop * (1 - backdrop)
        : backdrop + (2 * source - 1) * (d - backdrop);
    }
    case "difference": return Math.abs(backdrop - source);
    case "exclusion": return backdrop + source - 2 * backdrop * source;
    default: return source;
  }
}

/**
 * Composite `layer` over `base`, in place, honouring alpha, a mask, an opacity and a blend mode.
 *
 * The source-over formula with a blend function substituted for the source colour, which is what the
 * spec calls it. Written out rather than delegated to canvas `globalCompositeOperation` because the
 * renderer works on buffers, and because a mask has to multiply into the alpha before the blend --
 * something the canvas API cannot express without a second offscreen pass per layer.
 */
export function compositeOver(
  base: Uint8ClampedArray, layer: Uint8ClampedArray,
  mode: BlendMode, opacity: number, coverage?: Uint8Array,
): void {
  const strength = Math.min(1, Math.max(0, opacity));
  if (strength === 0) return;
  for (let pixel = 0, at = 0; at < base.length; at += 4, pixel++) {
    let alpha = (layer[at + 3] / 255) * strength;
    if (coverage) alpha *= coverage[pixel] / 255;
    if (alpha <= 0) continue;

    const backdropAlpha = base[at + 3] / 255;
    const outAlpha = alpha + backdropAlpha * (1 - alpha);
    if (outAlpha <= 0) { base[at + 3] = 0; continue; }

    for (let channel = 0; channel < 3; channel++) {
      const backdrop = base[at + channel] / 255;
      const source = layer[at + channel] / 255;
      // Where the backdrop is transparent there is nothing to blend against, so the blend fades out
      // with it -- otherwise Multiply over empty canvas would render black instead of the layer.
      const blended = mode === "normal" ? source
        : source + backdropAlpha * (blendChannel(mode, backdrop, source) - source);
      const value = (alpha * blended + backdropAlpha * backdrop * (1 - alpha)) / outAlpha;
      base[at + channel] = value * 255;
    }
    base[at + 3] = outAlpha * 255;
  }
}
