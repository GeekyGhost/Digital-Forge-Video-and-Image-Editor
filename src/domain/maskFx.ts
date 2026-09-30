/**
 * VFX Phase 7: the matte-driven mask VFX (`ai-runtime/aiv_runtime/mask_effects.py`), ported to the
 * compositor.
 *
 * One function renders one frame of a `MaskVfxRecipeV1` onto a layer's working pixels, from the
 * subject matte placed the way the layer is placed. It is pure JavaScript -- no canvas -- so the CPU
 * compositor and the GPU compositor (which runs it on its CPU path, like the envelope and liquify)
 * produce the same bytes, and the unit tests run it in Node.
 *
 * The old renderer worked on the clip's own frames at source (or preview) resolution. Here the
 * frame is wherever the layer landed: `toWork` maps source pixels to working pixels, so positions
 * (the effect anchor, the transition origin, the gradient, the imported effect clip) are measured in
 * the source frame exactly as before, and every radius the old code gave in pixels is scaled by the
 * layer's linear scale `k`. The passes, their order and their parameters are the old ones:
 *
 *   selected matte (feather/expand/invert) -> animated matte style or keyed effect clip, moved by
 *   the (keyframed, pulsing) transform -> inside/displacement/motion blur -> the effect at an
 *   (audio-reactive) strength -> effect opacity -> the effect clip or gradient blended in -> grain
 *   -> transition in/out reveal and energy overlay -> pass 1 behind the clean subject, pass 2 light
 *   wrap and spill on its edge, pass 3 a foreground copy through the matte.
 *
 * What is NOT byte-identical to the PIL renderer: PIL quantised to 8 bits after every step and used
 * numpy's random generators; this keeps floats between steps and uses a seeded hash, so sparks,
 * bolts and particles land in different (equally random, equally stable) places. Shapes are
 * anti-aliased where PIL's were not.
 */

import { compose, invert, rotateBy, scaleBy, translate, applyMatrix, type Matrix } from "./photoGeometry";
import { boxesForGauss, hash01, gaussPlanes } from "./photoClipEffects";
import { buildGradientLut, gradientPosition, rampParameter, BAYER, hexToRgb, type Gradient } from "./photoGradient";
import type { MaskVfxRecipeV1, MaskEffectTransition, MaskVfxBlendMode, MaskVfxAnimatedValues } from "./maskVfx";

export type Rgb = [number, number, number];

export type MaskFxInput = {
  /** The layer's working pixels, straight RGBA (not modified). */
  pixels: Uint8ClampedArray;
  width: number;
  height: number;
  recipe: MaskVfxRecipeV1;
  /** This frame's index in the layer's range, the range's length in frames, and its rate. */
  index: number;
  count: number;
  fps: number;
  /** The audio envelope at this frame, 0..1 (0 when the recipe is not audio-reactive). */
  envelope: number;
  /** The (stabilised) subject matte in working pixels, 0..255. */
  matte: Uint8Array;
  /** Source pixels -> working pixels (the layer's placement). */
  toWork: Matrix;
  sourceWidth: number;
  sourceHeight: number;
  /** The imported effect clip's frame for the "clip" matte style, straight RGBA at its own size. */
  effect?: { data: Uint8ClampedArray; width: number; height: number };
};

type Plane = Float32Array;

const TAU = Math.PI * 2;
const clamp = (value: number, low: number, high: number) => value < low ? low : value > high ? high : value;
const clamp01 = (value: number) => clamp(value, 0, 1);
const odd = (value: number) => (value | 0) % 2 === 0 ? (value | 0) + 1 : value | 0;

/* ================================================================ planes */

const plane = (width: number, height: number): Plane => new Float32Array(width * height);

/** The bounding box of the values above zero, or undefined when there are none. */
function nonzeroBox(p: Plane, width: number, height: number): [number, number, number, number] | undefined {
  let left = width, top = height, right = -1, bottom = -1;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let first = -1, lastX = -1;
    for (let x = 0; x < width; x++) if (p[row + x] > 0) { if (first < 0) first = x; lastX = x; }
    if (first < 0) continue;
    if (first < left) left = first;
    if (lastX > right) right = lastX;
    if (y < top) top = y;
    bottom = y;
  }
  return right < 0 ? undefined : [left, top, right + 1, bottom + 1];
}

/**
 * Run `fn` on the part of `source` within `margin` of its non-zero values only, copying the result
 * back into a zero frame. Exact for operations that keep zero at zero and reach at most `margin`
 * pixels (a dilation, an erosion, a blur whose boxes add up to less), because the crop's clamped edge
 * then only ever repeats zeros the full frame would have had there too.
 */
function onSupport(source: Plane, width: number, height: number, margin: number, fn: (sub: Plane, w: number, h: number) => Plane): Plane {
  const box = nonzeroBox(source, width, height);
  if (!box) return plane(width, height);
  const x0 = Math.max(0, box[0] - margin), y0 = Math.max(0, box[1] - margin);
  const x1 = Math.min(width, box[2] + margin), y1 = Math.min(height, box[3] + margin);
  if (x0 === 0 && y0 === 0 && x1 === width && y1 === height) return fn(new Float32Array(source), width, height);
  const w = x1 - x0, h = y1 - y0;
  const sub = plane(w, h);
  for (let y = 0; y < h; y++) sub.set(source.subarray((y + y0) * width + x0, (y + y0) * width + x1), y * w);
  const done = fn(sub, w, h);
  const out = plane(width, height);
  for (let y = 0; y < h; y++) out.set(done.subarray(y * w, (y + 1) * w), (y + y0) * width + x0);
  return out;
}

/** Rolling max over a (2r+1) window along rows then columns (van Herk / Gil-Werman), in place. */
function rollingMax(values: Plane, width: number, height: number, r: number): void {
  const w = 2 * r + 1;
  const pass = (lines: number, length: number, stride: number, step: number) => {
    const padded = length + 2 * r;
    const total = Math.ceil((padded) / w) * w + w;
    const line = new Float32Array(total), g = new Float32Array(total), h = new Float32Array(total);
    for (let l = 0; l < lines; l++) {
      const base = l * stride;
      line.fill(-Infinity);
      for (let i = 0; i < length; i++) line[i + r] = values[base + i * step];
      for (let start = 0; start < total; start += w) {
        const end = start + w;
        let running = line[start];
        g[start] = running;
        for (let i = start + 1; i < end; i++) { const v = line[i]; if (v > running) running = v; g[i] = running; }
        running = line[end - 1];
        h[end - 1] = running;
        for (let i = end - 2; i >= start; i--) { const v = line[i]; if (v > running) running = v; h[i] = running; }
      }
      for (let i = 0; i < length; i++) { const a = h[i], b = g[i + 2 * r]; values[base + i * step] = a > b ? a : b; }
    }
  };
  pass(height, width, width, 1);
  pass(width, height, 1, width);
}

/** Rolling max (dilation) or min (erosion) over a square (2r+1) window. */
function extreme(source: Plane, width: number, height: number, r: number, max: boolean): Plane {
  if (r < 1) return new Float32Array(source);
  if (max) {
    // Zero stays zero more than r pixels from the coverage, so only its neighbourhood is worked.
    return onSupport(source, width, height, r, (sub, w, h) => { rollingMax(sub, w, h, r); return sub; });
  }
  // An erosion is a dilation of the negative. The margin keeps the zeros around the coverage inside
  // the crop, so the coverage's edge still erodes against them.
  return onSupport(source, width, height, r, (sub, w, h) => {
    for (let i = 0; i < sub.length; i++) sub[i] = -sub[i];
    rollingMax(sub, w, h, r);
    for (let i = 0; i < sub.length; i++) sub[i] = -sub[i] + 0;
    return sub;
  });
}

/** PIL's MaxFilter/MinFilter(size) with `size` in source pixels (a square window). */
const maxFilter = (p: Plane, w: number, h: number, size: number, k: number) => extreme(p, w, h, Math.round(((size - 1) / 2) * k), true);
const minFilter = (p: Plane, w: number, h: number, size: number, k: number) => extreme(p, w, h, Math.round(((size - 1) / 2) * k), false);

/** Gaussian blur of planes worked at 1/d size: area-averaged down, blurred, bilinear back up. */
function gaussReduced(planes: Plane[], w: number, h: number, sigma: number): void {
  // A wide blur is smooth at a fraction of the size anyway, and the cost falls with the square.
  const d = Math.min(4, Math.floor(sigma / 3));
  if (d < 2 || w < d * 4 || h < d * 4) { gaussPlanes(planes, w, h, sigma); return; }
  const sw = Math.ceil(w / d), sh = Math.ceil(h / d);
  const counts = plane(sw, sh);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) counts[Math.floor(y / d) * sw + Math.floor(x / d)] += 1;
  const smalls = planes.map((sub) => {
    const small = plane(sw, sh);
    for (let y = 0; y < h; y++) {
      const row = Math.floor(y / d) * sw;
      for (let x = 0; x < w; x++) small[row + Math.floor(x / d)] += sub[y * w + x];
    }
    for (let i = 0; i < small.length; i++) small[i] /= counts[i];
    return small;
  });
  gaussPlanes(smalls, sw, sh, sigma / d);
  for (let y = 0; y < h; y++) {
    const fy = Math.min(sh - 1, Math.max(0, (y + 0.5) / d - 0.5));
    const y0 = Math.floor(fy), y1 = Math.min(sh - 1, y0 + 1), ty = fy - y0;
    for (let x = 0; x < w; x++) {
      const fx = Math.min(sw - 1, Math.max(0, (x + 0.5) / d - 0.5));
      const x0 = Math.floor(fx), x1 = Math.min(sw - 1, x0 + 1), tx = fx - x0;
      const a = y0 * sw + x0, b = y0 * sw + x1, c = y1 * sw + x0, e = y1 * sw + x1;
      for (let p = 0; p < planes.length; p++) {
        const small = smalls[p];
        planes[p][y * w + x] = (small[a] * (1 - tx) + small[b] * tx) * (1 - ty) + (small[c] * (1 - tx) + small[e] * tx) * ty;
      }
    }
  }
}

/**
 * Blur planes that are zero wherever `support` is (a matte and anything premultiplied by it), only
 * around the support: the three boxes reach the sum of their radii, and zero stays zero beyond.
 */
function blurOnSupport(planes: Plane[], support: Plane, width: number, height: number, sigma: number): Plane[] {
  if (sigma <= 0.05) return planes.map((p) => new Float32Array(p));
  const box = nonzeroBox(support, width, height);
  if (!box) return planes.map(() => plane(width, height));
  const reach = boxesForGauss(sigma).reduce((sum, radius) => sum + Math.floor(radius), 0) + 1;
  const x0 = Math.max(0, box[0] - reach), y0 = Math.max(0, box[1] - reach);
  const x1 = Math.min(width, box[2] + reach), y1 = Math.min(height, box[3] + reach);
  const w = x1 - x0, h = y1 - y0;
  const subs = planes.map((source) => {
    const sub = plane(w, h);
    for (let y = 0; y < h; y++) sub.set(source.subarray((y + y0) * width + x0, (y + y0) * width + x1), y * w);
    return sub;
  });
  gaussReduced(subs, w, h, sigma);
  return subs.map((sub) => {
    const out = plane(width, height);
    for (let y = 0; y < h; y++) out.set(sub.subarray(y * w, (y + 1) * w), (y + y0) * width + x0);
    return out;
  });
}

/** PIL's GaussianBlur(radius) (radius is the standard deviation), in working pixels. */
function gauss(source: Plane, width: number, height: number, sigma: number): Plane {
  return blurOnSupport([source], source, width, height, sigma)[0];
}

const subtract = (a: Plane, b: Plane): Plane => { const out = new Float32Array(a.length); for (let i = 0; i < a.length; i++) out[i] = Math.max(0, a[i] - b[i]); return out; };
const multiply = (a: Plane, b: Plane): Plane => { const out = new Float32Array(a.length); for (let i = 0; i < a.length; i++) out[i] = a[i] * b[i] / 255; return out; };
const scaled = (a: Plane, amount: number): Plane => { const out = new Float32Array(a.length); for (let i = 0; i < a.length; i++) out[i] = clamp(a[i] * amount, 0, 255); return out; };

/** `_edge`: dilation minus erosion with an odd window of 3..63 (source pixels). */
function edgeOf(mask: Plane, width: number, height: number, radius: number, k: number): Plane {
  const size = clamp(odd(radius), 3, 63);
  return subtract(maxFilter(mask, width, height, size, k), minFilter(mask, width, height, size, k));
}

/** `_mask_bounds`: the bounding box of the non-zero coverage, or the middle third. */
export function maskBounds(mask: Plane | Uint8Array, width: number, height: number): [number, number, number, number] {
  let left = width, top = height, right = -1, bottom = -1;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      if (mask[row + x] >= 0.5) {
        if (x < left) left = x;
        if (x > right) right = x;
        if (y < top) top = y;
        bottom = y;
      }
    }
  }
  if (right < 0) return [Math.floor(width / 3), Math.floor(height / 3), Math.floor((width * 2) / 3), Math.floor((height * 2) / 3)];
  return [left, top, right + 1, bottom + 1];
}

/** Sample a plane through `inverse` (working -> source-of-plane) with bilinear filtering; 0 outside. */
function warpPlane(source: Plane, width: number, height: number, inverse: Matrix): Plane {
  const out = plane(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const sx = inverse.a * (x + 0.5) + inverse.c * (y + 0.5) + inverse.e - 0.5;
      const sy = inverse.b * (x + 0.5) + inverse.d * (y + 0.5) + inverse.f - 0.5;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      if (x0 < -1 || y0 < -1 || x0 >= width || y0 >= height) continue;
      const fx = sx - x0, fy = sy - y0;
      const at = (xx: number, yy: number) => xx < 0 || yy < 0 || xx >= width || yy >= height ? 0 : source[yy * width + xx];
      out[y * width + x] = (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy;
    }
  }
  return out;
}

/** RGBA (premultiplied floats) of a picture sampled through `inverse` (working -> picture pixels). */
function warpRgba(data: Uint8ClampedArray, sw: number, sh: number, width: number, height: number, inverse: Matrix): Float32Array {
  const out = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const sx = inverse.a * (x + 0.5) + inverse.c * (y + 0.5) + inverse.e - 0.5;
      const sy = inverse.b * (x + 0.5) + inverse.d * (y + 0.5) + inverse.f - 0.5;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      if (x0 < -1 || y0 < -1 || x0 >= sw || y0 >= sh) continue;
      const fx = sx - x0, fy = sy - y0;
      const o = (y * width + x) * 4;
      for (let corner = 0; corner < 4; corner++) {
        const xx = x0 + (corner & 1), yy = y0 + (corner >> 1);
        const weight = (corner & 1 ? fx : 1 - fx) * (corner >> 1 ? fy : 1 - fy);
        if (xx < 0 || yy < 0 || xx >= sw || yy >= sh || weight <= 0) continue;
        const i = (yy * sw + xx) * 4;
        const a = data[i + 3] / 255;
        out[o] += data[i] * a * weight;
        out[o + 1] += data[i + 1] * a * weight;
        out[o + 2] += data[i + 2] * a * weight;
        out[o + 3] += a * weight;
      }
    }
  }
  // Back to straight colour, alpha 0..255.
  for (let o = 0; o < out.length; o += 4) {
    const a = out[o + 3];
    if (a > 1e-6) { out[o] /= a; out[o + 1] /= a; out[o + 2] /= a; }
    out[o + 3] = a * 255;
  }
  return out;
}

/* ============================================================ RGB images */

/** An opaque working picture: interleaved RGB floats 0..255. */
type Rgb3 = Float32Array;

/** A layer to put over it: straight RGB floats and an alpha plane 0..255. */
type Layer = { rgb: Float32Array; alpha: Plane };

const colourLayer = (count: number, color: Rgb, alpha: Plane): Layer => {
  const rgb = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) { rgb[i * 3] = color[0]; rgb[i * 3 + 1] = color[1]; rgb[i * 3 + 2] = color[2]; }
  return { rgb, alpha };
};

/** `Image.alpha_composite` of a layer over an opaque picture (and `Image.composite(layer, base, alpha)`). */
function over(base: Rgb3, layer: Layer): Rgb3 {
  const out = new Float32Array(base);
  for (let i = 0; i < layer.alpha.length; i++) {
    const a = layer.alpha[i] / 255;
    if (a <= 0) continue;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = base[i * 3 + c] * (1 - a) + layer.rgb[i * 3 + c] * a;
  }
  return out;
}

/** `_screen`: screen the layer's colour in, through its alpha. */
function screen(base: Rgb3, layer: Layer): Rgb3 {
  const out = new Float32Array(base);
  for (let i = 0; i < layer.alpha.length; i++) {
    const a = layer.alpha[i] / 255;
    if (a <= 0) continue;
    for (let c = 0; c < 3; c++) {
      const b = base[i * 3 + c], l = layer.rgb[i * 3 + c];
      const s = 255 - ((255 - b) * (255 - l)) / 255;
      out[i * 3 + c] = b * (1 - a) + s * a;
    }
  }
  return out;
}

/** `_blend`: editor-style alpha with a standard RGB blend mode. */
function blendLayer(base: Rgb3, layer: Layer, mode: MaskVfxBlendMode): Rgb3 {
  if (mode === "normal") return over(base, layer);
  const out = new Float32Array(base);
  for (let i = 0; i < layer.alpha.length; i++) {
    const a = layer.alpha[i] / 255;
    if (a <= 0) continue;
    for (let c = 0; c < 3; c++) {
      const f = base[i * 3 + c] / 255, s = layer.rgb[i * 3 + c] / 255;
      let mixed: number;
      if (mode === "screen") mixed = 1 - (1 - f) * (1 - s);
      else if (mode === "add") mixed = Math.min(1, f + s);
      else if (mode === "multiply") mixed = f * s;
      else if (mode === "overlay") mixed = f <= 0.5 ? 2 * f * s : 1 - 2 * (1 - f) * (1 - s);
      else mixed = (1 - 2 * s) * f * f + 2 * s * f;
      out[i * 3 + c] = (f * (1 - a) + clamp01(mixed) * a) * 255;
    }
  }
  return out;
}

/** `Image.composite(a, b, mask)`: a where the mask is, b elsewhere. */
function through(a: Rgb3, b: Rgb3, mask: Plane): Rgb3 {
  const out = new Float32Array(b);
  for (let i = 0; i < mask.length; i++) {
    const m = mask[i] / 255;
    if (m <= 0) continue;
    for (let c = 0; c < 3; c++) out[i * 3 + c] = a[i * 3 + c] * m + b[i * 3 + c] * (1 - m);
  }
  return out;
}

/** `Image.blend(a, b, amount)`. */
function mix(a: Rgb3, b: Rgb3, amount: number): Rgb3 {
  if (amount >= 1) return new Float32Array(b);
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] + (b[i] - a[i]) * amount;
  return out;
}

/** ITU-R 601 luma (PIL's `convert("L")`), 0..255. */
function luma601(base: Rgb3, count: number): Plane {
  const out = plane(count, 1);
  for (let i = 0; i < count; i++) out[i] = base[i * 3] * 0.299 + base[i * 3 + 1] * 0.587 + base[i * 3 + 2] * 0.114;
  return out;
}

/** An RGB picture blurred (PIL GaussianBlur on RGB). */
function blurRgb(base: Rgb3, width: number, height: number, sigma: number): Rgb3 {
  const count = width * height;
  const planes = [plane(width, height), plane(width, height), plane(width, height)];
  for (let i = 0; i < count; i++) for (let c = 0; c < 3; c++) planes[c][i] = base[i * 3 + c];
  gaussPlanes(planes, width, height, sigma);
  const out = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) for (let c = 0; c < 3; c++) out[i * 3 + c] = planes[c][i];
  return out;
}

/** A drawn RGBA layer (straight) blurred, alpha-weighted. */
function blurLayer(layer: Layer, width: number, height: number, sigma: number): Layer {
  const count = width * height;
  const premultiplied = [plane(width, height), plane(width, height), plane(width, height), layer.alpha];
  for (let i = 0; i < count; i++) for (let c = 0; c < 3; c++) premultiplied[c][i] = layer.rgb[i * 3 + c] * layer.alpha[i];
  const planes = blurOnSupport(premultiplied, layer.alpha, width, height, sigma);
  const rgb = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const a = planes[3][i];
    if (a > 1e-4) for (let c = 0; c < 3; c++) rgb[i * 3 + c] = planes[c][i] / a;
  }
  return { rgb, alpha: planes[3] };
}

/** `Image.alpha_composite(bottom, top)` of two straight layers. */
function layerOver(bottom: Layer, top: Layer): Layer {
  const count = top.alpha.length;
  const rgb = new Float32Array(count * 3), alpha = plane(count, 1);
  for (let i = 0; i < count; i++) {
    const at = top.alpha[i] / 255, ab = bottom.alpha[i] / 255;
    const a = at + ab * (1 - at);
    alpha[i] = a * 255;
    if (a <= 1e-6) continue;
    for (let c = 0; c < 3; c++) rgb[i * 3 + c] = (top.rgb[i * 3 + c] * at + bottom.rgb[i * 3 + c] * ab * (1 - at)) / a;
  }
  return { rgb, alpha };
}

/* ======================================================= drawing (pure) */

/** A drawing surface of straight RGBA floats; shapes are drawn with anti-aliased coverage. */
class Surface {
  readonly rgb: Float32Array;
  readonly alpha: Plane;
  constructor(readonly width: number, readonly height: number) {
    this.rgb = new Float32Array(width * height * 3);
    this.alpha = plane(width, height);
  }
  /** Paint `color` at `opacity` (0..255) where `coverage(x, y)` (0..1), over what is there. */
  paint(box: [number, number, number, number], color: Rgb, opacity: number, coverage: (x: number, y: number) => number): void {
    const x0 = Math.max(0, Math.floor(box[0])), y0 = Math.max(0, Math.floor(box[1]));
    const x1 = Math.min(this.width - 1, Math.ceil(box[2])), y1 = Math.min(this.height - 1, Math.ceil(box[3]));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const c = coverage(x + 0.5, y + 0.5);
        if (c <= 0) continue;
        const i = y * this.width + x;
        const at = (opacity / 255) * Math.min(1, c), ab = this.alpha[i] / 255;
        const a = at + ab * (1 - at);
        for (let k = 0; k < 3; k++) this.rgb[i * 3 + k] = (color[k] * at + this.rgb[i * 3 + k] * ab * (1 - at)) / a;
        this.alpha[i] = a * 255;
      }
    }
  }
  layer(): Layer { return { rgb: this.rgb, alpha: this.alpha }; }
}

/** Distance from p to the segment a-b. */
function segmentDistance(px: number, py: number, ax: number, ay: number, bx: number, by: number): number {
  const dx = bx - ax, dy = by - ay;
  const length = dx * dx + dy * dy;
  const t = length > 0 ? clamp(((px - ax) * dx + (py - ay) * dy) / length, 0, 1) : 0;
  return Math.hypot(px - ax - t * dx, py - ay - t * dy);
}

function drawLine(surface: Surface, points: Array<[number, number]>, color: Rgb, opacity: number, width: number): void {
  const half = Math.max(0.5, width / 2);
  for (let i = 0; i + 1 < points.length; i++) {
    const [ax, ay] = points[i], [bx, by] = points[i + 1];
    surface.paint([Math.min(ax, bx) - half - 1, Math.min(ay, by) - half - 1, Math.max(ax, bx) + half + 1, Math.max(ay, by) + half + 1],
      color, opacity, (x, y) => clamp01(half + 0.5 - segmentDistance(x, y, ax, ay, bx, by)));
  }
}

/** An ellipse outline `width` thick, inside the box (PIL `ellipse(outline, width)`). */
function drawEllipseOutline(surface: Surface, cx: number, cy: number, rx: number, ry: number, color: Rgb, opacity: number, width: number): void {
  if (rx <= 0 || ry <= 0) return;
  surface.paint([cx - rx - 1, cy - ry - 1, cx + rx + 1, cy + ry + 1], color, opacity, (x, y) => {
    // Approximate signed distance to the ellipse (gradient-normalised implicit function).
    const nx = (x - cx) / rx, ny = (y - cy) / ry;
    const f = Math.sqrt(nx * nx + ny * ny);
    // Well inside the ring (most of a big one's box): nothing to draw.
    if (f < 1 - (width + 2) / Math.min(rx, ry)) return 0;
    const grad = Math.hypot(nx / rx, ny / ry) / f;
    const distance = (f - 1) / Math.max(1e-6, grad); // negative inside
    return clamp01(Math.min(-distance + 0.5, distance + width + 0.5));
  });
}

function drawDisc(surface: Surface, cx: number, cy: number, r: number, color: Rgb, opacity: number): void {
  surface.paint([cx - r - 1, cy - r - 1, cx + r + 1, cy + r + 1], color, opacity, (x, y) => clamp01(r + 0.5 - Math.hypot(x - cx, y - cy)));
}

function drawRect(surface: Surface, x0: number, y0: number, x1: number, y1: number, color: Rgb, opacity: number): void {
  surface.paint([x0, y0, x1, y1], color, opacity, (x, y) => x >= x0 && x <= x1 + 1 && y >= y0 && y <= y1 + 1 ? 1 : 0);
}

/* ================================================================ random */

/** A seeded generator (mulberry32) with numpy-like helpers, for sparks, bolts and particles. */
export function seeded(seed: number) {
  let state = seed >>> 0;
  const next = () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return {
    random: next,
    normal: (mean: number, deviation: number) => mean + deviation * Math.sqrt(-2 * Math.log(Math.max(1e-12, next()))) * Math.cos(TAU * next()),
    /** Integer in [low, high). */
    integers: (low: number, high: number) => low + Math.floor(next() * Math.max(1, high - low)),
  };
}

/* ============================================================ animation */

export function ease(value: number, easing: string): number {
  const v = clamp01(value);
  if (easing === "smooth") return v * v * (3 - 2 * v);
  if (easing === "ease-in") return v * v;
  if (easing === "ease-out") return 1 - (1 - v) ** 2;
  return v;
}

/** `sample_animation`: the keyframed transform, opacity and intensity at `progress` of the range. */
export function sampleAnimation(recipe: MaskVfxRecipeV1, progress: number): MaskVfxAnimatedValues {
  const defaults: MaskVfxAnimatedValues = {
    effectX: recipe.effectX, effectY: recipe.effectY, effectScale: recipe.effectScale,
    effectRotation: recipe.effectRotation, effectOpacity: recipe.effectOpacity, intensity: recipe.intensity,
  };
  const keyframes = recipe.keyframes.slice().sort((a, b) => a.at - b.at);
  if (!keyframes.length) return defaults;
  if (progress <= keyframes[0].at) return { ...defaults, ...keyframes[0].values };
  if (progress >= keyframes[keyframes.length - 1].at) return { ...defaults, ...keyframes[keyframes.length - 1].values };
  for (let i = 0; i + 1 < keyframes.length; i++) {
    const left = keyframes[i], right = keyframes[i + 1];
    if (left.at <= progress && progress <= right.at) {
      const amount = ease((progress - left.at) / Math.max(1e-9, right.at - left.at), right.easing);
      const first = { ...defaults, ...left.values }, second = { ...defaults, ...right.values };
      const out = { ...defaults };
      for (const key of Object.keys(defaults) as Array<keyof MaskVfxAnimatedValues>) out[key] = first[key] + (second[key] - first[key]) * amount;
      return out;
    }
  }
  return defaults;
}

/* ================================================================ mattes */

/** `shape_mask`: grow or shrink, feather, invert (sizes in source pixels). */
export function shapeMask(mask: Plane, width: number, height: number, feather: number, expand: number, inverted: boolean, k: number): Plane {
  let value = mask;
  if (expand) {
    let size = Math.min(65, Math.abs(expand) * 2 + 1);
    if (size % 2 === 0) size += 1;
    value = expand > 0 ? maxFilter(value, width, height, size, k) : minFilter(value, width, height, size, k);
  }
  if (feather) value = gauss(value, width, height, feather * k);
  if (inverted) { const out = plane(width, height); for (let i = 0; i < out.length; i++) out[i] = 255 - value[i]; return out; }
  return value === mask ? new Float32Array(mask) : value;
}

/** `animated_matte`: the subject itself, or rings, a burst or flames derived from it. */
export function animatedMatte(subject: Plane, width: number, height: number, style: MaskVfxRecipeV1["matteStyle"], phase: number, k: number): Plane {
  if (style === "subject" || style === "clip") return subject;
  const [left, top, right, bottom] = maskBounds(subject, width, height);
  const w = Math.max(4, right - left), h = Math.max(4, bottom - top);
  if (style === "rings") {
    const surface = new Surface(width, height);
    const cx = (left + right) / 2;
    for (let ring = 0; ring < 4; ring++) {
      const progress = (phase + ring / 4) % 1;
      const rx = w * (0.48 + progress * 0.55), ry = h * (0.12 + progress * 0.26);
      const y = bottom - h * (0.04 + progress * 0.82);
      const alpha = Math.round(255 * Math.sin(progress * Math.PI));
      drawEllipseOutline(surface, cx, y, rx, ry, [255, 255, 255], alpha, Math.max(2, Math.round(Math.min(w, h) * 0.018)));
    }
    return gauss(surface.alpha, width, height, Math.max(1, Math.min(w, h) * 0.008));
  }
  if (style === "burst") {
    const surface = new Surface(width, height);
    const cx = (left + right) / 2, cy = (top + bottom) / 2;
    const maximum = Math.hypot(width, height) * (0.18 + 0.38 * phase);
    for (let ray = 0; ray < 24; ray++) {
      const angle = (ray / 24) * TAU + Math.sin(phase * TAU + ray) * 0.035;
      const start = maximum * (0.15 + (ray % 3) * 0.04), end = maximum * (0.7 + (ray % 5) * 0.08);
      drawLine(surface, [[cx + Math.cos(angle) * start, cy + Math.sin(angle) * start], [cx + Math.cos(angle) * end, cy + Math.sin(angle) * end]],
        [255, 255, 255], 95 + (ray % 4) * 40, Math.max(1, Math.round(Math.min(w, h) * (0.008 + (ray % 2) * 0.004))));
    }
    return gauss(surface.alpha, width, height, 1.2 * k);
  }
  // Flame: an outside rim with several vertically shifted, oscillating tongues.
  const radius = clamp(Math.round(Math.min(w, h) * 0.055) | 1, 5, Math.max(5, Math.round(63 * k) | 1));
  const outer = extreme(subject, width, height, (radius - 1) / 2, true);
  const inner = Math.max(3, (Math.floor(radius / 3) | 1));
  const flame = subtract(outer, extreme(subject, width, height, (inner - 1) / 2, false));
  for (let tongue = 1; tongue <= 4; tongue++) {
    const xShift = Math.round(Math.sin(phase * TAU * (1 + tongue * 0.13) + tongue * 1.7) * w * 0.018 * tongue);
    const yShift = -Math.round(h * (0.018 + tongue * 0.012));
    const gain = 0.28 / tongue;
    for (let y = 0; y < height; y++) {
      const sy = y - yShift;
      if (sy < 0 || sy >= height) continue;
      for (let x = 0; x < width; x++) {
        const sx = x - xShift;
        if (sx < 0 || sx >= width) continue;
        const value = Math.round(outer[sy * width + sx] * gain);
        const i = y * width + x;
        if (value > flame[i]) flame[i] = value;
      }
    }
  }
  const out = plane(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const turbulence = 0.74 + 0.26 * Math.sin((x / k) * 0.045 + (y / k) * 0.018 - phase * TAU * 2.3);
      out[y * width + x] = clamp(flame[y * width + x] * turbulence, 0, 255);
    }
  }
  return out;
}

/** `_extract_clip_matte`: transparency from alpha, luma, or distance from the key colour. */
export function clipMatte(layer: Float32Array, count: number, recipe: MaskVfxRecipeV1): Plane {
  const out = plane(count, 1);
  const key = hexToRgb(recipe.keyColor).map((value) => value / 255);
  for (let i = 0; i < count; i++) {
    const a = layer[i * 4 + 3];
    if (recipe.keyMode === "alpha") { out[i] = a; continue; }
    // The old worker keyed the RGB of the RGBA frame (PIL drops alpha; fully transparent is black).
    const f = a > 0 ? 1 / 255 : 0;
    const r = layer[i * 4] * f, g = layer[i * 4 + 1] * f, b = layer[i * 4 + 2] * f;
    let alpha: number;
    if (recipe.keyMode === "luma") {
      alpha = clamp01((r * 0.2126 + g * 0.7152 + b * 0.0722 - recipe.keyThreshold) / Math.max(0.001, recipe.keySoftness) + 0.5);
    } else {
      const distance = Math.sqrt(((r - key[0]) ** 2 + (g - key[1]) ** 2 + (b - key[2]) ** 2) / 3);
      alpha = clamp01((distance - recipe.keyThreshold) / Math.max(0.001, recipe.keySoftness) + 0.5);
    }
    out[i] = alpha * alpha * (3 - 2 * alpha) * 255;
  }
  return out;
}

/** `_displace`: each row shifted by a sine of its (source) row, `amount` source pixels. */
function displaceRows(values: Float32Array, width: number, height: number, channels: number, amount: number, phase: number, k: number): Float32Array {
  if (amount <= 0) return values;
  const out = new Float32Array(values.length);
  for (let y = 0; y < height; y++) {
    const offset = Math.round(Math.sin((y / k) * 0.045 + phase * TAU) * amount * k);
    for (let x = 0; x < width; x++) {
      // np.roll: element x comes from x - offset, wrapping.
      const from = ((x - offset) % width + width) % width;
      for (let c = 0; c < channels; c++) out[(y * width + x) * channels + c] = values[(y * width + from) * channels + c];
    }
  }
  return out;
}

/* =========================================================== transitions */

const softThreshold = (value: number, boundary: number, softness = 0.045) => {
  const ramp = clamp01((boundary - value) / Math.max(0.001, softness) + 0.5);
  return ramp * ramp * (3 - 2 * ramp);
};

/**
 * `transition_alpha`: a full-frame reveal matte (0..255) in working pixels. Positions are measured
 * in the source frame (`toSource` maps working pixels there), as the old worker measured them.
 */
export function transitionAlpha(
  width: number, height: number, kind: MaskEffectTransition, progress: number, frameIndex: number, speed: number,
  anchor: [number, number], toSource: Matrix, sourceWidth: number, sourceHeight: number,
): Plane {
  const out = plane(width, height);
  const amount = clamp01(progress);
  if (kind === "none") return out.fill(255);
  if (kind === "fade") return out.fill(Math.round(amount * 255));
  const [ax, ay] = anchor;
  const maximum = Math.max(Math.hypot(ax, ay), Math.hypot(1 - ax, ay), Math.hypot(ax, 1 - ay), Math.hypot(1 - ax, 1 - ay));
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [sx, sy] = applyMatrix(toSource, x + 0.5, y + 0.5);
      const u = (sx - 0.5) / Math.max(1, sourceWidth - 1), v = (sy - 0.5) / Math.max(1, sourceHeight - 1);
      let alpha: number;
      if (kind === "top-to-bottom") alpha = softThreshold(v, amount);
      else if (kind === "bottom-to-top") alpha = softThreshold(1 - v, amount);
      else {
        const dx = u - ax, dy = v - ay;
        const radius = Math.sqrt(dx * dx + dy * dy) / Math.max(0.001, maximum);
        if (kind === "edges-to-center") alpha = softThreshold(1 - radius, amount, 0.065);
        else if (kind === "explosion") {
          const turbulence = Math.sin(Math.atan2(dy, dx) * 11 + frameIndex * 0.31 * speed) * 0.045 * (1 - amount);
          alpha = softThreshold(radius, amount + turbulence, 0.065);
        } else if (kind === "power-up") {
          const wave = Math.sin(radius * 34 - frameIndex * 0.28 * speed) * 0.025;
          alpha = softThreshold(radius, amount + wave, 0.075) * amount;
        } else alpha = softThreshold(radius, amount, 0.06);
      }
      out[y * width + x] = clamp(alpha * 255, 0, 255);
    }
  }
  return out;
}

/** `transition_energy_overlay`: shockwave rings for the explosion and power-up transitions. */
function energyOverlay(
  width: number, height: number, kind: MaskEffectTransition, progress: number, primary: Rgb, secondary: Rgb,
  frameIndex: number, strength: number, anchor: [number, number], toWork: Matrix, sourceWidth: number, sourceHeight: number, k: number,
): Layer | undefined {
  if ((kind !== "explosion" && kind !== "power-up") || progress <= 0 || progress >= 1) return undefined;
  const surface = new Surface(width, height);
  const [cx, cy] = applyMatrix(toWork, sourceWidth * anchor[0], sourceHeight * anchor[1]);
  const scx = sourceWidth * anchor[0], scy = sourceHeight * anchor[1];
  const maximum = Math.max(Math.hypot(scx, scy), Math.hypot(sourceWidth - scx, scy), Math.hypot(scx, sourceHeight - scy), Math.hypot(sourceWidth - scx, sourceHeight - scy)) * k;
  if (kind === "explosion") {
    const radius = maximum * progress;
    drawEllipseOutline(surface, cx, cy, radius, radius, secondary, Math.trunc(230 * Math.sin(progress * Math.PI)), Math.max(2, Math.trunc(3 + 7 * strength)) * k);
    return layerOver(blurLayer(surface.layer(), width, height, 9 * k), surface.layer());
  }
  for (const [offset, color] of [[0, primary], [0.11, secondary], [0.22, primary]] as const) {
    const radius = maximum * ((progress + offset + frameIndex * 0.006) % 1);
    const fade = Math.sin(progress * Math.PI) * (1 - offset * 1.8);
    drawEllipseOutline(surface, cx, cy, radius, radius, color, Math.trunc(175 * fade), Math.max(2, Math.trunc(2 + 4 * strength)) * k);
  }
  return layerOver(blurLayer(surface.layer(), width, height, 12 * k), surface.layer());
}

/* =============================================================== effects */

function energyLines(width: number, height: number, bounds: [number, number, number, number], color: Rgb, seed: number, amount: number, k: number): Layer {
  const surface = new Surface(width, height);
  const random = seeded(seed);
  const [left, top, right, bottom] = bounds;
  const w = Math.max(2, right - left), h = Math.max(2, bottom - top);
  const lines = Math.max(2, Math.trunc(4 + amount * 7));
  for (let line = 0; line < lines; line++) {
    const side = line % 4;
    const points: Array<[number, number]> = [];
    for (let step = 0; step < 9; step++) {
      const t = step / 8;
      const jx = Math.trunc(random.normal(0, Math.max(2 * k, w * 0.035))), jy = Math.trunc(random.normal(0, Math.max(2 * k, h * 0.035)));
      if (side === 0) points.push([left + Math.trunc(w * t) + jx, top + jy]);
      else if (side === 1) points.push([right + jx, top + Math.trunc(h * t) + jy]);
      else if (side === 2) points.push([right - Math.trunc(w * t) + jx, bottom + jy]);
      else points.push([left + jx, bottom - Math.trunc(h * t) + jy]);
    }
    drawLine(surface, points, color, 230, Math.max(1, Math.trunc(1 + amount)) * k);
  }
  return surface.layer();
}

function fireballLayer(width: number, height: number, center: [number, number], radius: number, primary: Rgb, secondary: Rgb, phase: number, amount: number): Layer {
  const count = width * height;
  const rgb = new Float32Array(count * 3), alpha = plane(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - center[0], dy = y - center[1];
      const distance = Math.sqrt(dx * dx + dy * dy) / Math.max(1, radius);
      const turbulence = Math.sin(Math.atan2(dy, dx) * 7 + phase * 8 + distance * 13) * 0.09;
      const core = clamp01(1 - distance + turbulence);
      const i = y * width + x;
      alpha[i] = Math.trunc(clamp(core ** 0.62 * 255 * Math.min(1.5, amount), 0, 255));
      const m = clamp01(distance);
      for (let c = 0; c < 3; c++) rgb[i * 3 + c] = Math.trunc(clamp(primary[c] * (1 - m) + secondary[c] * m, 0, 255));
    }
  }
  const layer = { rgb, alpha };
  return layerOver(blurLayer(layer, width, height, Math.max(2, radius * 0.15)), layer);
}

type EffectContext = {
  width: number; height: number; k: number; count: number;
  color: Rgb; secondary: Rgb; frameIndex: number; fps: number; speed: number; particles: number;
};

/** `apply_effect`: one effect through `mask` (0..255) on an opaque RGB picture. */
export function applyEffect(base: Rgb3, mask: Plane, effect: MaskVfxRecipeV1["effect"], strength: number, ctx: EffectContext): Rgb3 {
  const { width, height, k, count, color, secondary, frameIndex } = ctx;
  const value = clamp(strength, 0, 3.5);
  const phase = (frameIndex / Math.max(1, ctx.fps)) * ctx.speed;
  const colour = (tint: Rgb, alpha: Plane) => colourLayer(count, tint, alpha);
  const g = (p: Plane, sigma: number) => gauss(p, width, height, sigma * k);
  switch (effect) {
    case "glow": {
      const halo = subtract(g(mask, 8 + value * 18), g(mask, 2));
      return over(base, colour(color, scaled(halo, Math.min(2, value * 1.25))));
    }
    case "outline": {
      const size = odd(clamp(Math.trunc(2 + value * 7), 1, 31));
      const edge = subtract(maxFilter(mask, width, height, size, k), minFilter(mask, width, height, size, k));
      return over(base, colour(color, scaled(edge, Math.min(2, value))));
    }
    case "neon-edge": {
      const edge = edgeOf(mask, width, height, Math.trunc(3 + value * 6), k);
      const glow = g(edge, 5 + value * 8);
      const result = screen(base, colour(secondary, scaled(glow, value * 0.9)));
      return screen(result, colour(color, scaled(edge, Math.min(2, value * 1.3))));
    }
    case "electric": {
      const edge = edgeOf(mask, width, height, 5, k);
      let result = screen(base, colour(color, scaled(g(edge, 8), value)));
      const bolts = energyLines(width, height, maskBounds(mask, width, height), secondary, Math.floor(frameIndex / 2) + 1049, ctx.particles * value, k);
      result = screen(result, blurLayer(bolts, width, height, 4 * k));
      return over(result, bolts);
    }
    case "energy-shield": {
      const surface = new Surface(width, height);
      const [left, top, right, bottom] = maskBounds(mask, width, height);
      const pulse = 1 + Math.sin(phase * TAU) * 0.035 * value;
      const cx = (left + right) / 2, cy = (top + bottom) / 2;
      const w = (right - left) * 1.18 * pulse, h = (bottom - top) * 1.12 * pulse;
      drawEllipseOutline(surface, cx, cy, w / 2, h / 2, color, 210, Math.max(2, Math.trunc(2 + value * 3)) * k);
      const result = screen(base, blurLayer(surface.layer(), width, height, 10 * k));
      return over(result, surface.layer());
    }
    case "aura": {
      let result = base;
      for (const [radius, amount] of [[38, 0.22], [20, 0.38], [8, 0.65]]) {
        result = over(result, colour(color, scaled(g(mask, Math.max(1, radius * value)), amount * value)));
      }
      return result;
    }
    case "spotlight": {
      const grey = luma601(base, count);
      const saturation = Math.max(0, 1 - value * 0.8), brightness = Math.max(0.08, 1 - value * 0.62);
      const dark = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) for (let c = 0; c < 3; c++) dark[i * 3 + c] = clamp((grey[i] + (base[i * 3 + c] - grey[i]) * saturation) * brightness, 0, 255);
      const result = through(base, dark, mask);
      return over(result, colour(color, scaled(g(mask, 10 + value * 10), value * 0.18)));
    }
    case "background-blur":
      return through(base, blurRgb(base, width, height, (2 + value * 18) * k), mask);
    case "background-pixelate": {
      const block = Math.max(1, Math.round(Math.max(3, Math.trunc(5 + value * 18)) * k));
      const background = new Float32Array(count * 3);
      for (let by = 0; by < height; by += block) {
        for (let bx = 0; bx < width; bx += block) {
          const sum = [0, 0, 0];
          let n = 0;
          for (let y = by; y < Math.min(height, by + block); y++) for (let x = bx; x < Math.min(width, bx + block); x++) {
            const i = y * width + x;
            sum[0] += base[i * 3]; sum[1] += base[i * 3 + 1]; sum[2] += base[i * 3 + 2]; n++;
          }
          for (let y = by; y < Math.min(height, by + block); y++) for (let x = bx; x < Math.min(width, bx + block); x++) {
            const i = y * width + x;
            for (let c = 0; c < 3; c++) background[i * 3 + c] = sum[c] / n;
          }
        }
      }
      return through(base, background, mask);
    }
    case "silhouette": {
      const result = over(base, colour(color, scaled(mask, Math.min(1, 0.45 + value * 0.55))));
      return over(result, colour(color, scaled(g(mask, 5 + value * 12), value * 0.35)));
    }
    case "hologram": {
      const grey = luma601(base, count);
      const tinted = new Float32Array(count * 3);
      const black: Rgb = [0, 18, 30];
      for (let i = 0; i < count; i++) for (let c = 0; c < 3; c++) tinted[i * 3 + c] = black[c] + (color[c] - black[c]) * (grey[i] / 255);
      const every = Math.max(2, Math.trunc(6 - Math.min(value, 2) * 2));
      const level = Math.trunc(80 + Math.min(value, 2) * 70);
      const subjectAlpha = plane(width, height);
      for (let y = 0; y < height; y++) {
        const scan = Math.floor(y / k) % every === 0 ? level : 0;
        for (let x = 0; x < width; x++) subjectAlpha[y * width + x] = (mask[y * width + x] * (255 - scan)) / 255;
      }
      const result = through(tinted, base, scaled(subjectAlpha, Math.min(1, 0.45 + value * 0.35)));
      return over(result, colour(color, scaled(g(mask, 7 + value * 8), value * 0.28)));
    }
    case "heat-vision": {
      const grey = luma601(base, count);
      const heat = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        const l = grey[i] / 255;
        heat[i * 3] = Math.trunc(clamp01(l * 3) * 255);
        heat[i * 3 + 1] = Math.trunc(clamp01((l - 0.28) * 2.2) * 255);
        heat[i * 3 + 2] = Math.trunc(clamp01((0.62 - l) * 2.4) * 255);
      }
      return through(heat, base, scaled(mask, Math.min(1, 0.5 + value * 0.5)));
    }
    case "rgb-split": {
      const offset = Math.max(1, Math.round(Math.max(1, Math.trunc(2 + value * 8)) * k));
      const split = new Float32Array(base);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const i = y * width + x;
          split[i * 3] = base[(y * width + ((x - offset) % width + width) % width) * 3];
          split[i * 3 + 2] = base[(y * width + (x + offset) % width) * 3 + 2];
        }
      }
      return through(split, base, scaled(mask, Math.min(1, 0.45 + value * 0.35)));
    }
    case "subject-echo": {
      let result = base;
      const echoes = Math.max(2, Math.trunc(2 + ctx.particles * 3));
      for (let echo = echoes; echo >= 1; echo--) {
        const shift = Math.round(Math.trunc(echo * (4 + value * 5)) * k);
        const tint = echo % 2 ? color : secondary;
        const amount = value * 0.14 * (1 - echo / (echoes + 1));
        const dy = Math.trunc(Math.sin(phase * 4 + echo) * shift * 0.25);
        const rgb = new Float32Array(count * 3), alpha = plane(width, height);
        for (let y = 0; y < height; y++) {
          const sy = y - dy;
          if (sy < 0 || sy >= height) continue;
          for (let x = 0; x < width; x++) {
            const sx = x + shift;
            if (sx < 0 || sx >= width) continue;
            const from = sy * width + sx, to = y * width + x;
            const m = mask[from] / 255;
            // The subject pasted through its matte (so its colour is weighted by it), blended 55% with the tint.
            for (let c = 0; c < 3; c++) rgb[to * 3 + c] = base[from * 3 + c] * m * 0.45 + tint[c] * 0.55;
            alpha[to] = clamp(mask[from] * amount, 0, 255);
          }
        }
        result = over(result, { rgb, alpha });
      }
      return result;
    }
    case "fireball": {
      const [left, top, right, bottom] = maskBounds(mask, width, height);
      const center: [number, number] = [(left + right) * 0.5, (top + bottom) * 0.5];
      const radius = Math.max(14 * k, Math.min(right - left, bottom - top) * (0.16 + value * 0.12));
      const orb = fireballLayer(width, height, center, radius, color, secondary, phase, ctx.particles * value);
      const result = screen(base, orb);
      const random = seeded(707 + frameIndex);
      const sparks = new Surface(width, height);
      for (let n = 0; n < Math.max(3, Math.trunc(8 * ctx.particles * value)); n++) {
        const angle = random.random() * TAU;
        const distance = radius * (0.7 + random.random() * 1.2);
        const r = Math.max(1, random.integers(1, Math.max(2, Math.trunc(radius * 0.045)) + 1));
        drawDisc(sparks, center[0] + Math.cos(angle) * distance, center[1] + Math.sin(angle) * distance, r, secondary, 220);
      }
      return over(result, blurLayer(sparks.layer(), width, height, 1 * k));
    }
    case "disintegrate": {
      let result = base;
      const seed = 1701 + Math.floor(frameIndex / 2);
      const chance = 0.018 * ctx.particles * value;
      const random = seeded(seed);
      const particles = new Surface(width, height);
      const drift = Math.round(Math.trunc((5 + 18 * value) * (0.35 + (phase % 1))) * k);
      const limit = Math.max(1, Math.round(4500 * Math.min(1, k * k)));
      let placed = 0;
      for (let y = 0; y < height && placed < limit; y++) {
        for (let x = 0; x < width && placed < limit; x++) {
          const i = y * width + x;
          if (mask[i] <= 100 || hash01(x, y, seed) >= chance) continue;
          const size = Math.round(random.integers(1, 4 + Math.max(1, Math.trunc(value))) * k);
          const rgb: Rgb = [base[i * 3], base[i * 3 + 1], base[i * 3 + 2]];
          const x0 = x + drift, y0 = y - Math.floor(drift / 3);
          drawRect(particles, x0, y0, x0 + size, y0 + size, rgb, 210);
          placed++;
        }
      }
      const edge = edgeOf(mask, width, height, 5, k);
      result = screen(result, colour(color, scaled(edge, value * 0.75)));
      return over(result, particles.layer());
    }
    case "comic-pop": {
      const bits = Math.max(2, 6 - Math.trunc(Math.min(value, 2) * 2));
      const keep = 256 - (1 << (8 - bits));
      const poster = new Float32Array(count * 3);
      for (let i = 0; i < count * 3; i++) poster[i] = Math.round(clamp(base[i], 0, 255)) & keep;
      // ImageEnhance.Contrast: towards the picture's mean grey.
      const grey = luma601(poster, count);
      let mean = 0;
      for (let i = 0; i < count; i++) mean += grey[i];
      mean = Math.trunc(mean / Math.max(1, count) + 0.5);
      const factor = 1.15 + value * 0.35;
      for (let i = 0; i < count * 3; i++) poster[i] = clamp(mean + (poster[i] - mean) * factor, 0, 255);
      let result = through(poster, base, mask);
      // FIND_EDGES of the grey picture, inverted, through the matte; dark where below 95.
      const lum = luma601(base, count);
      const ink = plane(width, height);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let edge = 0;
          if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
            const i = y * width + x;
            edge = 8 * lum[i] - lum[i - 1] - lum[i + 1] - lum[i - width] - lum[i + width] - lum[i - width - 1] - lum[i - width + 1] - lum[i + width - 1] - lum[i + width + 1];
          }
          const outlined = 255 - clamp(edge, 0, 255);
          ink[y * width + x] = (outlined * mask[y * width + x]) / 255 < 95 ? 255 : 0;
        }
      }
      result = over(result, colour([8, 8, 12], scaled(ink, 0.72)));
      const dots = new Surface(width, height);
      const spacing = Math.max(7, Math.trunc(13 - value * 3)) * k;
      for (let y = spacing / 2; y < height; y += spacing) {
        for (let x = spacing / 2; x < width; x += spacing) {
          if (mask[Math.floor(y) * width + Math.floor(x)] > 128) drawDisc(dots, x, y, 1 * k, secondary, 95);
        }
      }
      return over(result, dots.layer());
    }
    case "power-up": {
      const edge = edgeOf(mask, width, height, Math.max(5, Math.trunc(7 + value * 6)), k);
      let result = base;
      for (const [radius, amount, tint] of [[34, 0.26, secondary], [17, 0.46, color], [5, 0.92, [255, 255, 255] as Rgb]] as const) {
        result = screen(result, colour(tint, scaled(g(edge, Math.max(1, radius * Math.max(0.35, value))), amount * value)));
      }
      const bolts = energyLines(width, height, maskBounds(mask, width, height), color, 4200 + Math.floor(frameIndex / 2), ctx.particles * value, k);
      result = screen(result, blurLayer(bolts, width, height, 5 * k));
      return over(result, bolts);
    }
    case "animated-matte":
    default:
      return base;
  }
}

/* ================================================================ frame */

/** A gradient layer in working pixels, positioned in the source frame, travelling with `phase`. */
function gradientLayer(gradient: Gradient, phase: number, width: number, height: number, toSource: Matrix, sourceWidth: number, sourceHeight: number, where: Plane): Layer {
  const repeat = Math.max(1, gradient.repeat);
  const travel = repeat > 1 ? phase : Math.sin(phase * TAU) * 0.15;
  const moved = { ...gradient, offset: gradient.offset + travel };
  const lut = buildGradientLut(moved);
  const last = lut.length / 4 - 1;
  const dither = clamp01(moved.dither ?? 0);
  const count = width * height;
  const rgb = new Float32Array(count * 3), alpha = plane(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Only where it will show: the layer's alpha is multiplied by the matte afterwards.
      if (where[y * width + x] <= 0) continue;
      const sx = toSource.a * (x + 0.5) + toSource.c * (y + 0.5) + toSource.e;
      const sy = toSource.b * (x + 0.5) + toSource.d * (y + 0.5) + toSource.f;
      const p = rampParameter(moved, gradientPosition(moved, sx, sy, sourceWidth, sourceHeight));
      const index = Math.min(last, Math.max(0, Math.round(p * last))) * 4;
      const wobble = dither > 0 ? (BAYER[(Math.floor(sy) & 7) * 8 + (Math.floor(sx) & 7)] - 0.5) * dither : 0;
      const i = y * width + x;
      rgb[i * 3] = clamp(lut[index] + wobble, 0, 255);
      rgb[i * 3 + 1] = clamp(lut[index + 1] + wobble, 0, 255);
      rgb[i * 3 + 2] = clamp(lut[index + 2] + wobble, 0, 255);
      alpha[i] = lut[index + 3];
    }
  }
  return { rgb, alpha };
}

/** A standard normal from two hashed uniforms (the grain; stable per pixel and frame). */
const gaussianAt = (x: number, y: number, seed: number) =>
  Math.sqrt(-2 * Math.log(Math.max(1e-12, hash01(x, y, seed)))) * Math.cos(TAU * hash01(x, y, seed + 7919));

/** The linear scale of a placement (source pixels -> working pixels). */
export const matrixScale = (m: Matrix): number => Math.sqrt(Math.abs(m.a * m.d - m.b * m.c)) || 1;

/**
 * The source-frame transform of `_transform_canvas`: the whole frame resized by `scale` percent and
 * turned by `rotation` degrees, its centre placed at `center` (normalised source coordinates).
 */
export function effectTransform(center: [number, number], scale: number, rotation: number, sourceWidth: number, sourceHeight: number): Matrix {
  const factor = Math.max(0.01, scale / 100);
  let m = translate(-sourceWidth / 2, -sourceHeight / 2);
  m = compose(scaleBy(factor, factor), m);
  m = compose(rotateBy(rotation), m);
  return compose(translate(center[0] * sourceWidth, center[1] * sourceHeight), m);
}

/** One frame of a mask VFX recipe over the layer's working pixels: a new straight RGBA buffer. */
export function renderMaskFx(input: MaskFxInput): Uint8ClampedArray {
  const { pixels, width, height, recipe: r, index, fps, toWork, sourceWidth, sourceHeight } = input;
  const count = width * height;
  const frames = Math.max(1, input.count);
  const k = matrixScale(toWork);
  const toSource = invert(toWork);
  const primary = hexToRgb(r.color) as Rgb, secondary = hexToRgb(r.secondaryColor) as Rgb;

  const source: Rgb3 = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) for (let c = 0; c < 3; c++) source[i * 3 + c] = pixels[i * 4 + c];
  const raw = plane(width, height);
  for (let i = 0; i < count; i++) raw[i] = input.matte[i] ?? 0;

  // The holdout is deliberately independent of the stylised matte's growth and inversion.
  const subject = shapeMask(raw, width, height, Math.round(r.occlusionFeather), r.occlusionExpand, false, k);
  const selected = shapeMask(raw, width, height, r.effectFeather, r.effectExpand, r.invert, k);
  const elapsed = index / Math.max(1, fps);
  const animation = sampleAnimation(r, index / Math.max(1, frames - 1));
  const rawPhase = elapsed * r.effectSpeed;
  const phase = r.loop ? rawPhase % 1 : Math.min(1, rawPhase);
  const pulse = 1 + r.pulseAmount * Math.sin(elapsed * r.pulseRate * TAU);
  const center: [number, number] = [animation.effectX, animation.effectY];
  const moveBy = effectTransform(center, animation.effectScale * pulse, animation.effectRotation, sourceWidth, sourceHeight);

  let clip: Float32Array | undefined;
  let matte: Plane;
  if (r.matteStyle === "clip") {
    const effect = input.effect;
    if (effect) {
      // Fitted into the source frame (contain, centred, transparent bars), then moved.
      const s = Math.min(sourceWidth / effect.width, sourceHeight / effect.height);
      const fit = compose(translate((sourceWidth - effect.width * s) / 2, (sourceHeight - effect.height * s) / 2), scaleBy(s, s));
      clip = warpRgba(effect.data, effect.width, effect.height, width, height, invert(compose(toWork, compose(moveBy, fit))));
      matte = clipMatte(clip, count, r);
    } else matte = plane(width, height);
  } else {
    const styled = animatedMatte(selected, width, height, r.matteStyle, phase, k);
    // The style is drawn around the subject where the layer shows it, then moved in the source frame.
    matte = warpPlane(styled, width, height, compose(toWork, compose(invert(moveBy), toSource)));
  }
  if (r.placement === "inside") matte = multiply(matte, subject);
  matte = displaceRows(matte, width, height, 1, r.displacement, phase, k);
  if (r.motionBlur) matte = gauss(matte, width, height, r.motionBlur * k);
  if (clip && r.displacement) clip = displaceRows(clip, width, height, 4, r.displacement, phase, k);

  const strength = animation.intensity * pulse * (1 + r.reactiveAmount * clamp01(input.envelope));
  let effected = applyEffect(source, matte, r.effect, strength, {
    width, height, k, count, color: primary, secondary, frameIndex: index, fps, speed: r.effectSpeed, particles: r.particleAmount,
  });
  const opacity = clamp01(animation.effectOpacity);
  effected = mix(source, effected, opacity);
  if (clip && r.effectClipColor === "original") {
    const rgb = new Float32Array(count * 3);
    const key = hexToRgb(r.keyColor);
    const spill = r.keyMode === "chroma" ? r.colorSpill : 0;
    for (let i = 0; i < count; i++) {
      // The frame as the old worker held it: RGBA where transparent pixels are black.
      for (let c = 0; c < 3; c++) {
        const value = clip[i * 4 + c] * (clip[i * 4 + 3] > 0 ? 1 : 0);
        rgb[i * 3 + c] = spill ? clamp(value + (value - key[c]) * spill, 0, 255) : value;
      }
    }
    effected = blendLayer(effected, { rgb, alpha: scaled(matte, opacity) }, r.blendMode);
  }
  if (r.gradientEnabled && (!clip || r.effectClipColor === "gradient")) {
    const through = scaled(matte, opacity);
    const layer = gradientLayer(r.gradient, phase, width, height, toSource, sourceWidth, sourceHeight, through);
    for (let i = 0; i < count; i++) layer.alpha[i] = (layer.alpha[i] * through[i]) / 255;
    effected = blendLayer(effected, layer, r.blendMode);
  }
  if (r.grain) {
    const seed = 8009 + index;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        const m = matte[i] / 255;
        if (m <= 0) continue;
        const sx = toSource.a * x + toSource.c * y + toSource.e, sy = toSource.b * x + toSource.d * y + toSource.f;
        const noise = gaussianAt(Math.floor(sx), Math.floor(sy), seed) * 14 * r.grain * m;
        for (let c = 0; c < 3; c++) effected[i * 3 + c] = clamp(effected[i * 3 + c] + noise, 0, 255);
      }
    }
  }

  const remaining = Math.max(0, (frames - 1 - index) / Math.max(1, fps));
  const inProgress = r.transitionIn === "none" ? 1 : Math.min(1, elapsed / r.transitionInDuration);
  const outProgress = r.transitionOut === "none" ? 1 : Math.min(1, remaining / r.transitionOutDuration);
  let anchor: [number, number] = center;
  if (r.transitionOrigin === "subject") {
    const [left, top, right, bottom] = maskBounds(subject, width, height);
    const [sx, sy] = applyMatrix(toSource, (left + right) / 2, (top + bottom) / 2);
    anchor = [sx / sourceWidth, sy / sourceHeight];
  }
  const entrance = transitionAlpha(width, height, r.transitionIn, inProgress, index, r.effectSpeed, anchor, toSource, sourceWidth, sourceHeight);
  const exit = transitionAlpha(width, height, r.transitionOut, outProgress, index, r.effectSpeed, anchor, toSource, sourceWidth, sourceHeight);
  const reveal = plane(width, height);
  for (let i = 0; i < count; i++) reveal[i] = Math.min(entrance[i], exit[i]);
  let revealed = through(effected, source, reveal);
  const [activeKind, activeProgress] = inProgress < 1 ? [r.transitionIn, inProgress] : [r.transitionOut, outProgress];
  const energy = energyOverlay(width, height, activeKind, activeProgress, primary, secondary, index, strength, anchor, toWork, sourceWidth, sourceHeight, k);
  if (energy) revealed = blendLayer(revealed, energy, r.blendMode);

  // Pass 1: the effect behind the clean subject cutout (the sandwich).
  let result = mix(source, through(source, revealed, subject), r.rearAmount);
  // Pass 2: light wrap and controlled colour spill on the clean subject edge only.
  if (r.subjectAmount && (r.lightWrap || r.colorSpill)) {
    const size = clamp(Math.round(5 + r.lightWrap * 34) | 1, 3, 63);
    const inner = subtract(subject, minFilter(subject, width, height, size, k));
    const wrap = gauss(inner, width, height, (2 + r.lightWrap * 8) * k);
    result = blendLayer(result, colourLayer(count, primary, scaled(wrap, r.subjectAmount * r.lightWrap)), "screen");
    if (r.colorSpill) result = blendLayer(result, colourLayer(count, secondary, scaled(inner, r.subjectAmount * r.colorSpill * 0.55)), "soft-light");
  }
  // Pass 3: a controlled foreground copy can cross the silhouette for sparks and energy.
  if (r.frontAmount) result = through(mix(source, revealed, r.frontAmount), result, matte);

  const out = new Uint8ClampedArray(pixels.length);
  for (let i = 0; i < count; i++) {
    out[i * 4] = result[i * 3];
    out[i * 4 + 1] = result[i * 3 + 1];
    out[i * 4 + 2] = result[i * 3 + 2];
    out[i * 4 + 3] = pixels[i * 4 + 3];
  }
  return out;
}
