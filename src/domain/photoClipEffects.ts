/**
 * VFX Phase 7: a timeline clip's "Effects" (brightness, contrast, saturation, gamma, hue, blur,
 * sharpen, vignette, noise) as a compositor node, plus the few per-pixel treatments the timeline's
 * transitions need (grayscale, pixelate, dissolve).
 *
 * Until Phase 7 these had two implementations that disagreed: CSS filters in the monitor and an
 * ffmpeg `eq,hue,gblur,unsharp,vignette,noise` chain in the export. Now there is this one, and both
 * compositors run it (the GPU compositor on its CPU path), so the monitor and the export agree by
 * construction.
 *
 * Semantics, fixed here (they follow ffmpeg's filters, which is what exports have always used):
 * - `eq` in BT.709 Y'CbCr on the encoded bytes: Y = (Y - .5) * contrast + .5 + brightness, then
 *   Y^(1/gamma); Cb and Cr scaled by saturation; `hue` rotates (Cb, Cr) by degrees.
 * - blur: Gaussian of that sigma (canvas pixels) as three box passes, alpha-weighted so a placed
 *   picture's edge does not pick up dark fringes.
 * - sharpen: unsharp mask against a 5 x 5 box, `amount` 0..2.
 * - vignette: RGB times cos(a * d / dmax)^2, a = pi / (9 - 6.5 * vignette), d from the frame centre.
 * - noise: +-noise levels of integer-hash noise, different every `seed` (the frame number).
 * - pixelate: blocks of that many canvas pixels, averaged.
 * - dissolve: alpha kept where a fixed per-pixel hash is below the value (xfade's "dissolve").
 */

export type ClipEffectsParams = {
  brightness: number;
  contrast: number;
  saturation: number;
  gamma: number;
  hue: number;
  blur: number;
  sharpen: number;
  vignette: number;
  noise: number;
  /** 0..1: how far towards grey (the "fadegrays" transition). */
  grayscale?: number;
  /** Block size in canvas pixels; 0 or 1 is off. */
  pixelate?: number;
  /** 0..1 revealed share for the dissolve transition; absent is off. */
  dissolve?: number;
  /** The noise pattern: the frame number, so grain moves. */
  seed?: number;
};

export const NEUTRAL_CLIP_EFFECTS: ClipEffectsParams = {
  brightness: 0, contrast: 1, saturation: 1, gamma: 1, hue: 0, blur: 0, sharpen: 0, vignette: 0, noise: 0,
};

export function isNeutralClipEffects(p: ClipEffectsParams): boolean {
  return Math.abs(p.brightness) < 1e-6 && Math.abs(p.contrast - 1) < 1e-6 && Math.abs(p.saturation - 1) < 1e-6
    && Math.abs(p.gamma - 1) < 1e-6 && Math.abs(p.hue) < 1e-6 && p.blur <= 0.001 && p.sharpen <= 0.001
    && p.vignette <= 0.001 && p.noise <= 0.001 && !(p.grayscale && p.grayscale > 0.0005)
    && !(p.pixelate && p.pixelate > 1) && p.dissolve === undefined;
}

/** A 32-bit integer hash of (x, y, seed) in [0, 1). Integer arithmetic, so it is exact everywhere. */
export function hash01(x: number, y: number, seed: number): number {
  let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(seed | 0, -2048144777)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}

const clampByte = (value: number) => value < 0 ? 0 : value > 255 ? 255 : Math.round(value);

/** Radii of three box blurs approximating a Gaussian of `sigma` (the standard construction). */
export function boxesForGauss(sigma: number, n = 3): number[] {
  const ideal = Math.sqrt((12 * sigma * sigma) / n + 1);
  let lower = Math.floor(ideal);
  if (lower % 2 === 0) lower--;
  const upper = lower + 2;
  const m = Math.round((12 * sigma * sigma - n * lower * lower - 4 * n * lower - 3 * n) / (-4 * lower - 4));
  return Array.from({ length: n }, (_, i) => ((i < m ? lower : upper) - 1) / 2);
}

/**
 * One box pass along rows or columns of a float plane, in place (edges clamped). The column pass
 * runs row by row over a vector of running sums (cache-friendly); each column's arithmetic is the
 * same sequence either way.
 */
function boxPass(plane: Float32Array, scratch: Float32Array, width: number, height: number, radius: number, horizontal: boolean): void {
  if (radius < 1) return;
  const r = Math.floor(radius);
  const span = 2 * r + 1;
  if (horizontal) {
    const last = width - 1;
    for (let line = 0; line < height; line++) {
      const base = line * width;
      let sum = 0;
      for (let k = -r; k <= r; k++) sum += plane[base + (k < 0 ? 0 : k > last ? last : k)];
      for (let i = 0; i < width; i++) {
        scratch[base + i] = sum / span;
        const out = i - r < 0 ? 0 : i - r, into = i + r + 1 > last ? last : i + r + 1;
        sum += plane[base + into] - plane[base + out];
      }
    }
  } else {
    const last = height - 1;
    const sums = new Float64Array(width);
    for (let k = -r; k <= r; k++) {
      const row = (k < 0 ? 0 : k > last ? last : k) * width;
      for (let x = 0; x < width; x++) sums[x] += plane[row + x];
    }
    for (let i = 0; i < height; i++) {
      const base = i * width;
      const outRow = (i - r < 0 ? 0 : i - r) * width, intoRow = (i + r + 1 > last ? last : i + r + 1) * width;
      for (let x = 0; x < width; x++) {
        scratch[base + x] = sums[x] / span;
        sums[x] += plane[intoRow + x] - plane[outRow + x];
      }
    }
  }
  plane.set(scratch);
}

/** Gaussian blur of float planes (three box passes each way). */
export function gaussPlanes(planes: Float32Array[], width: number, height: number, sigma: number): void {
  if (sigma <= 0.01) return;
  const scratch = new Float32Array(width * height);
  for (const radius of boxesForGauss(sigma)) {
    for (const plane of planes) {
      boxPass(plane, scratch, width, height, radius, true);
      boxPass(plane, scratch, width, height, radius, false);
    }
  }
}

/** Alpha-weighted blur of straight RGBA, in place. */
export function blurRgba(data: Uint8ClampedArray, width: number, height: number, sigma: number): void {
  if (sigma <= 0.01) return;
  const count = width * height;
  const r = new Float32Array(count), g = new Float32Array(count), b = new Float32Array(count), a = new Float32Array(count);
  for (let i = 0, j = 0; i < count; i++, j += 4) {
    const alpha = data[j + 3];
    r[i] = data[j] * alpha; g[i] = data[j + 1] * alpha; b[i] = data[j + 2] * alpha; a[i] = alpha;
  }
  gaussPlanes([r, g, b, a], width, height, sigma);
  for (let i = 0, j = 0; i < count; i++, j += 4) {
    const alpha = a[i];
    if (alpha <= 0.0001) { data[j] = data[j + 1] = data[j + 2] = data[j + 3] = 0; continue; }
    data[j] = clampByte(r[i] / alpha); data[j + 1] = clampByte(g[i] / alpha); data[j + 2] = clampByte(b[i] / alpha);
    data[j + 3] = clampByte(alpha);
  }
}

const KR = 0.2126, KB = 0.0722, KG = 1 - KR - KB;

/** The clip effects, in place, on straight RGBA at `width` x `height` (canvas pixels * `canvasScale`). */
export function applyClipEffects(data: Uint8ClampedArray, width: number, height: number, p: ClipEffectsParams, canvasScale = 1): void {
  const contrast = Math.min(2, Math.max(0, p.contrast));
  const brightness = Math.min(1, Math.max(-1, p.brightness));
  const gamma = Math.min(3, Math.max(0.1, p.gamma));
  const saturation = Math.min(3, Math.max(0, p.saturation)) * (1 - Math.min(1, Math.max(0, p.grayscale ?? 0)));
  const hue = (Math.min(180, Math.max(-180, p.hue)) * Math.PI) / 180;
  const colourChanges = Math.abs(brightness) > 1e-6 || Math.abs(contrast - 1) > 1e-6 || Math.abs(gamma - 1) > 1e-6
    || Math.abs(saturation - 1) > 1e-6 || Math.abs(hue) > 1e-6;
  if (colourChanges) {
    const cos = Math.cos(hue) * saturation, sin = Math.sin(hue) * saturation;
    const inverseGamma = 1 / gamma;
    // Luma through a table: the curve depends on Y alone.
    const lut = new Float32Array(1024);
    for (let i = 0; i < 1024; i++) {
      let y = (i / 1023 - 0.5) * contrast + 0.5 + brightness;
      y = y < 0 ? 0 : y > 1 ? 1 : y;
      lut[i] = inverseGamma === 1 ? y : Math.pow(y, inverseGamma);
    }
    for (let j = 0; j < data.length; j += 4) {
      if (data[j + 3] === 0) continue;
      const R = data[j] / 255, G = data[j + 1] / 255, B = data[j + 2] / 255;
      const Y = KR * R + KG * G + KB * B;
      const cb = (B - Y) / (2 * (1 - KB)), cr = (R - Y) / (2 * (1 - KR));
      const y2 = lut[Math.round(Math.min(1, Math.max(0, Y)) * 1023)];
      const cb2 = cb * cos - cr * sin, cr2 = cb * sin + cr * cos;
      const r2 = y2 + 2 * (1 - KR) * cr2;
      const b2 = y2 + 2 * (1 - KB) * cb2;
      const g2 = (y2 - KR * r2 - KB * b2) / KG;
      data[j] = clampByte(r2 * 255); data[j + 1] = clampByte(g2 * 255); data[j + 2] = clampByte(b2 * 255);
    }
  }
  if (p.blur > 0.001) blurRgba(data, width, height, Math.min(40, p.blur) * canvasScale);
  if (p.sharpen > 0.001) {
    const amount = Math.min(2, p.sharpen);
    const count = width * height;
    const planes = [0, 1, 2].map((c) => { const plane = new Float32Array(count); for (let i = 0; i < count; i++) plane[i] = data[i * 4 + c]; return plane; });
    const blurred = planes.map((plane) => new Float32Array(plane));
    const scratch = new Float32Array(count);
    for (const plane of blurred) { boxPass(plane, scratch, width, height, 2, true); boxPass(plane, scratch, width, height, 2, false); }
    for (let i = 0; i < count; i++) {
      if (data[i * 4 + 3] === 0) continue;
      for (let c = 0; c < 3; c++) data[i * 4 + c] = clampByte(planes[c][i] + amount * (planes[c][i] - blurred[c][i]));
    }
  }
  const block = Math.round((p.pixelate ?? 0) * canvasScale);
  if (block > 1) {
    for (let by = 0; by < height; by += block) {
      for (let bx = 0; bx < width; bx += block) {
        let r = 0, g = 0, b = 0, a = 0, n = 0;
        const ey = Math.min(height, by + block), ex = Math.min(width, bx + block);
        for (let y = by; y < ey; y++) for (let x = bx; x < ex; x++) {
          const j = (y * width + x) * 4, alpha = data[j + 3];
          r += data[j] * alpha; g += data[j + 1] * alpha; b += data[j + 2] * alpha; a += alpha; n++;
        }
        const outA = a / n;
        const R = a > 0 ? clampByte(r / a) : 0, G = a > 0 ? clampByte(g / a) : 0, B = a > 0 ? clampByte(b / a) : 0, A = clampByte(outA);
        for (let y = by; y < ey; y++) for (let x = bx; x < ex; x++) {
          const j = (y * width + x) * 4;
          data[j] = R; data[j + 1] = G; data[j + 2] = B; data[j + 3] = A;
        }
      }
    }
  }
  if (p.vignette > 0.001) {
    const angle = Math.PI / (9 - Math.min(1, p.vignette) * 6.5);
    const cx = width / 2, cy = height / 2, dmax = Math.hypot(cx, cy);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const j = (y * width + x) * 4;
        if (data[j + 3] === 0) continue;
        const k = Math.cos(angle * Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / dmax);
        const f = k * k;
        data[j] = clampByte(data[j] * f); data[j + 1] = clampByte(data[j + 1] * f); data[j + 2] = clampByte(data[j + 2] * f);
      }
    }
  }
  if (p.noise > 0.001) {
    const amount = Math.min(50, p.noise), seed = p.seed ?? 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const j = (y * width + x) * 4;
        if (data[j + 3] === 0) continue;
        const n = (hash01(x, y, seed) - 0.5) * 2 * amount;
        data[j] = clampByte(data[j] + n); data[j + 1] = clampByte(data[j + 1] + n); data[j + 2] = clampByte(data[j + 2] + n);
      }
    }
  }
  if (p.dissolve !== undefined) {
    const keep = Math.min(1, Math.max(0, p.dissolve));
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        if (hash01(x, y, 7919) >= keep) data[(y * width + x) * 4 + 3] = 0;
      }
    }
  }
}
