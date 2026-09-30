/**
 * The Photo Edit grade: one implementation, applied to both the preview and the saved file.
 *
 * The preview runs this over a downscaled copy of the picture and the save runs it over the full
 * one, so there is no second renderer that could disagree with what the user is looking at -- the
 * only difference between the two is how many pixels went in. That is the whole reason this grade
 * lives in the webview instead of the Python worker: a Pillow implementation would have to match
 * this one op for op, and the ones that bit this project before (BT.601 vs 709 on the upscale, the
 * inverted H3 decode) were all two-implementations-disagreeing bugs.
 *
 * Everything works in linear light. Exposure is a multiply, contrast pivots around middle grey and
 * blend maths behaves, all of which are wrong on gamma-encoded bytes. The cost is a decode and an
 * encode per pixel, both of which are table lookups.
 */

export type CurvePoint = { x: number; y: number };

/**
 * One of the eight colour bands the HSL panel splits the wheel into.
 *
 * Bands rather than a continuous picker because that is how people actually think about it -- "the
 * blues are too cyan" -- and because eight fixed centres let the weights be a cheap smooth falloff
 * instead of a range selection the user has to define first.
 */
export type HslBand = { hue: number; saturation: number; luminance: number };

export const HSL_BAND_NAMES = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"] as const;

/** Band centres in degrees, matching the names above. */
export const HSL_CENTRES = [0, 30, 60, 120, 180, 240, 285, 315];

export const neutralHslBands = (): HslBand[] =>
  HSL_CENTRES.map(() => ({ hue: 0, saturation: 0, luminance: 0 }));

export type PhotoAdjustments = {
  /** Stops. A multiply in linear light. */
  exposure: number;
  /** -100..100, pivoting on middle grey. */
  contrast: number;
  /** -100..100. Luminance-masked, so they move one end of the range without shifting the other. */
  highlights: number;
  shadows: number;
  /** -100..100. Warm/cool and green/magenta channel gains. */
  temperature: number;
  tint: number;
  /** 0..200, where 100 changes nothing. */
  saturation: number;
  /** -100..100. Saturation weighted towards the already-dull pixels. */
  vibrance: number;
  /** 0..100. How hard the luminance grain is squeezed out. */
  denoise: number;
  /** 0..100. How much of the fine texture the squeeze took is handed back. */
  denoiseDetail: number;
  /** 0..100. Chroma smoothing, on its own control because colour noise needs a far wider radius. */
  denoiseColor: number;
  /** 0..100. Unsharp mask amount. */
  sharpen: number;
  /** 0..100, mapped to a blur radius in pixels. */
  blur: number;
  /** 0..100. Monochrome grain, added after encoding so it looks like film rather than noise. */
  grain: number;
  /** Id from `PHOTO_FILTERS`. */
  filter: string;
  /** 0..100. How much of the filter to mix in. */
  filterAmount: number;
  /** Three draggable interior points. The ends are pinned at black and white. */
  curve: CurvePoint[];
  /** Input black and white points, 0-255, and a midtone gamma where 1 changes nothing. */
  levelsBlack: number;
  levelsWhite: number;
  levelsGamma: number;
  /** Eight bands of hue rotation, saturation and luminance. */
  hsl: HslBand[];
  /** -100 darkens the corners, +100 lightens them. */
  vignette: number;
  /** Where the falloff begins, as a percentage of the way out from the centre. */
  vignetteMidpoint: number;
  /** How gradual the falloff is. 0 is an abrupt edge, 100 is a slow wash. */
  vignetteFeather: number;
  rotate: 0 | 90 | 180 | 270;
  flipHorizontal: boolean;
  flipVertical: boolean;
};

export const NEUTRAL_CURVE: CurvePoint[] = [{ x: 0.25, y: 0.25 }, { x: 0.5, y: 0.5 }, { x: 0.75, y: 0.75 }];

export const NEUTRAL_ADJUSTMENTS: PhotoAdjustments = {
  exposure: 0, contrast: 0, highlights: 0, shadows: 0,
  temperature: 0, tint: 0, saturation: 100, vibrance: 0,
  denoise: 0, denoiseDetail: 40, denoiseColor: 0,
  sharpen: 0, blur: 0, grain: 0,
  filter: "none", filterAmount: 100,
  curve: NEUTRAL_CURVE.map((point) => ({ ...point })),
  levelsBlack: 0, levelsWhite: 255, levelsGamma: 1,
  hsl: neutralHslBands(),
  vignette: 0, vignetteMidpoint: 55, vignetteFeather: 60,
  rotate: 0, flipHorizontal: false, flipVertical: false,
};

/**
 * A look, expressed as lift / gamma / gain per channel plus a saturation term.
 *
 * Every one of these is pointwise, which is what lets the whole set fold into the same lookup table
 * as the curve. A filter that needed neighbouring pixels would not belong in this list.
 */
export type PhotoFilter = {
  id: string;
  name: string;
  lift: [number, number, number];
  gain: [number, number, number];
  gamma: [number, number, number];
  saturation: number;
};

export const PHOTO_FILTERS: PhotoFilter[] = [
  { id: "none", name: "None", lift: [0, 0, 0], gain: [1, 1, 1], gamma: [1, 1, 1], saturation: 1 },
  { id: "aurora", name: "Aurora 64", lift: [-.010, 0, .005], gain: [1.07, 1.01, .95], gamma: [.94, .97, 1.04], saturation: 1.18 },
  { id: "halation", name: "Halation", lift: [.030, .014, .004], gain: [1.05, 1.00, .94], gamma: [.93, .98, 1.06], saturation: 1.06 },
  { id: "bleach", name: "Bleach", lift: [-.012, -.012, -.008], gain: [1.14, 1.14, 1.12], gamma: [.86, .86, .88], saturation: 0.46 },
  { id: "teal", name: "Teal Night", lift: [-.010, .006, .030], gain: [1.02, .99, .98], gamma: [1.04, 1.00, .92], saturation: 0.94 },
  { id: "mono", name: "Mono 400", lift: [.014, .014, .016], gain: [1.02, 1.02, 1.02], gamma: [.95, .95, .95], saturation: 0 },
  { id: "faded", name: "Faded Print", lift: [.048, .044, .052], gain: [.90, .90, .90], gamma: [1.02, 1.02, 1.00], saturation: 0.78 },
  { id: "sun", name: "Sun Bleach", lift: [.020, .012, -.004], gain: [1.10, 1.02, .90], gamma: [.90, .96, 1.10], saturation: 0.88 },
];

export const photoFilter = (id: string): PhotoFilter =>
  PHOTO_FILTERS.find((filter) => filter.id === id) ?? PHOTO_FILTERS[0];

/** Rec. 709 luma, matching the working space. */
const LUMA_R = 0.2126, LUMA_G = 0.7152, LUMA_B = 0.0722;

/** How finely the linear range is sampled on the way back out. 4096 keeps shadow banding invisible. */
export const ENCODE_STEPS = 4096;
/** Linear values above this clip to white, so the table only has to cover a sensible highlight range. */
export const ENCODE_CEILING = 4;

/** Exported so the GPU renderer uploads this exact float32 table instead of recomputing it. */
export const srgbToLinear = (() => {
  const table = new Float32Array(256);
  for (let index = 0; index < 256; index++) {
    const value = index / 255;
    table[index] = value <= 0.04045 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
  }
  return table;
})();

const linearToSrgb = (value: number) =>
  value <= 0.0031308 ? 12.92 * value : 1.055 * Math.pow(value, 1 / 2.4) - 0.055;

const clamp01 = (value: number) => value < 0 ? 0 : value > 1 ? 1 : value;

/**
 * True when the grade would not change a single pixel.
 *
 * Worth having as its own check rather than letting the maths work it out: it makes "no adjustments"
 * bit-exact rather than merely close, which is what "hold to compare" and an untouched save both
 * need to be trustworthy.
 */
export function isNeutralGrade(adjustments: PhotoAdjustments): boolean {
  const neutralCurve = adjustments.curve.length === NEUTRAL_CURVE.length
    && adjustments.curve.every((point, index) =>
      Math.abs(point.x - NEUTRAL_CURVE[index].x) < 1e-4 && Math.abs(point.y - NEUTRAL_CURVE[index].y) < 1e-4);
  const neutralLevels = adjustments.levelsBlack === 0 && adjustments.levelsWhite === 255
    && Math.abs(adjustments.levelsGamma - 1) < 1e-4;
  const neutralBands = !adjustments.hsl?.some(
    (band) => band.hue !== 0 || band.saturation !== 0 || band.luminance !== 0);
  return neutralCurve && neutralLevels && neutralBands
    && adjustments.exposure === 0 && adjustments.contrast === 0
    && adjustments.highlights === 0 && adjustments.shadows === 0
    && adjustments.temperature === 0 && adjustments.tint === 0
    && adjustments.saturation === 100 && adjustments.vibrance === 0
    && adjustments.sharpen === 0 && adjustments.blur === 0 && adjustments.grain === 0
    && (adjustments.denoise ?? 0) === 0 && (adjustments.denoiseColor ?? 0) === 0
    && adjustments.vignette === 0
    && (adjustments.filter === "none" || adjustments.filterAmount === 0);
}

/** True when anything in the grade changes the pixel grid rather than the pixel values. */
export function hasGeometry(adjustments: PhotoAdjustments): boolean {
  return adjustments.rotate !== 0 || adjustments.flipHorizontal || adjustments.flipVertical;
}

/**
 * The tone curve as a 256-entry table.
 *
 * Interpolation is Catmull-Rom through the pinned ends and the interior points, then forced
 * non-decreasing. A curve that dips backwards is never what someone dragging a point meant, and
 * letting it happen produces posterised bands that look like a bug in the renderer.
 */
export function buildCurveLut(points: CurvePoint[]): Float32Array {
  const knots = [{ x: 0, y: 0 }, ...points.map((point) => ({ ...point })), { x: 1, y: 1 }]
    .sort((left, right) => left.x - right.x);
  const table = new Float32Array(256);
  for (let index = 0; index < 256; index++) {
    const at = index / 255;
    let span = 0;
    while (span < knots.length - 2 && at > knots[span + 1].x) span++;
    const previous = knots[Math.max(0, span - 1)];
    const start = knots[span];
    const end = knots[span + 1];
    const next = knots[Math.min(knots.length - 1, span + 2)];
    const width = (end.x - start.x) || 1e-6;
    const t = Math.min(1, Math.max(0, (at - start.x) / width));
    // At the ends there is no point on the far side, and the usual trick of reusing the endpoint
    // halves the tangent -- which bows the identity curve away from the diagonal by about two
    // levels at the quarter tones. A one-sided difference there keeps a straight line straight.
    const m1 = span === 0 ? end.y - start.y : (end.y - previous.y) / 2;
    const m2 = span === knots.length - 2 ? end.y - start.y : (next.y - start.y) / 2;
    const t2 = t * t, t3 = t2 * t;
    table[index] = clamp01(
      (2 * t3 - 3 * t2 + 1) * start.y + (t3 - 2 * t2 + t) * m1
      + (-2 * t3 + 3 * t2) * end.y + (t3 - t2) * m2,
    );
  }
  for (let index = 1; index < 256; index++) {
    if (table[index] < table[index - 1]) table[index] = table[index - 1];
  }
  return table;
}

/**
 * One table per channel mapping a linear value to a finished byte: sRGB encode, then the curve,
 * then the filter's gamma / gain / lift.
 *
 * Folding them together is what keeps the per-pixel loop cheap -- the curve and every filter cost
 * one array lookup between them, no matter how many of them are switched on.
 */
export function buildOutputLuts(adjustments: PhotoAdjustments): [Uint8Array, Uint8Array, Uint8Array] {
  const curve = buildCurveLut(adjustments.curve);
  const filter = photoFilter(adjustments.filter);
  const mix = Math.min(1, Math.max(0, adjustments.filterAmount / 100));
  const blackPoint = Math.min(250, Math.max(0, adjustments.levelsBlack ?? 0));
  const whitePoint = Math.max(blackPoint + 5, Math.min(255, adjustments.levelsWhite ?? 255));
  const gamma = Math.min(4, Math.max(0.1, adjustments.levelsGamma ?? 1));
  const span = (whitePoint - blackPoint) / 255;
  const tables: [Uint8Array, Uint8Array, Uint8Array] = [
    new Uint8Array(ENCODE_STEPS), new Uint8Array(ENCODE_STEPS), new Uint8Array(ENCODE_STEPS),
  ];
  for (let channel = 0; channel < 3; channel++) {
    const lift = filter.lift[channel] * mix;
    const gain = 1 + (filter.gain[channel] - 1) * mix;
    const gamma = 1 + (filter.gamma[channel] - 1) * mix;
    const table = tables[channel];
    for (let index = 0; index < ENCODE_STEPS; index++) {
      const linear = index / (ENCODE_STEPS - 1) * ENCODE_CEILING;
      let value = clamp01(linearToSrgb(linear));
      // Levels first: it decides which part of the range the curve then gets to shape. Doing it
      // afterwards would mean the curve's control points no longer sit where the user put them.
      if (blackPoint > 0 || whitePoint < 255 || gamma !== 1) {
        value = clamp01((value - blackPoint / 255) / span);
        if (gamma !== 1) value = Math.pow(value, 1 / gamma);
      }
      value = curve[Math.round(value * 255)];
      value = Math.pow(clamp01(value), gamma);
      value = lift + gain * value;
      table[index] = Math.round(clamp01(value) * 255);
    }
  }
  return tables;
}

/**
 * Apply the whole grade to an RGBA buffer, in place.
 *
 * Order is deliberate and matches how the panel is laid out top to bottom: exposure and white
 * balance are scene-referred and come first, recovery and contrast reshape the range, saturation
 * works on the reshaped result, and the look is applied last on the way to display. Spatial passes
 * follow, because sharpening a picture before you have decided its contrast sharpens the wrong
 * edges, and grain is added after everything so the filter cannot smear it.
 */
export function applyAdjustments(
  data: Uint8ClampedArray, width: number, height: number, adjustments: PhotoAdjustments,
): void {
  if (isNeutralGrade(adjustments)) return;

  const exposure = Math.pow(2, adjustments.exposure);
  const contrast = 1 + adjustments.contrast / 100;
  const highlights = adjustments.highlights / 100;
  const shadows = adjustments.shadows / 100;
  const warm = adjustments.temperature / 100;
  const green = adjustments.tint / 100;
  const filter = photoFilter(adjustments.filter);
  const mix = Math.min(1, Math.max(0, adjustments.filterAmount / 100));
  const saturation = (adjustments.saturation / 100) * (1 + (filter.saturation - 1) * mix);
  const vibrance = adjustments.vibrance / 100;
  const gainR = exposure * (1 + 0.16 * warm);
  const gainG = exposure * (1 - 0.09 * green);
  const gainB = exposure * (1 - 0.16 * warm);
  const [lutR, lutG, lutB] = buildOutputLuts(adjustments);
  const scale = (ENCODE_STEPS - 1) / ENCODE_CEILING;
  const last = ENCODE_STEPS - 1;

  const bands = adjustments.hsl ?? [];
  const bandsActive = bands.some((band) => band.hue || band.saturation || band.luminance);
  const weights = new Float32Array(HSL_CENTRES.length);

  const vignette = (adjustments.vignette ?? 0) / 100;
  const centreX = width / 2, centreY = height / 2;
  const maxRadius = Math.hypot(centreX, centreY) || 1;
  const vignetteStart = (adjustments.vignetteMidpoint ?? 55) / 100;
  const vignetteSpread = Math.max(0.02, (adjustments.vignetteFeather ?? 60) / 100);

  for (let pixel = 0, index = 0; index < data.length; index += 4, pixel++) {
    let r = srgbToLinear[data[index]] * gainR;
    let g = srgbToLinear[data[index + 1]] * gainG;
    let b = srgbToLinear[data[index + 2]] * gainB;

    if (highlights !== 0 || shadows !== 0) {
      const luma = r * LUMA_R + g * LUMA_G + b * LUMA_B;
      // Two soft masks that overlap in the midtones, so pushing one end leaves the other alone.
      const top = Math.min(1, luma * 1.9);
      const factor = 1 + highlights * 0.55 * top + shadows * 0.55 * (1 - top) * (1 - top);
      r *= factor; g *= factor; b *= factor;
    }
    if (contrast !== 1) {
      r = (r - 0.16) * contrast + 0.16;
      g = (g - 0.16) * contrast + 0.16;
      b = (b - 0.16) * contrast + 0.16;
    }
    if (saturation !== 1 || vibrance !== 0) {
      const luma = r * LUMA_R + g * LUMA_G + b * LUMA_B;
      let amount = saturation;
      if (vibrance !== 0) {
        // Weight by how dull the pixel already is, so vibrance leaves saturated colour alone.
        const peak = Math.max(r, g, b);
        const dullness = peak <= 0 ? 1 : 1 - (peak - Math.min(r, g, b)) / peak;
        amount += vibrance * dullness;
      }
      r = luma + (r - luma) * amount;
      g = luma + (g - luma) * amount;
      b = luma + (b - luma) * amount;
    }
    if (bandsActive) {
      const total = bandWeights(r, g, b, weights);
      if (total > 0) {
        let hueShift = 0, saturate = 0, lighten = 0;
        for (let band = 0; band < weights.length; band++) {
          const weight = weights[band] / total;
          if (weight === 0) continue;
          hueShift += weight * bands[band].hue;
          saturate += weight * bands[band].saturation;
          lighten += weight * bands[band].luminance;
        }
        // Hue is rotated about the luma axis rather than through an HSL round trip: same result for
        // a rotation, and it keeps the whole band pass to arithmetic on the three channels.
        if (hueShift !== 0) {
          const angle = hueShift * 0.3 * Math.PI / 180;
          const cos = Math.cos(angle), sin = Math.sin(angle);
          const nr = r * (0.213 + cos * 0.787 - sin * 0.213) + g * (0.715 - cos * 0.715 - sin * 0.715) + b * (0.072 - cos * 0.072 + sin * 0.928);
          const ng = r * (0.213 - cos * 0.213 + sin * 0.143) + g * (0.715 + cos * 0.285 + sin * 0.140) + b * (0.072 - cos * 0.072 - sin * 0.283);
          const nb = r * (0.213 - cos * 0.213 - sin * 0.787) + g * (0.715 - cos * 0.715 + sin * 0.715) + b * (0.072 + cos * 0.928 + sin * 0.072);
          r = nr; g = ng; b = nb;
        }
        if (saturate !== 0) {
          const luma = r * LUMA_R + g * LUMA_G + b * LUMA_B;
          const amount = 1 + saturate / 100;
          r = luma + (r - luma) * amount;
          g = luma + (g - luma) * amount;
          b = luma + (b - luma) * amount;
        }
        if (lighten !== 0) {
          const amount = 1 + lighten / 100;
          r *= amount; g *= amount; b *= amount;
        }
      }
    }
    if (vignette !== 0) {
      const x = pixel % width, y = (pixel / width) | 0;
      const distance = Math.hypot(x - centreX, y - centreY) / maxRadius;
      const t = Math.min(1, Math.max(0, (distance - vignetteStart) / vignetteSpread));
      // Smoothstep, so the corner falloff has no visible edge where it begins.
      const falloff = t * t * (3 - 2 * t);
      const factor = 1 + vignette * falloff;
      r *= factor; g *= factor; b *= factor;
    }

    const ir = r <= 0 ? 0 : Math.min(last, (r * scale) | 0);
    const ig = g <= 0 ? 0 : Math.min(last, (g * scale) | 0);
    const ib = b <= 0 ? 0 : Math.min(last, (b * scale) | 0);
    data[index] = lutR[ir];
    data[index + 1] = lutG[ig];
    data[index + 2] = lutB[ib];
  }

  // Before the blur and well before the sharpen: denoising after a sharpen is trying to remove
  // grain that has just been amplified, which never works and costs the detail twice.
  if ((adjustments.denoise ?? 0) > 0 || (adjustments.denoiseColor ?? 0) > 0) {
    denoiseBuffer(data, width, height,
      adjustments.denoise ?? 0, adjustments.denoiseDetail ?? 0, adjustments.denoiseColor ?? 0);
  }
  if (adjustments.blur > 0) blurBuffer(data, width, height, adjustments.blur / 100 * 12);
  if (adjustments.sharpen > 0) sharpenBuffer(data, width, height, adjustments.sharpen / 100);
  if (adjustments.grain > 0) addGrain(data, width, adjustments.grain / 100 * 26);
}

/**
 * Weight each colour band for one pixel, writing into `into` and returning the total.
 *
 * The centres are unevenly spaced, so a fixed-width falloff is used and the weights are normalised
 * by the caller. A grey pixel has no hue and gets no weight at all, which is what keeps the bands
 * off skies and skin that happen to sit between two centres.
 */
function bandWeights(r: number, g: number, b: number, into: Float32Array): number {
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const delta = max - min;
  into.fill(0);
  if (delta <= 1e-6) return 0;
  let hue: number;
  if (max === r) hue = ((g - b) / delta) % 6;
  else if (max === g) hue = (b - r) / delta + 2;
  else hue = (r - g) / delta + 4;
  hue *= 60;
  if (hue < 0) hue += 360;
  // Desaturated pixels should barely move, so chroma scales the whole contribution.
  const chroma = Math.min(1, delta / Math.max(1e-6, max));
  let total = 0;
  for (let band = 0; band < HSL_CENTRES.length; band++) {
    let distance = Math.abs(hue - HSL_CENTRES[band]);
    if (distance > 180) distance = 360 - distance;
    if (distance >= HSL_BAND_WIDTH) continue;
    const weight = (1 - distance / HSL_BAND_WIDTH) * chroma;
    into[band] = weight;
    total += weight;
  }
  return total;
}

/** How far either side of a centre a band reaches, in degrees. */
export const HSL_BAND_WIDTH = 40;

/**
 * Counts per bin for the three channels and luma, for the histogram display.
 *
 * Read off the finished preview buffer rather than recomputed from the source, so what it shows is
 * what is actually on screen -- including everything the stack did to get there.
 */
export function histogram(data: Uint8ClampedArray, bins = 64): {
  red: Uint32Array; green: Uint32Array; blue: Uint32Array; luma: Uint32Array; peak: number;
} {
  const red = new Uint32Array(bins), green = new Uint32Array(bins);
  const blue = new Uint32Array(bins), luma = new Uint32Array(bins);
  const scale = bins / 256;
  for (let index = 0; index < data.length; index += 4) {
    const r = data[index], g = data[index + 1], b = data[index + 2];
    red[Math.min(bins - 1, (r * scale) | 0)]++;
    green[Math.min(bins - 1, (g * scale) | 0)]++;
    blue[Math.min(bins - 1, (b * scale) | 0)]++;
    luma[Math.min(bins - 1, ((r * 0.2126 + g * 0.7152 + b * 0.0722) * scale) | 0)]++;
  }
  let peak = 1;
  for (let bin = 0; bin < bins; bin++) {
    peak = Math.max(peak, red[bin], green[bin], blue[bin], luma[bin]);
  }
  return { red, green, blue, luma, peak };
}

/**
 * Temperature and tint that make the sampled pixel neutral.
 *
 * The grade multiplies red and blue against temperature and green against tint, so solving for grey
 * is just inverting those two gains. Returned rather than applied, because the caller has to fold it
 * into whichever node the user is editing.
 */
export function whiteBalanceFor(r: number, g: number, b: number): { temperature: number; tint: number } {
  const red = Math.max(1, r), green = Math.max(1, g), blue = Math.max(1, b);
  // Warmth is the red/blue imbalance; the 0.16 mirrors the gain the pipeline applies.
  const temperature = clamp(((blue - red) / Math.max(red, blue)) / 0.16 * 100, -100, 100);
  const grey = (red + blue) / 2;
  const tint = clamp(((green - grey) / Math.max(green, grey)) / 0.09 * 100, -100, 100);
  return { temperature: Math.round(temperature), tint: Math.round(tint) };
}

const clamp = (value: number, low: number, high: number) => value < low ? low : value > high ? high : value;

/**
 * Noise reduction, by shrinking the detail that looks like noise.
 *
 * Not a bilateral filter, which is the obvious choice and the wrong one here: a real bilateral is
 * quadratic in its radius and this runs on every frame of a slider drag at full preview size.
 *
 * Instead the picture is split into a blurred base and the detail on top of it, and each detail
 * value is scaled by `d² / (d² + σ²)` -- near zero for small wobbles, near one for real edges. That
 * is Wiener shrinkage, it is linear in the pixel count, and it has the property that matters: it
 * cannot invent an edge, only fade one that was already weak. `detail` then hands back a share of
 * what was taken, because a picture with every fine texture removed reads as plastic.
 *
 * Luma and chroma are separated and treated differently on purpose. Colour noise sits in wide, soft
 * blotches and needs a radius that would destroy luminance detail; luminance noise is fine and
 * needs barely any. Smoothing them together is why one-slider denoise looks like wax.
 */
export function denoiseBuffer(
  data: Uint8ClampedArray, width: number, height: number,
  amount: number, detail: number, colour: number,
): void {
  const luma = Math.min(1, Math.max(0, amount / 100));
  const chroma = Math.min(1, Math.max(0, colour / 100));
  if (luma <= 0 && chroma <= 0) return;
  const count = width * height;
  if (count < 4) return;
  const keep = Math.min(1, Math.max(0, detail / 100));

  // Six float planes at preview size is around thirty megabytes, and this runs on every frame of a
  // slider drag. Allocating them each time is the kind of garbage that shows up as stutter rather
  // than as slowness, so the set is kept and reused.
  const [y, cb, cr, scratch, spareA, spareB] = planes(count);
  for (let index = 0, at = 0; index < count; index++, at += 4) {
    // Rec. 709 luma, the same coefficients the rest of this file uses. Consistency matters more
    // than the choice: two different lumas in one pipeline is a bug waiting for a rotation.
    const value = data[at] * LUMA_R + data[at + 1] * LUMA_G + data[at + 2] * LUMA_B;
    y[index] = value;
    cb[index] = data[at + 2] - value;
    cr[index] = data[at] - value;
  }

  if (luma > 0) {
    const base = spareA;
    base.set(y);
    blurPlane(base, scratch, width, height, Math.max(1, Math.round(1 + luma * 2)));
    // Sigma in 0-255 units. Eight is about where sensor grain lives; the slider reaches well past
    // what any real picture needs, because "too much" has to be available to be judged.
    shrink(y, base, 2 + luma * 16, keep);
  }
  if (chroma > 0) {
    const radius = Math.max(1, Math.round(2 + chroma * 10));
    const baseCb = spareA;
    const baseCr = spareB;
    baseCb.set(cb);
    baseCr.set(cr);
    blurPlane(baseCb, scratch, width, height, radius);
    blurPlane(baseCr, scratch, width, height, radius);
    // Chroma keeps none of its detail back: colour texture at this scale is always noise, and the
    // edges that matter are carried by luma anyway.
    shrink(cb, baseCb, 3 + chroma * 30, 0);
    shrink(cr, baseCr, 3 + chroma * 30, 0);
  }

  for (let index = 0, at = 0; index < count; index++, at += 4) {
    const value = y[index];
    const red = value + cr[index];
    const blue = value + cb[index];
    // Green falls out of the luma identity, so the round trip is exact when nothing was changed.
    const green = (value - red * LUMA_R - blue * LUMA_B) / LUMA_G;
    data[at] = red;
    data[at + 1] = green;
    data[at + 2] = blue;
  }
}

/**
 * Six working planes of the right length, reused between calls.
 *
 * Grown but never shrunk: the sizes in play are the preview and the full render, and holding the
 * larger of the two costs less than reallocating on every switch between them.
 */
let pool: Float32Array[] = [];
function planes(count: number): Float32Array[] {
  if (!pool.length || pool[0].length < count) {
    pool = Array.from({ length: 6 }, () => new Float32Array(count));
  }
  return pool.map((plane) => plane.length === count ? plane : plane.subarray(0, count));
}

/** Move each value toward its local average by how much it looks like noise rather than structure. */
function shrink(plane: Float32Array, base: Float32Array, sigma: number, keep: number): void {
  const variance = sigma * sigma;
  for (let index = 0; index < plane.length; index++) {
    const smooth = base[index];
    const detail = plane[index] - smooth;
    const weight = detail * detail / (detail * detail + variance);
    plane[index] = smooth + detail * (weight + (1 - weight) * keep);
  }
}

/** The box blur above, for a single-channel float plane. Same transpose trick, same three passes. */
function blurPlane(
  plane: Float32Array, scratch: Float32Array, width: number, height: number, radius: number,
): void {
  for (let pass = 0; pass < 3; pass++) {
    planePass(plane, scratch, width, height, radius);
    planePass(scratch, plane, height, width, radius);
  }
}

function planePass(
  source: Float32Array, target: Float32Array, width: number, height: number, radius: number,
): void {
  const window = radius * 2 + 1;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let total = 0;
    for (let x = -radius; x <= radius; x++) total += source[row + Math.min(width - 1, Math.max(0, x))];
    for (let x = 0; x < width; x++) {
      target[x * height + y] = total / window;
      total += source[row + Math.min(width - 1, Math.max(0, x + radius + 1))]
        - source[row + Math.min(width - 1, Math.max(0, x - radius))];
    }
  }
}

/**
 * Separable box blur, run three times, which converges on a Gaussian closely enough that nobody can
 * tell and costs a fraction of a real convolution. Alpha is left alone: blurring it would eat into
 * a cutout's edge, and every picture that reaches here from Image Studio may be a cutout.
 */
export function blurBuffer(data: Uint8ClampedArray, width: number, height: number, radius: number): void {
  const size = Math.max(1, Math.round(radius));
  if (size < 1) return;
  const scratch = new Uint8ClampedArray(data.length);
  for (let pass = 0; pass < 3; pass++) {
    boxPass(data, scratch, width, height, size);
    boxPass(scratch, data, height, width, size);
  }
}

/**
 * One axis of the box blur, written out transposed. Running it twice with the dimensions swapped
 * covers both axes and lands back in the original orientation, so there is no separate vertical
 * implementation to keep in step with this one.
 */
function boxPass(
  source: Uint8ClampedArray, target: Uint8ClampedArray,
  width: number, height: number, radius: number,
): void {
  const window = radius * 2 + 1;
  for (let y = 0; y < height; y++) {
    const row = y * width * 4;
    for (let channel = 0; channel < 3; channel++) {
      let total = 0;
      for (let x = -radius; x <= radius; x++) {
        total += source[row + Math.min(width - 1, Math.max(0, x)) * 4 + channel];
      }
      for (let x = 0; x < width; x++) {
        // Transposed write: column x of the output row becomes row x, column y.
        target[(x * height + y) * 4 + channel] = total / window;
        const leaving = source[row + Math.min(width - 1, Math.max(0, x - radius)) * 4 + channel];
        const entering = source[row + Math.min(width - 1, Math.max(0, x + radius + 1)) * 4 + channel];
        total += entering - leaving;
      }
    }
    for (let x = 0; x < width; x++) {
      target[(x * height + y) * 4 + 3] = source[row + x * 4 + 3];
    }
  }
}

/**
 * Unsharp mask: the picture plus its own difference from a blurred copy.
 *
 * The blur radius is fixed at one pixel rather than exposed. A radius control is the difference
 * between sharpening and local contrast, and offering both without explaining them produces halos.
 */
export function sharpenBuffer(data: Uint8ClampedArray, width: number, height: number, amount: number): void {
  const blurred = new Uint8ClampedArray(data);
  blurBuffer(blurred, width, height, 1);
  const strength = amount * 1.4;
  for (let index = 0; index < data.length; index += 4) {
    for (let channel = 0; channel < 3; channel++) {
      const original = data[index + channel];
      data[index + channel] = original + (original - blurred[index + channel]) * strength;
    }
  }
}

/**
 * Monochrome grain. Deterministic from the pixel position so a redraw of the same picture produces
 * the same grain -- otherwise the preview would crawl every time a slider moved.
 */
export function addGrain(data: Uint8ClampedArray, width: number, strength: number): void {
  for (let index = 0, pixel = 0; index < data.length; index += 4, pixel++) {
    const x = pixel % width, y = (pixel / width) | 0;
    const noise = (Math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1;
    const shift = (noise - 0.5) * strength;
    data[index] += shift;
    data[index + 1] += shift;
    data[index + 2] += shift;
  }
}
