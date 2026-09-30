/**
 * Colour ramps.
 *
 * One definition serves two tools that look unrelated and are the same thing: a gradient FILL paints
 * the ramp across the frame in a direction, and a gradient MAP indexes the same ramp by how bright
 * each pixel already is. Writing the ramp twice would be this project's usual bug -- two
 * implementations of one piece of maths, agreeing until the day they do not -- so the ramp is built
 * once as a lookup table and both tools read it.
 *
 * Interpolation defaults to OKLab rather than sRGB, and that is the single decision here that most
 * changes what the tool produces. Mixing #0040ff into #ffcc00 channel by channel passes through a
 * dead grey-brown around the middle, because sRGB's numbers are not spaced the way seeing is. OKLab
 * is built so that equal steps look equal, and the same two colours cross through a live teal
 * instead. sRGB is kept as an option because matching a gradient made somewhere else sometimes
 * matters more than making the better one.
 */

export type GradientStop = {
  /** Where the stop sits along the ramp, 0 to 1. */
  at: number;
  /** "#rrggbb". */
  color: string;
  /** 0 to 1. Stops carry their own alpha, which is what makes a fade-to-nothing possible. */
  opacity: number;
};

/**
 * How the ramp is laid across the frame.
 *
 * `linear` runs one way; `reflected` mirrors it about the centre line, which is the one to reach for
 * when a light source is IN the frame rather than off it. `angular` sweeps around, `diamond` is the
 * square-cornered radial, and `radial` is the obvious one.
 */
export type GradientKind = "linear" | "reflected" | "radial" | "angular" | "diamond";
export type GradientSpace = "oklab" | "rgb";

export type Gradient = {
  stops: GradientStop[];
  kind: GradientKind;
  /** Degrees, clockwise, 0 pointing right. The direction the ramp travels. */
  angle: number;
  /** The point the ramp starts from, or turns about, normalised to the canvas. */
  cx: number;
  cy: number;
  /** 1 spans the frame along the chosen direction. Smaller compresses the ramp, larger stretches it. */
  scale: number;
  /** Slides the whole ramp along its axis without moving any stop. */
  offset: number;
  /** How many times the ramp repeats across that span. 1 clamps at both ends instead. */
  repeat: number;
  /** Ping-pong rather than restart, so a repeating ramp has no hard seam. */
  mirror: boolean;
  space: GradientSpace;
  /** Bends the ramp toward one end without moving stops. 1 is even. */
  ease: number;
  /**
   * Ordered dithering, 0 to 1.
   *
   * Not a nicety. A smooth ramp across two thousand pixels of an eight-bit picture has fewer
   * available values than it has pixels to cross, so it bands into visible stripes -- the single
   * most recognisable tell of a gradient made in software. Half a level of noise, laid out in a
   * fixed pattern rather than randomly, removes them completely and is invisible itself.
   */
  dither: number;
};

export type VisibleGradientHandle = {
  at: [number, number];
  clipped: boolean;
};

/**
 * Keep an editor handle reachable without changing the gradient it represents.
 *
 * Gradient centres intentionally range beyond the picture and a large spread can put the direction
 * handle several canvases away. The renderer must retain those values, but a clipped canvas cannot
 * offer an off-screen handle back to the user. This returns an edge proxy for the editor only; the
 * first drag immediately writes the real pointer position back into gradient space.
 */
export function visibleGradientHandle(point: [number, number], inset = 0.025): VisibleGradientHandle {
  const safeInset = clamp(inset, 0, 0.49);
  const x = clamp(Number.isFinite(point[0]) ? point[0] : 0.5, safeInset, 1 - safeInset);
  const y = clamp(Number.isFinite(point[1]) ? point[1] : 0.5, safeInset, 1 - safeInset);
  return { at: [x, y], clipped: Math.abs(x - point[0]) > 1e-6 || Math.abs(y - point[1]) > 1e-6 };
}

export const NEUTRAL_GRADIENT: Gradient = {
  stops: [
    { at: 0, color: "#000000", opacity: 1 },
    { at: 1, color: "#ffffff", opacity: 1 },
  ],
  kind: "linear",
  angle: 90,
  cx: 0.5, cy: 0.5,
  scale: 1, offset: 0,
  repeat: 1, mirror: false,
  space: "oklab",
  ease: 1,
  dither: 0.7,
};

/** Ready-made ramps. Starting points, not a palette: every one is editable the moment it lands. */
export const GRADIENT_PRESETS: Array<{ name: string; gradient: Partial<Gradient> }> = [
  { name: "Black to clear", gradient: { stops: [
    { at: 0, color: "#000000", opacity: 1 }, { at: 1, color: "#000000", opacity: 0 }] } },
  { name: "Sunset", gradient: { stops: [
    { at: 0, color: "#1b1035", opacity: 1 }, { at: 0.45, color: "#c8365b", opacity: 1 },
    { at: 0.75, color: "#ff8e2b", opacity: 1 }, { at: 1, color: "#ffd98a", opacity: 1 }] } },
  { name: "Dusk sky", gradient: { stops: [
    { at: 0, color: "#0b2a4a", opacity: 1 }, { at: 0.55, color: "#3f7fa6", opacity: 1 },
    { at: 1, color: "#f2c17a", opacity: 1 }] } },
  { name: "Teal and orange", gradient: { stops: [
    { at: 0, color: "#08313d", opacity: 1 }, { at: 0.5, color: "#2f8b93", opacity: 1 },
    { at: 1, color: "#ff9a3c", opacity: 1 }] } },
  { name: "Duotone", gradient: { stops: [
    { at: 0, color: "#101b3a", opacity: 1 }, { at: 1, color: "#ffd7a1", opacity: 1 }] } },
  { name: "Spectrum", gradient: { kind: "angular", stops: [
    { at: 0, color: "#ff3b3b", opacity: 1 }, { at: 0.17, color: "#ffd53b", opacity: 1 },
    { at: 0.34, color: "#4bff6a", opacity: 1 }, { at: 0.5, color: "#3bf0ff", opacity: 1 },
    { at: 0.67, color: "#3b6dff", opacity: 1 }, { at: 0.84, color: "#c23bff", opacity: 1 },
    { at: 1, color: "#ff3b3b", opacity: 1 }] } },
  { name: "Spotlight", gradient: { kind: "radial", stops: [
    { at: 0, color: "#ffffff", opacity: 0 }, { at: 1, color: "#000000", opacity: 0.85 }] } },
];

/* --------------------------------------------------------------- the paint */

/**
 * How anything is coloured: one flat colour, or a whole ramp.
 *
 * Introduced because "colour" turned out to be the wrong word in five places at once -- a text fill,
 * its outline, its drop shadow, a panel and a shape all wanted a gradient and all had a single hex
 * string. Rather than five gradient flags, there is one thing a colour can be, and every one of them
 * takes it.
 *
 * A gradient paint is drawn by the SAME renderer that fills a gradient layer, handed to the canvas
 * as a pattern. That is the point: a ramp on a headline and a ramp across the frame are one
 * implementation, so they cannot drift into looking different.
 */
export type Paint =
  | { kind: "solid"; color: string }
  | { kind: "gradient"; gradient: Gradient };

export const solidPaint = (color: string): Paint => ({ kind: "solid", color });

/**
 * Whatever was stored, as a paint.
 *
 * Documents written before this existed hold plain hex strings, and they are in project files on
 * disk. Read through rather than migrated, the same way single-source documents became layers:
 * nothing has to be upgraded, and nothing can be half upgraded.
 */
export function toPaint(value: unknown, fallback = "#ffffff"): Paint {
  if (typeof value === "string") return solidPaint(value);
  if (value && typeof value === "object") {
    const paint = value as Paint;
    if (paint.kind === "gradient" && paint.gradient) return paint;
    if (paint.kind === "solid" && typeof paint.color === "string") return paint;
  }
  return solidPaint(fallback);
}

/** A representative colour, for the places that can only take one -- a swatch, a CSS border. */
export function paintColor(paint: Paint): string {
  if (paint.kind === "solid") return paint.color;
  const stops = normalizeStops(paint.gradient.stops);
  return stops[Math.floor(stops.length / 2)].color;
}

export const isBlank = (paint: Paint): boolean =>
  paint.kind === "gradient" && normalizeStops(paint.gradient.stops).every((stop) => stop.opacity <= 0);

/* ---------------------------------------------------------------- the ramp */

/**
 * Stops in a state the sampler can rely on: sorted, in range, and at least two of them.
 *
 * Done once at the top of every entry point rather than defended against everywhere. A one-stop
 * ramp is a flat colour, which is a reasonable thing to have typed on the way to a second stop, so
 * it is completed rather than rejected.
 */
export function normalizeStops(stops: GradientStop[]): GradientStop[] {
  const cleaned = stops
    .filter((stop) => typeof stop?.color === "string")
    .map((stop) => ({
      at: clamp01(Number.isFinite(stop.at) ? stop.at : 0),
      color: stop.color,
      opacity: clamp01(Number.isFinite(stop.opacity) ? stop.opacity : 1),
    }))
    .sort((left, right) => left.at - right.at);
  if (!cleaned.length) return [{ at: 0, color: "#000000", opacity: 1 }, { at: 1, color: "#ffffff", opacity: 1 }];
  if (cleaned.length === 1) return [{ ...cleaned[0], at: 0 }, { ...cleaned[0], at: 1 }];
  return cleaned;
}

/** The ramp as an RGBA table. `steps` entries, evenly spaced from 0 to 1. */
export function buildGradientLut(gradient: Gradient, steps = 512): Uint8ClampedArray {
  const stops = normalizeStops(gradient.stops);
  const table = new Uint8ClampedArray(steps * 4);
  const oklab = gradient.space !== "rgb";
  // Converted once per stop rather than once per sample: a 512-entry table over eight stops would
  // otherwise do four thousand cube roots to produce the same numbers.
  const colors = stops.map((stop) => {
    const [r, g, b] = hexToRgb(stop.color);
    return oklab ? rgbToOklab(r, g, b) : ([r, g, b] as [number, number, number]);
  });

  let cursor = 0;
  for (let index = 0; index < steps; index++) {
    const t = steps === 1 ? 0 : index / (steps - 1);
    while (cursor < stops.length - 2 && t > stops[cursor + 1].at) cursor++;
    const from = stops[cursor];
    const to = stops[cursor + 1];
    const span = to.at - from.at;
    // Two stops at the same position are a hard edge, which is a legitimate thing to want; without
    // this guard it is a division by zero instead.
    const mix = span <= 1e-6 ? (t >= to.at ? 1 : 0) : clamp01((t - from.at) / span);
    const a = colors[cursor];
    const b = colors[cursor + 1];
    const x = a[0] + (b[0] - a[0]) * mix;
    const y = a[1] + (b[1] - a[1]) * mix;
    const z = a[2] + (b[2] - a[2]) * mix;
    const [r, g, bl] = oklab ? oklabToRgb(x, y, z) : [x, y, z];
    const at = index * 4;
    table[at] = r;
    table[at + 1] = g;
    table[at + 2] = bl;
    table[at + 3] = (from.opacity + (to.opacity - from.opacity) * mix) * 255;
  }
  return table;
}

/* ------------------------------------------------------------- the geometry */

/**
 * Where a pixel sits along the ramp, before repeat and easing.
 *
 * Extent is measured so that `scale` of 1 spans the frame ALONG THE CHOSEN DIRECTION -- the
 * projection of the frame's half-diagonal onto the axis, not the frame's width. Without that, a
 * gradient at forty-five degrees runs out well before the corners and every angle needs its own
 * scale to look the same, which is exactly the fiddling this is meant to remove.
 */
export function gradientPosition(
  gradient: Gradient, x: number, y: number, width: number, height: number,
): number {
  const radians = gradient.angle * Math.PI / 180;
  const cos = Math.cos(radians), sin = Math.sin(radians);
  const dx = x - gradient.cx * width;
  const dy = y - gradient.cy * height;
  // Into the ramp's own frame: u runs along the direction, v across it.
  const u = dx * cos + dy * sin;
  const v = -dx * sin + dy * cos;
  const scale = Math.max(0.01, gradient.scale);

  if (gradient.kind === "angular") {
    return (Math.atan2(v, u) / (Math.PI * 2) + 0.5) / scale + gradient.offset;
  }
  const along = Math.abs(width / 2 * cos) + Math.abs(height / 2 * sin);
  const across = Math.abs(width / 2 * sin) + Math.abs(height / 2 * cos);
  if (gradient.kind === "linear") {
    return u / (2 * Math.max(1e-6, along) * scale) + 0.5 + gradient.offset;
  }
  if (gradient.kind === "reflected") {
    return Math.abs(u) / (Math.max(1e-6, along) * scale) + gradient.offset;
  }
  if (gradient.kind === "diamond") {
    return (Math.abs(u) / Math.max(1e-6, along) + Math.abs(v) / Math.max(1e-6, across))
      / (2 * scale) + gradient.offset;
  }
  const radius = Math.hypot(width / 2, height / 2);
  return Math.hypot(u, v) / (Math.max(1e-6, radius) * scale) + gradient.offset;
}

/** Repeat, mirror and easing folded into one, so the fill and the map cannot disagree about them. */
export function rampParameter(gradient: Gradient, position: number): number {
  const repeat = Math.max(1, gradient.repeat);
  let p = position * repeat;
  if (repeat > 1) {
    // A repeating ramp wraps; a single one clamps. Ping-pong instead, and there is no seam where
    // the last stop meets the first.
    p = gradient.mirror ? pingPong(p) : p - Math.floor(p);
  } else {
    p = gradient.mirror ? pingPong(p) : clamp01(p);
  }
  const ease = gradient.ease;
  if (ease && Math.abs(ease - 1) > 1e-4) p = Math.pow(clamp01(p), Math.max(0.05, ease));
  return clamp01(p);
}

/* --------------------------------------------------------------- the fills */

/** An 8x8 ordered matrix. Fixed rather than random, so a still picture has no crawling grain. */
/** Exported for the GPU renderer, which uploads this exact table rather than rebuilding it. */
export const BAYER = buildBayer(8);

/**
 * Paint the ramp across a buffer, replacing whatever was there.
 *
 * Writes straight RGBA including alpha, because a gradient that fades to nothing is most of what
 * these are used for -- a sky graduate, a vignette, a wash that lets the plate through at one end.
 */
export function renderGradient(
  data: Uint8ClampedArray, width: number, height: number, gradient: Gradient,
): void {
  const lut = buildGradientLut(gradient);
  const steps = lut.length / 4;
  const last = steps - 1;
  const dither = clamp01(gradient.dither ?? 0);

  for (let y = 0, at = 0; y < height; y++) {
    for (let x = 0; x < width; x++, at += 4) {
      const p = rampParameter(gradient, gradientPosition(gradient, x + 0.5, y + 0.5, width, height));
      const index = Math.min(last, Math.max(0, Math.round(p * last))) * 4;
      if (dither > 0) {
        // Half a level either side, laid out by position. Below what anyone can see as noise, and
        // enough to break a band because neighbouring pixels round different ways.
        const wobble = (BAYER[(y & 7) * 8 + (x & 7)] - 0.5) * dither;
        data[at] = lut[index] + wobble;
        data[at + 1] = lut[index + 1] + wobble;
        data[at + 2] = lut[index + 2] + wobble;
      } else {
        data[at] = lut[index];
        data[at + 1] = lut[index + 1];
        data[at + 2] = lut[index + 2];
      }
      data[at + 3] = lut[index + 3];
    }
  }
}

/* ---------------------------------------------------------------- the map */

/**
 * Re-colour a picture by its own brightness.
 *
 * Shadows take the colour at one end of the ramp, highlights the colour at the other, and everything
 * between follows. This is duotone and split-toning, and it is the ramp doing a completely different
 * job with no second implementation: the same table, indexed by luminance rather than by position.
 *
 * Luminance is Rec. 709, the coefficients the rest of the pipeline uses. Alpha is left alone.
 */
export function applyGradientMap(
  data: Uint8ClampedArray, gradient: Gradient, amount: number,
): void {
  const strength = clamp01(amount / 100);
  if (strength <= 0) return;
  const lut = buildGradientLut(gradient, 256);
  for (let at = 0; at < data.length; at += 4) {
    const luma = data[at] * 0.2126 + data[at + 1] * 0.7152 + data[at + 2] * 0.0722;
    const p = rampParameter(gradient, clamp01(luma / 255));
    const index = Math.min(255, Math.max(0, Math.round(p * 255))) * 4;
    // The stop's own alpha weights how far this pixel travels, so a ramp that fades out also fades
    // out of the tone map rather than punching a transparent hole in the picture.
    const weight = strength * (lut[index + 3] / 255);
    data[at] += (lut[index] - data[at]) * weight;
    data[at + 1] += (lut[index + 1] - data[at + 1]) * weight;
    data[at + 2] += (lut[index + 2] - data[at + 2]) * weight;
  }
}

/* -------------------------------------------------------------- the colour */

/** "#rgb" or "#rrggbb" to 0-255. Anything unreadable is black, never NaN. */
export function hexToRgb(hex: string): [number, number, number] {
  const text = hex.trim().replace(/^#/, "");
  if (text.length === 3) {
    const r = parseInt(text[0] + text[0], 16);
    const g = parseInt(text[1] + text[1], 16);
    const b = parseInt(text[2] + text[2], 16);
    return [r || 0, g || 0, b || 0];
  }
  if (text.length !== 6) return [0, 0, 0];
  const value = parseInt(text, 16);
  if (!Number.isFinite(value)) return [0, 0, 0];
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export function rgbToHex(r: number, g: number, b: number): string {
  const part = (value: number) => Math.round(clamp(value, 0, 255)).toString(16).padStart(2, "0");
  return `#${part(r)}${part(g)}${part(b)}`;
}

const toLinear = (value: number): number => {
  const v = value / 255;
  return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
};
const toSrgb = (value: number): number => {
  const v = value <= 0.0031308 ? value * 12.92 : 1.055 * Math.pow(Math.max(0, value), 1 / 2.4) - 0.055;
  return clamp(v * 255, 0, 255);
};

/** Björn Ottosson's OKLab. Perceptually even, which is the whole reason it is here. */
export function rgbToOklab(r255: number, g255: number, b255: number): [number, number, number] {
  const r = toLinear(r255), g = toLinear(g255), b = toLinear(b255);
  const l = Math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b);
  const m = Math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b);
  const s = Math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b);
  return [
    0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
    1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
    0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
  ];
}

export function oklabToRgb(L: number, a: number, b: number): [number, number, number] {
  const l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3;
  const m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3;
  const s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3;
  return [
    toSrgb(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
    toSrgb(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
    toSrgb(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s),
  ];
}

/* --------------------------------------------------------------- plumbing */

const clamp = (value: number, low: number, high: number) =>
  value < low ? low : value > high ? high : value;
const clamp01 = (value: number) => clamp(value, 0, 1);
const pingPong = (value: number) => {
  const wrapped = Math.abs(value) % 2;
  return wrapped > 1 ? 2 - wrapped : wrapped;
};

/**
 * The recursive ordered-dither matrix, normalised to 0..1.
 *
 * Built by doubling rather than by bit-twiddling. The doubling rule is the definition of the thing
 * and reads as such; the clever version is four lines of shifts that nobody can check.
 */
function buildBayer(size: number): Float32Array {
  let matrix = [[0]];
  while (matrix.length < size) {
    const n = matrix.length;
    const next = Array.from({ length: n * 2 }, () => new Array<number>(n * 2).fill(0));
    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const value = matrix[y][x] * 4;
        next[y][x] = value;
        next[y][x + n] = value + 2;
        next[y + n][x] = value + 3;
        next[y + n][x + n] = value + 1;
      }
    }
    matrix = next;
  }
  const flat = new Float32Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) flat[y * size + x] = matrix[y][x] / (size * size);
  }
  return flat;
}
