/**
 * Shapes, and the balloons that are shapes with something to say.
 *
 * Every outline here is a POLYLINE -- a closed list of points -- and the rest of the file follows
 * from that. Beziers would be shorter to write and impossible to check: the only way to know a
 * balloon's tail meets its body would be to render one and look at it. Points can be counted,
 * measured, and asked whether they lie where they should, so the whole of this file is testable
 * without a canvas, which for the geometry under a drawing tool is worth far more than brevity.
 *
 * Sampling error is not a concern because outlines are generated at the shape's own pixel size: a
 * 400-pixel balloon sampled 192 times is accurate to about a hundredth of a pixel. Shapes with
 * corners emit their corners EXACTLY rather than hoping a sample lands on one, which is what keeps a
 * star's points sharp.
 *
 * There are four families and one mechanism. `bodyOutline` builds the closed body for every kind;
 * `bodyPoint` is a ray cast against that outline, so it is exact for a heart and a star alike; and
 * the tail is cut into the loop rather than added beside it, so a balloon and its tail arrive as one
 * closed path that strokes as a single line. Two paths would need their shared join painted over by
 * the fill, which works until the fill is semi-transparent and then draws a line through the middle
 * of the balloon.
 */

import { solidPaint, type Paint } from "./photoGradient";

export type Point = [number, number];

export type ShapeKind =
  /* Plain shapes. */
  | "rectangle" | "rounded" | "ellipse" | "polygon" | "star" | "burst" | "arrow" | "banner"
  /* Comic furniture: things that go behind and around the action. */
  | "heart" | "lightning" | "splat" | "impact" | "sunburst" | "speedlines"
  /* Balloons: bodies with a tail and somewhere to letter. */
  | "speech" | "thought" | "shout" | "scream" | "electric" | "icicle" | "wobble"
  | "whisper" | "caption";

export type ShapeBlock = {
  kind: ShapeKind;
  /** The body's size in canvas pixels, before the tail and the stroke are added. */
  width: number;
  height: number;
  /** 0 is a hard rectangle, 1 a full ellipse. Everything between is a rounded rectangle. */
  radius: number;
  /** Sides for a polygon, points for anything spiky, rays for a sunburst, lines for speed lines. */
  points: number;
  /** How far the spikes cut in, as a fraction of the outer radius. Low is jagged, high is gentle. */
  innerRadius: number;
  /** Turns the corners around the shape without turning the box they sit in. Degrees. */
  spin?: number;
  /**
   * How much the spikes vary in length, 0 to 1.
   *
   * The difference between a sale sticker and a comic. A regular starburst reads as printed; real
   * shout balloons have spikes of every length, and one number of irregularity is most of what
   * separates the two.
   */
  jitter?: number;
  /** Which irregularity. Fixed, so a shape does not crawl every time the frame is redrawn. */
  seed?: number;
  fill: Paint;
  stroke: Paint;
  strokeWidth: number;
  /** Dashes the outline. What makes a whisper balloon a whisper. */
  dashed: boolean;
  /** Which way the tail points, in degrees clockwise from right. */
  tailAngle: number;
  /** How far past the body it reaches, as a fraction of the body's radius in that direction. */
  tailLength: number;
  /** How wide it is where it leaves the body, in degrees of the body's own angle. */
  tailWidth: number;
  /** Bends the tail sideways, -1 to 1. A straight tail reads as a spike. */
  tailCurve: number;
  /** Space between the outline and the lettering inside it. */
  padding: number;
  shadow: number;
  shadowColor: string;
  shadowOffset: number;
};

export const NEUTRAL_SHAPE: ShapeBlock = {
  kind: "speech",
  width: 520, height: 300,
  radius: 1, points: 5, innerRadius: 0.5,
  spin: 0, jitter: 0, seed: 1,
  fill: solidPaint("#ffffff"),
  stroke: solidPaint("#000000"),
  strokeWidth: 6,
  dashed: false,
  tailAngle: 110, tailLength: 0.55, tailWidth: 22, tailCurve: 0.35,
  padding: 26,
  shadow: 0, shadowColor: "#000000", shadowOffset: 8,
};

/** The kinds that carry a tail, and so are balloons rather than shapes. */
export const BALLOON_KINDS: ShapeKind[] = [
  "speech", "thought", "shout", "scream", "electric", "icicle", "wobble", "whisper", "caption",
];

/** The kinds that are several separate strokes, and so cannot hold lettering. */
export const DECORATION_KINDS: ShapeKind[] = ["impact", "sunburst", "speedlines"];

export const hasTail = (shape: ShapeBlock): boolean =>
  BALLOON_KINDS.includes(shape.kind) && shape.tailLength > 0.02;
export const holdsText = (kind: ShapeKind): boolean => !DECORATION_KINDS.includes(kind);

/** Kinds whose outline is a run of corners rather than a curve, so the corners are emitted exactly. */
const CORNERED: ShapeKind[] = [
  "polygon", "star", "burst", "shout", "scream", "electric", "arrow", "banner", "lightning",
];

export const SHAPE_PRESETS: Array<{ name: string; group: string; shape: Partial<ShapeBlock> }> = [
  { name: "Speech", group: "Balloons", shape: {
    kind: "speech", radius: 1, strokeWidth: 6, dashed: false, tailCurve: 0.35 } },
  { name: "Thought", group: "Balloons", shape: {
    kind: "thought", radius: 1, strokeWidth: 5, dashed: false, tailCurve: 0 } },
  { name: "Shout", group: "Balloons", shape: {
    kind: "shout", points: 15, innerRadius: 0.7, jitter: 0.45, seed: 7, strokeWidth: 6,
    dashed: false, tailWidth: 16, tailCurve: 0.15 } },
  { name: "Scream", group: "Balloons", shape: {
    kind: "scream", points: 21, innerRadius: 0.5, jitter: 0.7, seed: 3, strokeWidth: 6,
    dashed: false, tailWidth: 12, tailCurve: 0 } },
  { name: "Broadcast", group: "Balloons", shape: {
    kind: "electric", points: 26, innerRadius: 0.86, jitter: 0.15, seed: 11, strokeWidth: 5,
    dashed: false, tailWidth: 14, tailCurve: 0 } },
  { name: "Dripping", group: "Balloons", shape: {
    kind: "icicle", points: 7, innerRadius: 0.68, radius: 0.55, strokeWidth: 5, dashed: false,
    tailAngle: 250, tailCurve: 0.3 } },
  { name: "Wobble", group: "Balloons", shape: {
    kind: "wobble", points: 9, innerRadius: 0.86, strokeWidth: 5, dashed: false, tailCurve: 0.5 } },
  { name: "Whisper", group: "Balloons", shape: {
    kind: "whisper", radius: 1, strokeWidth: 4, dashed: true, tailCurve: 0.2 } },
  { name: "Caption", group: "Balloons", shape: {
    kind: "caption", radius: 0.12, strokeWidth: 5, dashed: false, tailLength: 0 } },

  { name: "Box", group: "Shapes", shape: { kind: "rectangle", strokeWidth: 5, dashed: false } },
  { name: "Rounded", group: "Shapes", shape: { kind: "rounded", radius: 0.3, strokeWidth: 5, dashed: false } },
  { name: "Ellipse", group: "Shapes", shape: { kind: "ellipse", strokeWidth: 5, dashed: false } },
  { name: "Polygon", group: "Shapes", shape: { kind: "polygon", points: 6, spin: -90, strokeWidth: 5 } },
  { name: "Star", group: "Shapes", shape: {
    kind: "star", points: 5, innerRadius: 0.38, spin: -90, jitter: 0, strokeWidth: 5 } },
  { name: "Burst", group: "Shapes", shape: {
    kind: "burst", points: 16, innerRadius: 0.66, spin: -90, jitter: 0, strokeWidth: 5 } },
  { name: "Heart", group: "Shapes", shape: { kind: "heart", strokeWidth: 5 } },
  { name: "Bolt", group: "Shapes", shape: { kind: "lightning", strokeWidth: 5 } },
  { name: "Splat", group: "Shapes", shape: {
    kind: "splat", points: 11, innerRadius: 0.62, jitter: 0.55, seed: 5, strokeWidth: 5 } },
  { name: "Arrow", group: "Shapes", shape: { kind: "arrow", strokeWidth: 5 } },
  { name: "Banner", group: "Shapes", shape: { kind: "banner", strokeWidth: 5 } },

  { name: "Impact", group: "Behind the action", shape: {
    kind: "impact", points: 18, innerRadius: 0.34, jitter: 0.5, seed: 2, strokeWidth: 0 } },
  { name: "Sunburst", group: "Behind the action", shape: {
    kind: "sunburst", points: 24, innerRadius: 0.06, jitter: 0, strokeWidth: 0 } },
  { name: "Speed lines", group: "Behind the action", shape: {
    kind: "speedlines", points: 14, innerRadius: 0.5, jitter: 0.6, seed: 9, strokeWidth: 0 } },
];

/* --------------------------------------------------------------- the parts */

export type ShapePath = { points: Point[]; closed: boolean };

const TAU = Math.PI * 2;
const SMOOTH_SAMPLES = 192;

/**
 * A fixed pseudo-random value for one index.
 *
 * Deterministic on purpose. A shape that re-rolls its irregularity on every render would shimmer
 * as you drag any unrelated slider, and would export as something other than what you approved.
 */
function wobbleAt(index: number, seed: number): number {
  let hash = Math.imul(index ^ (seed * 0x9e3779b1), 2654435761);
  hash ^= hash >>> 15;
  hash = Math.imul(hash, 2246822519);
  hash ^= hash >>> 13;
  return ((hash >>> 0) % 4096) / 4096;
}

/** 2 is an ellipse, 30 is close enough to a rectangle that no corner survives. */
export function superExponent(shape: ShapeBlock): number {
  if (shape.kind === "ellipse" || shape.kind === "speech" || shape.kind === "wobble") return 2;
  if (shape.kind === "rectangle") return 30;
  const radius = Math.min(1, Math.max(0, shape.radius));
  return 2 + (1 - radius) * 28;
}

const spinOf = (shape: ShapeBlock) => (shape.spin ?? 0) * Math.PI / 180;
const jitterOf = (shape: ShapeBlock) => Math.min(1, Math.max(0, shape.jitter ?? 0));
const seedOf = (shape: ShapeBlock) => Math.max(1, Math.round(shape.seed ?? 1));
const spikes = (shape: ShapeBlock) => Math.max(3, Math.round(shape.points));
const depth = (shape: ShapeBlock) => Math.min(0.98, Math.max(0.05, shape.innerRadius));

/**
 * The corners of a spiky shape, as angle-and-radius pairs.
 *
 * Alternating outer and inner, with jitter SHORTENING spikes rather than lengthening them -- so
 * however irregular a burst gets, it still fits the box its size says it does.
 */
function spikeVertices(shape: ShapeBlock): Array<{ angle: number; scale: number }> {
  const count = spikes(shape);
  const inner = depth(shape);
  const jitter = jitterOf(shape);
  const seed = seedOf(shape);
  const phase = spinOf(shape);
  const out: Array<{ angle: number; scale: number }> = [];
  for (let index = 0; index < count * 2; index++) {
    const outer = index % 2 === 0;
    const base = outer ? 1 : inner;
    // Outer points vary a lot, inner ones only a little: a shout balloon's spikes are uneven, but
    // its middle is not, and jittering both makes a shape that reads as damaged rather than drawn.
    const vary = jitter * (outer ? 0.55 : 0.18) * wobbleAt(index, seed);
    out.push({ angle: phase + index * Math.PI / count, scale: base * (1 - vary) });
  }
  return out;
}

/** The closed body loop, with no tail. One builder for every kind. */
export function bodyOutline(shape: ShapeBlock): Point[] {
  const rx = Math.max(1, shape.width / 2);
  const ry = Math.max(1, shape.height / 2);
  const kind = shape.kind;

  if (kind === "arrow") return arrowPoints(rx, ry, depth(shape));
  if (kind === "banner") return bannerPoints(rx, ry);
  if (kind === "lightning") return boltPoints(rx, ry);
  if (kind === "heart") return heartPoints(rx, ry);

  if (kind === "polygon") {
    const count = spikes(shape);
    const phase = spinOf(shape);
    return Array.from({ length: count }, (_, index) => {
      const angle = phase + index * TAU / count;
      return [rx * Math.cos(angle), ry * Math.sin(angle)] as Point;
    });
  }

  if (CORNERED.includes(kind)) {
    return spikeVertices(shape).map(({ angle, scale }) =>
      [rx * scale * Math.cos(angle), ry * scale * Math.sin(angle)] as Point);
  }

  // Everything else is a radius that varies smoothly with the angle, so it is sampled.
  return Array.from({ length: SMOOTH_SAMPLES }, (_, index) => {
    const angle = (index / SMOOTH_SAMPLES) * TAU;
    const scale = smoothScale(shape, angle);
    if (kind === "ellipse" || kind === "speech" || kind === "thought" || kind === "wobble"
      || kind === "icicle" || kind === "splat") {
      return [rx * scale * Math.cos(angle), ry * scale * Math.sin(angle)] as Point;
    }
    // The superellipse family is not polar: its parametrisation is what gives it square corners.
    const exponent = superExponent(shape);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    return [
      rx * Math.sign(cos) * Math.pow(Math.abs(cos), 2 / exponent),
      ry * Math.sign(sin) * Math.pow(Math.abs(sin), 2 / exponent),
    ] as Point;
  });
}

/**
 * The radius multiplier for the smoothly-varying kinds.
 *
 * Every one of these cuts INWARD from the box rather than bulging out of it, so a shape's stated
 * width and height stay the truth however wobbly it is. Drips that hung outside the box would make
 * "300 pixels tall" mean something different for every kind.
 */
function smoothScale(shape: ShapeBlock, angle: number): number {
  if (shape.kind === "thought") {
    const lobes = Math.max(5, Math.round(spikes(shape) * 2 + 5));
    const bump = 0.11;
    return 1 - bump + bump * Math.abs(Math.cos(lobes * angle / 2));
  }
  if (shape.kind === "wobble") {
    const waves = spikes(shape);
    const amount = 1 - depth(shape);
    return 1 - amount * (1 - (Math.cos(waves * angle + spinOf(shape)) + 1) / 2);
  }
  if (shape.kind === "splat") {
    // Lobes of uneven size, interpolated smoothly between seeded values at each one.
    const lobes = spikes(shape);
    const seed = seedOf(shape);
    const jitter = jitterOf(shape);
    const at = (angle - spinOf(shape)) / TAU * lobes;
    const index = Math.floor(at);
    const t = at - index;
    const ease = t * t * (3 - 2 * t);
    const from = 1 - jitter * 0.55 * wobbleAt(((index % lobes) + lobes) % lobes, seed);
    const to = 1 - jitter * 0.55 * wobbleAt((((index + 1) % lobes) + lobes) % lobes, seed);
    return (from + (to - from) * ease) * (0.55 + depth(shape) * 0.45);
  }
  if (shape.kind === "icicle") {
    // Drips only along the bottom, cut up into the body so their tips reach the box's own edge.
    const down = Math.max(0, Math.sin(angle));
    if (down <= 0) return 1;
    const drips = spikes(shape);
    const cut = 1 - depth(shape);
    const wave = 1 - 2 * Math.abs(at1(drips * angle / Math.PI));
    return 1 - cut * down * (1 - Math.max(0, wave));
  }
  return 1;
}

/** Distance to the nearest whole number, as a signed fraction. A triangular wave in disguise. */
const at1 = (value: number) => value - Math.round(value);

/**
 * Where a ray from the centre leaves the shape.
 *
 * A cast against the outline itself rather than a per-kind formula, which is what makes it exact for
 * a heart and a star alike, and what stopped the bug this replaced: the old version picked the
 * NEAREST corner and then walked forward from it, so for half of all angles the ray missed the
 * segment it was given and fell back to an unrelated radius. Stars, bursts and shout balloons were
 * all drawn from that fallback.
 */
export function bodyPoint(shape: ShapeBlock, angle: number): Point {
  return rayHit(bodyOutline(shape), angle);
}

function rayHit(outline: Point[], angle: number): Point {
  const dx = Math.cos(angle), dy = Math.sin(angle);
  let best = -Infinity;
  let found: Point = [0, 0];
  for (let index = 0; index < outline.length; index++) {
    const a = outline[index];
    const b = outline[(index + 1) % outline.length];
    const ex = b[0] - a[0], ey = b[1] - a[1];
    const denominator = dx * ey - dy * ex;
    if (Math.abs(denominator) < 1e-12) continue;
    const t = (a[0] * dy - a[1] * dx) / denominator;
    if (t < -1e-9 || t > 1 + 1e-9) continue;
    const px = a[0] + ex * t, py = a[1] + ey * t;
    // Furthest along the ray, so a concave shape reports its outer edge rather than an inner one.
    const along = px * dx + py * dy;
    if (along > best) { best = along; found = [px, py]; }
  }
  return best > -Infinity ? found : [0, 0];
}

/* ------------------------------------------------------------- the outline */

export function shapePaths(shape: ShapeBlock): ShapePath[] {
  if (shape.kind === "impact" || shape.kind === "sunburst") return rayPaths(shape);
  if (shape.kind === "speedlines") return speedPaths(shape);

  const outline = bodyOutline(shape);
  const tailed = hasTail(shape) && shape.kind !== "thought";
  if (!tailed) {
    const paths: ShapePath[] = [{ points: outline, closed: true }];
    if (shape.kind === "thought" && hasTail(shape)) paths.push(...thoughtTrail(shape));
    return paths;
  }

  const half = Math.min(80, Math.max(3, shape.tailWidth)) * Math.PI / 180;
  const centre = shape.tailAngle * Math.PI / 180;
  const from = centre + half;
  const to = centre - half + TAU;
  const points = arcPoints(shape, outline, from, to);
  points.push(...tailPoints(shape, points[points.length - 1], points[0]));
  return [{ points, closed: true }];
}

/**
 * The outline's own points between two angles, in order, with the exact ends included.
 *
 * Taking the shape's real points rather than resampling is what keeps a shout balloon's spikes
 * sharp: a re-sample at even angles rounds every corner it does not happen to land on.
 */
function arcPoints(shape: ShapeBlock, outline: Point[], from: number, to: number): Point[] {
  const points: Point[] = [rayHit(outline, from)];
  const between: Array<{ angle: number; point: Point }> = [];
  for (const point of outline) {
    let angle = Math.atan2(point[1], point[0]);
    while (angle < from) angle += TAU;
    if (angle > from + 1e-9 && angle < to - 1e-9) between.push({ angle, point });
  }
  between.sort((left, right) => left.angle - right.angle);
  for (const entry of between) points.push(entry.point);
  points.push(rayHit(outline, to));
  return points;
}

/**
 * From one edge of the gap, out to the tip, and back to the other edge.
 *
 * Curved by default. A straight tail is a spike, and a spike does not read as coming out of a
 * mouth; the bend is most of what makes a balloon look drawn rather than generated.
 */
function tailPoints(shape: ShapeBlock, start: Point, end: Point): Point[] {
  const angle = shape.tailAngle * Math.PI / 180;
  const base = bodyPoint(shape, angle);
  const reach = 1 + Math.max(0, shape.tailLength);
  const tip: Point = [base[0] * reach, base[1] * reach];
  const away = Math.hypot(tip[0] - base[0], tip[1] - base[1]);
  const bend = (shape.tailCurve ?? 0) * away * 0.6;
  const nx = -Math.sin(angle) * bend, ny = Math.cos(angle) * bend;
  const out: Point[] = [];
  const steps = 14;
  for (let index = 1; index <= steps; index++) {
    out.push(quadratic(start, [(start[0] + tip[0]) / 2 + nx, (start[1] + tip[1]) / 2 + ny], tip, index / steps));
  }
  for (let index = 1; index < steps; index++) {
    out.push(quadratic(tip, [(tip[0] + end[0]) / 2 + nx * 0.35, (tip[1] + end[1]) / 2 + ny * 0.35], end, index / steps));
  }
  return out;
}

const quadratic = (a: Point, control: Point, b: Point, t: number): Point => {
  const inverse = 1 - t;
  return [
    inverse * inverse * a[0] + 2 * inverse * t * control[0] + t * t * b[0],
    inverse * inverse * a[1] + 2 * inverse * t * control[1] + t * t * b[1],
  ];
};

/** Three shrinking circles marching away from the cloud. */
function thoughtTrail(shape: ShapeBlock): ShapePath[] {
  const angle = shape.tailAngle * Math.PI / 180;
  const base = bodyPoint(shape, angle);
  const span = Math.hypot(base[0], base[1]) * Math.max(0, shape.tailLength);
  const start = Math.min(shape.width, shape.height) * 0.09;
  const paths: ShapePath[] = [];
  for (let index = 0; index < 3; index++) {
    const along = (index + 1) / 3.2;
    const radius = start * (1 - index * 0.28);
    if (radius < 1) continue;
    const cx = base[0] + Math.cos(angle) * (span * along + radius);
    const cy = base[1] + Math.sin(angle) * (span * along + radius);
    paths.push({ closed: true, points: Array.from({ length: 48 }, (_, step) => {
      const at = (step / 48) * TAU;
      return [cx + Math.cos(at) * radius, cy + Math.sin(at) * radius] as Point;
    }) });
  }
  return paths;
}

/**
 * Separate rays, for the two shapes that go behind the action rather than around it.
 *
 * A sunburst's rays meet in the middle; an impact's stop short of it, leaving the hole where the
 * thing that hit is. That gap is the entire difference between "look here" and "something just
 * landed", and it is why these are two shapes and not one with a slider.
 */
function rayPaths(shape: ShapeBlock): ShapePath[] {
  const rx = Math.max(1, shape.width / 2);
  const ry = Math.max(1, shape.height / 2);
  const count = spikes(shape);
  const seed = seedOf(shape);
  const jitter = jitterOf(shape);
  const hollow = shape.kind === "impact" ? Math.max(0.05, depth(shape)) : depth(shape) * 0.4;
  const step = TAU / count;
  const phase = spinOf(shape);
  const paths: ShapePath[] = [];
  for (let index = 0; index < count; index++) {
    const middle = phase + index * step;
    // Half the gap between rays, narrowed a touch so they never quite touch each other.
    const spread = step * 0.42 * (1 - jitter * 0.5 * wobbleAt(index, seed));
    const reach = 1 - jitter * 0.4 * wobbleAt(index + 500, seed);
    const inner = hollow * (1 + jitter * 0.25 * wobbleAt(index + 900, seed));
    const at = (angle: number, scale: number): Point =>
      [rx * scale * Math.cos(angle), ry * scale * Math.sin(angle)];
    paths.push({ closed: true, points: [
      at(middle - spread, reach), at(middle + spread, reach),
      at(middle + spread * 0.25, inner), at(middle - spread * 0.25, inner),
    ] });
  }
  return paths;
}

/** Tapered horizontal streaks of uneven length, for motion. */
function speedPaths(shape: ShapeBlock): ShapePath[] {
  const rx = Math.max(1, shape.width / 2);
  const ry = Math.max(1, shape.height / 2);
  const count = spikes(shape);
  const seed = seedOf(shape);
  const jitter = jitterOf(shape);
  const thickness = ry * 2 / count * Math.max(0.1, depth(shape)) * 0.8;
  const paths: ShapePath[] = [];
  for (let index = 0; index < count; index++) {
    const y = -ry + (index + 0.5) * (ry * 2 / count);
    const length = rx * 2 * (1 - jitter * wobbleAt(index, seed) * 0.75);
    const left = -rx;
    const right = left + length;
    const halfThick = thickness / 2;
    paths.push({ closed: true, points: [
      [left, y - halfThick], [right - halfThick * 2, y - halfThick * 0.25],
      [right, y], [right - halfThick * 2, y + halfThick * 0.25], [left, y + halfThick],
    ] });
  }
  return paths;
}

/* -------------------------------------------------------- the fixed shapes */

function arrowPoints(rx: number, ry: number, shaftFraction: number): Point[] {
  const head = Math.min(rx * 1.2, rx * 0.84);
  const shaft = ry * shaftFraction;
  return [
    [rx, 0], [rx - head, -ry], [rx - head, -shaft], [-rx, -shaft],
    [-rx, shaft], [rx - head, shaft], [rx - head, ry],
  ];
}

function bannerPoints(rx: number, ry: number): Point[] {
  const notch = Math.min(rx * 0.4, ry * 0.9);
  return [[-rx, -ry], [rx, -ry], [rx - notch, 0], [rx, ry], [-rx, ry], [-rx + notch, 0]];
}

/** A bolt drawn as a fixed silhouette, scaled to the box. Nobody wants to parametrise lightning. */
function boltPoints(rx: number, ry: number): Point[] {
  const unit: Point[] = [
    [0.15, -1], [0.95, -1], [0.25, -0.1], [0.8, -0.1],
    [-0.35, 1], [-0.05, 0.15], [-0.75, 0.15], [-0.2, -1],
  ];
  return unit.map(([x, y]) => [rx * x, ry * y] as Point);
}

/** The standard heart curve, normalised so it exactly fills its box. */
function heartPoints(rx: number, ry: number): Point[] {
  const raw: Point[] = [];
  let widest = 0, tallest = 0;
  for (let index = 0; index < SMOOTH_SAMPLES; index++) {
    const t = (index / SMOOTH_SAMPLES) * TAU;
    const x = Math.pow(Math.sin(t), 3) * 16;
    const y = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t));
    raw.push([x, y]);
    widest = Math.max(widest, Math.abs(x));
    tallest = Math.max(tallest, Math.abs(y));
  }
  return raw.map(([x, y]) => [rx * x / widest, ry * y / tallest] as Point);
}

/* -------------------------------------------------------------- the extent */

/**
 * The bitmap a shape needs, and where its centre sits inside it.
 *
 * Measured from the outline that will actually be drawn rather than from the body's dimensions,
 * because a tail, a thick stroke and a drop shadow all reach past the body and every one of them
 * gets clipped if the bitmap is sized to the body alone.
 */
export function shapeBounds(shape: ShapeBlock): {
  width: number; height: number; offsetX: number; offsetY: number;
} {
  let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
  for (const path of shapePaths(shape)) {
    for (const [x, y] of path.points) {
      if (x < left) left = x;
      if (x > right) right = x;
      if (y < top) top = y;
      if (y > bottom) bottom = y;
    }
  }
  if (!Number.isFinite(left)) return { width: 8, height: 8, offsetX: 4, offsetY: 4 };
  const pad = shape.strokeWidth / 2 + 2;
  const glow = shape.shadow > 0 ? shape.shadow + Math.abs(shape.shadowOffset) : 0;
  left -= pad + glow; top -= pad + glow;
  right += pad + glow; bottom += pad + glow;
  return {
    width: Math.max(8, Math.ceil(right - left)),
    height: Math.max(8, Math.ceil(bottom - top)),
    offsetX: -left,
    offsetY: -top,
  };
}

/** Ray casting: the definition of being inside a closed outline. */
function contains(outline: Point[], x: number, y: number): boolean {
  let within = false;
  for (let index = 0, previous = outline.length - 1; index < outline.length; previous = index++) {
    const [xi, yi] = outline[index];
    const [xj, yj] = outline[previous];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) within = !within;
  }
  return within;
}

/**
 * The rectangle inside the shape that lettering may use.
 *
 * SEARCHED rather than derived: the largest box of the shape's own proportions whose corners and
 * edge midpoints all lie inside the outline. The previous version used a hand-picked fraction per
 * kind, which meant every new shape needed a new guess and a five-pointed star got one that put its
 * words out past two of its points. A search cannot be wrong about a shape it has never seen, which
 * matters when the list of shapes keeps growing.
 */
export function shapeInnerBox(shape: ShapeBlock): { width: number; height: number } {
  const rx = Math.max(1, shape.width / 2);
  const ry = Math.max(1, shape.height / 2);
  const inset = shape.padding + shape.strokeWidth;
  if (!holdsText(shape.kind)) {
    return { width: Math.max(8, rx - inset), height: Math.max(8, ry - inset) };
  }
  const outline = bodyOutline(shape);
  const fits = (k: number) => {
    const w = rx * k, h = ry * k;
    // Corners and edge midpoints: a concave shape can admit all four corners and still cut through
    // the middle of a side, which is exactly what a star does.
    return [
      [-w, -h], [w, -h], [w, h], [-w, h], [0, -h], [w, 0], [0, h], [-w, 0],
    ].every(([x, y]) => contains(outline, x, y));
  };
  let low = 0, high = 1;
  for (let round = 0; round < 18; round++) {
    const middle = (low + high) / 2;
    if (fits(middle)) low = middle; else high = middle;
  }
  return {
    width: Math.max(8, rx * 2 * low - inset * 2),
    height: Math.max(8, ry * 2 * low - inset * 2),
  };
}

/** Where the tail's tip lands, for the handle that drags it. */
export function tailTip(shape: ShapeBlock): Point {
  const angle = shape.tailAngle * Math.PI / 180;
  const base = bodyPoint(shape, angle);
  const reach = 1 + Math.max(0, shape.tailLength);
  return [base[0] * reach, base[1] * reach];
}
