/**
 * VFX Phase 6: vector paths -- the free-form sibling of `photoShape.ts`.
 *
 * A path is cubic Bezier subpaths, open or closed, with a fill rule, an optional fill and stroke
 * `Paint`, and the canvas's own stroke parameters (width, join, cap, miter limit, dashes).
 *
 * Coordinates are CANVAS PIXELS around the path's own origin, which is the layer's centre: exactly
 * like `ShapeBlock`, so a path is placed, scaled and turned by the layer matrix like everything else,
 * and a path drawn at scale 1 lands on the canvas pixels it was drawn on. The raster the compositor
 * places is symmetric about that origin (`pathBox`), so the bitmap's centre IS the origin and no
 * offset has to be carried through placement: a path that grows to one side grows its bitmap on
 * both, which costs pixels and buys a placement that can never drift.
 *
 * Handles are stored RELATIVE to their anchor (`in`, `out`), so moving an anchor carries its handles
 * with it, and a corner is simply an anchor without handles. Everything that needs absolute control
 * points (flattening, bounds, the envelope, booleans) goes through `segmentsOf`.
 *
 * Everything here is pure geometry and runs without a canvas, so it is all unit-tested in node; the
 * two compositors rasterise through one Canvas2D function (`drawPath` in photoCompose.ts), which is
 * why their parity is by construction.
 */

import { envelopePath } from "./photoEnvelope";
import type { Paint } from "./photoGradient";

export type Point = [number, number];

export type PathNode = {
  x: number;
  y: number;
  /** The incoming handle, relative to the anchor. Absent: a corner on that side. */
  in?: Point;
  /** The outgoing handle, relative to the anchor. */
  out?: Point;
};

export type Subpath = { nodes: PathNode[]; closed: boolean };

export type PathJoin = "miter" | "round" | "bevel";
export type PathCap = "butt" | "round" | "square";
export type FillRule = "nonzero" | "evenodd";

/**
 * A live envelope distort: a tensor-product Bezier surface of (cols + 1) x (rows + 1) control points
 * over `rect` (the path's box when it was made). With the points evenly spaced it is the identity;
 * moving them bends everything inside. Applied to the anchors and handles after subdividing, so the
 * warped curves stay curves (see photoEnvelope.ts).
 */
export type Envelope = {
  cols: number;
  rows: number;
  rect: { x: number; y: number; width: number; height: number };
  /** Row-major, (cols + 1) * (rows + 1) points, in the path's own coordinates. */
  points: Point[];
};

export type PathBlock = {
  subpaths: Subpath[];
  fillRule: FillRule;
  /** Absent: not filled. */
  fill?: Paint;
  /** Absent (or `strokeWidth` 0): not stroked. */
  stroke?: Paint;
  strokeWidth: number;
  join: PathJoin;
  cap: PathCap;
  miterLimit: number;
  /** Dash lengths in canvas pixels, alternating on/off (the canvas's `setLineDash`). */
  dash?: number[];
  dashOffset?: number;
  /** "stroke": the stroke is drawn first, under the fill (SVG paint-order), so only its outer half shows. */
  paintOrder?: "fill" | "stroke";
  envelope?: Envelope;
};

export const NEUTRAL_PATH: Omit<PathBlock, "subpaths"> = {
  fillRule: "nonzero",
  fill: { kind: "solid", color: "#f2c230" },
  stroke: { kind: "solid", color: "#1b1b1b" },
  strokeWidth: 4,
  join: "miter",
  cap: "butt",
  miterLimit: 10,
};

/* ------------------------------------------------------------ segments */

export type Cubic = [Point, Point, Point, Point];

const add = (a: Point, b: Point | undefined): Point => b ? [a[0] + b[0], a[1] + b[1]] : [a[0], a[1]];

/** A subpath's segments as absolute cubics. A closed subpath gets its closing segment too. */
export function segmentsOf(subpath: Subpath): Cubic[] {
  const { nodes } = subpath;
  const out: Cubic[] = [];
  const count = subpath.closed ? nodes.length : nodes.length - 1;
  for (let index = 0; index < count; index++) {
    const a = nodes[index];
    const b = nodes[(index + 1) % nodes.length];
    const p0: Point = [a.x, a.y];
    const p3: Point = [b.x, b.y];
    out.push([p0, add(p0, a.out), add(p3, b.in), p3]);
  }
  return out;
}

/** The point at `t` on a cubic. */
export function cubicAt(c: Cubic, t: number): Point {
  const u = 1 - t;
  const a = u * u * u, b = 3 * u * u * t, d = 3 * u * t * t, e = t * t * t;
  return [a * c[0][0] + b * c[1][0] + d * c[2][0] + e * c[3][0], a * c[0][1] + b * c[1][1] + d * c[2][1] + e * c[3][1]];
}

/** The derivative at `t`. */
export function cubicTangent(c: Cubic, t: number): Point {
  const u = 1 - t;
  const x = 3 * u * u * (c[1][0] - c[0][0]) + 6 * u * t * (c[2][0] - c[1][0]) + 3 * t * t * (c[3][0] - c[2][0]);
  const y = 3 * u * u * (c[1][1] - c[0][1]) + 6 * u * t * (c[2][1] - c[1][1]) + 3 * t * t * (c[3][1] - c[2][1]);
  return [x, y];
}

/** de Casteljau split at `t`: the two halves, exactly the same curve. */
export function splitCubic(c: Cubic, t: number): [Cubic, Cubic] {
  const lerp = (p: Point, q: Point): Point => [p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t];
  const p01 = lerp(c[0], c[1]), p12 = lerp(c[1], c[2]), p23 = lerp(c[2], c[3]);
  const p012 = lerp(p01, p12), p123 = lerp(p12, p23);
  const mid = lerp(p012, p123);
  return [[c[0], p01, p012, mid], [mid, p123, p23, c[3]]];
}

/**
 * How many straight pieces make a cubic within `tolerance` pixels: Wang's formula,
 * n = ceil(sqrt(3/4 * max|p_i - 2 p_i+1 + p_i+2| / tolerance)). Deterministic, no recursion.
 */
export function flattenCount(c: Cubic, tolerance: number): number {
  // A segment between two corners is a straight line whatever its parameterisation: one piece.
  if (c[1][0] === c[0][0] && c[1][1] === c[0][1] && c[2][0] === c[3][0] && c[2][1] === c[3][1]) return 1;
  const dd = (i: number) => Math.hypot(c[i][0] - 2 * c[i + 1][0] + c[i + 2][0], c[i][1] - 2 * c[i + 1][1] + c[i + 2][1]);
  const m = Math.max(dd(0), dd(1));
  return Math.max(1, Math.min(4096, Math.ceil(Math.sqrt((0.75 * m) / Math.max(1e-6, tolerance)))));
}

/** A subpath as a polyline (a closed one without its repeated start point). */
export function flattenSubpath(subpath: Subpath, tolerance = 0.2): Point[] {
  if (!subpath.nodes.length) return [];
  const points: Point[] = [[subpath.nodes[0].x, subpath.nodes[0].y]];
  for (const c of segmentsOf(subpath)) {
    const n = flattenCount(c, tolerance);
    for (let i = 1; i <= n; i++) points.push(cubicAt(c, i / n));
  }
  if (subpath.closed && points.length > 1) points.pop();
  return points;
}

/* ---------------------------------------------------------- measuring */

export type Bounds = { minX: number; minY: number; maxX: number; maxY: number };

const EMPTY_BOUNDS: Bounds = { minX: 0, minY: 0, maxX: 0, maxY: 0 };

/** The tight box of a cubic: its end points and the extrema where the derivative is zero. */
function cubicBounds(c: Cubic, into: Bounds): void {
  const take = (p: Point) => {
    if (p[0] < into.minX) into.minX = p[0];
    if (p[0] > into.maxX) into.maxX = p[0];
    if (p[1] < into.minY) into.minY = p[1];
    if (p[1] > into.maxY) into.maxY = p[1];
  };
  take(c[0]);
  take(c[3]);
  for (const axis of [0, 1]) {
    // B'(t) / 3 = a t^2 + b t + k
    const p0 = c[0][axis], p1 = c[1][axis], p2 = c[2][axis], p3 = c[3][axis];
    const a = -p0 + 3 * p1 - 3 * p2 + p3, b = 2 * (p0 - 2 * p1 + p2), k = p1 - p0;
    const roots: number[] = [];
    if (Math.abs(a) < 1e-12) { if (Math.abs(b) > 1e-12) roots.push(-k / b); } else {
      const disc = b * b - 4 * a * k;
      if (disc >= 0) { const s = Math.sqrt(disc); roots.push((-b + s) / (2 * a), (-b - s) / (2 * a)); }
    }
    for (const t of roots) if (t > 0 && t < 1) take(cubicAt(c, t));
  }
}

/** The geometry's tight box (the envelope applied), without the stroke. */
export function pathBounds(path: PathBlock): Bounds {
  const shaped = shapedPath(path);
  let bounds: Bounds | undefined;
  for (const subpath of shaped.subpaths) {
    if (!subpath.nodes.length) continue;
    const first = subpath.nodes[0];
    bounds ??= { minX: first.x, minY: first.y, maxX: first.x, maxY: first.y };
    for (const c of segmentsOf(subpath)) cubicBounds(c, bounds);
  }
  return bounds ?? { ...EMPTY_BOUNDS };
}

/** How far the stroke can reach past the geometry: half the width, times the miter or the cap. */
export function strokeReach(path: PathBlock): number {
  if (!path.stroke || path.strokeWidth <= 0) return 0;
  const half = path.strokeWidth / 2;
  const join = path.join === "miter" ? Math.max(1, path.miterLimit) : 1;
  const cap = path.cap === "square" ? Math.SQRT2 : 1;
  return half * Math.max(join, cap);
}

/**
 * The raster a path is drawn into: symmetric about the origin (so the bitmap's centre is the layer's
 * centre), covering the geometry plus the stroke's reach plus a pixel of antialiasing, in whole
 * even pixels. `offsetX/Y` is where the origin sits in the bitmap (always its centre).
 */
export function pathBox(path: PathBlock): { width: number; height: number; offsetX: number; offsetY: number } {
  const b = pathBounds(path);
  const reach = strokeReach(path) + 2;
  const halfW = Math.ceil(Math.max(Math.abs(b.minX), Math.abs(b.maxX)) + reach);
  const halfH = Math.ceil(Math.max(Math.abs(b.minY), Math.abs(b.maxY)) + reach);
  const width = Math.max(2, Math.min(16384, halfW * 2));
  const height = Math.max(2, Math.min(16384, halfH * 2));
  return { width, height, offsetX: width / 2, offsetY: height / 2 };
}

/** Signed area of a closed polyline (positive = clockwise in y-down canvas coordinates). */
export function polygonArea(points: Point[]): number {
  let sum = 0;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    sum += (points[j][0] * points[i][1]) - (points[i][0] * points[j][1]);
  }
  return sum / 2;
}

/** Length of a polyline, closed or not. */
export function polylineLength(points: Point[], closed: boolean): number {
  let length = 0;
  for (let i = 1; i < points.length; i++) length += Math.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]);
  if (closed && points.length > 1) {
    const a = points[points.length - 1], b = points[0];
    length += Math.hypot(b[0] - a[0], b[1] - a[1]);
  }
  return length;
}

/**
 * The point and direction (radians) at arc length `s` along a polyline. `s` wraps on a closed one
 * and clamps on an open one.
 */
export function pointAlong(points: Point[], closed: boolean, s: number): { at: Point; angle: number } {
  if (!points.length) return { at: [0, 0], angle: 0 };
  if (points.length === 1) return { at: points[0], angle: 0 };
  const ring = closed ? [...points, points[0]] : points;
  const total = polylineLength(points, closed);
  let rest = closed ? ((s % total) + total) % total : Math.min(Math.max(0, s), total);
  for (let i = 1; i < ring.length; i++) {
    const a = ring[i - 1], b = ring[i];
    const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (rest <= length || i === ring.length - 1) {
      const t = length > 0 ? Math.min(1, rest / length) : 0;
      return { at: [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t], angle: Math.atan2(b[1] - a[1], b[0] - a[0]) };
    }
    rest -= length;
  }
  const a = ring[ring.length - 2], b = ring[ring.length - 1];
  return { at: b, angle: Math.atan2(b[1] - a[1], b[0] - a[0]) };
}

/* ------------------------------------------------------------ mapping */

/**
 * Every anchor and handle through `map` (handles as absolute points, then made relative again).
 * Exact for affine maps; for a curved map subdivide first (`subdivided`).
 */
export function mapPath(path: PathBlock, map: (p: Point) => Point): PathBlock {
  return {
    ...path,
    subpaths: path.subpaths.map((subpath) => ({
      closed: subpath.closed,
      nodes: subpath.nodes.map((node) => {
        const anchor = map([node.x, node.y]);
        const out: PathNode = { x: anchor[0], y: anchor[1] };
        if (node.in) { const h = map([node.x + node.in[0], node.y + node.in[1]]); out.in = [h[0] - anchor[0], h[1] - anchor[1]]; }
        if (node.out) { const h = map([node.x + node.out[0], node.y + node.out[1]]); out.out = [h[0] - anchor[0], h[1] - anchor[1]]; }
        return out;
      }),
    })),
  };
}

/** Nodes from absolute cubics (consecutive, `closed` joins the last to the first). */
export function subpathFromCubics(cubics: Cubic[], closed: boolean): Subpath {
  if (!cubics.length) return { nodes: [], closed };
  const handle = (from: Point, to: Point): Point | undefined => {
    const d: Point = [to[0] - from[0], to[1] - from[1]];
    return Math.abs(d[0]) < 1e-9 && Math.abs(d[1]) < 1e-9 ? undefined : d;
  };
  const nodes: PathNode[] = cubics.map((c) => ({ x: c[0][0], y: c[0][1] }));
  cubics.forEach((c, index) => {
    const a = nodes[index];
    const out = handle(c[0], c[1]);
    if (out) a.out = out;
    const next = closed ? nodes[(index + 1) % nodes.length] : nodes[index + 1];
    if (next) { const into = handle(c[3], c[2]); if (into) next.in = into; }
  });
  if (!closed) {
    const last = cubics[cubics.length - 1];
    const end: PathNode = { x: last[3][0], y: last[3][1] };
    const into = handle(last[3], last[2]);
    if (into) end.in = into;
    nodes.push(end);
  }
  return { nodes: nodes.map((node) => clean(node)), closed };
}

function clean(node: PathNode): PathNode {
  const out: PathNode = { x: node.x, y: node.y };
  if (node.in) out.in = node.in;
  if (node.out) out.out = node.out;
  return out;
}

/** Every segment split into `pieces` equal-parameter parts: the same curve, more anchors. */
export function subdivided(path: PathBlock, pieces: number): PathBlock {
  const k = Math.max(1, Math.floor(pieces));
  return {
    ...path,
    subpaths: path.subpaths.map((subpath) => {
      if (k === 1 || subpath.nodes.length < 2) return subpath;
      const cubics: Cubic[] = [];
      for (const c of segmentsOf(subpath)) {
        let rest = c;
        for (let i = k; i > 1; i--) {
          const [head, tail] = splitCubic(rest, 1 / i);
          cubics.push(head);
          rest = tail;
        }
        cubics.push(rest);
      }
      return subpathFromCubics(cubics, subpath.closed);
    }),
  };
}

export const translatePath = (path: PathBlock, dx: number, dy: number): PathBlock =>
  mapPath(path, ([x, y]) => [x + dx, y + dy]);

/** The geometry that is drawn: the envelope (if any) applied, and dropped from the result. */
export function shapedPath(path: PathBlock): PathBlock {
  if (!path.envelope) return path;
  const { envelope, ...rest } = path;
  return envelopePath(rest, envelope);
}

/* --------------------------------------------------------------- editing */

/** The nearest anchor within `radius`, as [subpath, node], or undefined. */
export function hitAnchor(path: PathBlock, at: Point, radius: number): [number, number] | undefined {
  let best: [number, number] | undefined;
  let bestDistance = radius;
  path.subpaths.forEach((subpath, s) => subpath.nodes.forEach((node, n) => {
    const d = Math.hypot(node.x - at[0], node.y - at[1]);
    if (d <= bestDistance) { bestDistance = d; best = [s, n]; }
  }));
  return best;
}

/** The nearest handle end within `radius`, as [subpath, node, "in" | "out"], or undefined. */
export function hitHandle(path: PathBlock, at: Point, radius: number): [number, number, "in" | "out"] | undefined {
  let best: [number, number, "in" | "out"] | undefined;
  let bestDistance = radius;
  path.subpaths.forEach((subpath, s) => subpath.nodes.forEach((node, n) => {
    for (const side of ["in", "out"] as const) {
      const h = node[side];
      if (!h) continue;
      const d = Math.hypot(node.x + h[0] - at[0], node.y + h[1] - at[1]);
      if (d <= bestDistance) { bestDistance = d; best = [s, n, side]; }
    }
  }));
  return best;
}

const withSubpath = (path: PathBlock, index: number, change: (subpath: Subpath) => Subpath): PathBlock => ({
  ...path, subpaths: path.subpaths.map((subpath, i) => i === index ? change(subpath) : subpath),
});

/** Move one anchor (its handles come with it). */
export function moveAnchor(path: PathBlock, s: number, n: number, to: Point): PathBlock {
  return withSubpath(path, s, (subpath) => ({
    ...subpath, nodes: subpath.nodes.map((node, i) => i === n ? { ...node, x: to[0], y: to[1] } : node),
  }));
}

/**
 * Set one handle to the absolute point `to`. `mirror` turns the opposite handle to stay colinear
 * (a smooth point), keeping its own length; "equal" also gives it the same length (pen-drag).
 */
export function setHandle(path: PathBlock, s: number, n: number, side: "in" | "out", to: Point, mirror: false | "angle" | "equal" = false): PathBlock {
  return withSubpath(path, s, (subpath) => ({
    ...subpath,
    nodes: subpath.nodes.map((node, i) => {
      if (i !== n) return node;
      const own: Point = [to[0] - node.x, to[1] - node.y];
      const next: PathNode = { ...node, [side]: own };
      const other = side === "in" ? "out" : "in";
      const length = Math.hypot(own[0], own[1]);
      if (mirror && length > 1e-9) {
        const keep = mirror === "equal" ? length : Math.hypot(...(node[other] ?? [0, 0]));
        if (keep > 1e-9) next[other] = [-own[0] / length * keep, -own[1] / length * keep];
      }
      if (length < 1e-9) delete next[side];
      return next;
    }),
  }));
}

/** Append an anchor to the end of a subpath (a new subpath when `s` is past the end). */
export function appendAnchor(path: PathBlock, s: number, at: Point): PathBlock {
  if (s >= path.subpaths.length) return { ...path, subpaths: [...path.subpaths, { nodes: [{ x: at[0], y: at[1] }], closed: false }] };
  return withSubpath(path, s, (subpath) => ({ ...subpath, nodes: [...subpath.nodes, { x: at[0], y: at[1] }] }));
}

export const closeSubpath = (path: PathBlock, s: number, closed = true): PathBlock =>
  withSubpath(path, s, (subpath) => ({ ...subpath, closed }));

/** Remove one anchor; a subpath left with none is removed. */
export function deleteAnchor(path: PathBlock, s: number, n: number): PathBlock {
  const next = withSubpath(path, s, (subpath) => ({ ...subpath, nodes: subpath.nodes.filter((_, i) => i !== n) }));
  return { ...next, subpaths: next.subpaths.filter((subpath) => subpath.nodes.length > 0) };
}

/** Corner <-> smooth: a corner gets handles along its neighbours' chord; a smooth point loses both. */
export function toggleSmooth(path: PathBlock, s: number, n: number): PathBlock {
  return withSubpath(path, s, (subpath) => ({
    ...subpath,
    nodes: subpath.nodes.map((node, i) => {
      if (i !== n) return node;
      if (node.in || node.out) return { x: node.x, y: node.y };
      const count = subpath.nodes.length;
      const prev = subpath.nodes[subpath.closed ? (i - 1 + count) % count : Math.max(0, i - 1)];
      const next = subpath.nodes[subpath.closed ? (i + 1) % count : Math.min(count - 1, i + 1)];
      const dx = next.x - prev.x, dy = next.y - prev.y;
      const length = Math.hypot(dx, dy);
      if (length < 1e-9) return node;
      const reach = length / 6;
      return { ...node, in: [-dx / length * reach, -dy / length * reach], out: [dx / length * reach, dy / length * reach] };
    }),
  }));
}

/* ------------------------------------------------------------- drawing */

/** SVG path data for the geometry, through `map` (e.g. path -> display percent). */
export function pathData(path: PathBlock, map: (p: Point) => Point = (p) => p): string {
  const f = (p: Point) => { const q = map(p); return `${round3(q[0])} ${round3(q[1])}`; };
  const parts: string[] = [];
  for (const subpath of path.subpaths) {
    if (!subpath.nodes.length) continue;
    parts.push(`M${f([subpath.nodes[0].x, subpath.nodes[0].y])}`);
    for (const c of segmentsOf(subpath)) parts.push(`C${f(c[1])} ${f(c[2])} ${f(c[3])}`);
    if (subpath.closed) parts.push("Z");
  }
  return parts.join("");
}

const round3 = (value: number) => Math.round(value * 1000) / 1000;

/** Anything drawable at all: at least one subpath with two anchors (or one closed with one). */
export const hasGeometry = (path: PathBlock): boolean =>
  path.subpaths.some((subpath) => subpath.nodes.length >= 2);
