/**
 * VFX Phase 6: what you do WITH paths -- booleans, compound paths, outline stroke, Make Blend, and
 * the mapping between a path layer's own coordinates and the canvas.
 *
 * Booleans run on flattened curves through Clipper2 (Angus Johnson, Boost Software License 1.0; the
 * `clipper2-ts` port). Curve-exact booleans need curve/curve intersection with all its numerical
 * traps; flattening at 0.1 px keeps the result within a tenth of a pixel of the true answer, which is
 * below what either compositor can show. The result is a path of corner anchors.
 *
 * Each operand is first normalised by its OWN fill rule (a union with itself), so an even-odd path
 * and a non-zero one can be combined: after that both are simple non-overlapping polygons and the
 * operation itself runs non-zero.
 */

import { Clipper, ClipType, EndType, FillRule as ClipperFill, JoinType, type PathsD } from "clipper2-ts";
import type { PhotoLayer } from "./photoDocument";
import { croppedRect, layerMatrix } from "./photoCompose";
import { applyMatrix, compose, invert, translate, type Matrix } from "./photoGeometry";
import type { Paint } from "./photoGradient";
import {
  cubicAt, flattenSubpath, mapPath, pathBox, pathBounds, polygonArea, segmentsOf, splitCubic, subpathFromCubics,
  type Cubic, type PathBlock, type PathNode, type Point, type Subpath,
} from "./photoPath";

/* ------------------------------------------------------------ placement */

/**
 * Path coordinates -> canvas pixels for a path layer: the layer matrix of its raster, after the
 * raster's own offset (its centre is the path's origin) and the layer's crop.
 */
export function pathToCanvas(layer: PhotoLayer, path: PathBlock, canvasWidth: number, canvasHeight: number): Matrix {
  const box = pathBox(path);
  const rect = croppedRect(layer, box);
  return compose(layerMatrix(layer, canvasWidth, canvasHeight, box), translate(box.offsetX - rect.left, box.offsetY - rect.top));
}

/** The path in canvas pixels (exact: the layer placement is affine). */
export function pathInCanvas(layer: PhotoLayer, path: PathBlock, canvasWidth: number, canvasHeight: number): PathBlock {
  const m = pathToCanvas(layer, path, canvasWidth, canvasHeight);
  const out = mapPath(path, ([x, y]) => applyMatrix(m, x, y));
  const scale = Math.sqrt(Math.abs(m.a * m.d - m.b * m.c));
  return { ...out, strokeWidth: path.strokeWidth * scale, dash: path.dash?.map((value) => value * scale), dashOffset: path.dashOffset === undefined ? undefined : path.dashOffset * scale };
}

/**
 * A canvas-pixel path as a layer's path: centred on its own box, with the layer placed at that
 * centre at scale 1 and no rotation. Returns the local path and the layer's normalised centre.
 */
export function centreOnBox(canvasPath: PathBlock, canvasWidth: number, canvasHeight: number): { path: PathBlock; x: number; y: number } {
  const b = pathBounds({ ...canvasPath, envelope: undefined });
  const cx = (b.minX + b.maxX) / 2, cy = (b.minY + b.maxY) / 2;
  return {
    path: mapPath(canvasPath, ([x, y]) => [x - cx, y - cy]),
    x: cx / Math.max(1, canvasWidth), y: cy / Math.max(1, canvasHeight),
  };
}

/** A canvas-pixel path expressed in `layer`'s own path coordinates (the inverse of `pathToCanvas`). */
export function canvasToLayerPath(canvasPath: PathBlock, layer: PhotoLayer, layerPath: PathBlock, canvasWidth: number, canvasHeight: number): PathBlock {
  const inverse = invert(pathToCanvas(layer, layerPath, canvasWidth, canvasHeight));
  return mapPath(canvasPath, ([x, y]) => applyMatrix(inverse, x, y));
}

/* -------------------------------------------------------------- booleans */

export type BooleanOp = "unite" | "subtract" | "intersect" | "exclude";

export const BOOLEAN_TOLERANCE = 0.1;
const PRECISION = 3;

/** The filled area of a path as Clipper polygons (open subpaths are closed, as a fill closes them). */
export function pathPolygons(path: PathBlock, tolerance = BOOLEAN_TOLERANCE): PathsD {
  const rings: PathsD = [];
  for (const subpath of path.subpaths) {
    const points = flattenSubpath({ ...subpath, closed: true }, tolerance);
    if (points.length >= 3) rings.push(points.map(([x, y]) => ({ x, y })));
  }
  return rings;
}

/** The path's fill as simple polygons, normalised by its own fill rule. */
export function normalisedPolygons(path: PathBlock, tolerance = BOOLEAN_TOLERANCE): PathsD {
  const rule = path.fillRule === "evenodd" ? ClipperFill.EvenOdd : ClipperFill.NonZero;
  return Clipper.unionD(pathPolygons(path, tolerance), [], rule, PRECISION);
}

const CLIP_TYPE: Record<BooleanOp, ClipType> = {
  unite: ClipType.Union, subtract: ClipType.Difference, intersect: ClipType.Intersection, exclude: ClipType.Xor,
};

/** `a op b`, both in the same coordinates. The result keeps `a`'s style, rule non-zero. */
export function booleanPaths(a: PathBlock, b: PathBlock, op: BooleanOp, tolerance = BOOLEAN_TOLERANCE): PathBlock {
  const result = Clipper.booleanOpD(CLIP_TYPE[op], normalisedPolygons(a, tolerance), normalisedPolygons(b, tolerance), ClipperFill.NonZero, PRECISION);
  return { ...a, fillRule: "nonzero", envelope: undefined, subpaths: polygonsToSubpaths(result) };
}

/** Polygons as closed subpaths of corner anchors. */
export function polygonsToSubpaths(polygons: PathsD): Subpath[] {
  return polygons.filter((ring) => ring.length >= 3).map((ring) => ({
    closed: true, nodes: ring.map((p): PathNode => ({ x: p.x, y: p.y })),
  }));
}

/** The filled area in square pixels (fill rule honoured, overlaps counted once). */
export function filledArea(path: PathBlock, tolerance = BOOLEAN_TOLERANCE): number {
  return Math.abs(normalisedPolygons(path, tolerance).reduce((sum, ring) => sum + polygonArea(ring.map((p): Point => [p.x, p.y])), 0));
}

/** Every subpath of both, in one path (a compound path): `a`'s style and fill rule. */
export function compoundPaths(a: PathBlock, b: PathBlock): PathBlock {
  return { ...a, subpaths: [...a.subpaths, ...b.subpaths] };
}

/**
 * The stroke as a filled shape ("Outline stroke"): the centre line offset both ways by half the
 * width with the path's join, cap and miter limit, filled with the stroke's paint.
 */
export function outlineStroke(path: PathBlock, tolerance = BOOLEAN_TOLERANCE): PathBlock | undefined {
  if (!path.stroke || path.strokeWidth <= 0) return undefined;
  const join = path.join === "round" ? JoinType.Round : path.join === "bevel" ? JoinType.Bevel : JoinType.Miter;
  const results: PathsD = [];
  for (const subpath of path.subpaths) {
    const points = flattenSubpath(subpath, tolerance).map(([x, y]) => ({ x, y }));
    if (points.length < 2) continue;
    const end = subpath.closed ? EndType.Joined
      : path.cap === "round" ? EndType.Round : path.cap === "square" ? EndType.Square : EndType.Butt;
    results.push(...Clipper.inflatePathsD([points], path.strokeWidth / 2, join, end, path.miterLimit, PRECISION, tolerance));
  }
  const united = Clipper.unionD(results, [], ClipperFill.NonZero, PRECISION);
  return {
    ...path, subpaths: polygonsToSubpaths(united), fillRule: "nonzero", fill: path.stroke, stroke: undefined,
    strokeWidth: 0, dash: undefined, dashOffset: undefined, envelope: undefined,
  };
}

/* ------------------------------------------------------------ Make Blend */

/** Segments with lengths, for picking which to split. */
const chord = (c: Cubic) => Math.hypot(c[3][0] - c[0][0], c[3][1] - c[0][1]) + Math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]) * 0.1;

/** Split the longest segments (at t = 0.5) until there are `count`: the same curve, more anchors. */
function toSegmentCount(cubics: Cubic[], count: number): Cubic[] {
  const out = cubics.slice();
  while (out.length < count) {
    let longest = 0;
    for (let i = 1; i < out.length; i++) if (chord(out[i]) > chord(out[longest])) longest = i;
    const [head, tail] = splitCubic(out[longest], 0.5);
    out.splice(longest, 1, head, tail);
  }
  return out;
}

const reverseCubics = (cubics: Cubic[]): Cubic[] => cubics.slice().reverse().map((c): Cubic => [c[3], c[2], c[1], c[0]]);

/** A point cubic at `p` (a subpath that shrinks to nothing when the other side has no partner). */
const pointCubics = (p: Point, count: number): Cubic[] => Array.from({ length: count }, (): Cubic => [p, p, p, p]);

function centroidOf(subpath: Subpath): Point {
  const points = flattenSubpath(subpath, 1);
  if (!points.length) return [0, 0];
  const sum = points.reduce((acc, p) => [acc[0] + p[0], acc[1] + p[1]] as Point, [0, 0] as Point);
  return [sum[0] / points.length, sum[1] / points.length];
}

/**
 * Two subpaths with the same number of segments, lined up for interpolation: same direction, and
 * (for closed ones) the rotation of `b`'s start that puts its anchors nearest `a`'s.
 */
export function matchSubpaths(a: Subpath, b: Subpath): [Cubic[], Cubic[], boolean] {
  let ca = segmentsOf(a), cb = segmentsOf(b);
  if (!ca.length && !cb.length) return [[], [], a.closed];
  if (!ca.length) ca = pointCubics(centroidOf(a.nodes.length ? a : b), cb.length);
  if (!cb.length) cb = pointCubics(centroidOf(b.nodes.length ? b : a), ca.length);
  const closed = a.closed && b.closed;
  if (closed) {
    const area = (cs: Cubic[]) => polygonArea(cs.map((c) => c[0]));
    if (Math.sign(area(ca)) !== Math.sign(area(cb)) && area(ca) !== 0 && area(cb) !== 0) cb = reverseCubics(cb);
  } else {
    const d = (p: Point, q: Point) => Math.hypot(p[0] - q[0], p[1] - q[1]);
    const straight = d(ca[0][0], cb[0][0]) + d(ca[ca.length - 1][3], cb[cb.length - 1][3]);
    const crossed = d(ca[0][0], cb[cb.length - 1][3]) + d(ca[ca.length - 1][3], cb[0][0]);
    if (crossed < straight) cb = reverseCubics(cb);
  }
  const count = Math.max(ca.length, cb.length);
  ca = toSegmentCount(ca, count);
  cb = toSegmentCount(cb, count);
  if (closed && count > 1) {
    let best = 0, bestCost = Infinity;
    for (let shift = 0; shift < count; shift++) {
      let cost = 0;
      for (let i = 0; i < count; i++) {
        const p = ca[i][0], q = cb[(i + shift) % count][0];
        cost += (p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2;
      }
      if (cost < bestCost) { bestCost = cost; best = shift; }
    }
    cb = cb.slice(best).concat(cb.slice(0, best));
  }
  return [ca, cb, closed];
}

const mix = (a: number, b: number, t: number) => a + (b - a) * t;
const mixPoint = (p: Point, q: Point, t: number): Point => [mix(p[0], q[0], t), mix(p[1], q[1], t)];

/** Colours mixed in linear light, which keeps a red-to-green blend from going muddy through brown. */
export function mixColour(a: string, b: string, t: number): string {
  const parse = (hex: string) => {
    const h = hex.replace("#", "");
    const full = h.length === 3 ? h.split("").map((c) => c + c).join("") : h.padEnd(6, "0");
    return [0, 2, 4].map((i) => parseInt(full.slice(i, i + 2), 16) / 255);
  };
  const toLinear = (c: number) => c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
  const toSrgb = (c: number) => c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055;
  const pa = parse(a), pb = parse(b);
  return `#${pa.map((c, i) => Math.round(Math.min(1, Math.max(0, toSrgb(mix(toLinear(c), toLinear(pb[i]), t)))) * 255)
    .toString(16).padStart(2, "0")).join("")}`;
}

/** Solid paints mix; anything else switches halfway. Absent (no fill) mixes as absent until halfway. */
export function mixPaint(a: Paint | undefined, b: Paint | undefined, t: number): Paint | undefined {
  if (a?.kind === "solid" && b?.kind === "solid") return { kind: "solid", color: mixColour(a.color, b.color, t) };
  return t < 0.5 ? a : b;
}

/** One in-between of two paths, `t` from 0 (`a`) to 1 (`b`), both in the same coordinates. */
export function blendPath(a: PathBlock, b: PathBlock, t: number): PathBlock {
  const count = Math.max(a.subpaths.length, b.subpaths.length);
  const subpaths: Subpath[] = [];
  for (let i = 0; i < count; i++) {
    const sa = a.subpaths[i] ?? { nodes: [], closed: b.subpaths[i].closed };
    const sb = b.subpaths[i] ?? { nodes: [], closed: a.subpaths[i].closed };
    const [ca, cb, closed] = matchSubpaths(sa, sb);
    if (!ca.length) continue;
    const cubics = ca.map((c, k): Cubic => [mixPoint(c[0], cb[k][0], t), mixPoint(c[1], cb[k][1], t), mixPoint(c[2], cb[k][2], t), mixPoint(c[3], cb[k][3], t)]);
    subpaths.push(subpathFromCubics(cubics, closed));
  }
  return {
    ...(t < 0.5 ? a : b),
    subpaths,
    fill: mixPaint(a.fill, b.fill, t),
    stroke: mixPaint(a.stroke, b.stroke, t),
    strokeWidth: mix(a.strokeWidth, b.strokeWidth, t),
    miterLimit: mix(a.miterLimit, b.miterLimit, t),
    envelope: undefined,
  };
}

/** `steps` in-betweens (not counting the ends), evenly spaced. */
export function blendSteps(a: PathBlock, b: PathBlock, steps: number): PathBlock[] {
  const n = Math.max(1, Math.min(256, Math.round(steps)));
  return Array.from({ length: n }, (_, i) => blendPath(a, b, (i + 1) / (n + 1)));
}

/** The midpoint of a cubic, for tests and overlays. */
export const cubicMidpoint = (c: Cubic): Point => cubicAt(c, 0.5);
