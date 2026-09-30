/**
 * VFX Phase 6: envelope distort.
 *
 * The envelope is a tensor-product Bezier surface: (cols + 1) x (rows + 1) control points over a
 * rectangle. Bernstein polynomials reproduce the identity when the points are evenly spaced (linear
 * precision), so a fresh envelope changes nothing, and moving any point bends everything smoothly --
 * a 4 x 4 grid (cols = rows = 3) is the bicubic patch Illustrator's "Make with Mesh" gives.
 *
 * Two uses, one surface:
 * - on a path (vector): the path is subdivided so each piece is short against the bend, and every
 *   anchor and handle is mapped through the surface. The result is still Bezier curves, so it stays
 *   sharp at any scale (`envelopePath`).
 * - on pixels (a node on any layer): the surface is tessellated into small triangles, each drawn
 *   with its own affine map back into the picture and a bilinear fetch (`applyEnvelopeRaster`).
 *   Nothing outside the rectangle moves unless the surface pulls something over it. The GPU
 *   compositor runs this very function on its CPU path, so the two compositors agree exactly.
 */

import type { Envelope, PathBlock, Point } from "./photoPath";
import { mapPath, subdivided } from "./photoPath";

const binomial = (n: number, k: number): number => {
  let value = 1;
  for (let i = 1; i <= k; i++) value = (value * (n - k + i)) / i;
  return value;
};

function bernstein(n: number, t: number): number[] {
  const out: number[] = [];
  for (let i = 0; i <= n; i++) out.push(binomial(n, i) * t ** i * (1 - t) ** (n - i));
  return out;
}

/** An identity envelope of (cols + 1) x (rows + 1) points over `rect`. */
export function createEnvelope(rect: Envelope["rect"], cols = 3, rows = 3): Envelope {
  const c = Math.max(1, Math.min(6, Math.round(cols)));
  const r = Math.max(1, Math.min(6, Math.round(rows)));
  const points: Point[] = [];
  for (let j = 0; j <= r; j++) {
    for (let i = 0; i <= c; i++) points.push([rect.x + (rect.width * i) / c, rect.y + (rect.height * j) / r]);
  }
  return { cols: c, rows: r, rect: { ...rect }, points };
}

/** Where the point `p` (in the rectangle's space) goes. Outside the rectangle it extrapolates. */
export function envelopeAt(envelope: Envelope, p: Point): Point {
  const { rect, cols, rows, points } = envelope;
  const u = rect.width ? (p[0] - rect.x) / rect.width : 0;
  const v = rect.height ? (p[1] - rect.y) / rect.height : 0;
  return envelopeUv(envelope, u, v, points, cols, rows);
}

function envelopeUv(_envelope: Envelope, u: number, v: number, points: Point[], cols: number, rows: number): Point {
  const bu = bernstein(cols, u), bv = bernstein(rows, v);
  let x = 0, y = 0;
  for (let j = 0; j <= rows; j++) {
    for (let i = 0; i <= cols; i++) {
      const w = bu[i] * bv[j];
      const p = points[j * (cols + 1) + i];
      x += w * p[0];
      y += w * p[1];
    }
  }
  return [x, y];
}

/** True when every point sits where the identity puts it (so the envelope does nothing). */
export function isIdentityEnvelope(envelope: Envelope): boolean {
  const rest = createEnvelope(envelope.rect, envelope.cols, envelope.rows);
  return envelope.points.length === rest.points.length
    && envelope.points.every((p, i) => Math.abs(p[0] - rest.points[i][0]) < 1e-9 && Math.abs(p[1] - rest.points[i][1]) < 1e-9);
}

/** Pieces each segment is cut into before warping: enough that a strong bend stays within ~0.1 px. */
export const ENVELOPE_PIECES = 8;

/** A path bent by the envelope, still as cubic Beziers. */
export function envelopePath(path: PathBlock, envelope: Envelope): PathBlock {
  if (envelope.points.length !== (envelope.cols + 1) * (envelope.rows + 1)) return path;
  return mapPath(subdivided(path, ENVELOPE_PIECES), (p) => envelopeAt(envelope, p));
}

/**
 * Pixels bent by an envelope whose rect and points are NORMALISED to the buffer (0..1), in place.
 *
 * The rectangle's content is lifted out (the area it leaves is cleared), and the surface, cut into
 * `grid` x `grid` quads of two triangles each, is drawn back with an affine map per triangle and a
 * bilinear fetch from the untouched source. Pixel centres decide coverage, so the result is exactly
 * repeatable.
 */
export function applyEnvelopeRaster(data: Uint8ClampedArray, width: number, height: number, envelope: Envelope, grid = 32): void {
  const { rect } = envelope;
  if (rect.width <= 0 || rect.height <= 0 || isIdentityEnvelope(envelope)) return;
  const source = new Uint8ClampedArray(data);
  const toPx = (p: Point): Point => [p[0] * width, p[1] * height];
  // Clear the lifted rectangle.
  const x0 = Math.max(0, Math.floor(rect.x * width)), x1 = Math.min(width, Math.ceil((rect.x + rect.width) * width));
  const y0 = Math.max(0, Math.floor(rect.y * height)), y1 = Math.min(height, Math.ceil((rect.y + rect.height) * height));
  for (let y = y0; y < y1; y++) data.fill(0, (y * width + x0) * 4, (y * width + x1) * 4);

  const n = Math.max(2, Math.min(128, Math.round(grid)));
  const warped: Point[] = [];
  const rest: Point[] = [];
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const u = i / n, v = j / n;
      warped.push(toPx(envelopeUv(envelope, u, v, envelope.points, envelope.cols, envelope.rows)));
      rest.push(toPx([rect.x + rect.width * u, rect.y + rect.height * v]));
    }
  }
  const at = (i: number, j: number) => j * (n + 1) + i;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = at(i, j), b = at(i + 1, j), c = at(i, j + 1), d = at(i + 1, j + 1);
      drawTriangle(data, source, width, height, [warped[a], warped[b], warped[d]], [rest[a], rest[b], rest[d]]);
      drawTriangle(data, source, width, height, [warped[a], warped[d], warped[c]], [rest[a], rest[d], rest[c]]);
    }
  }
}

/** One triangle: pixels whose centres fall inside `to`, fetched from the matching point of `from`. */
function drawTriangle(
  data: Uint8ClampedArray, source: Uint8ClampedArray, width: number, height: number, to: Point[], from: Point[],
): void {
  const [p0, p1, p2] = to;
  const area = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]);
  if (Math.abs(area) < 1e-9) return;
  const minX = Math.max(0, Math.floor(Math.min(p0[0], p1[0], p2[0]) - 0.5));
  const maxX = Math.min(width - 1, Math.ceil(Math.max(p0[0], p1[0], p2[0]) + 0.5));
  const minY = Math.max(0, Math.floor(Math.min(p0[1], p1[1], p2[1]) - 0.5));
  const maxY = Math.min(height - 1, Math.ceil(Math.max(p0[1], p1[1], p2[1]) + 0.5));
  for (let y = minY; y <= maxY; y++) {
    const cy = y + 0.5;
    for (let x = minX; x <= maxX; x++) {
      const cx = x + 0.5;
      // Barycentric weights. A centre exactly on a shared edge is drawn by both triangles; the later
      // one wins, which is deterministic, and both fetch the same source point there anyway.
      const w0 = ((p1[0] - cx) * (p2[1] - cy) - (p2[0] - cx) * (p1[1] - cy)) / area;
      const w1 = ((p2[0] - cx) * (p0[1] - cy) - (p0[0] - cx) * (p2[1] - cy)) / area;
      const w2 = 1 - w0 - w1;
      if (w0 < 0 || w1 < 0 || w2 < 0) continue;
      const sx = w0 * from[0][0] + w1 * from[1][0] + w2 * from[2][0] - 0.5;
      const sy = w0 * from[0][1] + w1 * from[1][1] + w2 * from[2][1] - 0.5;
      bilinear(source, width, height, sx, sy, data, (y * width + x) * 4);
    }
  }
}

/** A bilinear fetch of straight RGBA, weighting colour by alpha (no dark fringes), transparent outside. */
function bilinear(source: Uint8ClampedArray, width: number, height: number, x: number, y: number, out: Uint8ClampedArray, at: number): void {
  const fx = Math.floor(x), fy = Math.floor(y);
  const tx = x - fx, ty = y - fy;
  let r = 0, g = 0, b = 0, a = 0;
  for (let k = 0; k < 4; k++) {
    const px = fx + (k & 1), py = fy + (k >> 1);
    if (px < 0 || py < 0 || px >= width || py >= height) continue;
    const w = (k & 1 ? tx : 1 - tx) * (k >> 1 ? ty : 1 - ty);
    const i = (py * width + px) * 4;
    const alpha = source[i + 3] * w;
    r += source[i] * alpha; g += source[i + 1] * alpha; b += source[i + 2] * alpha; a += alpha;
  }
  if (a <= 0) { out[at] = 0; out[at + 1] = 0; out[at + 2] = 0; out[at + 3] = 0; return; }
  out[at] = Math.round(r / a); out[at + 1] = Math.round(g / a); out[at + 2] = Math.round(b / a); out[at + 3] = Math.round(a);
}
