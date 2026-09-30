import { describe, expect, it } from "vitest";
import { applyEnvelopeRaster, createEnvelope, envelopeAt, envelopePath, isIdentityEnvelope } from "./photoEnvelope";
import {
  appendAnchor, closeSubpath, deleteAnchor, flattenSubpath, hitAnchor, hitHandle, moveAnchor, NEUTRAL_PATH, pathBounds, pathBox,
  pathData, pointAlong, polygonArea, segmentsOf, setHandle, shapedPath, subdivided, toggleSmooth,
  type PathBlock, type Subpath,
} from "./photoPath";
import {
  blendPath, blendSteps, booleanPaths, compoundPaths, filledArea, matchSubpaths, mixColour, outlineStroke,
} from "./photoPathOps";

/** The standard 4-cubic circle (kappa 0.5523), centred at (cx, cy). */
function circle(cx: number, cy: number, r: number): Subpath {
  const k = 0.5522847498 * r;
  return {
    closed: true,
    nodes: [
      { x: cx + r, y: cy, in: [0, -k], out: [0, k] },
      { x: cx, y: cy + r, in: [k, 0], out: [-k, 0] },
      { x: cx - r, y: cy, in: [0, k], out: [0, -k] },
      { x: cx, y: cy - r, in: [-k, 0], out: [k, 0] },
    ],
  };
}

const square = (x: number, y: number, size: number, clockwise = true): Subpath => ({
  closed: true,
  nodes: (clockwise
    ? [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]
    : [[x, y], [x, y + size], [x + size, y + size], [x + size, y]]).map(([px, py]) => ({ x: px, y: py })),
});

const path = (subpaths: Subpath[], over: Partial<PathBlock> = {}): PathBlock => ({ ...NEUTRAL_PATH, subpaths, ...over });

describe("path geometry (VFX Phase 6)", () => {
  it("flattens curves within tolerance: a Bezier circle's area", () => {
    const points = flattenSubpath(circle(0, 0, 100), 0.05);
    // The 4-cubic circle is itself 0.027% larger than a true circle at its widest.
    expect(Math.abs(polygonArea(points)) / (Math.PI * 100 * 100)).toBeCloseTo(1, 3);
    // Booleans flatten at 0.1 px: chords sit inside the curve, so the area is a hair (< 0.2%) small.
    expect(filledArea(path([circle(0, 0, 100)])) / (Math.PI * 1e4)).toBeGreaterThan(0.998);
  });

  it("measures tight bounds and a symmetric raster box that includes the stroke", () => {
    const b = pathBounds(path([circle(10, -5, 50)]));
    expect(b.minX).toBeCloseTo(-40, 6); expect(b.maxX).toBeCloseTo(60, 6);
    expect(b.minY).toBeCloseTo(-55, 6); expect(b.maxY).toBeCloseTo(45, 6);
    const box = pathBox(path([circle(10, -5, 50)], { strokeWidth: 4, join: "round" }));
    // Half-width reach max(|-40|, 60) + 2 (stroke) + 2 (antialiasing) = 64 -> 128 wide, centred.
    expect(box).toEqual({ width: 128, height: 118, offsetX: 64, offsetY: 59 });
    const miter = pathBox(path([square(0, 0, 10)], { strokeWidth: 2, join: "miter", miterLimit: 10 }));
    expect(miter.width).toBe(2 * Math.ceil(10 + 10 + 2));
    expect(pathBox(path([square(0, 0, 10)], { stroke: undefined })).width).toBe(24);
  });

  it("walks a polyline by arc length, wrapping a closed one", () => {
    const ring: [number, number][] = [[0, 0], [10, 0], [10, 10], [0, 10]];
    expect(pointAlong(ring, true, 15)).toEqual({ at: [10, 5], angle: Math.PI / 2 });
    expect(pointAlong(ring, true, 45).at).toEqual([5, 0]);
    expect(pointAlong(ring, false, 100).at).toEqual([0, 10]);
  });

  it("subdivides without changing the curve", () => {
    const p = path([circle(0, 0, 80)]);
    const fine = subdivided(p, 3);
    expect(fine.subpaths[0].nodes).toHaveLength(12);
    const a = flattenSubpath(p.subpaths[0], 0.01), b = flattenSubpath(fine.subpaths[0], 0.01);
    expect(Math.abs(polygonArea(a) - polygonArea(b))).toBeLessThan(0.5);
  });

  it("edits anchors and handles like a pen tool", () => {
    let p = path([]);
    p = appendAnchor(p, 0, [0, 0]);
    p = appendAnchor(p, 0, [100, 0]);
    p = appendAnchor(p, 0, [100, 100]);
    expect(p.subpaths).toHaveLength(1);
    expect(p.subpaths[0].nodes).toHaveLength(3);
    // Pen drag on the last anchor: out handle to (130, 100), mirrored at equal length.
    p = setHandle(p, 0, 2, "out", [130, 100], "equal");
    expect(p.subpaths[0].nodes[2]).toEqual({ x: 100, y: 100, out: [30, 0], in: [-30, -0] });
    // A smooth point keeps its other handle's length when one is dragged ("angle").
    p = setHandle(p, 0, 2, "out", [100, 140], "angle");
    expect(p.subpaths[0].nodes[2].in![0]).toBeCloseTo(0, 9);
    expect(p.subpaths[0].nodes[2].in![1]).toBeCloseTo(-30, 9);
    expect(hitAnchor(p, [99, 101], 4)).toEqual([0, 2]);
    expect(hitHandle(p, [100, 139], 4)).toEqual([0, 2, "out"]);
    p = moveAnchor(p, 0, 2, [110, 110]);
    expect(p.subpaths[0].nodes[2].out).toEqual([0, 40]);
    p = closeSubpath(p, 0);
    expect(segmentsOf(p.subpaths[0])).toHaveLength(3);
    p = toggleSmooth(p, 0, 0);
    expect(p.subpaths[0].nodes[0].in && p.subpaths[0].nodes[0].out).toBeTruthy();
    p = toggleSmooth(p, 0, 0);
    expect(p.subpaths[0].nodes[0]).toEqual({ x: 0, y: 0 });
    p = deleteAnchor(p, 0, 1);
    expect(p.subpaths[0].nodes).toHaveLength(2);
    expect(pathData(p)).toBe("M0 0C0 0 110 80 110 110C110 150 0 0 0 0Z");
  });
});

describe("booleans and compound paths", () => {
  const a = path([square(0, 0, 100)]);
  const b = path([square(50, 50, 100)], { fill: { kind: "solid", color: "#ff0000" } });

  it("gives the known areas of two overlapping squares", () => {
    expect(filledArea(booleanPaths(a, b, "unite"))).toBeCloseTo(17500, 6);
    expect(filledArea(booleanPaths(a, b, "intersect"))).toBeCloseTo(2500, 6);
    expect(filledArea(booleanPaths(a, b, "subtract"))).toBeCloseTo(7500, 6);
    expect(filledArea(booleanPaths(a, b, "exclude"))).toBeCloseTo(15000, 6);
    expect(booleanPaths(a, b, "unite").fill).toEqual(a.fill);
    // Corner anchors only: an L-shaped octagon, no collinear extras.
    expect(booleanPaths(a, b, "unite").subpaths.map((sub) => sub.nodes.length)).toEqual([8]);
  });

  it("gives the lens area of two intersecting circles", () => {
    const r = 100, d = 100;
    const lens = 2 * r * r * Math.acos(d / (2 * r)) - (d / 2) * Math.sqrt(4 * r * r - d * d);
    const area = filledArea(booleanPaths(path([circle(0, 0, r)]), path([circle(d, 0, r)]), "intersect"));
    expect(Math.abs(area - lens) / lens).toBeLessThan(0.002);
  });

  it("honours each operand's own fill rule", () => {
    const twice = path([square(0, 0, 100), square(0, 0, 100)], { fillRule: "evenodd" });
    expect(filledArea(twice)).toBeCloseTo(0, 6);
    expect(filledArea({ ...twice, fillRule: "nonzero" })).toBeCloseTo(10000, 6);
    // A ring as a compound path: outer clockwise, inner anticlockwise, non-zero -> a hole.
    const ring = compoundPaths(path([square(0, 0, 100)]), path([square(25, 25, 50, false)]));
    expect(filledArea(ring)).toBeCloseTo(7500, 6);
    // Unite of an even-odd ring with a square filling its hole: the full square.
    const eoRing = path([square(0, 0, 100), square(25, 25, 50)], { fillRule: "evenodd" });
    expect(filledArea(booleanPaths(eoRing, path([square(25, 25, 50)]), "unite"))).toBeCloseTo(10000, 6);
  });

  it("outlines a stroke with its caps and joins", () => {
    const line = path([{ closed: false, nodes: [{ x: 0, y: 0 }, { x: 100, y: 0 }] }], { strokeWidth: 10, cap: "butt" });
    expect(filledArea(outlineStroke(line)!)).toBeCloseTo(1000, 3);
    expect(filledArea(outlineStroke({ ...line, cap: "square" })!)).toBeCloseTo(1100, 3);
    expect(filledArea(outlineStroke({ ...line, cap: "round" })!) / (1000 + Math.PI * 25)).toBeCloseTo(1, 2);
    const box = path([square(0, 0, 100)], { strokeWidth: 10, join: "miter" });
    expect(filledArea(outlineStroke(box)!)).toBeCloseTo(110 * 110 - 90 * 90, 3);
    expect(outlineStroke({ ...box, stroke: undefined })).toBeUndefined();
    expect(outlineStroke(box)!.fill).toEqual(box.stroke);
  });
});

describe("Make Blend", () => {
  it("matches anchor counts, direction and start, then interpolates", () => {
    const sq = square(-50, -50, 100);
    const ci = circle(0, 0, 50);
    const [ca, cb] = matchSubpaths({ ...sq, nodes: [...sq.nodes].reverse() }, ci);
    expect(ca.length).toBe(cb.length);
    const half = blendPath(path([sq]), path([ci], { strokeWidth: 10 }), 0.5);
    // Every anchor of the in-between is the midpoint of the matched anchors.
    const [ma, mb] = matchSubpaths(sq, ci);
    half.subpaths[0].nodes.forEach((node, k) => {
      expect(node.x).toBeCloseTo((ma[k][0][0] + mb[k][0][0]) / 2, 9);
      expect(node.y).toBeCloseTo((ma[k][0][1] + mb[k][0][1]) / 2, 9);
    });
    const area = filledArea(half);
    expect(area).toBeGreaterThan(7000);
    expect(area).toBeLessThan(10000);
    expect(half.strokeWidth).toBe(7);
    expect(filledArea(blendPath(path([sq]), path([ci]), 0))).toBeCloseTo(10000, 3);
    expect(filledArea(blendPath(path([sq]), path([ci]), 1))).toBeCloseTo(filledArea(path([ci])), 1);
  });

  it("makes N evenly spaced steps with mixed colours", () => {
    const a = path([square(0, 0, 10)], { fill: { kind: "solid", color: "#ff0000" } });
    const b = path([square(100, 0, 10)], { fill: { kind: "solid", color: "#0000ff" } });
    const steps = blendSteps(a, b, 3);
    expect(steps).toHaveLength(3);
    expect(steps.map((step) => step.subpaths[0].nodes[0].x)).toEqual([25, 50, 75]);
    expect(mixColour("#ff0000", "#0000ff", 0)).toBe("#ff0000");
    expect(mixColour("#ff0000", "#0000ff", 1)).toBe("#0000ff");
    expect(mixColour("#000000", "#ffffff", 0.5)).toBe("#bcbcbc");
  });
});

describe("envelope", () => {
  const rect = { x: -50, y: -50, width: 100, height: 100 };

  it("is the identity until a point moves", () => {
    const envelope = createEnvelope(rect, 3, 3);
    expect(isIdentityEnvelope(envelope)).toBe(true);
    const at = envelopeAt(envelope, [12.5, -30]);
    expect(at[0]).toBeCloseTo(12.5, 9); expect(at[1]).toBeCloseTo(-30, 9);
    const p = path([circle(0, 0, 40)]);
    const warped = envelopePath(p, envelope);
    const plain = subdivided(p, 8);
    warped.subpaths[0].nodes.forEach((node, k) => {
      const same = plain.subpaths[0].nodes[k];
      expect(node.x).toBeCloseTo(same.x, 9);
      expect(node.y).toBeCloseTo(same.y, 9);
      expect(node.out![0]).toBeCloseTo(same.out![0], 9);
    });
  });

  it("bends the path and keeps the curves", () => {
    const envelope = createEnvelope(rect, 1, 1);
    // Pull the top-right corner out: the square becomes a quadrilateral with a known area.
    envelope.points[1] = [100, -50];
    const p = path([square(-50, -50, 100)]);
    const warped = shapedPath({ ...p, envelope });
    expect(warped.envelope).toBeUndefined();
    // Bilinear with corners (-50,-50) (100,-50) (-50,50) (50,50): the area of that trapezoid.
    expect(filledArea(warped)).toBeCloseTo((150 + 100) / 2 * 100, 1);
    expect(pathBounds({ ...p, envelope }).maxX).toBeCloseTo(100, 6);
  });

  it("moves pixels with the surface, and changes nothing when it is the identity", () => {
    const width = 40, height = 20;
    const data = new Uint8ClampedArray(width * height * 4);
    // A 4x4 opaque red block at (10..14, 8..12).
    for (let y = 8; y < 12; y++) for (let x = 10; x < 14; x++) data.set([255, 0, 0, 255], (y * width + x) * 4);
    const copy = new Uint8ClampedArray(data);
    applyEnvelopeRaster(copy, width, height, createEnvelope({ x: 0, y: 0, width: 0.5, height: 1 }, 1, 1));
    expect(copy).toEqual(data);
    // Every point moved right by 5 px (0.125 of the width): the block moves 5 px right.
    const moved = createEnvelope({ x: 0, y: 0, width: 0.5, height: 1 }, 1, 1);
    moved.points = moved.points.map(([x, y]) => [x + 0.125, y]);
    applyEnvelopeRaster(copy, width, height, moved);
    const alphaAt = (x: number, y: number) => copy[(y * width + x) * 4 + 3];
    expect(alphaAt(15, 10)).toBe(255);
    expect(alphaAt(18, 9)).toBe(255);
    expect(alphaAt(12, 10)).toBe(0);
    expect(alphaAt(19, 10)).toBe(0);
  });
});

describe("flattening straight segments", () => {
  it("keeps a corner-to-corner segment as one piece", () => {
    expect(flattenSubpath(square(0, 0, 100))).toHaveLength(4);
  });
});
