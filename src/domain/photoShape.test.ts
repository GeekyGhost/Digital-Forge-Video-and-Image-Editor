import { describe, expect, it } from "vitest";
import {
  BALLOON_KINDS, DECORATION_KINDS, NEUTRAL_SHAPE, SHAPE_PRESETS, bodyOutline, bodyPoint, hasTail,
  holdsText, shapeBounds, shapeInnerBox, shapePaths, superExponent, tailTip,
  type ShapeBlock, type ShapeKind,
} from "./photoShape";
import { NEUTRAL_TEXT, fitFontSize, paintsOf, type TextBlock } from "./photoText";
import { solidPaint, toPaint } from "./photoGradient";

const shape = (over: Partial<ShapeBlock> = {}): ShapeBlock => ({ ...NEUTRAL_SHAPE, ...over });
const ALL_KINDS: ShapeKind[] = [
  "rectangle", "rounded", "ellipse", "polygon", "star", "burst", "arrow", "banner",
  "heart", "lightning", "splat", "impact", "sunburst", "speedlines",
  "speech", "thought", "shout", "scream", "electric", "icicle", "wobble", "whisper", "caption",
];
/** The kinds that are one closed loop, which is what "inside" is even a question about. */
const SOLID_KINDS = ALL_KINDS.filter((kind) => holdsText(kind));

/** Is this point inside the closed polyline? Ray casting, which is the definition of "inside". */
function inside(points: Array<[number, number]>, x: number, y: number): boolean {
  let within = false;
  for (let index = 0, previous = points.length - 1; index < points.length; previous = index++) {
    const [xi, yi] = points[index];
    const [xj, yj] = points[previous];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) within = !within;
  }
  return within;
}

/* ------------------------------------------------------------- the outline */

describe("shapePaths", () => {
  it("gives every kind a closed outline with real points", () => {
    for (const kind of ALL_KINDS) {
      const paths = shapePaths(shape({ kind }));
      expect(paths.length, kind).toBeGreaterThan(0);
      for (const path of paths) {
        expect(path.closed, kind).toBe(true);
        expect(path.points.length, kind).toBeGreaterThan(3);
        for (const [x, y] of path.points) {
          expect(Number.isFinite(x) && Number.isFinite(y), kind).toBe(true);
        }
      }
    }
  });

  it("keeps a balloon and its tail as ONE path, so the stroke has no seam", () => {
    // The whole reason the tail is cut into the body's loop rather than added beside it: two paths
    // need their shared join painted over by the fill, which shows the moment the fill is not
    // opaque.
    for (const kind of ["speech", "shout", "scream", "electric", "icicle", "wobble",
      "whisper", "caption"] as ShapeKind[]) {
      expect(shapePaths(shape({ kind, tailLength: 0.6 })), kind).toHaveLength(1);
    }
  });

  it("gives a thought balloon separate bubbles, because that is what makes it a thought", () => {
    const paths = shapePaths(shape({ kind: "thought", tailLength: 0.7 }));
    expect(paths.length).toBeGreaterThan(1);
  });

  it("drops the tail entirely when its length goes to nothing", () => {
    const withTail = shape({ kind: "speech", tailLength: 0.6 });
    const without = shape({ kind: "speech", tailLength: 0 });
    expect(hasTail(withTail)).toBe(true);
    expect(hasTail(without)).toBe(false);
    // Without a tail the outline is a plain closed body, so it is shorter.
    expect(shapePaths(without)[0].points.length)
      .toBeLessThan(shapePaths(withTail)[0].points.length);
  });

  it("has no tail on a shape that is not a balloon, however long the setting", () => {
    expect(hasTail(shape({ kind: "star", tailLength: 2 }))).toBe(false);
    expect(BALLOON_KINDS).not.toContain("star");
  });

  it("puts the tail's tip outside the body, on the side it was aimed at", () => {
    const balloon = shape({ kind: "speech", tailAngle: 90, tailLength: 0.8 });
    const tip = tailTip(balloon);
    // Ninety degrees is down the screen, and past the body's own edge.
    expect(tip[1]).toBeGreaterThan(balloon.height / 2);
    expect(Math.abs(tip[0])).toBeLessThan(balloon.width / 4);
    expect(inside(shapePaths(shape({ kind: "speech", tailLength: 0 }))[0].points, tip[0], tip[1]))
      .toBe(false);
  });

  it("turns the tail with its angle", () => {
    const right = tailTip(shape({ kind: "speech", tailAngle: 0, tailLength: 0.6 }));
    const left = tailTip(shape({ kind: "speech", tailAngle: 180, tailLength: 0.6 }));
    expect(right[0]).toBeGreaterThan(0);
    expect(left[0]).toBeLessThan(0);
  });

  it("reaches further as the tail lengthens", () => {
    const near = tailTip(shape({ kind: "speech", tailAngle: 90, tailLength: 0.2 }));
    const far = tailTip(shape({ kind: "speech", tailAngle: 90, tailLength: 1.2 }));
    expect(far[1]).toBeGreaterThan(near[1]);
  });

  it("reaches the tip whichever way the tail is bent", () => {
    for (const tailCurve of [-1, 0, 1]) {
      const balloon = shape({ kind: "speech", tailCurve, tailLength: 0.8, tailAngle: 90 });
      const tip = tailTip(balloon);
      const points = shapePaths(balloon)[0].points;
      const closest = points.reduce((best, [x, y]) =>
        Math.min(best, Math.hypot(x - tip[0], y - tip[1])), Infinity);
      expect(closest, `curve ${tailCurve}`).toBeLessThan(1);
    }
  });
});

describe("bodyPoint", () => {
  it("is an ellipse at the ends of its own axes", () => {
    const oval = shape({ kind: "ellipse", width: 400, height: 200 });
    const [rx] = bodyPoint(oval, 0);
    const [, ry] = bodyPoint(oval, Math.PI / 2);
    expect(rx).toBeCloseTo(200, 3);
    expect(ry).toBeCloseTo(100, 3);
  });

  it("reaches the corner on a rectangle, which an ellipse does not", () => {
    const box = bodyPoint(shape({ kind: "rectangle", width: 400, height: 200 }), Math.PI / 4);
    const oval = bodyPoint(shape({ kind: "ellipse", width: 400, height: 200 }), Math.PI / 4);
    expect(Math.hypot(box[0], box[1])).toBeGreaterThan(Math.hypot(oval[0], oval[1]));
  });

  it("moves smoothly from ellipse to rectangle as roundness falls", () => {
    // One family with one exponent, which is why there is no seam where "rounded" stops being
    // rounded and starts being a box.
    const corner = (radius: number) => {
      const point = bodyPoint(shape({ kind: "rounded", radius, width: 200, height: 200 }), Math.PI / 4);
      return Math.hypot(point[0], point[1]);
    };
    expect(superExponent(shape({ kind: "ellipse" }))).toBe(2);
    expect(corner(1)).toBeLessThan(corner(0.5));
    expect(corner(0.5)).toBeLessThan(corner(0));
  });

  it("stays within the box it was given, for every kind and all the way round", () => {
    for (const kind of SOLID_KINDS) {
      const body = shape({ kind, width: 300, height: 180, tailLength: 0 });
      for (let step = 0; step < 64; step++) {
        const [x, y] = bodyPoint(body, (step / 64) * Math.PI * 2);
        expect(Math.abs(x), `${kind} x`).toBeLessThanOrEqual(151);
        expect(Math.abs(y), `${kind} y`).toBeLessThanOrEqual(91);
      }
    }
  });

  it("lands on the outline at every angle, not just at the corners", () => {
    // The bug this replaced: the old version picked the NEAREST corner and walked forward from it,
    // so for half of all angles the ray missed its segment and fell back to an unrelated radius.
    // A star sampled between its points came back with the radius OF a point.
    const star = shape({ kind: "star", points: 5, innerRadius: 0.4, width: 300, height: 300, spin: 0 });
    const outline = bodyOutline(star);
    for (let step = 0; step < 97; step++) {
      const angle = (step / 97) * Math.PI * 2;
      const [x, y] = bodyPoint(star, angle);
      // Every returned point lies on some edge of the outline, to within a rounding error.
      let nearest = Infinity;
      for (let index = 0; index < outline.length; index++) {
        const a = outline[index], b = outline[(index + 1) % outline.length];
        const ex = b[0] - a[0], ey = b[1] - a[1];
        const length = ex * ex + ey * ey;
        const t = length ? Math.max(0, Math.min(1, ((x - a[0]) * ex + (y - a[1]) * ey) / length)) : 0;
        nearest = Math.min(nearest, Math.hypot(x - a[0] - ex * t, y - a[1] - ey * t));
      }
      expect(nearest, `angle ${step}`).toBeLessThan(0.01);
      // And it points the way it was asked to.
      expect(Math.atan2(y, x)).toBeCloseTo(Math.atan2(Math.sin(angle), Math.cos(angle)), 4);
    }
  });

  it("varies a spiky shape's radius smoothly between its points, rather than in jumps", () => {
    // The old fallback produced a sawtooth: correct on one sample, a vertex radius on the next.
    const star = shape({ kind: "star", points: 5, innerRadius: 0.4, width: 300, height: 300 });
    const radii: number[] = [];
    for (let step = 0; step < 200; step++) {
      radii.push(Math.hypot(...bodyPoint(star, (step / 200) * Math.PI * 2)));
    }
    let worst = 0;
    for (let index = 1; index < radii.length; index++) {
      worst = Math.max(worst, Math.abs(radii[index] - radii[index - 1]));
    }
    // A star this size has 90 units between its outer and inner radius, and the old fallback could
    // jump the whole way between two neighbouring samples. The honest maximum is the slope just off
    // a sharp point, which is about thirteen; anything near ninety is the sawtooth coming back.
    expect(worst).toBeLessThan(25);
  });

  it("makes a star's inner vertices closer in than its outer ones", () => {
    const star = shape({ kind: "star", points: 5, innerRadius: 0.4, width: 200, height: 200 });
    const step = Math.PI * 2 / 10;
    const outer = Math.hypot(...bodyPoint(star, 0));
    const inner = Math.hypot(...bodyPoint(star, step));
    expect(inner).toBeLessThan(outer);
    expect(inner / outer).toBeCloseTo(0.4, 1);
  });
});

/* -------------------------------------------------------------- the extent */

describe("shapeBounds", () => {
  it("is at least the body, plus room for the stroke", () => {
    const box = shapeBounds(shape({ kind: "ellipse", width: 400, height: 200, strokeWidth: 20, tailLength: 0 }));
    expect(box.width).toBeGreaterThanOrEqual(400 + 20);
    expect(box.height).toBeGreaterThanOrEqual(200 + 20);
  });

  it("grows to contain the tail rather than clipping it", () => {
    const bare = shapeBounds(shape({ kind: "speech", tailAngle: 90, tailLength: 0 }));
    const tailed = shapeBounds(shape({ kind: "speech", tailAngle: 90, tailLength: 1 }));
    expect(tailed.height).toBeGreaterThan(bare.height);
  });

  it("grows for a drop shadow, which also reaches past the body", () => {
    const flat = shapeBounds(shape({ shadow: 0 }));
    const lifted = shapeBounds(shape({ shadow: 40, shadowOffset: 20 }));
    expect(lifted.width).toBeGreaterThan(flat.width);
    expect(lifted.height).toBeGreaterThan(flat.height);
  });

  it("reports an offset that puts every point of the outline inside the bitmap", () => {
    for (const kind of ALL_KINDS) {
      const body = shape({ kind, tailLength: 0.8, strokeWidth: 8 });
      const box = shapeBounds(body);
      for (const path of shapePaths(body)) {
        for (const [x, y] of path.points) {
          expect(x + box.offsetX, `${kind} left`).toBeGreaterThanOrEqual(0);
          expect(y + box.offsetY, `${kind} top`).toBeGreaterThanOrEqual(0);
          expect(x + box.offsetX, `${kind} right`).toBeLessThanOrEqual(box.width);
          expect(y + box.offsetY, `${kind} bottom`).toBeLessThanOrEqual(box.height);
        }
      }
    }
  });

  it("never returns a degenerate bitmap", () => {
    const box = shapeBounds(shape({ width: 1, height: 1, strokeWidth: 0, tailLength: 0 }));
    expect(box.width).toBeGreaterThan(0);
    expect(box.height).toBeGreaterThan(0);
  });
});

/* ---------------------------------------------------------- the inner box */

describe("shapeInnerBox", () => {
  it("fits inside the outline, so words cannot poke through the side", () => {
    for (const kind of SOLID_KINDS) {
      const body = shape({ kind, width: 400, height: 260, padding: 0, strokeWidth: 0, tailLength: 0 });
      const box = shapeInnerBox(body);
      const outline = bodyOutline(body);
      // Every corner of the text box lands inside the shape it is meant to sit in.
      for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]] as Array<[number, number]>) {
        expect(inside(outline, sx * box.width / 2 * 0.99, sy * box.height / 2 * 0.99), kind).toBe(true);
      }
    }
  });

  it("finds a box for every kind that can hold words, including ones it has never seen", () => {
    // The box is searched rather than guessed per kind, which is the point: a new shape gets a
    // correct answer without anyone picking a fraction for it.
    for (const kind of SOLID_KINDS) {
      const box = shapeInnerBox(shape({ kind, width: 400, height: 260, padding: 0, strokeWidth: 0 }));
      expect(box.width, kind).toBeGreaterThan(8);
      expect(box.height, kind).toBeGreaterThan(8);
    }
  });

  it("gives a rectangle more room than an ellipse of the same size", () => {
    const box = shapeInnerBox(shape({ kind: "rectangle", width: 400, height: 200, padding: 0, strokeWidth: 0 }));
    const oval = shapeInnerBox(shape({ kind: "ellipse", width: 400, height: 200, padding: 0, strokeWidth: 0 }));
    expect(box.width).toBeGreaterThan(oval.width);
  });

  it("takes the padding and the stroke off both sides", () => {
    const bare = shapeInnerBox(shape({ kind: "rectangle", padding: 0, strokeWidth: 0 }));
    const inset = shapeInnerBox(shape({ kind: "rectangle", padding: 20, strokeWidth: 10 }));
    expect(bare.width - inset.width).toBeCloseTo(60, 5);
  });

  it("never collapses, however much padding it is given", () => {
    const box = shapeInnerBox(shape({ width: 60, height: 40, padding: 500, strokeWidth: 100 }));
    expect(box.width).toBeGreaterThan(0);
    expect(box.height).toBeGreaterThan(0);
  });
});

/* ------------------------------------------------------------- the letters */

describe("fitFontSize", () => {
  const block = (over: Partial<TextBlock> = {}): TextBlock => ({ ...NEUTRAL_TEXT, ...over });
  /** One line, ten units wide per character per unit of size. Enough to reason about exactly. */
  const measurer = (text: string, lines = 1) =>
    (size: number) => ({ width: text.length * size * 0.5, height: lines * size * 1.2 });

  it("leaves the size alone when it already fits", () => {
    const size = fitFontSize(block({ fontSize: 40 }), { width: 1000, height: 1000 }, measurer("abc"));
    expect(size).toBe(40);
  });

  it("shrinks until it fits", () => {
    const size = fitFontSize(block({ fontSize: 100 }), { width: 100, height: 1000 }, measurer("abcdefghij"));
    // Ten characters at half a unit each: the widest that fits 100 is a size of 20.
    expect(size).toBeLessThanOrEqual(20);
    expect(size).toBeGreaterThan(19);
  });

  it("is bounded below, so it gives up rather than vanishing", () => {
    const size = fitFontSize(block({ fontSize: 100, fitMin: 24 }), { width: 1, height: 1 }, measurer("abcdefghij"));
    expect(size).toBe(24);
  });

  it("respects the height as well as the width", () => {
    // fitMin has to come down too, or the floor is what answers rather than the height.
    const size = fitFontSize(block({ fontSize: 100, fitMin: 1 }), { width: 100000, height: 60 }, measurer("a", 10));
    // Ten lines at 1.2 per unit: 60 of height allows a size of 5.
    expect(size).toBeLessThanOrEqual(5);
    expect(size).toBeGreaterThan(4);
  });

  it("stops at the floor even when the floor does not fit, rather than disappearing", () => {
    // Deliberate: lettering too small to read is not a better answer than lettering that overflows,
    // because one can be seen and fixed and the other looks like the tool lost the words.
    const measure = measurer("a", 10);
    const size = fitFontSize(block({ fontSize: 100, fitMin: 8 }), { width: 1e6, height: 60 }, measure);
    expect(size).toBe(8);
    expect(measure(size).height).toBeGreaterThan(60);
  });

  it("never returns something that does not fit", () => {
    for (const width of [10, 55, 130, 400]) {
      const measure = measurer("hello there");
      const size = fitFontSize(block({ fontSize: 200, fitMin: 1 }), { width, height: 1e6 }, measure);
      expect(measure(size).width, `width ${width}`).toBeLessThanOrEqual(width + 0.5);
    }
  });
});

/* --------------------------------------------------------------- the paint */

describe("paints read through whatever was stored", () => {
  it("takes a plain string, which is what older documents hold", () => {
    // Text colours were hex strings before they could be ramps, and those documents are in project
    // files on disk. Read through rather than migrated: nothing can be half upgraded.
    expect(toPaint("#ff0000")).toEqual({ kind: "solid", color: "#ff0000" });
  });

  it("takes a paint unchanged", () => {
    const paint = solidPaint("#123456");
    expect(toPaint(paint)).toBe(paint);
  });

  it("falls back rather than producing something undrawable", () => {
    expect(toPaint(undefined, "#00ff00")).toEqual({ kind: "solid", color: "#00ff00" });
    expect(toPaint(null, "#00ff00")).toEqual({ kind: "solid", color: "#00ff00" });
    expect(toPaint({ kind: "nonsense" }, "#00ff00")).toEqual({ kind: "solid", color: "#00ff00" });
  });

  it("reads a whole text block written before paints existed", () => {
    const older = {
      ...NEUTRAL_TEXT,
      color: "#111111", strokeColor: "#222222", shadowColor: "#333333",
      panelColor: "#444444", panelBorderColor: "#555555",
    } as unknown as TextBlock;
    const paints = paintsOf(older);
    expect(paints.fill).toEqual({ kind: "solid", color: "#111111" });
    expect(paints.stroke).toEqual({ kind: "solid", color: "#222222" });
    expect(paints.shadow).toEqual({ kind: "solid", color: "#333333" });
    expect(paints.panel).toEqual({ kind: "solid", color: "#444444" });
    expect(paints.panelBorder).toEqual({ kind: "solid", color: "#555555" });
  });
});

/* ------------------------------------------------------------- the presets */

describe("the shape presets", () => {
  it("keeps irregular shapes fixed, so they do not crawl as the frame redraws", () => {
    const shout = shape({ kind: "shout", jitter: 0.8, seed: 4 });
    expect(shapePaths(shout)[0].points).toEqual(shapePaths(shout)[0].points);
    const first = JSON.stringify(shapePaths(shout));
    const again = JSON.stringify(shapePaths(shape({ kind: "shout", jitter: 0.8, seed: 4 })));
    expect(again).toBe(first);
  });

  it("gives a different arrangement for a different seed, and the same for the same one", () => {
    const at = (seed: number) => JSON.stringify(shapePaths(shape({ kind: "shout", jitter: 0.8, seed })));
    expect(at(4)).not.toBe(at(9));
    expect(at(4)).toBe(at(4));
  });

  it("only ever shortens a spike, so an irregular burst still fits its stated box", () => {
    for (const jitter of [0, 0.3, 0.7, 1]) {
      const burst = shape({ kind: "burst", jitter, seed: 3, width: 300, height: 180, tailLength: 0 });
      for (const [x, y] of bodyOutline(burst)) {
        expect(Math.abs(x), `jitter ${jitter}`).toBeLessThanOrEqual(150.001);
        expect(Math.abs(y), `jitter ${jitter}`).toBeLessThanOrEqual(90.001);
      }
    }
  });

  it("turns a shape's corners with its spin, without turning the box", () => {
    const upright = shape({ kind: "star", points: 5, spin: -90, width: 200, height: 200 });
    const outline = bodyOutline(upright);
    const highest = outline.reduce((best, point) => point[1] < best[1] ? point : best, outline[0]);
    // A point straight up, which is what everyone means by a star.
    expect(Math.abs(highest[0])).toBeLessThan(1);
    expect(highest[1]).toBeCloseTo(-100, 1);
  });

  it("gives the shapes that go behind the action several separate strokes", () => {
    for (const kind of DECORATION_KINDS) {
      expect(shapePaths(shape({ kind })).length, kind).toBeGreaterThan(3);
      expect(holdsText(kind), kind).toBe(false);
    }
  });

  it("leaves a hole in the middle of an impact and none in a sunburst", () => {
    const nearestToCentre = (kind: ShapeKind) => shapePaths(shape({ kind, width: 400, height: 400 }))
      .flatMap((path) => path.points)
      .reduce((best, [x, y]) => Math.min(best, Math.hypot(x, y)), Infinity);
    // The gap where the thing that hit is, which is the whole difference between the two.
    expect(nearestToCentre("impact")).toBeGreaterThan(nearestToCentre("sunburst"));
  });

  it("all name a real kind", () => {
    for (const preset of SHAPE_PRESETS) {
      expect(ALL_KINDS, preset.name).toContain(preset.shape.kind);
    }
  });

  it("all produce a drawable outline", () => {
    for (const preset of SHAPE_PRESETS) {
      const paths = shapePaths(shape(preset.shape));
      expect(paths.length, preset.name).toBeGreaterThan(0);
      const box = shapeBounds(shape(preset.shape));
      expect(box.width, preset.name).toBeGreaterThan(8);
      expect(box.height, preset.name).toBeGreaterThan(8);
    }
  });

  it("gives the caption no tail, because a caption is not spoken", () => {
    const caption = SHAPE_PRESETS.find((preset) => preset.name === "Caption")!;
    expect(hasTail(shape(caption.shape))).toBe(false);
  });

  it("groups every preset under a heading", () => {
    for (const preset of SHAPE_PRESETS) {
      expect(["Balloons", "Shapes", "Behind the action"], preset.name).toContain(preset.group);
    }
  });

  it("dashes only the whisper", () => {
    const dashed = SHAPE_PRESETS.filter((preset) => preset.shape.dashed);
    expect(dashed.map((preset) => preset.name)).toEqual(["Whisper"]);
  });
});
