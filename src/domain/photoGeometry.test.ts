import { describe, expect, it } from "vitest";
import {
  NEUTRAL_GEOMETRY, applyMatrix, displayToSource, forwardMatrix, framedSize, invert,
  isNeutralGeometry, outputSize, sourceToDisplay, type PhotoGeometry,
} from "./photoGeometry";

const geometry = (over: Partial<PhotoGeometry> = {}): PhotoGeometry => ({ ...NEUTRAL_GEOMETRY, ...over });

/** Every geometry worth checking the round trip against. */
const CASES: Array<[string, PhotoGeometry]> = [
  ["neutral", geometry()],
  ["rotate 90", geometry({ rotate: 90 })],
  ["rotate 180", geometry({ rotate: 180 })],
  ["rotate 270", geometry({ rotate: 270 })],
  ["flip h", geometry({ flipHorizontal: true })],
  ["flip v", geometry({ flipVertical: true })],
  ["both flips", geometry({ flipHorizontal: true, flipVertical: true })],
  ["straighten", geometry({ straighten: 7.5 })],
  ["straighten negative", geometry({ straighten: -12 })],
  ["scale", geometry({ scale: 1.4 })],
  ["crop", geometry({ crop: { x: 0.1, y: 0.2, width: 0.5, height: 0.6 } })],
  ["everything", geometry({
    rotate: 270, flipHorizontal: true, straighten: 4, scale: 1.2,
    crop: { x: 0.05, y: 0.15, width: 0.7, height: 0.6 },
  })],
];

describe("matrix", () => {
  it("inverts", () => {
    const matrix = forwardMatrix(800, 600, geometry({ rotate: 90, straighten: 5, scale: 1.3 }));
    const [x, y] = applyMatrix(matrix, 123, 456);
    const [back, backY] = applyMatrix(invert(matrix), x, y);
    expect(back).toBeCloseTo(123, 6);
    expect(backY).toBeCloseTo(456, 6);
  });

  it("survives a degenerate transform instead of producing NaN", () => {
    const broken = invert({ a: 0, b: 0, c: 0, d: 0, e: 0, f: 0 });
    expect(applyMatrix(broken, 5, 7)).toEqual([5, 7]);
  });
});

describe("framedSize", () => {
  it("swaps the axes on a quarter turn", () => {
    const framed = framedSize(800, 600, geometry({ rotate: 90 }));
    expect(framed.width).toBeCloseTo(600, 6);
    expect(framed.height).toBeCloseTo(800, 6);
  });

  it("grows when straightened, because the corners have to go somewhere", () => {
    const framed = framedSize(800, 600, geometry({ straighten: 10 }));
    expect(framed.width).toBeGreaterThan(800);
    expect(framed.height).toBeGreaterThan(600);
  });

  it("is unchanged by flips", () => {
    expect(framedSize(800, 600, geometry({ flipHorizontal: true, flipVertical: true })))
      .toEqual(framedSize(800, 600, geometry()));
  });
});

describe("outputSize", () => {
  it("is the source size when nothing is set", () => {
    expect(outputSize(800, 600, geometry())).toEqual({ width: 800, height: 600 });
  });

  it("takes its bite out of the framed size", () => {
    const size = outputSize(800, 600, geometry({ crop: { x: 0, y: 0, width: 0.5, height: 0.25 } }));
    expect(size).toEqual({ width: 400, height: 150 });
  });

  it("never returns a zero dimension", () => {
    const size = outputSize(800, 600, geometry({ crop: { x: 0, y: 0, width: 0.0001, height: 0.0001 } }));
    expect(size.width).toBeGreaterThanOrEqual(1);
    expect(size.height).toBeGreaterThanOrEqual(1);
  });
});

describe("display and source round trip", () => {
  it.each(CASES)("survives %s", (_name, value) => {
    for (const [u, v] of [[0.5, 0.5], [0.2, 0.7], [0.85, 0.1]]) {
      const [px, py] = sourceToDisplay(u, v, 800, 600, value);
      const [backU, backV] = displayToSource(px, py, 800, 600, value);
      expect(backU).toBeCloseTo(u, 6);
      expect(backV).toBeCloseTo(v, 6);
    }
  });

  it("puts the source centre at the display centre when there is no crop", () => {
    // Measured in pixels, not in normalised units: the output size is rounded to whole pixels, so
    // the centre of an odd-width frame legitimately sits half a pixel off exact 0.5.
    for (const [name, value] of CASES.filter(([label]) => label !== "crop" && label !== "everything")) {
      const size = outputSize(800, 600, value);
      const [px, py] = sourceToDisplay(0.5, 0.5, 800, 600, value);
      expect(Math.abs(px - 0.5) * size.width, name).toBeLessThan(0.5);
      expect(Math.abs(py - 0.5) * size.height, name).toBeLessThan(0.5);
    }
  });

  it("maps a quarter turn the way the canvas draws it", () => {
    // Canvas rotates clockwise on screen, so the source's top-left lands top-right.
    const [px, py] = sourceToDisplay(0, 0, 800, 600, geometry({ rotate: 90 }));
    expect(px).toBeCloseTo(1, 6);
    expect(py).toBeCloseTo(0, 6);
  });

  it("mirrors horizontally", () => {
    const [px] = sourceToDisplay(0.25, 0.5, 800, 600, geometry({ flipHorizontal: true }));
    expect(px).toBeCloseTo(0.75, 6);
  });

  it("puts the crop's top-left corner at the origin", () => {
    const crop = { x: 0.25, y: 0.5, width: 0.5, height: 0.5 };
    const [px, py] = sourceToDisplay(0.25, 0.5, 800, 600, geometry({ crop }));
    expect(px).toBeCloseTo(0, 6);
    expect(py).toBeCloseTo(0, 6);
  });
});

describe("isNeutralGeometry", () => {
  it("is true for the defaults", () => expect(isNeutralGeometry(geometry())).toBe(true));

  it("notices each control", () => {
    const changes: Array<Partial<PhotoGeometry>> = [
      { rotate: 90 }, { flipHorizontal: true }, { flipVertical: true },
      { straighten: 1 }, { scale: 1.01 }, { crop: { x: 0, y: 0, width: 1, height: 1 } },
    ];
    for (const change of changes) expect(isNeutralGeometry(geometry(change))).toBe(false);
  });
});
