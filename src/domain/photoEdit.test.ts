import { describe, expect, it } from "vitest";
import {
  NEUTRAL_ADJUSTMENTS, applyAdjustments, blurBuffer, buildCurveLut,
  hasGeometry, isNeutralGrade, photoFilter, type PhotoAdjustments,
} from "./photoEdit";

const grade = (over: Partial<PhotoAdjustments> = {}): PhotoAdjustments => ({
  ...NEUTRAL_ADJUSTMENTS, curve: NEUTRAL_ADJUSTMENTS.curve.map((point) => ({ ...point })), ...over,
});

/** A four-pixel strip: black, mid grey, a warm colour, white. */
const strip = () => new Uint8ClampedArray([
  0, 0, 0, 255, 128, 128, 128, 255, 200, 90, 40, 255, 255, 255, 255, 255,
]);

describe("isNeutralGrade", () => {
  it("is true for the defaults", () => {
    expect(isNeutralGrade(grade())).toBe(true);
  });

  it("notices every control that changes a pixel", () => {
    const controls: Partial<PhotoAdjustments>[] = [
      { exposure: 0.1 }, { contrast: 1 }, { highlights: -1 }, { shadows: 1 },
      { temperature: 1 }, { tint: -1 }, { saturation: 101 }, { vibrance: 1 },
      { sharpen: 1 }, { blur: 1 }, { grain: 1 }, { filter: "mono" },
      { curve: [{ x: 0.25, y: 0.3 }, { x: 0.5, y: 0.5 }, { x: 0.75, y: 0.75 }] },
    ];
    for (const control of controls) expect(isNeutralGrade(grade(control))).toBe(false);
  });

  it("ignores a filter that is mixed all the way out", () => {
    expect(isNeutralGrade(grade({ filter: "mono", filterAmount: 0 }))).toBe(true);
  });

  it("does not count geometry, which moves pixels rather than changing them", () => {
    expect(isNeutralGrade(grade({ rotate: 90, flipHorizontal: true }))).toBe(true);
    expect(hasGeometry(grade({ rotate: 90 }))).toBe(true);
    expect(hasGeometry(grade())).toBe(false);
  });
});

describe("applyAdjustments", () => {
  it("leaves the picture bit-identical when nothing is set", () => {
    const data = strip();
    applyAdjustments(data, 4, 1, grade());
    expect(Array.from(data)).toEqual(Array.from(strip()));
  });

  it("round-trips a neutral grade to within a quantisation step when one is forced", () => {
    // filterAmount 0 on a real filter still runs the whole pipeline, so this measures the decode /
    // encode error rather than short-circuiting past it.
    const data = strip();
    applyAdjustments(data, 4, 1, grade({ filter: "aurora", filterAmount: 0, exposure: 0 }));
    const original = strip();
    for (let index = 0; index < data.length; index++) {
      expect(Math.abs(data[index] - original[index])).toBeLessThanOrEqual(1);
    }
  });

  it("brightens with positive exposure and darkens with negative", () => {
    const up = strip();
    const down = strip();
    applyAdjustments(up, 4, 1, grade({ exposure: 1 }));
    applyAdjustments(down, 4, 1, grade({ exposure: -1 }));
    // Index 4 is the mid-grey pixel's red channel; black and white have nowhere to go.
    expect(up[4]).toBeGreaterThan(128);
    expect(down[4]).toBeLessThan(128);
  });

  it("preserves alpha", () => {
    const data = new Uint8ClampedArray([200, 90, 40, 0, 200, 90, 40, 128]);
    applyAdjustments(data, 2, 1, grade({ exposure: 1, saturation: 40 }));
    expect(data[3]).toBe(0);
    expect(data[7]).toBe(128);
  });

  it("collapses colour to grey at zero saturation", () => {
    const data = strip();
    applyAdjustments(data, 4, 1, grade({ saturation: 0 }));
    expect(Math.abs(data[8] - data[9])).toBeLessThanOrEqual(1);
    expect(Math.abs(data[9] - data[10])).toBeLessThanOrEqual(1);
  });

  it("desaturates through the mono filter without touching the saturation slider", () => {
    const data = strip();
    applyAdjustments(data, 4, 1, grade({ filter: "mono" }));
    expect(Math.abs(data[8] - data[10])).toBeLessThanOrEqual(2);
  });
});

describe("buildCurveLut", () => {
  it("never goes backwards, whatever the points say", () => {
    const table = buildCurveLut([{ x: 0.25, y: 0.9 }, { x: 0.5, y: 0.1 }, { x: 0.75, y: 0.8 }]);
    for (let index = 1; index < table.length; index++) {
      expect(table[index]).toBeGreaterThanOrEqual(table[index - 1]);
    }
  });

  it("pins the ends", () => {
    const table = buildCurveLut(NEUTRAL_ADJUSTMENTS.curve);
    expect(table[0]).toBeCloseTo(0, 3);
    expect(table[255]).toBeCloseTo(1, 3);
  });

  it("is close to a straight line when the points are on the diagonal", () => {
    const table = buildCurveLut(NEUTRAL_ADJUSTMENTS.curve);
    for (let index = 0; index < 256; index++) {
      // Tight on purpose: the end-segment tangents were once halved, which bowed this by ~0.019.
      expect(Math.abs(table[index] - index / 255)).toBeLessThan(0.002);
    }
  });
});

describe("blurBuffer", () => {
  it("leaves a flat picture flat, so the edge clamping is not darkening the border", () => {
    const data = new Uint8ClampedArray(8 * 8 * 4).fill(160);
    for (let index = 3; index < data.length; index += 4) data[index] = 255;
    blurBuffer(data, 8, 8, 3);
    for (let index = 0; index < data.length; index += 4) {
      expect(Math.abs(data[index] - 160)).toBeLessThanOrEqual(1);
    }
  });

  it("keeps the picture the right way round", () => {
    // A single bright column must stay a column: a transpose bug turns it into a row.
    const data = new Uint8ClampedArray(8 * 4 * 4);
    for (let y = 0; y < 4; y++) {
      const index = (y * 8 + 2) * 4;
      data[index] = data[index + 1] = data[index + 2] = 255;
    }
    blurBuffer(data, 8, 4, 1);
    const columnTotal = (x: number) => {
      let total = 0;
      for (let y = 0; y < 4; y++) total += data[(y * 8 + x) * 4];
      return total;
    };
    expect(columnTotal(2)).toBeGreaterThan(columnTotal(6));
  });
});

describe("photoFilter", () => {
  it("falls back to none for an id that is not in the list", () => {
    expect(photoFilter("does-not-exist").id).toBe("none");
  });
});
