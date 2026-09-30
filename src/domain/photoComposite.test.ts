import { describe, expect, it } from "vitest";
import { blendChannel, compositeOver } from "./photoBlend";
import { applyRetouch, type RetouchStroke } from "./photoRetouch";
import {
  NEUTRAL_EDGE, applyLightWrap, buildContactShadow, defringeLayer, erode, isNeutralEdge, matchColour,
} from "./photoEdge";

/** A solid RGBA field. */
const field = (width: number, height: number, r: number, g: number, b: number, a = 255) => {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let at = 0; at < data.length; at += 4) {
    data[at] = r; data[at + 1] = g; data[at + 2] = b; data[at + 3] = a;
  }
  return data;
};

const pixelAt = (data: Uint8ClampedArray, width: number, x: number, y: number) => {
  const at = (y * width + x) * 4;
  return [data[at], data[at + 1], data[at + 2], data[at + 3]];
};

describe("blendChannel", () => {
  it("matches the spec on the separable modes", () => {
    expect(blendChannel("normal", 0.2, 0.8)).toBeCloseTo(0.8, 6);
    expect(blendChannel("multiply", 0.5, 0.5)).toBeCloseTo(0.25, 6);
    expect(blendChannel("screen", 0.5, 0.5)).toBeCloseTo(0.75, 6);
    expect(blendChannel("darken", 0.3, 0.7)).toBeCloseTo(0.3, 6);
    expect(blendChannel("lighten", 0.3, 0.7)).toBeCloseTo(0.7, 6);
    expect(blendChannel("difference", 0.3, 0.7)).toBeCloseTo(0.4, 6);
    expect(blendChannel("exclusion", 0.5, 0.5)).toBeCloseTo(0.5, 6);
  });

  it("defines overlay as hard light with the operands swapped", () => {
    for (const backdrop of [0, 0.2, 0.5, 0.8, 1]) {
      for (const source of [0, 0.3, 0.6, 1]) {
        expect(blendChannel("overlay", backdrop, source))
          .toBeCloseTo(blendChannel("hardLight", source, backdrop), 6);
      }
    }
  });

  it("does not divide by zero in dodge and burn", () => {
    expect(blendChannel("colorDodge", 0, 0.9)).toBe(0);
    expect(blendChannel("colorDodge", 0.5, 1)).toBe(1);
    expect(blendChannel("colorBurn", 1, 0.1)).toBe(1);
    expect(blendChannel("colorBurn", 0.5, 0)).toBe(0);
    for (const mode of ["colorDodge", "colorBurn"] as const) {
      for (let step = 0; step <= 10; step++) {
        expect(Number.isFinite(blendChannel(mode, step / 10, 1 - step / 10))).toBe(true);
      }
    }
  });

  it("leaves a neutral soft light alone", () => {
    expect(blendChannel("softLight", 0.4, 0.5)).toBeCloseTo(0.4, 6);
  });
});

describe("compositeOver", () => {
  it("replaces the backdrop at full alpha in normal mode", () => {
    const base = field(2, 1, 10, 20, 30);
    compositeOver(base, field(2, 1, 200, 100, 50), "normal", 1);
    expect(pixelAt(base, 2, 0, 0)).toEqual([200, 100, 50, 255]);
  });

  it("honours opacity", () => {
    const base = field(1, 1, 0, 0, 0);
    compositeOver(base, field(1, 1, 200, 200, 200), "normal", 0.5);
    expect(base[0]).toBeGreaterThan(95);
    expect(base[0]).toBeLessThan(105);
  });

  it("honours a mask, leaving uncovered pixels alone", () => {
    const base = field(2, 1, 10, 10, 10);
    compositeOver(base, field(2, 1, 250, 250, 250), "normal", 1, new Uint8Array([255, 0]));
    expect(pixelAt(base, 2, 0, 0)[0]).toBe(250);
    expect(pixelAt(base, 2, 1, 0)[0]).toBe(10);
  });

  it("skips a fully transparent layer", () => {
    const base = field(1, 1, 40, 50, 60);
    compositeOver(base, field(1, 1, 200, 200, 200, 0), "multiply", 1);
    expect(pixelAt(base, 1, 0, 0)).toEqual([40, 50, 60, 255]);
  });

  it("multiplies against an opaque backdrop", () => {
    const base = field(1, 1, 128, 128, 128);
    compositeOver(base, field(1, 1, 128, 128, 128), "multiply", 1);
    expect(base[0]).toBeGreaterThan(58);
    expect(base[0]).toBeLessThan(70);
  });

  it("shows the layer rather than black when multiplied over empty canvas", () => {
    // The blend has to fade out with the backdrop's alpha, or a Multiply layer on a fresh document
    // renders black and looks broken.
    const base = field(1, 1, 0, 0, 0, 0);
    compositeOver(base, field(1, 1, 200, 120, 60), "multiply", 1);
    expect(pixelAt(base, 1, 0, 0)).toEqual([200, 120, 60, 255]);
  });
});

describe("applyRetouch", () => {
  const stroke = (over: Partial<RetouchStroke> = {}): RetouchStroke => ({
    points: [[0.75, 0.5]], radius: 0.1, hardness: 1, offsetX: -0.5, offsetY: 0, heal: false, ...over,
  });

  /** Left half dark, right half bright, so a clone across the middle is obvious. */
  const split = (width = 40, height = 20) => {
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const at = (y * width + x) * 4;
        const value = x < width / 2 ? 40 : 220;
        data[at] = data[at + 1] = data[at + 2] = value;
        data[at + 3] = 255;
      }
    }
    return data;
  };

  it("copies from the sampled offset", () => {
    const data = split();
    applyRetouch(data, 40, 20, [stroke()]);
    // Painted at x=30 sampling from x=10, which is the dark half.
    expect(pixelAt(data, 40, 30, 10)[0]).toBeLessThan(80);
  });

  it("leaves the picture alone when there is no offset and no healing", () => {
    const data = split();
    const before = Array.from(data);
    applyRetouch(data, 40, 20, [stroke({ offsetX: 0, offsetY: 0 })]);
    expect(Array.from(data)).toEqual(before);
  });

  it("matches the surrounding brightness when healing", () => {
    const data = split();
    applyRetouch(data, 40, 20, [stroke({ heal: true })]);
    // Sampled from the dark half but healed into the bright half, so it must stay bright.
    expect(pixelAt(data, 40, 30, 10)[0]).toBeGreaterThan(150);
  });

  it("never touches alpha, so a cutout keeps its matte", () => {
    const data = split();
    for (let at = 3; at < data.length; at += 4) data[at] = 128;
    applyRetouch(data, 40, 20, [stroke()]);
    for (let at = 3; at < data.length; at += 4) expect(data[at]).toBe(128);
  });

  it("paints along the whole path, not just at the samples", () => {
    const data = split();
    applyRetouch(data, 40, 20, [stroke({ points: [[0.6, 0.5], [0.9, 0.5]], radius: 0.05 })]);
    expect(pixelAt(data, 40, 30, 10)[0]).toBeLessThan(120);
  });

  it("does nothing with no strokes", () => {
    const data = split();
    const before = Array.from(data);
    applyRetouch(data, 40, 20, []);
    expect(Array.from(data)).toEqual(before);
  });
});

describe("erode", () => {
  it("shrinks a solid block by the radius", () => {
    const alpha = new Uint8Array(20 * 20);
    for (let y = 5; y < 15; y++) for (let x = 5; x < 15; x++) alpha[y * 20 + x] = 255;
    erode(alpha, 20, 20, 2);
    expect(alpha[10 * 20 + 10]).toBe(255);
    expect(alpha[5 * 20 + 5]).toBe(0);
  });

  it("is a no-op below one pixel", () => {
    const alpha = new Uint8Array([255, 255, 0, 255]);
    erode(alpha, 2, 2, 0);
    expect(Array.from(alpha)).toEqual([255, 255, 0, 255]);
  });
});

describe("defringeLayer", () => {
  it("pulls a contaminated rim toward the interior colour", () => {
    // A red square with one ring of green rim, which is what a cutout off grass looks like.
    const size = 32;
    const data = new Uint8ClampedArray(size * size * 4);
    for (let y = 8; y < 24; y++) {
      for (let x = 8; x < 24; x++) {
        const rim = x === 8 || x === 23 || y === 8 || y === 23;
        const at = (y * size + x) * 4;
        data[at] = rim ? 40 : 220;
        data[at + 1] = rim ? 200 : 30;
        data[at + 2] = 30;
        data[at + 3] = 255;
      }
    }
    const greenBefore = pixelAt(data, size, 8, 16)[1];
    defringeLayer(data, size, size, 1);
    const greenAfter = pixelAt(data, size, 8, 16)[1];
    // Either the rim was eaten or it was repainted; both count, neither leaves it green.
    const alphaAfter = pixelAt(data, size, 8, 16)[3];
    expect(alphaAfter === 0 || greenAfter < greenBefore).toBe(true);
  });

  it("is a no-op below one pixel", () => {
    const data = field(8, 8, 10, 20, 30);
    const before = Array.from(data);
    defringeLayer(data, 8, 8, 0);
    expect(Array.from(data)).toEqual(before);
  });
});

describe("applyLightWrap", () => {
  it("brightens the inside of the edge and leaves the middle alone", () => {
    const size = 40;
    const data = new Uint8ClampedArray(size * size * 4);
    for (let y = 10; y < 30; y++) {
      for (let x = 10; x < 30; x++) {
        const at = (y * size + x) * 4;
        data[at] = data[at + 1] = data[at + 2] = 30;
        data[at + 3] = 255;
      }
    }
    const centreBefore = pixelAt(data, size, 20, 20)[0];
    const rimBefore = pixelAt(data, size, 11, 20)[0];
    applyLightWrap(data, field(size, size, 255, 255, 255), size, size, 100, 4);
    expect(pixelAt(data, size, 11, 20)[0]).toBeGreaterThan(rimBefore);
    expect(pixelAt(data, size, 20, 20)[0]).toBe(centreBefore);
  });

  it("does nothing at zero", () => {
    const data = field(8, 8, 10, 10, 10);
    const before = Array.from(data);
    applyLightWrap(data, field(8, 8, 255, 255, 255), 8, 8, 0, 4);
    expect(Array.from(data)).toEqual(before);
  });
});

describe("buildContactShadow", () => {
  it("declines when the shadow is off", () => {
    expect(buildContactShadow(field(8, 8, 0, 0, 0), 8, 8, NEUTRAL_EDGE)).toBeUndefined();
  });

  it("puts opacity under the subject", () => {
    const size = 40;
    const subject = new Uint8ClampedArray(size * size * 4);
    for (let y = 8; y < 28; y++) {
      for (let x = 14; x < 26; x++) subject[(y * size + x) * 4 + 3] = 255;
    }
    const shadow = buildContactShadow(subject, size, size, { ...NEUTRAL_EDGE, contactShadow: 80 });
    expect(shadow).toBeDefined();
    let total = 0;
    for (let at = 3; at < shadow!.length; at += 4) total += shadow![at];
    expect(total).toBeGreaterThan(0);
  });
});

describe("isNeutralEdge", () => {
  it("is true for the defaults and for nothing at all", () => {
    expect(isNeutralEdge(NEUTRAL_EDGE)).toBe(true);
    expect(isNeutralEdge(undefined)).toBe(true);
  });

  it("notices each of the three", () => {
    expect(isNeutralEdge({ ...NEUTRAL_EDGE, defringe: 1 })).toBe(false);
    expect(isNeutralEdge({ ...NEUTRAL_EDGE, lightWrap: 1 })).toBe(false);
    expect(isNeutralEdge({ ...NEUTRAL_EDGE, contactShadow: 1 })).toBe(false);
  });
});

describe("matchColour", () => {
  it("raises exposure when the plate is brighter than the cutout", () => {
    const solved = matchColour(field(16, 16, 60, 60, 60), field(16, 16, 180, 180, 180));
    expect(solved).toBeDefined();
    expect(solved!.exposure).toBeGreaterThan(0);
  });

  it("lowers it when the plate is darker", () => {
    const solved = matchColour(field(16, 16, 200, 200, 200), field(16, 16, 60, 60, 60));
    expect(solved!.exposure).toBeLessThan(0);
  });

  it("warms a cutout that is bluer than its plate", () => {
    const solved = matchColour(field(16, 16, 80, 100, 180), field(16, 16, 180, 100, 80));
    expect(solved!.temperature).toBeGreaterThan(0);
  });

  it("declines when the cutout is empty", () => {
    expect(matchColour(field(16, 16, 0, 0, 0, 0), field(16, 16, 128, 128, 128))).toBeUndefined();
  });
});
