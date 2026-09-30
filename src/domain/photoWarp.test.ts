import { describe, expect, it } from "vitest";
import { applyWarp, buildField, type WarpStroke } from "./photoWarp";
import { denoiseBuffer } from "./photoEdit";
import { NEUTRAL_TEXT, fontSpec, lineOrigin, textBoxSize, wrapLines, type TextBlock } from "./photoText";

/* -------------------------------------------------------------------- warp */

const stroke = (over: Partial<WarpStroke> = {}): WarpStroke => ({
  points: [[0.5, 0.5]], radius: 0.3, strength: 1, mode: "push", ...over,
});

/** A flat field with one bright pixel, so a displacement is visible as the pixel having moved. */
function dot(width: number, height: number, x: number, y: number): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let at = 3; at < data.length; at += 4) data[at] = 255;
  const index = (y * width + x) * 4;
  data[index] = data[index + 1] = data[index + 2] = 255;
  return data;
}

describe("buildField", () => {
  it("has nothing to say about an empty stroke list", () => {
    expect(buildField(16, 16, [])).toBeUndefined();
  });

  it("ignores a stroke with no strength, rather than building a field of zeroes", () => {
    expect(buildField(16, 16, [stroke({ strength: 0 })])).toBeUndefined();
  });

  it("leaves everything outside the brush untouched", () => {
    const field = buildField(64, 64, [stroke({
      points: [[0.5, 0.5], [0.6, 0.5]], radius: 0.1, mode: "push",
    })])!;
    // The brush reaches 6.4px from the middle; a corner is nowhere near it.
    expect(field.x[0]).toBe(0);
    expect(field.y[0]).toBe(0);
  });

  it("points the fetch backwards along a push, so the content travels with the brush", () => {
    const field = buildField(64, 64, [stroke({
      points: [[0.4, 0.5], [0.6, 0.5]], radius: 0.25, mode: "push", strength: 1,
    })])!;
    const middle = 32 * 64 + 32;
    // Backward mapping: moving content to the right means reading from the left.
    expect(field.x[middle]).toBeLessThan(0);
    expect(Math.abs(field.y[middle])).toBeLessThan(1e-6);
  });

  it("reads inward for bloat and outward for pinch", () => {
    const at = (mode: "bloat" | "pinch") => {
      const field = buildField(64, 64, [stroke({ mode, radius: 0.4, strength: 1 })])!;
      // A pixel to the right of the brush centre.
      return field.x[32 * 64 + 40];
    };
    expect(at("bloat")).toBeLessThan(0);
    expect(at("pinch")).toBeGreaterThan(0);
  });

  it("turns the fetch sideways for a twirl", () => {
    const field = buildField(64, 64, [stroke({ mode: "twirl", radius: 0.4, strength: 1 })])!;
    const right = 32 * 64 + 40;
    // Straight right of the centre, a rotation shows up almost entirely in y.
    expect(Math.abs(field.y[right])).toBeGreaterThan(Math.abs(field.x[right]));
  });

  it("falls off to nothing at the brush edge", () => {
    const field = buildField(128, 128, [stroke({
      points: [[0.5, 0.5], [0.7, 0.5]], radius: 0.2, mode: "push",
    })])!;
    // Measured ACROSS the stroke rather than along it: both ends of the path are inside the brush,
    // so the falloff only shows up as you move away from the line the brush travelled.
    const onTheLine = Math.abs(field.x[64 * 128 + 76]);
    const offToTheSide = Math.abs(field.x[84 * 128 + 76]);
    expect(onTheLine).toBeGreaterThan(offToTheSide);
    expect(offToTheSide).toBeGreaterThan(0);
    // Past the radius there is nothing at all.
    expect(field.x[100 * 128 + 76]).toBe(0);
  });

  it("holds one stroke under a brush radius, so a long drag cannot tear the picture", () => {
    const field = buildField(200, 200, [stroke({
      // A drag four times longer than the brush is wide. Every dab along it accumulates, and the
      // total is what has to stay bounded -- past one radius, neighbouring pixels fetch across each
      // other and the picture folds over itself.
      points: [[0.05, 0.5], [0.95, 0.5]], radius: 0.05, strength: 1, mode: "push",
    })])!;
    const radius = 0.05 * 200;
    let worst = 0;
    for (let index = 0; index < field.x.length; index++) {
      worst = Math.max(worst, Math.hypot(field.x[index], field.y[index]));
    }
    expect(worst).toBeGreaterThan(radius * 0.5);
    expect(worst).toBeLessThanOrEqual(radius * 0.9 + 1e-6);
  });

  it("reports the rectangle the strokes reached, and nothing outside it moved", () => {
    const field = buildField(100, 100, [stroke({
      points: [[0.5, 0.5]], radius: 0.1, mode: "bloat", strength: 1,
    })])!;
    expect(field.left).toBeGreaterThan(30);
    expect(field.right).toBeLessThan(70);
    for (let y = 0; y < 100; y++) {
      for (let x = 0; x < 100; x++) {
        if (y >= field.top && y <= field.bottom && x >= field.left && x <= field.right) continue;
        expect(field.x[y * 100 + x]).toBe(0);
        expect(field.y[y * 100 + x]).toBe(0);
      }
    }
  });
});

describe("applyWarp", () => {
  it("does nothing with no strokes", () => {
    const before = dot(32, 32, 16, 16);
    const after = new Uint8ClampedArray(before);
    applyWarp(after, 32, 32, []);
    expect([...after]).toEqual([...before]);
  });

  it("goes further on a second pass over the same place", () => {
    const width = 64, height = 64;
    const once = dot(width, height, 32, 32);
    const twice = dot(width, height, 32, 32);
    const pass = stroke({ points: [[0.5, 0.5], [0.62, 0.5]], radius: 0.25, strength: 1, mode: "push" });
    applyWarp(once, width, height, [pass]);
    applyWarp(twice, width, height, [pass, pass]);
    const furthest = (data: Uint8ClampedArray) => {
      let at = 32;
      for (let x = 33; x < 64; x++) if (data[(32 * width + x) * 4] > 8) at = x;
      return at;
    };
    // Each stroke warps the result of the last, which is the only reason a second pass does
    // anything at all -- accumulated into one field it would hit the cap and stop.
    expect(furthest(twice)).toBeGreaterThan(furthest(once));
  });

  it("moves a bright pixel in the direction of the push", () => {
    const width = 64, height = 64;
    const data = dot(width, height, 32, 32);
    applyWarp(data, width, height, [stroke({
      points: [[0.5, 0.5], [0.62, 0.5]], radius: 0.25, strength: 1, mode: "push",
    })]);
    const brightness = (x: number) => data[(32 * width + x) * 4];
    // It has left where it was and turned up to the right of it.
    expect(brightness(32)).toBeLessThan(255);
    let moved = 0;
    for (let x = 33; x < 48; x++) moved = Math.max(moved, brightness(x));
    expect(moved).toBeGreaterThan(0);
  });

  it("invents no colour, because every output pixel is read from the input", () => {
    const width = 48, height = 48;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let at = 0; at < data.length; at += 4) {
      data[at] = 40; data[at + 1] = 90; data[at + 2] = 200; data[at + 3] = 255;
    }
    applyWarp(data, width, height, [stroke({ mode: "twirl", radius: 0.4, strength: 1 })]);
    for (let at = 0; at < data.length; at += 4) {
      expect(data[at]).toBe(40);
      expect(data[at + 1]).toBe(90);
      expect(data[at + 2]).toBe(200);
    }
  });

  it("keeps alpha, so warping a cutout does not eat its matte", () => {
    const width = 32, height = 32;
    const data = dot(width, height, 16, 16);
    for (let at = 3; at < data.length; at += 4) data[at] = 128;
    applyWarp(data, width, height, [stroke({ mode: "bloat", radius: 0.4, strength: 1 })]);
    for (let at = 3; at < data.length; at += 4) expect(data[at]).toBe(128);
  });
});

/* ----------------------------------------------------------------- denoise */

/** A flat grey with alternating noise on it, which is what denoise has to remove. */
function noisy(width: number, height: number, swing: number): Uint8ClampedArray {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let index = 0, at = 0; index < width * height; index++, at += 4) {
    const wobble = (index % 2 === 0 ? swing : -swing);
    data[at] = data[at + 1] = data[at + 2] = 128 + wobble;
    data[at + 3] = 255;
  }
  return data;
}

const spread = (data: Uint8ClampedArray): number => {
  let total = 0, squared = 0, count = 0;
  for (let at = 0; at < data.length; at += 4) {
    total += data[at]; squared += data[at] * data[at]; count++;
  }
  const mean = total / count;
  return Math.sqrt(Math.max(0, squared / count - mean * mean));
};

describe("denoiseBuffer", () => {
  it("does nothing at zero", () => {
    const before = noisy(24, 24, 9);
    const after = new Uint8ClampedArray(before);
    denoiseBuffer(after, 24, 24, 0, 0, 0);
    expect([...after]).toEqual([...before]);
  });

  it("flattens fine luminance noise", () => {
    const data = noisy(32, 32, 9);
    const before = spread(data);
    denoiseBuffer(data, 32, 32, 80, 0, 0);
    expect(spread(data)).toBeLessThan(before * 0.5);
  });

  it("keeps more of the texture when asked to", () => {
    const bare = noisy(32, 32, 9);
    const kept = noisy(32, 32, 9);
    denoiseBuffer(bare, 32, 32, 80, 0, 0);
    denoiseBuffer(kept, 32, 32, 80, 100, 0);
    expect(spread(kept)).toBeGreaterThan(spread(bare));
  });

  it("leaves a real edge standing", () => {
    const width = 40, height = 40;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let y = 0, at = 0; y < height; y++) {
      for (let x = 0; x < width; x++, at += 4) {
        const value = x < width / 2 ? 30 : 225;
        data[at] = data[at + 1] = data[at + 2] = value;
        data[at + 3] = 255;
      }
    }
    denoiseBuffer(data, width, height, 100, 0, 0);
    // Well away from the boundary the two sides are still the two values they were.
    expect(data[(20 * width + 4) * 4]).toBeLessThan(60);
    expect(data[(20 * width + 35) * 4]).toBeGreaterThan(195);
  });

  it("round-trips a flat picture unchanged, so the colour conversion is honest", () => {
    const width = 16, height = 16;
    const data = new Uint8ClampedArray(width * height * 4);
    for (let at = 0; at < data.length; at += 4) {
      data[at] = 70; data[at + 1] = 140; data[at + 2] = 210; data[at + 3] = 255;
    }
    denoiseBuffer(data, width, height, 100, 0, 100);
    for (let at = 0; at < data.length; at += 4) {
      expect(Math.abs(data[at] - 70)).toBeLessThanOrEqual(1);
      expect(Math.abs(data[at + 1] - 140)).toBeLessThanOrEqual(1);
      expect(Math.abs(data[at + 2] - 210)).toBeLessThanOrEqual(1);
    }
  });

  it("never touches alpha", () => {
    const data = noisy(16, 16, 9);
    for (let at = 3; at < data.length; at += 4) data[at] = 77;
    denoiseBuffer(data, 16, 16, 100, 0, 100);
    for (let at = 3; at < data.length; at += 4) expect(data[at]).toBe(77);
  });
});

/* -------------------------------------------------------------------- text */

const block = (over: Partial<TextBlock> = {}): TextBlock => ({ ...NEUTRAL_TEXT, ...over });
/** Every character is ten units wide, so line breaking can be reasoned about exactly. */
const tenWide = (line: string) => line.length * 10;

describe("wrapLines", () => {
  it("breaks on explicit newlines whatever the width", () => {
    const lines = wrapLines(block({ content: "one\ntwo", boxWidth: 10000, padding: 0 }), tenWide);
    expect(lines).toEqual(["one", "two"]);
  });

  it("keeps a blank line rather than swallowing it", () => {
    const lines = wrapLines(block({ content: "a\n\nb", boxWidth: 1000, padding: 0 }), tenWide);
    expect(lines).toEqual(["a", "", "b"]);
  });

  it("wraps greedily to the inner width", () => {
    // 60 wide inside: "aaa bbb" is 70 and does not fit, "aaa" is 30 and does.
    const lines = wrapLines(block({ content: "aaa bbb ccc", boxWidth: 60, padding: 0 }), tenWide);
    expect(lines).toEqual(["aaa", "bbb", "ccc"]);
  });

  it("takes the padding off both sides before wrapping", () => {
    const wide = wrapLines(block({ content: "aa bb", boxWidth: 60, padding: 0 }), tenWide);
    const padded = wrapLines(block({ content: "aa bb", boxWidth: 60, padding: 15 }), tenWide);
    expect(wide).toEqual(["aa bb"]);
    expect(padded).toEqual(["aa", "bb"]);
  });

  it("leaves a word too wide for the box alone rather than cutting it up", () => {
    const lines = wrapLines(block({ content: "abcdefghij", boxWidth: 30, padding: 0 }), tenWide);
    expect(lines).toEqual(["abcdefghij"]);
  });

  it("does not turn runs of spaces into empty words", () => {
    const lines = wrapLines(block({ content: "aa    bb", boxWidth: 1000, padding: 0 }), tenWide);
    expect(lines).toEqual(["aa bb"]);
  });
});

describe("textBoxSize", () => {
  it("takes its width from the column, not from the longest line", () => {
    expect(textBoxSize(block({ boxWidth: 400 }), 1).width).toBe(400);
    expect(textBoxSize(block({ boxWidth: 400 }), 9).width).toBe(400);
  });

  it("grows by one line height per line, plus the padding on both sides", () => {
    const text = block({ fontSize: 40, lineHeight: 1.5, padding: 10 });
    expect(textBoxSize(text, 1).height).toBe(60 + 20);
    expect(textBoxSize(text, 3).height).toBe(180 + 20);
  });

  it("still has a size with no lines at all", () => {
    expect(textBoxSize(block({ padding: 0 }), 0).height).toBeGreaterThan(0);
  });
});

describe("fontSpec", () => {
  it("quotes the family and gives it a fallback of the right kind", () => {
    expect(fontSpec(block({ fontFamily: "Comic Sans MS", weight: 700, fontSize: 40 })))
      .toBe('700 40px "Comic Sans MS", cursive');
    expect(fontSpec(block({ fontFamily: "Consolas" }))).toContain("monospace");
    expect(fontSpec(block({ fontFamily: "Georgia" }))).toContain("serif");
    expect(fontSpec(block({ fontFamily: "Inter" }))).toContain("sans-serif");
  });

  it("does not call a sans-serif a serif because the word is inside it", () => {
    expect(fontSpec(block({ fontFamily: "Source Sans Pro" }))).toContain("sans-serif");
  });

  it("leaves an explicit family list alone", () => {
    expect(fontSpec(block({ fontFamily: "Inter, sans-serif", weight: 400, fontSize: 12 })))
      .toBe("400 12px Inter, sans-serif");
  });

  it("carries italic", () => {
    expect(fontSpec(block({ italic: true }))).toMatch(/^italic /);
  });
});

describe("lineOrigin", () => {
  it("starts at the padding on the left and ends at it on the right", () => {
    expect(lineOrigin(block({ align: "left", padding: 12 }), 300)).toEqual({ x: 12, align: "left" });
    expect(lineOrigin(block({ align: "right", padding: 12 }), 300)).toEqual({ x: 288, align: "right" });
  });

  it("centres on the whole bitmap, padding included, so a centred column stays centred", () => {
    expect(lineOrigin(block({ align: "center", padding: 40 }), 300))
      .toEqual({ x: 150, align: "center" });
  });
});
