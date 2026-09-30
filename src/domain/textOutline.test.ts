import { describe, expect, it } from "vitest";
import { cubicAt, segmentsOf } from "./photoPath";
import { NEUTRAL_TEXT } from "./photoText";
import { fontRequest, glyphSubpaths, textPath, type FontOutlines } from "./textOutline";

describe("text to outlines (VFX Phase 6)", () => {
  it("places font units on the baseline, y flipped, quadratics as exact cubics", () => {
    // A 100-unit square with a quadratic bulge on top, at scale 0.5, x 10, baseline 200.
    const [sub] = glyphSubpaths([["M", [0, 0]], ["L", [100, 0]], ["Q", [50, 100, 0, 100]], ["Z", []]], 0.5, 10, 200);
    expect(sub.closed).toBe(true);
    expect(sub.nodes.map((node) => [node.x, node.y])).toEqual([[10, 200], [60, 200], [10, 150]]);
    // The quadratic from (60,200) via (35,150) to (10,150): its midpoint is 1/4 p0 + 1/2 q + 1/4 p2.
    const quad = segmentsOf(sub)[1];
    const mid = cubicAt(quad, 0.5);
    expect(mid[0]).toBeCloseTo(0.25 * 60 + 0.5 * 35 + 0.25 * 10, 9);
    expect(mid[1]).toBeCloseTo(0.25 * 200 + 0.5 * 150 + 0.25 * 150, 9);
  });

  it("folds a closing point that repeats the start, and keeps separate contours", () => {
    const subs = glyphSubpaths([["M", [0, 0]], ["L", [10, 0]], ["L", [10, 10]], ["L", [0, 0]], ["Z", []], ["M", [2, 2]], ["L", [4, 2]], ["L", [4, 4]], ["Z", []]], 1, 0, 0);
    expect(subs.map((sub) => sub.nodes.length)).toEqual([3, 3]);
  });

  it("builds the text's path around its bitmap centre with its paints", () => {
    const font: FontOutlines = {
      face: { path: "x.ttf", index: 0, family: "X", legacyFamily: "X", weight: 400, italic: false },
      unitsPerEm: 1000, ascender: 800, descender: -200, substituted: false, synthetic: false,
      glyphs: [{ ch: "I", advance: 300, found: true, commands: [["M", [0, 0]], ["L", [100, 0]], ["L", [100, 700]], ["L", [0, 700]], ["Z", []]] },
        { ch: " ", advance: 250, found: true, commands: [] }, { ch: "é", advance: 0, found: false, commands: [] }],
    };
    const text = { ...NEUTRAL_TEXT, strokeWidth: 3 };
    const { path, missing } = textPath(text, [{ ch: "I", x: 50, baseline: 80 }, { ch: " ", x: 80, baseline: 80 }, { ch: "é", x: 100, baseline: 80 }], { width: 200, height: 100 }, 100, font);
    expect(missing).toEqual(["é"]);
    expect(path.subpaths).toHaveLength(1);
    expect(path.subpaths[0].nodes[0]).toEqual({ x: -50, y: 30 });
    expect(path.subpaths[0].nodes[2]).toEqual({ x: -40, y: -40 });
    expect(path.strokeWidth).toBe(6);
    expect(path.paintOrder).toBe("stroke");
    expect(path.fill).toEqual(NEUTRAL_TEXT.color);
  });

  it("asks for the first real family and its generic", () => {
    const generic = () => "sans-serif";
    expect(fontRequest("Inter", generic)).toEqual({ family: "Inter", generic: "sans-serif" });
    expect(fontRequest('"Times New Roman", Georgia, serif', generic)).toEqual({ family: "Times New Roman", generic: "serif" });
  });
});
