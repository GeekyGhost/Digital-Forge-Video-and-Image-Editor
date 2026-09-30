import { describe, expect, it } from "vitest";
import { DEFAULT_MATRIX_RAIN } from "./photoDocument";
import { hash4, mix32, rainLayout, renderMatrixRain, type RainAtlas } from "./matrixRain";

/** A synthetic atlas: every glyph a solid block with a hole, so coverage is known without a canvas. */
function atlas(cell: number, count = 8): RainAtlas {
  const width = cell * count;
  const coverage = new Uint8Array(width * cell);
  for (let y = 0; y < cell; y++) for (let x = 0; x < width; x++) coverage[y * width + x] = (x % cell === 0 || y === 0) ? 0 : 200;
  return { cell, count, width, coverage };
}

const W = 160, H = 96;
const gen = { ...DEFAULT_MATRIX_RAIN, cellSize: 8, loopFrames: 48, trail: 5, density: 1 };
const render = (frame: number, over = {}) => {
  const out = new Uint8ClampedArray(W * H * 4);
  const g = { ...gen, ...over };
  renderMatrixRain(out, W, H, g, frame, 1, atlas(rainLayout(g, W, H, 1).cell));
  return out;
};

describe("matrix rain", () => {
  it("hashes like the GLSL (32-bit wrap)", () => {
    expect(mix32(0)).toBe(0);
    // Reference values from an independent uint32 implementation (Python, masked to 32 bits).
    expect(mix32(1)).toBe(1753845952);
    expect(hash4(1, 2, 3, 4)).toBe(2823244345);
    expect(hash4(1, 2, 3, 4)).not.toBe(hash4(1, 2, 3, 5));
    expect(hash4(0xffffffff, 7, 0, 0)).toBeGreaterThanOrEqual(0);
  });

  it("is deterministic and loops exactly", () => {
    expect(Array.from(render(5))).toEqual(Array.from(render(5)));
    expect(Array.from(render(0))).toEqual(Array.from(render(48)));
    expect(Array.from(render(7))).toEqual(Array.from(render(7 + 48 * 3)));
    expect(Array.from(render(1))).not.toEqual(Array.from(render(2)));
  });

  it("draws something, heads in the head colour, and nothing with density 0", () => {
    const out = render(10);
    let lit = 0, heads = 0;
    for (let at = 0; at < out.length; at += 4) {
      if (out[at + 3]) lit++;
      if (out[at + 3] && out[at] === 0xe8) heads++;
    }
    expect(lit).toBeGreaterThan(100);
    expect(heads).toBeGreaterThan(0);
    expect(render(10, { density: 0 }).every((value) => value === 0)).toBe(true);
  });

  it("changes seed, changes rain", () => {
    expect(Array.from(render(3, { seed: 1 }))).not.toEqual(Array.from(render(3, { seed: 2 })));
  });
});
