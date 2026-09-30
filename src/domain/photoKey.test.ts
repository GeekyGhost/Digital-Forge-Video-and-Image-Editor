import { describe, expect, it } from "vitest";
import { compositeOver } from "./photoBlend";
import { applyLumaKey, applyUnmultiply, lumaKeyAmount } from "./photoKey";

describe("applyUnmultiply", () => {
  it("makes black fully transparent and keeps the brightest channel at full", () => {
    const pixels = new Uint8ClampedArray([0, 0, 0, 255, 255, 0, 0, 255, 128, 64, 0, 255, 1, 0, 0, 255]);
    applyUnmultiply(pixels);
    expect(Array.from(pixels.subarray(0, 4))).toEqual([0, 0, 0, 0]);
    expect(Array.from(pixels.subarray(4, 8))).toEqual([255, 0, 0, 255]);
    // 64 * 255 / 128 = 127.5, which rounds half to even as a Uint8ClampedArray does.
    expect(Array.from(pixels.subarray(8, 12))).toEqual([255, 128, 0, 128]);
    expect(Array.from(pixels.subarray(12, 16))).toEqual([255, 0, 0, 1]);
  });

  it("composited normally, gives what Screen gives in the brightest channel", () => {
    const backdrop = [40, 90, 160];
    for (const colour of [[200, 120, 30], [90, 90, 90], [10, 250, 40]]) {
      const keyed = new Uint8ClampedArray([...colour, 255]);
      applyUnmultiply(keyed);
      const normal = new Uint8ClampedArray([...backdrop, 255]);
      compositeOver(normal, keyed, "normal", 1);
      const screened = new Uint8ClampedArray([...backdrop, 255]);
      compositeOver(screened, new Uint8ClampedArray([...colour, 255]), "screen", 1);
      const brightest = colour.indexOf(Math.max(...colour));
      expect(Math.abs(normal[brightest] - screened[brightest])).toBeLessThanOrEqual(1);
    }
  });

  it("scales existing alpha", () => {
    const pixels = new Uint8ClampedArray([255, 255, 255, 100, 51, 51, 51, 100]);
    applyUnmultiply(pixels);
    expect(pixels[3]).toBe(100);
    expect(pixels[7]).toBe(20);
  });
});

describe("applyLumaKey", () => {
  const key = { low: 20, high: 80, softness: 0, invert: false };

  it("keeps the range, hard", () => {
    expect(lumaKeyAmount(0.1 * 255, key)).toBe(0);
    expect(lumaKeyAmount(0.5 * 255, key)).toBe(1);
    expect(lumaKeyAmount(0.9 * 255, key)).toBe(0);
    expect(lumaKeyAmount(0.5 * 255, { ...key, invert: true })).toBe(0);
  });

  it("ramps over the softness outside the range", () => {
    const soft = { ...key, softness: 10 };
    expect(lumaKeyAmount(0.15 * 255, soft)).toBeCloseTo(0.5, 6);
    expect(lumaKeyAmount(0.85 * 255, soft)).toBeCloseTo(0.5, 6);
    expect(lumaKeyAmount(0.05 * 255, soft)).toBe(0);
  });

  it("multiplies alpha and leaves colour alone", () => {
    const pixels = new Uint8ClampedArray([0, 0, 0, 255, 128, 128, 128, 200]);
    applyLumaKey(pixels, { low: 10, high: 100, softness: 0, invert: false });
    expect(Array.from(pixels)).toEqual([0, 0, 0, 0, 128, 128, 128, 200]);
  });
});
