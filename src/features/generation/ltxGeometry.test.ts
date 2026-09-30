import { describe, expect, it } from "vitest";
import {
  LTX_AXIS_MULTIPLE,
  LTX_FRAMES_PER_LATENT,
  LTX_MAX_FRAMES,
  LTX_MAX_PIXELS,
  LTX_MIN_FRAMES,
  ltxGeometryError,
  ltxLatentFrames,
  ltxPixelFrames,
  snapLtxAxis,
  snapLtxCanvas,
  snapLtxFrames,
  ltxRetakeProblems,
  ltxFrameRange,
  LTX_MAX_PIXEL_FRAMES,
} from "./ltxGeometry";
import { fitLtxDimensions } from "./ltx";

/**
 * The mirror half of `ai-runtime/tests/test_ltx_geometry.py`. Same cases, same expected values --
 * the point of both files is that the panel and the worker cannot drift apart, so anything
 * asserted on one side is asserted on the other.
 */
describe("LTX geometry", () => {
  it("uses the one-stage axis multiple, not the two-stage one", () => {
    expect(LTX_AXIS_MULTIPLE).toBe(32);
    // 832x480 was retakeable but not generatable, purely because generation rounded to 64.
    expect(snapLtxCanvas(832, 480)).toEqual({ width: 832, height: 480 });
    expect(fitLtxDimensions(832, 480)).toEqual({ width: 832, height: 480, value: "832x480" });
  });

  it("rounds axes up and stays on the grid", () => {
    for (let value = 1; value <= 2000; value += 1) {
      const snapped = snapLtxAxis(value);
      expect(snapped % LTX_AXIS_MULTIPLE).toBe(0);
      expect(snapped).toBeGreaterThanOrEqual(value);
      expect(snapped - value).toBeLessThan(LTX_AXIS_MULTIPLE);
    }
  });

  it("produces a canvas that is always legal, not usually legal", () => {
    const cases: Array<[number, number]> = [
      [1920, 1080], [1080, 1920], [3840, 2160], [768, 512], [832, 480], [1, 1],
      [2000, 33], [33, 2000], [767, 511], [769, 513],
    ];
    for (const [width, height] of cases) {
      const fitted = snapLtxCanvas(width, height);
      expect(ltxGeometryError(fitted.width, fitted.height, 121)).toBeUndefined();
    }
  });

  it("agrees with the Python module on the shared cases", () => {
    // These values are asserted verbatim in test_ltx_geometry.py.
    expect(snapLtxCanvas(1920, 1080)).toEqual({ width: 1024, height: 576 });
    expect(snapLtxCanvas(768, 512)).toEqual({ width: 768, height: 512 });
    expect(snapLtxFrames(30)).toBe(33);
    expect(snapLtxFrames(30, "down")).toBe(25);
    expect(snapLtxFrames(30, "up")).toBe(33);
    expect(snapLtxFrames(28)).toBe(25);
    expect(ltxLatentFrames(49)).toBe(7);
    expect(ltxLatentFrames(121)).toBe(16);
    expect(ltxPixelFrames(768, 512, 49)).toBe(2_752_512);
  });

  it("snaps frame counts rather than rejecting them", () => {
    for (let requested = 1; requested <= 400; requested += 1) {
      const snapped = snapLtxFrames(requested);
      expect((snapped - 1) % LTX_FRAMES_PER_LATENT).toBe(0);
      expect(snapped).toBeGreaterThanOrEqual(LTX_MIN_FRAMES);
      if (requested >= LTX_MIN_FRAMES) {
        expect(Math.abs(snapped - requested)).toBeLessThanOrEqual(LTX_FRAMES_PER_LATENT / 2);
      }
    }
    for (const legal of [9, 17, 49, 121, 193]) {
      expect(snapLtxFrames(legal)).toBe(legal);
      expect(snapLtxFrames(legal, "up")).toBe(legal);
      expect(snapLtxFrames(legal, "down")).toBe(legal);
    }
  });

  it("holds the floor below one latent frame", () => {
    for (let requested = 1; requested <= LTX_MIN_FRAMES; requested += 1) {
      expect(snapLtxFrames(requested)).toBe(LTX_MIN_FRAMES);
    }
  });

  it("gives generation a frame cap at all", () => {
    // `--num-frames 481` was accepted and OOMed after loading 27 GB. It is on the grid, so only
    // the cap catches it.
    expect((481 - 1) % LTX_FRAMES_PER_LATENT).toBe(0);
    expect(LTX_MAX_FRAMES).toBe(193);
    expect(ltxGeometryError(768, 512, 481)).toMatch(/over the 193/);
    expect(ltxGeometryError(768, 512, 193)).toBeUndefined();
  });

  it("reports each way a request can be off the contract", () => {
    expect(ltxGeometryError(770, 512, 121)).toMatch(/multiple of 32/);
    expect(ltxGeometryError(1024, 1024, 121)).toMatch(/budget/);
    expect(ltxGeometryError(768, 512, 120)).toMatch(/8n \+ 1/);
    expect(ltxGeometryError(768, 512, 1)).toMatch(/at least 9/);
    expect(LTX_MAX_PIXELS).toBe(768 * 768);
  });
});

/**
 * The same cases as `ai-runtime/tests/test_ltx_adapters.py`, because the panel reported READY for
 * a clip the worker always rejected — and printed the limits it was not checking in its own help
 * text while doing it.
 */

/**
 * Mirrors `ai-runtime/tests/test_retake_window.py`. The worker windows around the selection now,
 * so the clip's length is not a limit — the canvas and the selection are.
 */
describe("retake limits", () => {
  it("accepts the clip that started this", () => {
    // rest42.mp4: 260 frames at 960x544, a 3-second selection. Refused outright before.
    expect(ltxRetakeProblems(960, 544, 260, 24, 0, 3)).toEqual([]);
  });

  it("accepts a twenty-minute source", () => {
    expect(ltxRetakeProblems(960, 544, 28_800, 24, 600, 603)).toEqual([]);
  });

  it("still refuses a canvas a window cannot shrink", () => {
    const problems = ltxRetakeProblems(3840, 2160, 260, 24, 0, 3);
    expect(problems.join(" ")).toMatch(/14\.1× too large/);
    expect(problems.join(" ")).toMatch(/shorter, not smaller/);
  });

  it("names only the axis that is off the grid", () => {
    const [offGrid] = ltxRetakeProblems(960, 500, 260, 24, 0, 3).filter((p) => p.includes("off the 32-pixel grid"));
    expect(offGrid).toMatch(/height 500/);
    expect(offGrid).not.toMatch(/width 960/);
  });

  it("refuses a selection longer than one window, with both numbers", () => {
    const [tooLong] = ltxRetakeProblems(960, 544, 260, 24, 0, 9).filter((p) => p.includes("most that can be retaken"));
    expect(tooLong).toMatch(/216 frames \(9\.0s\)/);
    expect(tooLong).toMatch(/137 \(5\.7s\)/);
  });

  it("refuses a clip too short to retake at all", () => {
    expect(ltxRetakeProblems(960, 544, 6, 24, 0, 0.2).join(" ")).toMatch(/at least 9/);
  });

  it.each([
    [24_000, 1_001],
    [30_000, 1_001],
  ])("uses a half-open frame contract at %i/%i fps", (numerator, denominator) => {
    const range = ltxFrameRange(1.001, 2.002, numerator, denominator, 10_000);
    // 1.001 is exactly 1001/1000; do not bake JavaScript's 23.999999999 rounding into the test.
    expect(range.startFrame).toBe(numerator / 1_000);
    expect(range.endFrame).toBe(numerator / 500);
    expect(range.frames).toBe(range.endFrame - range.startFrame);
    expect(range.startSeconds).toBe(range.startFrame * denominator / numerator);
    expect(range.endSeconds).toBe(range.endFrame * denominator / numerator);
  });

  it("mirrors the Python token budget exactly", () => {
    expect(LTX_MAX_PIXEL_FRAMES).toBe(9_609_216);
  });
});
