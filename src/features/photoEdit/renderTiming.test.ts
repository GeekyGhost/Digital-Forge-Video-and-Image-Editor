import { describe, expect, it } from "vitest";
import { encodableRgba, renderTimingLine } from "./PhotoEdit";

describe("Clip VFX render helpers (VFX Phase 5)", () => {
  it("passes compositor pixels through, blacking only fully transparent ones", () => {
    const pixels = new Uint8ClampedArray([10, 20, 30, 255, 40, 50, 60, 0, 70, 80, 90, 128]);
    const bytes = encodableRgba(pixels);
    expect(Array.from(bytes)).toEqual([10, 20, 30, 255, 0, 0, 0, 0, 70, 80, 90, 128]);
    expect(bytes.buffer).toBe(pixels.buffer); // no copy
  });

  it("names decode, compose, send and the decoder's work per frame", () => {
    const line = renderTimingLine({
      name: "shot-render.mp4", frames: 124, seconds: 45.2, gpu: true,
      spent: { decode: 124 * 12, wait: 124 * 2, compose: 124 * 40, send: 124 * 30, blocked: 124 * 5 },
      encoder: "h264_nvenc", encoderSeconds: 0.124, fedSamples: 124, resets: 1,
    });
    expect(line).toBe("Rendered shot-render.mp4 · 124 frames in 45.2 s (2.7 fps; per frame: decode 12 ms (waited 2), "
      + "GPU compose 40 ms, send 30 ms (blocked 5, h264_nvenc 1 ms of it) · decoder fed 124 samples for 124 frames, 1 reset)"
      + " — replace the range or insert it after the clip");
  });

  it("names the pack stage and the transport (Phase 6)", () => {
    const line = renderTimingLine({
      name: "b.mp4", frames: 10, seconds: 1, gpu: true,
      spent: { decode: 100, wait: 0, compose: 500, pack: 40, send: 80, blocked: 10 }, fedSamples: 0, resets: 0,
      encoder: "h264_nvenc", encoderSeconds: 0.01, transport: "HTTP as YUV 4:2:0",
    });
    expect(line).toContain("GPU compose 50 ms, pack 4 ms, send 8 ms over HTTP as YUV 4:2:0 (blocked 1, h264_nvenc 1 ms of it)");
  });

  it("leaves the decoder out for a render with no video layers", () => {
    const line = renderTimingLine({
      name: "a.mp4", frames: 2, seconds: 1, gpu: false, spent: { decode: 0, wait: 0, compose: 2, send: 2, blocked: 0 }, fedSamples: 0, resets: 0,
    });
    expect(line).not.toContain("decoder fed");
    expect(line).toContain("CPU compose 1 ms");
  });
});
