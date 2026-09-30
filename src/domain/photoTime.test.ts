import { describe, expect, it } from "vitest";
import { createMatrixRainLayer, createMediaLayer, createPhotoDocument, type LayerMedia } from "./photoDocument";
import {
  applyAlphaCurve, compositionDuration, compositionRate, crossfadeFrames, formatPattern, mediaFrameKey,
  mediaFramesAt, mediaKeysAt, parseVideoFrameKey, scopeFrame,
} from "./photoTime";

const media = (over: Partial<LayerMedia> = {}): LayerMedia => ({
  kind: "frames", path: "C:\\el\\smoke", frameCount: 81, fps: 16, width: 832, height: 480,
  start: 0, inPoint: 0, outPoint: 81, loop: false, crossfade: 0, ...over,
});

describe("mediaFramesAt", () => {
  it("is the identity at the same rate, hidden before start and after the out point", () => {
    const m = media({ start: 10, inPoint: 5, outPoint: 20 });
    expect(mediaFramesAt(m, 9, 16)).toBeUndefined();
    expect(mediaFramesAt(m, 10, 16)).toEqual([{ index: 5, weight: 1 }]);
    expect(mediaFramesAt(m, 24, 16)).toEqual([{ index: 19, weight: 1 }]);
    expect(mediaFramesAt(m, 25, 16)).toBeUndefined();
  });

  it("maps between rates without drifting on exact multiples", () => {
    const m = media({ fps: 12 });
    expect([0, 1, 2, 3, 4].map((f) => mediaFramesAt(m, f, 24)![0].index)).toEqual([0, 0, 1, 1, 2]);
    const ntsc = media({ fps: 24000 / 1001, frameCount: 10000, outPoint: 10000 });
    expect(mediaFramesAt(ntsc, 2400, 24000 / 1001)![0].index).toBe(2400);
  });

  it("loops with a crossfade whose weights are exact and continuous across the wrap", () => {
    const m = media({ inPoint: 0, outPoint: 20, loop: true, crossfade: 4 });
    // Period 16. p = 0..3 blend tail (16..19) into head (0..3).
    expect(mediaFramesAt(m, 0, 16)).toEqual([{ index: 16, weight: 0.8 }, { index: 0, weight: 0.2 }]);
    expect(mediaFramesAt(m, 3, 16)).toEqual([{ index: 19, weight: 0.19999999999999996 }, { index: 3, weight: 0.8 }]);
    expect(mediaFramesAt(m, 4, 16)).toEqual([{ index: 4, weight: 1 }]);
    expect(mediaFramesAt(m, 15, 16)).toEqual([{ index: 15, weight: 1 }]);
    expect(mediaFramesAt(m, 16, 16)).toEqual(mediaFramesAt(m, 0, 16));
    expect(mediaFramesAt(m, 16 * 7 + 5, 16)).toEqual(mediaFramesAt(m, 5, 16));
  });

  it("caps the crossfade at half the loop and loops plainly without one", () => {
    expect(mediaFramesAt(media({ outPoint: 6, loop: true, crossfade: 50 }), 0, 16)!.length).toBe(2);
    expect(mediaFramesAt(media({ outPoint: 6, loop: true }), 13, 16)).toEqual([{ index: 1, weight: 1 }]);
  });
});

describe("keys and rates", () => {
  it("names frames the loaders understand", () => {
    expect(formatPattern("frame_%04d.png", 12)).toBe("frame_0012.png");
    expect(mediaFrameKey(media(), 12)).toBe("C:\\el\\smoke\\frame_0012.png");
    expect(mediaFrameKey(media({ kind: "video", path: "/v/a.mp4" }), 7)).toBe("/v/a.mp4#7");
    expect(mediaFrameKey(media({ proxyPath: "/p.mp4" }), 7, true)).toBe("/p.mp4#7");
    expect(parseVideoFrameKey("C:/a#b.mp4#31")).toEqual({ path: "C:/a#b.mp4", index: 31 });
    expect(parseVideoFrameKey("C:/a.png")).toBeUndefined();
  });

  it("derives the rate and length from the layers when the document has none", () => {
    const element = createMediaLayer("s", media({ start: 8, outPoint: 40 }));
    const doc = { ...createPhotoDocument("a", "a.png"), layers: [element, createMatrixRainLayer({ loopFrames: 30 })] };
    expect(compositionRate(doc)).toBe(16);
    expect(compositionDuration(doc)).toBe(48);
    expect(compositionDuration({ ...doc, duration: 100 })).toBe(100);
    expect(mediaKeysAt(doc, 8)).toEqual(["C:\\el\\smoke\\frame_0000.png"]);
    expect(scopeFrame(element, mediaFramesAt(element.media!, 20, 16), 20)).toBe(12);
    expect(scopeFrame(undefined, undefined, 20)).toBe(20);
  });
});

describe("crossfadeFrames and applyAlphaCurve", () => {
  it("blends premultiplied, so a transparent frame's colour does not leak", () => {
    const a = new Uint8ClampedArray([255, 0, 0, 255, 0, 255, 0, 0]);
    const b = new Uint8ClampedArray([0, 0, 255, 255, 0, 0, 255, 200]);
    const out = crossfadeFrames(a, b, 0.25);
    expect(Array.from(out.subarray(0, 4))).toEqual([191, 0, 64, 255]);
    // First frame transparent: the colour is the second frame's alone.
    expect(Array.from(out.subarray(4, 8))).toEqual([0, 0, 255, 50]);
  });

  it("raises and reshapes alpha, leaving zero alone", () => {
    const pixels = new Uint8ClampedArray([9, 9, 9, 0, 9, 9, 9, 64, 9, 9, 9, 200]);
    applyAlphaCurve(pixels, 2, 1);
    expect([pixels[3], pixels[7], pixels[11]]).toEqual([0, 128, 255]);
  });
});
