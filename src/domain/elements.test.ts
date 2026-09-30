import { describe, expect, it } from "vitest";
import { mediaFromElement, mediaFromVideoAsset, WAN_ALPHA_PRESETS } from "./elements";

describe("element and video media", () => {
  it("places an element with its manifest's loop and stacked proxy", () => {
    const media = mediaFromElement({ dir: "C:\\AI\\elements\\smoke-1", manifest: {
      id: "smoke-1", frameCount: 81, fps: 16, width: 832, height: 480, pattern: "frame_%04d.png",
      proxy: "proxy.mp4", loopIn: 4, loopOut: 76, crossfade: 8,
    } }, 12);
    expect(media).toMatchObject({ kind: "frames", start: 12, inPoint: 4, outPoint: 76, loop: true, crossfade: 8,
      proxyPath: "C:\\AI\\elements\\smoke-1\\proxy.mp4", proxyStacked: true, elementId: "smoke-1" });
    const plain = mediaFromElement({ dir: "/e/x", manifest: { id: "x", frameCount: 33, fps: 16, width: 8, height: 8, pattern: "", loopIn: 0, crossfade: 0 } });
    expect(plain).toMatchObject({ loop: false, outPoint: 33, pattern: "frame_%04d.png", proxyPath: undefined });
  });

  it("turns a probed video asset into frames, and refuses an unprobed one", () => {
    const asset = { id: "v", name: "clip", path: "C:/v.mp4", kind: "video" as const,
      duration: { numerator: 5, denominator: 1 }, frameRate: { numerator: 24000, denominator: 1001 }, width: 1920, height: 1080 };
    expect(mediaFromVideoAsset(asset)).toMatchObject({ kind: "video", frameCount: 120, width: 1920 });
    expect(mediaFromVideoAsset({ ...asset, width: undefined })).toBeUndefined();
  });

  it("has presets that start the way Wan-Alpha expects", () => {
    for (const preset of WAN_ALPHA_PRESETS) expect(preset.prompt.startsWith("Transparent background.")).toBe(true);
  });
});
