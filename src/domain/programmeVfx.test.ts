import { describe, expect, it } from "vitest";
import { DEFAULT_MASK_VFX_RECIPE } from "./maskEffects";
import type { ClipVfxLayer } from "./maskVfx";
import { maskVfxHook, maskVfxNode } from "./programmeVfx";
import type { ProgrammeMedia } from "./programme";

const cache = { cacheId: "c", cachePath: "C:/cache/m1", manifestPath: "C:/cache/m1/manifest.json", frameCount: 48, width: 1920, height: 1080, fps: 24, model: "sam2" as const, createdAt: "" };
const layer = (over: Partial<ClipVfxLayer> = {}): ClipVfxLayer => ({
  id: "L", name: "fx", enabled: true, rangeStart: 1, rangeEnd: 3, recipe: { ...DEFAULT_MASK_VFX_RECIPE },
  matteCache: cache, createdAt: "", updatedAt: "", ...over,
});
const media: ProgrammeMedia = { assetId: "a", path: "C:/a.mp4", kind: "video", fps: 24, frameCount: 480, width: 1920, height: 1080, hasAudio: false };

describe("programme mask VFX", () => {
  it("resolves the matte frame, count and source size over the layer's source range", () => {
    // Clip starts at source 2 s; the layer covers source 3..5 s.
    expect(maskVfxNode("clip", 2, layer(), 2.9, 24)).toBeUndefined();
    const node = maskVfxNode("clip", 2, layer(), 3.5, 24)!;
    expect(node).toMatchObject({ kind: "maskVfx", id: "vfx:clip:L", index: 12, count: 48, fps: 24, sourceWidth: 1920, mattePath: "C:/cache/m1/masks/00012.png" });
    expect(maskVfxNode("clip", 2, layer(), 5, 24)).toBeUndefined();
  });

  it("uses the baked matte only when it matches the recipe's smoothing", () => {
    const baked = layer({ stableMatte: { cachePath: "C:/cache/m1/stable-s2-c300", temporalSmooth: 2, cleanup: 300 } });
    expect((maskVfxNode("c", 0, baked, 1.5, 24) as { mattePath: string }).mattePath).toBe("C:/cache/m1/stable-s2-c300/masks/00012.png");
    const stale = layer({ stableMatte: { cachePath: "C:/cache/m1/stable-s1-c300", temporalSmooth: 1, cleanup: 300 } });
    expect((maskVfxNode("c", 0, stale, 1.5, 24) as { mattePath: string }).mattePath).toBe("C:/cache/m1/masks/00012.png");
  });

  it("reads the envelope and the effect clip frame (retimed, looped)", () => {
    const clipLayer = layer({
      recipe: { ...DEFAULT_MASK_VFX_RECIPE, matteStyle: "clip", effectClipPath: "C:/fx/boom.mov", effectSpeed: 2, loop: true, audioReactive: "bass" },
      effectClipMedia: { frameCount: 10, fps: 24, width: 640, height: 360 },
      envelope: { mode: "bass", smoothing: 0.45, values: Array.from({ length: 48 }, (_, i) => i / 47) },
    });
    const node = maskVfxNode("c", 0, clipLayer, 1.5, 24) as { envelope: number; effectClip: { index: number; media: { path: string } } };
    expect(node.envelope).toBeCloseTo(12 / 47);
    expect(node.effectClip.media.path).toBe("C:/fx/boom.mov");
    expect(node.effectClip.index).toBe(24 % 10);
  });

  it("swaps in the rendered file for a layer without a matte, or everywhere in legacy mode", () => {
    const clip = { id: "c", media, sourceIn: 2, rate: 1, vfx: [layer({ matteCache: undefined, renderCachePath: "C:/cache/r.mp4" })] };
    expect(maskVfxHook()(clip, 1.5, 36, 24)).toEqual({ nodes: [], masks: [], sourcePath: "C:/cache/r.mp4", sourceSeconds: 0.5 });
    const live = { ...clip, vfx: [layer({ renderCachePath: "C:/cache/r.mp4" })] };
    expect(maskVfxHook()(live, 1.5, 36, 24)?.nodes).toHaveLength(1);
    expect(maskVfxHook("legacy")(live, 1.5, 36, 24)?.sourcePath).toBe("C:/cache/r.mp4");
    expect(maskVfxHook()(live, 0.5, 12, 24)).toBeUndefined();
  });
});

describe("the Mask Effects preview document", () => {
  it("is a one-clip programme showing the asked source frame with the layer's node", async () => {
    const { vfxPreviewDocument } = await import("./programmeVfx");
    const document = vfxPreviewDocument(media, "clip", 2, layer(), 3.5);
    expect(document.canvasWidth).toBe(1920);
    const [only] = document.layers;
    expect(only.media).toMatchObject({ inPoint: 84 });
    expect(only.nodes.map((node) => node.kind)).toEqual(["maskVfx"]);
    expect(vfxPreviewDocument(media, "clip", 2, layer(), 2.5).layers[0].nodes).toEqual([]);
  });
});
