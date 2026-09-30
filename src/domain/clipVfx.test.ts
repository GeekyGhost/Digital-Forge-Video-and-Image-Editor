/** VFX Phase 4b: a clip's range as the plate, and what a render of it replaces on the timeline. */
import { describe, expect, it } from "vitest";
import { clipPlateDocument, clipRenderPlan, clipVfxDocument, hasVideoContent, plateLayerId, withPlateRange } from "./clipVfx";
import { createMatrixRainLayer, createPhotoDocument, layersOf, withLayers } from "./photoDocument";
import { compositionDuration, compositionRate } from "./photoTime";
import { fromSeconds, rational } from "./time";
import type { MediaAsset } from "./types";

const asset: MediaAsset = {
  id: "a1", name: "smoker.mp4", path: "C:/m/smoker.mp4", kind: "video",
  duration: rational(10), width: 1920, height: 1080, frameRate: rational(24),
};
const clip = { id: "c1", sourceIn: fromSeconds(2), sourceOut: fromSeconds(8) };

describe("clip VFX documents", () => {
  it("makes the clip's range the plate at frame 0, at the clip's rate and length", () => {
    const document = clipPlateDocument(clip, asset, 1, 3)!;
    const [plate] = layersOf(document);
    expect(plate.id).toBe(plateLayerId("c1"));
    expect(plate.locked).toBe(true);
    expect(plate.media).toMatchObject({ kind: "video", path: asset.path, inPoint: 72, outPoint: 120, start: 0, loop: false });
    expect(compositionRate(document)).toBe(24);
    expect(compositionDuration(document)).toBe(48);
  });

  it("needs a probed asset", () => {
    expect(clipPlateDocument(clip, { ...asset, width: undefined }, 0, 1)).toBeUndefined();
  });

  it("keeps a stored document only while it still describes this clip's footage", () => {
    const stored = { ...clipPlateDocument(clip, asset, 0, 2)!, tracks: [] };
    expect(clipVfxDocument(stored, clip, asset, 0, 6)).toBe(stored);
    const other = { ...asset, id: "render" };
    expect(clipVfxDocument(stored, clip, other, 0, 6)?.assetId).toBe("render");
    expect(clipVfxDocument(stored, { ...clip, id: "c2" }, asset, 0, 6)).not.toBe(stored);
  });

  it("plans a render over the plate's range, with the plate's audio", () => {
    const document = clipPlateDocument(clip, asset, 1, 3)!;
    const plan = clipRenderPlan(document, clip)!;
    expect(plan.frames).toBe(48);
    expect(plan.fps).toBe(24);
    expect(plan.replaceStart).toBeCloseTo(1);
    expect(plan.replaceEnd).toBeCloseTo(3);
    expect(plan.audio).toMatchObject({ path: asset.path, start: 3, duration: 2 });
  });

  it("clamps a composition longer than the clip to the clip", () => {
    const document = { ...clipPlateDocument(clip, asset, 5, 6)!, duration: 240 };
    const plan = clipRenderPlan(document, clip)!;
    expect(plan.replaceStart).toBeCloseTo(5);
    expect(plan.replaceEnd).toBeCloseTo(6);
  });

  it("moves the plate to a new range and keeps everything else", () => {
    const document = { ...clipPlateDocument(clip, asset, 1, 3)!, tracks: [{ id: "t", name: "T", layerId: plateLayerId("c1"), first: 72, samples: [], source: "lucas-kanade" as const }] };
    const whole = withPlateRange(document, clip, 0, 6);
    expect(layersOf(whole)[0].media).toMatchObject({ inPoint: 48, outPoint: 192, start: 0 });
    expect(compositionDuration(whole)).toBe(144);
    expect(whole.tracks).toBe(document.tracks);
    expect(clipRenderPlan(whole, clip)).toMatchObject({ replaceStart: 0, replaceEnd: 6, frames: 144 });
  });

  it("flags Photo Edit documents that carry video content", () => {
    const still = createPhotoDocument("i1", "C:/i/still.png");
    expect(hasVideoContent(still)).toBe(false);
    const rain = withLayers(still);
    expect(hasVideoContent({ ...rain, layers: [...layersOf(rain), createMatrixRainLayer()] })).toBe(false);
    expect(hasVideoContent(clipPlateDocument(clip, asset, 0, 1)!)).toBe(true);
  });
});
