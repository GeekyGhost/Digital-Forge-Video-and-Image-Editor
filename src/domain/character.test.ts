import { describe, expect, it } from "vitest";
import { actorFrame, actorRegion, alignToPlate, animateRecord, characterLayer, cleanPlateRequest, drivenByPlate, drivingRange, insertAbove, isCleanPlate, seedPoints, sequenceMattes, type CharacterElement, type PlateCrop } from "./character";
import { createMask, createMaskEntry, createMediaLayer, createPhotoDocument, type PhotoLayer } from "./photoDocument";
import { croppedRect, layerMatrix } from "./photoCompose";
import { applyMatrix } from "./photoGeometry";

const plateMedia = {
  kind: "video" as const, path: "C:/clips/actor.mp4", frameCount: 200, fps: 24, width: 960, height: 544,
  inPoint: 30, outPoint: 154, start: 0,
};
const plate = (over: Partial<PhotoLayer> = {}) => ({ ...createMediaLayer("Plate", plateMedia, { locked: true }), id: "clip-plate", ...over });
const crop: PlateCrop = [4, 0, 952, 544];
const element: CharacterElement = {
  dir: "C:/proj/elements/hero-1",
  manifest: {
    id: "hero-1", name: "Hero", frameCount: 124, fps: 24, width: 672, height: 384, pattern: "frame_%04d.png",
    proxy: "proxy.mp4", loopIn: 0, crossfade: 0,
    animate: { crop, outputWidth: 672, outputHeight: 384, plateWidth: 960, plateHeight: 544, drivingVideo: "C:/clips/actor.mp4", firstFrame: 30, frameCount: 124 },
  },
};

/** Where element pixel (s, t) lands on the canvas, and where the plate pixel it should cover lands. */
function landing(layer: PhotoLayer, onto: PhotoLayer, canvas: { width: number; height: number }, s: number, t: number) {
  const size = { width: 672, height: 384 };
  const got = applyMatrix(layerMatrix(layer, canvas.width, canvas.height, size, 1), s, t);
  const rect = croppedRect(onto, { width: 960, height: 544 });
  const want = applyMatrix(layerMatrix(onto, canvas.width, canvas.height, { width: 960, height: 544 }, 1),
    crop[0] + s * crop[2] / 672 - rect.left, crop[1] + t * crop[3] / 384 - rect.top);
  return { got, want };
}

describe("Wan Animate 2 character insertion (VFX Phase 5)", () => {
  it("drives with the plate's source range", () => {
    expect(drivingRange(plate())).toEqual({ first: 30, count: 124 });
    expect(drivingRange(createMediaLayer("still", { ...plateMedia, kind: "frames" }))).toBeUndefined();
  });

  it("lays every element pixel over the plate pixel it was generated for, however the plate is placed", () => {
    const canvas = { width: 960, height: 544 };
    for (const placed of [plate(), plate({ x: 0.4, y: 0.62, scale: 0.7, rotation: 12 }), plate({ scale: 1.3, cropLeft: 10 })]) {
      const layer = { ...createMediaLayer("Hero", { ...plateMedia, width: 672, height: 384, kind: "frames" }), ...alignToPlate(placed, canvas, crop, 672, 384) };
      for (const [s, t] of [[0, 0], [672, 0], [0, 384], [336, 192], [600, 50]]) {
        const { got, want } = landing(layer, placed, canvas, s, t);
        // One scale for both axes: the crop's aspect is the output's to within a pixel (952/544 vs 672/384).
        expect(Math.abs(got[0] - want[0])).toBeLessThan(1.1);
        expect(Math.abs(got[1] - want[1])).toBeLessThan(1.1);
      }
    }
  });

  it("seeds SAM 2 on the actor's body, not between the legs, in output coordinates", () => {
    const width = 960, height = 544;
    const alpha = new Uint8Array(width * height);
    const fill = (x0: number, y0: number, x1: number, y1: number) => {
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) alpha[y * width + x] = 255;
    };
    fill(440, 100, 520, 330); // head and torso
    fill(440, 330, 470, 500); // left leg
    fill(490, 330, 520, 500); // right leg
    const points = seedPoints(alpha, width, height, crop);
    expect(points.length).toBe(3);
    for (const point of points) {
      const u = crop[0] + point.x * crop[2], v = crop[1] + point.y * crop[3];
      expect(alpha[Math.floor(v) * width + Math.floor(u)]).toBe(255);
      expect(point.include).toBe(true);
    }
    expect(seedPoints(new Uint8Array(width * height), width, height, crop)).toEqual([]);
  });

  it("makes an aligned frames layer that starts on the plate's first driving frame", () => {
    const document = { ...createPhotoDocument("asset", "C:/clips/actor.mp4"), layers: [plate()], frameRate: 24, duration: 124 };
    const layer = characterLayer(document, plate(), element, { width: 960, height: 544 })!;
    expect(layer.media).toMatchObject({ kind: "frames", path: element.dir, start: 0, inPoint: 0, outPoint: 124, fps: 24, elementId: "hero-1" });
    expect(layer.masks).toEqual([]);
    expect(layer.scale).toBeCloseTo(952 / 672 / 2 + 544 / 384 / 2, 6);
    // A plate trimmed to start 10 frames later: the character starts 10 frames early (before the plate).
    const later = plate({ media: { ...plate().media!, inPoint: 40 } });
    expect(characterLayer(document, later, element, { width: 960, height: 544 })!.media!.start).toBe(-10);
  });

  it("has no picture file, and is occluded by the plate's depth only when asked", () => {
    const withDepth = plate({ media: { ...plate().media!, depth: { cachePath: "C:/proj/.aiv/cache/depth/x", frameCount: 200 } } });
    const document = { ...createPhotoDocument("asset", "C:/clips/actor.mp4"), layers: [withDepth], frameRate: 24 };
    const plain = characterLayer(document, withDepth, element, { width: 960, height: 544 })!;
    // The loader opens a non-empty sourcePath as an image: a frames layer must leave it empty (4090 run).
    expect([plain.sourcePath, plain.assetId]).toEqual(["", ""]);
    expect(plain.masks).toEqual([]); // the actor is the nearest thing in the plate: no automatic occlusion
    const layer = characterLayer(document, withDepth, element, { width: 960, height: 544 }, { occlude: true })!;
    expect(layer.masks.length).toBe(1);
    expect(layer.maskId).toBe(layer.masks[0].id);
    expect(layer.masks[0].stack![0].source).toEqual({ kind: "depth", layerId: "clip-plate", plane: 55, softness: 6 });
  });

  it("measures the actor on its torso, its feet line and its standing height", () => {
    const width = 100, height = 100;
    const raster = (fill: (x: number, y: number) => boolean) => {
      const alpha = new Uint8Array(width * height);
      for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (fill(x, y)) alpha[y * width + x] = 255;
      return alpha;
    };
    // Frame 0: torso x 40..50, rows 30..90, plus a leg reaching out to x 80 on rows 80..90.
    const first = raster((x, y) => (x >= 40 && x < 50 && y >= 30 && y < 90) || (x >= 50 && x < 80 && y >= 80 && y < 90));
    const standing = raster((x, y) => x >= 40 && x < 50 && y >= 20 && y < 90); // 70 tall later
    const [cx, feet, tall] = actorFrame(first, [standing], width, height)!;
    expect(cx).toBeLessThan(0.55); // the leg pulls a box centre to 0.6; the centroid stays near the torso
    expect(feet).toBe(0.9);
    expect(tall).toBe(0.7);
    expect(actorFrame(new Uint8Array(width * height), [], width, height)).toBeUndefined();
    expect(actorRegion([first, standing], width, height)).toEqual([0.4, 0.2, 0.8, 0.9]);
    expect(actorRegion([new Uint8Array(width * height)], width, height)).toBeUndefined();
  });

  it("knows which elements came from this plate and lists its SAM 2 mattes", () => {
    expect(drivenByPlate(element, plate())).toBe(true);
    expect(drivenByPlate({ ...element, manifest: { ...element.manifest, animate: undefined } }, plate())).toBe(false);
    expect(animateRecord({ ...element, manifest: { ...element.manifest, animate: { crop: [1, 2] } } })).toBeUndefined();
    const mask = { ...createMask("Actor"), stack: [createMaskEntry({ kind: "sequence", cachePath: "C:/c", frameCount: 124, first: 30 }, { mode: "add" })] };
    const listed = sequenceMattes(plate({ masks: [mask] }));
    expect(listed.length).toBe(1);
    expect(listed[0].label).toBe("Actor");
    expect(listed[0].source.first).toBe(30);
  });

  it("asks for a clean plate over the plate's range and lays it between the plate and the character", () => {
    const request = cleanPlateRequest(plate(), { cachePath: "C:/cache/actor", first: 30 }, "Actor clean plate")!;
    expect(request).toEqual({
      name: "Actor clean plate", video: "C:/clips/actor.mp4", firstFrame: 30, frameCount: 124, totalFrames: 200, fps: 24,
      plateWidth: 960, plateHeight: 544, matteCache: "C:/cache/actor", matteFirst: 30,
    });
    expect(cleanPlateRequest(createMediaLayer("still", { ...plateMedia, kind: "frames" }), { cachePath: "C:/c" }, "x")).toBeUndefined();
    const patch: CharacterElement = { ...element, manifest: { ...element.manifest, source: "clean-plate", animate: { ...element.manifest.animate as object, crop: [300, 100, 400, 380], outputWidth: 400, outputHeight: 380 } } };
    expect(isCleanPlate(patch)).toBe(true);
    expect(isCleanPlate(element)).toBe(false);
    // Placed like a character: 1:1 over its crop of the plate.
    const document = { ...createPhotoDocument("asset", "C:/clips/actor.mp4"), layers: [plate()], frameRate: 24 };
    const layer = characterLayer(document, plate(), patch, { width: 960, height: 544 })!;
    expect(layer.scale).toBeCloseTo(1, 6);
    const hero = { ...layer, id: "hero" };
    const order = insertAbove([plate(), hero], "clip-plate", { ...layer, id: "clean" }).map((item) => item.id);
    expect(order).toEqual(["clip-plate", "clean", "hero"]);
    expect(insertAbove([hero], "missing", { ...layer, id: "clean" }).map((item) => item.id)).toEqual(["hero", "clean"]);
  });
});
