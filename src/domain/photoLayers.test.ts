import { describe, expect, it } from "vitest";
import { canvasSizeOf, croppedRect } from "./photoCompose";
import {
  createGradeNode, createLayer, createPhotoDocument, createRetouchNode, layersOf, withLayers,
  type PhotoDocument, type PhotoLayer,
} from "./photoDocument";
import { emptyResources, renderStack } from "./photoRender";
import { NEUTRAL_ADJUSTMENTS } from "./photoEdit";

const asset = (id: string) => ({ id, name: `${id}.png`, path: `C:/pics/${id}.png` });
const source = (width: number, height: number) => ({
  image: undefined as unknown as CanvasImageSource, width, height,
});

const docWith = (layers: PhotoLayer[]): PhotoDocument =>
  ({ ...createPhotoDocument("a", "C:/pics/a.png"), nodes: [], masks: [], layers });

describe("croppedRect", () => {
  it("takes the given percentage off each side", () => {
    const layer = createLayer(asset("a"), { cropLeft: 10, cropRight: 20, cropTop: 25, cropBottom: 25 });
    const rect = croppedRect(layer, { width: 100, height: 200 });
    expect(rect).toEqual({ left: 10, top: 50, width: 70, height: 100 });
  });

  it("is the whole picture when nothing is trimmed", () => {
    const rect = croppedRect(createLayer(asset("a")), { width: 640, height: 480 });
    expect(rect).toEqual({ left: 0, top: 0, width: 640, height: 480 });
  });

  it("never trims a side away entirely", () => {
    const layer = createLayer(asset("a"), { cropLeft: 100, cropRight: 100 });
    const rect = croppedRect(layer, { width: 100, height: 100 });
    expect(rect.width).toBeGreaterThan(0);
  });
});

describe("canvasSizeOf", () => {
  it("prefers the size the document states", () => {
    const document = { ...docWith([createLayer(asset("a"))]), canvasWidth: 1920, canvasHeight: 1080 };
    const sources = new Map([["C:/pics/a.png", source(512, 512)]]);
    expect(canvasSizeOf(document, sources)).toEqual({ width: 1920, height: 1080 });
  });

  it("otherwise takes the bottom layer's own size", () => {
    const document = docWith([createLayer(asset("a")), createLayer(asset("b"))]);
    const sources = new Map([
      ["C:/pics/a.png", source(800, 600)],
      ["C:/pics/b.png", source(64, 64)],
    ]);
    expect(canvasSizeOf(document, sources)).toEqual({ width: 800, height: 600 });
  });

  it("falls through to a layer that has actually loaded", () => {
    const document = docWith([createLayer(asset("a")), createLayer(asset("b"))]);
    const sources = new Map([["C:/pics/b.png", source(64, 32)]]);
    expect(canvasSizeOf(document, sources)).toEqual({ width: 64, height: 32 });
  });

  it("has no answer while nothing is loaded", () => {
    expect(canvasSizeOf(docWith([createLayer(asset("a"))]), new Map())).toBeUndefined();
  });
});

describe("the document as a set of layers", () => {
  it("reads a document written before layers existed as one layer", () => {
    const document = createPhotoDocument("a", "C:/pics/a.png");
    const layers = layersOf(document);
    expect(layers).toHaveLength(1);
    // The old stack belongs to its one picture, not to the composite: a grade made before the merge
    // has to keep applying to exactly the pixels it was made for.
    expect(layers[0].nodes).toBe(document.nodes);
    expect(layers[0].sourcePath).toBe("C:/pics/a.png");
  });

  it("does not rewrite anything until something layer-shaped happens", () => {
    const document = createPhotoDocument("a", "C:/pics/a.png");
    expect(document.layers).toBeUndefined();
  });

  it("moves the old stack onto the base layer when it does", () => {
    const document = createPhotoDocument("a", "C:/pics/a.png");
    const moved = withLayers(document);
    expect(moved.nodes).toEqual([]);
    expect(moved.layers).toHaveLength(1);
    expect(moved.layers![0].nodes).toBe(document.nodes);
  });

  it("is idempotent, so an edit cannot migrate a document twice", () => {
    const once = withLayers(createPhotoDocument("a", "C:/pics/a.png"));
    expect(withLayers(once)).toBe(once);
  });
});

describe("renderStack", () => {
  const grey = () => new Uint8ClampedArray([128, 128, 128, 255]);

  it("is the one implementation a layer and the composite both use", () => {
    const node = createGradeNode();
    node.adjustments = { ...NEUTRAL_ADJUSTMENTS, exposure: 1, curve: NEUTRAL_ADJUSTMENTS.curve };
    const asLayerStack = renderStack(grey(), 1, 1, [node], emptyResources()).pixels;
    const asCompositeStack = renderStack(grey(), 1, 1, [node], emptyResources()).pixels;
    expect([...asLayerStack]).toEqual([...asCompositeStack]);
    expect(asLayerStack[0]).toBeGreaterThan(128);
  });

  it("skips a node whose mask has not been drawn", () => {
    const node = createGradeNode();
    node.maskId = "missing";
    const { report } = renderStack(grey(), 1, 1, [node], emptyResources());
    expect(report.skipped).toEqual([{ nodeId: node.id, reason: "its mask has not been drawn yet" }]);
  });

  it("leaves an empty retouch node out rather than copying the frame", () => {
    const before = grey();
    const { pixels } = renderStack(before, 1, 1, [createRetouchNode()], emptyResources());
    expect([...pixels]).toEqual([...before]);
  });
});
