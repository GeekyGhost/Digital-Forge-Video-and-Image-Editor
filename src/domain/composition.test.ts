import { describe, expect, it } from "vitest";
import {
  COMPOSITION_VERSION, drawableLayers, fromPhotoDocument, fromPhotoLayer, photoLayerKind,
  toPhotoDocument, toPhotoLayer, type Composition,
} from "./composition";
import {
  createGradeNode, createGradientLayer, createGradientMapNode, createLayer, createMask,
  createPhotoDocument, createRelightNode, createRemoveNode, createRetouchNode, createShapeLayer,
  createTextLayer, createWarpNode, layersOf, type PhotoDocument, type PhotoLayer,
  createLumaKeyNode, createMaskEntry, createUnmultiplyNode,
  createAlphaCurveNode, createMatrixRainLayer, createMediaLayer, createTrailNode, createEmitterLayer,
  createPathLayer, createEnvelopeNode,
} from "./photoDocument";

/** JSON with sorted keys: what the project file would hold, independent of key order. */
function canonical(value: unknown): string {
  const sort = (item: unknown): unknown => {
    if (Array.isArray(item)) return item.map(sort);
    if (item && typeof item === "object") {
      return Object.fromEntries(Object.keys(item).sort()
        .filter((key) => (item as Record<string, unknown>)[key] !== undefined)
        .map((key) => [key, sort((item as Record<string, unknown>)[key])]));
    }
    return item;
  };
  return JSON.stringify(sort(value));
}

const asset = (n: number) => ({ id: `asset-${n}`, name: `Picture ${n}`, path: `C:/media/p${n}.png` });

function everyMask() {
  const strokes = createMask("strokes");
  strokes.strokes = [
    { points: [[0.1, 0.2], [0.3, 0.4]], radius: 0.05, hardness: 0.5, erase: false },
    { points: [[0.2, 0.2]], radius: 0.02, hardness: 1, erase: true },
  ];
  const linear = { ...createMask("linear"), shape: { kind: "linear" as const, x1: 0, y1: 0, x2: 1, y2: 1 } };
  const radial = {
    ...createMask("radial"), invert: true,
    shape: { kind: "radial" as const, cx: 0.5, cy: 0.5, radius: 0.3, aspect: 1.5, feather: 0.4 },
  };
  const colour = {
    ...createMask("colour"), feather: 0,
    shape: { kind: "colour" as const, r: 200, g: 40, b: 20, tolerance: 30, softness: 10 },
  };
  const sam = { ...createMask("sam"), sourcePath: "C:/masks/sam.png", enabled: false };
  return [strokes, linear, radial, colour, sam];
}

function everyNode(maskIds: string[]) {
  const grade = createGradeNode();
  grade.adjustments = { ...grade.adjustments, exposure: 0.4, contrast: 20, curve: [{ x: 0.2, y: 0.3 }, { x: 0.5, y: 0.5 }, { x: 0.8, y: 0.7 }] };
  grade.maskId = maskIds[0];
  const retouch = createRetouchNode();
  retouch.strokes = [{ points: [[0.1, 0.1]], radius: 0.02, hardness: 0.8, offset: [0.1, 0], mode: "clone" } as never];
  const warp = createWarpNode();
  warp.strokes = [{ points: [[0.5, 0.5]], radius: 0.1, strength: 0.5, mode: "push", direction: [0.1, 0] } as never];
  const map = createGradientMapNode();
  map.enabled = false;
  return [grade, createRelightNode(), createRemoveNode(maskIds[1]), retouch, warp, map];
}

function layeredDocument(): PhotoDocument {
  const base = createPhotoDocument("asset-1", "C:/media/p1.png");
  const masks = everyMask();
  const image = createLayer(asset(1), { name: "Plate", masks, nodes: everyNode(masks.map((m) => m.id)) });
  const cutout = createLayer(asset(2), {
    x: 0.3, y: 0.7, scale: 0.6, rotation: 12.5, cropLeft: 5, cropTop: 2, cropRight: 10, cropBottom: 0,
    opacity: 0.8, blendMode: "multiply", locked: true,
    edge: { defringe: 2, lightWrap: 30, lightWrapSpread: 12, contactShadow: 50, contactShadowSoftness: 40, contactShadowOffset: 0.02, contactShadowSquash: 0.18 },
  });
  const text = createTextLayer({ width: 1920, height: 1080 }, { content: "Hello" });
  const gradient = createGradientLayer({ kind: "radial", angle: 30 });
  const shape = createShapeLayer("star", { width: 1920, height: 1080 });
  const balloon = createShapeLayer("speech", { width: 1920, height: 1080 });
  const hidden = { ...createLayer(asset(3)), visible: false, blendMode: "screen" as const };
  return {
    ...base,
    layers: [image, cutout, text, gradient, shape, balloon, hidden],
    canvasWidth: 1920, canvasHeight: 1080, background: "white",
    nodes: [createGradeNode()],
    masks: [createMask("composite")],
    depthPath: "C:/cache/depth.png",
    rotate: 90, flipHorizontal: true, flipVertical: false,
    straighten: 1.5, scale: 1.1, crop: { x: 0.1, y: 0.1, width: 0.8, height: 0.7 },
  };
}

describe("composition round trip", () => {
  const cases: Array<[string, () => PhotoDocument]> = [
    ["a fresh single-source document", () => createPhotoDocument("a", "C:/a.png")],
    ["a single-source document with every mask and node", () => {
      const doc = createPhotoDocument("a", "C:/a.png");
      const masks = everyMask();
      return { ...doc, masks, nodes: everyNode(masks.map((m) => m.id)), depthPath: "C:/d.png" };
    }],
    ["a single-source document with an empty layers array", () => ({ ...createPhotoDocument("a", "C:/a.png"), layers: [] })],
    ["a document with every layer kind and all optional fields", layeredDocument],
    ["a one-layer layered document", () => ({ ...createPhotoDocument("a", "C:/a.png"), nodes: [], masks: [], layers: [createLayer(asset(1))] })],
    ["a layer written before blend modes and edges existed", () => {
      const layer = createLayer(asset(1)) as Partial<PhotoLayer>;
      delete layer.blendMode;
      delete layer.edge;
      return { ...createPhotoDocument("a", "C:/a.png"), nodes: [], masks: [], layers: [layer as PhotoLayer, createLayer(asset(2))] };
    }],
    ["fields this build does not know", () => {
      const doc = layeredDocument();
      (doc as Record<string, unknown>).futureDocField = { deep: [1, 2] };
      (doc.layers![1] as Record<string, unknown>).futureLayerField = "kept";
      return doc;
    }],
    ["VFX Phase 2: mask stacks, layer masks, clipping, track mattes, keys", () => {
      const plate = createLayer(asset(1), { name: "Plate" });
      const matte = createLayer(asset(4), { name: "Matte", visible: false });
      const stack = {
        ...createMask("stack"),
        stack: [
          createMaskEntry({ kind: "shape", shape: { kind: "radial", cx: 0.4, cy: 0.5, radius: 0.2, aspect: 1, feather: 0.3 } }, { mode: "subtract", opacity: 0.6, feather: 4, expand: -2 }),
          createMaskEntry({ kind: "file", path: "C:/mattes/imported.png", channel: "luma" }, { mode: "intersect", invert: true }),
          createMaskEntry({ kind: "sequence", cachePath: "C:/cache/sam2/abc", frameCount: 81, fps: 16 }, { mode: "difference", enabled: false }),
          createMaskEntry({ kind: "paint" }, { strokes: [{ points: [[0.2, 0.2]], radius: 0.03, hardness: 0.5, erase: false }] }),
        ],
      };
      const trackMatte = { ...createMask("track matte"), stack: [createMaskEntry({ kind: "layer", layerId: matte.id, channel: "luma" }, { invert: true })] };
      const cutout = createLayer(asset(2), {
        masks: [stack, trackMatte], maskId: trackMatte.id,
        nodes: [createUnmultiplyNode(), { ...createLumaKeyNode(), maskId: stack.id, invert: true }],
      });
      const clipped = createLayer(asset(3), { clip: true, opacity: 0.5 });
      const notClipped = createLayer(asset(5), { clip: false });
      return { ...createPhotoDocument("a", "C:/a.png"), nodes: [], masks: [], layers: [plate, matte, cutout, clipped, notClipped] };
    }],
    ["VFX Phase 2: a pre-layers document whose one mask has a stack", () => {
      const mask = { ...createMask("old"), strokes: [], stack: [createMaskEntry({ kind: "shape", shape: { kind: "linear", x1: 0, y1: 0, x2: 1, y2: 0 } })] };
      const grade = createGradeNode();
      grade.maskId = mask.id;
      return { ...createPhotoDocument("a", "C:/a.png"), masks: [mask], nodes: [grade] };
    }],
    ["VFX Phase 3: video, element and generator layers, and a document rate and length", () => {
      const video = createMediaLayer("Clip", { kind: "video", path: "C:/media/clip.mp4", frameCount: 120, fps: 23.976, width: 1920, height: 1080, proxyPath: "C:/media/clip.proxy.mp4" }, { scale: 0.5 });
      const element = createMediaLayer("Smoke", {
        kind: "frames", path: "C:/AI/elements/abc", pattern: "frame_%04d.png", frameCount: 81, fps: 16, width: 832, height: 480,
        start: 12, inPoint: 4, outPoint: 77, loop: true, crossfade: 8, proxyPath: "C:/AI/elements/abc/proxy.mp4", proxyStacked: true, elementId: "abc",
      }, { blendMode: "screen", nodes: [createAlphaCurveNode()] });
      const rain = createMatrixRainLayer({ seed: 7 });
      return { ...createPhotoDocument("a", "C:/a.png"), nodes: [], masks: [], frameRate: 24, duration: 240, layers: [createLayer(asset(1)), video, element, rain] };
    }],
    ["VFX Phase 4: tracks, attach, trail, emitter and depth occlusion", () => {
      const plate = createMediaLayer("Plate", {
        kind: "video", path: "C:/media/smoker.mp4", frameCount: 96, fps: 24, width: 1920, height: 1080,
        depth: { cachePath: "C:/AI/cache/depth/abc", frameCount: 96 },
      });
      const track = { id: "track-1", name: "Cigarette tip", layerId: plate.id, first: 3, source: "lucas-kanade" as const,
        samples: [[812.5, 400.25, 1], [815, 398, 1], [0, 0, 0], [820.75, 391.5, 1]] as Array<[number, number, 0 | 1]> };
      const occlusion = { ...createMask("occluded"), stack: [createMaskEntry({ kind: "depth", layerId: plate.id, plane: 55, softness: 8 }, { mode: "subtract" })] };
      const smoke = createMediaLayer("Smoke", {
        kind: "frames", path: "C:/AI/elements/smoke", pattern: "frame_%04d.png", frameCount: 81, fps: 16, width: 832, height: 480,
      }, { attach: { trackId: track.id, anchorFrame: 10, lag: 2 }, nodes: [createTrailNode(track.id)], masks: [occlusion], maskId: occlusion.id });
      const emitter = createEmitterLayer(smoke.media!, { trackId: track.id, seed: 99 });
      return { ...createPhotoDocument("a", "C:/a.png"), nodes: [], masks: [], frameRate: 24, tracks: [track], layers: [plate, smoke, emitter] };
    }],
    ["VFX Phase 6: path layers, a path mask, an envelope node and repeats", () => {
      const star = createPathLayer({
        subpaths: [
          { closed: true, nodes: [{ x: 0, y: -50 }, { x: 40, y: 30, in: [5, -5], out: [-3, 8] }, { x: -40, y: 30 }] },
          { closed: false, nodes: [{ x: -10, y: 0 }, { x: 10, y: 0, in: [-4, 4] }] },
        ],
        fillRule: "evenodd", dash: [6, 3], dashOffset: 2, cap: "round", join: "bevel",
        envelope: { cols: 1, rows: 1, rect: { x: -40, y: -50, width: 80, height: 80 }, points: [[-40, -50], [60, -50], [-40, 30], [40, 30]] },
      }, 0.3, 0.6, "Star");
      const noFill = { ...createPathLayer({ subpaths: [] }), path: { ...createPathLayer({ subpaths: [] }).path!, fill: undefined }, repeat: { kind: "radial" as const, count: 6, radius: 90, rotate: true } };
      const pathMask = { ...createMask("vector"), stack: [createMaskEntry({ kind: "path", path: { ...star.path!, subpaths: [{ closed: true, nodes: [{ x: 0.1, y: 0.1 }, { x: 0.5, y: 0.2 }, { x: 0.3, y: 0.7 }] }] } }, { mode: "intersect" })] };
      const image = createLayer(asset(1), { masks: [pathMask], maskId: pathMask.id, nodes: [createEnvelopeNode({ x: 0.2, y: 0.2, width: 0.5, height: 0.5 }, 2, 1)],
        repeat: { kind: "grid", columns: 3, rows: 2, gapX: 120, gapY: 80, shift: 0.5, scaleStep: 0.9, opacityStep: 0.8 } });
      const along = createLayer(asset(2), { repeat: { kind: "path", layerId: star.id, count: 12, start: 0.1, end: 0.9, rotate: false } });
      const both = { ...createPathLayer({ subpaths: [] }), shape: createShapeLayer("star", undefined).shape, text: createTextLayer(undefined).text };
      return { ...createPhotoDocument("a", "C:/a.png"), nodes: [], masks: [], layers: [image, star, noFill, along, both] };
    }],
    ["a gradient layer that also carries text", () => {
      const layer = { ...createGradientLayer(), text: createTextLayer(undefined).text };
      return { ...createPhotoDocument("a", "C:/a.png"), nodes: [], masks: [], layers: [createLayer(asset(1)), layer] };
    }],
  ];

  for (const [name, make] of cases) {
    it(`is lossless for ${name}`, () => {
      const document = make();
      const composition = fromPhotoDocument(document);
      expect(composition.version).toBe(COMPOSITION_VERSION);
      expect(canonical(toPhotoDocument(composition))).toBe(canonical(document));
      // And survives the project file: JSON out, JSON back in.
      const reloaded = JSON.parse(JSON.stringify(composition)) as Composition;
      expect(canonical(toPhotoDocument(reloaded))).toBe(canonical(document));
    });
  }

  it("reads a single-source document as one layer holding its stack, like layersOf", () => {
    const masks = everyMask();
    const doc = { ...createPhotoDocument("a", "C:/a.png"), masks, nodes: everyNode(masks.map((m) => m.id)) };
    const composition = fromPhotoDocument(doc);
    expect(composition.photoShape).toBe("single-source");
    expect(composition.layers).toHaveLength(1);
    expect(composition.layers[0].nodes).toBe(doc.nodes);
    expect(composition.nodes).toEqual([]);
    expect(canonical(toPhotoLayer(composition.layers[0]))).toBe(canonical(layersOf(doc)[0]));
  });

  it("does not write back the old shape once a second layer exists", () => {
    const composition = fromPhotoDocument(createPhotoDocument("a", "C:/a.png"));
    composition.layers.push(fromPhotoLayer(createLayer(asset(2))));
    const document = toPhotoDocument(composition);
    expect(document.layers).toHaveLength(2);
    expect(document.nodes).toEqual([]);
  });
});

describe("layer kinds", () => {
  it("follows the compositor's drawing priority", () => {
    expect(photoLayerKind(createLayer(asset(1)))).toBe("image");
    expect(photoLayerKind(createTextLayer(undefined))).toBe("text");
    expect(photoLayerKind(createShapeLayer("speech", undefined))).toBe("shape");
    expect(photoLayerKind({ ...createGradientLayer(), text: createTextLayer(undefined).text })).toBe("gradient");
    expect(photoLayerKind(createPathLayer({ subpaths: [] }))).toBe("vector");
    expect(photoLayerKind({ ...createPathLayer({ subpaths: [] }), shape: createShapeLayer("star", undefined).shape })).toBe("vector");
  });

  it("groups placement and source", () => {
    const layer = fromPhotoLayer(createLayer(asset(4), { x: 0.2, scale: 2, cropRight: 7 }));
    expect(layer.kind).toBe("image");
    expect(layer.source).toEqual({ assetId: "asset-4", sourcePath: "C:/media/p4.png" });
    expect(layer.transform).toMatchObject({ x: 0.2, scale: 2, cropRight: 7 });
  });

  it("refuses to store a kind the photo document cannot hold", () => {
    const layer = { ...fromPhotoLayer(createLayer(asset(1))), kind: "adjustment" } as const;
    expect(() => toPhotoLayer(layer)).toThrow(/adjustment/);
  });

  it("flattens groups for drawing", () => {
    const composition = fromPhotoDocument(layeredDocument());
    const [first, ...rest] = composition.layers;
    composition.layers = [first, { ...fromPhotoLayer(createLayer(asset(9))), kind: "group", children: rest }];
    expect(drawableLayers(composition).map((layer) => layer.id))
      .toEqual([first.id, ...rest.map((layer) => layer.id)]);
  });
});
