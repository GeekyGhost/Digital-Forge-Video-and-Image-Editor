/**
 * VFX Phase 2: mask stacks, layer masks, clipping and track mattes -- the CPU references the GPU
 * renderer is held to by the parity harness (src/render/parity.ts). Everything here runs in node:
 * compositing tests use gradient layers, which are generated rather than placed through a canvas.
 */
import { describe, expect, it } from "vitest";
import {
  COMPOSITE_SCOPE, MATTE_CYCLE, applyAlphaCut, clipBases, composeDocument, referencedLayers,
  type ComposeInput, type ComposeScope,
} from "./photoCompose";
import {
  createGradeNode, createGradientLayer, createMask, createMaskEntry, createPhotoDocument,
  type MaskEntry, type PhotoDocument, type PhotoLayer, type PhotoMask,
} from "./photoDocument";
import {
  combineCoverage, coverageFromFile, coverageFromImage, layerCoverage, maskFilePaths, maskReadsPixels,
  morphCoverage, rasterizeMask, rasterizeMaskStack, sequenceFramePath, type MaskContext,
} from "./photoMask";
import { buildScopeResources, wantedMasks } from "./photoResources";

const W = 40, H = 30;
const context = (over: Partial<MaskContext> = {}): MaskContext => ({ width: W, height: H, ...over });

/** Every mask shape the app wrote before VFX Phase 2. */
function legacyMasks(): PhotoMask[] {
  const strokes = createMask("strokes");
  strokes.strokes = [
    { points: [[0.1, 0.2], [0.6, 0.5], [0.9, 0.3]], radius: 0.08, hardness: 0.4, erase: false },
    { points: [[0.5, 0.45]], radius: 0.04, hardness: 1, erase: true },
  ];
  return [
    strokes,
    { ...createMask("linear"), shape: { kind: "linear", x1: 0.1, y1: 0.2, x2: 0.9, y2: 0.8 } },
    { ...createMask("radial"), invert: true, feather: 3, shape: { kind: "radial", cx: 0.4, cy: 0.5, radius: 0.3, aspect: 1.4, feather: 0.5 } },
    { ...createMask("off"), enabled: false },
    { ...createMask("empty"), feather: 0 },
  ];
}

const hard = (fill: (x: number, y: number) => boolean) => {
  const out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) out[y * W + x] = fill(x, y) ? 255 : 0;
  return out;
};

describe("mask stack: compatibility", () => {
  it("reads every pre-Phase-2 mask as a one-entry stack, byte for byte", () => {
    const sam2 = hard((x, y) => x > 10 && y < 20);
    for (const mask of legacyMasks()) {
      expect(Array.from(rasterizeMaskStack(mask, context()))).toEqual(Array.from(rasterizeMask(mask, W, H)));
      expect(Array.from(rasterizeMaskStack({ ...mask, sourcePath: "x.png" }, context({ legacyMatte: sam2 }))))
        .toEqual(Array.from(rasterizeMask({ ...mask, sourcePath: "x.png" }, W, H, sam2)));
    }
  });

  it("scales the base feather with the render: the preview matches the save, the save is unchanged", () => {
    const mask = { ...createMask("f"), feather: 6, shape: { kind: "radial" as const, cx: 0.5, cy: 0.5, radius: 0.2, aspect: 1, feather: 0 } };
    expect(Array.from(rasterizeMaskStack(mask, context({ canvasScale: 1 })))).toEqual(Array.from(rasterizeMask(mask, W, H)));
    expect(Array.from(rasterizeMaskStack(mask, context({ canvasScale: 0.5 }))))
      .toEqual(Array.from(rasterizeMask({ ...mask, feather: 3 }, W, H)));
  });

  it("treats an empty stack exactly like no stack", () => {
    for (const mask of legacyMasks()) {
      expect(Array.from(rasterizeMaskStack({ ...mask, stack: [] }, context()))).toEqual(Array.from(rasterizeMask(mask, W, H)));
    }
  });

  it("returns full coverage for a bypassed mask, whatever its stack holds", () => {
    const mask = { ...createMask("off"), enabled: false, stack: [createMaskEntry({ kind: "paint" }, { mode: "intersect" })] };
    expect(rasterizeMaskStack(mask, context()).every((value) => value === 255)).toBe(true);
  });
});

describe("mask stack: modes", () => {
  const left = hard((x) => x < 20);
  const top = hard((_, y) => y < 15);
  const run = (mode: MaskEntry["mode"], opacity = 1) => {
    const out = new Uint8Array(left);
    combineCoverage(out, top, mode, opacity);
    return out;
  };

  it("is boolean on hard mattes", () => {
    const union = run("add"), cut = run("subtract"), both = run("intersect"), either = run("difference");
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const a = x < 20, b = y < 15, at = y * W + x;
        expect(union[at]).toBe(a || b ? 255 : 0);
        expect(cut[at]).toBe(a && !b ? 255 : 0);
        expect(both[at]).toBe(a && b ? 255 : 0);
        expect(either[at]).toBe(a !== b ? 255 : 0);
      }
    }
  });

  it("scales the entry by its opacity before combining", () => {
    const out = run("add", 0.5);
    expect(out[(20) * W + 30]).toBe(0); // neither
    expect(out[(5) * W + 30]).toBe(128); // only the half-strength entry: round(127.5)
    expect(out[(5) * W + 5]).toBe(255);
  });

  it("adds soft coverage as a union, not a sum", () => {
    const a = new Uint8Array([128]), e = new Uint8Array([128]);
    combineCoverage(a, e, "add", 1);
    expect(a[0]).toBe(Math.round(128 + 128 - 128 * 128 / 255));
    expect(a[0]).toBeLessThan(255);
  });

  it("combines entries in order after the base", () => {
    const mask: PhotoMask = {
      ...createMask("stack"), feather: 0,
      shape: { kind: "linear", x1: 0, y1: 0, x2: 0, y2: 0 }, // zero length: covers everything
      stack: [
        createMaskEntry({ kind: "shape", shape: { kind: "radial", cx: 0.5, cy: 0.5, radius: 0.2, aspect: 1, feather: 0 } }, { mode: "subtract" }),
        createMaskEntry({ kind: "paint" }, { mode: "add", strokes: [{ points: [[0.5, 0.5]], radius: 0.05, hardness: 1, erase: false }] }),
      ],
    };
    const out = rasterizeMaskStack(mask, context());
    expect(out[0]).toBe(255); // corner: base only
    expect(out[15 * W + 20]).toBe(255); // centre: cut, then painted back
    expect(out[15 * W + 26]).toBe(0); // inside the hole, outside the dab
  });

  it("skips a disabled entry and an entry whose source cannot be drawn", () => {
    const base = legacyMasks()[1];
    const reference = rasterizeMask(base, W, H);
    for (const entry of [
      createMaskEntry({ kind: "paint" }, { enabled: false, mode: "intersect" }),
      createMaskEntry({ kind: "file", path: "missing.png" }, { mode: "intersect" }),
      createMaskEntry({ kind: "layer", layerId: "nowhere", channel: "alpha" }, { mode: "intersect" }),
    ]) {
      expect(Array.from(rasterizeMaskStack({ ...base, stack: [entry] }, context()))).toEqual(Array.from(reference));
    }
  });
});

describe("mask stack: entry processing", () => {
  it("grows and chokes by a square of the radius", () => {
    const dot = new Uint8Array(W * H);
    dot[15 * W + 20] = 255;
    const grown = new Uint8Array(dot);
    morphCoverage(grown, W, H, 2);
    expect(grown.filter((value) => value === 255).length).toBe(25);
    const choked = new Uint8Array(grown);
    morphCoverage(choked, W, H, -1);
    expect(choked.filter((value) => value === 255).length).toBe(9);
  });

  it("scales entry feather and expand with the render, so preview and save agree", () => {
    const entry = createMaskEntry({ kind: "shape", shape: { kind: "radial", cx: 0.5, cy: 0.5, radius: 0.25, aspect: 1, feather: 0 } },
      { feather: 4, expand: 2 });
    const mask = { ...createMask("m"), feather: 0, stack: [entry] };
    const half = rasterizeMaskStack(mask, context({ canvasScale: 0.5 }));
    const direct = rasterizeMaskStack({ ...mask, stack: [{ ...entry, feather: 2, expand: 1 }] }, context({ canvasScale: 1 }));
    expect(Array.from(half)).toEqual(Array.from(direct));
  });

  it("inverts after feathering, like the base", () => {
    const mask = { ...createMask("m"), feather: 0, stack: [createMaskEntry({ kind: "paint" }, { invert: true })] };
    // An empty paint entry inverted covers everything.
    expect(rasterizeMaskStack(mask, context()).every((value) => value === 255)).toBe(true);
  });
});

describe("matte sources", () => {
  const rgba = (pixels: Array<[number, number, number, number]>) => new Uint8ClampedArray(pixels.flat());

  it("decodes greys exactly with Rec. 709, where the legacy Rec. 601 truncation loses a level", () => {
    const greys = Array.from({ length: 256 }, (_, v) => [v, v, v, 255] as [number, number, number, number]);
    const exact = coverageFromFile(rgba(greys), 256, "auto");
    expect(Array.from(exact)).toEqual(greys.map(([v]) => v));
    // Kept on purpose for SAM 2 files so old masks do not move; documented in coverageFromFile.
    const legacy = coverageFromImage(rgba(greys), 256);
    expect(legacy[1]).toBe(0);
    expect(Array.from(legacy).filter((value, v) => value !== v).length).toBe(65);
  });

  it("picks alpha per file when any pixel is transparent, luma otherwise", () => {
    const cutout = rgba([[200, 10, 10, 255], [200, 10, 10, 0], [10, 10, 10, 128]]);
    expect(Array.from(coverageFromFile(cutout, 3, "auto"))).toEqual([255, 0, 128]);
    const matte = rgba([[200, 200, 200, 255], [0, 0, 0, 255]]);
    expect(Array.from(coverageFromFile(matte, 2, "auto"))).toEqual([200, 0]);
    expect(Array.from(coverageFromFile(cutout, 3, "luma"))[0]).toBe(Math.round(0.2126 * 200 + 0.7152 * 10 + 0.0722 * 10));
  });

  it("names sequence frames in the mask_cache layout and clamps the index", () => {
    const windows = { kind: "sequence" as const, cachePath: "C:\\AI\\cache\\sam2\\abc\\", frameCount: 10 };
    expect(sequenceFramePath(windows, 0)).toBe("C:\\AI\\cache\\sam2\\abc\\masks\\00000.png");
    expect(sequenceFramePath(windows, 42)).toBe("C:\\AI\\cache\\sam2\\abc\\masks\\00009.png");
    expect(sequenceFramePath({ ...windows, cachePath: "/x/y" }, 3)).toBe("/x/y/masks/00003.png");
  });

  it("lists every file a mask needs loaded", () => {
    const mask = { ...createMask("m"), sourcePath: "sam.png", stack: [
      createMaskEntry({ kind: "file", path: "a.png" }),
      createMaskEntry({ kind: "sequence", cachePath: "/c", frameCount: 3 }),
      createMaskEntry({ kind: "layer", layerId: "l", channel: "alpha" }),
    ] };
    expect(maskFilePaths(mask)).toEqual(["sam.png", "a.png", "/c/masks/00000.png"]);
    expect(maskReadsPixels(mask)).toBe(true);
    expect(maskReadsPixels({ ...mask, stack: mask.stack.slice(0, 2) })).toBe(false);
  });

  it("takes a layer's alpha or alpha-weighted luma, times its opacity", () => {
    const pixels = rgba([[255, 255, 255, 255], [255, 255, 255, 128], [0, 0, 0, 255], [255, 0, 0, 0]]);
    expect(Array.from(layerCoverage(pixels, "alpha", 1))).toEqual([255, 128, 255, 0]);
    expect(Array.from(layerCoverage(pixels, "alpha", 0.5))).toEqual([128, 64, 128, 0]);
    expect(Array.from(layerCoverage(pixels, "luma", 1))).toEqual([255, 128, 0, 0]);
  });
});

/* ------------------------------------------------------------------ compositing */

const solid = (id: string, colour: string, over: Partial<PhotoLayer> = {}): PhotoLayer => ({
  ...createGradientLayer({ stops: [{ at: 0, color: colour, opacity: 1 }, { at: 1, color: colour, opacity: 1 }] }),
  id, name: id, ...over,
});

/** A ramp whose alpha runs from nearly 0 at the left to nearly 1 at the right: a layer with every alpha. */
const ramp = (id: string, over: Partial<PhotoLayer> = {}): PhotoLayer => ({
  ...createGradientLayer({ angle: 0, stops: [{ at: 0, color: "#ffffff", opacity: 0 }, { at: 1, color: "#ffffff", opacity: 1 }] }),
  id, name: id, ...over,
});

function compose(document: PhotoDocument, over: Partial<ComposeInput> = {}) {
  return composeDocument({
    document, width: W, height: H, sources: new Map(),
    resourcesFor: (scope: ComposeScope, pixels: Uint8ClampedArray) =>
      buildScopeResources({ document, files: () => undefined }, scope, W, H, pixels),
    ...over,
  });
}

const doc = (layers: PhotoLayer[], over: Partial<PhotoDocument> = {}): PhotoDocument => ({
  ...createPhotoDocument("a", "a.png"), nodes: [], masks: [], layers, canvasWidth: W, canvasHeight: H, ...over,
});

const at = (pixels: Uint8ClampedArray, x: number, y: number) => Array.from(pixels.subarray((y * W + x) * 4, (y * W + x) * 4 + 4));

describe("layer masks, clipping and track mattes", () => {
  it("cuts a layer's alpha by its layer mask", () => {
    const mask = { ...createMask("half"), feather: 0, strokes: [], stack: [
      createMaskEntry({ kind: "shape", shape: { kind: "linear", x1: 0, y1: 0, x2: 0, y2: 0 } }), // everything
      createMaskEntry({ kind: "paint" }, { mode: "intersect", strokes: [{ points: [[0.25, 0.5]], radius: 0.3, hardness: 1, erase: false }] }),
    ] };
    const red = solid("red", "#ff0000", { masks: [mask], maskId: mask.id });
    const out = compose(doc([solid("blue", "#0000ff"), red])).pixels;
    expect(at(out, 10, 15)).toEqual([255, 0, 0, 255]);
    expect(at(out, 35, 15)).toEqual([0, 0, 255, 255]);
  });

  it("clips consecutive layers to the base below them, and hides them with it", () => {
    const layers = [solid("plate", "#000000"), ramp("base"), solid("c1", "#ff0000", { clip: true }), solid("c2", "#00ff00", { clip: true, opacity: 0.5 })];
    expect([...clipBases(layers).entries()].map(([id, base]) => [id, base.id])).toEqual([["c1", "base"], ["c2", "base"]]);
    const out = compose(doc(layers)).pixels;
    expect(Math.max(...at(out, 0, 5).slice(0, 3))).toBeLessThan(8); // base (nearly) transparent: (nearly) nothing clipped shows
    const right = at(out, W - 1, 5); // base opaque: red, then half green over it
    expect(right[0]).toBeGreaterThan(100);
    expect(right[1]).toBeGreaterThan(100);
    const hidden = compose(doc([layers[0], { ...layers[1], visible: false }, layers[2]])).pixels;
    expect(at(hidden, W - 1, 5)).toEqual([0, 0, 0, 255]);
  });

  it("ignores clip on the bottom layer", () => {
    const out = compose(doc([solid("red", "#ff0000", { clip: true })])).pixels;
    expect(at(out, 5, 5)).toEqual([255, 0, 0, 255]);
  });

  it("matches the four track-matte types, from a hidden matte layer after its own nodes", () => {
    const matte = ramp("matte", { visible: false });
    const cases: Array<["alpha" | "luma", boolean]> = [["alpha", false], ["alpha", true], ["luma", false], ["luma", true]];
    for (const [channel, invert] of cases) {
      const track = { ...createMask("track"), feather: 0, stack: [createMaskEntry({ kind: "layer", layerId: "matte", channel }, { invert })] };
      const fill = solid("fill", "#ffffff", { masks: [track], maskId: track.id });
      const out = compose(doc([solid("plate", "#000000"), matte, fill])).pixels;
      const leftValue = at(out, 0, 5)[0], rightValue = at(out, W - 1, 5)[0];
      expect(invert ? leftValue : rightValue).toBeGreaterThan(245);
      expect(invert ? rightValue : leftValue).toBeLessThan(10);
    }
    // The matte is read after its own node stack: a grade that darkens it to black kills a luma matte.
    const track = { ...createMask("track"), feather: 0, stack: [createMaskEntry({ kind: "layer", layerId: "matte", channel: "luma" })] };
    const dark = createGradeNode();
    dark.adjustments = { ...dark.adjustments, exposure: -20 };
    const out = compose(doc([solid("plate", "#000000"), { ...matte, nodes: [dark] }, solid("fill", "#ffffff", { masks: [track], maskId: track.id })])).pixels;
    expect(at(out, W - 1, 5)[0]).toBeLessThan(3);
  });

  it("draws a visible matte layer as well as using it", () => {
    const track = { ...createMask("track"), feather: 0, stack: [createMaskEntry({ kind: "layer", layerId: "matte", channel: "alpha" })] };
    const out = compose(doc([solid("plate", "#000000"), solid("fill", "#ff0000", { masks: [track], maskId: track.id }), ramp("matte")])).pixels;
    const top = at(out, W - 1, 5); // the matte layer itself, on top of the red it lets through
    expect(top[0]).toBe(255);
    expect(top[1]).toBeGreaterThan(245);
  });

  it("reports a matte cycle once and still renders", () => {
    const a = { ...createMask("a"), feather: 0, stack: [createMaskEntry({ kind: "layer", layerId: "B", channel: "alpha" })] };
    const b = { ...createMask("b"), feather: 0, stack: [createMaskEntry({ kind: "layer", layerId: "A", channel: "alpha" })] };
    const result = compose(doc([solid("A", "#ff0000", { masks: [a], maskId: a.id }), solid("B", "#00ff00", { masks: [b], maskId: b.id })]));
    expect(result.report.skipped).toEqual([{ nodeId: "A", reason: MATTE_CYCLE }]);
    expect(referencedLayers(doc([solid("A", "#ff0000", { masks: [a] }), solid("B", "#00ff00", { masks: [b] })]))).toEqual(new Set(["B", "A"]));
  });

  it("captures the combined coverage for the overlay, including layer entries", () => {
    const track = { ...createMask("track"), feather: 0, stack: [createMaskEntry({ kind: "layer", layerId: "matte", channel: "alpha" })] };
    const document = doc([solid("plate", "#000000"), ramp("matte", { visible: false })], { masks: [track], nodes: [{ ...createGradeNode(), maskId: track.id }] });
    const coverage = compose(document, { captureMask: track.id }).maskCoverage!;
    expect(coverage[5 * W]).toBeLessThan(8);
    expect(coverage[5 * W + W - 1]).toBeGreaterThan(247);
    expect(compose(document, { captureMask: track.id, capture: COMPOSITE_SCOPE }).maskCoverage).toEqual(coverage);
  });

  it("only rasterises the masks a scope reads", () => {
    const used = createMask("used"), unused = createMask("unused"), layerMask = createMask("layer");
    const node = { ...createGradeNode(), maskId: used.id };
    const layer = solid("x", "#ffffff", { maskId: layerMask.id });
    const scope: ComposeScope = { nodes: [node, { ...createGradeNode(), enabled: false, maskId: unused.id }], masks: [used, unused, layerMask], layer };
    expect(wantedMasks(scope).map((mask) => mask.name)).toEqual(["used", "layer"]);
  });

  it("cuts alpha as a * (mask * base * opacity), rounded once", () => {
    const pixels = new Uint8ClampedArray([10, 20, 30, 200]);
    applyAlphaCut(pixels, new Uint8Array([128]), new Uint8ClampedArray([0, 0, 0, 255]), 0.5);
    expect(Array.from(pixels)).toEqual([10, 20, 30, Math.round(200 * (128 / 255) * 0.5)]);
  });
});
