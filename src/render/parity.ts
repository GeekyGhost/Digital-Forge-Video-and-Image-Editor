/**
 * Parity harness: every fixture document rendered by the old CPU `composeDocument` (the golden) and
 * by the new `PhotoRenderer`, compared pixel by pixel.
 *
 * Runs in a browser: in the cloud container through headless Chromium (SwiftShader WebGL2), and on
 * a real GPU machine from the spike window ("Phase 1 parity"), because a real GPU can round
 * differently from SwiftShader. Source pictures are generated from a seeded generator, so both
 * places render identical inputs without shipping test images.
 *
 * Tolerance: max 1/255 per channel on alpha and on alpha-weighted colour. Colour is also reported
 * straight (unweighted), where nearly transparent pixels can legitimately differ more: the canvas
 * stores them premultiplied in 8 bits, so their straight colour is already quantised coarsely.
 */

import { composeDocument, croppedRect, layerMatrix, type ComposeInput, type ComposeScope, type LayerSource } from "../domain/photoCompose";
import {
  createGradeNode, createGradientLayer, createGradientMapNode, createLayer, createMask, createPhotoDocument,
  createRelightNode, createRemoveNode, createRetouchNode, createShapeLayer, createTextLayer, createWarpNode,
  freshAdjustments, type PhotoDocument, type PhotoLayer, type PhotoMask, type PhotoNode,
  createLumaKeyNode, createMaskEntry, createUnmultiplyNode, MASK_MODES, type MaskEntry, type MaskSource,
  createAlphaCurveNode, createMatrixRainLayer, createMediaLayer, type LayerMedia,
  createDepthOcclusionMask, createEmitterLayer, createTrailNode, type PointTrack, type TrackSample,
  createEnvelopeNode, createPathLayer,
} from "../domain/photoDocument";
import { createEnvelope } from "../domain/photoEnvelope";
import { NEUTRAL_GRADIENT } from "../domain/photoGradient";
import { NEUTRAL_PATH, type PathBlock, type Subpath } from "../domain/photoPath";
import { NEUTRAL_CLIP_EFFECTS, type ClipEffectsParams } from "../domain/photoClipEffects";
import {
  programmeDocument, transitionFrame, type Programme, type ProgrammeClip, type ProgrammeMedia, type ProgrammeOverlay,
} from "../domain/programme";
import type { ProgrammeMatte } from "../domain/clipMatte";
import { effectsFor } from "../domain/effects";
import { DEFAULT_OVERLAY } from "../domain/project";
import { DEFAULT_MASK_VFX_RECIPE, MASK_VFX_RECIPES } from "../domain/maskEffects";
import type { MaskVfxRecipeV1 } from "../domain/maskVfx";
import type { GraphicOverlay } from "../domain/types";
import { attachedLayer } from "../domain/photoTrack";
import { compositionRate } from "../domain/photoTime";
import { mediaFrameKey } from "../domain/photoTime";
import { BLEND_MODES, type BlendMode } from "../domain/photoBlend";
import type { PhotoAdjustments } from "../domain/photoEdit";
import { invert } from "../domain/photoGeometry";
import { emptyResources } from "../domain/photoRender";
import { buildScopeResources } from "../domain/photoResources";
import { PhotoRenderer, type RendererOptions } from "./photoRenderer";

type Difference = {
  /** Worst difference, 0..255. */
  maxAlpha: number;
  maxPremultiplied: number;
  maxStraight: number;
  /** Pixels whose alpha-weighted colour or alpha differs by more than 1. */
  over1: number;
  meanAbs: number;
  /** Where the worst pixel is, and both values there, for diagnosis. */
  worst?: { x: number; y: number; old: number[]; gpu: number[] };
};

/**
 * - exact: identical bytes.
 * - pass: within 1/255 on alpha and alpha-weighted colour.
 * - boundary: a handful of pixels (at most 0.5%) further off, which is what float32 (GPU) against
 *   float64 (JS) does where a value lands on a table index or rounding boundary. Mean error stays
 *   under 0.05 levels; a logic error shows up as widespread differences instead.
 * - fail: anything else.
 */
export type Verdict = "exact" | "pass" | "boundary" | "fail";

export type FixtureResult = {
  name: string;
  size: string;
  verdict: Verdict;
  /**
   * The renderer against the old compositor with layer placement done by the canvas in both
   * (`placement: "canvas"`): every stage after placement. The verdict is this.
   */
  pipeline: Difference;
  /**
   * The same with the GPU's own placement. Differs from `pipeline` only where a layer is scaled or
   * translated by a fraction: Chrome's software canvas resamples those with 4-bit weights (Skia's
   * legacy bitmap sampler), the shader with exact bilinear. See `placementAccuracy` for the shader
   * held against a float64 reference instead.
   */
  gpuPlacement?: Difference;
  /** VFX Phase 3: pixels that differ from the bottom layer rendered alone (see Fixture.changed). */
  changed?: number;
  /** VFX Phase 2: the captured mask's coverage, old against GPU (max difference, pixels over 1). */
  mask?: { maxDiff: number; over1: number; missing?: string };
  cpu: string[];
  oldMs: number;
  gpuMs: number;
  skippedMatch: boolean;
  error?: string;
};

export type PlacementResult = { name: string; interiorPixels: number; maxError: number; over1: number };

export type ParityReport = {
  renderer: string;
  fixtures: FixtureResult[];
  counts: Record<Verdict, number>;
  placement: PlacementResult[];
  performance?: PerformanceResult;
};

export type PerformanceResult = {
  size: string;
  frames: number;
  oldMsPerFrame: number;
  gpuMsPerFrame: number;
  /** VFX Phase 2: the same drag through a masked grade on a track-matted layer. */
  maskedOldMsPerFrame?: number;
  maskedGpuMsPerFrame?: number;
  note: string;
};

/* ------------------------------------------------------------ inputs */

type Loaded = { canvas: HTMLCanvasElement; width: number; height: number };

function rng(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function canvasFrom(width: number, height: number, paint: (x: number, y: number, random: () => number) => [number, number, number, number], seed: number): Loaded {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  const image = context.createImageData(width, height);
  const random = rng(seed);
  for (let y = 0, at = 0; y < height; y++) {
    for (let x = 0; x < width; x++, at += 4) {
      const [r, g, b, a] = paint(x, y, random);
      image.data[at] = r; image.data[at + 1] = g; image.data[at + 2] = b; image.data[at + 3] = a;
    }
  }
  context.putImageData(image, 0, 0);
  return { canvas, width, height };
}

/** A plate with ramps, saturated patches for the HSL bands, fine noise and hard edges. */
const plate = () => canvasFrom(480, 320, (x, y, random) => {
  const n = (random() - 0.5) * 18;
  let r = x / 479 * 255 + n, g = y / 319 * 255 + n, b = (1 - x / 479) * 200 + 30 + n;
  const cell = ((x / 60) | 0) + ((y / 60) | 0) * 8;
  if (y > 200 && y < 260) {
    const hue = (x / 480) * 360;
    [r, g, b] = hsv(hue, 0.85, 0.35 + 0.6 * ((cell % 3) / 2));
  }
  if (x > 400 && y < 60) { r = g = b = (x & 8) ? 250 : 5; }
  return [r, g, b, 255];
}, 1);

/** A cutout with a feathered edge, a hole and a colour ramp: every alpha value appears. */
const cutout = () => canvasFrom(200, 240, (x, y) => {
  const dx = (x - 100) / 90, dy = (y - 120) / 110;
  const d = Math.hypot(dx, dy);
  let a = Math.max(0, Math.min(1, (1 - d) * 6));
  if (Math.hypot(x - 100, y - 90) < 18) a = 0.25;
  return [230 - y * 0.6, 60 + x * 0.8, 40 + (x ^ y) % 90, Math.round(a * 255)];
}, 2);

const matte = () => canvasFrom(480, 320, (x, y) => {
  const v = Math.max(0, Math.min(255, 255 - Math.hypot(x - 240, y - 150) * 1.6));
  return [v, v, v, 255];
}, 3);

const depth = () => canvasFrom(480, 320, (x, y) => {
  const v = 60 + 120 * Math.exp(-((x - 200) ** 2 + (y - 160) ** 2) / 9000) + y * 0.1;
  return [v, v, v, 255];
}, 4);

const fill = () => canvasFrom(480, 320, (x, y, random) => [120 + random() * 20, 110 + (y & 15), 90 + (x & 15), 255], 5);

function hsv(h: number, s: number, v: number): [number, number, number] {
  const c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c;
  const [r, g, b] = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x] : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

/** Stock footage on black: a glow with saturated flames and pure-black surroundings. */
const onBlack = () => canvasFrom(260, 220, (x, y, random) => {
  const d = Math.hypot(x - 130, (y - 120) * 1.3) / 100;
  const heat = Math.max(0, 1 - d) ** 1.4 * (0.8 + random() * 0.2);
  if (heat <= 0.02) return [0, 0, 0, 255];
  return [255 * Math.min(1, heat * 1.6), 200 * heat * heat, 60 * heat ** 3, 255];
}, 6);

/** A white-on-black matte with an alpha hole, to test the per-file "auto" channel rule. */
const alphaMatte = () => canvasFrom(480, 320, (x, y) => {
  const inside = Math.abs(x - 300) < 110 && Math.abs(y - 170) < 90;
  return [255, 255, 255, inside ? Math.round(255 * Math.min(1, (110 - Math.abs(x - 300)) / 40)) : 0];
}, 7);

const PATHS = {
  plate: "fixture:plate", cutout: "fixture:cutout", matte: "fixture:matte", depth: "fixture:depth", fill: "fixture:fill",
  onBlack: "fixture:onblack", alphaMatte: "fixture:alphamatte", sequence: "fixture:seq",
};
/** Frame 0 of the sequence fixture, in the mask_cache.py layout. */
const SEQUENCE_FRAME = "fixture:seq/masks/00000.png";

/* VFX Phase 3: moving sources. Every frame differs, so an off-by-one frame is a visible failure. */
const ELEMENT_FRAMES = 24;
const ELEMENT: LayerMedia = {
  kind: "frames", path: "fixture:elem", pattern: "frame_%04d.png", frameCount: ELEMENT_FRAMES, fps: 24,
  width: 200, height: 150, start: 0, inPoint: 0, outPoint: ELEMENT_FRAMES, loop: false, crossfade: 0,
};
const CLIP: LayerMedia = {
  kind: "video", path: "fixture:clip.mp4", frameCount: 30, fps: 24, width: 240, height: 160,
  start: 0, inPoint: 0, outPoint: 30, loop: false, crossfade: 0,
};

/** Smoke-like: a soft blob that drifts and breathes with the frame, colour varying across it. */
const elementFrame = (frame: number) => canvasFrom(200, 150, (x, y) => {
  const cx = 60 + frame * 4, cy = 75 + Math.sin(frame / 3) * 20;
  const d = Math.hypot(x - cx, (y - cy) * 1.2) / (40 + frame);
  const a = Math.max(0, Math.min(1, 1 - d)) ** 1.5;
  return [200 + (frame * 3) % 55, 120 + (x & 31), 60 + frame * 5, Math.round(a * 255)];
}, 20 + frame);

/** An opaque "video" frame with the frame number in a bar, like the Phase 0 index strip. */
const clipFrame = (frame: number) => canvasFrom(240, 160, (x, y) => {
  if (y < 12) return [(x >> 3) === frame ? 255 : 0, 0, 0, 255];
  return [(x + frame * 7) & 255, (y * 2 + frame * 3) & 255, 128, 255];
}, 60 + frame);

/* VFX Phase 4: a synthetic point track on the clip, and a depth sequence for it. */
const TRACK_ID = "fixture-track";
const TRACKED_CLIP_ID = "clip-tracked";
const DEPTH_SEQUENCE = "fixture:depthseq";
/** Sub-pixel on purpose: x 60.13 + 3.37 t, y 70 + 9 sin(t / 4); unseen at t = 11 (held). */
const fixtureTrack = (): PointTrack => ({
  id: TRACK_ID, name: "Tip", layerId: TRACKED_CLIP_ID, first: 0, source: "manual",
  samples: Array.from({ length: 30 }, (_, t) => [60.13 + 3.37 * t, 70 + 9 * Math.sin(t / 4), t === 11 ? 0 : 1] as TrackSample),
});
/** Disparity: a far ramp, and a near "hand" (255) that sweeps across the clip with the frame. */
const depthFrame = (frame: number) => canvasFrom(240, 160, (x, y) => {
  const far = 40 + x * 0.3 + y * 0.2;
  const hand = Math.max(0, 1 - Math.hypot(x - 40 - frame * 6, y - 90) / 38);
  const v = Math.min(255, far + hand * 230);
  return [v, v, v, 255];
}, 120 + frame);

const sequenceFrame = (frame: number) => canvasFrom(480, 320, (x, y) => {
  const v = Math.max(0, Math.min(255, 255 - Math.hypot(x - 120 - frame * 9, y - 160) * 2));
  return [v, v, v, 255];
}, 90 + frame);

/* ------------------------------------------------------------ fixtures */

export type Fixture = {
  name: string; document: PhotoDocument; scale?: number; placement?: RendererOptions["placement"];
  /** VFX Phase 3: the composition frame to render. */
  frame?: number;
  /** Also report how many pixels differ from the bottom layer alone (proof something drew). */
  changed?: boolean;
  /**
   * VFX Phase 4: count `changed` against this document instead of the bottom layer alone -- the same
   * document without the thing under test (the attachment, the trail node, the emitter, the mask), so
   * the count proves that thing did something, not that the plate is different from the clip.
   */
  versus?: PhotoDocument;
  /** Compare this mask's combined coverage too (the overlay's `captureMask`). */
  captureMask?: string;
};

const plateLayer = (over: Partial<PhotoLayer> = {}) =>
  createLayer({ id: "plate", name: "Plate", path: PATHS.plate }, over);
const cutoutLayer = (over: Partial<PhotoLayer> = {}) =>
  createLayer({ id: "cutout", name: "Cutout", path: PATHS.cutout }, over);

function doc(layers: PhotoLayer[], over: Partial<PhotoDocument> = {}): PhotoDocument {
  return { ...createPhotoDocument("plate", PATHS.plate), nodes: [], masks: [], layers, ...over };
}

function gradeWith(over: Partial<PhotoAdjustments>, maskId?: string): PhotoNode {
  const node = createGradeNode();
  node.adjustments = { ...freshAdjustments(), ...over };
  if (maskId) node.maskId = maskId;
  return node;
}

function graded(name: string, over: Partial<PhotoAdjustments>): Fixture {
  return { name, document: doc([plateLayer({ nodes: [gradeWith(over)] })]) };
}

function maskFixture(name: string, mask: PhotoMask, scale?: number): Fixture {
  return {
    name: `mask: ${name}`, captureMask: mask.id, scale,
    document: doc([plateLayer({ masks: [mask], nodes: [gradeWith({ exposure: 1.2, saturation: 40 }, mask.id)] })]),
  };
}

const radialBase = (name: string, stack: MaskEntry[]): PhotoMask => ({
  ...createMask(name), feather: 3, stack,
  shape: { kind: "radial", cx: 0.45, cy: 0.5, radius: 0.28, aspect: 1.3, feather: 0.4 },
});

/** VFX Phase 3: media and generator layers at several frames. */
function phase3Fixtures(): Fixture[] {
  const list: Fixture[] = [];
  const element = (over: Partial<LayerMedia> = {}, layer: Partial<PhotoLayer> = {}) =>
    createMediaLayer("Smoke", { ...ELEMENT, ...over }, { blendMode: "screen", x: 0.45, y: 0.5, scale: 1.3, ...layer });
  for (const frame of [0, 5, 23]) {
    list.push({ name: `element: frame ${frame}`, frame, document: doc([plateLayer(), element()]) });
  }
  list.push({ name: "element: before its start (not drawn)", frame: 2, document: doc([plateLayer(), element({ start: 3 })]) });
  list.push({ name: "element: after its out point (not drawn)", frame: 20, document: doc([plateLayer(), element({ inPoint: 4, outPoint: 12 })]) });
  for (const frame of [0, 3, 6, 20, 16 + 1]) {
    list.push({ name: `element loop, crossfade 4: frame ${frame}`, frame, document: doc([plateLayer(), element({ inPoint: 2, outPoint: 22, loop: true, crossfade: 4 }, { rotation: 11 })]) });
  }
  list.push({ name: "element: alpha curve, grade, normal blend", frame: 9, document: doc([plateLayer(), element({}, {
    blendMode: "normal", nodes: [createAlphaCurveNode(), gradeWith({ exposure: 0.4, temperature: 30 })],
  })]) });
  list.push({ name: "element at 12 fps in a 24 fps composition: frame 7", frame: 7, document: doc([plateLayer(), element({ fps: 12 })], { frameRate: 24 }) });
  for (const frame of [0, 13, 29]) {
    list.push({ name: `video layer: frame ${frame}, scaled, rotated, graded`, frame, document: doc([plateLayer(), createMediaLayer("Clip", CLIP, {
      scale: 0.9, rotation: -6, x: 0.52, nodes: [gradeWith({ contrast: 25 })],
    })]) });
  }
  const text = createTextLayer({ width: 480, height: 320 }, { content: "CLIPPED" });
  list.push({ name: "text clipped to an element, frame 11", frame: 11, document: doc([plateLayer(), element({}, { blendMode: "normal" }), { ...text, clip: true }]) });
  const matte = { ...createMask("element matte"), feather: 0, stack: [createMaskEntry({ kind: "layer", layerId: "elem-matte", channel: "alpha" })] };
  const matteElement = { ...element({ start: 10 }, { visible: false }), id: "elem-matte" };
  for (const frame of [4, 14]) {
    list.push({ name: `track matte from an element (off screen at 4): frame ${frame}`, frame, captureMask: matte.id, document: doc([
      plateLayer(), matteElement, createLayer({ id: "fill-e", name: "Fill", path: PATHS.fill }, { masks: [matte], maskId: matte.id }),
    ]) });
  }
  const tracked = { ...createMask("tracked"), feather: 2, stack: [createMaskEntry({ kind: "sequence", cachePath: PATHS.sequence, frameCount: 24 })] };
  list.push({ name: "sequence mask follows the element's source frame (7)", frame: 10, captureMask: tracked.id, document: doc([plateLayer(),
    element({ start: 3 }, { masks: [tracked], nodes: [gradeWith({ exposure: 1, saturation: 0 }, tracked.id)] })]) });
  const plateTracked = { ...createMask("plate tracked"), feather: 0, stack: [createMaskEntry({ kind: "sequence", cachePath: PATHS.sequence, frameCount: 24 })] };
  list.push({ name: "sequence mask on a still layer follows the composition frame (13)", frame: 13, captureMask: plateTracked.id,
    document: doc([plateLayer({ masks: [plateTracked], nodes: [gradeWith({ exposure: -1 }, plateTracked.id)] }), element({ start: 0 })]) });
  for (const frame of [0, 17, 239]) {
    list.push({ name: `matrix rain: frame ${frame}`, frame, document: doc([plateLayer(), createMatrixRainLayer({ seed: 42, cellSize: 14 })]) });
  }
  list.push({ name: "matrix rain at proxy 0.5, mirrored off, frame 33", frame: 33, scale: 0.5, document: doc([plateLayer(), createMatrixRainLayer({ seed: 9, mirror: false, density: 1 })]) });
  list.push({ name: "matrix rain masked by an element's luma, frame 8", frame: 8, document: (() => {
    const m = { ...createMask("m"), id: "rain-luma", feather: 0, stack: [createMaskEntry({ kind: "layer", layerId: "rain-matte", channel: "luma" })] };
    return doc([plateLayer(), { ...element({}, { visible: false }), id: "rain-matte" }, { ...createMatrixRainLayer({ seed: 3, density: 1, trail: 40 }), masks: [m], maskId: m.id }]);
  })(), captureMask: "rain-luma" });
  return list;
}

/** VFX Phase 4: attach-to-track, the trail, the emitter and depth occlusion, at several frames. */
function phase4Fixtures(): Fixture[] {
  const list: Fixture[] = [];
  const clip = (over: Partial<PhotoLayer> = {}, media: Partial<LayerMedia> = {}) => ({
    ...createMediaLayer("Clip", { ...CLIP, ...media }, { scale: 1.6, rotation: 4, x: 0.49, y: 0.52, ...over }), id: TRACKED_CLIP_ID,
  });
  const smoke = (over: Partial<PhotoLayer> = {}) =>
    createMediaLayer("Smoke", ELEMENT, { blendMode: "screen", x: 0.41, y: 0.37, scale: 0.9, ...over });
  const withTrack = (layers: PhotoLayer[], over: Partial<PhotoDocument> = {}) => doc(layers, { tracks: [fixtureTrack()], ...over });
  /** The same document with the top layer changed (`strip`), for `versus`. */
  const without = (document: PhotoDocument, strip: (layer: PhotoLayer) => PhotoLayer | undefined): PhotoDocument => {
    const layers = document.layers!;
    const top = strip(layers[layers.length - 1]);
    return { ...document, layers: top ? [...layers.slice(0, -1), top] : layers.slice(0, -1) };
  };
  const unattached = (layer: PhotoLayer) => ({ ...layer, attach: undefined });
  const untrailed = (layer: PhotoLayer) => ({ ...layer, nodes: layer.nodes.filter((node) => node.kind !== "trail") });
  const unmasked = (layer: PhotoLayer) => ({ ...layer, maskId: undefined });

  const push = (fixture: Fixture, strip: (layer: PhotoLayer) => PhotoLayer | undefined) =>
    list.push({ ...fixture, versus: without(fixture.document, strip) });

  // Attach: `changed` counts against the same layer unattached, so it is 0 at the anchor frame.
  for (const frame of [0, 5, 17]) {
    push({ name: `attach: element follows the clip's track, frame ${frame}`, frame, document: withTrack([
      plateLayer(), clip(), smoke({ attach: { trackId: TRACK_ID, anchorFrame: 0 } }),
    ]) }, unattached);
  }
  push({ name: "attach: lag 3, held across an unseen sample, frame 14", frame: 14, document: withTrack([
    plateLayer(), clip(), smoke({ attach: { trackId: TRACK_ID, anchorFrame: 2, lag: 3 }, rotation: -12 }),
  ]) }, unattached);
  // A file matte is placed through its layer, so it travels with the attachment.
  const attachedMask = { ...createMask("attached file matte"), feather: 3, stack: [createMaskEntry({ kind: "file", path: PATHS.matte, channel: "luma" })] };
  push({ name: "attach: a cutout with a file-matte layer mask follows, frame 21", frame: 21, captureMask: attachedMask.id, document: withTrack([
    plateLayer(), clip({ visible: false }), cutoutLayer({ scale: 0.6, masks: [attachedMask], maskId: attachedMask.id, attach: { trackId: TRACK_ID, anchorFrame: 0 } }),
  ]) }, unattached);
  push({ name: "attach: missing track leaves the layer where it is (changed 0)", frame: 9, document: withTrack([
    plateLayer(), clip(), smoke({ attach: { trackId: "no-such-track", anchorFrame: 0 } }),
  ]) }, unattached);

  // Trail: `changed` counts against the same layer without its trail node.
  const trail = (over: Partial<ReturnType<typeof createTrailNode>> = {}) => ({ ...createTrailNode(TRACK_ID), maxLag: 8, strength: 1.5, ...over });
  for (const frame of [5, 17]) {
    push({ name: `trail: attached element, lag 8, frame ${frame}`, frame, document: withTrack([
      plateLayer(), clip(), smoke({ attach: { trackId: TRACK_ID, anchorFrame: 0 }, nodes: [trail()] }),
    ]) }, untrailed);
  }
  push({ name: "trail: rotated cutout, strength 3, then graded, frame 17", frame: 17, document: withTrack([
    plateLayer(), clip({ visible: false }), cutoutLayer({ rotation: 25, scale: 0.8, nodes: [trail({ strength: 3 }), gradeWith({ exposure: 0.3 })] }),
  ]) }, untrailed);
  push({ name: "trail at proxy 0.5, frame 17", frame: 17, scale: 0.5, document: withTrack([
    plateLayer(), clip(), smoke({ attach: { trackId: TRACK_ID, anchorFrame: 0 }, nodes: [trail()] }),
  ]) }, untrailed);
  push({ name: "trail: missing track is reported (changed 0)", frame: 4, document: withTrack([
    plateLayer(), smoke({ nodes: [trail({ trackId: "no-such-track" })] }),
  ]) }, untrailed);

  // Emitter: `changed` counts against the document without the emitter.
  const emitter = (over: Parameters<typeof createEmitterLayer>[1] = {}, layer: Partial<PhotoLayer> = {}) => ({
    ...createEmitterLayer(ELEMENT, {
      seed: 1234, rate: 1.5, lifetime: 8, trackId: TRACK_ID, velocityX: 0.004, velocityY: -0.012, spread: 0.003,
      scaleStart: 0.35, scaleEnd: 0.9, opacityStart: 1, opacityEnd: 0.2, spin: 25, ...over,
    }), ...layer,
  });
  for (const frame of [0, 9, 17]) {
    push({ name: `emitter on the track: frame ${frame}`, frame, document: withTrack([plateLayer(), clip(), emitter()]) }, () => undefined);
  }
  push({ name: "emitter at a point, screen, graded, proxy 0.5: frame 12", frame: 12, scale: 0.5, document: withTrack([
    plateLayer(), emitter({ trackId: undefined, x: 0.3, y: 0.8, rate: 0.5, lifetime: 12 }, { blendMode: "screen", nodes: [gradeWith({ temperature: 40 })] }),
  ]) }, () => undefined);

  // Depth occlusion: `changed` counts against the same layer with no layer mask.
  const depthClip = (over: Partial<PhotoLayer> = {}) => clip(over, { depth: { cachePath: DEPTH_SEQUENCE, frameCount: CLIP.frameCount } });
  for (const frame of [0, 5, 17]) {
    const occlusion = createDepthOcclusionMask(TRACKED_CLIP_ID, 50, 10);
    push({ name: `depth occlusion: the clip's near hand hides the element, frame ${frame}`, frame, captureMask: occlusion.id, document: withTrack([
      plateLayer(), depthClip(), smoke({ blendMode: "normal", scale: 1.8, x: 0.45, y: 0.55, masks: [occlusion], maskId: occlusion.id }),
    ]) }, unmasked);
  }
  const hard = { ...createDepthOcclusionMask(TRACKED_CLIP_ID, 62, 0), feather: 2 };
  push({ name: "depth occlusion: hard plane, feathered, attached + trail, frame 17", frame: 17, captureMask: hard.id, document: withTrack([
    plateLayer(), depthClip(), smoke({ blendMode: "normal", scale: 1.5, masks: [hard], maskId: hard.id, attach: { trackId: TRACK_ID, anchorFrame: 0 }, nodes: [trail()] }),
  ]) }, unmasked);
  const noDepth = createDepthOcclusionMask(TRACKED_CLIP_ID, 50, 10);
  push({ name: "depth occlusion: a clip without depth (entry skipped, changed 0)", frame: 5, captureMask: noDepth.id, document: withTrack([
    plateLayer(), clip(), smoke({ blendMode: "normal", masks: [noDepth], maskId: noDepth.id }),
  ]) }, unmasked);
  return list;
}

/** VFX Phase 2: stacks, layer masks, clipping, track mattes, imported mattes, keys. */
function phase2Fixtures(): Fixture[] {
  const list: Fixture[] = [];
  const linear: MaskSource = { kind: "shape", shape: { kind: "linear", x1: 0.2, y1: 0.1, x2: 0.8, y2: 0.9 } };
  for (const mode of MASK_MODES) {
    list.push(maskFixture(`stack ${mode}, opacity 0.7`, radialBase(mode, [createMaskEntry(linear, { mode, opacity: 0.7 })])));
  }
  list.push(maskFixture("stack: expand 3 + feather 4, choke 5 + invert", radialBase("morph", [
    createMaskEntry({ kind: "shape", shape: { kind: "radial", cx: 0.7, cy: 0.3, radius: 0.12, aspect: 1, feather: 0 } }, { expand: 3, feather: 4 }),
    createMaskEntry({ kind: "shape", shape: { kind: "radial", cx: 0.3, cy: 0.7, radius: 0.15, aspect: 0.8, feather: 0.1 } }, { mode: "subtract", expand: -5, invert: true }),
  ])));
  list.push(maskFixture("stack: paint entry with strokes, colour entry intersect", radialBase("paint", [
    createMaskEntry({ kind: "paint" }, { strokes: [
      { points: [[0.1, 0.1], [0.5, 0.2], [0.9, 0.15]], radius: 0.05, hardness: 0.6, erase: false },
      { points: [[0.5, 0.18]], radius: 0.02, hardness: 1, erase: true },
    ] }),
    createMaskEntry({ kind: "shape", shape: { kind: "colour", r: 200, g: 120, b: 60, tolerance: 45, softness: 60 } }, { mode: "intersect", opacity: 0.9 }),
  ])));
  list.push(maskFixture("stack: disabled entry is skipped", { ...radialBase("off", [createMaskEntry(linear, { enabled: false, mode: "subtract" })]) }));
  list.push(maskFixture("whole stack bypassed", { ...radialBase("bypassed", [createMaskEntry(linear)]), enabled: false }));
  list.push(maskFixture("stack at proxy 0.5 (entry feather/expand scale)", radialBase("proxy", [
    createMaskEntry(linear, { mode: "intersect", feather: 8, expand: 4 }),
  ]), 0.5));
  list.push(maskFixture("imported matte: luma (709)", { ...createMask("file"), stack: [
    createMaskEntry({ kind: "file", path: PATHS.matte, channel: "luma" }, { feather: 2 }),
  ] }));
  list.push(maskFixture("imported matte: auto picks alpha", { ...createMask("file alpha"), feather: 0, stack: [
    createMaskEntry({ kind: "file", path: PATHS.alphaMatte }),
  ] }));
  list.push(maskFixture("imported sequence, frame 0", { ...createMask("sequence"), feather: 0, stack: [
    createMaskEntry({ kind: "sequence", cachePath: PATHS.sequence, frameCount: 24 }, { mode: "add" }),
  ] }));
  list.push(maskFixture("imported matte not loaded (entry skipped)", radialBase("missing", [
    createMaskEntry({ kind: "file", path: "fixture:never-loaded" }, { mode: "intersect" }),
  ])));

  // Layer masks: cut the layer's alpha after its nodes, before the edge group and the blend.
  const layerMask = { ...createMask("layer mask"), feather: 4, shape: { kind: "linear" as const, x1: 0.3, y1: 0, x2: 0.7, y2: 0 } };
  list.push({ name: "layer mask on a cutout", captureMask: layerMask.id, document: doc([plateLayer(), cutoutLayer({
    x: 0.45, scale: 1.1, masks: [layerMask], maskId: layerMask.id, nodes: [gradeWith({ exposure: 0.3 })],
  })]) });
  const stackMask = radialBase("layer stack", [createMaskEntry(linear, { mode: "difference" })]);
  list.push({ name: "layer mask (stack) with edge group", captureMask: stackMask.id, document: doc([plateLayer(), cutoutLayer({
    masks: [stackMask], maskId: stackMask.id, blendMode: "screen",
    edge: { defringe: 1, lightWrap: 40, lightWrapSpread: 8, contactShadow: 50, contactShadowSoftness: 30, contactShadowOffset: 0.02, contactShadowSquash: 0.25 },
  })]) });

  // Clipping groups (Photoshop): consecutive clipped layers share the base below them.
  const canvas = { width: 480, height: 320 };
  const clippedGradient = { ...createGradientLayer({ kind: "linear", angle: 60, stops: [
    { at: 0, color: "#ff3300", opacity: 1 }, { at: 1, color: "#0033ff", opacity: 1 },
  ] }), clip: true, blendMode: "overlay" as const, opacity: 0.9 };
  const clippedText = { ...createTextLayer(canvas, { content: "CLIP" }), clip: true, x: 0.48 };
  list.push({ name: "clipping group: gradient + text clipped to a cutout", document: doc([
    plateLayer(), cutoutLayer({ opacity: 0.8, x: 0.47 }), clippedGradient, clippedText,
  ]) });
  list.push({ name: "clipping group: base hidden hides the group", document: doc([
    plateLayer(), cutoutLayer({ visible: false }), { ...clippedGradient, id: "g2" },
  ]) });
  list.push({ name: "clip on the bottom layer is ignored", document: doc([plateLayer({ clip: true }), cutoutLayer()]) });

  // Track mattes: a layer mask whose stack holds a layer entry. The matte layer is hidden (AE).
  const matteLayer = createLayer({ id: "matte-asset", name: "Matte", path: PATHS.cutout }, { visible: false, x: 0.55, y: 0.45, scale: 1.2, rotation: 8 });
  const filler = (id: string, over: Partial<PhotoLayer> = {}) => createLayer({ id, name: "Fill", path: PATHS.fill }, over);
  for (const [channel, invert] of [["alpha", false], ["alpha", true], ["luma", false], ["luma", true]] as const) {
    const track = { ...createMask(`track ${channel}${invert ? " inverted" : ""}`), feather: 0,
      stack: [createMaskEntry({ kind: "layer", layerId: matteLayer.id, channel }, { invert })] };
    list.push({ name: `track matte: ${channel}${invert ? " inverted" : ""}`, captureMask: track.id, document: doc([
      plateLayer(), { ...matteLayer, nodes: [gradeWith({ exposure: 0.5 })] }, filler(`fill-${channel}-${invert}`, { masks: [track], maskId: track.id, blendMode: "hardLight" }),
    ]) });
  }
  const visibleMatte = { ...createMask("visible matte"), stack: [createMaskEntry({ kind: "layer", layerId: "matte-visible", channel: "alpha" })] };
  list.push({ name: "track matte: matte layer also visible, above", document: doc([
    plateLayer(), filler("fill-v", { masks: [visibleMatte], maskId: visibleMatte.id }),
    { ...matteLayer, id: "matte-visible", visible: true, opacity: 0.6 },
  ]) });
  const loopA = { ...createMask("a"), stack: [createMaskEntry({ kind: "layer", layerId: "loop-b", channel: "alpha" })] };
  const loopB = { ...createMask("b"), stack: [createMaskEntry({ kind: "layer", layerId: "loop-a", channel: "alpha" }, { invert: true })] };
  list.push({ name: "track matte cycle is reported and broken", document: doc([
    plateLayer(), cutoutLayer({ id: "loop-a", masks: [loopA], maskId: loopA.id }),
    filler("loop-b", { masks: [loopB], maskId: loopB.id, opacity: 0.7 }),
  ]) });
  const nodeMatte = { ...createMask("node limited by a layer"), stack: [createMaskEntry({ kind: "layer", layerId: "cutout", channel: "luma" }, { feather: 3 })] };
  list.push({ name: "composite grade limited by a layer's luma", captureMask: nodeMatte.id, document: doc([plateLayer(), cutoutLayer({ x: 0.4 })], {
    masks: [nodeMatte], nodes: [gradeWith({ exposure: -1.5, saturation: 0 }, nodeMatte.id)],
  }) });

  // Keys for stock on black.
  const stock = (id: string, over: Partial<PhotoLayer>) => createLayer({ id, name: "Fire", path: PATHS.onBlack }, over);
  list.push({ name: "unmultiply: fire on black over the plate", document: doc([plateLayer(), stock("fire-u", { nodes: [createUnmultiplyNode()] })]) });
  list.push({ name: "unmultiply, then graded, scaled", document: doc([plateLayer(), stock("fire-g", {
    scale: 1.4, rotation: 12, nodes: [createUnmultiplyNode(), gradeWith({ temperature: 30, exposure: 0.3 })],
  })]) });
  list.push({ name: "luma key: soft", document: doc([plateLayer(), stock("fire-k", { nodes: [{ ...createLumaKeyNode(), low: 10, high: 95, softness: 12 }] })]) });
  list.push({ name: "luma key: hard, inverted, masked", document: (() => {
    const mask = { ...createMask("key mask"), shape: { kind: "linear" as const, x1: 0, y1: 0.5, x2: 1, y2: 0.5 } };
    return doc([plateLayer({ masks: [mask], nodes: [{ ...createLumaKeyNode(), low: 30, high: 70, softness: 0, invert: true, maskId: mask.id }] })]);
  })() });
  return list;
}

function fixtures(): Fixture[] {
  const list: Fixture[] = [
    { name: "plate, untouched", document: doc([plateLayer()]) },
    { name: "single-source document (pre-layers)", document: { ...createPhotoDocument("plate", PATHS.plate), nodes: [gradeWith({ exposure: 0.5 })] } },
    graded("grade: exposure/contrast/white balance", { exposure: 0.7, contrast: 35, temperature: 40, tint: -25 }),
    graded("grade: highlights/shadows", { highlights: -60, shadows: 70 }),
    graded("grade: saturation/vibrance", { saturation: 160, vibrance: 50 }),
    graded("grade: curve/levels/filter", {
      curve: [{ x: 0.2, y: 0.12 }, { x: 0.5, y: 0.55 }, { x: 0.8, y: 0.92 }],
      levelsBlack: 18, levelsWhite: 235, levelsGamma: 1.3, filter: "aurora", filterAmount: 80,
    }),
    graded("grade: HSL bands", {
      hsl: [
        { hue: 30, saturation: 20, luminance: 0 }, { hue: 0, saturation: -40, luminance: 10 },
        { hue: -20, saturation: 0, luminance: 0 }, { hue: 40, saturation: 30, luminance: -20 },
        { hue: 0, saturation: 0, luminance: 0 }, { hue: -35, saturation: 50, luminance: 15 },
        { hue: 10, saturation: 0, luminance: 0 }, { hue: 0, saturation: -100, luminance: 0 },
      ],
    }),
    graded("grade: vignette", { vignette: -70, vignetteMidpoint: 40, vignetteFeather: 50 }),
    graded("grade: vignette lighten", { vignette: 60 }),
    graded("grade: denoise luma+chroma", { denoise: 70, denoiseDetail: 30, denoiseColor: 60 }),
    graded("grade: denoise luma only", { denoise: 40 }),
    graded("grade: blur", { blur: 35 }),
    graded("grade: sharpen", { sharpen: 80 }),
    graded("grade: grain", { grain: 60 }),
    graded("grade: extremes", { exposure: 3, contrast: 100, saturation: 200, vibrance: 100, levelsBlack: 60, sharpen: 100 }),
    graded("grade: crush", { exposure: -4, contrast: -100, saturation: 0, filter: "mono" }),
    graded("grade: every filter step", { filter: "bleach", filterAmount: 100, temperature: -80 }),
    { name: "gradient map", document: doc([plateLayer({ nodes: [{ ...createGradientMapNode(), amount: 85 }] })]) },
    { name: "gradient map, repeat mirror ease", document: doc([plateLayer({ nodes: [{
      ...createGradientMapNode(), amount: 100,
      gradient: { ...createGradientMapNode().gradient, repeat: 3, mirror: true, ease: 1.8,
        stops: [{ at: 0, color: "#000000", opacity: 1 }, { at: 0.5, color: "#ff2200", opacity: 0.5 }, { at: 1, color: "#ffffff", opacity: 1 }] },
    }] })]) },
  ];

  const strokes = createMask("strokes");
  strokes.strokes = [
    { points: [[0.2, 0.3], [0.4, 0.35], [0.6, 0.5]], radius: 0.08, hardness: 0.3, erase: false },
    { points: [[0.45, 0.4]], radius: 0.03, hardness: 1, erase: true },
  ];
  list.push(maskFixture("strokes", strokes));
  list.push(maskFixture("linear", { ...createMask("linear"), shape: { kind: "linear", x1: 0.1, y1: 0.2, x2: 0.9, y2: 0.7 } }));
  list.push(maskFixture("radial inverted", { ...createMask("radial"), invert: true, shape: { kind: "radial", cx: 0.4, cy: 0.5, radius: 0.3, aspect: 1.4, feather: 0.5 } }));
  list.push(maskFixture("colour", { ...createMask("colour"), shape: { kind: "colour", r: 220, g: 60, b: 40, tolerance: 60, softness: 40 } }));
  list.push(maskFixture("SAM 2 file, feathered", { ...createMask("sam"), sourcePath: PATHS.matte, feather: 6 }));
  list.push(maskFixture("disabled", { ...createMask("off"), enabled: false, shape: { kind: "linear", x1: 0, y1: 0, x2: 1, y2: 0 } }));
  const missing = gradeWith({ exposure: 1 }, "no-such-mask");
  list.push({ name: "node with a missing mask", document: doc([plateLayer({ nodes: [missing] })]) });

  for (const mode of BLEND_MODES) {
    list.push({ name: `blend: ${mode.id}`, document: doc([plateLayer(), cutoutLayer({ blendMode: mode.id as BlendMode, opacity: 0.8, x: 0.45, y: 0.5 })]) });
  }
  list.push({ name: "place: identity cutout", document: doc([plateLayer(), cutoutLayer({ x: 0.5, y: 0.5 })]) });
  list.push({ name: "place: scale 0.7 rotate 17 crop", document: doc([plateLayer(), cutoutLayer({ scale: 0.7, rotation: 17, x: 0.37, y: 0.52, cropLeft: 8, cropTop: 5, cropRight: 12, cropBottom: 3 })]) });
  list.push({ name: "place: upscale 1.6, fractional", document: doc([plateLayer(), cutoutLayer({ scale: 1.6, x: 0.513, y: 0.487 })]) });
  list.push({ name: "place: rotate 90", document: doc([plateLayer(), cutoutLayer({ rotation: 90, x: 0.3, y: 0.4 })]) });
  list.push({ name: "place: via canvas (diagnostic)", placement: "canvas", document: doc([plateLayer(), cutoutLayer({ scale: 0.7, rotation: 17, x: 0.37, y: 0.52 })]) });

  const kinds = ["linear", "reflected", "radial", "angular", "diamond"] as const;
  for (const kind of kinds) {
    list.push({ name: `gradient layer: ${kind}`, document: doc([plateLayer(), createGradientLayer({
      kind, angle: 25, cx: 0.4, cy: 0.55, scale: 0.8, dither: 1, repeat: kind === "linear" ? 2 : 1, mirror: kind === "linear", ease: kind === "radial" ? 1.6 : 1,
      stops: [{ at: 0, color: "#1030ff", opacity: 1 }, { at: 0.6, color: "#ffcc00", opacity: 0.6 }, { at: 1, color: "#ff0060", opacity: 0 }],
    })]) });
  }
  const canvas = { width: 480, height: 320 };
  list.push({ name: "text layer", document: doc([plateLayer(), { ...createTextLayer(canvas, { content: "Parity 123", strokeWidth: 3, shadow: 4 }), rotation: -8 }]) });
  list.push({ name: "shape + balloon", document: doc([plateLayer(), { ...createShapeLayer("star", canvas), x: 0.3 }, { ...createShapeLayer("speech", canvas), x: 0.7, y: 0.35 }]) });

  for (const background of ["transparent", "black", "white"] as const) {
    list.push({ name: `background ${background}, cutout only`, document: doc([cutoutLayer({ rotation: 10 })], { background, canvasWidth: 480, canvasHeight: 320 }) });
  }

  const retouch = createRetouchNode();
  retouch.strokes = [{ points: [[0.3, 0.3], [0.35, 0.32]], radius: 0.05, hardness: 0.6, offsetX: 0.2, offsetY: 0.1, heal: true }];
  const warp = createWarpNode();
  warp.strokes = [{ points: [[0.5, 0.5], [0.55, 0.52]], radius: 0.15, strength: 0.6, mode: "push" }];
  const removeMask = { ...createMask("remove"), shape: { kind: "radial" as const, cx: 0.6, cy: 0.4, radius: 0.12, aspect: 1, feather: 0.3 } };
  const remove = { ...createRemoveNode(removeMask.id), cachePath: PATHS.fill };
  list.push({ name: "CPU nodes: relight, retouch, warp, remove", document: doc(
    [plateLayer({ masks: [removeMask], nodes: [createRelightNode(), retouch, warp, remove] })], { depthPath: PATHS.depth }) });
  list.push({ name: "relight without depth (skipped)", document: doc([plateLayer({ nodes: [createRelightNode()] })]) });

  list.push({ name: "edge group: defringe, shadow, wrap", document: doc([plateLayer(), cutoutLayer({
    scale: 0.9, edge: { defringe: 2, lightWrap: 60, lightWrapSpread: 10, contactShadow: 70, contactShadowSoftness: 40, contactShadowOffset: 0.03, contactShadowSquash: 0.2 },
  })]) });

  const compositeMask = { ...createMask("composite"), shape: { kind: "linear" as const, x1: 0, y1: 1, x2: 0, y2: 0 } };
  list.push({ name: "composite stack with a mask", document: doc([plateLayer(), cutoutLayer({ blendMode: "screen" })], {
    masks: [compositeMask], nodes: [gradeWith({ contrast: 40, saturation: 130 }), gradeWith({ exposure: -1 }, compositeMask.id)],
  }) });
  list.push({ name: "proxy render at 0.5", scale: 0.5, document: doc([plateLayer({ nodes: [gradeWith({ exposure: 0.4, blur: 10 })] }), cutoutLayer({ scale: 0.8, rotation: 30 })]) });
  list.push({ name: "hidden and zero-opacity layers", document: doc([plateLayer(), cutoutLayer({ visible: false }), cutoutLayer({ opacity: 0 })]) });
  list.push(...phase2Fixtures());
  list.push(...phase3Fixtures().map((fixture) => ({ ...fixture, changed: true })));
  list.push(...phase4Fixtures().map((fixture) => ({ ...fixture, changed: true })));
  list.push(...phase6Fixtures().map((fixture) => ({ ...fixture, changed: true })));
  list.push(...phase7Fixtures().map((fixture) => ({ ...fixture, changed: true })));
  list.push(...clipMatteFixtures().map((fixture) => ({ ...fixture, changed: true })));
  return list;
}

/**
 * Background removal on the timeline: real programme documents with a clip matte (a `sequence`
 * entry indexed by the clip's source frame) -- a cutout on V2 over V1, a turned and inverted V1
 * clip, a matte intersected with a transition's iris, and the feather at proxy 0.5.
 */
function clipMatteFixtures(): Fixture[] {
  const media: ProgrammeMedia = { assetId: "clip", path: CLIP.path, kind: "video", fps: 24, frameCount: CLIP.frameCount, width: CLIP.width, height: CLIP.height, hasAudio: false };
  const matte: ProgrammeMatte = { cachePath: PATHS.sequence, frameCount: 24, first: 0, feather: 3, expand: 2, invert: false };
  const clip = (id: string, over: Partial<ProgrammeClip> = {}): ProgrammeClip => ({
    id, name: id, media, start: 0, duration: 1, sourceIn: 0, rate: 1, x: 0.5, y: 0.5, scale: 100, rotation: 0, opacity: 1,
    blend: "normal", effects: effectsFor(undefined), conform: { fit: "contain", pad: "black" }, vfx: [], volumeDb: 0, ...over,
  });
  const overlay = (id: string, over: Partial<ProgrammeOverlay> = {}): ProgrammeOverlay => ({
    id, media, start: 0, duration: 1, sourceIn: 0, rate: 1, box: { ...DEFAULT_OVERLAY }, opacity: 1, effects: effectsFor(undefined),
    vfx: [], volumeDb: 0, muted: false, ...over,
  });
  const programme = (clips: ProgrammeClip[], overlays: ProgrammeOverlay[] = []): Programme => ({
    width: 480, height: 320, fps: 24, duration: 2, clips, overlays, graphics: [], audio: [], rippled: 0, pixelScale: 1,
  });
  const background = clip("bg", { sourceIn: 0.25 });
  const cutoutOnV2 = programmeDocument(programme([background], [overlay("fg", { matte })]), 5);
  const v1Only = programmeDocument(programme([background]), 5);
  const turned = { scale: 80, rotation: 10, x: 0.55 };
  const incoming = { start: 1, sourceIn: 0.5, incoming: { kind: "circleopen", duration: 0.5 } };
  return [
    { name: "clip matte: cutout on V2 over V1 (feather, grow)", frame: 5, document: cutoutOnV2, versus: v1Only },
    { name: "clip matte: cutout at proxy 0.5", frame: 5, scale: 0.5, document: cutoutOnV2, versus: v1Only },
    {
      name: "clip matte: inverted on a turned V1 clip", frame: 8,
      document: programmeDocument(programme([clip("a", { ...turned, matte: { ...matte, invert: true, expand: -3 } })]), 8),
      versus: programmeDocument(programme([clip("a", turned)]), 8),
    },
    {
      name: "clip matte: intersected with a circle-open transition", frame: 18,
      document: programmeDocument(programme([clip("a", { sourceIn: 0.2 }), clip("b", { ...incoming, matte })]), 18),
      versus: programmeDocument(programme([clip("a", { sourceIn: 0.2 }), clip("b", incoming)]), 18),
    },
  ];
}

/**
 * VFX Phase 7: the pieces the programme (timeline) documents are made of -- the clip Effects node,
 * a horizontal stretch, a timeline title layer, and a transition's vector mask over a clip.
 */
function phase7Fixtures(): Fixture[] {
  const list: Fixture[] = [];
  const bare = doc([plateLayer()]);
  const fx = (effects: Partial<ClipEffectsParams>): PhotoNode =>
    ({ id: "fx", kind: "clipEffects", enabled: true, effects: { ...NEUTRAL_CLIP_EFFECTS, ...effects } });
  const push = (name: string, document: PhotoDocument, versus: PhotoDocument) => list.push({ name: `programme: ${name}`, document, versus });
  push("clip effects: eq, hue, vignette, noise", doc([plateLayer({ nodes: [fx({ brightness: 0.08, contrast: 1.3, saturation: 1.4, gamma: 1.2, hue: 35, vignette: 0.6, noise: 12, seed: 3 })] })]), bare);
  push("clip effects: blur and sharpen", doc([plateLayer(), cutoutLayer({ nodes: [fx({ blur: 6, sharpen: 1.2 })] })]), doc([plateLayer(), cutoutLayer()]));
  push("clip effects: pixelate, dissolve, grayscale", doc([plateLayer(), cutoutLayer({ x: 0.45, nodes: [fx({ pixelate: 9, dissolve: 0.4, grayscale: 0.7, seed: 11 })] })]), doc([plateLayer(), cutoutLayer({ x: 0.45 })]));
  list.push({ name: "programme: clip effects at proxy 0.5", scale: 0.5, document: doc([plateLayer({ nodes: [fx({ blur: 8, vignette: 0.4 })] })]), versus: bare });
  push("stretch: a squeezed cutout and a stretched pad rectangle", doc([plateLayer(),
    { ...createPathLayer({ subpaths: [{ closed: true, nodes: [{ x: -60, y: -40 }, { x: 60, y: -40 }, { x: 60, y: 40 }, { x: -60, y: 40 }] }], stroke: undefined, strokeWidth: 0, fill: { kind: "solid", color: "#203050" } }, 0.3, 0.3, "Pad"), stretchX: 1.7, scale: 0.8 },
    cutoutLayer({ stretchX: 1.45, scale: 0.8, rotation: 12, x: 0.6 })]), doc([plateLayer(), cutoutLayer({ scale: 0.8, rotation: 12, x: 0.6 })]));
  const graphic = (template: GraphicOverlay["template"], local: number): PhotoLayer => ({
    ...createLayer({ id: `g-${template}`, name: "Graphic", path: "" }),
    timelineGraphic: { local, duration: 4, canvasWidth: 480, canvasHeight: 320, overlay: {
      template, text: "Parity title", subtitle: "Phase 7", textColor: "#ffffff", backgroundColor: "#101830", accentColor: "#ff3060",
      fontSize: 42, backgroundOpacity: 0.8, motionIn: "slide-up", motionDuration: 0.8, strokeWidth: 2, strokeColor: "#000000", shadowOffset: 3,
    } },
  });
  push("title: lower third, mid motion", doc([plateLayer(), graphic("lower-third", 0.4)]), bare);
  push("title: banner, settled", doc([plateLayer(), graphic("banner", 2)]), bare);
  const radial = transitionFrame("radial", 0.4, 480 / 320).incoming.mask!;
  const circle = transitionFrame("circleopen", 0.6, 480 / 320).incoming.mask!;
  const wipe = { ...createMask("wipe"), stack: [
    createMaskEntry({ kind: "path", path: { ...NEUTRAL_PATH, stroke: undefined, strokeWidth: 0, subpaths: radial } }),
    createMaskEntry({ kind: "path", path: { ...NEUTRAL_PATH, stroke: undefined, strokeWidth: 0, subpaths: circle } }, { mode: "intersect" }),
  ] };
  const vfx = (recipe: MaskVfxRecipeV1, index = 6): PhotoNode => ({
    id: `vfx-${recipe.presetId}`, kind: "maskVfx", enabled: true, recipe, index, count: 24, fps: 24, envelope: 0.4,
    mattePath: PATHS.matte, sourceWidth: 480, sourceHeight: 320,
  });
  const powerUp = MASK_VFX_RECIPES.find((item) => item.presetId === "anime-power-up")!;
  const neon = { ...DEFAULT_MASK_VFX_RECIPE, effect: "neon-edge" as const, transitionIn: "explosion" as const, transitionInDuration: 0.5 };
  push("mask VFX: anime power up (flame matte, gradient, passes)", doc([plateLayer({ nodes: [vfx(powerUp)] })]), bare);
  push("mask VFX: neon edge + explosion entrance on a scaled, turned plate", doc([plateLayer({ scale: 0.8, rotation: 9, x: 0.52, nodes: [vfx(neon, 4)] })]), doc([plateLayer({ scale: 0.8, rotation: 9, x: 0.52 })]));
  list.push({ name: "programme: mask VFX at proxy 0.5", scale: 0.5, document: doc([plateLayer({ nodes: [vfx({ ...DEFAULT_MASK_VFX_RECIPE, effect: "hologram" })] })]), versus: bare });
  push("transition: radial sweep intersect iris over a clip", doc([plateLayer(), cutoutLayer({ masks: [wipe], maskId: wipe.id, scale: 1.6 })]), doc([plateLayer(), cutoutLayer({ scale: 1.6 })]));
  return list;
}

/**
 * VFX Phase 6: paths (fill rules, strokes, joins, caps, dashes, a gradient paint, an envelope), a
 * layer clipped to a path, a path mask, the pixel envelope node and repeats. Each counts `changed`
 * against the same document without the thing under test.
 */
function phase6Fixtures(): Fixture[] {
  const list: Fixture[] = [];
  const k = 0.5522847498;
  const circle = (cx: number, cy: number, r: number): Subpath => ({ closed: true, nodes: [
    { x: cx + r, y: cy, in: [0, -k * r], out: [0, k * r] }, { x: cx, y: cy + r, in: [k * r, 0], out: [-k * r, 0] },
    { x: cx - r, y: cy, in: [0, k * r], out: [0, -k * r] }, { x: cx, y: cy - r, in: [-k * r, 0], out: [k * r, 0] },
  ] });
  const star: Subpath = { closed: true, nodes: Array.from({ length: 10 }, (_, i) => {
    const angle = -Math.PI / 2 + (i * Math.PI) / 5, radius = i % 2 ? 36 : 90;
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
  }) };
  const wave: Subpath = { closed: false, nodes: [{ x: -110, y: 20, out: [40, -80] }, { x: 0, y: 0, in: [-40, 40], out: [40, -40] }, { x: 110, y: -20, in: [-40, 80] }] };
  const path = (subpaths: Subpath[], over: Partial<PathBlock> = {}, layer: Partial<PhotoLayer> = {}) =>
    ({ ...createPathLayer({ subpaths, ...over }, 0.5, 0.5, "Path"), ...layer });
  const bare = doc([plateLayer()]);
  const push = (name: string, document: PhotoDocument, versus: PhotoDocument, captureMask?: string) =>
    list.push({ name: `path: ${name}`, document, versus, captureMask });

  push("star, non-zero, miter stroke", doc([plateLayer(), path([star], { strokeWidth: 6, join: "miter" }, { x: 0.4 })]), bare);
  push("compound ring, even-odd, dashed round-cap stroke", doc([plateLayer(),
    path([circle(0, 0, 80), circle(10, 0, 40)], { fillRule: "evenodd", strokeWidth: 5, cap: "round", join: "round", dash: [14, 9], dashOffset: 3,
      fill: { kind: "solid", color: "#30a0ff" }, stroke: { kind: "solid", color: "#ffffff" } }, { x: 0.55, y: 0.45 })]), bare);
  push("open wave, stroke only, square caps, rotated and scaled", doc([plateLayer(),
    path([wave], { fill: undefined, strokeWidth: 9, cap: "square", join: "bevel" }, { rotation: 23, scale: 1.4, x: 0.47, y: 0.55 })]), bare);
  push("gradient fill, screen blend", doc([plateLayer(), path([circle(0, 0, 70)], {
    fill: { kind: "gradient", gradient: { ...NEUTRAL_GRADIENT, kind: "linear", angle: 30, stops: [
      { at: 0, color: "#ff2060", opacity: 1 }, { at: 1, color: "#2040ff", opacity: 0.5 }] } }, stroke: undefined,
  }, { blendMode: "screen" })]), bare);
  const bent = createEnvelope({ x: -90, y: -90, width: 180, height: 180 }, 2, 2);
  bent.points[1] = [0, -150];
  bent.points[4] = [30, 20];
  push("envelope on a path", doc([plateLayer(), path([star], { envelope: bent, strokeWidth: 3 })]),
    doc([plateLayer(), path([star], { strokeWidth: 3 })]));
  const base = path([star], { strokeWidth: 0, stroke: undefined }, { x: 0.45 });
  push("cutout clipped to a path", doc([plateLayer(), base, cutoutLayer({ clip: true, scale: 0.9 })]),
    doc([plateLayer(), base, cutoutLayer({ scale: 0.9 })]));
  const vector = { ...createMask("vector"), stack: [createMaskEntry({ kind: "path", path: { ...NEUTRAL_PATH, fillRule: "evenodd",
    subpaths: [
      { closed: true, nodes: [{ x: 0.2, y: 0.2 }, { x: 0.8, y: 0.25, in: [0.1, -0.2] }, { x: 0.6, y: 0.85 }, { x: 0.25, y: 0.7, out: [-0.1, -0.1] }] },
      { closed: true, nodes: [{ x: 0.4, y: 0.4 }, { x: 0.6, y: 0.45 }, { x: 0.5, y: 0.6 }] },
    ] } }, { feather: 3 })] };
  const masked = (maskId?: string) => doc([plateLayer({ masks: [vector], nodes: [gradeWith({ exposure: 1.1, saturation: 30 }, maskId)] })]);
  list.push({ name: "path: vector mask (even-odd, feathered) limits a grade", captureMask: vector.id, document: masked(vector.id), versus: doc([plateLayer({ masks: [vector] })]) });
  list.push({ name: "path: vector mask at proxy 0.5", scale: 0.5, captureMask: vector.id, document: masked(vector.id), versus: doc([plateLayer({ masks: [vector] })]) });
  const envelopeNode = createEnvelopeNode({ x: 0.25, y: 0.2, width: 0.5, height: 0.6 }, 1, 1);
  envelopeNode.envelope.points[1] = [0.85, 0.1];
  envelopeNode.envelope.points[2] = [0.2, 0.9];
  push("envelope node on a cutout", doc([plateLayer(), cutoutLayer({ nodes: [envelopeNode], rotation: 6 })]), doc([plateLayer(), cutoutLayer({ rotation: 6 })]));
  const small = path([star], { strokeWidth: 3 }, { scale: 0.3, x: 0.2, y: 0.25 });
  push("repeat: brick grid with scale and opacity steps", doc([plateLayer(), { ...small, repeat: { kind: "grid", columns: 4, rows: 3, gapX: 90, gapY: 70, shift: 0.5, scaleStep: 0.95, opacityStep: 0.92 } }]), doc([plateLayer(), small]));
  push("repeat: radial, turned with the ring", doc([plateLayer(), { ...small, x: 0.7, y: 0.5, repeat: { kind: "radial", count: 7, radius: 90, startAngle: 0, rotate: true } }]), doc([plateLayer(), { ...small, x: 0.7, y: 0.5 }]));
  const guide = { ...path([wave], { fill: undefined, strokeWidth: 2 }), id: "guide-path", visible: false };
  push("repeat: cutout along a hidden guide path", doc([plateLayer(), guide, cutoutLayer({ scale: 0.25, repeat: { kind: "path", layerId: guide.id, count: 6, rotate: true } })]),
    doc([plateLayer(), guide, cutoutLayer({ scale: 0.25 })]));
  return list;
}

/* ------------------------------------------------------------ resources */

/** The same resource builder Photo Edit uses (domain/photoResources.ts), without its cache. */
function resourcesFor(document: PhotoDocument, files: Map<string, Loaded>, width: number, height: number) {
  return (scope: ComposeScope, pixels: Uint8ClampedArray) =>
    buildScopeResources({ document, files: (path) => files.get(path) }, scope, width, height, pixels);
}

/* ------------------------------------------------------------ compare */

const RANK: Verdict[] = ["exact", "pass", "boundary", "fail"];
const worse = (a: Verdict, b: Verdict): Verdict => RANK[Math.max(RANK.indexOf(a), RANK.indexOf(b))];

function compareCoverage(old: Uint8Array | undefined, gpu: Uint8Array | undefined): NonNullable<FixtureResult["mask"]> {
  if (!old || !gpu || old.length !== gpu.length) {
    return { maxDiff: 255, over1: -1, missing: `coverage missing (old ${Boolean(old)}, gpu ${Boolean(gpu)})` };
  }
  let maxDiff = 0, over1 = 0;
  for (let index = 0; index < old.length; index++) {
    const diff = Math.abs(old[index] - gpu[index]);
    if (diff > maxDiff) maxDiff = diff;
    if (diff > 1) over1++;
  }
  return { maxDiff, over1 };
}

function maskVerdict(mask: NonNullable<FixtureResult["mask"]>, pixels: number): Verdict {
  if (mask.missing) return "fail";
  if (mask.maxDiff === 0) return "exact";
  if (mask.maxDiff <= 1) return "pass";
  return mask.over1 <= pixels * 0.005 && mask.maxDiff <= 8 ? "boundary" : "fail";
}

function verdictOf(d: Difference, pixels: number, skippedMatch: boolean): Verdict {
  if (!skippedMatch) return "fail";
  if (d.maxAlpha === 0 && d.maxPremultiplied === 0 && d.maxStraight === 0) return "exact";
  if (d.maxAlpha <= 1 && d.maxPremultiplied <= 1) return "pass";
  if (d.over1 <= pixels * 0.005 && d.meanAbs < 0.05 && d.maxAlpha <= 8 && d.maxPremultiplied <= 8) return "boundary";
  return "fail";
}

function compare(old: Uint8ClampedArray, gpu: Uint8ClampedArray, width: number): Difference {
  let maxAlpha = 0, maxPremultiplied = 0, maxStraight = 0, over1 = 0, total = 0;
  let worst: Difference["worst"];
  let worstValue = -1;
  for (let at = 0; at < old.length; at += 4) {
    const alphaDiff = Math.abs(old[at + 3] - gpu[at + 3]);
    let premul = 0, straight = 0;
    for (let c = 0; c < 3; c++) {
      straight = Math.max(straight, Math.abs(old[at + c] - gpu[at + c]));
      premul = Math.max(premul, Math.abs(old[at + c] * old[at + 3] - gpu[at + c] * gpu[at + 3]) / 255);
      total += Math.abs(old[at + c] - gpu[at + c]);
    }
    total += alphaDiff;
    maxAlpha = Math.max(maxAlpha, alphaDiff);
    maxPremultiplied = Math.max(maxPremultiplied, premul);
    maxStraight = Math.max(maxStraight, straight);
    const score = Math.max(alphaDiff, premul);
    if (score > 1) over1++;
    if (score > worstValue) {
      worstValue = score;
      const pixel = at / 4;
      worst = { x: pixel % width, y: (pixel / width) | 0, old: Array.from(old.subarray(at, at + 4)), gpu: Array.from(gpu.subarray(at, at + 4)) };
    }
  }
  return {
    maxAlpha, maxPremultiplied: Math.round(maxPremultiplied * 100) / 100, maxStraight, over1,
    meanAbs: Math.round(total / old.length * 10000) / 10000, worst,
  };
}

/* ------------------------------------------------------------ run */

export async function runParity(
  log: (line: string) => void = () => {}, options: { performance?: boolean; filter?: string; extra?: Fixture[] } = {},
): Promise<ParityReport> {
  const files = new Map<string, Loaded>([
    [PATHS.plate, plate()], [PATHS.cutout, cutout()], [PATHS.matte, matte()], [PATHS.depth, depth()], [PATHS.fill, fill()],
    [PATHS.onBlack, onBlack()], [PATHS.alphaMatte, alphaMatte()], [SEQUENCE_FRAME, matte()],
  ]);
  for (let frame = 0; frame < ELEMENT_FRAMES; frame++) files.set(mediaFrameKey(ELEMENT, frame), elementFrame(frame));
  for (let frame = 0; frame < CLIP.frameCount; frame++) files.set(mediaFrameKey(CLIP, frame), clipFrame(frame));
  for (let frame = 1; frame < 24; frame++) files.set(`${PATHS.sequence}/masks/${String(frame).padStart(5, "0")}.png`, sequenceFrame(frame));
  for (let frame = 0; frame < CLIP.frameCount; frame++) files.set(`${DEPTH_SEQUENCE}/masks/${String(frame).padStart(5, "0")}.png`, depthFrame(frame));
  const sources = new Map<string, LayerSource>();
  for (const [path, held] of files) sources.set(path, { image: held.canvas, width: held.width, height: held.height });

  const renderer = new PhotoRenderer();
  const gl = renderer.gl.gl;
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  const rendererName = String(debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER));
  const results: FixtureResult[] = [];

  const selected = [...fixtures(), ...(options.extra ?? [])]
    .filter((fixture) => !options.filter || fixture.name.includes(options.filter));
  for (const fixture of selected) {
    const full = { width: fixture.document.canvasWidth ?? 480, height: fixture.document.canvasHeight ?? 320 };
    const scale = fixture.scale ?? 1;
    const width = Math.round(full.width * scale), height = Math.round(full.height * scale);
    const input: ComposeInput = {
      document: fixture.document, width, height, sources, captureMask: fixture.captureMask, frame: fixture.frame,
      resourcesFor: resourcesFor(fixture.document, files, width, height),
    };
    const empty = (): Difference => ({ maxAlpha: 0, maxPremultiplied: 0, maxStraight: 0, over1: 0, meanAbs: 0 });
    const result: FixtureResult = {
      name: fixture.name, size: `${width}x${height}`, verdict: "fail", pipeline: empty(),
      cpu: [], oldMs: 0, gpuMs: 0, skippedMatch: true,
    };
    try {
      // Warm-up, discarded. The first canvas placement of a freshly drawn source (text, a shape)
      // can resample differently from every later one -- Chrome changes how it holds that canvas
      // once it has been drawn from or uploaded -- so the first text fixture of a run disagreed
      // with itself: 2172 pixels off by up to 3, and exact when the same document ran again
      // (found 2026-09-27). Rendering once first puts the golden and the pipeline on the same side.
      composeDocument(input);
      renderer.compose(input, { placement: "canvas" });
      let started = performance.now();
      const gpu = renderer.compose(input, { placement: fixture.placement ?? "gpu" });
      result.gpuMs = Math.round((performance.now() - started) * 10) / 10;
      started = performance.now();
      const old = composeDocument(input);
      result.oldMs = Math.round((performance.now() - started) * 10) / 10;
      const pipeline = fixture.placement === "canvas" ? gpu : renderer.compose(input, { placement: "canvas" });
      result.pipeline = compare(old.pixels, pipeline.pixels, width);
      if (fixture.placement !== "canvas") result.gpuPlacement = compare(old.pixels, gpu.pixels, width);
      result.cpu = gpu.stats.cpu;
      result.skippedMatch = JSON.stringify(old.report.skipped) === JSON.stringify(gpu.report.skipped);
      result.verdict = verdictOf(result.pipeline, width * height, result.skippedMatch);
      if (fixture.changed && fixture.document.layers?.length) {
        const versus = fixture.versus ?? { ...fixture.document, layers: fixture.document.layers.slice(0, 1) };
        const alone = composeDocument({ ...input, document: versus, captureMask: undefined, resourcesFor: resourcesFor(versus, files, width, height) }).pixels;
        let changed = 0;
        for (let at = 0; at < alone.length; at += 4) {
          if (alone[at] !== old.pixels[at] || alone[at + 1] !== old.pixels[at + 1] || alone[at + 2] !== old.pixels[at + 2] || alone[at + 3] !== old.pixels[at + 3]) changed++;
        }
        result.changed = changed;
      }
      if (fixture.captureMask) {
        result.mask = compareCoverage(old.maskCoverage, pipeline.maskCoverage);
        result.verdict = worse(result.verdict, maskVerdict(result.mask, width * height));
      }
    } catch (error) {
      result.error = String((error as Error)?.stack ?? error);
    }
    results.push(result);
    const d = result.pipeline, g = result.gpuPlacement;
    log(`${result.verdict.toUpperCase()} ${fixture.name}: premul ${d.maxPremultiplied}, alpha ${d.maxAlpha}, >1 ${d.over1}, mean ${d.meanAbs}`
      + (result.mask ? ` | mask: max ${result.mask.maxDiff}, >1 ${result.mask.over1}${result.mask.missing ? ` ${result.mask.missing}` : ""}` : "")
      + (result.changed !== undefined ? ` | changed ${result.changed}` : "")
      + (result.skippedMatch ? "" : " | SKIPPED REPORTS DIFFER")
      + (g && (g.maxPremultiplied > 1 || g.maxAlpha > 1) ? ` | gpu placement: premul ${g.maxPremultiplied}, alpha ${g.maxAlpha}, >1 ${g.over1}` : "")
      + (result.error ? ` ERROR ${result.error.split("\n")[0]}` : ""));
    await new Promise((resolve) => setTimeout(resolve, 0));
  }

  const counts: Record<Verdict, number> = { exact: 0, pass: 0, boundary: 0, fail: 0 };
  for (const result of results) counts[result.verdict]++;
  const placement = placementAccuracy(renderer, log);
  const report: ParityReport = { renderer: rendererName, fixtures: results, counts, placement };
  if (options.performance !== false) report.performance = await measure(renderer, files, sources, log);
  renderer.dispose();
  return report;
}

/**
 * The shader's placement against exact float64 bilinear, on an opaque high-frequency source, at
 * interior pixels (the rectangle's antialiased rim is excluded: its coverage model is Skia-like but
 * not Skia's). This is what shows the GPU placement is right even where the canvas disagrees.
 */
function placementAccuracy(renderer: PhotoRenderer, log: (line: string) => void): PlacementResult[] {
  const W = 200, H = 240;
  const texture = canvasFrom(W, H, (x, y) => [(x * 37 + y * 11) % 256, (x ^ y) & 255, (x * y) % 251, 255], 11);
  const source = texture.canvas.getContext("2d")!.getImageData(0, 0, W, H).data;
  const path = "fixture:texture";
  const sources = new Map<string, LayerSource>([[path, { image: texture.canvas, width: W, height: H }]]);
  const cases: Array<[string, Partial<PhotoLayer>, number?]> = [
    ["scale 1.6, fractional", { scale: 1.6, x: 0.513, y: 0.487 }],
    ["scale 0.7, fractional", { scale: 0.7, x: 0.4731, y: 0.5129 }],
    ["scale 1, fractional", { x: 0.50313, y: 0.49687 }],
    ["rotate 17, scale 0.9", { rotation: 17, scale: 0.9, x: 0.45, y: 0.52 }],
    ["rotate 90", { rotation: 90, x: 0.5, y: 0.5 }],
    ["crop, scale 1.3", { scale: 1.3, cropLeft: 10, cropTop: 7, cropRight: 5, cropBottom: 12, x: 0.51, y: 0.49 }],
    // VFX Phase 4: a position that comes from a sub-pixel track (the reference uses the float64 attached placement).
    ["attached to a track, frame 17", { scale: 0.85, rotation: 3, attach: { trackId: TRACK_ID, anchorFrame: 0, lag: 1 } }, 17],
    ["attached to a track, frame 9", { scale: 1.2, attach: { trackId: TRACK_ID, anchorFrame: 4 } }, 9],
  ];
  const width = 480, height = 320;
  const results: PlacementResult[] = [];
  for (const [name, over, frame = 0] of cases) {
    const own = createLayer({ id: "texture", name: "Texture", path }, over);
    // The tracked clip is hidden: it only carries the track's placement.
    const document = doc([{ ...createMediaLayer("Clip", CLIP, { scale: 1.6, rotation: 4, visible: false }), id: TRACKED_CLIP_ID }, own],
      { canvasWidth: width, canvasHeight: height, tracks: [fixtureTrack()] });
    const layer = attachedLayer(document, own, frame, { width, height }, compositionRate(document));
    const out = renderer.compose({ document, width, height, sources, frame, resourcesFor: () => emptyResources() }).pixels;
    const rect = croppedRect(layer, { width: W, height: H });
    const inverse = invert(layerMatrix(layer, width, height, { width: W, height: H }, 1));
    const tex = (x: number, y: number, c: number) => {
      const cx = Math.min(Math.ceil(rect.left + rect.width) - 1, Math.max(Math.floor(rect.left), x));
      const cy = Math.min(Math.ceil(rect.top + rect.height) - 1, Math.max(Math.floor(rect.top), y));
      return source[(cy * W + cx) * 4 + c];
    };
    let interior = 0, maxError = 0, over1 = 0;
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const lx = inverse.a * (x + 0.5) + inverse.c * (y + 0.5) + inverse.e;
        const ly = inverse.b * (x + 0.5) + inverse.d * (y + 0.5) + inverse.f;
        const margin = 1.5 / Math.max(0.1, layer.scale);
        if (lx < margin || ly < margin || lx > rect.width - margin || ly > rect.height - margin) continue;
        interior++;
        const u = rect.left + lx - 0.5, v = rect.top + ly - 0.5;
        const x0 = Math.floor(u), y0 = Math.floor(v), fx = u - x0, fy = v - y0;
        for (let c = 0; c < 4; c++) {
          const want = (tex(x0, y0, c) * (1 - fx) + tex(x0 + 1, y0, c) * fx) * (1 - fy)
            + (tex(x0, y0 + 1, c) * (1 - fx) + tex(x0 + 1, y0 + 1, c) * fx) * fy;
          const error = Math.abs(out[(y * width + x) * 4 + c] - want);
          maxError = Math.max(maxError, error);
          if (error > 1) over1++;
        }
      }
    }
    results.push({ name, interiorPixels: interior, maxError: Math.round(maxError * 100) / 100, over1 });
    log(`placement ${name}: max ${maxError.toFixed(2)} against float64 bilinear over ${interior} interior pixels (>1: ${over1})`);
  }
  return results;
}

/**
 * Slider-drag frame time at the 1500 px preview: the same document with its exposure changing on
 * every frame, old path against new. Captures nothing, like a drag with no mask overlay.
 */
async function measure(renderer: PhotoRenderer, files: Map<string, Loaded>, sources: Map<string, LayerSource>, log: (line: string) => void): Promise<PerformanceResult> {
  const width = 1500, height = 1000;
  const big = canvasFrom(width, height, (x, y, random) => [x % 256, y % 256, (random() * 255) | 0, 255], 9);
  const path = "fixture:big";
  const bigFiles = new Map(files).set(path, big);
  const bigSources = new Map(sources).set(path, { image: big.canvas, width, height });
  const frames = 12;
  const documentAt = (frame: number) => ({
    ...createPhotoDocument("big", path), nodes: [], masks: [],
    layers: [createLayer({ id: "big", name: "Big", path }, { nodes: [gradeWith({ exposure: frame / frames, contrast: 20, saturation: 120 })] })],
  });
  const time = (run: (input: ComposeInput) => void) => {
    const started = performance.now();
    for (let frame = 0; frame < frames; frame++) {
      const document = documentAt(frame);
      run({ document, width, height, sources: bigSources, resourcesFor: resourcesFor(document, bigFiles, width, height) });
    }
    return (performance.now() - started) / frames;
  };
  renderer.compose({ document: documentAt(0), width, height, sources: bigSources, resourcesFor: resourcesFor(documentAt(0), bigFiles, width, height) });
  const oldMs = time((input) => composeDocument(input));
  const gpuMs = time((input) => renderer.compose(input));
  log(`performance ${width}x${height}: old ${oldMs.toFixed(1)} ms/frame, gpu ${gpuMs.toFixed(1)} ms/frame`);

  // VFX Phase 2: the same drag with a masked grade -- a three-entry stack with strokes, a colour range
  // (the GPU samples it, the CPU needs a readback) and a feather -- on a layer cut by a track matte.
  const stack = {
    ...createMask("perf"), feather: 6,
    strokes: [{ points: [[0.2, 0.3], [0.5, 0.4], [0.8, 0.35]] as Array<[number, number]>, radius: 0.06, hardness: 0.5, erase: false }],
    shape: { kind: "radial" as const, cx: 0.5, cy: 0.5, radius: 0.3, aspect: 1.2, feather: 0.4 },
    stack: [
      createMaskEntry({ kind: "shape", shape: { kind: "colour", r: 120, g: 120, b: 120, tolerance: 40, softness: 1 } }, { mode: "subtract", feather: 3 }),
      createMaskEntry({ kind: "shape", shape: { kind: "linear", x1: 0, y1: 0, x2: 1, y2: 1 } }, { mode: "intersect", expand: 2 }),
    ],
  };
  const matteLayer = createLayer({ id: "big-matte", name: "Matte", path }, { visible: false, nodes: [gradeWith({ exposure: 0.5 })] });
  const track = { ...createMask("track"), feather: 0, stack: [createMaskEntry({ kind: "layer", layerId: matteLayer.id, channel: "luma" })] };
  const maskedAt = (frame: number): PhotoDocument => ({
    ...createPhotoDocument("big", path), nodes: [], masks: [],
    layers: [matteLayer, createLayer({ id: "big", name: "Big", path }, {
      masks: [stack, track], maskId: track.id,
      nodes: [gradeWith({ exposure: frame / frames, contrast: 20, saturation: 120 }, stack.id)],
    })],
  });
  const timeMasked = (run: (input: ComposeInput) => void) => {
    const started = performance.now();
    for (let frame = 0; frame < frames; frame++) {
      const document = maskedAt(frame);
      run({ document, width, height, sources: bigSources, resourcesFor: resourcesFor(document, bigFiles, width, height) });
    }
    return (performance.now() - started) / frames;
  };
  renderer.compose({ document: maskedAt(0), width, height, sources: bigSources, resourcesFor: resourcesFor(maskedAt(0), bigFiles, width, height) });
  const maskedOld = timeMasked((input) => composeDocument(input));
  const maskedGpu = timeMasked((input) => renderer.compose(input));
  log(`performance ${width}x${height}, masked (3-entry stack + luma track matte): old ${maskedOld.toFixed(1)} ms/frame, gpu ${maskedGpu.toFixed(1)} ms/frame`);
  return {
    size: `${width}x${height}`, frames,
    oldMsPerFrame: Math.round(oldMs * 10) / 10, gpuMsPerFrame: Math.round(gpuMs * 10) / 10,
    maskedOldMsPerFrame: Math.round(maskedOld * 10) / 10, maskedGpuMsPerFrame: Math.round(maskedGpu * 10) / 10,
    note: "GPU time includes the synchronous readback Photo Edit's call site needs. In the container this is SwiftShader (CPU), so only numbers from a real GPU mean anything.",
  };
}

