/**
 * Walking the node stack.
 *
 * Every node produces a full-frame result and is then blended back through its mask, rather than
 * each node being handed only the pixels it is allowed to touch. That costs a little work on masked
 * nodes and buys correctness on the ones that read their neighbours: a blur or an unsharp mask
 * confined to a masked region would sample the unblurred surroundings and leave a seam exactly on
 * the mask edge, which is the classic way this goes wrong.
 *
 * The same function renders the preview and the saved file. Only the resolution differs -- see
 * `photoEdit.ts` for why there is no second implementation anywhere.
 */

import { applyEnvelopeRaster, isIdentityEnvelope } from "./photoEnvelope";
import { applyClipEffects, isNeutralClipEffects } from "./photoClipEffects";
import type { PhotoDocument, PhotoNode } from "./photoDocument";
import { applyAdjustments } from "./photoEdit";
import { applyRelight } from "./photoRelight";
import { applyRetouch } from "./photoRetouch";
import { applyWarp } from "./photoWarp";
import { applyGradientMap } from "./photoGradient";
import { applyLumaKey, applyUnmultiply } from "./photoKey";
import { applyAlphaCurve } from "./photoTime";
import { applyTrail, type TrailField } from "./photoTrack";
import { mediaFrameKey } from "./photoTime";
import { renderMaskFx } from "./maskFx";
import type { Matrix } from "./photoGeometry";

export type RenderResources = {
  /** Coverage per mask id, rasterised at the working resolution. */
  masks: Map<string, Uint8Array>;
  /** Depth for this picture at the working resolution, if it has been computed. */
  depth?: Uint8Array;
  /** Finished bitmaps from the removal worker, keyed by node id, at the working resolution. */
  removals: Map<string, Uint8ClampedArray>;
  /**
   * VFX Phase 2, for the GPU renderer: placed mask files by `matteKey`, when the scope asked for
   * files only (`ComposeScope.filesOnly`) and rasterises the stacks itself.
   */
  mattes?: Map<string, Uint8Array>;
  /**
   * VFX Phase 4: each trail node's displacement at this frame, by node id. Built by the compositor
   * (it knows the frame, the tracks and the layer's placement) with `trailField`, and read by both
   * compositors, so they share the motion exactly. A node with no entry has no track.
   */
  trails?: Map<string, TrailField>;
  /** VFX Phase 7: working pixels per canvas pixel, for nodes whose sizes are canvas pixels. */
  canvasScale?: number;
  /** VFX Phase 7: the composition frame being rendered. */
  frame?: number;
  /** VFX Phase 7: a loaded picture or media frame by key (the compose input's `sources`). */
  pictures?: (key: string) => { image: CanvasImageSource; width: number; height: number } | undefined;
  /** VFX Phase 7: each mask VFX node's placed subject matte and source -> working matrix, by node id. */
  maskFx?: Map<string, { matte: Uint8Array; toWork: Matrix }>;
};

/** Reported on a mask VFX node whose matte (or effect clip frame) has not loaded. */
export const MASK_FX_NO_MATTE = "its matte is not loaded yet";
export const MASK_FX_NO_CLIP = "its effect clip frame is not loaded yet";

/** Straight RGBA of a loaded picture, read once per picture (the last few are kept). */
const readPictures = new Map<string, { data: Uint8ClampedArray; width: number; height: number }>();
function pictureData(key: string, resources: RenderResources): { data: Uint8ClampedArray; width: number; height: number } | undefined {
  const held = readPictures.get(key);
  if (held) return held;
  const picture = resources.pictures?.(key);
  if (!picture || typeof window === "undefined") return undefined;
  const canvas = window.document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(picture.width));
  canvas.height = Math.max(1, Math.round(picture.height));
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.drawImage(picture.image, 0, 0, canvas.width, canvas.height);
  const read = { data: context.getImageData(0, 0, canvas.width, canvas.height).data, width: canvas.width, height: canvas.height };
  if (readPictures.size > 4) readPictures.clear();
  readPictures.set(key, read);
  return read;
}

/**
 * VFX Phase 7: one mask VFX node over `current` -- the SAME function in both compositors, since the
 * GPU runs it on its CPU path. Undefined (with a report) while the matte or clip frame is loading.
 */
export function runMaskVfx(
  node: Extract<PhotoNode, { kind: "maskVfx" }>, current: Uint8ClampedArray, width: number, height: number,
  resources: RenderResources, report: RenderReport,
): Uint8ClampedArray | undefined {
  const placed = resources.maskFx?.get(node.id);
  if (!placed || placed.matte.length !== width * height) {
    report.skipped.push({ nodeId: node.id, reason: MASK_FX_NO_MATTE });
    return undefined;
  }
  let effect: { data: Uint8ClampedArray; width: number; height: number } | undefined;
  if (node.recipe.matteStyle === "clip" && node.effectClip) {
    effect = pictureData(mediaFrameKey(node.effectClip.media, node.effectClip.index, false), resources);
    if (!effect) {
      report.skipped.push({ nodeId: node.id, reason: MASK_FX_NO_CLIP });
      return undefined;
    }
  }
  return renderMaskFx({
    pixels: current, width, height, recipe: node.recipe, index: node.index, count: node.count, fps: node.fps,
    envelope: node.envelope, matte: placed.matte, toWork: placed.toWork,
    sourceWidth: node.sourceWidth, sourceHeight: node.sourceHeight, effect,
  });
}

/** Reported on a trail node whose track is missing or was never seen. */
export const TRAIL_NO_TRACK = "its track is missing";

export const emptyResources = (): RenderResources => ({ masks: new Map(), removals: new Map() });

/**
 * Which nodes could not be drawn, and why.
 *
 * Returned rather than thrown: a relight node with no depth map yet is a normal state during the
 * seconds the worker is running, not an error, and the workspace shows it as a badge on the node
 * instead of an alert over the picture.
 */
export type RenderReport = { skipped: Array<{ nodeId: string; reason: string }> };

export function renderDocument(
  base: Uint8ClampedArray, width: number, height: number,
  document: PhotoDocument, resources: RenderResources,
): { pixels: Uint8ClampedArray; report: RenderReport } {
  return renderStack(base, width, height, document.nodes, resources);
}

/**
 * The same walk over an arbitrary stack.
 *
 * A layer owns a stack and so does the composite, and they are the same thing run over different
 * pixels. Taking the nodes rather than the document is what keeps that literally true instead of
 * approximately true -- there is one implementation, and a grade on a cutout and a grade on the
 * finished frame cannot drift apart.
 */
export function renderStack(
  base: Uint8ClampedArray, width: number, height: number,
  nodes: PhotoNode[], resources: RenderResources,
): { pixels: Uint8ClampedArray; report: RenderReport } {
  const current = new Uint8ClampedArray(base);
  const report: RenderReport = { skipped: [] };

  for (const node of nodes) {
    if (!node.enabled) continue;
    const layer = renderNode(node, current, width, height, resources, report);
    if (!layer) continue;
    const coverage = node.maskId ? resources.masks.get(node.maskId) : undefined;
    if (node.maskId && !coverage) {
      report.skipped.push({ nodeId: node.id, reason: "its mask has not been drawn yet" });
      continue;
    }
    blend(current, layer, coverage);
  }
  return { pixels: current, report };
}

/** Produce a node's full-frame result, or undefined when it cannot run yet. */
function renderNode(
  node: PhotoNode, current: Uint8ClampedArray, width: number, height: number,
  resources: RenderResources, report: RenderReport,
): Uint8ClampedArray | undefined {
  if (node.kind === "grade") {
    const layer = new Uint8ClampedArray(current);
    applyAdjustments(layer, width, height, node.adjustments);
    return layer;
  }
  if (node.kind === "retouch") {
    if (!node.strokes.length) return undefined;
    const layer = new Uint8ClampedArray(current);
    // Sampling from its own input is what makes a retouch node well defined wherever it sits in the
    // stack, and what lets a later stroke clean up after an earlier one.
    applyRetouch(layer, width, height, node.strokes);
    return layer;
  }
  if (node.kind === "warp") {
    if (!node.strokes.length) return undefined;
    const layer = new Uint8ClampedArray(current);
    // Warps its own input, so a warp reads whatever the stack has produced up to that point and can
    // be reordered against a grade without either one meaning something different.
    applyWarp(layer, width, height, node.strokes);
    return layer;
  }
  if (node.kind === "clipEffects") {
    // VFX Phase 7: the GPU compositor runs this same function on its CPU path.
    if (isNeutralClipEffects(node.effects)) return undefined;
    const layer = new Uint8ClampedArray(current);
    applyClipEffects(layer, width, height, node.effects, resources.canvasScale ?? 1);
    return layer;
  }
  if (node.kind === "maskVfx") return runMaskVfx(node, current, width, height, resources, report);
  if (node.kind === "envelope") {
    // VFX Phase 6: the GPU compositor runs this same function on its CPU path.
    if (isIdentityEnvelope(node.envelope)) return undefined;
    const layer = new Uint8ClampedArray(current);
    applyEnvelopeRaster(layer, width, height, node.envelope);
    return layer;
  }
  if (node.kind === "gradientMap") {
    if (node.amount <= 0) return undefined;
    const layer = new Uint8ClampedArray(current);
    applyGradientMap(layer, node.gradient, node.amount);
    return layer;
  }
  if (node.kind === "unmultiply") {
    const layer = new Uint8ClampedArray(current);
    applyUnmultiply(layer);
    return layer;
  }
  if (node.kind === "alphaCurve") {
    const layer = new Uint8ClampedArray(current);
    applyAlphaCurve(layer, node.gain, node.gamma);
    return layer;
  }
  if (node.kind === "lumaKey") {
    const layer = new Uint8ClampedArray(current);
    applyLumaKey(layer, node);
    return layer;
  }
  if (node.kind === "trail") {
    const field = resources.trails?.get(node.id);
    if (!field) {
      report.skipped.push({ nodeId: node.id, reason: TRAIL_NO_TRACK });
      return undefined;
    }
    // A track that has not moved over the lag moves nothing; the node is skipped silently.
    if (field.still) return undefined;
    return applyTrail(current, width, height, field);
  }
  if (node.kind === "relight") {
    if (!resources.depth) {
      report.skipped.push({ nodeId: node.id, reason: "no depth map yet" });
      return undefined;
    }
    const layer = new Uint8ClampedArray(current);
    applyRelight(layer, width, height, resources.depth, node.relight);
    return layer;
  }
  const removal = resources.removals.get(node.id);
  if (!removal || removal.length !== current.length) {
    report.skipped.push({ nodeId: node.id, reason: "not filled yet" });
    return undefined;
  }
  return removal;
}

/**
 * Composite `layer` over `current` through `coverage`.
 *
 * With no mask this is a straight copy, which is the common case and worth not paying for a
 * per-pixel lerp. Alpha is carried from the layer either way, so a node that changes transparency
 * still works inside a mask.
 */
export function blend(
  current: Uint8ClampedArray, layer: Uint8ClampedArray, coverage: Uint8Array | undefined,
): void {
  if (!coverage) {
    current.set(layer);
    return;
  }
  for (let pixel = 0, at = 0; pixel < coverage.length; pixel++, at += 4) {
    const amount = coverage[pixel];
    if (amount === 0) continue;
    if (amount === 255) {
      current[at] = layer[at];
      current[at + 1] = layer[at + 1];
      current[at + 2] = layer[at + 2];
      current[at + 3] = layer[at + 3];
      continue;
    }
    const mix = amount / 255;
    current[at] += (layer[at] - current[at]) * mix;
    current[at + 1] += (layer[at + 1] - current[at + 1]) * mix;
    current[at + 2] += (layer[at + 2] - current[at + 2]) * mix;
    current[at + 3] += (layer[at + 3] - current[at + 3]) * mix;
  }
}

/** Nearest-neighbour resample of a coverage or depth buffer. Good enough for both: a mask is about
 *  to be feathered anyway, and a depth map is already a smooth estimate. */
export function resampleSingleChannel(
  source: Uint8Array, sourceWidth: number, sourceHeight: number,
  width: number, height: number,
): Uint8Array {
  if (sourceWidth === width && sourceHeight === height) return source;
  const target = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) {
    const sourceY = Math.min(sourceHeight - 1, Math.floor(y * sourceHeight / height));
    for (let x = 0; x < width; x++) {
      const sourceX = Math.min(sourceWidth - 1, Math.floor(x * sourceWidth / width));
      target[y * width + x] = source[sourceY * sourceWidth + sourceX];
    }
  }
  return target;
}
