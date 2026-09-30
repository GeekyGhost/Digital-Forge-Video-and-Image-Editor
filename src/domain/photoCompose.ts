/**
 * Turning a stack of layers into one picture.
 *
 * This is the merge the whole Photo Edit design was pointing at. Before it there were two canvases
 * in this app -- Image Studio's, which stacked pictures but could not adjust them, and Photo Edit's,
 * which adjusted one picture but could not stack -- and each had its own idea of what a layer was.
 * Two implementations of the same maths is this project's recurring bug class, so there is now one:
 * every layer is placed, graded and blended here, and the preview and the saved file are this
 * function called at two resolutions.
 *
 * Placement uses the browser's own drawImage for resampling rather than a hand-written sampler.
 * That is a deliberate choice: the browser's is better than anything worth writing here, and it is
 * the same resampler at both resolutions, so a layer does not shift half a pixel between preview
 * and export.
 */

import { compositeOver } from "./photoBlend";
import { compose, rotateBy, scaleBy, translate, type Matrix } from "./photoGeometry";
import type { PhotoDocument, PhotoLayer, PhotoMask, PhotoNode } from "./photoDocument";
import { layersOf } from "./photoDocument";
import { applyLightWrap, buildContactShadow, defringeLayer, isNeutralEdge } from "./photoEdge";
import { renderStack, type RenderReport, type RenderResources } from "./photoRender";
import {
  fitFontSize, fontSpec, lineOrigin, paintsOf, textBoxSize, wrapLines, type TextBlock,
} from "./photoText";
import { renderGradient, type Gradient, type Paint } from "./photoGradient";
import { pathBounds, pathBox, segmentsOf, shapedPath, type PathBlock, type Point as PathPoint } from "./photoPath";
import { expandRepeats } from "./photoRepeat";
import { drawTimelineGraphic, type TimelineGraphicBlock } from "./timelineGraphic";
import { buildRainAtlas, rainLayout, renderMatrixRain } from "./matrixRain";
import {
  depthFrameOf, emitterParticles, placedLayers, renderEmitter, trailField, type CanvasSize, type DepthFrameRef,
} from "./photoTrack";
import {
  compositionRate, crossfadeFrames, mediaFrameKey, mediaFramesAt, scopeFrame, type MediaFrame,
} from "./photoTime";
import {
  shapeBounds, shapeInnerBox, shapePaths, type ShapeBlock,
} from "./photoShape";

/** Anything the browser can draw, with its intrinsic size. */
export type LayerSource = { image: CanvasImageSource; width: number; height: number };

/**
 * One stack and the masks it can reach, plus the layer it belongs to when it belongs to one.
 *
 * The layer travels with the scope because a cached generative fill was produced in the layer's own
 * pixels and has to be placed through the same matrix the layer was, or a fill lands correctly on an
 * untransformed plate and slides off a scaled cutout.
 */
export type ComposeScope = {
  nodes: PhotoNode[]; masks: PhotoMask[]; layer?: PhotoLayer;
  /** The proxy ratio this render is running at, so a placed file lands where its layer does. */
  canvasScale?: number;
  /**
   * VFX Phase 2. Set by the GPU renderer, which rasterises mask stacks itself: supply the placed
   * mask files (`RenderResources.mattes`) instead of rasterised masks.
   */
  filesOnly?: boolean;
  /**
   * VFX Phase 2. Another layer's finished pixels (after its placement, nodes and masks) and its
   * opacity, for a `layer` mask entry -- a track matte. Undefined when it is missing, still loading,
   * or would read the layer asking (a cycle).
   */
  layerSource?: (layerId: string) => { pixels: Uint8ClampedArray; opacity: number } | undefined;
  /** Rasterise this mask even when nothing in the scope reads it (the mask overlay). */
  captureMask?: string;
  /** Frame index for sequence mask sources. Photo Edit renders frame 0; Phase 3 passes the time. */
  frame?: number;
  /**
   * VFX Phase 4: the depth frame a `depth` mask entry reads for a layer at this render's composition
   * frame (with the layer as placed), or undefined when there is none. Set by the compositors.
   */
  depthFrame?: (layerId: string) => DepthFrameRef | undefined;
};

export type ComposeInput = {
  document: PhotoDocument;
  width: number;
  height: number;
  /** Loaded pictures by the path the layer names. A missing one leaves that layer out. */
  sources: Map<string, LayerSource>;
  /**
   * Masks, depth and fills for one scope, at the working resolution. `pixels` is that scope's
   * ungraded input, because a colour-range mask matches against it -- handing over the wrong buffer
   * is how a selection works on screen and vanishes from the export.
   */
  resourcesFor: (scope: ComposeScope, pixels: Uint8ClampedArray) => RenderResources;
  /**
   * Keep the ungraded pixels and the backdrop for ONE scope -- a layer id, or `COMPOSITE_SCOPE`.
   *
   * One rather than all: the eyedropper, the mask overlay and match-colour each want the exact
   * buffer the render used for a single scope, and copying every layer's frame twice on every
   * slider tick would cost more than the render does.
   */
  capture?: string;
  /**
   * VFX Phase 2: return this mask's combined coverage exactly as the render used it (the overlay
   * tint), including layer entries that only the compositor can resolve.
   */
  captureMask?: string;
  /**
   * The composition frame to render (VFX Phase 3). Media layers show the source frames
   * `mediaFramesAt` names -- which the caller must have loaded into `sources` under `mediaFrameKey`
   * -- generators render this frame, and mask sequences follow it. Undefined is frame 0: a still.
   */
  frame?: number;
  /** Read media frames from their proxies (`mediaFrameKey(..., true)`): scrubbing, not saving. */
  useProxy?: boolean;
};

export type ComposeResult = {
  pixels: Uint8ClampedArray;
  report: RenderReport;
  /** Per layer id, only when `capture` was asked for. */
  raw: Map<string, Uint8ClampedArray>;
  backdrops: Map<string, Uint8ClampedArray>;
  /** The `captureMask` coverage, one byte per pixel, when it was asked for and could be drawn. */
  maskCoverage?: Uint8Array;
};

/**
 * The canvas size a document renders at, at full resolution.
 *
 * Explicit if the document carries one, otherwise the bottom layer's own size: opening a picture and
 * dropping a cutout on it should give you the picture's dimensions, not a guess.
 */
export function canvasSizeOf(
  document: PhotoDocument, sources: Map<string, LayerSource>,
): { width: number; height: number } | undefined {
  if (document.canvasWidth && document.canvasHeight) {
    return { width: document.canvasWidth, height: document.canvasHeight };
  }
  for (const layer of layersOf(document)) {
    // A media layer's size is its own, known without loading a frame.
    if (layer.media) return { width: layer.media.width, height: layer.media.height };
    // Text, gradients and generators are drawn to fit whatever they are placed on, so none can be the
    // thing that decides how big that is -- asking them would make the canvas size depend on the caption.
    if (layer.text || layer.gradient || layer.shape || layer.path || layer.timelineGraphic || layer.generator) continue;
    const source = sources.get(layer.sourcePath);
    if (source) return { width: source.width, height: source.height };
  }
  return undefined;
}

/** The visible rectangle of the source, after the layer's per-side crop. */
export function croppedRect(layer: PhotoLayer, source: { width: number; height: number }) {
  const left = clamp01(layer.cropLeft / 100) * source.width;
  const top = clamp01(layer.cropTop / 100) * source.height;
  const width = Math.max(1, source.width - left - clamp01(layer.cropRight / 100) * source.width);
  const height = Math.max(1, source.height - top - clamp01(layer.cropBottom / 100) * source.height);
  return { left, top, width, height };
}

/**
 * Where a layer sits on the canvas, as a matrix.
 *
 * One matrix rather than a sequence of context calls, because the inverse is needed too: a mask
 * painted on the canvas has to be mapped back into the layer's own pixels before it can be sent to
 * a worker that only knows the source file. Deriving both from the same matrix means they cannot
 * disagree, which hand-written forward and backward transforms reliably do.
 */
export function layerMatrix(
  layer: PhotoLayer, canvasWidth: number, canvasHeight: number,
  source: { width: number; height: number }, canvasScale = 1,
): Matrix {
  const rect = croppedRect(layer, source);
  // `layer.scale` is measured in CANVAS pixels per source pixel, so rendering into a proxy buffer
  // has to carry the proxy's own ratio. Without it the placement is measured in buffer pixels and a
  // layer silently doubles in size between a half-scale preview and the exported file -- the
  // preview/output divergence this whole design is arranged to prevent.
  const k = layer.scale * canvasScale;
  let matrix = translate(-rect.width / 2, -rect.height / 2);
  // VFX Phase 7: `stretchX` scales the width alone (a timeline clip's "stretch" conform, a squeeze).
  matrix = compose(scaleBy(k * (layer.stretchX ?? 1), k), matrix);
  matrix = compose(rotateBy(layer.rotation), matrix);
  return compose(translate(layer.x * canvasWidth, layer.y * canvasHeight), matrix);
}

/** Draw one layer's source into a canvas-sized buffer, placed but otherwise untouched. */
export function placeLayer(
  layer: PhotoLayer, source: LayerSource, width: number, height: number, canvasScale = 1,
): Uint8ClampedArray {
  const canvas = scratch(width, height);
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.clearRect(0, 0, width, height);
  const rect = croppedRect(layer, source);
  const matrix = layerMatrix(layer, width, height, source, canvasScale);
  context.save();
  context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f);
  context.drawImage(source.image, rect.left, rect.top, rect.width, rect.height, 0, 0, rect.width, rect.height);
  context.restore();
  return context.getImageData(0, 0, width, height).data;
}

export function composeDocument(input: ComposeInput): ComposeResult {
  const { width, height, sources, resourcesFor } = input;
  // How much smaller this render is than the document's own canvas. Every placement is measured
  // against the canvas, so the proxy has to carry the ratio or the preview stops being a preview.
  const full = canvasSizeOf(input.document, sources);
  // VFX Phase 6: repeated layers become their copies before anything else reads the layers.
  const document = expandRepeats(input.document, full ?? { width, height });
  const canvasScale = full && full.width > 0 ? width / full.width : 1;
  const count = width * height;
  const pixels = new Uint8ClampedArray(count * 4);
  const report: RenderReport = { skipped: [] };
  const raw = new Map<string, Uint8ClampedArray>();
  const backdrops = new Map<string, Uint8ClampedArray>();
  let maskCoverage: Uint8Array | undefined;

  if (document.background === "black" || document.background === "white") {
    const value = document.background === "white" ? 255 : 0;
    for (let at = 0; at < pixels.length; at += 4) {
      pixels[at] = pixels[at + 1] = pixels[at + 2] = value;
      pixels[at + 3] = 255;
    }
  }

  const frame = Math.max(0, Math.floor(input.frame ?? 0));
  const rate = compositionRate(document);
  const canvas: CanvasSize = full ?? { width, height };
  // VFX Phase 4: every layer as placed at this frame -- attached layers follow their track. Done
  // here, before anything reads a layer, so masks, nodes, mattes and the alpha cut follow for free.
  const layers = placedLayers(document, frame, canvas, rate);
  const byId = new Map(layers.map((layer) => [layer.id, layer]));
  const depthFrame = (layerId: string) => depthFrameOf(byId, layerId, frame, rate);
  const bases = clipBases(layers);
  const referenced = referencedLayers(document);
  // A layer's result after placement, nodes and its alpha cut, computed at most once per render and
  // kept while another layer may read it (a clip base or a track matte). `null` is a layer that
  // could not be drawn; `visiting` is how a matte cycle is noticed.
  const prepared = new Map<string, Uint8ClampedArray | null>();
  const visiting = new Set<string>();

  const layerSource = (layerId: string) => {
    const layer = byId.get(layerId);
    const out = layer ? prepare(layer) : undefined;
    return layer && out ? { pixels: out, opacity: layer.opacity } : undefined;
  };

  function prepare(layer: PhotoLayer): Uint8ClampedArray | undefined {
    const held = prepared.get(layer.id);
    if (held !== undefined) return held ?? undefined;
    if (visiting.has(layer.id)) {
      report.skipped.push({ nodeId: layer.id, reason: MATTE_CYCLE });
      return undefined;
    }
    visiting.add(layer.id);
    const out = prepareOnce(layer);
    visiting.delete(layer.id);
    prepared.set(layer.id, out ?? null);
    return out;
  }

  function prepareOnce(layer: PhotoLayer): Uint8ClampedArray | undefined {
    // A gradient has no intrinsic pixels, so there is nothing to place: it is generated at the
    // working size directly, and its own angle, centre and scale are its placement. Sending it
    // through the layer matrix as well would give two controls for one thing.
    let placed: Uint8ClampedArray;
    let shown: MediaFrame[] | undefined;
    if (layer.media) {
      shown = mediaFramesAt(layer.media, frame, rate);
      if (!shown) {
        // Not on screen at this frame. Still a layer another may read (a matte that has ended is
        // empty, not absent), so it prepares as transparent; the main loop does not draw it.
        placed = new Uint8ClampedArray(count * 4);
      } else {
        const frames: Uint8ClampedArray[] = [];
        for (const one of shown) {
          const source = sources.get(mediaFrameKey(layer.media, one.index, input.useProxy));
          if (!source) {
            report.skipped.push({ nodeId: layer.id, reason: "its picture is still loading" });
            return undefined;
          }
          frames.push(placeLayer(layer, source, width, height, canvasScale));
        }
        placed = frames.length === 2 ? crossfadeFrames(frames[0], frames[1], shown[1].weight) : frames[0];
      }
    } else if (layer.generator?.kind === "emitter") {
      // VFX Phase 4: particles of an element, each placed like a layer, into a transparent buffer.
      placed = new Uint8ClampedArray(count * 4);
      const particles = emitterParticles(document, layer.generator, frame, rate, canvas);
      if (!renderEmitter(placed, particles, layer.generator.element, sources, width, height, canvasScale, input.useProxy)) {
        report.skipped.push({ nodeId: layer.id, reason: "its picture is still loading" });
        return undefined;
      }
    } else if (layer.generator) {
      // Generated at the working size, like a gradient: its cell size is its placement.
      placed = new Uint8ClampedArray(count * 4);
      const cell = rainLayout(layer.generator, width, height, canvasScale).cell;
      renderMatrixRain(placed, width, height, layer.generator, frame, canvasScale, buildRainAtlas(cell, layer.generator.mirror));
    } else if (layer.gradient) {
      placed = new Uint8ClampedArray(count * 4);
      renderGradient(placed, width, height, layer.gradient);
    } else {
      const source = isDrawnLayer(layer) ? drawnSource(layer) : sources.get(layer.sourcePath);
      if (!source) {
        report.skipped.push({ nodeId: layer.id, reason: "its picture is still loading" });
        return undefined;
      }
      placed = placeLayer(layer, source, width, height, canvasScale);
    }
    if (input.capture === layer.id) raw.set(layer.id, new Uint8ClampedArray(placed));

    const captureMask = layer.masks.some((mask) => mask.id === input.captureMask) ? input.captureMask : undefined;
    const resources = resourcesFor({
      nodes: layer.nodes, masks: layer.masks, layer, canvasScale, layerSource, captureMask,
      frame: scopeFrame(layer, shown, frame), depthFrame,
    }, placed);
    resources.trails = trailFields(document, layer, placementSize(layer, sources), frame, rate, canvas, width, height, canvasScale);
    Object.assign(resources, { canvasScale, frame, pictures: (key: string) => sources.get(key) });
    if (captureMask) maskCoverage = resources.masks.get(captureMask);
    const rendered = renderStack(placed, width, height, layer.nodes, resources);
    report.skipped.push(...rendered.report.skipped);
    const graded = rendered.pixels;

    // VFX Phase 2: the layer mask and the clipping base cut the layer's alpha, after its nodes and
    // before the edge group, so a contact shadow or a light wrap follows the shape that is left.
    const mask = layer.maskId ? resources.masks.get(layer.maskId) : undefined;
    const base = bases.get(layer.id);
    const basePixels = base ? prepare(base) : undefined;
    if (base && !basePixels) return undefined;
    if (mask || basePixels) applyAlphaCut(graded, mask, basePixels, base?.opacity ?? 1);
    return graded;
  }

  for (const layer of layers) {
    if (!layer.visible || layer.opacity <= 0) continue;
    // A media layer off screen at this frame draws nothing (it may still be read as a matte).
    if (layer.media && !mediaFramesAt(layer.media, frame, rate)) continue;
    const base = bases.get(layer.id);
    // A clipped layer shows only through its base, so it goes when the base does.
    if (base && (!base.visible || base.opacity <= 0)) continue;
    const prepared = prepare(layer);
    if (!prepared) continue;
    // The edge group writes into the layer, and a layer another one reads (as a clip base or a matte)
    // must be read as it was before that; so it works on a copy.
    const graded = referenced.has(layer.id) ? new Uint8ClampedArray(prepared) : prepared;

    const edge = layer.edge;
    if (edge && !isNeutralEdge(edge)) {
      if (edge.defringe > 0) defringeLayer(graded, width, height, edge.defringe);
      const shadow = buildContactShadow(graded, width, height, edge);
      // Under the subject, and before the wrap is measured, so a layer that casts a shadow also
      // catches light from it. That ordering is the difference between a composite and a sticker.
      if (shadow) compositeOver(pixels, shadow, "normal", 1);
      if (edge.lightWrap > 0) {
        applyLightWrap(graded, pixels, width, height, edge.lightWrap, edge.lightWrapSpread);
      }
    }

    if (input.capture === layer.id) backdrops.set(layer.id, new Uint8ClampedArray(pixels));
    compositeOver(pixels, graded, layer.blendMode ?? "normal", layer.opacity);
  }

  // The stack over the finished frame. A grade here sees every layer at once, which is the only
  // place a look that unifies a composite can honestly live.
  //
  // Only on a LAYERED document. A pre-layers document's top-level stack is read as its one layer's
  // stack by `layersOf`, so it has already been applied above; running it again here graded every
  // such picture twice (found by the VFX Phase 1 parity harness, 2026-09-27).
  if (input.capture === COMPOSITE_SCOPE) raw.set(COMPOSITE_SCOPE, new Uint8ClampedArray(pixels));
  const layered = Boolean(document.layers?.length);
  const compositeNodes = layered ? document.nodes : [];
  const compositeCapture = layered && document.masks.some((mask) => mask.id === input.captureMask)
    ? input.captureMask : undefined;
  if (compositeNodes.length || compositeCapture) {
    const resources = resourcesFor({
      nodes: compositeNodes, masks: document.masks, canvasScale, layerSource, captureMask: compositeCapture, frame, depthFrame,
    }, pixels);
    resources.trails = trailFields(document, undefined, undefined, frame, rate, canvas, width, height, canvasScale, compositeNodes);
    Object.assign(resources, { canvasScale, frame, pictures: (key: string) => sources.get(key) });
    if (compositeCapture) maskCoverage = resources.masks.get(compositeCapture);
    const rendered = renderStack(pixels, width, height, compositeNodes, resources);
    pixels.set(rendered.pixels);
    report.skipped.push(...rendered.report.skipped);
  }

  return { pixels, report, raw, backdrops, maskCoverage };
}

/**
 * VFX Phase 4: the trail fields of every enabled trail node in a scope (see `trailField`), keyed by
 * node id; undefined when the scope has none. Shared by both compositors.
 */
export function trailFields(
  document: PhotoDocument, layer: PhotoLayer | undefined, size: CanvasSize | undefined, frame: number, rate: number,
  canvas: CanvasSize, width: number, height: number, canvasScale: number, nodes: PhotoNode[] = layer?.nodes ?? [],
) {
  let out: Map<string, NonNullable<ReturnType<typeof trailField>>> | undefined;
  for (const node of nodes) {
    if (node.kind !== "trail" || !node.enabled) continue;
    const field = trailField(document, node, layer, size, frame, rate, canvas, width, height, canvasScale);
    if (field) (out ??= new Map()).set(node.id, field);
  }
  return out;
}

/**
 * The source size a layer is placed through `layerMatrix` with: its media's, its drawn text or
 * shape's, or its picture's. Undefined for layers drawn at the working size (gradients, generators)
 * and for a picture that has not loaded.
 */
export function placementSize(layer: PhotoLayer, sources: Map<string, LayerSource>): CanvasSize | undefined {
  if (layer.media) return { width: layer.media.width, height: layer.media.height };
  if (layer.generator || layer.gradient) return undefined;
  const source = isDrawnLayer(layer) ? drawnSource(layer) : sources.get(layer.sourcePath);
  return source ? { width: source.width, height: source.height } : undefined;
}

/** Reported on a layer whose track matte (or clip base) would, through other layers, read itself. */
export const MATTE_CYCLE = "its matte refers back to itself";

/**
 * The base of each clipped layer: the nearest layer below that is not itself clipped (Photoshop's
 * clipping group). A clipped bottom layer has no base and is drawn unclipped.
 */
export function clipBases(layers: PhotoLayer[]): Map<string, PhotoLayer> {
  const bases = new Map<string, PhotoLayer>();
  let base: PhotoLayer | undefined;
  for (const layer of layers) {
    if (layer.clip && base) bases.set(layer.id, base);
    else if (!layer.clip) base = layer;
  }
  return bases;
}

/** Layers another layer or the composite reads: clip bases and every `layer` mask entry's layer. */
export function referencedLayers(document: PhotoDocument): Set<string> {
  const layers = layersOf(document);
  const ids = new Set<string>();
  for (const base of clipBases(layers).values()) ids.add(base.id);
  const masks = [...document.masks, ...layers.flatMap((layer) => layer.masks)];
  for (const mask of masks) {
    for (const entry of mask.stack ?? []) if (entry.source.kind === "layer") ids.add(entry.source.layerId);
  }
  return ids;
}

/**
 * Cut a layer's alpha by its layer mask and its clipping base, in place:
 * a' = a * ((mask / 255) * (baseAlpha / 255 * baseOpacity)), rounded once as a Uint8ClampedArray
 * does. The GPU's ALPHA_CUT_FS multiplies in the same order.
 */
export function applyAlphaCut(
  pixels: Uint8ClampedArray, mask: Uint8Array | undefined, base: Uint8ClampedArray | undefined, baseOpacity: number,
): void {
  const opacity = Math.min(1, Math.max(0, baseOpacity));
  for (let pixel = 0, at = 0; at < pixels.length; pixel++, at += 4) {
    const m = mask ? mask[pixel] / 255 : 1;
    const c = base ? (base[at + 3] / 255) * opacity : 1;
    pixels[at + 3] = pixels[at + 3] * (m * c);
  }
}

/** The key the composite's own ungraded frame is captured under. */
export const COMPOSITE_SCOPE = "__composite__";

const clamp01 = (value: number) => value < 0 ? 0 : value > 0.95 ? 0.95 : value;

/**
 * One reusable canvas for placing layers into.
 *
 * Kept rather than allocated per layer per frame: at preview size that is several megabytes of
 * canvas backing store per layer on every slider tick, and the browser is not obliged to reclaim it
 * promptly. Safe to share because `placeLayer` reads the pixels out before it returns, so no two
 * callers ever hold the canvas at once.
 */
let shared: HTMLCanvasElement | undefined;

function scratch(width: number, height: number): HTMLCanvasElement {
  if (!shared) shared = window.document.createElement("canvas");
  if (shared.width !== width || shared.height !== height) {
    shared.width = width;
    shared.height = height;
  }
  return shared;
}

/* --------------------------------------------------- shapes, text and ink */

/**
 * A drawn layer -- a shape, some lettering, or a balloon which is both.
 *
 * Cached on everything that affects it. Drawing one means laying out a paragraph and building a
 * ramp per paint, and while someone is typing that happens on every keystroke; but placement,
 * opacity, blend mode and the layer's whole node stack change far more often than the words do,
 * and re-laying-out on every slider tick would be the expensive thing in the frame.
 */
const drawn = new Map<string, LayerSource>();
const graphics = new Map<string, LayerSource>();

export function drawnSource(layer: {
  text?: TextBlock; shape?: ShapeBlock; path?: PathBlock; timelineGraphic?: TimelineGraphicBlock;
}): LayerSource | undefined {
  // VFX Phase 7: a timeline graphic changes every frame and is frame-sized, so only the last few are
  // kept (placement, the CPU and the GPU compositor each ask for the same frame's bitmap).
  if (layer.timelineGraphic) {
    const key = JSON.stringify(layer.timelineGraphic);
    const held = graphics.get(key);
    if (held) return held;
    const made = drawTimelineGraphic(layer.timelineGraphic);
    if (graphics.size > 6) graphics.clear();
    graphics.set(key, made);
    return made;
  }
  if (!layer.text && !layer.shape && !layer.path) return undefined;
  // A path layer draws only its path (VFX Phase 6); its key leaves the others out.
  const key = layer.path ? JSON.stringify(["path", layer.path]) : JSON.stringify([layer.shape ?? 0, layer.text ?? 0]);
  const held = drawn.get(key);
  if (held) return held;

  const source = layer.path ? drawPath(layer.path) : layer.shape ? drawShape(layer.shape, layer.text) : drawText(layer.text!);
  if (drawn.size > 24) drawn.clear();
  drawn.set(key, source);
  return source;
}

/**
 * Layers the app draws into a bitmap of their own and then places like a picture: text, shapes,
 * balloons and (VFX Phase 6) paths. Media, generators and gradients are handled before this.
 */
export const isDrawnLayer = (layer: { text?: TextBlock; shape?: ShapeBlock; path?: PathBlock; timelineGraphic?: TimelineGraphicBlock }): boolean =>
  Boolean(layer.text || layer.shape || layer.path || layer.timelineGraphic);

/** Kept for the callers that only ever have text. */
export const textSource = (text: TextBlock): LayerSource => drawnSource({ text })!;

/**
 * A paint, as something the canvas can fill with.
 *
 * A gradient becomes a PATTERN drawn by `renderGradient` at the exact size of the thing being
 * painted, which is the whole reason gradients on text look like gradients on the canvas: it is
 * literally the same function. Going through the canvas's own `createLinearGradient` instead would
 * have been three lines and would have quietly dropped the angular and diamond shapes, the repeat,
 * the OKLab interpolation and the dithering.
 */
function paintStyle(
  context: CanvasRenderingContext2D, paint: Paint, width: number, height: number,
): string | CanvasPattern {
  if (paint.kind === "solid") return paint.color;
  const tile = window.document.createElement("canvas");
  tile.width = Math.max(1, Math.round(width));
  tile.height = Math.max(1, Math.round(height));
  const inner = tile.getContext("2d")!;
  const image = inner.createImageData(tile.width, tile.height);
  renderGradient(image.data, tile.width, tile.height, paint.gradient);
  inner.putImageData(image, 0, 0);
  return context.createPattern(tile, "no-repeat") ?? "#ffffff";
}

/** How many lines the text makes at a given size, and how wide the widest of them is. */
function measureAt(
  context: CanvasRenderingContext2D, text: TextBlock, size: number,
): { width: number; height: number; lines: string[] } {
  const sized = { ...text, fontSize: size, padding: 0 };
  context.font = fontSpec(sized);
  applySpacing(context, sized);
  const lines = wrapLines({ ...sized, boxWidth: text.boxWidth }, (line) => context.measureText(line).width);
  let widest = 0;
  for (const line of lines) widest = Math.max(widest, context.measureText(line).width);
  return {
    width: widest,
    height: lines.length * Math.max(1, size * Math.max(0.5, text.lineHeight)),
    lines,
  };
}

/**
 * Lay the lettering out and paint it, into a context whose origin is the text box's corner.
 *
 * Shared by the free-standing text layer and the balloon, so a caption and a speech bubble letter
 * identically -- which is the point of a balloon being a shape that contains text rather than a
 * separate kind of thing with its own text code.
 */
function paintText(
  context: CanvasRenderingContext2D, text: TextBlock,
  box: { x: number; y: number; width: number; height: number },
): void {
  const paints = paintsOf(text);
  const layout = textLayout(context, text, box);
  const { sized, x, top, step, lines } = layout;
  const wrapWidth = box.width;

  context.save();
  context.translate(box.x, box.y);
  layout.apply(context);

  if (text.shadow > 0) drawTextShadow(context, sized, lines, x, top, step, wrapWidth, box.height, paints.shadow);

  for (let index = 0; index < lines.length; index++) {
    const y = top + step * (index + 0.5);
    // Outline first and centred on the glyph edge, so half of it hides under the fill. Stroking on
    // top would eat into the letterforms and thin type would close up.
    if (text.strokeWidth > 0) {
      context.lineWidth = text.strokeWidth * 2;
      context.strokeStyle = paintStyle(context, paints.stroke, wrapWidth, box.height);
      context.lineJoin = "round";
      context.miterLimit = 2;
      context.strokeText(lines[index], x, y);
    }
    context.fillStyle = paintStyle(context, paints.fill, wrapWidth, box.height);
    context.fillText(lines[index], x, y);
  }
  context.restore();
}

/**
 * Where lettering goes in its box: the size (auto-fit or not), the wrapped lines, the line step,
 * the alignment origin and the first line's top. Shared by `paintText` and, in VFX Phase 6, by
 * "Convert text to outlines" (`textGlyphRuns`), so outlines land on the letters' own pixels.
 * `apply` sets the context's font, spacing, baseline and alignment to match.
 */
function textLayout(
  context: CanvasRenderingContext2D, text: TextBlock,
  box: { x: number; y: number; width: number; height: number },
) {
  const wrapWidth = box.width;
  const size = text.autoFit
    ? fitFontSize({ ...text, boxWidth: wrapWidth }, box,
      (at) => measureAt(context, { ...text, boxWidth: wrapWidth }, at))
    : text.fontSize;
  const laid = measureAt(context, { ...text, boxWidth: wrapWidth }, size);
  const step = Math.max(1, size * Math.max(0.5, text.lineHeight));
  const sized = { ...text, fontSize: size, boxWidth: wrapWidth, padding: 0 };
  const { x, align } = lineOrigin(sized, wrapWidth);
  // Centred vertically in the box it was given. A balloon's words sitting at the top of it is the
  // most common way hand-made lettering looks wrong.
  const top = Math.max(0, (box.height - laid.lines.length * step) / 2);
  const apply = (target: CanvasRenderingContext2D) => {
    target.font = fontSpec(sized);
    applySpacing(target, sized);
    target.textBaseline = "middle";
    target.textAlign = align;
  };
  return { size, sized, lines: laid.lines, step, x, align, top, apply };
}

/**
 * VFX Phase 6: every character of a text layer where `drawText` puts it, in the text bitmap's
 * pixels: its left edge `x` and its alphabetic baseline `baseline`, found by measuring with the same
 * canvas state the text is drawn with (prefix widths, so kerning and letter spacing are included).
 * Also the bitmap's size (the layer's placement size) and the font size used.
 */
export function textGlyphRuns(text: TextBlock): {
  runs: Array<{ ch: string; x: number; baseline: number }>; width: number; height: number; fontSize: number;
} {
  const measurer = window.document.createElement("canvas").getContext("2d")!;
  const inner = Math.max(1, text.boxWidth - text.padding * 2);
  const laid = measureAt(measurer, { ...text, boxWidth: text.boxWidth }, text.fontSize);
  const size = textBoxSize(text, laid.lines.length);
  const box = { x: text.padding, y: text.padding, width: inner, height: size.height - text.padding * 2 };
  const context = window.document.createElement("canvas").getContext("2d")!;
  const layout = textLayout(context, { ...text, autoFit: false }, box);
  layout.apply(context);
  const runs: Array<{ ch: string; x: number; baseline: number }> = [];
  layout.lines.forEach((line, index) => {
    const y = box.y + layout.top + layout.step * (index + 0.5);
    const metrics = context.measureText(line);
    // With textBaseline "middle", alphabeticBaseline is the (negative) distance up to the baseline.
    const baseline = y - (metrics.alphabeticBaseline ?? 0);
    const lineWidth = metrics.width;
    const start = box.x + (layout.align === "center" ? layout.x - lineWidth / 2 : layout.align === "right" ? layout.x - lineWidth : layout.x);
    const chars = Array.from(line);
    let prefix = "";
    for (const ch of chars) {
      prefix += ch;
      const x = start + context.measureText(prefix).width - context.measureText(ch).width;
      runs.push({ ch, x, baseline });
    }
  });
  return { runs, width: size.width, height: size.height, fontSize: layout.size };
}

/**
 * The drop shadow, drawn as its own blurred silhouette rather than with `shadowColor`.
 *
 * The canvas shadow takes a colour and only a colour, so a gradient shadow is impossible through it.
 * Drawing the text into a scratch canvas, blurring it, and then filling the blur THROUGH ITSELF with
 * the paint gives a shadow that can be a ramp, and gives exactly the same result as the built-in one
 * when the paint happens to be solid.
 */
function drawTextShadow(
  context: CanvasRenderingContext2D, text: TextBlock, lines: string[],
  x: number, top: number, step: number, width: number, height: number, paint: Paint,
): void {
  const pad = Math.ceil(text.shadow * 2 + 4);
  const scratch = window.document.createElement("canvas");
  scratch.width = Math.max(1, Math.ceil(width) + pad * 2);
  scratch.height = Math.max(1, Math.ceil(height) + pad * 2);
  const ink = scratch.getContext("2d")!;
  ink.translate(pad, pad);
  ink.font = context.font;
  ink.textAlign = context.textAlign;
  ink.textBaseline = "middle";
  ink.fillStyle = "#000000";
  for (let index = 0; index < lines.length; index++) {
    if (text.strokeWidth > 0) {
      ink.lineWidth = text.strokeWidth * 2;
      ink.strokeStyle = "#000000";
      ink.lineJoin = "round";
      ink.strokeText(lines[index], x, top + step * (index + 0.5));
    }
    ink.fillText(lines[index], x, top + step * (index + 0.5));
  }

  ink.setTransform(1, 0, 0, 1, 0, 0);
  ink.filter = `blur(${text.shadow}px)`;
  ink.drawImage(scratch, 0, 0);
  ink.filter = "none";
  // Paint the silhouette with itself as the mask, which is what lets it be a gradient.
  ink.globalCompositeOperation = "source-in";
  ink.fillStyle = paintStyle(ink, paint, scratch.width, scratch.height);
  ink.fillRect(0, 0, scratch.width, scratch.height);
  ink.globalCompositeOperation = "source-over";

  context.drawImage(scratch, -pad, -pad + text.shadow * 0.35);
}

/** A free-standing text layer: a column of words, sized by what it says. */
function drawText(text: TextBlock): LayerSource {
  const measurer = window.document.createElement("canvas").getContext("2d")!;
  const inner = Math.max(1, text.boxWidth - text.padding * 2);
  const laid = measureAt(measurer, { ...text, boxWidth: text.boxWidth }, text.fontSize);
  const size = textBoxSize(text, laid.lines.length);

  const canvas = window.document.createElement("canvas");
  canvas.width = size.width;
  canvas.height = size.height;
  const context = canvas.getContext("2d")!;

  if (text.panel !== "none") {
    const paints = paintsOf(text);
    const radius = text.panel === "rounded" ? Math.min(size.width, size.height) * 0.12 : 0;
    roundedPath(context, 0.5, 0.5, size.width - 1, size.height - 1, radius);
    context.fillStyle = paintStyle(context, paints.panel, size.width, size.height);
    context.fill();
    if (text.panelBorder > 0) {
      context.lineWidth = text.panelBorder;
      context.strokeStyle = paintStyle(context, paints.panelBorder, size.width, size.height);
      context.stroke();
    }
  }

  paintText(context, { ...text, autoFit: false }, {
    x: text.padding, y: text.padding, width: inner, height: size.height - text.padding * 2,
  });
  return { image: canvas, width: size.width, height: size.height };
}

/**
 * A shape, and the lettering inside it if it has any.
 *
 * Stroked before it is filled and filled on top, which is the ordinary order; what matters is that
 * the balloon and its tail arrive here as ONE closed path, so there is no seam to hide.
 */
function drawShape(shape: ShapeBlock, text?: TextBlock): LayerSource {
  const bounds = shapeBounds(shape);
  const canvas = window.document.createElement("canvas");
  canvas.width = bounds.width;
  canvas.height = bounds.height;
  const context = canvas.getContext("2d")!;
  context.translate(bounds.offsetX, bounds.offsetY);

  const paths = shapePaths(shape).map((path) => {
    const outline = new Path2D();
    path.points.forEach(([x, y], index) => index === 0 ? outline.moveTo(x, y) : outline.lineTo(x, y));
    if (path.closed) outline.closePath();
    return outline;
  });

  if (shape.shadow > 0) {
    context.save();
    context.shadowColor = shape.shadowColor;
    context.shadowBlur = shape.shadow;
    context.shadowOffsetY = shape.shadowOffset;
    context.fillStyle = "#000000";
    for (const path of paths) context.fill(path);
    context.restore();
  }

  context.fillStyle = paintStyle(context, shape.fill, shape.width, shape.height);
  for (const path of paths) context.fill(path);
  if (shape.strokeWidth > 0) {
    context.lineWidth = shape.strokeWidth;
    context.lineJoin = "round";
    context.strokeStyle = paintStyle(context, shape.stroke, shape.width, shape.height);
    if (shape.dashed) context.setLineDash([shape.strokeWidth * 2.6, shape.strokeWidth * 2]);
    for (const path of paths) context.stroke(path);
    context.setLineDash([]);
  }

  if (text) {
    const box = shapeInnerBox(shape);
    paintText(context, { ...text, boxWidth: box.width, padding: 0, panel: "none", autoFit: text.autoFit !== false }, {
      x: -box.width / 2, y: -box.height / 2, width: box.width, height: box.height,
    });
  }
  return { image: canvas, width: bounds.width, height: bounds.height };
}

/** A path (after its envelope) as a canvas `Path2D`, through an optional point map. */
export function pathToPath2D(path: PathBlock, map?: (p: PathPoint) => PathPoint): Path2D {
  const out = new Path2D();
  const f = map ?? ((p: PathPoint) => p);
  for (const subpath of shapedPath(path).subpaths) {
    if (!subpath.nodes.length) continue;
    const start = f([subpath.nodes[0].x, subpath.nodes[0].y]);
    out.moveTo(start[0], start[1]);
    for (const c of segmentsOf(subpath)) {
      const [c1, c2, end] = [f(c[1]), f(c[2]), f(c[3])];
      out.bezierCurveTo(c1[0], c1[1], c2[0], c2[1], end[0], end[1]);
    }
    if (subpath.closed) out.closePath();
  }
  return out;
}

/**
 * VFX Phase 6: a path layer's bitmap -- filled with its rule, then stroked, into the symmetric box of
 * `pathBox` so the bitmap's centre is the path's origin. Paints are laid over the geometry's own box
 * (a gradient spans the shape, as it does on a shape layer). Both compositors place this bitmap, so
 * a path looks the same in each by construction.
 */
function drawPath(path: PathBlock): LayerSource {
  const box = pathBox(path);
  const canvas = window.document.createElement("canvas");
  canvas.width = box.width;
  canvas.height = box.height;
  const context = canvas.getContext("2d")!;
  context.translate(box.offsetX, box.offsetY);
  const outline = pathToPath2D(path);
  const b = pathBounds(path);
  const paintOver = (paint: Paint) => {
    const style = paintStyle(context, paint, b.maxX - b.minX, b.maxY - b.minY);
    if (typeof style !== "string") style.setTransform(new DOMMatrix().translate(b.minX, b.minY));
    return style;
  };
  const fill = () => {
    if (!path.fill) return;
    context.fillStyle = paintOver(path.fill);
    context.fill(outline, path.fillRule);
  };
  if (path.paintOrder !== "stroke") fill();
  if (path.stroke && path.strokeWidth > 0) {
    context.lineWidth = path.strokeWidth;
    context.lineJoin = path.join;
    context.lineCap = path.cap;
    context.miterLimit = Math.max(1, path.miterLimit);
    if (path.dash?.length) {
      context.setLineDash(path.dash.map((value) => Math.max(0, value)));
      context.lineDashOffset = path.dashOffset ?? 0;
    }
    context.strokeStyle = paintOver(path.stroke);
    context.stroke(outline);
    context.setLineDash([]);
  }
  if (path.paintOrder === "stroke") fill();
  return { image: canvas, width: box.width, height: box.height };
}

/**
 * VFX Phase 6: a `path` mask entry's coverage at `width` x `height`. The path is normalised to the
 * canvas, so it is scaled to the working size here; coverage is the fill's alpha (antialiased by the
 * canvas, identical for both compositors because both are handed these bytes).
 */
export function pathCoverage(path: PathBlock, width: number, height: number): Uint8Array {
  const canvas = window.document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.fillStyle = "#ffffff";
  context.fill(pathToPath2D(path, ([x, y]) => [x * width, y * height]), path.fillRule);
  const data = context.getImageData(0, 0, width, height).data;
  const coverage = new Uint8Array(width * height);
  for (let index = 0; index < coverage.length; index++) coverage[index] = data[index * 4 + 3];
  return coverage;
}

/** Letter spacing where the browser supports it, ignored where it does not. Never throws. */
function applySpacing(context: CanvasRenderingContext2D, text: TextBlock): void {
  if (!text.letterSpacing) return;
  try {
    (context as CanvasRenderingContext2D & { letterSpacing: string }).letterSpacing =
      `${text.letterSpacing * text.fontSize}px`;
  } catch {
    // An older engine simply sets the text solid, which is a cosmetic loss rather than a broken
    // layout, because the same context does the measuring.
  }
}

function roundedPath(
  context: CanvasRenderingContext2D,
  x: number, y: number, width: number, height: number, radius: number,
): void {
  const r = Math.max(0, Math.min(radius, width / 2, height / 2));
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
}

export function gradientPreview(gradient: Gradient, width: number, height: number): string {
  const canvas = window.document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d")!;
  const image = context.createImageData(width, height);
  renderGradient(image.data, width, height, gradient);
  context.putImageData(image, 0, 0);
  return canvas.toDataURL("image/png");
}
