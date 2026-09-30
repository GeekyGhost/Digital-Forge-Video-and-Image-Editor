/**
 * The composition model: one description of "layers over a canvas" for Photo Edit now and for the
 * video editor later (a photo is a composition with no duration).
 *
 * VFX Phase 1 introduces it WITHOUT migrating anything. Photo documents stay exactly as they are in
 * `Project.photoEdits`, so save, load and undo keep working untouched; a document is converted to a
 * composition on the way into the GPU renderer and nowhere else. That is why the conversion has to
 * be lossless in both directions -- `toPhotoDocument(fromPhotoDocument(d))` gives back `d` field for
 * field -- and why there is a test that holds it to that over every document shape the app writes.
 *
 * What changes relative to `PhotoLayer`:
 * - a layer has an explicit `kind` instead of "whichever optional field is set";
 * - placement is grouped into `transform`, and the source file into `source`;
 * - kinds the photo editor cannot make yet (video, vector, generator, adjustment, group) exist as
 *   types only, so the renderer and later phases have a shape to grow into.
 *
 * Nothing here touches the DOM: it runs in vitest's node environment and in the renderer alike.
 */

import type { BlendMode } from "./photoBlend";
import type { LayerEdge } from "./photoEdge";
import type { Gradient } from "./photoGradient";
import type { CropRect } from "./photoGeometry";
import type {
  LayerAttach, LayerGenerator, LayerMedia, PhotoDocument, PhotoLayer, PhotoMask, PhotoNode, PointTrack, LayerRepeat,
} from "./photoDocument";
import { layersOf } from "./photoDocument";
import type { ShapeBlock } from "./photoShape";
import type { PathBlock } from "./photoPath";
import type { TimelineGraphicBlock } from "./timelineGraphic";
import type { TextBlock } from "./photoText";

export const COMPOSITION_VERSION = 1;

/** Where a layer sits. Normalised to the canvas exactly as `PhotoLayer` is; see `layerMatrix`. */
export type LayerTransform = {
  /** Centre of the layer, normalised to the canvas. */
  x: number;
  y: number;
  /** Canvas pixels per source pixel. */
  scale: number;
  /** Degrees. */
  rotation: number;
  /** VFX Phase 7: a horizontal factor on top of `scale` (see `PhotoLayer.stretchX`). */
  stretchX?: number;
  /** Percentages taken off each side of the source before it is placed. */
  cropLeft: number;
  cropTop: number;
  cropRight: number;
  cropBottom: number;
};

/** The file behind a layer. Both strings are empty on a layer the app draws (text, shape, gradient). */
export type LayerSourceRef = { assetId: string; sourcePath: string };

type LayerCommon = {
  id: string;
  name: string;
  source: LayerSourceRef;
  transform: LayerTransform;
  opacity: number;
  /** Optional because documents written before blend modes existed have none; the renderer reads "normal". */
  blend?: BlendMode;
  visible: boolean;
  locked: boolean;
  /** Applied in order to the placed layer, each through its own mask (see `photoRender.ts`). */
  nodes: PhotoNode[];
  masks: PhotoMask[];
  /** Optional for the same reason as `blend`. */
  edge?: LayerEdge;
  /** VFX Phase 2: the layer mask (one of `masks`, a stack) that cuts this layer's alpha. */
  maskId?: string;
  /** VFX Phase 2: clip to the base of the clipping group below (see `clipBases`). */
  clip?: boolean;
  /** VFX Phase 4: placement follows a point track (see `attachedLayer`). */
  attach?: LayerAttach;
  /** VFX Phase 6: copies in a grid, a ring or along a path (see photoRepeat.ts). */
  repeat?: LayerRepeat;
  /**
   * Fields this build does not know about, carried through untouched so that a document written by
   * a newer build survives a round trip through an older one.
   */
  extra?: Record<string, unknown>;
};

export type ImageLayer = LayerCommon & { kind: "image" };
export type TextLayer = LayerCommon & { kind: "text"; text: TextBlock };
/** A shape, lettered inside when it carries `text` (a speech balloon). */
export type ShapeLayer = LayerCommon & { kind: "shape"; shape: ShapeBlock; text?: TextBlock };
/**
 * A gradient fill. Its own angle, centre and scale are its placement, so `transform` is ignored.
 * `text` and `shape` are never drawn on a gradient layer (the photo compositor never drew them
 * either); they are only kept so a document that somehow carries both round-trips unchanged.
 */
export type GradientLayer = LayerCommon & {
  kind: "gradient"; gradient: Gradient; text?: TextBlock; shape?: ShapeBlock;
};

/**
 * VFX Phase 3: moving pictures -- a video file or a frame sequence (an element). `text`, `shape` and
 * `gradient` are never drawn on it; they are kept only so an odd document round-trips unchanged.
 */
export type VideoLayer = LayerCommon & {
  kind: "video"; media: LayerMedia; text?: TextBlock; shape?: ShapeBlock; gradient?: Gradient;
  generator?: LayerGenerator;
};
/** VFX Phase 3: a generated layer (matrix rain), drawn at the working size like a gradient. */
export type GeneratorLayer = LayerCommon & {
  kind: "generator"; generator: LayerGenerator; text?: TextBlock; shape?: ShapeBlock; gradient?: Gradient;
};
/* Kinds later phases fill in. Types only: nothing creates them and the renderer skips them. */
/**
 * VFX Phase 6: a vector path (see photoPath.ts). `text` and `shape` are never drawn on it; they are
 * kept only so an odd document round-trips unchanged.
 */
export type VectorLayer = LayerCommon & { kind: "vector"; path: PathBlock; text?: TextBlock; shape?: ShapeBlock };
/** VFX Phase 7: a timeline graphic at one moment (see timelineGraphic.ts); programme frames only. */
export type GraphicLayer = LayerCommon & { kind: "graphic"; timelineGraphic: TimelineGraphicBlock };
/** Applies its node stack to everything below it, like a grade on the composite but scoped by order. */
export type AdjustmentLayer = LayerCommon & { kind: "adjustment" };
export type GroupLayer = LayerCommon & { kind: "group"; children: CompositionLayer[] };

export type CompositionLayer =
  | ImageLayer | TextLayer | ShapeLayer | GradientLayer
  | VideoLayer | VectorLayer | GeneratorLayer | GraphicLayer | AdjustmentLayer | GroupLayer;

export type CompositionLayerKind = CompositionLayer["kind"];

/** The kinds the Phase 1 renderer draws, which are exactly the ones a photo document can express. */
export const PHOTO_LAYER_KINDS: CompositionLayerKind[] = ["image", "text", "shape", "gradient", "video", "generator", "vector", "graphic"];

/** Whole-picture orientation, applied after composing (display and save), exactly as in `PhotoDocument`. */
export type CompositionGeometry = {
  rotate: 0 | 90 | 180 | 270;
  flipHorizontal: boolean;
  flipVertical: boolean;
  straighten?: number;
  scale?: number;
  crop?: CropRect;
};

export type Composition = {
  version: typeof COMPOSITION_VERSION;
  /** The asset the document opened from. */
  source: LayerSourceRef;
  /** Undefined means the bottom image layer's size decides (see `canvasSizeOf`). */
  width?: number;
  height?: number;
  background?: "transparent" | "black" | "white";
  /** Bottom first, the order they composite in. */
  layers: CompositionLayer[];
  /** The stack over the finished composite. */
  nodes: PhotoNode[];
  masks: PhotoMask[];
  depthPath?: string;
  geometry: CompositionGeometry;
  updatedAt: string;
  /**
   * How the layers were stored in the photo document, so `toPhotoDocument` writes the same shape
   * back. A pre-layers document keeps its one stack at the top level (see `layersOf`); converting
   * must not silently upgrade it, or opening a project would rewrite it.
   */
  photoShape: "layers" | "single-source" | "single-source-empty-layers";
  /** VFX Phase 3: rate and length in frames (see `compositionRate` / `compositionDuration`). */
  frameRate?: number;
  duration?: number;
  /** VFX Phase 4: point tracks, stored inline (see `PointTrack`). */
  tracks?: PointTrack[];
  extra?: Record<string, unknown>;
};

/* ------------------------------------------------------------------ keys */

const LAYER_KEYS = new Set<string>([
  "id", "assetId", "sourcePath", "name", "x", "y", "scale", "rotation", "stretchX",
  "cropLeft", "cropTop", "cropRight", "cropBottom", "opacity", "blendMode", "visible", "locked",
  "nodes", "masks", "edge", "text", "gradient", "shape", "maskId", "clip", "media", "generator", "attach",
  "path", "repeat", "timelineGraphic",
]);

const DOCUMENT_KEYS = new Set<string>([
  "assetId", "sourcePath", "layers", "canvasWidth", "canvasHeight", "background", "nodes", "masks",
  "depthPath", "rotate", "flipHorizontal", "flipVertical", "straighten", "scale", "crop", "updatedAt",
  "frameRate", "duration", "tracks",
]);

function extraOf(value: object, known: Set<string>): Record<string, unknown> | undefined {
  let extra: Record<string, unknown> | undefined;
  for (const [key, field] of Object.entries(value)) {
    if (known.has(key)) continue;
    (extra ??= {})[key] = field;
  }
  return extra;
}

/** Assigns only the keys whose value is not undefined, so absent stays absent through a round trip. */
function assignDefined<T extends object>(target: T, fields: Record<string, unknown>): T {
  for (const [key, value] of Object.entries(fields)) {
    if (value !== undefined) (target as Record<string, unknown>)[key] = value;
  }
  return target;
}

/* ------------------------------------------------------------- converting */

/** Which kind a photo layer is, in the same priority the photo compositor draws it. */
export function photoLayerKind(layer: PhotoLayer): "image" | "text" | "shape" | "gradient" | "video" | "generator" | "vector" | "graphic" {
  if (layer.media) return "video";
  if (layer.generator) return "generator";
  if (layer.gradient) return "gradient";
  if (layer.timelineGraphic) return "graphic";
  if (layer.path) return "vector";
  if (layer.shape) return "shape";
  if (layer.text) return "text";
  return "image";
}

export function fromPhotoLayer(layer: PhotoLayer): CompositionLayer {
  const common: LayerCommon = assignDefined({
    id: layer.id,
    name: layer.name,
    source: { assetId: layer.assetId, sourcePath: layer.sourcePath },
    transform: assignDefined({
      x: layer.x, y: layer.y, scale: layer.scale, rotation: layer.rotation,
      cropLeft: layer.cropLeft, cropTop: layer.cropTop, cropRight: layer.cropRight, cropBottom: layer.cropBottom,
    } as LayerTransform, { stretchX: layer.stretchX }),
    opacity: layer.opacity,
    visible: layer.visible,
    locked: layer.locked,
    nodes: layer.nodes,
    masks: layer.masks,
  } as LayerCommon, {
    blend: layer.blendMode,
    edge: layer.edge,
    maskId: layer.maskId,
    clip: layer.clip,
    attach: layer.attach,
    repeat: layer.repeat,
    extra: extraOf(layer, LAYER_KEYS),
  });
  const kind = photoLayerKind(layer);
  if (kind === "video") {
    return assignDefined({ ...common, kind, media: layer.media! } as VideoLayer,
      { text: layer.text, shape: layer.shape, gradient: layer.gradient, generator: layer.generator });
  }
  if (kind === "generator") {
    return assignDefined({ ...common, kind, generator: layer.generator! } as GeneratorLayer,
      { text: layer.text, shape: layer.shape, gradient: layer.gradient });
  }
  if (kind === "gradient") {
    return assignDefined({ ...common, kind, gradient: layer.gradient! } as GradientLayer,
      { text: layer.text, shape: layer.shape });
  }
  if (kind === "graphic") return { ...common, kind, timelineGraphic: layer.timelineGraphic! } as GraphicLayer;
  if (kind === "vector") return assignDefined({ ...common, kind, path: layer.path! } as VectorLayer, { text: layer.text, shape: layer.shape });
  if (kind === "shape") return assignDefined({ ...common, kind, shape: layer.shape! } as ShapeLayer, { text: layer.text });
  if (kind === "text") return { ...common, kind, text: layer.text! };
  return { ...common, kind };
}

export function toPhotoLayer(layer: CompositionLayer): PhotoLayer {
  if (!PHOTO_LAYER_KINDS.includes(layer.kind)) {
    throw new Error(`a ${layer.kind} layer cannot be stored in a photo document`);
  }
  const { transform: t } = layer;
  const out = {
    ...(layer.extra ?? {}),
    id: layer.id,
    assetId: layer.source.assetId,
    sourcePath: layer.source.sourcePath,
    name: layer.name,
    x: t.x, y: t.y, scale: t.scale, rotation: t.rotation,
    cropLeft: t.cropLeft, cropTop: t.cropTop, cropRight: t.cropRight, cropBottom: t.cropBottom,
    opacity: layer.opacity,
  } as PhotoLayer;
  assignDefined(out, { stretchX: t.stretchX });
  assignDefined(out, { blendMode: layer.blend });
  out.visible = layer.visible;
  out.locked = layer.locked;
  out.nodes = layer.nodes;
  out.masks = layer.masks;
  assignDefined(out, { edge: layer.edge, maskId: layer.maskId, clip: layer.clip, attach: layer.attach, repeat: layer.repeat });
  if (layer.kind === "text") out.text = layer.text;
  if (layer.kind === "gradient") assignDefined(out, { text: layer.text, gradient: layer.gradient, shape: layer.shape });
  if (layer.kind === "shape") assignDefined(out, { text: layer.text, shape: layer.shape });
  if (layer.kind === "vector") assignDefined(out, { text: layer.text, shape: layer.shape, path: layer.path });
  if (layer.kind === "graphic") out.timelineGraphic = layer.timelineGraphic;
  if (layer.kind === "video") assignDefined(out, { text: layer.text, gradient: layer.gradient, shape: layer.shape, media: layer.media, generator: layer.generator });
  if (layer.kind === "generator") assignDefined(out, { text: layer.text, gradient: layer.gradient, shape: layer.shape, generator: layer.generator });
  return out;
}

export function fromPhotoDocument(document: PhotoDocument): Composition {
  const layered = Boolean(document.layers?.length);
  const photoShape: Composition["photoShape"] = layered ? "layers"
    : document.layers ? "single-source-empty-layers" : "single-source";
  const geometry: CompositionGeometry = assignDefined({
    rotate: document.rotate,
    flipHorizontal: document.flipHorizontal,
    flipVertical: document.flipVertical,
  } as CompositionGeometry, { straighten: document.straighten, scale: document.scale, crop: document.crop });
  return assignDefined({
    version: COMPOSITION_VERSION,
    source: { assetId: document.assetId, sourcePath: document.sourcePath },
    // A pre-layers document is READ as one layer holding its stack (layersOf), and so is this.
    layers: layersOf(document).map(fromPhotoLayer),
    nodes: layered ? document.nodes : [],
    masks: layered ? document.masks : [],
    geometry,
    updatedAt: document.updatedAt,
    photoShape,
  } as Composition, {
    width: document.canvasWidth,
    height: document.canvasHeight,
    background: document.background,
    depthPath: document.depthPath,
    frameRate: document.frameRate,
    duration: document.duration,
    tracks: document.tracks,
    extra: extraOf(document, DOCUMENT_KEYS),
  });
}

export function toPhotoDocument(composition: Composition): PhotoDocument {
  const single = composition.photoShape !== "layers" && composition.layers.length === 1
    && !composition.nodes.length && !composition.masks.length;
  const out = {
    ...(composition.extra ?? {}),
    assetId: composition.source.assetId,
    sourcePath: composition.source.sourcePath,
  } as PhotoDocument;
  if (single) {
    if (composition.photoShape === "single-source-empty-layers") out.layers = [];
  } else {
    out.layers = composition.layers.map(toPhotoLayer);
  }
  assignDefined(out, {
    canvasWidth: composition.width,
    canvasHeight: composition.height,
    background: composition.background,
  });
  out.nodes = single ? composition.layers[0].nodes : composition.nodes;
  out.masks = single ? composition.layers[0].masks : composition.masks;
  assignDefined(out, { depthPath: composition.depthPath });
  const g = composition.geometry;
  out.rotate = g.rotate;
  out.flipHorizontal = g.flipHorizontal;
  out.flipVertical = g.flipVertical;
  assignDefined(out, { straighten: g.straighten, scale: g.scale, crop: g.crop });
  out.updatedAt = composition.updatedAt;
  assignDefined(out, { frameRate: composition.frameRate, duration: composition.duration, tracks: composition.tracks });
  return out;
}

/** Every layer that the Phase 1 renderer draws, bottom first, with groups flattened. */
export function drawableLayers(composition: Composition): CompositionLayer[] {
  const out: CompositionLayer[] = [];
  const walk = (layers: CompositionLayer[]) => {
    for (const layer of layers) {
      if (layer.kind === "group") walk(layer.children);
      else out.push(layer);
    }
  };
  walk(composition.layers);
  return out;
}
