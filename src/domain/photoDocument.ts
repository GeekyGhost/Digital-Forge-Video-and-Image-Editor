/**
 * What Photo Edit remembers about a picture.
 *
 * A document is an ordered stack of nodes over an untouched source file. Nothing is ever baked into
 * the original: reopening a picture rebuilds its look from these parameters, so an exposure chosen
 * last week is still a number you can drag rather than pixels you have to live with.
 *
 * Documents live inside `Project`, not in a store of their own. That is deliberate and buys three
 * things for free: they are saved and loaded by the project file that already exists, they travel
 * with a project copied to another machine, and they are covered by the undo stack the editor
 * already keeps over `project`. A parallel document store would have had to reimplement all three.
 */

import { isNeutralGeometry, type CropRect, type PhotoGeometry } from "./photoGeometry";
import type { BlendMode } from "./photoBlend";
import { NEUTRAL_EDGE, type LayerEdge } from "./photoEdge";
import type { RetouchStroke } from "./photoRetouch";
import type { WarpStroke } from "./photoWarp";
import { NEUTRAL_TEXT, type TextBlock } from "./photoText";
import { BALLOON_KINDS, SHAPE_PRESETS, holdsText } from "./photoShape";
import { solidPaint } from "./photoGradient";
import { NEUTRAL_GRADIENT, type Gradient } from "./photoGradient";
import { NEUTRAL_SHAPE, type ShapeBlock, type ShapeKind } from "./photoShape";
import type { PhotoAdjustments } from "./photoEdit";
import { NEUTRAL_PATH, type Envelope, type PathBlock } from "./photoPath";
import type { ClipEffectsParams } from "./photoClipEffects";
import type { TimelineGraphicBlock } from "./timelineGraphic";
import type { MaskVfxRecipeV1 } from "./maskVfx";
import { NEUTRAL_ADJUSTMENTS, isNeutralGrade } from "./photoEdit";

/**
 * One dab of the mask brush, in coordinates normalised to the source picture.
 *
 * Normalised because the preview runs at a proxy resolution and the save runs at full size, and a
 * mask stored in preview pixels would land in the wrong place on the file that actually gets
 * written. Strokes rather than a bitmap for the same reason a curve is stored as points: they
 * rasterise correctly at any size, undo one step at a time, and cost bytes instead of megabytes.
 */
export type MaskStroke = {
  points: Array<[number, number]>;
  /** Brush radius as a fraction of the picture's long edge. */
  radius: number;
  /** 0 is a soft edge that fades across the whole radius, 1 is a hard disc. */
  hardness: number;
  /** An erase stroke subtracts from the mask instead of adding to it. */
  erase: boolean;
};

/**
 * A mask's generated starting point, before any brushwork.
 *
 * All three describe a rule rather than pixels, so they re-derive at whatever resolution is being
 * worked at and stay editable -- dragging a gradient's endpoints later is a change to two numbers,
 * not a repaint. Strokes still layer on top, which is what makes a generated mask correctable
 * instead of take-it-or-leave-it.
 */
export type MaskShape =
  /** Ramps from nothing at one end to full at the other. Normalised to the picture. */
  | { kind: "linear"; x1: number; y1: number; x2: number; y2: number }
  /** Full inside, fading out. `aspect` above 1 stretches it wide, below 1 tall. */
  | { kind: "radial"; cx: number; cy: number; radius: number; aspect: number; feather: number }
  /** Everything within `tolerance` of a sampled colour. The magic wand. */
  | { kind: "colour"; r: number; g: number; b: number; tolerance: number; softness: number };

/**
 * How a mask-stack entry combines with everything above it (VFX Phase 2), on coverage 0..1:
 * - add: a + e - a*e (a union that stays soft where two feathered entries overlap);
 * - subtract: a * (1 - e);
 * - intersect: a * e;
 * - difference: |a - e|.
 * With hard-edged entries these are exactly the boolean operations. `e` is the entry's coverage
 * after its own invert, times its opacity.
 */
export type MaskMode = "add" | "subtract" | "intersect" | "difference";

export const MASK_MODES: MaskMode[] = ["add", "subtract", "intersect", "difference"];

/**
 * Where a mask-stack entry's coverage comes from.
 *
 * Every source that reads a file or another layer is resolved at render time, so a frame index can be
 * threaded through later (VFX Phase 3) without changing this shape: `sequence` already names a
 * whole `MaskMatteCache`-style folder and the renderer picks the frame.
 */
export type MaskSource =
  /** Nothing but the entry's own strokes. */
  | { kind: "paint" }
  /** A generated linear, radial or colour-range shape, exactly as on the base. */
  | { kind: "shape"; shape: MaskShape }
  /**
   * An imported greyscale or alpha PNG, placed through the owning layer's matrix like a SAM 2 file.
   * `channel` "auto" takes alpha when the file has any transparency, otherwise Rec. 709 luma.
   */
  | { kind: "file"; path: string; channel?: "auto" | "alpha" | "luma" }
  /**
   * A matte sequence in the `mask_cache.py` layout: `<cachePath>/masks/00000.png ...`. Photo Edit
   * renders `frame` (default 0); a video layer will pass its own frame index instead.
   */
  | {
    kind: "sequence"; cachePath: string; frameCount: number; fps?: number; frame?: number;
    /**
     * The source frame `masks/00000.png` belongs to (0 when absent). A SAM 2 video track started
     * part-way into a clip (SAM 2 propagates forward from the frame the points were clicked on)
     * covers source frames first..first+frameCount-1; frames before it hold its first matte.
     */
    first?: number;
  }
  /**
   * Another layer, after its own placement, nodes and masks: its alpha, or its Rec. 709 luma
   * weighted by alpha. Times that layer's opacity. This is a track matte.
   */
  | { kind: "layer"; layerId: string; channel: "alpha" | "luma" }
  /**
   * VFX Phase 4: where another layer's depth (its `media.depth` sequence, disparity: 255 is near) is
   * NEARER than `plane` (0..100), ramped over `softness` (0..100). Subtracted from an element's own
   * mask, it is occlusion: the hand in the plate passes in front of the smoke.
   */
  | { kind: "depth"; layerId: string; plane: number; softness: number }
  /**
   * VFX Phase 6: a vector path's fill, in coordinates NORMALISED to the canvas (0..1 on each axis),
   * like the other canvas-space shapes. Only the geometry and fill rule are read; paint and stroke
   * are ignored. Rasterised once per size by the same Canvas2D call for both compositors.
   */
  | { kind: "path"; path: PathBlock };

/**
 * One entry of a mask stack after the base. The base is the mask's own legacy fields (SAM 2 file,
 * shape, strokes, feather, invert), which is how every mask written before VFX Phase 2 reads as a
 * one-entry stack with exactly the coverage it always had.
 *
 * `feather` and `expand` are in CANVAS pixels and scale with the render, like the base's feather, so
 * an entry softens the same amount in the 1500 px preview and in the full-size save.
 */
export type MaskEntry = {
  id: string;
  name?: string;
  enabled: boolean;
  source: MaskSource;
  /** Hand corrections stamped over the source, as on the base. */
  strokes: MaskStroke[];
  mode: MaskMode;
  /** 0..1, scales this entry's coverage before it is combined. */
  opacity: number;
  feather: number;
  invert: boolean;
  /** Positive grows the entry (a max filter), negative chokes it (a min filter). Before the feather. */
  expand: number;
};

export type PhotoMask = {
  id: string;
  name: string;
  shape?: MaskShape;
  strokes: MaskStroke[];
  /** A greyscale PNG from SAM 2, used as the starting point that strokes then correct. */
  sourcePath?: string;
  invert: boolean;
  /**
   * Softening in canvas pixels, scaled to the render, applied after the strokes and the SAM 2 mask
   * are combined. (Until VFX Phase 2 it was not scaled, so the preview was softer than the save.)
   */
  feather: number;
  enabled: boolean;
  /**
   * VFX Phase 2: further entries combined, in order, into the base above. Optional, so a document
   * written before it loads unchanged and an older build keeps it untouched.
   */
  stack?: MaskEntry[];
};

/** Where the light is, in coordinates normalised to the picture, plus how it behaves. */
export type RelightSettings = {
  x: number;
  y: number;
  /** How far in front of the picture the light sits, as a fraction of the long edge. */
  distance: number;
  intensity: number;
  /** 0 is a hard, contrasty falloff; 1 is broad and soft. */
  softness: number;
  color: [number, number, number];
  ambient: number;
  rim: number;
};

export const NEUTRAL_RELIGHT: RelightSettings = {
  x: 0.3, y: 0.25, distance: 0.55, intensity: 1, softness: 0.45,
  color: [1, 0.92, 0.82], ambient: 0.55, rim: 0.35,
};

export type PhotoNodeBase = {
  id: string;
  enabled: boolean;
  /** Which mask limits this node. Undefined means the whole picture. */
  maskId?: string;
};

export type GradeNode = PhotoNodeBase & { kind: "grade"; adjustments: PhotoAdjustments };
export type RelightNode = PhotoNodeBase & { kind: "relight"; relight: RelightSettings };
/**
 * A generative fill. `cachePath` is the file the worker produced for the current parameters -- the
 * node stays re-editable because the mask that produced it is still here, and clearing the path is
 * how it gets asked to run again.
 */
export type RemoveNode = PhotoNodeBase & { kind: "remove"; maskId: string; cachePath?: string };
/** Clone stamp and spot heal. Strokes carry where they sampled from, so the node stays re-editable. */
export type RetouchNode = PhotoNodeBase & { kind: "retouch"; strokes: RetouchStroke[] };
/**
 * Liquify. Strokes rather than a baked displacement map, for the same reason as everything else
 * here: a stored field would be megabytes, would be wrong at any other resolution, and could not be
 * undone one gesture at a time.
 */
export type WarpNode = PhotoNodeBase & { kind: "warp"; strokes: WarpStroke[] };
/**
 * Re-colour by brightness: shadows take one end of a ramp, highlights the other. Duotone and
 * split-toning. The same `Gradient` a fill layer paints with, doing a different job.
 */
export type GradientMapNode = PhotoNodeBase & {
  kind: "gradientMap"; gradient: Gradient; amount: number;
};

/**
 * Stock footage shot on black, made transparent (VFX Phase 2): alpha becomes max(r, g, b) and the
 * colour is divided by it, on ENCODED values. Composited normally over a backdrop that gives what a
 * Screen blend of the original would, because Screen here is defined on encoded values too
 * (photoBlend.ts). Black becomes fully transparent. See `applyUnmultiply` in photoKey.ts.
 */
export type UnmultiplyNode = PhotoNodeBase & { kind: "unmultiply" };
/**
 * Keep the pixels whose Rec. 709 luma (encoded) lies between `low` and `high` (0..100), ramping over
 * `softness` outside that range; `invert` keeps the rest instead. Multiplies alpha.
 */
export type LumaKeyNode = PhotoNodeBase & {
  kind: "lumaKey"; low: number; high: number; softness: number; invert: boolean;
};

/**
 * VFX Phase 3: reshape alpha -- a' = 255 * min(1, gain * (a / 255) ^ gamma). Wan Alpha elements come
 * out mostly translucent (opaque pixels are rare even in a fireball core); this is the per-use fix.
 */
export type AlphaCurveNode = PhotoNodeBase & { kind: "alphaCurve"; gain: number; gamma: number };

/**
 * VFX Phase 4: the time-lagged warp. The layer's far end (along its up axis) follows the track
 * `maxLag` frames late, its base on time, so smoke rising from a moving cigarette trails behind it.
 * `strength` scales the displacement (1 = exactly the track's lagged motion).
 */
export type TrailNode = PhotoNodeBase & { kind: "trail"; trackId: string; maxLag: number; strength: number };

/**
 * VFX Phase 6: envelope distort on pixels. `envelope.rect` and `points` are NORMALISED to the canvas
 * (0..1), like liquify strokes, so preview and export bend the same place (see photoEnvelope.ts).
 */
export type EnvelopeNode = PhotoNodeBase & { kind: "envelope"; envelope: Envelope };

/** VFX Phase 7: a timeline clip's effects, and the per-pixel parts of transitions (photoClipEffects.ts). */
export type ClipEffectsNode = PhotoNodeBase & { kind: "clipEffects"; effects: ClipEffectsParams };
/**
 * VFX Phase 7: one frame of a matte-driven mask VFX recipe (the old `mask_effects.py`), rendered by
 * `renderMaskFx` in both compositors. The programme builds one per frame from a clip's
 * `ClipVfxLayer`; everything that changes per frame is resolved into the node.
 */
export type MaskVfxNode = PhotoNodeBase & {
  kind: "maskVfx";
  recipe: MaskVfxRecipeV1;
  /** This frame's index in the layer's range, the range's length in frames, and its rate. */
  index: number;
  count: number;
  fps: number;
  /** The audio envelope at this frame (0..1). */
  envelope: number;
  /** The (stabilised) subject matte for this frame: a file the size of the source picture. */
  mattePath: string;
  /** The source picture's size, which the matte, the effect anchor and the gradient are measured in. */
  sourceWidth: number;
  sourceHeight: number;
  /** The imported effect clip and the frame of it this frame shows (the "clip" matte style). */
  effectClip?: { media: LayerMedia; index: number };
};
export type PhotoNode =
  | GradeNode | RelightNode | RemoveNode | RetouchNode | WarpNode | GradientMapNode
  | UnmultiplyNode | LumaKeyNode | AlphaCurveNode | TrailNode | EnvelopeNode | ClipEffectsNode | MaskVfxNode;

/** VFX Phase 4: one sample of a point track: source pixels of the tracked layer, and whether it was seen. */
export type TrackSample = [x: number, y: number, visible: 0 | 1];

/**
 * VFX Phase 4: a point followed through a media layer's frames. Measured in that layer's SOURCE
 * pixels and SOURCE frames, so it stays right however the layer is placed, trimmed or retimed.
 * Stored inline (a few numbers per frame): the compositors are synchronous and read it directly.
 */
export type PointTrack = {
  id: string;
  name: string;
  /** The media layer the track was measured on. */
  layerId: string;
  /** samples[i] is source frame `first + i`. */
  first: number;
  samples: TrackSample[];
  source: "cotracker3" | "lucas-kanade" | "sam2-centroid" | "manual";
};

/**
 * VFX Phase 4: attach a layer to a track. The layer's own x/y are where it sits at `anchorFrame`;
 * at any other frame it moves by however far the track moved since then, `lag` frames late.
 */
export type LayerAttach = { trackId: string; anchorFrame: number; lag?: number };

/**
 * VFX Phase 3: moving pictures on a layer. Units are FRAMES throughout: `start` in composition
 * frames (at the document's `frameRate`), `inPoint` / `outPoint` / `crossfade` in source frames at
 * `fps`. Frames rather than seconds because every consumer (the decoder's seek, a PNG file name, a
 * matte sequence) is indexed by frame, and a rounding step between seconds and frames is exactly
 * where off-by-one drift starts. See `mediaFramesAt` in photoTime.ts.
 */
export type LayerMedia = {
  /** "video": an MP4 decoded with WebCodecs. "frames": a folder of PNGs (an element master). */
  kind: "video" | "frames";
  /** The MP4, or the folder holding the frames. */
  path: string;
  /** File name pattern inside the folder for "frames", printf-style. Default "frame_%04d.png". */
  pattern?: string;
  frameCount: number;
  fps: number;
  width: number;
  height: number;
  /** Composition frame where source frame `inPoint` shows. */
  start: number;
  /** First source frame used, and one past the last. */
  inPoint: number;
  outPoint: number;
  /** Repeat [inPoint, outPoint) forever, with `crossfade` frames of the tail blended into the head. */
  loop: boolean;
  crossfade: number;
  /** VFX Phase 4: a per-frame depth sequence for this media (Video Depth Anything), `masks/00000.png` layout. */
  depth?: { cachePath: string; frameCount: number };
  /** A lighter file for scrubbing: an H.264 proxy, or an element's stacked colour|alpha proxy. */
  proxyPath?: string;
  proxyStacked?: boolean;
  /** The element library entry this came from, if any. */
  elementId?: string;
};

/**
 * VFX Phase 3: a layer the app generates per frame, like a gradient but moving. Deterministic in
 * (seed, frame), so a preview frame and a saved frame are the same picture.
 */
export type MatrixRainGenerator = {
  kind: "matrixRain";
  seed: number;
  /** Glyph cell size in canvas pixels. */
  cellSize: number;
  /** Composition frames after which the rain repeats exactly. */
  loopFrames: number;
  /** Fraction of columns that rain, 0..1. */
  density: number;
  /** Trail length in cells. */
  trail: number;
  /** Each column falls this many whole screens per loop (a range, per column). */
  minPasses: number;
  maxPasses: number;
  /** How many times each cell's glyph changes per loop. */
  glyphChanges: number;
  colour: string;
  headColour: string;
  /** Mirror the glyphs, as the film does. */
  mirror: boolean;
};

/**
 * VFX Phase 4: a sprite emitter. Spawns copies of an element (a frames `LayerMedia`) at a track or a
 * point, `rate` per composition frame, each living `lifetime` frames and playing the element from its
 * first frame. Every particle is a pure function of (seed, spawn frame, index), so any frame renders
 * the same whenever it is asked for.
 */
export type EmitterGenerator = {
  kind: "emitter";
  seed: number;
  element: LayerMedia;
  /** Particles per composition frame (fractions accumulate: 0.25 is one every four frames). */
  rate: number;
  lifetime: number;
  /** Spawn at this track, else at (x, y), normalised to the canvas. */
  trackId?: string;
  x: number;
  y: number;
  /** Velocity in canvas fractions per frame, plus a random spread of up to `spread` each way. */
  velocityX: number;
  velocityY: number;
  spread: number;
  /** Canvas pixels per element pixel at birth and at death. */
  scaleStart: number;
  scaleEnd: number;
  opacityStart: number;
  opacityEnd: number;
  /** Random initial rotation, up to this many degrees each way. */
  spin: number;
};

export type LayerGenerator = MatrixRainGenerator | EmitterGenerator;

/**
 * One picture on the canvas, with everything that is true only of it.
 *
 * A layer carries its own node stack and masks, which is the whole point of the merge: a cutout can
 * be graded and relit to match the plate it was dropped on, live, instead of being baked and
 * re-baked. Placement is normalised to the canvas so a document survives being rendered at proxy
 * size and at full size, same reason masks are.
 */
/**
 * VFX Phase 6: repeat a layer. Offsets are in CANVAS pixels and angles in degrees, so a repeat means
 * the same thing at preview and export size. Copy 0 is the layer itself; the others are placed
 * relative to it.
 * - grid: `columns` x `rows` copies, `gapX`/`gapY` apart (centre to centre); `shift` offsets every
 *   other row by that fraction of `gapX` (a brick pattern).
 * - radial: `count` copies around a centre `radius` pixels from the layer (the layer is the copy at
 *   `startAngle`), turned with the ring when `rotate`.
 * - path: `count` copies spread along another path layer's outline (from `start` to `end`, 0..1 of
 *   its length), turned to its direction when `rotate`.
 * `scaleStep` multiplies each next copy's scale, `opacityStep` its opacity (1 = unchanged).
 */
export type LayerRepeat =
  | { kind: "grid"; columns: number; rows: number; gapX: number; gapY: number; shift?: number; scaleStep?: number; opacityStep?: number }
  | { kind: "radial"; count: number; radius: number; startAngle?: number; sweep?: number; rotate: boolean; scaleStep?: number; opacityStep?: number }
  | { kind: "path"; layerId: string; count: number; start?: number; end?: number; rotate: boolean; scaleStep?: number; opacityStep?: number };

export type PhotoLayer = {
  id: string;
  /** Empty on a text layer, which has no asset behind it. */
  assetId: string;
  /** Empty on a text layer, for the same reason. */
  sourcePath: string;
  name: string;
  /** Centre of the layer, normalised to the canvas. */
  x: number;
  y: number;
  /** 1 draws the source at its own pixel size against the canvas. */
  scale: number;
  rotation: number;
  /**
   * VFX Phase 7: a horizontal factor on top of `scale` (1 or absent: uniform). A timeline clip's
   * "stretch" conform and the squeeze transition are the only writers.
   */
  stretchX?: number;
  /** Percentages taken off each side of the source before it is placed. */
  cropLeft: number;
  cropTop: number;
  cropRight: number;
  cropBottom: number;
  opacity: number;
  blendMode: BlendMode;
  visible: boolean;
  locked: boolean;
  nodes: PhotoNode[];
  masks: PhotoMask[];
  edge: LayerEdge;
  /**
   * Present on a text layer, which the app draws rather than loads.
   *
   * A field rather than a separate layer type: by the time the compositor has it, a text layer is a
   * bitmap of a known size like any other, and everything else about a layer -- placement, blend
   * mode, the edge group, its own node stack -- applies to it without a single special case.
   */
  text?: TextBlock;
  /**
   * Present on a gradient layer, which the app also draws rather than loads.
   *
   * Alongside `text` rather than replacing it, and for the same reason: what the compositor gets is
   * a bitmap of a known size, so blend modes, masks, the edge group and the layer's own node stack
   * all work on it with no special case. A wash of colour over a plate is just a layer.
   */
  gradient?: Gradient;
  /**
   * Present on a shape layer: a box, a star, or a speech balloon.
   *
   * A balloon that also has `text` letters INSIDE the shape rather than beside it, which is why
   * these are two fields on one layer and not two kinds of layer -- a balloon is a container with
   * something to say, and treating it as its own type would mean a second copy of all the lettering.
   */
  shape?: ShapeBlock;
  /**
   * VFX Phase 6: a vector path layer (see photoPath.ts): canvas pixels around the layer's centre,
   * drawn like a shape into a bitmap the layer places. Takes precedence over `shape` and `text`.
   */
  path?: PathBlock;
  /**
   * VFX Phase 6: copies of this layer in a grid, a ring or along another layer's path, expanded
   * before compositing into ordinary layers (see photoRepeat.ts).
   */
  repeat?: LayerRepeat;
  /**
   * VFX Phase 7: a timeline title/lower third/caption at one moment, drawn frame-sized by
   * `timelineGraphic.ts`. Only the programme renderer writes it; it is never saved in a document.
   */
  timelineGraphic?: TimelineGraphicBlock;
  /**
   * VFX Phase 2: the layer mask, naming one of this layer's own masks (a stack). Its coverage
   * multiplies the layer's alpha after the node stack, before the edge group and the blend. A track
   * matte is a layer mask whose stack holds a `layer` entry.
   */
  maskId?: string;
  /** VFX Phase 3: a video or frame-sequence layer (see LayerMedia). */
  media?: LayerMedia;
  /** VFX Phase 3: a generated layer (matrix rain). Drawn at the working size, like a gradient. */
  generator?: LayerGenerator;
  /** VFX Phase 4: follow a point track (see LayerAttach). */
  attach?: LayerAttach;
  /**
   * VFX Phase 2: clip to the layer below (Photoshop semantics). The layer's alpha is multiplied by
   * the alpha and opacity of the base of its clipping group, the nearest layer below that is not
   * itself clipped. Hidden with that base. Ignored on the bottom layer.
   */
  clip?: boolean;
};

export type PhotoDocument = {
  /**
   * The layer the document opened from. Retained after the move to layers so an existing document
   * still knows which asset it belongs to, and so Project Media can mark it edited.
   */
  assetId: string;
  /** Kept so a document can tell it is looking at a different file than the one it was built on. */
  sourcePath: string;
  /** Bottom first, the order they composite in. */
  layers?: PhotoLayer[];
  /** Undefined means the canvas takes the size of the bottom layer. */
  canvasWidth?: number;
  canvasHeight?: number;
  background?: "transparent" | "black" | "white";
  /** The stack over the finished composite. Layers have their own. */
  nodes: PhotoNode[];
  masks: PhotoMask[];
  /** Depth map for this picture, written once by the worker and reused by every relight node. */
  depthPath?: string;
  /**
   * VFX Phase 3: the composition's rate and length, in frames. Undefined on a still: the rate falls
   * back to the first media layer's (`compositionRate`), and the length to the longest layer.
   */
  frameRate?: number;
  duration?: number;
  /** VFX Phase 4: point tracks, measured on media layers. */
  tracks?: PointTrack[];
  /** Whole-picture orientation. Not a node: it moves pixels rather than changing them. */
  rotate: 0 | 90 | 180 | 270;
  flipHorizontal: boolean;
  flipVertical: boolean;
  /** Fine rotation, on top of the quarter turns. Optional: documents predate it. */
  straighten?: number;
  /** Free scale, 1 being none. Optional for the same reason. */
  scale?: number;
  /** Normalised to the straightened frame. */
  crop?: CropRect;
  updatedAt: string;
};

/**
 * The document's geometry, with the fields older documents do not carry filled in.
 *
 * Straighten, scale and crop arrived after the first documents were written, so reading them raw
 * would put `undefined` into the transform and turn every coordinate into NaN. One normaliser means
 * no caller has to remember that.
 */
export const geometryOf = (document: PhotoDocument): PhotoGeometry => ({
  rotate: document.rotate,
  flipHorizontal: document.flipHorizontal,
  flipVertical: document.flipVertical,
  straighten: document.straighten ?? 0,
  scale: document.scale ?? 1,
  crop: document.crop,
});

const newId = () => crypto.randomUUID();

export const freshAdjustments = (): PhotoAdjustments => ({
  ...NEUTRAL_ADJUSTMENTS,
  curve: NEUTRAL_ADJUSTMENTS.curve.map((point) => ({ ...point })),
});

export function createGradeNode(): GradeNode {
  return { id: newId(), kind: "grade", enabled: true, adjustments: freshAdjustments() };
}

export function createRelightNode(): RelightNode {
  const color: [number, number, number] = [...NEUTRAL_RELIGHT.color];
  return { id: newId(), kind: "relight", enabled: true, relight: { ...NEUTRAL_RELIGHT, color } };
}

export function createRemoveNode(maskId: string): RemoveNode {
  return { id: newId(), kind: "remove", enabled: true, maskId };
}

export function createRetouchNode(): RetouchNode {
  return { id: newId(), kind: "retouch", enabled: true, strokes: [] };
}

/** VFX Phase 6: an envelope node over the whole canvas, 4 x 4 points (a bicubic patch). */
export function createEnvelopeNode(rect = { x: 0, y: 0, width: 1, height: 1 }, cols = 3, rows = 3): EnvelopeNode {
  const points: Array<[number, number]> = [];
  for (let j = 0; j <= rows; j++) for (let i = 0; i <= cols; i++) points.push([rect.x + (rect.width * i) / cols, rect.y + (rect.height * j) / rows]);
  return { id: newId(), kind: "envelope", enabled: true, envelope: { cols, rows, rect: { ...rect }, points } };
}

export function createWarpNode(): WarpNode {
  return { id: newId(), kind: "warp", enabled: true, strokes: [] };
}

export function createGradientMapNode(): GradientMapNode {
  return {
    id: newId(), kind: "gradientMap", enabled: true, amount: 100,
    // Starts as a duotone rather than as black-to-white: mapping a picture onto greyscale is a
    // valid thing to want and a useless thing to be shown first.
    gradient: {
      ...NEUTRAL_GRADIENT,
      stops: [
        { at: 0, color: "#101b3a", opacity: 1 },
        { at: 1, color: "#ffd7a1", opacity: 1 },
      ],
    },
  };
}

export function createAlphaCurveNode(): AlphaCurveNode {
  return { id: newId(), kind: "alphaCurve", enabled: true, gain: 1.6, gamma: 0.8 };
}

export const DEFAULT_MATRIX_RAIN: MatrixRainGenerator = {
  kind: "matrixRain", seed: 1999, cellSize: 22, loopFrames: 240, density: 0.7, trail: 18,
  minPasses: 1, maxPasses: 3, glyphChanges: 12, colour: "#35ff6a", headColour: "#e8ffe8", mirror: true,
};

/** A matrix-rain layer. Like a gradient it has no placement of its own: it fills the canvas. */
export function createMatrixRainLayer(over: Partial<MatrixRainGenerator> = {}): PhotoLayer {
  return {
    id: newId(), assetId: "", sourcePath: "", name: "Matrix rain",
    x: 0.5, y: 0.5, scale: 1, rotation: 0,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
    opacity: 1, blendMode: "screen", visible: true, locked: false,
    nodes: [], masks: [], edge: { ...NEUTRAL_EDGE },
    generator: { ...DEFAULT_MATRIX_RAIN, seed: Math.floor(Math.random() * 1e6), ...over },
  };
}

/**
 * A video or frame-sequence layer. `sourcePath` is the file (video) or the first frame (frames), so
 * code that only knows about pictures (the canvas size, Project Media) still finds something real.
 */
export function createMediaLayer(
  name: string, media: Omit<LayerMedia, "start" | "inPoint" | "outPoint" | "loop" | "crossfade"> & Partial<LayerMedia>,
  over: Partial<PhotoLayer> = {},
): PhotoLayer {
  const full: LayerMedia = {
    start: 0, inPoint: 0, outPoint: media.frameCount, loop: false, crossfade: 0, ...media,
  };
  return {
    id: newId(), assetId: "", sourcePath: "", name,
    x: 0.5, y: 0.5, scale: 1, rotation: 0,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
    opacity: 1, blendMode: "normal", visible: true, locked: false,
    nodes: [], masks: [], edge: { ...NEUTRAL_EDGE },
    media: full,
    ...over,
  };
}

/**
 * A layer mask that hides the layer wherever `layerId`'s depth is nearer than `plane`: an inverted
 * empty base (everything) minus a depth entry. Occlusion is a mask like any other, so it can be
 * feathered, painted and combined.
 */
export function createDepthOcclusionMask(layerId: string, plane = 55, softness = 6): PhotoMask {
  return {
    ...createMask("Depth occlusion"), invert: true, feather: 0,
    stack: [createMaskEntry({ kind: "depth", layerId, plane, softness }, { mode: "subtract" })],
  };
}

export function createTrailNode(trackId: string): TrailNode {
  return { id: newId(), kind: "trail", enabled: true, trackId, maxLag: 12, strength: 1 };
}

/** An emitter of `element`, at the canvas centre until it is given a track. */
export function createEmitterLayer(element: LayerMedia, over: Partial<EmitterGenerator> = {}): PhotoLayer {
  return {
    id: newId(), assetId: "", sourcePath: "", name: "Emitter",
    x: 0.5, y: 0.5, scale: 1, rotation: 0,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
    opacity: 1, blendMode: "normal", visible: true, locked: false,
    nodes: [], masks: [], edge: { ...NEUTRAL_EDGE },
    generator: {
      kind: "emitter", seed: Math.floor(Math.random() * 1e6), element: { ...element, start: 0, loop: false, crossfade: 0 },
      rate: 0.25, lifetime: Math.min(48, element.frameCount), x: 0.5, y: 0.6,
      velocityX: 0, velocityY: -0.004, spread: 0.0015, scaleStart: 0.5, scaleEnd: 1, opacityStart: 1, opacityEnd: 0, spin: 10,
      ...over,
    },
  };
}

export function createUnmultiplyNode(): UnmultiplyNode {
  return { id: newId(), kind: "unmultiply", enabled: true };
}

export function createLumaKeyNode(): LumaKeyNode {
  return { id: newId(), kind: "lumaKey", enabled: true, low: 6, high: 100, softness: 10, invert: false };
}

/** A mask-stack entry. Defaults to the mode that adds, at full strength, with no softening. */
export function createMaskEntry(source: MaskSource, over: Partial<MaskEntry> = {}): MaskEntry {
  return {
    id: newId(), enabled: true, source, strokes: [], mode: "add", opacity: 1, feather: 0, invert: false, expand: 0,
    ...over,
  };
}

export function createMask(name: string): PhotoMask {
  return { id: newId(), name, strokes: [], invert: false, feather: 2, enabled: true };
}

/**
 * A text layer.
 *
 * Sized from the canvas rather than from a fixed default, because a 72px caption is a headline on a
 * 1024 plate and a footnote on a 6000px one, and nobody wants to start by fixing that.
 */
export function createTextLayer(
  canvas: { width: number; height: number } | undefined, over: Partial<TextBlock> = {},
): PhotoLayer {
  const long = canvas ? Math.max(canvas.width, canvas.height) : 1200;
  const text: TextBlock = {
    ...NEUTRAL_TEXT,
    fontSize: Math.round(long * 0.06),
    boxWidth: Math.round((canvas?.width ?? long) * 0.8),
    padding: Math.round(long * 0.02),
    ...over,
  };
  return {
    id: newId(), assetId: "", sourcePath: "", name: "Text",
    x: 0.5, y: 0.5, scale: 1, rotation: 0,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
    opacity: 1, blendMode: "normal", visible: true, locked: false,
    nodes: [], masks: [], edge: { ...NEUTRAL_EDGE },
    text,
  };
}

/**
 * A gradient fill layer.
 *
 * It has no placement of its own, because a gradient has no intrinsic pixels to place: its angle,
 * centre and scale ARE where it sits, and running it through the layer matrix as well would give
 * two controls for one thing that could disagree.
 */
export function createGradientLayer(over: Partial<Gradient> = {}): PhotoLayer {
  return {
    id: newId(), assetId: "", sourcePath: "", name: "Gradient",
    x: 0.5, y: 0.5, scale: 1, rotation: 0,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
    opacity: 1, blendMode: "normal", visible: true, locked: false,
    nodes: [], masks: [], edge: { ...NEUTRAL_EDGE },
    gradient: { ...NEUTRAL_GRADIENT, stops: NEUTRAL_GRADIENT.stops.map((stop) => ({ ...stop })), ...over },
  };
}

/**
 * A shape layer, sized against the canvas it is going onto.
 *
 * A balloon comes with lettering already in it, because an empty balloon is never what anyone
 * wanted; the plain shapes come without, because a star with "Double-click to edit" through it is
 * not a useful default.
 */
export function createShapeLayer(
  kind: ShapeKind, canvas: { width: number; height: number } | undefined,
): PhotoLayer {
  const long = canvas ? Math.max(canvas.width, canvas.height) : 1200;
  const balloon = BALLOON_KINDS.includes(kind);
  const preset = SHAPE_PRESETS.find((entry) => entry.shape.kind === kind)?.shape ?? {};
  // Things that go behind the action want to be big enough to sit behind it.
  const wide = !holdsText(kind);
  const shape: ShapeBlock = {
    ...NEUTRAL_SHAPE,
    ...preset,
    kind,
    width: Math.round(long * (wide ? 0.7 : 0.34)),
    height: Math.round(long * (wide ? 0.7 : 0.2)),
    strokeWidth: Math.max(2, Math.round(long * 0.005)),
    padding: Math.round(long * 0.018),
    shadowOffset: Math.round(long * 0.006),
  };
  return {
    id: newId(), assetId: "", sourcePath: "",
    name: `${kind[0].toUpperCase()}${kind.slice(1)}`,
    x: 0.5, y: 0.5, scale: 1, rotation: 0,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
    opacity: 1, blendMode: "normal", visible: true, locked: false,
    nodes: [], masks: [], edge: { ...NEUTRAL_EDGE },
    shape,
    text: balloon ? {
      ...NEUTRAL_TEXT,
      content: kind === "thought" ? "Hmm…" : "Say something",
      color: solidPaint("#000000"),
      fontSize: Math.round(long * 0.04),
      boxWidth: Math.round(long * 0.26),
      padding: 0,
      autoFit: true,
      fitMin: Math.max(6, Math.round(long * 0.008)),
    } : undefined,
  };
}

/**
 * VFX Phase 6: a vector path layer. `path` is in canvas pixels around the layer's centre, and the
 * layer is placed at (x, y) normalised, scale 1, no rotation, so the path lands where it was drawn.
 */
export function createPathLayer(
  path: Partial<PathBlock> & Pick<PathBlock, "subpaths">, x = 0.5, y = 0.5, name = "Path",
): PhotoLayer {
  return {
    id: newId(), assetId: "", sourcePath: "", name,
    x, y, scale: 1, rotation: 0,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
    opacity: 1, blendMode: "normal", visible: true, locked: false,
    nodes: [], masks: [], edge: { ...NEUTRAL_EDGE },
    path: { ...NEUTRAL_PATH, ...path },
  };
}

export function createLayer(
  asset: { id: string; name: string; path: string }, over: Partial<PhotoLayer> = {},
): PhotoLayer {
  return {
    id: newId(), assetId: asset.id, sourcePath: asset.path, name: asset.name,
    x: 0.5, y: 0.5, scale: 1, rotation: 0,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
    opacity: 1, blendMode: "normal", visible: true, locked: false,
    nodes: [], masks: [], edge: { ...NEUTRAL_EDGE },
    ...over,
  };
}

/**
 * The document's layers, with a single-source document folded into one.
 *
 * Documents predate layers, so the old shape -- one asset, one stack -- is still what is on disk for
 * anything edited before the merge. Rather than a migration pass that rewrites saved projects, the
 * old shape is READ as a one-layer document. Nothing has to be upgraded, nothing can be half
 * upgraded, and a project opened on an older build still makes sense.
 */
export function layersOf(document: PhotoDocument): PhotoLayer[] {
  if (document.layers?.length) return document.layers;
  return [{
    id: `${document.assetId}-base`,
    assetId: document.assetId,
    sourcePath: document.sourcePath,
    name: "Base",
    x: 0.5, y: 0.5, scale: 1, rotation: 0,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
    opacity: 1, blendMode: "normal", visible: true, locked: false,
    // The old document's stack belongs to its one layer, not to the composite: that keeps a grade
    // made before the merge applying to exactly the pixels it was made for.
    nodes: document.nodes, masks: document.masks, edge: { ...NEUTRAL_EDGE },
  }];
}

/**
 * Move a single-source document onto the layer model, in place of the read-through above.
 *
 * Called the moment something layer-shaped happens -- adding a second layer, changing a blend mode.
 * Until then the document is left in its old shape so that opening a project, looking at it and
 * closing it does not rewrite anything.
 */
export function withLayers(document: PhotoDocument): PhotoDocument {
  if (document.layers?.length) return document;
  return { ...document, layers: layersOf(document), nodes: [], masks: [] };
}

export const layerById = (document: PhotoDocument, layerId: string | undefined): PhotoLayer | undefined =>
  layerId ? layersOf(document).find((layer) => layer.id === layerId) : undefined;

/**
 * A new document starts with one grade node rather than an empty stack.
 *
 * An empty stack is technically correct and useless: the first thing anyone reaches for is
 * exposure, and making them press "add node" to find it is a worse editor for no benefit.
 */
export function createPhotoDocument(assetId: string, sourcePath: string): PhotoDocument {
  return {
    assetId, sourcePath,
    nodes: [createGradeNode()],
    masks: [],
    rotate: 0, flipHorizontal: false, flipVertical: false,
    updatedAt: new Date().toISOString(),
  };
}

/** Masks and nodes are looked up across the whole document, since either scope may own them. */
export const maskById = (document: PhotoDocument, maskId: string | undefined): PhotoMask | undefined => {
  if (!maskId) return undefined;
  const own = document.masks.find((mask) => mask.id === maskId);
  if (own) return own;
  for (const layer of layersOf(document)) {
    const found = layer.masks.find((mask) => mask.id === maskId);
    if (found) return found;
  }
  return undefined;
};

export const nodeById = (document: PhotoDocument, nodeId: string | undefined): PhotoNode | undefined => {
  if (!nodeId) return undefined;
  const own = document.nodes.find((node) => node.id === nodeId);
  if (own) return own;
  for (const layer of layersOf(document)) {
    const found = layer.nodes.find((node) => node.id === nodeId);
    if (found) return found;
  }
  return undefined;
};

// One definition of "this grade changes nothing", shared with the renderer that has to agree with
// it. There were two, and the copy here quietly did not know about levels, colour bands, the
// vignette or noise reduction -- so a picture with only those set reported itself untouched.
const isNodeNeutral = (node: PhotoNode): boolean =>
  !node.enabled || (node.kind === "grade" && isNeutralGrade(node.adjustments));

/** True when rebuilding this picture would hand back the source file unchanged. */
export function isUntouched(document: PhotoDocument): boolean {
  if (!isNeutralGeometry(geometryOf(document))) return false;
  if (document.nodes.some((node) => !isNodeNeutral(node))) return false;
  const layers = layersOf(document);
  if (layers.length > 1) return false;
  const base = layers[0];
  if (!base) return true;
  return base.visible && base.opacity === 1 && base.blendMode === "normal" && !base.maskId
    && base.scale === 1 && base.rotation === 0 && !base.attach && !base.generator
    && !base.nodes.some((node) => !isNodeNeutral(node));
}

/**
 * A stable name for a node's cached result.
 *
 * Keyed on everything that changes the output -- the source file, the mask that shaped it, and the
 * node itself -- so a cache entry can never be handed to a node that would have produced something
 * different. Changing any of them simply misses, which is the failure mode you want; the
 * alternative is a stale bitmap that looks almost right.
 */
export function removalCacheKey(document: PhotoDocument, node: RemoveNode): string {
  const mask = maskById(document, node.maskId);
  const strokes = mask?.strokes.map((stroke) =>
    `${stroke.erase ? "-" : "+"}${stroke.radius.toFixed(4)}:${stroke.hardness.toFixed(2)}:`
    + stroke.points.map(([x, y]) => `${x.toFixed(4)},${y.toFixed(4)}`).join(";")).join("|") ?? "";
  const key = [
    document.sourcePath,
    node.id,
    mask?.sourcePath ?? "",
    mask?.invert ? "inv" : "",
    String(mask?.feather ?? 0),
    strokes,
  ];
  // VFX Phase 2: a stack changes the region too. Only appended when there is one, so every key made
  // before stacks existed is unchanged and no finished fill is thrown away.
  if (mask?.stack?.length) key.push(JSON.stringify(mask.stack));
  return key.join(" ");
}
