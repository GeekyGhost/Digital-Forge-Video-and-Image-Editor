/**
 * VFX Phase 4: point tracks, and everything that follows one.
 *
 * - `trackAt` / `trackCanvasPoint`: a track's position at a (fractional) source frame, and where that
 *   lands on the canvas through the tracked layer's placement.
 * - `attachedLayer`: placement that follows a track. It returns a layer with adjusted x/y, and both
 *   compositors use it in `prepare`, so masks, nodes and the alpha cut follow for free.
 * - The trail (time-lagged warp): `trailField` builds one displacement table per layer per frame in
 *   TypeScript; `applyTrail` is the CPU reference and TRAIL_FS reads the same Float32 table, so the
 *   two compositors cannot disagree about the motion, only about float rounding in the resample.
 * - The emitter: `emitterParticles` turns (seed, frame) into particles with the integer hash matrix
 *   rain uses; `renderEmitter` is the CPU reference. The GPU draws the same particle list.
 * - Depth occlusion helpers: `depthFrameOf` and `depthThresholdLut`.
 * - Built-in trackers: `lucasKanade` (pyramidal, forward-backward checked) and `sam2Centroid`.
 *
 * Track time is the TRACKED layer's source frame (a track measured on a clip is indexed by that
 * clip's frames), so a track stays right however the clip is trimmed, retimed or moved in time.
 */

import { compositeOver } from "./photoBlend";
import { croppedRect, layerMatrix, placeLayer, type LayerSource } from "./photoCompose";
import type {
  EmitterGenerator, LayerAttach, LayerMedia, PhotoDocument, PhotoLayer, PointTrack, TrackSample, TrailNode,
} from "./photoDocument";
import { layersOf } from "./photoDocument";
import { applyMatrix, invert, type Matrix } from "./photoGeometry";
import { hash4 } from "./matrixRain";
import { sequenceFramePath } from "./photoMask";
export { depthThresholdLut } from "./photoMask";
import { emitterElementIndex, emitterSpawns, mediaFrameKey, mediaFramesAt } from "./photoTime";

export type CanvasSize = { width: number; height: number };

/* ================================================================ tracks */

export const trackById = (document: PhotoDocument, id: string | undefined): PointTrack | undefined =>
  id ? document.tracks?.find((track) => track.id === id) : undefined;

/**
 * Samples with every unseen one replaced by the nearest seen one (the earlier on a tie), as
 * [x, y] pairs; undefined when nothing was ever seen. Cached per track object: tracks are
 * immutable (the document is replaced, never mutated), so the object is the key.
 */
const held = new WeakMap<PointTrack, Float64Array | null>();

function heldSamples(track: PointTrack): Float64Array | undefined {
  const cached = held.get(track);
  if (cached !== undefined) return cached ?? undefined;
  const n = track.samples.length;
  const seen: number[] = [];
  for (let i = 0; i < n; i++) if (track.samples[i][2]) seen.push(i);
  if (!seen.length) {
    held.set(track, null);
    return undefined;
  }
  const out = new Float64Array(n * 2);
  let k = 0;
  for (let i = 0; i < n; i++) {
    while (k + 1 < seen.length && Math.abs(seen[k + 1] - i) < Math.abs(seen[k] - i)) k++;
    const [x, y] = track.samples[seen[k]];
    out[i * 2] = x;
    out[i * 2 + 1] = y;
  }
  held.set(track, out);
  return out;
}

/**
 * Where the track is at source frame `sourceFrame` (fractional allowed), in the tracked layer's
 * source pixels. Between samples it interpolates linearly; before the first and after the last, and
 * across frames where the point was not seen, it holds the nearest seen sample.
 */
export function trackAt(track: PointTrack, sourceFrame: number): [number, number] | undefined {
  const samples = heldSamples(track);
  if (!samples) return undefined;
  const last = track.samples.length - 1;
  const at = Math.min(last, Math.max(0, sourceFrame - track.first));
  const i0 = Math.floor(at), i1 = Math.min(last, i0 + 1), f = at - i0;
  return [
    samples[i0 * 2] * (1 - f) + samples[i1 * 2] * f,
    samples[i0 * 2 + 1] * (1 - f) + samples[i1 * 2 + 1] * f,
  ];
}

/**
 * The tracked layer's source frame at composition frame `frame`, `lag` composition frames earlier.
 *
 * "Now" is exactly the source frame the layer SHOWS (the crossfade's head), so an attached element
 * sits on the picture that is on screen. The lag is subtracted in source frames (it can be
 * fractional across rates, and the track interpolates). Off screen, the unclamped position is used,
 * and `trackAt` holds the end samples. A layer with no media counts in composition frames.
 */
export function trackTime(layer: PhotoLayer | undefined, frame: number, lag: number, rate: number): number {
  const media = layer?.media;
  if (!media) return frame - lag;
  const perFrame = media.fps / Math.max(1e-6, rate);
  const shown = mediaFramesAt(media, frame, rate);
  const now = shown
    ? shown[shown.length - 1].index
    : Math.floor(media.inPoint) + (Math.floor(frame) - Math.floor(media.start)) * perFrame;
  return now - lag * perFrame;
}

/** The size a layer's source pixels are measured in, for tracks and placement: its media, else undefined. */
const mediaSize = (layer: PhotoLayer): CanvasSize | undefined =>
  layer.media ? { width: layer.media.width, height: layer.media.height } : undefined;

/**
 * Where a track is on the canvas at composition frame `frame`, `lag` frames late, in CANVAS pixels
 * (full resolution): the track's source point through the tracked layer's own placement. The tracked
 * layer is taken as placed, NOT as attached (a plate is not attached to anything, and following its
 * attachment could recurse). A track on a layer without media is in canvas pixels already.
 */
export function trackCanvasPoint(
  document: PhotoDocument, track: PointTrack, frame: number, lag: number, canvas: CanvasSize, rate: number,
): [number, number] | undefined {
  const layer = layersOf(document).find((candidate) => candidate.id === track.layerId);
  const point = trackAt(track, trackTime(layer, frame, lag, rate));
  if (!point) return undefined;
  const size = layer ? mediaSize(layer) : undefined;
  if (!layer || !size) return point;
  const rect = croppedRect(layer, size);
  return applyMatrix(layerMatrix(layer, canvas.width, canvas.height, size, 1), point[0] - rect.left, point[1] - rect.top);
}

/**
 * A canvas point (normalised, as the pointer reports it) in a media layer's continuous source
 * pixels, through that layer's placement: the inverse of what `trackCanvasPoint` does to a sample.
 */
export function canvasToSourcePoint(layer: PhotoLayer, point: [number, number], canvas: CanvasSize): [number, number] | undefined {
  const size = mediaSize(layer);
  if (!size) return undefined;
  const rect = croppedRect(layer, size);
  const [x, y] = applyMatrix(invert(layerMatrix(layer, canvas.width, canvas.height, size, 1)), point[0] * canvas.width, point[1] * canvas.height);
  return [x + rect.left, y + rect.top];
}

/**
 * The layer as it is placed at `frame`: unchanged unless it is attached to a track, in which case
 * its x/y move by however far the track has moved since `anchorFrame`, `lag` frames late:
 *   x = x0 + (P(frame - lag) - P(anchor)).x / canvasWidth   (and y alike)
 * A missing track, or one never seen, leaves the layer where it is.
 */
export function attachedLayer(document: PhotoDocument, layer: PhotoLayer, frame: number, canvas: CanvasSize, rate: number): PhotoLayer {
  const offset = attachOffset(document, layer.attach, frame, canvas, rate);
  return offset ? { ...layer, x: layer.x + offset[0], y: layer.y + offset[1] } : layer;
}

/** The normalised offset an attachment adds at `frame`, or undefined when it adds nothing. */
export function attachOffset(
  document: PhotoDocument, attach: LayerAttach | undefined, frame: number, canvas: CanvasSize, rate: number,
): [number, number] | undefined {
  if (!attach) return undefined;
  const track = trackById(document, attach.trackId);
  if (!track) return undefined;
  const now = trackCanvasPoint(document, track, frame, attach.lag ?? 0, canvas, rate);
  const anchor = trackCanvasPoint(document, track, attach.anchorFrame, 0, canvas, rate);
  if (!now || !anchor) return undefined;
  return [(now[0] - anchor[0]) / canvas.width, (now[1] - anchor[1]) / canvas.height];
}

/** Every layer of the document as placed at `frame` (see `attachedLayer`). */
export function placedLayers(document: PhotoDocument, frame: number, canvas: CanvasSize, rate: number): PhotoLayer[] {
  return layersOf(document).map((layer) => attachedLayer(document, layer, frame, canvas, rate));
}

/* ================================================================ trail (time-lagged warp) */

export const TRAIL_BINS = 32;

/**
 * One trail node's displacement at one frame, ready for either compositor.
 *
 * `table` holds TRAIL_BINS (dx, dy) pairs in WORKING pixels: bin i is height h = i / (BINS - 1)
 * along the layer's up axis (0 its base, 1 its far end) and holds
 *   strength * (P(now - h * maxLag) - P(now))
 * where P is the track on the canvas. An output pixel samples the input at itself minus the
 * displacement for its height, so the far end shows where the layer was `maxLag` frames ago.
 * `inverse` maps a working pixel centre into the layer's cropped-source pixels and `rectHeight` is
 * that rectangle's height, which is how a pixel's height is found. Float32 so both compositors read
 * identical numbers.
 */
export type TrailField = { table: Float32Array; inverse: Matrix; rectHeight: number; still: boolean };

/**
 * Build the field for `node` on `layer` (already attached) at `frame`. `size` is the layer's source
 * size when it is placed through a matrix (a picture, text, a shape, media); undefined for layers
 * drawn at the working size (gradients, generators), whose frame is the canvas itself.
 * Undefined when the node's track is missing or was never seen.
 */
export function trailField(
  document: PhotoDocument, node: TrailNode, layer: PhotoLayer | undefined, size: CanvasSize | undefined,
  frame: number, rate: number, canvas: CanvasSize, width: number, height: number, canvasScale: number,
): TrailField | undefined {
  const track = trackById(document, node.trackId);
  if (!track) return undefined;
  const now = trackCanvasPoint(document, track, frame, 0, canvas, rate);
  if (!now) return undefined;
  const table = new Float32Array(TRAIL_BINS * 2);
  const lag = Math.max(0, node.maxLag);
  const strength = node.strength * canvasScale;
  let still = true;
  for (let bin = 0; bin < TRAIL_BINS; bin++) {
    const h = bin / (TRAIL_BINS - 1);
    const then = trackCanvasPoint(document, track, frame, h * lag, canvas, rate) ?? now;
    table[bin * 2] = (then[0] - now[0]) * strength;
    table[bin * 2 + 1] = (then[1] - now[1]) * strength;
    if (table[bin * 2] !== 0 || table[bin * 2 + 1] !== 0) still = false;
  }
  if (layer && size) {
    const rect = croppedRect(layer, size);
    return { table, inverse: invert(layerMatrix(layer, width, height, size, canvasScale)), rectHeight: rect.height, still };
  }
  return { table, inverse: { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }, rectHeight: height, still };
}

/** The displacement at a working pixel centre (px, py): the table interpolated at the pixel's height. */
export function trailDisplacement(field: TrailField, px: number, py: number): [number, number] {
  const { inverse: m, table } = field;
  const ly = m.b * px + m.d * py + m.f;
  const h = Math.min(1, Math.max(0, 1 - ly / field.rectHeight));
  const b = h * (TRAIL_BINS - 1);
  const i0 = Math.floor(b), i1 = Math.min(TRAIL_BINS - 1, i0 + 1), f = b - i0;
  return [
    table[i0 * 2] * (1 - f) + table[i1 * 2] * f,
    table[i0 * 2 + 1] * (1 - f) + table[i1 * 2 + 1] * f,
  ];
}

/**
 * The CPU reference for TRAIL_FS: resample `pixels` through the field. Bilinear on premultiplied
 * values (a transparent pixel's colour cannot bleed into smoke), texels outside the frame are
 * transparent, and the result is unpremultiplied and rounded as a Uint8ClampedArray rounds.
 */
export function applyTrail(pixels: Uint8ClampedArray, width: number, height: number, field: TrailField): Uint8ClampedArray {
  const out = new Uint8ClampedArray(pixels.length);
  const at = (x: number, y: number) => (x < 0 || y < 0 || x >= width || y >= height) ? -1 : (y * width + x) * 4;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const px = x + 0.5, py = y + 0.5;
      const [dx, dy] = trailDisplacement(field, px, py);
      const sx = px - dx - 0.5, sy = py - dy - 0.5;
      const x0 = Math.floor(sx), y0 = Math.floor(sy);
      const fx = sx - x0, fy = sy - y0;
      let r = 0, g = 0, b = 0, a = 0;
      const weights = [(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy];
      const corners = [at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1)];
      for (let k = 0; k < 4; k++) {
        const c = corners[k];
        if (c < 0) continue;
        const w = weights[k] * pixels[c + 3];
        r += pixels[c] * w; g += pixels[c + 1] * w; b += pixels[c + 2] * w; a += w;
      }
      if (a <= 0) continue;
      const o = (y * width + x) * 4;
      out[o] = r / a; out[o + 1] = g / a; out[o + 2] = b / a; out[o + 3] = a;
    }
  }
  return out;
}

/* ================================================================ emitter */

/** One living particle, placed: canvas-normalised centre, like a layer's x/y. */
export type Particle = {
  spawn: number; index: number; age: number;
  x: number; y: number; scale: number; rotation: number; opacity: number;
  /** The element's source frame it shows. */
  frameIndex: number;
};

const unit = (value: number) => value / 4294967296;
const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

/**
 * The particles alive at `frame`, oldest first. Every number is a pure function of (seed, spawn
 * frame, index): three 32-bit hashes give the x and y spread and the initial rotation. A particle
 * born on a track starts at the track's canvas point at its spawn frame, and then keeps its own
 * velocity -- it does not follow the track after it leaves.
 */
export function emitterParticles(
  document: PhotoDocument, generator: EmitterGenerator, frame: number, rate: number, canvas: CanvasSize,
): Particle[] {
  const seed = Math.round(generator.seed) >>> 0;
  const lifetime = Math.max(1, Math.floor(generator.lifetime));
  const track = trackById(document, generator.trackId);
  const out: Particle[] = [];
  const spawnPoints = new Map<number, [number, number]>();
  for (const { spawn, index, age } of emitterSpawns(generator, frame)) {
    let origin = spawnPoints.get(spawn);
    if (!origin) {
      const point = track ? trackCanvasPoint(document, track, spawn, 0, canvas, rate) : undefined;
      origin = point ? [point[0] / canvas.width, point[1] / canvas.height] : [generator.x, generator.y];
      spawnPoints.set(spawn, origin);
    }
    const ux = unit(hash4(seed, spawn, index, 0));
    const uy = unit(hash4(seed, spawn, index, 1));
    const ur = unit(hash4(seed, spawn, index, 2));
    const t = age / lifetime;
    out.push({
      spawn, index, age,
      x: origin[0] + age * (generator.velocityX + generator.spread * (2 * ux - 1)),
      y: origin[1] + age * (generator.velocityY + generator.spread * (2 * uy - 1)),
      scale: Math.max(0, lerp(generator.scaleStart, generator.scaleEnd, t)),
      rotation: generator.spin * (2 * ur - 1),
      opacity: Math.min(1, Math.max(0, lerp(generator.opacityStart, generator.opacityEnd, t))),
      frameIndex: emitterElementIndex(generator.element, age, rate),
    });
  }
  return out;
}

/** A particle as the placement fields of a layer, for `placeLayer` / `layerMatrix`. */
export function particleLayer(particle: Particle): PhotoLayer {
  return {
    id: "", assetId: "", sourcePath: "", name: "", visible: true, locked: false, opacity: 1, nodes: [], masks: [],
    x: particle.x, y: particle.y, scale: particle.scale, rotation: particle.rotation,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0, blendMode: "normal",
  } as unknown as PhotoLayer;
}

/** The source key every particle needs (the frame store fetches these; see `mediaNeedsAt`). */
export const particleKey = (element: LayerMedia, particle: Particle, useProxy = false) =>
  mediaFrameKey(element, particle.frameIndex, useProxy);

/**
 * CPU reference: every particle placed like a layer and composited Normal at its opacity, oldest
 * first, into `out` (transparent to start). False when a frame it needs has not loaded.
 */
export function renderEmitter(
  out: Uint8ClampedArray, particles: Particle[], element: LayerMedia, sources: Map<string, LayerSource>,
  width: number, height: number, canvasScale: number, useProxy = false,
): boolean {
  for (const particle of particles) {
    if (particle.opacity <= 0 || particle.scale <= 0) continue;
    const source = sources.get(particleKey(element, particle, useProxy));
    if (!source) return false;
    compositeOver(out, placeLayer(particleLayer(particle), source, width, height, canvasScale), "normal", particle.opacity);
  }
  return true;
}

/* ================================================================ depth occlusion */

/** The depth frame a `depth` mask entry reads: `layerId`'s depth sequence at the frame it shows. */
export type DepthFrameRef = { path: string; layer: PhotoLayer; width: number; height: number };

export function depthFrameOf(layers: Map<string, PhotoLayer>, layerId: string, frame: number, rate: number): DepthFrameRef | undefined {
  const layer = layers.get(layerId);
  const media = layer?.media;
  if (!layer || !media?.depth) return undefined;
  const shown = mediaFramesAt(media, frame, rate);
  if (!shown) return undefined;
  const path = sequenceFramePath({ kind: "sequence", cachePath: media.depth.cachePath, frameCount: media.depth.frameCount }, shown[shown.length - 1].index);
  return { path, layer, width: media.width, height: media.height };
}

/** Every depth frame the document's masks read at `frame`: the loader's fetch list. */
export function depthFramePaths(document: PhotoDocument, frame: number, rate: number): string[] {
  const layers = new Map(layersOf(document).map((layer) => [layer.id, layer]));
  const masks = [...document.masks, ...layersOf(document).flatMap((layer) => layer.masks)];
  const paths = new Set<string>();
  for (const mask of masks) {
    for (const entry of mask.stack ?? []) {
      if (entry.source.kind !== "depth") continue;
      const ref = depthFrameOf(layers, entry.source.layerId, frame, rate);
      if (ref) paths.add(ref.path);
    }
  }
  return [...paths];
}

/* ================================================================ built-in trackers */

/** A single-channel frame, 0..255 floats (luma). */
export type GreyFrame = { data: Float32Array; width: number; height: number };

/** Rec. 709 luma of an RGBA buffer. */
export function greyOf(rgba: Uint8ClampedArray | Uint8Array, width: number, height: number): GreyFrame {
  const data = new Float32Array(width * height);
  for (let i = 0, at = 0; i < data.length; i++, at += 4) data[i] = 0.2126 * rgba[at] + 0.7152 * rgba[at + 1] + 0.0722 * rgba[at + 2];
  return { data, width, height };
}

function downsample(frame: GreyFrame): GreyFrame {
  const width = Math.max(1, frame.width >> 1), height = Math.max(1, frame.height >> 1);
  const data = new Float32Array(width * height);
  const src = frame.data, w = frame.width, h = frame.height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const x0 = Math.min(w - 1, x * 2), x1 = Math.min(w - 1, x * 2 + 1);
      const y0 = Math.min(h - 1, y * 2), y1 = Math.min(h - 1, y * 2 + 1);
      data[y * width + x] = (src[y0 * w + x0] + src[y0 * w + x1] + src[y1 * w + x0] + src[y1 * w + x1]) / 4;
    }
  }
  return { data, width, height };
}

function sample(frame: GreyFrame, x: number, y: number): number {
  const { data, width, height } = frame;
  const cx = Math.min(width - 1.001, Math.max(0, x)), cy = Math.min(height - 1.001, Math.max(0, y));
  const x0 = Math.floor(cx), y0 = Math.floor(cy), fx = cx - x0, fy = cy - y0;
  const i = y0 * width + x0;
  const top = data[i] * (1 - fx) + data[i + 1] * fx;
  const bottom = data[i + width] * (1 - fx) + data[i + width + 1] * fx;
  return top * (1 - fy) + bottom * fy;
}

type Pyramid = GreyFrame[];
const pyramidOf = (frame: GreyFrame, levels: number): Pyramid => {
  const out = [frame];
  for (let level = 1; level < levels; level++) out.push(downsample(out[level - 1]));
  return out;
};

export type LucasKanadeOptions = { levels?: number; window?: number; iterations?: number; forwardBackward?: number; anchor?: number };

/**
 * One pyramidal Lucas-Kanade step: where the point at (x, y) in `from` is in `to`. Undefined when
 * the window has no texture to lock onto (a singular structure tensor) or it leaves the frame.
 */
function lkStep(
  from: Pyramid, to: Pyramid, x: number, y: number, half: number, iterations: number, guess: [number, number] = [0, 0],
): [number, number] | undefined {
  // The guess (a level-0 displacement) enters at the coarsest level, scaled down to it.
  let gx = guess[0] / 2 ** (from.length - 1), gy = guess[1] / 2 ** (from.length - 1);
  for (let level = from.length - 1; level >= 0; level--) {
    const scale = 2 ** level;
    const I = from[level], J = to[level];
    const px = x / scale, py = y / scale;
    let sxx = 0, sxy = 0, syy = 0;
    const ix: number[] = [], iy: number[] = [], iv: number[] = [];
    for (let v = -half; v <= half; v++) {
      for (let u = -half; u <= half; u++) {
        const dx = (sample(I, px + u + 1, py + v) - sample(I, px + u - 1, py + v)) / 2;
        const dy = (sample(I, px + u, py + v + 1) - sample(I, px + u, py + v - 1)) / 2;
        ix.push(dx); iy.push(dy); iv.push(sample(I, px + u, py + v));
        sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
      }
    }
    const det = sxx * syy - sxy * sxy;
    if (det < 1e-6 * (half * 2 + 1) ** 4) return undefined;
    let vx = 0, vy = 0;
    for (let iteration = 0; iteration < iterations; iteration++) {
      let bx = 0, by = 0, k = 0;
      for (let v = -half; v <= half; v++) {
        for (let u = -half; u <= half; u++, k++) {
          const diff = iv[k] - sample(J, px + gx + vx + u, py + gy + vy + v);
          bx += diff * ix[k]; by += diff * iy[k];
        }
      }
      const ex = (syy * bx - sxy * by) / det;
      const ey = (sxx * by - sxy * bx) / det;
      vx += ex; vy += ey;
      if (ex * ex + ey * ey < 1e-4) break;
    }
    if (level > 0) { gx = 2 * (gx + vx); gy = 2 * (gy + vy); } else { gx += vx; gy += vy; }
  }
  const nx = x + gx, ny = y + gy;
  const frame = to[0];
  if (!Number.isFinite(nx) || !Number.isFinite(ny) || nx < 0 || ny < 0 || nx > frame.width - 1 || ny > frame.height - 1) return undefined;
  return [nx, ny];
}

/**
 * Track one point through `frames` from `frames[0]`: pyramidal Lucas-Kanade (3 levels, 15 x 15
 * window) frame to frame, with a forward-backward check -- a step whose reverse lands more than
 * `forwardBackward` px (1) from where it started is marked unseen. An unseen step keeps going from
 * its best guess (so a brief occlusion can be recovered) or holds the last position when it failed
 * outright. Deterministic: plain float64 arithmetic, no randomness.
 *
 * Frame-to-frame steps drift (each one is pulled a little by whatever else is in the window, and
 * the errors add up), so every step is then refined against the ORIGINAL template -- the first
 * frame's window -- starting from the step's answer, and the refinement is kept when it lands within
 * `anchor` px (2) of it. That keeps the error bounded instead of growing with the length of the clip
 * while the point looks like it did (Matthews, Ishikawa and Baker's drift-corrected template update).
 */
export function lucasKanade(frames: GreyFrame[], start: [number, number], options: LucasKanadeOptions = {}): TrackSample[] {
  const levels = options.levels ?? 3, half = Math.floor((options.window ?? 15) / 2);
  const iterations = options.iterations ?? 20, limit = options.forwardBackward ?? 1, anchor = options.anchor ?? 2;
  if (!frames.length) return [];
  const out: TrackSample[] = [[start[0], start[1], 1]];
  const template = pyramidOf(frames[0], levels);
  const x0 = start[0] - 0.5, y0 = start[1] - 0.5;
  let previous = template;
  // Samples are continuous pixel coordinates (pixel i spans [i, i + 1), as placement measures them);
  // the solver works on texel centres, which sit at integers.
  let x = start[0] - 0.5, y = start[1] - 0.5;
  for (let index = 1; index < frames.length; index++) {
    const next = pyramidOf(frames[index], levels);
    const forward = lkStep(previous, next, x, y, half, iterations);
    if (!forward) {
      out.push([x + 0.5, y + 0.5, 0]);
    } else {
      const back = lkStep(next, previous, forward[0], forward[1], half, iterations);
      const ok = back && Math.hypot(back[0] - x, back[1] - y) <= limit;
      [x, y] = forward;
      const refined = index > 1 ? lkStep(template, next, x0, y0, half, iterations, [x - x0, y - y0]) : undefined;
      if (refined && Math.hypot(refined[0] - x, refined[1] - y) <= anchor) [x, y] = refined;
      out.push([x + 0.5, y + 0.5, ok ? 1 : 0]);
    }
    previous = next;
  }
  return out;
}

/**
 * Track from a frame in the middle: backwards to the first frame and forwards to the last, joined
 * at `query`. samples[i] is frames[i].
 */
export function lucasKanadeBothWays(frames: GreyFrame[], query: number, start: [number, number], options?: LucasKanadeOptions): TrackSample[] {
  const q = Math.min(frames.length - 1, Math.max(0, query));
  const forward = lucasKanade(frames.slice(q), start, options);
  const backward = lucasKanade(frames.slice(0, q + 1).reverse(), start, options).reverse();
  return [...backward.slice(0, -1), ...forward];
}

/**
 * The cheap source: the alpha-weighted centroid of each SAM 2 matte frame (coverage bytes), in
 * continuous pixel coordinates (a pixel's centre is i + 0.5). A frame with less than one pixel's
 * worth of coverage is unseen.
 */
export function sam2Centroid(masks: Array<{ coverage: Uint8Array; width: number; height: number }>): TrackSample[] {
  return masks.map(({ coverage, width }) => {
    let sum = 0, sx = 0, sy = 0;
    for (let i = 0; i < coverage.length; i++) {
      const w = coverage[i];
      if (!w) continue;
      sum += w; sx += w * (i % width); sy += w * Math.floor(i / width);
    }
    return sum < 255 ? [0, 0, 0] as TrackSample : [sx / sum + 0.5, sy / sum + 0.5, 1] as TrackSample;
  });
}

/** A new track's id. */
export const newTrackId = () => `track-${Math.random().toString(36).slice(2, 10)}`;
