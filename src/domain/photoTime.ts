/**
 * Time in a composition (VFX Phase 3): which source frames a media layer shows at a composition
 * frame, how those frames are named, and the loop crossfade that blends them.
 *
 * Pure and synchronous. Decoding lives elsewhere (src/render/video/): the caller fetches the frames
 * this module names, ahead of time, into the same `sources` map pictures already use, and both
 * compositors look them up by key. That keeps the compositors synchronous and identical, and it is
 * why a frame that has not arrived yet is simply "still loading", like a picture.
 */

import type { EmitterGenerator, LayerMedia, PhotoDocument, PhotoLayer } from "./photoDocument";
import { layersOf } from "./photoDocument";

/** One source frame a layer shows, and how much of it (1 unless a loop crossfade is running). */
export type MediaFrame = { index: number; weight: number };

export const DEFAULT_FRAME_PATTERN = "frame_%04d.png";

/** The composition's rate: the document's own, else the first media layer's, else 24. */
export function compositionRate(document: PhotoDocument): number {
  if (document.frameRate && document.frameRate > 0) return document.frameRate;
  for (const layer of layersOf(document)) if (layer.media?.fps) return layer.media.fps;
  return 24;
}

/** The composition's length in frames: the document's own, else the longest layer (a loop counts once). */
export function compositionDuration(document: PhotoDocument): number {
  if (document.duration && document.duration > 0) return Math.round(document.duration);
  const rate = compositionRate(document);
  let longest = 1;
  for (const layer of layersOf(document)) {
    const media = layer.media;
    if (media) {
      const length = Math.max(1, media.outPoint - media.inPoint);
      longest = Math.max(longest, media.start + Math.ceil(length * rate / Math.max(1e-6, media.fps)));
    }
    // Rain loops exactly at its loop length; an emitter runs for ever, so it asks for one lifetime.
    if (layer.generator?.kind === "matrixRain") longest = Math.max(longest, layer.generator.loopFrames);
    else if (layer.generator?.kind === "emitter") longest = Math.max(longest, Math.round(layer.generator.lifetime));
  }
  return longest;
}

/** True when anything in the document moves, so the time bar is worth showing. */
export function isAnimated(document: PhotoDocument): boolean {
  return layersOf(document).some((layer) => Boolean(layer.media || layer.generator));
}

/**
 * The source frames `media` shows at composition frame `frame`, or undefined when the layer is not
 * on screen then (before its start, or after its out point when it does not loop).
 *
 * A loop of length L = out - in with a crossfade of C frames repeats every L - C frames. In the
 * first C frames of each period the tail (in + L - C + p) is blended into the head (in + p), with
 * the head's weight rising (p + 1) / (C + 1), so the last frame before the wrap and the first frame
 * after it are neighbours in the source. The weights are exact fractions; no frame is dropped.
 */
export function mediaFramesAt(media: LayerMedia, frame: number, compositionFps: number): MediaFrame[] | undefined {
  const elapsed = Math.floor(frame) - Math.floor(media.start);
  if (elapsed < 0) return undefined;
  const inPoint = Math.max(0, Math.min(media.frameCount - 1, Math.floor(media.inPoint)));
  const outPoint = Math.max(inPoint + 1, Math.min(media.frameCount, Math.floor(media.outPoint)));
  const length = outPoint - inPoint;
  // Same rate: the identity, with no division to drift. Different rates: the source frame on screen
  // at that instant (a tiny epsilon keeps an exact multiple from landing one frame early).
  const position = media.fps === compositionFps
    ? elapsed
    : Math.floor(elapsed * media.fps / compositionFps + 1e-9);
  if (!media.loop) return position < length ? [{ index: inPoint + position, weight: 1 }] : undefined;
  const fade = Math.max(0, Math.min(Math.floor(media.crossfade || 0), Math.floor(length / 2)));
  const period = length - fade;
  const p = position % period;
  if (p < fade) {
    const head = (p + 1) / (fade + 1);
    return [{ index: inPoint + length - fade + p, weight: 1 - head }, { index: inPoint + p, weight: head }];
  }
  return [{ index: inPoint + p, weight: 1 }];
}

/** "frame_%04d.png" with 12 -> "frame_0012.png". Only %d and %0Nd are understood. */
export function formatPattern(pattern: string, index: number): string {
  return pattern.replace(/%(0?)(\d*)d/, (_, zero: string, width: string) =>
    String(index).padStart(Number(width || 0), zero ? "0" : " "));
}

function join(folder: string, name: string): string {
  const separator = folder.includes("\\") && !folder.includes("/") ? "\\" : "/";
  return `${folder.replace(/[\\/]+$/, "")}${separator}${name}`;
}

/**
 * The key a frame is loaded and looked up under. A frames layer's key is the PNG's own path, so the
 * ordinary picture loader serves it; a video frame is `<file>#<index>`, served by the frame store.
 */
export function mediaFrameKey(media: LayerMedia, index: number, useProxy = false): string {
  if (useProxy && media.proxyPath) return `${media.proxyPath}#${index}`;
  if (media.kind === "frames") return join(media.path, formatPattern(media.pattern ?? DEFAULT_FRAME_PATTERN, index));
  return `${media.path}#${index}`;
}

/** Parse a `<file>#<index>` key back, for the frame store. */
export function parseVideoFrameKey(key: string): { path: string; index: number } | undefined {
  const at = key.lastIndexOf("#");
  if (at < 0) return undefined;
  const index = Number(key.slice(at + 1));
  return Number.isInteger(index) && index >= 0 ? { path: key.slice(0, at), index } : undefined;
}

/**
 * Every source key the document needs at `frame`: the fetch list for the frame store and the
 * picture loader. The primary (head) frame of a crossfade comes last.
 */
export function mediaKeysAt(document: PhotoDocument, frame: number, useProxy = false): string[] {
  const rate = compositionRate(document);
  const keys: string[] = [];
  for (const need of mediaNeedsAt(document, frame, rate)) keys.push(mediaFrameKey(need.media, need.index, useProxy));
  return keys;
}

/**
 * Every (media, source frame) the document shows at `frame`: media layers' frames (the crossfade's
 * head last) and, VFX Phase 4, the element frames of every living emitter particle. The frame store
 * fetches exactly these.
 */
export function mediaNeedsAt(document: PhotoDocument, frame: number, rate = compositionRate(document)): Array<{ media: LayerMedia; index: number }> {
  const out: Array<{ media: LayerMedia; index: number }> = [];
  const seen = new Set<string>();
  for (const layer of layersOf(document)) {
    if (layer.media) {
      for (const shown of mediaFramesAt(layer.media, frame, rate) ?? []) out.push({ media: layer.media, index: shown.index });
    }
    // VFX Phase 7: a mask VFX node's imported effect clip, at the frame the node resolved.
    for (const node of layer.nodes) {
      if (node.kind === "maskVfx" && node.enabled && node.recipe.matteStyle === "clip" && node.effectClip) {
        out.push({ media: node.effectClip.media, index: node.effectClip.index });
      }
    }
    if (layer.generator?.kind === "emitter" && layer.visible) {
      const element = layer.generator.element;
      for (const spawn of emitterSpawns(layer.generator, frame)) {
        const index = emitterElementIndex(element, spawn.age, rate);
        const key = `${element.path}|${index}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({ media: element, index });
      }
    }
  }
  return out;
}

/* ================================================================ VFX Phase 4: emitter schedule */

/** More than this many particles alive at once keeps the newest: a runaway rate cannot hang a frame. */
export const MAX_PARTICLES = 400;

/** One particle: born at composition frame `spawn` as the `index`-th of that frame, `age` frames ago. */
export type EmitterSpawn = { spawn: number; index: number; age: number };

/**
 * The particles alive at `frame`, oldest first (the draw order). Frame s spawns
 * floor((s + 1) rate) - floor(s rate) particles, so a fractional rate accumulates (0.25 is one every
 * four frames) with no state: any frame can be asked for on its own. Nothing spawns before frame 0.
 */
export function emitterSpawns(generator: EmitterGenerator, frame: number): EmitterSpawn[] {
  const now = Math.max(0, Math.floor(frame));
  const lifetime = Math.max(1, Math.floor(generator.lifetime));
  const rate = Math.max(0, Math.min(32, generator.rate || 0));
  const out: EmitterSpawn[] = [];
  if (rate <= 0) return out;
  for (let spawn = Math.max(0, now - lifetime + 1); spawn <= now; spawn++) {
    const count = Math.floor((spawn + 1) * rate) - Math.floor(spawn * rate);
    for (let index = 0; index < count; index++) out.push({ spawn, index, age: now - spawn });
  }
  return out.length > MAX_PARTICLES ? out.slice(out.length - MAX_PARTICLES) : out;
}

/**
 * The element frame a particle of this age shows: the element played from its in point at its own
 * rate, looping (a particle can outlive its element), exactly as `mediaFramesAt` would play it.
 */
export function emitterElementIndex(element: LayerMedia, age: number, compositionFps: number): number {
  const shown = mediaFramesAt({ ...element, start: 0, loop: true, crossfade: 0 }, Math.max(0, Math.floor(age)), compositionFps);
  return shown ? shown[shown.length - 1].index : Math.max(0, Math.floor(element.inPoint));
}

/**
 * The frame index a layer's own mask sequences follow: the source frame on screen for a media
 * layer (a SAM 2 matte tracked on a clip is indexed by that clip's frames), the composition frame
 * for anything else.
 */
export function scopeFrame(layer: PhotoLayer | undefined, frames: MediaFrame[] | undefined, frame: number): number {
  if (layer?.media && frames?.length) return frames[frames.length - 1].index;
  return frame;
}

/**
 * Blend two placed frames of the same layer (a loop crossfade), premultiplied so the colour of a
 * transparent pixel in either frame cannot leak in:
 *   a = a0 (1 - w) + a1 w;  rgb = (rgb0 a0 (1 - w) + rgb1 a1 w) / a
 * in 0..1, written as a Uint8ClampedArray rounds. CROSSFADE_FS does the same arithmetic.
 */
export function crossfadeFrames(from: Uint8ClampedArray, to: Uint8ClampedArray, weight: number): Uint8ClampedArray {
  const out = new Uint8ClampedArray(from.length);
  const w = Math.min(1, Math.max(0, weight));
  for (let at = 0; at < out.length; at += 4) {
    const a0 = from[at + 3] / 255 * (1 - w);
    const a1 = to[at + 3] / 255 * w;
    const alpha = a0 + a1;
    if (alpha <= 0) continue;
    out[at] = (from[at] * a0 + to[at] * a1) / alpha;
    out[at + 1] = (from[at + 1] * a0 + to[at + 1] * a1) / alpha;
    out[at + 2] = (from[at + 2] * a0 + to[at + 2] * a1) / alpha;
    out[at + 3] = alpha * 255;
  }
  return out;
}

/** The alpha curve node: a' = 255 * min(1, gain * (a / 255) ^ gamma), rounded as a Uint8ClampedArray does. */
export function applyAlphaCurve(pixels: Uint8ClampedArray, gain: number, gamma: number): void {
  const g = Math.max(0, gain), power = Math.max(0.01, gamma);
  for (let at = 3; at < pixels.length; at += 4) {
    const a = pixels[at];
    if (a === 0) continue;
    pixels[at] = Math.min(1, g * Math.pow(a / 255, power)) * 255;
  }
}
