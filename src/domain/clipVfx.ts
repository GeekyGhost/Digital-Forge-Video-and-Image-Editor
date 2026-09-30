/**
 * VFX Phase 4b: the Clip Editor's VFX mode.
 *
 * Video VFX (elements, tracks, trails, emitters, depth occlusion, SAM 2 video mattes) used to live in
 * Photo Edit, which is for pictures and single frames. They now run on a timeline clip: the clip's
 * source range becomes the plate, a video layer at composition frame 0, and the finished composition
 * is rendered to an MP4 that goes back onto the timeline over the same range.
 *
 * Pure: the document shape is Photo Edit's own (`PhotoDocument`), so both compositors, every panel and
 * the parity fixtures work unchanged. The documents are kept in `Project.clipVfx`, by clip id.
 */

import { createMediaLayer, createPhotoDocument, layersOf, type PhotoDocument, type PhotoLayer } from "./photoDocument";
import { compositionDuration, compositionRate } from "./photoTime";
import { mediaFromVideoAsset } from "./elements";
import { seconds } from "./time";
import type { MediaAsset, TimelineClip } from "./types";

/** The plate layer's id for a clip, stable so a rebuilt document keeps its tracks' `layerId`. */
export const plateLayerId = (clipId: string) => `${clipId}-plate`;

/**
 * A new VFX document for `clip`: its source range [rangeStart, rangeEnd) (seconds, relative to the
 * clip's in point, the Clip Editor's own range) as a plate at frame 0, at the clip's frame rate, as
 * long as the range. Undefined when the asset has not been probed (no size or length).
 */
export function clipPlateDocument(
  clip: Pick<TimelineClip, "id" | "sourceIn">, asset: MediaAsset, rangeStart: number, rangeEnd: number,
): PhotoDocument | undefined {
  const media = mediaFromVideoAsset(asset, 0);
  if (!media) return undefined;
  const fps = media.fps;
  const base = seconds(clip.sourceIn);
  const inPoint = Math.min(media.frameCount - 1, Math.max(0, Math.round((base + rangeStart) * fps)));
  const outPoint = Math.min(media.frameCount, Math.max(inPoint + 1, Math.round((base + rangeEnd) * fps)));
  const plate: PhotoLayer = {
    ...createMediaLayer("Plate", { ...media, inPoint, outPoint, start: 0 }, { assetId: asset.id, locked: true }),
    id: plateLayerId(clip.id),
  };
  return {
    ...createPhotoDocument(asset.id, asset.path),
    layers: [plate],
    frameRate: fps,
    duration: outPoint - inPoint,
  };
}

/**
 * The stored document if it still belongs to this clip's footage, otherwise a fresh one. A clip whose
 * range was already replaced by a render points at the RENDER now; its old document described the
 * original footage, so reopening starts clean on what the clip actually shows.
 */
export function clipVfxDocument(
  stored: PhotoDocument | undefined, clip: Pick<TimelineClip, "id" | "sourceIn">, asset: MediaAsset,
  rangeStart: number, rangeEnd: number,
): PhotoDocument | undefined {
  if (stored && belongsToClip(stored, clip.id, asset.id)) return stored;
  return clipPlateDocument(clip, asset, rangeStart, rangeEnd);
}

/** A stored document is this clip's while it was built on the asset the clip still shows. */
export const belongsToClip = (document: PhotoDocument, clipId: string, assetId: string) =>
  document.assetId === assetId && layersOf(document).some((layer) => layer.id === plateLayerId(clipId));

/**
 * The same document over a different range of the clip: the plate's in and out points move and the
 * composition takes the new length. Tracks and SAM 2 mattes are stored in the plate's SOURCE frames,
 * so they stay where they were on the footage; an attached layer keeps following.
 */
export function withPlateRange(
  document: PhotoDocument, clip: Pick<TimelineClip, "id" | "sourceIn">, rangeStart: number, rangeEnd: number,
): PhotoDocument {
  const id = plateLayerId(clip.id);
  const plate = layersOf(document).find((layer) => layer.id === id);
  const media = plate?.media;
  if (!media) return document;
  const base = seconds(clip.sourceIn);
  const inPoint = Math.min(media.frameCount - 1, Math.max(0, Math.round((base + rangeStart) * media.fps)));
  const outPoint = Math.min(media.frameCount, Math.max(inPoint + 1, Math.round((base + rangeEnd) * media.fps)));
  const rate = compositionRate(document);
  return {
    ...document,
    duration: Math.max(1, Math.round((outPoint - inPoint) * rate / media.fps)),
    layers: layersOf(document).map((layer) => layer.id === id ? { ...layer, media: { ...media, inPoint, outPoint, start: 0 } } : layer),
  };
}

export type ClipRenderPlan = {
  /** Composition frames to render, 0 .. frames - 1, at `fps`. */
  frames: number;
  fps: number;
  /** The range the render replaces, in seconds relative to the clip's in point (Clip Editor units). */
  replaceStart: number;
  replaceEnd: number;
  /** The plate's own audio for the same span, muxed into the render so VFX never drops the sound. */
  audio?: { path: string; start: number; duration: number };
};

/**
 * What a render of `document` covers on the clip. Composition frame 0 is the plate's `start`, so the
 * replaced range begins at the plate's in point (plus any frames before the plate starts are its
 * lead-in, which count against nothing). The end is where the composition ends, clamped to the clip.
 */
export function clipRenderPlan(
  document: PhotoDocument, clip: Pick<TimelineClip, "id" | "sourceIn" | "sourceOut">,
): ClipRenderPlan | undefined {
  const plate = layersOf(document).find((layer) => layer.id === plateLayerId(clip.id));
  const media = plate?.media;
  if (!media) return undefined;
  const fps = compositionRate(document);
  const frames = compositionDuration(document);
  const base = seconds(clip.sourceIn);
  const span = seconds(clip.sourceOut) - base;
  const plateAt = (media.inPoint - media.start * media.fps / fps) / media.fps;
  const replaceStart = Math.min(span, Math.max(0, plateAt - base));
  const replaceEnd = Math.min(span, Math.max(replaceStart, plateAt - base + frames / fps));
  return {
    frames, fps, replaceStart, replaceEnd,
    audio: { path: media.path, start: Math.max(0, plateAt), duration: frames / fps },
  };
}

/** True when a Photo Edit document carries video-only content made before Phase 4b (for a notice). */
export function hasVideoContent(document: PhotoDocument): boolean {
  return Boolean(document.tracks?.length) || layersOf(document).some((layer) => Boolean(layer.media)
    || layer.generator?.kind === "emitter" || layer.attach !== undefined
    || layer.nodes.some((node) => node.kind === "trail"));
}
