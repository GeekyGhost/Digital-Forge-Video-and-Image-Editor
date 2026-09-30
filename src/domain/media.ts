import type { MediaAsset } from "./types";

// Audio files can contain embedded album art. FFprobe exposes that artwork as a video stream, so
// extension-first classification is intentional: an MP3 with a JPEG cover is still an audio asset.
const AUDIO_FILE = /\.(mp3|wav|flac|m4a|aac|ogg|opus|wma)$/i;
const IMAGE_FILE = /\.(png|jpe?g|webp|gif|bmp|tiff?)$/i;

export const isAudioPath = (path: string) => AUDIO_FILE.test(path);
export const isImagePath = (path: string) => IMAGE_FILE.test(path);

export function importedAssetKind(path: string, hasVideo: boolean): "image" | "video" | "audio" {
  if (isAudioPath(path)) return "audio";
  if (isImagePath(path)) return "image";
  return hasVideo ? "video" : "audio";
}

/** Also repairs routing for audio assets saved by an older importer as `video`. */
export const isAudioAsset = (asset: Pick<MediaAsset, "kind" | "path">) =>
  asset.kind === "audio" || isAudioPath(asset.path);

/**
 * A still picture: one frame, with no timeline of its own.
 *
 * Decided by the file rather than by `kind`, because `kind` cannot answer it. "generated" is used
 * for anything a model produced, and that includes video -- so treating it as a picture puts MP4s in
 * the Photo Edit bin, and treating it as video sends PNGs through a frame extractor. The extension
 * is the thing that actually knows.
 */
export const isStillAsset = (asset: Pick<MediaAsset, "kind" | "path">) =>
  !isAudioAsset(asset) && (asset.kind === "image" || isImagePath(asset.path));


/**
 * Keep a pair of frame-picker ids pointing at assets that still exist.
 *
 * Why this is not just "default when empty": a `<select>` is controlled by its `value`, but React
 * drives it by setting `selected` on each matching `<option>`. When the value matches nothing,
 * every option ends up unselected, and a single-line `<select>` in Chromium then **displays its
 * first option** — so a stale id looks exactly like a valid selection while the state behind it
 * points at an asset that is gone. The generate button then fails a guard telling the user to
 * choose the image the control appears to be showing.
 *
 * An id goes stale whenever an asset leaves the bin: removed by hand, replaced by a re-import, or
 * belonging to a project that has since been closed. A selection therefore survives only if its
 * asset is still present, and is otherwise replaced by what the control was already showing.
 *
 * `last` falls back to the first asset that is not `first`, because first-and-last-frame needs two
 * *distinct* frames — `assets[1]` is only that by coincidence.
 */
export function resolveFramePickerIds<T extends { id: string }>(
  assets: T[], first: string, last: string,
): { first: string; last: string } {
  const known = new Set(assets.map((asset) => asset.id));
  const resolvedFirst = known.has(first) ? first : assets[0]?.id ?? "";
  const resolvedLast = known.has(last)
    ? last
    : assets.find((asset) => asset.id !== resolvedFirst)?.id ?? "";
  return { first: resolvedFirst, last: resolvedLast };
}

/** The same rule for a single-selection picker. */
export function resolvePickerId<T extends { id: string }>(assets: T[], selected: string): string {
  return assets.some((asset) => asset.id === selected) ? selected : assets[0]?.id ?? "";
}

/**
 * Assets whose frames can condition a generation.
 *
 * A still counts. This used to be `["video", "generated"]`, with a comment reasoning that "a
 * graphic or caption has no frame to condition on" — true of a *track* kind, but an image asset
 * sitting on V1 has exactly one frame, and it is the obvious way to end a bridge on artwork. The
 * kind filter was doing the job the audio check should have been doing.
 */
export const canConditionFrames = (asset: Pick<MediaAsset, "kind" | "path">) =>
  !isAudioAsset(asset) && ["video", "generated", "image"].includes(asset.kind);

/** One enabled V1 clip, flattened to seconds so this logic never has to know about Rationals. */
export type FrameSourceClip = { id: string; assetId: string; startSeconds: number; endSeconds: number };

type Conditionable = Pick<MediaAsset, "kind" | "path" | "name">;

/** Half a frame at 24 fps. Two clips closer than this are touching, not separated by a gap. */
export const ADJACENT_TOLERANCE = 1 / 48;

/**
 * How a bridge is placed between two clips.
 *
 * `gapSeconds` is the room already there. A bridge shorter than the gap drops into it and nothing
 * downstream moves; a longer one takes the gap and pushes the rest along by the overflow only.
 * `rippleSeconds` is that overflow, and it is what the timeline shifts by — inserting the full
 * length regardless would push the incoming clip a gap's worth too far, which is the bug that
 * makes "insert between these two" quietly wrong whenever a gap exists.
 */
export type BridgePlacement = {
  gapSeconds: number;
  bridgeSeconds: number;
  rippleSeconds: number;
  /** True when the bridge exactly fills the room, to within half a frame. */
  fillsGap: boolean;
};

export function planBridgePlacement(gapSeconds: number, bridgeSeconds: number): BridgePlacement {
  const gap = Math.max(0, gapSeconds);
  const ripple = Math.max(0, bridgeSeconds - gap);
  return {
    gapSeconds: gap,
    bridgeSeconds,
    rippleSeconds: ripple,
    fillsGap: Math.abs(bridgeSeconds - gap) <= ADJACENT_TOLERANCE,
  };
}

/**
 * The clip a "continue from here" generation extends, or why there isn't one.
 *
 * Returning the *reason* is the point. Both editor workflows used to fail into a single notice —
 * "Select the outgoing V1 clip" — whether nothing was selected, a caption was selected, the
 * selection was on the wrong track, or the asset simply wasn't eligible. A guard that names a
 * remedy instead of the problem leaves the user re-doing the thing they already did.
 */
export function resolveExtendClip<A extends Conditionable>(
  clips: FrameSourceClip[], assetFor: (assetId: string) => A | undefined,
  selectedClipId: string | undefined, onV1: boolean,
): { clip: FrameSourceClip; asset: A } | { blocker: string } {
  if (!selectedClipId) return { blocker: "Select the clip on V1 to continue from." };
  if (!onV1) return { blocker: "The selected clip is not on V1. Pick the clip on the Primary track." };
  const clip = clips.find((item) => item.id === selectedClipId);
  if (!clip) return { blocker: "The selected clip is disabled, so it has no visible frame to continue from." };
  const asset = assetFor(clip.assetId);
  if (!asset) return { blocker: "The selected clip's media is missing from the project." };
  if (!canConditionFrames(asset)) {
    return { blocker: `${asset.name} is ${asset.kind === "audio" ? "audio" : "not video or still media"}, so it has no frame to continue from.` };
  }
  return { clip, asset };
}

/**
 * The (outgoing, incoming) pair a bridge is generated between, or why there isn't one.
 *
 * A gap between the two is reported rather than refused: it is the natural length for the bridge,
 * and the caller decides whether to fill it or to set a length and push the incoming clip along.
 */
export function resolveBridgePair<A extends Conditionable>(
  clips: FrameSourceClip[], assetFor: (assetId: string) => A | undefined,
  selectedClipId: string | undefined, onV1: boolean,
): { from: FrameSourceClip; fromAsset: A; to: FrameSourceClip; toAsset: A; gapSeconds: number }
  | { blocker: string } {
  const outgoing = resolveExtendClip(clips, assetFor, selectedClipId, onV1);
  if ("blocker" in outgoing) {
    return { blocker: outgoing.blocker.replace("to continue from.", "to bridge from.") };
  }
  const index = clips.findIndex((item) => item.id === outgoing.clip.id);
  const to = clips[index + 1];
  if (!to) return { blocker: "There is no enabled V1 clip after this one to bridge to." };
  const toAsset = assetFor(to.assetId);
  if (!toAsset) return { blocker: "The following clip's media is missing from the project." };
  if (!canConditionFrames(toAsset)) {
    return { blocker: `The following clip, ${toAsset.name}, has no frame to bridge to.` };
  }
  // A gap is not a fault. It is either the length the bridge should be, or room the bridge
  // partly fills — `planBridgePlacement` decides how far the rest of the timeline moves.
  const gapSeconds = Math.max(0, to.startSeconds - outgoing.clip.endSeconds);
  return { from: outgoing.clip, fromAsset: outgoing.asset, to, toAsset, gapSeconds };
}
