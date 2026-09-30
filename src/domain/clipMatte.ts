/**
 * Background removal for timeline clips: a matte sequence applied as the clip's alpha.
 *
 * The matte is made once per clip range (`clip_matte` in Rust, `clip_matte.py`) and stored on the
 * clip. The programme turns it into a `sequence` mask entry on the clip's layer, so the monitor and
 * the export cut the clip out the same way, through the same compositors: whatever is under the
 * clip (a V1 background under a V2 cutout, the other clip of a transition) shows through.
 *
 * Methods:
 *   - "birefnet": automatic foreground per frame, soft edges (hair, motion blur);
 *   - "sam2": the subject picked with points, followed through the clip, hard edges;
 *   - "sam2-birefnet": SAM 2 picks the subject, BiRefNet gives it soft edges.
 *
 * Points are stored in the clip's SOURCE (normalised picture coordinates and absolute source
 * frames), so moving, zooming or conforming the clip never invalidates them, and a pick made on
 * any frame of the clip is valid. "Track assist" follows the picked points with CoTracker3 and
 * gives SAM 2 the followed points again every few frames, which keeps it on a subject that turns,
 * is briefly covered, or leaves and re-enters the frame.
 */

import type { TrackSample } from "./photoDocument";
import { seconds } from "./time";
import type { MediaAsset, TimelineClip } from "./types";

export type ClipMatteMethod = "birefnet" | "sam2" | "sam2-birefnet";

/** A picked point: absolute source frame, position normalised to the source picture. */
export type ClipMattePoint = { frame: number; x: number; y: number; include: boolean };

/** The matte as made: a `mask_cache.py` folder covering source frames first..first+frameCount-1. */
export type ClipMatteCache = {
  cacheId: string;
  cachePath: string;
  frameCount: number;
  first: number;
  width: number;
  height: number;
  fps: number;
  method: ClipMatteMethod;
  /** The source range it was made for (seconds), to tell when a trim has left it short. */
  sourceIn: number;
  sourceOut: number;
  createdAt: string;
  /** `matteRecipe` of the settings it was made with, to tell when they have changed since. */
  recipe?: string;
};

export type ClipMatte = {
  enabled: boolean;
  method: ClipMatteMethod;
  trackAssist: boolean;
  points: ClipMattePoint[];
  /** BiRefNet's temporal smoothing, 0..0.6 (steadier edges, a little lag on fast motion). */
  smooth: number;
  /** Edge softness and grow/shrink, in sequence pixels (they scale with the render). */
  feather: number;
  expand: number;
  /** Keep the background and remove the subject instead. */
  invert: boolean;
  cache?: ClipMatteCache;
};

export const CLIP_MATTE_METHODS: { value: ClipMatteMethod; label: string; hint: string }[] = [
  { value: "birefnet", label: "Auto (BiRefNet)", hint: "Finds the foreground in every frame by itself. Soft, detailed edges; no points needed." },
  { value: "sam2", label: "Pick subject (SAM 2)", hint: "Click the subject in the monitor; SAM 2 follows it through the clip. Chooses exactly which subject, hard edges." },
  { value: "sam2-birefnet", label: "Pick + soft edges (both)", hint: "SAM 2 picks the subject, BiRefNet gives it soft edges (hair, motion blur). The best cutout; slowest." },
];

export const defaultClipMatte = (): ClipMatte => ({
  enabled: true, method: "birefnet", trackAssist: false, points: [], smooth: 0.2, feather: 1, expand: 0, invert: false,
});

/** SAM 2 keeps state for every frame in memory; the worker takes at most this many. */
export const MAX_SAM2_FRAMES = 900;
export const MAX_BIREFNET_FRAMES = 5400;
/** How often track assist gives SAM 2 the followed points again, in frames. */
export const ASSIST_EVERY = 12;
/** The Rust side's limit on points per request. */
export const MAX_PROMPTS = 600;

export const usesSam2 = (method: ClipMatteMethod) => method.startsWith("sam2");

const fpsOf = (asset: MediaAsset) => asset.frameRate ? asset.frameRate.numerator / Math.max(1, asset.frameRate.denominator) : 24;
const frameIndex = (secondsValue: number, fps: number) => Math.floor(Math.max(0, secondsValue) * fps + 0.5 - 1e-9);

/** The source frames a clip's matte covers, and the seconds the worker reads them from. */
export type MatteRange = { still: boolean; fps: number; first: number; frameCount: number; startTime: number; endTime: number };

export function matteRange(clip: Pick<TimelineClip, "sourceIn" | "sourceOut">, asset: MediaAsset): MatteRange {
  const fps = fpsOf(asset);
  if (asset.kind === "image") return { still: true, fps, first: 0, frameCount: 1, startTime: 0, endTime: 0 };
  const total = Math.max(1, Math.round(seconds(asset.duration) * fps));
  const first = Math.min(total - 1, frameIndex(seconds(clip.sourceIn), fps));
  const count = Math.max(1, Math.min(total - first, Math.round((seconds(clip.sourceOut) - seconds(clip.sourceIn)) * fps)));
  // Half a frame early: ffmpeg keeps frames at or after the seek point, so the first one extracted
  // is exactly `first` however the in-point falls between frames.
  const startTime = Math.max(0, (first - 0.5) / fps);
  return { still: false, fps, first, frameCount: count, startTime, endTime: startTime + count / fps };
}

/** Why the matte cannot be made as set up, or undefined. */
export function matteBlocker(matte: ClipMatte, range: MatteRange): string | undefined {
  if (usesSam2(matte.method)) {
    if (!matte.points.some((point) => point.include)) return "Pick the subject first: click it in the monitor";
    if (range.frameCount > MAX_SAM2_FRAMES) return `SAM 2 follows at most ${MAX_SAM2_FRAMES} frames; this clip has ${range.frameCount}. Split it, or use Auto`;
  } else if (range.frameCount > MAX_BIREFNET_FRAMES) {
    return `At most ${MAX_BIREFNET_FRAMES} frames at a time; this clip has ${range.frameCount}. Split it`;
  }
  return undefined;
}

/** The points SAM 2 is given, as the Rust request wants them: frames relative to the range. */
export function promptsFor(points: ClipMattePoint[], range: MatteRange): { frame: number; x: number; y: number; include: boolean }[] {
  return points
    .map((point) => ({ frame: range.still ? 0 : point.frame - range.first, x: clamp01(point.x), y: clamp01(point.y), include: point.include }))
    .filter((point) => point.frame >= 0 && point.frame < range.frameCount);
}

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

/**
 * Track assist: the picked points followed by CoTracker3 (`samples[k]` is source frame
 * `first + k`, in source pixels), given again every `every` frames where the point is visible.
 * The picked points themselves are kept as they are; exclude points are followed too (whatever
 * they excluded moves with the picture). Capped at `limit` points in all, thinning evenly.
 */
export function assistPoints(
  picked: ClipMattePoint[],
  tracks: { point: ClipMattePoint; first: number; samples: TrackSample[] }[],
  size: { width: number; height: number },
  range: MatteRange,
  every = ASSIST_EVERY,
  limit = MAX_PROMPTS,
): ClipMattePoint[] {
  const pickedFrames = new Set(picked.map((point) => point.frame));
  const last = range.first + range.frameCount - 1;
  const byFrame = new Map<number, ClipMattePoint[]>();
  for (const track of tracks) {
    for (let frame = range.first; frame <= last; frame += Math.max(1, every)) {
      if (pickedFrames.has(frame)) continue;
      const sample = track.samples[frame - track.first];
      if (!sample || !sample[2]) continue;
      const x = sample[0] / Math.max(1, size.width), y = sample[1] / Math.max(1, size.height);
      if (x < 0 || y < 0 || x > 1 || y > 1) continue;
      const list = byFrame.get(frame) ?? [];
      list.push({ frame, x, y, include: track.point.include });
      byFrame.set(frame, list);
    }
  }
  // A frame's points are only useful together when one of them is an include point.
  const frames = [...byFrame.entries()].filter(([, list]) => list.some((point) => point.include)).sort((a, b) => a[0] - b[0]);
  const room = Math.max(0, limit - picked.length);
  // Too many: keep every k-th re-prompted frame, the smallest k that fits.
  let kept = frames;
  for (let k = 2; kept.reduce((sum, [, list]) => sum + list.length, 0) > room; k++) {
    kept = frames.filter((_, index) => index % k === 0);
    if (k > frames.length) { kept = []; break; }
  }
  return [...picked, ...kept.flatMap(([, list]) => list)];
}

/** The matte the programme draws: only an enabled one that has been made. */
export type ProgrammeMatte = { cachePath: string; frameCount: number; first: number; feather: number; expand: number; invert: boolean };

export function programmeMatte(matte: ClipMatte | undefined): ProgrammeMatte | undefined {
  if (!matte?.enabled || !matte.cache) return undefined;
  const { cachePath, frameCount, first } = matte.cache;
  return { cachePath, frameCount, first, feather: Math.max(0, matte.feather), expand: matte.expand, invert: matte.invert };
}

/** What decides the matte's pixels (feather, expand and invert apply live and are not part of it). */
export function matteRecipe(matte: ClipMatte): string {
  const sam2 = usesSam2(matte.method);
  return JSON.stringify({
    method: matte.method,
    points: sam2 ? matte.points.map((point) => [point.frame, +point.x.toFixed(4), +point.y.toFixed(4), point.include ? 1 : 0]) : [],
    assist: sam2 && matte.trackAssist,
    smooth: matte.method === "sam2" ? 0 : +matte.smooth.toFixed(2),
  });
}

/** Whether the settings were changed after the matte was made (it needs remaking to show them). */
export const matteIsStale = (matte: ClipMatte | undefined): boolean =>
  Boolean(matte?.cache?.recipe && matte.cache.recipe !== matteRecipe(matte));

/** Whether the made matte no longer covers the clip's range (it was trimmed out past it). */
export function matteIsShort(matte: ClipMatte | undefined, clip: Pick<TimelineClip, "sourceIn" | "sourceOut">): boolean {
  const cache = matte?.cache;
  if (!cache || cache.frameCount <= 1 && cache.first === 0 && cache.sourceOut <= cache.sourceIn) return false;
  return seconds(clip.sourceIn) < cache.sourceIn - 1e-3 || seconds(clip.sourceOut) > cache.sourceOut + 1e-3;
}
