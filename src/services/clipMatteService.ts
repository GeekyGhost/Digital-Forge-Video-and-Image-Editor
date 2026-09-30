/**
 * Background removal for timeline clips (Rust `clip_matte`, `ai-runtime/aiv_runtime/clip_matte.py`).
 * The domain side -- ranges, points, track assist, how the programme draws it -- is
 * `domain/clipMatte.ts`.
 */

import { invoke } from "@tauri-apps/api/core";
import {
  assistPoints, matteRange, matteRecipe, promptsFor, usesSam2,
  type ClipMatte, type ClipMatteCache, type ClipMattePoint,
} from "../domain/clipMatte";
import { seconds } from "../domain/time";
import type { MediaAsset, TimelineClip } from "../domain/types";
import type { MaskTrackResult } from "./modelService";
import { trackPoints } from "./motionService";

export type ClipMatteRequest = {
  videoPath: string;
  startTime: number;
  endTime: number;
  method: ClipMatte["method"];
  prompts: { frame: number; x: number; y: number; include: boolean }[];
  smooth: number;
};

export const runClipMatte = (request: ClipMatteRequest) => invoke<MaskTrackResult>("clip_matte", { request });

/**
 * Track assist: each picked point followed through the clip's range by CoTracker3 (points picked on
 * the same frame are tracked together), then the SAM 2 points with the followed ones added.
 */
export async function assistedPoints(
  clip: Pick<TimelineClip, "sourceIn" | "sourceOut">, asset: MediaAsset, points: ClipMattePoint[],
  report: (text: string) => void = () => undefined,
): Promise<ClipMattePoint[]> {
  const range = matteRange(clip, asset);
  if (range.still || range.frameCount < 2) return points;
  const width = asset.width ?? 1920, height = asset.height ?? 1080;
  const byFrame = new Map<number, ClipMattePoint[]>();
  for (const point of points) byFrame.set(point.frame, [...(byFrame.get(point.frame) ?? []), point]);
  const tracks: Parameters<typeof assistPoints>[1] = [];
  let group = 0;
  for (const [frame, picked] of byFrame) {
    group += 1;
    report(`Track assist: CoTracker3 is following ${picked.length} point${picked.length === 1 ? "" : "s"} (${group} of ${byFrame.size})…`);
    const result = await trackPoints({
      videoPath: asset.path,
      queryFrame: Math.min(range.first + range.frameCount - 1, Math.max(range.first, frame)),
      points: picked.map((point) => [point.x * width, point.y * height] as [number, number]),
      startFrame: range.first, frameCount: range.frameCount,
    });
    // CoTracker3 measures in the file's own pixels, which is what the probe reported.
    const size = { width: result.width || width, height: result.height || height };
    result.points.forEach((track, index) => {
      if (!picked[index]) return;
      tracks.push({ point: picked[index], first: result.first, samples: track.samples.map(([x, y, seen]) => [x / size.width * width, y / size.height * height, seen]) });
    });
  }
  return assistPoints(points, tracks, { width, height }, range);
}

/** Make (or reuse) the clip's matte; the cache to store on the clip. */
export async function makeClipMatte(
  clip: Pick<TimelineClip, "sourceIn" | "sourceOut">, asset: MediaAsset, matte: ClipMatte,
  report: (text: string) => void = () => undefined,
): Promise<ClipMatteCache & { reused: boolean; prompts: number }> {
  const range = matteRange(clip, asset);
  let points = usesSam2(matte.method) ? matte.points : [];
  if (usesSam2(matte.method) && matte.trackAssist) points = await assistedPoints(clip, asset, points, report);
  const prompts = promptsFor(points, range);
  report(matte.method === "birefnet" ? "BiRefNet is removing the background…"
    : matte.method === "sam2" ? "SAM 2 is following the subject…" : "SAM 2 is following the subject, then BiRefNet refines the edges…");
  const result = await runClipMatte({
    videoPath: asset.path, startTime: range.startTime, endTime: range.still ? range.startTime : range.endTime,
    method: matte.method, prompts, smooth: Math.max(0, Math.min(0.6, matte.smooth)),
  });
  return {
    cacheId: result.cacheId, cachePath: result.cachePath, frameCount: result.frameCount, first: range.first,
    width: result.width, height: result.height, fps: result.fps || range.fps, method: matte.method,
    sourceIn: range.still ? 0 : seconds(clip.sourceIn), sourceOut: range.still ? 0 : seconds(clip.sourceOut),
    createdAt: result.createdAt, recipe: matteRecipe(matte), reused: result.reused, prompts: prompts.length,
  };
}
