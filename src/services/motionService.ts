/**
 * The motion runtime (VFX Phase 4): CoTracker3 point tracks and Video Depth Anything depth
 * sequences. Rust owns the workers and their caches (src-tauri/src/ai/motion.rs).
 */

import { invoke } from "@tauri-apps/api/core";
import type { TrackSample } from "../domain/photoDocument";

export type MotionStatus = {
  ready: boolean;
  cotracker: boolean;
  depth: boolean;
  cotrackerLicence: string;
  depthLicence: string;
  detail: string;
};

export type TrackPointsResult = {
  ok: boolean;
  source: "cotracker3";
  licence: string;
  width: number;
  height: number;
  fps: number;
  /** Source frame of samples[0]. */
  first: number;
  frames: number;
  points: Array<{ query: [number, number]; samples: TrackSample[] }>;
  seconds?: number;
  reused: boolean;
  cachePath: string;
};

export type VideoDepthResult = {
  ok: boolean;
  kind: "depth";
  frames: number;
  width: number;
  height: number;
  cachePath: string;
  seconds?: number;
  reused: boolean;
};

export const motionStatus = () => invoke<MotionStatus>("motion_runtime_status");

/** Points in continuous source pixels, picked on `queryFrame`. */
export const trackPoints = (request: { videoPath: string; queryFrame: number; points: Array<[number, number]>; startFrame?: number; frameCount?: number }) =>
  invoke<TrackPointsResult>("track_points", { request });

export const computeVideoDepth = (videoPath: string) => invoke<VideoDepthResult>("video_depth", { request: { videoPath } });
