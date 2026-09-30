/**
 * VFX Phase 5: Wan Animate 2 character insertion (src-tauri/src/ai/wan_animate.rs). One call animates,
 * mattes (SAM 2 + BiRefNet) and registers the character as an element; progress arrives as
 * `character-progress` events while it runs.
 */
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import type { ElementEntry } from "./elementService";

export type AnimateStatus = { ready: boolean; baked: boolean; licence: string; detail: string };

export type CharacterRequest = {
  name: string;
  characterPath: string;
  prompt: string;
  drivingVideo: string;
  firstFrame: number;
  frameCount: number;
  plateWidth: number;
  plateHeight: number;
  fps: number;
  steps?: number;
  seed?: number;
  area?: number;
  /** SAM 2 include points on the first driving frame, normalised to the driving clip. */
  actorPoints?: Array<{ x: number; y: number; include: boolean }>;
  /** The actor's (centroid x, feet line, standing height), normalised to the driving clip: the worker
   * frames the character picture to match (Animate 2 keeps the reference's framing). */
  actorFrame?: [number, number, number];
  /** The actor's union box over the range (x0, y0, x1, y1), normalised: the worker drives with that
   * neighbourhood of the clip instead of the whole frame, so the character gets the pixels. */
  actorRegion?: [number, number, number, number];
};

/** One worker event: `plan`, `stage-start`, `animate-start`, `step`, `stage`, `done`. */
export type CharacterProgress = {
  event: string;
  stage?: string;
  step?: number;
  total?: number;
  seconds?: number;
  segments?: number;
  frames?: number;
  width?: number;
  height?: number;
  ok?: boolean;
  error?: string;
};

/** VFX Phase 5: a clean plate patch for a locked-off shot (src-tauri/src/ai/clean_plate.rs). */
export type CleanPlateRequest = {
  name: string;
  video: string;
  firstFrame: number;
  frameCount: number;
  /** Frames in the whole source clip: the background is looked for either side of the range. */
  totalFrames: number;
  fps: number;
  plateWidth: number;
  plateHeight: number;
  /** The actor's SAM 2 track folder (`masks/00000.png ...`) and the source frame its first mask is. */
  matteCache: string;
  matteFirst: number;
};

export const buildCleanPlate = (request: CleanPlateRequest) => invoke<ElementEntry>("clean_plate_build", { request });

export const wanAnimateStatus = () => invoke<AnimateStatus>("wan_animate_status");
export const generateCharacter = (request: CharacterRequest) => invoke<ElementEntry>("character_generate", { request });
export const onCharacterProgress = (handler: (progress: CharacterProgress) => void): Promise<UnlistenFn> =>
  listen<CharacterProgress>("character-progress", (event) => handler(event.payload));

/** The status line for one progress event (undefined: keep the current line). */
export function describeCharacterProgress(progress: CharacterProgress, name: string): string | undefined {
  switch (progress.event) {
    case "plan": return `${name}: ${progress.frames} frames at ${progress.width} × ${progress.height}`;
    case "stage-start": return {
      bake: `${name}: baking the Animate 2 transformer to FP8 (first run only, several minutes)…`,
      encode: `${name}: encoding the prompt (UMT5)…`,
      animate: `${name}: loading Animate 2…`,
      sam2: `${name}: SAM 2 is tracking the character…`,
      matte: `${name}: cutting the character out (SAM 2 + BiRefNet)…`,
      proxy: `${name}: building the preview proxy…`,
    }[progress.stage ?? ""] ?? `${name}: ${progress.stage}…`;
    case "animate-start": return `${name}: animating ${progress.frames} frames in ${progress.segments} segment${progress.segments === 1 ? "" : "s"}…`;
    case "step": {
      const done = progress.step ?? 0, total = progress.total ?? 0;
      const left = done > 0 && progress.seconds ? Math.round(progress.seconds / done * (total - done)) : undefined;
      return `${name}: step ${done} of ${total}${left !== undefined ? ` · about ${formatSeconds(left)} left` : ""}`;
    }
    case "stage": return progress.ok === false ? `${name}: ${progress.stage} failed — ${progress.error ?? "see worker.log"}` : undefined;
    default: return undefined;
  }
}

const formatSeconds = (seconds: number) => seconds >= 90 ? `${Math.round(seconds / 60)} min` : `${seconds} s`;
