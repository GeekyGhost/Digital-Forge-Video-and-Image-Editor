/**
 * The Phase 1 switch between Photo Edit's CPU compositor and the GPU renderer.
 *
 * On by default since 2026-09-27, when the parity page (the spike window's "Phase 1 parity" button)
 * passed on an RTX 4090 Laptop: no failures, and 3.6 ms against 58.9 ms per 1500 px preview frame.
 * The toolbar toggle still switches back to the CPU compositor; the choice is kept per machine in
 * localStorage, and a failed read (private mode, cleared storage) means the default.
 *
 * Whichever is chosen, a GPU failure (no WebGL2, a lost context, a picture larger than the GPU's
 * texture limit, out of memory) falls back to the CPU compositor for that frame, so the flag can
 * never cost a save.
 */

import { composeDocument, type ComposeInput, type ComposeResult } from "../domain/photoCompose";
import { PhotoRenderer } from "./photoRenderer";

const KEY = "aiv-gpu-compositor";
export const GPU_COMPOSITOR_DEFAULT = true;

export function readCompositorFlag(): boolean {
  try {
    const value = window.localStorage.getItem(KEY);
    return value === null ? GPU_COMPOSITOR_DEFAULT : value === "1";
  } catch {
    return GPU_COMPOSITOR_DEFAULT;
  }
}

export function writeCompositorFlag(on: boolean): void {
  try {
    window.localStorage.setItem(KEY, on ? "1" : "0");
  } catch {
    // Not remembered; the toggle still works for this session.
  }
}

export type ComposeEngine = "gpu" | "cpu";

/**
 * One compositor for a Photo Edit session: the GPU renderer when enabled, the CPU one otherwise or
 * when the GPU fails. `lastError` says why the last GPU attempt fell back.
 */
export class FlaggedCompositor {
  private renderer?: PhotoRenderer;
  lastEngine: ComposeEngine = "cpu";
  lastError?: string;
  lastMilliseconds = 0;

  compose(input: ComposeInput, gpu: boolean): ComposeResult {
    const started = performance.now();
    try {
      if (gpu) {
        try {
          this.renderer ??= new PhotoRenderer();
          const result = this.renderer.compose(input);
          this.lastEngine = "gpu";
          this.lastError = undefined;
          return result;
        } catch (error) {
          this.lastError = String((error as Error)?.message ?? error);
          console.warn("[compositor] GPU render failed, using the CPU compositor:", error);
          // A lost or broken context is not coming back; the next frame builds a fresh one.
          this.renderer?.dispose();
          this.renderer = undefined;
        }
      }
      this.lastEngine = "cpu";
      return composeDocument(input);
    } finally {
      this.lastMilliseconds = performance.now() - started;
    }
  }

  /** Forget a GPU texture made from a media frame that is no longer cached (VFX Phase 3). */
  releaseSource(image: CanvasImageSource): void {
    this.renderer?.releaseSource(image);
  }

  dispose(): void {
    this.renderer?.dispose();
    this.renderer = undefined;
  }
}
