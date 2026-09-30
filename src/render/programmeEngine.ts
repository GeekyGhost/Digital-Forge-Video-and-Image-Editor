/**
 * VFX Phase 7: frames of a programme (a timeline, see domain/programme.ts) through the compositor --
 * the one engine behind the timeline monitor and the timeline export.
 *
 * It owns what a frame reads (pictures and mattes by path, media frames through the WebCodecs frame
 * store) and the compositor (GPU with the per-frame CPU fallback). `prepare` loads everything one
 * frame needs; `compose` is synchronous and draws only from what is loaded, so what the monitor shows
 * and what the export writes are the same function of the same inputs.
 */

import { buildProgramme, programmeDocument, programmeFrames, type Programme, type ProgrammeHooks } from "../domain/programme";
import { maskVfxHook, type VfxMode } from "../domain/programmeVfx";
import { filesWantedAt } from "../domain/documentFiles";
import type { LayerMedia, PhotoDocument } from "../domain/photoDocument";
import type { ComposeResult, LayerSource } from "../domain/photoCompose";
import { buildScopeResources, type LoadedPicture } from "../domain/photoResources";
import { compositionRate, mediaFrameKey, mediaNeedsAt } from "../domain/photoTime";
import type { Project } from "../domain/types";
import { FlaggedCompositor } from "./compositorFlag";
import { MediaFrameStore } from "./video/frameStore";
import { loadReadableImage } from "../services/readableImage";
import { readMediaBytes } from "../services/mediaBytes";

const canvasOf = (width: number, height: number) => {
  const canvas = window.document.createElement("canvas");
  canvas.width = Math.max(1, width);
  canvas.height = Math.max(1, height);
  return canvas;
};

async function loadPicture(path: string): Promise<LoadedPicture> {
  const image = await loadReadableImage(path);
  const width = image.naturalWidth || image.width, height = image.naturalHeight || image.height;
  const canvas = canvasOf(width, height);
  canvas.getContext("2d", { willReadFrequently: true })!.drawImage(image, 0, 0);
  return { canvas, width, height };
}

export type ProgrammeNeeds = { files: string[]; media: Array<{ media: LayerMedia; index: number }> };

export class ProgrammeEngine {
  readonly files = new Map<string, LoadedPicture>();
  private readonly pendingFiles = new Map<string, Promise<void>>();
  readonly frames: MediaFrameStore;
  readonly compositor = new FlaggedCompositor();
  private readonly maskCache = new Map<string, Uint8Array>();
  private cachedFrame = -1;
  private lastDocument?: { programme: Programme; frame: number; mode: VfxMode; document: PhotoDocument };
  /** Files that failed to load, and why (reported once each). */
  readonly failures = new Map<string, string>();

  constructor(options: { budgetBytes?: number; lookahead?: number } = {}) {
    this.frames = new MediaFrameStore({ readBytes: readMediaBytes, loadImage: loadPicture, ...options });
    this.frames.onEvict = (picture) => this.compositor.releaseSource(picture.canvas);
  }

  /** The compositor document for a frame (the last one is kept: the monitor asks repeatedly). */
  document(programme: Programme, frame: number, mode: VfxMode = "live"): PhotoDocument {
    const held = this.lastDocument;
    if (held && held.programme === programme && held.frame === frame && held.mode === mode) return held.document;
    const hooks: ProgrammeHooks = { vfx: maskVfxHook(mode) };
    const document = programmeDocument(programme, frame, hooks);
    this.lastDocument = { programme, frame, mode, document };
    return document;
  }

  needs(document: PhotoDocument, frame: number): ProgrammeNeeds {
    return {
      files: [...filesWantedAt(document, frame)],
      media: mediaNeedsAt(document, frame, compositionRate(document)),
    };
  }

  /** Whether everything the frame reads is loaded. */
  ready(needs: ProgrammeNeeds, useProxy: boolean): boolean {
    return needs.files.every((path) => this.files.has(path) || this.failures.has(path))
      && needs.media.every((need) => this.frames.get(mediaFrameKey(need.media, need.index, useProxy)) || this.failures.has(need.media.path));
  }

  /** Load what the frame reads. Failures are recorded (and that picture left out), not thrown. */
  async prepare(needs: ProgrammeNeeds, useProxy: boolean, pin = false): Promise<void> {
    if (pin) this.frames.pin(needs.media.map((need) => mediaFrameKey(need.media, need.index, useProxy)));
    await Promise.all([
      ...needs.files.filter((path) => !this.files.has(path) && !this.failures.has(path)).map((path) => {
        let job = this.pendingFiles.get(path);
        if (!job) {
          job = loadPicture(path).then((picture) => { this.files.set(path, picture); }, (error) => {
            this.failures.set(path, String((error as Error)?.message ?? error));
          }).finally(() => this.pendingFiles.delete(path));
          this.pendingFiles.set(path, job);
        }
        return job;
      }),
      ...needs.media.filter((need) => !this.failures.has(need.media.path)).map((need) =>
        this.frames.load(need.media, need.index, useProxy).then(() => undefined, (error) => {
          this.failures.set(need.media.path, String((error as Error)?.message ?? error));
        })),
    ]);
    // Keep the picture cache bounded: mattes are one file per frame.
    if (this.files.size > 96) {
      const keep = new Set(needs.files);
      for (const path of [...this.files.keys()]) if (!keep.has(path) && this.files.size > 64) this.files.delete(path);
    }
  }

  sources(): Map<string, LayerSource> {
    const map = new Map<string, LayerSource>();
    for (const [path, held] of this.files) map.set(path, { image: held.canvas, width: held.width, height: held.height });
    for (const [key, picture] of this.frames.entries()) map.set(key, { image: picture.canvas, width: picture.width, height: picture.height });
    return map;
  }

  /** One frame at `width` x `height`, from what is loaded. */
  compose(document: PhotoDocument, frame: number, width: number, height: number, useProxy: boolean, gpu = true): ComposeResult {
    // Sequence mattes change every frame; rasters kept for another frame are never read again.
    if (frame !== this.cachedFrame) { this.maskCache.clear(); this.cachedFrame = frame; }
    const files = this.files;
    return this.compositor.compose({
      document, width, height, sources: this.sources(), frame, useProxy,
      resourcesFor: (scope, pixels) => buildScopeResources({ document, files: (path) => files.get(path), cache: this.maskCache }, scope, width, height, pixels),
    }, gpu);
  }

  dispose(): void {
    this.frames.dispose();
    this.compositor.dispose();
    this.files.clear();
  }
}

export { buildProgramme, programmeFrames };
export type { Programme, Project };
