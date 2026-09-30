/**
 * VFX Phase 7: the timeline export on the compositor.
 *
 * Every frame is the programme document rendered by the engine the monitor uses, packed to I420 on
 * the page and streamed to one encode with the delivery settings (`programme_export.rs`), which mixes
 * the programme's sound first and publishes the file (after the upscale chain, when asked) at the
 * end. Previews (the timeline's quick/full preview) are the same path over a window of the programme.
 *
 * Nothing is left out silently: a picture that cannot be read stops the export and names it (the
 * legacy export can still render it), and nodes that were skipped are reported as notes.
 */

import { invoke } from "@tauri-apps/api/core";
import { buildProgramme, programmeFrames, type Programme, type ProgrammeAudio } from "../../domain/programme";
import type { ClipVfxLayer } from "../../domain/maskVfx";
import type { Project } from "../../domain/types";
import { ProgrammeEngine } from "../../render/programmeEngine";
import { readCompositorFlag } from "../../render/compositorFlag";
import { packYuv420 } from "../../render/yuv420";
import { FrameSender, type RenderStarted } from "../../services/renderService";
import { prepareProjectVfx } from "../../services/vfxPrepare";
import { readLegacyVfx } from "../monitor/ProgrammeMonitor";

const LEGACY_EXPORT_KEY = "aiv-legacy-export";

/** Whether the old filter-graph export is chosen on this machine. Off by default. */
export function readLegacyExport(): boolean {
  try { return window.localStorage.getItem(LEGACY_EXPORT_KEY) === "1"; } catch { return false; }
}
export function writeLegacyExport(on: boolean): void {
  try { window.localStorage.setItem(LEGACY_EXPORT_KEY, on ? "1" : "0"); } catch { /* not remembered */ }
}

export type ProgrammeExportSettings = {
  outputPath: string;
  width: number;
  height: number;
  frameRate: { numerator: number; denominator: number };
  videoCodec: string;
  audioCodec: string;
  videoBitrate: string;
  audioBitrate: string;
  qualityMode: string;
  quality: number;
  preset: string;
  audioSampleRate: number;
  audioChannels: number;
  upscale?: unknown;
};

/** The programme's sound over [from, from + length): entries shifted, trimmed and clipped to it. */
export function audioWindow(audio: ProgrammeAudio[], from: number, length: number): ProgrammeAudio[] {
  return audio.flatMap((entry) => {
    const start = Math.max(entry.start, from), end = Math.min(entry.start + entry.duration, from + length);
    if (end <= start) return [];
    const cut = start - entry.start;
    return [{
      ...entry, start: start - from, duration: end - start, sourceIn: entry.sourceIn + cut * entry.rate,
      // A fade the window starts inside is dropped rather than restarted.
      fadeIn: cut > 0 ? 0 : entry.fadeIn,
      fadeOut: entry.start + entry.duration > from + length ? 0 : entry.fadeOut,
    }];
  });
}

export type ProgrammeExportProgress = (text: string) => void;

/**
 * Render the active sequence (or a window of it) to `settings.outputPath`. `saveLayer` stores a VFX
 * layer that had to be prepared for live rendering (a baked matte, an envelope).
 */
export async function exportProgramme(
  project: Project, settings: ProgrammeExportSettings, progress: ProgrammeExportProgress,
  saveLayer: (clipId: string, layer: ClipVfxLayer) => void, readProject: () => Project,
  window?: { startSeconds: number; durationSeconds: number },
): Promise<{ outputPath: string; warnings: string[] }> {
  const warnings: string[] = [];
  const problems = await prepareProjectVfx(project, saveLayer, progress);
  for (const problem of problems) warnings.push(`A VFX layer could not be prepared for live rendering (${problem}); it renders with its raw matte.`);
  const fps = settings.frameRate.numerator / Math.max(1, settings.frameRate.denominator);
  // Built at the delivery size: sizes the timeline gives in pixels (a blur, a border) are delivery
  // pixels, as they were in the filter-graph export.
  const programme: Programme | undefined = buildProgramme(readProject(), { fps, width: settings.width, height: settings.height });
  if (!programme || !programme.clips.length) throw new Error("Render needs at least one ENABLED clip on the V1 track.");
  const all = programmeFrames(programme);
  const first = window ? Math.max(0, Math.min(all - 1, Math.floor(window.startSeconds * fps + 1e-6))) : 0;
  const count = window ? Math.max(1, Math.min(all - first, Math.round(window.durationSeconds * fps))) : all;
  const duration = count / fps;
  const audio = window ? audioWindow(programme.audio, first / fps, duration) : programme.audio;
  const head = !window || first === 0 ? programme.headFade?.duration ?? 0 : 0;
  const tail = !window || first + count >= all ? programme.tailFade?.duration ?? 0 : 0;

  const started = await invoke<RenderStarted>("programme_export_start", { request: {
    ...settings, duration, audio, headFade: head, tailFade: tail, pixelFormat: "yuv420", http: true,
  } });
  const sender = new FrameSender(started);
  const engine = new ProgrammeEngine({ lookahead: 24 });
  const mode = readLegacyVfx();
  const gpu = readCompositorFlag();
  const skipped = new Set<string>();
  const packed = [new Uint8Array(0), new Uint8Array(0)];
  const clock = performance.now();
  let lastShown = 0;
  try {
    const prepareFrame = async (frame: number) => {
      const document = engine.document(programme, frame, mode);
      const needs = engine.needs(document, frame);
      await engine.prepare(needs, false);
      return { document, needs };
    };
    let next = prepareFrame(first);
    let sending: Promise<void> = Promise.resolve();
    for (let at = 0; at < count; at++) {
      const frame = first + at;
      const { document, needs } = await next;
      await engine.prepare(needs, false, true);
      if (engine.failures.size) {
        const [path, reason] = [...engine.failures][0];
        throw new Error(`could not read ${path} (${reason}). The legacy export may still render it.`);
      }
      if (at + 1 < count) next = prepareFrame(frame + 1);
      const result = engine.compose(document, frame, settings.width, settings.height, false, gpu);
      for (const item of result.report.skipped) skipped.add(item.reason);
      await sending;
      let bytes: Uint8Array = new Uint8Array(result.pixels.buffer, result.pixels.byteOffset, result.pixels.byteLength);
      if (sender.format === "yuv420") {
        bytes = packYuv420(result.pixels, settings.width, settings.height, packed[at & 1]);
        packed[at & 1] = bytes;
      }
      sending = sender.send(bytes);
      const now = performance.now();
      if (now - lastShown > 400 || at + 1 === count) {
        lastShown = now;
        const rate = (at + 1) / Math.max(0.001, (now - clock) / 1000);
        progress(`Rendering frame ${at + 1} / ${count} · ${rate.toFixed(1)} fps · ${engine.compositor.lastEngine.toUpperCase()} · ${sender.describe()}`);
      }
    }
    await sending;
    progress("Finishing the encode…");
    const finished = await invoke<{ outputPath: string; warnings?: string[] }>("programme_export_finish", { id: started.id });
    for (const reason of skipped) warnings.push(`Some frames were drawn without a node that ${reason}.`);
    return { outputPath: finished.outputPath, warnings: [...warnings, ...(finished.warnings ?? [])] };
  } catch (error) {
    await invoke("programme_export_cancel", { id: started.id }).catch(() => undefined);
    throw error;
  } finally {
    engine.dispose();
  }
}
