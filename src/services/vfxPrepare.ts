/**
 * VFX Phase 7: what a mask VFX layer needs before the compositor can render it live, computed once
 * and kept on the layer: the matte with the recipe's smoothing and cleanup baked in, the audio
 * envelope for an audio-reactive recipe, and the imported effect clip's frame count and size.
 *
 * Each piece is only (re)computed when missing or made for other settings, so preparing a project
 * whose layers are all ready costs nothing. Layers without a matte cache are left alone: they can
 * only be shown as the file they were rendered to.
 */

import { invoke } from "@tauri-apps/api/core";
import type { ClipVfxLayer } from "../domain/maskVfx";
import type { Project } from "../domain/types";
import { seconds } from "../domain/time";
import { parseFrameRate, probeMedia } from "./modelService";

export function stabilizeMaskMatte(cachePath: string, temporalSmooth: number, cleanup: number) {
  return invoke<{ cachePath: string; temporalSmooth: number; cleanup: number; reused: boolean }>(
    "stabilize_mask_matte", { request: { cachePath, temporalSmooth, cleanup } });
}

export function maskAudioEnvelope(path: string, startSeconds: number, endSeconds: number, frameCount: number, mode: string, smoothing: number) {
  return invoke<number[]>("mask_audio_envelope", { request: { path, startSeconds, endSeconds, frameCount, mode, smoothing } });
}

/** What is still missing for this layer to render live (empty when it is ready). */
export function missingForLive(layer: ClipVfxLayer): Array<"matte" | "envelope" | "clip"> {
  const recipe = layer.recipe;
  if (!layer.matteCache) return [];
  const missing: Array<"matte" | "envelope" | "clip"> = [];
  const wantsBake = recipe.temporalSmooth > 0 || recipe.cleanup > 0;
  if (wantsBake && !(layer.stableMatte?.temporalSmooth === recipe.temporalSmooth && layer.stableMatte.cleanup === recipe.cleanup)) missing.push("matte");
  if (recipe.audioReactive !== "off" && !(layer.envelope?.mode === recipe.audioReactive && layer.envelope.smoothing === recipe.reactiveSmoothing)) missing.push("envelope");
  if (recipe.matteStyle === "clip" && recipe.effectClipPath && !layer.effectClipMedia) missing.push("clip");
  return missing;
}

/** The layer with whatever it was missing filled in. `sourcePath`/`sourceIn` are its clip's media. */
export async function prepareVfxLayer(layer: ClipVfxLayer, sourcePath: string, sourceIn: number): Promise<ClipVfxLayer> {
  const missing = missingForLive(layer);
  if (!missing.length || !layer.matteCache) return layer;
  const recipe = layer.recipe;
  const next: ClipVfxLayer = { ...layer };
  if (missing.includes("matte")) {
    const baked = await stabilizeMaskMatte(layer.matteCache.cachePath, recipe.temporalSmooth, recipe.cleanup);
    next.stableMatte = { cachePath: baked.cachePath, temporalSmooth: baked.temporalSmooth, cleanup: baked.cleanup };
  }
  if (missing.includes("envelope")) {
    const values = await maskAudioEnvelope(sourcePath, sourceIn + layer.rangeStart, sourceIn + layer.rangeEnd,
      Math.max(1, layer.matteCache.frameCount), recipe.audioReactive, recipe.reactiveSmoothing);
    next.envelope = { mode: recipe.audioReactive, smoothing: recipe.reactiveSmoothing, values };
  }
  if (missing.includes("clip")) {
    const probe = await probeMedia(recipe.effectClipPath);
    const rate = parseFrameRate(probe.frameRate);
    const fps = rate.numerator / Math.max(1, rate.denominator);
    next.effectClipMedia = {
      frameCount: Math.max(1, Math.round(probe.durationSeconds * fps)), fps,
      width: probe.width ?? 1920, height: probe.height ?? 1080,
    };
  }
  return next;
}

/**
 * Prepare every enabled VFX layer in the active sequence; `save` receives each changed layer.
 * Failures are collected, not thrown: a layer that cannot be prepared still renders (raw matte, no
 * reactivity) and the caller reports what went wrong.
 */
export async function prepareProjectVfx(
  project: Project, save: (clipId: string, layer: ClipVfxLayer) => void, progress?: (text: string) => void,
): Promise<string[]> {
  const problems: string[] = [];
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId);
  const assets = new Map(project.assets.map((asset) => [asset.id, asset]));
  for (const track of sequence?.tracks ?? []) {
    for (const clip of track.clips) {
      const asset = assets.get(clip.assetId);
      for (const layer of clip.vfxLayers ?? []) {
        if (!layer.enabled || !asset || !missingForLive(layer).length) continue;
        progress?.(`Preparing VFX "${layer.name}"`);
        try {
          save(clip.id, await prepareVfxLayer(layer, asset.path, seconds(clip.sourceIn)));
        } catch (error) {
          problems.push(`${layer.name}: ${error instanceof Error ? error.message : String(error)}`);
        }
      }
    }
  }
  return problems;
}
