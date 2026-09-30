/**
 * VFX Phase 7: a clip's mask VFX layers as compositor nodes, per programme frame (the programme's
 * `vfx` hook).
 *
 * A `ClipVfxLayer` covers source seconds [sourceIn + rangeStart, sourceIn + rangeEnd) of its clip --
 * the span its matte was tracked on, matte frame 0 at the start. Where the clip shows a source frame
 * inside that span, each enabled layer becomes one `maskVfx` node with that frame's matte, envelope
 * value and effect clip frame resolved in; later layers draw over earlier ones.
 *
 * A layer with no matte cache (made before mattes were cached) can only be shown as the file it was
 * rendered to: its source is swapped for that file over its range, as the old export did. The
 * "legacy" mode does that for every layer that has a rendered file, as a fallback while the live
 * renderer is being checked against the old one.
 */

import { effectsFor } from "./effects";
import type { LayerMedia, PhotoDocument, PhotoNode } from "./photoDocument";
import { sequenceFramePath } from "./photoMask";
import { programmeDocument, type Programme, type ProgrammeHooks, type ProgrammeMedia } from "./programme";
import type { ClipVfxLayer } from "./maskVfx";

export type VfxMode = "live" | "legacy";

const frameAt = (seconds: number, fps: number) => Math.floor(Math.max(0, seconds) * fps + 0.5 - 1e-9);

/** The node for one layer at source second `source`, or undefined outside its range or without a matte. */
export function maskVfxNode(clipId: string, sourceIn: number, layer: ClipVfxLayer, source: number, fallbackFps: number): PhotoNode | undefined {
  const cache = layer.matteCache;
  if (!cache) return undefined;
  const start = sourceIn + layer.rangeStart, end = sourceIn + layer.rangeEnd;
  if (source < start - 1e-9 || source >= end - 1e-9) return undefined;
  const fps = cache.fps > 0 ? cache.fps : fallbackFps;
  const count = Math.max(1, cache.frameCount || Math.round((end - start) * fps));
  const index = Math.min(count - 1, frameAt(source - start, fps));
  const recipe = layer.recipe;
  // The baked (smoothed, cleaned) matte only when it was baked with this recipe's settings.
  const stable = layer.stableMatte && layer.stableMatte.temporalSmooth === recipe.temporalSmooth && layer.stableMatte.cleanup === recipe.cleanup
    ? layer.stableMatte.cachePath : undefined;
  const mattePath = sequenceFramePath({ kind: "sequence", cachePath: stable ?? cache.cachePath, frameCount: count, first: 0 }, index);
  const envelope = recipe.audioReactive !== "off" && layer.envelope?.mode === recipe.audioReactive
    ? layer.envelope.values[Math.min(layer.envelope.values.length - 1, index)] ?? 0 : 0;
  let effectClip: { media: LayerMedia; index: number } | undefined;
  const probed = layer.effectClipMedia;
  if (recipe.matteStyle === "clip" && recipe.effectClipPath && probed && probed.frameCount > 0) {
    const media: LayerMedia = {
      kind: "video", path: recipe.effectClipPath, frameCount: probed.frameCount, fps: probed.fps, width: probed.width, height: probed.height,
      start: 0, inPoint: 0, outPoint: probed.frameCount, loop: false, crossfade: 0,
    };
    // The old worker: the clip retimed by the effect speed, resampled to the source rate, looped or held.
    const at = frameAt((index / fps) * recipe.effectSpeed, probed.fps);
    effectClip = { media, index: recipe.loop ? at % probed.frameCount : Math.min(at, probed.frameCount - 1) };
  }
  return {
    id: `vfx:${clipId}:${layer.id}`, kind: "maskVfx", enabled: true, recipe, index, count, fps, envelope, mattePath,
    sourceWidth: cache.width, sourceHeight: cache.height, effectClip,
  };
}

/** The programme hook: live nodes, or the rendered file where a layer can only be shown that way. */
export function maskVfxHook(mode: VfxMode = "live"): NonNullable<ProgrammeHooks["vfx"]> {
  return (clip, local) => {
    const source = clip.sourceIn + local * clip.rate;
    const active = clip.vfx.filter((layer) => layer.enabled
      && source >= clip.sourceIn + layer.rangeStart - 1e-9 && source < clip.sourceIn + layer.rangeEnd - 1e-9);
    if (!active.length) return undefined;
    // The topmost layer that can only be a file decides the picture, as the old export's segments did.
    const fileOnly = active.slice().reverse().find((layer) => layer.renderCachePath && (mode === "legacy" || !layer.matteCache));
    if (fileOnly) {
      return { nodes: [], masks: [], sourcePath: fileOnly.renderCachePath, sourceSeconds: source - (clip.sourceIn + fileOnly.rangeStart) };
    }
    const nodes = active.flatMap((layer) => maskVfxNode(clip.id, clip.sourceIn, layer, source, clip.media.fps) ?? []);
    return nodes.length ? { nodes, masks: [] } : undefined;
  };
}

/**
 * One frame of a clip with one VFX layer, at the clip's own size and nowhere else: the Mask Effects
 * workspace's live preview and its bake. It is a one-clip programme, so it goes through exactly the
 * code the timeline does. `source` is the source second to show.
 */
export function vfxPreviewDocument(media: ProgrammeMedia, clipId: string, clipSourceIn: number, layer: ClipVfxLayer, source: number): PhotoDocument {
  const offset = source - clipSourceIn;
  const programme: Programme = {
    width: media.width, height: media.height, fps: media.fps, duration: 1 / media.fps,
    clips: [{
      id: clipId, name: "preview", media, start: -offset, duration: offset + 1, sourceIn: clipSourceIn, rate: 1,
      x: 0.5, y: 0.5, scale: 100, rotation: 0, opacity: 1, blend: "normal", effects: effectsFor(undefined),
      conform: { fit: "contain", pad: "black" }, vfx: [{ ...layer, enabled: true }], volumeDb: 0,
    }],
    overlays: [], graphics: [], audio: [], rippled: 0, pixelScale: 1,
  };
  return programmeDocument(programme, 0, { vfx: maskVfxHook("live") });
}
