import { effectsFor } from "../../domain/effects";
import { clipDuration, clipPlaybackRate, seconds, timelineClipDuration } from "../../domain/time";
import { mapToProgramme, resolveJunctionLayout } from "../../domain/junctions";
import { DEFAULT_CONFORM } from "../../domain/types";
import type { ClipConform, Project } from "../../domain/types";

/**
 * Per-clip conform beats the sequence default, which beats plain `contain`.
 *
 * Resolved here rather than in Rust so the monitor preview and the render read the exact same
 * value -- a preview that disagrees with the export is how the pillarbox bars went unnoticed until
 * someone opened the file in VLC.
 */
export function resolveConform(clip: { conform?: ClipConform }, sequenceConform?: ClipConform): ClipConform {
  return clip.conform ?? sequenceConform ?? DEFAULT_CONFORM;
}

function vfxSegments<T extends { id: string; enabled: boolean; rangeStart: number; rangeEnd: number; renderCachePath?: string }>(
  duration: number, layers: T[] | undefined,
) {
  const usable = (layers ?? []).filter((layer) => layer.enabled && layer.renderCachePath
    && layer.rangeEnd > 0 && layer.rangeStart < duration);
  const boundaries = Array.from(new Set([0, duration, ...usable.flatMap((layer) => [
    Math.max(0, Math.min(duration, layer.rangeStart)), Math.max(0, Math.min(duration, layer.rangeEnd)),
  ])])).sort((a, b) => a - b);
  return boundaries.slice(0, -1).flatMap((from, index) => {
    const to = boundaries[index + 1];
    if (to <= from + .000_001) return [];
    const middle = (from + to) / 2;
    // Later layers are visually above earlier layers, matching their order in the clip inspector.
    const layer = usable.slice().reverse().find((candidate) => candidate.rangeStart <= middle && candidate.rangeEnd >= middle);
    return [{ from, to, layer }];
  });
}

export function buildTimelineExportPayload(project: Project, upscaleFactor: number) {
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId);
  const track = sequence?.tracks.find((item) => item.id === "v1");
  // Overlay picture tracks. V2 composites over V1; later tracks draw on top of earlier ones.
  const overlayTracks = (sequence?.tracks ?? [])
    .filter((item) => item.kind === "video" && item.id !== "v1" && item.overlay && !item.hidden);

  let v1 = (track?.hidden ? [] : track?.clips ?? []).filter((clip) => clip.enabled)
    .sort((a, b) => seconds(a.start) - seconds(b.start));

  // With nothing on V1 there is nothing to overlay ONTO, so the lowest overlay track becomes the
  // programme and renders full-frame -- its placement box would otherwise letterbox it against
  // black for no reason. Decided at TRACK level, not per-instant: a clip that changed size halfway
  // through because V1 ran out underneath it would read as a glitch.
  //
  // Doing this here rather than in Rust keeps the renderer simple: it still receives a programme
  // track plus overlays, and never has to know the rule exists.
  let promoted: (typeof overlayTracks)[number] | undefined;
  if (!v1.length) {
    promoted = overlayTracks.find((item) => item.clips.some((clip) => clip.enabled));
    if (promoted) {
      v1 = promoted.clips.filter((clip) => clip.enabled)
        .sort((a, b) => seconds(a.start) - seconds(b.start));
    }
  }
  // The SAME layout the monitor and the timeline read. Computing it here independently is exactly
  // how the preview and the exported file came to disagree.
  const layout = resolveJunctionLayout(v1);
  const rippleAt = (time: number) => mapToProgramme(layout, v1, time);
  const clips = v1.flatMap((clip) => {
      const asset = project.assets.find((item) => item.id === clip.assetId);
      if (!asset) return [];
      const duration = timelineClipDuration(clip);
      const rate = clipPlaybackRate(clip);
      return vfxSegments(duration, clip.vfxLayers).map(({ from, to, layer }) => ({
        path: layer?.renderCachePath ?? asset.path,
        startSeconds: rippleAt(seconds(clip.start) + from),
        sourceInSeconds: layer ? from - layer.rangeStart : seconds(clip.sourceIn) + from * rate,
        durationSeconds: to - from,
        sourceDurationSeconds: layer ? to - from : (to - from) * rate,
        speedRate: layer ? 1 : rate, hasAudio: Boolean(asset.hasAudio),
        positionX: clip.positionX ?? (sequence?.width ?? 1920) / 2,
        positionY: clip.positionY ?? (sequence?.height ?? 1080) / 2,
        scalePercent: clip.scale ?? 100, rotationDegrees: clip.rotation ?? 0, opacity: clip.opacity,
        volumeDb: track?.muted ? -60 : clip.volumeDb,
        transitionKind: from <= .000_001 ? clip.transitionIn ?? "none" : "none",
        transitionDuration: clip.transitionDuration ?? 0.5,
        transitionOutKind: to >= duration - .000_001 ? clip.transitionOut ?? "none" : "none",
        transitionOutDuration: clip.transitionOutDuration ?? clip.transitionDuration ?? 0.5,
        effects: effectsFor(clip.effects),
        conform: resolveConform(clip, sequence?.conform),
        // What the renderer may take from before this clip's in-point. Computed here, not in Rust,
        // so the shift applied to the other tracks and the handle used by the render can never
        // disagree -- they come from the same number.
        headHandleSeconds: from <= .000_001 ? layout.borrowed.get(clip.id) ?? 0 : 0,
        vfxLayerId: layer?.id,
      }));
    });
  const graphics = (sequence?.tracks.filter((item) => ["g1", "c1"].includes(item.id) && !item.hidden).flatMap((item) => item.clips) ?? [])
    .filter((clip) => clip.enabled && clip.graphic).map((clip) => ({
      startSeconds: rippleAt(seconds(clip.start)), durationSeconds: timelineClipDuration(clip), opacity: clip.opacity, ...clip.graphic!,
    }));
  const audioClips = (sequence?.tracks ?? []).filter((item) => item.kind === "audio" && !item.muted)
    .flatMap((item) => item.clips.filter((clip) => clip.enabled).map((clip) => ({ clip, track: item })))
    .sort((a, b) => seconds(a.clip.start) - seconds(b.clip.start)).flatMap(({ clip, track: audioTrack }) => {
      const asset = project.assets.find((item) => item.id === clip.assetId);
      return asset ? [{
        path: asset.path, sourceInSeconds: seconds(clip.sourceIn), durationSeconds: timelineClipDuration(clip),
        sourceDurationSeconds: clipDuration(clip.sourceIn, clip.sourceOut), speedRate: clipPlaybackRate(clip),
        startSeconds: rippleAt(seconds(clip.start)), volumeDb: clip.volumeDb + (audioTrack.gainDb ?? 0),
        pan: clip.audio?.pan ?? 0,
        fadeInSeconds: clip.audio?.fadeInSeconds ?? 0,
        fadeOutSeconds: clip.audio?.fadeOutSeconds ?? 0,
      }] : [];
    });
  const trackOverlays = overlayTracks.filter((item) => item.id !== promoted?.id)
    .slice().reverse().flatMap((item) => item.clips.filter((clip) => clip.enabled).flatMap((clip) => {
      const asset = project.assets.find((candidate) => candidate.id === clip.assetId);
      const box = item.overlay!;
      if (!asset) return [];
      const duration = timelineClipDuration(clip);
      const rate = clipPlaybackRate(clip);
      return vfxSegments(duration, clip.vfxLayers).map(({ from, to, layer }) => ({
        path: layer?.renderCachePath ?? asset.path,
        startSeconds: rippleAt(seconds(clip.start) + from),
        sourceInSeconds: layer ? from - layer.rangeStart : seconds(clip.sourceIn) + from * rate,
        durationSeconds: to - from, sourceDurationSeconds: layer ? to - from : (to - from) * rate,
        speedRate: layer ? 1 : rate, hasAudio: Boolean(asset.hasAudio), volumeDb: item.muted ? -60 : clip.volumeDb,
        boxX: box.x, boxY: box.y, boxWidth: box.width, boxHeight: box.height, fit: box.fit,
        opacity: box.opacity * clip.opacity, cornerRadius: box.cornerRadius, borderWidth: box.borderWidth,
        borderColor: box.borderColor, effects: effectsFor(clip.effects),
        vfxLayerId: layer?.id,
      }));
    }));
  const overlayClips = trackOverlays;
  return { sequence, clips, graphics, audioClips, overlayClips, upscaleFactor, rippleSeconds: layout.rippled };
}

/** Crop an export payload to a short programme-time window for rendered timeline previews. */
export function sliceTimelineExportPayload(
  payload: ReturnType<typeof buildTimelineExportPayload>, startSeconds: number, durationSeconds: number,
): ReturnType<typeof buildTimelineExportPayload> {
  const windowStart = Math.max(0, startSeconds);
  const windowEnd = windowStart + Math.max(0.04, durationSeconds);
  const intersection = (start: number, duration: number) => {
    const left = Math.max(windowStart, start);
    const right = Math.min(windowEnd, start + duration);
    return right > left + 0.000_001 ? { left, right, duration: right - left } : undefined;
  };
  const clips = payload.clips.flatMap((clip) => {
    const overlap = intersection(clip.startSeconds, clip.durationSeconds);
    if (!overlap) return [];
    const head = overlap.left - clip.startSeconds;
    const tailClipped = overlap.right < clip.startSeconds + clip.durationSeconds - 0.000_001;
    return [{ ...clip, startSeconds: overlap.left - windowStart,
      sourceInSeconds: clip.sourceInSeconds + head * clip.speedRate,
      sourceDurationSeconds: overlap.duration * clip.speedRate, durationSeconds: overlap.duration,
      transitionKind: head > 0 ? "none" : clip.transitionKind,
      transitionOutKind: tailClipped ? "none" : clip.transitionOutKind,
      headHandleSeconds: head > 0 ? 0 : clip.headHandleSeconds,
    }];
  });
  const audioClips = payload.audioClips.flatMap((clip) => {
    const overlap = intersection(clip.startSeconds, clip.durationSeconds);
    if (!overlap) return [];
    const head = overlap.left - clip.startSeconds;
    return [{ ...clip, startSeconds: overlap.left - windowStart,
      sourceInSeconds: clip.sourceInSeconds + head * clip.speedRate,
      sourceDurationSeconds: overlap.duration * clip.speedRate, durationSeconds: overlap.duration,
      fadeInSeconds: head > 0 ? 0 : Math.min(clip.fadeInSeconds, overlap.duration),
      fadeOutSeconds: overlap.right < clip.startSeconds + clip.durationSeconds - 0.000_001
        ? 0 : Math.min(clip.fadeOutSeconds, overlap.duration),
    }];
  });
  const overlayClips = payload.overlayClips.flatMap((clip) => {
    const overlap = intersection(clip.startSeconds, clip.durationSeconds);
    if (!overlap) return [];
    const head = overlap.left - clip.startSeconds;
    return [{ ...clip, startSeconds: overlap.left - windowStart,
      sourceInSeconds: clip.sourceInSeconds + head * clip.speedRate,
      sourceDurationSeconds: overlap.duration * clip.speedRate, durationSeconds: overlap.duration,
    }];
  });
  const graphics = payload.graphics.flatMap((graphic) => {
    const overlap = intersection(graphic.startSeconds, graphic.durationSeconds);
    return overlap ? [{ ...graphic, startSeconds: overlap.left - windowStart, durationSeconds: overlap.duration }] : [];
  });
  return { ...payload, clips, audioClips, overlayClips, graphics };
}
