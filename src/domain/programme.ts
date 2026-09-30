/**
 * VFX Phase 7: the programme -- a timeline as the compositor draws it.
 *
 * `buildProgramme` reads a project once (the same junction layout, conform resolution and ripple
 * mapping the old export payload used) into a flat, render-ready description. `programmeDocument`
 * then turns any frame of it into an ordinary compositor document:
 *
 *   - V1 clips as media (or picture) layers, conformed (fit, pad, static or moving crop), then the
 *     clip's own transform, opacity, blend mode and Effects (`clipEffects` node);
 *   - junction transitions between V1 clips (all of `TRANSITIONS`) as opacity, placement offsets,
 *     vector masks (Phase 6 `path` entries) and per-pixel treatments;
 *   - the first clip's "in" and the last clip's "out" as dips from and to a colour;
 *   - overlay tracks in their boxes (fit, rounded corners, border, opacity, Effects);
 *   - titles and captions (`timelineGraphic` layers);
 *   - live mask VFX on clips (`maskVfx` nodes), or the old rendered file where a layer has no matte.
 *
 * The monitor renders these documents at its size and the export at the delivery size, through the
 * same compositors, so the preview and the file agree by construction. Every id is derived from the
 * timeline, so consecutive frames' documents differ only where the picture does.
 *
 * Timing semantics (they are the old export's, which the monitor only approximated): a junction of
 * D seconds between A and B plays over [end(A) - D, end(A)); B's source time is always
 * sourceIn + (t - start(B)) * rate, so where B borrows a head handle it shows the frames before its
 * in-point, and where it could not (a clip starting at frame 0) the clips overlap on the timeline.
 */

import { effectsFor } from "./effects";
import { mapToProgramme, resolveJunctionLayout } from "./junctions";
import { NEUTRAL_EDGE } from "./photoEdge";
import type { LayerMedia, MaskEntry, PhotoDocument, PhotoLayer, PhotoMask, PhotoNode } from "./photoDocument";
import type { BlendMode as PhotoBlendMode } from "./photoBlend";
import { isNeutralClipEffects, type ClipEffectsParams } from "./photoClipEffects";
import { NEUTRAL_PATH, type PathBlock, type Subpath } from "./photoPath";
import { resolveEdge, resolveJunction } from "./transitions";
import { clipPlaybackRate, seconds, timelineClipDuration } from "./time";
import {
  DEFAULT_CONFORM, type BlendMode, type ClipConform, type ClipEffects, type GraphicOverlay, type MediaAsset,
  type OverlayPlacement, type Project, type SourceRect, type TimelineClip,
} from "./types";
import type { ClipVfxLayer } from "./maskVfx";
import { programmeMatte, type ProgrammeMatte } from "./clipMatte";
import { croppedRect, layerMatrix } from "./photoCompose";
import { applyMatrix, invert } from "./photoGeometry";

/* ================================================================= model */

export type ProgrammeMedia = {
  assetId: string;
  path: string;
  kind: "video" | "image";
  fps: number;
  frameCount: number;
  width: number;
  height: number;
  proxyPath?: string;
  hasAudio: boolean;
};

export type ProgrammeClip = {
  id: string;
  name: string;
  media: ProgrammeMedia;
  /** Programme seconds (after ripple). */
  start: number;
  duration: number;
  /** Source seconds at `start`, and source seconds per programme second. */
  sourceIn: number;
  rate: number;
  /** Centre of the conformed picture, normalised to the frame; scale in percent; degrees. */
  x: number;
  y: number;
  scale: number;
  rotation: number;
  opacity: number;
  blend: BlendMode;
  effects: ClipEffects;
  conform: ClipConform;
  /** The junction with the previous clip, when it has one. */
  incoming?: { kind: string; duration: number };
  /** Head or tail dip (first clip's "in", last clip's "out"). */
  headFade?: { kind: string; duration: number };
  tailFade?: { kind: string; duration: number };
  vfx: ClipVfxLayer[];
  /** Background removal: the clip's matte as its alpha. */
  matte?: ProgrammeMatte;
  volumeDb: number;
};

export type ProgrammeOverlay = {
  id: string;
  media: ProgrammeMedia;
  start: number;
  duration: number;
  sourceIn: number;
  rate: number;
  box: OverlayPlacement;
  opacity: number;
  effects: ClipEffects;
  vfx: ClipVfxLayer[];
  matte?: ProgrammeMatte;
  volumeDb: number;
  muted: boolean;
};

export type ProgrammeGraphic = { id: string; overlay: GraphicOverlay; start: number; duration: number; opacity: number };

/** Everything audible, on the programme clock (the export mixes these; the monitor plays them). */
export type ProgrammeAudio = {
  id: string;
  path: string;
  start: number;
  /** Programme seconds. */
  duration: number;
  sourceIn: number;
  rate: number;
  volumeDb: number;
  pan: number;
  fadeIn: number;
  fadeOut: number;
  kind: "picture" | "overlay" | "audio";
};

export type Programme = {
  width: number;
  height: number;
  fps: number;
  /** Seconds: the V1 programme, which is what the export has always delivered. */
  duration: number;
  clips: ProgrammeClip[];
  overlays: ProgrammeOverlay[];
  graphics: ProgrammeGraphic[];
  audio: ProgrammeAudio[];
  /** Colour of the head and tail dips of the whole programme, if any (also applied to the audio). */
  headFade?: { color: string; duration: number };
  tailFade?: { color: string; duration: number };
  /** Seconds taken out of the timeline by transitions that had no handles (see junctions.ts). */
  rippled: number;
  /** Canvas pixels per sequence pixel (a "none" conform keeps source pixels at the sequence's scale). */
  pixelScale: number;
};

const fpsOf = (asset: MediaAsset): number => asset.frameRate
  ? asset.frameRate.numerator / Math.max(1, asset.frameRate.denominator) : 24;

export function programmeMedia(asset: MediaAsset, path = asset.path): ProgrammeMedia {
  const fps = fpsOf(asset);
  const kind = asset.kind === "image" ? "image" : "video";
  return {
    assetId: asset.id, path, kind, fps,
    frameCount: kind === "image" ? 1 : Math.max(1, Math.round(seconds(asset.duration) * fps)),
    width: asset.width ?? 1920, height: asset.height ?? 1080,
    proxyPath: path === asset.path ? asset.proxyPath : undefined,
    hasAudio: Boolean(asset.hasAudio),
  };
}

/** The dip colour of a transition id at the head or tail of the programme. */
export const dipColour = (id: string | undefined): string =>
  id === "fadewhite" ? "#ffffff" : id === "fadegrays" ? "#808080" : "#000000";

export function resolveClipConform(clip: { conform?: ClipConform }, sequenceConform?: ClipConform): ClipConform {
  return clip.conform ?? sequenceConform ?? DEFAULT_CONFORM;
}

/**
 * The project's active sequence as a programme, at `width` x `height` and `fps` (the sequence's own
 * by default -- an export can ask for another size or rate; placements are normalised so they hold).
 */
export function buildProgramme(project: Project, options: { width?: number; height?: number; fps?: number } = {}): Programme | undefined {
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId);
  if (!sequence) return undefined;
  const sequenceFps = sequence.frameRate.numerator / Math.max(1, sequence.frameRate.denominator);
  const width = options.width || sequence.width, height = options.height || sequence.height;
  const fps = options.fps || sequenceFps;
  const assets = new Map(project.assets.map((asset) => [asset.id, asset]));
  const track = sequence.tracks.find((item) => item.id === "v1");
  const overlayTracks = sequence.tracks.filter((item) => item.kind === "video" && item.id !== "v1" && item.overlay && !item.hidden);
  let v1 = (track?.hidden ? [] : track?.clips ?? []).filter((clip) => clip.enabled).sort((a, b) => seconds(a.start) - seconds(b.start));
  let promoted: typeof overlayTracks[number] | undefined;
  let v1Muted = Boolean(track?.muted);
  // With nothing on V1 the lowest overlay track is the programme, full frame (as the export did).
  if (!v1.length) {
    promoted = overlayTracks.find((item) => item.clips.some((clip) => clip.enabled));
    if (promoted) {
      v1 = promoted.clips.filter((clip) => clip.enabled).sort((a, b) => seconds(a.start) - seconds(b.start));
      v1Muted = promoted.muted;
    }
  }
  v1 = v1.filter((clip) => assets.get(clip.assetId) && assets.get(clip.assetId)!.kind !== "audio");
  const layout = resolveJunctionLayout(v1);
  const rippleAt = (time: number) => mapToProgramme(layout, v1, time);

  const clips: ProgrammeClip[] = v1.map((clip, index) => {
    const asset = assets.get(clip.assetId)!;
    const previous = index > 0 ? v1[index - 1] : undefined;
    const junction = previous ? resolveJunction(previous, clip) : undefined;
    const junctionDuration = layout.transition.get(clip.id) ?? 0;
    const out: ProgrammeClip = {
      id: clip.id, name: clip.name, media: programmeMedia(asset),
      start: layout.start.get(clip.id) ?? seconds(clip.start),
      duration: timelineClipDuration(clip),
      sourceIn: seconds(clip.sourceIn), rate: clipPlaybackRate(clip),
      x: (clip.positionX ?? sequence.width / 2) / sequence.width,
      y: (clip.positionY ?? sequence.height / 2) / sequence.height,
      scale: clip.scale ?? 100, rotation: clip.rotation ?? 0, opacity: clip.opacity,
      blend: clip.blendMode ?? "normal",
      effects: effectsFor(clip.effects),
      conform: resolveClipConform(clip, sequence.conform),
      vfx: (clip.vfxLayers ?? []).filter((layer) => layer.enabled),
      volumeDb: v1Muted ? -60 : clip.volumeDb,
    };
    const matte = programmeMatte(clip.matte);
    if (matte) out.matte = matte;
    if (junction && junction.id !== "none" && junctionDuration > 0) out.incoming = { kind: junction.id, duration: junctionDuration };
    if (index === 0) {
      const edge = resolveEdge(clip.transitionIn, clip.transitionDuration);
      if (edge.id !== "none") out.headFade = { kind: edge.id, duration: Math.min(edge.duration, out.duration) };
    }
    if (index === v1.length - 1) {
      const edge = resolveEdge(clip.transitionOut, clip.transitionOutDuration ?? clip.transitionDuration);
      if (edge.id !== "none") out.tailFade = { kind: edge.id, duration: Math.min(edge.duration, out.duration) };
    }
    return out;
  });

  const overlays: ProgrammeOverlay[] = overlayTracks.filter((item) => item.id !== promoted?.id).slice().reverse()
    .flatMap((item) => item.clips.filter((clip) => clip.enabled).flatMap((clip): ProgrammeOverlay[] => {
      const asset = assets.get(clip.assetId);
      if (!asset || asset.kind === "audio") return [];
      const overlay: ProgrammeOverlay = {
        id: clip.id, media: programmeMedia(asset), start: rippleAt(seconds(clip.start)),
        duration: timelineClipDuration(clip), sourceIn: seconds(clip.sourceIn), rate: clipPlaybackRate(clip),
        box: item.overlay!, opacity: item.overlay!.opacity * clip.opacity, effects: effectsFor(clip.effects),
        vfx: (clip.vfxLayers ?? []).filter((layer) => layer.enabled), volumeDb: clip.volumeDb, muted: item.muted,
      };
      const matte = programmeMatte(clip.matte);
      if (matte) overlay.matte = matte;
      return [overlay];
    }));

  const graphics: ProgrammeGraphic[] = sequence.tracks.filter((item) => ["g1", "c1"].includes(item.id) && !item.hidden)
    .flatMap((item) => item.clips).filter((clip) => clip.enabled && clip.graphic)
    .map((clip) => ({ id: clip.id, overlay: clip.graphic!, start: rippleAt(seconds(clip.start)), duration: timelineClipDuration(clip), opacity: clip.opacity }));

  const duration = layout.duration;
  const first = clips[0], last = clips[clips.length - 1];
  const headFade = first?.headFade ? { color: dipColour(first.headFade.kind), duration: first.headFade.duration } : undefined;
  const tailFade = last?.tailFade ? { color: dipColour(last.tailFade.kind), duration: last.tailFade.duration } : undefined;

  const audio: ProgrammeAudio[] = [];
  // Picture audio: each V1 clip for its own span, with junctions crossfaded (the export's acrossfade):
  // the incoming clip starts at the window's start (inside its handle) and fades in over the window
  // while the outgoing one fades out.
  clips.forEach((clip, index) => {
    if (!clip.media.hasAudio || clip.media.kind !== "video") return;
    const next = clips[index + 1];
    const windowStart = clip.incoming ? Math.min(clip.start, (clips[index - 1].start + clips[index - 1].duration) - clip.incoming.duration) : clip.start;
    const lead = clip.start - windowStart;
    audio.push({
      id: `a:${clip.id}`, path: clip.media.path, start: windowStart, duration: clip.duration + lead,
      sourceIn: Math.max(0, clip.sourceIn - lead * clip.rate), rate: clip.rate, volumeDb: clip.volumeDb, pan: 0,
      fadeIn: clip.incoming?.duration ?? 0, fadeOut: next?.incoming?.duration ?? 0, kind: "picture",
    });
  });
  for (const overlay of overlays) {
    if (!overlay.media.hasAudio || overlay.media.kind !== "video") continue;
    audio.push({
      id: `a:${overlay.id}`, path: overlay.media.path, start: overlay.start, duration: overlay.duration,
      sourceIn: overlay.sourceIn, rate: overlay.rate, volumeDb: overlay.muted ? -60 : overlay.volumeDb, pan: 0, fadeIn: 0, fadeOut: 0, kind: "overlay",
    });
  }
  for (const audioTrack of sequence.tracks.filter((item) => item.kind === "audio" && !item.muted)) {
    for (const clip of audioTrack.clips.filter((item) => item.enabled)) {
      const asset = assets.get(clip.assetId);
      if (!asset) continue;
      audio.push({
        id: `a:${clip.id}`, path: asset.path, start: rippleAt(seconds(clip.start)), duration: timelineClipDuration(clip),
        sourceIn: seconds(clip.sourceIn), rate: clipPlaybackRate(clip), volumeDb: clip.volumeDb + (audioTrack.gainDb ?? 0),
        pan: clip.audio?.pan ?? 0, fadeIn: clip.audio?.fadeInSeconds ?? 0, fadeOut: clip.audio?.fadeOutSeconds ?? 0, kind: "audio",
      });
    }
  }
  return { width, height, fps, duration, clips, overlays, graphics, audio, headFade, tailFade, rippled: layout.rippled, pixelScale: width / Math.max(1, sequence.width) };
}

/* ======================================================== frame documents */

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
const smooth = (value: number) => value * value * (3 - 2 * value);

/** Photo blend ids for the timeline's. */
export const photoBlend = (mode: BlendMode | undefined): PhotoBlendMode =>
  mode === "hardlight" ? "hardLight" : (mode ?? "normal") as PhotoBlendMode;

/** A rectangle path in normalised canvas coordinates. */
function rectPath(x0: number, y0: number, x1: number, y1: number): Subpath {
  return { closed: true, nodes: [{ x: x0, y: y0 }, { x: x1, y: y0 }, { x: x1, y: y1 }, { x: x0, y: y1 }] };
}

/** An ellipse path (4 cubics) in normalised canvas coordinates. */
function ellipsePath(cx: number, cy: number, rx: number, ry: number): Subpath {
  const k = 0.5522847498;
  return {
    closed: true, nodes: [
      { x: cx + rx, y: cy, in: [0, -k * ry], out: [0, k * ry] },
      { x: cx, y: cy + ry, in: [k * rx, 0], out: [-k * rx, 0] },
      { x: cx - rx, y: cy, in: [0, k * ry], out: [0, -k * ry] },
      { x: cx, y: cy - ry, in: [-k * rx, 0], out: [k * rx, 0] },
    ],
  };
}

/** A rounded rectangle path in normalised canvas coordinates (`r` in x units, `ry` in y units). */
function roundedRectPath(x0: number, y0: number, x1: number, y1: number, rx: number, ry: number): Subpath {
  if (rx <= 1e-6 || ry <= 1e-6) return rectPath(x0, y0, x1, y1);
  const k = 0.5522847498;
  return {
    closed: true, nodes: [
      { x: x0 + rx, y: y0, in: [-k * rx, 0] }, { x: x1 - rx, y: y0, out: [k * rx, 0] },
      { x: x1, y: y0 + ry, in: [0, -k * ry] }, { x: x1, y: y1 - ry, out: [0, k * ry] },
      { x: x1 - rx, y: y1, in: [k * rx, 0] }, { x: x0 + rx, y: y1, out: [-k * rx, 0] },
      { x: x0, y: y1 - ry, in: [0, k * ry] }, { x: x0, y: y0 + ry, out: [0, -k * ry] },
    ],
  };
}

/** A pie sweep from 12 o'clock clockwise by `fraction` of a turn, covering the frame (normalised). */
function sweepPath(fraction: number, aspect: number): Subpath {
  const f = clamp01(fraction);
  if (f <= 0) return { closed: true, nodes: [] };
  const nodes = [{ x: 0.5, y: 0.5 }];
  const steps = Math.max(2, Math.ceil(f * 64));
  for (let i = 0; i <= steps; i++) {
    const a = -Math.PI / 2 + (f * Math.PI * 2 * i) / steps;
    // A radius that clears the corners at any aspect.
    nodes.push({ x: 0.5 + Math.cos(a) * 1.5, y: 0.5 + Math.sin(a) * 1.5 * aspect });
  }
  return { closed: true, nodes };
}

/** Bands for the slice and wind transitions: `count` stripes along the axis, each `reach` full. */
function stripesPath(reach: number, horizontal: boolean, fromEnd: boolean, count = 12): Subpath[] {
  const band = 1 / count, filled = clamp01(reach) * band;
  const out: Subpath[] = [];
  for (let i = 0; i < count; i++) {
    const a = i * band + (fromEnd ? band - filled : 0), b = a + filled;
    out.push(horizontal ? rectPath(a, 0, b, 1) : rectPath(0, a, 1, b));
  }
  return out;
}

/** How one clip is drawn at a moment of a transition. */
export type ClipTreatment = {
  opacity: number;
  /** Offset of the whole clip, in frame fractions. */
  dx: number;
  dy: number;
  /** Scale about the frame centre (x and y). */
  sx: number;
  sy: number;
  /** Vector mask in normalised canvas coordinates (fill = visible). */
  mask?: Subpath[];
  effects?: Partial<ClipEffectsParams>;
};

const plain = (): ClipTreatment => ({ opacity: 1, dx: 0, dy: 0, sx: 1, sy: 1 });

export type TransitionFrame = {
  outgoing: ClipTreatment;
  incoming: ClipTreatment;
  dip?: { color: string; opacity: number };
  /** The outgoing clip is drawn over the incoming one (it closes, slides away or squeezes off it). */
  outgoingOnTop?: boolean;
};

/** Transitions in which the outgoing picture is what moves or closes, over the incoming one. */
const OUTGOING_ON_TOP = new Set([
  "revealleft", "revealright", "revealup", "revealdown", "circleclose", "vertclose", "horzclose", "squeezeh",
]);

/**
 * One junction transition at progress `p` (0 all outgoing, 1 all incoming), as treatments for the
 * two clips. The geometry follows the monitor's former CSS versions (transitionPreview.ts), made
 * exact: wipes and irises are vector masks, pushes and covers are offsets, dips are colour layers.
 */
export function transitionFrame(id: string, rawProgress: number, aspect: number): TransitionFrame {
  const p = clamp01(rawProgress), s = smooth(p);
  const out = plain(), inc = plain();
  const result: TransitionFrame = { outgoing: out, incoming: inc };
  if (OUTGOING_ON_TOP.has(id)) result.outgoingOnTop = true;
  // CSS `circle(r)`: a radius of r * hypot(W, H) / sqrt(2) pixels, in normalised x and y.
  const circleR = (r: number) => ({ rx: r / Math.SQRT2 * Math.hypot(1, 1 / aspect), ry: r / Math.SQRT2 * Math.hypot(aspect, 1) });
  const circle = (r: number) => { const { rx, ry } = circleR(r); return [ellipsePath(0.5, 0.5, rx, ry)]; };
  const diagonal = (corner: "tl" | "tr" | "bl" | "br"): Subpath[] => {
    const q = p * 2;
    const tri = {
      tl: [[0, 0], [q, 0], [0, q]], tr: [[1, 0], [1 - q, 0], [1, q]],
      bl: [[0, 1], [q, 1], [0, 1 - q]], br: [[1, 1], [1 - q, 1], [1, 1 - q]],
    }[corner];
    return [{ closed: true, nodes: tri.map(([x, y]) => ({ x, y })) }];
  };
  switch (id) {
    case "none": inc.opacity = p < 0.5 ? 0 : 1; break;
    case "fade": case "distance": inc.opacity = p; break;
    case "dissolve": inc.effects = { dissolve: p }; break;
    case "fadefast": inc.opacity = clamp01(p * 1.6); break;
    case "fadeslow": inc.opacity = smooth(p) * p; break;
    case "fadeblack": case "fadewhite":
      inc.opacity = p < 0.5 ? 0 : 1;
      result.dip = { color: id === "fadewhite" ? "#ffffff" : "#000000", opacity: 1 - Math.abs(p - 0.5) * 2 };
      break;
    case "fadegrays": {
      const grey = 1 - Math.abs(p - 0.5) * 2;
      out.effects = { grayscale: grey };
      inc.effects = { grayscale: grey };
      inc.opacity = p;
      break;
    }
    case "hblur":
      out.effects = { blur: p * 14 };
      inc.effects = { blur: (1 - p) * 14 };
      inc.opacity = p;
      break;
    case "pixelize": {
      const bite = 1 - Math.abs(p - 0.5) * 2;
      const block = 1 + Math.round(bite * 48);
      out.effects = { pixelate: block };
      inc.effects = { pixelate: block };
      inc.opacity = p < 0.5 ? 0 : 1;
      break;
    }
    case "wipeleft": inc.mask = [rectPath(1 - p, 0, 1, 1)]; break;
    case "wiperight": inc.mask = [rectPath(0, 0, p, 1)]; break;
    case "wipeup": inc.mask = [rectPath(0, 1 - p, 1, 1)]; break;
    case "wipedown": inc.mask = [rectPath(0, 0, 1, p)]; break;
    case "wipetl": case "diagtl": inc.mask = diagonal("tl"); break;
    case "wipetr": case "diagtr": inc.mask = diagonal("tr"); break;
    case "wipebl": case "diagbl": inc.mask = diagonal("bl"); break;
    case "wipebr": case "diagbr": inc.mask = diagonal("br"); break;
    case "slideleft": out.dx = -p; inc.dx = 1 - p; break;
    case "slideright": out.dx = p; inc.dx = p - 1; break;
    case "slideup": out.dy = -p; inc.dy = 1 - p; break;
    case "slidedown": out.dy = p; inc.dy = p - 1; break;
    case "smoothleft": out.dx = -s; inc.dx = 1 - s; break;
    case "smoothright": out.dx = s; inc.dx = s - 1; break;
    case "smoothup": out.dy = -s; inc.dy = 1 - s; break;
    case "smoothdown": out.dy = s; inc.dy = s - 1; break;
    case "coverleft": inc.dx = 1 - s; break;
    case "coverright": inc.dx = s - 1; break;
    case "coverup": inc.dy = 1 - s; break;
    case "coverdown": inc.dy = s - 1; break;
    case "revealleft": out.dx = -s; break;
    case "revealright": out.dx = s; break;
    case "revealup": out.dy = -s; break;
    case "revealdown": out.dy = s; break;
    case "circleopen": inc.mask = circle(p * 0.75); break;
    case "circleclose": out.mask = circle((1 - p) * 0.75); break;
    case "circlecrop":
      // Through black: the programme's background is black, so the "dip" is simply what shows.
      if (p < 0.5) { out.mask = circle((1 - p * 2) * 0.75); inc.opacity = 0; } else { inc.mask = circle((p - 0.5) * 1.5); out.opacity = 0; }
      break;
    case "rectcrop": { const e = (1 - p) / 2; inc.mask = [rectPath(e, e, 1 - e, 1 - e)]; break; }
    case "radial": inc.mask = [sweepPath(p, aspect)]; break;
    case "vertopen": { const e = (1 - p) / 2; inc.mask = [rectPath(e, 0, 1 - e, 1)]; break; }
    case "vertclose": { const e = p / 2; out.mask = [rectPath(e, 0, 1 - e, 1)]; break; }
    case "horzopen": { const e = (1 - p) / 2; inc.mask = [rectPath(0, e, 1, 1 - e)]; break; }
    case "horzclose": { const e = p / 2; out.mask = [rectPath(0, e, 1, 1 - e)]; break; }
    case "hlslice": inc.mask = stripesPath(p, true, false); break;
    case "hrslice": inc.mask = stripesPath(p, true, true); break;
    case "vuslice": inc.mask = stripesPath(p, false, true); break;
    case "vdslice": inc.mask = stripesPath(p, false, false); break;
    case "hlwind": inc.mask = stripesPath(p * 1.45, true, false, 9); inc.opacity = clamp01(p * 1.4); break;
    case "hrwind": inc.mask = stripesPath(p * 1.45, true, true, 9); inc.opacity = clamp01(p * 1.4); break;
    case "vuwind": inc.mask = stripesPath(p * 1.45, false, true, 9); inc.opacity = clamp01(p * 1.4); break;
    case "vdwind": inc.mask = stripesPath(p * 1.45, false, false, 9); inc.opacity = clamp01(p * 1.4); break;
    case "squeezeh": out.sy = 1 - s; break;
    case "zoomin": inc.opacity = p; inc.sx = inc.sy = 0.4 + s * 0.6; break;
    default: inc.opacity = p;
  }
  return result;
}

/** A solid colour over the whole frame (a path layer), for dips and fades. */
function solidLayer(id: string, color: string, opacity: number, width: number, height: number): PhotoLayer {
  return {
    ...baseLayer(id, "Dip"),
    opacity: clamp01(opacity),
    path: { ...NEUTRAL_PATH, fill: { kind: "solid", color }, stroke: undefined, strokeWidth: 0, subpaths: [rectPathPx(width, height)] },
  };
}

function rectPathPx(width: number, height: number): Subpath {
  return rectPath(-width / 2, -height / 2, width / 2, height / 2);
}

function baseLayer(id: string, name: string): PhotoLayer {
  return {
    id, assetId: "", sourcePath: "", name, x: 0.5, y: 0.5, scale: 1, rotation: 0,
    cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0, opacity: 1, blendMode: "normal",
    visible: true, locked: false, nodes: [], masks: [], edge: { ...NEUTRAL_EDGE },
  };
}

/**
 * A mask of vector subpaths (normalised canvas), all intersected, as the layer mask -- after the
 * clip's background-removal matte when it has one (so a transition's wipe cuts the cutout).
 */
function vectorMask(id: string, groups: Subpath[][], matte?: MaskEntry): PhotoMask {
  const stack: MaskEntry[] = matte ? [matte] : [];
  groups.forEach((subpaths, index) => stack.push({
    id: `${id}:e${index}`, enabled: true, strokes: [], mode: stack.length === 0 ? "add" : "intersect", opacity: 1, feather: 0, invert: false, expand: 0,
    source: { kind: "path", path: { ...NEUTRAL_PATH, fill: { kind: "solid", color: "#ffffff" }, stroke: undefined, strokeWidth: 0, subpaths } },
  }));
  return { id, name: "Programme mask", strokes: [], invert: false, feather: 0, enabled: true, stack };
}

/**
 * A clip's matte as a mask entry for source frame `sourceFrame` (the frame is explicit, so it holds
 * even where a rendered VFX file stands in for the clip's source). Feather and expand are in
 * sequence pixels; `pixelScale` takes them to the programme's canvas.
 */
export function matteEntry(id: string, matte: ProgrammeMatte, sourceFrame: number, pixelScale: number): MaskEntry {
  return {
    id: `${id}:matte`, enabled: true, strokes: [], mode: "add", opacity: 1,
    feather: matte.feather * pixelScale, expand: matte.expand * pixelScale, invert: matte.invert,
    source: { kind: "sequence", cachePath: matte.cachePath, frameCount: matte.frameCount, first: matte.first, frame: sourceFrame },
  };
}

/** The source frame on screen at `sourceSeconds` (nearest frame, clamped to the file). */
export function sourceFrameIndex(media: ProgrammeMedia, sourceSeconds: number): number {
  return Math.max(0, Math.min(media.frameCount - 1, Math.floor(Math.max(0, sourceSeconds) * media.fps + 0.5 - 1e-9)));
}

/** A layer showing one exact source frame of `media` at composition frame `frame`. */
function mediaAt(media: ProgrammeMedia, index: number, frame: number): Partial<PhotoLayer> {
  if (media.kind === "image") return { assetId: media.assetId, sourcePath: media.path };
  const layerMedia: LayerMedia = {
    kind: "video", path: media.path, frameCount: media.frameCount, fps: media.fps, width: media.width, height: media.height,
    start: frame, inPoint: index, outPoint: index + 1, loop: false, crossfade: 0, proxyPath: media.proxyPath,
  };
  return { assetId: media.assetId, media: layerMedia };
}

const lerpRect = (a: SourceRect, b: SourceRect, u: number): SourceRect => ({
  x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u, width: a.width + (b.width - a.width) * u, height: a.height + (b.height - a.height) * u,
});

function saneRect(rect: SourceRect): SourceRect {
  const width = Math.min(1, Math.max(0.01, rect.width)), height = Math.min(1, Math.max(0.01, rect.height));
  return { x: Math.min(1 - width, Math.max(0, rect.x)), y: Math.min(1 - height, Math.max(0, rect.y)), width, height };
}

/** Programme-clip effects plus a transition's per-pixel treatment, as node params (or none). */
function effectsNode(id: string, effects: ClipEffects, extra: Partial<ClipEffectsParams> | undefined, seed: number): PhotoNode[] {
  const params: ClipEffectsParams = { ...effects, ...(extra ?? {}), seed };
  if (extra?.blur) params.blur = Math.max(effects.blur, extra.blur);
  return isNeutralClipEffects(params) ? [] : [{ id, kind: "clipEffects", enabled: true, effects: params }];
}

export type ProgrammeHooks = {
  /** Nodes and masks for a clip's live mask VFX at `local` clip seconds (see maskFx). */
  vfx?: (clip: { id: string; vfx: ClipVfxLayer[]; media: ProgrammeMedia; sourceIn: number; rate: number }, local: number, frame: number, fps: number) =>
    { nodes: PhotoNode[]; masks: PhotoMask[]; sourcePath?: string; sourceSeconds?: number } | undefined;
};

/**
 * The compositor document for programme frame `frame` at the programme's size. Deterministic in
 * (programme, frame).
 */
export function programmeDocument(programme: Programme, frame: number, hooks: ProgrammeHooks = {}): PhotoDocument {
  const { width: W, height: H, fps } = programme;
  const t = frame / fps;
  const layers: PhotoLayer[] = [];
  const aspect = W / Math.max(1, H);

  const clipLayers = (clip: ProgrammeClip, treat: ClipTreatment, role: string): PhotoLayer[] => {
    const local = t - clip.start;
    let media = clip.media;
    let sourceSeconds = clip.sourceIn + local * clip.rate;
    const matteFrame = sourceFrameIndex(media, sourceSeconds);
    const vfx = hooks.vfx?.(clip, local, frame, fps);
    if (vfx?.sourcePath) { media = { ...media, path: vfx.sourcePath, proxyPath: undefined }; sourceSeconds = vfx.sourceSeconds ?? sourceSeconds; }
    const index = sourceFrameIndex(media, sourceSeconds);
    const conform = clip.conform;
    const u = clamp01(local / Math.max(1 / 1000, clip.duration));
    const crop = conform.crop ? saneRect(conform.crop.end ? lerpRect(saneRect(conform.crop.start), saneRect(conform.crop.end), u) : conform.crop.start) : { x: 0, y: 0, width: 1, height: 1 };
    const cw = media.width * crop.width, ch = media.height * crop.height;
    // Conform into the frame (canvas pixels per source pixel), then the clip's own zoom.
    let kx: number, ky: number;
    if (conform.fit === "stretch") { kx = W / cw; ky = H / ch; }
    else if (conform.fit === "cover") { kx = ky = Math.max(W / cw, H / ch); }
    else if (conform.fit === "none") { kx = ky = programme.pixelScale; }
    else { kx = ky = Math.min(W / cw, H / ch); }
    const zoom = clip.scale / 100;
    // Transition scale is about the frame centre; the clip's own position is about its centre.
    const cx = 0.5 + (clip.x - 0.5) * treat.sx + treat.dx, cy = 0.5 + (clip.y - 0.5) * treat.sy + treat.dy;
    const place = {
      x: cx, y: cy, rotation: clip.rotation,
      cropLeft: crop.x * 100, cropTop: crop.y * 100,
      cropRight: (1 - crop.x - crop.width) * 100, cropBottom: (1 - crop.y - crop.height) * 100,
    };
    const opacity = clamp01(clip.opacity * treat.opacity);
    const masks: PhotoMask[] = [];
    const matte = clip.matte ? matteEntry(`${role}:${clip.id}`, clip.matte, matteFrame, programme.pixelScale) : undefined;
    if (treat.mask || matte) masks.push(vectorMask(`${role}:${clip.id}:mask`, treat.mask ? [treat.mask] : [], matte));
    const maskId = masks[0]?.id;
    const effectNodes = effectsNode(`${role}:${clip.id}:fx`, clip.effects, treat.effects, frame);
    const out: PhotoLayer[] = [];
    const frameScaleX = zoom * treat.sx, frameScaleY = zoom * treat.sy;
    // Padding of a contained picture is part of the clip (it moves with its transform).
    // "black" pads too: the old export padded to an opaque frame, which shows over the other clip
    // of a slide or cover transition and when the clip is zoomed out over an overlay-free frame.
    // Only where the contained picture leaves bars: a frame-filling one needs no pad.
    // A cutout has no frame to pad: what is under it shows through.
    const bars = !clip.matte && (cw * kx < W - 0.5 || ch * ky < H - 0.5);
    if (conform.fit === "contain" && conform.pad !== "blur" && bars) {
      out.push({
        ...baseLayer(`${role}:${clip.id}:pad`, `${clip.name} pad`), ...place, cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
        scale: frameScaleY, stretchX: frameScaleX / Math.max(1e-6, frameScaleY), opacity, masks, maskId,
        path: { ...NEUTRAL_PATH, fill: { kind: "solid", color: conform.pad === "color" ? conform.padColor ?? "#000000" : "#000000" }, stroke: undefined, strokeWidth: 0, subpaths: [rectPathPx(W, H)] },
      });
    }
    if (conform.fit === "contain" && conform.pad === "blur" && bars) {
      const coverK = Math.max(W / cw, H / ch);
      // Cropped to the clip's frame rectangle (the padded frame), wherever the clip transform puts it.
      const frameRect = maskForFrame(`${role}:${clip.id}:padmask`, cx, cy, frameScaleX, frameScaleY, clip.rotation, aspect, treat.mask);
      out.push({
        ...baseLayer(`${role}:${clip.id}:backdrop`, `${clip.name} backdrop`), ...place, ...mediaAt(media, index, frame),
        scale: coverK * frameScaleY, stretchX: frameScaleX / Math.max(1e-6, frameScaleY), opacity,
        nodes: effectsNode(`${role}:${clip.id}:bdfx`, { ...clip.effects, blur: Math.min(200, conform.padBlur ?? 40) }, treat.effects, frame),
        masks: [frameRect], maskId: frameRect.id,
      });
    }
    out.push({
      ...baseLayer(`${role}:${clip.id}`, clip.name), ...place, ...mediaAt(media, index, frame),
      scale: ky * frameScaleY, stretchX: (kx * frameScaleX) / Math.max(1e-6, ky * frameScaleY),
      opacity, blendMode: photoBlend(clip.blend),
      nodes: [...(vfx?.nodes ?? []), ...effectNodes],
      masks: [...masks, ...(vfx?.masks ?? [])], maskId,
    });
    return out;
  };

  // V1 in stacking order: clip i at 2i, the dip of the junction into clip i at 2i + 1.5 (over both
  // clips), and an outgoing clip that closes or slides off over its successor at 2i + 3.
  const stack: { z: number; layers: PhotoLayer[] }[] = [];
  const clips = programme.clips;
  clips.forEach((clip, index) => {
    const end = clip.start + clip.duration;
    const previous = clips[index - 1];
    const next = clips[index + 1];
    const inWindow = clip.incoming && previous
      ? { from: previous.start + previous.duration - clip.incoming.duration, to: previous.start + previous.duration, kind: clip.incoming.kind }
      : undefined;
    const outWindow = next?.incoming ? { from: end - next.incoming.duration, to: end, kind: next.incoming.kind } : undefined;
    const drawFrom = inWindow ? Math.min(clip.start, inWindow.from) : clip.start;
    if (t < drawFrom - 1e-9 || t >= end - 1e-9) return;
    let treat = plain();
    let z = index * 2;
    if (inWindow && t >= inWindow.from - 1e-9 && t < inWindow.to - 1e-9) {
      const junction = transitionFrame(inWindow.kind, (t - inWindow.from) / (inWindow.to - inWindow.from), aspect);
      treat = junction.incoming;
      if (junction.dip && junction.dip.opacity > 0) stack.push({ z: index * 2 + 1.5, layers: [solidLayer(`dip:${clip.id}`, junction.dip.color, junction.dip.opacity, W, H)] });
    } else if (outWindow && t >= outWindow.from - 1e-9) {
      const junction = transitionFrame(outWindow.kind, (t - outWindow.from) / (outWindow.to - outWindow.from), aspect);
      treat = junction.outgoing;
      if (junction.outgoingOnTop) z = index * 2 + 3;
    }
    if (treat.opacity <= 0) return;
    stack.push({ z, layers: clipLayers(clip, treat, "v1") });
  });
  // A dip whose incoming clip is not drawn yet (it starts later than the window: no handle).
  clips.forEach((clip, index) => {
    const previous = clips[index - 1];
    if (!clip.incoming || !previous) return;
    const from = previous.start + previous.duration - clip.incoming.duration, to = previous.start + previous.duration;
    if (t < from - 1e-9 || t >= to - 1e-9) return;
    if (stack.some((entry) => entry.layers.some((layer) => layer.id === `dip:${clip.id}`))) return;
    const junction = transitionFrame(clip.incoming.kind, (t - from) / (to - from), aspect);
    if (junction.dip && junction.dip.opacity > 0) stack.push({ z: index * 2 + 1.5, layers: [solidLayer(`dip:${clip.id}`, junction.dip.color, junction.dip.opacity, W, H)] });
  });
  stack.sort((a, b) => a.z - b.z);
  for (const entry of stack) layers.push(...entry.layers);

  // The programme's head and tail dips.
  if (programme.headFade && t < programme.headFade.duration) {
    layers.push(solidLayer("fade:head", programme.headFade.color, 1 - t / programme.headFade.duration, W, H));
  }
  if (programme.tailFade && t > programme.duration - programme.tailFade.duration) {
    layers.push(solidLayer("fade:tail", programme.tailFade.color, (t - (programme.duration - programme.tailFade.duration)) / programme.tailFade.duration, W, H));
  }

  for (const overlay of programme.overlays) {
    const local = t - overlay.start;
    if (local < -1e-9 || local >= overlay.duration - 1e-9) continue;
    layers.push(...overlayLayers(overlay, local, frame, programme, hooks));
  }

  for (const graphic of programme.graphics) {
    const local = t - graphic.start;
    if (local < -1e-9 || local >= graphic.duration - 1e-9) continue;
    layers.push({
      ...baseLayer(`g:${graphic.id}`, "Graphic"), opacity: clamp01(graphic.opacity),
      timelineGraphic: { overlay: graphic.overlay, local, duration: graphic.duration, canvasWidth: W, canvasHeight: H },
    });
  }

  return {
    assetId: "programme", sourcePath: "", canvasWidth: W, canvasHeight: H, background: "black",
    layers, nodes: [], masks: [], frameRate: fps, duration: Math.max(1, Math.round(programme.duration * fps)),
    rotate: 0, flipHorizontal: false, flipVertical: false, updatedAt: "",
  } as PhotoDocument;
}

/** A mask of the clip's (transformed) frame rectangle, for the blur backdrop; intersected with `extra`. */
function maskForFrame(id: string, cx: number, cy: number, fw: number, fh: number, rotation: number, aspect: number, extra?: Subpath[]): PhotoMask {
  const a = (rotation * Math.PI) / 180, cos = Math.cos(a), sin = Math.sin(a);
  // Corners in canvas pixels relative to the centre (aspect-correct rotation), back to normalised.
  const corners = [[-1, -1], [1, -1], [1, 1], [-1, 1]].map(([sx, sy]) => {
    const px = (sx * fw) / 2 * aspect, py = (sy * fh) / 2;
    return { x: cx + (px * cos - py * sin) / aspect, y: cy + (px * sin + py * cos) };
  });
  const groups: Subpath[][] = [[{ closed: true, nodes: corners }]];
  if (extra) groups.push(extra);
  return vectorMask(id, groups);
}

function overlayLayers(overlay: ProgrammeOverlay, local: number, frame: number, programme: Programme, hooks: ProgrammeHooks): PhotoLayer[] {
  const { width: W, height: H } = programme;
  let media = overlay.media;
  let sourceSeconds = overlay.sourceIn + local * overlay.rate;
  const matteFrame = sourceFrameIndex(media, sourceSeconds);
  const vfx = hooks.vfx?.(overlay, local, frame, programme.fps);
  if (vfx?.sourcePath) { media = { ...media, path: vfx.sourcePath, proxyPath: undefined }; sourceSeconds = vfx.sourceSeconds ?? sourceSeconds; }
  const index = sourceFrameIndex(media, sourceSeconds);
  const box = overlay.box;
  const bw = Math.max(0.01, Math.min(1, box.width)) * W, bh = Math.max(0.01, Math.min(1, box.height)) * H;
  const bx = Math.max(0, Math.min(1, box.x)) * W, by = Math.max(0, Math.min(1, box.y)) * H;
  let kx: number, ky: number;
  if (box.fit === "stretch") { kx = bw / media.width; ky = bh / media.height; }
  else if (box.fit === "cover") { kx = ky = Math.max(bw / media.width, bh / media.height); }
  else { kx = ky = Math.min(bw / media.width, bh / media.height); }
  const radius = Math.max(0, box.cornerRadius) * Math.min(bw, bh);
  const clip = roundedRectPath(bx / W, by / H, (bx + bw) / W, (by + bh) / H, radius / W, radius / H);
  const mask = vectorMask(`ov:${overlay.id}:box`, [[clip]],
    overlay.matte ? matteEntry(`ov:${overlay.id}`, overlay.matte, matteFrame, programme.pixelScale) : undefined);
  const out: PhotoLayer[] = [{
    ...baseLayer(`ov:${overlay.id}`, "Overlay"), ...mediaAt(media, index, frame),
    x: (bx + bw / 2) / W, y: (by + bh / 2) / H, scale: ky, stretchX: kx / ky, opacity: clamp01(overlay.opacity),
    nodes: [...(vfx?.nodes ?? []), ...effectsNode(`ov:${overlay.id}:fx`, overlay.effects, undefined, frame)],
    masks: [mask, ...(vfx?.masks ?? [])], maskId: mask.id,
  }];
  const border = Math.round(Math.max(0, Math.min(0.1, box.borderWidth)) * W);
  if (border > 0) {
    // The border is drawn inside the box, as `drawbox` drew it: a stroke centred half a width in.
    const inset = border / 2;
    const path: PathBlock = {
      ...NEUTRAL_PATH, fill: undefined, stroke: { kind: "solid", color: box.borderColor || "#00f0ff" }, strokeWidth: border, join: "miter",
      subpaths: [roundedRectPath(-bw / 2 + inset, -bh / 2 + inset, bw / 2 - inset, bh / 2 - inset, Math.max(0, radius - inset), Math.max(0, radius - inset))],
    };
    out.push({ ...baseLayer(`ov:${overlay.id}:border`, "Border"), x: (bx + bw / 2) / W, y: (by + bh / 2) / H, opacity: clamp01(overlay.opacity), path });
  }
  return out;
}

/**
 * A point on the programme picture (normalised to the frame) in clip `clipId`'s SOURCE: position
 * normalised to the source picture and the source frame on screen. Undefined when the clip is not
 * drawn at `frame` or the point is off its picture. Goes through the layer exactly as drawn, so
 * conform, crop, the clip's transform and a transition's offset are all undone.
 */
export function programmePick(programme: Programme, frame: number, clipId: string, point: [number, number], hooks: ProgrammeHooks = {}):
  { x: number; y: number; frame: number } | undefined {
  const clip: ProgrammeClip | ProgrammeOverlay | undefined = programme.clips.find((item) => item.id === clipId)
    ?? programme.overlays.find((item) => item.id === clipId);
  if (!clip) return undefined;
  const document = programmeDocument(programme, frame, hooks);
  const layer = (document.layers ?? []).find((item) => item.id === `v1:${clipId}` || item.id === `ov:${clipId}`);
  if (!layer) return undefined;
  const size = { width: clip.media.width, height: clip.media.height };
  const rect = croppedRect(layer, size);
  const [lx, ly] = applyMatrix(invert(layerMatrix(layer, programme.width, programme.height, size, 1)), point[0] * programme.width, point[1] * programme.height);
  const x = lx + rect.left, y = ly + rect.top;
  if (x < rect.left - 0.5 || y < rect.top - 0.5 || x > rect.left + rect.width + 0.5 || y > rect.top + rect.height + 0.5) return undefined;
  const local = frame / programme.fps - clip.start;
  return {
    x: clamp01(x / size.width), y: clamp01(y / size.height),
    frame: clip.media.kind === "image" ? 0 : sourceFrameIndex(clip.media, clip.sourceIn + local * clip.rate),
  };
}

/** The source frame clip `clipId` shows at programme frame `frame`, or undefined when it is not drawn there. */
export function programmeClipFrame(programme: Programme, frame: number, clipId: string, hooks: ProgrammeHooks = {}): number | undefined {
  const clip: ProgrammeClip | ProgrammeOverlay | undefined = programme.clips.find((item) => item.id === clipId)
    ?? programme.overlays.find((item) => item.id === clipId);
  if (!clip) return undefined;
  const drawn = (programmeDocument(programme, frame, hooks).layers ?? []).some((item) => item.id === `v1:${clipId}` || item.id === `ov:${clipId}`);
  if (!drawn) return undefined;
  return clip.media.kind === "image" ? 0 : sourceFrameIndex(clip.media, clip.sourceIn + (frame / programme.fps - clip.start) * clip.rate);
}

/** The inverse of `programmePick`: a source point of the clip on the programme picture (normalised). */
export function programmePlace(programme: Programme, frame: number, clipId: string, point: { x: number; y: number }, hooks: ProgrammeHooks = {}): [number, number] | undefined {
  const clip: ProgrammeClip | ProgrammeOverlay | undefined = programme.clips.find((item) => item.id === clipId)
    ?? programme.overlays.find((item) => item.id === clipId);
  if (!clip) return undefined;
  const layer = (programmeDocument(programme, frame, hooks).layers ?? []).find((item) => item.id === `v1:${clipId}` || item.id === `ov:${clipId}`);
  if (!layer) return undefined;
  const size = { width: clip.media.width, height: clip.media.height };
  const rect = croppedRect(layer, size);
  const [x, y] = applyMatrix(layerMatrix(layer, programme.width, programme.height, size, 1), point.x * size.width - rect.left, point.y * size.height - rect.top);
  return [x / programme.width, y / programme.height];
}

/** Every programme frame index, 0..n-1. */
export const programmeFrames = (programme: Programme): number => Math.max(0, Math.round(programme.duration * programme.fps));
