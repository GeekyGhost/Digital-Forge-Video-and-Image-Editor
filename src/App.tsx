import {
  AudioLines, Bot, Captions, ChevronDown, Download, Film, Gauge, Layers3,
  Magnet, MousePointer2, Pause, Play, Plus, Save, Scissors,
  SkipBack, SkipForward, SlidersHorizontal, Sparkles, WandSparkles,
  ZoomIn, ZoomOut, Eye, EyeOff, Lock, LockOpen, Undo2, Redo2, Palette, Type, Settings2,
  Blend, Clapperboard, X,
} from "lucide-react";
import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  useEffect, useLayoutEffect, useMemo, useRef, useState, type DragEvent as ReactDragEvent,
  type CSSProperties, type MouseEvent as ReactMouseEvent, type PointerEvent as ReactPointerEvent,
} from "react";
import { useEditor, type CaptionCue, type TimelineTool } from "./store";
import { EFFECT_PRESETS, effectsFor } from "./domain/effects";
import { RenameField } from "./features/media/RenameField";
import { PhotoEdit } from "./features/photoEdit/PhotoEdit";
import { mapToProgramme, resolveJunctionLayout } from "./domain/junctions";
import { visibleTimelineTicks } from "./domain/timelineScale";
import {
  boundedMediaTime, canDecodeWholeAssetForScrub, timelineFollowScroll,
} from "./domain/playback";
import { suggestedFileName } from "./domain/naming";
import {
  ASSIST_EVERY, CLIP_MATTE_METHODS, defaultClipMatte, matteBlocker, matteIsShort, matteIsStale, matteRange, usesSam2,
  type ClipMatte, type ClipMatteMethod,
} from "./domain/clipMatte";
import { makeClipMatte } from "./services/clipMatteService";
import {
  ADJACENT_TOLERANCE, importedAssetKind, isAudioAsset, isImagePath, isStillAsset, planBridgePlacement,
  resolveBridgePair, resolveExtendClip, resolveFramePickerIds, resolvePickerId,
} from "./domain/media";
import { clipDuration, clipPlaybackRate, formatTimecode, fromSeconds, seconds, timelineClipDuration } from "./domain/time";
import type {
  AudioCodec, BlendMode, ExportContainer, ExportSettings, GraphicOverlay, GraphicTemplate, MediaAsset,
  OverlayPlacement, Project, QualityMode, Sequence, TextBackdrop, TextFontFamily, TextMotionEasing,
  TextMotionIn, TextMotionLoop, TextMotionOut, TimelineClip, VideoEncoder,
} from "./domain/types";
import {
  CONTAINER_SUPPORT, DEFAULT_CONFORM, DEFAULT_EXPORT_SETTINGS, ENCODER_PRESETS, FULL_SOURCE_RECT,
  defaultPresetFor,
} from "./domain/types";
import type { ClipConform, ConformFit, ConformPad } from "./domain/types";
import {
  AI_UPSCALE_MODELS, ENCODER_CEILING_NOTE, ENCODER_MAX_DIMENSION, NVIDIA_UPSCALE_MODEL_ID,
  RESAMPLE_ALGOS, UPSCALE_TYPES, aiUpscaleModel, resolveUpscalePlan, upscaleFactorsFor,
  upscaleRequest, type ResampleAlgo, type UpscaleType,
} from "./domain/upscale";

const CONFORM_FIT_OPTIONS = [
  { value: "contain", label: "Fit · whole picture, may add bars" },
  { value: "cover", label: "Fill · crops the edges, no bars" },
  { value: "stretch", label: "Stretch · distorts to fill" },
  { value: "none", label: "None · leave at source size" },
];

const CONFORM_PAD_OPTIONS = [
  { value: "black", label: "Black bars" },
  { value: "blur", label: "Blurred copy · no bars, nothing cropped" },
  { value: "color", label: "Solid colour" },
];

const VIDEO_ENCODER_LABELS: Record<VideoEncoder, string> = {
  h264_nvenc: "H.264 · NVENC (GPU)",
  hevc_nvenc: "H.265 / HEVC · NVENC (GPU)",
  libx264: "H.264 · x264 (CPU)",
  libx265: "H.265 / HEVC · x265 (CPU)",
};

const AUDIO_CODEC_LABELS: Record<AudioCodec, string> = {
  aac: "AAC",
  libopus: "Opus",
  flac: "FLAC · lossless",
  pcm_s16le: "PCM 16-bit · uncompressed",
  none: "No audio",
};

// ---------------------------------------------------------------------------
// Bidirectional timeline scrub audio
// ---------------------------------------------------------------------------

type ScrubPlayable = {
  asset: MediaAsset;
  clip: TimelineClip;
  programStart: number;
  gainDb: number;
};

let scrubAudioContext: AudioContext | undefined;
const scrubAudioBuffers = new Map<string, Promise<AudioBuffer | null>>();
const SCRUB_AUDIO_CACHE_LIMIT = 3;
const SCRUB_SOURCE_BYTE_LIMIT = 96 * 1024 * 1024;
let scrubAudioRequest = 0;
let activeScrubSources: AudioBufferSourceNode[] = [];

function scrubContext(): AudioContext {
  scrubAudioContext ??= new AudioContext({ latencyHint: "interactive" });
  return scrubAudioContext;
}

function decodeScrubAudio(asset: MediaAsset): Promise<AudioBuffer | null> {
  if (!canDecodeWholeAssetForScrub(seconds(asset.duration), asset.kind === "audio" ? "audio" : "video")) {
    return Promise.resolve(null);
  }
  const cached = scrubAudioBuffers.get(asset.path);
  if (cached) {
    // Refresh insertion order so the small cache behaves like an LRU.
    scrubAudioBuffers.delete(asset.path);
    scrubAudioBuffers.set(asset.path, cached);
    return cached;
  }
  const pending = (async () => {
    try {
      const response = await fetch(convertFileSrc(asset.path));
      if (!response.ok) return null;
      const contentLength = Number(response.headers.get("content-length") ?? 0);
      if (contentLength > SCRUB_SOURCE_BYTE_LIMIT) return null;
      return await scrubContext().decodeAudioData(await response.arrayBuffer());
    } catch {
      // Some Windows codecs can be played by HTMLMediaElement but cannot be decoded by Web Audio.
      // The picture still scrubs correctly; the next decodable timeline layer remains audible.
      return null;
    }
  })();
  scrubAudioBuffers.set(asset.path, pending);
  while (scrubAudioBuffers.size > SCRUB_AUDIO_CACHE_LIMIT) {
    const oldest = scrubAudioBuffers.keys().next().value;
    if (oldest) scrubAudioBuffers.delete(oldest); else break;
  }
  return pending;
}

function scrubPlayablesAt(project: Project, sequence: Sequence, at: number): ScrubPlayable[] {
  const primary = sequence.tracks.find((track) => track.id === "v1");
  const primaryClips = (primary?.hidden ? [] : primary?.clips ?? [])
    .filter((clip) => clip.enabled).sort((a, b) => seconds(a.start) - seconds(b.start));
  const layout = resolveJunctionLayout(primaryClips);
  const result: ScrubPlayable[] = [];

  for (const track of sequence.tracks) {
    if (track.hidden || track.muted || !["video", "audio"].includes(track.kind)) continue;
    for (const clip of track.clips) {
      if (!clip.enabled) continue;
      const asset = project.assets.find((item) => item.id === clip.assetId);
      if (!asset?.path || asset.kind === "image" || (track.kind === "video" && asset.hasAudio === false)) continue;
      const programStart = track.id === "v1"
        ? layout.start.get(clip.id) ?? seconds(clip.start)
        : track.kind === "audio"
          ? mapToProgramme(layout, primaryClips, seconds(clip.start))
          : seconds(clip.start);
      if (at < programStart || at >= programStart + timelineClipDuration(clip)) continue;
      result.push({ asset, clip, programStart, gainDb: clip.volumeDb + (track.gainDb ?? 0) });
    }
  }
  return result;
}

function stopScrubSources(): void {
  for (const source of activeScrubSources) {
    try { source.stop(); } catch { /* already stopped */ }
  }
  activeScrubSources = [];
}

async function playTimelineScrubAudio(project: Project, sequence: Sequence, from: number, to: number): Promise<void> {
  const delta = to - from;
  if (Math.abs(delta) < 0.002) return;
  const direction = delta > 0 ? 1 : -1;
  const grainTimelineSeconds = Math.min(0.24, Math.max(0.045, Math.abs(delta)));
  const playables = scrubPlayablesAt(project, sequence, to);
  if (!playables.length) return;
  const request = ++scrubAudioRequest;
  const decoded = await Promise.all(playables.map(async (item) => ({ item, buffer: await decodeScrubAudio(item.asset) })));
  if (request !== scrubAudioRequest) return;

  const context = scrubContext();
  if (context.state === "suspended") await context.resume();
  stopScrubSources();
  const now = context.currentTime + 0.006;

  for (const { item, buffer } of decoded) {
    if (!buffer) continue;
    const rate = clipPlaybackRate(item.clip);
    const sourceAt = seconds(item.clip.sourceIn) + Math.max(0, to - item.programStart) * rate;
    const sourceSpan = grainTimelineSeconds * rate;
    const low = Math.max(seconds(item.clip.sourceIn), direction > 0 ? sourceAt - sourceSpan : sourceAt);
    const high = Math.min(seconds(item.clip.sourceOut), direction > 0 ? sourceAt : sourceAt + sourceSpan);
    const startFrame = Math.max(0, Math.floor(low * buffer.sampleRate));
    const endFrame = Math.min(buffer.length, Math.ceil(high * buffer.sampleRate));
    const frameCount = endFrame - startFrame;
    if (frameCount < 32) continue;

    const grain = context.createBuffer(buffer.numberOfChannels, frameCount, buffer.sampleRate);
    for (let channel = 0; channel < buffer.numberOfChannels; channel += 1) {
      const input = buffer.getChannelData(channel);
      const output = grain.getChannelData(channel);
      if (direction > 0) output.set(input.subarray(startFrame, endFrame));
      else for (let index = 0; index < frameCount; index += 1) output[index] = input[endFrame - 1 - index];
    }

    const source = context.createBufferSource();
    const gain = context.createGain();
    const duration = frameCount / buffer.sampleRate / rate;
    const level = Math.max(0, Math.min(4, 10 ** (item.gainDb / 20)));
    source.buffer = grain;
    source.playbackRate.value = rate;
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(level, now + Math.min(0.008, duration / 3));
    gain.gain.setValueAtTime(level, Math.max(now + 0.009, now + duration - 0.012));
    gain.gain.linearRampToValueAtTime(0.0001, now + duration);
    source.connect(gain).connect(context.destination);
    source.start(now);
    activeScrubSources.push(source);
  }
}
import {
  filterTransitions, resolveJunction, TRANSITION_CATEGORIES, transitionById, TRANSITIONS,
  type TransitionDefinition,
} from "./domain/transitions";
import { transitionLayers } from "./domain/transitionPreview";
import { ProgrammeMonitor, readLegacyMonitor, writeLegacyMonitor } from "./features/monitor/ProgrammeMonitor";
import { exportProgramme, readLegacyExport, writeLegacyExport } from "./features/export/programmeExport";
import { clipFilter, cssBlendMode } from "./features/effects/presentation";
import { planKiwiRange, type KiwiStrategy } from "./features/generation/kiwi";
import { fitLtxDimensions } from "./features/generation/ltx";
import { clampH3Frames, fitH3Dimensions, H3_ASPECTS, H3_FPS, H3_LOW_VRAM_SHORT_EDGE, H3_MAX_FRAMES, H3_MIN_FRAMES, H3_TRAINED_SHORT_EDGE, h3Seconds, buildH3RefRequest, H3_REF_DEFAULTS, H3_REF_IMAGE_DETAILS, type H3RefImageDetail } from "./features/generation/h3";
import {
  availableH3ReferenceIntents, buildH3ReferencePrompt, defaultH3ReferenceIntents, H3_REFERENCE_INTENTS, h3ReferenceLimits,
  h3ReferenceCapError, H3_MAX_AUDIOS, H3_MAX_IMAGES, H3_MAX_REFERENCES, H3_MAX_VIDEOS,
  type H3ReferenceDraft, type H3ReferenceIntent, type H3ReferenceKind,
} from "./features/generation/h3Ref";
import {
  LTX_FRAMES_PER_LATENT, LTX_MAX_FRAMES, LTX_MIN_FRAMES, snapLtxFrames, ltxRetakeProblems,
  ltxFrameRange,
} from "./features/generation/ltxGeometry";
import { buildTimelineExportPayload, sliceTimelineExportPayload } from "./features/export/request";
import { ImageStudio } from "./features/imageStudio/ImageStudio";
import { Ref2V } from "./features/ref2v/Ref2V";
import { SfxWorkspace } from "./features/clipEditor/SfxWorkspace";
import { MaskEffectsWorkspace, VisualModeSwitch, type VisualMode } from "./features/clipEditor/MaskEffectsWorkspace";
import { TextMotionWorkspace } from "./features/clipEditor/TextMotionWorkspace";
import { TEXT_MOTION_EASINGS, TEXT_MOTION_IN, TEXT_MOTION_LOOPS, TEXT_MOTION_OUT, textMotionFrame } from "./domain/textMotion";
import {
  generatedAsset, filename, parseFrameRate as parseRate, probeMedia, runtimeStatus,
  type AiBridgeResult as LtxBridgeResult, type AiRuntimeStatus, type LtxRetakeResult,
  type MediaProbe, type MiniMaxH3Result,
} from "./services/modelService";

function graphicColorWithOpacity(color: string, opacity: number) {
  const alpha = Math.round(Math.max(0, Math.min(1, opacity)) * 255).toString(16).padStart(2, "0");
  return color.slice(0, 7) + alpha;
}

function graphicFontStack(family: TextFontFamily | undefined) {
  if (family === "serif") return "Georgia, 'Times New Roman', serif";
  if (family === "mono") return "Consolas, 'Courier New', monospace";
  if (family === "sans") return "Arial, Helvetica, sans-serif";
  return "var(--font-display)";
}

/** Each clip-editor blocker needs a different fix, so name the fix rather than just "missing". */
const KIWI_STATE_COPY: Record<string, { pill: string; detail: string }> = {
  ready: { pill: "Pack ready", detail: "" },
  "pack-missing": {
    pill: "Pack missing",
    detail: "Install the model pack: .venv\\Scripts\\python.exe scripts\\models.py install kiwi-edit --accept-licenses",
  },
  "deps-missing": {
    pill: "Dependencies missing",
    detail: "The pack is installed but its Python libraries are not. Run: .venv\\Scripts\\python.exe -m pip install -e .\\ai-runtime[inference]",
  },
  "torch-cpu": {
    pill: "CPU-only PyTorch",
    detail: "The pack and its libraries are installed, but this environment's PyTorch cannot use the GPU. Relaunch through Launch-AIV-Studio.bat and it will install the CUDA build (~2.5 GB, once).",
  },
  "adapter-missing": {
    pill: "Adapter missing",
    detail: "ai-runtime\\aiv_runtime\\kiwi_edit_16gb.py is missing from this install.",
  },
  "venv-missing": {
    pill: "Python missing",
    detail: "The project virtual environment is missing. Run scripts\\bootstrap.ps1.",
  },
};

/** Turn `minimaxH3State` into a specific installation or runtime action. */
function h3Copy(state?: string): { title: string; detail: string; blocker: boolean } {
  switch (state) {
    case "ready":
      return { title: "", detail: "", blocker: false };
    case "encoder-missing":
      return {
        title: "The published H3 INT8 ConvRot conditioner is missing",
        detail: "models\\MiniMax-H3\\text_encoders\\qwen3vl_32b_minimax_h3_int8_convrot.safetensors "
          + "was not found. Install the H3 pack again. Production does not fall back to local .pt or .bin conversions.",
        blocker: true,
      };
    case "checkpoint-missing":
      return {
        title: "The H3 INT8 ConvRot denoiser is missing",
        detail: "models\\MiniMax-H3\\diffusion_models\\minimax_h3_fl2va_pruned_int8_convrot.safetensors was not found. The BF16 partition "
          + "will not load in this 16 GiB profile.\n\n  .venv\\Scripts\\python.exe scripts\\models.py "
          + "install minimax-h3 --accept-licenses",
        blocker: true,
      };
    case "pack-missing":
      return {
        title: "The MiniMax-H3 weights are not installed",
        detail: "The production pack uses published safetensors for both the denoiser and conditioner."
          + "\n\n  .venv\\Scripts\\python.exe scripts\\models.py install minimax-h3 --accept-licenses",
        blocker: true,
      };
    case "adapter-missing":
      return {
        title: "The MiniMax-H3 adapter is missing",
        detail: "ai-runtime\\aiv_runtime\\minimax_h3.py was not found.",
        blocker: true,
      };
    default:
      return {
        title: "The MiniMax-H3 environment has not been built",
        detail: "H3 gets its own virtualenv so it cannot disturb the LTX one.\n\n  powershell "
          + "-NoProfile -ExecutionPolicy Bypass -File .\\scripts\\setup-h3.ps1",
        blocker: true,
      };
  }
}

const kiwiCopy = (state: string | undefined) => KIWI_STATE_COPY[state ?? ""] ?? KIWI_STATE_COPY["pack-missing"];

type DetectedRange = { startSeconds: number; endSeconds: number; durationSeconds: number; kind: string };
type PauseAnalysisResult = { freezes: DetectedRange[]; silences: DetectedRange[]; candidates: DetectedRange[] };
type TranscriptCue = { startSeconds: number; endSeconds: number; text: string };
type DenoiseResult = { outputPath: string; filter: string; videoCopied: boolean };

const GRAPHIC_TEMPLATES: Array<{ id: GraphicTemplate; name: string; description: string }> = [
  { id: "title", name: "Centered title", description: "Large centered program, chapter, or interview title." },
  { id: "lower-third", name: "Lower third", description: "Name, role, location, or speaker identification." },
  { id: "banner", name: "News banner", description: "Bottom headline band with supporting text." },
  { id: "bug", name: "Corner bug", description: "Small LIVE, network, sponsor, or segment label." },
  { id: "caption", name: "Caption", description: "Readable timed subtitle placed on the dedicated C1 track." },
];

const BLEND_MODES: Array<{ value: BlendMode; label: string }> = [
  { value: "normal", label: "Normal" },
  { value: "screen", label: "Screen" },
  { value: "multiply", label: "Multiply" },
  { value: "overlay", label: "Overlay" },
  { value: "darken", label: "Darken" },
  { value: "lighten", label: "Lighten" },
  { value: "difference", label: "Difference" },
  { value: "hardlight", label: "Hard light" },
];

const Button = ({ children, active, title, onClick, disabled = false }: {
  children: React.ReactNode; active?: boolean; title: string; onClick?: () => void; disabled?: boolean;
}) => <button className={"icon-button " + (active ? "active" : "")} title={title} onClick={onClick} disabled={disabled}>{children}</button>;

const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);

/** Worker failures arrive as several lines. The status bar gets one; the panel keeps the rest. */
const errorHeadline = (error: unknown, limit = 160) => {
  const lines = errorText(error).split("\n").map((line) => line.trim()).filter(Boolean);
  const first = lines.at(-1) ?? lines[0] ?? "unknown error";
  return first.length > limit ? first.slice(0, limit - 1) + "…" : first;
};
const subtitleTimeSeconds = (value: string): number | undefined => {
  const match = value.trim().match(/^(\d{1,3}):(\d{2}):(\d{2})[,.](\d{1,3})/);
  if (!match) return undefined;
  return Number(match[1]) * 3600 + Number(match[2]) * 60 + Number(match[3]) + Number(match[4].padEnd(3, "0").slice(0, 3)) / 1000;
};

const parseSubtitleCues = (content: string): CaptionCue[] => content.replace(/^﻿/, "").replace(/\r/g, "")
  .split(/\n{2,}/).flatMap((block) => {
    const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);
    const timingIndex = lines.findIndex((line) => line.includes("-->"));
    if (timingIndex < 0) return [];
    const [left, rightWithSettings] = lines[timingIndex].split("-->");
    const startSeconds = subtitleTimeSeconds(left);
    const endSeconds = subtitleTimeSeconds(rightWithSettings?.trim().split(/\s+/)[0] ?? "");
    const text = lines.slice(timingIndex + 1).join(" ").replace(/<br\s*\/?>/gi, " ").replace(/<[^>]+>/g, "").trim();
    return startSeconds !== undefined && endSeconds !== undefined && endSeconds > startSeconds && text
      ? [{ startSeconds, endSeconds, text }] : [];
  });

// ---------------------------------------------------------------------------
// File and render operations
// ---------------------------------------------------------------------------

async function importSubtitles() {
  const state = useEditor.getState();
  const selected = await open({ multiple: false, filters: [{ name: "Subtitle files", extensions: ["srt", "vtt"] }] });
  if (typeof selected !== "string") return;
  state.setBusy(true);
  state.setStatus("Reading subtitles…");
  try {
    const content = await invoke<string>("read_subtitle_file", { path: selected });
    state.addCaptionCues(parseSubtitleCues(content));
  } catch (error) {
    state.setStatus("Subtitle import failed: " + errorText(error));
  } finally {
    state.setBusy(false);
  }
}

async function importMedia() {
  const state = useEditor.getState();
  const paths = await open({
    multiple: true,
    directory: false,
    filters: [{ name: "Media", extensions: ["mp4", "mov", "mkv", "webm", "avi", "m4v", "mp3", "wav", "flac", "m4a", "aac", "png", "jpg", "jpeg", "webp"] }],
  });
  if (!paths) return;
  state.setBusy(true);
  state.setStatus("Inspecting media…");
  try {
    const selected = Array.isArray(paths) ? paths : [paths];
    const probes = await Promise.all(selected.map((path) => invoke<MediaProbe>("probe_media", { path })));
    const assets: MediaAsset[] = probes.map((probe) => {
      const image = isImagePath(probe.path);
      return ({
        id: crypto.randomUUID(),
        name: filename(probe.path),
        path: probe.path,
        kind: importedAssetKind(probe.path, Boolean(probe.width)),
        duration: fromSeconds(image ? 5 : Math.max(probe.durationSeconds, 0.04)),
        width: probe.width,
        height: probe.height,
        frameRate: parseRate(probe.frameRate),
        colorSpace: probe.colorSpace,
        hasAudio: Boolean(probe.audioCodec),
      });
    });
    state.addAssets(assets);
  } catch (error) {
    state.setStatus("Import failed: " + errorText(error));
  } finally {
    state.setBusy(false);
  }
}

async function saveProject(saveAs = false) {
  const state = useEditor.getState();
  let path = saveAs ? undefined : state.projectPath;
  if (!path) {
    path = await save({
      defaultPath: state.project.name.replace(/[<>:"/\\|?*]/g, "_") + ".aivproj",
      filters: [{ name: "Digital Forge Project", extensions: ["aivproj"] }],
    }) ?? undefined;
  }
  if (!path) return;
  state.setBusy(true);
  state.setStatus("Saving project…");
  try {
    await invoke<string>("save_project", { path, project: state.project });
    state.setProjectPath(path);
    state.setStatus("Saved " + filename(path));
  } catch (error) {
    state.setStatus("Save failed: " + errorText(error));
  } finally {
    state.setBusy(false);
  }
}

async function openProject() {
  const state = useEditor.getState();
  const path = await open({ multiple: false, filters: [{ name: "Digital Forge Project", extensions: ["aivproj"] }] });
  if (!path || Array.isArray(path)) return;
  state.setBusy(true);
  state.setStatus("Opening project…");
  try {
    const project = await invoke<Project>("load_project", { path });
    state.replaceProject(project, path);
  } catch (error) {
    state.setStatus("Open failed: " + errorText(error));
  } finally {
    state.setBusy(false);
  }
}

/** The export settings as they actually stand, defaults underneath and stored values on top. */
export function resolveExportSettings(project: Project): ExportSettings {
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId);
  // Merged, not substituted. A project saved before a setting existed has no value for it, and
  // taking the stored object wholesale leaves the new keys undefined -- which reaches ffmpeg as
  // `interp_algo=undefined` and kills the render.
  return { ...DEFAULT_EXPORT_SETTINGS, ...(sequence?.exportSettings ?? {}) };
}

type TimelinePreviewMode = "quick" | "full";
type TimelinePreviewResult = { path: string; mode: TimelinePreviewMode; startSeconds: number; durationSeconds: number };
type TimelinePreviewSpec = {
  mode: TimelinePreviewMode;
  startSeconds: number;
  durationSeconds: number;
  onComplete: (result: TimelinePreviewResult) => void;
};

function quickPreviewSettings(settings: ExportSettings, sequence?: Sequence): ExportSettings {
  if (!sequence) return { ...DEFAULT_EXPORT_SETTINGS, upscaleEnabled: false };
  const scale = Math.min(1, 960 / Math.max(1, sequence.width), 540 / Math.max(1, sequence.height));
  const even = (value: number) => Math.max(2, Math.round(value / 2) * 2);
  const fps = sequence.frameRate.numerator / Math.max(1, sequence.frameRate.denominator);
  return {
    ...DEFAULT_EXPORT_SETTINGS,
    container: "mp4", encoder: "h264_nvenc", preset: "p1", qualityMode: "quality", quality: 28,
    width: even(sequence.width * scale), height: even(sequence.height * scale), frameRate: Math.min(30, fps),
    audioCodec: settings.audioCodec === "none" ? "none" : "aac", audioBitrate: "192k",
    audioSampleRate: 48_000, audioChannels: Math.min(2, settings.audioChannels), upscaleEnabled: false,
  };
}

// `settings` has NO default. It used to fall back to DEFAULT_EXPORT_SETTINGS, and the Sequence
// menu's "Render timeline..." called this with no arguments -- so that command rendered at the
// defaults every time, discarding the encoder, the bitrate, the resolution and, most visibly, the
// upscale. `upscaleFactor: 1` in the defaults means no chain is sent at all, which is why the
// upscale appeared to fail silently in every mode: it was never asked for.
async function exportTimeline(settings: ExportSettings, uiSequence?: Sequence, preview?: TimelinePreviewSpec) {
  const state = useEditor.getState();
  const activeSequence = state.project.sequences.find((item) => item.id === state.project.activeSequenceId);
  if (preview?.mode === "quick") settings = quickPreviewSettings(settings, activeSequence);
  // Through `upscaleRequest`, not from the raw fields: the panel asks the question as a toggle plus
  // a type, and this is the one place that turns that into the factor and model id the resolver
  // takes. Reading `settings.upscaleFactor` directly would ignore the toggle entirely.
  const request = upscaleRequest(settings);
  let { sequence, clips, graphics, audioClips, overlayClips } = buildTimelineExportPayload(state.project, request.factor);
  if (preview) {
    ({ sequence, clips, graphics, audioClips, overlayClips } = sliceTimelineExportPayload(
      { sequence, clips, graphics, audioClips, overlayClips, upscaleFactor: request.factor, rippleSeconds: 0 },
      preview.startSeconds, preview.durationSeconds,
    ));
  }
  // Resolve the chain from the size actually being encoded, not from the sequence: an export that
  // overrides width/height is what the upscale multiplies.
  const exportPlan = resolveUpscalePlan({
    source: {
      width: settings.width || sequence?.width || 0,
      height: settings.height || sequence?.height || 0,
    },
    factor: request.factor,
    aiModelId: request.aiModelId,
    resample: settings.upscaleResample,
    maxDimension: ENCODER_MAX_DIMENSION[settings.encoder],
    maxDimensionReason: ENCODER_CEILING_NOTE[settings.encoder],
    medium: "video",
  });
  void uiSequence;
  // Every audio track mixes in, not just A1 — stem separation produces several.

  // Overlay picture tracks composite over the programme. Later tracks in the list draw on top,
  // matching the timeline's own top-to-bottom order.
  if (!sequence || clips.length === 0) {
    // Named precisely: "nothing on V1" and "everything on V1 is disabled" look identical from
    // here but are fixed differently.
    state.setStatus(!sequence
      ? "Render needs an active sequence."
      : "Render needs at least one ENABLED clip on the V1 track. Check that V1 is not hidden and its clips are enabled.");
    return;
  }
  const containerLabels: Record<string, string> = { mp4: "MPEG-4 Video", mov: "QuickTime Movie", mkv: "Matroska Video" };
  // The dialog is awaited outside the render's own try block, so a rejection here used to be an
  // unhandled promise -- silent in a release build, and indistinguishable from a dead button.
  let outputPath: string | null;
  try {
    outputPath = preview
      ? await invoke<string>("timeline_preview_path", { kind: preview.mode, extension: settings.container })
      : await save({
        defaultPath: state.project.name.replace(/[<>:"/\\|?*]/g, "_") + "." + settings.container,
        filters: [{ name: containerLabels[settings.container] ?? "Video", extensions: [settings.container] }],
      });
  } catch (error) {
    state.setStatus(`Could not open the save dialog: ${error instanceof Error ? error.message : String(error)}`);
    return;
  }
  if (!outputPath) { state.setStatus("Render canceled"); return; }
  // The export size wins when it is set; 0 means follow the sequence. Both are forced even so the
  // encoders never see an odd dimension, which yuv420p cannot represent.
  const even = (value: number) => Math.max(2, Math.round(value / 2) * 2);
  const width = even(settings.width || sequence.width);
  const height = even(settings.height || sequence.height);
  const frameRate = settings.frameRate > 0
    ? { numerator: settings.frameRate, denominator: 1 }
    : sequence.frameRate;
  state.setBusy(true);
  state.setStatus(
    `${preview ? preview.mode === "quick" ? "Quick preview" : "Full-quality preview" : "Rendering"} · ${clips.length} clip${clips.length === 1 ? "" : "s"} · ${width}×${height} · ` +
    `${VIDEO_ENCODER_LABELS[settings.encoder]}${exportPlan.stages.length ? ` · ${exportPlan.readout}` : ""}…`,
  );
  try {
    const upscale = exportPlan.stages.length ? {
      aiBackend: exportPlan.stages.find((stage) => stage.kind === "ai")?.model.backend ?? "esrgan",
      aiModelFile: exportPlan.stages.find((stage) => stage.kind === "ai")?.model.file,
      aiQuality: settings.upscaleQuality,
      tile: settings.upscaleTile,
      resample: settings.upscaleResample,
      width: exportPlan.final.width,
      height: exportPlan.final.height,
    } : undefined;
    // VFX Phase 7: the compositor export (what the monitor shows) unless the legacy one is chosen.
    const result = !readLegacyExport()
      ? await exportProgramme(state.project, {
          outputPath, width, height, frameRate, videoCodec: settings.encoder, audioCodec: settings.audioCodec,
          videoBitrate: settings.bitrate, audioBitrate: settings.audioBitrate, qualityMode: settings.qualityMode,
          quality: settings.quality, preset: settings.preset, audioSampleRate: settings.audioSampleRate,
          audioChannels: settings.audioChannels, upscale,
        }, (text) => useEditor.getState().setStatus(text),
        (clipId, layer) => useEditor.getState().upsertClipVfxLayer(clipId, layer), () => useEditor.getState().project,
        preview ? { startSeconds: preview.startSeconds, durationSeconds: preview.durationSeconds } : undefined)
      : await invoke<{ outputPath: string; warnings?: string[] }>("export_timeline", { request: {
      clips, graphics, audioClips, overlayClips, outputPath, width, height, frameRate,
      videoCodec: settings.encoder,
      audioCodec: settings.audioCodec,
      videoBitrate: settings.bitrate,
      audioBitrate: settings.audioBitrate,
      qualityMode: settings.qualityMode,
      quality: settings.quality,
      preset: settings.preset,
      audioSampleRate: settings.audioSampleRate,
      audioChannels: settings.audioChannels,
      upscaleFactor: request.factor,
      upscale,
    } });
    // Surface non-fatal notes rather than swallowing them. A transition that quietly held a frame
    // because the clip had no handle is exactly the kind of thing that goes unnoticed until someone
    // watches the export frame by frame.
    const warnings = result?.warnings ?? [];
    state.setStatus(warnings.length
      ? `Export complete with ${warnings.length} note${warnings.length === 1 ? "" : "s"}: ${warnings[0]}`
      : preview ? `${preview.mode === "quick" ? "Quick" : "Full-quality"} preview ready` : "Export complete: " + outputPath);
    if (warnings.length) console.warn("[export]", warnings);
    if (preview) preview.onComplete({ path: outputPath, mode: preview.mode, startSeconds: preview.startSeconds, durationSeconds: preview.durationSeconds });
  } catch (error) {
    state.setStatus("Export failed: " + errorHeadline(error));
  } finally {
    state.setBusy(false);
  }
}

// ---------------------------------------------------------------------------
// Shared controls
// ---------------------------------------------------------------------------

/**
 * Kind of the asset currently being dragged from the media panel.
 *
 * `dragover` cannot read `dataTransfer.getData()` -- the drag data store is protected until drop --
 * so the only way to know what is in flight is to record it at `dragstart`. This used to test
 * `selectedAsset` instead, i.e. whatever happened to be selected in the panel rather than what the
 * user had hold of, so dragging an unselected audio file never called `preventDefault()` on the
 * audio lane and the browser refused the drop entirely.
 *
 * Module-level rather than React state on purpose: a re-render during a drag cancels it in WebView2.
 */
let draggedAssetKind: MediaAsset["kind"] | undefined;

type MenuItem = { label: string; action: () => void; disabled?: boolean; danger?: boolean; shortcut?: string };
type MenuState = { x: number; y: number; items: MenuItem[] } | null;

function ContextMenu({ menu, close }: { menu: MenuState; close: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  const [placed, setPlaced] = useState<
    { menu: MenuState; left: number; top: number; maxHeight?: number } | null
  >(null);
  useEffect(() => {
    if (!menu) return;
    const dismiss = () => close();
    const escape = (event: KeyboardEvent) => { if (event.key === "Escape") close(); };
    window.addEventListener("pointerdown", dismiss);
    window.addEventListener("keydown", escape);
    return () => {
      window.removeEventListener("pointerdown", dismiss);
      window.removeEventListener("keydown", escape);
    };
  }, [menu, close]);
  // Raw cursor coordinates run the menu off-screen when it opens near an edge -- which is exactly
  // what happens on the lower timeline tracks, where the bottom items became unreachable. Measure
  // the rendered menu, then flip it above the cursor if it does not fit below, clamp it into the
  // viewport if it fits neither way, and scroll it if it is simply taller than the screen.
  useLayoutEffect(() => {
    if (!menu || !ref.current) return;
    const MARGIN = 8;
    const { width, height } = ref.current.getBoundingClientRect();
    const room = window.innerHeight - MARGIN * 2;
    const maxHeight = height > room ? room : undefined;
    const effective = maxHeight ?? height;

    let top = menu.y;
    if (top + effective > window.innerHeight - MARGIN) {
      const flipped = menu.y - effective;
      top = flipped >= MARGIN ? flipped : Math.max(MARGIN, window.innerHeight - effective - MARGIN);
    }
    let left = menu.x;
    if (left + width > window.innerWidth - MARGIN) left = Math.max(MARGIN, menu.x - width);

    setPlaced({ menu, left, top, maxHeight });
  }, [menu]);

  if (!menu) return null;
  // Identity check, not just presence: a second right-click arrives with a fresh MenuState, and
  // reusing the previous placement would flash the menu at the old position for one frame.
  const ready = placed?.menu === menu;
  return <div ref={ref} className="context-menu" onPointerDown={(event) => event.stopPropagation()}
    style={{
      left: ready ? placed.left : menu.x,
      top: ready ? placed.top : menu.y,
      maxHeight: ready ? placed.maxHeight : undefined,
      visibility: ready ? "visible" : "hidden",
    }}>
    {menu.items.map((item) => <button key={item.label} className={item.danger ? "danger" : ""} disabled={item.disabled}
      onClick={() => { item.action(); close(); }}><span>{item.label}</span>{item.shortcut && <kbd>{item.shortcut}</kbd>}</button>)}
  </div>;
}

function MenuDropdown({ label, open, setOpen, items }: {
  label: string; open: boolean; setOpen: () => void; items: MenuItem[];
}) {
  return <div className="menu-root"><button className={open ? "open" : ""} onClick={setOpen}>{label}</button>
    {open && <div className="menu-dropdown">{items.map((item) => <button key={item.label} className={item.danger ? "danger" : ""}
      disabled={item.disabled} onClick={() => { item.action(); setOpen(); }}><span>{item.label}</span>{item.shortcut && <kbd>{item.shortcut}</kbd>}</button>)}</div>}
  </div>;
}

function Section({ title, children, accent, hint }: { title: string; children: React.ReactNode; accent?: string; hint?: string }) {
  return <section className="control-section"><div className="section-title" title={hint}><strong>{title}</strong>{accent ? <b>{accent}</b> : <ChevronDown size={14} />}</div>{children}</section>;
}

function Field({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return <label className="field" title={hint}><span>{label}</span><input value={value} readOnly /></label>;
}

/**
 * A number field that lets you finish typing before it argues with you.
 *
 * The previous version committed `Number(event.target.value)` on every keystroke, which broke in
 * three ways at once:
 *   - consumers that re-derive the value fought the keyboard. Frames snaps to 8n+1, so typing
 *     "121" became 9 (from "1"), then 89 (from "92"), then something else again;
 *   - clamping mid-word is wrong -- "9" on the way to "96" is under a min of 12 only momentarily;
 *   - clearing the box gave `Number("") === 0`, so you could never empty it to retype.
 *
 * Typing now goes into a local draft and commits on blur or Enter (Escape reverts). Spinner clicks
 * and arrow keys still apply immediately, because those are discrete steps the user expects to see
 * move -- they are told apart by `inputType`, which real typing sets and a step does not.
 */
function NumberControl({ label, value, onChange, disabled = false, min, max, step = 1, suffix, hint }: {
  label: string; value: number; onChange: (value: number) => void; disabled?: boolean;
  min?: number; max?: number; step?: number; suffix?: string; hint?: string;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const shown = draft ?? (Number.isFinite(value) ? String(value) : "0");
  const commit = (raw: string) => {
    setDraft(null);
    const parsed = Number(raw);
    // An empty or unparseable box reverts to the last good value rather than becoming 0.
    if (!raw.trim() || !Number.isFinite(parsed)) return;
    let next = parsed;
    if (min !== undefined) next = Math.max(min, next);
    if (max !== undefined) next = Math.min(max, next);
    if (next !== value) onChange(next);
  };
  return <label className={"field number-field" + (hint ? " has-hint" : "")} title={hint}><span>{label}</span><span className="number-input">
    <input type="number" value={shown} min={min} max={max} step={step} disabled={disabled}
      onChange={(event) => {
        const typed = Boolean((event.nativeEvent as InputEvent).inputType);
        if (typed) setDraft(event.target.value); else commit(event.target.value);
      }}
      onBlur={(event) => commit(event.target.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter") { commit(event.currentTarget.value); event.currentTarget.blur(); }
        if (event.key === "Escape") { setDraft(null); event.currentTarget.blur(); }
      }} />
    {suffix && <i>{suffix}</i>}</span></label>;
}

/**
 * Seed field with a dice. **0 means "draw a new seed every run"**; anything else is fixed.
 *
 * The resolved seed is reported in the status line after each run, so a result you liked from a
 * random draw can still be reproduced by typing that number back in.
 */
function SeedControl({ value, onChange, disabled = false }: {
  value: number; onChange: (value: number) => void; disabled?: boolean;
}) {
  return <div className="seed-control">
    <NumberControl label="Seed" value={value} onChange={onChange} disabled={disabled} min={0} max={2147483647}
      hint="0 draws a new random seed for every run. Any other value is fixed — the same seed and settings reproduce the same result." />
    <button type="button" className="seed-dice" disabled={disabled} title="Pick a random fixed seed"
      onClick={() => onChange(Math.floor(Math.random() * 2147483646) + 1)}>Randomise</button>
    <small>{value === 0 ? "New seed every run" : "Fixed · reproducible"}</small>
  </div>;
}

/** 0 is "surprise me". Resolved at run time so the seed actually used can be surfaced. */
function resolveSeed(seed: number): number {
  return seed > 0 ? seed : Math.floor(Math.random() * 2147483646) + 1;
}

/**
 * Fit, padding and source crop for one clip (or the sequence default).
 *
 * The crop rectangle is normalised 0..1 of the SOURCE, not the frame, so it survives a change of
 * sequence resolution or a swap to a proxy. `end` on the crop turns it into a move, which the
 * renderer expresses as an ffmpeg `crop` expression in `t` -- no keyframes involved.
 */
function ConformControls({ value, onChange, disabled = false }: {
  value: ClipConform; onChange: (next: ClipConform) => void; disabled?: boolean;
}) {
  const crop = value.crop?.start ?? FULL_SOURCE_RECT;
  const patchCrop = (patch: Partial<typeof crop>) => {
    const next = { ...crop, ...patch };
    // Clamp so the rect cannot leave the source or collapse -- ffmpeg's `crop` fails on a
    // degenerate rectangle with a message that gives no hint where it came from.
    next.width = Math.min(1, Math.max(0.05, next.width));
    next.height = Math.min(1, Math.max(0.05, next.height));
    next.x = Math.min(1 - next.width, Math.max(0, next.x));
    next.y = Math.min(1 - next.height, Math.max(0, next.y));
    onChange({ ...value, crop: { start: next, end: value.crop?.end } });
  };
  const zoom = Math.round(100 / crop.width);
  return <>
    <SelectControl label="Fit" value={value.fit} disabled={disabled}
      onChange={(fit) => onChange({ ...value, fit: fit as ConformFit })} options={CONFORM_FIT_OPTIONS}
      hint="How the clip is fitted into the sequence frame when the aspect ratios disagree." />
    {value.fit === "contain" && <SelectControl label="Fill bars with" value={value.pad} disabled={disabled}
      onChange={(pad) => onChange({ ...value, pad: pad as ConformPad })} options={CONFORM_PAD_OPTIONS}
      hint="Blurred copy is usually the one you want: the frame is filled and no part of the picture is cropped away." />}
    {value.fit === "contain" && value.pad === "color" && <TextControl label="Bar colour" disabled={disabled}
      value={value.padColor ?? "#000000"} onChange={(padColor) => onChange({ ...value, padColor })} placeholder="#000000" />}
    {value.fit === "contain" && value.pad === "blur" && <RangeControl label="Blur amount" disabled={disabled}
      value={Math.round(value.padBlur ?? 40)} min={5} max={120}
      onChange={(padBlur) => onChange({ ...value, padBlur })} />}
    <RangeControl label="Source zoom" value={zoom} disabled={disabled} min={100} max={400} suffix="%"
      onChange={(next) => {
        const size = Math.min(1, 100 / Math.max(100, next));
        // Zoom about the current centre so the framing does not jump when you drag the slider.
        patchCrop({ width: size, height: size, x: crop.x + (crop.width - size) / 2, y: crop.y + (crop.height - size) / 2 });
      }}
      hint="Crops into the source before fitting. 100% uses the whole frame." />
    {zoom > 100 && <>
      <RangeControl label="Pan X" value={Math.round(crop.width >= 1 ? 50 : (crop.x / (1 - crop.width)) * 100)} disabled={disabled}
        min={0} max={100} suffix="%" onChange={(pct) => patchCrop({ x: (pct / 100) * (1 - crop.width) })} />
      <RangeControl label="Pan Y" value={Math.round(crop.height >= 1 ? 50 : (crop.y / (1 - crop.height)) * 100)} disabled={disabled}
        min={0} max={100} suffix="%" onChange={(pct) => patchCrop({ y: (pct / 100) * (1 - crop.height) })} />
    </>}
  </>;
}

function TextControl({ label, value, onChange, disabled = false, hint, placeholder }: {
  label: string; value: string; onChange: (value: string) => void; disabled?: boolean; hint?: string; placeholder?: string;
}) {
  return <label className={"field" + (hint ? " has-hint" : "")} title={hint}><span>{label}</span>
    <input value={value} disabled={disabled} placeholder={placeholder} onChange={(event) => onChange(event.target.value)} /></label>;
}

function SelectControl({ label, value, onChange, disabled = false, options, hint }: {
  label: string; value: string; onChange: (value: string) => void; disabled?: boolean;
  options: Array<{ value: string; label: string }>; hint?: string;
}) {
  return <label className={"field select-field" + (hint ? " has-hint" : "")} title={hint}><span>{label}</span><select value={value} disabled={disabled} onChange={(event) => onChange(event.target.value)}>
    {options.map((option) => <option value={option.value} key={option.value}>{option.label}</option>)}
  </select></label>;
}

/**
 * A labelled on/off switch, shaped like the other `field` controls so it lines up with them.
 *
 * The hint sits on the wrapping label rather than the input, because a `title` on a *disabled*
 * input never appears -- Chromium fires no mouse events on one -- and this control is meant to be
 * the thing that explains what the options below it are for.
 */
function ToggleControl({ label, value, onChange, disabled = false, hint }: {
  label: string; value: boolean; onChange: (value: boolean) => void; disabled?: boolean; hint?: string;
}) {
  return <label className={"field toggle-field" + (hint ? " has-hint" : "")} title={hint}>
    <span>{label}</span>
    <div className="toggle-switch-wrapper">
      <input type="checkbox" role="switch" checked={value} disabled={disabled}
        onChange={(event) => onChange(event.target.checked)} />
      <span className={`toggle-state-badge ${value ? "on" : "off"}`}>
        {value ? "ON" : "OFF"}
      </span>
    </div>
  </label>;
}

function RangeControl({ label, value, onChange, min, max, step = 1, disabled = false, suffix = "", hint }: {
  label: string; value: number; onChange: (value: number) => void; min: number; max: number;
  step?: number; disabled?: boolean; suffix?: string; hint?: string;
}) {
  return <div className={"range-control" + (hint ? " has-hint" : "")} title={hint}><span>{label}</span><input type="range" value={value} min={min} max={max} step={step} disabled={disabled}
    onChange={(event) => onChange(Number(event.target.value))} /><label><input type="number" value={value} min={min} max={max} step={step} disabled={disabled}
      onChange={(event) => onChange(Number(event.target.value))} />{suffix}</label></div>;
}

// ---------------------------------------------------------------------------
// Media derivatives — thumbnails and waveforms are fetched once per asset
// ---------------------------------------------------------------------------

const pendingDerivatives = new Set<string>();

function useClipThumbnail(asset: MediaAsset | undefined, atSeconds: number) {
  const thumbnail = useEditor((state) => (asset ? state.thumbnails[asset.id] : undefined));
  const setAssetThumbnail = useEditor((state) => state.setAssetThumbnail);
  /**
   * A picture is its own thumbnail.
   *
   * Stills used to go through the same frame extractor as video, which seeks to a timestamp before
   * it decodes -- and a one-frame file has nothing at 0.2 seconds, so ffmpeg wrote an empty file and
   * every image in the bin and on the timeline came back blank. Nothing reported it, because a
   * missing thumbnail is a normal state while one is still being made. The webview decodes the file
   * directly, which is also instant and spawns no process.
   */
  const still = asset && isStillAsset(asset) ? convertFileSrc(asset.path) : undefined;
  useEffect(() => {
    if (!asset || still || thumbnail || isAudioAsset(asset)) return;
    const key = "thumb:" + asset.id;
    if (pendingDerivatives.has(key)) return;
    pendingDerivatives.add(key);
    void invoke<string>("clip_thumbnail", { request: { path: asset.path, atSeconds, width: 192 } })
      .then((value) => setAssetThumbnail(asset.id, value))
      .catch(() => undefined)
      .finally(() => pendingDerivatives.delete(key));
  }, [asset?.id, asset?.path, still, thumbnail, atSeconds, setAssetThumbnail]);
  return still ?? thumbnail;
}

function useClipWaveform(asset: MediaAsset | undefined, wanted: boolean) {
  const peaks = useEditor((state) => (asset ? state.peaks[asset.id] : undefined));
  const setAssetPeaks = useEditor((state) => state.setAssetPeaks);
  useEffect(() => {
    if (!asset || !wanted || peaks || !(isAudioAsset(asset) || asset.hasAudio)) return;
    const key = "wave:" + asset.id;
    if (pendingDerivatives.has(key)) return;
    pendingDerivatives.add(key);
    void invoke<number[]>("clip_waveform", { request: {
      path: asset.path, sourceInSeconds: 0, durationSeconds: seconds(asset.duration), buckets: 320,
    } })
      .then((value) => setAssetPeaks(asset.id, value))
      .catch(() => undefined)
      .finally(() => pendingDerivatives.delete(key));
  }, [asset?.id, asset?.path, wanted, peaks, setAssetPeaks]);
  return peaks;
}

function Waveform({ peaks, from, to }: { peaks: number[]; from: number; to: number }) {
  const path = useMemo(() => {
    if (!peaks.length) return "";
    const start = Math.max(0, Math.floor(from * peaks.length));
    const end = Math.min(peaks.length, Math.ceil(to * peaks.length));
    const window = peaks.slice(start, Math.max(start + 2, end));
    const step = 100 / Math.max(1, window.length - 1);
    const top = window.map((peak, index) => `${(index * step).toFixed(3)},${(50 - peak * 46).toFixed(2)}`);
    const bottom = window.map((peak, index) => `${((window.length - 1 - index) * step).toFixed(3)},${(50 + peak * 46).toFixed(2)}`);
    return `M${top.join(" L")} L${bottom.join(" L")} Z`;
  }, [peaks, from, to]);
  if (!path) return null;
  return <svg className="clip-wave" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
    <path d={path} />
  </svg>;
}

// ---------------------------------------------------------------------------
// Top bar
// ---------------------------------------------------------------------------

function TopBar() {
  const {
    project, snapping, toggleSnapping, busy, newProject, setActivePanel, selectedClipId,
    deleteClip, duplicateClip, splitClip, playhead, updateSelectedClip,
    undo, redo, past, future, setSequenceSettingsOpen, setClipEditorOpen,
  } = useEditor();
  const [openMenu, setOpenMenu] = useState<string | null>(null);
  useEffect(() => {
    if (!openMenu) return;
    const close = (event: PointerEvent) => {
      if (!(event.target as HTMLElement | null)?.closest(".menu-root")) setOpenMenu(null);
    };
    window.addEventListener("pointerdown", close);
    return () => window.removeEventListener("pointerdown", close);
  }, [openMenu]);
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId)!;
  const fps = sequence.frameRate.numerator / sequence.frameRate.denominator;
  const clip = sequence.tracks.flatMap((track) => track.clips).find((item) => item.id === selectedClipId);
  const toggle = (name: string) => () => setOpenMenu((current) => current === name ? null : name);
  return <header className="topbar">
    <div className="brand">
      <span className="brand-core">DF</span>
      <div><strong>DIGITAL FORGE</strong><small>Image + Video Studio</small></div>
    </div>
    <nav className="menu">
      <MenuDropdown label="File" open={openMenu === "file"} setOpen={toggle("file")} items={[
        { label: "New project", action: newProject, disabled: busy },
        { label: "Open project…", action: () => { void openProject(); }, disabled: busy, shortcut: "Ctrl+O" },
        { label: "Save", action: () => { void saveProject(); }, disabled: busy, shortcut: "Ctrl+S" },
        { label: "Save as…", action: () => { void saveProject(true); }, disabled: busy },
        { label: "Import media…", action: () => { void importMedia(); }, disabled: busy },
        { label: "Export…", action: () => setActivePanel("export"), disabled: busy },
      ]} />
      <MenuDropdown label="Edit" open={openMenu === "edit"} setOpen={toggle("edit")} items={[
        { label: "Undo", action: undo, disabled: past.length === 0, shortcut: "Ctrl+Z" },
        { label: "Redo", action: redo, disabled: future.length === 0, shortcut: "Ctrl+Y" },
        { label: "Duplicate clip", action: () => selectedClipId && duplicateClip(selectedClipId), disabled: !selectedClipId, shortcut: "Ctrl+D" },
        { label: "Split at playhead", action: () => selectedClipId && splitClip(selectedClipId, playhead), disabled: !selectedClipId, shortcut: "S" },
        { label: "Delete clip", action: () => selectedClipId && deleteClip(selectedClipId), disabled: !selectedClipId, danger: true, shortcut: "Del" },
      ]} />
      <MenuDropdown label="Clip" open={openMenu === "clip"} setOpen={toggle("clip")} items={[
        { label: "Open AI clip editor…", action: () => setClipEditorOpen(true), disabled: !clip || clip.trackId !== "v1", shortcut: "Ctrl+K" },
        { label: clip?.enabled === false ? "Enable clip" : "Disable clip", action: () => updateSelectedClip({ enabled: !(clip?.enabled ?? true) }), disabled: !clip },
        { label: "Reset transform", action: () => updateSelectedClip({ positionX: sequence.width / 2, positionY: sequence.height / 2, scale: 100, rotation: 0, opacity: 1 }), disabled: !clip },
        { label: "Clear transitions", action: () => updateSelectedClip({ transitionIn: "none", transitionOut: "none" }), disabled: !clip },
      ]} />
      <MenuDropdown label="Sequence" open={openMenu === "sequence"} setOpen={toggle("sequence")} items={[
        { label: "Render timeline…", action: () => { void exportTimeline(resolveExportSettings(useEditor.getState().project)); }, disabled: busy },
        { label: "Sequence settings", action: () => setSequenceSettingsOpen(true), disabled: busy },
      ]} />
      <MenuDropdown label="View" open={openMenu === "view"} setOpen={toggle("view")} items={[
        { label: snapping ? "Disable snapping" : "Enable snapping", action: toggleSnapping, shortcut: "N" },
        { label: "Show export panel", action: () => setActivePanel("export") },
        { label: "Show AI panel", action: () => setActivePanel("ai") },
      ]} />
    </nav>
    <div className="project-title"><span>{project.name}</span><small>{sequence.width} × {sequence.height} · {fps.toFixed(2)} fps</small></div>
    <div className="top-actions">
      <Button title="Undo (Ctrl+Z)" onClick={undo} disabled={past.length === 0}><Undo2 size={16} /></Button>
      <Button title="Redo (Ctrl+Y)" onClick={redo} disabled={future.length === 0}><Redo2 size={16} /></Button>
      <Button title="Toggle snapping (N)" active={snapping} onClick={toggleSnapping}><Magnet size={16} /></Button>
      <button className="save-button" onClick={() => { void saveProject(); }} disabled={busy}><Save size={15} /> Save</button>
      <button className="export-button" onClick={() => setActivePanel("export")} disabled={busy}><Download size={15} /> Export</button>
    </div>
    <SequenceSettingsDialog />
  </header>;
}

// ---------------------------------------------------------------------------
// Media bin
// ---------------------------------------------------------------------------

function AssetTile({ asset, index, selected, renaming, onSelect, onAdd, onMenu, onRenaming, onRename }: {
  asset: MediaAsset; index: number; selected: boolean; renaming: boolean;
  onSelect: () => void; onAdd: () => void; onMenu: (event: ReactMouseEvent) => void;
  onRenaming: (renaming: boolean) => void; onRename: (name: string) => void;
}) {
  const thumbnail = useClipThumbnail(asset, Math.min(1, seconds(asset.duration) / 3));
  return <button className={"asset " + (selected ? "selected" : "")}
    draggable={!renaming} onDragStart={(event) => {
      draggedAssetKind = asset.kind;
      event.dataTransfer.effectAllowed = "copy";
      event.dataTransfer.setData("application/x-aiv-asset", asset.id);
      event.dataTransfer.setData("text/plain", asset.id);
    }}
    onContextMenu={onMenu} onClick={onSelect} onDoubleClick={onAdd}>
    <div className={"asset-thumb thumb-" + (index % 2)} style={thumbnail ? { backgroundImage: `url("${thumbnail}")` } : undefined}>
      {isAudioAsset(asset) && <AudioLines size={20} className="asset-thumb-icon" />}
      <span className="duration">{formatTimecode(seconds(asset.duration)).slice(3)}</span>
      <span className="drag-badge">DRAG</span>
      {(asset.kind === "generated" || asset.origin) && <span className="ai-badge">AI</span>}
    </div>
    <RenameField value={asset.name} editing={renaming} onEditing={onRenaming} onCommit={onRename} />
    <small>{asset.width ? asset.width + " × " + asset.height : "Audio"} · {asset.frameRate ? (asset.frameRate.numerator / asset.frameRate.denominator).toFixed(2) + " fps" : "media"}</small>
    <span className="asset-add" onClick={(event) => { event.stopPropagation(); onAdd(); }}>+ Timeline</span>
  </button>;
}

function MediaBin() {
  const {
    project, selectedAssetId, selectedClipId, setSelectedAsset, addAssetToTimeline,
    removeAsset, renameAsset, updateSelectedClip, addGraphic, busy,
  } = useEditor();
  const [tab, setTab] = useState<"media" | "effects" | "graphics">("media");
  const [query, setQuery] = useState("");
  const [menu, setMenu] = useState<MenuState>(null);
  const [renamingAssetId, setRenamingAssetId] = useState<string>();
  const selectedClip = project.sequences.flatMap((sequence) => sequence.tracks).flatMap((track) => track.clips)
    .find((clip) => clip.id === selectedClipId);
  const selectedVideoClip = selectedClip?.trackId === "v1" ? selectedClip : undefined;
  const normalizedQuery = query.trim().toLowerCase();
  const assets = project.assets.filter((asset) => asset.name.toLowerCase().includes(normalizedQuery));
  const presets = EFFECT_PRESETS.filter((preset) => (preset.name + " " + preset.category + " " + preset.description).toLowerCase().includes(normalizedQuery));
  const graphics = GRAPHIC_TEMPLATES.filter((template) => (template.name + " " + template.description).toLowerCase().includes(normalizedQuery));
  const openAssetMenu = (event: ReactMouseEvent, asset: MediaAsset) => {
    event.preventDefault();
    setSelectedAsset(asset.id);
    setMenu({ x: event.clientX, y: event.clientY, items: [
      { label: "Preview", action: () => setSelectedAsset(asset.id), shortcut: "Enter" },
      { label: "Rename…", action: () => setRenamingAssetId(asset.id) },
      { label: "Add to end of timeline", action: () => addAssetToTimeline(asset.id) },
      { label: "Remove from project", action: () => removeAsset(asset.id), danger: true },
    ] });
  };
  const selectTab = (next: "media" | "effects" | "graphics") => { setTab(next); setQuery(""); };
  return <aside className="media-panel">
    <div className="panel-tabs">
      <button className={tab === "media" ? "selected" : ""} onClick={() => selectTab("media")}>Media</button>
      <button className={tab === "effects" ? "selected" : ""} onClick={() => selectTab("effects")}>Effects</button>
      <button className={tab === "graphics" ? "selected" : ""} onClick={() => selectTab("graphics")}>Graphics</button>
    </div>
    {tab === "media" && <div className="media-tools"><button className="import-button" onClick={() => { void importMedia(); }} disabled={busy}><Plus size={15} /> Import media</button></div>}
    <label className="search"><MousePointer2 size={13} /><input placeholder={tab === "media" ? "Search project media" : tab === "effects" ? "Search effects" : "Search graphics"} value={query} onChange={(event) => setQuery(event.target.value)} /></label>
    {tab === "media" && <>
      <div className="bin-title"><span>PROJECT MEDIA</span><small>{project.assets.length} items</small></div>
      <div className="asset-grid">
        {assets.map((asset, index) => <AssetTile key={asset.id} asset={asset} index={index}
          selected={selectedAssetId === asset.id}
          renaming={renamingAssetId === asset.id}
          onSelect={() => setSelectedAsset(asset.id)}
          onAdd={() => addAssetToTimeline(asset.id)}
          onMenu={(event) => openAssetMenu(event, asset)}
          onRenaming={(on) => setRenamingAssetId(on ? asset.id : undefined)}
          onRename={(name) => renameAsset(asset.id, name)} />)}
        <button className="empty-asset" onClick={() => { void importMedia(); }} disabled={busy}><Plus size={24} /><span>{project.assets.length ? "Add media" : "Import your first clip"}</span></button>
      </div>
      <p className="panel-help">Drag media directly onto V1, double-click to append, or right-click for actions.</p>
    </>}
    {tab === "effects" && <>
      <div className="bin-title"><span>CLIP EFFECTS</span><small>{presets.length} presets</small></div>
      {!selectedVideoClip && <div className="library-notice"><Palette size={22} /><strong>Select a V1 clip</strong><span>Effects apply immediately to preview and export.</span></div>}
      <div className="library-list">{presets.map((preset) => <button key={preset.id} disabled={!selectedVideoClip || busy}
        onClick={() => updateSelectedClip({ effects: { ...preset.effects } })}>
        <span className="library-icon"><Palette size={16} /></span><div><strong>{preset.name}</strong><small>{preset.category} · {preset.description}</small></div><i>Apply</i>
      </button>)}</div>
      <p className="panel-help">Presets are non-destructive. Fine-tune every parameter in the Inspector.</p>
    </>}
    {tab === "graphics" && <>
      <div className="bin-title"><span>ESSENTIAL GRAPHICS</span><small>{graphics.length} templates</small></div>
      <button className="import-button subtitle-import" onClick={() => { void importSubtitles(); }} disabled={busy}><Captions size={15} /> Import SRT / WebVTT</button>
      <div className="library-list">{graphics.map((template) => <button key={template.id} disabled={busy}
        onClick={() => addGraphic(template.id)}>
        <span className="library-icon">{template.id === "caption" ? <Captions size={16} /> : <Type size={16} />}</span><div><strong>{template.name}</strong><small>{template.description}</small></div><i>+ {template.id === "caption" ? "C1" : "G1"}</i>
      </button>)}</div>
      <p className="panel-help">Adds a five-second editable overlay at the playhead. Graphics use G1; captions use C1. Both preview live and render into export.</p>
    </>}
    <ContextMenu menu={menu} close={() => setMenu(null)} />
  </aside>;
}

/**
 * One V2 (or higher) picture layer composited over the programme.
 *
 * The monitor previously rendered V1 only, so an overlay clip was invisible until you exported --
 * the same preview/export divergence that hid the pillarbox bars and the transition ripple. The
 * geometry here mirrors `render_overlays` in `export.rs`: a normalised 0..1 box of the frame,
 * `contain`/`cover`/`stretch` mapped to `object-fit`, plus opacity, corner radius and border.
 *
 * `fullFrame` is set when V1 has no clips at all: with nothing underneath, the placement box would
 * only letterbox the picture against black, so the layer takes the whole frame instead.
 */
function OverlayLayer({ clip, asset, box, playhead, playing, muted, fullFrame }: {
  clip: TimelineClip; asset: MediaAsset; box: OverlayPlacement;
  playhead: number; playing: boolean; muted: boolean; fullFrame: boolean;
}) {
  const ref = useRef<HTMLVideoElement>(null);
  const rate = clipPlaybackRate(clip);
  const elapsed = Math.max(0, playhead - seconds(clip.start));
  useEffect(() => {
    const video = ref.current;
    if (!video) return;
    const target = seconds(clip.sourceIn) + elapsed * rate;
    // Same 0.12s tolerance the incoming-transition layer uses: tight enough to stay in sync, loose
    // enough not to fight the decoder with a seek on every frame.
    if (Math.abs(video.currentTime - target) > 0.12) video.currentTime = target;
    video.playbackRate = rate;
    if (playing && video.paused) void video.play().catch(() => undefined);
    if (!playing && !video.paused) video.pause();
  }, [elapsed, playing, rate, clip.id]);

  const geometry: React.CSSProperties = fullFrame
    ? { left: 0, top: 0, width: "100%", height: "100%" }
    : {
        left: `${box.x * 100}%`, top: `${box.y * 100}%`,
        width: `${box.width * 100}%`, height: `${box.height * 100}%`,
      };
  const fit = box.fit === "stretch" ? "fill" : box.fit;
  return <video ref={ref} key={clip.id} src={convertFileSrc(asset.path)} muted={muted} preload="metadata"
    className="preview-overlay"
    style={{
      ...geometry,
      objectFit: fullFrame ? "contain" : (fit as React.CSSProperties["objectFit"]),
      opacity: box.opacity * clip.opacity,
      borderRadius: fullFrame ? 0 : `${box.cornerRadius * 100}%`,
      border: !fullFrame && box.borderWidth > 0 ? `${Math.max(1, box.borderWidth * 100)}px solid ${box.borderColor}` : undefined,
    }} />;
}

/**
 * One audible timeline clip. The programme video remains the playback clock; this layer follows
 * it just like a DAW follows its transport, correcting only visible drift so the browser is not
 * forced to seek on every `timeupdate` tick.
 *
 * Audio clips are authored against the un-rippled timeline, while `programStart` is their mapped
 * position after picture transitions. Keeping that mapping outside this component makes preview
 * and export use the same clock.
 */
function TimelineAudioLayer({ clip, asset, programStart, playhead, playing, muted, gainDb, onError }: {
  clip: TimelineClip; asset: MediaAsset; programStart: number;
  playhead: number; playing: boolean; muted: boolean; gainDb: number;
  onError: () => void;
}) {
  const ref = useRef<HTMLAudioElement>(null);
  const rate = clipPlaybackRate(clip);
  const elapsed = Math.max(0, playhead - programStart);
  // HTMLMediaElement.volume cannot amplify above unity. Negative gain and mute remain exact;
  // positive preview gain saturates at 1 while the exporter applies the full dB value.
  const volume = Math.min(1, Math.max(0, 10 ** ((clip.volumeDb + gainDb) / 20)));

  const sync = (audio: HTMLAudioElement) => {
    const target = seconds(clip.sourceIn) + elapsed * rate;
    const end = Math.max(seconds(clip.sourceIn), seconds(clip.sourceOut) - 0.02);
    const bounded = Math.min(target, end);
    if (Math.abs(audio.currentTime - bounded) > 0.12) audio.currentTime = bounded;
    audio.playbackRate = rate;
    audio.volume = volume;
    audio.muted = muted;
    if (playing && audio.paused) void audio.play().catch(() => undefined);
    if (!playing && !audio.paused) audio.pause();
  };

  useEffect(() => {
    const audio = ref.current;
    if (!audio) return;
    if (audio.readyState >= 1) sync(audio);
    return () => audio.pause();
  }, [elapsed, playing, muted, rate, volume, clip.id]);

  return <audio ref={ref} key={clip.id} src={convertFileSrc(asset.path)} preload="metadata"
    onLoadedMetadata={(event) => sync(event.currentTarget)} onError={onError} />;
}

// ---------------------------------------------------------------------------
// Program monitor with live transition compositing
// ---------------------------------------------------------------------------

/**
 * VFX Phase 7: the timeline monitor is the compositor (`ProgrammeMonitor`) -- what it shows is what
 * the export writes. The old CSS-layer monitor stays behind the "Legacy" switch, and still previews a
 * media-bin source (nothing selected on the timeline), which is not a programme.
 */
function Monitor() {
  const { project, selectedAssetId, selectedClipId } = useEditor();
  const [legacy, setLegacy] = useState(readLegacyMonitor);
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId)!;
  const onTimeline = sequence.tracks.some((track) => track.clips.some((item) => item.id === selectedClipId));
  const sourcePreview = !onTimeline && Boolean(selectedAssetId);
  const toggle = <button type="button" className={"monitor-toggle" + (legacy ? " selected" : "")}
    title="Switch between the compositor monitor (what the export renders) and the legacy preview"
    onClick={() => { writeLegacyMonitor(!legacy); setLegacy(!legacy); }}>{legacy ? "Legacy monitor" : "Compositor"}</button>;
  return legacy || sourcePreview ? <LegacyMonitor toggle={sourcePreview && !legacy ? undefined : toggle} /> : <ProgrammeMonitor toggle={toggle} />;
}

function LegacyMonitor({ toggle }: { toggle?: React.ReactNode }) {
  const {
    playhead, playing, setPlaying, setPlayhead, project, selectedAssetId, selectedClipId,
    setSelectedClip, setStatus, maskPicking, maskPoints, addMaskPoint,
  } = useEditor();
  const videoRef = useRef<HTMLVideoElement>(null);
  const incomingRef = useRef<HTMLVideoElement>(null);
  const backdropRef = useRef<HTMLVideoElement>(null);
  const lastSync = useRef(0);
  const advancing = useRef(false);
  const pendingMainPlay = useRef(false);
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const [gapUntil, setGapUntil] = useState<number>();
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId)!;
  const videoTrack = sequence.tracks.find((track) => track.id === "v1");
  const graphicsTrack = sequence.tracks.find((track) => track.id === "g1");
  const captionTrack = sequence.tracks.find((track) => track.id === "c1");
  const timelineClips = (videoTrack?.hidden ? [] : videoTrack?.clips ?? [])
    .filter((item) => item.enabled).sort((a, b) => seconds(a.start) - seconds(b.start));
  // Programme positions after ripple. Authored `start` is what the user dragged; this is where the
  // frame actually lands once transitions have taken what they need. The monitor, the timeline and
  // the export all read this ONE layout -- computing it in two places is how the preview and the
  // exported file drifted apart.
  const layout = resolveJunctionLayout(timelineClips);
  const startOf = (item: TimelineClip | undefined) =>
    item ? layout.start.get(item.id) ?? seconds(item.start) : 0;
  const endOf = (item: TimelineClip | undefined) =>
    item ? startOf(item) + timelineClipDuration(item) : 0;
  const selectedTimelineClip = sequence.tracks.flatMap((track) => track.clips).find((item) => item.id === selectedClipId);
  const clip = selectedTimelineClip?.trackId === "v1"
    ? selectedTimelineClip
    : selectedTimelineClip
      ? timelineClips.find((item) => playhead >= startOf(item) && playhead < endOf(item)) ?? timelineClips[0]
      : undefined;
  // Overlay picture tracks composite over the programme. V2 is the lowest of them and draws first;
  // anything above it stacks on top, matching the export's ordering.
  const overlayTracks = sequence.tracks.filter((item) =>
    item.kind === "video" && item.id !== "v1" && item.overlay && !item.hidden);
  // With V1 empty the payload builder promotes the lowest overlay track to the programme, so the
  // monitor must not also draw it as a layer -- it would be composited twice.
  const promotedOverlayId = timelineClips.length === 0
    ? overlayTracks.find((item) => item.clips.some((c) => c.enabled))?.id
    : undefined;
  const overlayLayers = overlayTracks.filter((item) => item.id !== promotedOverlayId)
    .flatMap((item) => item.clips
      .filter((c) => c.enabled && playhead >= seconds(c.start) && playhead < seconds(c.start) + timelineClipDuration(c))
      .flatMap((c) => {
        const overlayAsset = project.assets.find((a) => a.id === c.assetId);
        return overlayAsset && overlayAsset.kind !== "audio"
          ? [{ clip: c, asset: overlayAsset, box: item.overlay!, muted: item.muted }] : [];
      }));

  // Separate A1/A2 media used to exist only in the export payload: the monitor never instantiated
  // it, so Play could only expose audio embedded in the V1 video. Map these clips through the same
  // transition ripple as export and mount every layer active at the current programme time.
  const audioLayers = sequence.tracks.filter((item) => item.kind === "audio")
    .flatMap((item) => item.clips.filter((c) => c.enabled).flatMap((c) => {
      const programStart = mapToProgramme(layout, timelineClips, seconds(c.start));
      const audioAsset = project.assets.find((a) => a.id === c.assetId);
      return audioAsset?.path
        && playhead >= programStart
        && playhead < programStart + timelineClipDuration(c)
        ? [{ clip: c, asset: audioAsset, programStart, muted: item.muted, gainDb: item.gainDb ?? 0 }]
        : [];
    }));

  const activeGraphics = [graphicsTrack, captionTrack].filter((track) => track && !track.hidden).flatMap((track) => track?.clips ?? []).filter((item) =>
    // Graphics and captions live on other tracks but are authored against the same programme, so
    // they map through the ripple too -- otherwise a lower third fires early by exactly that much.
    item.enabled && item.graphic
    && playhead >= mapToProgramme(layout, timelineClips, seconds(item.start))
    && playhead < mapToProgramme(layout, timelineClips, seconds(item.start)) + timelineClipDuration(item),
  );
  const asset = project.assets.find((item) => item.id === (clip?.assetId ?? selectedAssetId));
  const fps = sequence.frameRate.numerator / sequence.frameRate.denominator;
  const playbackRate = clip ? clipPlaybackRate(clip) : 1;
  const clipIndex = clip ? timelineClips.findIndex((item) => item.id === clip.id) : -1;
  const nextClip = clipIndex >= 0 ? timelineClips[clipIndex + 1] : undefined;
  const nextAsset = project.assets.find((item) => item.id === nextClip?.assetId);
  const edits = timelineClips.flatMap((item) => [startOf(item), endOf(item)]);

  // -- transition window ---------------------------------------------------
  // A junction transition occupies the tail of the outgoing clip, matching how xfade consumes
  // the overlap on export. Head and tail of the sequence degrade to a dip through a colour.
  const junction = clip && nextClip ? resolveJunction(clip, nextClip) : undefined;
  const clipEnd = endOf(clip);
  // Exactly the length the layout resolved -- same capping, same number the renderer uses.
  const junctionDuration = junction && junction.id !== "none" && nextClip
    ? layout.transition.get(nextClip.id) ?? 0
    : 0;
  const inJunction = junctionDuration > 0 && playhead >= clipEnd - junctionDuration && playhead < clipEnd;
  const junctionProgress = inJunction ? (playhead - (clipEnd - junctionDuration)) / junctionDuration : 0;
  const layers = transitionLayers(inJunction ? junction!.id : "none", junctionProgress);

  const isFirst = clipIndex === 0;
  const isLast = clipIndex >= 0 && clipIndex === timelineClips.length - 1;
  const headDuration = isFirst && clip?.transitionIn && clip.transitionIn !== "none"
    ? Math.min(clip.transitionDuration ?? 0.5, timelineClipDuration(clip)) : 0;
  const tailDuration = isLast && clip?.transitionOut && clip.transitionOut !== "none"
    ? Math.min(clip.transitionOutDuration ?? 0.5, timelineClipDuration(clip)) : 0;
  const headOpacity = headDuration > 0 && clip && playhead < startOf(clip) + headDuration
    ? 1 - (playhead - startOf(clip)) / headDuration : 0;
  const tailOpacity = tailDuration > 0 && playhead > clipEnd - tailDuration
    ? (playhead - (clipEnd - tailDuration)) / tailDuration : 0;
  const edgeColor = (id: string | undefined) => id === "fadewhite" ? "#ffffff" : id === "fadegrays" ? "#808080" : "#000000";

  const seekTarget = clip ? seconds(clip.sourceIn) + Math.max(0, playhead - startOf(clip)) * playbackRate : playhead;
  const boundedSeekTarget = (video: HTMLVideoElement, requested = seekTarget) => {
    const end = clip ? seconds(clip.sourceOut) : seconds(asset?.duration ?? fromSeconds(video.duration || 0));
    return boundedMediaTime(requested, clip ? seconds(clip.sourceIn) : 0, end);
  };
  const applySeek = (video: HTMLVideoElement, requested = seekTarget) => {
    try {
      video.currentTime = boundedSeekTarget(video, requested);
      return true;
    } catch (error) {
      pendingMainPlay.current = false;
      setPlaying(false);
      setStatus("Could not seek this media: " + errorText(error));
      return false;
    }
  };
  const beginMainPlayback = (video: HTMLVideoElement) => {
    pendingMainPlay.current = false;
    void video.play().catch((error) => {
      if (!playingRef.current) return;
      setPlaying(false);
      setStatus("Playback could not continue: " + errorText(error));
    });
  };
  const seekThenPlay = (video: HTMLVideoElement, requested = seekTarget) => {
    video.playbackRate = playbackRate;
    if (video.readyState < 1) {
      pendingMainPlay.current = true;
      return;
    }
    const target = boundedSeekTarget(video, requested);
    if (video.seeking || Math.abs(video.currentTime - target) > 0.08) {
      // Starting a second decoder while WebView2 is still resolving a distant keyframe seek is a
      // common long-form crash path. Park playback and resume only from seeked/canplay.
      pendingMainPlay.current = true;
      if (!video.paused) video.pause();
      if (!video.seeking && Math.abs(video.currentTime - target) > 0.08) applySeek(video, target);
      return;
    }
    beginMainPlayback(video);
  };
  useEffect(() => {
    advancing.current = false;
    setGapUntil(undefined);
    const video = videoRef.current;
    if (!video) return;
    if (video.readyState >= 1) applySeek(video);
  }, [clip?.id, asset?.path]);
  // Ruler scrubbing parks normal playback and drives the programme clock directly. Seek every
  // playhead update so dragging backward shows the matching reverse frame sequence as well.
  useEffect(() => {
    if (playing) return;
    const video = videoRef.current;
    if (video?.readyState && Math.abs(video.currentTime - seekTarget) > 0.015) applySeek(video);
  }, [playhead, playing, seekTarget, clip?.id, asset?.path]);
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    video.playbackRate = playbackRate;
    if (playing) {
      seekThenPlay(video);
    } else {
      pendingMainPlay.current = false;
      video.pause();
    }
  }, [playing, asset?.path, clip?.id, playbackRate, seekTarget, setPlaying, setStatus]);

  // Keep the incoming layer locked to the transition window so the preview matches the render.
  useEffect(() => {
    const incoming = incomingRef.current;
    if (!incoming || !nextClip) return;
    if (!inJunction) {
      incoming.pause();
      return;
    }
    const elapsed = playhead - (clipEnd - junctionDuration);
    const target = seconds(nextClip.sourceIn) + Math.max(0, elapsed) * clipPlaybackRate(nextClip);
    if (Math.abs(incoming.currentTime - target) > 0.12) incoming.currentTime = target;
    incoming.playbackRate = clipPlaybackRate(nextClip);
    if (playing && incoming.paused) void incoming.play().catch(() => undefined);
    if (!playing && !incoming.paused) incoming.pause();
  }, [inJunction, playhead, playing, nextClip?.id, clipEnd, junctionDuration]);

  // The blur backdrop follows the main video, but loosely: it is blurred past recognition, so
  // chasing it frame-accurately would burn decode time for nothing. Re-sync only on a visible
  // drift, and mirror play/pause so it does not run on while the programme is parked.
  useEffect(() => {
    const backdrop = backdropRef.current;
    const main = videoRef.current;
    if (!backdrop || !main) return;
    if (Math.abs(backdrop.currentTime - main.currentTime) > 0.35) backdrop.currentTime = main.currentTime;
    backdrop.playbackRate = main.playbackRate || 1;
    if (playing && backdrop.paused) void backdrop.play().catch(() => undefined);
    if (!playing && !backdrop.paused) backdrop.pause();
  }, [playhead, playing, asset?.path]);

  const advance = () => {
    if (advancing.current) return;
    advancing.current = true;
    if (clip && nextClip) {
      const nextStart = startOf(nextClip);
      if (nextStart > clipEnd + 1 / fps / 2) {
        setPlayhead(clipEnd);
        setGapUntil(nextStart);
      } else {
        setSelectedClip(nextClip.id);
        setTimeout(() => setPlaying(true), 0);
      }
    } else {
      setPlaying(false);
      setStatus(clip ? "Timeline playback complete" : "Source playback complete");
    }
  };
  useEffect(() => {
    if (gapUntil === undefined || !playing || !nextClip) return;
    const timelineStart = clipEnd;
    const wallStart = performance.now();
    let frameId = 0;
    const tick = (now: number) => {
      const value = Math.min(gapUntil, timelineStart + (now - wallStart) / 1000);
      setPlayhead(value);
      if (value >= gapUntil - 0.0005) {
        setGapUntil(undefined);
        advancing.current = false;
        setSelectedClip(nextClip.id);
        setTimeout(() => setPlaying(true), 0);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    };
    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, [gapUntil, playing, nextClip?.id, clipEnd, setPlayhead, setSelectedClip, setPlaying]);
  const syncTime = (video: HTMLVideoElement) => {
    if (clip && video.currentTime >= seconds(clip.sourceOut) - 0.04) {
      advance();
      return;
    }
    const now = performance.now();
    if (now - lastSync.current < 60) return;
    lastSync.current = now;
    setPlayhead(clip ? startOf(clip) + Math.max(0, video.currentTime - seconds(clip.sourceIn)) / playbackRate : video.currentTime);
  };
  const jump = (direction: -1 | 1) => {
    const candidates = direction < 0 ? edits.filter((value) => value < playhead - 0.01) : edits.filter((value) => value > playhead + 0.01);
    const target = direction < 0 ? candidates.at(-1) ?? 0 : candidates[0] ?? playhead;
    setPlayhead(target);
    const targetClip = timelineClips.find((item) => target >= startOf(item) && target < endOf(item));
    if (targetClip && targetClip.id !== clip?.id) setSelectedClip(targetClip.id);
    else if (videoRef.current && clip) {
      const sourceTarget = seconds(clip.sourceIn) + (target - startOf(clip)) * playbackRate;
      if (playingRef.current) seekThenPlay(videoRef.current, sourceTarget); else applySeek(videoRef.current, sourceTarget);
    }
  };

  // Mirror the render's conform rules in CSS. The preview and the export MUST agree here: the
  // pillarbox bars nobody noticed until VLC existed only because the monitor happened to letterbox
  // the same way the renderer did, by coincidence rather than by construction.
  //
  //   contain -> object-fit: contain   (renderer: force_original_aspect_ratio=decrease + pad)
  //   cover   -> object-fit: cover     (renderer: ...=increase + crop)
  //   stretch -> object-fit: fill      (renderer: plain scale)
  //   none    -> object-fit: none      (renderer: no scale at all)
  //
  // The source crop maps to `object-view-box`, which is the CSS equivalent of the renderer's
  // `crop=` filter: an inset rectangle of the SOURCE, before fitting. On a webview too old to
  // support it the property is ignored and the preview simply shows the uncropped frame.
  const conformOf = (item: TimelineClip | undefined): ClipConform =>
    item?.conform ?? sequence.conform ?? DEFAULT_CONFORM;

  const conformStyle = (item: TimelineClip | undefined): React.CSSProperties => {
    const conform = conformOf(item);
    const fit = conform.fit === "stretch" ? "fill" : conform.fit === "none" ? "none" : conform.fit;
    const style: React.CSSProperties = { objectFit: fit as React.CSSProperties["objectFit"] };
    const crop = conform.crop?.start;
    if (crop && (crop.x > 0 || crop.y > 0 || crop.width < 1 || crop.height < 1)) {
      const pct = (value: number) => (value * 100).toFixed(4) + "%";
      // inset() order is top right bottom left.
      (style as Record<string, string>).objectViewBox =
        `inset(${pct(crop.y)} ${pct(1 - crop.x - crop.width)} ${pct(1 - crop.y - crop.height)} ${pct(crop.x)})`;
    }
    return style;
  };

  const geometry = (item: TimelineClip | undefined): React.CSSProperties => {
    if (!item) return {};
    const dx = (((item.positionX ?? sequence.width / 2) - sequence.width / 2) / sequence.width) * 100;
    const dy = (((item.positionY ?? sequence.height / 2) - sequence.height / 2) / sequence.height) * 100;
    return {
      transform: "translate(" + dx + "%, " + dy + "%) scale(" + (item.scale ?? 100) / 100 + ") rotate(" + (item.rotation ?? 0) + "deg)",
      opacity: item.opacity,
      mixBlendMode: cssBlendMode(item.blendMode),
      filter: clipFilter(item),
      ...conformStyle(item),
    };
  };
  const outgoingStyle: React.CSSProperties = { ...geometry(clip), ...layers.outgoing };
  const incomingStyle: React.CSSProperties = { ...geometry(nextClip), ...layers.incoming };
  const effect = effectsFor(clip?.effects);

  return <main className="monitor-column">
    <div className="monitor-header">
      <span>{clip ? "PROGRAM" : "SOURCE"}</span>
      <div>
        {inJunction && <span className="transition-live">{transitionById(junction!.id).name} · {Math.round(junctionProgress * 100)}%</span>}
        <span>{asset?.name ?? "No media selected"}</span>
        {toggle}
      </div>
    </div>
    <div className="monitor"><div className={"preview-frame" + (maskPicking ? " picking" : "")}
      onClick={(event) => {
        if (!maskPicking) return;
        // Normalise against the frame box so the worker can scale to the source resolution.
        const rect = event.currentTarget.getBoundingClientRect();
        addMaskPoint({
          x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
          y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
          include: !event.shiftKey,
        });
      }}>
      {/* Blur-fill backdrop: the same frame, zoomed to cover and blurred, sitting behind the
          contained picture so the bars are filled exactly as the renderer fills them. Muted and
          inert -- it is decoration, and being a few frames adrift is invisible through a 40px
          blur. Only mounted when the clip actually asks for it. */}
      {asset?.path && asset.kind !== "audio" && conformOf(clip).fit === "contain" && conformOf(clip).pad === "blur" &&
        (asset.kind === "image"
          ? <img className="preview-backdrop" src={convertFileSrc(asset.path)} alt=""
              style={{ filter: `blur(${(conformOf(clip).padBlur ?? 40) / 4}px)`, ...conformStyle(clip), objectFit: "cover" }} />
          : <video ref={backdropRef} key={"backdrop-" + asset.path} src={convertFileSrc(asset.path)} muted preload="metadata"
              className="preview-backdrop"
              style={{ filter: `blur(${(conformOf(clip).padBlur ?? 40) / 4}px)`, ...conformStyle(clip), objectFit: "cover" }} />)}

      {asset?.path && asset.kind === "image"
        ? <img className="preview-image" src={convertFileSrc(asset.path)} alt={asset.name} style={outgoingStyle} />
        : asset?.path && asset.kind !== "audio"
          ? <video ref={videoRef} key={asset.path} src={convertFileSrc(asset.path)} style={outgoingStyle} preload="metadata" muted={Boolean(videoTrack?.muted)}
              onLoadedMetadata={(event) => {
                event.currentTarget.playbackRate = playbackRate;
                if (playingRef.current) seekThenPlay(event.currentTarget); else applySeek(event.currentTarget);
              }}
              onSeeked={(event) => {
                if (pendingMainPlay.current && playingRef.current) beginMainPlayback(event.currentTarget);
              }}
              onCanPlay={(event) => {
                if (pendingMainPlay.current && playingRef.current && !event.currentTarget.seeking) beginMainPlayback(event.currentTarget);
              }}
              onWaiting={() => setStatus("Buffering local media…")}
              onPlaying={() => setStatus(clip ? "Playing timeline" : "Playing source")}
              onError={() => { setPlaying(false); setStatus("This media codec could not be decoded by the Windows webview"); }}
              onTimeUpdate={(event) => syncTime(event.currentTarget)} onEnded={advance} />
          : <div className="preview-empty"><Film size={34} /><strong>Import media to begin</strong><small>Choose a source or timeline clip to preview it</small></div>}

      {/* Incoming layer for the active junction transition. */}
      {nextAsset?.path && nextAsset.kind !== "audio" && <video
        ref={incomingRef} key={"incoming-" + nextAsset.path} src={convertFileSrc(nextAsset.path)} preload="metadata" muted
        className={"preview-incoming" + (inJunction ? " active" : "")} style={incomingStyle} />}
      {overlayLayers.map(({ clip: overlayClip, asset: overlayAsset, box, muted }) =>
        <OverlayLayer key={overlayClip.id} clip={overlayClip} asset={overlayAsset} box={box}
          playhead={playhead} playing={playing} muted={muted} fullFrame={false} />)}
      {audioLayers.map(({ clip: audioClip, asset: audioAsset, programStart, muted, gainDb }) =>
        <TimelineAudioLayer key={audioClip.id} clip={audioClip} asset={audioAsset}
          programStart={programStart} playhead={playhead} playing={playing} muted={muted} gainDb={gainDb}
          onError={() => setStatus(`Audio preview could not decode ${audioAsset.name}`)} />)}

      {layers.dip && inJunction && <div className="preview-dip" style={{ background: layers.dip.color, opacity: layers.dip.opacity }} />}
      {headOpacity > 0 && <div className="preview-dip" style={{ background: edgeColor(clip?.transitionIn), opacity: headOpacity }} />}
      {tailOpacity > 0 && <div className="preview-dip" style={{ background: edgeColor(clip?.transitionOut), opacity: tailOpacity }} />}
      {gapUntil !== undefined && <div className="preview-dip" style={{ background: "#000", opacity: 1 }} />}

      {clip && effect.vignette > 0 && <div className="preview-vignette" style={{ opacity: effect.vignette }} />}
      {activeGraphics.map((graphicClip) => {
        const graphic = graphicClip.graphic!;
        const duration = timelineClipDuration(graphicClip);
        const frame = textMotionFrame(graphic, playhead - seconds(graphicClip.start), duration);
        const glow = frame.glow > 0 ? `0 0 ${8 + frame.glow * 22}px ${graphic.accentColor}` : undefined;
        const backdrop = graphic.backdrop ?? "box";
        const shimmerStyle: CSSProperties = graphic.motionLoop === "shimmer" ? {
          backgroundImage: `linear-gradient(105deg, ${graphic.textColor} 34%, ${graphic.accentColor} 49%, ${graphic.textColor} 64%)`,
          backgroundSize: "240% 100%", backgroundPosition: `${100 - frame.shimmer * 200}% 0`,
          WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent",
        } : { color: graphic.textColor };
        return <div className={"graphic-preview graphic-" + graphic.template + " backdrop-" + backdrop} key={graphicClip.id} style={{
          opacity: graphicClip.opacity * frame.opacity,
        }}>
          <div style={{ backgroundColor: backdrop === "box" ? graphicColorWithOpacity(graphic.backgroundColor, graphic.backgroundOpacity ?? 1) : "transparent",
            borderColor: graphic.accentColor, borderWidth: backdrop === "outline" ? graphic.backgroundBorderWidth ?? 3 : undefined,
            padding: `${graphic.backgroundPadding ?? 16}px`,
            transform: `translate(${frame.translateX}px, ${frame.translateY}px) scale(${frame.scale}) rotate(${frame.rotation}deg)`,
            filter: `blur(${frame.blur}px) hue-rotate(${frame.hue}deg)`, transformOrigin: "center",
            clipPath: `inset(0 ${(1 - frame.reveal) * 100}% 0 0)`,
          }}>
            <strong style={{ ...shimmerStyle, fontFamily: graphicFontStack(graphic.fontFamily), fontWeight: graphic.fontWeight ?? 700,
              fontStyle: graphic.italic ? "italic" : "normal", fontSize: graphic.fontSize, letterSpacing: `${frame.tracking}px`,
              WebkitTextStroke: `${graphic.strokeWidth ?? 0}px ${graphic.strokeColor ?? "#000000"}`,
              textShadow: [glow, (graphic.shadowOffset ?? 0) > 0 ? `${graphic.shadowOffset}px ${graphic.shadowOffset}px ${Math.max(2, (graphic.shadowOffset ?? 0) * 1.6)}px ${graphic.shadowColor ?? "#000000"}` : undefined].filter(Boolean).join(", ") || undefined,
            }}>{frame.text || "\u00a0"}</strong>
            {graphic.subtitle && <span style={{ color: graphic.textColor, fontFamily: graphicFontStack(graphic.fontFamily) }}>{graphic.subtitle}</span>}
          </div>
        </div>;
      })}
      {maskPoints.map((point, index) => <span key={index}
        className={"mask-point" + (point.include ? "" : " exclude")}
        style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
        title={(point.include ? "Include" : "Exclude") + ` point ${index + 1}`} />)}
      <div className="safe-area" />
    </div></div>
    <div className="transport">
      <div className="transport-left"><span className="timecode">{formatTimecode(playhead, fps)}</span></div>
      <div className="transport-center"><Button title="Previous edit" onClick={() => jump(-1)}><SkipBack size={18} /></Button>
        <button className="play" onClick={() => setPlaying(!playing)} disabled={!asset?.path || asset.kind === "image"}>{playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}</button>
        <Button title="Next edit" onClick={() => jump(1)}><SkipForward size={18} /></Button></div>
      <div className="transport-right"><span>{clip ? formatTimecode(clipEnd, fps) : asset ? formatTimecode(seconds(asset.duration), fps) : "00:00:00:00"}</span>
        <button onClick={() => {
          const target = clip ? startOf(clip) : 0;
          setPlaying(false); setPlayhead(target);
          if (videoRef.current) applySeek(videoRef.current, clip ? seconds(clip.sourceIn) : 0);
        }}>Reset</button></div>
    </div>
  </main>;
}

// ---------------------------------------------------------------------------
// Transition inspector — incoming and outgoing
// ---------------------------------------------------------------------------

function TransitionPicker({ label, value, duration, disabled, onChoose, onDuration, hint }: {
  label: string; value: string; duration: number; disabled: boolean;
  onChoose: (definition: TransitionDefinition) => void; onDuration: (value: number) => void; hint: string;
}) {
  const [query, setQuery] = useState("");
  const matches = filterTransitions(query);
  const selected = transitionById(value);
  return <div className="transition-side">
    <div className="transition-side-head"><strong>{label}</strong><span>{hint}</span></div>
    <label className="transition-search"><span>Find</span><input value={query} disabled={disabled} placeholder={`Search ${TRANSITIONS.length} transitions`}
      onChange={(event) => setQuery(event.target.value)} /></label>
    <label className="field select-field"><span>Transition</span><select value={selected.id} disabled={disabled}
      onChange={(event) => onChoose(transitionById(event.target.value))}>
      {TRANSITION_CATEGORIES.map((category) => {
        const options = matches.filter((item) => item.category === category);
        return options.length ? <optgroup label={category} key={category}>
          {options.map((item) => <option value={item.id} key={item.id}>{item.name}</option>)}
        </optgroup> : null;
      })}
    </select></label>
    <div className="transition-summary"><strong>{selected.name}</strong><span>{selected.category}</span><p>{selected.description}</p></div>
    <NumberControl label="Duration" value={Math.max(duration, selected.minDuration)} disabled={disabled || selected.id === "none"}
      min={selected.minDuration} max={10} step={0.05} suffix="s" onChange={(next) => onDuration(Math.max(next, selected.minDuration))} />
  </div>;
}

function TransitionControl({ clip, neighbours }: {
  clip?: TimelineClip;
  neighbours: { previous?: TimelineClip; next?: TimelineClip };
}) {
  const { updateSelectedClip, applyTransitionToAllCuts } = useEditor();
  const disabled = !clip;
  const inDefinition = transitionById(clip?.transitionIn);
  const outDefinition = transitionById(clip?.transitionOut);
  const chooseIn = (definition: TransitionDefinition) => updateSelectedClip({
    transitionIn: definition.id,
    transitionDuration: definition.id === "none" ? 0
      : inDefinition.id === "none" ? definition.defaultDuration
        : Math.max(clip?.transitionDuration ?? definition.defaultDuration, definition.minDuration),
  });
  const chooseOut = (definition: TransitionDefinition) => updateSelectedClip({
    transitionOut: definition.id,
    transitionOutDuration: definition.id === "none" ? 0
      : outDefinition.id === "none" ? definition.defaultDuration
        : Math.max(clip?.transitionOutDuration ?? definition.defaultDuration, definition.minDuration),
  });
  const headJunction = clip && neighbours.previous ? resolveJunction(neighbours.previous, clip) : undefined;
  const tailJunction = clip && neighbours.next ? resolveJunction(clip, neighbours.next) : undefined;
  return <div className="transition-control">
    <TransitionPicker label="Transition in" value={inDefinition.id} duration={clip?.transitionDuration ?? inDefinition.defaultDuration}
      disabled={disabled} onChoose={chooseIn} onDuration={(transitionDuration) => updateSelectedClip({ transitionDuration })}
      hint={neighbours.previous ? "blends from the previous clip" : "fades up from black"} />
    <TransitionPicker label="Transition out" value={outDefinition.id} duration={clip?.transitionOutDuration ?? outDefinition.defaultDuration}
      disabled={disabled} onChoose={chooseOut} onDuration={(transitionOutDuration) => updateSelectedClip({ transitionOutDuration })}
      hint={neighbours.next ? "blends into the next clip" : "fades down to black"} />
    <div className="junction-report">
      <div><b>Head cut</b><span>{!clip ? "—" : !neighbours.previous ? (inDefinition.id === "none" ? "Hard start" : `Fade up · ${inDefinition.name}`)
        : headJunction?.id === "none" ? "Hard cut" : `${transitionById(headJunction?.id).name} · ${headJunction?.duration.toFixed(2)}s`}</span></div>
      <div><b>Tail cut</b><span>{!clip ? "—" : !neighbours.next ? (outDefinition.id === "none" ? "Hard end" : `Fade down · ${outDefinition.name}`)
        : tailJunction?.id === "none" ? "Hard cut" : `${transitionById(tailJunction?.id).name} · ${tailJunction?.duration.toFixed(2)}s`}</span></div>
    </div>
    <div className="transition-actions">
      <button disabled={disabled || (inDefinition.id === "none" && outDefinition.id === "none")}
        onClick={() => applyTransitionToAllCuts(
          inDefinition.id !== "none" ? inDefinition.id : outDefinition.id,
          inDefinition.id !== "none" ? clip?.transitionDuration ?? inDefinition.defaultDuration : clip?.transitionOutDuration ?? outDefinition.defaultDuration,
        )}>Apply to all cuts</button>
      <button disabled={disabled} onClick={() => updateSelectedClip({ transitionIn: "none", transitionDuration: 0, transitionOut: "none", transitionOutDuration: 0 })}>Clear</button>
    </div>
    <small className="transition-count">
      Both sides fuse at a cut: the incoming choice sets the look and the longer duration wins, so a transition never gets truncated.
      Audio is crossfaded to match, and the first and last clips fade from and to black.
    </small>
  </div>;
}

// ---------------------------------------------------------------------------
// AI panels
// ---------------------------------------------------------------------------

function Sam2Section({ runtime }: { runtime?: AiRuntimeStatus }) {
  const {
    project, selectedClipId, busy, setBusy, setStatus, addAssets, playhead,
  } = useEditor();
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId)!;
  const clip = sequence.tracks.flatMap((track) => track.clips).find((item) => item.id === selectedClipId);
  const clipAsset = project.assets.find((asset) => asset.id === clip?.assetId);
  const [kind, setKind] = useState("matte");
  const [feather, setFeather] = useState(0);
  const [running, setRunning] = useState(false);
  const [failure, setFailure] = useState<string>();
  const points = useEditor((state) => state.maskPoints);
  const clearMaskPoints = useEditor((state) => state.clearMaskPoints);
  const setMaskPicking = useEditor((state) => state.setMaskPicking);
  const picking = useEditor((state) => state.maskPicking);

  const usable = Boolean(clip && clipAsset && clip.trackId === "v1");
  const run = async () => {
    if (!clip || !clipAsset) return;
    const outputPath = await save({
      defaultPath: clip.name.replace(/\.[^.]+$/, "") + "-" + kind + ".mp4",
      filters: [{ name: "MPEG-4 Video", extensions: ["mp4"] }],
    });
    if (!outputPath) return;
    setBusy(true); setRunning(true); setFailure(undefined);
    setStatus("SAM 2 is tracking the subject through the clip…");
    try {
      await invoke<LtxBridgeResult>("track_sam2", { request: {
        videoPath: clipAsset.path, outputPath,
        startTime: seconds(clip.sourceIn),
        endTime: seconds(clip.sourceOut),
        points: points.map((point) => ({ x: point.x, y: point.y, include: point.include })),
        outputKind: kind, feather,
      } });
      const probe = await invoke<MediaProbe>("probe_media", { path: outputPath });
      const asset: MediaAsset = {
        id: crypto.randomUUID(), name: filename(outputPath), path: outputPath, kind: "generated",
        duration: fromSeconds(Math.max(probe.durationSeconds, 0.04)), width: probe.width, height: probe.height,
        frameRate: parseRate(probe.frameRate), colorSpace: probe.colorSpace, hasAudio: false,
      };
      addAssets([asset]);
      setStatus("SAM 2 wrote " + asset.name + " into the media bin");
    } catch (error) {
      setFailure(errorText(error));
      setStatus("SAM 2 tracking failed: " + errorHeadline(error));
    } finally { setRunning(false); setBusy(false); }
  };

  return <Section title="SAM 2 subject tracking" accent={runtime?.sam2Ready ? "READY" : "UNAVAILABLE"}
    hint="Click a subject in the program monitor and SAM 2 follows it through the clip, producing a matte, a cutout, or the inverse.">
    <p className="ai-note">Select a V1 clip, arm picking, then click the subject in the monitor. Shift-click to exclude a region. The result lands in the media bin as a new asset.</p>
    {!usable && <div className="library-notice"><Layers3 size={22} /><strong>Select a V1 clip</strong><span>Tracking runs over the selected clip's trimmed range.</span></div>}
    <div className="mask-points">
      <button className={picking ? "arm active" : "arm"} disabled={!usable || running} onClick={() => setMaskPicking(!picking)}
        title="While armed, clicking the program monitor adds a tracking point. Shift-click adds an exclude point.">
        {picking ? "Picking — click the monitor" : "Add points from monitor"}
      </button>
      <span>{points.length} point{points.length === 1 ? "" : "s"}{points.some((p) => !p.include) ? " (some excluding)" : ""}</span>
      <button disabled={!points.length || running} onClick={clearMaskPoints} title="Remove every tracking point and start again.">Clear</button>
    </div>
    <SelectControl label="Output" value={kind} disabled={running} onChange={setKind}
      hint="Matte is a black-and-white mask for keying. Cutout isolates the subject on black. Removed blacks the subject out, for clean-plate work."
      options={[{ value: "matte", label: "Matte · black & white mask" }, { value: "cutout", label: "Cutout · subject on black" }, { value: "removed", label: "Removed · subject blacked out" }]} />
    <RangeControl label="Feather" value={feather} disabled={running} min={0} max={32} suffix="px"
      hint="Softens the mask edge. A little feather hides aliasing when compositing." onChange={setFeather} />
    <button className="render-button" disabled={!usable || running || busy || !runtime?.sam2Ready || !points.length} onClick={() => { void run(); }}>
      <Layers3 size={16} /> {running ? "Tracking…" : "Track subject + add to bin"}
    </button>
    {running && <div className="ltx-job"><span className="ltx-spinner" /><div><strong>SAM 2 running</strong><small>Propagating the selection frame by frame.</small></div></div>}
    {failure && <div className="worker-failure"><div><strong>SAM 2 failed</strong><button onClick={() => setFailure(undefined)} aria-label="Dismiss"><X size={12} /></button></div><pre>{failure}</pre></div>}
    {runtime && !runtime.sam2Ready && <small className="runtime-checks">SAM 2 state: {runtime.sam2State}. Run scripts\setup-image-studio.ps1 to install the isolated runtime and package.</small>}
  </Section>;
}

function AiPanel({ active }: { active: boolean }) {
  const {
    project, busy, setBusy, setStatus, addAssets, addAssetToTimeline, addGeneratedAvPair,
    insertGeneratedAvPairAfterClip, insertAssetAfterClip, setClipEditorOpen, selectedClipId,
  } = useEditor();
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId)!;
  const selectedClip = sequence.tracks.flatMap((track) => track.clips).find((item) => item.id === selectedClipId);
  const selectedClipAsset = project.assets.find((asset) => asset.id === selectedClip?.assetId);
  // The enabled V1 clips in timeline order, flattened to seconds. Both editor workflows read the
  // timeline selection rather than a dropdown, so this is the whole input to their eligibility.
  const v1Clips = (sequence.tracks.find((track) => track.id === "v1")?.clips ?? [])
    .filter((item) => item.enabled)
    .slice().sort((a, b) => seconds(a.start) - seconds(b.start));
  const frameSourceClips = v1Clips.map((clip) => ({
    id: clip.id, assetId: clip.assetId,
    startSeconds: seconds(clip.start),
    endSeconds: seconds(clip.start) + timelineClipDuration(clip),
  }));
  const assetFor = (assetId: string) => project.assets.find((asset) => asset.id === assetId);
  const onV1 = selectedClip?.trackId === "v1";
  // These now report *why* they are unavailable. The notice used to say "Select the outgoing V1
  // clip" whether nothing was selected, a caption was, the selection was the last clip on the
  // track, or the next clip was a still — so a user who had selected a clip was told to select a
  // clip. See `resolveExtendClip` / `resolveBridgePair`.
  const extendResult = resolveExtendClip(frameSourceClips, assetFor, selectedClipId, onV1);
  const bridgeResult = resolveBridgePair(frameSourceClips, assetFor, selectedClipId, onV1);
  const extendBlocker = "blocker" in extendResult ? extendResult.blocker : undefined;
  const bridgeBlocker = "blocker" in bridgeResult ? bridgeResult.blocker : undefined;
  const extendableClip = "blocker" in extendResult ? undefined
    : { clip: v1Clips.find((item) => item.id === extendResult.clip.id)!, asset: extendResult.asset };
  const bridgePair = "blocker" in bridgeResult ? undefined : {
    from: v1Clips.find((item) => item.id === bridgeResult.from.id)!,
    fromAsset: bridgeResult.fromAsset,
    to: v1Clips.find((item) => item.id === bridgeResult.to.id)!,
    toAsset: bridgeResult.toAsset,
    gapSeconds: bridgeResult.gapSeconds,
  };
  const images = project.assets.filter((asset) => asset.kind === "image");
  const videoAssets = project.assets.filter((asset) => asset.kind === "video" || asset.kind === "generated");
  const h3ReferenceAssets = project.assets.filter((asset) =>
    isAudioAsset(asset) || ["image", "video", "generated"].includes(asset.kind));
  const [runtime, setRuntime] = useState<AiRuntimeStatus>();
  // Which generator the panel is driving. H3 is a different model with a different geometry
  // contract, so the fields below swap rather than being reinterpreted.
  const [engine, setEngine] = useState<"ltx" | "h3">("ltx");
  const [ltxMode, setLtxMode] = useState("first-last");
  const [h3Mode, setH3Mode] = useState("text");
  const [h3Aspect, setH3Aspect] = useState("16:9");
  const [h3ShortEdge, setH3ShortEdge] = useState(H3_LOW_VRAM_SHORT_EDGE);
  const [h3Frames, setH3Frames] = useState(H3_MIN_FRAMES);
  /** Bridge length: take it from the hole on the timeline, or set it and push the rest along. */
  const [h3BridgeLength, setH3BridgeLength] = useState<"gap" | "custom">("gap");
  const [h3Steps, setH3Steps] = useState(50);
  const [h3SamplingMode, setH3SamplingMode] = useState<"base" | "turbo-4" | "turbo-8">("turbo-8");
  const [h3AttentionBackend, setH3AttentionBackend] = useState<"native" | "sage">("sage");
  // H3 does not share LTX's cpu/disk axis, and sharing the state would carry an LTX
  // value across when the model selector is switched.
  const [h3Offload, setH3Offload] = useState("balanced");
  const [h3RefQuality, setH3RefQuality] = useState<H3RefImageDetail>(H3_REF_DEFAULTS.refImageDetail);
  const [h3ReferenceAssetId, setH3ReferenceAssetId] = useState("");
  const [h3References, setH3References] = useState<H3ReferenceDraft[]>([]);
  const [h3ReferenceBrief, setH3ReferenceBrief] = useState("");
  const [h3ReferencePrompt, setH3ReferencePrompt] = useState("");
  const [firstId, setFirstId] = useState("");
  const [lastId, setLastId] = useState("");
  const [sourceClipId, setSourceClipId] = useState("");
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [resolution, setResolution] = useState("768x512");
  const [frames, setFrames] = useState(121);
  const [seed, setSeed] = useState(42);
  const [guideStrength, setGuideStrength] = useState(0.8);
  const [cfgScale, setCfgScale] = useState(1);
  const [audioCfgScale, setAudioCfgScale] = useState(1.5);
  const [loraStrength, setLoraStrength] = useState(1);
  const [referenceStrength, setReferenceStrength] = useState(1);
  const [offload, setOffload] = useState("cpu");
  const [ltxAttention, setLtxAttention] = useState<"native" | "sage">("native");
  const [generating, setGenerating] = useState(false);
  const [startedAt, setStartedAt] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [lastResult, setLastResult] = useState<LtxBridgeResult>();
  // The status bar only ever gets a single truncated line, which is useless for diagnosing a
  // worker that died. Keep the whole tail here, the way the clip editor already does.
  const [failure, setFailure] = useState<string>();
  // The hole between the two clips, and what filling it would cost. H3 only produces 17n+5 frames
  // between 5 and 15 seconds, so a gap is *snapped* to a legal count rather than matched exactly —
  // `placement` then says whether that still fills the gap and how far the rest of the timeline
  // has to move. A gap smaller than the 5-second floor cannot be filled exactly by this model.
  const bridgeGap = bridgePair?.gapSeconds ?? 0;
  const hasBridgeGap = bridgeGap > ADJACENT_TOLERANCE;
  const useBridgeGap = Boolean(bridgePair) && h3BridgeLength === "gap" && hasBridgeGap;
  const bridgeFrames = useBridgeGap
    ? clampH3Frames(Math.round(bridgeGap * H3_FPS)) : clampH3Frames(h3Frames);
  const bridgePlacement = planBridgePlacement(bridgeGap, h3Seconds(bridgeFrames));
  const clipExtendDimensions = fitLtxDimensions(extendableClip?.asset.width, extendableClip?.asset.height);
  const bridgeDimensions = fitLtxDimensions(bridgePair?.fromAsset.width, bridgePair?.fromAsset.height);
  useEffect(() => {
    void runtimeStatus().then(setRuntime).catch((error) => {
      setStatus("AI runtime check failed: " + errorText(error));
    });
  }, [setStatus]);
  /**
   * Keep the frame pickers pointing at assets that still exist.
   *
   * Defaulting an empty id was only half of it. An id whose asset has since been removed from the
   * bin -- or that came from a different project, or from an asset whose kind changed -- stayed in
   * state forever. A `<select>` whose `value` matches no `<option>` gets `selectedIndex = -1` and
   * renders blank in Chromium, so the control looks unset while the state is not, and `generateH3`
   * then fails a guard telling the user to choose the image they believe they already chose.
   *
   * The clip picker below carried a comment about the sibling half of exactly this fault -- "the
   * select displays the first video while the state stays empty" -- so the shape was known; only
   * the empty case was handled. Both halves now go through the same rule: a selection survives if
   * the asset is still there, and is replaced if it is not.
   *
   * `lastId` falls back to the first asset that is not `firstId`, because first-last needs two
   * distinct frames and `images[1]` is only that by accident.
   */
  useEffect(() => {
    const resolved = resolveFramePickerIds(images, firstId, lastId);
    if (resolved.first !== firstId) setFirstId(resolved.first);
    if (resolved.last !== lastId) setLastId(resolved.last);
  }, [images, firstId, lastId]);
  useEffect(() => {
    const resolved = resolvePickerId(videoAssets, sourceClipId);
    if (resolved !== sourceClipId) setSourceClipId(resolved);
  }, [videoAssets, sourceClipId]);
  useEffect(() => {
    const resolved = resolvePickerId(h3ReferenceAssets, h3ReferenceAssetId);
    if (resolved !== h3ReferenceAssetId) setH3ReferenceAssetId(resolved);
  }, [h3ReferenceAssets, h3ReferenceAssetId]);
  useEffect(() => {
    // Projects imported before the album-art fix can already contain an MP3 tagged as video.
    // Correct any live Ref2VA tray entry without requiring the user to remove and re-add it.
    setH3References((current) => {
      let changed = false;
      const next = current.map((reference) => {
        const asset = project.assets.find((item) => item.id === reference.assetId);
        if (!asset) return reference;
        const kind: H3ReferenceKind = isAudioAsset(asset) ? "audio" : asset.kind === "image" ? "image" : "video";
        const includeEmbeddedAudio = kind === "video" ? reference.includeEmbeddedAudio && Boolean(asset.hasAudio) : true;
        const allowed = availableH3ReferenceIntents(kind, kind === "audio" || includeEmbeddedAudio);
        const requested = kind === reference.kind ? reference.intents : defaultH3ReferenceIntents(kind);
        const intents = requested.filter((intent) => allowed.includes(intent));
        if (!intents.length) intents.push(...defaultH3ReferenceIntents(kind));
        if (kind === reference.kind && includeEmbeddedAudio === reference.includeEmbeddedAudio
          && intents.length === reference.intents.length && intents.every((intent, index) => intent === reference.intents[index])) return reference;
        changed = true;
        return { ...reference, kind, intents, includeEmbeddedAudio };
      });
      return changed ? next : current;
    });
  }, [project.assets]);
  useEffect(() => {
    if (!generating) return;
    const update = () => setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    update();
    const timer = window.setInterval(update, 500);
    return () => window.clearInterval(timer);
  }, [generating, startedAt]);
  const generate = async () => {
    const first = images.find((asset) => asset.id === firstId);
    const last = images.find((asset) => asset.id === lastId);
    const sourceClip = videoAssets.find((asset) => asset.id === sourceClipId);
    if (ltxMode === "first-last" && (!first || !last || first.id === last.id)) {
      setStatus("Choose two different conditioning images");
      return;
    }
    if (ltxMode === "image" && !first) {
      setStatus("Choose a conditioning image");
      return;
    }
    if (ltxMode === "video-audio" && !sourceClip) {
      setStatus("Choose a source clip to score");
      return;
    }
    if (ltxMode === "clip-extend" && !extendableClip) {
      setStatus(extendBlocker ?? "Select an enabled V1 clip before using Clip extend");
      return;
    }
    if (ltxMode === "clip-to-clip" && !bridgePair) {
      setStatus("Select a V1 video clip that has another video clip after it");
      return;
    }
    const outputPath = await save({
      // The prompt is the only thing that makes one generation recognisable from another, so it
      // names the file. Without one this falls back to a timestamp, as it always did.
      defaultPath: suggestedFileName(ltxMode === "video-audio" ? "LTX-scored"
        : ltxMode === "clip-extend" ? "LTX-extension"
        : ltxMode === "clip-to-clip" ? "LTX-bridge"
        : "LTX-" + ltxMode, prompt, "mp4"),
      filters: [{ name: "MPEG-4 Video", extensions: ["mp4"] }],
    });
    if (!outputPath) return;
    const [chosenWidth, chosenHeight] = resolution.split("x").map(Number);
    const width = ltxMode === "clip-extend" ? clipExtendDimensions.width
      : ltxMode === "clip-to-clip" ? bridgeDimensions.width : chosenWidth;
    const height = ltxMode === "clip-extend" ? clipExtendDimensions.height
      : ltxMode === "clip-to-clip" ? bridgeDimensions.height : chosenHeight;
    setBusy(true);
    setGenerating(true);
    setStartedAt(Date.now());
    setElapsed(0);
    setLastResult(undefined);
    setFailure(undefined);
    const usedSeed = resolveSeed(seed);
    setStatus((ltxMode === "video-audio"
      ? "LTX is scoring the clip; only audio is generated and the picture is copied through"
      : ltxMode === "clip-to-clip"
        ? "LTX is generating the bridge between the two clips"
        : "LTX is generating a single-pass clip") + ` (seed ${usedSeed})…`);
    try {
      const result = ltxMode === "clip-extend" && extendableClip
        ? await invoke<LtxBridgeResult>("extend_clip_ltx", { request: {
          videoPath: extendableClip.asset.path,
          sourceInSeconds: seconds(extendableClip.clip.sourceIn),
          sourceOutSeconds: seconds(extendableClip.clip.sourceOut),
          negativePrompt, outputPath, prompt, width, height, numFrames: frames, frameRate: 24,
          seed: usedSeed, guideStrength, cfgScale, audioCfgScale, offload,
          attentionBackend: ltxAttention,
        } })
        : ltxMode === "clip-to-clip" && bridgePair
        ? await invoke<LtxBridgeResult>("bridge_clips_ltx", { request: {
          fromVideoPath: bridgePair.fromAsset.path,
          fromSourceInSeconds: seconds(bridgePair.from.sourceIn),
          fromSourceOutSeconds: seconds(bridgePair.from.sourceOut),
          toVideoPath: bridgePair.toAsset.path,
          toSourceInSeconds: seconds(bridgePair.to.sourceIn),
          negativePrompt, outputPath, prompt, width, height, numFrames: frames, frameRate: 24,
          seed: usedSeed, guideStrength, cfgScale, audioCfgScale, offload,
          attentionBackend: ltxAttention,
        } })
        : await invoke<LtxBridgeResult>("generate_ltx_bridge", { request: {
          mode: ltxMode,
          firstFrame: ltxMode === "first-last" || ltxMode === "image" ? first?.path ?? null : null,
          lastFrame: ltxMode === "first-last" ? last?.path ?? null : null,
          videoPath: ltxMode === "video-audio" ? sourceClip?.path ?? null : null,
          negativePrompt,
          outputPath, prompt,
          width, height, numFrames: frames, frameRate: 24,
          seed: usedSeed, guideStrength, cfgScale, audioCfgScale, loraStrength, referenceStrength, offload,
          attentionBackend: ltxAttention,
        } });
      setLastResult(result);
      const probe = await probeMedia(outputPath);
      const generated = generatedAsset(outputPath, probe);
      if (ltxMode === "clip-extend" && extendableClip) {
        insertAssetAfterClip(extendableClip.clip.id, generated);
        setStatus("Extended " + extendableClip.clip.name + " and ripple-inserted " + generated.name);
      } else if (ltxMode === "clip-to-clip" && bridgePair) {
        // Inserting after the outgoing clip ripples the incoming clip and everything past it, so
        // the bridge lands exactly at the cut it was generated for.
        insertAssetAfterClip(bridgePair.from.id, generated);
        setStatus("Bridged " + bridgePair.from.name + " → " + bridgePair.to.name + " with " + generated.name);
      } else {
        addAssets([generated]);
        addAssetToTimeline(generated.id);
        setStatus("Generated and placed " + generated.name + " on V1");
      }
    } catch (error) {
      // Keep the full worker tail in the panel; the status bar only gets a single line.
      setFailure(errorText(error));
      setStatus("LTX generation failed: " + errorHeadline(error));
    } finally {
      setGenerating(false);
      setBusy(false);
    }
  };
  // A keyframe decides the canvas: H3 stretches the first frame onto it, so resolving from the
  // aspect picker instead would silently distort every image-conditioned generation. `last-only`
  // has no first frame, so there the LAST frame is the one that sets it.
  const h3CanvasSource = (() => {
    if (h3Mode === "text" || h3Mode === "reference") return undefined;
    if (h3Mode === "clip-extend" || h3Mode === "clip-to-clip") {
      return extendableClip?.asset.width && extendableClip.asset.height ? extendableClip.asset : undefined;
    }
    const preferred = h3Mode === "last-only" ? lastId : firstId;
    const asset = images.find((item) => item.id === preferred);
    return asset?.width && asset?.height ? asset : undefined;
  })();
  // The dedicated Draft adapter was distilled at 544p mixed aspect ratios. Keep Draft on that
  // canvas rather than offering a 768p control whose distribution it was not trained for.
  const h3EffectiveShortEdge = h3Mode === "reference" || h3SamplingMode === "turbo-4" ? H3_LOW_VRAM_SHORT_EDGE : h3ShortEdge;
  const h3Dimensions = (() => {
    if (h3CanvasSource) return fitH3Dimensions(h3CanvasSource.width, h3CanvasSource.height, h3EffectiveShortEdge);
    const [left, right] = h3Aspect.split(":").map(Number);
    return fitH3Dimensions(left, right, h3EffectiveShortEdge);
  })();
  const h3EffectiveSteps = h3Mode === "reference" ? H3_REF_DEFAULTS.steps
    : h3SamplingMode === "turbo-4" ? 4 : h3SamplingMode === "turbo-8" ? 8 : h3Steps;
  const h3SamplingReady = h3Mode === "reference" ? Boolean(runtime?.minimaxH3RefReady)
    : h3SamplingMode === "turbo-4" ? Boolean(runtime?.minimaxH3DraftReady)
    : h3SamplingMode === "turbo-8" ? Boolean(runtime?.minimaxH3TurboReady) : true;
  const h3AssetInfo = (assetId: string) => {
    const asset = project.assets.find((item) => item.id === assetId);
    return asset ? { name: asset.name, hasAudio: Boolean(asset.hasAudio) } : undefined;
  };
  const h3RefLimits = h3ReferenceLimits(h3References, h3AssetInfo);
  // Interim adapter for the six-field builder. The brief becomes the single shot's description
  // until the Ref2V workspace lands its own style / shot-list / soundscape fields, at which point
  // this whole block moves out of AiPanel.
  const localH3ReferencePrompt = buildH3ReferencePrompt({
    style: "",
    summary: "",
    shots: [{ id: "shot-1", timecode: "", description: h3ReferenceBrief, dialogue: [] }],
    soundscape: "",
    music: "",
    references: h3References,
    assetInfo: (assetId: string) => {
      const asset = project.assets.find((item) => item.id === assetId);
      return { name: asset?.name ?? "missing asset", hasAudio: Boolean(asset?.hasAudio) };
    },
  });
  const effectiveH3ReferencePrompt = h3ReferencePrompt.trim() || localH3ReferencePrompt;
  const h3ReferenceInputs = h3References.map((reference) => ({
    ...reference,
    asset: project.assets.find((asset) => asset.id === reference.assetId),
  }));
  const h3ReferencePreflight = (() => {
    if (h3Mode !== "reference") return undefined;
    if (!effectiveH3ReferencePrompt.trim()) return "Describe the Ref2VA shot or build a structured prompt.";
    if (!h3References.length) return "Add at least one image or video reference.";
    if (h3ReferenceInputs.some((item) => !item.asset)) return "Remove the missing reference asset from the tray.";
    // The model card's caps and the checkpoint's pairing rule, mirrored from the Rust validator.
    // These used to be the old conservative 3/1/1/4 profile, which the backend stopped enforcing
    // when the caps were raised -- so the panel was refusing trays the worker would have accepted.
    const capError = h3ReferenceCapError(h3RefLimits);
    if (capError) return capError;
    const timed = h3ReferenceInputs.filter((item) => item.kind !== "image");
    const invalid = timed.find((item) => item.asset && (seconds(item.asset.duration) < 2 || seconds(item.asset.duration) > 15));
    if (invalid?.asset) return `${invalid.asset.name} is ${seconds(invalid.asset.duration).toFixed(1)}s; every video and audio reference must be 2–15 seconds.`;
    const videoSeconds = timed.reduce((total, item) => total + (item.kind === "video" && item.asset ? seconds(item.asset.duration) : 0), 0);
    const audioSeconds = timed.reduce((total, item) => total + (item.asset && (item.kind === "audio"
      || (item.kind === "video" && item.includeEmbeddedAudio && item.asset.hasAudio)) ? seconds(item.asset.duration) : 0), 0);
    if (videoSeconds > 15 || audioSeconds > 15) {
      return `Reference totals must stay within 15 seconds per modality (video ${videoSeconds.toFixed(1)}s, audio ${audioSeconds.toFixed(1)}s).`;
    }
    return undefined;
  })();
  const addH3Reference = () => {
    const asset = h3ReferenceAssets.find((item) => item.id === h3ReferenceAssetId);
    if (!asset || h3References.some((item) => item.assetId === asset.id)) return;
    const kind: H3ReferenceKind = isAudioAsset(asset) ? "audio" : asset.kind === "image" ? "image" : "video";
    const draft: H3ReferenceDraft = {
      id: crypto.randomUUID(), assetId: asset.id, kind,
      intents: defaultH3ReferenceIntents(kind), includeEmbeddedAudio: kind === "video" ? Boolean(asset.hasAudio) : true,
    };
    // Refuse the add itself only on the hard caps -- the pairing rule is a property of the whole
    // tray and belongs in the preflight, where the user can fix it by muting a soundtrack rather
    // than being blocked from adding the visual that would satisfy it.
    const next = h3ReferenceLimits([...h3References, draft], h3AssetInfo);
    if (next.total > H3_MAX_REFERENCES || next.images > H3_MAX_IMAGES
        || next.videos > H3_MAX_VIDEOS || next.effectiveAudios > H3_MAX_AUDIOS) {
      setStatus(
        `Ref2VA reference limits: ${H3_MAX_IMAGES} images, ${H3_MAX_VIDEOS} videos, `
        + `${H3_MAX_AUDIOS} audio (a video's own soundtrack counts), ${H3_MAX_REFERENCES} total`
      );
      return;
    }
    setH3References((current) => [...current, draft]);
    setH3ReferencePrompt("");
  };
  const moveH3Reference = (index: number, direction: -1 | 1) => {
    const target = index + direction;
    if (target < 0 || target >= h3References.length) return;
    setH3References((current) => {
      const next = current.slice();
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
    setH3ReferencePrompt("");
  };
  const generateH3 = async () => {
    const first = images.find((asset) => asset.id === firstId);
    const last = images.find((asset) => asset.id === lastId);
    const referenceInputs = h3ReferenceInputs;
    if (h3Mode === "reference") {
      if (h3ReferencePreflight) { setStatus("Ref2VA needs attention: " + h3ReferencePreflight); return; }
    }
    const frameName = h3Mode === "image" ? "source image" : "first frame";
    if ((h3Mode === "image" || h3Mode === "first-last") && !first) {
      setStatus(firstId
        ? `The ${frameName} that was selected is no longer in the media bin — choose another.`
        : images.length ? `Choose a ${frameName}` : "Import an image to condition on first");
      return;
    }
    if ((h3Mode === "last-only" || h3Mode === "first-last") && !last) {
      setStatus(lastId
        ? "The last frame that was selected is no longer in the media bin — choose another."
        : images.length > 1 || h3Mode === "last-only"
          ? "Choose the frame the clip should end on"
          : "First and last frame needs two images in the bin; import or generate a second one.");
      return;
    }
    if (h3Mode === "first-last" && first && last && first.id === last.id) {
      setStatus("Choose two different conditioning images");
      return;
    }
    if (h3Mode === "clip-extend" && !extendableClip) {
      setStatus(extendBlocker ?? "Select an enabled V1 clip before using Clip to video");
      return;
    }
    if (h3Mode === "clip-to-clip" && !bridgePair) {
      setStatus(bridgeBlocker ?? "Select a V1 clip that has another enabled clip after it");
      return;
    }
    setBusy(true);
    setFailure(undefined);
    setStatus("Choose where to save the generated H3 clip…");
    let outputPath: string | null;
    try {
      outputPath = await save({
        defaultPath: suggestedFileName("H3-" + h3Mode, prompt, "mp4"),
        filters: [{ name: "MPEG-4 Video", extensions: ["mp4"] }],
      });
    } catch (error) {
      setFailure(errorText(error));
      setStatus("Could not open the H3 output picker: " + errorHeadline(error));
      setBusy(false);
      return;
    }
    if (!outputPath) {
      setStatus("H3 generation canceled before it started");
      setBusy(false);
      return;
    }
    // Clip-to-clip owns its own length: it may be filling a measured hole rather than using the
    // Frames control, and the two must never diverge from what the notice just promised.
    const numFrames = h3Mode === "clip-to-clip" ? bridgeFrames : clampH3Frames(h3Frames);
    setGenerating(true);
    setStartedAt(Date.now());
    setElapsed(0);
    setLastResult(undefined);
    setFailure(undefined);
    const usedSeed = resolveSeed(seed);
    const speedLabel = h3Mode === "reference" ? ` Ref2VA at ${H3_REF_DEFAULTS.steps} NFE Base`
      : h3SamplingMode === "turbo-4" ? " with LightX2V 4-NFE Draft"
      : h3SamplingMode === "turbo-8" ? " with LightX2V 8-NFE Turbo" : "";
    const attentionLabel = h3AttentionBackend === "sage" ? " + Sage attention" : "";
    setStatus(`MiniMax-H3${speedLabel}${attentionLabel} is generating ${h3Seconds(numFrames).toFixed(1)}s of video and audio (seed ${usedSeed})…`);
    try {
      const result = h3Mode === "reference"
        // The same builder the Ref2V workspace uses. This branch used to send no sampling fields
        // at all and inherit whatever Rust defaulted to, which is how the two paths ended up
        // running different recipes against the same checkpoint.
        ? await invoke<MiniMaxH3Result>("generate_minimax_h3_ref", { request: buildH3RefRequest({
          prompt: effectiveH3ReferencePrompt,
          references: referenceInputs.map((item) => ({
            kind: item.kind, path: item.asset!.path, includeEmbeddedAudio: item.includeEmbeddedAudio,
          })),
          outputPath,
          width: h3Dimensions.width, height: h3Dimensions.height,
          numFrames, seed: usedSeed, attentionBackend: h3AttentionBackend, offload: h3Offload,
          refImageDetail: h3RefQuality,
        }) })
        : await invoke<MiniMaxH3Result>("generate_minimax_h3", { request: {
        mode: h3Mode,
        firstFrame: h3Mode === "image" || h3Mode === "first-last" ? first?.path ?? null : null,
        lastFrame: h3Mode === "last-only" || h3Mode === "first-last" ? last?.path ?? null : null,
        videoPath: h3Mode === "clip-extend" || h3Mode === "clip-to-clip" ? extendableClip?.asset.path ?? null : null,
        sourceInSeconds: h3Mode === "clip-extend" || h3Mode === "clip-to-clip" ? seconds(extendableClip!.clip.sourceIn) : null,
        sourceOutSeconds: h3Mode === "clip-extend" || h3Mode === "clip-to-clip" ? seconds(extendableClip!.clip.sourceOut) : null,
        toVideoPath: h3Mode === "clip-to-clip" ? bridgePair?.toAsset.path ?? null : null,
        toSourceInSeconds: h3Mode === "clip-to-clip" ? seconds(bridgePair!.to.sourceIn) : null,
        prompt, outputPath, audioPath: null,
        width: h3Dimensions.width, height: h3Dimensions.height,
        numFrames, steps: h3EffectiveSteps, seed: usedSeed,
        samplingMode: h3SamplingMode, attentionBackend: h3AttentionBackend, offload: h3Offload,
      } });
      setLastResult({ outputPath: result.outputPath, log: result.log });
      const [videoProbe, audioProbe] = await Promise.all([
        probeMedia(result.outputPath), probeMedia(result.audioPath),
      ]);
      const video = generatedAsset(result.outputPath, videoProbe);
      const audio = generatedAsset(result.audioPath, audioProbe, "audio");
      addAssets([video, audio]);
      if ((h3Mode === "clip-extend" || h3Mode === "clip-to-clip") && extendableClip) {
        insertGeneratedAvPairAfterClip(extendableClip.clip.id, video.id, audio.id);
      } else {
        addGeneratedAvPair(video.id, audio.id);
      }
    } catch (error) {
      setFailure(errorText(error));
      setStatus("MiniMax-H3 generation failed: " + errorHeadline(error));
    } finally {
      setGenerating(false);
      setBusy(false);
    }
  };
  // Keep this component mounted while another inspector tab is visible. Generation owns an async
  // worker session plus prompt/settings/timer state; conditionally removing the component used to
  // throw all of that state away even though the worker itself continued running. `hidden` removes
  // it from layout without ending the React session. A real app relaunch still mounts fresh defaults.
  return <div className="inspector-content" hidden={!active}>
    <div className="ai-heading"><span><Sparkles size={17} /></span><div><strong>Local AI runtime</strong><small>{runtime?.profile ?? "Checking local runtime…"}</small></div><b className={runtime?.singlePassReady ? "ready" : "not-ready"}>{runtime?.singlePassReady ? "Ready" : "Incomplete"}</b></div>
    <Section title="Clip editor" accent={kiwiCopy(runtime?.kiwiEditState).pill.toUpperCase()}>
      <p className="ai-note">Kiwi-Edit rewrites a range of an existing timeline clip from a plain-language instruction, optionally guided by a reference image. The result drops straight back onto V1.</p>
      <button className="render-button" disabled={!selectedClipId || busy} onClick={() => setClipEditorOpen(true)}>
        <Clapperboard size={16} /> Open the clip editor
      </button>
      <small className="runtime-checks">Select a V1 clip first. Ctrl+K opens the editor from anywhere.</small>
    </Section>
    <Sam2Section runtime={runtime} />
    <Section title="Clip generation" accent={(engine === "h3" ? runtime?.minimaxH3Ready : runtime?.singlePassReady) ? "READY" : "INCOMPLETE"}>
      <SelectControl label="Model" value={engine} disabled={busy} onChange={(value) => setEngine(value as "ltx" | "h3")} options={[
        { value: "ltx", label: "LTX 2.3 distilled · 8n+1 frames" },
        { value: "h3", label: "MiniMax-H3 INT8 ConvRot · joint video + audio" },
      ]} hint="Two different models with different geometry rules, so the fields below change with this. LTX is the validated 16 GB path; H3 generates its own synchronized soundtrack." />
      {/* `runtime` is undefined until the status call returns, and h3Copy falls through to
          "environment not built" for undefined -- so gate on it or the panel flashes a
          scary warning on every mount. */}
      {engine === "h3" && runtime && h3Copy(runtime.minimaxH3State).blocker && <div className="worker-failure">
        <div><strong>{h3Copy(runtime.minimaxH3State).title}</strong></div>
        <pre>{h3Copy(runtime.minimaxH3State).detail}</pre>
      </div>}
      {engine === "h3" ? <>
        <SelectControl label="Mode" value={h3Mode} disabled={busy} onChange={setH3Mode} options={[
          { value: "text", label: "Text → video + audio" },
          { value: "image", label: "Image → video + audio" },
          { value: "first-last", label: "First + last frame → video + audio" },
          { value: "last-only", label: "Generate up to a final frame" },
          { value: "clip-extend", label: "Clip → video + audio · continue last frame" },
          { value: "clip-to-clip", label: "Clip → clip + audio · bridge the cut" },
          { value: "reference", label: "References → video + audio · Ref2VA" },
        ]} hint="Clip → video extracts the selected clip's last visible frame. Clip → clip also extracts the following clip's first visible frame. H3 is guidance-distilled and always generates a soundtrack." />
        <p className="ai-note">MiniMax-H3 generates picture and sound in one denoising loop, so the soundtrack is locked to the action rather than scored afterwards. The result lands as two linked clips — video on V1, its 32 kHz stereo audio on A1 — so the generated sound can be muted, replaced or mixed without touching the picture. Guidance is baked into the weights: there is no CFG and no “avoid” prompt to set.</p>
        {h3Mode === "clip-extend" && <p className="ai-note">Uses the selected V1 clip's exact last visible frame as H3's first frame, removes the duplicated boundary frame and matching audio, then ripple-inserts the generated pair immediately after it.</p>}
        {h3Mode === "clip-to-clip" && <p className="ai-note">Uses the selected clip's last visible frame and the following clip's first visible frame as H3's first/last constraints. Both duplicated boundaries are removed before the synchronized pair is ripple-inserted at the cut.</p>}
        {h3Mode === "clip-extend" && <div className="library-notice"><Clapperboard size={20} />
          <strong>{extendableClip ? extendableClip.clip.name : "Nothing to continue from"}</strong>
          <span>{extendableClip ? `Continue from source frame ${seconds(extendableClip.clip.sourceOut).toFixed(3)}s · ${h3Dimensions.width} × ${h3Dimensions.height}` : extendBlocker}</span>
        </div>}
        {h3Mode === "clip-to-clip" && <div className="library-notice"><Clapperboard size={20} />
          <strong>{bridgePair ? `${bridgePair.from.name} → ${bridgePair.to.name}` : "Nothing to bridge"}</strong>
          <span>{bridgePair
            ? `${bridgeFrames} frames · ${h3Seconds(bridgeFrames).toFixed(2)}s · ${h3Dimensions.width} × ${h3Dimensions.height}. `
              + (hasBridgeGap ? `The ${bridgeGap.toFixed(2)}s gap ` : "The clips touch, so ")
              + (bridgePlacement.rippleSeconds <= 0.0001
                ? bridgePlacement.fillsGap ? "is filled exactly; nothing downstream moves."
                  : `absorbs it, leaving ${(bridgeGap - h3Seconds(bridgeFrames)).toFixed(2)}s still open.`
                : `${hasBridgeGap ? "takes part of it and " : ""}${bridgePair.to.name} and everything after move ${bridgePlacement.rippleSeconds.toFixed(2)}s later.`)
            : bridgeBlocker}</span>
        </div>}
        {h3Mode === "clip-to-clip" && bridgePair && <SelectControl label="Bridge length"
          value={hasBridgeGap ? h3BridgeLength : "custom"} disabled={busy || !hasBridgeGap}
          onChange={(value) => setH3BridgeLength(value as "gap" | "custom")}
          options={hasBridgeGap ? [
            { value: "gap", label: `Fill the ${bridgeGap.toFixed(2)}s gap · ${clampH3Frames(Math.round(bridgeGap * H3_FPS))} frames` },
            { value: "custom", label: "Set the duration below" },
          ] : [{ value: "custom", label: "Set the duration below · the clips already touch" }]}
          hint={hasBridgeGap
            ? "H3 only generates 17n+5 frames between 5 and 15 seconds, so the gap is snapped to the nearest legal count. Anything longer than the gap pushes the following clips along by the overflow only."
            : "There is no gap to fill, so the bridge is inserted at the cut and everything after it moves by its full length."} />}
        {h3Mode === "reference" ? <div className="library-notice"><Layers3 size={20} />
          <strong>Ref2VA · published INT8 ConvRot · 20-NFE Base</strong>
          <span>Dedicated community checkpoint, official MiniMax components, fixed 544p canvas. FL2VA remains isolated.</span>
        </div> : <SelectControl label="Sampling" value={h3SamplingMode} disabled={busy}
          onChange={(value) => setH3SamplingMode(value as "base" | "turbo-4" | "turbo-8")} options={[
            { value: "turbo-8", label: "LightX2V Turbo v1.0 · fixed 8 NFE · Recommended" },
            { value: "base", label: "Base H3 · 50 NFE quality reference" },
            { value: "turbo-4", label: "LightX2V Draft v0.1 · fixed 4 NFE, 544p" },
          ]} hint="Draft and Turbo load separate official LightX2V FL2VA adapters. Draft is fixed to its trained 544p canvas and four evaluations; Turbo uses eight. Both use 12/3 video/audio scheduler shifts. Base H3 is unchanged." />}
        {!h3SamplingReady && runtime && <div className="worker-failure"><div><strong>{h3Mode === "reference" ? `Optional Ref2VA pack is not ready (${runtime.minimaxH3RefState})` : "Selected LightX2V adapter is missing"}</strong></div>
          <pre>{h3Mode === "reference"
            ? runtime.minimaxH3RefState === "encoder-missing"
              ? "Run: .venv\\Scripts\\python.exe scripts\\models.py install minimax-h3 --accept-licenses"
              : "Run: .venv\\Scripts\\python.exe scripts\\models.py install minimax-h3-ref2va --accept-licenses"
            : "Run: .venv\\Scripts\\python.exe scripts\\models.py install minimax-h3 --accept-licenses"}</pre>
        </div>}
        <SelectControl label="Attention" value={h3AttentionBackend} disabled={busy}
          onChange={(value) => setH3AttentionBackend(value as "native" | "sage")} options={[
            { value: "sage", label: "Sage Attention 2 · Default" },
            { value: "native", label: "Native PyTorch SDPA" },
          ]} hint="Independent of Base/Turbo and memory profile. Sage is probed before denoising and again per call; a missing or incompatible kernel falls back to native attention and records why in the worker log. Pick native to rule attention out when a result looks wrong." />
        {(h3Mode === "image" || h3Mode === "first-last") && <SelectControl label={h3Mode === "image" ? "Source image" : "First frame"} value={firstId} disabled={busy || images.length < 1} onChange={setFirstId} options={images.length ? images.map((asset) => ({ value: asset.id, label: asset.name })) : [{ value: "", label: "Import an image" }]} />}
        {(h3Mode === "last-only" || h3Mode === "first-last") && <SelectControl label="Last frame" value={lastId} disabled={busy || images.length < 1} onChange={setLastId} options={images.length ? images.map((asset) => ({ value: asset.id, label: asset.name })) : [{ value: "", label: "Import an image" }]} />}
        {h3Mode === "first-last" && <button className="swap-frames" disabled={busy || !firstId || !lastId} onClick={() => { setFirstId(lastId); setLastId(firstId); }}>Swap first and last</button>}
        {h3Mode === "reference" && <SelectControl
          label="Reference quality"
          value={h3RefQuality}
          disabled={busy}
          onChange={(value) => setH3RefQuality(value as H3RefImageDetail)}
          options={H3_REF_IMAGE_DETAILS.map((item) => ({ value: item.id, label: item.label }))}
          hint="How large each reference image reaches the Qwen3-VL conditioner. Match is ComfyUI's own default and scales the reference to the generation's budget; 2048 is Diffusers' default and buys identity fidelity at a large cost in tokens and time, on every sampling step."
        />}
        {h3Mode === "reference" && <div className="h3-reference-tray">
          <div className="h3-reference-summary"><strong>Reference tray</strong><span>{h3RefLimits.images}/{H3_MAX_IMAGES} images · {h3RefLimits.videos}/{H3_MAX_VIDEOS} videos · {h3RefLimits.effectiveAudios}/{H3_MAX_AUDIOS} audio · {h3RefLimits.total}/{H3_MAX_REFERENCES} total</span></div>
          <div className="h3-reference-add">
            <select value={h3ReferenceAssetId} disabled={busy || !h3ReferenceAssets.length} onChange={(event) => setH3ReferenceAssetId(event.target.value)}>
              {h3ReferenceAssets.length ? h3ReferenceAssets.map((asset) => <option key={asset.id} value={asset.id}>{asset.name}</option>) : <option value="">Import reference media first</option>}
            </select>
            <button disabled={busy || !h3ReferenceAssetId} onClick={addH3Reference}><Plus size={14} /> Add</button>
          </div>
          {h3References.map((reference, index) => {
            const asset = project.assets.find((item) => item.id === reference.assetId);
            const sourceFps = asset?.frameRate
              ? asset.frameRate.numerator / asset.frameRate.denominator
              : undefined;
            return <div className="h3-reference-card" key={reference.id}>
              <span className="h3-reference-index">{index + 1}</span>
              <div><strong>{asset?.name ?? "Missing asset"}</strong><small>{reference.kind}{asset && reference.kind !== "image" ? ` · ${seconds(asset.duration).toFixed(1)}s` : ""}{reference.kind === "video" ? ` · ${sourceFps ? `${sourceFps.toFixed(2)} fps source` : "source fps"} → ordered RGB frames` : ""}</small></div>
              <div className="h3-reference-intents" aria-label={`Guidance roles for ${asset?.name ?? "reference"}`}>
                {availableH3ReferenceIntents(reference.kind, reference.kind === "audio" || Boolean(asset?.hasAudio && reference.includeEmbeddedAudio)).map((intent) => {
                  const definition = H3_REFERENCE_INTENTS.find((item) => item.value === intent)!;
                  const checked = reference.intents.includes(intent);
                  return <label key={intent} className={`${checked ? "selected" : ""}${intent === "continuation" ? " continuation" : ""}`}><input type="checkbox" checked={checked} disabled={busy || (checked && reference.intents.length === 1)} onChange={(event) => {
                    const enabled = event.target.checked;
                    setH3References((current) => current.map((item) => item.id === reference.id ? {
                      ...item,
                      intents: enabled ? [...item.intents, intent] : item.intents.filter((value) => value !== intent),
                      includeEmbeddedAudio: enabled && intent === "continuation" && Boolean(asset?.hasAudio) ? true : item.includeEmbeddedAudio,
                    } : item));
                    setH3ReferencePrompt("");
                  }} /><span>{definition.label}</span></label>;
                })}
              </div>
              {reference.kind === "video" && (asset?.hasAudio ? <label className="h3-audio-toggle"><input type="checkbox" checked={reference.includeEmbeddedAudio} disabled={busy} onChange={(event) => {
                const includeEmbeddedAudio = event.target.checked;
                setH3References((current) => current.map((item) => {
                  if (item.id !== reference.id) return item;
                  const allowed = availableH3ReferenceIntents("video", includeEmbeddedAudio);
                  const intents = item.intents.filter((intent) => allowed.includes(intent));
                  return { ...item, includeEmbeddedAudio, intents: intents.length ? intents : defaultH3ReferenceIntents("video") };
                }));
                setH3ReferencePrompt("");
              }} /> use clip audio</label> : <span className="h3-audio-toggle">no embedded audio</span>)}
              <div className="h3-reference-actions"><button disabled={busy || index === 0} onClick={() => moveH3Reference(index, -1)}>↑</button><button disabled={busy || index === h3References.length - 1} onClick={() => moveH3Reference(index, 1)}>↓</button><button disabled={busy} aria-label="Remove reference" onClick={() => { setH3References((current) => current.filter((item) => item.id !== reference.id)); setH3ReferencePrompt(""); }}><X size={13} /></button></div>
            </div>;
          })}
          <small className="panel-help">MP4 is an official Ref2VA input. It is decoded locally to ordered RGB frames at its source FPS, normalized by H3 to 24 fps, then sampled by the Qwen conditioner at 2 fps. Keep motion references as video; separate stills lose their temporal ordering. Choose as many guidance roles as apply. Continue clip includes the enabled soundtrack. Tray order defines model reference numbering. Timed references must each be 2–15 seconds.</small>
        </div>}
        {h3Mode === "reference" ? <>
          <label className="ai-prompt"><span>Shot brief</span><textarea value={h3ReferenceBrief} onChange={(event) => { setH3ReferenceBrief(event.target.value); setH3ReferencePrompt(""); }} placeholder="Describe the shot, action, dialogue and sound…" /></label>
          <div className="h3-prompt-tools"><button disabled={busy || !h3ReferenceBrief.trim()} onClick={() => setH3ReferencePrompt(localH3ReferencePrompt)}>Build local structured prompt</button><small>Local by default; editable before generation.</small></div>
          <label className="ai-prompt"><span>Structured prompt</span><textarea value={effectiveH3ReferencePrompt} onChange={(event) => setH3ReferencePrompt(event.target.value)} /></label>
        </> : <label className="ai-prompt"><span>Prompt</span><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder="Describe the shot and what it should sound like — H3 generates both together…" /></label>}
        <SelectControl label="Aspect" value={h3Aspect} disabled={busy || Boolean(h3CanvasSource)} onChange={setH3Aspect}
          options={H3_ASPECTS.map((item) => ({ value: item.ratio.join(":"), label: item.label }))}
          hint={h3CanvasSource ? `Taken from ${h3CanvasSource.name}, which H3 stretches onto the canvas.` : "H3 resolves its own canvas from the ratio, on a 32-pixel grid."} />
        <SelectControl label="Canvas" value={String(h3EffectiveShortEdge)} disabled={busy || h3Mode === "reference" || h3SamplingMode === "turbo-4"} onChange={(value) => setH3ShortEdge(Number(value))} options={[
          { value: String(H3_LOW_VRAM_SHORT_EDGE), label: `${H3_LOW_VRAM_SHORT_EDGE} short edge · 16 GB profile` },
          { value: String(H3_TRAINED_SHORT_EDGE), label: `${H3_TRAINED_SHORT_EDGE} short edge · trained size, much slower` },
        ]} hint={h3Mode === "reference"
          ? `Ref2VA is fixed to the conservative 544p canvas: ${h3Dimensions.width} × ${h3Dimensions.height}.`
          : h3SamplingMode === "turbo-4"
          ? `Draft is trained at 544p and resolves this aspect to ${h3Dimensions.width} × ${h3Dimensions.height}.`
          : `Resolves to ${h3Dimensions.width} × ${h3Dimensions.height}. The trained size costs roughly 2.3× the time per step on this card.`} />
        <NumberControl label="Frames" value={useBridgeGap ? bridgeFrames : h3Frames}
          disabled={busy || useBridgeGap} min={H3_MIN_FRAMES} max={H3_MAX_FRAMES} step={17}
          onChange={(value) => setH3Frames(clampH3Frames(value))}
          hint={useBridgeGap
            ? `Taken from the ${bridgeGap.toFixed(2)}s gap on the timeline. Switch Bridge length to "Set the duration below" to choose it here.`
            : `H3 requires 17n+5 frames at ${H3_FPS} fps and generates 5 to 15 seconds, so values snap into ${H3_MIN_FRAMES}–${H3_MAX_FRAMES}. Currently ${h3Seconds(clampH3Frames(h3Frames)).toFixed(2)}s.`} />
        <NumberControl label={h3Mode !== "reference" && h3SamplingMode === "base" ? "Steps" : "Evaluations"} value={h3EffectiveSteps}
          disabled={busy || h3Mode === "reference" || h3SamplingMode !== "base"} min={4} max={60} step={1}
          onChange={(value) => setH3Steps(Math.min(60, Math.max(4, Math.round(value))))}
          hint={h3Mode === "reference" ? "Base Ref2VA uses 20 transformer evaluations with no preview adapter."
            : h3SamplingMode === "turbo-4" ? "Fixed at four transformer evaluations by the dedicated LightX2V Draft checkpoint."
            : h3SamplingMode === "turbo-8" ? "Fixed at eight transformer evaluations by the selected LightX2V Turbo checkpoint."
            : "Transformer evaluations. The released Base reference recipe uses 50 NFE."} />
        <SeedControl value={seed} onChange={setSeed} disabled={busy} />
        <SelectControl label="Memory profile" value={h3Offload} disabled={busy} onChange={setH3Offload} options={[
          { value: "performance", label: "Performance · 4 blocks, VAE offloaded" },
          { value: "balanced", label: "Balanced · 1 block, streaming when supported (recommended)" },
          { value: "minimal", label: "Minimal · 1 block, no streaming" },
        ]} hint="All 16 GB profiles isolate output decoding and keep one VAE resident at a time. Balanced streams one block group of ordinary INT8 ConvRot tensors; Performance moves four groups without prefetch." />
        <small className="panel-help">The conditioner, denoiser, and decoder run as <b>three separate processes</b>, one after the other, so peak memory is the largest stage rather than their sum. The worker reports host RAM, VRAM, temporal decode progress, and peak allocation.</small>
        {h3Mode === "reference" && h3ReferencePreflight && <div className="worker-failure"><div><strong>Ref2VA needs attention</strong></div><pre>{h3ReferencePreflight}</pre></div>}
        <button className="render-button" onClick={() => { void generateH3(); }} disabled={busy || !(h3Mode === "reference" ? effectiveH3ReferencePrompt.trim() : prompt.trim()) || !runtime?.minimaxH3Ready || !h3SamplingReady
          || (h3Mode === "image" && images.length < 1)
          || (h3Mode === "last-only" && images.length < 1)
          || (h3Mode === "first-last" && images.length < 2)
          || (h3Mode === "clip-extend" && !extendableClip)
          || (h3Mode === "clip-to-clip" && !bridgePair)
          || (h3Mode === "reference" && Boolean(h3ReferencePreflight))}><WandSparkles size={16} />
          {h3Mode === "clip-extend" ? "Continue selected clip + insert"
            : h3Mode === "clip-to-clip" ? "Bridge selected clip to next + insert"
            : "Generate video + audio, place on V1 and A1"}</button>
      </> : <>
      <SelectControl label="Mode" value={ltxMode} disabled={busy} onChange={setLtxMode} options={[
        { value: "first-last", label: "First + last frame → video" },
        { value: "image", label: "Image → video" },
        { value: "text", label: "Text → video" },
        { value: "clip-extend", label: "Clip extend · last frame + prompt" },
        { value: "clip-to-clip", label: "Clip to clip · bridge two clips" },
        { value: "video-audio", label: "Video → audio (score a clip)" },
      ]} hint="Which LTX pipeline to run. Video to audio generates only a soundtrack and leaves your picture untouched." />
      <p className="ai-note">{ltxMode === "video-audio"
        ? "Generates only a soundtrack for the chosen clip and muxes it onto the untouched picture — no video is regenerated. The Foley V2A LoRA conditions the model on your footage so the audio lands on the action, and only DubIt's first stage runs, because that is where its audio actually comes from — the upsampler and second stage are picture-only."
          : ltxMode === "clip-extend"
            ? "Extracts the selected clip's exact last visible frame as a lossless PNG, generates forward with LTX, removes the duplicated guide frame and matching audio, then ripple-inserts the extension immediately after the clip."
          : ltxMode === "clip-to-clip"
            ? "Generates the missing motion between the selected V1 clip and the one after it. The outgoing clip's last visible frame and the incoming clip's first visible frame are extracted losslessly and used as the first/last conditioning pair, so the bridge starts and ends on frames already in your edit. Both duplicated boundary frames are trimmed off before the result is ripple-inserted at the cut."
          : ltxMode === "first-last"
          ? "The original hand-tuned single pass: both ends pinned, native resolution, synchronized audio, upscalers forcibly disabled."
          : ltxMode === "image"
            ? "Pins frame zero to your image and generates forward from it, with synchronized audio."
            : "Pure text conditioning with synchronized audio. No image is required."}</p>
      {(ltxMode === "first-last" || ltxMode === "image") && <SelectControl label={ltxMode === "image" ? "Source image" : "First frame"} value={firstId} disabled={busy || images.length < 1} onChange={setFirstId} options={images.length ? images.map((asset) => ({ value: asset.id, label: asset.name })) : [{ value: "", label: "Import an image" }]} />}
      {ltxMode === "first-last" && <SelectControl label="Last frame" value={lastId} disabled={busy || images.length < 2} onChange={setLastId} options={images.length ? images.map((asset) => ({ value: asset.id, label: asset.name })) : [{ value: "", label: "Import another image" }]} />}
      {ltxMode === "first-last" && <button className="swap-frames" disabled={busy || !firstId || !lastId} onClick={() => { setFirstId(lastId); setLastId(firstId); }}>Swap first and last</button>}
      {ltxMode === "video-audio" && <SelectControl label="Source clip" value={sourceClipId} disabled={busy || videoAssets.length < 1} onChange={setSourceClipId} options={videoAssets.length ? videoAssets.map((asset) => ({ value: asset.id, label: asset.name })) : [{ value: "", label: "Import a video" }]} />}
      {ltxMode === "clip-extend" && <div className="library-notice"><Clapperboard size={20} /><strong>{extendableClip ? extendableClip.clip.name : "Nothing to continue from"}</strong><span>{extendableClip ? `Last visible source frame at ${seconds(extendableClip.clip.sourceOut).toFixed(3)}s · ${clipExtendDimensions.width} × ${clipExtendDimensions.height}` : extendBlocker}</span></div>}
      {ltxMode === "clip-to-clip" && <div className="library-notice"><Clapperboard size={20} />
        <strong>{bridgePair ? `${bridgePair.from.name} → ${bridgePair.to.name}` : "Nothing to bridge"}</strong>
        <span>{bridgePair
          ? `Out at ${seconds(bridgePair.from.sourceOut).toFixed(3)}s → in at ${seconds(bridgePair.to.sourceIn).toFixed(3)}s · ${bridgeDimensions.width} × ${bridgeDimensions.height}`
          : bridgeBlocker}</span></div>}
      {ltxMode === "video-audio" && runtime && !runtime.foleyLoraReady && <div className="worker-failure">
        <div><strong>Foley V2A LoRA not installed</strong></div>
        <pre>{"Without it the model cannot see the picture, so the audio would not sync.\n\nPlace ltx-2.3-22b-lora-foley-v2a-1.0.safetensors in models\\loras\\ltx\\, or install the pack:\n  .venv\\Scripts\\python.exe scripts\\models.py install ltx-loras --accept-licenses\n\nAlso requires models\\LTX-2.3\\ltx-2.3-22b-distilled-1.1.safetensors — the FP8 checkpoint cannot fuse LoRAs."}</pre>
      </div>}
      <label className="ai-prompt"><span>Prompt</span><textarea value={prompt} onChange={(event) => setPrompt(event.target.value)} placeholder={ltxMode === "video-audio" ? "Describe the soundtrack: ambience, foley, music, dialogue tone…" : "Describe motion, camera, action, and audio…"} /></label>
      <label className="ai-prompt"><span>Avoid</span><textarea value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)} placeholder="Optional — what to steer away from" /></label>
      {/* Guidance is what applies a negative prompt, and there is none at 1.0. The worker refuses
          the combination rather than accepting the text and dropping it, so say so here first. */}
      {ltxMode !== "video-audio" && negativePrompt.trim() !== "" && cfgScale <= 1 && audioCfgScale <= 1 && <small className="panel-help">
        <b>Avoid needs guidance.</b> Both CFG scales are at 1.0, so there is no unconditional pass for this text to steer against and the generation will refuse to start. Raise <b>CFG (video)</b> or <b>CFG (audio)</b> above 1.0, or clear this field.
      </small>}
      {ltxMode !== "video-audio" && ltxMode !== "clip-extend" && ltxMode !== "clip-to-clip" && <SelectControl label="Resolution" value={resolution} disabled={busy} onChange={setResolution} options={[
        { value: "768x512", label: "768 × 512 landscape" }, { value: "512x768", label: "512 × 768 portrait" }, { value: "512x512", label: "512 × 512 square" },
      ]} hint="Generation size. The 16 GB single-pass profile is capped at 768x768 pixels." />}
      {ltxMode !== "video-audio" && <NumberControl label="Frames" value={frames} disabled={busy} min={LTX_MIN_FRAMES} max={LTX_MAX_FRAMES} step={LTX_FRAMES_PER_LATENT} onChange={(value) => setFrames(snapLtxFrames(value))} hint={`Length in frames. The model requires 8n+1, so values snap to the nearest valid count. The 16 GB profile has been run up to ${LTX_MAX_FRAMES}.`} />}
      <SeedControl value={seed} onChange={setSeed} disabled={busy} />
      {(ltxMode === "first-last" || ltxMode === "image" || ltxMode === "clip-extend" || ltxMode === "clip-to-clip") && <RangeControl label="Guide strength" value={guideStrength} disabled={busy} min={0.1} max={1} step={0.05} onChange={setGuideStrength} hint="How tightly the result is pinned to your conditioning image. Lower allows more invention." />}
      {ltxMode !== "video-audio" && <RangeControl label="CFG (video)" value={cfgScale} disabled={busy} min={1} max={8} step={0.1} onChange={setCfgScale} hint="Classifier-free guidance for picture. 1.0 is unguided. Above 1.0 costs an extra pass per step." />}
      <RangeControl label="CFG (audio)" value={audioCfgScale} disabled={busy} min={1} max={8} step={0.1} onChange={setAudioCfgScale} hint="Guidance for sound. Audio only starts responding to the prompt around 1.5." />
      <small className="panel-help">1.0 is unguided, which is what the distilled schedule assumes. Audio only starts responding to the prompt around <b>1.5</b>. Anything above 1.0 adds an unconditional pass per step, so it costs roughly double the time and VRAM.</small>
      {ltxMode === "video-audio" && <>
        <RangeControl label="Reference hold" value={referenceStrength} disabled={busy} min={0.1} max={2} step={0.05} onChange={setReferenceStrength} hint="How hard the generated audio is pinned to the footage. Raise this if the sound ignores the action." />
        <RangeControl label="LoRA strength" value={loraStrength} disabled={busy} min={0.1} max={2} step={0.05} onChange={setLoraStrength} hint="Scales the Foley adapter. Lower it if the result sounds over-processed." />
        <small className="panel-help">If the audio sounds right but ignores the action, raise <b>reference hold</b> — it controls how hard the sound is pinned to the footage. <b>LoRA strength</b> scales the foley adapter itself; lower it if the result sounds over-processed.</small>
      </>}
      <SelectControl label="Offload" value={offload} disabled={busy} onChange={setOffload} options={[
        { value: "cpu", label: "CPU streaming · faster, recommended · ~19 GiB pinned" },
        { value: "disk-cached", label: "Disk + cache · ~6 GiB pinned" },
        { value: "disk", label: "Disk streaming · lowest RAM, unstable here" },
      ]} hint="Where the 48 transformer blocks live between steps. CPU pins all of them and is fastest when RAM allows; disk + cache keeps a third pinned; plain disk keeps two and is the configuration that has crashed on the 16 GB test machine." />
      <SelectControl label="Attention" value={ltxAttention} disabled={busy} onChange={(value) => setLtxAttention(value as "native" | "sage")} options={[
        { value: "native", label: "Native PyTorch SDPA" },
        { value: "sage", label: "Sage Attention 2" },
      ]} hint="LTX only ever reaches FlashAttention on Hopper and datacenter Blackwell cards, so on consumer RTX cards such as the RTX 4090 (Ada, sm89) it runs on SDPA whatever is installed. Sage is the kernel built for sm89, and needs scripts\install-ltx-attention.ps1 first; without it this falls back to native and says so in the worker log rather than failing the run. Sage quantises Q, K and V, so it is faster and not identical — the install measured 3.6% mean deviation from SDPA. The first call of a run compiles a triton kernel, so judge speed on the second run." />
      <button className="render-button" onClick={() => { void generate(); }} disabled={busy || !prompt.trim()
        || (ltxMode === "video-audio"
          ? !sourceClipId || !runtime?.foleyLoraReady
          : !runtime?.singlePassReady
            || (ltxMode === "first-last" && images.length < 2)
            || (ltxMode === "image" && images.length < 1)
            || (ltxMode === "clip-extend" && !extendableClip)
            || (ltxMode === "clip-to-clip" && !bridgePair))}><WandSparkles size={16} /> {ltxMode === "video-audio" ? "Score clip + place on timeline" : ltxMode === "clip-extend" ? "Extend selected clip + insert after" : ltxMode === "clip-to-clip" ? "Bridge the two clips + insert between" : "Generate + place on timeline"}</button>
      </>}
      {generating && <div className="ltx-job"><span className="ltx-spinner" /><div><strong>Worker running · {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}</strong><small>Loading, denoising, decoding, and muxing locally. The UI remains responsive.</small></div></div>}
      {lastResult && <div className="ltx-job complete"><span>✓</span><div><strong>Generation completed</strong><small>{lastResult.outputPath}</small></div></div>}
      {failure && <div className="worker-failure">
        <div><strong>{engine === "h3" ? "The MiniMax-H3 worker failed" : "The LTX worker failed"}</strong><button onClick={() => setFailure(undefined)} aria-label="Dismiss"><X size={12} /></button></div>
        <pre>{failure}</pre>
      </div>}
      <small className="runtime-checks">
        Python {runtime?.pythonReady ? "✓" : "×"} · LTX {runtime?.ltxSourceReady ? "✓" : "×"} · checkpoint {runtime?.checkpointReady ? "✓" : "×"} ·
        Gemma {runtime?.gemmaReady ? "✓" : "×"} · Foley LoRA {runtime?.foleyLoraReady ? "✓" : "×"} ·
        MOSS SFX {runtime?.mossSfxReady ? "✓" : "×"} ·
        Kiwi-Edit {runtime?.kiwiEditReady ? "✓" : "×"} · Whisper {runtime?.whisperReady ? "✓" : "×"} ·
        MiniMax-H3 {runtime?.minimaxH3Ready ? "✓" : "×"}{runtime?.minimaxH3State && runtime.minimaxH3State !== "ready" ? " (" + runtime.minimaxH3State + ")" : ""} ·
        H3 Draft {runtime?.minimaxH3DraftReady ? "✓" : "×"} · H3 Turbo {runtime?.minimaxH3TurboReady ? "✓" : "×"} · Ref2VA {runtime?.minimaxH3RefReady ? "✓" : "×"}
      </small>
    </Section>
  </div>;
}

/**
 * The AI clip editor: a focused workspace for rewriting a range of one timeline clip with
 * Kiwi-Edit. It scrubs the clip's own source, keeps the range in source time, and can either
 * replace the clip's media in place or drop the result in as a new clip.
 */
function ClipEditor() {
  const {
    clipEditorOpen, setClipEditorOpen, project, selectedClipId, playhead,
    replaceClipRangeWithAsset, insertAssetAfterClip, setStatus, setBusy, busy,
  } = useEditor();
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId)!;
  const clip = sequence.tracks.flatMap((track) => track.clips).find((item) => item.id === selectedClipId);
  const asset = project.assets.find((item) => item.id === clip?.assetId);
  const previewRef = useRef<HTMLVideoElement>(null);
  const [runtime, setRuntime] = useState<AiRuntimeStatus>();
  // Which generator the popout drives. Both rewrite the source media behind a V1 clip over a
  // chosen range, so they share the preview, the scrub strip, the range and the apply actions --
  // only the model and its parameters differ.
  const [engine, setEngine] = useState<VisualMode>("kiwi");
  const [retakePrompt, setRetakePrompt] = useState("");
  const [retakeMode, setRetakeMode] = useState("video-audio");
  const [retakeSeed, setRetakeSeed] = useState(42);
  const [retakeOffload, setRetakeOffload] = useState("cpu");
  const [retakeNegative, setRetakeNegative] = useState("");
  const [retakeCfg, setRetakeCfg] = useState(2.5);
  const [retakeGuidance, setRetakeGuidance] = useState<"fast" | "experimental-guided-distilled">("fast");
  const [retakeEnhancePrompt, setRetakeEnhancePrompt] = useState(false);
  const [retakeAttention, setRetakeAttention] = useState<"native" | "sage">("native");
  const [mode, setMode] = useState("instruction");
  const [strategy, setStrategy] = useState<KiwiStrategy>("stable");
  const [instruction, setInstruction] = useState("");
  const [kiwiGuidance, setKiwiGuidance] = useState(4);
  const [kiwiNegative, setKiwiNegative] = useState("");
  const [referenceImage, setReferenceImage] = useState("");
  const [rangeStart, setRangeStart] = useState(0);
  const [rangeEnd, setRangeEnd] = useState(2);
  const [seed, setSeed] = useState(42);
  const [steps, setSteps] = useState(24);
  const [offload, setOffload] = useState("balanced");
  const [scrub, setScrub] = useState(0);
  const [running, setRunning] = useState(false);
  const [startedAt, setStartedAt] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<{
    asset: MediaAsset; log: string; previewPath?: string;
    effectiveStartOffset?: number; effectiveEndOffset?: number;
  }>();
  const [contextPreview, setContextPreview] = useState(true);
  const [failure, setFailure] = useState<string>();
  const [workspace, setWorkspace] = useState<"visual" | "sfx">("visual");
  const [sfxRunning, setSfxRunning] = useState(false);
  const [maskRunning, setMaskRunning] = useState(false);

  const sourceDuration = clip ? clipDuration(clip.sourceIn, clip.sourceOut) : 0;

  useEffect(() => {
    if (!clipEditorOpen) return;
    void runtimeStatus().then(setRuntime).catch(() => undefined);
  }, [clipEditorOpen]);
  useEffect(() => {
    if (!clipEditorOpen || !clip) return;
    // Seed the range from the playhead so "edit what I'm looking at" is the default.
    const atPlayhead = Math.max(0, Math.min(sourceDuration, (playhead - seconds(clip.start)) * clipPlaybackRate(clip)));
    setRangeStart(atPlayhead);
    setRangeEnd(Math.min(sourceDuration, atPlayhead + Math.min(3, sourceDuration)));
    setScrub(atPlayhead);
    setResult(undefined);
    setFailure(undefined);
  }, [clipEditorOpen, clip?.id]);
  useEffect(() => {
    if (!running) return;
    const update = () => setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    update();
    const timer = window.setInterval(update, 500);
    return () => window.clearInterval(timer);
  }, [running, startedAt]);
  useEffect(() => {
    const video = previewRef.current;
    if (!video || !clip) return;
    const target = seconds(clip.sourceIn) + scrub;
    if (Math.abs(video.currentTime - target) > 0.05) video.currentTime = target;
  }, [scrub, clip?.id]);

  if (!clipEditorOpen) return null;

  const pickImage = async (setter: (path: string) => void) => {
    const picked = await open({ multiple: false, filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }] });
    if (typeof picked === "string") setter(picked);
  };

  const unusable = !clip || clip.trackId !== "v1" || !asset;
  const rangePlan = planKiwiRange(rangeEnd - rangeStart, strategy);
  const engineReady = engine === "kiwi" ? Boolean(runtime?.kiwiEditReady)
    : engine === "retake" ? Boolean(runtime?.retakeReady)
      : engine === "mask" ? Boolean(runtime?.sam2Ready || runtime?.birefnetReady) : true;
  // One latent frame is 8 source frames, and the retake mask works in latent time -- a shorter
  // range marks a single token and barely moves no matter how hard it is guided.
  const assetFps = asset?.frameRate ? asset.frameRate.numerator / asset.frameRate.denominator : 24;
  const clipSourceInSeconds = clip ? seconds(clip.sourceIn) : 0;
  const absoluteRange = ltxFrameRange(
    clipSourceInSeconds + rangeStart,
    clipSourceInSeconds + rangeEnd,
    asset?.frameRate?.numerator ?? 24,
    asset?.frameRate?.denominator ?? 1,
  );
  const minRetakeRange = 8 / Math.max(1, assetFps);
  const retakeRangeOk = absoluteRange.frames >= LTX_FRAMES_PER_LATENT;
  // Retake works on the WHOLE clip's latent, so the source has to satisfy the profile before the
  // range is worth discussing. The panel used to report READY, take a range, spawn the worker and
  // let it refuse -- while printing the very limits it was not checking in its own help text.
  const retakeSourceFrames = asset ? Math.round(seconds(asset.duration) * assetFps) : 0;
  const retakeSourceProblems = engine === "retake" && asset && asset.width && asset.height
    ? ltxRetakeProblems(
      asset.width, asset.height, retakeSourceFrames, assetFps,
      clipSourceInSeconds + rangeStart,
      clipSourceInSeconds + rangeEnd,
    )
    : [];
  const canRun = engine !== "mask" && engine !== "text" && engine !== "vfx" && !unusable && !busy && engineReady && rangeEnd > rangeStart && (engine === "kiwi"
    ? instruction.trim().length > 0 && rangePlan.supported && (mode !== "reference" || Boolean(referenceImage))
    : retakePrompt.trim().length > 0 && retakeRangeOk && retakeSourceProblems.length === 0);

  const run = async () => {
    if (!clip || !asset) return;
    const outputPath = await save({
      defaultPath: clip.name.replace(/\.[^.]+$/, "") + "-edit.mp4",
      filters: [{ name: "MPEG-4 Video", extensions: ["mp4"] }],
    });
    if (!outputPath) return;
    setBusy(true);
    setRunning(true);
    setStartedAt(Date.now());
    setElapsed(0);
    setResult(undefined);
    setFailure(undefined);
    const usedSeed = resolveSeed(seed);
    setStatus(`Kiwi-Edit is rewriting the selected range (seed ${usedSeed}); this can take several minutes…`);
    try {
      const bridge = await invoke<LtxBridgeResult>("edit_clip_kiwi", { request: {
        videoPath: asset.path, outputPath, instruction,
        referenceImage: mode === "reference" ? referenceImage : null,
        startTime: seconds(clip.sourceIn) + rangeStart,
        endTime: seconds(clip.sourceIn) + rangeEnd,
        seed: usedSeed, mode, strategy, steps, offload,
        guidanceScale: kiwiGuidance, negativePrompt: kiwiNegative,
      } });
      const probe = await probeMedia(outputPath);
      const edited = generatedAsset(outputPath, probe);
      setResult({ asset: edited, log: bridge.log });
      setStatus(`Clip edit complete (seed ${usedSeed}) — review it, then replace or insert`);
    } catch (error) {
      // Keep the whole worker tail in the panel; the status bar only gets a single line.
      setFailure(errorText(error));
      setStatus("Clip edit failed: " + errorHeadline(error));
    } finally {
      setRunning(false);
      setBusy(false);
    }
  };

  // Deliberately mirrors `run()` rather than replacing the clip outright, which is what the old
  // inspector version did. Producing a `result` first means Retake gets the same preview-then-apply
  // review as Kiwi-Edit, and the same choice of replacing the range or inserting after the clip.
  const runRetake = async () => {
    if (!clip || !asset) return;
    const outputPath = await save({
      defaultPath: clip.name.replace(/\.[^.]+$/, "") + "-retake.mp4",
      filters: [{ name: "MPEG-4 Video", extensions: ["mp4"] }],
    });
    if (!outputPath) return;
    setBusy(true);
    setRunning(true);
    setStartedAt(Date.now());
    setElapsed(0);
    setResult(undefined);
    setFailure(undefined);
    const usedSeed = resolveSeed(retakeSeed);
    setStatus(`LTX Retake is regenerating the selected range (seed ${usedSeed})…`);
    try {
      const bridge = await invoke<LtxRetakeResult>("retake_ltx_bridge", { request: {
        videoPath: asset.path, outputPath, prompt: retakePrompt,
        startTime: seconds(clip.sourceIn) + rangeStart,
        endTime: seconds(clip.sourceIn) + rangeEnd,
        seed: usedSeed, mode: retakeMode, offload: retakeOffload,
        negativePrompt: retakeNegative,
        guidance: retakeGuidance,
        enhancePrompt: retakeEnhancePrompt,
        attentionBackend: retakeAttention,
        // Only under the experimental guided mode, where the slider is on screen. Sending a number under `fast`
        // would quietly turn its one pass per step into two.
        ...(retakeGuidance === "experimental-guided-distilled" ? {
          cfgScale: retakeCfg,
          audioCfgScale: retakeMode === "video" ? 1.0 : Math.max(1.5, Math.min(retakeCfg, 4)),
        } : {}),
      } });
      const probe = await probeMedia(outputPath);
      const sourceInSeconds = seconds(clip.sourceIn);
      setContextPreview(true);
      setResult({
        asset: generatedAsset(outputPath, probe), log: bridge.log, previewPath: bridge.previewPath,
        effectiveStartOffset: Math.max(0, bridge.effectiveStartTime - sourceInSeconds),
        effectiveEndOffset: Math.min(sourceDuration, bridge.effectiveEndTime - sourceInSeconds),
      });
      setStatus(`LTX Retake complete (seed ${usedSeed}) — review it, then replace or insert`);
    } catch (error) {
      setFailure(errorText(error));
      setStatus("LTX Retake failed: " + errorHeadline(error));
    } finally {
      setRunning(false);
      setBusy(false);
    }
  };

  const rangeLeft = sourceDuration > 0 ? (rangeStart / sourceDuration) * 100 : 0;
  const rangeWidth = sourceDuration > 0 ? ((rangeEnd - rangeStart) / sourceDuration) * 100 : 0;
  const modalRunning = running || sfxRunning || maskRunning;
  const changeVisualMode = (next: VisualMode) => {
    setEngine(next); setResult(undefined); setFailure(undefined);
  };

  return <div className="modal-backdrop" onPointerDown={() => !modalRunning && setClipEditorOpen(false)}>
    <section className={`clip-editor${workspace === "visual" && engine === "vfx" && !unusable ? " vfx" : ""}`} role="dialog" aria-modal="true" aria-label="AI clip editor" onPointerDown={(event) => event.stopPropagation()}>
      <div className="dialog-heading">
        <div>
          <strong><Clapperboard size={15} /> AI clip editor</strong>
          <small>{clip ? clip.name : "No clip selected"} · {workspace === "sfx" ? "MOSS designed SFX and LTX picture Foley" : engine === "kiwi"
            ? "Kiwi-Edit 5B · instruction and reference-guided editing"
            : engine === "retake" ? "LTX Retake · regenerate a range from a prompt"
              : engine === "mask" ? "BiRefNet / SAM 2 · mask-driven visual effects"
                : engine === "vfx" ? "Layers, elements, tracking and depth over the clip · renders back to the timeline"
                  : "Editable animated text on the graphics timeline"}</small>
        </div>
        <span className={"runtime-pill " + (workspace === "sfx"
          ? runtime?.mossSfxReady || runtime?.foleyLoraReady ? "ready" : "not-ready"
          : engineReady ? "ready" : "not-ready")}>
          {workspace === "sfx" ? runtime?.mossSfxReady || runtime?.foleyLoraReady ? "SFX ready" : "SFX setup"
            : engine === "kiwi" ? kiwiCopy(runtime?.kiwiEditState).pill
              : engine === "retake" ? runtime?.retakeReady ? "ready" : "not ready"
                : engine === "mask" ? engineReady ? "mask ready" : "mask setup" : engine === "vfx" ? "compositor" : "NLE native"}
        </span>
        <button onClick={() => setClipEditorOpen(false)} aria-label="Close" disabled={modalRunning}><X size={16} /></button>
      </div>

      {unusable ? <div className="library-notice"><Clapperboard size={22} /><strong>Select a V1 media clip</strong>
        <span>The clip editor rewrites the source media behind a primary-track clip.</span></div> : <>
        <div className="clip-workspace-switch" role="tablist" aria-label="AI clip editor mode">
          <button role="tab" aria-selected={workspace === "visual"} className={workspace === "visual" ? "active" : ""}
            disabled={modalRunning} onClick={() => setWorkspace("visual")}>Visual Edit</button>
          <button role="tab" aria-selected={workspace === "sfx"} className={workspace === "sfx" ? "active" : ""}
            disabled={modalRunning} onClick={() => setWorkspace("sfx")}>SFX</button>
        </div>
        {workspace === "sfx" ? <SfxWorkspace clip={clip!} asset={asset!} rangeStart={rangeStart} rangeEnd={rangeEnd}
          sourceDuration={sourceDuration} setRangeStart={setRangeStart} setRangeEnd={setRangeEnd}
          onRunningChange={setSfxRunning} onClose={() => setClipEditorOpen(false)} />
        : engine === "mask" ? <MaskEffectsWorkspace clip={clip!} asset={asset!} runtime={runtime}
          rangeStart={rangeStart} rangeEnd={rangeEnd} sourceDuration={sourceDuration}
          setRangeStart={setRangeStart} setRangeEnd={setRangeEnd} onModeChange={changeVisualMode}
          onRunningChange={setMaskRunning} onClose={() => setClipEditorOpen(false)} />
        : engine === "vfx" ? <PhotoEdit key={clip!.id} clipTarget={{
          clip: clip!, asset: asset!, rangeStart, rangeEnd,
          modeSwitch: <VisualModeSwitch mode="vfx" disabled={maskRunning} onChange={changeVisualMode} />,
          onRunningChange: setMaskRunning, onClose: () => setClipEditorOpen(false),
        }} />
        : engine === "text" ? <TextMotionWorkspace clip={clip!} asset={asset!}
          rangeStart={rangeStart} rangeEnd={rangeEnd} sourceDuration={sourceDuration}
          setRangeStart={setRangeStart} setRangeEnd={setRangeEnd} onModeChange={changeVisualMode}
          onClose={() => setClipEditorOpen(false)} /> : <>
        <div className="clip-editor-body">
          <div className="clip-editor-preview">
            <video ref={previewRef}
              key={(result && contextPreview && result.previewPath ? result.previewPath : result?.asset.path) ?? asset!.path}
              src={convertFileSrc((result && contextPreview && result.previewPath ? result.previewPath : result?.asset.path) ?? asset!.path)}
              preload="auto" muted={!result || Boolean(contextPreview && result.previewPath)} controls={Boolean(result)} />
            {result?.previewPath && <div className="scrub-actions">
              <button className={contextPreview ? "active" : ""} onClick={() => setContextPreview(true)}>Context + seams</button>
              <button className={!contextPreview ? "active" : ""} onClick={() => setContextPreview(false)}>Generated range</button>
              <span className="range-readout">One second of source context appears on each available side.</span>
            </div>}
            {!result && <div className="scrub-strip">
              <div className="scrub-range" style={{ left: `${rangeLeft}%`, width: `${rangeWidth}%` }} />
              <input type="range" min={0} max={Math.max(0.1, sourceDuration)} step={0.02} value={scrub}
                onChange={(event) => setScrub(Number(event.target.value))} />
            </div>}
            <div className="scrub-actions">
              <span className="timecode">{formatTimecode(scrub)}</span>
              <button disabled={Boolean(result)} onClick={() => setRangeStart(Math.min(scrub, rangeEnd - 0.1))}>Set in</button>
              <button disabled={Boolean(result)} onClick={() => setRangeEnd(Math.max(scrub, rangeStart + 0.1))}>Set out</button>
              <button disabled={Boolean(result)} onClick={() => { setRangeStart(0); setRangeEnd(sourceDuration); }}>Whole clip</button>
              <span className="range-readout">{rangeStart.toFixed(2)}s – {rangeEnd.toFixed(2)}s · {(rangeEnd - rangeStart).toFixed(2)}s</span>
              {engine === "kiwi" && (() => {
                const fps = 15;
                const plan = planKiwiRange(rangeEnd - rangeStart, strategy);
                if (!plan.supported) return <span className="range-warn">{plan.message}</span>;
                return <span className="range-plan">{plan.windows} pass{plan.windows === 1 ? "" : "es"} · {(plan.frames / fps).toFixed(2)}s output</span>;
              })()}
            </div>
          </div>

          <div className="clip-editor-controls">
            <VisualModeSwitch mode={engine} disabled={running} onChange={changeVisualMode} />
            {engine === "kiwi" && <SelectControl label="Mode" value={mode} disabled={running} onChange={setMode} options={[
              { value: "instruction", label: "Instruction only" },
              { value: "reference", label: "Reference image guided" },
            ]} />}
            {engine === "kiwi" && <SelectControl label="Edit length" value={strategy} disabled={running} onChange={(value) => setStrategy(value as KiwiStrategy)} options={[
              { value: "stable", label: "Stable - one 81-frame pass" },
              { value: "long", label: "Long - overlap + seam analysis" },
            ]} hint="Stable is the supported single-window quality mode. Long edits overlap 81-frame passes, condition each pass on the preceding result, choose the least-disruptive seam, and blend only around that cut." />}
            {engine === "kiwi" ? <label className="ai-prompt"><span>Instruction</span><textarea value={instruction} disabled={running}
              placeholder="Remove the microphone. Make it snow. Change the jacket to red…"
              onChange={(event) => setInstruction(event.target.value)} /></label>
            : <label className="ai-prompt"><span>Prompt</span><textarea value={retakePrompt} disabled={running}
              placeholder="Subject and appearance; chronological action; camera; lighting/style; sound and timing…"
              onChange={(event) => setRetakePrompt(event.target.value)} /></label>}
            {engine === "kiwi" && <RangeControl label="Instruction strength" value={kiwiGuidance} disabled={running}
              min={1} max={10} step={0.5} onChange={setKiwiGuidance}
              hint="Classifier-free guidance. The pack shipped with this branch commented out entirely, so every edit before now ran at 1.0 — the instruction described the change but nothing pushed the result toward it. 1.0 restores that old behaviour; 4 is a sensible start. Costs a second pass per step." />}
            {engine === "kiwi" && kiwiGuidance === 1 && <small className="range-warn">
              At 1.0 the instruction only conditions the model and never steers it, so the edit will tend to come back looking like the input. Raise this to see the instruction take effect.
            </small>}
            {engine === "kiwi" && <label className="ai-prompt"><span>Avoid</span>
              <textarea value={kiwiNegative} disabled={running} placeholder="What the edited range should not contain (optional)"
                onChange={(event) => setKiwiNegative(event.target.value)} /></label>}
            {engine === "retake" && <label className="ai-prompt"><span>Avoid</span>
              <textarea value={retakeNegative} disabled={running} placeholder="What the regenerated range should not contain (optional)"
                onChange={(event) => setRetakeNegative(event.target.value)} /></label>}
            {/* Same refusal the single-pass panel carries, for the same reason: without an
                unconditional branch there is nothing for this text to be subtracted from, so the
                worker refuses rather than running for minutes and ignoring it. */}
            {engine === "retake" && retakeGuidance === "fast" && retakeNegative.trim() !== "" && <small className="panel-help">
              <b>Avoid needs guidance.</b> Fast leaves every guidance axis inert, so there is no unconditional pass for this text to steer against and the retake will refuse to start. Switch <b>Guidance</b> to Experimental guided-distilled, or clear this field.
            </small>}
            {engine === "retake" && <SelectControl label="Attention" value={retakeAttention} disabled={running}
              onChange={(value) => setRetakeAttention(value as "native" | "sage")} options={[
                { value: "native", label: "Native PyTorch SDPA" },
                { value: "sage", label: "Sage Attention 2" },
              ]}
              hint="The LTX transformer only ever reaches FlashAttention on Hopper and datacenter Blackwell cards, so on consumer RTX cards such as the RTX 4090 (Ada, sm89) it runs on SDPA whatever is installed. Sage is the kernel built for sm89. It needs scripts\install-ltx-attention.ps1 first; without it, this falls back to native and says so in the worker log rather than failing the run. The first call of a run compiles a triton kernel, so judge the speed on the second run." />}
            {engine === "retake" && <SelectControl label="Guidance" value={retakeGuidance} disabled={running}
              onChange={(value) => setRetakeGuidance(value as "fast" | "experimental-guided-distilled")} options={[
                { value: "fast", label: "Fast · 1 pass per step" },
                { value: "experimental-guided-distilled", label: "Experimental guided-distilled · 4 passes/step" },
              ]}
              hint="Fast is the documented path for the LTX 2.3 distilled checkpoint. Experimental guided-distilled keeps its eight-step distilled schedule but forces the guided denoiser, STG, modality guidance and rescaling. It is diagnostic, not an official LTX reference configuration." />}
            {engine === "retake" && retakeGuidance === "experimental-guided-distilled" && <RangeControl label="Prompt strength" value={retakeCfg} disabled={running}
              min={1} max={8} step={0.5}
              onChange={setRetakeCfg}
              hint="Classifier-free guidance contributes (strength − 1) × (prompt prediction − no-prompt prediction). This checkpoint is guidance-distilled, so that difference may already be very small; the worker measures it." />}
            {engine === "retake" && <ToggleControl label="Enhance prompt" value={retakeEnhancePrompt} disabled={running}
              onChange={setRetakeEnhancePrompt}
              hint="Uses LTX 2.3's built-in Gemma prompt enhancer. Use it for short intent prompts; leave it off when subject, chronological action, camera, lighting and sound are already precise." />}
            {engine === "retake" && retakeSourceProblems.length > 0 && <div className="worker-failure">
              <div><strong>This retake cannot run on the 16 GB profile</strong></div>
              <ul>{retakeSourceProblems.map((problem) => <li key={problem}>{problem}</li>)}</ul>
              <small>The worker cuts a window around your selection and regenerates inside that, so the clip's length is not a limit — but the canvas is, and so is the length of the selection itself.</small>
            </div>}
            {engine === "retake" && !retakeRangeOk && <small className="range-warn">
              Range is shorter than one latent frame ({minRetakeRange.toFixed(2)}s at {assetFps.toFixed(0)} fps). Nothing meaningful can be regenerated below that.
            </small>}
            {engine === "retake" && retakeRangeOk && rangeEnd - rangeStart < 2 && <small className="range-warn">
              {(rangeEnd - rangeStart).toFixed(2)}s is under the 2s minimum Lightricks' own hosted Retake enforces. It will run, but the untouched frames on either side dominate the attention over a window this short and the usual result is that the original comes back essentially unchanged. Widen the range before concluding the prompt is being ignored.
            </small>}
            {engine === "retake" && <SelectControl label="Regenerate" value={retakeMode} disabled={running} onChange={setRetakeMode} options={[
              { value: "video-audio", label: "Video + audio" },
              { value: "video", label: "Video only · preserve audio" },
              { value: "audio", label: "Audio only · preserve video" },
            ]} hint="The preserved modality is rebuilt directly from source frames or samples after LTX finishes; it never comes from a frozen-and-decoded LTX latent." />}
            {engine === "kiwi" && mode === "reference" && <div className="guidance-picker">
              <button disabled={running} onClick={() => { void pickImage(setReferenceImage); }}>Choose reference image…</button>
              <small>{referenceImage ? filename(referenceImage) : "No reference selected"}</small>
            </div>}
            <div className="retake-range">
              <NumberControl label="Range in" value={rangeStart} disabled={running} min={0} max={Math.max(0.1, sourceDuration - 0.1)} step={1 / assetFps} suffix="s" onChange={setRangeStart} />
              <NumberControl label="Range out" value={rangeEnd} disabled={running} min={0.1} max={Math.max(0.1, sourceDuration)} step={1 / assetFps} suffix="s" onChange={setRangeEnd} />
            </div>
            {engine === "retake" && <small className="panel-help">
              Frame contract: source frames {absoluteRange.startFrame}–{absoluteRange.endFrame} (exclusive), {absoluteRange.frames} frames at {asset?.frameRate?.numerator ?? 24}/{asset?.frameRate?.denominator ?? 1} fps. Non-frame-aligned selections expand to cover every touched frame.
            </small>}
            {engine === "kiwi" && <RangeControl label="Steps" value={steps} disabled={running} min={4} max={60} onChange={setSteps} hint="Denoising iterations. More is slower and usually cleaner, with diminishing returns past ~30." />}
            <SeedControl value={engine === "kiwi" ? seed : retakeSeed} disabled={running}
              onChange={engine === "kiwi" ? setSeed : setRetakeSeed} />
            {engine === "kiwi" ? <SelectControl label="Placement" value={offload} disabled={running} onChange={setOffload} options={[
              { value: "balanced", label: "Staged · fits 16 GB VRAM" },
              { value: "minimal", label: "Block streaming · under 16 GB" },
            ]} hint="How the clip editor divides models between GPU and CPU for your VRAM budget." />
            : <SelectControl label="16 GB offload" value={retakeOffload} disabled={running} onChange={setRetakeOffload} options={[
              { value: "cpu", label: "CPU streaming · qualified · ~19 GiB pinned" },
              { value: "disk-cached", label: "Disk + cache · experimental · ~6 GiB pinned" },
              { value: "disk", label: "Disk streaming · unqualified / unstable here" },
            ]} hint="The 16-GB GPU envelope is qualified through 9,609,216 pixel-frames on CPU streaming. Disk-cached is available for RAM pressure but has not completed the same qualification matrix; disk has repeatedly faulted on the 16 GB test machine." />}
            {engine === "retake" && <small className="panel-help">LTX 2.3 Retake returns only the frame-aligned interval. For reliable prompts, describe subject + chronological action, then camera motion, lighting/style, and the sound event. Audio-only preserves decoded source frames; video-only preserves decoded source samples. The adapter enforces the qualified 16 GB geometry limits.</small>}
            {engine === "kiwi" && <small className="panel-help">Staged holds one model on the GPU at a time — the encoder runs first at 7.1 GiB, is evicted, then the transformer and VAE take its place at 11.9 GiB. Peak is the larger stage rather than the 19 GiB total, and both run at full GPU speed. Block streaming additionally pages transformer blocks in from system RAM; slower per step, but it fits a smaller card.</small>}
            <button className="render-button" disabled={!canRun || running}
              onClick={() => { void (engine === "kiwi" ? run() : runRetake()); }}>
              <WandSparkles size={16} /> {running ? "Worker running…"
                : engine === "kiwi" ? "Edit the selected range" : "Retake the selected range"}
            </button>
            {running && <div className="ltx-job"><span className="ltx-spinner" /><div>
              <strong>{engine === "kiwi" ? "Kiwi-Edit" : "LTX Retake"} running · {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}</strong>
              <small>{engine === "retake" ? "The LTX 2.3 worker and compatible pipeline stay resident for the next retake." : "Encoding the range, denoising, and decoding locally. Nothing leaves this machine."}</small>
            </div></div>}
            {result && <div className="ltx-job complete"><span>✓</span><div>
              <strong>Edit ready</strong><small>{result.asset.path}</small>
            </div></div>}
            {/* The worker prints its window plan, offload profile, peak VRAM and calibration
                line on the way through. Until now the panel held that text and never showed it,
                so a run that succeeded left no trace of what it cost. Also appended to
                .aiv/worker.log, which survives closing this dialog. */}
            {result?.log && <details className="worker-log">
              <summary>What the worker reported</summary>
              <pre>{result.log}</pre>
            </details>}
            {failure && <div className="worker-failure">
              <div><strong>The worker failed</strong><button onClick={() => setFailure(undefined)} aria-label="Dismiss"><X size={12} /></button></div>
              <pre>{failure}</pre>
            </div>}
            {runtime && !engineReady && <small className="runtime-checks">
              <code>{engine === "kiwi" ? kiwiCopy(runtime.kiwiEditState).detail
                : "LTX Retake needs the LTX runtime, the FP8 checkpoint and the Gemma encoder installed."}</code>
            </small>}
          </div>
        </div>

        <div className="dialog-actions">
          <button onClick={() => setClipEditorOpen(false)} disabled={running}>Close</button>
          <button disabled={!result || running} onClick={() => {
            if (result && clip) { insertAssetAfterClip(clip.id, result.asset); setClipEditorOpen(false); }
          }}>Insert after clip</button>
          <button className="primary" disabled={!result || running} onClick={() => {
            if (result && clip) {
              replaceClipRangeWithAsset(
                clip.id,
                result.effectiveStartOffset ?? rangeStart,
                result.effectiveEndOffset ?? rangeEnd,
                result.asset,
              );
              setClipEditorOpen(false);
            }
          }}>Replace selected range</button>
        </div>
        </>}
      </>}
    </section>
  </div>;
}

// ---------------------------------------------------------------------------
// Inspector
// ---------------------------------------------------------------------------

/**
 * The upscale chain, as two independent stages plus the readout that resolves them.
 *
 * The order between the stages is not offered as a choice. Real-ESRGAN always runs first and the
 * resample always second, because resampling up and then asking the model to reconstruct detail
 * from interpolated pixels is worse than either stage alone -- so there is no reorder control here
 * to get wrong. What the user picks is the target factor and, separately, whether a model is
 * involved at all; `resolveUpscalePlan` turns that into the concrete chain shown in the readout,
 * and the exact same call builds the arguments the exporter receives.
 */
function UpscaleChainControls({ settings, patch, source }: {
  settings: ExportSettings;
  patch: (patch: Partial<ExportSettings>) => void;
  source: { width: number; height: number };
}) {
  const [runtime, setRuntime] = useState<AiRuntimeStatus>();
  useEffect(() => {
    let cancelled = false;
    void runtimeStatus().then((value) => { if (!cancelled) setRuntime(value); }).catch(() => undefined);
    return () => { cancelled = true; };
  }, []);

  // No NVENC gate. Neither stage requires it: the AI stage decodes to PNG, runs a network and
  // re-encodes with whatever codec is selected, and the resample stage falls back to swscale where
  // scale_cuda is unavailable. Gating the control on NVENC made the entire upscale panel dead on
  // an x264 export -- and a disabled control cannot explain itself, because Chromium fires no
  // mouse events on one, so its tooltip never appeared either.
  const installedEsrgan = runtime?.realEsrganModels ?? [];
  const esrganModels = AI_UPSCALE_MODELS.filter(
    (model) => model.backend === "esrgan" && model.file !== undefined && installedEsrgan.includes(model.file),
  );
  // Each backend is installed independently -- Real-ESRGAN is a model pack, NVIDIA VFX is a pip
  // wheel -- so each type reports its own readiness rather than sharing one verdict.
  const ready: Record<UpscaleType, boolean> = {
    nvidia: Boolean(runtime?.nvvfxReady),
    esrgan: esrganModels.length > 0,
    resample: true,
  };
  const unavailable = (type: UpscaleType) => !runtime
    ? "Checking which upscalers are installed…"
    : type === "nvidia"
      ? `NVIDIA VFX is not installed (${runtime.nvvfxState}). Run Install-RTX-Upscale.bat -AcceptLicense.`
      : `No Real-ESRGAN weights are installed (${runtime.realEsrganState}). Run .venv\\Scripts\\python.exe scripts\\models.py install real-esrgan.`;

  const factors = upscaleFactorsFor(settings.upscaleType);
  const request = upscaleRequest(settings);
  const plan = resolveUpscalePlan({
    source,
    factor: request.factor,
    aiModelId: request.aiModelId,
    resample: settings.upscaleResample,
    maxDimension: ENCODER_MAX_DIMENSION[settings.encoder],
    maxDimensionReason: ENCODER_CEILING_NOTE[settings.encoder],
    medium: "video",
  });
  const selected = aiUpscaleModel(request.aiModelId);
  const typeHint = UPSCALE_TYPES.find((item) => item.value === settings.upscaleType)?.hint;

  return <>
    <ToggleControl label="Upscale" value={settings.upscaleEnabled}
      onChange={(value) => patch({ upscaleEnabled: value })}
      hint="A second full pass over the finished master: decode every frame, run the model, re-encode. It runs on any encoder, and it is not a filter — expect it to take longer than the render itself." />

    {settings.upscaleEnabled && <>
      <SelectControl label="Upscaler" value={settings.upscaleType}
        onChange={(value) => patch({ upscaleType: value as UpscaleType })}
        options={UPSCALE_TYPES.map((item) => ({
          value: item.value,
          label: ready[item.value] ? item.label : `${item.label} · not installed`,
        }))}
        hint={ready[settings.upscaleType] ? typeHint : unavailable(settings.upscaleType)} />

      <SelectControl label="Amount" value={String(settings.upscaleFactor)}
        onChange={(value) => patch({ upscaleFactor: Number(value) })}
        options={factors.map((value) => ({
          value: String(value),
          label: `${value}× · ${Math.round(source.width * value / 2) * 2}×${Math.round(source.height * value / 2) * 2}`,
        }))}
        hint={settings.upscaleType === "nvidia"
          ? "NVIDIA's own tooling exposes 2×, 3× and 4×, and nothing between them — so those are the only ones offered here rather than accepted and then refused by the worker."
          : "How much larger than the export resolution the finished master should be."} />

      {settings.upscaleType === "esrgan" && <SelectControl label="Model"
        value={request.aiModelId ?? ""}
        onChange={(value) => patch({ upscaleAiModel: value })}
        disabled={!ready.esrgan}
        options={esrganModels.map((model) => ({ value: model.id, label: model.label }))}
        hint={ready.esrgan
          ? selected?.hint ?? "Which weights to load. Each is trained for a kind of source."
          : unavailable("esrgan")} />}

      {settings.upscaleType === "nvidia" && <SelectControl label="Quality" value={settings.upscaleQuality}
        onChange={(value) => patch({ upscaleQuality: value })}
        options={(aiUpscaleModel(NVIDIA_UPSCALE_MODEL_ID)?.qualityLevels ?? []).map((level) => ({
          value: level,
          label: level.startsWith("HIGHBITRATE_")
            ? `${level.replace("HIGHBITRATE_", "").toLowerCase()} · clean source`
            : level.toLowerCase(),
        }))}
        hint="How hard the model works, and which source it assumes. The plain levels also remove compression artefacts; the clean-source levels skip that and keep more of what is already there — right for footage that was never heavily compressed." />}

      {settings.upscaleType === "esrgan" && <SelectControl label="Tile" value={String(settings.upscaleTile)}
        onChange={(value) => patch({ upscaleTile: Number(value) })}
        options={[
          { value: "0", label: "Off · whole frame at once" },
          { value: "256", label: "256 px" }, { value: "512", label: "512 px" },
          { value: "1024", label: "1024 px" },
        ]}
        hint="VRAM scales with the square of this, so a 4K frame at 4× will not fit un-tiled on 16 GB. Too small and the overlap has less to blend across, which shows as a faint grid — very visible once it crawls across a moving frame." />}

      {/* Only shown when a resample stage can actually exist. NVIDIA reaches the target directly,
          so on that path this control would govern a stage that never runs. */}
      {settings.upscaleType !== "nvidia" && <SelectControl label="Resample" value={settings.upscaleResample}
        onChange={(value) => patch({ upscaleResample: value as ResampleAlgo })}
        options={RESAMPLE_ALGOS.map((item) => ({ value: item.value, label: item.label }))}
        hint={RESAMPLE_ALGOS.find((item) => item.value === settings.upscaleResample)?.hint} />}
    </>}

    {settings.upscaleEnabled && <div className="upscale-readout">
      <span>Resolved chain</span>
      <code>{plan.readout}</code>
      {plan.clamped && <p className="upscale-clamped">
        Requested ×{plan.clamped.requested}, capped at ×{plan.clamped.applied}. {plan.clamped.reason}
      </p>}
      {plan.warnings.map((warning) => <p key={warning} className="upscale-warning">{warning}</p>)}
    </div>}
  </>;
}

/**
 * Background removal for the selected clip: BiRefNet (automatic), SAM 2 (picked subject), or both
 * (SAM 2 picks, BiRefNet refines the edge), optionally with CoTracker3 track assist. The matte is
 * the clip's alpha in the monitor and the export; see domain/clipMatte.ts.
 */
function BackgroundSection({ clip, asset }: { clip?: TimelineClip; asset?: MediaAsset }) {
  const { busy, setBusy, setStatus, setClipMatte, mattePickClipId, setMattePick } = useEditor();
  const [runtime, setRuntime] = useState<AiRuntimeStatus>();
  const [running, setRunning] = useState(false);
  const [failure, setFailure] = useState<string>();
  useEffect(() => { void runtimeStatus().then(setRuntime).catch(() => undefined); }, []);
  useEffect(() => setFailure(undefined), [clip?.id]);
  const picture = Boolean(clip && asset && asset.kind !== "audio" && !clip.graphic);
  if (!clip || !asset || !picture) {
    return <Section title="Background removal" accent="—" hint="Cut the subject out of a clip with BiRefNet and/or SAM 2.">
      <small className="panel-help">Select a video or image clip to remove its background.</small>
    </Section>;
  }
  const matte = clip.matte ?? defaultClipMatte();
  const range = matteRange(clip, asset);
  const sam2 = usesSam2(matte.method);
  const blocker = matteBlocker(matte, range);
  const needs = sam2 ? (matte.method === "sam2" ? runtime?.sam2Ready : runtime?.sam2Ready && runtime?.birefnetReady) : runtime?.birefnetReady;
  const picking = mattePickClipId === clip.id;
  const made = Boolean(matte.cache);
  const stale = matteIsStale(clip.matte);
  const short = matteIsShort(clip.matte, clip);
  const keep = matte.points.filter((point) => point.include).length;
  const frames = new Set(matte.points.map((point) => point.frame)).size;
  const set = (patch: Partial<ClipMatte>, status?: string) => setClipMatte(clip.id, { ...matte, ...patch }, status);
  const accent = !clip.matte ? "OFF" : !made ? "NOT MADE" : !matte.enabled ? "DISABLED" : stale || short ? "REMAKE" : "ON";

  const run = async () => {
    setBusy(true); setRunning(true); setFailure(undefined); setMattePick(undefined);
    try {
      const result = await makeClipMatte(clip, asset, matte, setStatus);
      const { reused, prompts, ...cache } = result;
      // The clip as it is now: sliders moved while the job ran are kept.
      const now = useEditor.getState().project.sequences.flatMap((sequence) => sequence.tracks.flatMap((track) => track.clips)).find((item) => item.id === clip.id);
      setClipMatte(clip.id, { ...(now?.matte ?? matte), enabled: true, cache },
        `Background removed from ${clip.name} · ${cache.frameCount} frame${cache.frameCount === 1 ? "" : "s"}${sam2 ? ` · ${prompts} SAM 2 point${prompts === 1 ? "" : "s"}` : ""}${reused ? " · reused the cached matte" : ""}`);
    } catch (error) {
      setFailure(errorText(error));
      setStatus("Background removal failed: " + errorHeadline(error));
    } finally { setRunning(false); setBusy(false); }
  };

  return <Section title="Background removal" accent={accent}
    hint="Cuts the subject out of this clip. Whatever is under it shows through: put the clip on V2 over a background on V1.">
    <SelectControl label="Method" value={matte.method} disabled={running}
      onChange={(method) => set({ method: method as ClipMatteMethod })}
      options={CLIP_MATTE_METHODS.map((item) => ({ value: item.value, label: item.label }))}
      hint={CLIP_MATTE_METHODS.map((item) => `${item.label}: ${item.hint}`).join("\n")} />
    {sam2 && <>
      <div className="mask-points">
        <button className={picking ? "arm active" : "arm"} disabled={running} onClick={() => setMattePick(picking ? undefined : clip.id)}
          title="While armed, clicking this clip in the program monitor adds a point. Shift-click marks something to remove. Points can go on any frame: move the playhead and click again where SAM 2 needs help.">
          {picking ? "Picking — click the monitor" : "Pick subject in monitor"}
        </button>
        <span>{keep} keep · {matte.points.length - keep} remove{frames > 1 ? ` · ${frames} frames` : ""}</span>
        <button disabled={!matte.points.length || running} onClick={() => set({ points: [] }, "Points cleared")}>Clear</button>
      </div>
      <ToggleControl label="Track assist (CoTracker3)" value={matte.trackAssist} disabled={running}
        onChange={(trackAssist) => set({ trackAssist })}
        hint={`CoTracker3 follows your points through the clip and SAM 2 is given them again every ${ASSIST_EVERY} frames where they are visible. Keeps SAM 2 on a subject that turns, is briefly covered, or leaves and comes back. Adds a tracking pass before SAM 2.`} />
    </>}
    {matte.method !== "sam2" && <RangeControl label="Edge smoothing" value={Math.round(matte.smooth * 100)} disabled={running} min={0} max={60} suffix="%"
      onChange={(value) => set({ smooth: value / 100 })}
      hint="Blends each frame's BiRefNet edge with the previous one: steadier hair and edges, a little lag on fast motion. Needs a remake." />}
    <button className="render-button" disabled={running || busy || Boolean(blocker) || !needs} onClick={() => { void run(); }}
      title={blocker ?? (needs ? undefined : "The model this method needs is not installed")}>
      <Layers3 size={16} /> {running ? "Removing background…" : made ? (stale || short ? "Remake with these settings" : "Remake") : "Remove background"}
    </button>
    {blocker && !running && <small className="panel-help">{blocker}</small>}
    {runtime && !needs && <small className="runtime-checks">{sam2 && !runtime.sam2Ready ? `SAM 2: ${runtime.sam2State}. ` : ""}{matte.method !== "sam2" && !runtime.birefnetReady ? "BiRefNet is not installed. " : ""}Run scripts\setup-image-studio.ps1.</small>}
    {running && <div className="ltx-job"><span className="ltx-spinner" /><div><strong>Removing the background</strong><small>{range.frameCount} frame{range.frameCount === 1 ? "" : "s"}; the matte is cached, so a remake with the same settings is instant.</small></div></div>}
    {failure && <div className="worker-failure"><div><strong>Background removal failed</strong><button onClick={() => setFailure(undefined)} aria-label="Dismiss"><X size={12} /></button></div><pre>{failure}</pre></div>}
    {made && <>
      {(stale || short) && <small className="panel-help">{short ? "The clip was trimmed past the matte: frames outside it hold the nearest matte. Remake to cover the new range." : "Settings changed since the matte was made: Remake to apply them."}</small>}
      <ToggleControl label="Show cutout" value={matte.enabled} onChange={(enabled) => set({ enabled })}
        hint="Off shows the clip whole without throwing the matte away." />
      <RangeControl label="Edge feather" value={matte.feather} min={0} max={40} step={0.5} suffix="px" onChange={(feather) => set({ feather })}
        hint="Softens the cut edge. Applies live, no remake." />
      <RangeControl label="Grow / shrink" value={matte.expand} min={-30} max={30} suffix="px" onChange={(expand) => set({ expand })}
        hint="Grows the kept area outward (positive) or eats into it (negative) to lose a halo. Applies live." />
      <ToggleControl label="Invert (remove the subject)" value={matte.invert} onChange={(invert) => set({ invert })}
        hint="Keeps the background and cuts the subject out instead." />
    </>}
    {clip.matte && <button className="ghost-button" disabled={running} onClick={() => { setMattePick(undefined); setClipMatte(clip.id, undefined); }}>Remove background removal</button>}
    <small className="panel-help">{clip.trackId === "v1"
      ? "Tip: on V1 the cutout is over black. Put the clip on V2 and a background clip on V1 to composite; keep V2's box at Full frame (the default)."
      : "The cutout composites over the tracks below it, in the monitor and the export."}</small>
  </Section>;
}

function Inspector() {
  const {
    activePanel, setActivePanel, selectedClipId, project, updateSelectedClip, removeSourceRangeFromClip,
    replaceSelectedClipWithAsset, repointClipToAsset, addCaptionCues, setClipEditorOpen, setBusy, setStatus, busy,
  } = useEditor();
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId)!;
  const clip = sequence.tracks.flatMap((track) => track.clips).find((item) => item.id === selectedClipId);
  const clipAsset = project.assets.find((asset) => asset.id === clip?.assetId);
  const clipTrack = sequence.tracks.find((track) => track.id === clip?.trackId);
  const siblings = (clipTrack?.clips ?? []).slice().sort((a, b) => seconds(a.start) - seconds(b.start));
  const position = clip ? siblings.findIndex((item) => item.id === clip.id) : -1;
  const neighbours = { previous: position > 0 ? siblings[position - 1] : undefined, next: position >= 0 ? siblings[position + 1] : undefined };
  const [pauseAnalysis, setPauseAnalysis] = useState<PauseAnalysisResult>();
  const [legacyExport, setLegacyExport] = useState(readLegacyExport);
  const [denoiseStrength, setDenoiseStrength] = useState("medium");
  const [denoiseNormalize, setDenoiseNormalize] = useState(false);
  const [denoiseResult, setDenoiseResult] = useState<DenoiseResult>();
  const persistExportSettings = useEditor((s) => s.setExportSettings);
  const [exportSettings, setExportSettings] = useState<ExportSettings>(
    () => resolveExportSettings(useEditor.getState().project),
  );
  const patchExport = (patch: Partial<ExportSettings>) => setExportSettings((current) => {
    const next = { ...current, ...patch };
    // Changing container or encoder can invalidate the other choices, so repair them here rather
    // than letting ffmpeg fail on pass four of five. NVENC and x26x do not share preset names.
    const support = CONTAINER_SUPPORT[next.container];
    if (!support.video.includes(next.encoder)) next.encoder = support.video[0];
    if (!support.audio.includes(next.audioCodec)) next.audioCodec = support.audio[0];
    if (!ENCODER_PRESETS[next.encoder].some((option) => option.value === next.preset)) {
      next.preset = defaultPresetFor(next.encoder);
    }
    // The upscale used to be wiped here on any non-NVENC encoder, silently, in the middle of an
    // unrelated patch -- pick libx264 and the factor you set went back to 1 with nothing said. It
    // was never true that both stages need CUDA: the AI stage decodes to PNG, runs a network and
    // re-encodes with whatever codec is chosen, and the resample stage now falls back to swscale
    // when scale_cuda is unavailable. Nothing to clamp.
    persistExportSettings(next);
    return next;
  });
  // 0 x 0 means "follow the sequence", so the dropdown key is derived rather than stored -- that
  // way changing the sequence size does not silently strand the export at an old resolution.
  const resolutionKey = exportSettings.width === 0 || exportSettings.height === 0
    ? "sequence"
    : (["3840x2160", "2560x1440", "1920x1080", "1280x720", "1080x1920", "1080x1080"]
        .includes(`${exportSettings.width}x${exportSettings.height}`)
        ? `${exportSettings.width}x${exportSettings.height}`
        : "custom");
  const applyResolutionPreset = (value: string) => {
    if (value === "sequence") return patchExport({ width: 0, height: 0 });
    if (value === "custom") return patchExport({ width: sequence.width, height: sequence.height });
    const [width, height] = value.split("x").map(Number);
    patchExport({ width, height });
  };
  const effect = effectsFor(clip?.effects);
  const speedRate = clip ? clipPlaybackRate(clip) : 1;
  const setSpeedRate = (rate: number) => updateSelectedClip({
    speed: [{ at: fromSeconds(0), rate: Math.max(0.25, Math.min(4, rate)), easing: "linear" }],
  });
  // Retake moved into the clip editor popout, which seeds its own range from the playhead.
  useEffect(() => { setPauseAnalysis(undefined); setDenoiseResult(undefined); }, [clip?.id]);
  // Non-destructive: the cleaned audio is a NEW file beside the source, and the clip is re-pointed
  // at it. The original is never written to, so undo — or a second denoise at a different strength —
  // costs nothing and loses nothing.
  const denoiseClipAudio = async () => {
    if (!clip || !clipAsset) return;
    setBusy(true);
    setStatus("Cleaning the clip audio with FFmpeg's tracking spectral denoiser…");
    try {
      const result = await invoke<DenoiseResult>("denoise_clip_audio", { request: {
        path: clipAsset.path, strength: denoiseStrength, highpassHz: 60, normalize: denoiseNormalize,
      } });
      const probe = await probeMedia(result.outputPath);
      const cleaned = generatedAsset(result.outputPath, probe);
      setDenoiseResult(result);
      repointClipToAsset(clip.id, cleaned, `Denoised ${clip.name} (${denoiseStrength}) — original untouched`);
    } catch (error) {
      setStatus("Denoise failed: " + errorHeadline(error));
    } finally {
      setBusy(false);
    }
  };
  const analyzePauses = async () => {
    if (!clip || !clipAsset || clip.trackId !== "v1") return;
    setBusy(true);
    setStatus("Analyzing repeated frames and silence locally…");
    try {
      const result = await invoke<PauseAnalysisResult>("analyze_clip_pauses", { request: {
        path: clipAsset.path, sourceInSeconds: seconds(clip.sourceIn),
        durationSeconds: clipDuration(clip.sourceIn, clip.sourceOut), hasAudio: Boolean(clipAsset.hasAudio),
        minimumDuration: 0.35,
      } });
      setPauseAnalysis(result);
      setStatus(result.candidates.length ? "Found " + result.candidates.length + " removable pause candidate" + (result.candidates.length === 1 ? "" : "s") : "No matching freeze and silence pauses found");
    } catch (error) {
      setStatus("Pause analysis failed: " + errorHeadline(error));
    } finally {
      setBusy(false);
    }
  };
  const transcribeClip = async () => {
    if (!clip || !clipAsset) return;
    setBusy(true);
    setStatus("Transcribing with Whisper large-v3-turbo locally…");
    try {
      const cues = await invoke<TranscriptCue[]>("transcribe_media", { request: { path: clipAsset.path, translate: false } });
      // Whisper timings are source-relative; shift them onto the clip's timeline position.
      const offset = seconds(clip.start);
      const from = seconds(clip.sourceIn);
      const to = seconds(clip.sourceOut);
      const rate = clipPlaybackRate(clip);
      const placed = cues
        .filter((cue) => cue.endSeconds > from && cue.startSeconds < to)
        .map((cue) => ({
          startSeconds: offset + (Math.max(cue.startSeconds, from) - from) / rate,
          endSeconds: offset + (Math.min(cue.endSeconds, to) - from) / rate,
          text: cue.text,
        }));
      if (!placed.length) {
        setStatus("Whisper found no speech inside this clip's range");
        return;
      }
      addCaptionCues(placed);
    } catch (error) {
      setStatus("Transcription failed: " + errorHeadline(error));
    } finally {
      setBusy(false);
    }
  };
  const updateGraphic = (patch: Partial<GraphicOverlay>) => {
    if (!clip?.graphic) return;
    const graphic = { ...clip.graphic, ...patch };
    updateSelectedClip({ graphic, name: patch.text ?? clip.name });
  };
  return <aside className="inspector">
    <div className="inspector-tabs">
      <button className={activePanel === "inspector" ? "selected" : ""} onClick={() => setActivePanel("inspector")}><SlidersHorizontal size={14} /> Inspector</button>
      <button className={activePanel === "ai" ? "selected" : ""} onClick={() => setActivePanel("ai")}><Bot size={14} /> AI</button>
      <button className={activePanel === "export" ? "selected" : ""} onClick={() => setActivePanel("export")}><Download size={14} /> Export</button>
    </div>
    {activePanel === "inspector" && <div className="inspector-content">
      <div className="clip-heading"><div className="mini-thumb" /><div><strong>{clip?.name ?? "No clip selected"}</strong><small>{clip ? "Changes preview live and render on export" : "Select or right-click a timeline clip"}</small></div></div>
      {clip?.graphic ? <Section title="Graphic">
        <SelectControl label="Template" value={clip.graphic.template} onChange={(template) => updateGraphic({ template: template as GraphicTemplate })} options={GRAPHIC_TEMPLATES.map((item) => ({ value: item.id, label: item.name }))} />
        <label className="field text-field"><span>Text</span><input value={clip.graphic.text} onChange={(event) => updateGraphic({ text: event.target.value })} /></label>
        <label className="field text-field"><span>Subtitle</span><input value={clip.graphic.subtitle} onChange={(event) => updateGraphic({ subtitle: event.target.value })} /></label>
        <NumberControl label="Duration" value={clipDuration(clip.sourceIn, clip.sourceOut)} min={1} max={60} step={0.25} suffix="s" onChange={(duration) => updateSelectedClip({ sourceOut: fromSeconds(seconds(clip.sourceIn) + duration) })} />
        <RangeControl label="Font size" value={clip.graphic.fontSize} min={12} max={240} suffix="px" onChange={(fontSize) => updateGraphic({ fontSize })} />
        <SelectControl label="Backdrop" value={clip.graphic.backdrop ?? "box"}
          onChange={(backdrop) => updateGraphic({ backdrop: backdrop as TextBackdrop })} options={[
            { value: "none", label: "Transparent" }, { value: "box", label: "Filled box" },
            { value: "outline", label: "Outline frame" }, { value: "underline", label: "Underline" },
          ]} />
        <SelectControl label="Font" value={clip.graphic.fontFamily ?? "display"}
          onChange={(fontFamily) => updateGraphic({ fontFamily: fontFamily as TextFontFamily })} options={[
            { value: "display", label: "Display" }, { value: "sans", label: "Clean sans" },
            { value: "serif", label: "Cinema serif" }, { value: "mono", label: "Tech mono" },
          ]} />
        <SelectControl label="Weight" value={String(clip.graphic.fontWeight ?? 700)}
          onChange={(fontWeight) => updateGraphic({ fontWeight: Number(fontWeight) as 400 | 700 | 900 })} options={[
            { value: "400", label: "Regular" }, { value: "700", label: "Bold" }, { value: "900", label: "Black" },
          ]} />
        <RangeControl label="Text outline" value={clip.graphic.strokeWidth ?? 0} min={0} max={12} suffix="px" onChange={(strokeWidth) => updateGraphic({ strokeWidth })} />
        <RangeControl label="Shadow" value={clip.graphic.shadowOffset ?? 0} min={0} max={20} suffix="px" onChange={(shadowOffset) => updateGraphic({ shadowOffset })} />
        {(clip.graphic.backdrop ?? "box") !== "none" && <RangeControl label="Backdrop opacity" value={Math.round((clip.graphic.backgroundOpacity ?? 1) * 100)} min={0} max={100} suffix="%" onChange={(backgroundOpacity) => updateGraphic({ backgroundOpacity: backgroundOpacity / 100 })} />}
        <SelectControl label="Animate in" value={clip.graphic.motionIn ?? "none"}
          onChange={(motionIn) => updateGraphic({ motionIn: motionIn as TextMotionIn })} options={TEXT_MOTION_IN} />
        <SelectControl label="Animate out" value={clip.graphic.motionOut ?? "none"}
          onChange={(motionOut) => updateGraphic({ motionOut: motionOut as TextMotionOut })} options={TEXT_MOTION_OUT} />
        <SelectControl label="While visible" value={clip.graphic.motionLoop ?? "none"}
          onChange={(motionLoop) => updateGraphic({ motionLoop: motionLoop as TextMotionLoop })} options={TEXT_MOTION_LOOPS} />
        <SelectControl label="Easing" value={clip.graphic.motionEasing ?? "smooth"}
          onChange={(motionEasing) => updateGraphic({ motionEasing: motionEasing as TextMotionEasing })} options={TEXT_MOTION_EASINGS} />
        <RangeControl label="Animation time" value={clip.graphic.motionDuration ?? 0.6} min={0.1}
          max={Math.max(0.1, Math.min(3, clipDuration(clip.sourceIn, clip.sourceOut) / 2))} step={0.05} suffix="s"
          onChange={(motionDuration) => updateGraphic({ motionDuration })} />
        <label className="color-control"><span>Text color</span><input type="color" value={clip.graphic.textColor.slice(0, 7)} onChange={(event) => updateGraphic({ textColor: event.target.value })} /></label>
        <label className="color-control"><span>Backdrop</span><input type="color" value={clip.graphic.backgroundColor.slice(0, 7)} onChange={(event) => updateGraphic({ backgroundColor: event.target.value })} /></label>
        <label className="color-control"><span>Accent</span><input type="color" value={clip.graphic.accentColor.slice(0, 7)} onChange={(event) => updateGraphic({ accentColor: event.target.value })} /></label>
        <label className="color-control"><span>Text outline</span><input type="color" value={(clip.graphic.strokeColor ?? "#000000").slice(0, 7)} onChange={(event) => updateGraphic({ strokeColor: event.target.value })} /></label>
        <RangeControl label="Opacity" value={Math.round(clip.opacity * 100)} min={0} max={100} suffix="%" onChange={(opacity) => updateSelectedClip({ opacity: opacity / 100 })} />
        <label className="check-row"><input type="checkbox" checked={clip.enabled} onChange={(event) => updateSelectedClip({ enabled: event.target.checked })} /> Enabled in preview and export</label>
      </Section> : <>
        <Section title="Transform">
          <NumberControl label="Position X" value={clip?.positionX ?? sequence.width / 2} disabled={!clip} min={-sequence.width} max={sequence.width * 2} onChange={(positionX) => updateSelectedClip({ positionX })} hint="Horizontal centre of the clip in sequence pixels. Half the sequence width is centred." />
          <NumberControl label="Position Y" value={clip?.positionY ?? sequence.height / 2} disabled={!clip} min={-sequence.height} max={sequence.height * 2} onChange={(positionY) => updateSelectedClip({ positionY })} hint="Vertical centre of the clip in sequence pixels. Half the sequence height is centred." />
          <RangeControl label="Scale" value={clip?.scale ?? 100} disabled={!clip} min={10} max={300} suffix="%" onChange={(scale) => updateSelectedClip({ scale })} hint="Uniform zoom. 100% fits the clip to the sequence frame." />
          <NumberControl label="Rotation" value={clip?.rotation ?? 0} disabled={!clip} min={-360} max={360} step={0.1} suffix="°" onChange={(rotation) => updateSelectedClip({ rotation })} hint="Clockwise rotation in degrees, applied around the clip centre." />
          <RangeControl label="Opacity" value={Math.round((clip?.opacity ?? 1) * 100)} disabled={!clip} min={0} max={100} suffix="%" onChange={(opacity) => updateSelectedClip({ opacity: opacity / 100 })} />
        </Section>
        <Section title="Aspect and crop" accent={clip?.conform ? "override" : "sequence"}>
          <ConformControls disabled={!clip} value={clip?.conform ?? sequence.conform ?? DEFAULT_CONFORM}
            onChange={(conform) => updateSelectedClip({ conform })} />
          {clip?.conform && <button className="ghost-button" onClick={() => updateSelectedClip({ conform: undefined })}>
            Reset to sequence default
          </button>}
        </Section>
        <BackgroundSection clip={clip} asset={clipAsset} />
        <Section title="Transitions" accent={clip ? [clip.transitionIn, clip.transitionOut].filter((id) => id && id !== "none").length + "/2" : undefined}>
          <TransitionControl clip={clip} neighbours={neighbours} />
        </Section>
        <Section title="Effects">
          <RangeControl label="Brightness" value={effect.brightness} disabled={!clip} min={-1} max={1} step={0.01} onChange={(brightness) => updateSelectedClip({ effects: { ...effect, brightness } })} hint="Adds or subtracts light. 0 is unchanged; negative darkens." />
          <RangeControl label="Contrast" value={effect.contrast} disabled={!clip} min={0} max={2} step={0.01} onChange={(contrast) => updateSelectedClip({ effects: { ...effect, contrast } })} hint="Separation between lights and darks. 1.0 is unchanged." />
          <RangeControl label="Saturation" value={effect.saturation} disabled={!clip} min={0} max={3} step={0.01} onChange={(saturation) => updateSelectedClip({ effects: { ...effect, saturation } })} hint="Colour intensity. 0 is monochrome, 1.0 unchanged." />
          <RangeControl label="Gamma" value={effect.gamma} disabled={!clip} min={0.1} max={3} step={0.01} onChange={(gamma) => updateSelectedClip({ effects: { ...effect, gamma } })} hint="Midtone response without moving black or white points." />
          <RangeControl label="Hue" value={effect.hue} disabled={!clip} min={-180} max={180} suffix="°" onChange={(hue) => updateSelectedClip({ effects: { ...effect, hue } })} hint="Rotates every colour around the wheel, in degrees." />
          <RangeControl label="Blur" value={effect.blur} disabled={!clip} min={0} max={20} step={0.1} suffix="px" onChange={(blur) => updateSelectedClip({ effects: { ...effect, blur } })} hint="Gaussian softening radius in pixels." />
          <RangeControl label="Sharpen" value={effect.sharpen} disabled={!clip} min={0} max={2} step={0.05} onChange={(sharpen) => updateSelectedClip({ effects: { ...effect, sharpen } })} hint="Unsharp mask amount. Small values are usually enough." />
          <RangeControl label="Vignette" value={effect.vignette} disabled={!clip} min={0} max={1} step={0.01} onChange={(vignette) => updateSelectedClip({ effects: { ...effect, vignette } })} hint="Darkens the frame edges to draw the eye inward." />
          <RangeControl label="Noise" value={effect.noise} disabled={!clip} min={0} max={50} step={1} onChange={(noise) => updateSelectedClip({ effects: { ...effect, noise } })} hint="Adds film-like grain. Useful for matching generated footage to real footage." />
          <button className="reset-control" disabled={!clip} onClick={() => updateSelectedClip({ effects: effectsFor() })}>Reset effects</button>
        </Section>
        <Section title="Speed">
          <RangeControl label="Playback speed" value={Math.round(speedRate * 100)} disabled={!clip} min={25} max={400} step={1} suffix="%" onChange={(percent) => setSpeedRate(percent / 100)} hint="Retimes the clip. Audio is time-stretched without changing pitch." />
          <NumberControl label="Timeline duration" value={clip ? timelineClipDuration(clip) : 0} disabled={!clip} min={0.1} max={3600} step={0.1} suffix="s" onChange={(duration) => clip && setSpeedRate(clipDuration(clip.sourceIn, clip.sourceOut) / Math.max(0.1, duration))} hint="How long the clip occupies the timeline. Editing this sets the speed to match." />
          <div className="speed-presets">{[25, 50, 100, 200, 400].map((percent) => <button className={Math.round(speedRate * 100) === percent ? "active" : ""} disabled={!clip} key={percent} onClick={() => setSpeedRate(percent / 100)}>{percent}%</button>)}</div>
          <small className="panel-help">Speed changes update timeline duration, preview playback, source trimming, audio pitch-preserving time stretch, and export.</small>
        </Section>
        <Section title="AI tools">
          <button className="render-button cleanup-analyze" disabled={!clip || clip.trackId !== "v1" || !clipAsset || busy} onClick={() => setClipEditorOpen(true)}>
            <Clapperboard size={15} /> Open the AI clip editor
          </button>
          <button className="render-button cleanup-analyze" disabled={!clip || clip.trackId !== "v1" || !clipAsset || busy} onClick={() => { void analyzePauses(); }}>
            <Gauge size={15} /> {busy ? "Working…" : "Find frozen and silent pauses"}
          </button>
          <button className="render-button cleanup-analyze" disabled={!clip || !clipAsset?.hasAudio || busy} onClick={() => { void transcribeClip(); }}>
            <Captions size={15} /> Auto-caption this clip
          </button>
          {pauseAnalysis && <div className="pause-results">
            <small>{pauseAnalysis.freezes.length} frozen range{pauseAnalysis.freezes.length === 1 ? "" : "s"} · {pauseAnalysis.silences.length} silent range{pauseAnalysis.silences.length === 1 ? "" : "s"}</small>
            {pauseAnalysis.candidates.length === 0 ? <p>No overlapping freeze/silence range met the 0.35s threshold.</p> : pauseAnalysis.candidates.map((range, index) => {
              const handle = Math.min(0.08, range.durationSeconds * 0.2);
              return <div className="pause-candidate" key={range.startSeconds + ":" + range.endSeconds}><div><strong>Pause {index + 1}</strong><span>{range.startSeconds.toFixed(2)}s – {range.endSeconds.toFixed(2)}s · {range.durationSeconds.toFixed(2)}s</span></div><button onClick={() => { removeSourceRangeFromClip(clip!.id, range.startSeconds + handle, range.endSeconds - handle); setPauseAnalysis(undefined); }}>Remove + ripple</button></div>;
            })}
          </div>}
          <small className="panel-help">Pause removal intersects FFmpeg frozen-frame and silence detection, preserves short boundary handles and ripple-closes the timeline. Captions land on C1 in timeline time.</small>
        </Section>
        <Section title="Audio">
          <RangeControl label="Volume" value={clip?.volumeDb ?? 0} disabled={!clip} min={-60} max={12} step={0.5} suffix="dB" onChange={(volumeDb) => updateSelectedClip({ volumeDb })} hint="Clip gain in decibels. 0 dB is unchanged; -60 is silence." />
          <SelectControl label="Denoise strength" value={denoiseStrength} disabled={!clip || !clipAsset?.hasAudio || busy}
            onChange={setDenoiseStrength} options={[
              { value: "light", label: "Light · 6 dB, safest" },
              { value: "medium", label: "Medium · 12 dB" },
              { value: "strong", label: "Strong · 24 dB + patch pass" },
            ]}
            hint="How much noise to subtract. Past roughly 24 dB the residue starts to sound watery, which is why this is three deliberate steps rather than a slider. Strong adds a slower non-local-means pass to clean up what the spectral pass leaves behind." />
          <label className="check-row"><input type="checkbox" disabled={!clip || !clipAsset?.hasAudio || busy}
            checked={denoiseNormalize} onChange={(event) => setDenoiseNormalize(event.target.checked)} /> Normalise to −16 LUFS afterwards</label>
          <button className="render-button cleanup-analyze" disabled={!clip || !clipAsset?.hasAudio || busy}
            onClick={() => { void denoiseClipAudio(); }}>
            <AudioLines size={15} /> {busy ? "Working…" : "Denoise audio"}
          </button>
          {denoiseResult && <div className="pause-results">
            <small>Cleaned to {filename(denoiseResult.outputPath)}{denoiseResult.videoCopied ? " · picture stream-copied" : ""}</small>
            <p><code>{denoiseResult.filter}</code></p>
          </div>}
          <small className="panel-help">Tracks the noise floor continuously instead of needing a silent noise print, which is what generated foley and V2A output need — their artefacts drift with the content. Writes a new file beside the source and re-points the clip at it, so the original is never modified and undo restores it.</small>
        </Section>
        <Section title="Compositing">
          <SelectControl label="Blend mode" disabled={!clip} value={clip?.blendMode ?? "normal"}
            onChange={(blendMode) => updateSelectedClip({ blendMode: blendMode as BlendMode })}
            options={BLEND_MODES.map((item) => ({ value: item.value, label: item.label }))} hint="How this clip composites over the layer beneath it. Previews live and renders in the export; the legacy export renders every clip as Normal." />
          <small className="panel-help"><Blend size={10} /> Blend modes composite this clip against the layer beneath it in the program monitor, including across a transition. The FFmpeg export path is a linear concatenation, so it renders blend modes as Normal — check the monitor for the blended look.</small>
          <label className="check-row"><input type="checkbox" disabled={!clip} checked={clip?.enabled ?? false} onChange={(event) => updateSelectedClip({ enabled: event.target.checked })} /> Enabled in preview and export</label>
        </Section>
      </>}
    </div>}
    <AiPanel active={activePanel === "ai"} />
    {activePanel === "export" && <div className="inspector-content">
      <Section title="Master export">
        <SelectControl label="Format" value={exportSettings.container} onChange={(value) => patchExport({ container: value as ExportContainer })}
          options={[{ value: "mp4", label: "MP4 · widest support" }, { value: "mov", label: "MOV · QuickTime, PCM audio" }, { value: "mkv", label: "MKV · anything goes" }]}
          hint="Governs which codecs are legal. Changing it repairs any pairing the new container cannot hold." />
        <SelectControl label="Video codec" value={exportSettings.encoder} onChange={(value) => patchExport({ encoder: value as VideoEncoder })}
          options={CONTAINER_SUPPORT[exportSettings.container].video.map((value) => ({ value, label: VIDEO_ENCODER_LABELS[value] }))}
          hint="NVENC uses the GPU and is far faster. The libx entries are the CPU fallback: slower, slightly better per bit." />
        <SelectControl label="Rate control" value={exportSettings.qualityMode} onChange={(value) => patchExport({ qualityMode: value as QualityMode })}
          options={[{ value: "quality", label: "Constant quality · recommended" }, { value: "bitrate", label: "Target bitrate" }]}
          hint="Constant quality spends bits where the picture needs them. Pick target bitrate only when a delivery spec demands one." />
        {exportSettings.qualityMode === "quality"
          ? <NumberControl label={exportSettings.encoder.includes("nvenc") ? "Quality (CQ)" : "Quality (CRF)"} value={exportSettings.quality}
              onChange={(value) => patchExport({ quality: value })} min={12} max={40} step={1}
              hint="Lower is better and bigger. 19-23 is visually transparent for delivery; 28+ starts to show." />
          : <TextControl label="Bitrate" value={exportSettings.bitrate} onChange={(value) => patchExport({ bitrate: value })}
              hint="ffmpeg syntax, e.g. 20M or 8000k." />}
        <SelectControl label="Encoder speed" value={exportSettings.preset} onChange={(value) => patchExport({ preset: value })}
          options={ENCODER_PRESETS[exportSettings.encoder]}
          hint="Slower presets buy real quality at the same file size. This was previously left at the encoder default, which is its fastest and worst." />
      </Section>

      <Section title="Frame">
        <SelectControl label="Resolution" value={resolutionKey} onChange={applyResolutionPreset}
          options={[{ value: "sequence", label: `Sequence · ${sequence.width} × ${sequence.height}` }, { value: "3840x2160", label: "2160p · 3840 × 2160" },
            { value: "2560x1440", label: "1440p · 2560 × 1440" }, { value: "1920x1080", label: "1080p · 1920 × 1080" },
            { value: "1280x720", label: "720p · 1280 × 720" }, { value: "1080x1920", label: "Vertical · 1080 × 1920" },
            { value: "1080x1080", label: "Square · 1080 × 1080" }, { value: "custom", label: "Custom…" }]} />
        {resolutionKey === "custom" && <>
          <NumberControl label="Width" value={exportSettings.width || sequence.width} onChange={(value) => patchExport({ width: Math.max(2, Math.round(value / 2) * 2) })} min={2} step={2} />
          <NumberControl label="Height" value={exportSettings.height || sequence.height} onChange={(value) => patchExport({ height: Math.max(2, Math.round(value / 2) * 2) })} min={2} step={2} />
        </>}
        <SelectControl label="Frame rate" value={String(exportSettings.frameRate)} onChange={(value) => patchExport({ frameRate: Number(value) })}
          options={[{ value: "0", label: `Sequence · ${(sequence.frameRate.numerator / sequence.frameRate.denominator).toFixed(2)} fps` },
            { value: "24", label: "24" }, { value: "25", label: "25" }, { value: "30", label: "30" }, { value: "50", label: "50" }, { value: "60", label: "60" }]} />
        <UpscaleChainControls
          settings={exportSettings} patch={patchExport}
          source={{ width: exportSettings.width || sequence.width, height: exportSettings.height || sequence.height }} />
      </Section>

      <Section title="Audio">
        <SelectControl label="Codec" value={exportSettings.audioCodec} onChange={(value) => patchExport({ audioCodec: value as AudioCodec })}
          options={CONTAINER_SUPPORT[exportSettings.container].audio.map((value) => ({ value, label: AUDIO_CODEC_LABELS[value] }))} />
        {!["flac", "pcm_s16le", "none"].includes(exportSettings.audioCodec) &&
          <SelectControl label="Bitrate" value={exportSettings.audioBitrate} onChange={(value) => patchExport({ audioBitrate: value })}
            options={[{ value: "128k", label: "128 kb/s" }, { value: "192k", label: "192 kb/s" }, { value: "256k", label: "256 kb/s" }, { value: "320k", label: "320 kb/s" }]} />}
        <SelectControl label="Sample rate" value={String(exportSettings.audioSampleRate)} onChange={(value) => patchExport({ audioSampleRate: Number(value) })}
          options={[{ value: "44100", label: "44.1 kHz" }, { value: "48000", label: "48 kHz · video standard" }, { value: "96000", label: "96 kHz" }]} />
        <SelectControl label="Channels" value={String(exportSettings.audioChannels)} onChange={(value) => patchExport({ audioChannels: Number(value) })}
          options={[{ value: "1", label: "Mono" }, { value: "2", label: "Stereo" }]} />
      </Section>

      <Section title="Render">
        <button className="render-button" onClick={() => { void exportTimeline(exportSettings, sequence); }} disabled={busy}>
          <Download size={16} /> {busy ? "Working…" : "Choose file and render"}
        </button>
        <ToggleControl label="Legacy export (ffmpeg filter graphs)" value={legacyExport}
          onChange={(value) => { writeLegacyExport(value); setLegacyExport(value); }}
          hint="Off: every frame is rendered by the compositor the monitor uses (titles, transitions, overlays, clip effects and live mask VFX), then encoded once with the settings above. On: the previous multi-pass ffmpeg export, kept as a fallback." />
        <p className="panel-help">{legacyExport
          ? "Legacy: V1 timing, transforms, effects and transitions as ffmpeg filter graphs; V2 overlays; titles through drawtext; mixed audio. Intermediate passes run near-lossless."
          : "Renders exactly what the program monitor shows, frame by frame, and mixes every audio track, then encodes once with the settings above."}</p>
      </Section>
    </div>}
  </aside>;
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

function Clip({ clip, zoom, locked, trackKind, openMenu }: {
  clip: TimelineClip; zoom: number; locked: boolean; trackKind: string;
  openMenu: (event: ReactMouseEvent, clip: TimelineClip) => void;
}) {
  const { selectedClipId, groupSelection, setSelectedClip, moveClip, trimClip, splitClip, tool, project } = useEditor();
  const asset = project.assets.find((item) => item.id === clip.assetId);
  // Draw where the frame actually lands, not where it was authored. A dissolve with no handles is
  // paid for out of visible material, so the clips genuinely close up -- Shotcut and Kdenlive draw
  // the same way. Keeping this butt-joined while the export rippled is what made the timeline
  // ruler disagree with the exported file.
  const programStart = useMemo(() => {
    const sequence = project.sequences.find((item) => item.id === project.activeSequenceId);
    const v1 = (sequence?.tracks.find((item) => item.id === "v1")?.clips ?? [])
      .filter((item) => item.enabled).sort((a, b) => seconds(a.start) - seconds(b.start));
    const layout = resolveJunctionLayout(v1);
    return clip.trackId === "v1"
      ? layout.start.get(clip.id) ?? seconds(clip.start)
      : mapToProgramme(layout, v1, seconds(clip.start));
  }, [project.sequences, project.activeSequenceId, clip.trackId, clip.id, clip.start]);
  const start = programStart;
  const authoredStart = seconds(clip.start);
  const duration = timelineClipDuration(clip);
  const isAudioLane = trackKind === "audio";
  const thumbnail = useClipThumbnail(isAudioLane ? undefined : asset, seconds(clip.sourceIn) + 0.2);
  const peaks = useClipWaveform(asset, isAudioLane || Boolean(asset?.hasAudio));
  const sourceSpan = asset ? Math.max(0.001, seconds(asset.duration)) : 1;
  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    if (tool === "blade") {
      event.stopPropagation();
      const rect = event.currentTarget.getBoundingClientRect();
      splitClip(clip.id, authoredStart + (event.clientX - rect.left) / zoom);
      return;
    }
    if (locked) return;
    if ((event.target as HTMLElement).closest(".trim-left, .trim-right")) return;
    const originX = event.clientX;
    const originStart = authoredStart;
    const element = event.currentTarget;
    element.setPointerCapture(event.pointerId);
    const originY = event.clientY;
    const move = (next: PointerEvent) => {
      // Follow the pointer in both axes so it is obvious the clip can change lanes.
      element.style.transform = `translate(${next.clientX - originX}px, ${next.clientY - originY}px)`;
      element.style.zIndex = "40";
    };
    const up = (next: PointerEvent) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      element.style.transform = "";
      element.style.zIndex = "";
      // The dragged clip sits under the cursor and would be its own hit-test result, so take it out
      // of the hit test for the one call that asks what is underneath.
      element.style.pointerEvents = "none";
      const lane = (document.elementFromPoint(next.clientX, next.clientY) as HTMLElement | null)?.closest("[data-track-id]");
      element.style.pointerEvents = "";
      const targetTrack = lane?.getAttribute("data-track-id") ?? undefined;
      const movedX = Math.abs(next.clientX - originX) > 2;
      const changedTrack = Boolean(targetTrack && targetTrack !== clip.trackId);
      if (movedX || changedTrack) {
        moveClip(clip.id, originStart + (next.clientX - originX) / zoom, targetTrack);
      }
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  const startTrim = (edge: "left" | "right", event: ReactPointerEvent<HTMLElement>) => {
    event.preventDefault();
    event.stopPropagation();
    if (locked) return;
    const originX = event.clientX;
    const handle = event.currentTarget;
    handle.setPointerCapture(event.pointerId);
    const move = (next: PointerEvent) => {
      const delta = (next.clientX - originX) / zoom;
      handle.style.transform = "translateX(" + (edge === "left" ? Math.max(-30, delta * zoom) : Math.min(30, delta * zoom)) + "px)";
    };
    const up = (next: PointerEvent) => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      handle.style.transform = "";
      trimClip(clip.id, edge, (next.clientX - originX) / zoom);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };
  const width = Math.max(42, duration * zoom);
  const inWidth = clip.transitionIn && clip.transitionIn !== "none" ? Math.min(width / 2, (clip.transitionDuration ?? 0.5) * zoom) : 0;
  const outWidth = clip.transitionOut && clip.transitionOut !== "none" ? Math.min(width / 2, (clip.transitionOutDuration ?? 0.5) * zoom) : 0;
  return <div className={"timeline-clip lane-" + trackKind
      + (selectedClipId === clip.id ? " selected" : "")
      + (groupSelection.includes(clip.id) ? " grouped" : "")
      + (clip.enabled ? "" : " disabled-clip") + (locked ? " locked-clip" : "")}
    style={{ left: start * zoom, width, background: clip.color }}
    onContextMenu={(event) => openMenu(event, clip)} onPointerDown={onPointerDown}
    onClick={(event) => { event.stopPropagation(); if (tool !== "blade") setSelectedClip(clip.id); }}>
    {thumbnail
      ? <div className="clip-filmstrip" style={{ backgroundImage: `url("${thumbnail}")` }} />
      : <div className="clip-filmstrip" />}
    {peaks && <Waveform peaks={peaks} from={seconds(clip.sourceIn) / sourceSpan} to={seconds(clip.sourceOut) / sourceSpan} />}
    {inWidth > 0 && <i className="transition-ramp in" style={{ width: inWidth }} title={"In · " + transitionById(clip.transitionIn).name} />}
    {outWidth > 0 && <i className="transition-ramp out" style={{ width: outWidth }} title={"Out · " + transitionById(clip.transitionOut).name} />}
    {clip.vfxLayers?.some((layer) => layer.enabled) && <em className="clip-vfx-badge"
      title={`${clip.vfxLayers.filter((layer) => layer.enabled).length} non-destructive mask VFX layer(s)`}>VFX</em>}
    <strong>{clip.name}</strong><span>{formatTimecode(duration).slice(3)}</span>
    <i className="trim-left" title="Trim clip in" onPointerDown={(event) => startTrim("left", event)} />
    <i className="trim-right" title="Trim clip out" onPointerDown={(event) => startTrim("right", event)} />
  </div>;
}

const TOOLS: Array<{ id: TimelineTool; icon: React.ReactNode; title: string }> = [
  { id: "select", icon: <MousePointer2 size={16} />, title: "Selection tool (V)" },
  { id: "blade", icon: <Scissors size={16} />, title: "Blade tool (B) — click a clip to cut there" },
  { id: "track", icon: <Layers3 size={16} />, title: "Track select (A) — click a lane to select everything after that point" },
];

function Timeline() {
  const {
    project, zoom, setZoom, playhead, playing, setPlayhead, setSelectedClip, selectedAssetId, selectedClipId,
    addAssetToTimelineAt, deleteClip, rippleDeleteClip, duplicateClip, splitClip, setPlaying,
    updateSelectedClip, updateTrack, tool, setTool, selectTrackFrom, clearGroupSelection,
    groupSelection, setClipEditorOpen, addAssets, busy, setBusy, setStatus,
  } = useEditor();
  const [menu, setMenu] = useState<MenuState>(null);
  const [scrubbing, setScrubbing] = useState(false);
  const [renderedPreview, setRenderedPreview] = useState<TimelinePreviewResult>();
  const timelineGridRef = useRef<HTMLDivElement>(null);
  const viewportFrame = useRef<number | undefined>(undefined);
  const followPlayback = useRef(false);
  const manualScrollUntil = useRef(0);
  const [timelineViewport, setTimelineViewport] = useState({ left: 0, width: 1200 });
  const scrub = useRef<{ pointerId: number; lastAt: number; lastAudibleAt: number; lastAudioMs: number } | undefined>(undefined);
  const scrubSelectedClip = useRef(selectedClipId);
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId)!;
  const selectedAsset = project.assets.find((asset) => asset.id === selectedAssetId);
  const maxEnd = sequence.tracks.flatMap((track) => track.clips).reduce((end, clip) => Math.max(end, seconds(clip.start) + timelineClipDuration(clip)), 0);
  const duration = Math.max(36, Math.ceil(maxEnd + 10));
  const rulerTicks = useMemo(() => visibleTimelineTicks(duration, zoom, timelineViewport.left, timelineViewport.width),
    [duration, zoom, timelineViewport.left, timelineViewport.width]);
  const primaryClips = (sequence.tracks.find((track) => track.id === "v1")?.clips ?? [])
    .filter((clip) => clip.enabled).sort((a, b) => seconds(a.start) - seconds(b.start));
  const primaryLayout = resolveJunctionLayout(primaryClips);
  useLayoutEffect(() => {
    const element = timelineGridRef.current;
    if (!element) return;
    const update = () => setTimelineViewport({ left: element.scrollLeft, width: element.clientWidth });
    update();
    const observer = new ResizeObserver(update); observer.observe(element);
    return () => { observer.disconnect(); if (viewportFrame.current) cancelAnimationFrame(viewportFrame.current); };
  }, []);
  const updateTimelineViewport = (element: HTMLDivElement) => {
    if (viewportFrame.current) return;
    viewportFrame.current = requestAnimationFrame(() => {
      viewportFrame.current = undefined;
      setTimelineViewport({ left: element.scrollLeft, width: element.clientWidth });
    });
  };
  useEffect(() => {
    const element = timelineGridRef.current;
    if (!element || !playing) {
      followPlayback.current = false;
      return;
    }
    // A deliberate wheel/trackpad movement temporarily wins over follow. Playback reclaims the
    // viewport after a short inspection window, or as soon as the playhead reaches the edge again.
    if (performance.now() < manualScrollUntil.current) return;
    const result = timelineFollowScroll({
      playheadSeconds: playhead, zoom, scrollLeft: element.scrollLeft,
      viewportWidth: element.clientWidth, scrollWidth: element.scrollWidth,
      following: followPlayback.current,
    });
    followPlayback.current = result.following;
    if (Math.abs(element.scrollLeft - result.scrollLeft) > 0.5) {
      element.scrollLeft = result.scrollLeft;
      updateTimelineViewport(element);
    }
  }, [playing, playhead, zoom]);
  const previewAtPlayhead = (mode: TimelinePreviewMode) => {
    if (busy || maxEnd <= 0.04) return;
    // One second of context before the playhead makes entrances and edits visible while keeping
    // the temporary render short. Move the playhead and run it again to inspect another moment.
    const startSeconds = Math.min(Math.max(0, playhead - 1), Math.max(0, maxEnd - 0.04));
    const durationSeconds = Math.min(6, Math.max(0.04, maxEnd - startSeconds));
    setRenderedPreview(undefined);
    void exportTimeline(resolveExportSettings(project), sequence, {
      mode, startSeconds, durationSeconds, onComplete: setRenderedPreview,
    });
  };
  const scrubAt = (element: HTMLElement, clientX: number) => {
    const rect = element.getBoundingClientRect();
    return Math.max(0, Math.min(duration, (clientX - rect.left) / zoom));
  };
  const commitScrubPosition = (at: number) => {
    const active = primaryClips.find((clip) => {
      const start = primaryLayout.start.get(clip.id) ?? seconds(clip.start);
      return at >= start && at < start + timelineClipDuration(clip);
    });
    // setSelectedClip intentionally jumps the ordinary edit playhead to the clip head. During a
    // scrub, select only when crossing an edit and then write the dragged position last.
    if (active && active.id !== scrubSelectedClip.current) {
      scrubSelectedClip.current = active.id;
      setSelectedClip(active.id);
    }
    setPlayhead(at);
  };
  const beginScrub = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    setPlaying(false);
    setScrubbing(true);
    scrubSelectedClip.current = selectedClipId;
    const at = scrubAt(event.currentTarget, event.clientX);
    scrub.current = { pointerId: event.pointerId, lastAt: at, lastAudibleAt: at, lastAudioMs: performance.now() };
    commitScrubPosition(at);
    const audible = scrubPlayablesAt(project, sequence, at).some((item) =>
      canDecodeWholeAssetForScrub(seconds(item.asset.duration), item.asset.kind === "audio" ? "audio" : "video"));
    setStatus(audible
      ? "Scrubbing timeline · drag left for reverse audio, right for forward audio"
      : "Scrubbing timeline picture · long-form audio resumes on Play");
  };
  const moveScrub = (event: ReactPointerEvent<HTMLDivElement>) => {
    const state = scrub.current;
    if (!state || state.pointerId !== event.pointerId) return;
    const at = scrubAt(event.currentTarget, event.clientX);
    commitScrubPosition(at);
    const now = performance.now();
    if (now - state.lastAudioMs >= 42 && Math.abs(at - state.lastAudibleAt) >= 0.003) {
      void playTimelineScrubAudio(project, sequence, state.lastAudibleAt, at);
      state.lastAudibleAt = at;
      state.lastAudioMs = now;
    }
    state.lastAt = at;
  };
  const endScrub = (event: ReactPointerEvent<HTMLDivElement>) => {
    const state = scrub.current;
    if (!state || state.pointerId !== event.pointerId) return;
    const at = scrubAt(event.currentTarget, event.clientX);
    commitScrubPosition(at);
    if (Math.abs(at - state.lastAudibleAt) >= 0.003) void playTimelineScrubAudio(project, sequence, state.lastAudibleAt, at);
    scrub.current = undefined;
    setScrubbing(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    setStatus(`Playhead parked at ${formatTimecode(at, sequence.frameRate.numerator / sequence.frameRate.denominator)}`);
  };
  const cancelScrub = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (scrub.current?.pointerId !== event.pointerId) return;
    scrub.current = undefined;
    scrubAudioRequest += 1;
    stopScrubSources();
    setScrubbing(false);
  };
  const saveTrimmedClipToMedia = async (clip: TimelineClip) => {
    const asset = project.assets.find((item) => item.id === clip.assetId);
    const track = sequence.tracks.find((item) => item.id === clip.trackId);
    const kind = track?.kind === "audio" ? "audio" : track?.kind === "video" ? "video" : undefined;
    if (!asset || !kind || (kind === "video" && asset.kind === "image")) {
      setStatus("Only timeline audio and video clips can be saved as media");
      return;
    }
    const extension = kind === "audio" ? "wav" : "mp4";
    const sourceName = filename(asset.path);
    const stem = sourceName.replace(/\.[^.]+$/, "").replace(/[<>:\"/\\|?*]/g, "_");
    const slash = Math.max(asset.path.lastIndexOf("\\"), asset.path.lastIndexOf("/"));
    const parent = slash >= 0 ? asset.path.slice(0, slash + 1) : "";
    setBusy(true);
    setStatus("Choose where to save the trimmed " + kind + " asset…");
    let outputPath: string | null;
    try {
      outputPath = await save({
        defaultPath: `${parent}${stem}-trimmed.${extension}`,
        filters: [{ name: kind === "audio" ? "Wave audio" : "MPEG-4 Video", extensions: [extension] }],
      });
    } catch (error) {
      setStatus("Could not open the trimmed-asset picker: " + errorHeadline(error));
      setBusy(false);
      return;
    }
    if (!outputPath) {
      setStatus("Save trimmed clip canceled");
      setBusy(false);
      return;
    }
    setStatus(`Saving ${clipDuration(clip.sourceIn, clip.sourceOut).toFixed(1)}s source range as project media…`);
    try {
      const path = await invoke<string>("materialize_trimmed_clip", { request: {
        path: asset.path, outputPath,
        sourceInSeconds: seconds(clip.sourceIn), sourceOutSeconds: seconds(clip.sourceOut),
        playbackRate: clipPlaybackRate(clip), kind,
      } });
      const probe = await probeMedia(path);
      const saved = generatedAsset(path, probe, kind);
      addAssets([saved]);
      setStatus(`Saved ${saved.name} to project media · ${seconds(saved.duration).toFixed(1)}s · ready for Ref2VA`);
    } catch (error) {
      setStatus("Could not save the trimmed clip: " + errorHeadline(error));
    } finally {
      setBusy(false);
    }
  };
  const openClipMenu = (event: ReactMouseEvent, clip: TimelineClip) => {
    event.preventDefault();
    event.stopPropagation();
    const splitAt = playhead;
    const clipTrack = sequence.tracks.find((track) => track.id === clip.trackId);
    const clipAsset = project.assets.find((asset) => asset.id === clip.assetId);
    const locked = Boolean(clipTrack?.locked);
    setSelectedClip(clip.id);
    setMenu({ x: event.clientX, y: event.clientY, items: [
      { label: "Play from clip", action: () => { setSelectedClip(clip.id); setTimeout(() => setPlaying(true), 0); }, shortcut: "Space" },
      { label: "Open AI clip editor…", action: () => setClipEditorOpen(true), disabled: clip.trackId !== "v1", shortcut: "Ctrl+K" },
      { label: "Save trimmed clip to project media…", action: () => { void saveTrimmedClipToMedia(clip); },
        disabled: busy || !clipAsset || !["audio", "video"].includes(clipTrack?.kind ?? "") || (clipTrack?.kind === "video" && clipAsset.kind === "image") },
      { label: "Split at playhead", action: () => splitClip(clip.id, splitAt), disabled: locked, shortcut: "S" },
      { label: "Duplicate", action: () => duplicateClip(clip.id), disabled: locked, shortcut: "Ctrl+D" },
      { label: clip.enabled ? "Disable clip" : "Enable clip", action: () => updateSelectedClip({ enabled: !clip.enabled }), disabled: locked },
      { label: "Clear transitions", action: () => updateSelectedClip({ transitionIn: "none", transitionOut: "none" }), disabled: locked },
      { label: "Ripple delete", action: () => rippleDeleteClip(clip.id), disabled: locked, danger: true, shortcut: "Shift+Del" },
      { label: "Delete", action: () => deleteClip(clip.id), disabled: locked, danger: true, shortcut: "Del" },
    ] });
  };
  const onDrop = (event: ReactDragEvent<HTMLDivElement>, trackId: string) => {
    event.preventDefault();
    event.currentTarget.classList.remove("drag-over");
    draggedAssetKind = undefined;
    const assetId = event.dataTransfer.getData("application/x-aiv-asset") || event.dataTransfer.getData("text/plain");
    if (!assetId) return;
    const rect = event.currentTarget.getBoundingClientRect();
    addAssetToTimelineAt(assetId, (event.clientX - rect.left) / zoom, trackId);
  };
  return <><section className="timeline-shell">
    <div className="timeline-toolbar">
      <div className="tool-group">
        {TOOLS.map((item) => <Button key={item.id} active={tool === item.id} title={item.title} onClick={() => setTool(item.id)}>{item.icon}</Button>)}
        {groupSelection.length > 0 && <button className="group-badge" onClick={clearGroupSelection} title="Clear the track selection">{groupSelection.length} selected ✕</button>}
        <span className="timeline-tool-divider" />
        <button className="timeline-preview-button quick" disabled={busy || maxEnd <= 0.04} onClick={() => previewAtPlayhead("quick")}
          title="Render a fast six-second 960×540 proxy around the playhead, including the timeline mix"><Gauge size={13} /> Quick preview</button>
        <button className="timeline-preview-button" disabled={busy || maxEnd <= 0.04} onClick={() => previewAtPlayhead("full")}
          title="Render six seconds around the playhead with the current Master Export settings"><Film size={13} /> Full preview</button>
      </div>
      <div className="sequence-name"><span className="status-dot" />{sequence.name}</div>
      <div className="zoom-controls"><ZoomOut size={14} /><input type="range" min="28" max="140" value={zoom} onChange={(event) => setZoom(Number(event.target.value))} /><ZoomIn size={14} /></div>
    </div>
    <div className="timeline-grid" ref={timelineGridRef}
      onWheel={() => { manualScrollUntil.current = performance.now() + 1400; followPlayback.current = false; }}
      onScroll={(event) => updateTimelineViewport(event.currentTarget)}>
      <div className="track-controls ruler-corner"><span>TRACKS</span></div>
      <div className={"ruler" + (scrubbing ? " scrubbing" : "")} style={{ minWidth: duration * zoom }}
        aria-label="Timeline scrub ruler" title="Drag left/right to scrub picture and audio forward or backward"
        onPointerDown={beginScrub} onPointerMove={moveScrub} onPointerUp={endScrub} onPointerCancel={cancelScrub}>
        {rulerTicks.map((tick) => <span key={tick.time} style={{ left: tick.time * zoom }} className={tick.major ? "major" : ""}>
          {tick.major ? formatTimecode(tick.time, sequence.frameRate.numerator / sequence.frameRate.denominator).split(":").slice(0, -1).join(":") : ""}</span>)}
      </div>
      {sequence.tracks.map((track) => <div className="track-row-pair" key={track.id}>
        <div className="track-controls"><div><strong>{track.name.split(" · ")[0]}</strong><span>{track.name.split(" · ")[1]}</span></div>
          <button title={track.hidden ? "Show track" : "Hide track"} className={track.hidden ? "active" : ""} onClick={() => updateTrack(track.id, { hidden: !track.hidden })}>{track.hidden ? <EyeOff size={13} /> : <Eye size={13} />}</button>
          <button title={track.muted ? "Unmute track" : "Mute track"} className={track.muted ? "active" : ""} onClick={() => updateTrack(track.id, { muted: !track.muted })}>M</button>
          <button title={track.locked ? "Unlock track" : "Lock track"} className={track.locked ? "active" : ""} onClick={() => updateTrack(track.id, { locked: !track.locked })}>{track.locked ? <Lock size={11} /> : <LockOpen size={11} />}</button>
        </div>
        <div data-track-id={track.id} className={"track-lane kind-" + track.kind + (tool === "blade" ? " blade-cursor" : tool === "track" ? " track-cursor" : "")}
          style={{ minWidth: duration * zoom, backgroundSize: zoom + "px 100%" }}
          onDragOver={(event) => {
            // Fall back to the selected asset only when the drag did not originate in the media
            // panel (e.g. a re-entrant drag), so an unknown kind still lands somewhere sensible.
            const kind = draggedAssetKind ?? selectedAsset?.kind;
            const compatible = kind === "audio" ? track.kind === "audio" : track.kind === "video";
            if (compatible && !track.locked) { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; event.currentTarget.classList.add("drag-over"); }
          }}
          onDragLeave={(event) => event.currentTarget.classList.remove("drag-over")}
          onDrop={(event) => onDrop(event, track.id)}
          onContextMenu={(event) => {
            event.preventDefault();
            const rect = event.currentTarget.getBoundingClientRect();
            const at = (event.clientX - rect.left) / zoom;
            setMenu({ x: event.clientX, y: event.clientY, items: [
              { label: "Move playhead here", action: () => setPlayhead(at) },
              { label: "Select from here on this track", action: () => selectTrackFrom(track.id, at) },
              { label: "Place selected media here", action: () => selectedAssetId && addAssetToTimelineAt(selectedAssetId, at, track.id), disabled: !selectedAssetId || (selectedAsset && isAudioAsset(selectedAsset) ? track.kind !== "audio" : track.kind !== "video") || track.locked },
            ] });
          }}
          onClick={(event) => {
            if (tool === "track") {
              const rect = event.currentTarget.getBoundingClientRect();
              selectTrackFrom(track.id, (event.clientX - rect.left) / zoom);
              return;
            }
            setSelectedClip();
            clearGroupSelection();
          }}>
          {track.clips.map((clip) => <Clip clip={clip} zoom={zoom} locked={track.locked} trackKind={track.kind} key={clip.id} openMenu={openClipMenu} />)}
        </div>
      </div>)}
      <div className="playhead" style={{ left: "calc(164px + " + playhead * zoom + "px)" }}><span /><i /></div>
    </div>
    <ContextMenu menu={menu} close={() => setMenu(null)} />
  </section>
  {renderedPreview && <div className="modal-backdrop" onPointerDown={() => setRenderedPreview(undefined)}>
    <section className="timeline-preview-dialog" role="dialog" aria-modal="true" aria-label="Rendered timeline preview" onPointerDown={(event) => event.stopPropagation()}>
      <div className="dialog-heading"><div><strong>{renderedPreview.mode === "quick" ? "Quick rendered preview" : "Export-quality rendered preview"}</strong>
        <small>{formatTimecode(renderedPreview.startSeconds, sequence.frameRate.numerator / sequence.frameRate.denominator)} · {renderedPreview.durationSeconds.toFixed(1)} seconds · picture and final timeline audio mix</small></div>
        <button onClick={() => setRenderedPreview(undefined)} aria-label="Close"><X size={16} /></button></div>
      <video key={renderedPreview.path} src={convertFileSrc(renderedPreview.path)} controls autoPlay />
      <div className="timeline-preview-note"><span>{renderedPreview.mode === "quick"
        ? "Fast proxy: 960×540 maximum, up to 30 fps, H.264 NVENC p1, no upscale."
        : "Exact current Master Export resolution, frame rate, codecs, quality, preset, audio, and upscale chain."}</span>
        <button onClick={() => previewAtPlayhead(renderedPreview.mode)} disabled={busy}>Render again at playhead</button></div>
    </section>
  </div>}
  </>;
}

const FRAME_RATE_OPTIONS = [
  { value: "24000/1001", label: "23.976 fps" }, { value: "24/1", label: "24 fps" },
  { value: "25/1", label: "25 fps" }, { value: "30000/1001", label: "29.97 fps" },
  { value: "30/1", label: "30 fps" }, { value: "50/1", label: "50 fps" },
  { value: "60000/1001", label: "59.94 fps" }, { value: "60/1", label: "60 fps" },
];

function SequenceSettingsDialog() {
  const { project, sequenceSettingsOpen, setSequenceSettingsOpen, updateSequenceSettings } = useEditor();
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId)!;
  const [name, setName] = useState(sequence.name);
  const [width, setWidth] = useState(sequence.width);
  const [height, setHeight] = useState(sequence.height);
  const [rate, setRate] = useState(`${sequence.frameRate.numerator}/${sequence.frameRate.denominator}`);
  const [sampleRate, setSampleRate] = useState(sequence.sampleRate);
  const [conform, setConform] = useState<ClipConform>(sequence.conform ?? DEFAULT_CONFORM);
  useEffect(() => {
    if (!sequenceSettingsOpen) return;
    setName(sequence.name);
    setWidth(sequence.width);
    setHeight(sequence.height);
    setRate(`${sequence.frameRate.numerator}/${sequence.frameRate.denominator}`);
    setSampleRate(sequence.sampleRate);
    setConform(sequence.conform ?? DEFAULT_CONFORM);
  }, [sequenceSettingsOpen, sequence]);
  if (!sequenceSettingsOpen) return null;
  const saveSettings = () => {
    const [numerator, denominator] = rate.split("/").map(Number);
    updateSequenceSettings({ name, width, height, frameRate: { numerator, denominator }, sampleRate, conform });
  };
  return <div className="modal-backdrop" onPointerDown={() => setSequenceSettingsOpen(false)}>
    <section className="settings-dialog" role="dialog" aria-modal="true" aria-label="Sequence settings" onPointerDown={(event) => event.stopPropagation()}>
      <div className="dialog-heading"><div><strong>Sequence settings</strong><small>Controls preview and master export.</small></div><button onClick={() => setSequenceSettingsOpen(false)} aria-label="Close"><X size={16} /></button></div>
      <label><span>Name</span><input value={name} onChange={(event) => setName(event.target.value)} /></label>
      <div className="dialog-pair"><label><span>Width</span><input type="number" min={64} max={8192} step={2} value={width} onChange={(event) => setWidth(Number(event.target.value))} /></label><label><span>Height</span><input type="number" min={64} max={8192} step={2} value={height} onChange={(event) => setHeight(Number(event.target.value))} /></label></div>
      <label><span>Frame rate</span><select value={rate} onChange={(event) => setRate(event.target.value)}>{FRAME_RATE_OPTIONS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
      <label><span>Aspect preset</span><select value={`${width}x${height}`} onChange={(event) => {
        const [nextWidth, nextHeight] = event.target.value.split("x").map(Number);
        if (nextWidth) { setWidth(nextWidth); setHeight(nextHeight); }
      }}>
        <option value={`${width}x${height}`}>{`Current · ${width} × ${height}`}</option>
        <option value="1920x1080">16:9 · 1920 × 1080</option>
        <option value="3840x2160">16:9 · 3840 × 2160</option>
        <option value="1080x1920">9:16 vertical · 1080 × 1920</option>
        <option value="1080x1080">1:1 square · 1080 × 1080</option>
        <option value="1440x1080">4:3 · 1440 × 1080</option>
        <option value="1536x1024">3:2 · 1536 × 1024 (LTX native)</option>
        <option value="2048x858">2.39:1 scope · 2048 × 858</option>
      </select></label>
      <label><span>Audio sample rate</span><select value={sampleRate} onChange={(event) => setSampleRate(Number(event.target.value))}><option value={44100}>44.1 kHz</option><option value={48000}>48 kHz</option><option value={96000}>96 kHz</option></select></label>
      <div className="dialog-conform">
        <strong>Default fit for clips</strong>
        <small>Applies to every clip that has not been given its own setting in the inspector.</small>
        <ConformControls value={conform} onChange={setConform} />
      </div>
      <div className="dialog-actions"><button onClick={() => setSequenceSettingsOpen(false)}>Cancel</button><button className="primary" onClick={saveSettings}>Apply settings</button></div>
    </section>
  </div>;
}

export default function App() {
  const { project, projectEpoch, status, busy, setSequenceSettingsOpen, activeWorkspace, setActiveWorkspace, setPlaying } = useEditor();
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target?.matches("input, select, textarea")) return;
      const state = useEditor.getState();
      const key = event.key.toLowerCase();
      if (event.ctrlKey && key === "s") {
        event.preventDefault();
        void saveProject();
      } else if (event.ctrlKey && key === "o") {
        event.preventDefault();
        void openProject();
      } else if (state.activeWorkspace === "image") {
        return;
      } else if (state.clipEditorOpen && !(event.ctrlKey && (key === "z" || key === "y")) && event.key !== "Escape") {
        // The Clip Editor is modal: its canvas and fields own the keyboard. Timeline shortcuts must
        // not fire underneath it (Delete would remove the very clip being edited, S would split it).
        return;
      } else if (event.ctrlKey && key === "k") {
        event.preventDefault();
        if (state.selectedClipId) state.setClipEditorOpen(true);
      } else if (event.ctrlKey && key === "z") {
        event.preventDefault();
        if (event.shiftKey) state.redo(); else state.undo();
      } else if (event.ctrlKey && key === "y") {
        event.preventDefault();
        state.redo();
      } else if (event.ctrlKey && key === "d" && state.selectedClipId) {
        event.preventDefault();
        state.duplicateClip(state.selectedClipId);
      } else if (event.ctrlKey) {
        return;
      } else if (event.key === "Escape") {
        state.setClipEditorOpen(false);
        state.setSequenceSettingsOpen(false);
      } else if (event.key === "Delete" && state.selectedClipId) {
        if (event.shiftKey) state.rippleDeleteClip(state.selectedClipId); else state.deleteClip(state.selectedClipId);
      } else if (key === "s" && state.selectedClipId) {
        state.splitClip(state.selectedClipId, state.playhead);
      } else if (key === "v") {
        state.setTool("select");
      } else if (key === "b") {
        state.setTool("blade");
      } else if (key === "a") {
        state.setTool("track");
      } else if (key === "n") {
        state.toggleSnapping();
      } else if (event.code === "Space") {
        event.preventDefault();
        state.togglePlaying();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  return <div className={`app ${activeWorkspace}-workspace`}>
    <TopBar />
    <nav className="workspace-switcher" aria-label="Workspace">
      <button className={activeWorkspace === "image" ? "selected" : ""} onClick={() => { setPlaying(false); setActiveWorkspace("image"); }}><Sparkles size={15} /> Image Studio</button>
      <button className={activeWorkspace === "photo" ? "selected" : ""} onClick={() => { setPlaying(false); setActiveWorkspace("photo"); }}><SlidersHorizontal size={15} /> Photo Edit</button>
      <button className={activeWorkspace === "video" ? "selected" : ""} onClick={() => setActiveWorkspace("video")}><Film size={15} /> Video Editor</button>
      <button className={activeWorkspace === "ref2v" ? "selected" : ""} onClick={() => { setPlaying(false); setActiveWorkspace("ref2v"); }}><Layers3 size={15} /> Ref2V</button>
    </nav>
    <ImageStudio />
    <PhotoEdit />
    <Ref2V key={`${project.id}:${projectEpoch}`} />
    <div className="workspace"><MediaBin /><Monitor /><Inspector /></div>
    <Timeline />
    <footer className="statusbar">
      <span className={busy ? "busy" : ""}><span className="status-dot" /> {status}</span>
      <span>GPU target · 16 GB VRAM</span>
      <span>Local processing</span>
      <button onClick={() => setSequenceSettingsOpen(true)}><Settings2 size={13} /> Sequence settings</button>
    </footer>
    <div className="video-modal-host"><ClipEditor /></div>
  </div>;
}
