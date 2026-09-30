import type { PhotoDocument } from "./photoDocument";
import type { H3DirectorWorkspace } from "./h3Director";
import type { ResampleAlgo, UpscaleType } from "./upscale";
import type { ClipVfxLayer } from "./maskVfx";
import type { ClipMatte } from "./clipMatte";

export type Rational = { numerator: number; denominator: number };

/** How an overlay clip is fitted into its box when the aspect ratios disagree. */
export type OverlayFit = "contain" | "cover" | "stretch";

/**
 * Placement of an overlay picture track over the programme, in normalised 0..1 coordinates of
 * the sequence frame. Normalised so a placement survives a change of sequence resolution.
 */
export type OverlayPlacement = {
  x: number;
  y: number;
  width: number;
  height: number;
  fit: OverlayFit;
  opacity: number;
  /** Rounded corners as a fraction of the smaller box edge. */
  cornerRadius: number;
  /** Border drawn around the overlay box; width is a fraction of the sequence width. */
  borderWidth: number;
  borderColor: string;
};

/**
 * How a clip's picture is fitted into the sequence frame when the aspect ratios disagree.
 *
 * Until now V1 was permanently `contain` -- `export.rs` hardcoded
 * `force_original_aspect_ratio=decrease` -- which is why a 768x512 clip in a 1080p sequence always
 * gained pillarbox bars nobody had asked for. `none` leaves the source at its own size and lets the
 * clip transform place it.
 */
export type ConformFit = "contain" | "cover" | "stretch" | "none";

/**
 * What fills the frame where a `contain` conform leaves the picture short.
 *
 * `blur` is the one worth reaching for: it fills the bars with a blurred, zoomed copy of the same
 * frame, so nothing is cropped away AND there are no bars. `color` uses `padColor`.
 */
export type ConformPad = "black" | "blur" | "color";

/** A normalised 0..1 rectangle of the SOURCE frame, so it survives a change of source resolution. */
export type SourceRect = { x: number; y: number; width: number; height: number };

export type ClipConform = {
  fit: ConformFit;
  pad: ConformPad;
  /** Used when `pad` is `color`. `#rrggbb`. */
  padColor?: string;
  /** Gaussian sigma for `blur` padding. */
  padBlur?: number;
  /**
   * Which part of the source to keep, and where it travels.
   *
   * `end` absent means a static crop. `end` present makes it a move (Ken Burns, push-in, drift):
   * ffmpeg's `crop` accepts `t` in its expressions, so interpolating start -> end is a string
   * change in the render path rather than a keyframe engine. Model it this way from the start and
   * animation stays cheap; bolt it on later and the render path has to be rewritten.
   */
  crop?: { start: SourceRect; end?: SourceRect };
};

export const FULL_SOURCE_RECT: SourceRect = { x: 0, y: 0, width: 1, height: 1 };

export const DEFAULT_CONFORM: ClipConform = { fit: "contain", pad: "black" };

/** NVENC entries need an NVIDIA card; the lib* entries are the CPU fallback. */
export type VideoEncoder = "h264_nvenc" | "hevc_nvenc" | "libx264" | "libx265";

/**
 * `quality` is constant-quality (NVENC `-cq`, x26x `-crf`) and is what you almost always want --
 * it spends bits where the picture needs them. `bitrate` pins a target for delivery specs that
 * demand one.
 */
export type QualityMode = "quality" | "bitrate";

/** Container. Governs which codecs are legal, so the UI filters the codec lists on it. */
export type ExportContainer = "mp4" | "mov" | "mkv";

export type AudioCodec = "aac" | "libopus" | "flac" | "pcm_s16le" | "none";

export type ExportSettings = {
  container: ExportContainer;
  encoder: VideoEncoder;
  qualityMode: QualityMode;
  /** CQ/CRF. Lower is better quality and a bigger file. Sane range 16..32. */
  quality: number;
  /** Target bitrate when `qualityMode` is `bitrate`, e.g. "20M". */
  bitrate: string;
  /** NVENC `p1`..`p7` (p1 fastest) or x26x `ultrafast`..`veryslow`. */
  preset: string;
  /** 0 means follow the sequence. Set both to export at a different size. */
  width: number;
  height: number;
  /** 0 means follow the sequence. */
  frameRate: number;
  audioCodec: AudioCodec;
  audioBitrate: string;
  audioSampleRate: number;
  audioChannels: number;
  /**
   * The upscale chain applied to the finished master. See `domain/upscale.ts` -- the panel readout
   * and the arguments the exporter receives are resolved by the same function, so what the panel
   * promises is what runs.
   */
  /** The toggle. Everything below it is inert while this is false. */
  upscaleEnabled: boolean;
  /** Which family runs the enlargement. Chosen before any of its own options are shown. */
  upscaleType: UpscaleType;
  upscaleFactor: number;
  /**
   * A model id from AI_UPSCALE_MODELS. Only meaningful for `upscaleType: "esrgan"`, which is the
   * one family with more than one model to choose between.
   */
  upscaleAiModel?: string;
  upscaleResample: ResampleAlgo;
  /** Tile edge handed to Real-ESRGAN. VRAM scales with the square of this. */
  upscaleTile: number;
  /** A VideoSuperRes.QualityLevel name. Only used when the chosen model runs on nvvfx. */
  upscaleQuality: string;
};

export const DEFAULT_EXPORT_SETTINGS: ExportSettings = {
  container: "mp4",
  encoder: "h264_nvenc",
  qualityMode: "quality",
  quality: 21,
  bitrate: "20M",
  preset: "p5",
  width: 0,
  height: 0,
  frameRate: 0,
  audioCodec: "aac",
  audioBitrate: "320k",
  audioSampleRate: 48000,
  audioChannels: 2,
  upscaleEnabled: false,
  upscaleType: "nvidia",
  upscaleFactor: 2,
  upscaleAiModel: undefined,
  upscaleResample: "lanczos",
  upscaleTile: 512,
  upscaleQuality: "HIGH",
};

/** Codecs each container will actually accept, so the UI cannot offer an illegal pairing. */
export const CONTAINER_SUPPORT: Record<ExportContainer, { video: VideoEncoder[]; audio: AudioCodec[] }> = {
  mp4: { video: ["h264_nvenc", "hevc_nvenc", "libx264", "libx265"], audio: ["aac", "none"] },
  mov: { video: ["h264_nvenc", "hevc_nvenc", "libx264", "libx265"], audio: ["aac", "pcm_s16le", "none"] },
  mkv: {
    video: ["h264_nvenc", "hevc_nvenc", "libx264", "libx265"],
    audio: ["aac", "libopus", "flac", "pcm_s16le", "none"],
  },
};

/**
 * NVENC and the x26x encoders do not share a preset vocabulary -- passing `veryslow` to NVENC or
 * `p5` to x264 is a hard failure, not a fallback. Keep them apart at the source.
 */
export const ENCODER_PRESETS: Record<VideoEncoder, Array<{ value: string; label: string }>> = {
  h264_nvenc: NVENC_PRESETS(),
  hevc_nvenc: NVENC_PRESETS(),
  libx264: X26X_PRESETS(),
  libx265: X26X_PRESETS(),
};

function NVENC_PRESETS() {
  return [
    { value: "p1", label: "p1 · fastest" },
    { value: "p3", label: "p3 · fast" },
    { value: "p5", label: "p5 · balanced" },
    { value: "p6", label: "p6 · slow" },
    { value: "p7", label: "p7 · slowest, best" },
  ];
}

function X26X_PRESETS() {
  return [
    { value: "veryfast", label: "veryfast" },
    { value: "faster", label: "faster" },
    { value: "medium", label: "medium" },
    { value: "slow", label: "slow" },
    { value: "veryslow", label: "veryslow · best" },
  ];
}

/** The preset to fall back to when the encoder changes family and the current one is illegal. */
export function defaultPresetFor(encoder: VideoEncoder): string {
  return encoder.includes("nvenc") ? "p5" : "medium";
}

export type MediaKind = "video" | "audio" | "image" | "generated";
export type TrackKind = "video" | "audio" | "caption" | "graphics";

export type MediaAsset = {
  id: string;
  name: string;
  path: string;
  kind: MediaKind;
  duration: Rational;
  width?: number;
  height?: number;
  frameRate?: Rational;
  proxyPath?: string;
  colorSpace?: string;
  hasAudio?: boolean;
  /** AI/file lineage is separate from physical media type so generated PNGs remain images. */
  origin?: {
    operation: "zit-generate" | "qwen-edit" | "birefnet-cutout" | "birefnet-mask" | "canvas-flatten" | "video-frame" | "esrgan-upscale" | "moss-sfx" | "ltx-foley" | "mask-effect" | "clip-vfx";
    model: string;
    prompt?: string;
    seed?: number;
    sourceAssetIds?: string[];
    parameters?: Record<string, unknown>;
    createdAt: string;
  };
};

export type SpeedPoint = { at: Rational; rate: number; easing: "linear" | "smooth" };

export type ClipEffects = {
  brightness: number;
  contrast: number;
  saturation: number;
  gamma: number;
  hue: number;
  blur: number;
  sharpen: number;
  vignette: number;
  noise: number;
};

export type BlendMode = "normal" | "screen" | "multiply" | "overlay" | "darken" | "lighten" | "difference" | "hardlight";

export type AudioClipSettings = {
  /** Stereo position, -1 left through 1 right. */
  pan: number;
  fadeInSeconds: number;
  fadeOutSeconds: number;
  /** Generated SFX can intentionally ring out beyond the picture selection. */
  preserveTail?: boolean;
};

export type GraphicTemplate = "title" | "lower-third" | "banner" | "bug" | "caption";

export type TextBackdrop = "none" | "box" | "outline" | "underline";
export type TextFontFamily = "display" | "sans" | "serif" | "mono";
export type TextMotionEasing = "linear" | "smooth" | "ease-out" | "back" | "bounce";
export type TextMotionIn =
  | "none" | "fade" | "typewriter" | "word-reveal"
  | "slide-up" | "slide-down" | "slide-left" | "slide-right"
  | "pop" | "bounce" | "zoom-blur" | "tracking" | "cinematic" | "magic" | "glitch";
export type TextMotionOut =
  | "none" | "fade" | "typewriter" | "word-reveal"
  | "slide-up" | "slide-down" | "slide-left" | "slide-right"
  | "shrink" | "zoom-blur" | "tracking" | "cinematic" | "magic" | "glitch";
export type TextMotionLoop = "none" | "pulse" | "float" | "flicker" | "breathe" | "shimmer" | "jitter";

export type GraphicOverlay = {
  template: GraphicTemplate;
  text: string;
  subtitle: string;
  textColor: string;
  backgroundColor: string;
  accentColor: string;
  fontSize: number;
  /** Optional production typography fields keep older project files valid. */
  backdrop?: TextBackdrop;
  backgroundOpacity?: number;
  backgroundPadding?: number;
  backgroundBorderWidth?: number;
  fontFamily?: TextFontFamily;
  fontWeight?: 400 | 700 | 900;
  italic?: boolean;
  strokeWidth?: number;
  strokeColor?: string;
  shadowOffset?: number;
  shadowColor?: string;
  lineSpacing?: number;
  /** Optional so projects created before animated graphics keep loading unchanged. */
  motionIn?: TextMotionIn;
  motionOut?: TextMotionOut;
  motionDuration?: number;
  motionLoop?: TextMotionLoop;
  motionEasing?: TextMotionEasing;
};

export type TimelineClip = {
  id: string;
  assetId: string;
  name: string;
  trackId: string;
  start: Rational;
  sourceIn: Rational;
  sourceOut: Rational;
  enabled: boolean;
  color: string;
  opacity: number;
  volumeDb: number;
  audio?: AudioClipSettings;
  positionX?: number;
  positionY?: number;
  scale?: number;
  rotation?: number;
  /** Overrides the sequence's `conform` for this clip only. Absent = inherit the sequence default. */
  conform?: ClipConform;
  blendMode?: BlendMode;
  /** Transition played as this clip enters. At a butt-joint it fuses with the previous clip's transitionOut. */
  transitionIn?: string;
  transitionDuration?: number;
  /** Transition played as this clip leaves. On the last clip of a track it renders as a fade to black. */
  transitionOut?: string;
  transitionOutDuration?: number;
  effects?: ClipEffects;
  graphic?: GraphicOverlay;
  /** Non-destructive subject-tracked VFX. Cached renders may be deleted and regenerated. */
  vfxLayers?: ClipVfxLayer[];
  /** Background removal: a matte (BiRefNet and/or SAM 2) applied as the clip's alpha. */
  matte?: ClipMatte;
  speed: SpeedPoint[];
};

export type Track = {
  id: string;
  name: string;
  kind: TrackKind;
  locked: boolean;
  muted: boolean;
  hidden: boolean;
  clips: TimelineClip[];
  /**
   * Present on picture tracks that composite over the primary track. V1 has none — it *is* the
   * programme. Overlay tracks carry their own box so picture-in-picture, corner bugs and
   * background-removed cutouts each get an independent placement.
   */
  overlay?: OverlayPlacement;
  /** Volume trim for a whole audio track, which stem separation makes worth having. */
  gainDb?: number;
};

export type Sequence = {
  id: string;
  name: string;
  width: number;
  height: number;
  frameRate: Rational;
  sampleRate: number;
  tracks: Track[];
  /**
   * Default conform for every clip in the sequence. Set it once here; per-clip `conform` is the
   * exception, not the rule. Optional so projects saved before this existed still load.
   */
  conform?: ClipConform;
  /** Master export settings. Optional for the same reason. */
  exportSettings?: ExportSettings;
};

export type Project = {
  schemaVersion: 1;
  id: string;
  name: string;
  createdAt: string;
  updatedAt: string;
  assets: MediaAsset[];
  sequences: Sequence[];
  activeSequenceId: string;
  /**
   * Photo Edit documents, keyed by the asset they describe.
   *
   * They live in the project rather than a store of their own so they are saved, loaded, copied
   * between machines and undone by machinery that already exists. Optional because every project
   * written before Photo Edit existed has none, and reading one back must not fail over it.
   */
  photoEdits?: Record<string, PhotoDocument>;
  /**
   * VFX Phase 4b: Clip Editor VFX compositions, keyed by the timeline clip they were built on. The
   * plate is that clip's source range as a video layer; rendering writes a new MP4 asset.
   */
  clipVfx?: Record<string, PhotoDocument>;
  /**
   * Persistent Ref2V/Director workspace, including reproducible take snapshots.
   * Optional so every project written before v0.8 opens unchanged.
   */
  h3Director?: H3DirectorWorkspace;
};

export type AiJobKind =
  | "ltx-keyframes"
  | "ltx-video-audio"
  | "ltx-retake"
  | "moss-sfx"
  | "ltx-foley"
  | "video-edit"
  | "transcribe"
  | "remove-background"
  | "interpolate"
  | "upscale"
  | "detect-stalls";

export type AiJob = {
  id: string;
  kind: AiJobKind;
  title: string;
  state: "queued" | "running" | "complete" | "failed" | "canceled";
  progress: number;
  message?: string;
};
