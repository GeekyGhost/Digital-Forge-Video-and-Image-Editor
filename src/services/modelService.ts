import { invoke } from "@tauri-apps/api/core";
import { fromSeconds } from "../domain/time";
import type { MediaAsset } from "../domain/types";
import type { Gradient } from "../domain/photoGradient";
import type {
  MaskEffectId, MaskEffectMatteStyle, MaskEffectPlacement, MaskEffectTransition, MaskMatteCache,
  MaskSource, MaskVfxBlendMode, MaskVfxKeyMode, MaskVfxKeyframe, MaskVfxTransitionOrigin,
} from "../domain/maskVfx";
export type {
  MaskEffectMatteStyle, MaskEffectPlacement, MaskEffectTransition, MaskMatteCache,
  MaskSource, MaskVfxBlendMode, MaskVfxKeyMode, MaskVfxKeyframe,
} from "../domain/maskVfx";

export type MediaProbe = {
  path: string; durationSeconds: number; width?: number; height?: number;
  frameRate?: string; videoCodec?: string; audioCodec?: string; colorSpace?: string;
};

export type AiRuntimeStatus = {
  root: string; pythonReady: boolean; ltxSourceReady: boolean; checkpointReady: boolean;
  gemmaReady: boolean; singlePassReady: boolean; retakeReady: boolean;
  foleyLoraReady: boolean; sam2Ready: boolean; sam2State: string;
  mossSfxState: string; mossSfxReady: boolean;
  kiwiEditReady: boolean; kiwiEditState: string; whisperReady: boolean;
  minimaxH3State: string; minimaxH3Ready: boolean;
  minimaxH3DraftReady: boolean; minimaxH3TurboReady: boolean;
  minimaxH3RefState: string; minimaxH3RefReady: boolean;
  imageStudioState: string; zitImageReady: boolean; qwenImageEditReady: boolean; birefnetReady: boolean;
  zimageBaseReady: boolean; zimageTrainerReady: boolean; zimageCaptionerReady: boolean;
  /** The Qwen edit prompt rewriter (Rust `prompt_rewrite`). Optional: older builds lack it. */
  qwenEditRewriterReady?: boolean;
  photoDepthReady: boolean; photoInpaintReady: boolean; photoSelectReady: boolean;
  /** Real-ESRGAN, the AI stage of the upscale chain. `realEsrganModels` lists what is on disk. */
  realEsrganState: string; realEsrganReady: boolean; realEsrganModels: string[];
  /** NVIDIA VideoSuperRes, installed as a pip wheel rather than a model pack. */
  nvvfxState: string; nvvfxReady: boolean;
  profile: string;
};

export type ImageExportOptions = {
  sourcePath: string;
  outputPath: string;
  format?: "png" | "jpeg" | "webp" | "tiff";
  quality?: number;
  lossless?: boolean;
  /** `#rrggbb`. Required to flatten transparency into a format that cannot carry it. */
  matte?: string;
  width?: number;
  height?: number;
};

export function exportImage(options: ImageExportOptions) {
  return invoke<ImageModelResult>("export_image", { request: options });
}

/**
 * Read a project image as base64. Only used when the asset protocol will not let the canvas read
 * its own pixels back -- see `read_image_base64` on the Rust side for why that can happen.
 */
/** Depth for one picture. Cached on the source path, so a second call on the same file is free. */
export function photoDepthMap(options: { projectId: string; imagePath: string }) {
  return invoke<ImageModelResult>("photo_depth_map", { request: options });
}

/**
 * Fill a masked region with LaMa. `maskPngBase64` is the rasterised mask with the `data:` prefix
 * stripped; `cacheKey` is everything about the document that shaped it, so an unchanged mask on an
 * unchanged picture never runs the model twice.
 */
export function photoInpaint(options: {
  projectId: string; imagePath: string; maskPngBase64: string; cacheKey: string;
}) {
  return invoke<ImageModelResult>("photo_inpaint", { request: options });
}

/** SAM 2 click-to-select on a still. Points are normalised to the picture. */
export function photoSelect(options: {
  projectId: string; imagePath: string;
  points: Array<{ x: number; y: number; include: boolean }>;
}) {
  return invoke<ImageModelResult>("photo_select", { request: options });
}

export function readImageBase64(path: string) {
  return invoke<string>("read_image_base64", { request: { path } });
}

/**
 * Hand a canvas-rendered PNG back to the app for storage. `pngBase64` is the payload of a
 * `toDataURL("image/png")` with the `data:` prefix already stripped.
 */
export function writeEditedImage(options: {
  projectId: string; pngBase64: string; sourcePath: string; width: number; height: number;
}) {
  return invoke<ImageModelResult>("write_edited_image", { request: options });
}

export function upscaleImage(options: {
  sourcePath: string; projectId: string;
  /** `esrgan` needs modelFile; `nvvfx` needs quality plus a target size. */
  backend: "esrgan" | "nvvfx";
  modelFile?: string; quality?: string;
  tile?: number; targetWidth?: number; targetHeight?: number;
}) {
  return invoke<ImageModelResult>("upscale_image", { request: options });
}

export type AiBridgeResult = { outputPath: string; log: string };

export type LtxRetakeResult = AiBridgeResult & {
  previewPath: string;
  effectiveStartTime: number;
  effectiveEndTime: number;
  startFrame: number;
  endFrame: number;
  fpsNumerator: number;
  fpsDenominator: number;
};

export type SfxGenerationResult = {
  outputPath: string;
  sampleRate: number;
  channels: number;
  durationSeconds: number;
  seed: number;
  log: string;
};

export type MossSfxRequest = {
  projectId: string;
  prompt: string;
  negativePrompt?: string;
  durationSeconds: number;
  seed: number;
  quality: "draft" | "final";
};

export type LtxFoleyRequest = {
  projectId: string;
  videoPath: string;
  sourceInSeconds: number;
  sourceOutSeconds: number;
  playbackRate: number;
  prompt: string;
  negativePrompt?: string;
  seed: number;
  density: "minimal" | "normal" | "detailed";
  offload: string;
  attentionBackend: "native" | "sage";
};

export type MaskEffectRequest = {
  videoPath: string;
  outputPath: string;
  startTime: number;
  endTime: number;
  model: MaskSource;
  points: Array<{ x: number; y: number; include: boolean }>;
  effect: MaskEffectId;
  color: string;
  secondaryColor: string;
  intensity: number;
  feather: number;
  expand: number;
  temporalSmooth: number;
  cleanup: number;
  effectSpeed: number;
  particleAmount: number;
  transitionIn: MaskEffectTransition;
  transitionOut: MaskEffectTransition;
  transitionInDuration: number;
  transitionOutDuration: number;
  invert: boolean;
  audioReactive: "off" | "amplitude" | "bass" | "transients";
  reactiveAmount: number;
  reactiveSmoothing: number;
  placement: MaskEffectPlacement;
  matteStyle: MaskEffectMatteStyle;
  effectClipPath: string;
  effectClipColor: "original" | "gradient";
  effectOpacity: number;
  effectX: number;
  effectY: number;
  effectScale: number;
  effectRotation: number;
  loop: boolean;
  pulseAmount: number;
  pulseRate: number;
  gradientEnabled: boolean;
  gradient: MaskEffectGradient;
  matteCachePath: string;
  occlusionFeather: number;
  occlusionExpand: number;
  transitionOrigin: MaskVfxTransitionOrigin;
  blendMode: MaskVfxBlendMode;
  keyMode: MaskVfxKeyMode;
  keyThreshold: number;
  keySoftness: number;
  keyColor: string;
  rearAmount: number;
  subjectAmount: number;
  frontAmount: number;
  motionBlur: number;
  lightWrap: number;
  colorSpill: number;
  grain: number;
  displacement: number;
  keyframes: MaskVfxKeyframe[];
  /** Zero for final quality; otherwise the worker renders an even proxy width. */
  previewWidth?: number;
};

export type MaskEffectGradient = Gradient;

export type MaskTrackRequest = {
  videoPath: string;
  startTime: number;
  endTime: number;
  model: MaskSource;
  points: Array<{ x: number; y: number; include: boolean }>;
};

export type MaskTrackResult = MaskMatteCache & { reused: boolean; log: string };

/** H3 returns picture and sound separately so they can land on V1 and A1 as a linked pair. */
export type MiniMaxH3Result = { outputPath: string; audioPath: string; log: string };
export type ImageModelResult = {
  outputPath: string; maskPath?: string; seed?: number; width?: number; height?: number;
  loraActive?: boolean; loraPath?: string; log: string;
};

export type CaptionedLoraImage = {
  imagePath: string; caption: string; tags: string[]; qualityFlags: string[];
  confidence: number; width: number; height: number; duplicateOf?: string; rawCaption: string;
};

export type LoraTrainingJob = {
  id: string; state: "running" | "pausing" | "paused" | "canceling" | "canceled" | "complete" | "failed";
  progress: number; step: number; total: number; message: string;
  outputPath?: string; checkpointPath?: string;
  checkpoints: Array<{ step: number; path: string; weights: string }>;
  error?: string; processId?: number; runDirectory: string; configPath: string;
};

export type ZImageLoraInfo = {
  folder: string; weights: string; name: string; triggerPhrase?: string; trigger?: string;
  className?: string; baseModel?: string; profile?: string; rank?: number; alpha?: number;
  step?: number; kind?: string;
};

export function generateZImages(request: {
  projectId: string; prompt: string; negativePrompt: string; model: "turbo" | "base";
  width: number; height: number; steps: number; guidanceScale: number; seed: number;
  candidates: 1 | 4; loraPath?: string; loraStrength: number;
}) {
  return invoke<{ images: ImageModelResult[] }>("generate_zimage_images", { request });
}

export function inspectZImageLora(path: string) {
  return invoke<ZImageLoraInfo>("inspect_zimage_lora", { path });
}

export function listZImageLoras() {
  return invoke<ZImageLoraInfo[]>("list_zimage_loras");
}

export function captionZImageLoraImages(request: {
  imagePaths: string[]; profile: "subject" | "character" | "style" | "product" | "concept";
  trigger: string; className: string; backend: "qwen" | "gemma";
}) {
  return invoke<{ items: CaptionedLoraImage[]; log: string }>("caption_zimage_lora_images", { request });
}

export function startZImageLoraTraining(request: {
  projectId: string; name: string; profile: "subject" | "character" | "style" | "product" | "concept";
  trigger: string; className: string; images: CaptionedLoraImage[];
  resolution: 512 | 768 | 1024; steps: number; rank: number; alpha: number;
  learningRate: number; seed: number; checkpointEvery: number; checkpointLimit: number;
  initialLoraPath?: string;
  /** "turbo" trains on Z-Image-Turbo with the frozen training adapter for 8-9 step generation. */
  trainingModel: "base" | "turbo";
}) {
  return invoke<LoraTrainingJob>("start_zimage_lora_training", { request });
}

export function zImageLoraTrainingStatus(id: string) {
  return invoke<LoraTrainingJob>("zimage_lora_training_status", { id });
}

export function pauseZImageLoraTraining(id: string) {
  return invoke<LoraTrainingJob>("pause_zimage_lora_training", { id });
}

export function resumeZImageLoraTraining(id: string, targetSteps: number) {
  return invoke<LoraTrainingJob>("resume_zimage_lora_training", { id, targetSteps });
}

export function cancelZImageLoraTraining(id: string) {
  return invoke<LoraTrainingJob>("cancel_zimage_lora_training", { id });
}

export const filename = (path: string) => path.split(/[\\/]/).pop() ?? path;

export function parseFrameRate(rate?: string) {
  if (!rate) return { numerator: 24, denominator: 1 };
  const [left, right = "1"] = rate.split("/");
  const numerator = Number(left);
  const denominator = Number(right);
  return {
    numerator: Number.isFinite(numerator) ? numerator : 24,
    denominator: Number.isFinite(denominator) && denominator !== 0 ? denominator : 1,
  };
}

export async function probeMedia(path: string): Promise<MediaProbe> {
  return invoke<MediaProbe>("probe_media", { path });
}

export function generatedAsset(path: string, probe: MediaProbe, kind: MediaAsset["kind"] = "generated"): MediaAsset {
  return {
    id: crypto.randomUUID(), name: filename(path), path, kind,
    duration: fromSeconds(Math.max(probe.durationSeconds, 0.04)), width: probe.width, height: probe.height,
    frameRate: parseFrameRate(probe.frameRate), colorSpace: probe.colorSpace, hasAudio: Boolean(probe.audioCodec),
  };
}

/**
 * Rewrite a Qwen-Image-Edit instruction with Qwen's own prompt-enhancer rules, on the local
 * Qwen2.5-VL-7B (Rust `prompt_rewrite`). `imagePaths` are Picture 1 (the edited image) and the
 * references, exactly as the edit will send them.
 */
export async function rewriteQwenEditPrompt(prompt: string, imagePaths: string[]): Promise<{ prompt: string; raw: string; log: string }> {
  return invoke("rewrite_qwen_edit_prompt", { request: { prompt, imagePaths } });
}

export async function runtimeStatus(): Promise<AiRuntimeStatus> {
  return invoke<AiRuntimeStatus>("ai_runtime_status");
}

export function generateMossSfx(request: MossSfxRequest) {
  return invoke<SfxGenerationResult>("generate_moss_sfx", { request });
}

export function generateLtxFoley(request: LtxFoleyRequest) {
  return invoke<SfxGenerationResult>("generate_ltx_foley", { request });
}

export function renderMaskEffect(request: MaskEffectRequest) {
  return invoke<AiBridgeResult>("render_mask_effect", { request });
}

export function cacheMaskTrack(request: MaskTrackRequest) {
  return invoke<MaskTrackResult>("cache_mask_track", { request });
}

export function previewMaskEffect(request: MaskEffectRequest) {
  return invoke<AiBridgeResult>("preview_mask_effect", { request });
}

export function clipWaveform(path: string, durationSeconds: number, sourceInSeconds = 0, buckets = 320) {
  return invoke<number[]>("clip_waveform", { request: { path, sourceInSeconds, durationSeconds, buckets } });
}
