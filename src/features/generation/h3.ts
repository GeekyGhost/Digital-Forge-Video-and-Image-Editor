/**
 * MiniMax-H3 geometry, mirroring `ai-runtime/aiv_runtime/h3_geometry.py`.
 *
 * H3 does NOT share LTX's rules: LTX wants 64-pixel axes and 8n+1 frames, H3 wants 32-pixel axes
 * and 17n+5. Keeping the panel honest about that matters more here than it does for LTX, because
 * an H3 worker that rejects its geometry has already spent minutes loading 20 GB of weights.
 */

export const H3_CANVAS_MULTIPLE = 32;
export const H3_FRAMES_PER_CHUNK = 17;
export const H3_LATENTS_PER_CHUNK = 5;
export const H3_FPS = 24;
export const H3_MAX_PIXELS = 768 * 1344;

/**
 * The short edge H3-Base actually generates at. MiniMax's model card: "The shorter side is set to
 * 768 pixels by default"; ComfyUI's node default is 1344x768. This is the canvas to compare our
 * output against MiniMax's own published reference clips on.
 */
export const H3_TRAINED_SHORT_EDGE = 768;
/** 544 lands 16:9 on 960x544 -- below the trained canvas, but what a 16 GB card finishes in a sane time. */
export const H3_LOW_VRAM_SHORT_EDGE = 544;
/** The longest axis any in-budget canvas can have: 768 * 1344 with a 768 short edge. */
export const H3_MAX_LONG_EDGE = 1344;

/** 124 frames is 5.167 s. */
export const H3_MIN_FRAMES = 124;
/** 345 frames is 14.375 s. The next valid count, 362, is 15.083 s and the checkpoint rejects it. */
export const H3_MAX_FRAMES = 345;

export type H3Aspect = { label: string; ratio: [number, number] };

/** The six aspect ratios the checkpoint declares. */
export const H3_ASPECTS: H3Aspect[] = [
  { label: "21:9 cinemascope", ratio: [21, 9] },
  { label: "16:9 landscape", ratio: [16, 9] },
  { label: "4:3 classic", ratio: [4, 3] },
  { label: "1:1 square", ratio: [1, 1] },
  { label: "3:4 portrait", ratio: [3, 4] },
  { label: "9:16 vertical", ratio: [9, 16] },
];

/** Snap a frame count up to the next `17n + 5`, the only counts the video VAE can decode. */
export function alignH3Frames(numFrames: number): number {
  let frames = Math.max(1, Math.round(numFrames));
  while (frames % H3_FRAMES_PER_CHUNK !== H3_LATENTS_PER_CHUNK) frames += 1;
  return frames;
}

/** Align, then hold inside the checkpoint's 5-15 second window. */
export function clampH3Frames(numFrames: number): number {
  return Math.min(H3_MAX_FRAMES, Math.max(H3_MIN_FRAMES, alignH3Frames(numFrames)));
}

export function h3Seconds(numFrames: number): number {
  return numFrames / H3_FPS;
}

export type H3Dimensions = { width: number; height: number; value: string };

/**
 * Resolve an aspect ratio into an H3 canvas. Mirrors `resolve_canvas_size` in diffusers: the short
 * edge starts at `shortEdge`, the area is capped, and both axes then round to the 32-grid -- so the
 * final area can land slightly above the pre-rounding budget, which is the model's own behaviour.
 *
 * Only the ratio of the first two arguments matters, so a keyframe's pixel dimensions work
 * directly.
 */
export function fitH3Dimensions(
  aspectWidth = 16,
  aspectHeight = 9,
  shortEdge = H3_LOW_VRAM_SHORT_EDGE,
): H3Dimensions {
  const safeWidth = Math.max(1, aspectWidth);
  const safeHeight = Math.max(1, aspectHeight);
  const ratio = safeWidth / safeHeight;
  let width = ratio >= 1 ? shortEdge * ratio : shortEdge;
  let height = ratio >= 1 ? shortEdge : shortEdge / ratio;
  const area = width * height;
  if (area > H3_MAX_PIXELS) {
    const scale = Math.sqrt(H3_MAX_PIXELS / area);
    width *= scale;
    height *= scale;
  }
  const snap = (value: number) =>
    Math.max(H3_CANVAS_MULTIPLE, Math.round(value / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE);
  const fittedWidth = snap(width);
  const fittedHeight = snap(height);
  return { width: fittedWidth, height: fittedHeight, value: `${fittedWidth}x${fittedHeight}` };
}

export type H3ResolutionTier = "native" | "standard" | "balanced" | "compact" | "custom";

/**
 * Named canvases, honestly labelled. 768 is the model's own short edge, not a luxury tier; 544 and
 * below are VRAM concessions, and calling 544 "high fidelity" hid that for a while.
 */
export const H3_RESOLUTION_TIERS: { id: H3ResolutionTier; label: string; description: string; shortEdge: number }[] = [
  { id: "native", label: "Native 768p · MiniMax's own canvas", description: "1344x768 at 16:9 — the resolution H3-Base generates at. Slowest, and the only canvas comparable to MiniMax's published clips. Needs headroom.", shortEdge: 768 },
  { id: "standard", label: "544p · 16 GB profile", description: "960x544. Below the trained canvas, but the geometry a 16 GB card finishes in a sane time.", shortEdge: 544 },
  { id: "balanced", label: "480p · safer", description: "15–20% fewer tokens. Stable with two references.", shortEdge: 480 },
  { id: "compact", label: "416p · low VRAM", description: "35–45% fewer tokens. For three references or 10s of video on 16 GB.", shortEdge: 416 },
  { id: "custom", label: "Custom (exact width x height)", description: "Set width and height directly on the 32px grid.", shortEdge: 0 },
];

/** True when a canvas is inside the checkpoint's own area budget. */
export function h3CanvasFits(width: number, height: number): boolean {
  return width * height <= H3_MAX_PIXELS;
}

/** Snap arbitrary user width/height to the required 32-pixel grid. */
export function snapH3Dimension(value: number, min = 128, max = H3_MAX_LONG_EDGE): number {
  const clamped = Math.max(min, Math.min(max, value));
  return Math.max(H3_CANVAS_MULTIPLE, Math.round(clamped / H3_CANVAS_MULTIPLE) * H3_CANVAS_MULTIPLE);
}

/** Calculate spatial tokens per frame: (W / 16) * (H / 16) */
export function calculateH3Tokens(width: number, height: number): number {
  return Math.round((width / 16) * (height / 16));
}

export type H3RefImageDetail = "draft" | "balanced" | "match" | "high" | "max";

/**
 * How large a reference IMAGE is encoded, in short-edge pixels. `match` is ComfyUI's own default
 * for its shipped R2V workflow and scales the reference to the generation's own budget; `max` is
 * ComfyUI's other mode and Diffusers' default (`reference_image_short_edge = 2048`), documented as
 * "stronger identity fidelity".
 *
 * Reference tokens ride through every sampling step, so the small end is a real speed dial -- but
 * it is a draft setting, not a recommended one. Nothing upstream conditions references at 256.
 */
export const H3_REF_IMAGE_DETAILS: { id: H3RefImageDetail; label: string; shortEdge: number | "match" }[] = [
  { id: "match", label: "Match canvas (upstream default)", shortEdge: "match" },
  { id: "high", label: "1024 px · stronger identity", shortEdge: 1024 },
  { id: "max", label: "2048 px · max fidelity, much slower", shortEdge: 2048 },
  { id: "balanced", label: "320 px · fast, weaker identity", shortEdge: 320 },
  { id: "draft", label: "256 px · draft only, weak identity", shortEdge: 256 },
];

/** Resolve a reference-image tier against the canvas. `match` is the canvas's SHORT edge. */
export function h3RefImageShortEdge(detail: H3RefImageDetail, width: number, height: number): number {
  const tier = H3_REF_IMAGE_DETAILS.find((item) => item.id === detail);
  if (!tier || tier.shortEdge === "match") return Math.min(width, height);
  return tier.shortEdge;
}

export type H3RefVideoDetail = "safe" | "match" | "native";

/**
 * How large a reference VIDEO is encoded. This is a different question from reference images:
 * ComfyUI's node ignores `ref_image_size` for videos entirely and fits them under the same
 * 768x1344 budget as the generation. Ours has been pinned at 384 because Qwen3-VL's vision tower
 * attends over the unmerged patches and a 768p reference asked SDPA for 21+ GiB in one allocation.
 *
 * So the cap stays -- as a tier the user can raise, with the risk stated, rather than a constant.
 */
export const H3_REF_VIDEO_DETAILS: { id: H3RefVideoDetail; label: string; shortEdge: number }[] = [
  { id: "safe", label: "384 px · safe on 16 GB", shortEdge: 384 },
  { id: "match", label: "544 px · matches a 544p canvas", shortEdge: 544 },
  { id: "native", label: "768 px · upstream parity, may exhaust VRAM", shortEdge: 768 },
];

export function h3RefVideoShortEdge(detail: H3RefVideoDetail): number {
  return H3_REF_VIDEO_DETAILS.find((item) => item.id === detail)?.shortEdge ?? 384;
}

/**
 * The Ref2VA recipe, in one place.
 *
 * Two call sites build this request -- the dedicated Ref2V workspace and the older reference mode
 * in the AI panel -- and they had drifted into two different recipes against the same checkpoint.
 * The panel sent no sampling fields at all and inherited Rust's defaults; the workspace sent a
 * motion shift of 8.0 and a 256 px reference image. Same model, different output, and the worse
 * one was the tab people actually used. Anything a caller omits comes from here.
 */
export const H3_REF_DEFAULTS = {
  samplingMode: "base" as "base" | "turbo-4",
  steps: 20,
  /** The value both released scheduler configs declare and the adapter was distilled against. */
  videoShift: 12.0,
  vaeTile: 256,
  refImageDetail: "match" as H3RefImageDetail,
  refVideoDetail: "safe" as H3RefVideoDetail,
};

export type H3RefRequestInput = {
  prompt: string;
  references: { kind: string; path: string; includeEmbeddedAudio: boolean }[];
  outputPath: string;
  width: number;
  height: number;
  numFrames: number;
  seed: number;
  attentionBackend: string;
  offload: string;
  samplingMode?: "base" | "turbo-4";
  steps?: number;
  videoShift?: number;
  vaeTile?: number;
  refImageDetail?: H3RefImageDetail;
  refVideoDetail?: H3RefVideoDetail;
};

/** Build the `generate_minimax_h3_ref` payload. Every field is explicit; nothing is left to a default. */
export function buildH3RefRequest(input: H3RefRequestInput) {
  const samplingMode = input.samplingMode ?? H3_REF_DEFAULTS.samplingMode;
  return {
    prompt: input.prompt,
    references: input.references,
    outputPath: input.outputPath,
    audioPath: null,
    width: input.width,
    height: input.height,
    numFrames: input.numFrames,
    seed: input.seed,
    attentionBackend: input.attentionBackend,
    offload: input.offload,
    samplingMode,
    steps: input.steps ?? (samplingMode === "turbo-4" ? 4 : H3_REF_DEFAULTS.steps),
    videoShift: input.videoShift ?? H3_REF_DEFAULTS.videoShift,
    vaeTile: input.vaeTile ?? H3_REF_DEFAULTS.vaeTile,
    refImageShortEdge: h3RefImageShortEdge(
      input.refImageDetail ?? H3_REF_DEFAULTS.refImageDetail, input.width, input.height,
    ),
    refVideoShortEdge: h3RefVideoShortEdge(input.refVideoDetail ?? H3_REF_DEFAULTS.refVideoDetail),
  };
}

export type H3AttentionBackend = "native" | "sage" | "sage1" | "sage2_stock";

export interface H3AttentionBackendOption {
  id: H3AttentionBackend;
  label: string;
  badge: string;
  description: string;
}

export const H3_ATTENTION_BACKENDS: H3AttentionBackendOption[] = [
  {
    id: "native",
    label: "Native / cuDNN SDPA (Lossless · Rock Solid)",
    badge: "ROCK SOLID",
    description: "Official NVIDIA PyTorch C++ engine. Lossless BF16 with zero quantization noise and 100% crash resistance.",
  },
  {
    id: "sage",
    label: "Sage Attention 2.0 (Patched · Fast FP8)",
    badge: "SAFE FP8",
    description: "INT8 Q/K + FP8 V with 32-bit accumulator and 64-stride padding. ~15% faster without kernel crashes.",
  },
  {
    id: "sage1",
    label: "Sage Attention 1.0 (High Quality · INT8/FP16)",
    badge: "DEEP IDENTITY",
    description: "INT8 Q/K with unquantized FP16 Values. Superior face and character consistency with fast throughput.",
  },
  {
    id: "sage2_stock",
    label: "Sage Attention 2.0 (Stock / Experimental)",
    badge: "EXPERIMENTAL",
    description: "Unpatched stock SageAttention 2++. Diagnostic mode for single-reference benchmarks.",
  },
];


