/**
 * The upscale chain, shared by the video export and Image Studio.
 *
 * Two stages, and the order between them is a correctness rule rather than a preference:
 *
 *   AI upscale (Real-ESRGAN)  ->  GPU/CPU resample
 *
 * Resampling up and *then* running Real-ESRGAN asks the model to reconstruct detail from pixels an
 * interpolator already invented, and produces worse output than either stage alone. So the order is
 * fixed here, in code, and is deliberately not exposed as something the user can reorder.
 *
 * The useful shape is therefore "overshoot, then come back down":
 *
 *   1920x1080 -> ESRGAN x4 -> 7680x4320 -> lanczos -> 3840x2160
 *
 * Real-ESRGAN can only produce the integer factor its weights were trained at, so hitting an exact
 * target means letting the model overshoot and resampling down to the requested size. Downsampling
 * after an AI pass is mildly beneficial anyway -- it averages away the model's high-frequency
 * artefacts.
 *
 * NVIDIA's VideoSuperRes is different in the one way that matters here: it accepts an arbitrary
 * output size up to 4x, so it lands on the target directly and the chain has no resample stage at
 * all. That is a real quality difference, not just a shorter pipeline -- there is no second
 * interpolation of the model's output. Models declare this with `variableFactor`.
 *
 * Everything here is pure. The plan is computed identically for the live readout in the UI and for
 * the arguments handed to the workers, so what the panel shows is what actually runs.
 */

export type Size = { width: number; height: number };

/** Which worker implements a given AI stage. */
export type UpscaleBackend = "esrgan" | "nvvfx";

/** `scale_cuda`'s documented interpolation algorithms. Its default (0) is outside this range. */
export type ResampleAlgo = "lanczos" | "bicubic" | "bilinear" | "nearest";

export const RESAMPLE_ALGOS: { value: ResampleAlgo; label: string; hint: string }[] = [
  { value: "lanczos", label: "Lanczos", hint: "Sharpest. The right default for enlargement and for coming back down after an AI pass." },
  { value: "bicubic", label: "Bicubic", hint: "Slightly softer than Lanczos, with less ringing on hard edges." },
  { value: "bilinear", label: "Bilinear", hint: "Soft. Cheap, and rarely what you want for a finished master." },
  { value: "nearest", label: "Nearest", hint: "No blending at all. For pixel art, and nothing else." },
];

export type AiUpscaleModel = {
  id: string;
  /**
   * Which worker path runs this model.
   *
   * `esrgan` loads a weight file from models/real-esrgan/ through spandrel. `nvvfx` calls
   * NVIDIA's VideoSuperRes effect, whose weights live inside the pip-installed wheel -- so it has
   * no `file` and no model pack.
   */
  backend: UpscaleBackend;
  /** Weight file inside models/real-esrgan/. Absent for backends that carry their own weights. */
  file?: string;
  label: string;
  /**
   * For a fixed-factor model this is what the network physically produces and cannot be asked to
   * change. For a variable-factor one it is the largest enlargement the effect supports; anything
   * up to it can be requested exactly. See `variableFactor`.
   */
  factor: number;
  /**
   * Whether the model can hit an arbitrary output size up to `factor`.
   *
   * This is the difference that matters in the chain. A fixed-factor model must overshoot to its
   * trained factor and be resampled back down to the target -- two stages. A variable-factor one
   * lands on the target directly, so there is no resample stage and no second interpolation at all.
   */
  variableFactor?: boolean;
  /** Quality levels the effect exposes, most conservative first. Only meaningful for `nvvfx`. */
  qualityLevels?: string[];
  /** Reported by the backend, kept only so the UI can explain the speed difference. */
  architecture: "ESRGAN" | "RealESRGAN Compact" | "NVIDIA VFX";
  hint: string;
};

/**
 * Verified against spandrel 0.4.2 on torch 2.13 -- every one of these loads from its published
 * state dict with the architecture and scale auto-detected, so nothing here has to hardcode
 * network parameters.
 */
export const AI_UPSCALE_MODELS: AiUpscaleModel[] = [
  {
    id: "photo-x4", backend: "esrgan", file: "RealESRGAN_x4plus.pth", label: "Photographic · 4×", factor: 4,
    architecture: "ESRGAN",
    hint: "The general-purpose model. Best on live footage and photographs; the slowest of the set.",
  },
  {
    id: "photo-x2", backend: "esrgan", file: "RealESRGAN_x2plus.pth", label: "Photographic · 2×", factor: 2,
    architecture: "ESRGAN",
    hint: "The same network trained at 2×. Use it when you only want 2× — it beats running the 4× model and resampling back down.",
  },
  {
    id: "lite-x4", backend: "esrgan", file: "realesr-general-x4v3.pth", label: "Lightweight · 4×", factor: 4,
    architecture: "RealESRGAN Compact",
    hint: "A far smaller network — roughly an order of magnitude faster, with less reconstructed detail. The practical choice for video.",
  },
  {
    id: "illustration-x4", backend: "esrgan", file: "RealESRGAN_x4plus_anime_6B.pth", label: "Illustration · 4×", factor: 4,
    architecture: "ESRGAN",
    hint: "Tuned for drawn and cel-shaded art. On photographs it flattens skin and texture.",
  },
  {
    id: "illustration-video-x4", backend: "esrgan", file: "realesr-animevideov3.pth", label: "Illustration · video · 4×", factor: 4,
    architecture: "RealESRGAN Compact",
    hint: "The compact illustration model, trained on video frames. Still per-frame, but visibly steadier than the 6B model on motion.",
  },
  {
    id: "nvidia-vsr", backend: "nvvfx", label: "NVIDIA VFX · up to 4×", factor: 4,
    variableFactor: true,
    architecture: "NVIDIA VFX",
    qualityLevels: [
      "BICUBIC", "LOW", "MEDIUM", "HIGH", "ULTRA",
      "HIGHBITRATE_LOW", "HIGHBITRATE_MEDIUM", "HIGHBITRATE_HIGH", "HIGHBITRATE_ULTRA",
    ],
    hint: "NVIDIA's VideoSuperRes, from the pip-installed nvidia-vfx wheel rather than a model pack. Unlike Real-ESRGAN it takes an arbitrary output size, so it lands on the target exactly and needs no resample afterwards. Despite the name it is still a per-frame model — measured, not assumed — so it is no steadier in motion than Real-ESRGAN. The HIGHBITRATE levels are for clean sources; the plain ones also remove compression artefacts.",
  },
];

/** The single nvvfx entry. Named rather than searched for, so a table edit cannot silently orphan it. */
export const NVIDIA_UPSCALE_MODEL_ID = "nvidia-vsr";

/** What "Real-ESRGAN" means with no more specific choice made. */
export const DEFAULT_ESRGAN_MODEL = AI_UPSCALE_MODELS.find((model) => model.id === "photo-x4")!;

/** Quality levels only apply to nvvfx; spandrel models have no equivalent knob. */
export const NVVFX_QUALITY_DEFAULT = "HIGH";

export const aiUpscaleModel = (id: string | undefined) =>
  id ? AI_UPSCALE_MODELS.find((item) => item.id === id) : undefined;

/**
 * Hard encode ceilings, per encoder.
 *
 * Provenance matters here, because getting it wrong means an export that dies minutes in:
 *
 *  - HEVC and AV1 NVENC at 8192x8192 on Ada is stated in NVIDIA's own NVENC Application Note.
 *  - H.264 NVENC at 4096x4096 is NOT published in NVIDIA's support matrix, which lists only
 *    per-codec yes/no. 4096 is the figure NVIDIA forum staff and the OBS/ffmpeg community
 *    consistently report, and the true value is queryable at runtime via NV_ENC_CAPS_WIDTH_MAX.
 *    It is treated as a floor here on purpose: refusing a 4100px export that might have worked is
 *    a far cheaper mistake than accepting a 6000px one that certainly will not.
 *  - libx264 / libx265 are level 6.2 reference ceilings (8192x4320). Both encoders will go past
 *    this if level conformance is not enforced, so it is advisory rather than physical.
 */
export const ENCODER_MAX_DIMENSION: Record<string, number> = {
  h264_nvenc: 4096,
  hevc_nvenc: 8192,
  av1_nvenc: 8192,
  libx264: 8192,
  libx265: 8192,
};

export const ENCODER_CEILING_NOTE: Record<string, string> = {
  h264_nvenc: "H.264 NVENC will not encode past 4096 px on either axis.",
  hevc_nvenc: "HEVC NVENC tops out at 8192 px per axis.",
  av1_nvenc: "AV1 NVENC tops out at 8192 px per axis.",
  libx264: "H.264 level 6.2 is defined up to 8192 px wide.",
  libx265: "HEVC level 6.2 is defined up to 8192 px wide.",
};

export type UpscaleStage =
  | { kind: "ai"; model: AiUpscaleModel; width: number; height: number }
  | { kind: "resample"; algo: ResampleAlgo; width: number; height: number; direction: "up" | "down" };

export type UpscalePlan = {
  source: Size;
  stages: UpscaleStage[];
  final: Size;
  /** The live one-line summary the panel shows. Never empty. */
  readout: string;
  warnings: string[];
  /** Present when the requested factor could not be honoured as asked. */
  clamped?: { requested: number; applied: number; reason: string };
};

export type UpscaleInput = {
  source: Size;
  /** Requested enlargement of the source. 1 means "leave it alone". */
  factor: number;
  aiModelId?: string;
  resample: ResampleAlgo;
  /** Encoder ceiling for the *final* size. Intermediates are uncompressed, so they are exempt. */
  maxDimension?: number;
  /** Why the ceiling exists, quoted back to the user rather than left as a bare number. */
  maxDimensionReason?: string;
  /** Video adds the per-frame shimmer warning, and a much larger cost warning, to any AI stage. */
  medium: "image" | "video";
};

/** Encoders reject odd dimensions often enough that every size in the chain is forced even. */
const even = (value: number) => Math.max(2, Math.round(value / 2) * 2);
const fmt = (size: Size) => `${size.width}×${size.height}`;

/**
 * Which family runs the enlargement, chosen before any of its own options are shown.
 *
 * This exists because the panel used to ask the two questions in the wrong order and in the wrong
 * shape: a factor select whose first entry was "Off", and beside it a flat list of every model from
 * both backends. Nothing said which options belonged to which model, the NVIDIA entry sat among
 * four Real-ESRGAN ones, and "off" was a value of the same control that chose how much.
 */
export type UpscaleType = "nvidia" | "esrgan" | "resample";

export const UPSCALE_TYPES: { value: UpscaleType; label: string; hint: string }[] = [
  {
    value: "nvidia", label: "NVIDIA RTX",
    hint: "VideoSuperRes from the nvidia-vfx wheel. It reaches the target size directly, so there is no second interpolation of the model's output. Installed by Install-RTX-Upscale.bat.",
  },
  {
    value: "esrgan", label: "Real-ESRGAN",
    hint: "A choice of weights for photographic or illustrated sources. Each produces exactly its trained factor, so anything else is reached by resampling the model's output afterwards.",
  },
  {
    value: "resample", label: "Resample only",
    hint: "No model. A good Lanczos resize and nothing else — no reconstructed detail, but no invented detail either, no model pack to install, and a fraction of the time.",
  },
];

/**
 * The factors a given type can actually deliver.
 *
 * Currently the same list for all three, and that is a *measured* answer rather than an assumption.
 * NVIDIA's own `rtx-upscale` tooling exposes only `--scale 2x, 3x, 4x`, which read like a hard
 * constraint, so this briefly offered `[2, 3, 4]` for the nvidia type. `scripts/probe-nvvfx.py`
 * against the installed wheel on 2026-08-19 says otherwise: from a 640x360 frame it accepted
 * 1.33x, 1.5x, 2x, 3x, 4x, a non-uniform 1600x720, and an odd 1281x721, returning exactly the
 * requested size every time. So the CLI's list is that CLI's list, not the effect's.
 *
 * Two things the probe did NOT establish, so do not read them into the above:
 *
 * - It accepted 5x as well, which is past both the documented ceiling and anything the model was
 *   trained for. It checked the returned *shape*, not whether the pixels were any good. The 4x cap
 *   is ours, deliberately, and lives in the model table as `factor`.
 * - 1.33x works, so 1080p to 1440p is reachable, but it is not offered here because `1.33` misses
 *   1440 by four pixels. Adding it means adding `4 / 3`, not `1.33`.
 */
export function upscaleFactorsFor(type: UpscaleType): number[] {
  void type;
  return [1.5, 2, 3, 4];
}

/**
 * The panel's settings as `resolveUpscalePlan` wants them.
 *
 * Keeping this separate is what lets the resolver stay the one source of truth: the UI gained a
 * toggle and a type, and the resolver still takes a factor and a model id and is still covered by
 * its own tests. A disabled toggle resolves to factor 1, which produces no stages, which sends no
 * chain -- the same "do nothing" the exporter already understands.
 */
export function upscaleRequest(settings: {
  upscaleEnabled: boolean;
  upscaleType: UpscaleType;
  upscaleFactor: number;
  upscaleAiModel?: string;
}): { factor: number; aiModelId?: string } {
  if (!settings.upscaleEnabled) return { factor: 1, aiModelId: undefined };
  const factors = upscaleFactorsFor(settings.upscaleType);
  // A factor left over from another type -- 1.5x carried across to NVIDIA -- is snapped to the
  // nearest one this type can deliver rather than passed through to fail in the worker.
  const factor = factors.includes(settings.upscaleFactor)
    ? settings.upscaleFactor
    : factors.reduce((best, value) =>
        Math.abs(value - settings.upscaleFactor) < Math.abs(best - settings.upscaleFactor) ? value : best);
  if (settings.upscaleType === "resample") return { factor, aiModelId: undefined };
  if (settings.upscaleType === "nvidia") return { factor, aiModelId: NVIDIA_UPSCALE_MODEL_ID };
  // Real-ESRGAN is the only family with a choice of weights, so it is the only one that reads the
  // stored model id -- and it falls back to the default rather than silently becoming resample-only.
  const chosen = AI_UPSCALE_MODELS.find(
    (model) => model.backend === "esrgan" && model.id === settings.upscaleAiModel,
  );
  return { factor, aiModelId: (chosen ?? DEFAULT_ESRGAN_MODEL).id };
}

export const UPSCALE_FACTORS = [1, 1.5, 2, 3, 4];

/**
 * Resolve a request into the concrete chain that will run.
 *
 * Note what is deliberately NOT done: no attempt is made to pick the model for the user by
 * matching factors. Choosing "Photographic 4×" and a 2× target is a legitimate request -- it
 * overshoots and comes back down, which is the higher-quality route -- and silently swapping to
 * the 2× model would change the look of the output without saying so.
 */
export function resolveUpscalePlan(input: UpscaleInput): UpscalePlan {
  const { source, resample, medium } = input;
  const warnings: string[] = [];
  const stages: UpscaleStage[] = [];
  const model = aiUpscaleModel(input.aiModelId);

  let factor = Math.max(1, input.factor);
  let clamped: UpscalePlan["clamped"];

  // The ceiling binds the final encode, so it caps the requested factor before anything else is
  // resolved -- otherwise the readout would promise a size the encoder is going to refuse.
  if (input.maxDimension) {
    const longest = Math.max(source.width, source.height);
    const ceiling = input.maxDimension / longest;
    if (factor > ceiling) {
      const applied = Math.max(1, Math.floor(ceiling * 100) / 100);
      clamped = {
        requested: factor, applied,
        reason: input.maxDimensionReason ?? `The encoder cannot go past ${input.maxDimension} px.`,
      };
      factor = applied;
    }
  }

  const target: Size = factor === 1
    ? { width: even(source.width), height: even(source.height) }
    : { width: even(source.width * factor), height: even(source.height * factor) };

  if (model) {
    // A variable-factor model is asked for the target directly, capped at what it supports. A
    // fixed-factor one produces its trained factor and nothing else, so it overshoots and the
    // resample below brings it back down. This is the whole practical difference between the two
    // backends: one chain interpolates the model's output a second time, the other does not.
    const reach = model.variableFactor ? Math.min(factor, model.factor) : model.factor;
    const aiOut: Size = {
      width: even(source.width * reach),
      height: even(source.height * reach),
    };
    stages.push({ kind: "ai", model, ...aiOut });

    if (aiOut.width !== target.width || aiOut.height !== target.height) {
      stages.push({
        kind: "resample", algo: resample, ...target,
        direction: target.width < aiOut.width ? "down" : "up",
      });
      if (target.width > aiOut.width) {
        warnings.push(
          `${model.label} reaches ×${model.factor} at most. Everything beyond that is interpolation, ` +
          `not reconstructed detail — a model with a larger factor would give a better result at this size.`,
        );
      }
    }

    if (medium === "video") {
      // This used to name Real-ESRGAN whichever model was selected, which read as a reason to
      // switch to the NVIDIA one. It is not: probing the installed wheel on 2026-08-19 -- the same
      // frame twice, then again after an intervening different frame, then through a fresh
      // instance -- gave bit-identical output every time. VideoSuperRes holds no temporal state
      // either. Both models are per-frame, and the shimmer belongs to both.
      warnings.push(
        `${model.label} is a per-frame model, so on video it runs frame by frame with no knowledge ` +
        "of the frames either side. Fine detail can shimmer or crawl in motion. It also decodes the " +
        "master to frames and re-encodes, which costs disk and a great deal of time at this factor.",
      );
    }
  } else if (target.width !== even(source.width) || target.height !== even(source.height)) {
    stages.push({
      kind: "resample", algo: resample, ...target,
      direction: target.width < source.width ? "down" : "up",
    });
  }

  const final = stages.length ? { width: stages[stages.length - 1].width, height: stages[stages.length - 1].height } : target;

  const readout = stages.length
    ? [fmt(source), ...stages.map((stage) => stage.kind === "ai"
        ? `ESRGAN ×${stage.model.factor} → ${fmt(stage)}`
        : `${stage.algo} → ${fmt(stage)}`)].join(" → ")
    : `${fmt(source)} → unchanged`;

  return { source, stages, final, readout, warnings, clamped };
}

/** The `-vf` fragment for the resample stage. `interp_algo` names map 1:1 onto scale_cuda's. */
export function scaleCudaFilter(stage: Extract<UpscaleStage, { kind: "resample" }>): string {
  return `scale_cuda=${stage.width}:${stage.height}:interp_algo=${stage.algo}`;
}
