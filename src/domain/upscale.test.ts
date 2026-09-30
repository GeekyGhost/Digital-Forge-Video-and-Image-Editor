import { describe, expect, it } from "vitest";
import {
  AI_UPSCALE_MODELS,
  DEFAULT_ESRGAN_MODEL,
  ENCODER_MAX_DIMENSION,
  NVIDIA_UPSCALE_MODEL_ID,
  UPSCALE_TYPES,
  resolveUpscalePlan,
  scaleCudaFilter,
  upscaleFactorsFor,
  upscaleRequest,
  type UpscaleInput,
} from "./upscale";

const HD = { width: 1920, height: 1080 };
const base = (over: Partial<UpscaleInput> = {}): UpscaleInput => ({
  source: HD, factor: 1, resample: "lanczos", medium: "video", ...over,
});

/**
 * The chain resolver is the one piece of this workstream that can be tested without a GPU, an
 * encoder or a model on disk -- and it is also the piece where a mistake is invisible, because a
 * wrong plan still produces a plausible-looking file at some other resolution.
 */
describe("no-op", () => {
  it("reports unchanged rather than an empty chain", () => {
    const plan = resolveUpscalePlan(base());
    expect(plan.stages).toEqual([]);
    expect(plan.readout).toBe("1920×1080 → unchanged");
    expect(plan.final).toEqual(HD);
  });

  it("forces even dimensions even when nothing else happens", () => {
    const plan = resolveUpscalePlan(base({ source: { width: 1921, height: 1081 } }));
    expect(plan.final).toEqual({ width: 1922, height: 1082 });
  });
});

describe("resample only", () => {
  it("goes straight to the target", () => {
    const plan = resolveUpscalePlan(base({ factor: 2 }));
    expect(plan.stages).toHaveLength(1);
    expect(plan.final).toEqual({ width: 3840, height: 2160 });
    expect(plan.readout).toBe("1920×1080 → lanczos → 3840×2160");
  });

  it("handles the fractional factor without landing on an odd size", () => {
    const plan = resolveUpscalePlan(base({ factor: 1.5, source: { width: 1280, height: 719 } }));
    expect(plan.final.width % 2).toBe(0);
    expect(plan.final.height % 2).toBe(0);
  });

  it("emits the interp_algo the default would otherwise skip", () => {
    const plan = resolveUpscalePlan(base({ factor: 2 }));
    const stage = plan.stages[0];
    if (stage.kind !== "resample") throw new Error("expected a resample stage");
    expect(scaleCudaFilter(stage)).toBe("scale_cuda=3840:2160:interp_algo=lanczos");
  });
});

describe("AI first, resample second", () => {
  it("overshoots with the model and comes back down to the exact target", () => {
    const plan = resolveUpscalePlan(base({ factor: 2, aiModelId: "photo-x4" }));
    expect(plan.stages.map((s) => s.kind)).toEqual(["ai", "resample"]);
    expect(plan.stages[0]).toMatchObject({ width: 7680, height: 4320 });
    expect(plan.final).toEqual({ width: 3840, height: 2160 });
    expect(plan.readout).toBe("1920×1080 → ESRGAN ×4 → 7680×4320 → lanczos → 3840×2160");
  });

  it("never puts a resample before the AI stage, whatever the factor", () => {
    for (const factor of [1, 1.5, 2, 3, 4]) {
      for (const model of AI_UPSCALE_MODELS) {
        const plan = resolveUpscalePlan(base({ factor, aiModelId: model.id, maxDimension: undefined }));
        expect(plan.stages[0].kind).toBe("ai");
        expect(plan.stages.filter((s) => s.kind === "ai")).toHaveLength(1);
      }
    }
  });

  it("drops the resample when the model lands exactly on the target", () => {
    const plan = resolveUpscalePlan(base({ factor: 4, aiModelId: "photo-x4" }));
    expect(plan.stages.map((s) => s.kind)).toEqual(["ai"]);
    expect(plan.final).toEqual({ width: 7680, height: 4320 });
  });

  it("says so when the model cannot reach the requested factor", () => {
    const plan = resolveUpscalePlan(base({ factor: 4, aiModelId: "photo-x2" }));
    expect(plan.stages.map((s) => s.kind)).toEqual(["ai", "resample"]);
    expect(plan.stages[1]).toMatchObject({ direction: "up" });
    expect(plan.warnings.some((w) => w.includes("reaches ×2 at most"))).toBe(true);
  });

  it("does not warn about interpolation when it is resampling down", () => {
    const plan = resolveUpscalePlan(base({ factor: 2, aiModelId: "photo-x4" }));
    expect(plan.warnings.some((w) => w.includes("at most"))).toBe(false);
  });

  it("warns about per-frame shimmer on video but not on stills", () => {
    const video = resolveUpscalePlan(base({ factor: 2, aiModelId: "photo-x4", medium: "video" }));
    const image = resolveUpscalePlan(base({ factor: 2, aiModelId: "photo-x4", medium: "image" }));
    expect(video.warnings.some((w) => w.includes("frame by frame"))).toBe(true);
    expect(image.warnings.some((w) => w.includes("frame by frame"))).toBe(false);
  });
});

describe("encoder ceilings", () => {
  it("clamps 4x of 1080p down for H.264 NVENC and says why", () => {
    const plan = resolveUpscalePlan(base({
      factor: 4,
      maxDimension: ENCODER_MAX_DIMENSION.h264_nvenc,
      maxDimensionReason: "H.264 NVENC will not encode past 4096 px on either axis.",
    }));
    expect(plan.clamped?.requested).toBe(4);
    expect(plan.final.width).toBeLessThanOrEqual(4096);
    expect(plan.clamped?.reason).toContain("4096");
  });

  it("leaves the same request alone on HEVC", () => {
    const plan = resolveUpscalePlan(base({ factor: 4, maxDimension: ENCODER_MAX_DIMENSION.hevc_nvenc }));
    expect(plan.clamped).toBeUndefined();
    expect(plan.final).toEqual({ width: 7680, height: 4320 });
  });

  it("caps the encoded result but lets the AI intermediate exceed the ceiling", () => {
    // The intermediate is uncompressed frames, so the encoder's limit does not apply to it --
    // clamping it there would throw away detail for no reason.
    const plan = resolveUpscalePlan(base({ factor: 2, aiModelId: "photo-x4", maxDimension: 4096 }));
    const ai = plan.stages[0];
    expect(ai.kind).toBe("ai");
    expect(ai.width).toBe(7680);
    expect(plan.final.width).toBeLessThanOrEqual(4096);
  });

  it("never clamps below 1x", () => {
    const plan = resolveUpscalePlan(base({ factor: 4, source: { width: 8000, height: 4500 }, maxDimension: 4096 }));
    expect(plan.clamped?.applied).toBe(1);
  });
});

describe("model table", () => {
  it("has unique ids and only integer factors", () => {
    const ids = AI_UPSCALE_MODELS.map((m) => m.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const model of AI_UPSCALE_MODELS) {
      expect(Number.isInteger(model.factor)).toBe(true);
      // Only the spandrel backend loads a weight file; nvvfx carries its weights in the wheel.
      if (model.file !== undefined) expect(model.file.endsWith(".pth")).toBe(true);
    }
  });
});

describe("variable-factor models (nvvfx)", () => {
  it("lands on the target directly, with no resample stage at all", () => {
    // This is the whole point of the backend: a fixed-factor model needs ESRGAN x4 followed by a
    // lanczos pass back down to 2x, which interpolates the model's output a second time. A
    // variable-factor one is simply asked for 2x.
    const plan = resolveUpscalePlan(base({ factor: 2, aiModelId: "nvidia-vsr" }));
    expect(plan.stages.map((s) => s.kind)).toEqual(["ai"]);
    expect(plan.final).toEqual({ width: 3840, height: 2160 });
    expect(plan.readout).toBe("1920×1080 → ESRGAN ×4 → 3840×2160");
  });

  it("handles a fractional factor without a second stage", () => {
    const plan = resolveUpscalePlan(base({ factor: 1.5, aiModelId: "nvidia-vsr" }));
    expect(plan.stages).toHaveLength(1);
    expect(plan.final).toEqual({ width: 2880, height: 1620 });
  });

  it("caps at the model's ceiling and resamples up beyond it, saying so", () => {
    // 1920 x 6 is past what the effect supports, so it does its 4x and the rest is interpolation.
    const plan = resolveUpscalePlan(base({ factor: 6, aiModelId: "nvidia-vsr" }));
    expect(plan.stages.map((s) => s.kind)).toEqual(["ai", "resample"]);
    expect(plan.stages[0]).toMatchObject({ width: 7680, height: 4320 });
    expect(plan.stages[1]).toMatchObject({ direction: "up" });
    expect(plan.warnings.some((w) => w.includes("reaches ×4 at most"))).toBe(true);
  });

  it("still puts the AI stage first", () => {
    for (const factor of [1.5, 2, 3, 4, 6]) {
      const plan = resolveUpscalePlan(base({ factor, aiModelId: "nvidia-vsr" }));
      expect(plan.stages[0].kind).toBe("ai");
    }
  });

  it("respects the encoder ceiling like any other model", () => {
    const plan = resolveUpscalePlan(base({
      factor: 4, aiModelId: "nvidia-vsr", maxDimension: ENCODER_MAX_DIMENSION.h264_nvenc,
    }));
    expect(plan.final.width).toBeLessThanOrEqual(4096);
    expect(plan.clamped?.requested).toBe(4);
    // Clamped to roughly 2.13x, which the effect can produce directly -- so still one stage.
    expect(plan.stages.map((s) => s.kind)).toEqual(["ai"]);
  });

  it("produces one fewer interpolation than the fixed-factor route at the same target", () => {
    const fixed = resolveUpscalePlan(base({ factor: 2, aiModelId: "photo-x4" }));
    const variable = resolveUpscalePlan(base({ factor: 2, aiModelId: "nvidia-vsr" }));
    expect(fixed.final).toEqual(variable.final);
    expect(fixed.stages.filter((s) => s.kind === "resample")).toHaveLength(1);
    expect(variable.stages.filter((s) => s.kind === "resample")).toHaveLength(0);
  });
});

describe("backends", () => {
  it("every model declares one, and only esrgan models carry a weight file", () => {
    for (const model of AI_UPSCALE_MODELS) {
      expect(["esrgan", "nvvfx"]).toContain(model.backend);
      if (model.backend === "esrgan") {
        expect(model.file, `${model.id} needs a weight file`).toBeTruthy();
        expect(model.variableFactor ?? false).toBe(false);
      } else {
        // nvvfx carries its weights inside the pip wheel, so there is no pack and no file.
        expect(model.file).toBeUndefined();
        expect(model.qualityLevels?.length).toBeGreaterThan(0);
      }
    }
  });
});

/**
 * The panel asks "on or off", then "which upscaler", then that upscaler's own options. The
 * resolver still takes a factor and a model id. `upscaleRequest` is the whole of the translation
 * between the two, which makes it the place a wrong answer becomes an export that silently does
 * nothing -- exactly the failure this shape was introduced to remove.
 */
describe("panel settings to a chain request", () => {
  const settings = (over: Partial<Parameters<typeof upscaleRequest>[0]> = {}) => ({
    upscaleEnabled: true, upscaleType: "nvidia" as const, upscaleFactor: 2, ...over,
  });

  it("resolves to no work at all when the toggle is off", () => {
    const request = upscaleRequest(settings({ upscaleEnabled: false, upscaleFactor: 4 }));
    expect(request).toEqual({ factor: 1, aiModelId: undefined });
    // Factor 1 has to produce an empty plan, because an empty plan is what stops a chain being
    // sent at all. If this ever produced a stage, the toggle would be decorative.
    expect(resolveUpscalePlan({ source: HD, resample: "lanczos", medium: "video", ...request }).stages)
      .toHaveLength(0);
  });

  it("sends the NVIDIA model without the user naming it", () => {
    expect(upscaleRequest(settings()).aiModelId).toBe(NVIDIA_UPSCALE_MODEL_ID);
  });

  it("ignores a stored Real-ESRGAN model while the type is NVIDIA", () => {
    const request = upscaleRequest(settings({ upscaleAiModel: "illustration-x4" }));
    expect(request.aiModelId).toBe(NVIDIA_UPSCALE_MODEL_ID);
  });

  it("falls back to a default model rather than silently becoming resample-only", () => {
    const request = upscaleRequest(settings({ upscaleType: "esrgan", upscaleAiModel: undefined }));
    expect(request.aiModelId).toBe(DEFAULT_ESRGAN_MODEL.id);
    // And a stale id from the other backend must not be honoured just because it is a valid id.
    expect(upscaleRequest(settings({ upscaleType: "esrgan", upscaleAiModel: NVIDIA_UPSCALE_MODEL_ID })).aiModelId)
      .toBe(DEFAULT_ESRGAN_MODEL.id);
  });

  it("runs no model at all on the resample-only type", () => {
    const request = upscaleRequest(settings({ upscaleType: "resample", upscaleFactor: 3 }));
    expect(request).toEqual({ factor: 3, aiModelId: undefined });
  });

  it("snaps a factor the chosen upscaler cannot deliver", () => {
    // The lists happen to agree today, so nothing is snapped. The mechanism is still pinned
    // because it is what stops a per-type constraint -- if one ever turns out to be real -- from
    // reaching the worker as a request it refuses after the whole master has already rendered.
    for (const type of UPSCALE_TYPES) {
      const factors = upscaleFactorsFor(type.value);
      const request = upscaleRequest(settings({ upscaleType: type.value, upscaleFactor: 2.5 }));
      expect(factors).toContain(request.factor);
    }
  });

  it("offers the same factors for every type, which is a measured answer", () => {
    // NVIDIA's own CLI exposes only 2x/3x/4x, so this briefly restricted the nvidia type to those.
    // probe-nvvfx.py against the installed wheel accepted 1.33x, 1.5x, non-uniform and odd targets
    // and returned the requested size every time, so the CLI's list is the CLI's, not the effect's.
    expect(upscaleFactorsFor("nvidia")).toEqual(upscaleFactorsFor("esrgan"));
    expect(upscaleFactorsFor("nvidia")).toContain(1.5);
    for (const type of UPSCALE_TYPES) {
      // 1 is "off" now, and off is the toggle. A factor list containing it would give two controls
      // that both mean off, which is the shape this replaced.
      expect(upscaleFactorsFor(type.value)).not.toContain(1);
      // And nothing past the 4x the model table caps `reach` at. The wheel accepted 5x, but only
      // the returned shape was checked -- it is well past what the model was trained for.
      expect(Math.max(...upscaleFactorsFor(type.value))).toBeLessThanOrEqual(4);
    }
  });

  it("names a type for every upscaler the model table can reach", () => {
    const backends = new Set(AI_UPSCALE_MODELS.map((model) => model.backend));
    expect(backends).toEqual(new Set(["esrgan", "nvvfx"]));
    expect(UPSCALE_TYPES.map((type) => type.value)).toEqual(["nvidia", "esrgan", "resample"]);
  });
});
