import { describe, expect, it } from "vitest";
import {
  H3_MAX_PIXELS,
  H3_MAX_LONG_EDGE,
  H3_TRAINED_SHORT_EDGE,
  H3_LOW_VRAM_SHORT_EDGE,
  H3_REF_DEFAULTS,
  H3_REF_IMAGE_DETAILS,
  H3_REF_VIDEO_DETAILS,
  H3_RESOLUTION_TIERS,
  buildH3RefRequest,
  fitH3Dimensions,
  h3CanvasFits,
  h3RefImageShortEdge,
  h3RefVideoShortEdge,
} from "./h3";

/**
 * The Ref2VA recipe, pinned against the published sources rather than against whatever the panel
 * happened to be initialised with. Three of these values had drifted away from what MiniMax,
 * ComfyUI and LightX2V actually specify, and the drift was invisible because nothing asserted it.
 */
describe("the published Ref2VA recipe", () => {
  it("uses the scheduler shift the adapter was distilled against", () => {
    // Both released scheduler configs declare video 12.0 / audio 3.0; ComfyUI's MiniMaxH3SigmaShift
    // node defaults to the same pair; LightX2V's own inference script hardcodes
    // DEFAULT_VIDEO_SHIFT = 12.0. The panel defaulted to 8.0 and sent it on every request.
    expect(H3_REF_DEFAULTS.videoShift).toBe(12.0);
  });

  it("defaults to Base rather than a preview adapter", () => {
    // LightX2V's Ref2V adapter is v0.1, published as a preview, and no 8-step Ref2V adapter
    // exists. ComfyUI's shipped R2V workflow uses no LoRA and 20 steps.
    expect(H3_REF_DEFAULTS.samplingMode).toBe("base");
    expect(H3_REF_DEFAULTS.steps).toBe(20);
  });

  it("conditions reference images at the canvas, not at a draft size", () => {
    expect(H3_REF_DEFAULTS.refImageDetail).toBe("match");
    expect(H3_REF_IMAGE_DETAILS[0].id).toBe("match");
  });

  it("uses MiniMax's released VAE tile geometry", () => {
    expect(H3_REF_DEFAULTS.vaeTile).toBe(256);
  });
});

describe("reference geometry", () => {
  it("resolves `match` to the canvas's short edge on either orientation", () => {
    // This was `dimensions.height`, which is the short edge only when the canvas is landscape.
    expect(h3RefImageShortEdge("match", 960, 544)).toBe(544);
    expect(h3RefImageShortEdge("match", 544, 960)).toBe(544);
    expect(h3RefImageShortEdge("match", 768, 768)).toBe(768);
  });

  it("keeps the absolute tiers absolute", () => {
    expect(h3RefImageShortEdge("max", 960, 544)).toBe(2048);
    expect(h3RefImageShortEdge("high", 960, 544)).toBe(1024);
    expect(h3RefImageShortEdge("draft", 960, 544)).toBe(256);
  });

  it("offers a reference-video tier that reaches upstream's own budget", () => {
    expect(h3RefVideoShortEdge("safe")).toBe(384);
    expect(h3RefVideoShortEdge("native")).toBe(H3_TRAINED_SHORT_EDGE);
    expect(H3_REF_VIDEO_DETAILS.map((item) => item.shortEdge)).toEqual([384, 544, 768]);
  });
});

describe("the canvas contract", () => {
  it("accepts MiniMax's own trained canvas", () => {
    expect(h3CanvasFits(1344, 768)).toBe(true);
    expect(fitH3Dimensions(16, 9, H3_TRAINED_SHORT_EDGE)).toEqual({
      width: 1344, height: 768, value: "1344x768",
    });
  });

  it("refuses the 1.0 MP selection ComfyUI's tutorial warns about", () => {
    // "Avoid the 1.0 Megapixel setting (yields 1376x768, exceeding the model's 768x1344 cap)."
    expect(h3CanvasFits(1376, 768)).toBe(false);
    expect(1376 * 768).toBeGreaterThan(H3_MAX_PIXELS);
  });

  it("exposes 768 as a tier and stops calling 544 high fidelity", () => {
    const native = H3_RESOLUTION_TIERS.find((tier) => tier.id === "native");
    expect(native?.shortEdge).toBe(H3_TRAINED_SHORT_EDGE);
    const standard = H3_RESOLUTION_TIERS.find((tier) => tier.id === "standard");
    expect(standard?.shortEdge).toBe(H3_LOW_VRAM_SHORT_EDGE);
    expect(standard?.label).not.toMatch(/high fidelity/i);
  });

  it("allows the long edge the area budget allows", () => {
    expect(H3_MAX_LONG_EDGE).toBe(1344);
    expect(H3_MAX_PIXELS).toBe(768 * 1344);
  });
});

describe("the shared request builder", () => {
  const inputs = {
    prompt: "a shot",
    references: [{ kind: "image", path: "a.png", includeEmbeddedAudio: false }],
    outputPath: "out.mp4",
    width: 960,
    height: 544,
    numFrames: 124,
    seed: 7,
    attentionBackend: "native",
    offload: "balanced",
  };

  it("sends a complete recipe when the caller specifies nothing", () => {
    // The AI panel's reference mode used to omit every sampling field and inherit Rust's defaults,
    // which is how two call sites ended up running different recipes against one checkpoint.
    expect(buildH3RefRequest(inputs)).toEqual({
      ...inputs,
      audioPath: null,
      samplingMode: "base",
      steps: 20,
      videoShift: 12.0,
      vaeTile: 256,
      refImageShortEdge: 544,
      refVideoShortEdge: 384,
    });
  });

  it("gives Turbo the step count its distillation fixes", () => {
    const request = buildH3RefRequest({ ...inputs, samplingMode: "turbo-4" });
    expect(request.steps).toBe(4);
  });

  it("lets a caller override any part of it", () => {
    const request = buildH3RefRequest({
      ...inputs, samplingMode: "base", steps: 50, videoShift: 8, refImageDetail: "max",
      refVideoDetail: "native", vaeTile: 384,
    });
    expect(request).toMatchObject({
      samplingMode: "base", steps: 50, videoShift: 8,
      refImageShortEdge: 2048, refVideoShortEdge: 768, vaeTile: 384,
    });
  });
});
