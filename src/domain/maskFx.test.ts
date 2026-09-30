import { describe, expect, it } from "vitest";
import { DEFAULT_MASK_VFX_RECIPE, MASK_EFFECTS, MASK_VFX_RECIPES } from "./maskEffects";
import { maskBounds, renderMaskFx, sampleAnimation, shapeMask, transitionAlpha, type MaskFxInput } from "./maskFx";
import { translate, scaleBy, type Matrix } from "./photoGeometry";
import type { MaskVfxRecipeV1 } from "./maskVfx";

const W = 64, H = 48;

function frame(width = W, height = H, dx = 0, dy = 0): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const i = (y * width + x) * 4;
    const sx = x - dx, sy = y - dy;
    out[i] = (sx * 4) & 255; out[i + 1] = (sy * 5) & 255; out[i + 2] = 90 + ((sx ^ sy) & 31); out[i + 3] = 255;
  }
  return out;
}

function disc(width = W, height = H, cx = 30, cy = 24, r = 10): Uint8Array {
  const out = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    out[y * width + x] = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) <= r ? 255 : 0;
  }
  return out;
}

const recipe = (over: Partial<MaskVfxRecipeV1> = {}): MaskVfxRecipeV1 => ({
  ...DEFAULT_MASK_VFX_RECIPE, transitionIn: "none", transitionOut: "none", pulseAmount: 0, grain: 0, ...over,
});

const input = (over: Partial<MaskFxInput> = {}): MaskFxInput => ({
  pixels: frame(), width: W, height: H, recipe: recipe(), index: 5, count: 24, fps: 24, envelope: 0,
  matte: disc(), toWork: { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }, sourceWidth: W, sourceHeight: H, ...over,
});

const changed = (a: Uint8ClampedArray, b: Uint8ClampedArray) => { let n = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++; return n; };

describe("mask VFX (compositor port of mask_effects.py)", () => {
  it("renders every effect: finite, alpha kept, deterministic, and visible", () => {
    for (const effect of MASK_EFFECTS) {
      const run = () => renderMaskFx(input({ recipe: recipe({ effect: effect.id, intensity: 1.2 }) }));
      const out = run();
      expect(out.length).toBe(W * H * 4);
      for (let i = 3; i < out.length; i += 4) expect(out[i]).toBe(255);
      expect(changed(out, run()), effect.id).toBe(0);
      // The default recipe blends a gradient through the matte, so even "VFX Matte" changes the frame.
      expect(changed(out, input().pixels), effect.id).toBeGreaterThan(0);
    }
  });

  it("renders every preset, including the flame, rings and burst matte styles", () => {
    for (const preset of MASK_VFX_RECIPES.filter((item) => item.matteStyle !== "clip")) {
      const out = renderMaskFx(input({ recipe: preset, index: 10 }));
      expect(changed(out, input().pixels), preset.presetId).toBeGreaterThan(0);
    }
  });

  it("keys an imported effect clip by luma and puts it behind the subject", () => {
    const effect = { data: new Uint8ClampedArray(32 * 24 * 4), width: 32, height: 24 };
    for (let i = 0; i < 32 * 24; i++) { const bright = (i % 32) < 16 ? 255 : 0; effect.data.set([bright, bright * 0.5, 0, 255], i * 4); }
    const r = recipe({ effect: "animated-matte", matteStyle: "clip", gradientEnabled: false, effectClipColor: "original",
      keyMode: "luma", blendMode: "normal", effectOpacity: 1, rearAmount: 1, frontAmount: 0, subjectAmount: 0 });
    const out = renderMaskFx(input({ recipe: r, effect }));
    const at = (x: number, y: number) => Array.from(out.slice((y * W + x) * 4, (y * W + x) * 4 + 3));
    // Left half of the frame (away from the subject): the orange clip. Right half: the plate.
    expect(at(4, 4)).toEqual([255, 128, 0]);
    expect(at(60, 4)).toEqual(Array.from(frame().slice((4 * W + 60) * 4, (4 * W + 60) * 4 + 3)));
    // The subject (a disc at 30,24) stays in front: untouched.
    expect(at(26, 24)).toEqual(Array.from(frame().slice((24 * W + 26) * 4, (24 * W + 26) * 4 + 3)));
  });

  it("moves with the layer: the same frame translated gives the same picture translated", () => {
    const r = recipe({ effect: "outline", gradientEnabled: false, rearAmount: 1, frontAmount: 0.2, lightWrap: 0.3 });
    const base = renderMaskFx(input({ recipe: r }));
    const width = W + 16, height = H + 12;
    const moved = renderMaskFx(input({
      recipe: r, width, height, pixels: frame(width, height, 16, 12), matte: disc(width, height, 46, 36),
      toWork: translate(16, 12),
    }));
    let worst = 0;
    for (let y = 14; y < H - 14; y++) for (let x = 14; x < W - 14; x++) for (let c = 0; c < 3; c++) {
      worst = Math.max(worst, Math.abs(base[(y * W + x) * 4 + c] - moved[((y + 12) * width + x + 16) * 4 + c]));
    }
    expect(worst).toBeLessThanOrEqual(1);
  });

  it("scales its radii with the layer (a half-size placement is close to the full one, halved)", () => {
    const r = recipe({ effect: "neon-edge", gradientEnabled: false });
    const full = renderMaskFx(input({ recipe: r, width: W * 2, height: H * 2, pixels: frame(W * 2, H * 2), matte: disc(W * 2, H * 2, 60, 48, 20), sourceWidth: W * 2, sourceHeight: H * 2 }));
    const half = renderMaskFx(input({ recipe: r, matte: disc(), sourceWidth: W * 2, sourceHeight: H * 2, toWork: scaleBy(0.5, 0.5) }));
    // The light the effect adds, summed: a half-size placement adds a quarter of it (same look, smaller).
    const added = (out: Uint8ClampedArray, plate: Uint8ClampedArray) => { let sum = 0; for (let i = 0; i < out.length; i++) if (i % 4 !== 3) sum += Math.max(0, out[i] - plate[i]); return sum; };
    const ratio = added(full, frame(W * 2, H * 2)) / (4 * added(half, frame()));
    expect(ratio).toBeGreaterThan(0.6);
    expect(ratio).toBeLessThan(1.6);
  });

  it("reveals through its transitions and keyframes its transform", () => {
    const fade = transitionAlpha(8, 8, "fade", 0.5, 0, 1, [0.5, 0.5], { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }, 8, 8);
    expect(fade[0]).toBe(128);
    const wipe = transitionAlpha(8, 8, "top-to-bottom", 0.5, 0, 1, [0.5, 0.5], { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }, 8, 8);
    expect(wipe[0]).toBe(255);
    expect(wipe[63]).toBe(0);
    const keyed = recipe({ keyframes: [
      { id: "a", at: 0, easing: "linear", values: { effectX: 0.2, effectY: 0.5, effectScale: 100, effectRotation: 0, effectOpacity: 1, intensity: 1 } },
      { id: "b", at: 1, easing: "linear", values: { effectX: 0.8, effectY: 0.5, effectScale: 200, effectRotation: 0, effectOpacity: 0.5, intensity: 2 } },
    ] });
    expect(sampleAnimation(keyed, 0.5)).toMatchObject({ effectX: 0.5, effectScale: 150, effectOpacity: 0.75, intensity: 1.5 });
    // The whole effect fades in: frame 0 of a "fade" entrance is the plate.
    const entering = renderMaskFx(input({ recipe: recipe({ effect: "silhouette", transitionIn: "fade", transitionInDuration: 1, rearAmount: 1, frontAmount: 0, subjectAmount: 0 }), index: 0 }));
    expect(changed(entering, input().pixels)).toBe(0);
  });

  it("grows, feathers and inverts the matte like shape_mask", () => {
    const matte = Float32Array.from(disc(), (value) => value);
    const grown = shapeMask(matte, W, H, 0, 4, false, 1);
    expect(maskBounds(matte, W, H)).toEqual([20, 14, 40, 34]);
    expect(maskBounds(grown, W, H)).toEqual([16, 10, 44, 38]);
    const inverted = shapeMask(matte, W, H, 0, 0, true, 1);
    expect(inverted[0]).toBe(255);
  });
});
