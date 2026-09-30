import { describe, expect, it } from "vitest";
import {
  DEFAULT_MASK_VFX_RECIPE, MASK_EFFECTS, MASK_VFX_RECIPES, cloneVfxGradient,
  maskEffectFileSuffix, maskEffectReady, maskVfxRenderFields,
} from "./maskEffects";

describe("mask effects catalog", () => {
  it("keeps every preset id unique", () => {
    expect(new Set(MASK_EFFECTS.map((effect) => effect.id)).size).toBe(MASK_EFFECTS.length);
  });

  it("checks only the selected model runtime", () => {
    expect(maskEffectReady("sam2", { sam2Ready: true, birefnetReady: false })).toBe(true);
    expect(maskEffectReady("birefnet", { sam2Ready: true, birefnetReady: false })).toBe(false);
  });

  it("produces compact safe filename suffixes", () => {
    expect(maskEffectFileSuffix("background-pixelate")).toBe("bg-pixelate");
  });

  it("clones editable gradient stops instead of sharing the preset", () => {
    const first = cloneVfxGradient();
    const second = cloneVfxGradient();
    first.stops[0].color = "#000000";
    expect(second.stops[0].color).toBe("#ffffff");
  });

  it("ships complete versioned recipes instead of partial patches", () => {
    const required = Object.keys(DEFAULT_MASK_VFX_RECIPE).sort();
    for (const recipe of MASK_VFX_RECIPES) {
      expect(recipe.schemaVersion).toBe(1);
      expect(Object.keys(recipe).sort()).toEqual(required);
      expect(maskVfxRenderFields(recipe).keyframes).not.toBe(recipe.keyframes);
      expect(maskVfxRenderFields(recipe).gradient.stops).not.toBe(recipe.gradient.stops);
    }
  });

  it("keeps the clean occlusion edge independent from the stylised effect edge", () => {
    const fields = maskVfxRenderFields({
      ...DEFAULT_MASK_VFX_RECIPE, effectFeather: 42, effectExpand: 30,
      occlusionFeather: 1, occlusionExpand: -1,
    });
    expect(fields).toMatchObject({ feather: 42, expand: 30, occlusionFeather: 1, occlusionExpand: -1 });
  });
});
