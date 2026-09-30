import { describe, expect, it } from "vitest";
import { EDIT_AREA_MAX, EDIT_SIZE_PRESETS, defaultEditSize, resolveEditSize, snapEditSize } from "./qwenEditSize";

describe("Qwen edit output size", () => {
  it("defaults to Image 1's shape at about a megapixel, snapped to 32, like the worker", () => {
    // canvas_output_size: sqrt(1024^2 * 1.5) = 1254.1 -> 1248, 1254.1 / 1.5 = 836.1 -> 832.
    expect(defaultEditSize({ width: 3000, height: 2000 })).toEqual({ width: 1248, height: 832 });
    expect(defaultEditSize({ width: 512, height: 512 })).toEqual({ width: 1024, height: 1024 });
    expect(defaultEditSize({ width: 1080, height: 1920 })).toEqual({ width: 768, height: 1376 });
    expect(defaultEditSize(undefined)).toBeUndefined();
  });

  it("snaps a manual size to what the pipeline accepts", () => {
    expect(snapEditSize({ width: 1000, height: 750 })).toEqual({ width: 1008, height: 752 });
    expect(snapEditSize({ width: 100, height: 5000 })).toEqual({ width: 256, height: 2048 });
    const huge = snapEditSize({ width: 4096, height: 4096 });
    expect(huge.width * huge.height).toBeLessThanOrEqual(EDIT_AREA_MAX);
    expect(huge).toEqual({ width: 2048, height: 2048 });
    const wide = snapEditSize({ width: 3840, height: 2160 });
    expect(wide.width * wide.height).toBeLessThanOrEqual(EDIT_AREA_MAX);
    expect(Math.abs(wide.width / wide.height - 16 / 9)).toBeLessThan(0.02);
    expect(snapEditSize({ width: Number.NaN, height: 512 })).toEqual({ width: 1024, height: 512 });
  });

  it("resolves the dropdown: default sends nothing, presets and manual send a size", () => {
    expect(resolveEditSize("default", { width: 1, height: 1 })).toBeUndefined();
    expect(resolveEditSize("1344x768", { width: 1, height: 1 })).toEqual({ width: 1344, height: 768 });
    expect(resolveEditSize("manual", { width: 1001, height: 999 })).toEqual({ width: 1008, height: 992 });
    expect(resolveEditSize("nonsense", { width: 1, height: 1 })).toBeUndefined();
    for (const preset of EDIT_SIZE_PRESETS) {
      expect(snapEditSize(preset.size)).toEqual(preset.size);
    }
  });
});
