import { describe, expect, it } from "vitest";
import { defaultGraphicOverlay, textMotionFrame } from "./textMotion";

describe("text motion", () => {
  it("types characters in and back out against the clip clock", () => {
    const graphic = { ...defaultGraphicOverlay("title"), text: "FORGE", motionIn: "typewriter" as const,
      motionOut: "typewriter" as const, motionDuration: 1 };
    expect(textMotionFrame(graphic, 0.3, 4).text).toBe("FO");
    expect(textMotionFrame(graphic, 2, 4).text).toBe("FORGE");
    expect(textMotionFrame(graphic, 3.7, 4).text).toBe("F");
  });

  it("caps animation time at half a short graphic", () => {
    const graphic = { ...defaultGraphicOverlay("title"), text: "A", motionIn: "fade" as const,
      motionOut: "fade" as const, motionDuration: 3 };
    expect(textMotionFrame(graphic, 0.25, 1).opacity).toBeCloseTo(0.5);
    expect(textMotionFrame(graphic, 0.75, 1).opacity).toBeCloseTo(0.5);
  });

  it("keeps static legacy graphics fully visible", () => {
    const graphic = defaultGraphicOverlay("lower-third");
    const frame = textMotionFrame(graphic, 1, 5);
    expect(frame.text).toBe(graphic.text);
    expect(frame.opacity).toBe(1);
    expect(frame.scale).toBe(1);
  });
});
