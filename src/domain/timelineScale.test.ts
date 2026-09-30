import { describe, expect, it } from "vitest";
import { visibleTimelineTicks } from "./timelineScale";

describe("long timeline ruler virtualization", () => {
  it("keeps an all-day timeline bounded to the viewport", () => {
    const ticks = visibleTimelineTicks(24 * 60 * 60, 28, 0, 1600);
    expect(ticks.length).toBeGreaterThan(5);
    expect(ticks.length).toBeLessThan(200);
  });

  it("moves the generated marks with horizontal scrolling", () => {
    const first = visibleTimelineTicks(10_000, 40, 0, 1200);
    const later = visibleTimelineTicks(10_000, 40, 200_000, 1200);
    expect(first[0].time).toBeLessThan(10);
    expect(later[0].time).toBeGreaterThan(4_900);
    expect(later.length).toBeLessThan(200);
  });
});
