import { describe, expect, it } from "vitest";
import {
  SCRUB_AUDIO_DECODE_LIMIT_SECONDS, boundedMediaTime, canDecodeWholeAssetForScrub,
  timelineFollowScroll,
} from "./playback";

describe("long-form playback safeguards", () => {
  it("does not expand a long compressed source into a whole-file Web Audio buffer", () => {
    expect(canDecodeWholeAssetForScrub(30, "video")).toBe(true);
    expect(canDecodeWholeAssetForScrub(31, "video")).toBe(false);
    expect(canDecodeWholeAssetForScrub(SCRUB_AUDIO_DECODE_LIMIT_SECONDS, "audio")).toBe(true);
    expect(canDecodeWholeAssetForScrub(SCRUB_AUDIO_DECODE_LIMIT_SECONDS + 0.01, "audio")).toBe(false);
    expect(canDecodeWholeAssetForScrub(2 * 60 * 60, "audio")).toBe(false);
  });

  it("starts following near the right edge and keeps the programme clock visible", () => {
    const started = timelineFollowScroll({
      playheadSeconds: 28, zoom: 40, scrollLeft: 0, viewportWidth: 1200, scrollWidth: 50_000,
      following: false,
    });
    expect(started.following).toBe(true);
    expect(started.scrollLeft).toBeGreaterThan(0);

    const continued = timelineFollowScroll({
      playheadSeconds: 29, zoom: 40, scrollLeft: started.scrollLeft, viewportWidth: 1200, scrollWidth: 50_000,
      following: true,
    });
    expect(continued.scrollLeft).toBeGreaterThan(started.scrollLeft);
  });

  it("does not disturb a parked playhead already inside the viewport", () => {
    expect(timelineFollowScroll({
      playheadSeconds: 10, zoom: 40, scrollLeft: 0, viewportWidth: 1200, scrollWidth: 50_000,
      following: false,
    })).toEqual({ scrollLeft: 0, following: false });
  });

  it("clamps decoder seeks to the selected source range", () => {
    expect(boundedMediaTime(500, 20, 100)).toBeCloseTo(99.98);
    expect(boundedMediaTime(-5, 20, 100)).toBe(20);
  });
});
