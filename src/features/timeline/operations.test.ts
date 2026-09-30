import { describe, expect, it } from "vitest";
import { createProject } from "../../domain/project";
import { fromSeconds, seconds } from "../../domain/time";
import type { MediaAsset, TimelineClip } from "../../domain/types";
import { replaceClipRange } from "./operations";

function fixture(rate = 1) {
  const project = createProject("Retake range contract");
  const source: MediaAsset = {
    id: "source", name: "source.mp4", path: "source.mp4", kind: "video",
    duration: fromSeconds(10), width: 160, height: 96,
    frameRate: { numerator: 30_000, denominator: 1_001 }, hasAudio: true,
  };
  const replacement: MediaAsset = {
    id: "replacement", name: "retake.mp4", path: "retake.mp4", kind: "generated",
    duration: fromSeconds(1.5015), width: 160, height: 96,
    frameRate: { numerator: 30_000, denominator: 1_001 }, hasAudio: true,
  };
  const clip: TimelineClip = {
    id: "clip", assetId: source.id, name: source.name, trackId: "v1",
    start: fromSeconds(4), sourceIn: fromSeconds(0), sourceOut: fromSeconds(10),
    enabled: true, color: "#000", opacity: 1, volumeDb: 0,
    speed: [{ at: fromSeconds(0), rate, easing: "linear" }],
  };
  project.assets.push(source);
  project.sequences[0].tracks.find((track) => track.id === "v1")!.clips.push(clip);
  return { project, replacement };
}

describe("frame-aligned retake replacement", () => {
  it.each([1, 2, 0.5])("preserves clip speed %sx and maps source offsets to timeline time", (rate) => {
    const { project, replacement } = fixture(rate);
    const startFrame = 30;
    const endFrame = 75;
    const start = startFrame * 1_001 / 30_000;
    const end = endFrame * 1_001 / 30_000;
    const result = replaceClipRange(project, project.activeSequenceId, "clip", start, end, replacement);
    const pieces = result.project.sequences[0].tracks.find((track) => track.id === "v1")!.clips;
    expect(pieces).toHaveLength(3);
    expect(seconds(pieces[1].start)).toBeCloseTo(4 + start / rate, 6);
    expect(seconds(pieces[2].start)).toBeCloseTo(4 + end / rate, 6);
    expect(pieces.every((piece) => piece.speed[0].rate === rate)).toBe(true);
    expect(seconds(pieces[0].sourceOut)).toBeCloseTo(start, 6);
    expect(seconds(pieces[2].sourceIn)).toBeCloseTo(end, 6);
  });

  it("uses the returned frame boundary rather than the user's non-aligned decimal", () => {
    const { project, replacement } = fixture();
    const effectiveStart = 23 * 1_001 / 30_000;
    const effectiveEnd = 61 * 1_001 / 30_000;
    const result = replaceClipRange(
      project, project.activeSequenceId, "clip", effectiveStart, effectiveEnd, replacement,
    );
    const pieces = result.project.sequences[0].tracks.find((track) => track.id === "v1")!.clips;
    expect(seconds(pieces[0].sourceOut)).toBeCloseTo(effectiveStart, 6);
    expect(seconds(pieces[2].sourceIn)).toBeCloseTo(effectiveEnd, 6);
  });
});
