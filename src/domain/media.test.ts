import { describe, expect, it } from "vitest";
import { isStillAsset, resolveFramePickerIds, resolvePickerId } from "./media";

const asset = (id: string) => ({ id });

/**
 * The 2026-08-27 report: "I have two images selected, but it won't do anything when I click
 * generate", with the status bar saying "Choose a conditioning image".
 *
 * React drives a controlled `<select>` by setting `selected` on each matching `<option>`. When the
 * value matches nothing, every option is left unselected and a single-line select in Chromium
 * displays its FIRST option — so a stale id is indistinguishable from a real selection on screen
 * while `images.find(a => a.id === firstId)` returns undefined behind it. The picker looked set,
 * the state was not, and the guard told the user to do the thing they had already done.
 *
 * The old effect only filled an id that was empty, so an id could never recover once its asset
 * left the bin.
 */
describe("frame picker ids survive the bin changing under them", () => {
  const bin = [asset("a"), asset("b"), asset("c")];

  it("keeps selections that are still in the bin", () => {
    expect(resolveFramePickerIds(bin, "c", "a")).toEqual({ first: "c", last: "a" });
  });

  it("replaces a first frame whose asset has gone", () => {
    // What the select was already displaying — its first option — is what the state becomes, so
    // the control and the state agree instead of silently disagreeing.
    expect(resolveFramePickerIds(bin, "deleted", "b")).toEqual({ first: "a", last: "b" });
  });

  it("replaces a last frame whose asset has gone, and never duplicates the first", () => {
    expect(resolveFramePickerIds(bin, "b", "deleted")).toEqual({ first: "b", last: "a" });
    // `assets[1]` would have been "b" here, which is the first frame. First-and-last needs two.
    expect(resolveFramePickerIds([asset("b"), asset("z")], "b", "gone")).toEqual({
      first: "b", last: "z",
    });
  });

  it("fills empty ids from the bin", () => {
    expect(resolveFramePickerIds(bin, "", "")).toEqual({ first: "a", last: "b" });
  });

  it("leaves last empty when there is only one image, rather than aliasing first", () => {
    expect(resolveFramePickerIds([asset("only")], "", "")).toEqual({ first: "only", last: "" });
  });

  it("clears both when the bin empties", () => {
    expect(resolveFramePickerIds([], "a", "b")).toEqual({ first: "", last: "" });
  });

  it("is idempotent, so the effect that runs it cannot loop", () => {
    const once = resolveFramePickerIds(bin, "gone", "also-gone");
    expect(resolveFramePickerIds(bin, once.first, once.last)).toEqual(once);
  });
});

describe("single pickers follow the same rule", () => {
  it("keeps a live selection and replaces a dead one", () => {
    expect(resolvePickerId([asset("x"), asset("y")], "y")).toBe("y");
    expect(resolvePickerId([asset("x"), asset("y")], "removed")).toBe("x");
    expect(resolvePickerId([], "removed")).toBe("");
  });
});

import { canConditionFrames, resolveBridgePair, resolveExtendClip, type FrameSourceClip } from "./media";

/**
 * The 2026-08-27 report: "Clip to clip using FL2VA isn't working, it won't let me select the clips
 * on the timeline and there is no drop down."
 *
 * There is no dropdown by design — both editor workflows read the timeline selection. What was
 * actually wrong is that the eligibility rule excluded still images (`["video", "generated"]`),
 * and the notice said "Select the outgoing V1 clip" for every possible reason it could not run.
 * A user whose second V1 clip was a canvas PNG was told, repeatedly, to select a clip they had
 * already selected.
 */
const clip = (id: string, assetId: string, start: number, end: number): FrameSourceClip =>
  ({ id, assetId, startSeconds: start, endSeconds: end });

const KINDS = {
  shot: { kind: "video" as const, path: "shot.mp4", name: "shot.mp4" },
  made: { kind: "generated" as const, path: "H3-first-last.mp4", name: "H3-first-last.mp4" },
  still: { kind: "image" as const, path: "canvas-20260827.png", name: "canvas-20260827.png" },
  song: { kind: "audio" as const, path: "score.wav", name: "score.wav" },
  // An audio file the importer mis-tagged as video — the case `isAudioAsset` exists to repair.
  mistagged: { kind: "video" as const, path: "voice.mp3", name: "voice.mp3" },
};
const assetFor = (id: string) => (KINDS as Record<string, typeof KINDS.shot | undefined>)[id];

describe("what can supply a conditioning frame", () => {
  it("accepts footage, generated video and stills", () => {
    expect(canConditionFrames(KINDS.shot)).toBe(true);
    expect(canConditionFrames(KINDS.made)).toBe(true);
    // The whole point of the fix: a PNG on V1 has exactly one frame, and it is a fine one.
    expect(canConditionFrames(KINDS.still)).toBe(true);
  });

  it("rejects audio, including a file mis-tagged as video", () => {
    expect(canConditionFrames(KINDS.song)).toBe(false);
    expect(canConditionFrames(KINDS.mistagged)).toBe(false);
  });
});

describe("continuing from a clip says why it cannot", () => {
  const clips = [clip("a", "made", 0, 5.167), clip("b", "still", 5.167, 10.167)];

  it("resolves a selected V1 clip", () => {
    const result = resolveExtendClip(clips, assetFor, "a", true);
    expect("blocker" in result).toBe(false);
  });

  it("names each reason instead of restating the remedy", () => {
    expect(resolveExtendClip(clips, assetFor, undefined, false)).toEqual({
      blocker: "Select the clip on V1 to continue from.",
    });
    expect(resolveExtendClip(clips, assetFor, "a", false)).toMatchObject({
      blocker: expect.stringContaining("not on V1"),
    });
    expect(resolveExtendClip(clips, assetFor, "disabled-clip", true)).toMatchObject({
      blocker: expect.stringContaining("disabled"),
    });
    expect(resolveExtendClip([clip("c", "gone", 0, 5)], assetFor, "c", true)).toMatchObject({
      blocker: expect.stringContaining("missing from the project"),
    });
    expect(resolveExtendClip([clip("d", "song", 0, 5)], assetFor, "d", true)).toMatchObject({
      blocker: expect.stringContaining("audio"),
    });
  });
});

describe("bridging a cut says why it cannot", () => {
  it("bridges a generated clip into a still, which is what was reported broken", () => {
    const clips = [clip("a", "made", 0, 5.167), clip("b", "still", 5.167, 10.167)];
    const result = resolveBridgePair(clips, assetFor, "a", true);
    expect("blocker" in result).toBe(false);
    expect(result).toMatchObject({ from: { id: "a" }, to: { id: "b" } });
  });

  it("reports a gap as a length to fill, not a fault", () => {
    // The reported timeline: the second clip started around 00:10 with the first ending at 5.167s.
    // This first refused the gap outright, which was wrong — a hole between two clips is exactly
    // where a bridge goes, and its size is the natural length for one.
    const clips = [clip("a", "made", 0, 5.167), clip("b", "still", 10, 15)];
    const result = resolveBridgePair(clips, assetFor, "a", true);
    expect("blocker" in result).toBe(false);
    expect(result).toMatchObject({ gapSeconds: expect.closeTo(4.833, 3) });
  });

  it("treats a sub-frame seam as touching, not as a gap", () => {
    const clips = [clip("a", "made", 0, 5.167), clip("b", "shot", 5.18, 10)];
    const result = resolveBridgePair(clips, assetFor, "a", true);
    expect("blocker" in result).toBe(false);
    expect((result as { gapSeconds: number }).gapSeconds).toBeLessThan(1 / 48);
  });

  it("says when there is nothing after the selection", () => {
    expect(resolveBridgePair([clip("a", "made", 0, 5)], assetFor, "a", true)).toMatchObject({
      blocker: expect.stringContaining("no enabled V1 clip after this one"),
    });
  });

  it("names the following clip when it cannot supply a frame", () => {
    const clips = [clip("a", "made", 0, 5), clip("b", "song", 5, 10)];
    expect(resolveBridgePair(clips, assetFor, "a", true)).toMatchObject({
      blocker: expect.stringContaining("score.wav"),
    });
  });

  it("rewords the outgoing reason for the bridge", () => {
    expect(resolveBridgePair([], assetFor, undefined, false)).toEqual({
      blocker: "Select the clip on V1 to bridge from.",
    });
  });
});

import { planBridgePlacement } from "./media";

/**
 * "If we set a duration it should insert the new clip between them and push the last frame image
 * or clip over more to make room if needed."
 *
 * Room that is already empty is room the insert does not have to make. The ripple is the overflow
 * alone — shifting by the whole bridge regardless would move the incoming clip a gap's worth too
 * far, so "put this between those two" silently became "put this after that one and shove
 * everything right" whenever the two were not already touching.
 */
describe("placing a bridge between two clips", () => {
  it("pushes by the full length when the clips already touch", () => {
    expect(planBridgePlacement(0, 5.167)).toMatchObject({ rippleSeconds: 5.167, fillsGap: false });
  });

  it("pushes by the overflow only when there is a gap to absorb", () => {
    const plan = planBridgePlacement(4.833, 5.167);
    expect(plan.rippleSeconds).toBeCloseTo(0.334, 3);
  });

  it("moves nothing when the bridge is shorter than the gap", () => {
    expect(planBridgePlacement(10, 5.167).rippleSeconds).toBe(0);
  });

  it("recognises an exact fill, to within half a frame", () => {
    expect(planBridgePlacement(5.167, 5.167).fillsGap).toBe(true);
    expect(planBridgePlacement(5.167, 5.18).fillsGap).toBe(true);
    expect(planBridgePlacement(5.167, 6).fillsGap).toBe(false);
  });

  it("never treats a negative gap as room", () => {
    // Overlapping clips are not a licence to shrink the ripple.
    expect(planBridgePlacement(-2, 5).rippleSeconds).toBe(5);
  });
});


/**
 * The 2026-09-04 report: "pictures show up blank in the asset preview in the video editor".
 *
 * Thumbnails were made by seeking to a timestamp and grabbing a frame, which is right for video and
 * meaningless for a picture: a one-frame file has nothing at 0.2 seconds, so the extractor wrote an
 * empty file and the tile stayed blank. Nothing surfaced it, because "no thumbnail yet" is a normal
 * state while one is being made.
 *
 * The fix hangs on telling a still from a moving picture, and `kind` cannot do it -- "generated"
 * covers everything a model produced, video included. The file extension can.
 */
describe("telling a still from a moving picture", () => {
  it("accepts a plain image", () => {
    expect(isStillAsset({ kind: "image", path: "C:/pics/plate.png" })).toBe(true);
  });

  it("accepts a generated still, which `kind` alone would call video", () => {
    expect(isStillAsset({ kind: "generated", path: "C:/out/zit-mushroom.png" })).toBe(true);
  });

  it("rejects a generated video, which `kind` alone would call a picture", () => {
    expect(isStillAsset({ kind: "generated", path: "C:/out/h3-shot.mp4" })).toBe(false);
  });

  it("rejects footage and audio", () => {
    expect(isStillAsset({ kind: "video", path: "C:/in/take-01.mov" })).toBe(false);
    expect(isStillAsset({ kind: "audio", path: "C:/in/voice.wav" })).toBe(false);
  });

  it("rejects an audio file mis-tagged as an image by an older importer", () => {
    expect(isStillAsset({ kind: "image", path: "C:/in/voice.wav" })).toBe(false);
  });
});
