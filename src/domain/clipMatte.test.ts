import { describe, expect, it } from "vitest";
import {
  assistPoints, defaultClipMatte, matteBlocker, matteIsShort, matteRange, programmeMatte, promptsFor,
  type ClipMatte, type ClipMattePoint,
} from "./clipMatte";
import { createProject } from "./project";
import { buildProgramme, programmeDocument, programmePick } from "./programme";
import { fromSeconds, rational } from "./time";
import type { MediaAsset, TimelineClip } from "./types";
import type { TrackSample } from "./photoDocument";

const video = (over: Partial<MediaAsset> = {}): MediaAsset => ({
  id: "a", name: "a.mp4", path: "C:/media/a.mp4", kind: "video", duration: fromSeconds(10),
  width: 1920, height: 1080, frameRate: rational(24), ...over,
});
const range = (sourceIn: number, sourceOut: number, asset = video()) =>
  matteRange({ sourceIn: fromSeconds(sourceIn), sourceOut: fromSeconds(sourceOut) }, asset);

const made = (over: Partial<ClipMatte> = {}): ClipMatte => ({
  ...defaultClipMatte(), ...over,
  cache: { cacheId: "c", cachePath: "C:/p/.aiv/cache/masks/c", frameCount: 48, first: 24, width: 1920, height: 1080, fps: 24, method: "birefnet", sourceIn: 1, sourceOut: 3, createdAt: "" },
});

describe("matteRange", () => {
  it("covers the clip's source frames and seeks half a frame early", () => {
    const r = range(1, 3);
    expect(r).toMatchObject({ still: false, first: 24, frameCount: 48 });
    expect(r.startTime).toBeCloseTo(23.5 / 24);
    expect(Math.round((r.endTime - r.startTime) * 24)).toBe(48);
    // An in-point between frames rounds to the frame the programme shows there.
    expect(range(1.03, 2).first).toBe(25);
    // Never past the file.
    expect(range(9.5, 12).frameCount).toBe(12);
  });

  it("is one frame for a still", () => {
    expect(range(0, 5, video({ kind: "image" }))).toMatchObject({ still: true, first: 0, frameCount: 1, startTime: 0, endTime: 0 });
  });

  it("says what stops a matte being made", () => {
    const sam2 = { ...defaultClipMatte(), method: "sam2" as const };
    expect(matteBlocker(sam2, range(0, 2))).toMatch(/Pick the subject/);
    expect(matteBlocker({ ...sam2, points: [{ frame: 0, x: 0.5, y: 0.5, include: true }] }, range(0, 2))).toBeUndefined();
    expect(matteBlocker({ ...sam2, points: [{ frame: 0, x: 0.5, y: 0.5, include: true }] }, matteRange({ sourceIn: fromSeconds(0), sourceOut: fromSeconds(60) }, video({ duration: fromSeconds(90) })))).toMatch(/900/);
    expect(matteBlocker(defaultClipMatte(), range(0, 2))).toBeUndefined();
  });
});

describe("points", () => {
  it("become range-relative prompts, and ones outside the range are dropped", () => {
    const points: ClipMattePoint[] = [
      { frame: 30, x: 0.5, y: 0.4, include: true }, { frame: 10, x: 0.1, y: 0.1, include: false }, { frame: 71, x: 1.2, y: 0.5, include: false },
    ];
    expect(promptsFor(points, range(1, 3))).toEqual([
      { frame: 6, x: 0.5, y: 0.4, include: true }, { frame: 47, x: 1, y: 0.5, include: false },
    ]);
  });

  it("track assist re-prompts every few frames where the followed point is visible", () => {
    const r = range(1, 3); // frames 24..71
    const picked: ClipMattePoint[] = [{ frame: 30, x: 0.5, y: 0.5, include: true }, { frame: 30, x: 0.1, y: 0.9, include: false }];
    const moving = (seenFrom: number): TrackSample[] => Array.from({ length: 48 }, (_, k) => [960 + k * 10, 540, (24 + k >= seenFrom ? 1 : 0) as 0 | 1]);
    const still = (): TrackSample[] => Array.from({ length: 48 }, () => [192, 972, 1] as TrackSample);
    const out = assistPoints(picked, [
      { point: picked[0], first: 24, samples: moving(40) },
      { point: picked[1], first: 24, samples: still() },
    ], { width: 1920, height: 1080 }, r, 12);
    expect(out.slice(0, 2)).toEqual(picked);
    const added = out.slice(2);
    // Frames 24, 36, 48, 60: the include point is unseen at 24 and 36, so only the exclude point
    // would be there -- those frames are left out.
    expect([...new Set(added.map((point) => point.frame))]).toEqual([48, 60]);
    const at48 = added.filter((point) => point.frame === 48);
    expect(at48).toContainEqual({ frame: 48, x: (960 + 24 * 10) / 1920, y: 0.5, include: true });
    expect(at48).toContainEqual({ frame: 48, x: 0.1, y: 0.9, include: false });
  });

  it("track assist thins evenly to the point limit", () => {
    const r = range(0, 10); // 240 frames
    const picked: ClipMattePoint[] = [{ frame: 0, x: 0.5, y: 0.5, include: true }];
    const samples: TrackSample[] = Array.from({ length: 240 }, () => [960, 540, 1]);
    const out = assistPoints(picked, [{ point: picked[0], first: 0, samples }], { width: 1920, height: 1080 }, r, 2, 40);
    expect(out.length).toBeLessThanOrEqual(40);
    expect(out.length).toBeGreaterThan(30);
    const frames = out.slice(1).map((point) => point.frame);
    expect(frames[frames.length - 1]).toBeGreaterThan(200);
  });
});

describe("the programme's matte", () => {
  const clip = (over: Partial<TimelineClip> = {}): TimelineClip => ({
    id: "A", assetId: "a", name: "A", trackId: "v1", start: fromSeconds(0), sourceIn: fromSeconds(1),
    sourceOut: fromSeconds(3), enabled: true, color: "#888", opacity: 1, volumeDb: 0, speed: [], ...over,
  });
  const programmeOf = (clips: TimelineClip[], overlays: TimelineClip[] = [], assets = [video()]) => {
    const project = createProject("matte");
    project.assets = assets;
    project.sequences[0].tracks.find((track) => track.id === "v1")!.clips = clips;
    project.sequences[0].tracks.find((track) => track.id === "v2")!.clips = overlays;
    return buildProgramme(project)!;
  };

  it("is drawn only when enabled and made", () => {
    expect(programmeMatte(undefined)).toBeUndefined();
    expect(programmeMatte({ ...defaultClipMatte() })).toBeUndefined();
    expect(programmeMatte(made({ enabled: false }))).toBeUndefined();
    expect(programmeMatte(made({ feather: 3, expand: -2, invert: true }))).toEqual({
      cachePath: "C:/p/.aiv/cache/masks/c", frameCount: 48, first: 24, feather: 3, expand: -2, invert: true,
    });
  });

  it("is the clip layer's mask, indexed by the source frame, with no pad", () => {
    const programme = programmeOf([clip({ matte: made({ feather: 2 }) })], [], [video({ width: 1440, height: 1080 })]);
    const document = programmeDocument(programme, 12);
    const layer = document.layers.find((item) => item.id === "v1:A")!;
    expect(document.layers.map((item) => item.id)).toEqual(["v1:A"]);
    const mask = layer.masks.find((item) => item.id === layer.maskId)!;
    expect(mask.stack).toHaveLength(1);
    expect(mask.stack![0]).toMatchObject({ mode: "add", feather: 2, invert: false });
    expect(mask.stack![0].source).toEqual({ kind: "sequence", cachePath: "C:/p/.aiv/cache/masks/c", frameCount: 48, first: 24, frame: 36 });
    // Without a matte the 4:3 clip is padded to the frame.
    const plain = programmeDocument(programmeOf([clip()], [], [video({ width: 1440, height: 1080 })]), 12);
    expect(plain.layers.map((item) => item.id)).toEqual(["v1:A:pad", "v1:A"]);
  });

  it("is intersected with a transition's wipe", () => {
    const programme = programmeOf([
      clip({ id: "B0", sourceIn: fromSeconds(4), sourceOut: fromSeconds(6) }),
      clip({ id: "A", start: fromSeconds(2), transitionIn: "wipeleft", transitionDuration: 1, matte: made() }),
    ]);
    const document = programmeDocument(programme, 36);
    const layer = document.layers.find((item) => item.id === "v1:A")!;
    const mask = layer.masks.find((item) => item.id === layer.maskId)!;
    expect(mask.stack!.map((entry) => [entry.source.kind, entry.mode])).toEqual([["sequence", "add"], ["path", "intersect"]]);
  });

  it("applies on overlay tracks inside the box, and feather follows the export size", () => {
    const programme = programmeOf([clip({ id: "BG" })], [clip({ id: "O", trackId: "v2", matte: made({ feather: 4 }) })]);
    const layer = programmeDocument(programme, 0).layers.find((item) => item.id === "ov:O")!;
    const mask = layer.masks.find((item) => item.id === layer.maskId)!;
    expect(mask.stack!.map((entry) => entry.source.kind)).toEqual(["sequence", "path"]);
    expect(mask.stack![0].source).toMatchObject({ frame: 24 });
    const project = createProject("matte");
    project.assets = [video()];
    project.sequences[0].tracks.find((track) => track.id === "v1")!.clips = [clip({ matte: made({ feather: 4 }) })];
    const half = buildProgramme(project, { width: 960, height: 540 })!;
    const entry = programmeDocument(half, 0).layers[0].masks[0].stack![0];
    expect(entry.feather).toBe(2);
  });

  it("maps a monitor click into the clip's source, through conform and transform", () => {
    // A 4:3 clip contained in 16:9 (pillarboxed), zoomed 200% about its centre.
    const programme = programmeOf([clip({ scale: 200 })], [], [video({ width: 1440, height: 1080 })]);
    // The frame centre is the source centre.
    expect(programmePick(programme, 12, "A", [0.5, 0.5])).toMatchObject({ x: 0.5, y: 0.5, frame: 36 });
    // At 200% the frame's left quarter point is the source's... contained width 1440/1920 = 0.75
    // of the frame, doubled = 1.5 frames wide: x = 0.25 of the frame is 0.5 - 0.25 / 1.5.
    const pick = programmePick(programme, 12, "A", [0.25, 0.5])!;
    expect(pick.x).toBeCloseTo(0.5 - 0.25 / 1.5, 4);
    // Off the clip.
    const small = programmeOf([clip({ scale: 50 })], [], [video({ width: 1440, height: 1080 })]);
    expect(programmePick(small, 12, "A", [0.05, 0.5])).toBeUndefined();
    expect(programmePick(small, 12, "nope", [0.5, 0.5])).toBeUndefined();
  });

  it("knows when a trim has left the matte short", () => {
    const c = clip({ matte: made() });
    expect(matteIsShort(c.matte, c)).toBe(false);
    expect(matteIsShort(c.matte, { ...c, sourceOut: fromSeconds(4) })).toBe(true);
    expect(matteIsShort(c.matte, { ...c, sourceIn: fromSeconds(2) })).toBe(false);
  });
});
