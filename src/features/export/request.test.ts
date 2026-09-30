import { describe, expect, it } from "vitest";
import { DEFAULT_MASK_VFX_RECIPE } from "../../domain/maskEffects";
import { createProject } from "../../domain/project";
import { fromSeconds } from "../../domain/time";
import { useEditor } from "../../store";
import { buildTimelineExportPayload, sliceTimelineExportPayload } from "./request";

describe("timeline rendered preview windows", () => {
  it("shifts programme time and source time without mutating the master payload", () => {
    const clip = {
      path: "shot.mp4", startSeconds: 4, sourceInSeconds: 10, durationSeconds: 8,
      sourceDurationSeconds: 16, speedRate: 2, hasAudio: true, positionX: 960, positionY: 540,
      scalePercent: 100, rotationDegrees: 0, opacity: 1, volumeDb: 0,
      transitionKind: "fade", transitionDuration: 0.5, transitionOutKind: "dissolve",
      transitionOutDuration: 0.5, effects: {}, conform: { fit: "contain", pad: "black" }, headHandleSeconds: 0.5,
    };
    const payload = {
      sequence: undefined, clips: [clip], graphics: [], audioClips: [], overlayClips: [],
      upscaleFactor: 1, rippleSeconds: 0,
    } as unknown as ReturnType<typeof buildTimelineExportPayload>;
    const preview = sliceTimelineExportPayload(payload, 6, 3);
    expect(preview.clips).toHaveLength(1);
    expect(preview.clips[0]).toMatchObject({
      startSeconds: 0, sourceInSeconds: 14, sourceDurationSeconds: 6, durationSeconds: 3,
      transitionKind: "none", transitionOutKind: "none", headHandleSeconds: 0,
    });
    expect(payload.clips[0]).toBe(clip);
  });

  it("drops items outside the preview window", () => {
    const payload = {
      sequence: undefined,
      clips: [{ startSeconds: 20, durationSeconds: 2 }], graphics: [{ startSeconds: 2, durationSeconds: 1 }],
      audioClips: [{ startSeconds: 15, durationSeconds: 1 }], overlayClips: [], upscaleFactor: 1, rippleSeconds: 0,
    } as unknown as ReturnType<typeof buildTimelineExportPayload>;
    const preview = sliceTimelineExportPayload(payload, 4, 5);
    expect(preview.clips).toEqual([]);
    expect(preview.audioClips).toEqual([]);
    expect(preview.graphics).toEqual([]);
  });

  it("exports a cached VFX layer over the original clip without replacing its source", () => {
    useEditor.setState({ project: createProject("VFX export"), past: [], future: [], playhead: 0 });
    const asset = {
      id: "source", name: "source.mp4", path: "C:/media/source.mp4", kind: "video" as const,
      duration: fromSeconds(6), hasAudio: true,
    };
    useEditor.getState().addAssets([asset]);
    useEditor.getState().addAssetToTimelineAt(asset.id, 2, "v1");
    const clip = useEditor.getState().project.sequences[0].tracks.find((track) => track.id === "v1")!.clips[0];
    useEditor.getState().upsertClipVfxLayer(clip.id, {
      id: "vfx-1", name: "Anime Power Up", enabled: true, rangeStart: 1, rangeEnd: 3,
      recipe: DEFAULT_MASK_VFX_RECIPE, renderCachePath: "C:/cache/vfx.mp4",
      createdAt: "2026-09-15T00:00:00Z", updatedAt: "2026-09-15T00:00:00Z",
    });
    const payload = buildTimelineExportPayload(useEditor.getState().project, 1);
    expect(payload.clips.filter((item) => item.path === asset.path)).toHaveLength(2);
    expect(payload.clips).toContainEqual(expect.objectContaining({
      path: "C:/cache/vfx.mp4", startSeconds: 3, durationSeconds: 2,
      sourceInSeconds: 0, hasAudio: true, vfxLayerId: "vfx-1",
    }));
  });
});
