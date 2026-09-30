import { beforeEach, describe, expect, it } from "vitest";
import { buildTimelineExportPayload } from "../features/export/request";
import { useEditor } from "../store";
import { createProject } from "./project";
import { fromSeconds } from "./time";
import type { MediaAsset } from "./types";

const generatedSfx = (): MediaAsset => ({
  id: crypto.randomUUID(), name: "door-slam.wav", path: "C:/project/door-slam.wav", kind: "audio",
  duration: fromSeconds(2), hasAudio: true,
  origin: {
    operation: "moss-sfx", model: "MOSS-SoundEffect-v2", prompt: "metal door slam", seed: 42,
    createdAt: "2026-09-12T00:00:00.000Z",
  },
});

describe("SFX timeline placement", () => {
  beforeEach(() => {
    useEditor.setState({ project: createProject("SFX test"), past: [], future: [], playhead: 0 });
  });

  it("registers a generated WAV and places it on the dedicated SFX track", () => {
    const asset = generatedSfx();
    useEditor.getState().addSfxToTimeline(asset, 1.25, {
      gainDb: -3, pan: 0.4, fadeInSeconds: 0.1, fadeOutSeconds: 0.2,
      trimInSeconds: 0.05, trimOutSeconds: 0.15, preserveTail: true,
    });
    const project = useEditor.getState().project;
    const sequence = project.sequences[0];
    const track = sequence.tracks.find((item) => item.id === "a2");
    expect(project.assets.find((item) => item.id === asset.id)?.origin?.operation).toBe("moss-sfx");
    expect(track?.name).toBe("A2 · SFX");
    expect(track?.clips).toHaveLength(1);
    expect(track?.clips[0]).toMatchObject({ volumeDb: -3, audio: {
      pan: 0.4, fadeInSeconds: 0.1, fadeOutSeconds: 0.2, preserveTail: true,
    } });
  });

  it("threads SFX gain, pan, and fades into final export", () => {
    const asset = generatedSfx();
    useEditor.getState().addSfxToTimeline(asset, 0.5, {
      gainDb: -4.5, pan: -0.25, fadeInSeconds: 0.12, fadeOutSeconds: 0.3,
      trimInSeconds: 0, trimOutSeconds: 0, preserveTail: true,
    });
    const payload = buildTimelineExportPayload(useEditor.getState().project, 1);
    expect(payload.audioClips).toEqual([expect.objectContaining({
      path: asset.path, startSeconds: 0.5, volumeDb: -4.5, pan: -0.25,
      fadeInSeconds: 0.12, fadeOutSeconds: 0.3,
    })]);
  });
});
