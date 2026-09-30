/**
 * VFX Phase 4b: the same editor on two surfaces. The Photo Edit tab is pictures and frames only; the
 * Clip Editor's VFX mode has every video tool and renders back to the timeline.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderToString } from "react-dom/server";

vi.mock("@tauri-apps/api/core", () => ({
  convertFileSrc: (path: string) => `asset://${path}`,
  invoke: () => Promise.reject(new Error("no backend in tests")),
}));
vi.mock("@tauri-apps/api/event", () => ({ listen: async () => () => undefined }));
vi.mock("@tauri-apps/plugin-dialog", () => ({ open: async () => null, save: async () => null }));

import { PhotoEdit } from "./PhotoEdit";
import { useEditor } from "../../store";
import { createProject } from "../../domain/project";
import { clipPlateDocument } from "../../domain/clipVfx";
import { createPhotoDocument, createMediaLayer, withLayers, layersOf } from "../../domain/photoDocument";
import { fromSeconds, rational } from "../../domain/time";
import type { MediaAsset, TimelineClip } from "../../domain/types";

/** Server rendering reads zustand's INITIAL state (its server snapshot), so a test state goes into both. */
const live = (partial: Partial<ReturnType<typeof useEditor.getState>>) => {
  useEditor.setState(partial);
  Object.assign(useEditor.getInitialState(), partial);
};
const render = (element: Parameters<typeof renderToString>[0]) => renderToString(element).replace(/<!-- -->/g, "");
const video: MediaAsset = {
  id: "v1", name: "smoker.mp4", path: "C:/m/smoker.mp4", kind: "video",
  duration: rational(10), width: 1920, height: 1080, frameRate: rational(24), hasAudio: true,
};
const still: MediaAsset = { id: "i1", name: "still.png", path: "C:/m/still.png", kind: "image", duration: rational(5), width: 800, height: 600 };
const clip = { id: "c1", name: "Smoker", assetId: "v1", trackId: "v1", sourceIn: fromSeconds(2), sourceOut: fromSeconds(8), start: fromSeconds(0) } as unknown as TimelineClip;

describe("Photo Edit surfaces", () => {
  beforeEach(() => {
    live({ project: { ...createProject("Surface test"), assets: [video] }, past: [], future: [], activeWorkspace: "photo" });
  });

  it("puts every video tool in the Clip Editor's VFX mode, with Render and no picture export", () => {
    const html = render(<PhotoEdit clipTarget={{ clip, asset: video, rangeStart: 1, rangeEnd: 3, onClose: () => undefined, modeSwitch: <b>MODE-SWITCH</b> }} />);
    expect(html).toContain("Clip VFX");
    expect(html).toContain("MODE-SWITCH");
    expect(html).toContain("48 frames at 24 fps");
    for (const card of ["Timing", "Tracks", "Elements", "Insert character"]) expect(html).toContain(`<strong>${card}</strong>`);
    expect(html).toContain("Render VFX");
    expect(html).toContain("Export spritesheet · 48 frames");
    expect(html).toContain("Whole clip");
    expect(html).toContain("1.00–3.00s");
    expect(html).not.toContain("Save to Project Media");
    expect(html).not.toContain("Back to Image Studio");
    // VFX Phase 6: vectors are pictures, so the pen is on both surfaces.
    expect(html).toContain("lucide-pen-tool");
  });

  it("keeps the Photo Edit tab to pictures: no video bin, elements, timing, tracks or render", () => {
    live({ project: { ...useEditor.getState().project, assets: [video] }, photoEditAssetId: undefined });
    const html = render(<PhotoEdit />);
    expect(html).toContain("Photo Edit");
    for (const gone of ["<strong>Timing</strong>", "<strong>Tracks</strong>", "<strong>Elements</strong>", "<strong>Insert character</strong>", "Animate and insert", "Export spritesheet", "Render VFX", "add as layer"]) {
      expect(html).not.toContain(gone);
    }
    expect(html).toContain("lives in the Clip Editor");
    expect(html).toContain("lucide-pen-tool");
  });

  it("shows an older picture with video layers as its first frame, and says where the tools went", () => {
    const base = withLayers(createPhotoDocument(still.id, still.path));
    const plate = createMediaLayer("Clip", { kind: "video", path: video.path, frameCount: 240, fps: 24, width: 1920, height: 1080 });
    const document = { ...base, layers: [...layersOf(base), plate], tracks: [] };
    live({
      project: { ...useEditor.getState().project, assets: [still, video], photoEdits: { [still.id]: document } },
      photoEditAssetId: still.id,
    });
    const html = render(<PhotoEdit />);
    expect(html).toContain("from before video VFX moved to the Clip Editor");
    expect(html).not.toContain("<strong>Timing</strong>");
    expect(html).not.toContain("photo-timebar");
    expect(html).toContain("<strong>Save and export</strong>");
  });

  it("reopens a clip's stored composition, and ignores one built on other footage", () => {
    const stored = { ...clipPlateDocument(clip, video, 0, 6)!, tracks: [] };
    live({ project: { ...useEditor.getState().project, clipVfx: { [clip.id]: stored } } });
    expect(render(<PhotoEdit clipTarget={{ clip, asset: video, rangeStart: 1, rangeEnd: 3, onClose: () => undefined }} />)).toContain("144 frames");
    const render2 = { ...video, id: "render" };
    expect(render(<PhotoEdit clipTarget={{ clip: { ...clip, assetId: "render" }, asset: render2, rangeStart: 1, rangeEnd: 3, onClose: () => undefined }} />)).toContain("48 frames");
  });
});
