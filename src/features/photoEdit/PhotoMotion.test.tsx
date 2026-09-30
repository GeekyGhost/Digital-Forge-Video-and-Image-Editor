/** VFX Phase 4 panels render without throwing, and say what they must (the licence, the attach state). */
import { describe, expect, it } from "vitest";
import { renderToString } from "react-dom/server";
import {
  createDepthOcclusionMask, createEmitterLayer, createMediaLayer, createTrailNode, type LayerMedia, type PointTrack,
} from "../../domain/photoDocument";
import { EmitterEditor, TracksPanel, TrailEditor } from "./PhotoMotion";

const noop = () => undefined;
/** Server rendering puts `<!-- -->` between adjacent text parts; the text a user reads has none. */
const render = (element: Parameters<typeof renderToString>[0]) => renderToString(element).replace(/<!-- -->/g, "");
const clip: LayerMedia = {
  kind: "video", path: "C:/m/smoker.mp4", frameCount: 96, fps: 24, width: 1920, height: 1080,
  start: 0, inPoint: 0, outPoint: 96, loop: false, crossfade: 0, depth: { cachePath: "C:/c/d", frameCount: 96 },
};
const element: LayerMedia = { kind: "frames", path: "C:/el/smoke", frameCount: 33, fps: 16, width: 832, height: 480, start: 0, inPoint: 0, outPoint: 33, loop: false, crossfade: 0 };
const track: PointTrack = { id: "t1", name: "Tip", layerId: "plate", first: 0, source: "cotracker3", samples: [[1, 2, 1], [3, 4, 0]] };

describe("Phase 4 panels", () => {
  it("renders the tracks panel with the licence, the list and the attach controls", () => {
    const plate = { ...createMediaLayer("Plate", clip), id: "plate" };
    const smoke = createMediaLayer("Smoke", element, { attach: { trackId: "t1", anchorFrame: 5, lag: 2 } });
    const html = render(<TracksPanel tracks={[track]} plate={plate} activeLayer={smoke} frame={7} engine="cotracker3"
      motion={{ ready: true, cotracker: true, depth: true, cotrackerLicence: "CoTracker3: CC-BY-NC 4.0", depthLicence: "Apache-2.0", detail: "" }}
      tracking={false} sam2Sources={[{ label: "Hand", cachePath: "C:/c/m", frameCount: 96 }]} occluded={false}
      onEngine={noop} onStartTrack={noop} onSam2Centroid={noop} onRename={noop} onDelete={noop} onAttach={noop}
      onComputeDepth={noop} onOcclude={noop} />);
    expect(html).toContain("CC-BY-NC");
    expect(html).toContain("Track a point + attach the selected layer");
    expect(html).toContain("seen 1/2");
    expect(html).toContain("Smoke follows");
    expect(html).toContain("2 frames late");
    expect(html).toContain("Occlude by depth");
    expect(html).toContain("Recompute depth");
  });

  it("says what is missing without a plate or a runtime", () => {
    const html = render(<TracksPanel tracks={[]} frame={0} engine="lucas-kanade" tracking={false} sam2Sources={[]} occluded={false}
      onEngine={noop} onStartTrack={noop} onSam2Centroid={noop} onRename={noop} onDelete={noop} onAttach={noop}
      onComputeDepth={noop} onOcclude={noop} />);
    expect(html).toContain("Add a video");
    expect(html).toContain("No tracks yet");
  });

  it("renders the trail and emitter editors", () => {
    expect(render(<TrailEditor node={createTrailNode("t1")} tracks={[track]} onChange={noop} />)).toContain("Lag at the far end");
    expect(render(<TrailEditor node={createTrailNode("gone")} tracks={[track]} onChange={noop} />)).toContain("Missing track");
    const emitter = createEmitterLayer(element, { trackId: "t1" }).generator;
    if (emitter?.kind !== "emitter") throw new Error("not an emitter");
    const html = render(<EmitterEditor generator={emitter} tracks={[track]} onChange={noop} />);
    expect(html).toContain("Lifetime");
    expect(html).not.toContain("Point X");
    expect(render(<EmitterEditor generator={{ ...emitter, trackId: undefined }} tracks={[]} onChange={noop} />)).toContain("Point X");
  });

  it("makes an occlusion mask that covers everything until the depth says otherwise", () => {
    const mask = createDepthOcclusionMask("plate", 40, 4);
    expect(mask.invert).toBe(true);
    expect(mask.stack?.[0]).toMatchObject({ mode: "subtract", source: { kind: "depth", layerId: "plate", plane: 40, softness: 4 } });
  });
});
