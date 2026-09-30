/** VFX Phase 3 panels render without throwing. */
import { describe, expect, it } from "vitest";
import { renderToString } from "react-dom/server";
import { DEFAULT_MATRIX_RAIN, createAlphaCurveNode, type LayerMedia } from "../../domain/photoDocument";
import { AlphaCurveEditor, ElementsPanel, MatrixRainEditor, MediaEditor, TimeBar, timecode } from "./PhotoPanels";

const noop = () => undefined;
const media: LayerMedia = {
  kind: "frames", path: "C:/el/a", frameCount: 81, fps: 16, width: 832, height: 480,
  start: 4, inPoint: 2, outPoint: 70, loop: true, crossfade: 6, elementId: "a",
};

describe("Phase 3 panels", () => {
  it("formats timecode at the composition rate", () => {
    expect(timecode(0, 24)).toBe("00:00+00");
    expect(timecode(24 * 61 + 5, 24)).toBe("01:01+05");
    expect(timecode(30, 29.97)).toBe("00:01+00");
  });

  it("renders the time bar, timing, rain, alpha curve and elements", () => {
    expect(renderToString(<TimeBar frame={12} duration={81} rate={16} playing={false} useProxy={false} hasProxy waiting={2}
      onFrame={noop} onPlay={noop} onProxy={noop} onDuration={noop} />)).toContain("Frame 12 of 81");
    expect(renderToString(<MediaEditor media={media} rate={24} onChange={noop} onSaveLoop={noop} />)).toContain("Save loop to element");
    expect(renderToString(<MatrixRainEditor generator={DEFAULT_MATRIX_RAIN} onChange={noop} />)).toContain("Loop length");
    expect(renderToString(<AlphaCurveEditor node={createAlphaCurveNode()} onChange={noop} />)).toContain("Gamma");
    const html = renderToString(<ElementsPanel elements={[{ dir: "C:/el/a", manifest: {
      version: 1, id: "a", name: "Smoke", frameCount: 81, fps: 16, width: 832, height: 480, pattern: "frame_%04d.png",
      loopIn: 0, crossfade: 0, source: "wan-alpha", createdAt: "2026-09-28T00:00:00Z", proxyError: "no ffmpeg",
    } }]} onRefresh={noop} onAdd={noop} onImport={noop} onGenerate={noop} onBuildProxy={noop} />);
    expect(html).toContain("Smoke");
    expect(html).toContain("no proxy (ffmpeg failed)");
    expect(html).toContain("Generate element");
  });
});
