/** VFX Phase 2 panels render without throwing, for a mask with every kind of stack entry. */
import { describe, expect, it } from "vitest";
import { renderToString } from "react-dom/server";
import {
  createLayer, createLumaKeyNode, createMask, createMaskEntry, createPhotoDocument,
} from "../../domain/photoDocument";
import { LayerEditor, LayerList, LumaKeyEditor, MaskEditor } from "./PhotoPanels";

describe("Phase 2 panels", () => {
  const plate = createLayer({ id: "p", name: "Plate", path: "p.png" });
  const matte = createLayer({ id: "m", name: "Matte", path: "m.png" }, { visible: false });
  const mask = {
    ...createMask("stack"),
    stack: [
      createMaskEntry({ kind: "paint" }),
      createMaskEntry({ kind: "shape", shape: { kind: "colour", r: 1, g: 2, b: 3, tolerance: 10, softness: 1 } }),
      createMaskEntry({ kind: "file", path: "C:\\\\mattes\\\\a.png" }, { mode: "subtract" }),
      createMaskEntry({ kind: "sequence", cachePath: "C:/c", frameCount: 5 }),
      createMaskEntry({ kind: "layer", layerId: matte.id, channel: "luma" }, { invert: true }),
    ],
  };
  const cutout = createLayer({ id: "c", name: "Cutout", path: "c.png" }, { masks: [mask], maskId: mask.id, clip: true });
  const document = { ...createPhotoDocument("p", "p.png"), layers: [plate, matte, cutout], masks: [mask] };
  const noop = () => undefined;

  it("renders the mask stack with each entry selected", () => {
    for (const entry of [undefined, ...mask.stack]) {
      const html = renderToString(<MaskEditor document={document} activeMaskId={mask.id} brush={{ radius: 0.05, hardness: 0.5, erase: false }}
        selectReady selecting={false} points={0} picking={false} patch={noop}
        onActivate={noop} onCreateMask={() => undefined} onPickColour={noop} onBrush={noop}
        onStartSelect={noop} onRunSelect={noop} onClearPoints={noop}
        layers={[plate, matte, cutout]} ownLayerId={cutout.id} activeEntryId={entry?.id} onActivateEntry={noop} onImportMatte={noop} />);
      expect(html).toContain("Stack");
      expect(html).toContain("Luma of Matte");
    }
  });

  it("renders the layer's mask, clip and matte controls, and the luma key", () => {
    const html = renderToString(<LayerEditor layer={cutout} layers={[plate, matte, cutout]} matching={false}
      onChange={noop} onMatchColour={noop} onTrackMatte={noop} />);
    expect(html).toContain("Clipped to below");
    expect(html).toContain("Track matte");
    expect(renderToString(<LayerList document={document} activeId={cutout.id} onSelect={noop} onChange={noop} />)).toContain("clipped");
    expect(renderToString(<LumaKeyEditor node={createLumaKeyNode()} onChange={noop} />)).toContain("Softness");
  });
});
