import { describe, expect, it } from "vitest";
import { createLayer, createPathLayer, createPhotoDocument, layersOf, type PhotoDocument, type PhotoLayer } from "./photoDocument";
import { expandRepeats, repeatPlacements } from "./photoRepeat";

const canvas = { width: 1000, height: 500 };
const asset = { id: "a1", name: "Star", path: "C:/s.png" };
const doc = (layers: PhotoLayer[]): PhotoDocument => ({ ...createPhotoDocument("a", "C:/a.png"), nodes: [], masks: [], layers });
const close = (value: number, expected: number) => expect(value).toBeCloseTo(expected, 9);

describe("repeat (VFX Phase 6)", () => {
  it("lays out a brick grid in canvas pixels, with scale and opacity steps", () => {
    const layer = createLayer(asset, { x: 0.1, y: 0.2, scale: 2, opacity: 1,
      repeat: { kind: "grid", columns: 3, rows: 2, gapX: 100, gapY: 50, shift: 0.5, scaleStep: 0.5, opacityStep: 0.5 } });
    const at = repeatPlacements(doc([layer]), layer, canvas);
    expect(at).toHaveLength(6);
    expect(at.map((p) => [Math.round(p.x * 1000), Math.round(p.y * 500)])).toEqual([
      [100, 100], [200, 100], [300, 100], [150, 150], [250, 150], [350, 150],
    ]);
    expect(at.map((p) => p.scale)).toEqual([2, 1, 0.5, 0.25, 0.125, 0.0625]);
    expect(at[2].opacity).toBe(0.25);
  });

  it("puts copies on a ring through the layer, turned with it", () => {
    const layer = createLayer(asset, { x: 0.5, y: 0.5, rotation: 10, repeat: { kind: "radial", count: 4, radius: 100, startAngle: 0, rotate: true } });
    const at = repeatPlacements(doc([layer]), layer, canvas);
    // Centre 100 px left of the layer: (400, 250). Copies at 0, 90, 180, 270 degrees (y down).
    const px = at.map((p) => [p.x * 1000, p.y * 500]);
    close(px[0][0], 500); close(px[0][1], 250);
    close(px[1][0], 400); close(px[1][1], 350);
    close(px[2][0], 300); close(px[2][1], 250);
    close(px[3][0], 400); close(px[3][1], 150);
    expect(at.map((p) => p.rotation)).toEqual([10, 100, 190, 280]);
    const arc = repeatPlacements(doc([layer]), { ...layer, repeat: { kind: "radial", count: 3, radius: 100, sweep: 180, rotate: false } }, canvas);
    close(arc[2].x * 1000, 300);
    expect(arc[2].rotation).toBe(10);
  });

  it("spreads copies along another layer's path, in canvas pixels", () => {
    // A horizontal 400 px line centred on (500, 250), turned 90 degrees by its layer: vertical.
    const guide = { ...createPathLayer({ subpaths: [{ closed: false, nodes: [{ x: -200, y: 0 }, { x: 200, y: 0 }] }] }, 0.5, 0.5), rotation: 90 };
    const layer = createLayer(asset, { repeat: { kind: "path", layerId: guide.id, count: 5, rotate: true } });
    const at = repeatPlacements(doc([guide, layer]), layer, canvas);
    const px = at.map((p) => [p.x * 1000, p.y * 500]);
    [50, 150, 250, 350, 450].forEach((y, k) => { close(px[k][0], 500); close(px[k][1], y); });
    expect(at.every((p) => Math.abs(p.rotation - 90) < 1e-9)).toBe(true);
    // A missing guide leaves the one layer where it is.
    const lost = repeatPlacements(doc([layer]), layer, canvas);
    expect(lost).toHaveLength(1);
  });

  it("expands into ordinary layers, copy 0 keeping the id", () => {
    const layer = createLayer(asset, { repeat: { kind: "grid", columns: 2, rows: 1, gapX: 10, gapY: 0 } });
    const other = createLayer({ ...asset, id: "b" });
    const document = doc([other, layer]);
    const expanded = layersOf(expandRepeats(document, canvas));
    expect(expanded.map((item) => item.id)).toEqual([other.id, layer.id, `${layer.id}~1`]);
    expect(expanded.every((item) => item.repeat === undefined)).toBe(true);
    expect(expandRepeats(doc([other]), canvas)).toEqual(doc([other]));
  });
});
