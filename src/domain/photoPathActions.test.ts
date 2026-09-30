import { describe, expect, it } from "vitest";
import { createLayer, createPathLayer, type PhotoLayer } from "./photoDocument";
import { pathBounds, type Subpath } from "./photoPath";
import { filledArea, pathInCanvas } from "./photoPathOps";
import {
  blendPathLayers, clipAboveToPath, combinePathLayers, envelopeFor, maskBelowWithPath, outlineLayerStroke, pathLayerBelow,
} from "./photoPathActions";

const canvas = { width: 1000, height: 800 };
const square = (x: number, y: number, size: number): Subpath => ({ closed: true, nodes: [[x, y], [x + size, y], [x + size, y + size], [x, y + size]].map(([px, py]) => ({ x: px, y: py })) });
const red = { kind: "solid" as const, color: "#ff0000" }, blue = { kind: "solid" as const, color: "#0000ff" };

/** A 100 px square centred on the layer, which sits at canvas pixel (cx, cy). */
const squareLayer = (cx: number, cy: number, over: Partial<PhotoLayer> = {}, fill = red) =>
  ({ ...createPathLayer({ subpaths: [square(-50, -50, 100)], fill, strokeWidth: 4 }, cx / canvas.width, cy / canvas.height), ...over });

describe("path layer actions (VFX Phase 6)", () => {
  it("combines with the path layer below, in canvas space, into the lower layer's placement", () => {
    const plate = createLayer({ id: "p", name: "Plate", path: "C:/p.png" });
    const bottom = squareLayer(400, 400, { scale: 2 }); // 200 px square on the canvas: 300..500
    const top = squareLayer(500, 500, {}, blue);        // 100 px square: 450..550
    const layers = [plate, bottom, top];
    expect(pathLayerBelow(layers, top.id)?.id).toBe(bottom.id);
    const united = combinePathLayers(layers, top.id, "unite", canvas)!;
    expect(united.map((layer) => layer.id)).toEqual([plate.id, bottom.id]);
    const kept = united[1];
    const inCanvas = pathInCanvas(kept, kept.path!, canvas.width, canvas.height);
    expect(filledArea(inCanvas)).toBeCloseTo(200 * 200 + 100 * 100 - 50 * 50, 3);
    // Unite takes the top's paint; its 4 px stroke is 2 units in the kept layer (scale 2).
    expect(kept.path!.fill).toEqual(blue);
    expect(kept.path!.strokeWidth).toBeCloseTo(2, 9);
    const minus = combinePathLayers(layers, top.id, "subtract", canvas)![1];
    expect(minus.path!.fill).toEqual(red);
    expect(filledArea(pathInCanvas(minus, minus.path!, canvas.width, canvas.height))).toBeCloseTo(40000 - 2500, 3);
    const compound = combinePathLayers(layers, top.id, "compound", canvas)![1];
    expect(compound.path!.subpaths).toHaveLength(2);
    expect(combinePathLayers([plate, top], top.id, "unite", canvas)).toBeUndefined();
  });

  it("outlines a stroke into its own layer above", () => {
    const layer = squareLayer(500, 400);
    const out = outlineLayerStroke([layer], layer.id)!;
    expect(out).toHaveLength(2);
    expect(out[0].path!.strokeWidth).toBe(0);
    expect(out[1].path!.fill).toEqual(layer.path!.stroke);
    expect(filledArea(out[1].path!)).toBeCloseTo(104 * 104 - 96 * 96, 3);
  });

  it("clips the layer above and masks the layer below", () => {
    const plate = createLayer({ id: "p", name: "Plate", path: "C:/p.png" });
    const shape = squareLayer(500, 400);
    const photo = createLayer({ id: "q", name: "Photo", path: "C:/q.png" });
    expect(clipAboveToPath([plate, shape, photo], shape.id)![2].clip).toBe(true);
    expect(clipAboveToPath([plate, shape], shape.id)).toBeUndefined();
    const masked = maskBelowWithPath([plate, shape], shape.id, canvas)!;
    const mask = masked[0].masks[0];
    expect(masked[0].maskId).toBe(mask.id);
    expect(masked[1].visible).toBe(false);
    const entry = mask.stack![0];
    expect(entry.source.kind).toBe("path");
    const b = pathBounds(entry.source.kind === "path" ? entry.source.path : ({} as never));
    expect([b.minX, b.minY, b.maxX, b.maxY].map((v) => Math.round(v * 1000) / 1000)).toEqual([0.45, 0.438, 0.55, 0.563]);
  });

  it("blends two layers where they sit, inserting the steps above the lower one", () => {
    const a = squareLayer(200, 400, {}, red);
    const b = squareLayer(800, 400, { rotation: 90 }, blue);
    const top = createLayer({ id: "t", name: "Top", path: "C:/t.png" });
    const out = blendPathLayers([a, b, top], b.id, a.id, 3, canvas)!;
    expect(out.map((layer) => layer.name)).toEqual(["Path", "Path → Path 1", "Path → Path 2", "Path → Path 3", "Path", "Top"]);
    expect(out.slice(1, 4).map((layer) => Math.round(layer.x * canvas.width))).toEqual([350, 500, 650]);
    expect(out[2].path!.fill).toEqual({ kind: "solid", color: "#bc00bc" });
  });

  it("gives a path an identity envelope over its box", () => {
    const layer = squareLayer(500, 400);
    const shaped = envelopeFor(layer.path!);
    expect(shaped.envelope!.points).toHaveLength(16);
    expect(shaped.envelope!.rect).toEqual({ x: -51, y: -51, width: 102, height: 102 });
    expect(pathBounds(shaped).maxX).toBeCloseTo(50, 9);
  });
});
