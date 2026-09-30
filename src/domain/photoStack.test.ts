import { describe, expect, it } from "vitest";
import {
  createGradeNode, createMask, createPhotoDocument, createRelightNode, createRemoveNode,
  isUntouched, removalCacheKey, type PhotoDocument,
} from "./photoDocument";
import { rasterizeMask } from "./photoMask";
import { normalsFromDepth, applyRelight } from "./photoRelight";
import { NEUTRAL_RELIGHT } from "./photoDocument";
import { blend, emptyResources, renderDocument, resampleSingleChannel } from "./photoRender";

const document = (): PhotoDocument => createPhotoDocument("asset-1", "C:/pics/one.png");

const solidMask = (name = "Subject") => {
  const mask = createMask(name);
  mask.feather = 0;
  return mask;
};

/** A flat mid-grey picture, which makes any change obvious. */
const flat = (width: number, height: number, value = 128) => {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let index = 0; index < data.length; index += 4) {
    data[index] = data[index + 1] = data[index + 2] = value;
    data[index + 3] = 255;
  }
  return data;
};

describe("rasterizeMask", () => {
  it("covers everything when the mask is switched off, so the node still applies", () => {
    const mask = solidMask();
    mask.enabled = false;
    const coverage = rasterizeMask(mask, 8, 8);
    expect(Array.from(coverage).every((value) => value === 255)).toBe(true);
  });

  it("paints where the stroke is and nowhere else", () => {
    const mask = solidMask();
    mask.strokes.push({ points: [[0.5, 0.5]], radius: 0.1, hardness: 1, erase: false });
    const coverage = rasterizeMask(mask, 40, 40);
    expect(coverage[20 * 40 + 20]).toBe(255);
    expect(coverage[0]).toBe(0);
  });

  it("puts a normalised stroke in the right place on a non-square picture", () => {
    const mask = solidMask();
    mask.strokes.push({ points: [[0.25, 0.5]], radius: 0.05, hardness: 1, erase: false });
    const coverage = rasterizeMask(mask, 80, 40);
    expect(coverage[20 * 80 + 20]).toBe(255);
    expect(coverage[20 * 80 + 60]).toBe(0);
  });

  it("fills the gap between two far-apart samples, so a fast drag is not dotted", () => {
    const mask = solidMask();
    mask.strokes.push({ points: [[0.1, 0.5], [0.9, 0.5]], radius: 0.03, hardness: 1, erase: false });
    const coverage = rasterizeMask(mask, 100, 20);
    expect(coverage[10 * 100 + 50]).toBe(255);
  });

  it("erases what an earlier stroke painted", () => {
    const mask = solidMask();
    mask.strokes.push({ points: [[0.5, 0.5]], radius: 0.3, hardness: 1, erase: false });
    mask.strokes.push({ points: [[0.5, 0.5]], radius: 0.15, hardness: 1, erase: true });
    const coverage = rasterizeMask(mask, 40, 40);
    expect(coverage[20 * 40 + 20]).toBe(0);
    // Still painted in the ring between the two radii.
    expect(coverage[20 * 40 + 31]).toBeGreaterThan(200);
  });

  it("inverts", () => {
    const mask = solidMask();
    mask.invert = true;
    mask.strokes.push({ points: [[0.5, 0.5]], radius: 0.1, hardness: 1, erase: false });
    const coverage = rasterizeMask(mask, 40, 40);
    expect(coverage[20 * 40 + 20]).toBe(0);
    expect(coverage[0]).toBe(255);
  });

  it("feathers the edge instead of leaving it hard", () => {
    const hard = solidMask();
    hard.strokes.push({ points: [[0.5, 0.5]], radius: 0.25, hardness: 1, erase: false });
    const soft = solidMask();
    soft.feather = 4;
    soft.strokes.push({ points: [[0.5, 0.5]], radius: 0.25, hardness: 1, erase: false });
    const at = 20 * 40 + 30; // right on the rim
    const hardEdge = rasterizeMask(hard, 40, 40)[at];
    const softEdge = rasterizeMask(soft, 40, 40)[at];
    expect(hardEdge === 0 || hardEdge === 255).toBe(true);
    expect(softEdge).toBeGreaterThan(0);
    expect(softEdge).toBeLessThan(255);
  });

  it("starts from the model's mask and lets strokes correct it", () => {
    const mask = solidMask();
    const sam2 = new Uint8Array(40 * 40).fill(255);
    mask.strokes.push({ points: [[0.5, 0.5]], radius: 0.1, hardness: 1, erase: true });
    const coverage = rasterizeMask(mask, 40, 40, sam2);
    expect(coverage[0]).toBe(255);
    expect(coverage[20 * 40 + 20]).toBe(0);
  });
});

describe("normalsFromDepth", () => {
  it("faces the camera where the depth is flat", () => {
    const depth = new Uint8Array(16 * 16).fill(120);
    const normals = normalsFromDepth(depth, 16, 16);
    for (let index = 0; index < 16 * 16; index++) {
      expect(normals[index * 3]).toBeCloseTo(0, 5);
      expect(normals[index * 3 + 1]).toBeCloseTo(0, 5);
      expect(normals[index * 3 + 2]).toBeCloseTo(1, 5);
    }
  });

  it("tilts away from a rising edge", () => {
    const depth = new Uint8Array(8 * 8);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 8; x++) depth[y * 8 + x] = x * 30;
    const normals = normalsFromDepth(depth, 8, 8);
    // Depth rises to the right, so the surface leans left.
    expect(normals[(4 * 8 + 4) * 3]).toBeLessThan(0);
  });
});

describe("applyRelight", () => {
  it("lights the side the lamp is on", () => {
    // A ridge running down the middle, so the two halves face opposite ways.
    const width = 40, height = 8;
    const depth = new Uint8Array(width * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) depth[y * width + x] = x < 20 ? x * 6 : (39 - x) * 6;
    }
    const left = flat(width, height);
    const right = flat(width, height);
    applyRelight(left, width, height, depth, { ...NEUTRAL_RELIGHT, x: 0, ambient: 0.2, rim: 0 });
    applyRelight(right, width, height, depth, { ...NEUTRAL_RELIGHT, x: 1, ambient: 0.2, rim: 0 });
    const sample = (data: Uint8ClampedArray, x: number) => data[(4 * width + x) * 4];
    expect(sample(left, 8)).toBeGreaterThan(sample(right, 8));
    expect(sample(right, 32)).toBeGreaterThan(sample(left, 32));
  });

  it("does nothing when the depth map is the wrong size, rather than reading past it", () => {
    const data = flat(8, 8);
    const before = Array.from(data);
    applyRelight(data, 8, 8, new Uint8Array(9), NEUTRAL_RELIGHT);
    expect(Array.from(data)).toEqual(before);
  });
});

describe("renderDocument", () => {
  it("hands back the source untouched for a fresh document", () => {
    const base = flat(8, 8);
    const { pixels, report } = renderDocument(base, 8, 8, document(), emptyResources());
    expect(Array.from(pixels)).toEqual(Array.from(base));
    expect(report.skipped).toEqual([]);
  });

  it("skips a disabled node", () => {
    const doc = document();
    (doc.nodes[0] as { adjustments: { exposure: number } }).adjustments.exposure = 1;
    doc.nodes[0].enabled = false;
    const base = flat(8, 8);
    const { pixels } = renderDocument(base, 8, 8, doc, emptyResources());
    expect(Array.from(pixels)).toEqual(Array.from(base));
  });

  it("only changes the masked pixels", () => {
    const doc = document();
    const mask = solidMask();
    mask.strokes.push({ points: [[0.25, 0.5]], radius: 0.12, hardness: 1, erase: false });
    doc.masks.push(mask);
    doc.nodes[0].maskId = mask.id;
    (doc.nodes[0] as { adjustments: { exposure: number } }).adjustments.exposure = 2;

    const width = 40, height = 40;
    const resources = emptyResources();
    resources.masks.set(mask.id, rasterizeMask(mask, width, height));
    const base = flat(width, height);
    const { pixels } = renderDocument(base, width, height, doc, resources);
    expect(pixels[(20 * width + 10) * 4]).toBeGreaterThan(128);
    expect(pixels[(20 * width + 35) * 4]).toBe(128);
  });

  it("reports a relight node with no depth instead of drawing something wrong", () => {
    const doc = document();
    doc.nodes.push(createRelightNode());
    const { pixels, report } = renderDocument(flat(8, 8), 8, 8, doc, emptyResources());
    expect(report.skipped).toHaveLength(1);
    expect(report.skipped[0].reason).toContain("depth");
    expect(pixels[0]).toBe(128);
  });

  it("reports a removal that has not been filled yet", () => {
    const doc = document();
    const mask = solidMask("Sign");
    doc.masks.push(mask);
    doc.nodes.push(createRemoveNode(mask.id));
    const resources = emptyResources();
    resources.masks.set(mask.id, rasterizeMask(mask, 8, 8));
    const { report } = renderDocument(flat(8, 8), 8, 8, doc, resources);
    expect(report.skipped[0].reason).toContain("filled");
  });

  it("uses a filled removal through its mask", () => {
    const doc = document();
    const mask = solidMask("Sign");
    mask.strokes.push({ points: [[0.5, 0.5]], radius: 0.4, hardness: 1, erase: false });
    doc.masks.push(mask);
    const node = createRemoveNode(mask.id);
    doc.nodes.push(node);
    const resources = emptyResources();
    resources.masks.set(mask.id, rasterizeMask(mask, 20, 20));
    resources.removals.set(node.id, flat(20, 20, 250));
    const { pixels } = renderDocument(flat(20, 20), 20, 20, doc, resources);
    expect(pixels[(10 * 20 + 10) * 4]).toBe(250);
    expect(pixels[0]).toBe(128);
  });

  it("runs nodes in order, so a later one sees what the earlier one did", () => {
    const doc = document();
    (doc.nodes[0] as { adjustments: { exposure: number } }).adjustments.exposure = 1;
    const second = createGradeNode();
    second.adjustments.exposure = 1;
    doc.nodes.push(second);
    const once = document();
    (once.nodes[0] as { adjustments: { exposure: number } }).adjustments.exposure = 1;
    const twice = renderDocument(flat(8, 8), 8, 8, doc, emptyResources()).pixels[0];
    const single = renderDocument(flat(8, 8), 8, 8, once, emptyResources()).pixels[0];
    expect(twice).toBeGreaterThan(single);
  });
});

describe("blend", () => {
  it("copies straight through with no mask", () => {
    const current = flat(4, 1, 0);
    blend(current, flat(4, 1, 200), undefined);
    expect(current[0]).toBe(200);
  });

  it("mixes proportionally at partial coverage", () => {
    const current = flat(1, 1, 0);
    blend(current, flat(1, 1, 200), new Uint8Array([128]));
    expect(current[0]).toBeGreaterThan(90);
    expect(current[0]).toBeLessThan(110);
  });
});

describe("resampleSingleChannel", () => {
  it("returns the same buffer when nothing needs to change", () => {
    const source = new Uint8Array([1, 2, 3, 4]);
    expect(resampleSingleChannel(source, 2, 2, 2, 2)).toBe(source);
  });

  it("scales up without going out of bounds", () => {
    const source = new Uint8Array([0, 255, 255, 0]);
    const scaled = resampleSingleChannel(source, 2, 2, 4, 4);
    expect(scaled).toHaveLength(16);
    expect(scaled[0]).toBe(0);
    expect(scaled[15]).toBe(0);
  });
});

describe("photoDocument", () => {
  it("knows a fresh document would rebuild the original", () => {
    expect(isUntouched(document())).toBe(true);
  });

  it("notices an enabled node that would change something", () => {
    const doc = document();
    (doc.nodes[0] as { adjustments: { exposure: number } }).adjustments.exposure = 0.5;
    expect(isUntouched(doc)).toBe(false);
  });

  it("counts geometry as a change", () => {
    const doc = document();
    doc.rotate = 90;
    expect(isUntouched(doc)).toBe(false);
  });

  it("changes a removal's cache key when its mask changes", () => {
    const doc = document();
    const mask = solidMask("Sign");
    doc.masks.push(mask);
    const node = createRemoveNode(mask.id);
    doc.nodes.push(node);
    const before = removalCacheKey(doc, node);
    mask.strokes.push({ points: [[0.5, 0.5]], radius: 0.1, hardness: 1, erase: false });
    expect(removalCacheKey(doc, node)).not.toBe(before);
  });
});
