import { describe, expect, it } from "vitest";
import {
  createEmitterLayer, createLayer, createMediaLayer, createPhotoDocument, createTrailNode,
  type LayerMedia, type PhotoDocument, type PhotoLayer, type PointTrack, type TrackSample,
} from "./photoDocument";
import { depthThresholdLut, entryCoverage } from "./photoMask";
import { emitterElementIndex, emitterSpawns, mediaKeysAt, compositionDuration } from "./photoTime";
import {
  TRAIL_BINS, applyTrail, attachedLayer, canvasToSourcePoint, depthFrameOf, depthFramePaths, emitterParticles, greyOf, lucasKanade,
  lucasKanadeBothWays, sam2Centroid, trackAt, trackCanvasPoint, trackTime, trailDisplacement, trailField,
  type GreyFrame,
} from "./photoTrack";

const CANVAS = { width: 1920, height: 1080 };

const plateMedia = (over: Partial<LayerMedia> = {}): LayerMedia => ({
  kind: "video", path: "C:/m/plate.mp4", frameCount: 48, fps: 24, width: 1920, height: 1080,
  start: 0, inPoint: 0, outPoint: 48, loop: false, crossfade: 0, ...over,
});

const smokeMedia: LayerMedia = {
  kind: "frames", path: "C:/el/smoke", pattern: "frame_%04d.png", frameCount: 20, fps: 24, width: 200, height: 300,
  start: 0, inPoint: 0, outPoint: 20, loop: false, crossfade: 0,
};

/** A track moving +10 px in x per source frame from x = 100, seen everywhere except frame 5. */
function linearTrack(layerId: string, first = 0, n = 20): PointTrack {
  const samples: TrackSample[] = [];
  for (let i = 0; i < n; i++) samples.push([100 + (first + i) * 10, 500, i === 5 ? 0 : 1]);
  return { id: "t", name: "t", layerId, first, samples, source: "manual" };
}

function documentWith(plate: PhotoLayer, others: PhotoLayer[] = [], track = linearTrack(plate.id)): PhotoDocument {
  return { ...createPhotoDocument("a", "C:/a.png"), nodes: [], masks: [], frameRate: 24, tracks: [track], layers: [plate, ...others] };
}

describe("trackAt", () => {
  const track = linearTrack("p", 2);
  it("returns samples, interpolates between them and holds the ends", () => {
    expect(trackAt(track, 2)).toEqual([120, 500]);
    expect(trackAt(track, 3.5)).toEqual([135, 500]);
    expect(trackAt(track, -40)).toEqual([120, 500]);
    expect(trackAt(track, 1000)).toEqual([100 + 21 * 10, 500]);
  });
  it("holds the nearest seen sample across an unseen one (earlier on a tie)", () => {
    // Sample index 5 (source frame 7) is unseen: its neighbours are equally near, the earlier wins.
    expect(trackAt(track, 7)).toEqual([100 + 6 * 10, 500]);
  });
  it("is undefined when nothing was ever seen", () => {
    expect(trackAt({ ...track, samples: [[1, 2, 0], [3, 4, 0]] }, 0)).toBeUndefined();
  });
});

describe("trackTime", () => {
  it("is the source frame the layer shows, minus the lag in source frames", () => {
    const layer = createMediaLayer("p", plateMedia({ start: 4, inPoint: 2, fps: 12 }));
    // Composition at 24: at frame 10 the clip is 6 frames in, showing source 2 + 3 = 5.
    expect(trackTime(layer, 10, 0, 24)).toBe(5);
    expect(trackTime(layer, 10, 4, 24)).toBe(3);
    // Before its start: the unclamped position, which trackAt then holds.
    expect(trackTime(layer, 0, 0, 24)).toBe(0);
  });
  it("counts composition frames for a layer without media", () => {
    expect(trackTime(createLayer({ id: "a", name: "a", path: "x" }), 7, 2, 24)).toBe(5);
  });
});

describe("attachedLayer", () => {
  it("moves by the track's canvas motion since the anchor, through the tracked layer's placement", () => {
    const plate = createMediaLayer("Plate", plateMedia(), { x: 0.5, y: 0.5, scale: 1 });
    const smoke = createMediaLayer("Smoke", smokeMedia, { x: 0.3, y: 0.4, attach: { trackId: "t", anchorFrame: 2 } });
    const document = documentWith(plate, [smoke]);
    const at = (frame: number) => attachedLayer(document, smoke, frame, CANVAS, 24);
    expect(at(2).x).toBeCloseTo(0.3, 12);
    expect(at(12).x).toBeCloseTo(0.3 + 100 / 1920, 12);
    expect(at(12).y).toBeCloseTo(0.4, 12);
  });
  it("scales the motion by the plate's own scale, and lags", () => {
    const plate = createMediaLayer("Plate", plateMedia(), { x: 0.5, y: 0.5, scale: 0.5 });
    const smoke = createMediaLayer("Smoke", smokeMedia, { x: 0.3, y: 0.4, attach: { trackId: "t", anchorFrame: 0, lag: 3 } });
    const document = documentWith(plate, [smoke]);
    // At frame 10 with lag 3 the track is read at source frame 7: 70 px from the anchor, halved by the plate's scale.
    expect(attachedLayer(document, smoke, 10, CANVAS, 24).x).toBeCloseTo(0.3 + (70 * 0.5) / 1920, 12);
  });
  it("leaves the layer alone without a track", () => {
    const plate = createMediaLayer("Plate", plateMedia());
    const smoke = createMediaLayer("Smoke", smokeMedia, { attach: { trackId: "missing", anchorFrame: 0 } });
    const document = documentWith(plate, [smoke]);
    expect(attachedLayer(document, smoke, 9, CANVAS, 24)).toBe(smoke);
  });
  it("maps through crop and rotation", () => {
    const plate = createMediaLayer("Plate", plateMedia(), { rotation: 90, cropLeft: 10 });
    const document = documentWith(plate);
    const point = trackCanvasPoint(document, document.tracks![0], 0, 0, CANVAS, 24)!;
    // Rotated a quarter turn: a source x motion becomes a canvas y motion.
    const later = trackCanvasPoint(document, document.tracks![0], 3, 0, CANVAS, 24)!;
    expect(later[0] - point[0]).toBeCloseTo(0, 9);
    expect(later[1] - point[1]).toBeCloseTo(30, 9);
  });
});

describe("canvasToSourcePoint", () => {
  it("is the inverse of the track's canvas mapping, through crop, scale and rotation", () => {
    const plate = createMediaLayer("Plate", plateMedia(), { rotation: 17, scale: 0.7, cropLeft: 8, cropTop: 4, x: 0.45, y: 0.55 });
    const document = documentWith(plate);
    const point = trackCanvasPoint(document, document.tracks![0], 3, 0, CANVAS, 24)!;
    const back = canvasToSourcePoint(plate, [point[0] / CANVAS.width, point[1] / CANVAS.height], CANVAS)!;
    expect(back[0]).toBeCloseTo(130, 9);
    expect(back[1]).toBeCloseTo(500, 9);
    expect(canvasToSourcePoint(createLayer({ id: "a", name: "a", path: "x" }), [0.5, 0.5], CANVAS)).toBeUndefined();
  });
});

describe("trail", () => {
  const plate = createMediaLayer("Plate", plateMedia());
  const smoke = createMediaLayer("Smoke", smokeMedia, { x: 0.5, y: 0.5 });
  const document = documentWith(plate, [smoke]);
  const node = { ...createTrailNode("t"), maxLag: 4, strength: 1 };

  it("holds the lagged track motion per height, in working pixels", () => {
    const field = trailField(document, node, smoke, { width: 200, height: 300 }, 12, 24, CANVAS, 960, 540, 0.5)!;
    expect(field.table).toHaveLength(TRAIL_BINS * 2);
    expect(field.table[0]).toBe(0);
    // The far end: 4 frames late is 40 source px (= canvas px at scale 1) behind, halved for the proxy.
    expect(field.table[(TRAIL_BINS - 1) * 2]).toBeCloseTo(-20, 5);
    expect(field.still).toBe(false);
    expect(field.rectHeight).toBe(300);
  });
  it("is undefined for a missing track and still for a track that did not move", () => {
    expect(trailField(document, { ...node, trackId: "nope" }, smoke, undefined, 3, 24, CANVAS, 100, 100, 1)).toBeUndefined();
    const parked = documentWith(plate, [smoke], { ...linearTrack(plate.id), samples: [[5, 5, 1], [5, 5, 1]] });
    expect(trailField(parked, node, smoke, undefined, 1, 24, CANVAS, 100, 100, 1)!.still).toBe(true);
  });
  it("displaces the far end, keeps the base and does not bleed transparent colour", () => {
    const width = 8, height = 8;
    const table = new Float32Array(TRAIL_BINS * 2);
    for (let bin = 0; bin < TRAIL_BINS; bin++) table[bin * 2] = -2 * bin / (TRAIL_BINS - 1);
    const field = { table, inverse: { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }, rectHeight: height, still: false };
    expect(trailDisplacement(field, 3.5, 7.5)[0]).toBeCloseTo(-2 * (1 / 16), 6);
    const pixels = new Uint8ClampedArray(width * height * 4);
    // A vertical opaque white bar at x = 3, and a red transparent pixel beside it.
    for (let y = 0; y < height; y++) {
      pixels.set([255, 255, 255, 255], (y * width + 3) * 4);
      pixels.set([255, 0, 0, 0], (y * width + 4) * 4);
    }
    const out = applyTrail(pixels, width, height, field);
    // A negative displacement is where the far end WAS: out(p) = in(p - d), so the top row (far end)
    // shows the bar ~1.9 px to the left, near x = 1.
    expect(out[(0 * width + 1) * 4 + 3]).toBeGreaterThan(200);
    expect(out[(0 * width + 3) * 4 + 3]).toBe(0);
    // Bottom row stays near x = 3.
    expect(out[(7 * width + 3) * 4 + 3]).toBeGreaterThan(200);
    // Whatever is visible stays white: the red of a transparent texel never enters.
    for (let at = 0; at < out.length; at += 4) if (out[at + 3] > 0) expect(out[at + 1]).toBe(255);
  });
});

describe("emitter", () => {
  const generator = createEmitterLayer(smokeMedia, { seed: 5, rate: 0.5, lifetime: 6 }).generator as Extract<PhotoLayer["generator"], { kind: "emitter" }>;

  it("spawns a fractional rate without state, oldest first, only while alive", () => {
    expect(emitterSpawns(generator, 0)).toEqual([]);
    expect(emitterSpawns(generator, 1)).toEqual([{ spawn: 1, index: 0, age: 0 }]);
    const at9 = emitterSpawns(generator, 9);
    expect(at9.map((s) => s.spawn)).toEqual([5, 7, 9]);
    expect(emitterSpawns({ ...generator, rate: 2.5 }, 0)).toHaveLength(2);
    expect(emitterSpawns({ ...generator, rate: 1000, lifetime: 1000 }, 999).length).toBeLessThanOrEqual(400);
  });

  it("is a pure function of (seed, frame)", () => {
    const document = documentWith(createMediaLayer("Plate", plateMedia()));
    const a = emitterParticles(document, generator, 17, 24, CANVAS);
    const b = emitterParticles(document, { ...generator }, 17, 24, CANVAS);
    expect(a).toEqual(b);
    expect(emitterParticles(document, { ...generator, seed: 6 }, 17, 24, CANVAS)).not.toEqual(a);
    // A particle keeps its identity: the same (spawn, index) at a later frame has moved by its velocity.
    const later = emitterParticles(document, generator, 18, 24, CANVAS);
    const first = a.find((p) => p.spawn === 13)!, again = later.find((p) => p.spawn === 13)!;
    expect(again.age).toBe(first.age + 1);
    expect(again.rotation).toBe(first.rotation);
    expect(again.y - first.y).toBeCloseTo(generator.velocityY + (again.y - first.y - generator.velocityY), 12);
  });

  it("starts particles on the track at their spawn frame and plays the element from its first frame", () => {
    const plate = createMediaLayer("Plate", plateMedia());
    const document = documentWith(plate);
    const onTrack = { ...generator, trackId: "t", spread: 0, velocityX: 0, velocityY: 0 };
    const particles = emitterParticles(document, onTrack, 9, 24, CANVAS);
    const newest = particles[particles.length - 1];
    expect(newest.age).toBe(0);
    // Source frame 9 of the plate: x = 190 px (plate centred at scale 1 on a same-size canvas).
    expect(newest.x * 1920).toBeCloseTo(190, 9);
    expect(newest.frameIndex).toBe(0);
    expect(emitterElementIndex(smokeMedia, 23, 24)).toBe(3);
  });

  it("asks the frame store for the element frames of the living particles", () => {
    const emitter = createEmitterLayer(smokeMedia, { rate: 1, lifetime: 3 });
    const document = { ...documentWith(createMediaLayer("Plate", plateMedia())), layers: [emitter] };
    expect(mediaKeysAt(document, 10)).toEqual(["C:/el/smoke/frame_0002.png", "C:/el/smoke/frame_0001.png", "C:/el/smoke/frame_0000.png"]);
    expect(compositionDuration(document)).toBe(3);
  });
});

describe("sequence mattes that start part-way into a clip", () => {
  it("offset their frames by `first` and hold the first matte before it", async () => {
    const { sequenceFramePath } = await import("./photoMask");
    const source = { kind: "sequence" as const, cachePath: "C:/c/sam", frameCount: 10, first: 30 };
    expect(sequenceFramePath(source, 30)).toBe("C:/c/sam/masks/00000.png");
    expect(sequenceFramePath(source, 34)).toBe("C:/c/sam/masks/00004.png");
    expect(sequenceFramePath(source, 12)).toBe("C:/c/sam/masks/00000.png");
    expect(sequenceFramePath(source, 99)).toBe("C:/c/sam/masks/00009.png");
    expect(sequenceFramePath({ ...source, first: undefined }, 4)).toBe("C:/c/sam/masks/00004.png");
  });
});

describe("depth occlusion", () => {
  it("thresholds by a table: nearer than the plane, smoothstepped over the softness", () => {
    const hard = depthThresholdLut(50, 0);
    expect(hard[127]).toBe(0);
    expect(hard[128]).toBe(255);
    const soft = depthThresholdLut(50, 20);
    expect(soft[Math.floor(0.39 * 255)]).toBe(0);
    expect(soft[Math.ceil(0.61 * 255)]).toBe(255);
    expect(soft[128]).toBeGreaterThan(100);
    expect(soft[128]).toBeLessThan(155);
  });
  it("reads the depth frame the layer shows, and is a mask entry like any other", () => {
    const plate = createMediaLayer("Plate", plateMedia({ start: 2, depth: { cachePath: "C:/cache/depth/x", frameCount: 48 } }));
    const layers = new Map([[plate.id, plate]]);
    expect(depthFrameOf(layers, plate.id, 7, 24)?.path).toBe("C:/cache/depth/x/masks/00005.png");
    expect(depthFrameOf(layers, plate.id, 1, 24)).toBeUndefined();
    const document = { ...documentWith(plate), masks: [{
      id: "m", name: "m", enabled: true, invert: false, feather: 0, strokes: [],
      stack: [{ id: "e", enabled: true, mode: "subtract" as const, opacity: 1, invert: false, feather: 0, expand: 0, strokes: [], source: { kind: "depth" as const, layerId: plate.id, plane: 50, softness: 0 } }],
    }] };
    expect(depthFramePaths(document, 3, 24)).toEqual(["C:/cache/depth/x/masks/00001.png"]);
    const coverage = entryCoverage(document.masks[0].stack[0], { width: 2, height: 1, depth: () => new Uint8Array([10, 250]) });
    expect(Array.from(coverage!)).toEqual([0, 255]);
  });
});

/* ------------------------------------------------------------ trackers */

/** A textured blob on a textured background, centred at (cx, cy). */
function frameAt(cx: number, cy: number, width = 96, height = 72): GreyFrame {
  const data = new Float32Array(width * height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const background = 60 + 20 * Math.sin(x * 0.45) * Math.cos(y * 0.37);
      const d = Math.hypot(x - cx, y - cy);
      const blob = d < 9 ? 120 + 60 * Math.cos((x - cx) * 0.9) * Math.sin((y - cy) * 0.8) : 0;
      data[y * width + x] = background + blob;
    }
  }
  return { data, width, height };
}

describe("built-in trackers", () => {
  it("Lucas-Kanade follows a moving blob to within half a pixel", () => {
    const path = (t: number): [number, number] => [30 + t * 2.5, 30 + Math.sin(t / 3) * 6];
    const frames = Array.from({ length: 12 }, (_, t) => frameAt(...path(t)));
    const start = path(0);
    const samples = lucasKanade(frames, [start[0] + 0.5, start[1] + 0.5]);
    expect(samples).toHaveLength(12);
    for (let t = 0; t < 12; t++) {
      const [x, y, seen] = samples[t];
      expect(seen).toBe(1);
      expect(Math.hypot(x - 0.5 - path(t)[0], y - 0.5 - path(t)[1])).toBeLessThan(0.5);
    }
  });
  it("tracks both ways from a middle frame", () => {
    const path = (t: number): [number, number] => [20 + t * 3, 36];
    const frames = Array.from({ length: 9 }, (_, t) => frameAt(...path(t)));
    const samples = lucasKanadeBothWays(frames, 4, [path(4)[0] + 0.5, 36.5]);
    expect(samples).toHaveLength(9);
    expect(samples[0][0] - 0.5).toBeCloseTo(path(0)[0], 0);
    expect(samples[8][0] - 0.5).toBeCloseTo(path(8)[0], 0);
  });
  it("marks a point unseen when the forward-backward check fails", () => {
    const frames = [frameAt(40, 36), { data: new Float32Array(96 * 72).map((_, i) => (i * 7919) % 251), width: 96, height: 72 }];
    const samples = lucasKanade(frames, [40.5, 36.5]);
    expect(samples[1][2]).toBe(0);
  });
  it("SAM 2 centroid is the coverage-weighted centre, unseen when empty", () => {
    const width = 10, height = 6;
    const coverage = new Uint8Array(width * height);
    coverage[2 * width + 3] = 255;
    coverage[2 * width + 5] = 255;
    const samples = sam2Centroid([{ coverage, width, height }, { coverage: new Uint8Array(width * height), width, height }]);
    expect(samples[0]).toEqual([4.5, 2.5, 1]);
    expect(samples[1][2]).toBe(0);
  });
  it("greyOf is Rec. 709 luma", () => {
    expect(greyOf(new Uint8ClampedArray([255, 0, 0, 255]), 1, 1).data[0]).toBeCloseTo(0.2126 * 255, 3);
  });
});
