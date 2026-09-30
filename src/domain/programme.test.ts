import { describe, expect, it } from "vitest";
import { createProject, DEFAULT_OVERLAY } from "./project";
import { buildProgramme, programmeDocument, programmeFrames, sourceFrameIndex, transitionFrame, type Programme } from "./programme";
import { TRANSITIONS } from "./transitions";
import { fromSeconds, rational } from "./time";
import type { ClipConform, GraphicOverlay, MediaAsset, Project, TimelineClip } from "./types";

const asset = (id: string, over: Partial<MediaAsset> = {}): MediaAsset => ({
  id, name: `${id}.mp4`, path: `C:/media/${id}.mp4`, kind: "video", duration: fromSeconds(20),
  width: 1920, height: 1080, frameRate: rational(24), hasAudio: true, ...over,
});

const clip = (id: string, assetId: string, start: number, sourceIn: number, duration: number, over: Partial<TimelineClip> = {}): TimelineClip => ({
  id, assetId, name: id, trackId: "v1", start: fromSeconds(start), sourceIn: fromSeconds(sourceIn),
  sourceOut: fromSeconds(sourceIn + duration), enabled: true, color: "#888", opacity: 1, volumeDb: 0, speed: [], ...over,
});

function project(clips: TimelineClip[], assets: MediaAsset[] = [asset("a"), asset("b")], edit?: (project: Project) => void): Project {
  const made = createProject("programme");
  made.assets = assets;
  made.sequences[0].tracks.find((track) => track.id === "v1")!.clips = clips;
  edit?.(made);
  return made;
}

const layerIds = (programme: Programme, frame: number) => programmeDocument(programme, frame).layers.map((layer) => layer.id);

describe("buildProgramme", () => {
  it("lays V1 out with junction handles and ripples what had none", () => {
    // B has a 1 s head handle and asks for 1 s: it borrows it, so nothing moves.
    const borrowed = buildProgramme(project([
      clip("A", "a", 0, 0, 4), clip("B", "b", 4, 1, 4, { transitionIn: "fade", transitionDuration: 1 }),
    ]))!;
    expect(borrowed.clips.map((item) => item.start)).toEqual([0, 4]);
    expect(borrowed.clips[1].incoming).toEqual({ kind: "fade", duration: 1 });
    expect(borrowed.duration).toBe(8);
    expect(borrowed.rippled).toBe(0);
    // B starts at source 0: no handle, so B moves 1 s earlier and everything after it follows.
    const rippled = buildProgramme(project([
      clip("A", "a", 0, 0, 4), clip("B", "b", 4, 0, 4, { transitionIn: "fade", transitionDuration: 1 }),
    ]))!;
    expect(rippled.clips[1].start).toBe(3);
    expect(rippled.duration).toBe(7);
    expect(rippled.rippled).toBe(1);
  });

  it("promotes the lowest overlay track when V1 is empty and keeps export sizes normalised", () => {
    const made = project([], [asset("a")], (p) => {
      p.sequences[0].tracks.find((track) => track.id === "v2")!.clips = [clip("O", "a", 0, 0, 3, { trackId: "v2" })];
    });
    const programme = buildProgramme(made, { width: 1280, height: 720, fps: 30 })!;
    expect(programme.clips.map((item) => item.id)).toEqual(["O"]);
    expect(programme.overlays).toEqual([]);
    expect(programme.width).toBe(1280);
    expect(programme.pixelScale).toBeCloseTo(1280 / 1920);
    expect(programmeFrames(programme)).toBe(90);
  });

  it("crossfades picture audio across a junction and maps audio tracks through the ripple", () => {
    const made = project([
      clip("A", "a", 0, 0, 4), clip("B", "b", 4, 0, 4, { transitionIn: "fade", transitionDuration: 1 }),
    ], [asset("a"), asset("b"), asset("m", { kind: "audio", hasAudio: true })], (p) => {
      p.sequences[0].tracks.find((track) => track.id === "a3")!.clips = [clip("M", "m", 6, 0, 2, { trackId: "a3", volumeDb: -3 })];
    });
    const programme = buildProgramme(made)!;
    const a = programme.audio.find((item) => item.id === "a:A")!;
    const b = programme.audio.find((item) => item.id === "a:B")!;
    expect(a).toMatchObject({ start: 0, duration: 4, fadeIn: 0, fadeOut: 1 });
    expect(b).toMatchObject({ start: 3, duration: 4, fadeIn: 1, sourceIn: 0 });
    // The music clip authored at 6 s sits after the junction, so it moves with the picture.
    expect(programme.audio.find((item) => item.id === "a:M")).toMatchObject({ start: 5, volumeDb: -3, kind: "audio" });
  });

  it("starts a handle-borrowing clip's audio inside its handle", () => {
    const programme = buildProgramme(project([
      clip("A", "a", 0, 0, 4), clip("B", "b", 4, 2, 4, { transitionIn: "fade", transitionDuration: 1 }),
    ]))!;
    expect(programme.audio.find((item) => item.id === "a:B")).toMatchObject({ start: 3, duration: 5, sourceIn: 1, fadeIn: 1 });
  });
});

describe("transitionFrame", () => {
  it("covers every transition id with finite, bounded treatments", () => {
    for (const transition of TRANSITIONS) {
      for (const p of [0, 0.25, 0.5, 0.75, 1]) {
        const frame = transitionFrame(transition.id, p, 16 / 9);
        for (const treat of [frame.outgoing, frame.incoming]) {
          expect(treat.opacity).toBeGreaterThanOrEqual(0);
          expect(treat.opacity).toBeLessThanOrEqual(1);
          for (const value of [treat.dx, treat.dy, treat.sx, treat.sy]) expect(Number.isFinite(value)).toBe(true);
        }
      }
    }
  });

  it("ends every junction on the incoming picture alone", () => {
    for (const transition of TRANSITIONS.filter((item) => item.id !== "none")) {
      const end = transitionFrame(transition.id, 1, 16 / 9);
      expect(end.incoming.opacity, transition.id).toBe(1);
      expect([end.incoming.dx, end.incoming.dy], transition.id).toEqual([0, 0]);
      expect(end.dip?.opacity ?? 0, transition.id).toBeCloseTo(0);
    }
  });

  it("wipes with rectangles, slides with offsets and dips through a colour", () => {
    expect(transitionFrame("wipeleft", 0.25, 1).incoming.mask?.[0].nodes.map((node) => node.x)).toEqual([0.75, 1, 1, 0.75]);
    expect(transitionFrame("slideleft", 0.25, 1)).toMatchObject({ outgoing: { dx: -0.25 }, incoming: { dx: 0.75 } });
    expect(transitionFrame("fadeblack", 0.5, 1).dip).toEqual({ color: "#000000", opacity: 1 });
    expect(transitionFrame("circleclose", 0.5, 1).outgoingOnTop).toBe(true);
    expect(transitionFrame("dissolve", 0.3, 1).incoming.effects).toEqual({ dissolve: 0.3 });
  });
});

describe("programmeDocument", () => {
  it("shows one exact source frame per programme frame, clamped to the file", () => {
    const programme = buildProgramme(project([clip("A", "a", 0, 2, 4)]))!;
    const layer = programmeDocument(programme, 12).layers.find((item) => item.id === "v1:A")!;
    expect(layer.media).toMatchObject({ path: "C:/media/a.mp4", start: 12, inPoint: 60, outPoint: 61, fps: 24 });
    expect(sourceFrameIndex(programme.clips[0].media, 99)).toBe(479);
  });

  it("draws both clips through a junction, the incoming one over the outgoing one", () => {
    const programme = buildProgramme(project([
      clip("A", "a", 0, 0, 4), clip("B", "b", 4, 1, 4, { transitionIn: "wipeleft", transitionDuration: 1 }),
    ]))!;
    expect(layerIds(programme, 3 * 24 - 1)).toEqual(["v1:A"]);
    const mid = programmeDocument(programme, Math.round(3.5 * 24));
    expect(mid.layers.map((layer) => layer.id)).toEqual(["v1:A", "v1:B"]);
    const incoming = mid.layers[1];
    expect(incoming.masks[0].stack?.[0].source.kind).toBe("path");
    expect(incoming.maskId).toBe(incoming.masks[0].id);
    // In the window B shows its handle: source 1 s - 0.5 s.
    expect(incoming.media?.inPoint).toBe(12);
    expect(layerIds(programme, 4 * 24)).toEqual(["v1:B"]);
  });

  it("lifts an outgoing clip that closes over the incoming one, with the dip above both", () => {
    const close = buildProgramme(project([
      clip("A", "a", 0, 0, 4), clip("B", "b", 4, 1, 4, { transitionIn: "circleclose", transitionDuration: 1 }),
    ]))!;
    expect(layerIds(close, Math.round(3.5 * 24))).toEqual(["v1:B", "v1:A"]);
    const dip = buildProgramme(project([
      clip("A", "a", 0, 0, 4), clip("B", "b", 4, 0, 4, { transitionIn: "fadeblack", transitionDuration: 1 }),
    ]))!;
    // B has no handle, so it starts at the window; before the midpoint only A and the dip show.
    expect(layerIds(dip, Math.round(3.25 * 24))).toEqual(["v1:A", "dip:B"]);
    expect(layerIds(dip, Math.round(3.75 * 24))).toEqual(["v1:A", "v1:B", "dip:B"]);
  });

  it("conforms contain, cover, stretch and none, with padding moving with the clip", () => {
    const at = (conform: ClipConform, media = asset("a", { width: 1000, height: 1000 })) => {
      const programme = buildProgramme(project([clip("A", "a", 0, 0, 2, { conform, scale: 50 })], [media]))!;
      return programmeDocument(programme, 0).layers;
    };
    const contain = at({ fit: "contain", pad: "black" });
    expect(contain.map((layer) => layer.id)).toEqual(["v1:A:pad", "v1:A"]);
    expect(contain[0]).toMatchObject({ scale: 0.5, stretchX: 1 });
    expect(contain[1].scale).toBeCloseTo(1.08 * 0.5);
    expect(at({ fit: "cover", pad: "black" })[0].scale).toBeCloseTo(1.92 * 0.5);
    const stretch = at({ fit: "stretch", pad: "black" })[0];
    expect(stretch.scale * (stretch.stretchX ?? 1)).toBeCloseTo(1.92 * 0.5);
    expect(stretch.scale).toBeCloseTo(1.08 * 0.5);
    expect(at({ fit: "none", pad: "black" })[0].scale).toBeCloseTo(0.5);
    const blur = at({ fit: "contain", pad: "blur", padBlur: 30 });
    expect(blur.map((layer) => layer.id)).toEqual(["v1:A:backdrop", "v1:A"]);
    expect(blur[0].nodes[0]).toMatchObject({ kind: "clipEffects", effects: { blur: 30 } });
    expect(blur[0].masks[0].stack?.[0].source.kind).toBe("path");
  });

  it("moves a crop from start to end over the clip", () => {
    const conform: ClipConform = {
      fit: "cover", pad: "black",
      crop: { start: { x: 0, y: 0, width: 0.5, height: 0.5 }, end: { x: 0.5, y: 0.5, width: 0.5, height: 0.5 } },
    };
    const programme = buildProgramme(project([clip("A", "a", 0, 0, 2, { conform })]))!;
    const first = programmeDocument(programme, 0).layers[0];
    const middle = programmeDocument(programme, 24).layers[0];
    expect([first.cropLeft, first.cropRight]).toEqual([0, 50]);
    expect(middle.cropLeft).toBeCloseTo(25);
    expect(middle.cropBottom).toBeCloseTo(25);
  });

  it("puts clip effects on a node, seeded by the frame, and skips neutral ones", () => {
    const programme = buildProgramme(project([
      clip("A", "a", 0, 0, 2, { effects: { brightness: 0.2, contrast: 1, saturation: 1, gamma: 1, hue: 0, blur: 0, sharpen: 0, vignette: 0, noise: 10 } }),
      clip("B", "b", 2, 0, 2),
    ]))!;
    expect(programmeDocument(programme, 5).layers[0].nodes[0]).toMatchObject({ kind: "clipEffects", effects: { brightness: 0.2, noise: 10, seed: 5 } });
    expect(programmeDocument(programme, 50).layers[0].nodes).toEqual([]);
  });

  it("dips the head and tail of the programme", () => {
    const programme = buildProgramme(project([
      clip("A", "a", 0, 0, 4, { transitionIn: "fadewhite", transitionDuration: 1, transitionOut: "fadeblack", transitionOutDuration: 2 }),
    ]))!;
    const head = programmeDocument(programme, 6).layers.find((layer) => layer.id === "fade:head")!;
    expect(head.opacity).toBeCloseTo(0.75);
    expect(head.path?.fill).toEqual({ kind: "solid", color: "#ffffff" });
    expect(programmeDocument(programme, 72).layers.find((layer) => layer.id === "fade:tail")?.opacity).toBeCloseTo(0.5);
    expect(layerIds(programme, 40)).toEqual(["v1:A"]);
  });

  it("places overlays in their boxes with a rounded clip and an inside border", () => {
    const made = project([clip("A", "a", 0, 0, 4)], [asset("a"), asset("p", { width: 1000, height: 500 })], (p) => {
      const track = p.sequences[0].tracks.find((item) => item.id === "v2")!;
      track.overlay = { ...DEFAULT_OVERLAY, x: 0.5, y: 0.5, width: 0.25, height: 0.25, fit: "cover", cornerRadius: 0.1, borderWidth: 0.01, opacity: 0.8 };
      track.clips = [clip("P", "p", 1, 0, 2, { trackId: "v2", opacity: 0.5 })];
    });
    const programme = buildProgramme(made)!;
    expect(layerIds(programme, 0)).toEqual(["v1:A"]);
    const layers = programmeDocument(programme, 30).layers;
    expect(layers.map((layer) => layer.id)).toEqual(["v1:A", "ov:P", "ov:P:border"]);
    const overlay = layers[1];
    expect(overlay.x).toBeCloseTo(0.625);
    expect(overlay.y).toBeCloseTo(0.625);
    // Cover: 480 x 270 box from a 2:1 source -> height-limited at 0.54, and 0.8 * 0.5 opacity.
    expect(overlay.scale).toBeCloseTo(0.54);
    expect(overlay.opacity).toBeCloseTo(0.4);
    expect(overlay.masks[0].stack?.[0].source.kind).toBe("path");
    expect(layers[2].path?.strokeWidth).toBe(19);
  });

  it("draws titles as frame-sized timeline graphic layers on the programme clock", () => {
    const graphic: GraphicOverlay = {
      template: "lower-third", text: "Hello", subtitle: "World", textColor: "#ffffff", backgroundColor: "#000000",
      accentColor: "#ff0000", fontSize: 48,
    };
    const made = project([clip("A", "a", 0, 0, 4)], undefined, (p) => {
      p.sequences[0].tracks.find((item) => item.id === "g1")!.clips = [clip("T", "a", 1, 0, 2, { trackId: "g1", graphic })];
    });
    const programme = buildProgramme(made)!;
    const layer = programmeDocument(programme, 36).layers.find((item) => item.id === "g:T")!;
    expect(layer.timelineGraphic).toMatchObject({ local: 0.5, duration: 2, canvasWidth: 1920, canvasHeight: 1080 });
    expect(programmeDocument(programme, 72).layers.some((item) => item.id === "g:T")).toBe(false);
  });

  it("asks the VFX hook for each clip and swaps in a rendered file when it returns one", () => {
    const programme = buildProgramme(project([clip("A", "a", 0, 0, 4)]))!;
    const calls: number[] = [];
    const doc = programmeDocument(programme, 24, {
      vfx: (item, local, frame, fps) => {
        calls.push(local, frame, fps);
        return { nodes: [], masks: [], sourcePath: "C:/cache/vfx.mp4", sourceSeconds: 0.5 };
      },
    });
    expect(calls).toEqual([1, 24, 24]);
    expect(doc.layers[0].media).toMatchObject({ path: "C:/cache/vfx.mp4", inPoint: 12 });
  });

  it("is deterministic in (programme, frame)", () => {
    const programme = buildProgramme(project([
      clip("A", "a", 0, 0, 4), clip("B", "b", 4, 1, 4, { transitionIn: "radial", transitionDuration: 1 }),
    ]))!;
    expect(JSON.stringify(programmeDocument(programme, 80))).toBe(JSON.stringify(programmeDocument(programme, 80)));
  });
});
