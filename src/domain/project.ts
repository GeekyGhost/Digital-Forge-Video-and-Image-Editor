import type { OverlayPlacement, Project, Track } from "./types";
import { rational } from "./time";

/** A full-frame overlay: the sensible default, since a background-removed clip wants to sit
 *  exactly over the programme rather than in a corner. */
export const DEFAULT_OVERLAY: OverlayPlacement = {
  x: 0, y: 0, width: 1, height: 1,
  fit: "contain", opacity: 1, cornerRadius: 0, borderWidth: 0, borderColor: "#00f0ff",
};

/** Common picture-in-picture placements, as fractions of the sequence frame. */
export const OVERLAY_PRESETS: Array<{ id: string; name: string; placement: Partial<OverlayPlacement> }> = [
  { id: "full", name: "Full frame", placement: { x: 0, y: 0, width: 1, height: 1, fit: "contain" } },
  { id: "pip-tr", name: "PiP · top right", placement: { x: 0.62, y: 0.05, width: 0.33, height: 0.33, fit: "cover" } },
  { id: "pip-tl", name: "PiP · top left", placement: { x: 0.05, y: 0.05, width: 0.33, height: 0.33, fit: "cover" } },
  { id: "pip-br", name: "PiP · bottom right", placement: { x: 0.62, y: 0.62, width: 0.33, height: 0.33, fit: "cover" } },
  { id: "pip-bl", name: "PiP · bottom left", placement: { x: 0.05, y: 0.62, width: 0.33, height: 0.33, fit: "cover" } },
  { id: "side-left", name: "Split · left half", placement: { x: 0, y: 0, width: 0.5, height: 1, fit: "cover" } },
  { id: "side-right", name: "Split · right half", placement: { x: 0.5, y: 0, width: 0.5, height: 1, fit: "cover" } },
  { id: "lower-left", name: "Presenter · lower left", placement: { x: 0.02, y: 0.45, width: 0.3, height: 0.53, fit: "contain" } },
];

export const audioTrack = (id: string, name: string): Track => ({
  id, name, kind: "audio", locked: false, muted: false, hidden: false, clips: [], gainDb: 0,
});

const id = () => crypto.randomUUID();

export const createProject = (name = "Untitled Film"): Project => {
  const now = new Date().toISOString();
  const sequenceId = id();
  return {
    schemaVersion: 1,
    id: id(),
    name,
    createdAt: now,
    updatedAt: now,
    assets: [],
    sequences: [{
      id: sequenceId,
      name: "Main Sequence",
      width: 1920,
      height: 1080,
      frameRate: rational(24),
      sampleRate: 48_000,
      tracks: [
        { id: "g1", name: "G1 · Graphics", kind: "graphics", locked: false, muted: false, hidden: false, clips: [] },
        // A real picture track that composites over V1: picture-in-picture, split screen, or a
        // background-removed cutout sitting full frame.
        { id: "v2", name: "V2 · Overlay", kind: "video", locked: false, muted: false, hidden: false, clips: [], overlay: { ...DEFAULT_OVERLAY } },
        { id: "v1", name: "V1 · Primary", kind: "video", locked: false, muted: false, hidden: false, clips: [] },
        audioTrack("a1", "A1 · Clip Audio"),
        audioTrack("a2", "A2 · SFX"),
        audioTrack("a3", "A3 · Music"),
        { id: "c1", name: "C1 · Captions", kind: "caption", locked: false, muted: false, hidden: false, clips: [] },
      ],
    }],
    activeSequenceId: sequenceId,
  };
};
