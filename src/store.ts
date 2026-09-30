import { create } from "zustand";
import { createProject } from "./domain/project";
import type { PhotoDocument } from "./domain/photoDocument";
import type { H3DirectorWorkspace } from "./domain/h3Director";
import { DEFAULT_EFFECTS } from "./domain/effects";
import { clipDuration, clipPlaybackRate, fromSeconds, rippleShiftFor, seconds, snapTime, timelineClipDuration } from "./domain/time";
import type { AudioClipSettings, ClipConform, ExportSettings, GraphicOverlay, GraphicTemplate, MediaAsset, OverlayPlacement, Project, TimelineClip, Track } from "./domain/types";
import { audioTrack, DEFAULT_OVERLAY } from "./domain/project";
import { replaceClipRange } from "./features/timeline/operations";
import { isAudioAsset } from "./domain/media";
import { defaultGraphicOverlay } from "./domain/textMotion";
import type { ClipVfxLayer } from "./domain/maskVfx";
import type { ClipMatte } from "./domain/clipMatte";


type ClipPatch = Partial<Omit<TimelineClip, "id" | "assetId" | "trackId">>;
export type CaptionCue = { startSeconds: number; endSeconds: number; text: string };
export type TimelineTool = "select" | "blade" | "track";
/** A SAM 2 tracking point, in normalised 0..1 coordinates of the program frame. */
export type MaskPoint = { x: number; y: number; include: boolean };

type EditorState = {
  project: Project;
  /** Increments when the whole document is replaced, even if it has the same persistent id. */
  projectEpoch: number;
  projectPath?: string;
  selectedAssetId?: string;
  selectedClipId?: string;
  playhead: number;
  playing: boolean;
  snapping: boolean;
  zoom: number;
  tool: TimelineTool;
  groupSelection: string[];
  activePanel: "inspector" | "ai" | "export";
  sequenceSettingsOpen: boolean;
  clipEditorOpen: boolean;
  peaks: Record<string, number[]>;
  thumbnails: Record<string, string>;
  maskPoints: MaskPoint[];
  maskPicking: boolean;
  /** Background removal: the clip whose subject monitor clicks pick (points in its source). */
  mattePickClipId?: string;
  status: string;
  busy: boolean;
  activeWorkspace: "image" | "photo" | "ref2v" | "video";
  /** Which Project Media image Photo Edit is working on. Set from either workspace. */
  photoEditAssetId?: string;
  setPhotoEditAsset: (id: string | undefined) => void;
  openPhotoEdit: (id: string) => void;
  /** An asset Photo Edit asked to be dropped onto the Image Studio canvas, consumed once there. */
  pendingCanvasAssetId?: string;
  sendToImageStudio: (id: string) => void;
  clearPendingCanvasAsset: () => void;
  /**
   * Write a Photo Edit document into the project.
   *
   * `coalesce` folds a run of rapid changes into one undo entry, which is what a slider drag is:
   * two hundred input events that the user experienced as a single act. Structural edits -- adding
   * a node, deleting a mask -- pass false so they always land as their own step.
   */
  setPhotoDocument: (document: PhotoDocument, options?: { coalesce?: boolean }) => void;
  /**
   * VFX Phase 4b: the Clip Editor's VFX composition for one timeline clip, kept in `project.clipVfx`
   * by clip id. Same undo behaviour as `setPhotoDocument`.
   */
  setClipVfxDocument: (clipId: string, document: PhotoDocument, options?: { coalesce?: boolean }) => void;
  /** Save the Director working session and take ledger into the ordinary project document. */
  setH3DirectorWorkspace: (workspace: H3DirectorWorkspace) => void;
  past: Project[];
  future: Project[];
  setStatus: (status: string) => void;
  setBusy: (busy: boolean) => void;
  setActiveWorkspace: (workspace: "image" | "photo" | "ref2v" | "video") => void;
  setProjectPath: (path?: string) => void;
  replaceProject: (project: Project, path?: string) => void;
  newProject: () => void;
  /** `keepSelection`: register without deselecting the timeline clip (the Clip Editor lives on it). */
  addAssets: (assets: MediaAsset[], options?: { keepSelection?: boolean }) => void;
  renameAsset: (id: string, name: string) => void;
  removeAsset: (id: string) => void;
  setSelectedAsset: (id?: string) => void;
  addAssetToTimeline: (assetId: string) => void;
  addAssetToTimelineAt: (assetId: string, startSeconds: number, trackId?: string) => void;
  /** Register a generated WAV and place it on the sequence's SFX track in one undoable action. */
  addSfxToTimeline: (asset: MediaAsset, startSeconds: number, settings: AudioClipSettings & {
    gainDb: number; trimInSeconds: number; trimOutSeconds: number;
  }) => void;
  /** Place a generated picture/soundtrack pair at the same time on V1 and A1, unsnapped. */
  addGeneratedAvPair: (videoAssetId: string, audioAssetId: string) => void;
  /** Ripple-insert a generated H3 picture/soundtrack pair immediately after a V1 clip. */
  insertGeneratedAvPairAfterClip: (clipId: string, videoAssetId: string, audioAssetId: string) => void;
  addGraphic: (template: GraphicTemplate) => void;
  /** Add an editable graphic at an exact timeline range (used by Clip Editor Text Motion). */
  addGraphicToTimeline: (graphic: GraphicOverlay, startSeconds: number, durationSeconds: number) => void;
  addCaptionCues: (cues: CaptionCue[]) => void;
  replaceSelectedClipWithAsset: (asset: MediaAsset) => void;
  replaceClipWithAsset: (clipId: string, asset: MediaAsset) => void;
  repointClipToAsset: (clipId: string, asset: MediaAsset, status?: string) => void;
  replaceClipRangeWithAsset: (clipId: string, startOffsetSeconds: number, endOffsetSeconds: number, asset: MediaAsset) => void;
  insertAssetAfterClip: (clipId: string, asset: MediaAsset) => void;
  setSelectedClip: (id?: string) => void;
  setPlayhead: (value: number) => void;
  setPlaying: (value: boolean) => void;
  togglePlaying: () => void;
  toggleSnapping: () => void;
  setZoom: (value: number) => void;
  setTool: (tool: TimelineTool) => void;
  selectTrackFrom: (trackId: string, atSeconds: number) => void;
  clearGroupSelection: () => void;
  addMaskPoint: (point: MaskPoint) => void;
  clearMaskPoints: () => void;
  setMaskPicking: (picking: boolean) => void;
  setMattePick: (clipId?: string) => void;
  setAssetPeaks: (assetId: string, peaks: number[]) => void;
  setAssetThumbnail: (assetId: string, dataUrl: string) => void;
  setActivePanel: (panel: EditorState["activePanel"]) => void;
  setSequenceSettingsOpen: (open: boolean) => void;
  setClipEditorOpen: (open: boolean) => void;
  updateSequenceSettings: (settings: { name: string; width: number; height: number; frameRate: { numerator: number; denominator: number }; sampleRate: number; conform?: ClipConform }) => void;
  /** Persist the export panel's settings on the active sequence. Deliberately not undoable. */
  setExportSettings: (settings: ExportSettings) => void;
  moveClip: (clipId: string, startSeconds: number, targetTrackId?: string) => void;
  trimClip: (clipId: string, edge: "left" | "right", deltaSeconds: number) => void;
  deleteClip: (clipId: string) => void;
  rippleDeleteClip: (clipId: string) => void;
  duplicateClip: (clipId: string) => void;
  splitClip: (clipId: string, atSeconds: number) => void;
  removeSourceRangeFromClip: (clipId: string, startOffsetSeconds: number, endOffsetSeconds: number) => void;
  applyTransitionToAllCuts: (transitionId: string, duration: number) => void;
  updateTrack: (trackId: string, patch: { muted?: boolean; hidden?: boolean; locked?: boolean; gainDb?: number; name?: string }) => void;
  updateOverlay: (trackId: string, patch: Partial<OverlayPlacement>) => void;
  addAudioTrack: (name?: string) => string;
  addOverlayTrack: () => string;
  removeTrack: (trackId: string) => void;
  updateSelectedClip: (patch: ClipPatch) => void;
  upsertClipVfxLayer: (clipId: string, layer: ClipVfxLayer) => void;
  removeClipVfxLayer: (clipId: string, layerId: string) => void;
  /** Background removal settings and matte for a clip (undefined removes it). */
  setClipMatte: (clipId: string, matte: ClipMatte | undefined, status?: string) => void;
  undo: () => void;
  redo: () => void;
};

const currentSequence = (project: Project) =>
  project.sequences.find((sequence) => sequence.id === project.activeSequenceId)!;
const palette = ["#cf6b4f", "#477e91", "#7c6da8", "#57916d", "#ad7b45"];

const newClip = (asset: MediaAsset, trackId: string, start: number, color: string): TimelineClip => ({
  id: crypto.randomUUID(),
  assetId: asset.id,
  name: asset.name,
  trackId,
  start: fromSeconds(start),
  sourceIn: fromSeconds(0),
  sourceOut: asset.duration,
  enabled: true,
  color,
  opacity: 1,
  volumeDb: 0,
  positionX: 960,
  positionY: 540,
  scale: 100,
  rotation: 0,
  blendMode: "normal",
  transitionIn: "none",
  transitionDuration: 0.5,
  transitionOut: "none",
  transitionOutDuration: 0.5,
  effects: { ...DEFAULT_EFFECTS },
  speed: [{ at: fromSeconds(0), rate: 1, easing: "linear" }],
});

const graphicDefaults = defaultGraphicOverlay;

/**
 * When the last Photo Edit change landed, used only to decide whether the next one continues the
 * same undo step. Kept outside the store on purpose: it is a property of the gesture in progress,
 * not of the project, and putting it in state would snapshot it into the undo stack it governs.
 */
let photoHistoryStamp: { assetId: string; at: number } | undefined;

export const useEditor = create<EditorState>((set, get) => ({
  project: createProject(),
  projectEpoch: 0,
  playhead: 0,
  playing: false,
  snapping: true,
  zoom: 56,
  tool: "select",
  groupSelection: [],
  activePanel: "inspector",
  sequenceSettingsOpen: false,
  clipEditorOpen: false,
  peaks: {},
  thumbnails: {},
  maskPoints: [],
  maskPicking: false,
  status: "Ready — import media to begin",
  busy: false,
  activeWorkspace: "image",
  past: [],
  future: [],
  setStatus: (status) => set({ status }),
  setBusy: (busy) => set({ busy }),
  setActiveWorkspace: (activeWorkspace) => set({ activeWorkspace }),
  setPhotoEditAsset: (photoEditAssetId) => set({ photoEditAssetId }),
  /** The round trip from either library: pick a picture and land in Photo Edit with it loaded. */
  openPhotoEdit: (id) => set({ photoEditAssetId: id, activeWorkspace: "photo", playing: false }),
  /**
   * Photo Edit cannot reach into Image Studio's layer list -- those layers are its own local state.
   * So the request travels through here as an id, and Image Studio picks it up on arrival.
   */
  sendToImageStudio: (id) => set({ pendingCanvasAssetId: id, activeWorkspace: "image", playing: false }),
  clearPendingCanvasAsset: () => set({ pendingCanvasAssetId: undefined }),
  setPhotoDocument: (document, options) => set((state) => {
    const now = Date.now();
    const last = photoHistoryStamp;
    // Same document, still moving, within the window: keep extending the step already on the stack.
    const continuing = options?.coalesce !== false
      && last !== undefined && last.assetId === document.assetId && now - last.at < 700;
    photoHistoryStamp = { assetId: document.assetId, at: now };
    const stamped = { ...document, updatedAt: new Date().toISOString() };
    return {
      past: continuing ? state.past : [...state.past.slice(-49), state.project],
      future: continuing ? state.future : [],
      project: {
        ...state.project,
        updatedAt: stamped.updatedAt,
        photoEdits: { ...state.project.photoEdits, [document.assetId]: stamped },
      },
    };
  }),
  setClipVfxDocument: (clipId, document, options) => set((state) => {
    const now = Date.now();
    const last = photoHistoryStamp;
    const key = `clip:${clipId}`;
    const continuing = options?.coalesce !== false && last !== undefined && last.assetId === key && now - last.at < 700;
    photoHistoryStamp = { assetId: key, at: now };
    const stamped = { ...document, updatedAt: new Date().toISOString() };
    return {
      past: continuing ? state.past : [...state.past.slice(-49), state.project],
      future: continuing ? state.future : [],
      project: { ...state.project, updatedAt: stamped.updatedAt, clipVfx: { ...state.project.clipVfx, [clipId]: stamped } },
    };
  }),
  setH3DirectorWorkspace: (workspace) => set((state) => {
    const updatedAt = new Date().toISOString();
    const h3Director = { ...workspace, updatedAt };
    // Director typing deliberately is not a timeline undo step. Keep the latest Director document
    // on both history branches so undoing a blade/trim cannot also erase an unrelated shot plan.
    const withDirector = (project: Project): Project => ({ ...project, h3Director });
    return {
      project: { ...withDirector(state.project), updatedAt },
      past: state.past.map(withDirector),
      future: state.future.map(withDirector),
    };
  }),
  setProjectPath: (projectPath) => set({ projectPath }),
  replaceProject: (project, projectPath) => set((state) => ({
    project, projectEpoch: state.projectEpoch + 1,
    projectPath, selectedAssetId: undefined, selectedClipId: undefined, groupSelection: [],
    playhead: 0, playing: false, status: "Opened " + project.name, past: [], future: [],
  })),
  newProject: () => set((state) => ({
    project: createProject(), projectEpoch: state.projectEpoch + 1,
    projectPath: undefined, selectedAssetId: undefined, groupSelection: [],
    selectedClipId: undefined, playhead: 0, playing: false, status: "New project", past: [], future: [],
  })),
  addAssets: (assets, options) => set((state) => ({
    past: [...state.past.slice(-49), state.project],
    future: [],
    project: {
      ...state.project,
      updatedAt: new Date().toISOString(),
      assets: [...state.project.assets.filter((old) => !assets.some((item) => item.path === old.path)), ...assets],
    },
    ...(options?.keepSelection ? {} : {
      selectedAssetId: assets.at(-1)?.id ?? state.selectedAssetId,
      selectedClipId: undefined,
      status: "Imported " + assets.length + " media file" + (assets.length === 1 ? "" : "s"),
    }),
  })),
  /**
   * Rename an asset for display. The file on disk keeps its path: clips, generation inputs and the
   * saved project all address media by path, so renaming the file would be a rename plus a
   * migration, and a half-finished one would break a project. What the user is actually complaining
   * about is the label, and this is the label.
   */
  renameAsset: (id, name) => set((state) => {
    const clean = name.trim();
    const asset = state.project.assets.find((item) => item.id === id);
    if (!asset || !clean || clean === asset.name) return {};
    const previous = asset.name;
    return {
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        assets: state.project.assets.map((item) => item.id === id ? { ...item, name: clean } : item),
        // A clip still carrying the asset's old name was never retitled on its own, so it follows
        // the asset. One the user has already named keeps the name they chose.
        sequences: state.project.sequences.map((sequence) => ({
          ...sequence,
          tracks: sequence.tracks.map((track) => ({
            ...track,
            clips: track.clips.map((clip) => clip.assetId === id && clip.name === previous
              ? { ...clip, name: clean } : clip),
          })),
        })),
      },
      status: `Renamed to ${clean}`,
    };
  }),
  removeAsset: (id) => set((state) => ({
    past: [...state.past.slice(-49), state.project],
    future: [],
    project: {
      ...state.project,
      updatedAt: new Date().toISOString(),
      assets: state.project.assets.filter((asset) => asset.id !== id),
      sequences: state.project.sequences.map((sequence) => ({
        ...sequence,
        tracks: sequence.tracks.map((track) => ({ ...track, clips: track.clips.filter((clip) => clip.assetId !== id) })),
      })),
    },
    selectedAssetId: state.selectedAssetId === id ? undefined : state.selectedAssetId,
    selectedClipId: undefined,
    status: "Removed media and its timeline instances",
  })),
  setSelectedAsset: (selectedAssetId) => set({ selectedAssetId, selectedClipId: undefined, playing: false }),
  addAssetToTimeline: (assetId) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const asset = state.project.assets.find((item) => item.id === assetId);
    const targetTrackId = asset && isAudioAsset(asset) ? "a1" : "v1";
    const track = sequence.tracks.find((item) => item.id === targetTrackId);
    const end = track?.clips.reduce((value, clip) => Math.max(value, seconds(clip.start) + timelineClipDuration(clip)), 0) ?? 0;
    state.addAssetToTimelineAt(assetId, end, targetTrackId);
  },
  addAssetToTimelineAt: (assetId, requestedStart, requestedTrackId = "v1") => {
    const state = get();
    const asset = state.project.assets.find((item) => item.id === assetId);
    const sequence = currentSequence(state.project);
    const audioAsset = asset ? isAudioAsset(asset) : false;
    const expectedTrackKind = audioAsset ? "audio" : "video";
    const track = sequence.tracks.find((item) => item.id === requestedTrackId && item.kind === expectedTrackKind);
    const compatible = asset && (audioAsset ? track?.kind === "audio" : ["image", "video", "generated"].includes(asset.kind) && track?.kind === "video");
    if (!asset || !track || !compatible || track.locked) {
      if (track?.locked) {
        set({ status: track.name + " is locked" });
        return;
      }
      set({ status: audioAsset ? "Audio clips can only be placed on A1" : "Video clips can only be placed on V1" });
      return;
    }
    const points = track.clips.flatMap((clip) => [seconds(clip.start), seconds(clip.start) + timelineClipDuration(clip)]);
    points.push(state.playhead, 0);
    const start = state.snapping ? snapTime(requestedStart, points, 10 / state.zoom).value : Math.max(0, requestedStart);
    const clip = newClip(asset, track.id, start, palette[track.clips.length % palette.length]);
    clip.positionX = sequence.width / 2;
    clip.positionY = sequence.height / 2;
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item,
          tracks: item.tracks.map((candidate) => candidate.id === track.id ? { ...candidate, clips: [...candidate.clips, clip] } : candidate),
        } : item),
      },
      selectedClipId: clip.id,
      selectedAssetId: asset.id,
      playhead: start,
      playing: false,
      status: "Placed " + asset.name + " at " + start.toFixed(2) + "s",
    });
  },
  addSfxToTimeline: (asset, requestedStart, settings) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const audioTracks = sequence.tracks.filter((track) => track.kind === "audio");
    let target = audioTracks.find((track) => /\b(sfx|effects?|foley)\b/i.test(track.name));
    const newTrack = !target;
    if (!target) {
      const ids = new Set(sequence.tracks.map((track) => track.id));
      let index = 2;
      while (ids.has(`a${index}`)) index += 1;
      target = audioTrack(`a${index}`, `A${index} · SFX`);
    }
    if (target.locked) {
      set({ status: target.name + " is locked" });
      return;
    }
    const start = Math.max(0, requestedStart);
    const clip = newClip(asset, target.id, start, palette[target.clips.length % palette.length]);
    const trimIn = Math.max(0, Math.min(settings.trimInSeconds, seconds(asset.duration) - 0.04));
    const trimOut = Math.max(0, Math.min(settings.trimOutSeconds, seconds(asset.duration) - trimIn - 0.04));
    clip.sourceIn = fromSeconds(trimIn);
    clip.sourceOut = fromSeconds(seconds(asset.duration) - trimOut);
    clip.volumeDb = Math.max(-60, Math.min(24, settings.gainDb));
    clip.audio = {
      pan: Math.max(-1, Math.min(1, settings.pan)),
      fadeInSeconds: Math.max(0, Math.min(settings.fadeInSeconds, timelineClipDuration(clip))),
      fadeOutSeconds: Math.max(0, Math.min(settings.fadeOutSeconds, timelineClipDuration(clip))),
      preserveTail: settings.preserveTail,
    };
    const assets = state.project.assets.some((item) => item.id === asset.id)
      ? state.project.assets
      : [...state.project.assets, asset];
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        assets,
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item,
          tracks: newTrack
            ? [...item.tracks, { ...target!, clips: [clip] }]
            : item.tracks.map((track) => track.id === target!.id ? { ...track, clips: [...track.clips, clip] } : track),
        } : item),
      },
      selectedClipId: clip.id,
      selectedAssetId: asset.id,
      playhead: start,
      playing: false,
      status: `Added ${asset.name} to ${target.name} at ${start.toFixed(2)}s`,
    });
  },
  // MiniMax-H3 returns picture and sound as two files. They have to land on the SAME frame or the
  // generated audio drifts against its own video, so this deliberately bypasses `addAssetToTimelineAt`
  // -- that snaps each placement against its own track's edges, and V1 and A1 do not share edges.
  addGeneratedAvPair: (videoAssetId, audioAssetId) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const video = state.project.assets.find((item) => item.id === videoAssetId);
    const audio = state.project.assets.find((item) => item.id === audioAssetId);
    const videoTrack = sequence.tracks.find((item) => item.id === "v1");
    const audioTrackTarget = sequence.tracks.find((item) => item.id === "a1");
    if (!video || !audio || !videoTrack || !audioTrackTarget) {
      set({ status: "Could not place the generated clip: V1 or A1 is missing" });
      return;
    }
    if (videoTrack.locked || audioTrackTarget.locked) {
      set({ status: (videoTrack.locked ? videoTrack.name : audioTrackTarget.name) + " is locked" });
      return;
    }
    const end = videoTrack.clips.reduce(
      (value, clip) => Math.max(value, seconds(clip.start) + timelineClipDuration(clip)), 0);
    const videoClip = newClip(video, videoTrack.id, end, palette[videoTrack.clips.length % palette.length]);
    videoClip.positionX = sequence.width / 2;
    videoClip.positionY = sequence.height / 2;
    const audioClip = newClip(audio, audioTrackTarget.id, end, palette[audioTrackTarget.clips.length % palette.length]);
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item,
          tracks: item.tracks.map((candidate) =>
            candidate.id === videoTrack.id ? { ...candidate, clips: [...candidate.clips, videoClip] }
            : candidate.id === audioTrackTarget.id ? { ...candidate, clips: [...candidate.clips, audioClip] }
            : candidate),
        } : item),
      },
      selectedClipId: videoClip.id,
      selectedAssetId: video.id,
      playhead: end,
      playing: false,
      status: "Placed " + video.name + " on V1 with its soundtrack on A1 at " + end.toFixed(2) + "s",
    });
  },
  insertGeneratedAvPairAfterClip: (clipId, videoAssetId, audioAssetId) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const anchor = sequence.tracks.find((item) => item.id === "v1")?.clips.find((item) => item.id === clipId);
    const video = state.project.assets.find((item) => item.id === videoAssetId);
    const audio = state.project.assets.find((item) => item.id === audioAssetId);
    const videoTrack = sequence.tracks.find((item) => item.id === "v1");
    const audioTrackTarget = sequence.tracks.find((item) => item.id === "a1");
    if (!anchor || !video || !audio || !videoTrack || !audioTrackTarget) {
      set({ status: "Could not insert the generated H3 pair: the anchor, V1, or A1 is missing" });
      return;
    }
    if (videoTrack.locked || audioTrackTarget.locked) {
      set({ status: (videoTrack.locked ? videoTrack.name : audioTrackTarget.name) + " is locked" });
      return;
    }
    const start = seconds(anchor.start) + timelineClipDuration(anchor);
    const lockedRippleTrack = sequence.tracks.find((item) => item.locked
      && item.clips.some((existing) => seconds(existing.start) >= start - 0.0001));
    if (lockedRippleTrack) {
      set({ status: lockedRippleTrack.name + " is locked and blocks the ripple insert" });
      return;
    }
    const span = seconds(video.duration);
    // Anything already-empty between the anchor and what follows is room this insert can use.
    const shift = rippleShiftFor(sequence.tracks, start, span);
    const videoClip = newClip(video, videoTrack.id, start, palette[videoTrack.clips.length % palette.length]);
    videoClip.positionX = anchor.positionX ?? sequence.width / 2;
    videoClip.positionY = anchor.positionY ?? sequence.height / 2;
    const audioClip = newClip(audio, audioTrackTarget.id, start, palette[audioTrackTarget.clips.length % palette.length]);
    // H3's independent WAV can differ from the video probe by a few samples. The generated pair is
    // one editorial unit, so cap the timeline audio to the picture duration instead of leaving a
    // sliver that collides with the next clip.
    audioClip.sourceOut = fromSeconds(Math.min(seconds(audio.duration), span));
    set({
      past: [...state.past.slice(-49), state.project], future: [],
      project: {
        ...state.project, updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item,
          // This is a real ripple insert, so every downstream track moves with the picture. Moving
          // V1 alone would leave the following clip's A1 soundtrack behind and break synchronization.
          tracks: item.tracks.map((candidate) => {
            const shifted = candidate.clips.map((existing) => seconds(existing.start) >= start - 0.0001
              ? { ...existing, start: fromSeconds(seconds(existing.start) + shift) } : existing);
            return candidate.id === videoTrack.id ? { ...candidate, clips: [...shifted, videoClip] }
              : candidate.id === audioTrackTarget.id ? { ...candidate, clips: [...shifted, audioClip] }
              : { ...candidate, clips: shifted };
          }),
        } : item),
      },
      selectedClipId: videoClip.id, selectedAssetId: video.id, playhead: start, playing: false,
      status: "Inserted " + video.name + " with synchronized audio after " + anchor.name
        + (shift < span - 0.0001
          ? shift <= 0.0001 ? " — it fitted the existing gap"
            : ` — the following clips moved ${shift.toFixed(2)}s to make room`
          : ""),
    });
  },
  addGraphic: (template) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const targetTrackId = template === "caption" ? "c1" : "g1";
    const track = sequence.tracks.find((item) => item.id === targetTrackId && item.kind === (template === "caption" ? "caption" : "graphics"));
    if (!track || track.locked) {
      set({ status: track?.locked ? track.name + " is locked" : targetTrackId.toUpperCase() + " track is missing" });
      return;
    }
    const id = crypto.randomUUID();
    const start = Math.max(0, state.playhead);
    const clip: TimelineClip = {
      id,
      assetId: (template === "caption" ? "caption:" : "graphic:") + id,
      name: graphicDefaults(template).text,
      trackId: track.id,
      start: fromSeconds(start),
      sourceIn: fromSeconds(0),
      sourceOut: fromSeconds(5),
      enabled: true,
      color: template === "caption" ? "#4971a8" : template === "title" ? "#a34f78" : template === "bug" ? "#bb5544" : "#8a5fa5",
      opacity: 1,
      volumeDb: 0,
      positionX: sequence.width / 2,
      positionY: sequence.height / 2,
      scale: 100,
      rotation: 0,
      blendMode: "normal",
      transitionIn: "none",
      transitionDuration: 0,
      transitionOut: "none",
      transitionOutDuration: 0,
      effects: { ...DEFAULT_EFFECTS },
      graphic: graphicDefaults(template),
      speed: [{ at: fromSeconds(0), rate: 1, easing: "linear" }],
    };
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item,
          tracks: item.tracks.map((candidate) => candidate.id === track.id ? { ...candidate, clips: [...candidate.clips, clip] } : candidate),
        } : item),
      },
      selectedClipId: clip.id,
      playhead: start,
      playing: false,
      status: "Added " + template.replace("-", " ") + " on " + targetTrackId.toUpperCase(),
    });
  },
  addGraphicToTimeline: (graphic, startSeconds, durationSeconds) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const targetTrackId = graphic.template === "caption" ? "c1" : "g1";
    const kind = graphic.template === "caption" ? "caption" : "graphics";
    const track = sequence.tracks.find((item) => item.id === targetTrackId && item.kind === kind);
    if (!track || track.locked) {
      set({ status: track?.locked ? track.name + " is locked" : targetTrackId.toUpperCase() + " track is missing" });
      return;
    }
    const id = crypto.randomUUID();
    const start = Math.max(0, Number.isFinite(startSeconds) ? startSeconds : 0);
    const duration = Math.max(0.1, Math.min(3600, Number.isFinite(durationSeconds) ? durationSeconds : 5));
    const clip: TimelineClip = {
      id, assetId: (kind === "caption" ? "caption:" : "graphic:") + id,
      name: graphic.text.trim().slice(0, 80) || "Animated text", trackId: track.id,
      start: fromSeconds(start), sourceIn: fromSeconds(0), sourceOut: fromSeconds(duration),
      enabled: true, color: graphic.template === "caption" ? "#4971a8" : graphic.template === "title" ? "#a34f78" : graphic.template === "bug" ? "#bb5544" : "#8a5fa5",
      opacity: 1, volumeDb: 0, positionX: sequence.width / 2, positionY: sequence.height / 2,
      scale: 100, rotation: 0, blendMode: "normal", transitionIn: "none", transitionDuration: 0,
      transitionOut: "none", transitionOutDuration: 0, effects: { ...DEFAULT_EFFECTS }, graphic: { ...graphic },
      speed: [{ at: fromSeconds(0), rate: 1, easing: "linear" }],
    };
    set({
      past: [...state.past.slice(-49), state.project], future: [],
      project: {
        ...state.project, updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item, tracks: item.tracks.map((candidate) => candidate.id === track.id
            ? { ...candidate, clips: [...candidate.clips, clip] } : candidate),
        } : item),
      },
      selectedClipId: clip.id, playhead: start, playing: false,
      status: "Added editable animated text on " + targetTrackId.toUpperCase(),
    });
  },
  addCaptionCues: (cues) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const track = sequence.tracks.find((item) => item.id === "c1" && item.kind === "caption");
    if (!track || track.locked) {
      set({ status: track?.locked ? track.name + " is locked" : "C1 caption track is missing" });
      return;
    }
    const valid = cues.filter((cue) => Number.isFinite(cue.startSeconds) && Number.isFinite(cue.endSeconds)
      && cue.endSeconds > cue.startSeconds && cue.text.trim()).slice(0, 10_000);
    if (!valid.length) {
      set({ status: "The subtitle file contained no valid timed cues" });
      return;
    }
    const added = valid.map((cue): TimelineClip => {
      const id = crypto.randomUUID();
      const graphic = { ...graphicDefaults("caption"), text: cue.text.trim() };
      return {
        id, assetId: "caption:" + id, name: cue.text.trim().slice(0, 80), trackId: track.id,
        start: fromSeconds(Math.max(0, cue.startSeconds)), sourceIn: fromSeconds(0),
        sourceOut: fromSeconds(cue.endSeconds - cue.startSeconds), enabled: true, color: "#4971a8",
        opacity: 1, volumeDb: 0, positionX: sequence.width / 2, positionY: sequence.height / 2,
        scale: 100, rotation: 0, blendMode: "normal", transitionIn: "none", transitionDuration: 0,
        transitionOut: "none", transitionOutDuration: 0,
        effects: { ...DEFAULT_EFFECTS }, graphic, speed: [{ at: fromSeconds(0), rate: 1, easing: "linear" }],
      };
    });
    set({
      past: [...state.past.slice(-49), state.project], future: [],
      project: {
        ...state.project, updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item, tracks: item.tracks.map((candidate) => candidate.id === track.id
            ? { ...candidate, clips: [...candidate.clips, ...added] } : candidate),
        } : item),
      },
      selectedClipId: added[0].id, playhead: seconds(added[0].start), playing: false,
      status: "Imported " + added.length + " caption cue" + (added.length === 1 ? "" : "s") + " on C1",
    });
  },
  replaceSelectedClipWithAsset: (asset) => {
    const state = get();
    if (!state.selectedClipId) return;
    state.replaceClipWithAsset(state.selectedClipId, asset);
  },
  replaceClipWithAsset: (clipId, asset) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const selected = sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === clipId);
    const track = sequence.tracks.find((item) => item.id === selected?.trackId);
    if (!selected || track?.locked || selected.trackId !== "v1") {
      set({ status: track?.locked ? track.name + " is locked" : "AI results can only replace V1 media clips" });
      return;
    }
    // Full-source AI operations such as Retake preserve existing source trims. Range-only results
    // use replaceClipRangeWithAsset instead of entering this path.
    const timelineDuration = timelineClipDuration(selected);
    const rate = clipPlaybackRate(selected);
    const available = seconds(asset.duration);
    const sourceSpan = Math.min(available, Math.max(0.04, timelineDuration * rate));
    const preserveTrims = available + 0.001 >= seconds(selected.sourceOut);
    set({
      past: [...state.past.slice(-49), state.project], future: [],
      project: {
        ...state.project, updatedAt: new Date().toISOString(),
        assets: state.project.assets.some((item) => item.id === asset.id) ? state.project.assets : [...state.project.assets, asset],
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item, tracks: item.tracks.map((candidate) => ({
            ...candidate, clips: candidate.clips.map((clip) => clip.id === selected.id ? {
              ...clip, assetId: asset.id, name: asset.name,
              // A matte made from the old picture does not fit the new one: keep the points, drop it.
              matte: clip.matte ? { ...clip.matte, cache: undefined } : undefined,
              sourceIn: preserveTrims ? selected.sourceIn : fromSeconds(0),
              sourceOut: preserveTrims ? selected.sourceOut : fromSeconds(sourceSpan),
            } : clip),
          })),
        } : item),
      },
      selectedClipId: selected.id, selectedAssetId: asset.id, playing: false,
      status: "Replaced " + selected.name + " with " + asset.name,
    });
  },
  // Swap the media behind a clip while keeping every timing decision the edit already carries:
  // start, source in/out, speed, transitions. Deliberately NOT `replaceClipWithAsset`, which is for
  // AI results of a different length and therefore rebuilds the trims — and which refuses anything
  // outside V1. A denoise pass is frame-for-frame identical to its source and audio clips live on
  // A1/A2, so it needs neither the retrim nor the V1 restriction.
  repointClipToAsset: (clipId, asset, status) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const selected = sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === clipId);
    const track = sequence.tracks.find((item) => item.id === selected?.trackId);
    if (!selected || !track) {
      set({ status: "Select a timeline clip first" });
      return;
    }
    if (track.locked) {
      set({ status: track.name + " is locked" });
      return;
    }
    set({
      past: [...state.past.slice(-49), state.project], future: [],
      project: {
        ...state.project, updatedAt: new Date().toISOString(),
        assets: state.project.assets.some((item) => item.id === asset.id)
          ? state.project.assets : [...state.project.assets, asset],
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item, tracks: item.tracks.map((candidate) => ({
            ...candidate, clips: candidate.clips.map((clip) => clip.id === selected.id
              ? { ...clip, assetId: asset.id } : clip),
          })),
        } : item),
      },
      selectedClipId: selected.id, selectedAssetId: asset.id,
      status: status ?? "Repointed " + selected.name + " at " + asset.name,
    });
  },
  replaceClipRangeWithAsset: (clipId, startOffsetSeconds, endOffsetSeconds, asset) => {
    const state = get();
    try {
      const result = replaceClipRange(
        state.project,
        state.project.activeSequenceId,
        clipId,
        startOffsetSeconds,
        endOffsetSeconds,
        asset,
      );
      set({
        past: [...state.past.slice(-49), state.project],
        future: [],
        project: result.project,
        selectedClipId: result.selectedClipId,
        selectedAssetId: asset.id,
        playing: false,
        status: result.status,
      });
    } catch (error) {
      set({ status: error instanceof Error ? error.message : String(error) });
    }
  },
  insertAssetAfterClip: (clipId, asset) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const anchor = sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === clipId);
    const track = sequence.tracks.find((item) => item.id === anchor?.trackId);
    if (!anchor || !track || track.locked) {
      set({ status: track?.locked ? track.name + " is locked" : "Select a timeline clip to insert after" });
      return;
    }
    const start = seconds(anchor.start) + timelineClipDuration(anchor);
    const span = seconds(asset.duration);
    const shift = rippleShiftFor([track], start, span);
    const clip: TimelineClip = {
      ...newClip(asset, track.id, start, palette[track.clips.length % palette.length]),
      positionX: anchor.positionX, positionY: anchor.positionY,
    };
    set({
      past: [...state.past.slice(-49), state.project], future: [],
      project: {
        ...state.project, updatedAt: new Date().toISOString(),
        assets: state.project.assets.some((item) => item.id === asset.id) ? state.project.assets : [...state.project.assets, asset],
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item, tracks: item.tracks.map((candidate) => candidate.id === track.id ? {
            ...candidate,
            clips: [
              ...candidate.clips.map((existing) => existing.id !== anchor.id && seconds(existing.start) >= start - 0.0001
                ? { ...existing, start: fromSeconds(seconds(existing.start) + shift) } : existing),
              clip,
            ],
          } : candidate),
        } : item),
      },
      selectedClipId: clip.id, selectedAssetId: asset.id, playhead: start, playing: false,
      status: "Inserted " + asset.name + " after " + anchor.name,
    });
  },
  setSelectedClip: (selectedClipId) => {
    const state = get();
    if (selectedClipId !== state.selectedClipId && state.maskPoints.length) {
      set({ maskPoints: [], maskPicking: false });
    }
    if (state.mattePickClipId && state.mattePickClipId !== selectedClipId) set({ mattePickClipId: undefined });
    const clip = currentSequence(state.project).tracks.flatMap((track) => track.clips).find((item) => item.id === selectedClipId);
    set({
      selectedClipId,
      selectedAssetId: clip?.assetId ?? state.selectedAssetId,
      playhead: clip ? seconds(clip.start) : state.playhead,
      playing: false,
    });
  },
  setPlayhead: (playhead) => set({ playhead: Math.max(0, playhead) }),
  setPlaying: (playing) => set({ playing }),
  togglePlaying: () => set((state) => ({ playing: !state.playing })),
  toggleSnapping: () => set((state) => ({ snapping: !state.snapping, status: "Snapping " + (state.snapping ? "off" : "on") })),
  setZoom: (zoom) => set({ zoom: Math.min(140, Math.max(28, zoom)) }),
  setTool: (tool) => set({
    tool,
    groupSelection: tool === "track" ? get().groupSelection : [],
    status: tool === "select" ? "Selection tool" : tool === "blade" ? "Blade tool — click a clip to cut at that point"
      : "Track select — click a lane to select everything from that point forward",
  }),
  selectTrackFrom: (trackId, atSeconds) => {
    const state = get();
    const track = currentSequence(state.project).tracks.find((item) => item.id === trackId);
    if (!track) return;
    const matched = track.clips
      .filter((clip) => seconds(clip.start) + timelineClipDuration(clip) > atSeconds)
      .sort((a, b) => seconds(a.start) - seconds(b.start));
    set({
      groupSelection: matched.map((clip) => clip.id),
      selectedClipId: matched[0]?.id,
      status: matched.length ? "Selected " + matched.length + " clip" + (matched.length === 1 ? "" : "s") + " on " + track.name
        : "No clips at or after that point on " + track.name,
    });
  },
  clearGroupSelection: () => set({ groupSelection: [] }),
  addMaskPoint: (point) => set((state) => ({
    maskPoints: [...state.maskPoints, point],
    status: (point.include ? "Added tracking point " : "Added exclude point ")
      + (state.maskPoints.length + 1),
  })),
  clearMaskPoints: () => set({ maskPoints: [], status: "Tracking points cleared" }),
  setMattePick: (mattePickClipId) => set({
    mattePickClipId, maskPicking: false, playing: false,
    status: mattePickClipId ? "Click the subject in the monitor to keep it · shift-click what to remove · any frame of the clip" : "Point picking off",
  }),
  setMaskPicking: (maskPicking) => set({
    mattePickClipId: maskPicking ? undefined : get().mattePickClipId,
    maskPicking,
    status: maskPicking ? "Click the subject in the monitor · shift-click to exclude" : "Point picking off",
  }),
  setAssetPeaks: (assetId, peaks) => set((state) => ({ peaks: { ...state.peaks, [assetId]: peaks } })),
  setAssetThumbnail: (assetId, dataUrl) => set((state) => ({ thumbnails: { ...state.thumbnails, [assetId]: dataUrl } })),
  setActivePanel: (activePanel) => set({ activePanel }),
  setSequenceSettingsOpen: (sequenceSettingsOpen) => set({ sequenceSettingsOpen }),
  setClipEditorOpen: (clipEditorOpen) => set({ clipEditorOpen }),
  // Not on the undo stack: an export setting is a property of how the project is delivered, not an
  // edit to it, and putting it there would make Ctrl+Z step back through encoder choices. It does
  // need to live on the PROJECT rather than in the panel's own React state -- as component state it
  // vanished the moment the panel unmounted, and the menu's "Render timeline..." never saw it at
  // all, so an upscale set in the panel silently rendered at 1x.
  setExportSettings: (settings) => set((state) => {
    const sequence = currentSequence(state.project);
    return {
      project: {
        ...state.project,
        sequences: state.project.sequences.map(
          (item) => item.id === sequence.id ? { ...item, exportSettings: settings } : item,
        ),
      },
    };
  }),
  updateSequenceSettings: (settings) => set((state) => {
    const sequence = currentSequence(state.project);
    const width = Math.max(64, Math.min(8192, Math.round(settings.width / 2) * 2));
    const height = Math.max(64, Math.min(8192, Math.round(settings.height / 2) * 2));
    const frameRate = settings.frameRate.numerator > 0 && settings.frameRate.denominator > 0
      ? settings.frameRate : sequence.frameRate;
    return {
      past: [...state.past.slice(-49), state.project],
      future: [],
      sequenceSettingsOpen: false,
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item,
          name: settings.name.trim() || item.name,
          width,
          height,
          frameRate,
          sampleRate: [44_100, 48_000, 96_000].includes(settings.sampleRate) ? settings.sampleRate : item.sampleRate,
          conform: settings.conform ?? item.conform,
        } : item),
      },
      status: `Sequence updated to ${width} x ${height}`,
    };
  }),
  moveClip: (clipId, requestedStart, targetTrackId) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const moving = sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === clipId);
    const movingTrack = sequence.tracks.find((track) => track.id === moving?.trackId);
    if (!moving || movingTrack?.locked) {
      if (movingTrack?.locked) set({ status: movingTrack.name + " is locked" });
      return;
    }
    // A cross-track move only makes sense for a single clip and only between tracks of the same
    // kind -- picture into picture, audio into audio. Anything else is rejected with a reason
    // rather than silently ignored, because a clip that neither moves nor explains itself reads
    // as the editor being broken.
    const destination = targetTrackId && targetTrackId !== moving.trackId
      ? sequence.tracks.find((track) => track.id === targetTrackId)
      : undefined;
    if (targetTrackId && targetTrackId !== moving.trackId) {
      const sourceKind = movingTrack?.kind;
      if (!destination) return;
      if (destination.locked) { set({ status: destination.name + " is locked" }); return; }
      if (destination.kind !== sourceKind) {
        set({ status: "A " + (sourceKind ?? "video") + " clip cannot go on " + destination.name });
        return;
      }
      if (state.groupSelection.length > 1 && state.groupSelection.includes(clipId)) {
        set({ status: "Move one clip at a time when changing tracks" });
        return;
      }
    }
    const group = state.groupSelection.includes(clipId) ? state.groupSelection : [clipId];
    const points = sequence.tracks.flatMap((track) =>
      track.clips.filter((clip) => !group.includes(clip.id))
        .flatMap((clip) => [seconds(clip.start), seconds(clip.start) + timelineClipDuration(clip)]),
    );
    points.push(state.playhead, 0);
    const start = state.snapping ? snapTime(requestedStart, points, 10 / state.zoom).value : Math.max(0, requestedStart);
    // Group moves are rigid: every member shifts by the same delta and none may cross zero.
    const rawDelta = start - seconds(moving.start);
    const earliest = Math.min(...group.map((id) =>
      sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === id)).filter(Boolean)
      .map((clip) => seconds(clip!.start)));
    const delta = Math.max(rawDelta, -earliest);
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => ({
          ...item,
          tracks: item.tracks.map((track) => {
            const shifted = (clip: TimelineClip) => group.includes(clip.id)
              ? { ...clip, start: fromSeconds(Math.max(0, seconds(clip.start) + delta)) } : clip;
            if (!destination) return { ...track, clips: track.clips.map(shifted) };
            // Lift out of the old track...
            if (track.id === moving.trackId) {
              return { ...track, clips: track.clips.filter((clip) => clip.id !== clipId) };
            }
            // ...and land in the new one, carrying its new trackId so every consumer agrees.
            if (track.id === destination.id) {
              const landed = { ...shifted(moving), trackId: destination.id };
              return { ...track, clips: [...track.clips, landed].sort((a, b) => seconds(a.start) - seconds(b.start)) };
            }
            return { ...track, clips: track.clips.map(shifted) };
          }),
        })),
      },
      status: destination ? "Moved to " + destination.name.split(" · ")[0]
        : group.length > 1 ? "Moved " + group.length + " clips"
        : state.snapping && start !== requestedStart ? "Clip snapped to edit point" : "Clip moved",
    });
  },
  trimClip: (clipId, edge, deltaSeconds) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const source = sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === clipId);
    const sourceTrack = sequence.tracks.find((track) => track.id === source?.trackId);
    if (!source || sourceTrack?.locked) {
      if (sourceTrack?.locked) set({ status: sourceTrack.name + " is locked" });
      return;
    }
    const frame = sequence.frameRate.denominator / sequence.frameRate.numerator;
    const rate = clipPlaybackRate(source);
    let next = source;
    if (edge === "left") {
      const maximum = timelineClipDuration(source) - frame;
      const delta = Math.max(-seconds(source.sourceIn) / rate, Math.min(deltaSeconds, maximum));
      next = {
        ...source,
        start: fromSeconds(Math.max(0, seconds(source.start) + delta)),
        sourceIn: fromSeconds(seconds(source.sourceIn) + delta * rate),
      };
    } else {
      const asset = state.project.assets.find((item) => item.id === source.assetId);
      const maximum = asset ? (seconds(asset.duration) - seconds(source.sourceOut)) / rate : Number.POSITIVE_INFINITY;
      const minimum = -(timelineClipDuration(source) - frame);
      const delta = Math.max(minimum, Math.min(deltaSeconds, maximum));
      next = { ...source, sourceOut: fromSeconds(seconds(source.sourceOut) + delta * rate) };
    }
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => ({
          ...item,
          tracks: item.tracks.map((track) => ({
            ...track,
            clips: track.clips.map((clip) => clip.id === clipId ? next : clip),
          })),
        })),
      },
      status: "Trimmed " + edge + " edge",
    });
  },
  deleteClip: (clipId) => set((state) => {
    const sourceTrack = currentSequence(state.project).tracks.find((track) =>
      track.clips.some((clip) => clip.id === clipId),
    );
    if (sourceTrack?.locked) return { status: sourceTrack.name + " is locked" };
    return {
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((sequence) => ({
          ...sequence,
          tracks: sequence.tracks.map((track) => ({ ...track, clips: track.clips.filter((clip) => clip.id !== clipId) })),
        })),
      },
      selectedClipId: state.selectedClipId === clipId ? undefined : state.selectedClipId,
      groupSelection: state.groupSelection.filter((id) => id !== clipId),
      playing: false,
      status: "Clip deleted",
    };
  }),
  rippleDeleteClip: (clipId) => set((state) => {
    const sequence = currentSequence(state.project);
    const source = sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === clipId);
    if (!source) return {};
    const sourceTrack = sequence.tracks.find((track) => track.id === source.trackId);
    if (sourceTrack?.locked) return { status: sourceTrack.name + " is locked" };
    const removedStart = seconds(source.start);
    const removedDuration = timelineClipDuration(source);
    return {
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => ({
          ...item,
          tracks: item.tracks.map((track) => track.id === source.trackId ? {
            ...track,
            clips: track.clips.filter((clip) => clip.id !== clipId).map((clip) =>
              seconds(clip.start) > removedStart ? { ...clip, start: fromSeconds(Math.max(removedStart, seconds(clip.start) - removedDuration)) } : clip,
            ),
          } : track),
        })),
      },
      selectedClipId: undefined,
      groupSelection: [],
      playing: false,
      status: "Ripple-deleted clip and closed the gap",
    };
  }),
  duplicateClip: (clipId) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const source = sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === clipId);
    const sourceTrack = sequence.tracks.find((track) => track.id === source?.trackId);
    if (!source || sourceTrack?.locked) {
      if (sourceTrack?.locked) set({ status: sourceTrack.name + " is locked" });
      return;
    }
    const copy = {
      ...source,
      id: crypto.randomUUID(),
      name: source.name + " copy",
      start: fromSeconds(seconds(source.start) + timelineClipDuration(source)),
    };
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => ({
          ...item,
          tracks: item.tracks.map((track) => track.id === source.trackId ? { ...track, clips: [...track.clips, copy] } : track),
        })),
      },
      selectedClipId: copy.id,
      playhead: seconds(copy.start),
      status: "Clip duplicated",
    });
  },
  splitClip: (clipId, atSeconds) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const source = sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === clipId);
    const sourceTrack = sequence.tracks.find((track) => track.id === source?.trackId);
    if (!source || sourceTrack?.locked) {
      if (sourceTrack?.locked) set({ status: sourceTrack.name + " is locked" });
      return;
    }
    const offset = atSeconds - seconds(source.start);
    const duration = timelineClipDuration(source);
    if (offset <= 0.04 || offset >= duration - 0.04) {
      set({ status: "Move the playhead inside the clip before splitting" });
      return;
    }
    const cutSource = seconds(source.sourceIn) + offset * clipPlaybackRate(source);
    // The cut is hard: the left half keeps the incoming transition, the right half keeps the outgoing one.
    const right: TimelineClip = {
      ...source,
      id: crypto.randomUUID(),
      start: fromSeconds(atSeconds),
      sourceIn: fromSeconds(cutSource),
      transitionIn: "none",
      transitionDuration: source.transitionDuration ?? 0.5,
    };
    const left: TimelineClip = {
      ...source,
      sourceOut: fromSeconds(cutSource),
      transitionOut: "none",
      transitionOutDuration: source.transitionOutDuration ?? 0.5,
    };
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => ({
          ...item,
          tracks: item.tracks.map((track) => track.id === source.trackId ? {
            ...track,
            clips: track.clips.flatMap((clip) => clip.id === source.id ? [left, right] : [clip]),
          } : track),
        })),
      },
      selectedClipId: right.id,
      status: "Clip split at playhead",
    });
  },
  removeSourceRangeFromClip: (clipId, requestedStart, requestedEnd) => {
    const state = get();
    const sequence = currentSequence(state.project);
    const source = sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === clipId);
    const sourceTrack = sequence.tracks.find((track) => track.id === source?.trackId);
    if (!source || sourceTrack?.locked || source.graphic) {
      set({ status: sourceTrack?.locked ? sourceTrack.name + " is locked" : "Pause removal needs a media clip" });
      return;
    }
    const sourceDuration = clipDuration(source.sourceIn, source.sourceOut);
    const frame = sequence.frameRate.denominator / sequence.frameRate.numerator;
    const rate = clipPlaybackRate(source);
    const sourceFrame = frame * rate;
    const startOffset = Math.max(0, Math.min(requestedStart, sourceDuration));
    const endOffset = Math.max(startOffset, Math.min(requestedEnd, sourceDuration));
    if (endOffset - startOffset < sourceFrame) {
      set({ status: "Detected pause is shorter than one timeline frame" });
      return;
    }
    const sourceStart = seconds(source.sourceIn);
    const clipStart = seconds(source.start);
    const removedTimelineDuration = (endOffset - startOffset) / rate;
    const cutEndOnTimeline = clipStart + endOffset / rate;
    const replacements: TimelineClip[] = [];
    if (startOffset >= sourceFrame) {
      replacements.push({
        ...source,
        sourceOut: fromSeconds(sourceStart + startOffset),
        transitionOut: sourceDuration - endOffset >= sourceFrame ? "none" : source.transitionOut,
      });
    }
    if (sourceDuration - endOffset >= sourceFrame) {
      replacements.push({
        ...source,
        id: replacements.length ? crypto.randomUUID() : source.id,
        start: fromSeconds(clipStart + startOffset / rate),
        sourceIn: fromSeconds(sourceStart + endOffset),
        transitionIn: replacements.length ? "none" : source.transitionIn,
      });
    }
    set({
      past: [...state.past.slice(-49), state.project], future: [],
      project: {
        ...state.project, updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => item.id === sequence.id ? {
          ...item,
          tracks: item.tracks.map((track) => ({
            ...track,
            clips: track.clips.flatMap((clip) => clip.id === source.id ? replacements : [{
              ...clip,
              start: seconds(clip.start) >= cutEndOnTimeline
                ? fromSeconds(Math.max(0, seconds(clip.start) - removedTimelineDuration)) : clip.start,
            }]),
          })),
        } : item),
      },
      selectedClipId: replacements.at(-1)?.id,
      playhead: clipStart + startOffset / rate,
      playing: false,
      status: "Removed " + removedTimelineDuration.toFixed(2) + "s pause and ripple-closed the timeline",
    });
  },
  applyTransitionToAllCuts: (transitionId, duration) => set((state) => {
    let touched = 0;
    const project = {
      ...state.project,
      updatedAt: new Date().toISOString(),
      sequences: state.project.sequences.map((sequence) => ({
        ...sequence,
        tracks: sequence.tracks.map((track) => {
          // Only the picture tracks carry visual transitions; captions and audio are untouched.
          if (track.kind !== "video" || track.locked) return track;
          const ordered = track.clips.slice().sort((a, b) => seconds(a.start) - seconds(b.start));
          const order = new Map(ordered.map((clip, index) => [clip.id, index]));
          const last = ordered.length - 1;
          if (last > 0) touched += last;
          return {
            ...track,
            clips: track.clips.map((clip) => {
              const index = order.get(clip.id) ?? 0;
              return {
                ...clip,
                transitionIn: index === 0 ? "none" : transitionId,
                transitionDuration: duration,
                transitionOut: index === last ? "none" : transitionId,
                transitionOutDuration: duration,
              };
            }),
          };
        }),
      })),
    };
    return {
      past: [...state.past.slice(-49), state.project],
      future: [],
      project,
      status: touched ? "Applied the transition to " + touched + " cut" + (touched === 1 ? "" : "s")
        : "There are no cuts on a picture track to apply a transition to",
    };
  }),
  updateOverlay: (trackId, patch) => set((state) => ({
    past: [...state.past.slice(-49), state.project],
    future: [],
    project: {
      ...state.project,
      updatedAt: new Date().toISOString(),
      sequences: state.project.sequences.map((sequence) => ({
        ...sequence,
        tracks: sequence.tracks.map((track) => track.id === trackId
          ? { ...track, overlay: { ...(track.overlay ?? DEFAULT_OVERLAY), ...patch } } : track),
      })),
    },
    status: "Overlay placement updated",
  })),
  addAudioTrack: (name) => {
    const state = get();
    const sequence = currentSequence(state.project);
    // Number from the existing audio tracks so stems can be added repeatedly without collisions.
    const existing = sequence.tracks.filter((track) => track.kind === "audio").length;
    const id = "a" + (existing + 1);
    const track = audioTrack(id, name ?? "A" + (existing + 1) + " · Audio");
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => {
          if (item.id !== sequence.id) return item;
          // Keep audio grouped: insert after the last audio track rather than at the end.
          // (findLastIndex would need ES2023, so scan manually.)
          let insertAt = item.tracks.length;
          for (let index = item.tracks.length - 1; index >= 0; index -= 1) {
            if (item.tracks[index].kind === "audio") { insertAt = index + 1; break; }
          }
          return {
            ...item,
            tracks: [...item.tracks.slice(0, insertAt), track, ...item.tracks.slice(insertAt)],
          };
        }),
      },
      status: "Added " + track.name,
    });
    return id;
  },
  addOverlayTrack: () => {
    const state = get();
    const sequence = currentSequence(state.project);
    const existing = sequence.tracks.filter((track) => track.kind === "video").length;
    const id = "v" + (existing + 1);
    const track: Track = {
      id, name: "V" + (existing + 1) + " · Overlay", kind: "video",
      locked: false, muted: false, hidden: false, clips: [], overlay: { ...DEFAULT_OVERLAY },
    };
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        // Overlays composite in track order, so a new one goes on top: before the existing ones.
        sequences: state.project.sequences.map((item) => item.id === sequence.id
          ? { ...item, tracks: [track, ...item.tracks] } : item),
      },
      status: "Added " + track.name,
    });
    return id;
  },
  removeTrack: (trackId) => set((state) => {
    const sequence = currentSequence(state.project);
    // V1 is the programme and C1 anchors captions; neither can be removed.
    if (trackId === "v1" || trackId === "c1") return { status: "That track cannot be removed" };
    const track = sequence.tracks.find((item) => item.id === trackId);
    if (!track) return {};
    if (sequence.tracks.filter((item) => item.kind === "audio").length <= 1 && track.kind === "audio") {
      return { status: "At least one audio track is required" };
    }
    return {
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((item) => ({
          ...item, tracks: item.tracks.filter((candidate) => candidate.id !== trackId),
        })),
      },
      selectedClipId: undefined,
      status: "Removed " + track.name,
    };
  }),
  updateTrack: (trackId, patch) => set((state) => ({
    past: [...state.past.slice(-49), state.project],
    future: [],
    project: {
      ...state.project,
      updatedAt: new Date().toISOString(),
      sequences: state.project.sequences.map((sequence) => ({
        ...sequence,
        tracks: sequence.tracks.map((track) => track.id === trackId ? { ...track, ...patch } : track),
      })),
    },
    status: "Track settings updated",
  })),
  updateSelectedClip: (patch) => {
    const state = get();
    if (!state.selectedClipId) return;
    const selectedTrack = currentSequence(state.project).tracks.find((track) =>
      track.clips.some((clip) => clip.id === state.selectedClipId),
    );
    if (selectedTrack?.locked) {
      set({ status: selectedTrack.name + " is locked" });
      return;
    }
    set({
      past: [...state.past.slice(-49), state.project],
      future: [],
      project: {
        ...state.project,
        updatedAt: new Date().toISOString(),
        sequences: state.project.sequences.map((sequence) => ({
          ...sequence,
          tracks: sequence.tracks.map((track) => ({
            ...track,
            clips: track.clips.map((clip) => clip.id === state.selectedClipId ? { ...clip, ...patch } : clip),
          })),
        })),
      },
      status: "Clip settings updated",
    });
  },
  upsertClipVfxLayer: (clipId, layer) => set((state) => ({
    past: [...state.past.slice(-49), state.project], future: [],
    project: {
      ...state.project, updatedAt: new Date().toISOString(),
      sequences: state.project.sequences.map((sequence) => ({
        ...sequence,
        tracks: sequence.tracks.map((track) => ({
          ...track,
          clips: track.clips.map((clip) => {
            if (clip.id !== clipId) return clip;
            const layers = clip.vfxLayers ?? [];
            const exists = layers.some((candidate) => candidate.id === layer.id);
            return { ...clip, vfxLayers: exists
              ? layers.map((candidate) => candidate.id === layer.id ? layer : candidate)
              : [...layers, layer] };
          }),
        })),
      })),
    },
    status: `VFX layer “${layer.name}” saved non-destructively`,
  })),
  setClipMatte: (clipId, matte, status) => set((state) => ({
    past: [...state.past.slice(-49), state.project], future: [],
    project: {
      ...state.project, updatedAt: new Date().toISOString(),
      sequences: state.project.sequences.map((sequence) => ({
        ...sequence,
        tracks: sequence.tracks.map((track) => ({ ...track,
          clips: track.clips.map((clip) => clip.id === clipId ? { ...clip, matte } : clip),
        })),
      })),
    },
    status: status ?? (matte ? "Background removal updated" : "Background removal removed"),
  })),
  removeClipVfxLayer: (clipId, layerId) => set((state) => ({
    past: [...state.past.slice(-49), state.project], future: [],
    project: {
      ...state.project, updatedAt: new Date().toISOString(),
      sequences: state.project.sequences.map((sequence) => ({
        ...sequence,
        tracks: sequence.tracks.map((track) => ({ ...track,
          clips: track.clips.map((clip) => clip.id === clipId
            ? { ...clip, vfxLayers: (clip.vfxLayers ?? []).filter((layer) => layer.id !== layerId) }
            : clip),
        })),
      })),
    },
    status: "VFX layer removed",
  })),
  undo: () => set((state) => {
    const project = state.past.at(-1);
    if (!project) return {};
    return {
      project,
      past: state.past.slice(0, -1),
      future: [state.project, ...state.future.slice(0, 49)],
      selectedClipId: undefined,
      groupSelection: [],
      playing: false,
      status: "Undo",
    };
  }),
  redo: () => set((state) => {
    const project = state.future[0];
    if (!project) return {};
    return {
      project,
      past: [...state.past.slice(-49), state.project],
      future: state.future.slice(1),
      selectedClipId: undefined,
      groupSelection: [],
      playing: false,
      status: "Redo",
    };
  }),
}));
