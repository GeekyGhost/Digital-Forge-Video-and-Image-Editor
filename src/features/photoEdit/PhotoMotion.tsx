/**
 * VFX Phase 4 panels: point tracks, attach-to-track, the trail node, the sprite emitter and depth.
 *
 * Presentational only. PhotoEdit owns the document, the frame store and the workers; these render
 * controls and report intent through callbacks, like the rest of PhotoPanels.
 */

import { Crosshair, Link2, Mountain, RefreshCw, Sparkles, Trash2, Unlink } from "lucide-react";
import type { EmitterGenerator, LayerAttach, PhotoLayer, PointTrack, TrailNode } from "../../domain/photoDocument";
import type { MotionStatus } from "../../services/motionService";
import { Slider } from "./PhotoPanels";

export type TrackEngine = "lucas-kanade" | "cotracker3";

const SOURCE_NAMES: Record<PointTrack["source"], string> = {
  "lucas-kanade": "built-in", cotracker3: "CoTracker3", "sam2-centroid": "SAM 2 centroid", manual: "manual",
};

function TrackPicker({ tracks, value, allowNone, onChange }: {
  tracks: PointTrack[]; value?: string; allowNone?: string; onChange: (id: string | undefined) => void;
}) {
  return <select value={value ?? ""} onChange={(event) => onChange(event.target.value || undefined)}>
    {allowNone !== undefined && <option value="">{allowNone}</option>}
    {tracks.map((track) => <option key={track.id} value={track.id}>{track.name}</option>)}
  </select>;
}

/**
 * Tracks: make one by clicking the plate (and attach the selected layer in the same action), list
 * them, attach the active layer, and compute the plate's depth for occlusion.
 */
export function TracksPanel({
  tracks, plate, activeLayer, frame, engine, motion, busy, tracking, sam2Sources, occluded,
  onEngine, onStartTrack, onSam2Centroid, onRename, onDelete, onAttach, onComputeDepth, onOcclude,
}: {
  tracks: PointTrack[];
  /** The media layer new tracks are measured on (the plate). */
  plate?: PhotoLayer;
  activeLayer?: PhotoLayer;
  frame: number;
  engine: TrackEngine;
  motion?: MotionStatus;
  busy?: string;
  tracking: boolean;
  /** Sequence mask entries on the plate (SAM 2 mattes), for the centroid source. */
  sam2Sources: Array<{ label: string; cachePath: string; frameCount: number; first?: number }>;
  /** The active layer already has a depth entry against the plate. */
  occluded: boolean;
  onEngine: (engine: TrackEngine) => void;
  onStartTrack: () => void;
  onSam2Centroid: (source: { label: string; cachePath: string; frameCount: number; first?: number }) => void;
  onRename: (id: string, name: string) => void;
  onDelete: (id: string) => void;
  onAttach: (attach: LayerAttach | undefined, label: string, coalesce?: boolean) => void;
  onComputeDepth: () => void;
  onOcclude: () => void;
}) {
  const attachable = activeLayer && plate && activeLayer.id !== plate.id && !activeLayer.generator;
  const attach = activeLayer?.attach;
  return <>
    {!plate && <p className="photo-note">Add a video (Media bin → Video) to track points on it.</p>}
    {plate && <>
      <label className="studio-field"><span>Engine</span>
        <select value={engine} onChange={(event) => onEngine(event.target.value as TrackEngine)}>
          <option value="lucas-kanade">Built-in (Lucas-Kanade, no model)</option>
          <option value="cotracker3" disabled={!motion?.cotracker}>
            CoTracker3{motion?.cotracker ? "" : " (run Setup-Motion.bat)"}
          </option>
        </select>
      </label>
      {engine === "cotracker3" && <p className="photo-note">{motion?.cotrackerLicence ?? "CoTracker3: CC-BY-NC 4.0, non-commercial use only"}</p>}
      <button className={`photo-inline wide${tracking ? " selected" : ""}`} type="button" disabled={Boolean(busy)} onClick={onStartTrack}>
        {busy === "Tracking" ? <><RefreshCw className="spin" size={12} /> Tracking…</> : <><Crosshair size={12} /> {tracking ? "Click the point on the picture…" : attachable ? "Track a point + attach the selected layer" : "Track a point"}</>}
      </button>
      <p className="photo-note">
        Measured on <strong>{plate.name}</strong> from its first frame to its last, starting where you click at
        frame {frame}.{attachable ? ` ${activeLayer!.name} is attached in the same step, its base on the point.` : ""}
      </p>
      {sam2Sources.length > 0 && <label className="studio-field"><span>From a SAM 2 matte</span>
        <select value="" disabled={Boolean(busy)} onChange={(event) => {
          const picked = sam2Sources[Number(event.target.value)];
          if (picked) onSam2Centroid(picked);
        }}>
          <option value="">Track the matte's centre…</option>
          {sam2Sources.map((source, index) => <option key={source.cachePath} value={index}>{source.label} · {source.frameCount} frames</option>)}
        </select>
      </label>}
    </>}

    <div className="photo-nodes">
      {tracks.map((track) => {
        const seen = track.samples.filter((sample) => sample[2]).length;
        return <div key={track.id} className={`photo-node${attach?.trackId === track.id ? " selected" : ""}`}>
          <div className="photo-node-main">
            <i><Crosshair size={11} /></i>
            <span>
              <input className="photo-rename" value={track.name} onChange={(event) => onRename(track.id, event.target.value)} />
              <small>{SOURCE_NAMES[track.source]} · frames {track.first}–{track.first + track.samples.length - 1} · seen {seen}/{track.samples.length}</small>
            </span>
          </div>
          {attachable && <i className="photo-node-act" title={`Attach ${activeLayer!.name} here, anchored at frame ${frame}`}
            onClick={() => onAttach({ trackId: track.id, anchorFrame: frame, lag: attach?.lag }, `Attached to ${track.name}`)}><Link2 size={12} /></i>}
          <i className="photo-node-act" title="Delete the track" onClick={() => onDelete(track.id)}><Trash2 size={12} /></i>
        </div>;
      })}
      {!tracks.length && <p className="photo-note">No tracks yet.</p>}
    </div>

    {attachable && attach && <>
      <h4>{activeLayer!.name} follows</h4>
      <label className="studio-field"><span>Track</span>
        <TrackPicker tracks={tracks} value={attach.trackId} onChange={(id) => id && onAttach({ ...attach, trackId: id }, "Track changed")} />
      </label>
      <Slider label="Lag" value={attach.lag ?? 0} min={0} max={24} step={1} format={(value) => `${value} frames`}
        onChange={(value) => onAttach({ ...attach, lag: value }, "Attach lag", true)} />
      <div className="photo-buttons">
        <button type="button" title="Keep the layer where it is now and measure motion from this frame"
          onClick={() => onAttach({ ...attach, anchorFrame: frame }, `Anchored at frame ${frame}`)}>Anchor at frame {frame}</button>
        <button type="button" onClick={() => onAttach(undefined, "Detached")}><Unlink size={12} /> Detach</button>
      </div>
      <p className="photo-note">
        Its position is where it sits at frame {attach.anchorFrame}; at any other frame it moves by
        however far the track has moved since, {attach.lag ? `${attach.lag} frames late` : "on time"}.
      </p>
    </>}

    {plate?.media?.kind === "video" && <>
      <h4>Depth</h4>
      <div className="photo-buttons">
        <button type="button" disabled={Boolean(busy) || !motion?.depth} title={motion?.depth ? motion.depthLicence : motion?.detail}
          onClick={onComputeDepth}>
          {busy === "Computing depth" ? <><RefreshCw className="spin" size={12} /> Computing…</> : <><Mountain size={12} /> {plate.media.depth ? "Recompute depth" : "Compute depth"}</>}
        </button>
        {attachable && plate.media.depth && <button type="button" disabled={occluded}
          title={occluded ? "Already occluded: edit the depth entry in Masks" : `Hide ${activeLayer!.name} where ${plate.name} is nearer`}
          onClick={onOcclude}>Occlude by depth</button>}
      </div>
      <p className="photo-note">
        {plate.media.depth ? `${plate.media.depth.frameCount} frames of depth. ` : ""}
        Video Depth Anything Small, once per clip (cached). Occlusion is a Depth entry in the
        selected layer's mask: set its plane and softness in Masks.
        {!motion?.depth ? " Needs Setup-Motion.bat." : ""}
      </p>
    </>}
  </>;
}

export function TrailEditor({ node, tracks, onChange }: {
  node: TrailNode; tracks: PointTrack[];
  onChange: (over: Partial<TrailNode>, label: string, coalesce?: boolean) => void;
}) {
  return <>
    <label className="studio-field"><span>Track</span>
      <TrackPicker tracks={tracks} value={node.trackId} allowNone={tracks.some((track) => track.id === node.trackId) ? undefined : "Missing track"}
        onChange={(id) => id && onChange({ trackId: id }, "Trail track", false)} />
    </label>
    <Slider label="Lag at the far end" value={node.maxLag} min={0} max={48} step={1} format={(value) => `${value} frames`}
      onChange={(value) => onChange({ maxLag: value }, "Trail lag", true)} />
    <Slider label="Strength" value={node.strength} min={0} max={3} step={0.05} format={(value) => value.toFixed(2)}
      onChange={(value) => onChange({ strength: value }, "Trail strength", true)} />
    <p className="photo-note">
      The layer's base moves with the track; its far end (the top of the picture) follows where the
      track was up to {node.maxLag} frames ago, so smoke bends behind the motion. Works after any
      attachment, before masks.
    </p>
  </>;
}

export function EmitterEditor({ generator, tracks, onChange }: {
  generator: EmitterGenerator; tracks: PointTrack[];
  onChange: (generator: EmitterGenerator, label: string, coalesce?: boolean) => void;
}) {
  const set = (over: Partial<EmitterGenerator>, label: string, coalesce = true) => onChange({ ...generator, ...over }, label, coalesce);
  const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
  return <>
    <label className="studio-field"><span>Spawn at</span>
      <TrackPicker tracks={tracks} value={generator.trackId} allowNone="A fixed point (below)"
        onChange={(id) => set({ trackId: id }, "Emitter source", false)} />
    </label>
    {!generator.trackId && <>
      <Slider label="Point X" value={generator.x} min={0} max={1} step={0.001} format={percent} onChange={(value) => set({ x: value }, "Emitter point")} />
      <Slider label="Point Y" value={generator.y} min={0} max={1} step={0.001} format={percent} onChange={(value) => set({ y: value }, "Emitter point")} />
    </>}
    <Slider label="Rate" value={generator.rate} min={0.02} max={4} step={0.01} format={(value) => `${value.toFixed(2)} / frame`}
      onChange={(value) => set({ rate: value }, "Emitter rate")} />
    <Slider label="Lifetime" value={generator.lifetime} min={1} max={240} step={1} format={(value) => `${value} frames`}
      onChange={(value) => set({ lifetime: Math.round(value) }, "Emitter lifetime")} />
    <Slider label="Velocity X" value={generator.velocityX} min={-0.03} max={0.03} step={0.0005} format={percent}
      onChange={(value) => set({ velocityX: value }, "Emitter velocity")} />
    <Slider label="Velocity Y" value={generator.velocityY} min={-0.03} max={0.03} step={0.0005} format={percent}
      onChange={(value) => set({ velocityY: value }, "Emitter velocity")} />
    <Slider label="Spread" value={generator.spread} min={0} max={0.02} step={0.0002} format={percent}
      onChange={(value) => set({ spread: value }, "Emitter spread")} />
    <Slider label="Scale at birth" value={generator.scaleStart} min={0.02} max={3} step={0.01} format={(value) => `${value.toFixed(2)}×`}
      onChange={(value) => set({ scaleStart: value }, "Emitter scale")} />
    <Slider label="Scale at death" value={generator.scaleEnd} min={0.02} max={3} step={0.01} format={(value) => `${value.toFixed(2)}×`}
      onChange={(value) => set({ scaleEnd: value }, "Emitter scale")} />
    <Slider label="Opacity at birth" value={generator.opacityStart} min={0} max={1} step={0.01} format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => set({ opacityStart: value }, "Emitter opacity")} />
    <Slider label="Opacity at death" value={generator.opacityEnd} min={0} max={1} step={0.01} format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => set({ opacityEnd: value }, "Emitter opacity")} />
    <Slider label="Spin" value={generator.spin} min={0} max={180} step={1} format={(value) => `±${value}°`}
      onChange={(value) => set({ spin: value }, "Emitter spin")} />
    <label className="studio-field"><span>Seed</span>
      <input type="number" value={generator.seed} onChange={(event) => set({ seed: Math.max(0, Math.round(Number(event.target.value) || 0)) }, "Emitter seed", false)} />
    </label>
    <p className="photo-note">
      <Sparkles size={11} /> {generator.element.path.split(/[\\/]/).pop()} · each particle plays it from its first frame. Every
      frame is a pure function of the seed and the frame number, so scrubbing, playback and the saved
      file agree. Velocity and spread are fractions of the canvas per frame; scale is canvas pixels per
      element pixel.
    </p>
  </>;
}
