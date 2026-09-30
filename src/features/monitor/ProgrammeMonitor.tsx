/**
 * VFX Phase 7: the timeline monitor on the compositor.
 *
 * Every frame is the programme document (domain/programme.ts) rendered by the same engine the export
 * uses (render/programmeEngine.ts): conform, transitions, overlays, titles, clip effects and live
 * mask VFX are one picture drawn one way, so the monitor shows what the file will contain.
 *
 * Clock: while playing, wall time drives the programme and the audio follows it; each display frame
 * shows the newest programme frame whose media has decoded (a slow frame is dropped, never waited
 * for, so sound and picture stay together). Playback renders at a reduced size (the frame store's
 * proxies where a clip has one); paused, the frame is drawn at the monitor's full resolution.
 */

import { convertFileSrc } from "@tauri-apps/api/core";
import { Pause, Play, SkipBack, SkipForward } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { buildProgramme, programmeClipFrame, programmeFrames, programmePick, programmePlace, type Programme, type ProgrammeAudio } from "../../domain/programme";
import { defaultClipMatte } from "../../domain/clipMatte";
import { formatTimecode } from "../../domain/time";
import { ProgrammeEngine } from "../../render/programmeEngine";
import { readCompositorFlag } from "../../render/compositorFlag";
import type { VfxMode } from "../../domain/programmeVfx";
import { useEditor } from "../../store";

const LEGACY_KEY = "aiv-legacy-monitor";
const VFX_KEY = "aiv-legacy-vfx";

/** Whether the old (CSS layers) monitor is chosen on this machine. Off by default. */
export function readLegacyMonitor(): boolean {
  try { return window.localStorage.getItem(LEGACY_KEY) === "1"; } catch { return false; }
}
export function writeLegacyMonitor(on: boolean): void {
  try { window.localStorage.setItem(LEGACY_KEY, on ? "1" : "0"); } catch { /* not remembered */ }
}
/** Whether mask VFX are shown from their rendered files rather than live. Off by default. */
export function readLegacyVfx(): VfxMode {
  try { return window.localStorage.getItem(VFX_KEY) === "1" ? "legacy" : "live"; } catch { return "live"; }
}
export function writeLegacyVfx(mode: VfxMode): void {
  try { window.localStorage.setItem(VFX_KEY, mode === "legacy" ? "1" : "0"); } catch { /* not remembered */ }
}

/** Linear gain of one audio entry at programme time `t` (fades, the programme's head/tail dips). */
export function audioGainAt(entry: ProgrammeAudio, programme: Programme, t: number): number {
  const local = t - entry.start;
  let gain = 10 ** (Math.max(-60, Math.min(24, entry.volumeDb)) / 20);
  if (entry.volumeDb <= -60) return 0;
  if (entry.fadeIn > 0 && local < entry.fadeIn) gain *= Math.max(0, local / entry.fadeIn);
  if (entry.fadeOut > 0 && local > entry.duration - entry.fadeOut) gain *= Math.max(0, (entry.duration - local) / entry.fadeOut);
  if (programme.headFade && t < programme.headFade.duration) gain *= t / programme.headFade.duration;
  if (programme.tailFade && t > programme.duration - programme.tailFade.duration) gain *= Math.max(0, (programme.duration - t) / programme.tailFade.duration);
  return gain;
}

/** One audible programme entry, following the monitor's clock like a DAW track follows its transport. */
function ProgrammeAudioLayer({ entry, programme, playhead, playing }: {
  entry: ProgrammeAudio; programme: Programme; playhead: number; playing: boolean;
}) {
  const ref = useRef<HTMLAudioElement>(null);
  const sync = (audio: HTMLAudioElement) => {
    const target = entry.sourceIn + Math.max(0, playhead - entry.start) * entry.rate;
    if (Math.abs(audio.currentTime - target) > 0.12) audio.currentTime = target;
    audio.playbackRate = entry.rate;
    // An audio element cannot amplify: positive gain saturates here, the export applies it in full.
    audio.volume = Math.min(1, audioGainAt(entry, programme, playhead));
    if (playing && audio.paused) void audio.play().catch(() => undefined);
    if (!playing && !audio.paused) audio.pause();
  };
  useEffect(() => {
    const audio = ref.current;
    if (audio && audio.readyState >= 1) sync(audio);
  });
  useEffect(() => () => ref.current?.pause(), []);
  return <audio ref={ref} src={convertFileSrc(entry.path)} preload="auto" onLoadedMetadata={(event) => sync(event.currentTarget)} />;
}

export function ProgrammeMonitor({ toggle }: { toggle: ReactNode }) {
  const {
    playhead, playing, setPlaying, setPlayhead, project, setStatus, maskPicking, maskPoints, addMaskPoint,
    mattePickClipId, setClipMatte,
  } = useEditor();
  const sequence = project.sequences.find((item) => item.id === project.activeSequenceId)!;
  const fps = sequence.frameRate.numerator / Math.max(1, sequence.frameRate.denominator);
  const programme = useMemo(() => buildProgramme(project), [project]);
  const total = programme ? programmeFrames(programme) : 0;
  const duration = programme?.duration ?? 0;
  const [vfxMode, setVfxMode] = useState<VfxMode>(readLegacyVfx);
  const engine = useRef<ProgrammeEngine | undefined>(undefined);
  engine.current ??= new ProgrammeEngine({ lookahead: 12 });
  useEffect(() => {
    engine.current ??= new ProgrammeEngine({ lookahead: 12 });
    return () => { engine.current?.dispose(); engine.current = undefined; };
  }, []);

  const frameBox = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [box, setBox] = useState({ width: 640, height: 360 });
  useEffect(() => {
    const element = frameBox.current;
    if (!element) return;
    const observer = new ResizeObserver(() => setBox({ width: element.clientWidth, height: element.clientHeight }));
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  const aspect = sequence.width / Math.max(1, sequence.height);
  /** Render size: the monitor's pixels (never above the sequence), halved while playing. */
  const sizeFor = (moving: boolean) => {
    const dpr = window.devicePixelRatio || 1;
    let width = Math.min(sequence.width, Math.round(box.width * dpr));
    if (moving) width = Math.max(Math.min(480, sequence.width), Math.round(width / 2));
    width = Math.max(2, width - (width % 2));
    return { width, height: Math.max(2, Math.round(width / aspect / 2) * 2) };
  };

  const frameOf = (t: number) => Math.max(0, Math.min(Math.max(0, total - 1), Math.floor(t * fps + 1e-6)));
  const [info, setInfo] = useState("");
  const drawn = useRef<{ frame: number; key: string; programme?: Programme }>({ frame: -1, key: "" });
  const reported = useRef(new Set<string>());

  /** Draw `frame` if its media is loaded (else load it and draw when it arrives, if still wanted). */
  const draw = (frame: number, moving: boolean, wanted: () => boolean): boolean => {
    const held = engine.current;
    const canvas = canvasRef.current;
    if (!held || !canvas || !programme) return false;
    const { width, height } = sizeFor(moving);
    const key = `${frame}|${width}|${vfxMode}`;
    if (drawn.current.key === key && drawn.current.frame === frame && drawn.current.programme === programme) return true;
    const document = held.document(programme, frame, vfxMode);
    const needs = held.needs(document, frame);
    if (!held.ready(needs, moving)) {
      void held.prepare(needs, moving, true).then(() => { if (wanted()) draw(frame, moving, wanted); });
      return false;
    }
    const result = held.compose(document, frame, width, height, moving, readCompositorFlag());
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    canvas.getContext("2d")!.putImageData(new ImageData(result.pixels, width, height), 0, 0);
    drawn.current = { frame, key, programme };
    const skipped = result.report.skipped.map((item) => item.reason);
    setInfo(`${held.compositor.lastEngine.toUpperCase()} · ${held.compositor.lastMilliseconds.toFixed(0)} ms${skipped.length ? ` · waiting: ${[...new Set(skipped)].join(", ")}` : ""}`);
    for (const [path, reason] of held.failures) {
      if (reported.current.has(path)) continue;
      reported.current.add(path);
      setStatus(`The monitor could not read ${path.split(/[\\/]/).pop()} (${reason}). The Legacy monitor can still preview it.`);
    }
    return true;
  };

  // Paused: draw the playhead's frame at full monitor resolution whenever anything changes.
  const clock = useRef(playhead);
  useEffect(() => {
    if (playing) return;
    clock.current = playhead;
    const frame = frameOf(playhead);
    draw(frame, false, () => !playingRef.current && frameOf(clock.current) === frame);
  });
  const playingRef = useRef(playing);
  playingRef.current = playing;

  // Playing: wall clock, newest ready frame, prefetch ahead, playhead published ~12 times a second.
  useEffect(() => {
    if (!playing || !programme) return;
    if (playhead >= duration - 1 / fps) setPlayhead(0);
    const from = playhead >= duration - 1 / fps ? 0 : playhead;
    const started = performance.now();
    let published = 0;
    let handle = 0;
    const tick = (now: number) => {
      const t = from + (now - started) / 1000;
      if (t >= duration) {
        setPlayhead(duration);
        setPlaying(false);
        setStatus("Timeline playback complete");
        return;
      }
      clock.current = t;
      const frame = frameOf(t);
      const held = engine.current;
      if (held) {
        // Prefetch the next frames so decoding runs ahead of the clock.
        for (let ahead = 1; ahead <= 6 && frame + ahead < total; ahead++) {
          const document = held.document(programme, frame + ahead, vfxMode);
          const needs = held.needs(document, frame + ahead);
          if (!held.ready(needs, true)) { void held.prepare(needs, true); break; }
        }
      }
      draw(frame, true, () => false);
      if (now - published > 80) { published = now; setPlayhead(t); }
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing, programme, vfxMode]);

  const edits = useMemo(() => [...new Set((programme?.clips ?? []).flatMap((item) => [item.start, item.start + item.duration]))].sort((a, b) => a - b), [programme]);
  const jump = (direction: -1 | 1) => {
    const candidates = direction < 0 ? edits.filter((value) => value < playhead - 0.01) : edits.filter((value) => value > playhead + 0.01);
    setPlayhead(direction < 0 ? candidates.at(-1) ?? 0 : candidates[0] ?? playhead);
  };
  const audible = (programme?.audio ?? []).filter((entry) => playhead >= entry.start - 0.25 && playhead < entry.start + entry.duration);

  // Background removal: clicks pick the subject in the clip's own source (any frame of the clip).
  const pickClip = mattePickClipId ? sequence.tracks.flatMap((track) => track.clips).find((clip) => clip.id === mattePickClipId) : undefined;
  const picking = maskPicking || Boolean(pickClip);
  const pickFrame = frameOf(playhead);
  const mattePoints = useMemo(() => {
    if (!programme || !pickClip?.matte || playing) return [];
    const shown = programmeClipFrame(programme, pickFrame, pickClip.id);
    return pickClip.matte.points.flatMap((point, index) => {
      if (shown === undefined || point.frame !== shown) return [];
      const at = programmePlace(programme, pickFrame, pickClip.id, point);
      return at ? [{ index, include: point.include, x: at[0], y: at[1] }] : [];
    });
  }, [programme, pickClip, pickFrame, playing]);
  const pickMatte = (x: number, y: number, include: boolean) => {
    if (!programme || !pickClip) return;
    const picked = programmePick(programme, pickFrame, pickClip.id, [x, y]);
    if (!picked) {
      const drawn = programmeClipFrame(programme, pickFrame, pickClip.id) !== undefined;
      setStatus(drawn ?`Click on ${pickClip.name} itself` : `${pickClip.name} is not on screen at the playhead: move the playhead into it`);
      return;
    }
    const matte = pickClip.matte ?? defaultClipMatte();
    const method = matte.method === "birefnet" ? "sam2-birefnet" : matte.method;
    setClipMatte(pickClip.id, { ...matte, method, points: [...matte.points, { frame: picked.frame, x: picked.x, y: picked.y, include }] },
      `${include ? "Keep" : "Remove"} point ${matte.points.length + 1} on source frame ${picked.frame} · run Remove background to apply`);
  };

  return <main className="monitor-column">
    <div className="monitor-header">
      <span>PROGRAM · COMPOSITOR</span>
      <div>
        <span className="monitor-engine" title="Which compositor drew the frame, and how long it took">{info}</span>
        <button type="button" className={"monitor-toggle" + (vfxMode === "legacy" ? " selected" : "")}
          title="Show mask VFX from their rendered files instead of rendering them live"
          onClick={() => { const next = vfxMode === "legacy" ? "live" : "legacy"; writeLegacyVfx(next); setVfxMode(next); drawn.current.key = ""; }}>
          VFX: {vfxMode === "legacy" ? "rendered files" : "live"}
        </button>
        {toggle}
      </div>
    </div>
    <div className="monitor"><div ref={frameBox} className={"preview-frame" + (picking ? " picking" : "")} style={{ aspectRatio: `${sequence.width} / ${sequence.height}` }}
      onClick={(event) => {
        if (!picking) return;
        const rect = event.currentTarget.getBoundingClientRect();
        if (pickClip) {
          pickMatte((event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height, !event.shiftKey);
          return;
        }
        addMaskPoint({
          x: Math.min(1, Math.max(0, (event.clientX - rect.left) / rect.width)),
          y: Math.min(1, Math.max(0, (event.clientY - rect.top) / rect.height)),
          include: !event.shiftKey,
        });
      }}>
      <canvas ref={canvasRef} className="programme-canvas" />
      {!programme?.clips.length && !programme?.overlays.length && <div className="preview-empty"><strong>The timeline is empty</strong><small>Add clips to V1 to preview the programme</small></div>}
      {programme && audible.map((entry) => <ProgrammeAudioLayer key={entry.id} entry={entry} programme={programme} playhead={playhead} playing={playing} />)}
      {maskPoints.map((point, index) => <span key={index}
        className={"mask-point" + (point.include ? "" : " exclude")}
        style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
        title={(point.include ? "Include" : "Exclude") + ` point ${index + 1}`} />)}
      {mattePoints.map((point) => <span key={`m${point.index}`}
        className={"mask-point" + (point.include ? "" : " exclude")}
        style={{ left: `${point.x * 100}%`, top: `${point.y * 100}%` }}
        title={(point.include ? "Keep" : "Remove") + ` point ${point.index + 1}`} />)}
      <div className="safe-area" />
    </div></div>
    <div className="transport">
      <div className="transport-left"><span className="timecode">{formatTimecode(playhead, fps)}</span></div>
      <div className="transport-center"><button className="icon-button" title="Previous edit" onClick={() => jump(-1)}><SkipBack size={18} /></button>
        <button className="play" onClick={() => setPlaying(!playing)} disabled={!duration}>{playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}</button>
        <button className="icon-button" title="Next edit" onClick={() => jump(1)}><SkipForward size={18} /></button></div>
      <div className="transport-right"><span>{formatTimecode(duration, fps)}</span>
        <button onClick={() => { setPlaying(false); setPlayhead(0); }}>Reset</button></div>
    </div>
  </main>;
}
