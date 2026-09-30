import { convertFileSrc } from "@tauri-apps/api/core";
import { useEffect, useMemo, useRef, useState } from "react";
import { clipPlaybackRate, fromSeconds, seconds } from "../../domain/time";
import type { MediaAsset, TimelineClip } from "../../domain/types";
import { useEditor } from "../../store";
import {
  clipWaveform, filename, generateLtxFoley, generateMossSfx, runtimeStatus,
  type AiRuntimeStatus, type SfxGenerationResult,
} from "../../services/modelService";

type SfxEngine = "moss" | "foley";
type SfxTake = {
  id: string;
  engine: SfxEngine;
  asset: MediaAsset;
  prompt: string;
  negativePrompt: string;
  seed: number;
  cueOffset: number;
  gainDb: number;
  pan: number;
  fadeInSeconds: number;
  fadeOutSeconds: number;
  trimInSeconds: number;
  trimOutSeconds: number;
  preserveTail: boolean;
  peaks: number[];
  log: string;
};

export function SfxWorkspace({ clip, asset, rangeStart, rangeEnd, sourceDuration, setRangeStart, setRangeEnd, onRunningChange, onClose }: {
  clip: TimelineClip;
  asset: MediaAsset;
  rangeStart: number;
  rangeEnd: number;
  sourceDuration: number;
  setRangeStart: (value: number) => void;
  setRangeEnd: (value: number) => void;
  onRunningChange: (running: boolean) => void;
  onClose: () => void;
}) {
  const { project, addSfxToTimeline, setBusy, setStatus, busy } = useEditor();
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [engine, setEngine] = useState<SfxEngine>("moss");
  const [runtime, setRuntime] = useState<AiRuntimeStatus>();
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [quality, setQuality] = useState<"draft" | "final">("draft");
  const [durationMode, setDurationMode] = useState<"range" | "custom">("range");
  const [customDuration, setCustomDuration] = useState(2);
  const [density, setDensity] = useState<"minimal" | "normal" | "detailed">("normal");
  const [seed, setSeed] = useState(42);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [startedAt, setStartedAt] = useState(0);
  const [failure, setFailure] = useState<string>();
  const [takes, setTakes] = useState<SfxTake[]>([]);
  const [activeTakeId, setActiveTakeId] = useState<string>();
  const [originalPeaks, setOriginalPeaks] = useState<number[]>([]);
  const [scrub, setScrub] = useState(rangeStart);
  const [loop, setLoop] = useState(true);
  const [originalAudio, setOriginalAudio] = useState(true);
  const [generatedAudio, setGeneratedAudio] = useState(true);

  const rangeDuration = Math.max(0.1, rangeEnd - rangeStart);
  const playbackRate = clipPlaybackRate(clip);
  const programmeRangeDuration = rangeDuration / playbackRate;
  const rangeSourceStart = seconds(clip.sourceIn) + rangeStart;
  const rangeSourceEnd = seconds(clip.sourceIn) + rangeEnd;
  const fps = asset.frameRate ? asset.frameRate.numerator / asset.frameRate.denominator : 24;
  const frame = 1 / Math.max(1, fps);
  const activeTake = takes.find((take) => take.id === activeTakeId);
  const engineReady = engine === "moss" ? Boolean(runtime?.mossSfxReady) : Boolean(runtime?.foleyLoraReady);

  useEffect(() => { void runtimeStatus().then(setRuntime).catch(() => undefined); }, []);
  useEffect(() => {
    setScrub(rangeStart);
    if (!asset.hasAudio) { setOriginalPeaks([]); return; }
    void clipWaveform(asset.path, rangeDuration, rangeSourceStart, 180)
      .then(setOriginalPeaks).catch(() => setOriginalPeaks([]));
  }, [asset.path, asset.hasAudio, rangeDuration, rangeSourceStart, rangeStart]);
  useEffect(() => {
    onRunningChange(running);
    if (!running) return;
    const update = () => setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    update();
    const timer = window.setInterval(update, 500);
    return () => window.clearInterval(timer);
  }, [running, startedAt, onRunningChange]);
  useEffect(() => () => onRunningChange(false), [onRunningChange]);
  useEffect(() => {
    const video = videoRef.current;
    if (video && Math.abs(video.currentTime - (seconds(clip.sourceIn) + scrub)) > 0.04) {
      video.currentTime = seconds(clip.sourceIn) + scrub;
    }
    if (video) video.playbackRate = playbackRate;
  }, [scrub, clip.sourceIn, playbackRate]);
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio || !activeTake) return;
    audio.volume = previewVolume(activeTake, audio.currentTime);
  }, [activeTake?.gainDb, activeTake?.fadeInSeconds, activeTake?.fadeOutSeconds, activeTake?.trimInSeconds, activeTake?.trimOutSeconds, activeTake?.id]);

  const updateTake = (patch: Partial<SfxTake>) => {
    if (!activeTakeId) return;
    setTakes((current) => current.map((take) => take.id === activeTakeId ? { ...take, ...patch } : take));
  };

  const syncGenerated = (shouldPlay: boolean) => {
    const video = videoRef.current;
    const audio = audioRef.current;
    if (!video || !audio || !activeTake || !generatedAudio) { audio?.pause(); return; }
    const relative = (video.currentTime - rangeSourceStart) / playbackRate - activeTake.cueOffset + activeTake.trimInSeconds;
    const playableEnd = seconds(activeTake.asset.duration) - activeTake.trimOutSeconds;
    if (relative < activeTake.trimInSeconds || relative >= playableEnd) { audio.pause(); return; }
    if (Math.abs(audio.currentTime - relative) > 0.08) audio.currentTime = relative;
    audio.volume = previewVolume(activeTake, relative);
    if (shouldPlay && audio.paused) void audio.play().catch(() => undefined);
    if (!shouldPlay) audio.pause();
  };

  const onTimeUpdate = () => {
    const video = videoRef.current;
    if (!video) return;
    const relative = video.currentTime - seconds(clip.sourceIn);
    setScrub(Math.max(0, Math.min(sourceDuration, relative)));
    if (video.currentTime >= rangeSourceEnd - 0.01) {
      if (loop) {
        video.currentTime = rangeSourceStart;
        syncGenerated(true);
      } else {
        video.pause();
        audioRef.current?.pause();
      }
      return;
    }
    syncGenerated(!video.paused);
  };

  const resultAsset = (result: SfxGenerationResult, usedEngine: SfxEngine, usedPrompt: string, usedNegative: string): MediaAsset => ({
    id: crypto.randomUUID(), name: filename(result.outputPath), path: result.outputPath, kind: "audio",
    duration: fromSeconds(Math.max(0.04, result.durationSeconds)), hasAudio: true,
    origin: {
      operation: usedEngine === "moss" ? "moss-sfx" : "ltx-foley",
      model: usedEngine === "moss" ? "MOSS-SoundEffect-v2" : "LTX-2.3 Foley V2A",
      prompt: usedPrompt, seed: result.seed, sourceAssetIds: usedEngine === "foley" ? [asset.id] : undefined,
      parameters: { negativePrompt: usedNegative, quality, density, rangeStart, rangeEnd },
      createdAt: new Date().toISOString(),
    },
  });

  const generate = async () => {
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt || running || busy || !engineReady) return;
    const usedEngine = engine;
    const usedSeed = seed === 0 ? Math.floor(Math.random() * 2_147_483_647) : seed;
    setRunning(true); setBusy(true); setStartedAt(Date.now()); setElapsed(0); setFailure(undefined);
    setStatus(`${usedEngine === "moss" ? "MOSS" : "LTX Foley"} is generating a sound effect…`);
    try {
      const result = usedEngine === "moss"
        ? await generateMossSfx({
            projectId: project.id, prompt: cleanPrompt, negativePrompt,
            durationSeconds: durationMode === "range" ? Math.min(30, programmeRangeDuration) : customDuration,
            seed: usedSeed, quality,
          })
        : await generateLtxFoley({
            projectId: project.id, videoPath: asset.path,
            sourceInSeconds: rangeSourceStart, sourceOutSeconds: rangeSourceEnd,
            playbackRate, prompt: cleanPrompt, negativePrompt,
            seed: usedSeed, density, offload: "cpu", attentionBackend: "native",
          });
      const generated = resultAsset(result, usedEngine, cleanPrompt, negativePrompt);
      const peaks = await clipWaveform(generated.path, result.durationSeconds, 0, 180).catch(() => []);
      const take: SfxTake = {
        id: crypto.randomUUID(), engine: usedEngine, asset: generated, prompt: cleanPrompt, negativePrompt,
        seed: usedSeed, cueOffset: 0, gainDb: 0, pan: 0, fadeInSeconds: 0,
        fadeOutSeconds: Math.min(0.1, result.durationSeconds / 4), trimInSeconds: 0, trimOutSeconds: 0,
        preserveTail: true, peaks, log: result.log,
      };
      setTakes((current) => [...current, take]);
      setActiveTakeId(take.id);
      setStatus(`${usedEngine === "moss" ? "MOSS sound" : "Picture Foley"} ready — audition and add it to the SFX track`);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setFailure(message); setStatus("SFX generation failed: " + message.split("\n")[0]);
    } finally {
      setRunning(false); setBusy(false);
      void runtimeStatus().then(setRuntime).catch(() => undefined);
    }
  };

  const placeTake = () => {
    if (!activeTake) return;
    const requestedStart = seconds(clip.start) + rangeStart / playbackRate + activeTake.cueOffset;
    const clippedPreRoll = Math.max(0, -requestedStart);
    const start = Math.max(0, requestedStart);
    const trimInSeconds = activeTake.trimInSeconds + clippedPreRoll;
    const assetDuration = seconds(activeTake.asset.duration);
    const rangeEndOnTimeline = seconds(clip.start) + rangeStart / playbackRate + programmeRangeDuration;
    const playableUntilRangeEnd = Math.max(0.04, rangeEndOnTimeline - start);
    const rangeTrimOut = Math.max(0, assetDuration - trimInSeconds - playableUntilRangeEnd);
    addSfxToTimeline(activeTake.asset, start, {
      gainDb: activeTake.gainDb, pan: activeTake.pan,
      fadeInSeconds: activeTake.fadeInSeconds, fadeOutSeconds: activeTake.fadeOutSeconds,
      trimInSeconds,
      trimOutSeconds: activeTake.preserveTail ? activeTake.trimOutSeconds : Math.max(activeTake.trimOutSeconds, rangeTrimOut),
      preserveTail: activeTake.preserveTail,
    });
    onClose();
  };

  const laneDuration = useMemo(() => Math.max(programmeRangeDuration, (activeTake?.cueOffset ?? 0) + (activeTake ? seconds(activeTake.asset.duration) : 0), 0.1), [programmeRangeDuration, activeTake]);
  const mossState = runtime?.mossSfxState ?? "checking";

  return <>
    <div className="clip-editor-body sfx-editor-body">
      <div className="clip-editor-preview">
        <video ref={videoRef} src={convertFileSrc(asset.path)} preload="auto" muted={!originalAudio}
          onPlay={() => syncGenerated(true)} onPause={() => syncGenerated(false)}
          onSeeked={() => syncGenerated(!videoRef.current?.paused)} onTimeUpdate={onTimeUpdate} controls />
        <audio ref={audioRef} key={activeTake?.asset.path ?? "no-sfx"} src={activeTake ? convertFileSrc(activeTake.asset.path) : undefined} />
        <div className="sfx-waveforms">
          <div className="sfx-wave-row"><span>Video</span><div className="sfx-wave-track"><div className="sfx-video-range" /></div></div>
          <div className="sfx-wave-row"><span>Original</span><Waveform peaks={originalPeaks} className="original" /></div>
          <div className="sfx-wave-row"><span>Generated</span><div className="sfx-wave-track">
            {activeTake && <div className="sfx-take-wave" style={{
              left: `${Math.max(0, activeTake.cueOffset) / laneDuration * 100}%`,
              width: `${Math.min(100, seconds(activeTake.asset.duration) / laneDuration * 100)}%`,
            }}><WaveBars peaks={activeTake.peaks} /></div>}
            {activeTake && <div className="sfx-cue-line" style={{ left: `${Math.max(0, activeTake.cueOffset) / laneDuration * 100}%` }} />}
          </div></div>
        </div>
        <div className="scrub-strip"><div className="scrub-range" style={{ left: `${rangeStart / Math.max(0.1, sourceDuration) * 100}%`, width: `${rangeDuration / Math.max(0.1, sourceDuration) * 100}%` }} />
          <input type="range" min={0} max={Math.max(0.1, sourceDuration)} step={frame} value={scrub}
            onChange={(event) => { setScrub(Number(event.target.value)); syncGenerated(false); }} /></div>
        <div className="scrub-actions">
          <button onClick={() => setRangeStart(Math.min(scrub, rangeEnd - frame))} disabled={running}>Set in</button>
          <button onClick={() => setRangeEnd(Math.max(scrub, rangeStart + frame))} disabled={running}>Set out</button>
          <button onClick={() => { setRangeStart(0); setRangeEnd(sourceDuration); }} disabled={running}>Whole clip</button>
          <label className="sfx-check"><input type="checkbox" checked={loop} onChange={(event) => setLoop(event.target.checked)} /> Loop range</label>
          <span className="range-readout">{rangeStart.toFixed(2)}s – {rangeEnd.toFixed(2)}s</span>
        </div>
      </div>

      <div className="clip-editor-controls">
        <div className="engine-switch" role="tablist" aria-label="Sound effect model">
          <button role="tab" aria-selected={engine === "moss"} className={engine === "moss" ? "active" : ""} disabled={running}
            onClick={() => setEngine("moss")}>Designed SFX · MOSS</button>
          <button role="tab" aria-selected={engine === "foley"} className={engine === "foley" ? "active" : ""} disabled={running}
            onClick={() => setEngine("foley")}>Picture Foley · LTX</button>
        </div>
        <label className="ai-prompt"><span>{engine === "moss" ? "Describe the sound" : "What should be heard?"}</span>
          <textarea value={prompt} disabled={running} onChange={(event) => setPrompt(event.target.value)}
            placeholder={engine === "moss" ? "A heavy metal door slamming in a concrete bunker…" : "Boots on wet concrete, cloth movement, and the landing impact…"} /></label>
        <label className="ai-prompt"><span>Exclude</span><textarea value={negativePrompt} disabled={running}
          onChange={(event) => setNegativePrompt(event.target.value)} placeholder="voices, music, background ambience…" /></label>
        {engine === "moss" ? <>
          <label className="studio-field"><span>Duration</span><select value={durationMode} disabled={running} onChange={(event) => setDurationMode(event.target.value as "range" | "custom")}>
            <option value="range">Match selected range · {Math.min(30, programmeRangeDuration).toFixed(2)}s</option><option value="custom">Custom</option>
          </select></label>
          {durationMode === "custom" && <label className="studio-field"><span>Custom duration · {customDuration.toFixed(1)}s</span>
            <input type="range" min={0.5} max={30} step={0.5} value={customDuration} disabled={running} onChange={(event) => setCustomDuration(Number(event.target.value))} /></label>}
          <label className="studio-field"><span>Quality</span><select value={quality} disabled={running} onChange={(event) => setQuality(event.target.value as "draft" | "final")}>
            <option value="draft">Draft · 30 steps</option><option value="final">Final · 100 steps</option>
          </select></label>
        </> : <label className="studio-field"><span>Foley amount</span><select value={density} disabled={running} onChange={(event) => setDensity(event.target.value as typeof density)}>
          <option value="minimal">Minimal</option><option value="normal">Normal</option><option value="detailed">Detailed</option>
        </select></label>}
        <label className="studio-field"><span>Seed · 0 randomizes</span><input type="number" value={seed} disabled={running}
          onChange={(event) => setSeed(Number(event.target.value))} /></label>
        <button className="render-button" disabled={running || busy || !engineReady || !prompt.trim()} onClick={() => { void generate(); }}>
          {running ? `Generating · ${Math.floor(elapsed / 60)}:${String(elapsed % 60).padStart(2, "0")}` : engine === "moss" ? "Generate designed SFX" : "Generate picture Foley"}
        </button>
        {!engineReady && <small className="runtime-checks">{engine === "moss"
          ? `MOSS ${mossState}. ${mossState === "dependencies-missing" ? "Run scripts\\install-moss-sfx.ps1." : "Check the app-owned model and runtime."}`
          : "LTX Foley needs the LTX runtime, Gemma, Foley LoRA, and adapter."}</small>}

        {activeTake && <div className="sfx-take-controls">
          <div className="sfx-take-tabs">{takes.map((take, index) => <button key={take.id} className={take.id === activeTakeId ? "active" : ""}
            onClick={() => setActiveTakeId(take.id)}>Take {index + 1} · {take.engine === "moss" ? "MOSS" : "Foley"}</button>)}</div>
          <label className="studio-field"><span>Cue offset · {activeTake.cueOffset >= 0 ? "+" : ""}{activeTake.cueOffset.toFixed(3)}s</span>
            <input type="range" min={-Math.min(2, programmeRangeDuration)} max={programmeRangeDuration} step={frame} value={activeTake.cueOffset}
              onChange={(event) => updateTake({ cueOffset: Number(event.target.value) })} /></label>
          <div className="sfx-nudge"><button onClick={() => updateTake({ cueOffset: activeTake.cueOffset - frame })}>← 1 frame</button>
            <button onClick={() => updateTake({ cueOffset: 0 })}>Range start</button><button onClick={() => updateTake({ cueOffset: activeTake.cueOffset + frame })}>1 frame →</button></div>
          <div className="retake-range"><label className="studio-field"><span>Gain · {activeTake.gainDb.toFixed(1)} dB</span><input type="range" min={-30} max={6} step={0.5} value={activeTake.gainDb}
            onChange={(event) => updateTake({ gainDb: Number(event.target.value) })} /></label>
            <label className="studio-field"><span>Pan · {Math.round(activeTake.pan * 100)}</span><input type="range" min={-1} max={1} step={0.05} value={activeTake.pan}
              onChange={(event) => updateTake({ pan: Number(event.target.value) })} /></label></div>
          <div className="retake-range"><label className="studio-field"><span>Fade in · {activeTake.fadeInSeconds.toFixed(2)}s</span><input type="number" min={0} step={0.05} value={activeTake.fadeInSeconds}
            onChange={(event) => updateTake({ fadeInSeconds: Math.max(0, Number(event.target.value)) })} /></label>
            <label className="studio-field"><span>Fade out · {activeTake.fadeOutSeconds.toFixed(2)}s</span><input type="number" min={0} step={0.05} value={activeTake.fadeOutSeconds}
              onChange={(event) => updateTake({ fadeOutSeconds: Math.max(0, Number(event.target.value)) })} /></label></div>
          <div className="retake-range"><label className="studio-field"><span>Trim start · {activeTake.trimInSeconds.toFixed(2)}s</span><input type="number" min={0} step={0.01} value={activeTake.trimInSeconds}
            onChange={(event) => updateTake({ trimInSeconds: Math.max(0, Math.min(Number(event.target.value), seconds(activeTake.asset.duration) - activeTake.trimOutSeconds - 0.04)) })} /></label>
            <label className="studio-field"><span>Trim end · {activeTake.trimOutSeconds.toFixed(2)}s</span><input type="number" min={0} step={0.01} value={activeTake.trimOutSeconds}
              onChange={(event) => updateTake({ trimOutSeconds: Math.max(0, Math.min(Number(event.target.value), seconds(activeTake.asset.duration) - activeTake.trimInSeconds - 0.04)) })} /></label></div>
          <div className="sfx-monitor-switches"><label><input type="checkbox" checked={originalAudio} onChange={(event) => setOriginalAudio(event.target.checked)} /> Original audio</label>
            <label><input type="checkbox" checked={generatedAudio} onChange={(event) => { setGeneratedAudio(event.target.checked); if (!event.target.checked) audioRef.current?.pause(); }} /> Generated sound</label>
            <label><input type="checkbox" checked={activeTake.preserveTail} onChange={(event) => updateTake({ preserveTail: event.target.checked })} /> Keep tail</label></div>
          <small className="runtime-checks">Gain and fades are auditioned here; pan is applied on the timeline and in export.</small>
          {activeTake.log && <details className="worker-log"><summary>What the worker reported</summary><pre>{activeTake.log}</pre></details>}
        </div>}
        {failure && <div className="worker-failure"><div><strong>SFX generation failed</strong><button onClick={() => setFailure(undefined)}>×</button></div><pre>{failure}</pre></div>}
      </div>
    </div>
    <div className="dialog-actions">
      <button onClick={onClose} disabled={running}>Close</button>
      <button disabled={!activeTake || running || busy || !engineReady || !prompt.trim()} onClick={() => { void generate(); }}>Try another</button>
      <button className="primary" disabled={!activeTake || running} onClick={placeTake}>Add to SFX track</button>
    </div>
  </>;
}

function previewVolume(take: SfxTake, position: number) {
  const playableEnd = seconds(take.asset.duration) - take.trimOutSeconds;
  const fromStart = Math.max(0, position - take.trimInSeconds);
  const toEnd = Math.max(0, playableEnd - position);
  const fadeIn = take.fadeInSeconds > 0 ? Math.min(1, fromStart / take.fadeInSeconds) : 1;
  const fadeOut = take.fadeOutSeconds > 0 ? Math.min(1, toEnd / take.fadeOutSeconds) : 1;
  return Math.max(0, Math.min(1, 10 ** (take.gainDb / 20) * fadeIn * fadeOut));
}

function WaveBars({ peaks }: { peaks: number[] }) {
  return <>{peaks.map((peak, index) => <i key={index} style={{ height: `${Math.max(4, peak * 100)}%` }} />)}</>;
}

function Waveform({ peaks, className }: { peaks: number[]; className?: string }) {
  return <div className={`sfx-wave-track ${className ?? ""}`}><div className="sfx-wave-bars"><WaveBars peaks={peaks} /></div></div>;
}
