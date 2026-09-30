import { convertFileSrc } from "@tauri-apps/api/core";
import { Play, Type } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import {
  defaultGraphicOverlay, TEXT_MOTION_EASINGS, TEXT_MOTION_IN, TEXT_MOTION_LOOPS, TEXT_MOTION_OUT,
  TEXT_STYLE_PRESETS, textMotionFrame,
} from "../../domain/textMotion";
import { clipPlaybackRate, formatTimecode, seconds } from "../../domain/time";
import type {
  GraphicOverlay, GraphicTemplate, MediaAsset, TextBackdrop, TextFontFamily, TextMotionEasing,
  TextMotionIn, TextMotionLoop, TextMotionOut, TimelineClip,
} from "../../domain/types";
import { useEditor } from "../../store";
import { VisualModeSwitch, type VisualMode } from "./MaskEffectsWorkspace";

const TEMPLATES: Array<{ value: GraphicTemplate; label: string }> = [
  { value: "title", label: "Centered title" },
  { value: "lower-third", label: "Lower third" },
  { value: "banner", label: "Bottom banner" },
  { value: "bug", label: "Corner label" },
  { value: "caption", label: "Caption" },
];

const BACKDROPS: Array<{ value: TextBackdrop; label: string }> = [
  { value: "none", label: "Transparent" }, { value: "box", label: "Box" },
  { value: "outline", label: "Outline frame" }, { value: "underline", label: "Underline" },
];

const FONT_FAMILIES: Array<{ value: TextFontFamily; label: string }> = [
  { value: "display", label: "Display" }, { value: "sans", label: "Clean sans" },
  { value: "serif", label: "Cinema serif" }, { value: "mono", label: "Tech mono" },
];

function colorWithOpacity(color: string, opacity: number) {
  const raw = color.slice(0, 7);
  const alpha = Math.round(Math.max(0, Math.min(1, opacity)) * 255).toString(16).padStart(2, "0");
  return raw + alpha;
}

function fontStack(family: TextFontFamily | undefined) {
  if (family === "serif") return "Georgia, 'Times New Roman', serif";
  if (family === "mono") return "Consolas, 'Courier New', monospace";
  if (family === "sans") return "Arial, Helvetica, sans-serif";
  return "var(--font-display)";
}

export function TextMotionWorkspace({
  clip, asset, rangeStart, rangeEnd, sourceDuration, setRangeStart, setRangeEnd, onModeChange, onClose,
}: {
  clip: TimelineClip; asset: MediaAsset; rangeStart: number; rangeEnd: number; sourceDuration: number;
  setRangeStart: (value: number) => void; setRangeEnd: (value: number) => void;
  onModeChange: (mode: VisualMode) => void; onClose: () => void;
}) {
  const { addGraphicToTimeline } = useEditor();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [graphic, setGraphic] = useState<GraphicOverlay>(() => ({
    ...defaultGraphicOverlay("title"), text: "A cinematic moment", motionIn: "fade", motionOut: "fade",
  }));
  const [scrub, setScrub] = useState(rangeStart);
  const [localTime, setLocalTime] = useState(0);
  const playbackRate = clipPlaybackRate(clip);
  const programmeDuration = Math.max(0.1, (rangeEnd - rangeStart) / playbackRate);
  const sourceStart = seconds(clip.sourceIn) + rangeStart;
  const sourceEnd = seconds(clip.sourceIn) + rangeEnd;
  const motion = textMotionFrame(graphic, localTime, programmeDuration);

  useEffect(() => {
    const video = videoRef.current; if (!video) return;
    video.playbackRate = playbackRate;
    const target = seconds(clip.sourceIn) + scrub;
    if (Math.abs(video.currentTime - target) > 0.04) video.currentTime = target;
  }, [scrub, clip.sourceIn, playbackRate]);

  const update = (patch: Partial<GraphicOverlay>) => setGraphic((current) => ({ ...current, ...patch }));
  const changeTemplate = (template: GraphicTemplate) => setGraphic((current) => ({ ...current, template }));
  const replay = () => {
    const video = videoRef.current; if (!video) return;
    video.currentTime = sourceStart; setLocalTime(0); void video.play().catch(() => undefined);
  };
  const onTime = () => {
    const video = videoRef.current; if (!video) return;
    if (video.currentTime >= sourceEnd - 0.01) { video.pause(); video.currentTime = sourceStart; setLocalTime(0); return; }
    const sourceRelative = Math.max(0, video.currentTime - sourceStart);
    setLocalTime(sourceRelative / playbackRate);
    setScrub(Math.max(0, Math.min(sourceDuration, video.currentTime - seconds(clip.sourceIn))));
  };
  const place = () => {
    const start = seconds(clip.start) + rangeStart / playbackRate;
    addGraphicToTimeline({ ...graphic, text: graphic.text.trim() || "Animated text" }, start, programmeDuration);
    onClose();
  };
  const rangeLeft = sourceDuration ? rangeStart / sourceDuration * 100 : 0;
  const rangeWidth = sourceDuration ? (rangeEnd - rangeStart) / sourceDuration * 100 : 0;
  const glow = motion.glow > 0 ? `0 0 ${8 + motion.glow * 22}px ${graphic.accentColor}` : undefined;
  const backdrop = graphic.backdrop ?? "box";
  const backdropColor = colorWithOpacity(graphic.backgroundColor, graphic.backgroundOpacity ?? 1);
  const borderWidth = graphic.backgroundBorderWidth ?? 3;
  const shimmerStyle = graphic.motionLoop === "shimmer" ? {
    backgroundImage: `linear-gradient(105deg, ${graphic.textColor} 34%, ${graphic.accentColor} 49%, ${graphic.textColor} 64%)`,
    backgroundSize: "240% 100%", backgroundPosition: `${100 - motion.shimmer * 200}% 0`,
    WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent",
  } : { color: graphic.textColor };

  return <>
    <div className="clip-editor-body text-motion-body">
      <div className="clip-editor-preview">
        <div className="text-motion-preview">
          <video ref={videoRef} src={convertFileSrc(asset.path)} preload="auto" muted onTimeUpdate={onTime} />
          <div className={"graphic-preview graphic-" + graphic.template + " backdrop-" + backdrop} style={{
            opacity: motion.opacity,
          }}><div style={{
            backgroundColor: backdrop === "box" ? backdropColor : "transparent", borderColor: graphic.accentColor,
            borderWidth: backdrop === "outline" ? borderWidth : undefined,
            padding: `${graphic.backgroundPadding ?? 16}px`,
            transform: `translate(${motion.translateX}px, ${motion.translateY}px) scale(${motion.scale}) rotate(${motion.rotation}deg)`,
            filter: `blur(${motion.blur}px) hue-rotate(${motion.hue}deg)`, transformOrigin: "center",
            clipPath: `inset(0 ${(1 - motion.reveal) * 100}% 0 0)`,
          }}>
            <strong style={{ ...shimmerStyle, fontFamily: fontStack(graphic.fontFamily), fontWeight: graphic.fontWeight ?? 700,
              fontStyle: graphic.italic ? "italic" : "normal", fontSize: graphic.fontSize,
              letterSpacing: `${motion.tracking}px`, lineHeight: `${1 + (graphic.lineSpacing ?? 4) / Math.max(16, graphic.fontSize)}`,
              WebkitTextStroke: `${graphic.strokeWidth ?? 0}px ${graphic.strokeColor ?? "#000000"}`,
              textShadow: [glow, (graphic.shadowOffset ?? 0) > 0 ? `${graphic.shadowOffset}px ${graphic.shadowOffset}px ${Math.max(2, (graphic.shadowOffset ?? 0) * 1.6)}px ${graphic.shadowColor ?? "#000000"}` : undefined].filter(Boolean).join(", ") || undefined,
            }}>{motion.text || "\u00a0"}</strong>
            {graphic.subtitle && <span style={{ color: graphic.textColor, fontFamily: fontStack(graphic.fontFamily) }}>{graphic.subtitle}</span>}
          </div></div>
        </div>
        <div className="scrub-strip"><div className="scrub-range" style={{ left: `${rangeLeft}%`, width: `${rangeWidth}%` }} />
          <input type="range" min={0} max={Math.max(0.1, sourceDuration)} step={0.02} value={scrub}
            onChange={(event) => { const value = Number(event.target.value); setScrub(value); setLocalTime(Math.max(0, (value - rangeStart) / playbackRate)); }} /></div>
        <div className="scrub-actions">
          <span className="timecode">{formatTimecode(scrub)}</span>
          <button onClick={() => setRangeStart(Math.min(scrub, rangeEnd - 0.1))}>Set in</button>
          <button onClick={() => setRangeEnd(Math.max(scrub, rangeStart + 0.1))}>Set out</button>
          <button onClick={() => { setRangeStart(0); setRangeEnd(sourceDuration); }}>Whole clip</button>
          <button onClick={replay}><Play size={12} /> Replay</button>
          <span className="range-readout">{programmeDuration.toFixed(2)}s on G1</span>
        </div>
      </div>
      <div className="clip-editor-controls">
        <VisualModeSwitch mode="text" disabled={false} onChange={onModeChange} />
        <section className="text-control-section"><header><strong>Style</strong><small>Editable preset</small></header>
          <div className="text-style-presets">{TEXT_STYLE_PRESETS.map((preset) => <button key={preset.id}
            onClick={() => update(preset.values)} title={preset.description}><strong>{preset.name}</strong><small>{preset.description}</small></button>)}</div>
        </section>
        <label className="studio-field"><span>Layout</span><select value={graphic.template} onChange={(event) => changeTemplate(event.target.value as GraphicTemplate)}>
          {TEMPLATES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
        <label className="ai-prompt"><span>Text</span><textarea value={graphic.text} maxLength={120} onChange={(event) => update({ text: event.target.value })} placeholder="Type the title…" /></label>
        <label className="field text-field"><span>Subtitle</span><input value={graphic.subtitle} maxLength={160} onChange={(event) => update({ subtitle: event.target.value })} /></label>
        <div className="text-motion-grid">
          <label className="studio-field"><span>Animate in</span><select value={graphic.motionIn ?? "none"} onChange={(event) => update({ motionIn: event.target.value as TextMotionIn })}>
            {TEXT_MOTION_IN.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
          <label className="studio-field"><span>Animate out</span><select value={graphic.motionOut ?? "none"} onChange={(event) => update({ motionOut: event.target.value as TextMotionOut })}>
            {TEXT_MOTION_OUT.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
          <label className="studio-field"><span>While visible</span><select value={graphic.motionLoop ?? "none"} onChange={(event) => update({ motionLoop: event.target.value as TextMotionLoop })}>
            {TEXT_MOTION_LOOPS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
          <label className="studio-field"><span>Easing</span><select value={graphic.motionEasing ?? "smooth"} onChange={(event) => update({ motionEasing: event.target.value as TextMotionEasing })}>
            {TEXT_MOTION_EASINGS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
        </div>
        <MotionRange label="Animation time" value={graphic.motionDuration ?? 0.6} min={0.1} max={Math.max(0.1, Math.min(3, programmeDuration / 2))} step={0.05} suffix="s" onChange={(motionDuration) => update({ motionDuration })} />
        <section className="text-control-section"><header><strong>Typography</strong><small>Fill, outline and shadow</small></header>
          <div className="text-motion-grid">
            <label className="studio-field"><span>Font</span><select value={graphic.fontFamily ?? "display"} onChange={(event) => update({ fontFamily: event.target.value as TextFontFamily })}>
              {FONT_FAMILIES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
            <label className="studio-field"><span>Weight</span><select value={graphic.fontWeight ?? 700} onChange={(event) => update({ fontWeight: Number(event.target.value) as 400 | 700 | 900 })}>
              <option value={400}>Regular</option><option value={700}>Bold</option><option value={900}>Black</option></select></label>
          </div>
          <MotionRange label="Font size" value={graphic.fontSize} min={12} max={240} step={1} suffix="px" onChange={(fontSize) => update({ fontSize })} />
          <MotionRange label="Text outline" value={graphic.strokeWidth ?? 0} min={0} max={12} step={1} suffix="px" onChange={(strokeWidth) => update({ strokeWidth })} />
          <MotionRange label="Shadow" value={graphic.shadowOffset ?? 0} min={0} max={20} step={1} suffix="px" onChange={(shadowOffset) => update({ shadowOffset })} />
          <label className="check-row"><input type="checkbox" checked={graphic.italic ?? false} onChange={(event) => update({ italic: event.target.checked })} /> Italic</label>
        </section>
        <section className="text-control-section"><header><strong>Backdrop</strong><small>Transparent, fill or frame</small></header>
          <div className="text-backdrop-switch">{BACKDROPS.map((item) => <button key={item.value} className={backdrop === item.value ? "active" : ""}
            onClick={() => update({ backdrop: item.value })}>{item.label}</button>)}</div>
          {backdrop !== "none" && <><MotionRange label="Backdrop opacity" value={graphic.backgroundOpacity ?? 1} min={0} max={1} step={0.05} suffix="" onChange={(backgroundOpacity) => update({ backgroundOpacity })} />
            <MotionRange label="Padding" value={graphic.backgroundPadding ?? 16} min={0} max={64} step={1} suffix="px" onChange={(backgroundPadding) => update({ backgroundPadding })} /></>}
          {backdrop === "outline" && <MotionRange label="Frame width" value={graphic.backgroundBorderWidth ?? 3} min={1} max={12} step={1} suffix="px" onChange={(backgroundBorderWidth) => update({ backgroundBorderWidth })} />}
        </section>
        <div className="text-color-grid">
          <label className="mask-color"><span>Text</span><input type="color" value={graphic.textColor.slice(0, 7)} onChange={(event) => update({ textColor: event.target.value })} /></label>
          <label className="mask-color"><span>Backdrop</span><input type="color" value={graphic.backgroundColor.slice(0, 7)} onChange={(event) => update({ backgroundColor: event.target.value })} /></label>
          <label className="mask-color"><span>Accent / magic</span><input type="color" value={graphic.accentColor.slice(0, 7)} onChange={(event) => update({ accentColor: event.target.value })} /></label>
          <label className="mask-color"><span>Text outline</span><input type="color" value={(graphic.strokeColor ?? "#000000").slice(0, 7)} onChange={(event) => update({ strokeColor: event.target.value })} /></label>
          <label className="mask-color"><span>Shadow</span><input type="color" value={(graphic.shadowColor ?? "#000000").slice(0, 7)} onChange={(event) => update({ shadowColor: event.target.value })} /></label>
        </div>
        <small className="panel-help">This creates an editable graphics clip. The text is not burned into the selected video, so you can trim, move, recolor, or change its animation later in the NLE Inspector.</small>
        <button className="render-button" disabled={!graphic.text.trim() || rangeEnd <= rangeStart} onClick={place}><Type size={16} /> Add editable text to timeline</button>
      </div>
    </div>
    <div className="dialog-actions"><button onClick={onClose}>Close</button><button className="primary" disabled={!graphic.text.trim()} onClick={place}>Add to G1</button></div>
  </>;
}

function MotionRange({ label, value, min, max, step, suffix, onChange }: {
  label: string; value: number; min: number; max: number; step: number; suffix: string; onChange: (value: number) => void;
}) {
  return <label className="mask-range"><span>{label}<b>{Number(value.toFixed(2))}{suffix}</b></span>
    <input type="range" value={Math.min(max, value)} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}
