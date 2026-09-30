import { convertFileSrc } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { AudioLines, Layers3, Move, Sparkles, Upload, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import {
  AUDIO_REACTIVE_MODES, MASK_EFFECTS, MASK_MATTE_STYLES, MASK_PLACEMENTS, MASK_TRANSITIONS,
  DEFAULT_MASK_VFX_RECIPE, MASK_VFX_RECIPES, cloneVfxGradient,
  maskEffectFileSuffix, maskEffectReady,
  type ClipVfxLayer, type MaskMatteCache, type MaskSource, type MaskVfxBlendMode,
  type MaskVfxKeyMode, type MaskVfxKeyframe, type MaskVfxRecipeV1,
} from "../../domain/maskEffects";
import { programmeMedia } from "../../domain/programme";
import { vfxPreviewDocument } from "../../domain/programmeVfx";
import { ProgrammeEngine } from "../../render/programmeEngine";
import { readCompositorFlag } from "../../render/compositorFlag";
import { packYuv420 } from "../../render/yuv420";
import { FrameSender, renderVideoCancel, renderVideoFinish, renderVideoStart } from "../../services/renderService";
import { maskAudioEnvelope, prepareVfxLayer } from "../../services/vfxPrepare";
import { cloneMaskVfxRecipe, maskVfxAnimatedValues } from "../../domain/maskVfx";
import { gradientPreview } from "../../domain/photoCompose";
import type { Gradient } from "../../domain/photoGradient";
import { clipPlaybackRate, formatTimecode, fromSeconds, seconds } from "../../domain/time";
import type { MediaAsset, TimelineClip } from "../../domain/types";
import {
  cacheMaskTrack, generatedAsset, parseFrameRate, probeMedia, runtimeStatus,
  type AiRuntimeStatus, type MaskTrackResult,
  type MaskEffectMatteStyle, type MaskEffectPlacement, type MaskEffectRequest, type MaskEffectTransition,
} from "../../services/modelService";
import { useEditor } from "../../store";
import { GradientEditor } from "../photoEdit/PhotoPanels";

export type VisualMode = "kiwi" | "retake" | "mask" | "text" | "vfx";
type PointMode = "off" | "include" | "exclude";

export function VisualModeSwitch({ mode, disabled, onChange }: {
  mode: VisualMode; disabled: boolean; onChange: (mode: VisualMode) => void;
}) {
  return <div className="engine-switch visual-mode-switch" role="tablist" aria-label="Visual editing mode">
    {([
      ["kiwi", "Kiwi-Edit"], ["retake", "LTX Retake"], ["mask", "Mask Effects"], ["text", "Text Motion"], ["vfx", "VFX"],
    ] as Array<[VisualMode, string]>).map(([value, label]) => <button key={value} role="tab"
      aria-selected={mode === value} className={mode === value ? "active" : ""} disabled={disabled}
      onClick={() => onChange(value)}>{label}</button>)}
  </div>;
}

export function MaskEffectsWorkspace({
  clip, asset, runtime, rangeStart, rangeEnd, sourceDuration, setRangeStart, setRangeEnd,
  onModeChange, onRunningChange, onClose,
}: {
  clip: TimelineClip; asset: MediaAsset; runtime?: AiRuntimeStatus;
  rangeStart: number; rangeEnd: number; sourceDuration: number;
  setRangeStart: (value: number) => void; setRangeEnd: (value: number) => void;
  onModeChange: (mode: VisualMode) => void; onRunningChange: (running: boolean) => void; onClose: () => void;
}) {
  const {
    busy, setBusy, setStatus, replaceClipRangeWithAsset, insertAssetAfterClip,
    upsertClipVfxLayer, removeClipVfxLayer, addAssets, project,
  } = useEditor();
  const videoRef = useRef<HTMLVideoElement>(null);
  const effectVideoRef = useRef<HTMLVideoElement>(null);
  const placementDrag = useRef(false);
  const [source, setSource] = useState<MaskSource>("sam2");
  const [effect, setEffect] = useState<MaskEffectRequest["effect"]>("neon-edge");
  const [color, setColor] = useState("#00f0ff");
  const [secondaryColor, setSecondaryColor] = useState("#ff6b1a");
  const [intensity, setIntensity] = useState(1);
  const [feather, setFeather] = useState(6);
  const [expand, setExpand] = useState(4);
  const [temporalSmooth, setTemporalSmooth] = useState(2);
  const [cleanup, setCleanup] = useState(300);
  const [effectSpeed, setEffectSpeed] = useState(1);
  const [particleAmount, setParticleAmount] = useState(1);
  const [transitionIn, setTransitionIn] = useState<MaskEffectTransition>("none");
  const [transitionOut, setTransitionOut] = useState<MaskEffectTransition>("none");
  const [transitionInDuration, setTransitionInDuration] = useState(0.6);
  const [transitionOutDuration, setTransitionOutDuration] = useState(0.45);
  const [invert, setInvert] = useState(false);
  const [reactive, setReactive] = useState<MaskEffectRequest["audioReactive"]>("off");
  const [reactiveAmount, setReactiveAmount] = useState(0.75);
  const [reactiveSmoothing, setReactiveSmoothing] = useState(0.45);
  const [placement, setPlacement] = useState<MaskEffectPlacement>("behind");
  const [matteStyle, setMatteStyle] = useState<MaskEffectMatteStyle>("subject");
  const [effectClipPath, setEffectClipPath] = useState("");
  const [effectClipColor, setEffectClipColor] = useState<"original" | "gradient">("original");
  const [effectOpacity, setEffectOpacity] = useState(0.9);
  const [effectX, setEffectX] = useState(0.5);
  const [effectY, setEffectY] = useState(0.5);
  const [effectScale, setEffectScale] = useState(100);
  const [effectRotation, setEffectRotation] = useState(0);
  const [loop, setLoop] = useState(true);
  const [pulseAmount, setPulseAmount] = useState(0.18);
  const [pulseRate, setPulseRate] = useState(1.25);
  const [gradientEnabled, setGradientEnabled] = useState(true);
  const [gradient, setGradient] = useState<Gradient>(cloneVfxGradient);
  const [presetId, setPresetId] = useState("custom");
  const [recipeName, setRecipeName] = useState("Custom VFX");
  const [occlusionFeather, setOcclusionFeather] = useState(DEFAULT_MASK_VFX_RECIPE.occlusionFeather);
  const [occlusionExpand, setOcclusionExpand] = useState(DEFAULT_MASK_VFX_RECIPE.occlusionExpand);
  const [transitionOrigin, setTransitionOrigin] = useState<MaskVfxRecipeV1["transitionOrigin"]>("subject");
  const [blendMode, setBlendMode] = useState<MaskVfxBlendMode>("screen");
  const [keyMode, setKeyMode] = useState<MaskVfxKeyMode>("luma");
  const [keyThreshold, setKeyThreshold] = useState(DEFAULT_MASK_VFX_RECIPE.keyThreshold);
  const [keySoftness, setKeySoftness] = useState(DEFAULT_MASK_VFX_RECIPE.keySoftness);
  const [keyColor, setKeyColor] = useState(DEFAULT_MASK_VFX_RECIPE.keyColor);
  const [rearAmount, setRearAmount] = useState(DEFAULT_MASK_VFX_RECIPE.rearAmount);
  const [subjectAmount, setSubjectAmount] = useState(DEFAULT_MASK_VFX_RECIPE.subjectAmount);
  const [frontAmount, setFrontAmount] = useState(DEFAULT_MASK_VFX_RECIPE.frontAmount);
  const [motionBlur, setMotionBlur] = useState(DEFAULT_MASK_VFX_RECIPE.motionBlur);
  const [lightWrap, setLightWrap] = useState(DEFAULT_MASK_VFX_RECIPE.lightWrap);
  const [colorSpill, setColorSpill] = useState(DEFAULT_MASK_VFX_RECIPE.colorSpill);
  const [grain, setGrain] = useState(DEFAULT_MASK_VFX_RECIPE.grain);
  const [displacement, setDisplacement] = useState(DEFAULT_MASK_VFX_RECIPE.displacement);
  const [keyframes, setKeyframes] = useState<MaskVfxKeyframe[]>([]);
  const [matteCache, setMatteCache] = useState<MaskMatteCache>();
  // VFX Phase 7: the look is rendered live by the compositor (no preview file any more); `proxy`
  // stays as "showing something other than the setup view" for the layout below and is never set.
  const proxy = undefined as { path: string; log: string } | undefined;
  const setProxy = (_: undefined) => undefined;
  const [live, setLive] = useState(true);
  const [previewPlaying, setPreviewPlaying] = useState(false);
  const [previewInfo, setPreviewInfo] = useState("");
  const [saved, setSaved] = useState(false);
  /** What a live render needs besides the recipe, computed once per setting (envelope, clip probe). */
  const [extras, setExtras] = useState<Pick<ClipVfxLayer, "envelope" | "effectClipMedia" | "stableMatte">>({});
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const engine = useRef<ProgrammeEngine | undefined>(undefined);
  const [points, setPoints] = useState<Array<{ x: number; y: number; include: boolean }>>([]);
  const [pointMode, setPointMode] = useState<PointMode>("include");
  const [scrub, setScrub] = useState(rangeStart);
  const [running, setRunning] = useState(false);
  const [startedAt, setStartedAt] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResultState] = useState<{ asset: MediaAsset; log: string }>();
  // Any change to the look makes a saved layer and a baked file stale.
  const setResult = (value: { asset: MediaAsset; log: string } | undefined) => { setResultState(value); if (!value) setSaved(false); };
  const [failure, setFailure] = useState<string>();
  const [runtimeInfo, setRuntimeInfo] = useState<AiRuntimeStatus | undefined>(runtime);
  const [checkingRuntime, setCheckingRuntime] = useState(false);
  const [, setLayoutRevision] = useState(0);
  const gradientStrip = useMemo(() => gradientPreview({
    ...gradient, kind: "linear", angle: 0, cx: 0.5, cy: 0.5, scale: 1, offset: 0,
  }, 220, 1), [gradient]);
  const playbackRate = clipPlaybackRate(clip);
  const sourceStart = seconds(clip.sourceIn) + rangeStart;
  const sourceEnd = seconds(clip.sourceIn) + rangeEnd;
  const fps = asset.frameRate ? asset.frameRate.numerator / Math.max(1, asset.frameRate.denominator) : 24;
  const requestedFrames = Math.round((rangeEnd - rangeStart) * fps);
  const rangeTooLong = requestedFrames > 900;
  const ready = maskEffectReady(source, runtimeInfo);
  const effectClipMissing = matteStyle === "clip" && !effectClipPath;
  const canRun = !running && !busy && !checkingRuntime && rangeEnd > rangeStart && !rangeTooLong && !effectClipMissing;
  const canTrack = canRun && source === "sam2" && ready && points.some((point) => point.include);
  const canPreview = canRun && source === "sam2" && ready && Boolean(matteCache || points.some((point) => point.include));
  const preflightMessage = !ready
    ? source === "sam2" ? `SAM 2 unavailable (${runtimeInfo?.sam2State ?? "checking"})` : "BiRefNet runtime is unavailable"
    : source === "sam2" && !points.some((point) => point.include) ? "Add at least one green Include point on the subject"
    : effectClipMissing ? "Choose an effect video for the imported VFX clip matte" : undefined;

  const currentRecipe = (): MaskVfxRecipeV1 => ({
    schemaVersion: 1, presetId, name: recipeName, effect, matteStyle, placement,
    effectClipPath, effectClipColor, blendMode, keyMode, keyThreshold, keySoftness, keyColor,
    color, secondaryColor, gradientEnabled,
    gradient: { ...gradient, stops: gradient.stops.map((stop) => ({ ...stop })) },
    intensity, effectOpacity, effectX, effectY, effectScale, effectRotation, effectSpeed,
    particleAmount, loop, pulseAmount, pulseRate, effectFeather: feather, effectExpand: expand,
    occlusionFeather, occlusionExpand, temporalSmooth, cleanup, invert,
    transitionIn, transitionOut, transitionInDuration, transitionOutDuration, transitionOrigin,
    audioReactive: reactive, reactiveAmount, reactiveSmoothing,
    rearAmount, subjectAmount, frontAmount, motionBlur, lightWrap, colorSpill, grain, displacement,
    keyframes: keyframes.map((keyframe) => ({ ...keyframe, values: { ...keyframe.values } })),
  });
  const recipeFingerprint = JSON.stringify(currentRecipe());

  const applyRecipe = (value: MaskVfxRecipeV1) => {
    const recipe = cloneMaskVfxRecipe(value);
    setPresetId(recipe.presetId); setRecipeName(recipe.name); setEffect(recipe.effect);
    setMatteStyle(recipe.matteStyle); setPlacement(recipe.placement);
    setEffectClipPath(recipe.effectClipPath); setEffectClipColor(recipe.effectClipColor);
    setBlendMode(recipe.blendMode); setKeyMode(recipe.keyMode); setKeyThreshold(recipe.keyThreshold);
    setKeySoftness(recipe.keySoftness); setKeyColor(recipe.keyColor); setColor(recipe.color);
    setSecondaryColor(recipe.secondaryColor); setGradientEnabled(recipe.gradientEnabled);
    setGradient(recipe.gradient); setIntensity(recipe.intensity); setEffectOpacity(recipe.effectOpacity);
    setEffectX(recipe.effectX); setEffectY(recipe.effectY); setEffectScale(recipe.effectScale);
    setEffectRotation(recipe.effectRotation); setEffectSpeed(recipe.effectSpeed);
    setParticleAmount(recipe.particleAmount); setLoop(recipe.loop); setPulseAmount(recipe.pulseAmount);
    setPulseRate(recipe.pulseRate); setFeather(recipe.effectFeather); setExpand(recipe.effectExpand);
    setOcclusionFeather(recipe.occlusionFeather); setOcclusionExpand(recipe.occlusionExpand);
    setTemporalSmooth(recipe.temporalSmooth); setCleanup(recipe.cleanup); setInvert(recipe.invert);
    setTransitionIn(recipe.transitionIn); setTransitionOut(recipe.transitionOut);
    setTransitionInDuration(recipe.transitionInDuration); setTransitionOutDuration(recipe.transitionOutDuration);
    setTransitionOrigin(recipe.transitionOrigin); setReactive(recipe.audioReactive);
    setReactiveAmount(recipe.reactiveAmount); setReactiveSmoothing(recipe.reactiveSmoothing);
    setRearAmount(recipe.rearAmount); setSubjectAmount(recipe.subjectAmount); setFrontAmount(recipe.frontAmount);
    setMotionBlur(recipe.motionBlur); setLightWrap(recipe.lightWrap); setColorSpill(recipe.colorSpill);
    setGrain(recipe.grain); setDisplacement(recipe.displacement); setKeyframes(recipe.keyframes);
    setResult(undefined); setProxy(undefined); setFailure(undefined);
  };

  const trackingIdentity = JSON.stringify({ clip: clip.id, source, sourceStart, sourceEnd, points });
  const trackingIdentityRef = useRef(trackingIdentity);
  const trackingClipRef = useRef(clip.id);
  const skipTrackingInvalidationRef = useRef(true);

  useEffect(() => {
    const layer = clip.vfxLayers?.at(-1);
    skipTrackingInvalidationRef.current = true;
    setPoints([]); setResult(undefined); setProxy(undefined); setFailure(undefined);
    if (layer) {
      const loadedSource: MaskSource = "sam2";
      applyRecipe(layer.recipe); setMatteCache(layer.matteCache?.model === "sam2" ? layer.matteCache : undefined); setSource(loadedSource);
      setExtras({ envelope: layer.envelope, effectClipMedia: layer.effectClipMedia, stableMatte: layer.stableMatte });
      setSaved(true);
      setRangeStart(layer.rangeStart); setRangeEnd(layer.rangeEnd); setScrub(layer.rangeStart);
      trackingIdentityRef.current = JSON.stringify({
        clip: clip.id, source: loadedSource,
        sourceStart: seconds(clip.sourceIn) + layer.rangeStart,
        sourceEnd: seconds(clip.sourceIn) + layer.rangeEnd,
        points: [],
      });
    } else {
      applyRecipe(DEFAULT_MASK_VFX_RECIPE); setMatteCache(undefined);
      trackingIdentityRef.current = JSON.stringify({
        clip: clip.id, source: "sam2", sourceStart, sourceEnd,
        points: [],
      });
    }
  // Loading is intentionally tied to clip identity; edits stay local until saved as a layer.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clip.id]);
  useEffect(() => { setProxy(undefined); }, [recipeFingerprint]);
  useEffect(() => {
    if (skipTrackingInvalidationRef.current) {
      skipTrackingInvalidationRef.current = false;
      return;
    }
    if (trackingClipRef.current !== clip.id) {
      trackingClipRef.current = clip.id;
      return;
    }
    if (trackingIdentityRef.current !== trackingIdentity) {
      trackingIdentityRef.current = trackingIdentity;
      setMatteCache(undefined); setProxy(undefined);
    }
  }, [trackingIdentity]);
  useEffect(() => {
    if (runtime) setRuntimeInfo(runtime);
  }, [runtime]);
  useEffect(() => {
    let live = true;
    setCheckingRuntime(true);
    void runtimeStatus().then((status) => { if (live) setRuntimeInfo(status); })
      .catch(() => undefined).finally(() => { if (live) setCheckingRuntime(false); });
    return () => { live = false; };
  }, []);
  useEffect(() => { onRunningChange(running); return () => onRunningChange(false); }, [running, onRunningChange]);
  useEffect(() => {
    if (!running) return;
    const update = () => setElapsed(Math.floor((Date.now() - startedAt) / 1000));
    update(); const timer = window.setInterval(update, 500); return () => window.clearInterval(timer);
  }, [running, startedAt]);
  useEffect(() => {
    const video = videoRef.current;
    if (!video || result || proxy) return;
    const target = seconds(clip.sourceIn) + scrub;
    if (Math.abs(video.currentTime - target) > 0.04) video.currentTime = target;
  }, [scrub, clip.sourceIn, result, proxy]);
  useEffect(() => {
    const resize = () => setLayoutRevision((value) => value + 1);
    window.addEventListener("resize", resize); return () => window.removeEventListener("resize", resize);
  }, []);
  useEffect(() => {
    if (effectVideoRef.current) effectVideoRef.current.playbackRate = effectSpeed;
  }, [effectSpeed, effectClipPath]);

  const videoBox = () => {
    const video = videoRef.current;
    if (!video || !video.videoWidth || !video.videoHeight) return { left: 0, top: 0, width: 1, height: 1 };
    const width = video.clientWidth; const height = video.clientHeight;
    const scale = Math.min(width / video.videoWidth, height / video.videoHeight);
    const drawnWidth = video.videoWidth * scale; const drawnHeight = video.videoHeight * scale;
    return { left: (width - drawnWidth) / 2, top: (height - drawnHeight) / 2, width: drawnWidth, height: drawnHeight };
  };
  const markerStyle = (point: { x: number; y: number }) => {
    const box = videoBox(); const video = videoRef.current;
    return { left: `${(box.left + point.x * box.width) / Math.max(1, video?.clientWidth ?? 1) * 100}%`, top: `${(box.top + point.y * box.height) / Math.max(1, video?.clientHeight ?? 1) * 100}%` };
  };
  const addPoint = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (source !== "sam2" || pointMode === "off" || result || proxy || running) return;
    const video = videoRef.current; if (!video) return;
    const rect = video.getBoundingClientRect(); const box = videoBox();
    const x = (event.clientX - rect.left - box.left) / box.width;
    const y = (event.clientY - rect.top - box.top) / box.height;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    setPoints((current) => [...current, { x, y, include: pointMode === "include" }]);
  };
  const positionEffect = (event: ReactPointerEvent<HTMLDivElement>) => {
    const video = videoRef.current; if (!video) return;
    const rect = video.getBoundingClientRect(); const box = videoBox();
    setEffectX(Math.max(0, Math.min(1, (event.clientX - rect.left - box.left) / box.width)));
    setEffectY(Math.max(0, Math.min(1, (event.clientY - rect.top - box.top) / box.height)));
    setResult(undefined);
  };
  const previewPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (result || proxy || running) return;
    if (source === "sam2" && pointMode !== "off") { addPoint(event); return; }
    placementDrag.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    positionEffect(event);
  };
  const importEffectClip = async () => {
    const selected = await open({
      multiple: false,
      filters: [{ name: "Effect video", extensions: ["mp4", "mov", "webm", "mkv", "avi", "gif"] }],
    });
    if (typeof selected !== "string") return;
    setEffectClipPath(selected); setMatteStyle("clip"); setPointMode("off");
    setEffect("animated-matte"); setResult(undefined); setFailure(undefined);
  };

  const acquireMatte = async (): Promise<MaskTrackResult> => {
    if (matteCache) return { ...matteCache, reused: true, log: "Reused cached matte." };
    if (source !== "sam2") throw new Error("Cached VFX previews currently use SAM 2 tracking.");
    if (!points.some((point) => point.include)) throw new Error("Add a green Include point before tracking.");
    setStatus("SAM 2 is tracking the subject once and caching a lossless matte…");
    const tracked = await cacheMaskTrack({
      videoPath: asset.path, startTime: sourceStart, endTime: sourceEnd,
      model: "sam2", points,
    });
    setMatteCache(tracked);
    return tracked;
  };

  const trackSubject = async () => {
    setRunning(true); setBusy(true); setStartedAt(Date.now()); setElapsed(0); setFailure(undefined);
    try {
      const tracked = await acquireMatte();
      setStatus(tracked.reused ? "Reused the cached SAM 2 matte" : "SAM 2 matte cached — looks no longer retrack it");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setFailure(message); setStatus("Subject tracking failed: " + message.split("\n").at(-1));
    } finally { setRunning(false); setBusy(false); }
  };

  /* ------------------------------------------------ VFX Phase 7: live compositor preview */

  useEffect(() => {
    engine.current ??= new ProgrammeEngine({ lookahead: 16 });
    return () => { engine.current?.dispose(); engine.current = undefined; };
  }, []);
  const media = useMemo(() => programmeMedia(asset), [asset]);
  const draftLayer = (): ClipVfxLayer | undefined => {
    if (!matteCache) return undefined;
    const previous = clip.vfxLayers?.at(-1);
    const now = new Date().toISOString();
    return {
      id: previous?.id ?? "draft", name: recipeName, enabled: true, rangeStart, rangeEnd, recipe: currentRecipe(),
      matteCache, ...extras, createdAt: previous?.createdAt ?? now, updatedAt: now,
    };
  };
  // The audio envelope and the effect clip's frame count, fetched when the look needs them.
  const envelopeKey = asset.hasAudio && reactive !== "off" && matteCache ? `${reactive}|${reactiveSmoothing}|${sourceStart}|${sourceEnd}|${matteCache.frameCount}` : "";
  useEffect(() => {
    if (!envelopeKey || !matteCache) return;
    if (extras.envelope?.mode === reactive && extras.envelope.smoothing === reactiveSmoothing && extras.envelope.values.length === matteCache.frameCount) return;
    let live = true;
    void maskAudioEnvelope(asset.path, sourceStart, sourceEnd, matteCache.frameCount, reactive, reactiveSmoothing)
      .then((values) => { if (live) setExtras((current) => ({ ...current, envelope: { mode: reactive, smoothing: reactiveSmoothing, values } })); })
      .catch((error) => setStatus("Audio envelope failed: " + (error instanceof Error ? error.message : String(error))));
    return () => { live = false; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [envelopeKey]);
  useEffect(() => {
    if (matteStyle !== "clip" || !effectClipPath) return;
    let live = true;
    void probeMedia(effectClipPath).then((probe) => {
      if (!live) return;
      const rate = parseFrameRate(probe.frameRate);
      const clipFps = rate.numerator / Math.max(1, rate.denominator);
      setExtras((current) => ({ ...current, effectClipMedia: {
        frameCount: Math.max(1, Math.round(probe.durationSeconds * clipFps)), fps: clipFps, width: probe.width ?? 1920, height: probe.height ?? 1080,
      } }));
    }).catch((error) => setStatus("Could not read the effect clip: " + (error instanceof Error ? error.message : String(error))));
    return () => { live = false; };
  }, [matteStyle, effectClipPath]);

  const previewSize = () => {
    const width = Math.min(media.width, 960);
    return { width: width - (width % 2), height: Math.max(2, Math.round((width * media.height) / media.width / 2) * 2) };
  };
  const drawn = useRef("");
  /** Draw the look at source second `source` onto the preview canvas once its inputs are loaded. */
  const drawPreview = (source: number, wanted: () => boolean) => {
    const held = engine.current, canvas = canvasRef.current, layer = draftLayer();
    if (!held || !canvas || !layer) return;
    const key = `${source}|${JSON.stringify(layer)}`;
    if (drawn.current === key) return;
    const document = vfxPreviewDocument(media, clip.id, seconds(clip.sourceIn), layer, source);
    const needs = held.needs(document, 0);
    if (!held.ready(needs, false)) {
      void held.prepare(needs, false, true).then(() => { if (wanted()) drawPreview(source, wanted); });
      return;
    }
    const { width, height } = previewSize();
    const started = performance.now();
    const result = held.compose(document, Math.round(source * media.fps), width, height, false, readCompositorFlag());
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    canvas.getContext("2d")!.putImageData(new ImageData(result.pixels, width, height), 0, 0);
    drawn.current = key;
    const waiting = result.report.skipped.map((item) => item.reason);
    setPreviewInfo(`${held.compositor.lastEngine.toUpperCase()} · ${(performance.now() - started).toFixed(0)} ms at ${width}×${height}${waiting.length ? ` · waiting: ${waiting[0]}` : ""}${held.failures.size ? ` · could not read ${[...held.failures.keys()][0]}` : ""}`);
  };
  const showLive = live && Boolean(matteCache) && !result;
  const scrubRef = useRef(scrub);
  scrubRef.current = scrub;
  useEffect(() => {
    if (!showLive || previewPlaying) return;
    const at = scrub;
    drawPreview(seconds(clip.sourceIn) + at, () => scrubRef.current === at);
  });
  // Play the range in the live preview: wall clock, newest frame that is ready.
  useEffect(() => {
    if (!previewPlaying || !showLive) return;
    const began = performance.now(), from = scrub >= rangeEnd - 0.05 || scrub < rangeStart ? rangeStart : scrub;
    let handle = 0, published = 0;
    const tick = (now: number) => {
      let at = from + (now - began) / 1000;
      if (at >= rangeEnd) at = rangeStart + ((at - rangeStart) % Math.max(0.05, rangeEnd - rangeStart));
      const frameAt = Math.floor(at * media.fps) / media.fps;
      drawPreview(seconds(clip.sourceIn) + frameAt, () => false);
      if (now - published > 120) { published = now; setScrub(frameAt); }
      handle = requestAnimationFrame(tick);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [previewPlaying, showLive, recipeFingerprint]);

  /** Save the look as a timeline VFX layer: the matte smoothing is baked, then it renders live. */
  const saveLayer = async () => {
    if (source === "sam2" && !matteCache && !points.some((point) => point.include)) {
      const message = "SAM 2 needs a green Include point. Choose + Include, then click the subject in the preview.";
      setFailure(message); setStatus(message); return;
    }
    setRunning(true); setBusy(true); setStartedAt(Date.now()); setElapsed(0); setFailure(undefined);
    try {
      const cache = await acquireMatte();
      const recipe = { ...currentRecipe(), audioReactive: asset.hasAudio ? reactive : "off" as const };
      const previous = clip.vfxLayers?.at(-1);
      const now = new Date().toISOString();
      setStatus("Baking the matte's smoothing and cleanup once, so every frame renders live…");
      const layer = await prepareVfxLayer({
        id: previous?.id ?? crypto.randomUUID(), name: recipe.name, enabled: true,
        rangeStart, rangeEnd, recipe, matteCache: cache, ...extras,
        createdAt: previous?.createdAt ?? now, updatedAt: now,
      }, asset.path, seconds(clip.sourceIn));
      upsertClipVfxLayer(clip.id, layer);
      setExtras({ envelope: layer.envelope, effectClipMedia: layer.effectClipMedia, stableMatte: layer.stableMatte });
      setSaved(true);
      setStatus("VFX layer saved on the clip — the timeline monitor and export render it live");
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setFailure(message); setStatus("Saving the VFX layer failed: " + message.split("\n").at(-1));
    } finally { setRunning(false); setBusy(false); }
  };

  /**
   * Bake the saved layer's range to a video file through the compositor (for "Insert baked copy" and
   * "Replace range"), full size, with the clip's own sound for the span.
   */
  const bake = async (): Promise<MediaAsset | undefined> => {
    const layer = clip.vfxLayers?.find((item) => item.id === draftLayer()?.id) ?? draftLayer();
    const held = engine.current;
    if (!layer || !held || !layer.matteCache) return undefined;
    const outputName = clip.name.replace(/\.[^.]+$/, "") + "-" + maskEffectFileSuffix(layer.recipe.effect);
    setRunning(true); setBusy(true); setStartedAt(Date.now()); setElapsed(0); setFailure(undefined);
    let session: string | undefined;
    try {
      const frameFps = layer.matteCache.fps || media.fps;
      const count = Math.max(1, layer.matteCache.frameCount);
      const width = media.width - (media.width % 2), height = media.height - (media.height % 2);
      const opened = await renderVideoStart({
        projectId: project.id, name: outputName, width, height, fps: frameFps,
        audio: asset.hasAudio ? { path: asset.path, start: sourceStart, duration: count / frameFps } : undefined,
        pixelFormat: "yuv420", http: true,
      });
      session = opened.id;
      const sender = new FrameSender(opened);
      const packed = [new Uint8Array(0), new Uint8Array(0)];
      let sending: Promise<void> = Promise.resolve();
      for (let index = 0; index < count; index++) {
        const sourceAt = seconds(clip.sourceIn) + layer.rangeStart + index / frameFps;
        const document = vfxPreviewDocument(media, clip.id, seconds(clip.sourceIn), layer, sourceAt);
        const needs = held.needs(document, 0);
        await held.prepare(needs, false, true);
        if (held.failures.size) throw new Error(`could not read ${[...held.failures.keys()][0]}`);
        const frame = held.compose(document, index, width, height, false, readCompositorFlag());
        await sending;
        let bytes: Uint8Array = new Uint8Array(frame.pixels.buffer, frame.pixels.byteOffset, frame.pixels.byteLength);
        if (sender.format === "yuv420") { bytes = packYuv420(frame.pixels, width, height, packed[index & 1]); packed[index & 1] = bytes; }
        sending = sender.send(bytes);
        if (index % 12 === 0) setStatus(`Baking VFX ${index + 1} / ${count} · ${sender.describe()}`);
      }
      await sending;
      const finished = await renderVideoFinish(session);
      session = undefined;
      const probe = await probeMedia(finished.outputPath);
      const created: MediaAsset = {
        ...generatedAsset(finished.outputPath, probe),
        origin: {
          operation: "mask-effect", model: "Compositor (live mask VFX)", sourceAssetIds: [asset.id],
          parameters: { recipe: layer.recipe, rangeStart: layer.rangeStart, rangeEnd: layer.rangeEnd, matteCacheId: layer.matteCache.cacheId },
          createdAt: new Date().toISOString(),
        },
      };
      addAssets([created], { keepSelection: true });
      setResultState({ asset: created, log: `${finished.frames} frames · ${finished.encoder}` });
      setStatus("Baked " + created.name);
      return created;
    } catch (error) {
      if (session) await renderVideoCancel(session).catch(() => undefined);
      const message = error instanceof Error ? error.message : String(error);
      setFailure(message); setStatus("Baking the VFX failed: " + message.split("\n").at(-1));
      return undefined;
    } finally { setRunning(false); setBusy(false); }
  };

  const rangeLeft = sourceDuration ? rangeStart / sourceDuration * 100 : 0;
  const rangeWidth = sourceDuration ? (rangeEnd - rangeStart) / sourceDuration * 100 : 0;
  return <>
    <div className="clip-editor-body mask-editor-body">
      <div className="clip-editor-preview">
        <div className={"mask-preview-stage " + (source === "sam2" && pointMode !== "off" && !result && !proxy ? "picking" : !result && !proxy ? "placing" : "")}
          onPointerDown={previewPointerDown}
          onPointerMove={(event) => { if (placementDrag.current) positionEffect(event); }}
          onPointerUp={() => { placementDrag.current = false; }}
          onPointerCancel={() => { placementDrag.current = false; }}>
          <video ref={videoRef} key={result?.asset.path ?? proxy?.path ?? asset.path} src={convertFileSrc(result?.asset.path ?? proxy?.path ?? asset.path)}
            preload="auto" controls={Boolean(result || proxy)} muted={!result && !proxy} onLoadedMetadata={() => setLayoutRevision((value) => value + 1)} />
          {!result && !proxy && matteStyle === "clip" && effectClipPath && <video ref={effectVideoRef}
            className={`mask-vfx-clip placement-${placement}`} src={convertFileSrc(effectClipPath)}
            autoPlay muted playsInline loop={loop} style={{ ...markerStyle({ x: effectX, y: effectY }),
              opacity: effectOpacity, transform: `translate(-50%, -50%) scale(${effectScale / 100}) rotate(${effectRotation}deg)` }} />}
          {!result && !proxy && matteStyle !== "clip" && <div
            className={`mask-vfx-proxy ${matteStyle} placement-${placement}${loop ? " looping" : ""}`}
            style={{ ...markerStyle({ x: effectX, y: effectY }), opacity: effectOpacity,
              transform: `translate(-50%, -50%) scale(${effectScale / 100}) rotate(${effectRotation}deg)`,
              backgroundImage: `url("${gradientStrip}")`, animationDuration: `${Math.max(.2, 1 / effectSpeed)}s` }} />}
          {!result && !proxy && (source !== "sam2" || pointMode === "off") && <div className="mask-placement-guide"
            style={{ ...markerStyle({ x: effectX, y: effectY }),
              width: `${Math.min(95, effectScale * .62)}%`, height: `${Math.min(95, effectScale * .62)}%`,
              transform: `translate(-50%, -50%) rotate(${effectRotation}deg)` }}><Move size={14} /></div>}
          {!result && !proxy && points.map((point, index) => <span key={index} className={"mask-point" + (point.include ? "" : " exclude")}
            style={markerStyle(point)} title={(point.include ? "Include" : "Exclude") + ` point ${index + 1}`} />)}
          {!result && !proxy && source === "birefnet" && <div className="mask-auto-badge"><Sparkles size={13} /> Automatic foreground matte</div>}
          {showLive && <canvas ref={canvasRef} className="mask-live-canvas" />}
        </div>
        {!result && !proxy && <div className="scrub-strip"><div className="scrub-range" style={{ left: `${rangeLeft}%`, width: `${rangeWidth}%` }} />
          <input type="range" min={0} max={Math.max(0.1, sourceDuration)} step={0.02} value={scrub} onChange={(event) => setScrub(Number(event.target.value))} /></div>}
        <div className="scrub-actions">
          <span className="timecode">{formatTimecode(scrub)}</span>
          <button disabled={Boolean(result) || running} onClick={() => setRangeStart(Math.min(scrub, rangeEnd - 0.1))}>Set in</button>
          <button disabled={Boolean(result) || running} onClick={() => setRangeEnd(Math.max(scrub, rangeStart + 0.1))}>Set out</button>
          <button disabled={Boolean(result) || running} onClick={() => { setRangeStart(0); setRangeEnd(sourceDuration); }}>Whole clip</button>
          <span className="range-readout">{rangeStart.toFixed(2)}s – {rangeEnd.toFixed(2)}s</span>
          {matteCache && !result && <>
            <label className="check-row"><input type="checkbox" checked={live} onChange={(event) => { setLive(event.target.checked); setPreviewPlaying(false); drawn.current = ""; }} /> Live look</label>
            <button disabled={!showLive} onClick={() => setPreviewPlaying(!previewPlaying)}>{previewPlaying ? "Pause" : "Play range"}</button>
            {showLive && <span className="range-readout" title="Which compositor drew the preview, and how long it took">{previewInfo}</span>}
          </>}
          {rangeTooLong && <span className="range-warn">{requestedFrames} frames · maximum 900</span>}
        </div>
      </div>
      <div className="clip-editor-controls">
        <VisualModeSwitch mode="mask" disabled={running} onChange={onModeChange} />
        <div className="mask-source-switch">
          <button className="active" disabled>SAM 2 · {runtimeInfo?.sam2Ready ? "Ready" : "Check setup"}</button>
          <small>Track once; previews and final looks reuse the cached matte.</small>
        </div>
        {Boolean(clip.vfxLayers?.length) && <section className="mask-reactive">
          <div className="mask-reactive-title"><Layers3 size={14} /><span>Timeline VFX layers</span></div>
          {clip.vfxLayers!.map((layer) => <div className="mask-import-row" key={layer.id}>
            <label className="check-row"><input type="checkbox" checked={layer.enabled} disabled={running}
              onChange={(event) => upsertClipVfxLayer(clip.id, {
                ...layer, enabled: event.target.checked, updatedAt: new Date().toISOString(),
              })} /> {layer.name}</label>
            <button disabled={running} onClick={() => removeClipVfxLayer(clip.id, layer.id)}>Remove</button>
          </div>)}
          <small>These recipes stay attached to the source clip. Cached render files can be regenerated without replacing the edit.</small>
        </section>}
        {source === "sam2" ? <div className="mask-point-tools">
          <button className={pointMode === "include" ? "active include" : ""} disabled={running} onClick={() => setPointMode("include")}>+ Include</button>
          <button className={pointMode === "exclude" ? "active exclude" : ""} disabled={running} onClick={() => setPointMode("exclude")}>− Exclude</button>
          <button className={pointMode === "off" ? "active place" : ""} disabled={running} onClick={() => setPointMode("off")}><Move size={12} /> Place VFX</button>
          <button disabled={!points.length || running} onClick={() => setPoints((current) => current.slice(0, -1))}>Undo point</button>
          <button disabled={!points.length || running} onClick={() => setPoints([])}>Clear</button>
          <small>{points.length ? `${points.length} point${points.length === 1 ? "" : "s"}` : "Click the subject in the preview"}</small>
        </div> : <small className="panel-help">BiRefNet finds the dominant foreground automatically and keeps its soft hair/fur edges. SAM 2 is better when you need to choose a specific object.</small>}
        <div className="mask-catalog-heading"><strong>Complete VFX recipes</strong><small>Versioned · every control is replaced</small></div>
        <div className="mask-effect-grid">{MASK_VFX_RECIPES.map((item) => <button key={item.presetId}
          className={presetId === item.presetId ? "active" : ""} disabled={running}
          onClick={() => applyRecipe(item)} title={`Apply the complete ${item.name} recipe`}>
          <strong>{item.name}</strong><small>Recipe v{item.schemaVersion}</small></button>)}</div>
        <div className="mask-catalog-heading"><strong>Effect algorithm</strong><small>Changes only the core look</small></div>
        <div className="mask-effect-grid">{MASK_EFFECTS.map((item) => <button key={item.id} className={effect === item.id ? "active" : ""}
          disabled={running} onClick={() => {
            setEffect(item.id); setPresetId("custom"); setRecipeName("Custom VFX"); setResult(undefined);
            if (item.id === "animated-matte") setPointMode("off");
          }} title={item.description}>
          <strong>{item.name}</strong><small>{item.group}</small></button>)}</div>
        <section className="mask-reactive mask-vfx-builder">
          <div className="mask-reactive-title"><Layers3 size={14} /><span>VFX layer builder</span></div>
          <div className="mask-placement-switch">{MASK_PLACEMENTS.map((item) => <button key={item.value}
            className={placement === item.value ? "active" : ""} title={item.description} disabled={running}
            onClick={() => { setPlacement(item.value); setResult(undefined); }}>{item.label}</button>)}</div>
          <label className="studio-field"><span>Effect matte</span><select value={matteStyle} disabled={running}
            onChange={(event) => { setMatteStyle(event.target.value as MaskEffectMatteStyle); setResult(undefined); }}>
            {MASK_MATTE_STYLES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
          <small>{MASK_MATTE_STYLES.find((item) => item.value === matteStyle)?.description}</small>
          <div className="mask-import-row">
            <button type="button" disabled={running} onClick={() => { void importEffectClip(); }}><Upload size={13} /> Choose effect clip</button>
            <span title={effectClipPath}>{effectClipPath ? effectClipPath.split(/[\\/]/).pop() : "No clip selected"}</span>
          </div>
          {matteStyle === "clip" && <label className="studio-field"><span>Clip color</span><select value={effectClipColor} disabled={running}
            onChange={(event) => { setEffectClipColor(event.target.value as "original" | "gradient"); setResult(undefined); }}>
            <option value="original">Keep original VFX colors</option><option value="gradient">Use clip as gradient matte</option>
          </select></label>}
          <div className="mask-control-grid">
            <label className="studio-field"><span>Blend mode</span><select value={blendMode} disabled={running}
              onChange={(event) => setBlendMode(event.target.value as MaskVfxBlendMode)}>
              <option value="normal">Normal</option><option value="screen">Screen</option><option value="add">Add</option>
              <option value="multiply">Multiply</option><option value="overlay">Overlay</option><option value="soft-light">Soft light</option>
            </select></label>
            {matteStyle === "clip" && <label className="studio-field"><span>Extract matte</span><select value={keyMode} disabled={running}
              onChange={(event) => setKeyMode(event.target.value as MaskVfxKeyMode)}>
              <option value="alpha">Alpha</option><option value="luma">Luminance</option><option value="chroma">Chroma key</option>
            </select></label>}
          </div>
          {matteStyle === "clip" && keyMode !== "alpha" && <div className="mask-control-grid">
            <RangeField label="Key threshold" value={keyThreshold} min={0} max={1} step={0.01} onChange={setKeyThreshold} />
            <RangeField label="Key softness" value={keySoftness} min={0} max={1} step={0.01} onChange={setKeySoftness} />
            {keyMode === "chroma" && <label className="mask-color"><span>Key color</span><input type="color" value={keyColor} disabled={running} onChange={(event) => setKeyColor(event.target.value)} /></label>}
          </div>}
          <div className="mask-control-grid">
            <RangeField label="Opacity" value={effectOpacity} min={0} max={1} step={0.01} onChange={(value) => { setEffectOpacity(value); setResult(undefined); }} />
            <RangeField label="Scale" value={effectScale} min={10} max={300} step={1} suffix="%" onChange={(value) => { setEffectScale(value); setResult(undefined); }} />
            <RangeField label="Across" value={effectX} min={0} max={1} step={0.005} onChange={(value) => { setEffectX(value); setResult(undefined); }} />
            <RangeField label="Down" value={effectY} min={0} max={1} step={0.005} onChange={(value) => { setEffectY(value); setResult(undefined); }} />
            <RangeField label="Rotation" value={effectRotation} min={-180} max={180} step={1} suffix="°" onChange={(value) => { setEffectRotation(value); setResult(undefined); }} />
            <RangeField label="Pulse" value={pulseAmount} min={0} max={1} step={0.01} onChange={(value) => { setPulseAmount(value); setResult(undefined); }} />
            <RangeField label="Pulse rate" value={pulseRate} min={0.1} max={8} step={0.05} suffix="Hz" onChange={(value) => { setPulseRate(value); setResult(undefined); }} />
          </div>
          <label className="check-row"><input type="checkbox" checked={loop} disabled={running}
            onChange={(event) => { setLoop(event.target.checked); setResult(undefined); }} /> Loop animation</label>
          <small>Choose Place VFX, then drag directly in the preview. “Behind subject” uses the clean SAM 2 cutout as the top layer.</small>
        </section>
        <div className="mask-control-grid">
          <RangeField label="Strength" value={intensity} min={0} max={2} step={0.05} onChange={(value) => { setIntensity(value); setResult(undefined); }} />
          <RangeField label="Feather" value={feather} min={0} max={64} step={1} suffix="px" onChange={(value) => { setFeather(value); setResult(undefined); }} />
          <RangeField label="Expand edge" value={expand} min={-32} max={64} step={1} suffix="px" onChange={(value) => { setExpand(value); setResult(undefined); }} />
          <label className="mask-color"><span>Effect color</span><input type="color" value={color} disabled={running} onChange={(event) => { setColor(event.target.value); setResult(undefined); }} /></label>
          <label className="mask-color"><span>Secondary color</span><input type="color" value={secondaryColor} disabled={running} onChange={(event) => { setSecondaryColor(event.target.value); setResult(undefined); }} /></label>
        </div>
        <details className="mask-advanced" open><summary>Matte quality + motion</summary><div className="mask-control-grid">
          <RangeField label="Temporal smoothing" value={temporalSmooth} min={0} max={5} step={1} onChange={(value) => { setTemporalSmooth(value); setResult(undefined); }} />
          <RangeField label="Noise cleanup" value={cleanup} min={0} max={2000} step={50} suffix="px²" onChange={(value) => { setCleanup(value); setResult(undefined); }} />
          <RangeField label="Animation speed" value={effectSpeed} min={0.1} max={3} step={0.1} suffix="×" onChange={(value) => { setEffectSpeed(value); setResult(undefined); }} />
          <RangeField label="Particles / copies" value={particleAmount} min={0} max={2} step={0.05} onChange={(value) => { setParticleAmount(value); setResult(undefined); }} />
          <RangeField label="Clean holdout feather" value={occlusionFeather} min={0} max={16} step={0.5} suffix="px" onChange={setOcclusionFeather} />
          <RangeField label="Clean holdout expand" value={occlusionExpand} min={-8} max={8} step={1} suffix="px" onChange={setOcclusionExpand} />
        </div><small>Effect feather/expand shapes the look. Clean holdout controls only the original subject placed above rear VFX.</small></details>
        <details className="mask-advanced" open><summary>Passes + shot integration</summary>
          <div className="mask-control-grid">
            <RangeField label="Rear pass" value={rearAmount} min={0} max={1} step={0.01} onChange={setRearAmount} />
            <RangeField label="Subject lighting" value={subjectAmount} min={0} max={1} step={0.01} onChange={setSubjectAmount} />
            <RangeField label="Foreground pass" value={frontAmount} min={0} max={1} step={0.01} onChange={setFrontAmount} />
            <RangeField label="Motion blur" value={motionBlur} min={0} max={24} step={0.5} suffix="px" onChange={setMotionBlur} />
            <RangeField label="Light wrap" value={lightWrap} min={0} max={1} step={0.01} onChange={setLightWrap} />
            <RangeField label="Color spill" value={colorSpill} min={0} max={1} step={0.01} onChange={setColorSpill} />
            <RangeField label="Grain match" value={grain} min={0} max={1} step={0.01} onChange={setGrain} />
            <RangeField label="Displacement" value={displacement} min={0} max={24} step={0.5} suffix="px" onChange={setDisplacement} />
          </div><small>Rear preserves a clean subject sandwich; lighting integrates the effect with the subject; foreground lets sparks and energy cross the silhouette.</small>
        </details>
        <details className="mask-advanced mask-gradient-editor" open><summary>Animated gradient</summary>
          <label className="check-row"><input type="checkbox" checked={gradientEnabled} disabled={running}
            onChange={(event) => { setGradientEnabled(event.target.checked); setResult(undefined); }} /> Color the effect with this gradient</label>
          {gradientEnabled && <GradientEditor gradient={gradient} preview={gradientStrip}
            onChange={(over) => { setGradient((current) => ({ ...current, ...over,
              stops: over.stops ? over.stops.map((stop) => ({ ...stop })) : current.stops })); setResult(undefined); }} />}
          <small>The gradient’s Shift value is animated by Animation speed. Repeat and Mirror control how that motion cycles.</small>
        </details>
        <section className="mask-reactive mask-transitions"><div className="mask-reactive-title"><Sparkles size={14} /><span>Effect transitions</span></div>
          <div className="mask-control-grid">
            <label className="studio-field"><span>Animate in</span><select value={transitionIn} disabled={running} onChange={(event) => { setTransitionIn(event.target.value as MaskEffectTransition); setResult(undefined); }}>
              {MASK_TRANSITIONS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
            <label className="studio-field"><span>Animate out</span><select value={transitionOut} disabled={running} onChange={(event) => { setTransitionOut(event.target.value as MaskEffectTransition); setResult(undefined); }}>
              {MASK_TRANSITIONS.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
            <label className="studio-field"><span>Origin</span><select value={transitionOrigin} disabled={running}
              onChange={(event) => setTransitionOrigin(event.target.value as MaskVfxRecipeV1["transitionOrigin"])}>
              <option value="subject">Tracked subject</option><option value="effect">Effect anchor</option>
            </select></label>
          </div>
          {(transitionIn !== "none" || transitionOut !== "none") && <div className="mask-control-grid">
            {transitionIn !== "none" && <RangeField label="In duration" value={transitionInDuration} min={0.05} max={Math.max(0.05, Math.min(3, (rangeEnd - rangeStart) / 2))} step={0.05} suffix="s" onChange={(value) => { setTransitionInDuration(value); setResult(undefined); }} />}
            {transitionOut !== "none" && <RangeField label="Out duration" value={transitionOutDuration} min={0.05} max={Math.max(0.05, Math.min(3, (rangeEnd - rangeStart) / 2))} step={0.05} suffix="s" onChange={(value) => { setTransitionOutDuration(value); setResult(undefined); }} />}
          </div>}
          <small>{transitionIn === "none" && transitionOut === "none"
            ? "Static effect: fully active from the first frame through the last frame of the selected range."
            : `${MASK_TRANSITIONS.find((item) => item.value === transitionIn)?.description ?? ""} Out transitions run the selected motion in reverse.`}</small>
        </section>
        <section className="mask-reactive"><div className="mask-reactive-title"><Move size={14} /><span>Transform keyframes</span></div>
          <small>Scrub to a time, adjust position, scale, rotation, opacity, or strength, then capture that pose.</small>
          <div className="mask-point-tools">
            <button disabled={running || rangeEnd <= rangeStart} onClick={() => {
              const at = Math.max(0, Math.min(1, (scrub - rangeStart) / Math.max(.001, rangeEnd - rangeStart)));
              const frame: MaskVfxKeyframe = { id: crypto.randomUUID(), at, easing: "smooth", values: maskVfxAnimatedValues(currentRecipe()) };
              setKeyframes((current) => [...current.filter((item) => Math.abs(item.at - at) > .002), frame].sort((a, b) => a.at - b.at));
            }}>Capture at {Math.round(Math.max(0, Math.min(1, (scrub - rangeStart) / Math.max(.001, rangeEnd - rangeStart))) * 100)}%</button>
            <button disabled={running || !keyframes.length} onClick={() => setKeyframes([])}>Clear keyframes</button>
            <small>{keyframes.length} keyframe{keyframes.length === 1 ? "" : "s"}</small>
          </div>
        </section>
        <label className="check-row"><input type="checkbox" checked={invert} disabled={running} onChange={(event) => { setInvert(event.target.checked); setResult(undefined); }} /> Invert mask (affect everything outside the selection)</label>
        <div className="mask-reactive">
          <div className="mask-reactive-title"><AudioLines size={14} /><span>Audio reactive</span></div>
          <label className="studio-field"><span>Driver</span><select value={reactive} disabled={running || !asset.hasAudio} onChange={(event) => { setReactive(event.target.value as MaskEffectRequest["audioReactive"]); setResult(undefined); }}>
            {AUDIO_REACTIVE_MODES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></label>
          {reactive !== "off" && asset.hasAudio && <><RangeField label="Reaction amount" value={reactiveAmount} min={0} max={2} step={0.05} onChange={(value) => { setReactiveAmount(value); setResult(undefined); }} />
            <RangeField label="Release smoothing" value={reactiveSmoothing} min={0} max={1} step={0.05} onChange={(value) => { setReactiveSmoothing(value); setResult(undefined); }} /></>}
          {!asset.hasAudio && <small>This clip has no audio stream, so the effect uses constant strength.</small>}
        </div>
        <div className="mask-point-tools">
          <button disabled={!canTrack || Boolean(matteCache)} onClick={() => { void trackSubject(); }}>
            {matteCache ? "Matte cached" : "1 · Track subject"}
          </button>
          {result && <button onClick={() => setResultState(undefined)}>Back to setup</button>}
        </div>
        {matteCache && <small className="runtime-checks">Cached {matteCache.frameCount} lossless matte frames at {matteCache.width}×{matteCache.height}. The preview renders the look live on them; temporal smoothing and cleanup are baked when you save.</small>}
        <button className="render-button" disabled={!canPreview && !matteCache} onClick={() => { void saveLayer(); }}><Layers3 size={16} /> {running ? "Working…" : checkingRuntime ? "Checking mask runtime…" : saved ? "2 · Saved · save again" : "2 · Save VFX layer (renders live)"}</button>
        {preflightMessage && !matteCache && <small className="runtime-checks mask-preflight">{preflightMessage}.</small>}
        {running && <div className="ltx-job"><span className="ltx-spinner" /><div><strong>Mask Effects running · {Math.floor(elapsed / 60)}:{String(elapsed % 60).padStart(2, "0")}</strong><small>Tracking, baking the matte, or rendering through the compositor.</small></div></div>}
        {proxy?.log && <details className="worker-log"><summary>Preview renderer report</summary><pre>{proxy.log}</pre></details>}
        {result && <div className="ltx-job complete"><span>✓</span><div><strong>Baked</strong><small>{result.asset.path}</small></div></div>}
        {result?.log && <details className="worker-log"><summary>What the worker reported</summary><pre>{result.log}</pre></details>}
        {failure && <div className="worker-failure"><div><strong>Mask Effects failed</strong><button onClick={() => setFailure(undefined)} aria-label="Dismiss"><X size={12} /></button></div><pre>{failure}</pre></div>}
        {!ready && <small className="runtime-checks">{source === "sam2"
          ? `SAM 2 unavailable (${runtimeInfo?.sam2State ?? "checking"}). Its checkpoint and a working Python runtime with the sam2 package are both required.`
          : "BiRefNet needs models\\BiRefNet and the Image Studio runtime."}</small>}
      </div>
    </div>
    <div className="dialog-actions">
      <button onClick={onClose} disabled={running}>Close</button>
      <button disabled={!saved || running} title="Render the saved layer's range to a file with the compositor, and insert it after the clip"
        onClick={() => { void (async () => { const baked = result?.asset ?? await bake(); if (baked) { insertAssetAfterClip(clip.id, baked); onClose(); } })(); }}>Insert baked copy</button>
      <button disabled={!saved || running} title="Render the saved layer's range to a file with the compositor, and put it in place of that range"
        onClick={() => { void (async () => { const baked = result?.asset ?? await bake(); if (baked) { replaceClipRangeWithAsset(clip.id, rangeStart, rangeEnd, baked); onClose(); } })(); }}>Replace range (bake)</button>
      <button className="primary" disabled={!saved || running} onClick={onClose}>Done · keep VFX layer</button>
    </div>
  </>;
}

function RangeField({ label, value, min, max, step, suffix = "", onChange }: {
  label: string; value: number; min: number; max: number; step: number; suffix?: string; onChange: (value: number) => void;
}) {
  return <label className="mask-range"><span>{label}<b>{Number(value.toFixed(2))}{suffix}</b></span>
    <input type="range" value={value} min={min} max={max} step={step} onChange={(event) => onChange(Number(event.target.value))} /></label>;
}
