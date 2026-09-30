import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown, ArrowUp, ChevronDown, ChevronUp, ClipboardCopy, Film, Layers3, Link, Plus, RefreshCw, Scissors, Trash2, Unlink, WandSparkles, X,
} from "lucide-react";
import {
  H3_ASPECTS, H3_LOW_VRAM_SHORT_EDGE, H3_TRAINED_SHORT_EDGE, H3_MAX_LONG_EDGE,
  clampH3Frames, fitH3Dimensions, h3Seconds, h3CanvasFits,
  H3_MIN_FRAMES, H3_MAX_FRAMES, H3_FRAMES_PER_CHUNK, H3_RESOLUTION_TIERS,
  snapH3Dimension, calculateH3Tokens, type H3ResolutionTier,
  H3_REF_IMAGE_DETAILS, type H3RefImageDetail,
  H3_REF_VIDEO_DETAILS, h3RefVideoShortEdge, type H3RefVideoDetail,
  buildH3RefRequest,
  H3_ATTENTION_BACKENDS, type H3AttentionBackend,
} from "../generation/h3";
import {
  H3_REFERENCE_INTENTS, H3_STRENGTHS, allocateH3Labels, allocateH3Subjects,
  availableH3ReferenceIntents, buildH3ReferencePrompt, defaultH3ReferenceIntents,
  h3DescriptionWordCount, h3Marker, h3ReferenceCapError, h3ReferenceLimits, h3ShotTimingIssues,
  isAudioIntent, strengthOf,
  type H3ReferenceDraft, type H3ReferenceIntent, type H3ReferenceKind, type H3Shot,
  type H3Strength,
} from "../generation/h3Ref";
import { seconds } from "../../domain/time";
import { isAudioAsset } from "../../domain/media";
import {
  cloneH3DirectorSession, h3DirectorSessionFingerprint, normalizeH3DirectorWorkspace,
  type H3DirectorSession, type H3DirectorTake,
} from "../../domain/h3Director";
import { generatedAsset, probeMedia, runtimeStatus, type AiRuntimeStatus, type MiniMaxH3Result } from "../../services/modelService";
import { useEditor } from "../../store";

/** One completed generation. Ref2V runs take minutes, so a re-roll must never overwrite one. */
type Take = H3DirectorTake;

const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const uid = () => Math.random().toString(36).slice(2, 10);
const resolveSeed = (value: number) => value >= 0 ? value : Math.floor(Math.random() * 2_147_483_647);

/** Conservative first-release limits, mirrored from src-tauri/src/ai/h3_ref.rs. */
// The checkpoint's own limits, from MiniMax's model card: "Images: <= 9 images. Videos: <= 3
// clips. Audio: <= 3 clips. Maximum number of files across all input types is 12." These used to
// be 3/1/1/4 -- a 16 GB profile hard-coded as if it were the model's contract, which put most of
// what Ref2VA can do out of reach. The conservative numbers survive as an advisory below.
const MAX_IMAGES = 9;
const MAX_VIDEOS = 3;
const MAX_AUDIOS = 3;
const MAX_REFERENCES = 12;
/** Above this, a 16 GB card is likely to struggle. Advisory, not a blocker. */
const SAFE_REFERENCES = 4;
const MIN_TIMED_SECONDS = 2;
const MAX_TIMED_SECONDS = 15;

export function Ref2V() {
  const {
    project, addAssets, addGeneratedAvPair, busy, setBusy, setStatus, activeWorkspace,
    setH3DirectorWorkspace,
  } = useEditor();
  // App keys this workspace by project id, so this initializer runs once per opened project and
  // never leaks the previous film's cast or takes into the next one.
  const [savedWorkspace] = useState(() => normalizeH3DirectorWorkspace(project.h3Director));
  const saved = savedWorkspace.session;
  const [runtime, setRuntime] = useState<AiRuntimeStatus>();

  // Reference cast
  const [references, setReferences] = useState<H3ReferenceDraft[]>(saved.references);
  const [pickAssetId, setPickAssetId] = useState("");

  // The six fields. `rawPrompt` overrides the assembled result once edited, matching the
  // "local by default, editable before generation" contract the old panel already had.
  const [style, setStyle] = useState(saved.style);
  const [summary, setSummary] = useState(saved.summary);
  const [shots, setShots] = useState<H3Shot[]>(saved.shots);
  const [soundscape, setSoundscape] = useState(saved.soundscape);
  const [music, setMusic] = useState(saved.music);
  const [rawPrompt, setRawPrompt] = useState(saved.rawPrompt);
  const [promptTab, setPromptTab] = useState<"built" | "raw">(saved.promptTab);

  // Output settings. These are deliberately Ref2V's OWN state, not shared with the FL2VA panel.
  // AiPanel already documents the same reasoning one level up: sharing carries a value across
  // when the model selector switches. Share these and generating here would silently change the
  const [aspect, setAspect] = useState(saved.aspect);
  const [tier, setTier] = useState<H3ResolutionTier>(saved.tier);
  const [customWidth, setCustomWidth] = useState(saved.customWidth);
  const [customHeight, setCustomHeight] = useState(saved.customHeight);
  const [lockRatio, setLockRatio] = useState(saved.lockRatio);
  const [frames, setFrames] = useState(saved.frames);
  const [seed, setSeed] = useState(saved.seed);
  const [attentionBackend, setAttentionBackend] = useState<H3AttentionBackend>(saved.attentionBackend);
  // Base, not Turbo. The Ref2V adapter LightX2V published is v0.1, which they describe as a
  // preview whose "image details still need improvement", and there is no 8-step Ref2V adapter at
  // all. ComfyUI's own shipped R2V workflow uses no LoRA and 20 steps -- that is the recipe to
  // default to; Turbo stays as the draft gear.
  const [samplingMode, setSamplingMode] = useState<"turbo-4" | "base">(saved.samplingMode);
  const [steps, setSteps] = useState(saved.steps);
  // Turbo's count is fixed by the distillation. Base's is a real choice, so switching modes moves
  // the step count with it rather than carrying a number the other mode would reject.
  const chooseSampling = (mode: "turbo-4" | "base") => {
    setSamplingMode(mode);
    setSteps(mode === "turbo-4" ? 4 : 20);
  };
  // 12.0, because that is what the adapter was distilled against and what the released scheduler
  // config declares. This defaulted to 8.0 and was sent on every request, so the Ref2V tab was
  // running a four-step distillation on a sigma grid it had never seen -- soft, artefacted output,
  // and a video/audio shift ratio of 8:3 where the two-clock design expects 12:3.
  const [videoShift, setVideoShift] = useState(saved.videoShift);
  const [refImageDetail, setRefImageDetail] = useState<H3RefImageDetail>(saved.refImageDetail);
  const [refVideoDetail, setRefVideoDetail] = useState<H3RefVideoDetail>(saved.refVideoDetail);
  const [vaeTile, setVaeTile] = useState(saved.vaeTile);
  const [offload, setOffload] = useState(saved.offload);

  // Takes
  const [takes, setTakes] = useState<Take[]>(savedWorkspace.takes);
  const [selectedTakeId, setSelectedTakeId] = useState<string | undefined>(savedWorkspace.selectedTakeId);
  const [generating, setGenerating] = useState(false);
  const [failure, setFailure] = useState<string>();
  const [elapsed, setElapsed] = useState(0);
  const startedAt = useRef<number | undefined>(undefined);

  const videoRef = useRef<HTMLVideoElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  useEffect(() => {
    if (activeWorkspace !== "ref2v") return;
    let cancelled = false;
    void runtimeStatus().then((value) => { if (!cancelled) setRuntime(value); }).catch(() => undefined);
    return () => { cancelled = true; };
  }, [activeWorkspace]);

  useEffect(() => {
    if (!generating) return;
    const timer = window.setInterval(() => {
      setElapsed(startedAt.current ? (Date.now() - startedAt.current) / 1000 : 0);
    }, 500);
    return () => window.clearInterval(timer);
  }, [generating]);

  const referenceAssets = project.assets.filter((asset) =>
    isAudioAsset(asset) || ["image", "video", "generated"].includes(asset.kind));

  useEffect(() => {
    if (!pickAssetId && referenceAssets[0]) setPickAssetId(referenceAssets[0].id);
  }, [referenceAssets, pickAssetId]);

  const assetOf = (assetId: string) => project.assets.find((item) => item.id === assetId);
  const assetInfo = (assetId: string) => {
    const asset = assetOf(assetId);
    return { name: asset?.name ?? "missing asset", hasAudio: Boolean(asset?.hasAudio) };
  };

  const applyAspectAndTier = (newAspect: string, newTier: H3ResolutionTier) => {
    setAspect(newAspect);
    setTier(newTier);
    if (newTier === "custom") return;
    const tierConfig = H3_RESOLUTION_TIERS.find((t) => t.id === newTier);
    const shortEdge = tierConfig?.shortEdge ?? H3_LOW_VRAM_SHORT_EDGE;
    const [left, right] = newAspect.split(":").map(Number);
    const dims = fitH3Dimensions(left, right, shortEdge);
    setCustomWidth(dims.width);
    setCustomHeight(dims.height);
  };

  // Both axes run to the checkpoint's own limits now (768 short edge, 1344 long edge). The area
  // budget is what actually binds, and `canvasFits` below reports it rather than silently snapping.
  const updateCustomWidth = (raw: number) => {
    const snappedW = snapH3Dimension(raw, 128, H3_MAX_LONG_EDGE);
    setCustomWidth(snappedW);
    setTier("custom");
    if (lockRatio && customWidth > 0) {
      const ratio = customHeight / customWidth;
      const snappedH = snapH3Dimension(Math.round(snappedW * ratio), 128, H3_MAX_LONG_EDGE);
      setCustomHeight(snappedH);
    }
  };

  const updateCustomHeight = (raw: number) => {
    const snappedH = snapH3Dimension(raw, 128, H3_MAX_LONG_EDGE);
    setCustomHeight(snappedH);
    setTier("custom");
    if (lockRatio && customHeight > 0) {
      const ratio = customWidth / customHeight;
      const snappedW = snapH3Dimension(Math.round(snappedH * ratio), 128, H3_MAX_LONG_EDGE);
      setCustomWidth(snappedW);
    }
  };

  const labelled = allocateH3Labels(references, assetInfo);
  const subjects = allocateH3Subjects(labelled);
  const limits = h3ReferenceLimits(references);
  const dimensions = useMemo(() => ({
    width: customWidth,
    height: customHeight,
    value: `${customWidth}x${customHeight}`,
  }), [customWidth, customHeight]);
  const alignedFrames = clampH3Frames(frames);
  const wordCount = h3DescriptionWordCount({ style, shots });

  const builtPrompt = buildH3ReferencePrompt({
    style, summary, shots, soundscape, music, references, assetInfo,
  });
  const effectivePrompt = rawPrompt.trim() || builtPrompt;
  const timingIssues = h3ShotTimingIssues(shots, h3Seconds(alignedFrames));

  const directorSession: H3DirectorSession = {
    references, style, summary, shots, soundscape, music, rawPrompt, promptTab,
    aspect, tier, customWidth, customHeight, lockRatio, frames, seed, attentionBackend,
    samplingMode, steps, videoShift, refImageDetail, refVideoDetail, vaeTile, offload,
  };
  const sessionFingerprint = h3DirectorSessionFingerprint(directorSession);

  // The Director document follows the project, not the component lifetime. Write synchronously
  // after each committed React change so Ctrl+S on the next event can never serialize stale text.
  useEffect(() => {
    setH3DirectorWorkspace({
      schemaVersion: 1,
      updatedAt: new Date().toISOString(),
      session: cloneH3DirectorSession(directorSession),
      takes,
      selectedTakeId,
    });
  }, [sessionFingerprint, takes, selectedTakeId, setH3DirectorWorkspace]);

  /** Audio budget counts a video's enabled soundtrack, matching the Rust validator. */
  const timedSeconds = (() => {
    let video = 0, audio = 0;
    for (const entry of labelled) {
      const asset = assetOf(entry.reference.assetId);
      const duration = asset ? seconds(asset.duration) : 0;
      if (entry.video !== undefined) video += duration;
      if (entry.audio !== undefined) audio += duration;
    }
    return { video, audio };
  })();

  type PreflightIssue = {
    severity: "block" | "warning";
    message: string;
    field?: "cast" | "shot" | "prompt";
  };
  const preflightIssues: PreflightIssue[] = (() => {
    const issues: PreflightIssue[] = [];
    const block = (message: string, field?: PreflightIssue["field"]) =>
      issues.push({ severity: "block", message, field });
    const warn = (message: string, field?: PreflightIssue["field"]) =>
      issues.push({ severity: "warning", message, field });
    if (!runtime) block("Checking the Ref2VA runtime…");
    else if (!runtime.minimaxH3RefReady) {
      block(`The Ref2VA pack is not ready (${runtime.minimaxH3RefState}). Run: .venv\\Scripts\\python.exe scripts\\models.py install minimax-h3-ref2va --accept-licenses`);
    }
    if (!references.length) block("Add at least one image or video reference.", "cast");
    if (references.some((item) => !assetOf(item.assetId))) block("Remove the missing reference asset from the tray.", "cast");
    const referenceCapError = references.length ? h3ReferenceCapError(limits) : undefined;
    if (referenceCapError) block(referenceCapError, "cast");
    for (const reference of references) {
      if (reference.kind === "image") continue;
      const asset = assetOf(reference.assetId);
      const duration = asset ? seconds(asset.duration) : 0;
      if (duration < MIN_TIMED_SECONDS || duration > MAX_TIMED_SECONDS) {
        block(`${asset?.name ?? "A reference"} is ${duration.toFixed(1)}s; every video and audio reference must be ${MIN_TIMED_SECONDS}–${MAX_TIMED_SECONDS} seconds.`, "cast");
      }
    }
    if (timedSeconds.video > MAX_TIMED_SECONDS || timedSeconds.audio > MAX_TIMED_SECONDS) {
      block(`Reference totals must stay within ${MAX_TIMED_SECONDS}s per modality (video ${timedSeconds.video.toFixed(1)}s, audio ${timedSeconds.audio.toFixed(1)}s).`, "cast");
    }
    if (!h3CanvasFits(dimensions.width, dimensions.height)) {
      block(`${dimensions.value} is ${(dimensions.width * dimensions.height / 1000).toFixed(0)}k pixels; MiniMax-H3 caps a canvas at 768x1344 (1032k). Lower one axis.`);
    }
    if (!effectivePrompt.trim()) block("Describe the shot before generating.", "prompt");
    if (!shots.some((shot) => shot.description.trim())) block("Write what happens in Shot 1 — that is the only field the model truly needs.", "shot");
    for (const issue of timingIssues) {
      if (issue.severity === "block") block(issue.message, "shot");
      else warn(issue.message, "shot");
    }
    if (limits.total > SAFE_REFERENCES) {
      warn(`${limits.total} references is above the ${SAFE_REFERENCES} this pack was profiled on; expect heavy VRAM use.`, "cast");
    }
    if (Math.min(dimensions.width, dimensions.height) > H3_LOW_VRAM_SHORT_EDGE) {
      warn(`${dimensions.value} is above the 544p profile. This is the canvas MiniMax generates at, and it is much slower.`);
    }
    if (refVideoDetail !== "safe" && limits.videos > 0) {
      warn(`Reference video at ${h3RefVideoShortEdge(refVideoDetail)} px can exhaust VRAM in the conditioner; 384 is the safe tier.`, "cast");
    }
    return issues;
  })();
  const blockers = preflightIssues.filter((issue) => issue.severity === "block");
  const warnings = preflightIssues.filter((issue) => issue.severity === "warning");
  const preflight = blockers[0];

  // ---------------------------------------------------------------- cast editing

  const patchReference = (id: string, patch: Partial<H3ReferenceDraft>) => {
    setReferences((current) => current.map((item) => item.id === id ? { ...item, ...patch } : item));
    setRawPrompt("");
  };

  const addReference = () => {
    const asset = referenceAssets.find((item) => item.id === pickAssetId);
    if (!asset || references.some((item) => item.assetId === asset.id)) return;
    const kind: H3ReferenceKind = isAudioAsset(asset) ? "audio" : asset.kind === "image" ? "image" : "video";
    setReferences((current) => [...current, {
      id: uid(), assetId: asset.id, kind,
      intents: defaultH3ReferenceIntents(kind),
      strengths: {},
      includeEmbeddedAudio: true,
      subjects: [],
    }]);
    setRawPrompt("");
  };

  const moveReference = (id: string, delta: number) => {
    setReferences((current) => {
      const index = current.findIndex((item) => item.id === id);
      const target = index + delta;
      if (index < 0 || target < 0 || target >= current.length) return current;
      const next = current.slice();
      [next[index], next[target]] = [next[target], next[index]];
      return next;
    });
    setRawPrompt("");
  };

  const toggleIntent = (reference: H3ReferenceDraft, intent: H3ReferenceIntent) => {
    const has = reference.intents.includes(intent);
    const intents = has
      ? reference.intents.filter((value) => value !== intent)
      : [...reference.intents, intent];
    patchReference(reference.id, {
      intents: intents.length ? intents : defaultH3ReferenceIntents(reference.kind),
      // Continuation only makes sense with the soundtrack carried across the cut.
      includeEmbeddedAudio: !has && intent === "continuation" ? true : reference.includeEmbeddedAudio,
    });
  };

  const setAudioLane = (reference: H3ReferenceDraft, on: boolean) => {
    // Roles are NOT filtered out when the lane closes. The old panel dropped them, so toggling
    // audio off and on again silently lost the user's voice/ambience/music choices.
    patchReference(reference.id, { includeEmbeddedAudio: on });
  };

  /** The audio reference, if any, that was demuxed out of this video reference. */
  const splitSiblingOf = (reference: H3ReferenceDraft) =>
    references.find((item) => item.splitAudioFrom === reference.id);

  /**
   * Demux a video reference's soundtrack into a standalone wav and add it back as its own audio
   * reference, with the video's own soundtrack switched off.
   *
   * This is NOT equivalent to leaving the soundtrack embedded. A sound-bearing video packs its
   * audio rows immediately before its video rows and shares their rotary origin, so the two are
   * time-aligned exactly as the generated audio and picture are. Split apart, they become two
   * sequential reference blocks with no alignment between them: good for carrying a voice, a
   * room tone or a music bed across, wrong for anything that has to stay in sync with the
   * picture it came from. The wav is inserted *before* the video so the presentation labels
   * (<Audio 1> then <Video 1>) match what the embedded form produced and any prompt text that
   * already names them keeps pointing at the same media.
   */
  const splitSoundtrack = async (reference: H3ReferenceDraft) => {
    const asset = assetOf(reference.assetId);
    if (!asset || splitSiblingOf(reference)) return;
    const duration = seconds(asset.duration);
    const stem = asset.name.replace(/\.[^.]+$/, "");
    setBusy(true);
    setStatus("Choose where to save the extracted soundtrack…");
    try {
      const target = await save({
        defaultPath: `${stem}-soundtrack.wav`,
        filters: [{ name: "WAV audio", extensions: ["wav"] }],
      });
      if (!target) { setStatus("Soundtrack extraction canceled"); return; }
      setStatus(`Extracting ${asset.name}'s soundtrack…`);
      await invoke<string>("materialize_trimmed_clip", {
        request: {
          path: asset.path, outputPath: target,
          sourceInSeconds: 0, sourceOutSeconds: duration,
          kind: "audio",
        },
      });
      const probe = await probeMedia(target);
      const wav = generatedAsset(target, probe, "audio");
      addAssets([wav]);
      // The audio roles the soundtrack already carried move with it; the picture keeps its own.
      const carried = reference.intents.filter((intent) => isAudioIntent(intent));
      const audioReference: H3ReferenceDraft = {
        id: uid(), assetId: wav.id, kind: "audio",
        intents: carried.length ? carried : defaultH3ReferenceIntents("audio"),
        strengths: reference.strengths,
        includeEmbeddedAudio: true,
        subjects: [],
        splitAudioFrom: reference.id,
      };
      setReferences((current) => {
        const index = current.findIndex((item) => item.id === reference.id);
        if (index < 0) return current;
        const next = current.slice();
        // The picture keeps only its visual roles. Falling back to the defaults matters: a video
        // whose every selected role was an audio one would otherwise be left with no role at all.
        const kept = next[index].intents.filter((intent) => !isAudioIntent(intent));
        next[index] = {
          ...next[index],
          includeEmbeddedAudio: false,
          intents: kept.length ? kept : defaultH3ReferenceIntents("video"),
        };
        next.splice(index, 0, audioReference);
        return next;
      });
      setRawPrompt("");
      setStatus(`${asset.name}'s soundtrack is now a separate reference and is in Project Media`);
    } catch (error) {
      setFailure(errorText(error));
      setStatus("Could not extract the soundtrack");
    } finally {
      setBusy(false);
    }
  };

  // ---------------------------------------------------------------- shots

  const patchShot = (id: string, patch: Partial<H3Shot>) => {
    setShots((current) => current.map((shot) => shot.id === id ? { ...shot, ...patch } : shot));
    setRawPrompt("");
  };

  // ---------------------------------------------------------------- generate

  const generate = async () => {
    if (preflight) { setStatus("Ref2V needs attention: " + preflight.message); return; }
    setBusy(true);
    setFailure(undefined);
    setStatus("Choose where to save the Ref2V take…");
    let outputPath: string | null;
    try {
      outputPath = await save({
        defaultPath: `Ref2V-${Date.now()}.mp4`,
        filters: [{ name: "MPEG-4 Video", extensions: ["mp4"] }],
      });
    } catch (error) {
      setFailure(errorText(error));
      setStatus("Could not open the Ref2V output picker");
      setBusy(false);
      return;
    }
    if (!outputPath) { setStatus("Ref2V generation canceled"); setBusy(false); return; }

    const usedSeed = resolveSeed(seed);
    setGenerating(true);
    startedAt.current = Date.now();
    setElapsed(0);
    setStatus(`MiniMax-H3 Ref2VA is generating ${h3Seconds(alignedFrames).toFixed(1)}s of video and audio (seed ${usedSeed})…`);

    try {
      const result = await invoke<MiniMaxH3Result>("generate_minimax_h3_ref", {
        // One builder, shared with the AI panel's reference mode. See `buildH3RefRequest`.
        request: buildH3RefRequest({
          prompt: effectivePrompt,
          references: references.map((item) => ({
            kind: item.kind,
            path: assetOf(item.assetId)!.path,
            includeEmbeddedAudio: item.includeEmbeddedAudio,
          })),
          outputPath,
          width: dimensions.width, height: dimensions.height,
          numFrames: alignedFrames, seed: usedSeed,
          attentionBackend, offload,
          samplingMode,
          steps,
          videoShift,
          refImageDetail,
          refVideoDetail,
          vaeTile,
        }),
      });
      const takeSession = cloneH3DirectorSession({ ...directorSession, seed: usedSeed });
      const take: Take = {
        id: uid(), index: takes.length + 1,
        videoPath: result.outputPath, audioPath: result.audioPath,
        seed: usedSeed, frames: alignedFrames,
        width: dimensions.width, height: dimensions.height,
        at: new Date().toLocaleTimeString(), createdAt: new Date().toISOString(),
        log: result.log, onTimeline: false,
        session: takeSession,
        sessionFingerprint: h3DirectorSessionFingerprint(takeSession),
      };
      setTakes((current) => [...current, take]);
      setSelectedTakeId(take.id);
      setStatus(`Ref2V take ${take.index} finished in ${elapsed.toFixed(0)}s`);
    } catch (error) {
      setFailure(errorText(error));
      setStatus("Ref2V generation failed");
    } finally {
      setGenerating(false);
      setBusy(false);
    }
  };

  const selectedTake = takes.find((take) => take.id === selectedTakeId);

  const restoreTakeSession = (take: Take) => {
    const session = cloneH3DirectorSession(take.session);
    setReferences(session.references);
    setStyle(session.style);
    setSummary(session.summary);
    setShots(session.shots);
    setSoundscape(session.soundscape);
    setMusic(session.music);
    setRawPrompt(session.rawPrompt);
    setPromptTab(session.promptTab);
    setAspect(session.aspect);
    setTier(session.tier);
    setCustomWidth(session.customWidth);
    setCustomHeight(session.customHeight);
    setLockRatio(session.lockRatio);
    setFrames(session.frames);
    setSeed(session.seed);
    setAttentionBackend(session.attentionBackend);
    setSamplingMode(session.samplingMode);
    setSteps(session.steps);
    setVideoShift(session.videoShift);
    setRefImageDetail(session.refImageDetail);
    setRefVideoDetail(session.refVideoDetail);
    setVaeTile(session.vaeTile);
    setOffload(session.offload);
    setStatus(`Restored the exact cast, prompt and sampler setup from take ${take.index}`);
  };

  /**
   * The adapter writes an mp4 with NO audio stream plus a sidecar wav, deliberately, so exports
   * never double the audio. So the preview is two elements driven as one -- a plain <video> here
   * would play silent and read as a foley failure.
   */
  const syncAudioTo = (action: "play" | "pause" | "seek") => {
    const video = videoRef.current, audio = audioRef.current;
    if (!video || !audio) return;
    if (action === "seek") { audio.currentTime = video.currentTime; return; }
    if (action === "play") { audio.currentTime = video.currentTime; void audio.play(); return; }
    audio.pause();
  };

  const sendToTimeline = async (take: Take) => {
    if (take.onTimeline) return;
    setBusy(true);
    try {
      const [videoProbe, audioProbe] = await Promise.all([
        probeMedia(take.videoPath), probeMedia(take.audioPath),
      ]);
      const videoAsset = generatedAsset(take.videoPath, videoProbe);
      const audioAsset = generatedAsset(take.audioPath, audioProbe, "audio");
      addAssets([videoAsset, audioAsset]);
      // Bypasses snapping on purpose: snapping each track independently would drift the
      // soundtrack against its own picture.
      addGeneratedAvPair(videoAsset.id, audioAsset.id);
      setTakes((current) => current.map((item) => item.id === take.id
        ? { ...item, onTimeline: true, videoAssetId: videoAsset.id, audioAssetId: audioAsset.id }
        : item));
      setStatus(`Take ${take.index} placed on V1 and A1 as a linked pair`);
    } catch (error) {
      setFailure(errorText(error));
      setStatus("Could not place the take on the timeline");
    } finally {
      setBusy(false);
    }
  };

  const deleteTake = (take: Take) => {
    setTakes((current) => current.filter((item) => item.id !== take.id));
    if (selectedTakeId === take.id) {
      const remaining = takes.filter((item) => item.id !== take.id);
      setSelectedTakeId(remaining.length ? remaining[remaining.length - 1].id : undefined);
    }
    // Both files are left on disk on purpose: they are in a location the user chose, and a UI
    // list is the wrong place to delete someone's chosen output from.
    setStatus(`Take ${take.index} removed from the list; its mp4 and wav are still on disk`);
  };

  // ---------------------------------------------------------------- render

  const intentOf = (intent: H3ReferenceIntent) => H3_REFERENCE_INTENTS.find((item) => item.value === intent);
  const intentLabel = (intent: H3ReferenceIntent) => intentOf(intent)?.label ?? intent;
  const intentHint = (intent: H3ReferenceIntent) => intentOf(intent)?.hint;

  /* A worker traceback is longer than the box, and reading it by scrolling is how a real cause
     gets missed. Copy takes the whole string, not the part that happens to be on screen. */
  const failureBox = (title: string, body: string) => <div className="worker-failure">
    <div>
      <strong>{title}</strong>
      <button aria-label="Copy the full message" title="Copy the whole message, including the part scrolled out of view"
        onClick={() => { void navigator.clipboard.writeText(body); }}><ClipboardCopy size={12} /></button>
    </div>
    <pre>{body}</pre>
  </div>;

  return <main className="ref2v-studio">
    <aside className="ref2v-cast">
      <div className="ref2v-title">
        <span><Layers3 size={18} /></span>
        <div><strong>Ref2V</strong><small>References to video and audio</small></div>
      </div>

      <section className="studio-card">
        <div className="studio-card-heading">
          <strong>Reference cast</strong>
          <span>{limits.images}/{MAX_IMAGES} img · {limits.videos}/{MAX_VIDEOS} vid · {limits.effectiveAudios}/{MAX_AUDIOS} aud · {limits.total}/{MAX_REFERENCES}</span>
        </div>
        <div className="ref2v-add">
          <select value={pickAssetId} disabled={busy || !referenceAssets.length}
            onChange={(event) => setPickAssetId(event.target.value)}>
            {referenceAssets.length
              ? referenceAssets.map((asset) => <option key={asset.id} value={asset.id}>{asset.name}</option>)
              : <option value="">Import reference media first</option>}
          </select>
          <button disabled={busy || !pickAssetId} onClick={addReference}><Plus size={14} /> Add</button>
        </div>

        {labelled.map((entry, index) => {
          const reference = entry.reference;
          const asset = assetOf(reference.assetId);
          const hasAudio = Boolean(asset?.hasAudio);
          const audioOn = entry.audio !== undefined;
          const splitSibling = splitSiblingOf(reference);
          // Continuation carries the soundtrack across the cut, so it pins the lane on -- unless
          // the soundtrack has been split out, in which case it is still present, just as its own
          // reference. Removing that audio card releases the pin again.
          const forcedAudio = reference.kind === "video"
            && reference.intents.includes("continuation") && !splitSibling;
          const splitBlocked = limits.audios >= MAX_AUDIOS
            ? `This profile allows ${MAX_AUDIOS} separate audio reference, and one is already in the cast.`
            : limits.total >= MAX_REFERENCES
              ? `Splitting adds a reference, and the cast is already at the ${MAX_REFERENCES} limit.`
              : undefined;
          const visualIntents = availableH3ReferenceIntents(reference.kind, false);
          const audioIntents = availableH3ReferenceIntents(reference.kind, true)
            .filter((intent) => isAudioIntent(intent));

          const lane = (
            kindLabel: string, badge: string | undefined, meta: string,
            intents: H3ReferenceIntent[], on: boolean, toggle?: React.ReactNode,
          ) => <div className={`ref2v-lane${on ? "" : " off"}`} key={kindLabel}>
            <div className="ref2v-lane-head">
              <span className="ref2v-lane-kind">{kindLabel}</span>
              <span className={`ref2v-label ${kindLabel}${on && badge ? "" : " off"}`}
                title={on && badge
                  ? `The model sees this reference as ${badge}. Numbering follows tray order, per modality — reordering the cast renumbers it.`
                  : "Not sent to the model."}>{badge ?? `<${kindLabel}>`}</span>
              <span className="ref2v-lane-meta">{meta}</span>
              {toggle ?? <span />}
            </div>
            {on && <div className="ref2v-lane-body">
              <div className="ref2v-intents">
                {intents.map((intent) => <label key={intent} title={intentHint(intent)}
                  className={reference.intents.includes(intent) ? "selected" : ""}>
                  <input type="checkbox" checked={reference.intents.includes(intent)} disabled={busy}
                    onChange={() => toggleIntent(reference, intent)} />
                  <span>{intentLabel(intent)}</span>
                </label>)}
              </div>
              {reference.intents.filter((intent) => intents.includes(intent)).map((intent) => {
                const current = strengthOf(reference, intent);
                const transferred = reference.transferOnto !== undefined && !isAudioIntent(intent);
                return <div className="ref2v-strength" key={intent}>
                  <span>{intentLabel(intent)}</span>
                  <span className="ref2v-strength-dial">
                    {H3_STRENGTHS.map((step) => <button key={step.value} disabled={busy || transferred}
                      title={transferred ? "Overridden: this reference is being transferred onto another subject." : step.hint}
                      className={!transferred && current === step.value ? `on ${step.value}` : ""}
                      onClick={() => patchReference(reference.id, {
                        strengths: { ...reference.strengths, [intent]: step.value as H3Strength },
                      })}>{step.label}</button>)}
                    <b className={transferred ? "transfer" : ""}
                      title="The retention_analysis marker this produces in the prompt.">
                      {transferred ? "attribute_transfer" : h3Marker(intent, current)}
                    </b>
                  </span>
                </div>;
              })}
            </div>}
          </div>;

          return <article className="ref2v-card" key={reference.id}>
            <div className="ref2v-card-head">
              <span className="ref2v-index">{index + 1}</span>
              <div>
                <strong>{asset?.name ?? "Missing asset"}</strong>
                <small>{reference.kind}{hasAudio ? " + audio" : ""}{asset && reference.kind !== "image" ? ` · ${seconds(asset.duration).toFixed(1)}s` : ""}</small>
              </div>
              <div className="ref2v-card-actions">
                <button disabled={busy || index === 0} onClick={() => moveReference(reference.id, -1)} aria-label="Move up"><ArrowUp size={13} /></button>
                <button disabled={busy || index === labelled.length - 1} onClick={() => moveReference(reference.id, 1)} aria-label="Move down"><ArrowDown size={13} /></button>
                <button disabled={busy} aria-label="Remove reference"
                  onClick={() => { setReferences((c) => c.filter((i) => i.id !== reference.id)); setRawPrompt(""); }}><X size={13} /></button>
              </div>
            </div>

            {entry.picture !== undefined && lane("Picture", `<Picture ${entry.picture}>`,
              asset?.width && asset?.height ? `${asset.width}×${asset.height}` : "image",
              visualIntents, true)}

            {entry.video !== undefined && lane("Video", `<Video ${entry.video}>`,
              asset ? `${seconds(asset.duration).toFixed(1)}s` : "video", visualIntents, true)}

            {reference.kind === "video" && (hasAudio
              ? lane("Audio",
                  splitSibling ? undefined : audioOn ? `<Audio ${entry.audio}>` : undefined,
                  splitSibling
                    ? "split to its own reference"
                    : audioOn ? `${asset ? seconds(asset.duration).toFixed(1) : "?"}s of the ${MAX_TIMED_SECONDS}s budget` : "costs nothing",
                  audioIntents, audioOn && !splitSibling,
                  <span className="ref2v-lane-tools">
                    <label className="ref2v-switch" title={forcedAudio
                      ? "Continue clip carries the soundtrack across the cut, so the audio lane stays on."
                      : splitSibling
                        ? "This clip's soundtrack is now a separate reference. Remove that audio card to fold it back in."
                        : "Carry this clip's soundtrack as part of the video reference. It stays rotary-aligned with the picture, takes an audio slot and counts against the 15s audio budget."}>
                      <input type="checkbox" checked={audioOn && !splitSibling}
                        disabled={busy || forcedAudio || Boolean(splitSibling)}
                        onChange={(event) => setAudioLane(reference, event.target.checked)} />
                      <span className="track" /><span>{forcedAudio ? "required" : "audio"}</span>
                    </label>
                    {/* The title has to live on a WRAPPER, not on the button. A disabled button
                        fires no mouse events in Chromium, so its own tooltip never appears -- which
                        is how "split is unclickable" ended up with no visible explanation. */}
                    {!splitSibling && <span className="ref2v-split"
                      title={splitBlocked ?? "Extract the soundtrack to a wav and add it as its own <Audio> reference. Use this when a video with sound crashes the run, or when you want the soundtrack's roles set independently of the picture. Note it stops being time-aligned with the picture it came from."}>
                      <button className="ghost-button" disabled={busy || Boolean(splitBlocked)}
                        onClick={() => { void splitSoundtrack(reference); }}><Scissors size={12} /> split</button>
                    </span>}
                  </span>)
              : <div className="ref2v-lane off" key="Audio">
                  <div className="ref2v-lane-head">
                    <span className="ref2v-lane-kind">Audio</span>
                    <span className="ref2v-label Audio off">&lt;Audio&gt;</span>
                    <span className="ref2v-lane-meta">no embedded soundtrack</span><span />
                  </div>
                </div>)}

            {reference.kind === "audio" && lane("Audio", `<Audio ${entry.audio}>`,
              asset ? `${seconds(asset.duration).toFixed(1)}s` : "audio", audioIntents, true)}

            {splitBlocked && reference.kind === "video" && hasAudio && !splitSibling &&
              <p className="ref2v-blocked-note">Cannot split this soundtrack: {splitBlocked}</p>}

            {entry.picture !== undefined && <div className="ref2v-subjects">
              <div className="ref2v-subject-head">
                <span title="A subject is a name you invent for reusable visible content, so the description can refer back to it without describing it again.">Subjects defined here</span>
                <button className="ghost-button" disabled={busy}
                  onClick={() => patchReference(reference.id, {
                    subjects: [...(reference.subjects ?? []), { id: uid(), text: "" }],
                  })}>+ define subject</button>
              </div>
              {(reference.subjects ?? []).map((subject) => {
                const numbered = subjects.find((item) => item.text === subject.text.trim());
                return <div className="ref2v-subject" key={subject.id}>
                  <span className="ref2v-subject-tag"
                  title="Prompt-only label. Numbered across the whole cast, in tray order.">&lt;Subject {numbered?.n ?? "—"}&gt;</span>
                  <input type="text" value={subject.text} disabled={busy}
                    placeholder="what this subject is, and what to preserve…"
                    onChange={(event) => patchReference(reference.id, {
                      subjects: (reference.subjects ?? []).map((item) =>
                        item.id === subject.id ? { ...item, text: event.target.value } : item),
                    })} />
                  <button className="ghost-button" disabled={busy}
                    onClick={() => patchReference(reference.id, {
                      subjects: (reference.subjects ?? []).filter((item) => item.id !== subject.id),
                    })}><X size={12} /></button>
                </div>;
              })}
              {subjects.length > 0 && <label className="ref2v-transfer"
                title="Move this reference's quality onto a different subject — attribute_transfer. The source itself will not appear in the shot.">
                <span>apply onto</span>
                <select value={reference.transferOnto ?? ""} disabled={busy}
                  onChange={(event) => patchReference(reference.id, {
                    transferOnto: event.target.value ? Number(event.target.value) : undefined,
                  })}>
                  <option value="">— nothing, it appears directly —</option>
                  {subjects.filter((item) => item.entry.reference.id !== reference.id)
                    .map((item) => <option key={item.n} value={item.n}>&lt;Subject {item.n}&gt;</option>)}
                </select>
              </label>}
            </div>}
          </article>;
        })}

        <ul className="ref2v-help">
          <li>Tray order is the model's numbering — reordering the cast renumbers the labels.</li>
          <li>A video with its soundtrack on takes an Audio slot <strong>before</strong> its own Video slot.</li>
          <li>Every video and audio reference must be {MIN_TIMED_SECONDS}–{MAX_TIMED_SECONDS} seconds.</li>
        </ul>
      </section>

      <section className="studio-card">
        <div className="studio-card-heading"><strong>Output</strong></div>
        <label className="studio-field"><span>Duration</span>
          <select value={alignedFrames} disabled={busy}
            onChange={(event) => setFrames(Number(event.target.value))}>
            {Array.from({ length: Math.floor((H3_MAX_FRAMES - H3_MIN_FRAMES) / H3_FRAMES_PER_CHUNK) + 1 },
              (_, i) => H3_MIN_FRAMES + i * H3_FRAMES_PER_CHUNK)
              .map((value) => <option key={value} value={value}>{value} frames · {h3Seconds(value).toFixed(2)}s</option>)}
          </select>
        </label>
        <label className="studio-field"><span>Aspect</span>
          <select value={aspect} disabled={busy} onChange={(event) => applyAspectAndTier(event.target.value, tier)}>
            {H3_ASPECTS.map((item) => <option key={item.label} value={item.ratio.join(":")}>{item.label}</option>)}
          </select>
        </label>
        <label className="studio-field" title="Canvas tier: adjusts width and height while preserving aspect ratio. Compact reduces tokens by 35–45% for multi-reference safety on 16 GB GPUs.">
          <span>Resolution Tier</span>
          <select value={tier} disabled={busy} onChange={(event) => applyAspectAndTier(aspect, event.target.value as H3ResolutionTier)}>
            {H3_RESOLUTION_TIERS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
          </select>
        </label>
        <div className="ref2v-dim-grid">
          <div className="ref2v-dim-control">
            <span className="dim-label">Width</span>
            <div className="numeric-stepper-box">
              <button type="button" className="stepper-btn" disabled={busy || customWidth <= 128} title="Decrease by 32px" onClick={() => updateCustomWidth(customWidth - 32)}>
                <ChevronDown size={11} />
              </button>
              <input
                type="number"
                step={32}
                min={128}
                max={H3_MAX_LONG_EDGE}
                className="stepper-input"
                value={customWidth}
                disabled={busy}
                onChange={(e) => setCustomWidth(Number(e.target.value))}
                onBlur={() => updateCustomWidth(customWidth)}
              />
              <button type="button" className="stepper-btn" disabled={busy || customWidth >= H3_MAX_LONG_EDGE} title="Increase by 32px" onClick={() => updateCustomWidth(customWidth + 32)}>
                <ChevronUp size={11} />
              </button>
              <span className="unit-badge">px</span>
            </div>
          </div>
          <button
            type="button"
            className={`ref2v-ratio-lock ${lockRatio ? "active" : ""}`}
            title={lockRatio ? "Aspect ratio locked" : "Aspect ratio unlocked"}
            disabled={busy}
            onClick={() => setLockRatio(!lockRatio)}
          >
            {lockRatio ? <Link size={13} /> : <Unlink size={13} />}
          </button>
          <div className="ref2v-dim-control">
            <span className="dim-label">Height</span>
            <div className="numeric-stepper-box">
              <button type="button" className="stepper-btn" disabled={busy || customHeight <= 128} title="Decrease by 32px" onClick={() => updateCustomHeight(customHeight - 32)}>
                <ChevronDown size={11} />
              </button>
              <input
                type="number"
                step={32}
                min={128}
                max={H3_MAX_LONG_EDGE}
                className="stepper-input"
                value={customHeight}
                disabled={busy}
                onChange={(e) => setCustomHeight(Number(e.target.value))}
                onBlur={() => updateCustomHeight(customHeight)}
              />
              <button type="button" className="stepper-btn" disabled={busy || customHeight >= H3_MAX_LONG_EDGE} title="Increase by 32px" onClick={() => updateCustomHeight(customHeight + 32)}>
                <ChevronUp size={11} />
              </button>
              <span className="unit-badge">px</span>
            </div>
          </div>
        </div>
        <div className="ref2v-token-row">
          <span>Canvas: <b>{customWidth}×{customHeight}</b></span>
          <span className={`token-badge ${calculateH3Tokens(customWidth, customHeight) <= 1400 ? "safe" : calculateH3Tokens(customWidth, customHeight) <= 2040 ? "balanced" : "heavy"}`}>
            {calculateH3Tokens(customWidth, customHeight)} tokens/frame
          </span>
        </div>
        <div className="studio-grid">
          <label><span>Seed</span>
            <input type="number" value={seed} disabled={busy}
              onChange={(event) => setSeed(Math.trunc(Number(event.target.value)))} />
          </label>
          <button className="ghost-button" disabled={busy}
            onClick={() => setSeed(Math.floor(Math.random() * 2_147_483_647))}><RefreshCw size={13} /> Roll</button>
        </div>
        <label className="studio-field" title="Transformer attention engine: Native cuDNN SDPA is 100% crash-proof and lossless; Sage 2 Patched gives ~15% speedup with FP32 accumulator; Sage 1 preserves deep facial fidelity.">
          <span>Attention</span>
          <select value={attentionBackend} disabled={busy}
            onChange={(event) => setAttentionBackend(event.target.value as H3AttentionBackend)}>
            {H3_ATTENTION_BACKENDS.map((item) => (
              <option key={item.id} value={item.id} title={item.description}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <label className="studio-field"><span>Sampling</span>
          <select value={samplingMode} disabled={busy}
            onChange={(event) => chooseSampling(event.target.value as "turbo-4" | "base")}>
            <option value="base">Base H3 · no adapter · quality reference</option>
            <option value="turbo-4">LightX2V Turbo v0.1 preview · 4 NFE · fast draft</option>
          </select>
        </label>
        {/* This used to offer 4, 6 and 8 steps while the worker was hardwired to `turbo-4`, which
            accepts exactly 4. Picking 6 or 8 ran LightX2V's four-step distillation outside the
            schedule it was distilled for -- labelled "Enhanced Texture" and "Maximum Clarity"
            while doing the opposite. Steps now belong to Base, which genuinely has a choice. */}
        {samplingMode === "base" && <label className="studio-field"><span>Steps</span>
          <select value={steps} disabled={busy}
            onChange={(event) => setSteps(Number(event.target.value))}>
            <option value={20}>20 · ComfyUI's R2V default</option>
            <option value={30}>30 · Comfy suggests 25+ for motion</option>
            <option value={50}>50 · MiniMax's published recipe</option>
          </select>
        </label>}
        {/* 12.0 is not a "stock" fallback, it is the number: both released scheduler configs
            declare shift 12 for video and 3 for audio, ComfyUI's SigmaShift node defaults to the
            same pair, and LightX2V's own inference script hardcodes DEFAULT_VIDEO_SHIFT = 12.0.
            The adapter is distilled against that grid. Anything else is an experiment, and this
            control shipped defaulting to 8.0 while calling it "Recommended". */}
        <label className="studio-field" title="The sigma-grid shift for the video stream. The audio stream stays at its own published 3.0.">
          <span>Motion Shift</span>
          <select value={videoShift} disabled={busy || samplingMode === "base"}
            onChange={(event) => setVideoShift(Number(event.target.value))}>
            <option value={12.0}>12.0 · published (default)</option>
            <option value={8.0}>8.0 · experimental</option>
            <option value={6.0}>6.0 · experimental (FL2V 768p's value, not Ref2V's)</option>
          </select>
        </label>
        {samplingMode === "base" && <p className="ref2v-hint">
          Base keeps the released scheduler shifts (12 / 3). The control above applies to Turbo only.
        </p>}
        {/* Reference tokens ride through every sampling step, so the small tiers are a real speed
            dial -- but nothing upstream conditions references at 256, and this defaulted there
            while labelling it "Recommended". ComfyUI's shipped R2V workflow uses `match`;
            Diffusers' own default is 2048. */}
        <label className="studio-field" title="How large each reference IMAGE is encoded. Larger carries more identity and costs time on every step.">
          <span>Reference Image Detail</span>
          <select value={refImageDetail} disabled={busy}
            onChange={(event) => setRefImageDetail(event.target.value as H3RefImageDetail)}>
            {H3_REF_IMAGE_DETAILS.map((item) => (
              <option key={item.id} value={item.id}>{item.label}</option>
            ))}
          </select>
        </label>
        {limits.videos > 0 && <label className="studio-field" title="Reference VIDEO is a separate budget: Qwen3-VL attends over unmerged patches, so this is the setting that has exhausted VRAM in the past.">
          <span>Reference Video Detail</span>
          <select value={refVideoDetail} disabled={busy}
            onChange={(event) => setRefVideoDetail(event.target.value as H3RefVideoDetail)}>
            {H3_REF_VIDEO_DETAILS.map((item) => (
              <option key={item.id} value={item.id}>{item.label}</option>
            ))}
          </select>
        </label>}
        <label className="studio-field"><span>VAE Tile Size</span>
          <select value={vaeTile} disabled={busy}
            onChange={(event) => setVaeTile(Number(event.target.value))}>
            {/* No sub-256 option. 256 is MiniMax's own released geometry (`vae_tile_size: 256`,
                `vae_tile_overlap_min: 64`) and the worker refuses anything smaller, so a 192 px
                choice failed at validation with a Python traceback. It would not have helped
                anyway: peak decode VRAM is dominated by the temporal chunk buffers, not the
                spatial tiles, so shrinking tiles costs seam quality and saves nothing. */}
            <option value={256}>256 px · Released geometry (recommended)</option>
            <option value={384}>384 px · High Quality (16 GB+)</option>
            <option value={512}>512 px · Seamless (24 GB+)</option>
          </select>
        </label>
        <label className="studio-field"><span>Memory</span>
          <select value={offload} disabled={busy} onChange={(event) => setOffload(event.target.value)}>
            <option value="balanced">balanced · 1 block, streamed</option>
            <option value="performance">performance · 4 blocks</option>
            <option value="minimal">minimal</option>
          </select>
        </label>
        <div className="ref2v-locked">
          <span title="MiniMax-H3 canvases are multiples of 32, up to a 768 short edge and 768x1344 pixels.">
            <b>canvas</b> {dimensions.value}{Math.min(dimensions.width, dimensions.height) >= H3_TRAINED_SHORT_EDGE ? " · native" : ""}</span>
          <span title={samplingMode === "turbo-4"
            ? "LightX2V's Ref2V v0.1 adapter, distilled to exactly four evaluations."
            : "Base H3 with no acceleration adapter. MiniMax's published recipe is 50 evaluations."}>
            <b>sampling</b> {steps} NFE ({samplingMode === "turbo-4" ? "Turbo v0.1" : "Base"})</span>
          <span title="H3 is guidance-distilled, so there is no CFG scale and no negative prompt to write.">
            <b>guidance</b> distilled — no negative prompt</span>
        </div>
      </section>
    </aside>

    <section className="ref2v-stage">
      <div className="ref2v-preview">
        {selectedTake ? <>
          <video ref={videoRef} src={convertFileSrc(selectedTake.videoPath)} controls
            onPlay={() => syncAudioTo("play")} onPause={() => syncAudioTo("pause")}
            onSeeked={() => syncAudioTo("seek")} />
          <audio ref={audioRef} src={convertFileSrc(selectedTake.audioPath)} preload="auto" />
          <div className="ref2v-take-meta">
            <span>take {selectedTake.index}</span>
            <span>{selectedTake.width}×{selectedTake.height}</span>
            <span>{selectedTake.frames}f · {h3Seconds(selectedTake.frames).toFixed(2)}s</span>
            <span>seed {selectedTake.seed}</span>
            {selectedTake.onTimeline && <span className="ok">on timeline</span>}
          </div>
        </> : <div className="ref2v-empty">
          <Film size={44} />
          <strong>{generating ? "Generating…" : "No take yet"}</strong>
          <span>{generating
            ? `${elapsed.toFixed(0)}s elapsed · encode then denoise`
            : "Build the cast and description, then generate. Every run is kept as a take."}</span>
        </div>}
      </div>

      <div className="ref2v-actions">
        <button className="render-button" disabled={busy || generating || Boolean(preflight)}
          title={preflight?.message} onClick={() => { void generate(); }}>
          <WandSparkles size={16} /> {generating ? "Generating…" : "Generate Ref2V"}
        </button>
        {preflight && <span className="ref2v-blocker">{preflight.message}</span>}
        {!preflight && warnings.map((issue) => (
          <span key={issue.message} className="ref2v-advisory">{issue.message}</span>
        ))}
        {selectedTake && <>
          <button className="ghost-button" disabled={busy || selectedTake.sessionFingerprint === sessionFingerprint}
            title="Restore every cast, prompt, output and sampling control captured when this take was rendered."
            onClick={() => restoreTakeSession(selectedTake)}>
            <RefreshCw size={14} /> {selectedTake.sessionFingerprint === sessionFingerprint ? "Setup restored" : "Restore take setup"}
          </button>
          <button className="ghost-button" disabled={busy || selectedTake.onTimeline}
            onClick={() => { void sendToTimeline(selectedTake); }}>
            <Film size={14} /> {selectedTake.onTimeline ? "Already on timeline" : "Send to timeline"}
          </button>
          <button className="ghost-button" disabled={busy} onClick={() => deleteTake(selectedTake)}>
            <Trash2 size={14} /> Remove take
          </button>
        </>}
      </div>

      {takes.length > 0 && <div className="ref2v-takes">
        {takes.map((take) => <button key={take.id}
          className={take.id === selectedTakeId ? "selected" : ""}
          onClick={() => setSelectedTakeId(take.id)}>
          <strong>take {take.index}</strong>
          <small>seed {take.seed} · {take.at}{take.sessionFingerprint === sessionFingerprint ? " · current setup" : " · setup changed"}{take.onTimeline ? " · placed" : ""}</small>
        </button>)}
      </div>}

      <div className="ref2v-ledger-note">
        <strong>Director ledger</strong>
        <span>Cast, shots, prompt, output controls and take recipes are saved inside this project. A take can restore the exact setup that produced it.</span>
      </div>

      {preflightIssues.length > 0 && <div className="ref2v-preflight-report">
        <div className="ref2v-preflight-head">
          <strong>Production preflight</strong>
          <span className={blockers.length ? "blocked" : "ready"}>{blockers.length} block{blockers.length === 1 ? "" : "s"}</span>
          <span>{warnings.length} warning{warnings.length === 1 ? "" : "s"}</span>
        </div>
        <ul>{preflightIssues.map((issue) => <li key={`${issue.severity}:${issue.message}`} className={issue.severity}>
          <b>{issue.severity === "block" ? "BLOCK" : "CHECK"}</b>{issue.message}
        </li>)}</ul>
      </div>}
      {failure && failureBox("Last run failed", failure)}

      <div className="ref2v-prompt">
        <div className="ref2v-prompt-tabs">
          <button className={promptTab === "built" ? "on" : ""} onClick={() => setPromptTab("built")}>Guided</button>
          <button className={promptTab === "raw" ? "on" : ""} onClick={() => { setPromptTab("raw"); if (!rawPrompt) setRawPrompt(builtPrompt); }}>Raw prompt</button>
          <span className="ref2v-wordcount"
            title="MiniMax's guide targets 350-500 words of description for reference tasks. Shorter still works.">{wordCount} / 350–500 words</span>
        </div>

        {promptTab === "built" ? <div className="ref2v-fields">
          <label className="ai-prompt" title="One or two sentences of overall look. Emitted before Shot 1, not inside it."><span>Style</span>
            <textarea value={style} disabled={busy} rows={2}
              placeholder="One or two sentences of overall look, before Shot 1…"
              onChange={(event) => { setStyle(event.target.value); setRawPrompt(""); }} /></label>
          <label className="ai-prompt" title="Optional. The bracketed task type is derived from the cast and added for you."><span>Summary</span>
            <textarea value={summary} disabled={busy} rows={2}
              placeholder="Optional. The bracketed task type is added automatically."
              onChange={(event) => { setSummary(event.target.value); setRawPrompt(""); }} /></label>

          {shots.map((shot, index) => <div key={shot.id}
            className={`ref2v-shot${timingIssues.some((issue) => issue.shotId === shot.id && issue.severity === "block")
              || (index === 0 && preflight?.field === "shot") ? " blocking" : ""}`}>
            <div className="ref2v-shot-head">
              <span className="ref2v-shot-tag"
                title={index === 0 ? "Shot 1 carries no timestamp, by the writing guide." : "Later shots are stamped MM:SS.mmm."}>[Shot {index + 1}]</span>
              {index === 0
                ? <small>{shot.description.trim() ? "no timestamp" : "required · no timestamp"}</small>
                : <input type="text" className="ref2v-timecode" value={shot.timecode} disabled={busy}
                    placeholder="MM:SS.mmm"
                    onChange={(event) => patchShot(shot.id, { timecode: event.target.value })} />}
              <span />
              <button className="ghost-button" disabled={busy || shots.length === 1}
                onClick={() => { setShots((c) => c.filter((s) => s.id !== shot.id)); setRawPrompt(""); }}><X size={12} /></button>
            </div>
            <textarea value={shot.description} disabled={busy} rows={3}
              placeholder="What happens, in playback order. Reference labels where their roles apply…"
              onChange={(event) => patchShot(shot.id, { description: event.target.value })} />
            {shot.dialogue.map((line) => <div className="ref2v-dialogue" key={line.id}>
              <select className="dialogue-language" value={line.language} disabled={busy}
                onChange={(event) => patchShot(shot.id, {
                  dialogue: shot.dialogue.map((item) => item.id === line.id ? { ...item, language: event.target.value } : item),
                })}>
                {["English", "Spanish", "Mandarin", "Japanese", "French", "German"].map((value) =>
                  <option key={value} value={value}>{value}</option>)}
              </select>
              {/* Who is speaking, not which line this is. Lines that share a speaker share one
                  (Sn) tag; without this every line became its own speaker and the model was asked
                  for a new voice per sentence. */}
              <select className="dialogue-speaker" value={line.speaker === undefined ? "" : String(line.speaker)} disabled={busy}
                title="Which subject speaks this line. Lines with the same speaker keep one voice."
                onChange={(event) => patchShot(shot.id, {
                  dialogue: shot.dialogue.map((item) => item.id === line.id
                    ? { ...item, speaker: event.target.value === "" ? undefined : Number(event.target.value) }
                    : item),
                })}>
                <option value="">Main speaker</option>
                {subjects.map((subject) => (
                  <option key={subject.n} value={subject.n}>{`Subject ${subject.n}`}</option>
                ))}
              </select>
              <select className="dialogue-overlap" value={line.overlapPolicy ?? "separate"} disabled={busy}
                title="How this line relates to the preceding voice in the same shot."
                onChange={(event) => patchShot(shot.id, {
                  dialogue: shot.dialogue.map((item) => item.id === line.id
                    ? { ...item, overlapPolicy: event.target.value as "separate" | "overlap" | "interrupt" }
                    : item),
                })}>
                <option value="separate">Separate</option>
                <option value="overlap">Overlap</option>
                <option value="interrupt">Interrupt</option>
              </select>
              <button className="ghost-button dialogue-remove" disabled={busy}
                onClick={() => patchShot(shot.id, { dialogue: shot.dialogue.filter((item) => item.id !== line.id) })}><X size={12} /></button>
              <input className="dialogue-delivery" type="text" value={line.delivery ?? ""} disabled={busy}
                placeholder="delivery: hushed, urgent, breathless…"
                onChange={(event) => patchShot(shot.id, {
                  dialogue: shot.dialogue.map((item) => item.id === line.id ? { ...item, delivery: event.target.value } : item),
                })} />
              <input className="dialogue-text" type="text" value={line.text} disabled={busy} placeholder="exact spoken line…"
                onChange={(event) => patchShot(shot.id, {
                  dialogue: shot.dialogue.map((item) => item.id === line.id ? { ...item, text: event.target.value } : item),
                })} />
            </div>)}
            <button className="ghost-button" disabled={busy}
              onClick={() => patchShot(shot.id, { dialogue: [...shot.dialogue, { id: uid(), language: "English", text: "" }] })}>+ dialogue line</button>
          </div>)}
          <button className="ghost-button" disabled={busy}
            onClick={() => { setShots((c) => [...c, { id: uid(), timecode: "", description: "", dialogue: [] }]); setRawPrompt(""); }}>+ Add shot</button>

          <label className="ai-prompt" title="Ambience and physical sound. Dialogue belongs in the shot, not here."><span>Soundscape</span>
            <textarea value={soundscape} disabled={busy} rows={2}
              placeholder="Ambience and physical sound. No dialogue here."
              onChange={(event) => { setSoundscape(event.target.value); setRawPrompt(""); }} /></label>
          <label className="ai-prompt" title="Audience-only score. Blank becomes N/A, which is a valid answer."><span>Music</span>
            <textarea value={music} disabled={busy} rows={2}
              placeholder="Audience-only score, or leave blank for N/A."
              onChange={(event) => { setMusic(event.target.value); setRawPrompt(""); }} /></label>

          <div className="ref2v-assembled">
            <span>Assembled prompt</span>
            <pre>{builtPrompt}</pre>
          </div>
        </div> : <textarea className="ref2v-raw" value={effectivePrompt} disabled={busy}
          onChange={(event) => setRawPrompt(event.target.value)} />}
      </div>
    </section>
  </main>;
}
