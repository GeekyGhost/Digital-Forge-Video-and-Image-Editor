import { convertFileSrc, invoke } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { ArrowDown, ArrowUp, ChevronDown, ChevronLeft, ChevronRight, ChevronUp, Download, Eye, EyeOff, Film, FolderOpen, Hand, ImagePlus, Layers3, Lock, LockOpen, Maximize2, Move, Pencil, Plus, Sliders, RefreshCw, RotateCcw, Sparkles, Trash2, WandSparkles, X, ZoomIn, ZoomOut } from "lucide-react";
import { looksGenerated, nameFromPrompt } from "../../domain/naming";
import { fromSeconds } from "../../domain/time";
import type { MediaAsset } from "../../domain/types";
import { exportImage, filename, parseFrameRate, probeMedia, rewriteQwenEditPrompt, runtimeStatus, upscaleImage,
  generateZImages, inspectZImageLora, listZImageLoras, writeEditedImage, type AiRuntimeStatus, type ImageModelResult,
  type ZImageLoraInfo } from "../../services/modelService";
import { loadLayerSources } from "../../services/readableImage";
import type { LayerSource } from "../../domain/photoCompose";
import { composeDocument } from "../../domain/photoCompose";
import { emptyResources } from "../../domain/photoRender";
import { createLayer, layersOf, type PhotoDocument, type PhotoLayer } from "../../domain/photoDocument";
import { EDIT_AREA_COMFORT, EDIT_SIZE_MAX, EDIT_SIZE_MIN, EDIT_SIZE_PRESETS, defaultEditSize, resolveEditSize, type EditSize } from "../../domain/qwenEditSize";
import { AI_UPSCALE_MODELS, NVVFX_QUALITY_DEFAULT, aiUpscaleModel, resolveUpscalePlan } from "../../domain/upscale";
import { useEditor } from "../../store";
import { RenameField } from "../media/RenameField";
import { LoraTraining } from "./LoraTraining";

type StudioMode = "generate" | "edit" | "train";
type ZImageMode = "turbo" | "base";
type CanvasTool = "layer" | "canvas";
type FlattenMode = "tight" | "frame";
type Layer = { id: string; assetId: string; name: string; path: string; x: number; y: number; scale: number; rotation: number; opacity: number; visible: boolean; locked: boolean; cropLeft: number; cropTop: number; cropRight: number; cropBottom: number };
type FramePickerSource = { path: string; name: string; duration: number; width: number; height: number; fps: number };
type PickedVideoFrame = { frameIndex: number; timestampSeconds: number; thumbnail: string };
type ExtractedVideoFrame = { outputPath: string; timestampSeconds: number; frameIndex: number; width: number; height: number };

const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
/** `png` from `zit-a1b2c3.png`. Keeps a suggested name on the same format as the file it renames. */
const extensionOf = (name: string) => name.split(".").length > 1 ? name.split(".").pop() ?? "png" : "png";
const numberSeed = (value: string) => Number.isFinite(Number(value)) ? Math.trunc(Number(value)) : -1;
const clock = (value: number) => {
  const totalMilliseconds = Math.max(0, Math.round(value * 1000));
  const minutes = Math.floor(totalMilliseconds / 60_000);
  const seconds = Math.floor(totalMilliseconds % 60_000 / 1000);
  const milliseconds = totalMilliseconds % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(milliseconds).padStart(3, "0")}`;
};

function imageAsset(result: ImageModelResult, operation: NonNullable<MediaAsset["origin"]>["operation"], model: string,
  prompt: string | undefined, sourceAssetIds: string[], parameters: Record<string, unknown>): MediaAsset {
  return {
    id: crypto.randomUUID(), name: filename(result.outputPath), path: result.outputPath, kind: "image",
    duration: fromSeconds(5), width: result.width, height: result.height,
    origin: { operation, model, prompt, seed: result.seed, sourceAssetIds, parameters, createdAt: new Date().toISOString() },
  };
}

const layerFrom = (asset: MediaAsset): Layer => ({
  id: crypto.randomUUID(), assetId: asset.id, name: asset.name, path: asset.path,
  x: 50, y: 50, scale: 100, rotation: 0, opacity: 100, visible: true, locked: false,
  cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
});

function NumericControl({
  label, value, min, max, step = 1, unit = "%", disabled, onChange,
}: {
  label: string; value: number; min: number; max: number; step?: number; unit?: string; disabled?: boolean; onChange: (value: number) => void;
}) {
  const [localText, setLocalText] = useState<string>(String(Math.round(value)));

  useEffect(() => {
    setLocalText(String(Math.round(value)));
  }, [value]);

  const commitValue = (raw: number) => {
    if (isNaN(raw)) {
      setLocalText(String(Math.round(value)));
      return;
    }
    const clamped = Math.max(min, Math.min(max, Math.round(raw)));
    setLocalText(String(clamped));
    onChange(clamped);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    const delta = e.shiftKey ? 5 : 1;
    if (e.key === "ArrowUp") {
      e.preventDefault();
      commitValue(value + delta);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      commitValue(value - delta);
    } else if (e.key === "Enter") {
      commitValue(Number(localText));
      (e.target as HTMLInputElement).blur();
    }
  };

  const nudge = (direction: number, e: React.MouseEvent) => {
    const delta = (e.shiftKey ? 5 : 1) * direction;
    commitValue(value + delta);
  };

  return (
    <div className="layer-numeric-control">
      <span className="control-label">{label}</span>
      <input
        disabled={disabled}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
      <div className="numeric-stepper-box">
        <button
          type="button"
          tabIndex={-1}
          disabled={disabled || value <= min}
          className="stepper-btn"
          title="Decrease by 1 (Shift: -5)"
          onClick={(e) => nudge(-1, e)}
        >
          <ChevronDown size={11} />
        </button>
        <input
          disabled={disabled}
          type="number"
          className="stepper-input"
          min={min}
          max={max}
          value={localText}
          onChange={(e) => setLocalText(e.target.value)}
          onBlur={() => commitValue(Number(localText))}
          onKeyDown={handleKeyDown}
        />
        <button
          type="button"
          tabIndex={-1}
          disabled={disabled || value >= max}
          className="stepper-btn"
          title="Increase by 1 (Shift: +5)"
          onClick={(e) => nudge(1, e)}
        >
          <ChevronUp size={11} />
        </button>
        <span className="unit-badge">{unit}</span>
      </div>
    </div>
  );
}

export function ImageStudio() {
  const { project, addAssets, removeAsset, renameAsset, addAssetToTimeline, openPhotoEdit,
    setPhotoDocument, pendingCanvasAssetId, clearPendingCanvasAsset, busy, setBusy, setStatus } = useEditor();
  const imageAssets = project.assets.filter((asset) => asset.kind === "image");
  const [runtime, setRuntime] = useState<AiRuntimeStatus>();
  const [mode, setMode] = useState<StudioMode>("generate");
  const [zImageMode, setZImageMode] = useState<ZImageMode>("turbo");
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [steps, setSteps] = useState(9);
  const [cfg, setCfg] = useState(1);
  /** Qwen edit "Output size": "default" follows Image 1, a preset value, or "manual". */
  const [editSizeChoice, setEditSizeChoice] = useState("default");
  const [editManualSize, setEditManualSize] = useState<EditSize>({ width: 1024, height: 1024 });
  const [seed, setSeed] = useState("-1");
  const [candidateCount, setCandidateCount] = useState<1 | 4>(1);
  const [zImageLora, setZImageLora] = useState("");
  const [zImageLoraInfo, setZImageLoraInfo] = useState<ZImageLoraInfo>();
  const [installedZImageLoras, setInstalledZImageLoras] = useState<ZImageLoraInfo[]>([]);
  const [verifiedZImageLora, setVerifiedZImageLora] = useState("");
  const [zImageLoraStrength, setZImageLoraStrength] = useState(1);
  const [layers, setLayers] = useState<Layer[]>([]);
  const [selectedLayerId, setSelectedLayerId] = useState<string>();
  // Only one name is editable at a time, in each list. Undefined means nothing is being renamed.
  const [renamingAssetId, setRenamingAssetId] = useState<string>();
  const [renamingLayerId, setRenamingLayerId] = useState<string>();
  const [exportFormat, setExportFormat] = useState<"png" | "jpeg" | "webp" | "tiff">("png");
  const [exportQuality, setExportQuality] = useState(92);
  const [exportMatte, setExportMatte] = useState("#ffffff");
  const [upscaleModelId, setUpscaleModelId] = useState("");
  const [upscaleTile, setUpscaleTile] = useState(512);
  const [upscaleQuality, setUpscaleQuality] = useState(NVVFX_QUALITY_DEFAULT);
  const [reference1, setReference1] = useState("");
  const [reference2, setReference2] = useState("");
  const [failure, setFailure] = useState<string>();
  const [runningLabel, setRunningLabel] = useState("");
  /** The last rewrite and the instruction it replaced, so the rewrite can be undone until the text is edited. */
  const [rewriteUndo, setRewriteUndo] = useState<{ original: string; rewritten: string }>();
  const [bgBlur, setBgBlur] = useState(0);
  const [bgDilate, setBgDilate] = useState(0);
  const [bgErode, setBgErode] = useState(0);
  const [canvasTool, setCanvasTool] = useState<CanvasTool>("layer");
  const [canvasPan, setCanvasPan] = useState({ x: 0, y: 0 });
  const [canvasZoom, setCanvasZoom] = useState(100);
  const [panDrag, setPanDrag] = useState<{ startX: number; startY: number; x: number; y: number }>();
  const [flattenMode, setFlattenMode] = useState<FlattenMode>("tight");
  const [flattenPadding, setFlattenPadding] = useState(0);
  const [flattenBackground, setFlattenBackground] = useState("transparent");
  const [frameSource, setFrameSource] = useState<FramePickerSource>();
  const [frameIndex, setFrameIndex] = useState(0);
  const [pickedFrames, setPickedFrames] = useState<PickedVideoFrame[]>([]);
  const [frameFailure, setFrameFailure] = useState<string>();
  const [drag, setDrag] = useState<{ layerId: string; startX: number; startY: number; x: number; y: number }>();
  const stageRef = useRef<HTMLDivElement>(null);
  const frameVideoRef = useRef<HTMLVideoElement>(null);
  const loadedProjectId = useRef(project.id);
  const selected = layers.find((layer) => layer.id === selectedLayerId);
  const selectedAsset = project.assets.find((asset) => asset.id === selected?.assetId);
  // What the edit will make: Image 1's shape at ~1 MP by default (the worker's canvas_output_size),
  // or the override, snapped the way the worker snaps it.
  const editDefaultSize = defaultEditSize(selectedAsset?.width && selectedAsset?.height ? { width: selectedAsset.width, height: selectedAsset.height } : undefined);
  const editOverride = resolveEditSize(editSizeChoice, editManualSize);
  const editOutputSize = editOverride ?? editDefaultSize;

  // The two backends install independently, so each is offered only when its own runtime is there.
  const upscaleModels = AI_UPSCALE_MODELS.filter((model) => model.backend === "nvvfx"
    ? Boolean(runtime?.nvvfxReady)
    : model.file !== undefined && (runtime?.realEsrganModels ?? []).includes(model.file));
  const upscalerReady = upscaleModels.length > 0;
  const upscaleModel = aiUpscaleModel(upscaleModelId);
  // Resolved from the source's real dimensions, so the readout is the size that will be written.
  const upscalePlan = upscaleModel && selectedAsset?.width && selectedAsset?.height
    ? resolveUpscalePlan({
        source: { width: selectedAsset.width, height: selectedAsset.height },
        factor: upscaleModel.factor,
        aiModelId: upscaleModel.id,
        resample: "lanczos",
        medium: "image",
      })
    : undefined;
  const referenceAsset1 = imageAssets.find((asset) => asset.id === reference1);
  const referenceAsset2 = imageAssets.find((asset) => asset.id === reference2);
  const visibleLayers = useMemo(() => layers.filter((layer) => layer.visible), [layers]);
  const maxVideoFrame = frameSource ? Math.max(0, Math.floor(frameSource.duration * frameSource.fps - 0.001)) : 0;
  const currentFrameSeconds = frameSource ? frameIndex / frameSource.fps : 0;

  useEffect(() => {
    void runtimeStatus().then(setRuntime).catch((error) => setFailure(errorText(error)));
  }, []);
  useEffect(() => {
    if (mode !== "generate") return;
    void listZImageLoras().then(setInstalledZImageLoras).catch(() => setInstalledZImageLoras([]));
  }, [mode]);
  useEffect(() => {
    if (loadedProjectId.current === project.id) return;
    loadedProjectId.current = project.id;
    setMode("generate"); setPrompt(""); setNegativePrompt(""); setWidth(1024); setHeight(1024);
    setSteps(9); setCfg(1); setSeed("-1"); setZImageMode("turbo"); setCandidateCount(1);
    setZImageLora(""); setZImageLoraInfo(undefined); setZImageLoraStrength(1); setVerifiedZImageLora("");
    setLayers([]); setSelectedLayerId(undefined); setReference1(""); setReference2("");
    setFailure(undefined); setRunningLabel(""); setBgBlur(0); setBgDilate(0); setBgErode(0);
    setCanvasTool("layer"); setCanvasPan({ x: 0, y: 0 }); setCanvasZoom(100); setPanDrag(undefined); setDrag(undefined);
    setFlattenMode("tight"); setFlattenPadding(0); setFlattenBackground("transparent");
    setFrameSource(undefined); setFrameIndex(0); setPickedFrames([]); setFrameFailure(undefined);
  }, [project.id]);

  useEffect(() => {
    const handleGlobalKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.tagName === "SELECT")) {
        return;
      }
      if (!selectedLayerId) return;
      const layer = layers.find((l) => l.id === selectedLayerId);
      if (!layer || layer.locked) return;

      const step = event.shiftKey ? 5 : 1;
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        patchLayer(layer.id, { x: Math.max(0, layer.x - step) });
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        patchLayer(layer.id, { x: Math.min(100, layer.x + step) });
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        patchLayer(layer.id, { y: Math.max(0, layer.y - step) });
      } else if (event.key === "ArrowDown") {
        event.preventDefault();
        patchLayer(layer.id, { y: Math.min(100, layer.y + step) });
      }
    };
    window.addEventListener("keydown", handleGlobalKeyDown);
    return () => window.removeEventListener("keydown", handleGlobalKeyDown);
  }, [selectedLayerId, layers]);

  const addLayer = (asset: MediaAsset) => { const layer = layerFrom(asset); setLayers((current) => [...current, layer]); setSelectedLayerId(layer.id); };

  // Photo Edit saves its result into Project Media and then asks for it on the canvas. The request
  // is cleared as soon as it is honoured, so switching back later does not add the layer twice.
  useEffect(() => {
    if (!pendingCanvasAssetId) return;
    const incoming = project.assets.find((item) => item.id === pendingCanvasAssetId);
    if (incoming) { addLayer(incoming); setStatus(`${incoming.name} added to the canvas`); }
    clearPendingCanvasAsset();
  }, [pendingCanvasAssetId, project.assets]);
  const patchLayer = (id: string, patch: Partial<Layer>) => setLayers((current) => current.map((layer) => layer.id === id ? { ...layer, ...patch } : layer));
  const moveLayer = (id: string, direction: -1 | 1) => setLayers((current) => {
    const index = current.findIndex((layer) => layer.id === id); const target = index + direction;
    if (index < 0 || target < 0 || target >= current.length) return current;
    const next = current.slice(); [next[index], next[target]] = [next[target], next[index]]; return next;
  });
  const startDrag = (event: ReactPointerEvent, layer: Layer) => {
    if (canvasTool !== "layer" || layer.locked || layer.id !== selectedLayerId) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrag({ layerId: layer.id, startX: event.clientX, startY: event.clientY, x: layer.x, y: layer.y });
  };
  const movePointer = (event: ReactPointerEvent) => {
    if (panDrag) {
      setCanvasPan({ x: panDrag.x + event.clientX - panDrag.startX, y: panDrag.y + event.clientY - panDrag.startY });
      return;
    }
    if (!drag || !stageRef.current) return; const rect = stageRef.current.getBoundingClientRect();
    patchLayer(drag.layerId, { x: Math.max(0, Math.min(100, drag.x + (event.clientX - drag.startX) / rect.width * 100)), y: Math.max(0, Math.min(100, drag.y + (event.clientY - drag.startY) / rect.height * 100)) });
  };
  const startCanvasPan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (canvasTool !== "canvas") return;
    event.preventDefault(); event.currentTarget.setPointerCapture(event.pointerId);
    setPanDrag({ startX: event.clientX, startY: event.clientY, x: canvasPan.x, y: canvasPan.y });
  };
  const stopPointer = () => { setDrag(undefined); setPanDrag(undefined); };
  const handleCanvasWheel = (event: React.WheelEvent<HTMLDivElement>) => {
    if (event.ctrlKey || canvasTool === "canvas") {
      event.preventDefault();
      const delta = event.deltaY < 0 ? 15 : -15;
      setCanvasZoom((current) => Math.max(25, Math.min(500, current + delta)));
    }
  };

  const runGenerate = async () => {
    if (!prompt.trim()) { setStatus("Describe the frame you want Z-Image to generate"); return; }
    const modelName = zImageMode === "base" ? "Z-Image Base" : "Z-Image-Turbo";
    const requiredTrigger = zImageLoraInfo?.trigger?.trim() || zImageLoraInfo?.triggerPhrase?.trim();
    const generationPrompt = zImageLora && requiredTrigger
      && !prompt.toLocaleLowerCase().includes(requiredTrigger.toLocaleLowerCase())
      ? `${zImageLoraInfo?.triggerPhrase ?? requiredTrigger}, ${prompt.trim()}` : prompt.trim();
    if (generationPrompt !== prompt.trim()) setPrompt(generationPrompt);
    setBusy(true); setFailure(undefined); setRunningLabel(`Loading ${modelName} and generating ${candidateCount === 4 ? "four candidates" : "the frame"}`); setStatus(`${modelName} is generating…`);
    try {
      const result = await generateZImages({
        projectId: project.id, prompt: generationPrompt, negativePrompt: zImageMode === "base" ? negativePrompt : "",
        model: zImageMode, width, height, steps, guidanceScale: zImageMode === "base" ? cfg : 0,
        seed: numberSeed(seed), candidates: candidateCount, loraPath: zImageLora || undefined,
        loraStrength: zImageLoraStrength,
      });
      const assets = result.images.map((image, index) => imageAsset(
        image, "zit-generate", modelName, generationPrompt, [], {
          width, height, steps, guidanceScale: zImageMode === "base" ? cfg : 0,
          negativePrompt: zImageMode === "base" ? negativePrompt : undefined,
          candidate: index + 1, candidates: candidateCount, loraPath: zImageLora || undefined,
          loraStrength: zImageLora ? zImageLoraStrength : undefined,
        },
      ));
      addAssets(assets); if (assets[0]) addLayer(assets[0]);
      const loraVerified = Boolean(zImageLora) && result.images.every((image) => image.loraActive && image.loraPath);
      setVerifiedZImageLora(loraVerified ? result.images[0]?.loraPath ?? "" : "");
      const loraStatus = zImageLora ? (loraVerified ? " · LoRA verified active" : " · LoRA verification failed") : "";
      setStatus(`Generated ${assets.length} ${modelName} candidate${assets.length === 1 ? "" : "s"} in Project Media${loraStatus}`);
    } catch (error) { setFailure(errorText(error)); setStatus("Z-Image generation failed"); } finally { setBusy(false); setRunningLabel(""); }
  };

  const chooseZImageLora = async () => {
    const root = runtime?.root.replace(/[\\/]+$/, "");
    const selected = await open({
      multiple: false,
      directory: true,
      defaultPath: root ? `${root}/models/loras/z-image` : undefined,
      title: "Choose a Z-Image LoRA folder",
    });
    if (typeof selected !== "string") return;
    try {
      const info = await inspectZImageLora(selected);
      setZImageLora(info.folder); setZImageLoraInfo(info); setVerifiedZImageLora(""); setFailure(undefined);
      setStatus(`Selected ${info.name}${info.triggerPhrase ? ` · trigger: ${info.triggerPhrase}` : ""}`);
    } catch (error) {
      setZImageLora(""); setZImageLoraInfo(undefined); setFailure(errorText(error));
      setStatus("Could not load the selected LoRA folder");
    }
  };

  const selectInstalledZImageLora = (folder: string) => {
    if (!folder) {
      setZImageLora(""); setZImageLoraInfo(undefined); setVerifiedZImageLora("");
      return;
    }
    const info = installedZImageLoras.find((candidate) => candidate.folder === folder);
    if (!info) return;
    setZImageLora(info.folder); setZImageLoraInfo(info); setVerifiedZImageLora(""); setFailure(undefined);
    setStatus(`Selected ${info.name}${info.triggerPhrase ? ` · trigger: ${info.triggerPhrase}` : ""}`);
  };

  const insertLoraTrigger = () => {
    const trigger = zImageLoraInfo?.triggerPhrase?.trim();
    if (!trigger || prompt.toLocaleLowerCase().includes(trigger.toLocaleLowerCase())) return;
    setPrompt((current) => `${trigger}${current.trim() ? `, ${current.trim()}` : ""}`);
  };
  const requiredLoraTrigger = zImageLoraInfo?.trigger?.trim() || zImageLoraInfo?.triggerPhrase?.trim();
  const promptHasLoraTrigger = !requiredLoraTrigger
    || prompt.toLocaleLowerCase().includes(requiredLoraTrigger.toLocaleLowerCase());
  /**
   * Get a picture out of the project folder and onto a path the user chose.
   *
   * Everything Image Studio produces lands in projects/<id>/media/images as PNG, which is app
   * storage nobody is expected to browse. Until this existed, getting a finished image out meant
   * finding it in there by hand.
   *
   * The alpha question is asked, never assumed. Pillow's default for RGBA-to-JPEG composites onto
   * black, so a tight cutout exported to JPEG would come back with a black halo that reads as a
   * bad matte rather than a format limit -- the worker refuses without an explicit matte colour,
   * and this is where that colour comes from.
   */
  /**
   * A variable-factor backend is told the exact size to produce; a fixed-factor one produces its
   * trained factor and needs no target at all. NVIDIA VFX is built around a fixed output size, so
   * for it this is required rather than optional.
   */
  const upscaleTarget = (model: { variableFactor?: boolean; factor: number }) =>
    model.variableFactor && selectedAsset?.width && selectedAsset?.height
      ? {
          targetWidth: Math.round(selectedAsset.width * model.factor / 2) * 2,
          targetHeight: Math.round(selectedAsset.height * model.factor / 2) * 2,
        }
      : {};

  const runExport = async () => {
    if (!selected) { setStatus("Select a layer to export"); return; }
    const stem = (selectedAsset?.name ?? selected.name).replace(/\.[^.]+$/, "");
    const extension = exportFormat === "jpeg" ? "jpg" : exportFormat;
    const target = await save({
      defaultPath: `${stem}.${extension}`,
      filters: [{ name: exportFormat.toUpperCase(), extensions: [extension] }],
    });
    if (!target) { setStatus("Export canceled"); return; }
    setBusy(true); setFailure(undefined);
    try {
      // The upscale is part of the export, not a separate errand: whatever the panel resolves is
      // what gets written to the path the user just chose. Running it first also keeps the
      // AI-before-resample rule -- the exporter's own resize is the resample stage.
      let sourcePath = selected.path;
      if (upscaleModel) {
        setRunningLabel(`${upscaleModel.label} on ${selected.name}`);
        setStatus(`Upscaling before export with ${upscaleModel.label}…`);
        const upscaled = await upscaleImage({
          sourcePath: selected.path, projectId: project.id,
          backend: upscaleModel.backend, modelFile: upscaleModel.file, quality: upscaleQuality,
          tile: upscaleTile, ...upscaleTarget(upscaleModel),
        });
        sourcePath = upscaled.outputPath;
      }
      setRunningLabel("Writing the exported image");
      const result = await exportImage({
        sourcePath, outputPath: target,
        format: exportFormat, quality: exportQuality,
        lossless: exportFormat === "webp" && exportQuality >= 100,
        // Only sent for formats that cannot carry alpha; PNG/TIFF keep it either way.
        matte: exportFormat === "jpeg" ? exportMatte : undefined,
      });
      setStatus(`Exported ${filename(result.outputPath)} · ${result.width}×${result.height}`);
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
      setStatus("Image export failed");
    } finally { setBusy(false); setRunningLabel(""); }
  };

  /**
   * Real-ESRGAN on one still. The result goes to Project Media as a new asset rather than onto
   * the canvas or the timeline -- an upscale is a new picture, and where it ends up is the user's
   * next decision, not this button's.
   */
  const runUpscale = async () => {
    const model = aiUpscaleModel(upscaleModelId);
    if (!selected || !model) { setStatus("Select a layer and an upscale model"); return; }
    setBusy(true); setFailure(undefined);
    setRunningLabel(`Real-ESRGAN ×${model.factor} on ${selected.name}`);
    setStatus(`Upscaling ${selected.name} at ×${model.factor}…`);
    try {
      const result = await upscaleImage({
        sourcePath: selected.path, projectId: project.id,
        backend: model.backend, modelFile: model.file, quality: upscaleQuality,
        tile: upscaleTile, ...upscaleTarget(model),
      });
      // Named after the source and the operation. Forty batch outputs arriving in the bin at once
      // are only useful if they can be told apart, and `origin` already carries the lineage.
      const asset: MediaAsset = {
        // nvvfx has no weight file, so the model label stands in as the recorded "model".
        ...imageAsset(result, "esrgan-upscale", model.file ?? model.label, undefined, [selected.assetId],
          { factor: model.factor, tile: upscaleTile, architecture: model.architecture }),
        name: `${(selectedAsset?.name ?? selected.name).replace(/\.[^.]+$/, "")} · ESRGAN ×${model.factor}`,
      };
      addAssets([asset]);
      setStatus(`${asset.name} added to Project Media · ${result.width}×${result.height}`);
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
      setStatus("Upscale failed");
    } finally { setBusy(false); setRunningLabel(""); }
  };

  // The model reads this list positionally: index 0 becomes "Picture 1" in the
  // prompt it is given, index 1 "Picture 2", and so on. Deduped by path, since
  // the same file in two slots costs a second ~4096-token reference block and
  // doubles that content's pull on the result for nothing.
  const qwenPaths = () => {
    if (!selected) return [];
    const paths = [selected.path];
    for (const id of [reference1, reference2]) {
      const path = project.assets.find((asset) => asset.id === id)?.path;
      if (path && !paths.includes(path)) paths.push(path);
    }
    return paths.slice(0, 3);
  };
  // Reference slots fill top-down so a box label always matches the picture
  // number the model is given. A hole would send "Image 3" as Picture 2, and a
  // prompt that says "the jacket in image 3" would then point at nothing.
  const setQwenReference = (slot: 2 | 3, assetId: string): 2 | 3 => {
    if (reference1 === assetId) return 2;
    if (reference2 === assetId) return 3;
    if (slot === 3 && !reference1) { setReference1(assetId); return 2; }
    if (slot === 2) setReference1(assetId); else setReference2(assetId);
    return slot;
  };
  const clearQwenReference = (slot: 2 | 3) => {
    if (slot === 2) { setReference1(reference2); setReference2(""); } else setReference2("");
  };
  const pickQwenReference = async (slot: 2 | 3) => {
    const picked = await open({ multiple: false, directory: false, filters: [{ name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] }] });
    if (typeof picked !== "string") return;
    setFailure(undefined);
    try {
      let asset = imageAssets.find((candidate) => candidate.path === picked);
      if (!asset) {
        const probe = await probeMedia(picked);
        asset = {
          id: crypto.randomUUID(), name: filename(probe.path), path: probe.path, kind: "image",
          duration: fromSeconds(5), width: probe.width, height: probe.height, colorSpace: probe.colorSpace,
        };
        addAssets([asset]);
      }
      const landed = setQwenReference(slot, asset.id);
      setStatus(`Selected ${asset.name} as Qwen Image ${landed} and added it to Project Media`);
    } catch (error) {
      setFailure(errorText(error));
      setStatus(`Could not load Qwen Image ${slot}`);
    }
  };
  /**
   * Rewrite: Qwen's own edit-prompt enhancer rules on the local Qwen2.5-VL-7B, shown the same
   * pictures the edit will get (Picture 1 = the selected canvas image, then Images 2 and 3). The
   * result replaces the instruction; Undo puts the original back until the text is edited.
   */
  const rewriteInstruction = async () => {
    const original = prompt.trim();
    if (!original) { setStatus("Write an edit instruction to rewrite"); return; }
    const pictures = qwenPaths();
    setBusy(true); setFailure(undefined); setRunningLabel("Qwen2.5-VL is rewriting the instruction");
    setStatus(`Rewriting the instruction${pictures.length ? ` with ${pictures.length} image${pictures.length === 1 ? "" : "s"}` : " (no image selected: text only)"}…`);
    try {
      const result = await rewriteQwenEditPrompt(original, pictures);
      setPrompt(result.prompt);
      setRewriteUndo({ original: prompt, rewritten: result.prompt });
      setStatus("Instruction rewritten · Undo restores yours");
    } catch (error) { setFailure(errorText(error)); setStatus("Rewriting the instruction failed"); } finally { setBusy(false); setRunningLabel(""); }
  };
  const runEdit = async () => {
    if (!selected) { setStatus("Add and select an image layer before editing"); return; }
    if (!prompt.trim()) { setStatus("Describe the Qwen edit"); return; }
    setBusy(true); setFailure(undefined); setRunningLabel("Qwen is streaming transformer blocks and editing"); setStatus("Qwen Image Edit is running with native Efficient SDPA…");
    try {
      const result = await invoke<ImageModelResult>("edit_qwen_image", { request: {
        projectId: project.id, prompt, negativePrompt, imagePaths: qwenPaths(), steps, trueCfgScale: cfg, seed: numberSeed(seed),
        outputWidth: editOverride?.width ?? 0, outputHeight: editOverride?.height ?? 0,
      } });
      const sources = [selectedAsset?.id, reference1, reference2].filter(Boolean) as string[];
      const asset = imageAsset(result, "qwen-edit", "Qwen-Image-Edit-2511 FP8 + Lightning", prompt, sources, { steps, trueCfgScale: cfg, ...(editOverride ? { outputWidth: editOverride.width, outputHeight: editOverride.height } : {}) });
      addAssets([asset]); addLayer(asset); setStatus("Qwen edit added to the canvas and Project Media");
    } catch (error) { setFailure(errorText(error)); setStatus("Qwen Image Edit failed"); } finally { setBusy(false); setRunningLabel(""); }
  };
  const removeBackground = async () => {
    if (!selected || !selectedAsset) { setStatus("Select an image layer before removing its background"); return; }
    setBusy(true); setFailure(undefined); setRunningLabel("BiRefNet is creating a soft alpha matte"); setStatus("BiRefNet is removing the selected layer background…");
    try {
      const result = await invoke<ImageModelResult>("remove_image_background", { request: { projectId: project.id, imagePath: selected.path, blur: bgBlur, dilate: bgDilate, erode: bgErode, threshold: -1, invert: false } });
      const cutout = imageAsset(result, "birefnet-cutout", "BiRefNet", undefined, [selectedAsset.id], { blur: bgBlur, dilate: bgDilate, erode: bgErode });
      const assets = [cutout];
      if (result.maskPath) assets.push({ ...imageAsset({ ...result, outputPath: result.maskPath }, "birefnet-mask", "BiRefNet", undefined, [selectedAsset.id], {}), name: filename(result.maskPath) });
      addAssets(assets); patchLayer(selected.id, { assetId: cutout.id, path: cutout.path, name: cutout.name }); setStatus("Background removed; cutout and mask added to Project Media");
    } catch (error) { setFailure(errorText(error)); setStatus("BiRefNet background removal failed"); } finally { setBusy(false); setRunningLabel(""); }
  };
  const deleteProjectImage = (asset: MediaAsset) => {
    setLayers((current) => current.filter((layer) => layer.assetId !== asset.id));
    if (selectedAsset?.id === asset.id) setSelectedLayerId(undefined);
    if (reference1 === asset.id || reference2 === asset.id) {
      // Close the gap rather than leaving a hole that would renumber the pictures.
      const kept = [reference1, reference2].filter((id) => id && id !== asset.id);
      setReference1(kept[0] ?? ""); setReference2(kept[1] ?? "");
    }
    removeAsset(asset.id);
    setStatus(`Removed ${asset.name} from Project Media and the canvas`);
  };
  /**
   * The canvas as a Photo Edit document.
   *
   * This is the bridge that ended the app having two layer systems. Placement here is measured off
   * the DOM stage, which is letterboxed by whatever shape the window happens to be, so the same
   * canvas could flatten to slightly different pixels depending on the panel size. A layer's scale
   * is instead resolved against the canvas itself: 100% is the picture contained in the canvas, and
   * the answer no longer depends on how wide the window is.
   */
  const canvasDocument = async (
    background: PhotoDocument["background"],
  ): Promise<{ document: PhotoDocument; sources: Map<string, LayerSource> } | undefined> => {
    const visible = layers.filter((layer) => layer.visible);
    if (!visible.length) return undefined;
    const bottom = visible[0];
    // Sizes come from the decoded pictures rather than from what the asset record claims, because a
    // record imported without dimensions would otherwise place its layer at a different size than
    // the canvas preview shows -- and the preview is what the user arranged against.
    const sources = await loadLayerSources(layers.map((layer) => layer.path));
    const photoLayers: PhotoLayer[] = layers.map((layer) => {
      const source = sources.get(layer.path);
      const fit = source
        ? Math.min(width / source.width, height / source.height)
        : 1;
      return createLayer({ id: layer.assetId, name: layer.name, path: layer.path }, {
        id: layer.id,
        x: layer.x / 100, y: layer.y / 100,
        scale: fit * layer.scale / 100,
        rotation: layer.rotation,
        cropLeft: layer.cropLeft, cropTop: layer.cropTop,
        cropRight: layer.cropRight, cropBottom: layer.cropBottom,
        opacity: layer.opacity / 100,
        visible: layer.visible, locked: layer.locked,
      });
    });
    return {
      document: {
        assetId: bottom.assetId, sourcePath: bottom.path,
        layers: photoLayers, canvasWidth: width, canvasHeight: height, background,
        nodes: [], masks: [],
        rotate: 0, flipHorizontal: false, flipVertical: false,
        updatedAt: new Date().toISOString(),
      },
      sources,
    };
  };

  /**
   * Composite the canvas into a new picture, in the webview.
   *
   * The same function Photo Edit renders with, at the canvas's own size -- not a second compositor
   * in Python. Two implementations of the same placement maths is this project's recurring bug
   * class, and the flatten worker was the last one left.
   */
  const flattenCanvas = async () => {
    if (!layers.some((layer) => layer.visible)) { setStatus("Add at least one visible layer before creating a canvas image"); return; }
    setBusy(true); setFailure(undefined); setRunningLabel("Compositing the visible canvas layers into a new image");
    setStatus(flattenMode === "tight" ? "Creating a tightly cropped transparent canvas image…" : "Creating a framed canvas image…");
    try {
      const built = await canvasDocument(
        flattenMode === "tight" ? "transparent" : flattenBackground as PhotoDocument["background"]);
      if (!built) throw new Error("Add at least one visible layer before creating a canvas image");
      const { document, sources } = built;
      const { pixels } = composeDocument({
        document, width, height, sources, resourcesFor: () => emptyResources(),
      });
      const full = window.document.createElement("canvas");
      full.width = width;
      full.height = height;
      full.getContext("2d")!.putImageData(new ImageData(pixels, width, height), 0, 0);

      // Tight trims to what is actually visible, which is what alpha already knows; framed keeps the
      // canvas it was composed at.
      const box = flattenMode === "tight"
        ? padded(alphaBounds(pixels, width, height), Math.max(0, Math.trunc(flattenPadding)), width, height)
        : { left: 0, top: 0, width, height };
      if (!box.width || !box.height) throw new Error("Every visible layer is fully transparent");
      const out = window.document.createElement("canvas");
      out.width = box.width;
      out.height = box.height;
      out.getContext("2d")!.drawImage(full, box.left, box.top, box.width, box.height, 0, 0, box.width, box.height);

      const url = out.toDataURL("image/png");
      // A canvas past the browser's size limit returns "data:," rather than throwing, which would
      // otherwise reach the Rust side as an empty PNG and fail there with something unhelpful.
      if (url.length < 128) throw new Error(`${out.width} × ${out.height} is past what the renderer can encode in one image`);
      const written = await writeEditedImage({
        projectId: project.id, pngBase64: url.slice(url.indexOf(",") + 1),
        sourcePath: document.sourcePath, width: out.width, height: out.height,
      });
      const probe = await probeMedia(written.outputPath);
      const sourceAssetIds = [...new Set(layers.filter((layer) => layer.visible).map((layer) => layer.assetId))];
      const result: ImageModelResult = {
        outputPath: written.outputPath,
        width: written.width ?? out.width,
        height: written.height ?? out.height,
        log: `Composited ${document.layers?.filter((layer) => layer.visible).length ?? 0} layers at ${width} × ${height}`,
      };
      const asset = imageAsset(result, "canvas-flatten", "Canvas compositor", undefined, sourceAssetIds, {
        crop: flattenMode, padding: flattenPadding, background: flattenMode === "tight" ? "transparent" : flattenBackground,
        canvasWidth: width, canvasHeight: height, colorSpace: probe.colorSpace,
      });
      addAssets([asset]);
      setStatus(`Created ${asset.name} and added it to Project Media`);
    } catch (error) { setFailure(errorText(error)); setStatus("Canvas image creation failed"); } finally { setBusy(false); setRunningLabel(""); }
  };

  /**
   * Hand the whole canvas to Photo Edit, layers intact.
   *
   * The same document the flatten composites, stored in the project instead of rendered. From there
   * every layer gains its own grade, relight, mask and edge treatment, and the arrangement is saved,
   * undone and copied between machines by the project file rather than living in this component.
   */
  const sendCanvasToPhotoEdit = async () => {
    setBusy(true); setFailure(undefined); setRunningLabel("Handing the canvas to Photo Edit");
    try {
      const built = await canvasDocument("transparent");
      if (!built) { setStatus("Add at least one visible layer first"); return; }
      // Anything already edited over there is kept. Overwriting a document because a canvas happens
      // to share its bottom picture would throw away grades and masks the user cannot get back.
      const merged = mergeIntoExisting(built.document, project.photoEdits?.[built.document.assetId]);
      setPhotoDocument(merged, { coalesce: false });
      openPhotoEdit(merged.assetId);
      setStatus(`Canvas moved to Photo Edit · ${merged.layers?.length ?? 0} layers`);
    } catch (error) { setFailure(errorText(error)); setStatus("Could not hand the canvas to Photo Edit"); }
    finally { setBusy(false); setRunningLabel(""); }
  };

  const closeFramePicker = () => { setFrameSource(undefined); setFrameIndex(0); setPickedFrames([]); setFrameFailure(undefined); };
  const openFramePicker = async () => {
    const picked = await open({
      multiple: false, directory: false,
      filters: [{ name: "Video files", extensions: ["mp4", "mov", "mkv", "webm", "avi", "m4v"] }],
    });
    if (typeof picked !== "string") return;
    setBusy(true); setFailure(undefined); setStatus("Inspecting video for frame import…");
    try {
      const probe = await probeMedia(picked);
      if (!probe.videoCodec || !probe.width || !probe.height || probe.durationSeconds <= 0) throw new Error("The selected file has no usable video stream");
      const rate = parseFrameRate(probe.frameRate);
      const fps = Math.max(1, Math.min(240, rate.numerator / rate.denominator));
      setFrameSource({ path: probe.path, name: filename(probe.path), duration: probe.durationSeconds, width: probe.width, height: probe.height, fps });
      setFrameIndex(0); setPickedFrames([]); setFrameFailure(undefined); setStatus("Choose one or more video frames");
    } catch (error) { setFailure(errorText(error)); setStatus("Could not open the video frame picker"); } finally { setBusy(false); }
  };
  const seekVideoFrame = (requested: number) => {
    if (!frameSource) return;
    const next = Math.max(0, Math.min(maxVideoFrame, Math.trunc(requested)));
    setFrameIndex(next);
    if (frameVideoRef.current) {
      frameVideoRef.current.pause();
      frameVideoRef.current.currentTime = next / frameSource.fps;
    }
  };
  const pickCurrentVideoFrame = async () => {
    if (!frameSource || pickedFrames.some((item) => item.frameIndex === frameIndex)) return;
    if (pickedFrames.length >= 32) { setFrameFailure("At most 32 frames can be imported at once"); return; }
    setFrameFailure(undefined);
    try {
      const thumbnail = await invoke<string>("clip_thumbnail", { request: { path: frameSource.path, atSeconds: currentFrameSeconds, width: 180 } });
      setPickedFrames((current) => [...current, { frameIndex, timestampSeconds: currentFrameSeconds, thumbnail }].sort((left, right) => left.frameIndex - right.frameIndex));
    } catch (error) { setFrameFailure(errorText(error)); }
  };
  const importPickedVideoFrames = async () => {
    if (!frameSource || !pickedFrames.length) return;
    setBusy(true); setFrameFailure(undefined); setRunningLabel(`Extracting ${pickedFrames.length} selected video frame${pickedFrames.length === 1 ? "" : "s"}`);
    setStatus("Extracting selected frames as lossless PNG images…");
    try {
      const extracted = await invoke<ExtractedVideoFrame[]>("extract_video_frames", { request: {
        projectId: project.id, path: frameSource.path,
        timestamps: pickedFrames.map((item) => item.timestampSeconds),
      } });
      const sourceAsset = project.assets.find((asset) => asset.path === frameSource.path);
      const stem = frameSource.name.replace(/\.[^.]+$/, "");
      const assets = extracted.map((frame) => ({
        ...imageAsset(
          { outputPath: frame.outputPath, width: frame.width, height: frame.height, log: "" },
          "video-frame", "FFmpeg frame extraction", undefined, sourceAsset ? [sourceAsset.id] : [],
          { sourcePath: frameSource.path, timestampSeconds: frame.timestampSeconds, frameIndex: frame.frameIndex, sourceFps: frameSource.fps },
        ),
        name: `${stem}-frame-${String(frame.frameIndex).padStart(6, "0")}.png`,
      }));
      addAssets(assets); assets.forEach(addLayer); closeFramePicker();
      setStatus(`Added ${assets.length} video frame${assets.length === 1 ? "" : "s"} to Project Media and the canvas`);
    } catch (error) { setFrameFailure(errorText(error)); setStatus("Video frame extraction failed"); } finally { setBusy(false); setRunningLabel(""); }
  };
  const preset = (w: number, h: number) => { setWidth(w); setHeight(h); };
  const qwenPreset = (kind: "lightning" | "balanced" | "quality") => { if (kind === "lightning") { setSteps(4); setCfg(1); } else if (kind === "balanced") { setSteps(8); setCfg(1); } else { setSteps(40); setCfg(4); } };
  const selectZImageMode = (value: ZImageMode) => {
    setZImageMode(value);
    if (value === "turbo") { setSteps(9); setCfg(0); setNegativePrompt(""); }
    else { setSteps(36); setCfg(4); }
  };

  const testZImageCheckpoint = async (path: string) => {
    try {
      const info = await inspectZImageLora(path);
      setZImageLora(info.folder); setZImageLoraInfo(info); setVerifiedZImageLora("");
      // A LoRA trained for Turbo is tested in Fast mode; everything else in Creative/Base.
      const turboLora = (info.baseModel ?? "").toLocaleLowerCase().includes("turbo");
      setMode("generate"); selectZImageMode(turboLora ? "turbo" : "base");
      const trigger = info.triggerPhrase?.trim();
      if (trigger && !prompt.toLocaleLowerCase().includes((info.trigger ?? trigger).toLocaleLowerCase())) {
        setPrompt((current) => `${trigger}${current.trim() ? `, ${current.trim()}` : ""}`);
      }
      setStatus(`Checkpoint ${info.step ?? ""} selected in ${turboLora ? "Fast/Turbo" : "Creative/Base"} mode for testing`);
    } catch (error) { setFailure(errorText(error)); setStatus("Could not open the training checkpoint"); }
  };

  if (mode === "train") return <main className="image-studio lora-training-mode">
    <aside className="image-studio-controls lora-mode-sidebar">
      <div className="image-studio-title"><span><Sparkles size={18} /></span><div><strong>Image Studio</strong><small>Create, edit, and train locally</small></div></div>
      <div className="studio-mode-tabs">
        <button onClick={() => { setMode("generate"); selectZImageMode("turbo"); }}><ImagePlus size={15} /> Generate</button>
        <button onClick={() => { setMode("edit"); qwenPreset("lightning"); }}><WandSparkles size={15} /> Edit</button>
        <button className="selected"><Sparkles size={15} /> Train LoRA</button>
      </div>
      <section className="studio-card"><strong>Training workflow</strong><p>Choose references, create editable captions with the Qwen 2511 vision encoder, and train a reusable adapter against Z-Image Base.</p><small>Captioning and training run as separate one-model GPU jobs.</small></section>
    </aside>
    <LoraTraining projectId={project.id} runtime={runtime} setStatus={setStatus} onTestCheckpoint={testZImageCheckpoint} />
  </main>;

  return <main className="image-studio">
    <aside className="image-studio-controls">
      <div className="image-studio-title"><span><Sparkles size={18} /></span><div><strong>Image Studio</strong><small>Create the frames before generating video</small></div></div>
      <div className="studio-mode-tabs"><button className={mode === "generate" ? "selected" : ""} onClick={() => { setMode("generate"); selectZImageMode("turbo"); }}><ImagePlus size={15} /> Generate</button><button className={mode === "edit" ? "selected" : ""} onClick={() => { setMode("edit"); qwenPreset("lightning"); }}><WandSparkles size={15} /> Edit</button><button onClick={() => setMode("train")}><Sparkles size={15} /> Train LoRA</button></div>
      <section className="studio-card">
        <div className="studio-card-heading"><strong>{mode === "generate" ? (zImageMode === "base" ? "Z-Image Creative" : "ZiT Fast Generator") : "Qwen Image Edit 2511"}</strong><b className={(mode === "generate" ? (zImageMode === "base" ? runtime?.zimageBaseReady : runtime?.zitImageReady) : runtime?.qwenImageEditReady) ? "ready" : "not-ready"}>{(mode === "generate" ? (zImageMode === "base" ? runtime?.zimageBaseReady : runtime?.zitImageReady) : runtime?.qwenImageEditReady) ? "READY" : "MISSING"}</b></div>
        {mode === "generate" && <div className="zimage-mode-picker"><button className={zImageMode === "turbo" ? "selected" : ""} onClick={() => selectZImageMode("turbo")}><strong>Fast</strong><small>ZiT · 9 steps</small></button><button className={zImageMode === "base" ? "selected" : ""} onClick={() => selectZImageMode("base")}><strong>Creative</strong><small>Base · diverse seeds</small></button></div>}
        <div className="studio-field"><div className="studio-field-head"><span>{mode === "generate" ? "Frame prompt" : "Edit instruction"}</span>
          {mode === "edit" && <span className="rewrite-actions">
            {rewriteUndo && rewriteUndo.rewritten === prompt && <button type="button" className="rewrite-undo" disabled={busy} onClick={() => { setPrompt(rewriteUndo.original); setRewriteUndo(undefined); setStatus("Restored your instruction"); }} title="Put back the instruction you wrote">Undo</button>}
            <button type="button" className="rewrite-button" disabled={busy || !prompt.trim() || !runtime?.qwenEditRewriterReady} onClick={() => void rewriteInstruction()}
              title={runtime?.qwenEditRewriterReady
                ? "Rewrites the instruction the way Qwen's own prompt enhancer does: direct and specific, naming what changes and what stays, and referring to Images 2 and 3 as picture 2 and 3. Qwen2.5-VL-7B looks at the selected image and any reference images first."
                : "Needs the Z-Image trainer runtime and Qwen-Image-Edit-2511's text encoder: run Setup-ZImage-Trainer.bat"}>
              {runningLabel.startsWith("Qwen2.5-VL") ? <RefreshCw className="spin" size={12} /> : <Sparkles size={12} />} Rewrite
            </button>
          </span>}</div>
          <textarea rows={5} value={prompt} aria-label={mode === "generate" ? "Frame prompt" : "Edit instruction"} onChange={(event) => setPrompt(event.target.value)} placeholder={mode === "generate" ? "Subject, environment, composition, lighting and lens…" : "Change only the selected image while preserving…"} /></div>
        {(mode === "edit" || zImageMode === "base") && <label className="studio-field"><span>Negative prompt{mode === "edit" ? " · Quality mode" : " · Creative mode"}</span><textarea rows={2} value={negativePrompt} onChange={(event) => setNegativePrompt(event.target.value)} /></label>}
        {mode === "generate" ? <><div className="studio-presets"><button onClick={() => preset(1344, 768)}>16:9</button><button onClick={() => preset(768, 1344)}>9:16</button><button onClick={() => preset(1024, 1024)}>1:1</button><button onClick={() => preset(1536, 1024)}>3:2</button></div><div className="studio-grid"><label><span>Width</span><input type="number" min={256} step={16} value={width} onChange={(event) => setWidth(Number(event.target.value))} /></label><label><span>Height</span><input type="number" min={256} step={16} value={height} onChange={(event) => setHeight(Number(event.target.value))} /></label></div></> : <><div className="studio-presets"><button onClick={() => qwenPreset("lightning")}>Lightning</button><button onClick={() => qwenPreset("balanced")}>Balanced</button><button onClick={() => qwenPreset("quality")}>Quality</button></div><div className="qwen-reference-section"><span className="qwen-reference-title">Optional inputs · selected canvas image is Image 1</span><div className="qwen-reference-boxes">{([{ slot: 2 as const, asset: referenceAsset1 }, { slot: 3 as const, asset: referenceAsset2 }]).map(({ slot, asset }) => <div className="qwen-reference-picker" key={slot}><button type="button" className={`qwen-reference-box${asset ? " has-image" : ""}`} title={`Click to choose Qwen Image ${slot}`} onClick={() => void pickQwenReference(slot)}>{asset ? <img src={convertFileSrc(asset.path)} alt={`Qwen Image ${slot}: ${asset.name}`} /> : <span><ImagePlus size={24} /><strong>Image {slot}</strong><small>Click to browse</small></span>}</button>{asset && <button type="button" className="qwen-reference-remove" title={`Remove Qwen Image ${slot}`} aria-label={`Remove Qwen Image ${slot}`} onClick={() => clearQwenReference(slot)}><X size={12} /></button>}<small className="qwen-reference-name">{asset?.name ?? `Optional Image ${slot}`}</small></div>)}</div></div></>}
        <div className="studio-grid"><label><span>Steps</span><input type="number" min={1} max={50} value={steps} onChange={(event) => setSteps(Number(event.target.value))} /></label>{(mode === "edit" || zImageMode === "base") && <label><span>{mode === "edit" ? "True CFG" : "CFG"}</span><input type="number" min={0} max={20} step={0.1} value={cfg} onChange={(event) => setCfg(Number(event.target.value))} /></label>}<label><span>Seed · -1 random</span><input value={seed} onChange={(event) => setSeed(event.target.value)} /></label></div>
        {mode === "edit" && <div className="edit-size">
          <label className="studio-field" title="The size of the edited image. Default follows Image 1 (the selected canvas image): its shape, at about one megapixel, which is what Qwen was trained on. A different shape from Image 1 re-frames the picture (Qwen fills or trims the edges); sizes above about 1.6 megapixels are slower and drift more.">
            <span>Output size</span>
            <select value={editSizeChoice} onChange={(event) => {
              const choice = event.target.value;
              // Manual starts from what would have been made, so there is something sensible to adjust.
              if (choice === "manual" && editSizeChoice !== "manual" && editOutputSize) setEditManualSize(editOutputSize);
              setEditSizeChoice(choice);
            }}>
              <option value="default">Default · same as Image 1{editDefaultSize ? ` (${editDefaultSize.width} × ${editDefaultSize.height})` : ""}</option>
              {EDIT_SIZE_PRESETS.map((preset) => <option key={preset.value} value={preset.value}>{preset.label}</option>)}
              <option value="manual">Manual…</option>
            </select>
          </label>
          {editSizeChoice === "manual" && <div className="studio-grid edit-size-manual">
            <label><span>Width</span><input type="number" min={EDIT_SIZE_MIN} max={EDIT_SIZE_MAX} step={16} value={editManualSize.width}
              onChange={(event) => setEditManualSize((current) => ({ ...current, width: Number(event.target.value) }))} /></label>
            <label><span>Height</span><input type="number" min={EDIT_SIZE_MIN} max={EDIT_SIZE_MAX} step={16} value={editManualSize.height}
              onChange={(event) => setEditManualSize((current) => ({ ...current, height: Number(event.target.value) }))} /></label>
          </div>}
          {editSizeChoice !== "default" && editOverride && <small className={"edit-size-note" + (editOverride.width * editOverride.height > EDIT_AREA_COMFORT ? " warn" : "")}>
            Makes {editOverride.width} × {editOverride.height}{editSizeChoice === "manual" && (editOverride.width !== editManualSize.width || editOverride.height !== editManualSize.height) ? " (rounded to multiples of 16, 256–2048)" : ""}
            {editOverride.width * editOverride.height > EDIT_AREA_COMFORT ? " · above ~1.6 MP: slower, and Qwen drifts more" : ""}
          </small>}
        </div>}
        {mode === "generate" && <><div className="studio-grid"><label><span>Candidates</span><select value={candidateCount} onChange={(event) => setCandidateCount(Number(event.target.value) as 1 | 4)}><option value={1}>1 image</option><option value={4}>4-seed grid</option></select></label><label><span>LoRA strength</span><input type="number" min={0} max={3} step={0.05} value={zImageLoraStrength} disabled={!zImageLora} onChange={(event) => setZImageLoraStrength(Number(event.target.value))} /></label></div><label className="studio-field"><span>Installed Z-Image LoRA</span><select value={zImageLora} onChange={(event) => selectInstalledZImageLora(event.target.value)}><option value="">None</option>{installedZImageLoras.map((lora) => <option key={lora.folder} value={lora.folder}>{lora.name} · {filename(lora.weights)}</option>)}</select></label><div className="zimage-lora-row"><button onClick={() => void chooseZImageLora()}><FolderOpen size={14} /> Browse LoRA folder</button>{zImageLora && <button title="Unload LoRA" onClick={() => { setZImageLora(""); setZImageLoraInfo(undefined); setVerifiedZImageLora(""); }}><X size={13} /></button>}</div>{zImageLoraInfo && <div className="zimage-lora-details"><strong>{zImageLoraInfo.name}</strong><small title={zImageLoraInfo.weights}>Weights: {filename(zImageLoraInfo.weights)}</small>{zImageLoraInfo.triggerPhrase && <small>Trigger: <code>{zImageLoraInfo.triggerPhrase}</code>{!promptHasLoraTrigger && <button type="button" onClick={insertLoraTrigger}>Insert trigger</button>}</small>}<small className={verifiedZImageLora ? "ready" : "pending"}>{verifiedZImageLora ? `Verified active: ${filename(verifiedZImageLora)}` : "Selected · activation is verified on the next generation"}</small>{zImageMode === "turbo" && !(zImageLoraInfo.baseModel ?? "").toLocaleLowerCase().includes("turbo") && <small className="lora-warning">Trained on Base. On Turbo the LoRA lands weaker — try strength 1.5–2.5, or use Creative/Base for the strongest match.</small>}</div>}</>}
        <button className={"studio-run" + (mode === "edit" ? " edit-run" : "")} disabled={busy || (mode === "generate" ? !(zImageMode === "base" ? runtime?.zimageBaseReady : runtime?.zitImageReady) : !runtime?.qwenImageEditReady) || (mode === "edit" && !selected)} onClick={() => void (mode === "generate" ? runGenerate() : runEdit())}>{busy ? <RefreshCw className="spin" size={16} /> : <WandSparkles size={16} />}{mode === "generate" ? (candidateCount === 4 ? "Generate 4 variations" : "Generate frame") : "Generate edited image"}</button>
      </section>
      <section className="studio-card"><div className="studio-card-heading"><strong>BiRefNet background remover</strong><b className={runtime?.birefnetReady ? "ready" : "not-ready"}>{runtime?.birefnetReady ? "READY" : "MISSING"}</b></div><div className="studio-grid"><label><span>Blur</span><input type="number" min={0} max={5} value={bgBlur} onChange={(event) => setBgBlur(Number(event.target.value))} /></label><label><span>Dilate</span><input type="number" min={0} max={10} value={bgDilate} onChange={(event) => setBgDilate(Number(event.target.value))} /></label><label><span>Erode</span><input type="number" min={0} max={10} value={bgErode} onChange={(event) => setBgErode(Number(event.target.value))} /></label></div><button disabled={busy || !selected || !runtime?.birefnetReady} onClick={() => void removeBackground()}><Layers3 size={15} /> Remove selected background</button></section>
      <section className="studio-card">
        <div className="studio-card-heading">
          <strong>Export and upscale</strong>
          <b className={upscalerReady ? "ready" : "not-ready"}>{upscalerReady ? "AI READY" : "AI MISSING"}</b>
        </div>

        {/* Upscale is a pre-step OF the export, not a separate errand -- it sits above the format
            controls because that is the order it runs in, and the readout below shows the size the
            file will actually be written at. */}
        <label className="studio-field" title={upscaleModel?.hint
          ?? "An AI upscale reconstructs detail rather than redistributing existing pixels. It runs before the export, so the file you save is the upscaled one."}>
          <span>Upscale first</span>
          <select value={upscaleModelId} disabled={busy || !upscalerReady}
            onChange={(event) => setUpscaleModelId(event.target.value)}>
            <option value="">Off · export at original size</option>
            {upscaleModels.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
          </select>
        </label>
        {upscaleModel?.backend === "nvvfx" && <label className="studio-field"
          title="How hard the model works, and which source it assumes. The plain levels also remove compression artefacts; the clean-source levels keep more of what is already there.">
          <span>Quality</span>
          <select value={upscaleQuality} disabled={busy}
            onChange={(event) => setUpscaleQuality(event.target.value)}>
            {(upscaleModel.qualityLevels ?? []).map((level) => <option key={level} value={level}>
              {level.startsWith("HIGHBITRATE_") ? `${level.replace("HIGHBITRATE_", "").toLowerCase()} · clean source` : level.toLowerCase()}
            </option>)}
          </select>
        </label>}
        {upscaleModel?.backend === "esrgan" && <label className="studio-field"
          title="VRAM scales with the square of this. Too small and tile seams print as a faint grid.">
          <span>Tile</span>
          <select value={upscaleTile} disabled={busy}
            onChange={(event) => setUpscaleTile(Number(event.target.value))}>
            <option value={0}>Off · whole image at once</option>
            <option value={256}>256 px</option>
            <option value={512}>512 px</option>
            <option value={1024}>1024 px</option>
          </select>
        </label>}
        {upscalePlan && <div className="upscale-readout">
          <span>Resolved chain</span>
          <code>{upscalePlan.readout}</code>
        </div>}
        {runtime && !upscalerReady && <small className="panel-help">
          No AI upscaler installed. Real-ESRGAN ({runtime.realEsrganState}): {runtime.realEsrganState === "spandrel-missing"
            ? "re-run scripts\setup-image-studio.ps1 — the loader it needs is not installed"
            : ".venv\Scripts\python.exe scripts\models.py install real-esrgan"} — NVIDIA VFX ({runtime.nvvfxState}): Install-RTX-Upscale.bat -AcceptLicense
        </small>}

        <div className="studio-grid">
          <label><span>Format</span>
            <select value={exportFormat} disabled={busy}
              onChange={(event) => setExportFormat(event.target.value as typeof exportFormat)}>
              <option value="png">PNG · lossless, keeps alpha</option>
              <option value="jpeg">JPEG · no alpha</option>
              <option value="webp">WebP · alpha, lossy or lossless</option>
              <option value="tiff">TIFF · lossless, keeps alpha</option>
            </select>
          </label>
          {(exportFormat === "jpeg" || exportFormat === "webp") && <label>
            <span>Quality</span>
            <input type="number" min={1} max={100} value={exportQuality} disabled={busy}
              onChange={(event) => setExportQuality(Number(event.target.value))} />
          </label>}
          {exportFormat === "jpeg" && <label
            title="JPEG cannot store transparency. Anything transparent is composited onto this colour — the export refuses rather than silently using black.">
            <span>Matte</span>
            <input type="color" value={exportMatte} disabled={busy}
              onChange={(event) => setExportMatte(event.target.value)} />
          </label>}
        </div>

        {/* A disabled button that does nothing when clicked reads as a broken feature, so the
            reason is always on the control itself. */}
        <button className="studio-run" disabled={busy || !selected} onClick={() => void runExport()}
          title={selected ? undefined : "Select a layer on the canvas first — this exports the selected layer."}>
          <Download size={15} /> {upscaleModel ? "Upscale and export…" : "Export selected layer…"}
        </button>
        <button disabled={busy || !selected || !upscaleModelId} onClick={() => void runUpscale()}
          title={!selected ? "Select a layer on the canvas first."
            : !upscaleModelId ? "Choose an upscaler above first."
            : "Upscale without exporting — the result lands in Project Media, ready to drag onto the timeline."}>
          <Maximize2 size={15} /> Upscale to Project Media only
        </button>
        {!selected && <small className="panel-help">
          Both actions work on the selected canvas layer. Add an image to the canvas and click it first.
        </small>}
      </section>

      <section className="studio-card frame-import-card">
        <div className="studio-card-heading"><strong>Import video frames</strong></div>
        <p>Scrub a local video frame by frame, then add one or more lossless stills to the canvas and Project Media.</p>
        <button disabled={busy} onClick={() => void openFramePicker()}><Film size={15} /> Choose video</button>
      </section>
      <section className="studio-card canvas-output-card"><div className="studio-card-heading"><strong>Create canvas image</strong></div><div className="canvas-output-modes"><button className={flattenMode === "tight" ? "selected" : ""} onClick={() => setFlattenMode("tight")}>Tight alpha</button><button className={flattenMode === "frame" ? "selected" : ""} onClick={() => setFlattenMode("frame")}>Framed output</button></div>{flattenMode === "tight" ? <><p>Preserves layer alpha and trims the result to its visible pixels.</p><label className="studio-field"><span>Transparent padding</span><input type="number" min={0} max={4096} value={flattenPadding} onChange={(event) => setFlattenPadding(Number(event.target.value))} /></label></> : <><div className="studio-presets"><button onClick={() => preset(1344, 768)}>16:9</button><button onClick={() => preset(768, 1344)}>9:16</button><button onClick={() => preset(1024, 1024)}>1:1</button><button onClick={() => preset(1536, 1024)}>3:2</button></div><div className="studio-grid"><label><span>Width</span><input type="number" min={256} max={8192} step={16} value={width} onChange={(event) => setWidth(Number(event.target.value))} /></label><label><span>Height</span><input type="number" min={256} max={8192} step={16} value={height} onChange={(event) => setHeight(Number(event.target.value))} /></label></div><label className="studio-field"><span>Background</span><select value={flattenBackground} onChange={(event) => setFlattenBackground(event.target.value)}><option value="transparent">Transparent</option><option value="black">Black</option><option value="white">White</option></select></label></>}<button className="canvas-output-run" disabled={busy || !visibleLayers.length} onClick={() => void flattenCanvas()}><Layers3 size={15} /> Create image in Project Media</button><button className="canvas-output-run" disabled={busy || !visibleLayers.length} onClick={() => void sendCanvasToPhotoEdit()}><Sliders size={15} /> Open this canvas in Photo Edit</button><p>Photo Edit keeps the layers. Each one gets its own grade, relight, masks and edge treatment, and the arrangement is saved with the project.</p></section>
      {runningLabel && <div className="studio-running"><RefreshCw className="spin" size={16} /><div><strong>{runningLabel}</strong><small>Only one local GPU model runs at a time.</small></div></div>}
      {failure && <div className="worker-failure"><div><strong>Image worker failed</strong><button onClick={() => setFailure(undefined)}><X size={12} /></button></div><pre>{failure}</pre></div>}
    </aside>
    <section className="image-canvas-panel">
      <div className="image-canvas-toolbar">
        <div className="canvas-tool-modes">
          <button className={canvasTool === "layer" ? "selected" : ""} onClick={() => setCanvasTool("layer")}>
            <Move size={14} /> Move layer
          </button>
          <button className={canvasTool === "canvas" ? "selected" : ""} onClick={() => setCanvasTool("canvas")}>
            <Hand size={14} /> Pan canvas
          </button>
        </div>
        <div className="canvas-zoom-controls">
          <button
            type="button"
            className="zoom-btn"
            title="Zoom out (Ctrl + Wheel Down)"
            onClick={() => setCanvasZoom((z) => Math.max(25, Math.round((z - 25) / 25) * 25))}
          >
            <ZoomOut size={13} />
          </button>
          <select
            className="zoom-select"
            value={canvasZoom}
            onChange={(e) => setCanvasZoom(Number(e.target.value))}
          >
            <option value={25}>25%</option>
            <option value={50}>50%</option>
            <option value={75}>75%</option>
            <option value={100}>100%</option>
            <option value={125}>125%</option>
            <option value={150}>150%</option>
            <option value={200}>200%</option>
            <option value={300}>300%</option>
            <option value={400}>400%</option>
            <option value={500}>500%</option>
          </select>
          <button
            type="button"
            className="zoom-btn"
            title="Zoom in (Ctrl + Wheel Up)"
            onClick={() => setCanvasZoom((z) => Math.min(500, Math.round((z + 25) / 25) * 25))}
          >
            <ZoomIn size={13} />
          </button>
        </div>
        <b>{width} × {height}</b>
        <button
          disabled={canvasTool === "layer" && (!selected || selected.locked)}
          onClick={() => {
            if (canvasTool === "canvas") {
              setCanvasPan({ x: 0, y: 0 });
              setCanvasZoom(100);
            } else if (selected && !selected.locked) {
              patchLayer(selected.id, {
                x: 50, y: 50, scale: 100, rotation: 0, opacity: 100,
                cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
              });
            }
          }}
        >
          <RotateCcw size={14} /> Reset {canvasTool === "canvas" ? "view" : "layer"}
        </button>
      </div>
      <div className={`image-canvas-viewport ${canvasTool === "canvas" ? "pan-tool" : "layer-tool"}`} onWheel={handleCanvasWheel}>
        <div
          ref={stageRef}
          className="image-canvas-stage"
          style={{
            aspectRatio: `${width} / ${height}`,
            width: `min(100cqw, calc(100cqh * ${width / height}))`,
            height: `min(100cqh, calc(100cqw / ${width / height}))`,
            transform: `translate(${canvasPan.x}px, ${canvasPan.y}px) scale(${canvasZoom / 100})`,
            transformOrigin: "center center",
          }}
          onPointerDown={startCanvasPan}
          onPointerMove={movePointer}
          onPointerUp={stopPointer}
          onPointerCancel={stopPointer}
        >
          {!visibleLayers.length && (
            <div className="image-canvas-empty">
              <ImagePlus size={46} />
              <strong>Create or add a frame</strong>
              <span>ZiT outputs, Qwen edits, and project images appear here.</span>
            </div>
          )}
          {layers.map((layer, index) => layer.visible && (
            <img
              key={layer.id}
              data-layer-id={layer.id}
              src={convertFileSrc(layer.path)}
              draggable={false}
              onPointerDown={(event) => startDrag(event, layer)}
              className={`${selectedLayerId === layer.id ? "selected" : ""}${layer.locked ? " locked" : ""}`}
              style={{
                left: `${layer.x}%`,
                top: `${layer.y}%`,
                opacity: layer.opacity / 100,
                clipPath: `inset(${layer.cropTop}% ${layer.cropRight}% ${layer.cropBottom}% ${layer.cropLeft}%)`,
                transform: `translate(-50%, -50%) scale(${layer.scale / 100}) rotate(${layer.rotation}deg)`,
                zIndex: index + 1,
              }}
            />
          ))}
        </div>
      </div>
      {selected && (
        <div className={`layer-transform-bar${selected.locked ? " locked" : ""}`}>
          <NumericControl
            label="Scale"
            min={10}
            max={300}
            value={selected.scale}
            unit="%"
            disabled={selected.locked}
            onChange={(val) => patchLayer(selected.id, { scale: val })}
          />
          <NumericControl
            label="Rotate"
            min={-180}
            max={180}
            value={selected.rotation}
            unit="°"
            disabled={selected.locked}
            onChange={(val) => patchLayer(selected.id, { rotation: val })}
          />
          <NumericControl
            label="Opacity"
            min={0}
            max={100}
            value={selected.opacity}
            unit="%"
            disabled={selected.locked}
            onChange={(val) => patchLayer(selected.id, { opacity: val })}
          />
          <div className="layer-crop-controls">
            <header>
              <span>Crop edges</span>
              <button
                disabled={selected.locked || !(selected.cropLeft || selected.cropTop || selected.cropRight || selected.cropBottom)}
                onClick={() => patchLayer(selected.id, { cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0 })}
              >
                <RotateCcw size={11} /> Reset crop
              </button>
            </header>
            <div>
              <NumericControl
                label="Left"
                min={0}
                max={99 - selected.cropRight}
                value={selected.cropLeft}
                unit="%"
                disabled={selected.locked}
                onChange={(val) => patchLayer(selected.id, { cropLeft: val })}
              />
              <NumericControl
                label="Top"
                min={0}
                max={99 - selected.cropBottom}
                value={selected.cropTop}
                unit="%"
                disabled={selected.locked}
                onChange={(val) => patchLayer(selected.id, { cropTop: val })}
              />
              <NumericControl
                label="Right"
                min={0}
                max={99 - selected.cropLeft}
                value={selected.cropRight}
                unit="%"
                disabled={selected.locked}
                onChange={(val) => patchLayer(selected.id, { cropRight: val })}
              />
              <NumericControl
                label="Bottom"
                min={0}
                max={99 - selected.cropTop}
                value={selected.cropBottom}
                unit="%"
                disabled={selected.locked}
                onChange={(val) => patchLayer(selected.id, { cropBottom: val })}
              />
            </div>
          </div>
          {selected.locked && <span className="layer-locked-note"><Lock size={12} /> Layer locked</span>}
        </div>
      )}
    </section>
    <aside className="image-studio-library">
      <div className="studio-library-section"><div className="studio-library-heading"><strong>Layers</strong><span>{layers.length}</span></div><div className="studio-layer-list">{layers.slice().reverse().map((layer) => <button key={layer.id} className={`${selectedLayerId === layer.id ? "selected" : ""}${layer.locked ? " locked" : ""}`} title="Double-click the name to rename this layer" onClick={() => setSelectedLayerId(layer.id)} onDoubleClick={() => setRenamingLayerId(layer.id)}><img src={convertFileSrc(layer.path)} /><span><RenameField value={layer.name} editing={renamingLayerId === layer.id} onEditing={(on) => setRenamingLayerId(on ? layer.id : undefined)} onCommit={(next) => patchLayer(layer.id, { name: next.trim() || layer.name })} /><small>{layer.locked ? "Locked · " : ""}{Math.round(layer.scale)}% · {Math.round(layer.opacity)}%</small></span><i className={layer.locked ? "lock-active" : ""} title={layer.locked ? "Unlock layer" : "Lock layer"} onClick={(event) => { event.stopPropagation(); patchLayer(layer.id, { locked: !layer.locked }); }}>{layer.locked ? <Lock size={13} /> : <LockOpen size={13} />}</i><i title={layer.visible ? "Hide layer" : "Show layer"} onClick={(event) => { event.stopPropagation(); patchLayer(layer.id, { visible: !layer.visible }); }}>{layer.visible ? <Eye size={14} /> : <EyeOff size={14} />}</i><i title="Move layer up" onClick={(event) => { event.stopPropagation(); moveLayer(layer.id, 1); }}><ArrowUp size={13} /></i><i title="Move layer down" onClick={(event) => { event.stopPropagation(); moveLayer(layer.id, -1); }}><ArrowDown size={13} /></i><i title="Remove layer from canvas" onClick={(event) => { event.stopPropagation(); setLayers((current) => current.filter((item) => item.id !== layer.id)); if (selectedLayerId === layer.id) setSelectedLayerId(undefined); }}><Trash2 size={13} /></i></button>)}</div></div>
      <div className="studio-library-section project-images"><div className="studio-library-heading"><strong>Project Media</strong><span>{imageAssets.length} images</span></div><div className="studio-media-grid">{imageAssets.map((asset) => <article key={asset.id} onDoubleClick={() => setRenamingAssetId(asset.id)}>{asset.origin?.prompt && looksGenerated(asset.name) && <button className="studio-media-suggest" title={`Rename from its prompt: ${nameFromPrompt(asset.origin.prompt, extensionOf(asset.name))}`} aria-label={`Rename ${asset.name} from its prompt`} onClick={() => { const suggestion = nameFromPrompt(asset.origin?.prompt, extensionOf(asset.name)); if (suggestion) renameAsset(asset.id, suggestion); }}><WandSparkles size={12} /></button>}<button className="studio-media-rename" title="Rename" aria-label={`Rename ${asset.name}`} onClick={() => setRenamingAssetId(asset.id)}><Pencil size={12} /></button><button className="studio-media-delete" title="Remove from Project Media" aria-label={`Remove ${asset.name} from Project Media`} onClick={() => deleteProjectImage(asset)}><Trash2 size={12} /></button><img src={convertFileSrc(asset.path)} /><RenameField value={asset.name} editing={renamingAssetId === asset.id} onEditing={(on) => setRenamingAssetId(on ? asset.id : undefined)} onCommit={(next) => renameAsset(asset.id, next)} /><small>{asset.width && asset.height ? `${asset.width} × ${asset.height}` : "Image"}{asset.origin ? ` · ${asset.origin.model}` : ""}</small><div><button onClick={() => addLayer(asset)}><Plus size={13} /> Canvas</button><button title="Open in Photo Edit" onClick={() => openPhotoEdit(asset.id)}><Sliders size={13} /> Edit</button><button onClick={() => addAssetToTimeline(asset.id)}>+ Timeline</button></div></article>)}</div></div>
    </aside>
    {frameSource && <div className="frame-picker-backdrop" onPointerDown={(event) => { if (event.target === event.currentTarget && !busy) closeFramePicker(); }}>
      <section className="frame-picker-dialog" role="dialog" aria-modal="true" aria-label="Import video frames">
        <header className="frame-picker-header">
          <div><strong><Film size={18} /> Import video frames</strong><small>{frameSource.name} · {frameSource.width} × {frameSource.height} · {frameSource.fps.toFixed(3).replace(/\.000$/, "")} fps</small></div>
          <button title="Close frame picker" aria-label="Close frame picker" disabled={busy} onClick={closeFramePicker}><X size={16} /></button>
        </header>
        <div className="frame-picker-preview">
          <video ref={frameVideoRef} src={convertFileSrc(frameSource.path)} muted preload="auto" onLoadedMetadata={() => seekVideoFrame(frameIndex)} onError={() => setFrameFailure("The selected video could not be decoded for preview")} />
        </div>
        <div className="frame-picker-scrubber">
          <button title="Previous frame" aria-label="Previous frame" disabled={frameIndex <= 0} onClick={() => seekVideoFrame(frameIndex - 1)}><ChevronLeft size={18} /></button>
          <input aria-label="Video frame" type="range" min={0} max={maxVideoFrame} step={1} value={frameIndex} onChange={(event) => seekVideoFrame(Number(event.target.value))} />
          <button title="Next frame" aria-label="Next frame" disabled={frameIndex >= maxVideoFrame} onClick={() => seekVideoFrame(frameIndex + 1)}><ChevronRight size={18} /></button>
        </div>
        <div className="frame-picker-readout">
          <span>Frame {(frameIndex + 1).toLocaleString()} of {(maxVideoFrame + 1).toLocaleString()}</span>
          <b>{clock(currentFrameSeconds)}</b>
          <button disabled={busy || pickedFrames.some((item) => item.frameIndex === frameIndex)} onClick={() => void pickCurrentVideoFrame()}><Plus size={15} /> {pickedFrames.some((item) => item.frameIndex === frameIndex) ? "Selected" : "Select current frame"}</button>
        </div>
        <div className="frame-picker-selection">
          {pickedFrames.length ? pickedFrames.map((frame) => <article key={frame.frameIndex}>
            <img src={frame.thumbnail} alt={`Selected frame ${frame.frameIndex + 1}`} />
            <span><strong>Frame {(frame.frameIndex + 1).toLocaleString()}</strong><small>{clock(frame.timestampSeconds)}</small></span>
            <button title="Remove selected frame" aria-label={`Remove frame ${frame.frameIndex + 1}`} disabled={busy} onClick={() => setPickedFrames((current) => current.filter((item) => item.frameIndex !== frame.frameIndex))}><X size={13} /></button>
          </article>) : <div className="frame-picker-empty">No frames selected yet. Scrub to a frame and choose <strong>Select current frame</strong>.</div>}
        </div>
        {frameFailure && <div className="frame-picker-error">{frameFailure}</div>}
        <footer className="frame-picker-actions">
          <span>{pickedFrames.length} selected · up to 32</span>
          <button disabled={busy} onClick={closeFramePicker}>Cancel</button>
          <button className="primary" disabled={busy || !pickedFrames.length} onClick={() => void importPickedVideoFrames()}>{busy ? <RefreshCw className="spin" size={15} /> : <ImagePlus size={15} />} Add selected to canvas</button>
        </footer>
      </section>
    </div>}
  </main>;
}

/* ------------------------------------------------------------------ helpers */

/**
 * A canvas handed to Photo Edit, keeping whatever was already edited there.
 *
 * Placement comes from the canvas the user just arranged, because that is the thing they are
 * carrying across. Everything that is work rather than arrangement -- node stacks, masks, edge
 * treatment, the depth map, the crop -- is taken from the document already stored, matched layer to
 * layer by the file each one points at. Without this, moving a canvas across would silently discard
 * an afternoon of grading because the two happened to share a bottom picture.
 */
function mergeIntoExisting(incoming: PhotoDocument, existing: PhotoDocument | undefined): PhotoDocument {
  if (!existing) return incoming;
  const held = new Map(layersOf(existing).map((layer) => [layer.sourcePath, layer]));
  return {
    ...incoming,
    nodes: existing.nodes,
    masks: existing.masks,
    depthPath: existing.depthPath,
    rotate: existing.rotate,
    flipHorizontal: existing.flipHorizontal,
    flipVertical: existing.flipVertical,
    straighten: existing.straighten,
    scale: existing.scale,
    crop: existing.crop,
    layers: (incoming.layers ?? []).map((layer) => {
      const previous = held.get(layer.sourcePath);
      if (!previous) return layer;
      held.delete(layer.sourcePath);
      return { ...layer, nodes: previous.nodes, masks: previous.masks, edge: previous.edge };
    }),
  };
}

/** The smallest rectangle holding every pixel that is not fully transparent. */
function alphaBounds(pixels: Uint8ClampedArray, width: number, height: number) {
  let left = width, top = height, right = -1, bottom = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (pixels[(y * width + x) * 4 + 3] === 0) continue;
      if (x < left) left = x;
      if (x > right) right = x;
      if (y < top) top = y;
      if (y > bottom) bottom = y;
    }
  }
  if (right < left || bottom < top) return { left: 0, top: 0, width: 0, height: 0 };
  return { left, top, width: right - left + 1, height: bottom - top + 1 };
}

/** Grow a rectangle by a margin, without leaving the canvas it was measured in. */
function padded(
  box: { left: number; top: number; width: number; height: number },
  margin: number, width: number, height: number,
) {
  if (!box.width || !box.height) return box;
  const left = Math.max(0, box.left - margin);
  const top = Math.max(0, box.top - margin);
  return {
    left, top,
    width: Math.min(width, box.left + box.width + margin) - left,
    height: Math.min(height, box.top + box.height + margin) - top,
  };
}
