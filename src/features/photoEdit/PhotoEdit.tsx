import { convertFileSrc } from "@tauri-apps/api/core";
import { open, save } from "@tauri-apps/plugin-dialog";
import {
  useCallback, useEffect, useMemo, useReducer, useRef, useState, type PointerEvent as ReactPointerEvent,
} from "react";
import {
  ChevronDown, Circle, Contrast, Crop, Download, FlipHorizontal, FlipVertical, Image as ImageIcon,
  Hand, Layers3, Lightbulb, Maximize2, Move, Palette, Plus, RefreshCw, RotateCw, Sliders, Sparkles,
  Stamp, Type, Wand2, Waves, Film, PenTool, Grid3x3, Spline,
} from "lucide-react";
import { isStillAsset } from "../../domain/media";
import { slugify } from "../../domain/naming";
import { fromSeconds } from "../../domain/time";
import {
  createGradeNode, createLayer, createMask, createPhotoDocument, createRelightNode, createRemoveNode,
  createGradientLayer, createGradientMapNode, createRetouchNode, createShapeLayer, createTextLayer,
  createWarpNode, geometryOf, isUntouched, layersOf, maskById, removalCacheKey, withLayers,
  type PhotoDocument, type PhotoLayer, type PhotoMask, type PhotoNode, type RemoveNode,
  type RetouchNode, type WarpNode,
  createLumaKeyNode, createMaskEntry, createUnmultiplyNode, type MaskEntry,
  createAlphaCurveNode, createMatrixRainLayer, createMediaLayer, type LayerMedia,
  createDepthOcclusionMask, createEmitterLayer, createTrailNode, type EmitterGenerator, type LayerAttach,
  type PointTrack, type TrackSample,
  createEnvelopeNode, createPathLayer, type EnvelopeNode,
} from "../../domain/photoDocument";
import { appendAnchor, setHandle, type PathBlock } from "../../domain/photoPath";
import {
  blendPathLayers, clipAboveToPath, combinePathLayers, envelopeFor, maskBelowWithPath, outlineLayerStroke, pathLayerBelow,
} from "../../domain/photoPathActions";
import { EnvelopeNodeEditor, EnvelopeNodeOverlay, PathEditor, PathOverlay, RepeatEditor, pathMapping, withoutAnchor } from "./PhotoPath";
import { textLayerToPath } from "../../services/textOutlineService";
import {
  attachedLayer, canvasToSourcePoint, depthFramePaths, greyOf, lucasKanadeBothWays, newTrackId, sam2Centroid,
  type GreyFrame,
} from "../../domain/photoTrack";
import { coverageFromImage, sequenceFramePath } from "../../domain/photoMask";
import { computeVideoDepth, motionStatus, trackPoints, type MotionStatus } from "../../services/motionService";
import { EmitterEditor, TracksPanel, TrailEditor, type TrackEngine } from "./PhotoMotion";
import {
  compositionDuration, compositionRate, isAnimated, mediaFrameKey, mediaFramesAt, mediaKeysAt, mediaNeedsAt, scopeFrame,
} from "../../domain/photoTime";
import { mediaFromElement, mediaFromVideoAsset } from "../../domain/elements";
import { MediaFrameStore } from "../../render/video/frameStore";
import { readMediaBytes } from "../../services/mediaBytes";
import {
  buildElementProxy, generateElement, importElement, listElements, updateElement, type ElementEntry,
} from "../../services/elementService";
import { maskFilePaths } from "../../domain/photoMask";
import type { MaskShape } from "../../domain/photoDocument";
import {
  COMPOSITE_SCOPE, canvasSizeOf, croppedRect, drawnSource, gradientPreview, isDrawnLayer,
  layerMatrix, placementSize, type ComposeScope, type LayerSource,
} from "../../domain/photoCompose";
import { toPaint, visibleGradientHandle, type Gradient, type Paint } from "../../domain/photoGradient";
import {
  BALLOON_KINDS, SHAPE_PRESETS, holdsText, shapeBounds, tailTip,
  type ShapeBlock, type ShapeKind,
} from "../../domain/photoShape";
import { matchColour } from "../../domain/photoEdge";
import {
  NEUTRAL_GEOMETRY, applyMatrix, compose, displayToSource, forwardMatrix, framedSize, invert,
  isNeutralGeometry, outputSize, sourceToDisplay, translate, type CropRect, type PhotoGeometry,
} from "../../domain/photoGeometry";
import { emptyResources, type RenderResources } from "../../domain/photoRender";
import { buildScopeResources } from "../../domain/photoResources";
import { filesWantedAt } from "../../domain/documentFiles";
import type { MediaAsset, TimelineClip } from "../../domain/types";
import { belongsToClip, clipPlateDocument, clipRenderPlan, hasVideoContent, plateLayerId, withPlateRange } from "../../domain/clipVfx";
import { FrameSender, renderVideoCancel, renderVideoFinish, renderVideoStart } from "../../services/renderService";
import { packYuv420 } from "../../render/yuv420";
import { histogram, whiteBalanceFor } from "../../domain/photoEdit";
import { FlaggedCompositor, readCompositorFlag, writeCompositorFlag } from "../../render/compositorFlag";
import {
  cacheMaskTrack, exportImage, filename, generatedAsset, photoDepthMap, photoInpaint, photoSelect, probeMedia,
  runtimeStatus, writeEditedImage, type AiRuntimeStatus,
} from "../../services/modelService";
import { loadReadableImage } from "../../services/readableImage";
import { useEditor } from "../../store";
import { RenameField } from "../media/RenameField";
import {
  GradeEditor, LayerEditor, LayerList, MaskEditor, MaskPicker, NodeList, RelightEditor, RemoveEditor,
  RetouchEditor, Slider, type HistogramData,
} from "./PhotoPanels";
import { CropOverlay, Handle, type Point } from "./PhotoHandles";
import { CharacterPanel, SpritesheetPanel, type CharacterOptions } from "./PhotoCharacter";
import { packAtlas, sheetFrames, sheetJson, sheetLayout, sheetSlug, spriteFramesTres, unionAlphaBounds } from "../../domain/spritesheet";
import { encodePng } from "../../domain/png";
import { spritesheetFolder, spritesheetWrite } from "../../services/spritesheetService";
import { actorFrame, actorRegion, characterLayer, cleanPlateRequest, drivenByPlate, drivingRange, insertAbove, isCleanPlate, seedPoints } from "../../domain/character";
import { buildCleanPlate, describeCharacterProgress, generateCharacter, onCharacterProgress, wanAnimateStatus, type AnimateStatus } from "../../services/characterService";
import {
  GradientEditor, GradientMapEditor, LumaKeyEditor, ShapeEditor, TextEditor, WarpEditor,
  AlphaCurveEditor, ElementsPanel, MatrixRainEditor, MediaEditor, TimeBar, type TrackedMatte,
} from "./PhotoPanels";
import type { WarpMode } from "../../domain/photoWarp";
import { availableFonts, type FontGroup } from "../../services/fonts";
import type { TextBlock } from "../../domain/photoText";

/**
 * Photo Edit: a non-destructive compositing and grading workspace.
 *
 * Nothing here is ever written back over a picture. What the user builds is a document -- an ordered
 * stack of LAYERS, each carrying its own stack of nodes over an untouched source file -- and that
 * document lives in the project, so it is saved, undone and copied between machines by machinery
 * that already existed. Saving renders the whole thing at full size into a NEW asset; every original
 * stays exactly where the clips using it expect to find it.
 *
 * The compositor runs in the webview and the same code draws the preview and the saved file, at two
 * resolutions. See `domain/photoCompose.ts` for why there is no second implementation.
 */

/** Long edge of the working copy. Big enough to judge a grade, small enough to stay interactive. */
const PREVIEW_LIMIT = 1500;

type Tool = "view" | "move" | "crop" | "brush" | "select" | "picker" | "retouch" | "warp" | "track" | "pen";
const FULL_CROP: CropRect = { x: 0, y: 0, width: 1, height: 1 };
type Loaded = { canvas: HTMLCanvasElement; width: number; height: number };
type SelectPoint = { x: number; y: number; include: boolean };
type RetouchBrush = { radius: number; hardness: number; heal: boolean };

/**
 * VFX Phase 4b: where this editor is running. Without a target it is the Photo Edit tab, for pictures
 * and single frames. With one it is the Clip Editor's VFX mode on a timeline clip: the clip's range is
 * the plate, every motion tool is on, and the output is a rendered MP4 that goes back on the timeline.
 */
export type ClipVfxTarget = {
  clip: TimelineClip;
  asset: MediaAsset;
  /** The Clip Editor's range, seconds from the clip's in point. Seeds a new document's plate. */
  rangeStart: number;
  rangeEnd: number;
  /** The Clip Editor's mode switch, shown in the title so VFX mode can be left. */
  modeSwitch?: React.ReactNode;
  onRunningChange?: (running: boolean) => void;
  onClose: () => void;
};

export function PhotoEdit({ clipTarget }: { clipTarget?: ClipVfxTarget } = {}) {
  const {
    project, activeWorkspace, photoEditAssetId, setPhotoEditAsset, sendToImageStudio,
    setPhotoDocument, setClipVfxDocument, addAssets, renameAsset, addAssetToTimeline, setActiveWorkspace,
    replaceClipRangeWithAsset, insertAssetAfterClip,
    busy, setBusy, setStatus, status: editorStatus,
  } = useEditor();
  /** The Clip Editor's VFX mode (video tools on), rather than the Photo Edit tab (pictures only). */
  const onClip = Boolean(clipTarget);
  /** Whether this instance is on screen: the Clip Editor mounts it only while VFX mode is open. */
  const live = onClip || activeWorkspace === "photo";

  const images = useMemo(
    // By the file, not by `kind`: a model-made video is also "generated", and listing an MP4 here
    // put a picture in the bin that could never be opened as one.
    () => project.assets.filter((asset) => isStillAsset(asset)),
    [project.assets],
  );
  const videos = useMemo(() => project.assets.filter((item) => item.kind === "video"), [project.assets]);
  // A remembered id can outlive the asset it names, and a deleted picture must not leave the
  // workspace pointing at nothing.
  const asset: MediaAsset | undefined = clipTarget
    ? clipTarget.asset
    : images.find((item) => item.id === photoEditAssetId) ?? images[0];
  const clipId = clipTarget?.clip.id;
  const clipSourceIn = clipTarget ? clipTarget.clip.sourceIn.numerator / clipTarget.clip.sourceIn.denominator : 0;

  // A document is only written to the project once something is actually changed. Creating one on
  // open would mark the project dirty just for looking at a picture. In the Clip Editor the new
  // document is the clip's range as a plate (domain/clipVfx.ts).
  const fallback = useMemo(
    () => clipTarget
      ? clipPlateDocument(clipTarget.clip, clipTarget.asset, clipTarget.rangeStart, clipTarget.rangeEnd)
      : asset ? createPhotoDocument(asset.id, asset.path) : undefined,
    [clipId, clipSourceIn, clipTarget?.rangeStart, clipTarget?.rangeEnd, asset?.id, asset?.path],
  );
  const clipStored = clipId ? project.clipVfx?.[clipId] : undefined;
  const stored = clipTarget
    ? clipStored && belongsToClip(clipStored, clipTarget.clip.id, clipTarget.asset.id) ? clipStored : undefined
    : asset ? project.photoEdits?.[asset.id] : undefined;
  const document = stored ?? fallback;
  const patch = useCallback((next: PhotoDocument, options?: { coalesce?: boolean }) => {
    if (clipId) setClipVfxDocument(clipId, next, options);
    else setPhotoDocument(next, options);
  }, [clipId, setClipVfxDocument, setPhotoDocument]);

  const [runtime, setRuntime] = useState<AiRuntimeStatus>();
  const [scopeId, setScopeId] = useState<string>();
  const [selectedNodeId, setSelectedNodeId] = useState<string>();
  const [activeMaskId, setActiveMaskId] = useState<string>();
  /** VFX Phase 2: the stack entry of the active mask that the brush and colour picker work on. */
  const [activeEntryId, setActiveEntryId] = useState<string>();
  const [tool, setTool] = useState<Tool>("view");
  const [brush, setBrush] = useState({ radius: 0.06, hardness: 0.7, erase: false });
  const [retouchBrush, setRetouchBrush] = useState<RetouchBrush>({ radius: 0.03, hardness: 0.6, heal: true });
  const [warpBrush, setWarpBrush] = useState<{ radius: number; strength: number; mode: WarpMode }>(
    { radius: 0.09, strength: 0.35, mode: "push" });
  // Where the clone brush copies FROM, normalised to the canvas. State rather than a ref because
  // the button that sets it has to say whether it has one.
  const [retouchSource, setRetouchSource] = useState<[number, number]>();
  // A locked ratio for the crop rectangle. Undefined is free-form, which is the honest default:
  // most crops are "a bit off the left", not "make this 16:9".
  const [cropAspect, setCropAspect] = useState<number>();
  const [fonts, setFonts] = useState<FontGroup[]>([]);
  const [points, setPoints] = useState<SelectPoint[]>([]);
  const [openCards, setOpenCards] = useState<Record<string, boolean>>({ layers: true, nodes: true, settings: true });
  const [comparing, setComparing] = useState(false);
  const [fit, setFit] = useState(true);
  const [showMask, setShowMask] = useState(true);
  const [levels, setLevels] = useState<HistogramData>();
  // What the next canvas click is for. Sampling serves three jobs now, and which one is running has
  // to be explicit or a click meant for white balance silently redefines a colour-range mask.
  const [picker, setPicker] = useState<"whiteBalance" | "colour" | "cloneSource">();
  const [failure, setFailure] = useState<string>();
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState<string>();
  const [skipped, setSkipped] = useState<Map<string, string>>(new Map());
  // A tiny copy that drives the look thumbnails, so eight previews cost eight small loops rather
  // than eight full composites every time a slider moves.
  const [thumbBase, setThumbBase] = useState<ImageData>();
  const [renamingId, setRenamingId] = useState<string>();
  const [exportFormat, setExportFormat] = useState<"png" | "jpeg" | "webp" | "tiff">("png");
  const [exportQuality, setExportQuality] = useState(92);
  const [exportMatte, setExportMatte] = useState("#ffffff");
  const [fileVersion, bumpFiles] = useReducer((value: number) => value + 1, 0);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  /**
   * VFX Phase 1: the GPU renderer behind a per-machine switch (see render/compositorFlag.ts). One
   * compositor object for the session, so the GPU context and its source textures live as long as
   * the workspace does, and are released with it.
   */
  const [gpuCompositor, setGpuCompositor] = useState(readCompositorFlag);
  const compositor = useRef<FlaggedCompositor>(undefined as unknown as FlaggedCompositor);
  compositor.current ??= new FlaggedCompositor();
  const frameTime = useRef<HTMLSpanElement>(null);
  useEffect(() => () => compositor.current.dispose(), []);

  /* VFX Phase 3: time. `compFrame` is the composition frame on screen. */
  const [compFrame, setCompFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [useProxy, setUseProxy] = useState(false);
  const [mediaVersion, bumpMedia] = useReducer((value: number) => value + 1, 0);
  const [elements, setElements] = useState<ElementEntry[]>([]);
  // VFX Phase 4: which tracker "Track a point" uses, and what the motion runtime has installed.
  const [trackEngine, setTrackEngine] = useState<TrackEngine>("lucas-kanade");
  const [motion, setMotion] = useState<MotionStatus>();
  /**
   * Video and element frames, decoded ahead of each render and kept by bytes (LRU). An evicted frame's
   * GPU texture is released with it, so scrubbing a long clip does not pile textures up.
   */
  const frameStore = useRef<MediaFrameStore>(undefined as unknown as MediaFrameStore);
  frameStore.current ??= (() => {
    const store = new MediaFrameStore({
      readBytes: readMediaBytes,
      loadImage: async (path) => {
        const image = await loadReadableImage(path);
        const width = image.naturalWidth || image.width, height = image.naturalHeight || image.height;
        const canvas = canvasOf(width, height);
        canvas.getContext("2d", { willReadFrequently: true })!.drawImage(image, 0, 0);
        return { canvas, width, height };
      },
    });
    store.onEvict = (picture) => compositor.current.releaseSource(picture.canvas);
    return store;
  })();
  useEffect(() => () => frameStore.current.dispose(), []);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const workBuffer = useRef<HTMLCanvasElement | undefined>(undefined);
  const overlayWork = useRef<HTMLCanvasElement | undefined>(undefined);
  const frame = useRef<number | undefined>(undefined);
  /** Every picture the document points at -- layer sources, SAM mattes, depth maps, cached fills. */
  const files = useRef(new Map<string, Loaded>());
  const pending = useRef(new Set<string>());
  const maskCache = useRef(new Map<string, Uint8Array>());
  // The strokes the mask had when this gesture started. Rebuilding from that snapshot every flush
  // means a frame that lands before React has committed the previous one cannot eat a stroke, which
  // slicing the live list could.
  const stroke = useRef<{
    maskId: string; entryId?: string; points: Array<[number, number]>; baseStrokes: PhotoMask["strokes"];
  } | undefined>(undefined);
  const retouch = useRef<{
    nodeId: string; points: Array<[number, number]>; offset: [number, number];
    baseStrokes: RetouchNode["strokes"];
  } | undefined>(undefined);
  const warp = useRef<{
    nodeId: string; points: Array<[number, number]>; baseStrokes: WarpNode["strokes"];
  } | undefined>(undefined);
  const strokeFrame = useRef<number | undefined>(undefined);
  /**
   * Where the pointer is, and everything the overlay needs to draw itself.
   *
   * Deliberately refs rather than state: the brush ring has to follow the pointer, and putting the
   * pointer in state would re-render this whole workspace -- both sidebars, every card -- sixty
   * times a second while someone paints. The overlay is its own canvas, so it can be redrawn on its
   * own without touching React at all.
   */
  const hoverAt = useRef<[number, number] | undefined>(undefined);
  const coverage = useRef<Uint8Array | undefined>(undefined);
  const overlayFrame = useRef<number | undefined>(undefined);
  const overlayInputs = useRef<{
    geometry: PhotoGeometry; width: number; height: number;
    brushRadius?: number; retouchRadius?: number; warpRadius?: number;
    retouchSource?: [number, number]; hardness: number;
  } | undefined>(undefined);
  const lightDrag = useRef<string | undefined>(undefined);
  const layerDrag = useRef<{ id: string; from: [number, number]; at: [number, number] } | undefined>(undefined);
  /**
   * VFX Phase 6: the pen's drawing -- which layer and subpath clicks append to, and the anchor whose
   * out-handle a drag is pulling (with where the drag began). `penDrawing` mirrors it for the overlay.
   */
  const pen = useRef<{ layerId: string; subpath: number; dragging?: number; from?: [number, number] } | undefined>(undefined);
  const [penDrawing, setPenDrawing] = useState<{ layerId: string; subpath: number }>();
  const [touchedAnchor, setTouchedAnchor] = useState<[number, number]>();
  const [envelopeEditing, setEnvelopeEditing] = useState(false);
  const [envelopeNodeEditing, setEnvelopeNodeEditing] = useState(false);

  /* ------------------------------------------------------------------ scope */

  const layers = useMemo(() => document ? layersOf(document) : [], [document]);
  /**
   * The geometry the picture is CURRENTLY DRAWN WITH, which is not always the document's.
   *
   * While the crop tool is up the frame is shown uncropped, so every mapping between the screen and
   * the canvas -- the brush, the handles, the overlay -- has to use the same exception the renderer
   * does. One value they all read is the only way that stays true.
   */
  const shownGeometry = useMemo(() => document
    ? geometryOf(tool === "crop" ? { ...document, crop: undefined } : document)
    : NEUTRAL_GEOMETRY, [document, tool]);
  /**
   * What the node stack, the masks and the brush are pointed at.
   *
   * Either one layer or the finished frame. Everything below reads `scopeDocument`, a view of the
   * document with that scope's nodes and masks in the places the panels already expect them, and
   * writes through `patchScope`, which puts them back where they came from. One indirection here is
   * what let every existing panel keep working unchanged when layers arrived.
   */
  const activeLayer: PhotoLayer | undefined = scopeId === COMPOSITE_SCOPE
    ? undefined
    : layers.find((layer) => layer.id === scopeId) ?? layers[layers.length - 1];
  const scopeKey = activeLayer?.id ?? COMPOSITE_SCOPE;

  const scopeDocument: PhotoDocument | undefined = document && (activeLayer
    ? { ...document, nodes: activeLayer.nodes, masks: activeLayer.masks }
    : document);

  const patchScope = useCallback((next: PhotoDocument, options?: { coalesce?: boolean }) => {
    if (!document) return;
    if (!activeLayer) { patch(next, options); return; }
    // Restore the document's own stack before materialising layers, so an old single-source document
    // migrates with its nodes intact and only the scope's copy is replaced.
    const base = withLayers({ ...next, nodes: document.nodes, masks: document.masks });
    patch({
      ...base,
      layers: layersOf(base).map((layer) => {
        if (layer.id !== activeLayer.id) return layer;
        const updated: PhotoLayer = { ...layer, nodes: next.nodes, masks: next.masks };
        // Deleting the mask a layer is cut by leaves the whole layer showing, not a dangling id.
        if (updated.maskId && !next.masks.some((mask) => mask.id === updated.maskId)) delete updated.maskId;
        return updated;
      }),
    }, options);
  }, [document, activeLayer, patch]);

  const patchLayer = useCallback((id: string, over: Partial<PhotoLayer>, options?: { coalesce?: boolean }) => {
    if (!document) return;
    const base = withLayers(document);
    patch({
      ...base,
      layers: layersOf(base).map((layer) => layer.id === id ? { ...layer, ...over } : layer),
    }, options);
  }, [document, patch]);

  const setLayers = useCallback((next: PhotoLayer[]) => {
    if (!document) return;
    patch({ ...withLayers(document), layers: next }, { coalesce: false });
  }, [document, patch]);

  const selectedNode = scopeDocument?.nodes.find((node) => node.id === selectedNodeId)
    ?? scopeDocument?.nodes[scopeDocument.nodes.length - 1];
  const activeMask = document ? maskById(document, activeMaskId) : undefined;
  const activeEntry = activeMask?.stack?.find((entry) => entry.id === activeEntryId);
  const dirty = document ? !isUntouched(document) : false;
  const animated = document ? isAnimated(document) : false;
  /**
   * Time is a Clip Editor thing (VFX Phase 4b). In Photo Edit a picture is one frame: a document that
   * still carries video layers from before the move shows its frame 0, and matrix rain is a still.
   */
  const timed = onClip && animated;
  const rate = document ? compositionRate(document) : 24;
  const duration = document ? compositionDuration(document) : 1;
  const shownFrame = timed ? Math.max(0, Math.min(compFrame, duration - 1)) : 0;

  useEffect(() => { void runtimeStatus().then(setRuntime).catch(() => undefined); }, [activeWorkspace]);
  useEffect(() => { void availableFonts().then(setFonts).catch(() => setFonts([])); }, []);
  useEffect(() => {
    setScopeId(undefined); setSelectedNodeId(undefined); setActiveMaskId(undefined);
    setTool("view"); setPoints([]); setRetouchSource(undefined);
  }, [asset?.id]);

  /* ---------------------------------------------------------------- loading */

  /** Every file the document names, decoded once and kept by path. */
  useEffect(() => {
    if (!live || !document) return;
    const wanted = filesWantedAt(document, shownFrame);

    let cancelled = false;
    const missing = [...wanted].filter((path) => !files.current.has(path) && !pending.current.has(path));
    // Only a missing LAYER blanks the viewport. A depth map or a cached fill arriving later is a
    // node that reports itself unrendered for a moment, not a reason to hide the picture.
    const layerPaths = new Set(layersOf(document).map((layer) => layer.sourcePath).filter(Boolean));
    const blocking = () => [...pending.current].some((path) => layerPaths.has(path));
    if (missing.some((path) => layerPaths.has(path))) setLoading(true);
    for (const path of missing) {
      pending.current.add(path);
      void loadReadableImage(path).then((image) => {
        const width = image.naturalWidth || image.width;
        const height = image.naturalHeight || image.height;
        const canvas = canvasOf(width, height);
        canvas.getContext("2d", { willReadFrequently: true })!.drawImage(image, 0, 0);
        files.current.set(path, { canvas, width, height });
        maskCache.current.clear();
      }).catch((error) => {
        // A missing derived file is not fatal: the node reports itself unrendered and the user can
        // run it again. A missing layer source is, so it says which one.
        if (layersOf(document).some((layer) => layer.sourcePath === path)) {
          setFailure(`Could not open ${filename(path)} for editing. ${errorText(error)}`);
        }
      }).finally(() => {
        pending.current.delete(path);
        if (!cancelled) { setLoading(blocking()); bumpFiles(); }
      });
    }
    return () => { cancelled = true; };
    // Deliberately not depending on `fileVersion`: this effect bumps it when a load finishes, and
    // depending on it would restart the effect that caused the bump. The render path picks the new
    // file up instead, because `sources` does depend on it.
  }, [document, live, shownFrame]);

  /** Pictures by path, plus the media frames the store holds (by `mediaFrameKey`), as of now. */
  const sourcesNow = useCallback(() => {
    const map = new Map<string, LayerSource>();
    for (const [path, held] of files.current) {
      map.set(path, { image: held.canvas, width: held.width, height: held.height });
    }
    for (const [key, picture] of frameStore.current.entries()) {
      map.set(key, { image: picture.canvas, width: picture.width, height: picture.height });
    }
    return map;
  }, []);
  const sources = useMemo(() => sourcesNow(), [fileVersion, loading, mediaVersion, sourcesNow]);

  /**
   * VFX Phase 3: fetch the media frames the current frame needs (and a few after it, more while
   * playing) into the frame store. The current frame's keys are pinned against eviction.
   */
  useEffect(() => {
    if (!live || !document || !animated) return;
    const store = frameStore.current;
    const ahead = !timed ? 0 : playing ? 8 : 2;
    const wanted: Array<{ media: LayerMedia; index: number; now: boolean }> = [];
    for (let at = shownFrame; at <= Math.min(duration - 1, shownFrame + ahead); at++) {
      // Media layers' frames and (VFX Phase 4) every living emitter particle's element frame.
      for (const need of mediaNeedsAt(document, at, rate)) wanted.push({ ...need, now: at === shownFrame });
    }
    store.pin(wanted.filter((item) => item.now).map((item) => mediaFrameKey(item.media, item.index, useProxy)));
    let cancelled = false;
    for (const item of wanted) {
      if (store.get(mediaFrameKey(item.media, item.index, useProxy))) continue;
      store.load(item.media, item.index, useProxy).then(() => {
        if (!cancelled && item.now) bumpMedia();
      }, (error) => {
        if (!cancelled && item.now) setStatus(`Could not read frame ${item.index} of ${filename(item.media.path)}: ${errorText(error)}`);
      });
    }
    return () => { cancelled = true; };
  }, [document, shownFrame, useProxy, playing, live, animated, timed, duration, rate]);

  /** Media frames the current frame still waits for (the time bar shows the count). */
  const waitingFrames = useMemo(() => document && animated
    ? mediaKeysAt(document, shownFrame, useProxy).filter((key) => !frameStore.current.get(key)).length : 0,
  [document, animated, shownFrame, useProxy, mediaVersion]);

  /**
   * Playback at the composition rate that only ever advances to a frame whose media is loaded: it
   * slows down rather than showing a layer missing.
   */
  useEffect(() => {
    if (!playing || !document || !timed) return;
    let handle = 0;
    let last = performance.now();
    const tick = (now: number) => {
      handle = requestAnimationFrame(tick);
      if (now - last < 1000 / rate) return;
      const next = shownFrame + 1 >= duration ? 0 : shownFrame + 1;
      if (!mediaKeysAt(document, next, useProxy).every((key) => frameStore.current.get(key))) return;
      last = now;
      setCompFrame(next);
    };
    handle = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(handle);
  }, [playing, document, timed, shownFrame, duration, rate, useProxy]);

  /** Load the full-quality frames of the current frame (saving never renders from a proxy). */
  const ensureMasterFrames = async () => {
    if (!document || !animated) return;
    await Promise.all(mediaNeedsAt(document, shownFrame, rate).map((need) => frameStore.current.load(need.media, need.index, false)));
  };

  const canvasSize = useMemo(
    () => document ? canvasSizeOf(document, sources) : undefined,
    [document, sources],
  );
  const previewSize = useMemo(() => {
    if (!canvasSize) return undefined;
    const scale = Math.min(1, PREVIEW_LIMIT / Math.max(canvasSize.width, canvasSize.height));
    return {
      width: Math.max(1, Math.round(canvasSize.width * scale)),
      height: Math.max(1, Math.round(canvasSize.height * scale)),
    };
  }, [canvasSize]);

  /** A small copy of the scope's own picture, so the filter previews cost almost nothing. */
  useEffect(() => {
    const held = activeLayer ? files.current.get(activeLayer.sourcePath) : undefined;
    const source = held ?? (layers[0] ? files.current.get(layers[0].sourcePath) : undefined);
    if (!source) { setThumbBase(undefined); return; }
    const thumb = canvasOf(72, 46);
    const context = thumb.getContext("2d", { willReadFrequently: true })!;
    context.drawImage(source.canvas, 0, 0, 72, 46);
    setThumbBase(context.getImageData(0, 0, 72, 46));
  }, [activeLayer?.sourcePath, layers, fileVersion]);

  /* -------------------------------------------------------------- resources */

  /**
   * Rasterised masks, depth and fills for one scope, all at one working resolution.
   *
   * `pixels` must be that scope's UNGRADED input at that resolution, because a colour-range mask
   * matches against it. Handing it the preview buffer while rendering at full size would silently
   * produce an empty mask -- the selection would work on screen and vanish from the saved file,
   * which is the exact class of preview/output divergence this design exists to avoid.
   *
   * A derived file belonging to a layer is PLACED through that layer's matrix rather than stretched
   * to the canvas: a matte and a fill were both produced in the layer's own pixels, so anything else
   * lands them correctly on an untransformed plate and slides them off a scaled cutout.
   */
  const buildResources = useCallback((
    scope: ComposeScope, width: number, height: number, pixels?: Uint8ClampedArray,
  ): RenderResources => document
    ? buildScopeResources({ document, files: (path) => files.current.get(path), cache: maskCache.current },
      scope, width, height, pixels)
    : emptyResources(), [document, fileVersion]);

  /** Compose the document at a given size. One function for the preview and for what gets saved. */
  const compose = useCallback((
    width: number, height: number, source: PhotoDocument, capture?: string, captureMask?: string,
    /** A save or a model input: full-quality frames, fetched fresh from the store. */
    full = false,
  ) => compositor.current.compose({
    document: source, width, height, sources: full ? sourcesNow() : sources, capture, captureMask,
    frame: shownFrame, useProxy: full ? false : useProxy,
    resourcesFor: (scope, pixels) => buildResources(scope, width, height, pixels),
  }, gpuCompositor), [sources, sourcesNow, buildResources, gpuCompositor, shownFrame, useProxy]);

  /* ---------------------------------------------------------------- preview */

  const paint = useCallback(() => {
    const target = canvasRef.current;
    const size = previewSize;
    if (!target || !size || !document || !canvasSize) return;
    if (!workBuffer.current || workBuffer.current.width !== size.width || workBuffer.current.height !== size.height) {
      workBuffer.current = canvasOf(size.width, size.height);
      overlayWork.current = canvasOf(size.width, size.height);
    }
    const work = workBuffer.current!;
    // With the crop tool up the picture is shown UNCROPPED. A crop drawn over its own result can
    // only ever shrink, because everything it would grow back into has already been cut off screen.
    const framed = tool === "crop" ? { ...document, crop: undefined } : document;
    const shown = comparing ? bypassed(framed) : framed;
    const overlayWanted = !comparing && showMask && Boolean(activeMaskId);
    // The overlay tints the mask exactly as this render combined it (VFX Phase 2): layer entries and
    // placed files included, which only the compositor can resolve.
    const result = compose(size.width, size.height, shown, undefined, overlayWanted ? activeMaskId : undefined);
    // Written straight to the element rather than through state: this runs on every frame of a
    // slider drag, and a re-render per frame to show a number would cost more than it measures.
    if (frameTime.current) {
      const engine = compositor.current.lastEngine.toUpperCase();
      frameTime.current.textContent = `${engine} ${compositor.current.lastMilliseconds.toFixed(0)} ms`;
      frameTime.current.title = compositor.current.lastError
        ? `The GPU renderer failed and the CPU compositor drew this frame: ${compositor.current.lastError}`
        : "Time to compose the preview. Click to switch between the CPU compositor and the GPU renderer (VFX Phase 1).";
    }

    // Off the finished composite, so it reflects everything rather than the file on disk.
    setLevels(histogram(result.pixels, 64));
    work.getContext("2d")!.putImageData(new ImageData(result.pixels, size.width, size.height), 0, 0);
    drawOriented(target, work, shownGeometry);

    coverage.current = overlayWanted ? result.maskCoverage : undefined;
    drawOverlay();

    const next = new Map(result.report.skipped.map((entry) => [entry.nodeId, entry.reason]));
    setSkipped((current) => sameKeys(current, next) ? current : next);
  }, [document, comparing, compose, showMask, activeMaskId, previewSize, canvasSize,
      scopeKey, scopeDocument, activeLayer, buildResources, tool, shownGeometry]);

  useEffect(() => {
    if (loading) return;
    if (frame.current !== undefined) cancelAnimationFrame(frame.current);
    frame.current = requestAnimationFrame(() => { frame.current = undefined; paint(); });
    return () => { if (frame.current !== undefined) cancelAnimationFrame(frame.current); };
  }, [paint, loading]);

  /* ---------------------------------------------------------------- overlay */

  /**
   * The masked area, the brush and where the clone is sampling from.
   *
   * Its own canvas and its own paint, split off the composite for one reason: you cannot paint into
   * something you cannot see, and until now the brush had no size on screen -- you found the radius
   * by making a mark and undoing it. A ring that follows the pointer has to redraw at pointer rate,
   * and recomposing every layer at pointer rate is not affordable, so the two were separated.
   *
   * Everything is drawn in canvas coordinates and then put through the document's geometry, the same
   * transform the picture goes through, so the ring stays the right size and in the right place on a
   * rotated, straightened or cropped canvas.
   */
  const drawOverlay = useCallback(() => {
    const target = overlayRef.current;
    const work = overlayWork.current;
    const state = overlayInputs.current;
    if (!target || !work || !state) { if (target) target.hidden = true; return; }
    const { width, height } = state;
    if (work.width !== width || work.height !== height) return;
    const context = work.getContext("2d")!;
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, width, height);
    let drew = false;

    const tint = coverage.current;
    if (tint && tint.length === width * height) {
      const image = context.createImageData(width, height);
      for (let pixel = 0, at = 0; pixel < tint.length; pixel++, at += 4) {
        const amount = tint[pixel];
        if (amount === 0) continue;
        image.data[at] = 255;
        image.data[at + 1] = 51;
        image.data[at + 2] = 133;
        image.data[at + 3] = amount * 0.42;
      }
      context.putImageData(image, 0, 0);
      drew = true;
    }

    const hover = hoverAt.current;
    const long = Math.max(width, height);
    const line = Math.max(1, long / 600);
    const radius = state.brushRadius ?? state.retouchRadius ?? state.warpRadius;
    if (radius !== undefined && hover) {
      const pixels = Math.max(1.5, radius * long);
      ring(context, hover[0] * width, hover[1] * height, pixels, line);
      // The solid core, so the difference between a hard and a soft brush is visible before the
      // first stroke rather than after undoing one.
      if (state.hardness < 0.95) {
        ring(context, hover[0] * width, hover[1] * height,
          Math.max(1, pixels * state.hardness), line * 0.7, true);
      }
      drew = true;
    }
    if (state.retouchRadius !== undefined) {
      // While a stroke is running the offset is frozen, so the source ring travels with the brush --
      // which is the whole point of a clone stamp and the only way to see what you are about to copy.
      const offset = retouch.current?.offset
        ?? (state.retouchSource && hover
          ? [state.retouchSource[0] - hover[0], state.retouchSource[1] - hover[1]] as [number, number]
          : undefined);
      const source = offset && hover
        ? [hover[0] + offset[0], hover[1] + offset[1]] as [number, number]
        : state.retouchSource;
      if (source) {
        ring(context, source[0] * width, source[1] * height,
          Math.max(1.5, state.retouchRadius * long), line, true);
        crosshair(context, source[0] * width, source[1] * height, long / 60, line);
        drew = true;
      }
    }

    if (!drew) { target.hidden = true; return; }
    drawOriented(target, work, state.geometry);
    target.hidden = false;
  }, []);

  const scheduleOverlay = useCallback(() => {
    if (overlayFrame.current !== undefined) return;
    overlayFrame.current = requestAnimationFrame(() => {
      overlayFrame.current = undefined;
      drawOverlay();
    });
  }, [drawOverlay]);

  // Refreshed every render, read by a painter that runs outside React. Anything the overlay draws
  // has to arrive this way, because the painter never re-reads props.
  useEffect(() => {
    if (!document || !previewSize) { overlayInputs.current = undefined; return; }
    overlayInputs.current = {
      geometry: shownGeometry,
      width: previewSize.width, height: previewSize.height,
      brushRadius: tool === "brush" && activeMask ? brush.radius : undefined,
      retouchRadius: tool === "retouch" && selectedNode?.kind === "retouch"
        ? retouchBrush.radius : undefined,
      warpRadius: tool === "warp" && selectedNode?.kind === "warp" ? warpBrush.radius : undefined,
      retouchSource,
      // Liquify has no hard core, so its ring shows no inner circle rather than a misleading one.
      hardness: tool === "warp" ? 1 : tool === "retouch" ? retouchBrush.hardness : brush.hardness,
    };
    drawOverlay();
  }, [document, previewSize, tool, activeMask, brush.radius, brush.hardness, selectedNode?.kind,
      retouchBrush.radius, retouchBrush.hardness, retouchSource, warpBrush.radius, shownGeometry,
      drawOverlay]);

  /* ------------------------------------------------------------ interaction */

  /** Pointer position on the shown canvas, as coordinates normalised to the unrotated canvas. */
  const toCanvas = (event: ReactPointerEvent<Element>): [number, number] | undefined => {
    const canvas = canvasRef.current;
    if (!canvas || !document || !previewSize) return undefined;
    const box = canvas.getBoundingClientRect();
    return displayToSource(
      (event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height,
      previewSize.width, previewSize.height, shownGeometry,
    );
  };

  /**
   * Write the in-progress mask stroke into the document.
   *
   * One pointer gesture is one stroke: the growing line is rebuilt on top of the strokes the mask
   * had when the drag began, so it stays a single item and "undo stroke" removes the whole drag
   * rather than the last few samples of it.
   */
  const flushStroke = () => {
    const active = stroke.current;
    if (!active || !scopeDocument || active.points.length === 0) return;
    const mask = scopeDocument.masks.find((item) => item.id === active.maskId);
    if (!mask) return;
    const strokes = [...active.baseStrokes, {
      points: active.points.slice(),
      radius: brush.radius, hardness: brush.hardness, erase: brush.erase,
    }];
    // Into the selected stack entry when there is one, otherwise the base as always.
    const written: PhotoMask = active.entryId
      ? { ...mask, stack: (mask.stack ?? []).map((entry) => entry.id === active.entryId ? { ...entry, strokes } : entry) }
      : { ...mask, strokes };
    patchScope({
      ...scopeDocument,
      masks: scopeDocument.masks.map((item) => item.id === mask.id ? written : item),
    });
  };

  /** The same idea for the clone and heal brush: one gesture, one re-editable stroke. */
  const flushRetouch = () => {
    const active = retouch.current;
    if (!active || !scopeDocument || active.points.length === 0) return;
    const strokes = [...active.baseStrokes, {
      points: active.points.slice(),
      radius: retouchBrush.radius, hardness: retouchBrush.hardness, heal: retouchBrush.heal,
      offsetX: active.offset[0], offsetY: active.offset[1],
    }];
    patchScope({
      ...scopeDocument,
      nodes: scopeDocument.nodes.map((item) => item.id === active.nodeId && item.kind === "retouch"
        ? { ...item, strokes } : item),
    });
  };

  /** A stored point, as a fraction of the shown canvas. The inverse of `toCanvas`. */
  const atDisplay = (u: number, v: number): Point => previewSize
    ? sourceToDisplay(u, v, previewSize.width, previewSize.height, shownGeometry) as Point
    : [u, v];

  /** The same point in canvas PIXELS, which is the space angles and distances have to be measured in. */
  const inPixels = (at: Point): Point => canvasSize
    ? [at[0] * canvasSize.width, at[1] * canvasSize.height] : at;

  /** Change the active mask's generated shape. Coalesced, so one drag is one undo step. */
  const patchMaskShape = (shape: MaskShape) => scopeDocument && activeMask && patchScope({
    ...scopeDocument,
    masks: scopeDocument.masks.map((mask) => mask.id !== activeMask.id ? mask
      // The selected stack entry's shape when it has one (VFX Phase 2), otherwise the base's.
      : activeEntry ? { ...mask, stack: (mask.stack ?? []).map((entry) => entry.id === activeEntry.id
        ? { ...entry, source: { kind: "shape" as const, shape } } : entry) }
        : { ...mask, shape }),
  });
  /** The shape the on-canvas handles drag: the selected entry's, or the base's when none is selected. */
  const handleShape: MaskShape | undefined = activeEntry
    ? (activeEntry.source.kind === "shape" ? activeEntry.source.shape : undefined)
    : activeMask?.shape;

  /**
   * The four corners of a placed layer, where they appear on screen.
   *
   * Straight through the layer's own matrix and then the document's, rather than an approximation
   * from x/y/scale: a rotated layer's box has to sit on the rotated layer, and reconstructing that
   * by hand is how a handle ends up half a picture away from the thing it moves.
   */
  const layerFrame = (): Point[] | undefined => {
    if (!activeLayer || !canvasSize) return undefined;
    const source = isDrawnLayer(activeLayer) ? drawnSource(activeLayer) : sources.get(activeLayer.sourcePath);
    if (!source) return undefined;
    const rect = croppedRect(activeLayer, source);
    const matrix = layerMatrix(activeLayer, canvasSize.width, canvasSize.height, source);
    return ([[0, 0], [rect.width, 0], [rect.width, rect.height], [0, rect.height]] as Point[])
      .map(([x, y]) => {
        const [px, py] = applyMatrix(matrix, x, y);
        return atDisplay(px / canvasSize.width, py / canvasSize.height);
      });
  };

  const framePinch = useRef<{ scale: number; rotation: number; grab: number; angle: number } | undefined>(undefined);

  const beginFrame = (event: ReactPointerEvent<Element>) => {
    const at = toCanvas(event);
    if (!at || !activeLayer || !canvasSize) return;
    const centre = inPixels([activeLayer.x, activeLayer.y]);
    const [px, py] = inPixels(at);
    framePinch.current = {
      scale: activeLayer.scale, rotation: activeLayer.rotation,
      grab: Math.max(1, Math.hypot(px - centre[0], py - centre[1])),
      angle: Math.atan2(py - centre[1], px - centre[0]),
    };
  };

  const dragCorner = (event: ReactPointerEvent<Element>) => {
    const start = framePinch.current;
    const at = toCanvas(event);
    if (!start || !at || !activeLayer || activeLayer.locked) return;
    const centre = inPixels([activeLayer.x, activeLayer.y]);
    const [px, py] = inPixels(at);
    // Uniform and about the layer's own centre, because the centre is what x and y mean. Scaling
    // from the opposite corner instead would move the layer as a side effect of resizing it.
    const distance = Math.hypot(px - centre[0], py - centre[1]);
    patchLayer(activeLayer.id, { scale: Math.max(0.02, start.scale * distance / start.grab) });
  };

  const dragRotate = (event: ReactPointerEvent<Element>) => {
    const start = framePinch.current;
    const at = toCanvas(event);
    if (!start || !at || !activeLayer || activeLayer.locked) return;
    const centre = inPixels([activeLayer.x, activeLayer.y]);
    const [px, py] = inPixels(at);
    const turned = (Math.atan2(py - centre[1], px - centre[0]) - start.angle) * 180 / Math.PI;
    const next = start.rotation + turned;
    // Shift snaps to fifteen degrees, which is what makes a horizon or an upright edge reachable.
    patchLayer(activeLayer.id, {
      rotation: event.shiftKey ? Math.round(next / 15) * 15 : Math.round(next * 10) / 10,
    });
  };

  /** And again for liquify. Three gestures, three snapshots, one shape of code. */
  const flushWarp = () => {
    const active = warp.current;
    if (!active || !scopeDocument || active.points.length === 0) return;
    const strokes = [...active.baseStrokes, {
      points: active.points.slice(),
      radius: warpBrush.radius, strength: warpBrush.strength, mode: warpBrush.mode,
    }];
    patchScope({
      ...scopeDocument,
      nodes: scopeDocument.nodes.map((item) => item.id === active.nodeId && item.kind === "warp"
        ? { ...item, strokes } : item),
    });
  };

  const onCanvasDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!document || !scopeDocument) return;
    const at = toCanvas(event);
    if (!at) return;

    if (tool === "picker") {
      if (picker === "cloneSource") {
        setRetouchSource(at);
        setPicker(undefined);
        setTool("retouch");
        setStatus("Source point set — paint where it should be copied to");
        return;
      }
      // Sampled from the scope's UNGRADED pixels: white balance is a correction of what the camera
      // recorded, so solving it from something already graded would chase its own tail.
      const size = previewSize;
      if (!size) return;
      const raw = compose(size.width, size.height, document, scopeKey).raw.get(scopeKey);
      if (!raw) return;
      const x = Math.min(size.width - 1, Math.max(0, Math.round(at[0] * size.width)));
      const y = Math.min(size.height - 1, Math.max(0, Math.round(at[1] * size.height)));
      const offset = (y * size.width + x) * 4;
      const rgb: [number, number, number] = [raw[offset], raw[offset + 1], raw[offset + 2]];
      if (picker === "whiteBalance" && selectedNode?.kind === "grade") {
        const balance = whiteBalanceFor(rgb[0], rgb[1], rgb[2]);
        patchScope({
          ...scopeDocument,
          nodes: scopeDocument.nodes.map((item) => item.id === selectedNode.id && item.kind === "grade"
            ? { ...item, adjustments: { ...item.adjustments, ...balance } } : item),
        }, { coalesce: false });
        setStatus(`White balance set from a sample · ${signed(balance.temperature)} / ${signed(balance.tint)}`);
      } else if (picker === "colour" && activeMask) {
        const sampled: MaskShape = { kind: "colour", r: rgb[0], g: rgb[1], b: rgb[2], tolerance: 18, softness: 0.6 };
        const entryId = activeEntry?.source.kind === "shape" ? activeEntry.id : undefined;
        patchScope({
          ...scopeDocument,
          masks: scopeDocument.masks.map((mask) => mask.id !== activeMask.id ? mask
            : entryId ? { ...mask, stack: (mask.stack ?? []).map((entry) => entry.id === entryId
              ? { ...entry, source: { kind: "shape" as const, shape: sampled } } : entry) }
              : { ...mask, shape: sampled }),
        }, { coalesce: false });
        setStatus(`Colour range sampled · rgb(${rgb.join(", ")})`);
      }
      setPicker(undefined);
      setTool(activeMask ? "brush" : "view");
      return;
    }

    if (tool === "pen" && canvasSize) {
      // Appending to the path being drawn; shift starts another subpath in the active path layer;
      // otherwise a click starts a new path layer at that point.
      const drawing = pen.current;
      const target = drawing ? layers.find((layer) => layer.id === drawing.layerId) : undefined;
      if (drawing && target?.path && !target.locked) {
        const next = appendAnchor(target.path, drawing.subpath, pathMapping(target, target.path, canvasSize).toLocal(at));
        patchLayer(target.id, { path: next }, { coalesce: false });
        pen.current = { ...drawing, dragging: next.subpaths[drawing.subpath].nodes.length - 1, from: at };
        setTouchedAnchor([drawing.subpath, next.subpaths[drawing.subpath].nodes.length - 1]);
      } else if (event.shiftKey && activeLayer?.path && !activeLayer.locked) {
        const subpath = activeLayer.path.subpaths.length;
        patchLayer(activeLayer.id, { path: appendAnchor(activeLayer.path, subpath, pathMapping(activeLayer, activeLayer.path, canvasSize).toLocal(at)) }, { coalesce: false });
        pen.current = { layerId: activeLayer.id, subpath, dragging: 0, from: at };
        setPenDrawing({ layerId: activeLayer.id, subpath });
      } else {
        const layer = createPathLayer({ subpaths: [{ closed: false, nodes: [{ x: 0, y: 0 }] }] }, at[0], at[1],
          `Path ${layers.filter((item) => item.path).length + 1}`);
        setLayers([...layers, layer]);
        setScopeId(layer.id);
        setOpenCards((current) => ({ ...current, path: true }));
        pen.current = { layerId: layer.id, subpath: 0, dragging: 0, from: at };
        setPenDrawing({ layerId: layer.id, subpath: 0 });
        setStatus("Drawing a path — click for corners, drag for curves, click the first point to close, Esc to finish");
      }
      event.currentTarget.setPointerCapture(event.pointerId);
      return;
    }

    if (tool === "move" && activeLayer && !activeLayer.locked) {
      event.currentTarget.setPointerCapture(event.pointerId);
      layerDrag.current = { id: activeLayer.id, from: [activeLayer.x, activeLayer.y], at };
      return;
    }

    if (tool === "select") {
      setPoints((current) => [...current, { x: at[0], y: at[1], include: !event.shiftKey }]);
      return;
    }

    if (tool === "track") {
      setTool("view");
      runTrack(at);
      return;
    }

    if (tool === "retouch" && selectedNode?.kind === "retouch") {
      // Alt-click sets the source point, the way every clone tool in existence does.
      if (event.altKey) {
        setRetouchSource(at);
        setStatus("Source point set — paint where it should be copied to");
        return;
      }
      if (!retouchSource && !retouchBrush.heal) {
        setStatus("Set a source point first — alt-click where the clone should copy from");
        return;
      }
      event.currentTarget.setPointerCapture(event.pointerId);
      const offset: [number, number] = retouchSource
        ? [retouchSource[0] - at[0], retouchSource[1] - at[1]] : [0, 0];
      retouch.current = {
        nodeId: selectedNode.id, points: [at], offset, baseStrokes: selectedNode.strokes,
      };
      flushRetouch();
      return;
    }

    if (tool === "warp" && selectedNode?.kind === "warp") {
      event.currentTarget.setPointerCapture(event.pointerId);
      warp.current = { nodeId: selectedNode.id, points: [at], baseStrokes: selectedNode.strokes };
      return;
    }

    if (tool === "brush" && activeMask) {
      event.currentTarget.setPointerCapture(event.pointerId);
      stroke.current = activeEntry
        ? { maskId: activeMask.id, entryId: activeEntry.id, points: [at], baseStrokes: activeEntry.strokes }
        : { maskId: activeMask.id, points: [at], baseStrokes: activeMask.strokes };
      flushStroke();
    }
  };

  const onCanvasMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const at = toCanvas(event);
    if (!at) return;
    hoverAt.current = at;
    scheduleOverlay();
    if (pen.current?.dragging !== undefined && pen.current.from && canvasSize) {
      // Dragging after a click pulls the new anchor's handles out (a smooth point). The layer made by
      // this click may not have rendered yet: nothing happens until it has.
      const drawing = pen.current;
      const target = layers.find((layer) => layer.id === drawing.layerId);
      const moved = Math.hypot((at[0] - drawing.from![0]) * canvasSize.width, (at[1] - drawing.from![1]) * canvasSize.height);
      if (!target?.path || moved < 3 || !target.path.subpaths[drawing.subpath]?.nodes[drawing.dragging!]) return;
      const local = pathMapping(target, target.path, canvasSize).toLocal(at);
      patchLayer(target.id, { path: setHandle(target.path, drawing.subpath, drawing.dragging!, "out", local, "equal") });
      return;
    }
    if (layerDrag.current) {
      const drag = layerDrag.current;
      patchLayer(drag.id, {
        x: drag.from[0] + (at[0] - drag.at[0]),
        y: drag.from[1] + (at[1] - drag.at[1]),
      });
      return;
    }
    if (stroke.current) stroke.current.points.push(at);
    else if (retouch.current) retouch.current.points.push(at);
    else if (warp.current) warp.current.points.push(at);
    else return;
    // Painting patches the document, which recomposes everything, so it is held to one frame rather
    // than one pointer event -- otherwise a fast drag queues dozens of full composites.
    if (strokeFrame.current === undefined) {
      strokeFrame.current = requestAnimationFrame(() => {
        strokeFrame.current = undefined;
        flushStroke();
        flushRetouch();
        flushWarp();
      });
    }
  };

  const onCanvasUp = () => {
    if (strokeFrame.current !== undefined) {
      cancelAnimationFrame(strokeFrame.current);
      strokeFrame.current = undefined;
    }
    flushStroke();
    flushRetouch();
    flushWarp();
    stroke.current = undefined;
    retouch.current = undefined;
    warp.current = undefined;
    layerDrag.current = undefined;
    if (pen.current) pen.current = { ...pen.current, dragging: undefined, from: undefined };
  };

  const finishPen = useCallback(() => {
    pen.current = undefined;
    setPenDrawing(undefined);
  }, []);
  // Leaving the pen finishes the path being drawn.
  useEffect(() => { if (tool !== "pen") finishPen(); }, [tool, finishPen]);
  // With the pen: Esc or Enter finishes the path, Delete removes the last point touched.
  useEffect(() => {
    if (tool !== "pen") return;
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable)) return;
      if (event.key === "Escape" || event.key === "Enter") {
        finishPen();
        setStatus("Path finished");
      } else if ((event.key === "Delete" || event.key === "Backspace") && touchedAnchor && activeLayer?.path && !activeLayer.locked) {
        event.preventDefault();
        patchLayer(activeLayer.id, { path: withoutAnchor(activeLayer.path, touchedAnchor) }, { coalesce: false });
        setTouchedAnchor(undefined);
        finishPen();
        setStatus("Point deleted");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [tool, touchedAnchor, activeLayer, patchLayer, finishPen]);

  /* --------------------------------------------------------------- actions */

  const guard = async (label: string, work: () => Promise<void>) => {
    setBusy(true); setRunning(label); setFailure(undefined);
    try { await work(); } catch (error) {
      setFailure(errorText(error));
      setStatus(`${label} failed`);
    } finally { setBusy(false); setRunning(undefined); }
  };

  /** The picture a worker should be pointed at: the layer being edited, or the plate underneath. */
  // A video plate has no picture file (its sourcePath is ""), and the clip's MP4 must never reach them.
  const workingPath = activeLayer?.sourcePath ?? layers[0]?.sourcePath ?? (onClip ? undefined : asset?.path);

  const computeDepth = () => workingPath && document && void guard("Measuring depth", async () => {
    setStatus("Measuring depth…");
    const result = await photoDepthMap({ projectId: project.id, imagePath: workingPath });
    files.current.delete(result.outputPath);
    patch({ ...document, depthPath: result.outputPath }, { coalesce: false });
    bumpFiles();
    setStatus("Depth map ready");
  });

  /**
   * SAM 2 on a VIDEO layer (found 2026-09-28: Run did nothing there, because a video layer has no
   * picture file for the still-image selector). The clicks are mapped into the clip's own pixels and
   * sent to the cached SAM 2 video tracker Mask VFX uses, starting at the source frame on screen
   * (SAM 2 propagates forward from the clicked frame). The matte sequence joins the active mask as
   * an entry, so it follows the subject frame by frame, and it is a source for a centroid track.
   */
  const runVideoSelect = (layer: PhotoLayer, mask: PhotoMask) => void guard("Selecting", async () => {
    const media = layer.media;
    if (!media || !document || !scopeDocument || !canvasSize) return;
    const placed = attachedLayer(document, layer, shownFrame, canvasSize, rate);
    const shown = mediaFramesAt(media, shownFrame, rate);
    if (!shown) { setStatus(`${layer.name} is not on screen at frame ${shownFrame}`); return; }
    const query = shown[shown.length - 1].index;
    const clamp01 = (value: number) => Math.min(1, Math.max(0, value));
    const mapped = points.flatMap((point) => {
      const inSource = canvasToSourcePoint(placed, [point.x, point.y], canvasSize);
      return inSource ? [{ x: clamp01(inSource[0] / media.width), y: clamp01(inSource[1] / media.height), include: point.include }] : [];
    });
    if (!mapped.some((point) => point.include)) { setStatus("Add at least one include point on the video"); return; }
    const last = Math.min(media.frameCount, Math.floor(media.outPoint), query + 900);
    setStatus(`SAM 2 is tracking ${layer.name} from frame ${query} to ${last - 1} — the resource guard checks memory first`);
    // A quarter frame early, so float rounding can never make ffmpeg start one frame late.
    const tracked = await cacheMaskTrack({
      videoPath: media.path, startTime: Math.max(0, (query - 0.25) / media.fps), endTime: last / media.fps,
      model: "sam2", points: mapped,
    });
    const entry = createMaskEntry({
      kind: "sequence", cachePath: tracked.cachePath, frameCount: tracked.frameCount, fps: tracked.fps, first: query,
    });
    maskCache.current.clear();
    patchScope({
      ...scopeDocument,
      masks: scopeDocument.masks.map((item) => item.id === mask.id ? { ...item, stack: [...(item.stack ?? []), entry] } : item),
    }, { coalesce: false });
    setActiveEntryId(entry.id);
    setPoints([]);
    setTool("brush");
    setStatus(`SAM 2 matte ${tracked.reused ? "reused" : "tracked"} · frames ${query}–${query + tracked.frameCount - 1} — it follows the subject; Tracks can follow its centre`);
  });

  const runSelect = () => onClip && activeLayer?.media?.kind === "video" && activeMask
    ? runVideoSelect(activeLayer, activeMask)
    : workingPath && scopeDocument && activeMask && void guard("Selecting", async () => {
    setStatus("Asking SAM 2 for the subject…");
    // The clicks are in canvas coordinates; SAM sees the layer's own file, so they are mapped back
    // through the same matrix the layer is drawn with rather than sent as-is.
    const mapped = points.map((point) => {
      const inSource = toLayerPoint(point.x, point.y, activeLayer, sources, canvasSize);
      return { x: inSource[0], y: inSource[1], include: point.include };
    });
    const result = await photoSelect({ projectId: project.id, imagePath: workingPath, points: mapped });
    files.current.delete(result.outputPath);
    maskCache.current.clear();
    patchScope({
      ...scopeDocument,
      masks: scopeDocument.masks.map((mask) => mask.id === activeMask.id
        ? { ...mask, sourcePath: result.outputPath } : mask),
    }, { coalesce: false });
    setPoints([]);
    setTool("brush");
    bumpFiles();
    setStatus("Selection added to the mask");
  });

  /**
   * Send the masked area to the fill worker.
   *
   * The mask is painted in canvas coordinates but LaMa is handed the layer's own file, so the
   * coverage is mapped back through the inverse of the layer's placement. Doing it any other way
   * works perfectly on an untransformed plate and quietly misaligns on a scaled cutout, which is the
   * kind of bug that only shows up in the one case you did not test.
   */
  const runFill = (node: RemoveNode) => scopeDocument && canvasSize && void guard("Filling", async () => {
    const mask = maskById(scopeDocument, node.maskId);
    if (!mask) throw new Error("this node has no mask");
    // A fill is a model run against one file, so it belongs to a layer. On the composite there is no
    // single file to send, and guessing one would fill the wrong picture.
    if (!activeLayer) throw new Error("select a layer to fill — Remove works on one picture, not on the composite");
    if (activeLayer.text) throw new Error("Remove works on a picture; edit the words instead");
    const target = sources.get(activeLayer.sourcePath);
    if (!target) throw new Error("the layer's picture is still loading");
    setStatus("Rasterising the mask at full size…");
    // The mask exactly as the full-size render combines it, stack and layer entries included.
    const coverage = compose(canvasSize.width, canvasSize.height, document!, undefined, mask.id).maskCoverage;
    if (!coverage) throw new Error("the mask is empty");

    const canvasMask = canvasOf(canvasSize.width, canvasSize.height);
    const context = canvasMask.getContext("2d")!;
    const data = context.createImageData(canvasSize.width, canvasSize.height);
    for (let index = 0, at = 0; index < coverage.length; index++, at += 4) {
      // LaMa reads a white-on-black matte and binarises it, so the coverage goes into all three
      // channels rather than into alpha, which the worker sets aside.
      data.data[at] = data.data[at + 1] = data.data[at + 2] = coverage[index];
      data.data[at + 3] = 255;
    }
    context.putImageData(data, 0, 0);

    const inLayer = unplace(canvasMask, activeLayer, canvasSize, target);
    const url = inLayer.toDataURL("image/png");
    setStatus("Filling the masked area…");
    const result = await photoInpaint({
      projectId: project.id, imagePath: workingPath!,
      maskPngBase64: url.slice(url.indexOf(",") + 1),
      cacheKey: removalCacheKey(scopeDocument, node),
    });
    files.current.delete(result.outputPath);
    patchScope({
      ...scopeDocument,
      nodes: scopeDocument.nodes.map((item) => item.id === node.id
        ? { ...item, cachePath: result.outputPath } as PhotoNode : item),
    }, { coalesce: false });
    bumpFiles();
    setStatus("Filled");
  });

  /**
   * Solve the grade that puts this layer in the same tonal range as what is behind it.
   *
   * This is the one that most directly replaces a round trip through an image model: closed form,
   * instant, and it lands as ordinary slider values on a new grade step the user can then disagree
   * with, rather than as pixels they have to accept.
   */
  const runMatchColour = () => {
    if (!activeLayer || !document || !previewSize) return;
    const result = compose(previewSize.width, previewSize.height, document, activeLayer.id);
    const backdrop = result.backdrops.get(activeLayer.id);
    const raw = result.raw.get(activeLayer.id);
    if (!backdrop || !raw) { setStatus("Nothing behind this layer to match"); return; }
    const solved = matchColour(raw, backdrop);
    if (!solved) { setStatus("Not enough overlap between this layer and what is behind it"); return; }
    const node = createGradeNode();
    node.adjustments = { ...node.adjustments, ...solved };
    patchScope({ ...scopeDocument!, nodes: [...scopeDocument!.nodes, node] }, { coalesce: false });
    setSelectedNodeId(node.id);
    setStatus(`Matched · ${signed(solved.exposure)} EV, temp ${signed(solved.temperature)}, contrast ${signed(solved.contrast)}`);
  };

  /* --------------------------------------------------------------- output */

  /** Compose at full resolution and store the result as a new asset. */
  const renderToProjectMedia = async (): Promise<MediaAsset | undefined> => {
    if (!asset || !document || !canvasSize) { setFailure("Load a picture first"); return undefined; }
    await ensureMasterFrames();
    const result = compose(canvasSize.width, canvasSize.height, document, undefined, undefined, true);
    const flat = canvasOf(canvasSize.width, canvasSize.height);
    flat.getContext("2d")!.putImageData(
      new ImageData(result.pixels, canvasSize.width, canvasSize.height), 0, 0);
    const oriented = canvasOf(1, 1);
    drawOriented(oriented, flat, geometryOf(document));

    const dataUrl = oriented.toDataURL("image/png");
    // A canvas past the browser's size limit returns "data:," rather than throwing, which would
    // otherwise reach the Rust side as an empty PNG and fail there with something unhelpful.
    if (dataUrl.length < 128) {
      throw new Error(`${oriented.width} × ${oriented.height} is past what the renderer can encode in one image`);
    }
    const written = await writeEditedImage({
      projectId: project.id,
      pngBase64: dataUrl.slice(dataUrl.indexOf(",") + 1),
      sourcePath: asset.path,
      width: oriented.width, height: oriented.height,
    });
    const probe = await probeMedia(written.outputPath);
    const created: MediaAsset = {
      id: crypto.randomUUID(),
      name: filename(written.outputPath),
      path: written.outputPath,
      kind: "image",
      duration: fromSeconds(5),
      width: written.width ?? oriented.width,
      height: written.height ?? oriented.height,
      colorSpace: probe.colorSpace,
      origin: {
        operation: "canvas-flatten", model: "Photo Edit",
        prompt: describe(document),
        sourceAssetIds: layersOf(document).map((layer) => layer.assetId).filter(Boolean),
        createdAt: new Date().toISOString(),
      },
    };
    addAssets([created]);
    return created;
  };

  const runSave = () => void guard("Saving", async () => {
    setStatus("Rendering at full size…");
    const created = await renderToProjectMedia();
    if (created) {
      setPhotoEditAsset(created.id);
      setStatus(`Saved ${created.name} to Project Media`);
    }
  });

  const runSendToCanvas = () => void guard("Saving", async () => {
    setStatus("Rendering at full size…");
    const created = await renderToProjectMedia();
    if (created) {
      sendToImageStudio(created.id);
      setStatus(`${created.name} added to the Image Studio canvas`);
    }
  });

  const runExport = async () => {
    if (!asset) return;
    const extension = exportFormat === "jpeg" ? "jpg" : exportFormat;
    const stem = slugify(asset.name.replace(/\.[^.]+$/, ""), 6) || "edit";
    const target = await save({
      defaultPath: `${stem}-edit.${extension}`,
      filters: [{ name: exportFormat.toUpperCase(), extensions: [extension] }],
    });
    if (!target) { setStatus("Export canceled"); return; }
    await guard("Export", async () => {
      setStatus("Rendering at full size…");
      // The edit is stored in the project first and then converted, so the exported file and the
      // copy you keep are the same pixels, and format work stays in the one exporter that already
      // knows about matte, quality and lossless WebP.
      const created = await renderToProjectMedia();
      if (!created) return;
      setStatus("Writing the exported file…");
      const result = await exportImage({
        sourcePath: created.path, outputPath: target,
        format: exportFormat, quality: exportQuality,
        lossless: exportFormat === "webp" && exportQuality >= 100,
        matte: exportFormat === "jpeg" ? exportMatte : undefined,
      });
      setPhotoEditAsset(created.id);
      setStatus(`Exported ${filename(result.outputPath)} · ${result.width}×${result.height}`);
    });
  };

  /* ------------------------------------------------- Clip Editor render (4b) */

  const [renderProgress, setRenderProgress] = useState<{ done: number; total: number; fps?: number }>();
  const [rendered, setRendered] = useState<{ asset: MediaAsset; replaceStart: number; replaceEnd: number }>();
  const renderCancel = useRef(false);
  const renderPlan = document && clipTarget ? clipRenderPlan(document, clipTarget.clip) : undefined;
  // The Clip Editor must not close under a render or a tracker run.
  useEffect(() => { clipTarget?.onRunningChange?.(Boolean(running)); }, [running, clipTarget?.onRunningChange]);
  useEffect(() => () => clipTarget?.onRunningChange?.(false), [clipTarget?.onRunningChange]);

  /**
   * Everything one composition frame reads, loaded: the files (mattes and depth follow the frame) and
   * the full-quality media frames. Called for the NEXT frame while the current one composes and
   * encodes, so decoding overlaps the rest of the pipeline.
   */
  const prepareRenderFrame = async (source: PhotoDocument, at: number) => {
    const began = performance.now();
    const needed = filesWantedAt(source, at);
    const loadedHere: string[] = [];
    await Promise.all([...needed].filter((path) => !files.current.has(path)).map(async (path) => {
      const image = await loadReadableImage(path);
      const w = image.naturalWidth || image.width, h = image.naturalHeight || image.height;
      const canvas = canvasOf(w, h);
      canvas.getContext("2d", { willReadFrequently: true })!.drawImage(image, 0, 0);
      files.current.set(path, { canvas, width: w, height: h });
      loadedHere.push(path);
    }));
    const needs = mediaNeedsAt(source, at, compositionRate(source));
    const keys = needs.map((need) => mediaFrameKey(need.media, need.index, false));
    await Promise.all(needs.map((need) => frameStore.current.load(need.media, need.index, false)));
    return { needed, loadedHere, keys, needs, decodeMs: performance.now() - began };
  };

  /**
   * One composition frame at full size, oriented, as RGBA: the same compositor call as the preview
   * and Save. Synchronous, so nothing a prefetch finishes can evict this frame's pictures mid-compose.
   */
  const composeForRender = (source: PhotoDocument, at: number, width: number, height: number, oriented: HTMLCanvasElement) => {
    // Sequence mattes and depth share cache slots across frames, so every frame rasterises afresh.
    maskCache.current.clear();
    const result = compositor.current.compose({
      document: source, width, height, sources: sourcesNow(), frame: at, useProxy: false,
      resourcesFor: (scope, pixels) => buildScopeResources(
        { document: source, files: (path) => files.current.get(path), cache: maskCache.current }, scope, width, height, pixels),
    }, gpuCompositor);
    // Without crop, rotation or flip the compositor's pixels are the frame: no canvas round trip
    // (three 4-byte-per-pixel copies and a canvas per frame).
    const geometry = geometryOf(source);
    if (isNeutralGeometry(geometry)) {
      oriented.width = width;
      oriented.height = height;
      return encodableRgba(result.pixels);
    }
    const flat = canvasOf(width, height);
    flat.getContext("2d")!.putImageData(new ImageData(result.pixels, width, height), 0, 0);
    drawOriented(oriented, flat, geometry);
    const context = oriented.getContext("2d", { willReadFrequently: true })!;
    const data = context.getImageData(0, 0, oriented.width, oriented.height).data;
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  };

  /**
   * Render every frame of the composition to an MP4 in the project's media folder. Pipelined: frame
   * n+1 is being decoded while n composes, and n is sent while n+1 composes. The time each stage
   * takes is reported at the end, so a slow render says where its time went.
   */
  const runRender = () => clipTarget && document && canvasSize && renderPlan && void guard("Rendering", async () => {
    const plan = renderPlan;
    const source = document;
    const { width, height } = canvasSize;
    setPlaying(false);
    setRendered(undefined);
    renderCancel.current = false;
    const oriented = canvasOf(1, 1);
    const started = performance.now();
    // decode: each frame's own decode time (it runs while the previous frame composes and sends);
    // wait: how long the loop then still blocked on it. send: the IPC round trip until its promise
    // settled (late when the page is busy); blocked: how long the loop waited on the previous send.
    // pack: the page's RGB -> YUV 4:2:0 conversion (Phase 6), timed apart from compose.
    const spent = { decode: 0, wait: 0, compose: 0, pack: 0, send: 0, blocked: 0 };
    let sender: FrameSender | undefined;
    // Two I420 buffers: frame n+1 is packed while frame n may still be in flight.
    const packed = [new Uint8Array(0), new Uint8Array(0)];
    const decoderAtStart = frameStore.current.decoderStats();
    let lastShown = 0;
    setRenderProgress({ done: 0, total: plan.frames });
    setStatus(`Rendering ${plan.frames} frames at ${width} × ${height} (${gpuCompositor ? "GPU" : "CPU"} compositor)…`);
    let session: string | undefined;
    let sending: Promise<unknown> = Promise.resolve();
    const layerFiles = new Set(layersOf(source).map((layer) => layer.sourcePath));
    try {
      let next = prepareRenderFrame(source, 0);
      for (let at = 0; at < plan.frames; at++) {
        if (renderCancel.current) throw new Error("cancelled");
        let mark = performance.now();
        const ready = await next;
        spent.decode += ready.decodeMs;
        // Pinned before anything else loads; a frame the store let go while unpinned is fetched again.
        frameStore.current.pin(ready.keys);
        await Promise.all(ready.needs.filter((need, index) => !frameStore.current.get(ready.keys[index]))
          .map((need) => frameStore.current.load(need.media, need.index, false)));
        spent.wait += performance.now() - mark;
        if (at + 1 < plan.frames) next = prepareRenderFrame(source, at + 1);
        mark = performance.now();
        const rgba = composeForRender(source, at, width, height, oriented);
        spent.compose += performance.now() - mark;
        // Per-frame files (a matte frame, a depth frame) are dropped again once no later frame needs
        // them: a long clip would otherwise hold every frame of every sequence in memory by the end.
        const upcoming = at + 1 < plan.frames ? filesWantedAt(source, at + 1) : new Set<string>();
        for (const path of ready.loadedHere) {
          if (!layerFiles.has(path) && path !== source.depthPath && !upcoming.has(path)) files.current.delete(path);
        }
        mark = performance.now();
        await sending;
        if (!session) {
          const opened = await renderVideoStart({
            projectId: project.id, name: `${clipTarget.clip.name} vfx`,
            width: oriented.width, height: oriented.height, fps: plan.fps,
            audio: clipTarget.asset.hasAudio === false ? undefined : plan.audio,
            pixelFormat: "yuv420", http: true,
          });
          session = opened.id;
          sender = new FrameSender(opened);
        }
        spent.blocked += performance.now() - mark;
        let bytes = rgba;
        if (sender!.format === "yuv420") {
          mark = performance.now();
          const slot = at & 1;
          bytes = packYuv420(rgba, oriented.width, oriented.height, packed[slot]);
          packed[slot] = bytes;
          spent.pack += performance.now() - mark;
        }
        const sendStart = performance.now();
        sending = sender!.send(bytes).then(() => { spent.send += performance.now() - sendStart; });
        const now = performance.now();
        if (now - lastShown > 250 || at + 1 === plan.frames) {
          lastShown = now;
          setRenderProgress({ done: at + 1, total: plan.frames, fps: (at + 1) / Math.max(0.001, (now - started) / 1000) });
        }
      }
      await sending;
      const finished = await renderVideoFinish(session!);
      session = undefined;
      const probe = await probeMedia(finished.outputPath);
      const created: MediaAsset = {
        ...generatedAsset(finished.outputPath, probe),
        origin: {
          operation: "clip-vfx", model: "Clip Editor VFX",
          prompt: describe(source),
          sourceAssetIds: [clipTarget.asset.id, ...layersOf(source).map((layer) => layer.assetId).filter(Boolean)]
            .filter((id, index, all) => all.indexOf(id) === index),
          parameters: { frames: plan.frames, fps: plan.fps, rangeStart: plan.replaceStart, rangeEnd: plan.replaceEnd },
          createdAt: new Date().toISOString(),
        },
      };
      // Registered without touching the selection: `addAssets` clears the selected clip, and the
      // Clip Editor closes on its own when its clip is deselected (found on the 4090, 2026-09-28).
      addAssets([created], { keepSelection: true });
      setRendered({ asset: created, replaceStart: plan.replaceStart, replaceEnd: plan.replaceEnd });
      const total = (performance.now() - started) / 1000;
      const decoder = frameStore.current.decoderStats();
      setStatus(renderTimingLine({
        name: created.name, frames: finished.frames, seconds: total, gpu: gpuCompositor, spent,
        encoder: finished.encoder, encoderSeconds: finished.writeSeconds,
        fedSamples: decoder.fedSamples - decoderAtStart.fedSamples, resets: decoder.resets - decoderAtStart.resets,
        transport: sender?.describe(),
      }));
    } catch (error) {
      await sending.catch(() => undefined);
      if (session) await renderVideoCancel(session).catch(() => undefined);
      if (renderCancel.current) { setStatus("Render cancelled"); return; }
      throw error;
    } finally {
      frameStore.current.pin([]);
      setRenderProgress(undefined);
      bumpMedia();
    }
  });

  /**
   * VFX Phase 5: a spritesheet of the composition, or of one layer alone, over composition frames
   * [first, last] every `step`, at `scale` of the canvas: rendered by the same compositor call as
   * Render VFX (full-quality frames, mattes and depth per frame), trimmed to the union of the frames'
   * alpha, packed into one atlas with a Godot 4 SpriteFrames .tres and a JSON sidecar.
   */
  const runSpritesheet = (options: { layerId?: string; first: number; last: number; step: number; scale: number; padding: number; trim: boolean; loop: boolean }) =>
    clipTarget && document && canvasSize && void guard("Exporting spritesheet", async () => {
      const layer = options.layerId ? layers.find((item) => item.id === options.layerId) : undefined;
      const name = layer ? layer.name : clipTarget.clip.name;
      const source: PhotoDocument = {
        ...document, background: "transparent",
        layers: options.layerId ? layers.map((item) => ({ ...item, visible: item.id === options.layerId })) : layers,
      };
      const frames = sheetFrames(options.first, Math.min(options.last, duration - 1), options.step);
      if (!frames.length) throw new Error("No frames in that range");
      const width = Math.max(1, Math.round(canvasSize.width * options.scale));
      const height = Math.max(1, Math.round(canvasSize.height * options.scale));
      const oriented = canvasOf(1, 1);
      const rendered: Uint8Array[] = [];
      const layerFiles = new Set(layersOf(source).map((item) => item.sourcePath));
      try {
        for (const at of frames) {
          const ready = await prepareRenderFrame(source, at);
          frameStore.current.pin(ready.keys);
          await Promise.all(ready.needs.filter((need, index) => !frameStore.current.get(ready.keys[index]))
            .map((need) => frameStore.current.load(need.media, need.index, false)));
          rendered.push(composeForRender(source, at, width, height, oriented));
          for (const path of ready.loadedHere) if (!layerFiles.has(path) && path !== source.depthPath) files.current.delete(path);
          if (rendered.length % 8 === 0) setStatus(`Spritesheet: rendered ${rendered.length} of ${frames.length} frames`);
        }
      } finally {
        frameStore.current.pin([]);
      }
      const frameWidth = oriented.width, frameHeight = oriented.height;
      const full = { x: 0, y: 0, width: frameWidth, height: frameHeight };
      const trim = options.trim ? unionAlphaBounds(rendered, frameWidth, frameHeight) : undefined;
      if (options.trim && !trim) throw new Error(`${name} is fully transparent over those frames`);
      const crop = trim ?? full;
      const layout = sheetLayout(rendered.length, crop.width, crop.height, options.padding);
      if (layout.width > 16384 || layout.height > 16384) {
        throw new Error(`The atlas would be ${layout.width} × ${layout.height}, over Godot's 16384 limit — use a smaller scale or a larger step`);
      }
      const png = await encodePng(packAtlas(rendered, frameWidth, crop, layout), layout.width, layout.height);
      const slug = sheetSlug(name);
      const folder = await spritesheetFolder(name);
      const fps = compositionRate(document) / options.step;
      await spritesheetWrite(folder, `${slug}.png`, png);
      const text = new TextEncoder();
      await spritesheetWrite(folder, `${slug}.tres`, text.encode(spriteFramesTres({ texturePath: `${slug}.png`, fps, loop: options.loop, regions: layout.cells })));
      const saved = await spritesheetWrite(folder, `${slug}.json`, text.encode(sheetJson({
        image: `${slug}.png`, layout, fps, loop: options.loop, sourceSize: { width: frameWidth, height: frameHeight }, trim, source: name, frames,
      })));
      setStatus(`Spritesheet saved: ${saved.replace(/[\\/][^\\/]+$/, "")} — ${frames.length} frames of ${crop.width} × ${crop.height}, atlas ${layout.width} × ${layout.height} (${slug}.png, .tres for Godot 4, .json)`);
    });

  // Leaving VFX mode (Escape, another mode) mid-render stops the render rather than orphaning ffmpeg.
  useEffect(() => () => { renderCancel.current = true; }, []);

  const clipSpan = clipTarget
    ? clipTarget.clip.sourceOut.numerator / clipTarget.clip.sourceOut.denominator - clipSourceIn : 0;
  const setPlateRange = (start: number, end: number, label: string) => {
    if (!clipTarget || !document) return;
    patch(withPlateRange(document, clipTarget.clip, start, end), { coalesce: false });
    setCompFrame(0);
    setRendered(undefined);
    setStatus(`Plate set to ${label} · ${start.toFixed(2)}–${end.toFixed(2)}s`);
  };

  const placeRender = (how: "replace" | "insert") => {
    if (!clipTarget || !rendered) return;
    if (how === "replace") replaceClipRangeWithAsset(clipTarget.clip.id, rendered.replaceStart, rendered.replaceEnd, rendered.asset);
    else insertAssetAfterClip(clipTarget.clip.id, rendered.asset);
    clipTarget.onClose();
  };

  /* ---------------------------------------------------------------- markup */

  const card = (name: string, title: string, state: string, body: React.ReactNode) => {
    const open = openCards[name] ?? false;
    return <section className={`studio-card photo-card${open ? " open" : ""}`}>
      <button className="photo-card-head" type="button"
        onClick={() => setOpenCards((current) => ({ ...current, [name]: !open }))}>
        <ChevronDown className="photo-chev" size={14} />
        <strong>{title}</strong>
        <em>{state}</em>
      </button>
      {open && <div className="photo-card-body">{body}</div>}
    </section>;
  };

  const addNode = (node: PhotoNode) => scopeDocument && patchScope(
    { ...scopeDocument, nodes: [...scopeDocument.nodes, node] }, { coalesce: false },
  );

  const addTextLayer = () => {
    if (!document) return;
    const layer = createTextLayer(canvasSize, fonts[0]?.families[0]
      ? { fontFamily: fonts[0].families[0] } : {});
    setLayers([...layers, layer]);
    setScopeId(layer.id);
    setOpenCards((current) => ({ ...current, text: true }));
    setTool("move");
    setStatus("Text layer added — type into the Text card, drag it into place");
  };

  const addGradientLayer = () => {
    if (!document) return;
    const layer = createGradientLayer();
    setLayers([...layers, layer]);
    setScopeId(layer.id);
    setOpenCards((current) => ({ ...current, gradient: true }));
    setTool("view");
    setStatus("Gradient layer added — set its colours and direction in the Gradient card");
  };

  const patchGradient = (over: Partial<Gradient>, label: string, coalesce = false) => {
    if (!activeLayer?.gradient) return;
    patchLayer(activeLayer.id, { gradient: { ...activeLayer.gradient, ...over } },
      coalesce ? undefined : { coalesce: false });
    setStatus(label);
  };

  const patchText = (over: Partial<TextBlock>, label: string, coalesce = false) => {
    if (!activeLayer?.text) return;
    patchLayer(activeLayer.id, { text: { ...activeLayer.text, ...over } },
      coalesce ? undefined : { coalesce: false });
    setStatus(label);
  };

  const patchShape = (over: Partial<ShapeBlock>, label: string, coalesce = false) => {
    if (!activeLayer?.shape) return;
    patchLayer(activeLayer.id, { shape: { ...activeLayer.shape, ...over } },
      coalesce ? undefined : { coalesce: false });
    setStatus(label);
  };

  /* ------------------------------------------------------ VFX Phase 6 paths */

  const patchPath = (over: Partial<PathBlock>, label: string, coalesce = false) => {
    if (!activeLayer?.path) return;
    patchLayer(activeLayer.id, { path: { ...activeLayer.path, ...over } }, coalesce ? undefined : { coalesce: false });
    setStatus(label);
  };

  /** Apply a layer-list action; `undefined` means it had nothing to act on. */
  const pathAction = (next: PhotoLayer[] | undefined, label: string, failure: string, select?: string) => {
    if (!next) { setStatus(failure); return; }
    setLayers(next);
    if (select) setScopeId(select);
    setStatus(label);
  };

  /** The rectangle a new envelope node covers: the active layer's placed box, else the canvas. */
  const envelopeRect = () => {
    const whole = { x: 0, y: 0, width: 1, height: 1 };
    if (!activeLayer || !canvasSize || activeLayer.gradient || activeLayer.generator) return whole;
    const size = placementSize(activeLayer, sources);
    if (!size) return whole;
    const matrix = layerMatrix(activeLayer, canvasSize.width, canvasSize.height, size);
    const rect = croppedRect(activeLayer, size);
    const corners = [[0, 0], [rect.width, 0], [rect.width, rect.height], [0, rect.height]].map(([x, y]) => applyMatrix(matrix, x, y));
    const xs = corners.map((c) => c[0] / canvasSize.width), ys = corners.map((c) => c[1] / canvasSize.height);
    const x0 = Math.max(0, Math.min(...xs)), y0 = Math.max(0, Math.min(...ys));
    const x1 = Math.min(1, Math.max(...xs)), y1 = Math.min(1, Math.max(...ys));
    return x1 > x0 && y1 > y0 ? { x: x0, y: y0, width: x1 - x0, height: y1 - y0 } : whole;
  };

  const convertTextToOutlines = () => activeLayer?.text && void guard("Converting to outlines", async () => {
    const source = activeLayer;
    const { layer, notes } = await textLayerToPath(source);
    updateLayers((current) => current.map((item) => item.id === source.id ? layer : item));
    setScopeId(layer.id);
    setOpenCards((current) => ({ ...current, path: true }));
    setStatus(`${source.name} is now a path${notes.length ? ` — ${notes.join("; ")}` : ""}`);
  });

  const addShapeLayer = (kind: ShapeKind) => {
    if (!document) return;
    const layer = createShapeLayer(kind, canvasSize);
    setLayers([...layers, layer]);
    setScopeId(layer.id);
    setOpenCards((current) => ({ ...current, shape: true, text: true }));
    setTool("move");
    setStatus(`${layer.name} added — drag it into place, drag its tail at whoever is speaking`);
  };

  const addLayerFrom = (item: MediaAsset) => {
    if (!document) return;
    // A picture bigger than the canvas comes in fitted rather than at source pixels. Dropping a
    // 4K cutout onto a 1024 plate and seeing one of its elbows is technically "100%" and useless.
    const size = item.width && item.height ? { width: item.width, height: item.height } : undefined;
    const fit = size && canvasSize
      ? Math.min(1, canvasSize.width / size.width, canvasSize.height / size.height) : 1;
    const layer = createLayer(item, { scale: fit });
    setLayers([...layers, layer]);
    setScopeId(layer.id);
    setTool("move");
    setStatus(`${item.name} added as a layer — drag it into place`);
  };

  /* ---------------------------------------------------------- VFX Phase 3 */

  const refreshElements = useCallback(async () => {
    try { setElements(await listElements()); } catch (error) { console.warn("element_list failed", error); }
  }, []);
  useEffect(() => { if (live && onClip) void refreshElements(); }, [live, onClip, refreshElements]);

  /** Fitted like a picture: never bigger than the canvas, and a small element comes in at a usable size. */
  const mediaScale = (media: LayerMedia) => canvasSize
    ? Math.min(canvasSize.width * 0.6 / media.width, canvasSize.height * 0.8 / media.height, canvasSize.width / media.width)
    : 1;

  const addMediaLayer = (name: string, media: LayerMedia, over: Partial<PhotoLayer> = {}) => {
    if (!document) return;
    const layer = createMediaLayer(name, media, { scale: mediaScale(media), ...over });
    setLayers([...layers, layer]);
    setScopeId(layer.id);
    setOpenCards((current) => ({ ...current, timing: true }));
    setTool("move");
  };

  const addElementLayer = (element: ElementEntry) => {
    addMediaLayer(element.manifest.name, mediaFromElement(element, shownFrame));
    setStatus(`${element.manifest.name} added at frame ${shownFrame} — drag it into place; Timing sets its loop`);
  };

  const addVideoLayer = (item: MediaAsset) => {
    const media = mediaFromVideoAsset(item, shownFrame);
    if (!media) { setStatus(`${item.name} has no size or length yet — open it in the Video Editor once so it is probed`); return; }
    addMediaLayer(item.name, media, { assetId: item.id });
    setStatus(`${item.name} added as a video layer at frame ${shownFrame}`);
  };

  const addRainLayer = () => {
    if (!document) return;
    const layer = createMatrixRainLayer();
    setLayers([...layers, layer]);
    setScopeId(layer.id);
    setOpenCards((current) => ({ ...current, rain: true }));
    setTool("view");
    setStatus("Matrix rain added (Screen blend) — it loops exactly; the Matrix rain card has its controls");
  };

  const runImportElement = async () => {
    const picked = await open({ directory: true, multiple: false, title: "A folder of RGBA PNG frames" });
    if (typeof picked !== "string") return;
    const name = picked.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || "Element";
    await guard("Importing element", async () => {
      setStatus(`Copying ${name} into the element library…`);
      const made = await importElement(picked, name, 24);
      await refreshElements();
      setStatus(`${made.manifest.name} imported · ${made.manifest.frameCount} frames at 24 fps (change it in Timing)`
        + (made.manifest.proxy ? "" : " · no preview proxy"));
    });
  };

  const runGenerateElement = (request: { name: string; prompt: string; frames: number; seed: number }) =>
    void guard("Generating element", async () => {
      setStatus(`Wan Alpha: ${request.frames} frames of "${request.name}" — the resource guard checks memory first`);
      const made = await generateElement(request);
      await refreshElements();
      setStatus(`${made.manifest.name} generated — click it in Elements to add it as a layer`);
    });

  const runBuildProxy = (id: string) => void guard("Building proxy", async () => {
    await buildElementProxy(id);
    await refreshElements();
    setStatus("Proxy built");
  });

  /** SAM 2 matte sequences cached by Mask VFX on timeline clips, for sequence mask entries. */
  const trackedMattes = useMemo<TrackedMatte[]>(() => project.sequences.flatMap((sequence) =>
    sequence.tracks.flatMap((track) => track.clips.flatMap((clip) => (clip.vfxLayers ?? []).flatMap((vfx) => vfx.matteCache ? [{
      label: `${vfx.name}`, cachePath: vfx.matteCache.cachePath, frameCount: vfx.matteCache.frameCount, fps: vfx.matteCache.fps,
    }] : []))))
    .filter((item, index, all) => all.findIndex((other) => other.cachePath === item.cachePath) === index), [project.sequences]);

  /* ---------------------------------------------------------- VFX Phase 4 */

  useEffect(() => { if (live && onClip) void motionStatus().then(setMotion).catch(() => undefined); }, [live, onClip]);

  const tracks = useMemo(() => document?.tracks ?? [], [document]);
  /** The media layer new tracks are measured on: the selected video, else the first video, else any media. */
  const trackPlate = (activeLayer?.media?.kind === "video" ? activeLayer : undefined)
    ?? layers.find((layer) => layer.media?.kind === "video") ?? layers.find((layer) => layer.media);

  /**
   * Store a new track, and attach the selected layer to it in the same step ("creating and attaching
   * stay one action"): an emitter spawns from it, any other layer follows it with its base (bottom
   * centre) on the clicked point, anchored at this frame. The plate itself is never attached.
   */
  const commitTrack = (track: PointTrack, at?: [number, number]) => {
    if (!document) return;
    const base = withLayers(document);
    const target = activeLayer && trackPlate && activeLayer.id !== trackPlate.id && !activeLayer.locked ? activeLayer : undefined;
    const next = layersOf(base).map((layer) => {
      if (layer.id !== target?.id) return layer;
      if (layer.generator?.kind === "emitter") return { ...layer, generator: { ...layer.generator, trackId: track.id } };
      if (layer.generator) return layer;
      const attached: PhotoLayer = { ...layer, attach: { trackId: track.id, anchorFrame: shownFrame } };
      if (!at || !canvasSize) return attached;
      const size = placementSize(layer, sources);
      const half = size ? size.height * layer.scale / 2 / canvasSize.height : 0;
      return { ...attached, x: at[0], y: at[1] - half };
    });
    patch({ ...base, tracks: [...(base.tracks ?? []), track], layers: next }, { coalesce: false });
    setOpenCards((current) => ({ ...current, tracks: true }));
  };

  /** Frames [from, to) of `media` as luma, at most `maxWidth` wide, for the built-in tracker. */
  const readGreyFrames = async (media: LayerMedia, from: number, to: number, maxWidth: number) => {
    const k = Math.min(1, maxWidth / media.width);
    const width = Math.max(8, Math.round(media.width * k)), height = Math.max(8, Math.round(media.height * k));
    const scratch = canvasOf(width, height);
    const context = scratch.getContext("2d", { willReadFrequently: true })!;
    const frames: GreyFrame[] = [];
    for (let index = from; index < to; index++) {
      const picture = await frameStore.current.load(media, index, false);
      context.clearRect(0, 0, width, height);
      context.drawImage(picture.canvas, 0, 0, width, height);
      frames.push(greyOf(context.getImageData(0, 0, width, height).data, width, height));
      if ((index - from) % 24 === 0) setStatus(`Reading frames for the tracker… ${index - from + 1} of ${to - from}`);
    }
    return { frames, kx: width / media.width, ky: height / media.height };
  };

  /** "Track a point": the click is mapped into the plate's source pixels and tracked both ways. */
  const runTrack = (at: [number, number]) => {
    if (!document || !trackPlate?.media || !canvasSize) { setStatus("Add a video layer to track a point on it"); return; }
    const plate = attachedLayer(document, trackPlate, shownFrame, canvasSize, rate);
    const media = plate.media!;
    const point = canvasToSourcePoint(plate, at, canvasSize);
    const shown = mediaFramesAt(media, shownFrame, rate);
    if (!point || !shown) { setStatus(`${plate.name} is not on screen at frame ${shownFrame}`); return; }
    if (point[0] < 0 || point[1] < 0 || point[0] > media.width || point[1] > media.height) {
      setStatus(`Click on ${plate.name} itself`);
      return;
    }
    const query = shown[shown.length - 1].index;
    const from = Math.max(0, Math.floor(media.inPoint));
    const to = Math.min(media.frameCount, Math.floor(media.outPoint), from + 600);
    const name = `Track ${tracks.length + 1}`;
    void guard("Tracking", async () => {
      let track: PointTrack;
      if (trackEngine === "cotracker3") {
        setStatus("CoTracker3: the resource guard checks memory first…");
        const result = await trackPoints({ videoPath: media.path, queryFrame: query, points: [point], startFrame: from, frameCount: to - from });
        track = { id: newTrackId(), name, layerId: trackPlate.id, first: result.first, samples: result.points[0].samples, source: "cotracker3" };
      } else {
        const { frames, kx, ky } = await readGreyFrames(media, from, to, 640);
        setStatus(`Tracking ${frames.length} frames…`);
        await new Promise((resolve) => setTimeout(resolve, 0));
        const samples = lucasKanadeBothWays(frames, query - from, [point[0] * kx, point[1] * ky])
          .map(([x, y, seen]) => [x / kx, y / ky, seen] as TrackSample);
        track = { id: newTrackId(), name, layerId: trackPlate.id, first: from, samples, source: "lucas-kanade" };
      }
      commitTrack(track, at);
      const seen = track.samples.filter((sample) => sample[2]).length;
      setStatus(`${name}: seen in ${seen} of ${track.samples.length} frames${seen < track.samples.length ? " (unseen frames hold the nearest seen position)" : ""}`);
    });
  };

  /** SAM 2 mattes the plate can be tracked from: its own sequence entries, and Mask VFX caches. */
  const sam2Sources = useMemo(() => {
    const own = (trackPlate?.masks ?? []).flatMap((mask) => (mask.stack ?? []).flatMap((entry) => entry.source.kind === "sequence"
      ? [{ label: mask.name, cachePath: entry.source.cachePath, frameCount: entry.source.frameCount, first: entry.source.first ?? 0 }] : []));
    return [...own, ...trackedMattes].filter((item, index, all) => all.findIndex((other) => other.cachePath === item.cachePath) === index);
  }, [trackPlate, trackedMattes]);

  /* ---------------------------------------------------------- VFX Phase 5 */

  const [animate, setAnimate] = useState<AnimateStatus>();
  useEffect(() => { if (live && onClip) void wanAnimateStatus().then(setAnimate).catch(() => undefined); }, [live, onClip]);
  /** The clip's own plate: what drives an aligned character. */
  const clipPlate = clipTarget ? layers.find((layer) => layer.id === plateLayerId(clipTarget.clip.id)) : undefined;

  /**
   * The document as last rendered, for steps that place layers after a long await (a character,
   * then its clean plate): the closure's `document` would be the one from before the first placement.
   */
  const latestDocument = useRef(document);
  latestDocument.current = document;
  const updateLayers = (update: (current: PhotoLayer[]) => PhotoLayer[]) => {
    const current = latestDocument.current;
    if (!current) return;
    const base = withLayers(current);
    patch({ ...base, layers: update(layersOf(base)) }, { coalesce: false });
  };

  /** The element as a layer: over the plate and aligned when the plate drove it, else at this frame. */
  const placeCharacter = (element: ElementEntry, relight: boolean) => {
    const current = latestDocument.current;
    if (!current || !canvasSize) return;
    const aligned = clipPlate && drivenByPlate(element, clipPlate) ? characterLayer(current, clipPlate, element, canvasSize) : undefined;
    if (!aligned) {
      addElementLayer(element);
      return;
    }
    if (isCleanPlate(element)) {
      // Between the plate and everything on it: it paints the actor out, the character paints over.
      updateLayers((current) => insertAbove(current, clipPlate!.id, aligned));
      setStatus(`${element.manifest.name} placed just above ${clipPlate!.name}: the actor is painted out where the character does not cover him`);
      return;
    }
    const layer = relight ? { ...aligned, nodes: [createRelightNode()] } : aligned;
    updateLayers((current) => [...current, layer]);
    setScopeId(layer.id);
    setTool("view");
    setStatus(`${element.manifest.name} placed over ${clipPlate!.name}, aligned frame for frame`
      + (relight ? "; Stack has a Relight step" : "")
      + " — for a foreground object in front of it, Tracks → Occlude by depth and set the plane in Masks");
  };

  /**
   * The plate's actor matte -> SAM 2 include points on the first driving frame and the actor's
   * framing (centroid x, feet line, standing height), all normalised to the plate. The standing
   * height is the tallest box over up to 8 frames spread across the range.
   */
  const actorPrompt = async (cachePath: string, plate: PhotoLayer, range: { first: number; count: number }) => {
    const source = sam2Sources.find((item) => item.cachePath === cachePath);
    const media = plate.media;
    if (!source || !media) return { points: [], frame: undefined, region: undefined };
    const sequence = { kind: "sequence" as const, cachePath: source.cachePath, frameCount: source.frameCount, first: "first" in source ? source.first : 0 };
    const canvas = canvasOf(media.width, media.height);
    const context = canvas.getContext("2d", { willReadFrequently: true })!;
    const coverageAt = async (sourceFrame: number) => {
      const image = await loadReadableImage(sequenceFramePath(sequence, sourceFrame));
      context.clearRect(0, 0, media.width, media.height);
      context.drawImage(image, 0, 0, media.width, media.height);
      return coverageFromImage(context.getImageData(0, 0, media.width, media.height).data, media.width * media.height);
    };
    const first = await coverageAt(range.first);
    const samples: Uint8Array[] = [];
    const step = Math.max(1, Math.floor(range.count / 12));
    for (let at = range.first + step; at < range.first + range.count; at += step) samples.push(await coverageAt(at));
    return {
      points: seedPoints(first, media.width, media.height, [0, 0, media.width, media.height]),
      frame: actorFrame(first, samples, media.width, media.height),
      region: actorRegion([first, ...samples], media.width, media.height),
    };
  };

  const runCharacter = (options: CharacterOptions) => void guard("Animating character", async () => {
    const picture = images.find((item) => item.id === options.imageId);
    if (!picture || !document) return;
    let request;
    if (options.drive === "plate") {
      const range = clipPlate ? drivingRange(clipPlate) : undefined;
      const media = clipPlate?.media;
      if (!range || !media || !clipPlate) throw new Error("The plate has no footage to drive with");
      const actor = options.matteId ? await actorPrompt(options.matteId, clipPlate, range) : { points: [], frame: undefined, region: undefined };
      if (options.matteId && !actor.points.length) setStatus("The actor matte is empty on the first frame — the character keeps its picture's framing, and BiRefNet alone cuts it out");
      request = {
        drivingVideo: media.path, firstFrame: range.first, frameCount: range.count,
        plateWidth: media.width, plateHeight: media.height, fps: media.fps, actorPoints: actor.points, actorFrame: actor.frame, actorRegion: actor.region,
      };
    } else {
      const clip = videos.find((item) => item.id === options.drive);
      const media = clip ? mediaFromVideoAsset(clip) : undefined;
      if (!clip || !media) throw new Error("That clip has no size or length yet — open it in the Video Editor once so it is probed");
      request = {
        drivingVideo: media.path, firstFrame: 0, frameCount: Math.min(media.frameCount, Math.max(5, duration)),
        plateWidth: media.width, plateHeight: media.height, fps: media.fps, actorPoints: [],
      };
    }
    setStatus(`${options.name}: Wan Animate 2 — the resource guard checks memory first`);
    const stop = await onCharacterProgress((progress) => {
      const line = describeCharacterProgress(progress, options.name);
      if (line) setStatus(line);
    });
    try {
      const made = await generateCharacter({
        name: options.name, characterPath: picture.path, prompt: options.prompt,
        steps: options.steps, seed: options.seed, area: options.area, ...request,
      });
      await refreshElements();
      placeCharacter(made, options.relight);
      if (options.drive === "plate" && options.hideActor && options.matteId) await paintActorOut(options.matteId);
    } finally {
      stop();
    }
  });

  /**
   * A clean plate patch for this plate from the actor's SAM 2 track (locked-off shots), placed just
   * above the plate. Wan Animate 2 has no replacement mode, so without it the actor shows wherever
   * the character does not cover him.
   */
  const paintActorOut = async (matteCachePath: string) => {
    const source = sam2Sources.find((item) => item.cachePath === matteCachePath);
    const request = clipPlate && source ? cleanPlateRequest(clipPlate, source, `${clipPlate.name} clean plate`) : undefined;
    if (!request) throw new Error("The clean plate needs the plate's footage and an actor matte (Masks → Click-select → Run)");
    setStatus(`${request.name}: building the background behind the actor from the frames around the shot…`);
    const made = await buildCleanPlate(request);
    await refreshElements();
    placeCharacter(made, false);
    const stats = made.manifest.cleanPlate as { contextKept?: number; holes?: number; fill?: string } | undefined;
    if (stats?.holes) setStatus(`${request.name} placed; ${stats.holes} pixels no frame showed were filled by ${stats.fill === "lama" ? "LaMa" : "inpainting"} — check behind the character's edges`);
  };
  const runPaintActorOut = (matteCachePath: string) => void guard("Painting the actor out", () => paintActorOut(matteCachePath));

  const runSam2Centroid = (source: { label: string; cachePath: string; frameCount: number; first?: number }) => {
    const plate = trackPlate;
    if (!plate?.media) return;
    const media = plate.media;
    void guard("Tracking", async () => {
      const masks: Array<{ coverage: Uint8Array; width: number; height: number }> = [];
      for (let index = 0; index < source.frameCount; index++) {
        const image = await loadReadableImage(sequenceFramePath({ kind: "sequence", cachePath: source.cachePath, frameCount: source.frameCount }, index));
        const width = image.naturalWidth || image.width, height = image.naturalHeight || image.height;
        const canvas = canvasOf(width, height);
        const context = canvas.getContext("2d", { willReadFrequently: true })!;
        context.drawImage(image, 0, 0);
        masks.push({ coverage: coverageFromImage(context.getImageData(0, 0, width, height).data, width * height), width, height });
        if (index % 24 === 0) setStatus(`Reading the matte… ${index + 1} of ${source.frameCount}`);
      }
      const sx = masks.length ? media.width / masks[0].width : 1, sy = masks.length ? media.height / masks[0].height : 1;
      const samples = sam2Centroid(masks).map(([x, y, seen]) => [x * sx, y * sy, seen] as TrackSample);
      commitTrack({ id: newTrackId(), name: `${source.label} centre`, layerId: plate.id, first: source.first ?? 0, samples, source: "sam2-centroid" });
      setStatus(`${source.label}: centre tracked over ${samples.length} frames`);
    });
  };

  const renameTrack = (id: string, name: string) => document && patch({
    ...document, tracks: tracks.map((track) => track.id === id ? { ...track, name } : track),
  });
  const deleteTrack = (id: string) => {
    if (!document) return;
    patch({ ...document, tracks: tracks.filter((track) => track.id !== id) }, { coalesce: false });
    setStatus("Track deleted (layers attached to it stay where they are)");
  };
  const setAttach = (attach: LayerAttach | undefined, label: string, coalesce?: boolean) => {
    if (!activeLayer) return;
    patchLayer(activeLayer.id, { attach }, coalesce ? undefined : { coalesce: false });
    setStatus(label);
  };

  const runVideoDepth = () => {
    const plate = trackPlate;
    if (!plate?.media) return;
    const media = plate.media;
    void guard("Computing depth", async () => {
      setStatus("Video Depth Anything: the resource guard checks memory first…");
      const result = await computeVideoDepth(media.path);
      patchLayer(plate.id, { media: { ...media, depth: { cachePath: result.cachePath, frameCount: result.frames } } }, { coalesce: false });
      maskCache.current.clear();
      setStatus(`Depth ready · ${result.frames} frames${result.reused ? " (cached)" : ""} — Occlude by depth hides the selected layer behind it`);
    });
  };

  const occluded = Boolean(activeLayer && trackPlate && activeLayer.masks.some((mask) => mask.id === activeLayer.maskId
    && mask.stack?.some((entry) => entry.source.kind === "depth" && entry.source.layerId === trackPlate.id)));

  /** Hide the selected layer where the plate is nearer: a Depth entry in its layer mask (made if needed). */
  const occludeByDepth = () => {
    if (!activeLayer || !trackPlate) return;
    const existing = activeLayer.maskId ? activeLayer.masks.find((mask) => mask.id === activeLayer.maskId) : undefined;
    if (existing) {
      const entry = createMaskEntry({ kind: "depth", layerId: trackPlate.id, plane: 55, softness: 6 }, { mode: "subtract" });
      patchLayer(activeLayer.id, { masks: activeLayer.masks.map((mask) => mask.id === existing.id
        ? { ...mask, stack: [...(mask.stack ?? []), entry] } : mask) }, { coalesce: false });
      setActiveMaskId(existing.id);
      setActiveEntryId(entry.id);
    } else {
      const mask = createDepthOcclusionMask(trackPlate.id);
      patchLayer(activeLayer.id, { masks: [...activeLayer.masks, mask], maskId: mask.id }, { coalesce: false });
      setActiveMaskId(mask.id);
      setActiveEntryId(mask.stack![0].id);
    }
    setOpenCards((current) => ({ ...current, masks: true }));
    setShowMask(true);
    setStatus(`${activeLayer.name} is hidden where ${trackPlate.name} is nearer — set the plane in Masks while scrubbing`);
  };

  const addEmitterLayer = (element: ElementEntry) => {
    if (!document) return;
    const media = mediaFromElement(element, 0);
    const fit = mediaScale(media);
    const trackId = tracks.length ? tracks[tracks.length - 1].id : undefined;
    const layer = { ...createEmitterLayer(media, { trackId, scaleStart: fit * 0.4, scaleEnd: fit * 0.9 }), name: `${element.manifest.name} emitter` };
    setLayers([...layers, layer]);
    setScopeId(layer.id);
    setOpenCards((current) => ({ ...current, emitter: true }));
    setTool("view");
    setStatus(`${element.manifest.name} emitter added${trackId ? " on the latest track" : " at a point"} — the Emitter card has its controls`);
  };

  /**
   * Make a mask and wire it up in one step.
   *
   * Creating a mask used to leave it attached to nothing, so painting changed the picture everywhere
   * and looked broken. A mask exists in order to limit something, so the node being edited claims it
   * unless it already has one of its own.
   */
  const createAndAttachMask = (stack?: MaskEntry[]): string | undefined => {
    if (!scopeDocument) return undefined;
    const mask = { ...createMask(`Mask ${scopeDocument.masks.length + 1}`), ...(stack ? { stack, feather: 0 } : {}) };
    const target = selectedNode && !selectedNode.maskId ? selectedNode.id : undefined;
    patchScope({
      ...scopeDocument,
      masks: [...scopeDocument.masks, mask],
      nodes: target
        ? scopeDocument.nodes.map((node) => node.id === target ? { ...node, maskId: mask.id } as PhotoNode : node)
        : scopeDocument.nodes,
    }, { coalesce: false });
    setActiveMaskId(mask.id);
    setActiveEntryId(stack?.[0]?.id);
    setTool("brush");
    setShowMask(true);
    setStatus(target ? `${mask.name} created and linked to this step — paint where it should apply`
      : `${mask.name} created — paint where it should apply`);
    return mask.id;
  };

  /**
   * Import a greyscale or alpha PNG as a mask entry (VFX Phase 2): into the active mask when there is
   * one, otherwise as a new mask that the selected step claims -- creating and attaching stay one action.
   */
  const importMatte = async () => {
    if (!scopeDocument) return;
    const picked = await open({ multiple: false, filters: [{ name: "Matte images", extensions: ["png", "webp", "jpg", "jpeg"] }] });
    if (typeof picked !== "string") return;
    const entry = createMaskEntry({ kind: "file", path: picked });
    const target = activeMask && scopeDocument.masks.some((mask) => mask.id === activeMask.id) ? activeMask : undefined;
    if (!target) {
      createAndAttachMask([entry]);
      setStatus(`Matte imported as a new mask · ${filename(picked)}`);
      return;
    }
    patchScope({
      ...scopeDocument,
      masks: scopeDocument.masks.map((mask) => mask.id === target.id ? { ...mask, stack: [...(mask.stack ?? []), entry] } : mask),
    }, { coalesce: false });
    setActiveEntryId(entry.id);
    setShowMask(true);
    setStatus(`Matte added to ${target.name} · ${filename(picked)}`);
  };

  /**
   * A track matte, After Effects style, as ONE action: a layer mask holding one `layer` entry, attached
   * as this layer's mask, with the matte layer hidden. Everything lands in a single patch (one undo).
   */
  const makeTrackMatte = (matteLayerId: string, channel: "alpha" | "luma", invert: boolean) => {
    if (!document || !activeLayer) return;
    const matteLayer = layers.find((layer) => layer.id === matteLayerId);
    if (!matteLayer) return;
    const mask = {
      ...createMask(`Matte · ${matteLayer.name}${invert ? " inverted" : ""}`), feather: 0,
      stack: [createMaskEntry({ kind: "layer", layerId: matteLayerId, channel }, { invert })],
    };
    const base = withLayers(document);
    patch({
      ...base,
      layers: layersOf(base).map((layer) => layer.id === activeLayer.id
        ? { ...layer, masks: [...layer.masks, mask], maskId: mask.id }
        : layer.id === matteLayerId ? { ...layer, visible: false } : layer),
    }, { coalesce: false });
    setActiveMaskId(mask.id);
    setActiveEntryId(undefined);
    setStatus(`${matteLayer.name} is now a ${channel}${invert ? " inverted" : ""} matte for ${activeLayer.name}, and hidden`);
  };

  /**
   * The swatches under the two gradient editors.
   *
   * Memoised on the ramp itself: drawing one is a canvas and a data URL, and the panel re-renders on
   * every pointer move over the picture. Rendered as a strip 1 pixel tall with a neutral direction,
   * because a swatch shows the COLOURS -- where they land on the frame is what the picture is for.
   */
  const rampPreview = useMemo(() => activeLayer?.gradient
    ? gradientPreview({ ...activeLayer.gradient, kind: "linear", angle: 0, cx: 0.5, cy: 0.5, scale: 1, offset: 0 }, 220, 1)
    : undefined, [activeLayer?.gradient]);
  const mapPreview = useMemo(() => selectedNode?.kind === "gradientMap"
    ? gradientPreview({ ...selectedNode.gradient, kind: "linear", angle: 0, cx: 0.5, cy: 0.5, scale: 1, offset: 0 }, 220, 1)
    : undefined, [selectedNode]);

  /**
   * A swatch for every paint the selected layer carries.
   *
   * All drawn by the ramp renderer as a flat strip, so what the picker shows is the actual
   * interpolation rather than a CSS approximation of it. Memoised together because they change
   * together and each one is a canvas and a data URL.
   */
  const paintPreview = (paint: Paint): string | undefined => paint.kind === "gradient"
    ? gradientPreview({ ...paint.gradient, kind: "linear", angle: 0, cx: 0.5, cy: 0.5, scale: 1, offset: 0 }, 220, 1)
    : undefined;
  const textPreviews = useMemo(() => {
    const text = activeLayer?.text;
    if (!text) return undefined;
    return {
      fill: paintPreview(toPaint(text.color, "#ffffff")),
      stroke: paintPreview(toPaint(text.strokeColor, "#000000")),
      shadow: paintPreview(toPaint(text.shadowColor, "#000000")),
      panel: paintPreview(toPaint(text.panelColor, "#000000")),
      panelBorder: paintPreview(toPaint(text.panelBorderColor, "#ffffff")),
    };
  }, [activeLayer?.text]);
  const shapePreviews = useMemo(() => {
    const shape = activeLayer?.shape;
    if (!shape) return undefined;
    return {
      fill: paintPreview(toPaint(shape.fill, "#ffffff")),
      stroke: paintPreview(toPaint(shape.stroke, "#000000")),
    };
  }, [activeLayer?.shape]);

  const light = selectedNode?.kind === "relight" ? selectedNode.relight : undefined;
  const lightAt = light ? atDisplay(light.x, light.y) : undefined;
  const depthState = running === "Measuring depth" ? "running" : document?.depthPath ? "ready" : "missing";
  const scopeName = activeLayer ? activeLayer.name : "Composite";

  const selectReady = Boolean(onClip && activeLayer?.media?.kind === "video" ? runtime?.sam2Ready : runtime?.photoSelectReady);
  const legacyVideo = !onClip && document ? hasVideoContent(document) : false;

  return <section className={`photo-studio${onClip ? " clip-vfx" : ""}`}>
    <aside className="image-studio-controls">
      <div className="image-studio-title">
        <span>{onClip ? <Film size={18} /> : <Sliders size={18} />}</span>
        <div>
          <strong>{onClip ? "Clip VFX" : "Photo Edit"}</strong>
          <small>{onClip ? `${clipTarget!.clip.name} · ${duration} frames at ${Number(rate.toFixed(3))} fps` : asset ? asset.name : "Nothing loaded"}</small>
        </div>
      </div>
      {clipTarget?.modeSwitch}

      {legacyVideo && <p className="photo-note photo-legacy-video">
        This picture carries video layers or tracks from before video VFX moved to the Clip Editor.
        Photo Edit shows its first frame. To animate, track or render it, select the clip on the
        timeline and open Clip Editor → VFX.
      </p>}

      {document && card("layers", "Canvas", `${layers.length} layer${layers.length === 1 ? "" : "s"}`, <>
        <LayerList document={document} activeId={scopeKey} onSelect={(id) => {
          setScopeId(id);
          setSelectedNodeId(undefined);
          setActiveMaskId(undefined);
          setTool("view");
        }} onChange={(next, label) => { setLayers(next); setStatus(label); }} />
        <div className="photo-buttons">
          <button type="button" onClick={addTextLayer}><Type size={12} /> Add text</button>
          <button type="button" onClick={addGradientLayer}><Palette size={12} /> Add gradient</button>
          <button type="button" onClick={addRainLayer}
            title={onClip ? "Procedural, loops exactly" : "Procedural glyph texture (a still here; it animates in Clip Editor → VFX)"}>
            <Sparkles size={12} /> Matrix rain</button>
          {onClip && <button type="button" onClick={() => setOpenCards((current) => ({ ...current, elements: true }))}
            title="Wan Alpha and imported elements"><Film size={12} /> Elements</button>}
        </div>
{[...new Set(SHAPE_PRESETS.map((preset) => preset.group))].map((group) => <div key={group}>
          <h4>{group}</h4>
          <div className="photo-buttons photo-shape-menu">
            {SHAPE_PRESETS.filter((preset) => preset.group === group).map((preset) =>
              <button key={preset.name} type="button"
                title={BALLOON_KINDS.includes(preset.shape.kind!)
                  ? "A balloon, with lettering already in it"
                  : holdsText(preset.shape.kind!) ? "A shape you can letter inside"
                    : "Goes behind the action"}
                onClick={() => addShapeLayer(preset.shape.kind!)}>{preset.name}</button>)}
          </div>
        </div>)}
        <p className="photo-note">
          {onClip
            ? "The plate is the clip's range. Add elements (Wan Alpha or imported), pictures and text over it, track a point to attach them, then Render to put the result back on the timeline."
            : "Add a picture from Project Media on the right, or drop in text. Each layer keeps its own steps and masks, so a cutout can be graded and relit to match the plate it landed on instead of being baked in."}
        </p>
      </>)}

      {document && activeLayer?.shape && card("shape", "Shape", activeLayer.shape.kind,
        <ShapeEditor shape={activeLayer.shape}
          fillPreview={shapePreviews?.fill} strokePreview={shapePreviews?.stroke}
          onChange={patchShape} />)}

      {document && activeLayer?.path && canvasSize && card("path", "Path",
        `${activeLayer.path.subpaths.length} subpath${activeLayer.path.subpaths.length === 1 ? "" : "s"}`, <PathEditor
          layer={activeLayer} layers={layers} busy={busy} envelopeEditing={envelopeEditing}
          onChange={patchPath}
          onCombine={(op) => pathAction(combinePathLayers(layers, activeLayer.id, op, canvasSize),
            op === "compound" ? "Compound path made" : `${op[0].toUpperCase()}${op.slice(1)} done`,
            "Needs a path layer below this one", pathLayerBelow(layers, activeLayer.id)?.id)}
          onOutlineStroke={() => pathAction(outlineLayerStroke(layers, activeLayer.id), "Stroke outlined into its own layer", "No stroke to outline")}
          onClipAbove={() => pathAction(clipAboveToPath(layers, activeLayer.id), "The layer above is clipped to this path", "No layer above")}
          onMaskBelow={() => pathAction(maskBelowWithPath(layers, activeLayer.id, canvasSize), "Vector mask added to the layer below; the path layer is hidden", "No layer below")}
          onBlend={(otherId, steps) => pathAction(blendPathLayers(layers, activeLayer.id, otherId, steps, canvasSize), `Blend made: ${steps} steps`, "Nothing to blend with")}
          onEnvelope={(on) => {
            patchPath({ envelope: on ? envelopeFor(activeLayer.path!).envelope : undefined }, on ? "Envelope added — drag its points" : "Envelope removed");
            setEnvelopeEditing(on);
            if (on) setTool("view");
          }}
          onEditEnvelope={(on) => { setEnvelopeEditing(on); if (on) setTool("view"); }} />)}

      {document && activeLayer && !activeLayer.gradient && card("repeat", "Repeat", activeLayer.repeat?.kind ?? "off", <RepeatEditor
        layer={activeLayer} layers={layers}
        onChange={(repeat, label, coalesce) => {
          patchLayer(activeLayer.id, { repeat }, coalesce ? undefined : { coalesce: false });
          setStatus(label);
        }} />)}

      {document && activeLayer?.text && card("text", "Text",
        activeLayer.text.fontFamily, <>
          <TextEditor text={activeLayer.text} fonts={fonts}
            previews={textPreviews} inShape={Boolean(activeLayer.shape)} onChange={patchText} />
          {!activeLayer.shape && <div className="photo-buttons">
            <button type="button" disabled={busy}
              title="Turn this text into a path with the same letters in the same place (reads the installed font)"
              onClick={convertTextToOutlines}><Spline size={12} /> Convert to outlines</button>
          </div>}
        </>)}

      {document && activeLayer?.gradient && card("gradient", "Gradient",
        `${activeLayer.gradient.stops.length} colours`, <GradientEditor
          gradient={activeLayer.gradient} preview={rampPreview} onChange={patchGradient} />)}

      {document && activeLayer && card("layer", "Layer", activeLayer.name, <LayerEditor
        layer={activeLayer} layers={layers} matching={false} onTrackMatte={makeTrackMatte}
        onChange={(over, label) => { patchLayer(activeLayer.id, over); setStatus(label); }}
        onMatchColour={runMatchColour} />)}

      {onClip && document && activeLayer?.media && card("timing", "Timing",
        `${activeLayer.media.loop ? "loops" : "plays once"} · from ${activeLayer.media.start}`, <MediaEditor
          media={activeLayer.media} rate={rate}
          onChange={(media, label, coalesce) => {
            patchLayer(activeLayer.id, { media }, coalesce ? undefined : { coalesce: false });
            setStatus(label);
          }}
          onSaveLoop={activeLayer.media.elementId ? () => {
            const media = activeLayer.media!;
            void guard("Saving loop", async () => {
              await updateElement({ id: media.elementId!, loopIn: media.inPoint, loopOut: media.outPoint, crossfade: media.loop ? media.crossfade : 0 });
              await refreshElements();
              setStatus("Loop saved as the element's default");
            });
          } : undefined} />)}

      {document && activeLayer?.generator?.kind === "matrixRain" && card("rain", "Matrix rain", `seed ${activeLayer.generator.seed}`, <MatrixRainEditor
        generator={activeLayer.generator}
        onChange={(generator, label, coalesce) => {
          patchLayer(activeLayer.id, { generator }, coalesce ? undefined : { coalesce: false });
          setStatus(label);
        }} />)}

      {onClip && document && (trackPlate || tracks.length > 0) && card("tracks", "Tracks", `${tracks.length} · ${trackEngine === "cotracker3" ? "CoTracker3" : "built-in"}`, <TracksPanel
        tracks={tracks} plate={trackPlate} activeLayer={activeLayer} frame={shownFrame} engine={trackEngine} motion={motion}
        busy={running} tracking={tool === "track"} sam2Sources={sam2Sources} occluded={occluded}
        onEngine={setTrackEngine}
        onStartTrack={() => {
          const on = tool === "track";
          setTool(on ? "view" : "track");
          if (!on) setStatus(`Click the point to track on ${trackPlate?.name ?? "the video"} (frame ${shownFrame})`);
        }}
        onSam2Centroid={runSam2Centroid} onRename={renameTrack} onDelete={deleteTrack} onAttach={setAttach}
        onComputeDepth={runVideoDepth} onOcclude={occludeByDepth} />)}

      {onClip && document && activeLayer?.generator?.kind === "emitter" && card("emitter", "Emitter",
        `${activeLayer.generator.rate.toFixed(2)}/frame · ${activeLayer.generator.lifetime} f`, <EmitterEditor
          generator={activeLayer.generator} tracks={tracks}
          onChange={(generator: EmitterGenerator, label, coalesce) => {
            patchLayer(activeLayer.id, { generator }, coalesce ? undefined : { coalesce: false });
            setStatus(label);
          }} />)}

      {onClip && document && card("elements", "Elements", `${elements.length} in the library`, <ElementsPanel
        elements={elements} busy={running === "Generating element" || running === "Importing element" ? running : undefined}
        onRefresh={() => void refreshElements()} onAdd={addElementLayer} onAddEmitter={addEmitterLayer}
        onImport={() => void runImportElement()} onGenerate={runGenerateElement} onBuildProxy={runBuildProxy} />)}

      {onClip && document && card("character", "Insert character", animate?.ready ? "Wan Animate 2" : "not set up", <CharacterPanel
        status={animate} images={images} videos={videos.filter((item) => item.id !== asset?.id)}
        mattes={sam2Sources.map((item) => ({ id: item.cachePath, label: item.label }))}
        plateName={clipPlate?.name} plateFrames={clipPlate ? drivingRange(clipPlate)?.count : undefined}
        elements={elements} busy={busy} onGenerate={runCharacter} onPlace={placeCharacter} onHideActor={runPaintActorOut} />)}

      {scopeDocument && card("nodes", "Stack", `${scopeName} · ${scopeDocument.nodes.length} steps`, <>
        <NodeList document={scopeDocument} selectedId={selectedNode?.id} skipped={skipped}
          onSelect={setSelectedNodeId} patch={patchScope} />
        <div className="photo-buttons">
          <button type="button" onClick={() => addNode(createGradeNode())}><Plus size={12} /> Grade</button>
          <button type="button" onClick={() => addNode(createRelightNode())}><Lightbulb size={12} /> Relight</button>
          <button type="button" title="Clone and heal, without a model run"
            onClick={() => { const node = createRetouchNode(); addNode(node); setSelectedNodeId(node.id); setTool("retouch"); }}>
            <Stamp size={12} /> Retouch
          </button>
          <button type="button" title="Push, bloat, pinch and twirl"
            onClick={() => { const node = createWarpNode(); addNode(node); setSelectedNodeId(node.id); setTool("warp"); }}>
            <Waves size={12} /> Liquify
          </button>
          <button type="button" title="Bend the pixels with a 4 × 4 mesh over this layer (envelope distort)"
            onClick={() => {
              const node = createEnvelopeNode(envelopeRect());
              addNode(node); setSelectedNodeId(node.id); setEnvelopeNodeEditing(true); setTool("view");
              setOpenCards((current) => ({ ...current, settings: true }));
            }}>
            <Grid3x3 size={12} /> Envelope
          </button>
          <button type="button" title="Re-colour by brightness — duotone and split-toning"
            onClick={() => { const node = createGradientMapNode(); addNode(node); setSelectedNodeId(node.id); }}>
            <Palette size={12} /> Gradient map
          </button>
          <button type="button" title="Stock footage shot on black: black becomes transparent, the glow stays"
            onClick={() => { const node = createUnmultiplyNode(); addNode(node); setSelectedNodeId(node.id); }}>
            <Plus size={12} /> Unmultiply
          </button>
          <button type="button" title="Keep what lies between two brightness levels"
            onClick={() => { const node = createLumaKeyNode(); addNode(node); setSelectedNodeId(node.id); }}>
            <Plus size={12} /> Luma key
          </button>
          <button type="button" title="Thicken or thin an element's alpha (Wan Alpha elements are mostly translucent)"
            onClick={() => { const node = createAlphaCurveNode(); addNode(node); setSelectedNodeId(node.id); }}>
            <Plus size={12} /> Alpha curve
          </button>
          {onClip && <button type="button" disabled={!tracks.length}
            title={tracks.length ? "Bend the layer's far end behind a track's motion (smoke trailing a moving hand)" : "Track a point first (Tracks card)"}
            onClick={() => {
              const emitterTrack = activeLayer?.generator?.kind === "emitter" ? activeLayer.generator.trackId : undefined;
              const trackId = activeLayer?.attach?.trackId ?? emitterTrack ?? tracks[tracks.length - 1]?.id;
              if (!trackId) return;
              const node = createTrailNode(trackId);
              addNode(node);
              setSelectedNodeId(node.id);
            }}><Plus size={12} /> Trail</button>}
          <button type="button" disabled={!activeLayer}
            title={activeLayer ? "Paint over what should go, then fill it"
              : "Remove works on one layer — select one first"}
            onClick={() => {
              // A fill needs a region, so rather than sitting disabled this makes one and drops the
              // user straight into the brush with it.
              const maskId = scopeDocument.masks.length ? scopeDocument.masks[0].id : createAndAttachMask();
              if (maskId) addNode(createRemoveNode(maskId));
            }}><Wand2 size={12} /> Remove</button>
        </div>
      </>)}

      {scopeDocument && selectedNode && card("settings",
        selectedNode.kind === "grade" ? "Grade" : selectedNode.kind === "relight" ? "Relight"
          : selectedNode.kind === "retouch" ? "Clone and heal"
            : selectedNode.kind === "warp" ? "Liquify"
              : selectedNode.kind === "gradientMap" ? "Gradient map"
                : selectedNode.kind === "unmultiply" ? "Unmultiply"
                  : selectedNode.kind === "lumaKey" ? "Luma key"
                    : selectedNode.kind === "alphaCurve" ? "Alpha curve"
                      : selectedNode.kind === "trail" ? "Trail"
                        : selectedNode.kind === "envelope" ? "Envelope"
                          : selectedNode.kind === "clipEffects" ? "Clip effects"
                            : selectedNode.kind === "maskVfx" ? "Mask VFX" : "Remove",
        skipped.get(selectedNode.id) ?? "active", <>
        <MaskPicker document={scopeDocument} node={selectedNode} patch={patchScope} />
        {selectedNode.kind === "envelope" && <EnvelopeNodeEditor node={selectedNode} editing={envelopeNodeEditing}
          onEdit={(on) => { setEnvelopeNodeEditing(on); if (on) setTool("view"); }}
          onChange={(node: EnvelopeNode, label) => {
            patchScope({ ...scopeDocument, nodes: scopeDocument.nodes.map((item) => item.id === node.id ? node : item) }, { coalesce: false });
            setStatus(label);
          }} />}
        {selectedNode.kind === "grade" && <GradeEditor document={scopeDocument} node={selectedNode}
          thumbBase={thumbBase} levels={levels} picking={tool === "picker" && picker === "whiteBalance"}
          patch={patchScope}
          onPickWhiteBalance={() => {
            const on = tool === "picker" && picker === "whiteBalance";
            setPicker(on ? undefined : "whiteBalance");
            setTool(on ? "view" : "picker");
          }} />}
        {selectedNode.kind === "relight" && <RelightEditor document={scopeDocument} node={selectedNode}
          patch={patchScope}
          depthReady={Boolean(runtime?.photoDepthReady)} depthState={depthState}
          onComputeDepth={() => computeDepth()} />}
        {selectedNode.kind === "retouch" && <RetouchEditor node={selectedNode} brush={retouchBrush}
          sampling={tool === "picker" && picker === "cloneSource"} hasSource={Boolean(retouchSource)}
          onBrush={(next) => { setRetouchBrush(next); setTool("retouch"); }}
          onSetSource={() => {
            const on = tool === "picker" && picker === "cloneSource";
            setPicker(on ? undefined : "cloneSource");
            setTool(on ? "retouch" : "picker");
          }}
          onUndo={() => patchScope({
            ...scopeDocument,
            nodes: scopeDocument.nodes.map((item) => item.id === selectedNode.id && item.kind === "retouch"
              ? { ...item, strokes: item.strokes.slice(0, -1) } : item),
          }, { coalesce: false })} />}
        {selectedNode.kind === "warp" && <WarpEditor node={selectedNode} brush={warpBrush}
          onBrush={(next) => { setWarpBrush(next); setTool("warp"); }}
          onUndo={() => patchScope({
            ...scopeDocument,
            nodes: scopeDocument.nodes.map((item) => item.id === selectedNode.id && item.kind === "warp"
              ? { ...item, strokes: item.strokes.slice(0, -1) } : item),
          }, { coalesce: false })}
          onClear={() => patchScope({
            ...scopeDocument,
            nodes: scopeDocument.nodes.map((item) => item.id === selectedNode.id && item.kind === "warp"
              ? { ...item, strokes: [] } : item),
          }, { coalesce: false })} />}
        {selectedNode.kind === "gradientMap" && <GradientMapEditor node={selectedNode}
          preview={mapPreview}
          onChange={(over, label, coalesce) => {
            patchScope({
              ...scopeDocument,
              nodes: scopeDocument.nodes.map((item) => item.id === selectedNode.id
                ? { ...item, ...over } as PhotoNode : item),
            }, coalesce ? undefined : { coalesce: false });
            setStatus(label);
          }} />}
        {selectedNode.kind === "alphaCurve" && <AlphaCurveEditor node={selectedNode}
          onChange={(over, label, coalesce) => {
            patchScope({
              ...scopeDocument,
              nodes: scopeDocument.nodes.map((item) => item.id === selectedNode.id ? { ...item, ...over } as PhotoNode : item),
            }, coalesce ? undefined : { coalesce: false });
            setStatus(label);
          }} />}
        {selectedNode.kind === "trail" && <TrailEditor node={selectedNode} tracks={tracks}
          onChange={(over, label, coalesce) => {
            patchScope({
              ...scopeDocument,
              nodes: scopeDocument.nodes.map((item) => item.id === selectedNode.id ? { ...item, ...over } as PhotoNode : item),
            }, coalesce ? undefined : { coalesce: false });
            setStatus(label);
          }} />}
        {selectedNode.kind === "lumaKey" && <LumaKeyEditor node={selectedNode}
          onChange={(over, label, coalesce) => {
            patchScope({
              ...scopeDocument,
              nodes: scopeDocument.nodes.map((item) => item.id === selectedNode.id ? { ...item, ...over } as PhotoNode : item),
            }, coalesce ? undefined : { coalesce: false });
            setStatus(label);
          }} />}
        {selectedNode.kind === "unmultiply" && <p className="photo-note">
          Alpha becomes the brightest channel and the colour is divided by it, so black goes fully
          transparent and a glow keeps its falloff. Composited normally it looks like Screen did.
        </p>}
        {selectedNode.kind === "remove" && <RemoveEditor document={scopeDocument} node={selectedNode}
          ready={Boolean(runtime?.photoInpaintReady)} running={running === "Filling"}
          filled={Boolean(selectedNode.cachePath)} onFill={() => runFill(selectedNode)} />}
      </>)}

      {scopeDocument && card("masks", "Masks", `${scopeName} · ${scopeDocument.masks.length}`, <MaskEditor
        document={scopeDocument} activeMaskId={activeMaskId} brush={brush}
        selectReady={selectReady}
        selecting={tool === "select"} points={points.length}
        patch={patchScope}
        onActivate={(id) => { setActiveMaskId(id); setActiveEntryId(undefined); setTool(id ? "brush" : "view"); }}
        onCreateMask={() => createAndAttachMask()}
        layers={layers} ownLayerId={activeLayer?.id}
        activeEntryId={activeEntryId}
        onActivateEntry={(id) => { setActiveEntryId(id); setTool("brush"); setShowMask(true); }}
        onImportMatte={() => { void importMatte(); }}
        trackedMattes={trackedMattes}
        picking={tool === "picker" && picker === "colour"}
        onPickColour={() => {
          const on = tool === "picker" && picker === "colour";
          setPicker(on ? undefined : "colour");
          setTool(on ? "brush" : "picker");
        }}
        onBrush={setBrush}
        onStartSelect={() => { setTool(tool === "select" ? "brush" : "select"); }}
        onRunSelect={() => runSelect()}
        onClearPoints={() => setPoints([])} />)}

      {document && card("geometry", "Geometry",
        isNeutralGeometry(geometryOf(document)) ? "upright" : "cropped", <>
        <div className="photo-buttons">
          <button type="button" onClick={() => patch({
            ...document, rotate: ((document.rotate + 90) % 360) as PhotoDocument["rotate"],
          }, { coalesce: false })}><RotateCw size={13} /> Rotate</button>
          <button type="button" className={document.flipHorizontal ? "selected" : ""}
            onClick={() => patch({ ...document, flipHorizontal: !document.flipHorizontal }, { coalesce: false })}>
            <FlipHorizontal size={13} /> Flip H
          </button>
          <button type="button" className={document.flipVertical ? "selected" : ""}
            onClick={() => patch({ ...document, flipVertical: !document.flipVertical }, { coalesce: false })}>
            <FlipVertical size={13} /> Flip V
          </button>
        </div>
        <Slider label="Straighten" value={document.straighten ?? 0} min={-45} max={45} step={0.1}
          format={(value) => `${value.toFixed(1)}°`}
          onChange={(value) => patch({ ...document, straighten: value })} />
        <Slider label="Scale" value={document.scale ?? 1} min={0.25} max={4} step={0.01}
          format={(value) => `${Math.round(value * 100)}%`}
          onChange={(value) => patch({ ...document, scale: value })} />

        <h4>Crop</h4>
        <div className="photo-buttons">
          <button type="button" className={cropAspect === undefined ? "selected" : ""}
            onClick={() => setCropAspect(undefined)}>Free</button>
          {CROP_PRESETS.map((preset) => <button key={preset.label} type="button"
            className={cropAspect === preset.ratio ? "selected" : ""}
            onClick={() => {
              setCropAspect(preset.ratio);
              setTool("crop");
              patch({ ...document, crop: cropForAspect(preset.ratio, document, canvasSize) },
                { coalesce: false });
            }}>{preset.label}</button>)}
          <button type="button" disabled={!document.crop}
            onClick={() => patch({ ...document, crop: undefined }, { coalesce: false })}>Reset</button>
        </div>
        {document.crop && <>
          <Slider label="Left" value={document.crop.x} min={0} max={0.95} step={0.002}
            format={(value) => `${Math.round(value * 100)}%`}
            onChange={(value) => patch({ ...document,
              crop: { ...document.crop!, x: value, width: Math.min(document.crop!.width, 1 - value) } })} />
          <Slider label="Top" value={document.crop.y} min={0} max={0.95} step={0.002}
            format={(value) => `${Math.round(value * 100)}%`}
            onChange={(value) => patch({ ...document,
              crop: { ...document.crop!, y: value, height: Math.min(document.crop!.height, 1 - value) } })} />
          <Slider label="Width" value={document.crop.width} min={0.05} max={1} step={0.002}
            format={(value) => `${Math.round(value * 100)}%`}
            onChange={(value) => patch({ ...document,
              crop: { ...document.crop!, width: Math.min(value, 1 - document.crop!.x) } })} />
          <Slider label="Height" value={document.crop.height} min={0.05} max={1} step={0.002}
            format={(value) => `${Math.round(value * 100)}%`}
            onChange={(value) => patch({ ...document,
              crop: { ...document.crop!, height: Math.min(value, 1 - document.crop!.y) } })} />
        </>}
        <h4>Behind the layers</h4>
        <div className="photo-buttons">
          {(["transparent", "black", "white"] as const).map((option) => <button key={option} type="button"
            className={(document.background ?? "transparent") === option ? "selected" : ""}
            onClick={() => patch({ ...document, background: option }, { coalesce: false })}>{option}</button>)}
        </div>
        <p className="photo-note">
          Straightening grows the frame, because the corners have to go somewhere. Crop takes it back.
        </p>
      </>)}

      {!onClip && card("output", "Save and export", exportFormat.toUpperCase(), <>
        <label className="studio-field"><span>Format</span>
          <select value={exportFormat} disabled={busy}
            onChange={(event) => setExportFormat(event.target.value as typeof exportFormat)}>
            <option value="png">PNG · lossless, keeps alpha</option>
            <option value="jpeg">JPEG · smallest, no alpha</option>
            <option value="webp">WebP · small, keeps alpha</option>
            <option value="tiff">TIFF · archival, keeps alpha</option>
          </select>
        </label>
        {exportFormat !== "png" && exportFormat !== "tiff" &&
          <Slider label="Quality" value={exportQuality} min={40} max={100}
            format={String} onChange={setExportQuality} />}
        {exportFormat === "jpeg" && <label className="studio-field"><span>Matte behind transparency</span>
          <input type="color" value={exportMatte} onChange={(event) => setExportMatte(event.target.value)} />
        </label>}
        <button className="studio-run" disabled={busy || !asset} onClick={runSave}>
          {busy ? <RefreshCw className="spin" size={15} /> : <ImageIcon size={15} />} Save to Project Media
        </button>
        <button disabled={busy || !asset} onClick={runSendToCanvas}>
          <Layers3 size={15} /> Save and add to Image Studio canvas
        </button>
        <button disabled={busy || !asset} onClick={() => void runExport()}>
          <Download size={15} /> Export to a file…
        </button>
        <p className="photo-note">
          Each of these composites every layer at full resolution into a new picture. The originals
          are never changed, and the document stays here to edit again.
        </p>
      </>)}
    </aside>

    <section className={`photo-stage${document && timed ? " timed" : ""}`}>
      <div className="photo-bar">
        <button type="button" className={tool === "view" ? "selected" : ""} onClick={() => setTool("view")}>
          <Hand size={13} /> View
        </button>
        <button type="button" className={tool === "move" ? "selected" : ""}
          disabled={!activeLayer || Boolean(activeLayer.gradient)}
          title={!activeLayer ? "Select a layer first"
            : activeLayer.gradient ? "A gradient is moved by its own handles on the picture"
              : "Drag the selected layer"}
          onClick={() => setTool("move")}>
          <Move size={13} /> Move
        </button>
        <button type="button" className={tool === "pen" ? "selected" : ""}
          title="Draw a vector path: click for corners, drag for curves, click the first point to close"
          onClick={() => setTool("pen")}>
          <PenTool size={13} /> Pen
        </button>
        <button type="button" className={tool === "crop" ? "selected" : ""}
          title="Drag the rectangle on the picture" onClick={() => setTool("crop")}>
          <Crop size={13} /> Crop
        </button>
        <button type="button" className={tool === "brush" ? "selected" : ""} disabled={!activeMask}
          title={activeMask ? "" : "Pick a mask first"} onClick={() => setTool("brush")}>
          <Sparkles size={13} /> {brush.erase ? "Erase" : "Paint"}
        </button>
        <button type="button" className={tool === "retouch" ? "selected" : ""}
          disabled={selectedNode?.kind !== "retouch"}
          title={selectedNode?.kind === "retouch" ? "Alt-click to set the source, then paint"
            : "Add a Retouch step first"}
          onClick={() => setTool("retouch")}>
          <Stamp size={13} /> {retouchBrush.heal ? "Heal" : "Clone"}
        </button>
        <button type="button" className={tool === "warp" ? "selected" : ""}
          disabled={selectedNode?.kind !== "warp"}
          title={selectedNode?.kind === "warp" ? "Drag to push the pixels around"
            : "Add a Liquify step first"}
          onClick={() => setTool("warp")}>
          <Waves size={13} /> Liquify
        </button>
        <button type="button" className={tool === "select" ? "selected" : ""}
          disabled={!activeMask || !selectReady}
          title={selectReady ? "" : "SAM 2 is not installed"}
          onClick={() => setTool("select")}>
          <Wand2 size={13} /> Click-select
        </button>
        <span className="photo-sep" />
        <button type="button" className={comparing ? "selected" : ""} disabled={!dirty}
          onPointerDown={() => setComparing(true)} onPointerUp={() => setComparing(false)}
          onPointerLeave={() => setComparing(false)}>
          <Contrast size={13} /> Compare
        </button>
        <button type="button" className={showMask ? "selected" : ""} disabled={!activeMask}
          title={activeMask ? "Tint the masked area on the picture" : "No mask selected"}
          onClick={() => setShowMask(!showMask)}>
          <Circle size={13} /> Mask
        </button>
        <button type="button" className={fit ? "selected" : ""} onClick={() => setFit(!fit)}>
          <Maximize2 size={13} /> {fit ? "Fit" : "100%"}
        </button>
        <span className="photo-spacer" />
        {running && <span className="photo-running"><RefreshCw className="spin" size={12} /> {running}…</span>}
        {canvasSize && <span className="photo-dims">{canvasSize.width} × {canvasSize.height}</span>}
        <button type="button" className={`photo-engine ${gpuCompositor ? "selected" : ""}`}
          onClick={() => { writeCompositorFlag(!gpuCompositor); setGpuCompositor(!gpuCompositor); }}>
          {/* Filled in by paint(), not by React, so the per-frame number costs no re-render. */}
          <span ref={frameTime} />
        </button>
        <span className={`photo-state ${dirty ? "edited" : ""}`}>
          {comparing ? "ORIGINAL" : dirty ? "EDITED" : "UNCHANGED"}
        </span>
      </div>

      {document && timed && <TimeBar frame={shownFrame} duration={duration} rate={rate} playing={playing}
        useProxy={useProxy} waiting={waitingFrames}
        hasProxy={layers.some((layer) => Boolean(layer.media?.proxyPath))}
        onFrame={(value) => { setPlaying(false); setCompFrame(value); }}
        onPlay={setPlaying} onProxy={setUseProxy}
        onDuration={(value) => patch({ ...document, duration: value })} />}

      <div className={`photo-viewport${fit ? " fit" : ""}`}>
        {!asset && <div className="photo-empty">
          <Sparkles size={26} />
          <strong>Nothing to edit yet</strong>
          <small>Generate a frame in Image Studio, or import an image, then pick one on the right.</small>
        </div>}
        {asset && !document && onClip && <div className="photo-empty">
          <Film size={26} />
          <strong>{asset.name} has not been probed yet</strong>
          <small>Its size and length are unknown. Play it once in the Video Editor, then open VFX again.</small>
        </div>}
        {asset && loading && <div className="photo-empty">
          <RefreshCw className="spin" size={22} /><strong>Reading {asset.name}…</strong>
        </div>}
        {asset && !loading && failure && <div className="photo-empty error"><strong>{failure}</strong></div>}
        <div className="photo-canvas-wrap" hidden={!asset || loading || Boolean(failure)}>
          <canvas ref={canvasRef} className={`tool-${tool}`}
            onPointerDown={onCanvasDown} onPointerMove={onCanvasMove}
            onPointerUp={onCanvasUp} onPointerCancel={onCanvasUp}
            onPointerLeave={() => { hoverAt.current = undefined; scheduleOverlay(); }} />
          <canvas ref={overlayRef} className="photo-mask-overlay" />
          {tool === "crop" && <CropOverlay
            rect={document?.crop ?? FULL_CROP} aspect={cropAspect}
            frameOf={() => canvasRef.current?.getBoundingClientRect()}
            onChange={(rect) => document && patch({ ...document, crop: rect })} />}

          {tool === "move" && activeLayer && !activeLayer.locked && (() => {
            const corners = layerFrame();
            if (!corners) return null;
            const top = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2] as Point;
            const centre = atDisplay(activeLayer.x, activeLayer.y);
            // The rotate grip sits beyond the top edge, on the line out from the centre, so it stays
            // outside the layer however far the layer has been turned.
            const away = Math.hypot(top[0] - centre[0], top[1] - centre[1]) || 1;
            const grip: Point = [
              top[0] + (top[0] - centre[0]) / away * 0.06,
              top[1] + (top[1] - centre[1]) / away * 0.06,
            ];
            return <>
              <svg className="photo-frame" viewBox="0 0 100 100" preserveAspectRatio="none">
                <polygon points={corners.map(([x, y]) => `${x * 100},${y * 100}`).join(" ")} />
              </svg>
              {corners.map((corner, index) => <Handle key={index} at={corner} className="corner"
                title="Drag to resize" onStart={beginFrame} onMove={dragCorner} />)}
              <Handle at={grip} className="rotate" title="Drag to rotate · hold shift to snap"
                onStart={beginFrame} onMove={dragRotate} />
            </>;
          })()}

          {activeLayer?.path && canvasSize && !activeLayer.locked && tool !== "crop"
            && (tool === "pen" || (envelopeEditing && activeLayer.path.envelope)) && <PathOverlay
            layer={activeLayer} canvas={canvasSize} atDisplay={atDisplay} toCanvas={toCanvas}
            envelope={envelopeEditing && Boolean(activeLayer.path.envelope)}
            drawing={penDrawing?.layerId === activeLayer.id ? penDrawing.subpath : undefined}
            onChange={(path, label, coalesce) => {
              patchLayer(activeLayer.id, { path }, coalesce ? undefined : { coalesce: false });
              setStatus(label);
            }}
            onTouch={setTouchedAnchor} onCloseDrawing={finishPen} />}

          {scopeDocument && selectedNode?.kind === "envelope" && envelopeNodeEditing && tool !== "crop" && <EnvelopeNodeOverlay
            node={selectedNode} atDisplay={atDisplay} toCanvas={toCanvas}
            onChange={(node) => patchScope({
              ...scopeDocument, nodes: scopeDocument.nodes.map((item) => item.id === node.id ? node : item),
            })} />}

          {handleShape?.kind === "linear" && tool !== "crop" && (() => {
            const shape = handleShape;
            return <>
              <svg className="photo-frame line" viewBox="0 0 100 100" preserveAspectRatio="none">
                <line
                  x1={atDisplay(shape.x1, shape.y1)[0] * 100} y1={atDisplay(shape.x1, shape.y1)[1] * 100}
                  x2={atDisplay(shape.x2, shape.y2)[0] * 100} y2={atDisplay(shape.x2, shape.y2)[1] * 100} />
              </svg>
              <Handle at={atDisplay(shape.x1, shape.y1)} className="gradient start"
                title="Where the gradient is empty" onStart={() => undefined}
                onMove={(event) => {
                  const at = toCanvas(event);
                  if (at) patchMaskShape({ ...shape, x1: at[0], y1: at[1] });
                }} />
              <Handle at={atDisplay(shape.x2, shape.y2)} className="gradient end"
                title="Where the gradient is full" onStart={() => undefined}
                onMove={(event) => {
                  const at = toCanvas(event);
                  if (at) patchMaskShape({ ...shape, x2: at[0], y2: at[1] });
                }} />
            </>;
          })()}

          {handleShape?.kind === "radial" && tool !== "crop" && canvasSize && (() => {
            const shape = handleShape;
            const long = Math.max(canvasSize.width, canvasSize.height);
            const aspect = shape.aspect || 1;
            // The two radii are stored as one radius and an aspect, so each handle sets its own
            // product and the pair is solved back out. Dragging one then leaves the other alone.
            const halfWidth = shape.radius * long * aspect / canvasSize.width;
            const halfHeight = shape.radius * long / aspect / canvasSize.height;
            const solve = (a: number, b: number) => ({
              radius: Math.max(0.005, Math.sqrt(Math.max(1e-9, a * b)) / long),
              aspect: Math.min(8, Math.max(0.125, Math.sqrt(Math.max(1e-9, a) / Math.max(1e-9, b)))),
            });
            return <>
              <Handle at={atDisplay(shape.cx, shape.cy)} className="gradient centre"
                title="Drag to move the centre" onStart={() => undefined}
                onMove={(event) => {
                  const at = toCanvas(event);
                  if (at) patchMaskShape({ ...shape, cx: at[0], cy: at[1] });
                }} />
              <Handle at={atDisplay(shape.cx + halfWidth, shape.cy)} className="gradient end"
                title="Drag to set the width" onStart={() => undefined}
                onMove={(event) => {
                  const at = toCanvas(event);
                  if (!at) return;
                  const across = Math.abs(at[0] - shape.cx) * canvasSize.width;
                  patchMaskShape({ ...shape, ...solve(across, shape.radius * long / aspect) });
                }} />
              <Handle at={atDisplay(shape.cx, shape.cy + halfHeight)} className="gradient end"
                title="Drag to set the height" onStart={() => undefined}
                onMove={(event) => {
                  const at = toCanvas(event);
                  if (!at) return;
                  const down = Math.abs(at[1] - shape.cy) * canvasSize.height;
                  patchMaskShape({ ...shape, ...solve(shape.radius * long * aspect, down) });
                }} />
            </>;
          })()}

          {activeLayer?.shape && tool !== "crop" && canvasSize
            && BALLOON_KINDS.includes(activeLayer.shape.kind)
            && activeLayer.shape.tailLength > 0.02 && (() => {
            /**
             * The tail's tip, where it actually lands on the picture.
             *
             * Read through the same functions that draw it -- the shape's own bounds and tip, then
             * the layer's matrix, then the document's geometry -- rather than reconstructed from the
             * angle. Dragging it therefore sets exactly the two numbers the outline is built from,
             * and the handle cannot drift away from the thing it moves.
             */
            const shape = activeLayer.shape;
            const bounds = shapeBounds(shape);
            const source = { width: bounds.width, height: bounds.height };
            const matrix = layerMatrix(activeLayer, canvasSize.width, canvasSize.height, source);
            const local = tailTip(shape);
            const [px, py] = applyMatrix(matrix, local[0] + bounds.offsetX, local[1] + bounds.offsetY);
            const at = atDisplay(px / canvasSize.width, py / canvasSize.height);
            return <Handle at={at} className="gradient end"
              title="Drag to point the tail · hold shift to snap"
              onStart={() => undefined}
              onMove={(event) => {
                const to = toCanvas(event);
                if (!to || activeLayer.locked) return;
                // Back through the layer's placement into the shape's own space, where the angle and
                // the length are measured from the body's centre.
                const inverse = invert(layerMatrix(activeLayer, canvasSize.width, canvasSize.height, source));
                const [lx, ly] = applyMatrix(inverse, to[0] * canvasSize.width, to[1] * canvasSize.height);
                const dx = lx - bounds.offsetX, dy = ly - bounds.offsetY;
                const distance = Math.hypot(dx, dy);
                if (distance < 1) return;
                const angle = (Math.atan2(dy, dx) * 180 / Math.PI + 360) % 360;
                const snapped = event.shiftKey ? Math.round(angle / 15) * 15 : Math.round(angle * 10) / 10;
                const body = Math.hypot(...bodyReach(shape, snapped));
                patchShape({
                  tailAngle: snapped,
                  tailLength: Math.min(2.5, Math.max(0, distance / Math.max(1, body) - 1)),
                }, "Tail", true);
              }} />;
          })()}

          {activeLayer?.gradient && tool !== "crop" && canvasSize && (() => {
            const ramp = activeLayer.gradient;
            const radians = ramp.angle * Math.PI / 180;
            const cos = Math.cos(radians), sin = Math.sin(radians);
            // The point the ramp reaches at its far end, in canvas pixels, using the same extent the
            // renderer measures -- so dragging the handle IS setting the number the fill uses.
            const along = Math.abs(canvasSize.width / 2 * cos) + Math.abs(canvasSize.height / 2 * sin);
            const reach = ramp.kind === "radial" || ramp.kind === "angular" || ramp.kind === "diamond"
              ? Math.hypot(canvasSize.width / 2, canvasSize.height / 2) * ramp.scale
              : along * ramp.scale;
            const centre: Point = [ramp.cx, ramp.cy];
            const end: Point = [
              ramp.cx + reach * cos / canvasSize.width,
              ramp.cy + reach * sin / canvasSize.height,
            ];
            // Clamp after the source-to-preview transform: crop/rotation can move an otherwise
            // in-bounds source point off the visible editor just as readily as an oversized ramp.
            const visibleCentre = visibleGradientHandle(atDisplay(centre[0], centre[1]));
            const visibleEnd = visibleGradientHandle(atDisplay(end[0], end[1]));
            const setFromEnd = (event: ReactPointerEvent<Element>) => {
              const at = toCanvas(event);
              if (!at) return;
              const dx = (at[0] - ramp.cx) * canvasSize.width;
              const dy = (at[1] - ramp.cy) * canvasSize.height;
              const distance = Math.hypot(dx, dy);
              if (distance < 1) return;
              const angle = (Math.atan2(dy, dx) * 180 / Math.PI + 360) % 360;
              const turned = Math.cos(angle * Math.PI / 180), tilted = Math.sin(angle * Math.PI / 180);
              const extent = ramp.kind === "radial" || ramp.kind === "angular" || ramp.kind === "diamond"
                ? Math.hypot(canvasSize.width / 2, canvasSize.height / 2)
                : Math.abs(canvasSize.width / 2 * turned) + Math.abs(canvasSize.height / 2 * tilted);
              patchGradient({
                // Shift snaps to fifteen degrees, which is what makes a level horizon reachable.
                angle: event.shiftKey ? Math.round(angle / 15) * 15 : Math.round(angle * 10) / 10,
                scale: Math.min(4, Math.max(0.05, distance / Math.max(1, extent))),
              }, "Direction", true);
            };
            return <>
              <svg className="photo-frame line" viewBox="0 0 100 100" preserveAspectRatio="none">
                <line
                  x1={visibleCentre.at[0] * 100} y1={visibleCentre.at[1] * 100}
                  x2={visibleEnd.at[0] * 100} y2={visibleEnd.at[1] * 100} />
              </svg>
              <Handle at={visibleCentre.at}
                className={`gradient centre${visibleCentre.clipped ? " clipped" : ""}`}
                title={visibleCentre.clipped ? "Gradient centre is outside the picture · drag this edge handle to bring it back" : "Drag to move where the ramp starts"} onStart={() => undefined}
                onMove={(event) => {
                  const at = toCanvas(event);
                  if (at) patchGradient({ cx: at[0], cy: at[1] }, "Centre", true);
                }} />
              <Handle at={visibleEnd.at}
                className={`gradient end${visibleEnd.clipped ? " clipped" : ""}`}
                title={visibleEnd.clipped ? "Gradient end is outside the picture · drag this edge handle to recover it" : "Drag to set the direction and spread · hold shift to snap"}
                onStart={() => undefined} onMove={setFromEnd} />
            </>;
          })()}

          {lightAt && selectedNode?.kind === "relight" && <div className="photo-light"
            style={{ left: `${lightAt[0] * 100}%`, top: `${lightAt[1] * 100}%` }}
            title="Drag to move the key light"
            onPointerDown={(event) => {
              event.currentTarget.setPointerCapture(event.pointerId);
              lightDrag.current = selectedNode.id;
            }}
            onPointerMove={(event) => {
              const canvas = canvasRef.current;
              if (!lightDrag.current || !document || !canvas || !previewSize || !scopeDocument) return;
              const box = canvas.getBoundingClientRect();
              const [u, v] = displayToSource(
                (event.clientX - box.left) / box.width,
                (event.clientY - box.top) / box.height,
                previewSize.width, previewSize.height, shownGeometry,
              );
              patchScope({
                ...scopeDocument,
                nodes: scopeDocument.nodes.map((node) => node.id === lightDrag.current && node.kind === "relight"
                  ? { ...node, relight: { ...node.relight, x: u, y: v } } : node),
              });
            }}
            onPointerUp={() => { lightDrag.current = undefined; }}
            onPointerCancel={() => { lightDrag.current = undefined; }} />}
          {points.map((point, index) => {
            const at = atDisplay(point.x, point.y);
            return <span key={index} className={`photo-point${point.include ? "" : " exclude"}`}
              style={{ left: `${at[0] * 100}%`, top: `${at[1] * 100}%` }} />;
          })}
        </div>
      </div>
      {failure && asset && !loading && <p className="photo-failure">{failure}</p>}
    </section>

    {onClip ? <aside className="image-studio-library">
      {/* The app's status bar sits under the full-window Clip Editor, so VFX mode repeats it here. */}
      {editorStatus && <p className="photo-note clip-vfx-status" role="status">{editorStatus}</p>}
      <div className="studio-library-section clip-vfx-render">
        <div className="studio-library-heading"><strong>Render</strong>
          <span>{renderPlan ? `${renderPlan.frames} f · ${renderPlan.replaceStart.toFixed(2)}–${renderPlan.replaceEnd.toFixed(2)}s` : "no plate"}</span></div>
        <p className="photo-note">
          Composes every frame at full size with the compositor you see here, and encodes an MP4
          (H.264, BT.709) with the clip's own sound for the same span. The original footage is never changed.
        </p>
        {!renderProgress && <div className="photo-buttons">
          <button type="button" disabled={busy || !document} onClick={() => setPlateRange(0, clipSpan, "the whole clip")}
            title="Put the whole clip on the plate">Whole clip</button>
          <button type="button" disabled={busy || !document || !clipTarget || clipTarget.rangeEnd <= clipTarget.rangeStart}
            onClick={() => setPlateRange(clipTarget!.rangeStart, clipTarget!.rangeEnd, "the Clip Editor range")}
            title="The range the Clip Editor opened with (from the playhead)">
            {clipTarget ? `${clipTarget.rangeStart.toFixed(2)}–${clipTarget.rangeEnd.toFixed(2)}s` : "Range"}</button>
        </div>}
        {renderProgress ? <>
          <div className="clip-vfx-progress"><div style={{ width: `${renderProgress.done / Math.max(1, renderProgress.total) * 100}%` }} /></div>
          <small className="photo-note">
            Frame {renderProgress.done} of {renderProgress.total}
            {renderProgress.fps ? ` · ${renderProgress.fps.toFixed(1)} frames/s` : ""}
          </small>
          <button type="button" onClick={() => { renderCancel.current = true; }}>Cancel render</button>
        </> : <button className="studio-run" type="button" disabled={busy || !renderPlan || !canvasSize || loading}
          onClick={runRender}>
          {running === "Rendering" ? <RefreshCw className="spin" size={15} /> : <Film size={15} />} Render VFX
        </button>}
        {rendered && !renderProgress && <>
          <video className="clip-vfx-result" src={convertFileSrc(rendered.asset.path)} controls preload="metadata" />
          <button className="studio-run" type="button" onClick={() => placeRender("replace")}
            title="The clip keeps its head and tail; this range now plays the render">
            Replace {rendered.replaceStart.toFixed(2)}–{rendered.replaceEnd.toFixed(2)}s on the timeline
          </button>
          <button type="button" onClick={() => placeRender("insert")}>Insert after the clip</button>
          <small className="photo-note">Saved in Project Media as {rendered.asset.name}.</small>
        </>}
      </div>
      <div className="studio-library-section clip-vfx-render">
        <div className="studio-library-heading"><strong>Spritesheet</strong><span>{duration} f</span></div>
        <SpritesheetPanel layers={layers.filter((layer) => !layer.locked).map((layer) => ({ id: layer.id, name: layer.name }))}
          frames={duration} busy={busy || Boolean(renderProgress) || !document} onExport={runSpritesheet} />
      </div>
      <div className="studio-library-section">
        <div className="studio-library-heading"><strong>Pictures</strong><span>{images.length}</span></div>
        <div className="photo-nodes">
          {images.map((item) => <div key={item.id} className="photo-node">
            <button className="photo-node-main" type="button" disabled={!document}
              title="Add as a picture layer over the plate" onClick={() => addLayerFrom(item)}>
              <i><ImageIcon size={12} /></i>
              <span><strong>{item.name}</strong><small>{item.width && item.height ? `${item.width} × ${item.height}` : "Image"} · add as layer</small></span>
            </button>
          </div>)}
          {!images.length && <p className="photo-note">No pictures in Project Media.</p>}
        </div>
      </div>
      {videos.length > 1 && <div className="studio-library-section">
        <div className="studio-library-heading"><strong>Other video</strong><span>{videos.length - 1}</span></div>
        <div className="photo-nodes">
          {videos.filter((item) => item.id !== asset?.id).map((item) => <div key={item.id} className="photo-node">
            <button className="photo-node-main" type="button" disabled={!document}
              title="Add as a video layer at the current frame (stock footage over the plate)"
              onClick={() => addVideoLayer(item)}>
              <i><Film size={12} /></i>
              <span><strong>{item.name}</strong><small>{item.width && item.height ? `${item.width} × ${item.height}` : "not probed"} · add as layer</small></span>
            </button>
          </div>)}
        </div>
      </div>}
      <div className="studio-library-section">
        <button className="photo-inline wide" type="button" disabled={Boolean(running)} onClick={clipTarget!.onClose}>
          Close the Clip Editor
        </button>
      </div>
    </aside> : <aside className="image-studio-library">
      <div className="studio-library-section project-images">
        <div className="studio-library-heading"><strong>Project Media</strong><span>{images.length} image{images.length === 1 ? "" : "s"}</span></div>
        <div className="studio-media-grid">
          {images.map((item) => <article key={item.id} className={item.id === asset?.id ? "selected" : ""}
            onDoubleClick={() => setRenamingId(item.id)}>
            <img src={convertFileSrc(item.path)} alt="" />
            <RenameField value={item.name} editing={renamingId === item.id}
              onEditing={(on) => setRenamingId(on ? item.id : undefined)}
              onCommit={(next) => renameAsset(item.id, next)} />
            <small>
              {item.width && item.height ? `${item.width} × ${item.height}` : "Image"}
              {project.photoEdits?.[item.id] ? " · edited" : ""}
            </small>
            <div>
              <button onClick={() => setPhotoEditAsset(item.id)}><Wand2 size={11} /> Open</button>
              <button disabled={!document} title="Add as a layer on this canvas"
                onClick={() => addLayerFrom(item)}><Layers3 size={11} /> Layer</button>
              <button onClick={() => addAssetToTimeline(item.id)}>+ Timeline</button>
            </div>
          </article>)}
          {!images.length && <p className="photo-note">
            No images yet. Generate one in Image Studio, or import media in the Video Editor.
          </p>}
        </div>
      </div>
      <div className="studio-library-section">
        <button className="photo-inline wide" onClick={() => setActiveWorkspace("image")}>
          Back to Image Studio
        </button>
        <p className="photo-note">
          Video VFX (elements, tracking, trails, emitters, depth occlusion) lives in the Clip Editor:
          select a clip on the timeline, open the Clip Editor and choose VFX.
        </p>
      </div>
    </aside>}
  </section>;
}

/* ------------------------------------------------------------------ helpers */

const errorText = (error: unknown) => error instanceof Error ? error.message : String(error);
const signed = (value: number) => `${value > 0 ? "+" : ""}${value}`;

const CROP_PRESETS = [
  { label: "1:1", ratio: 1 }, { label: "4:5", ratio: 4 / 5 }, { label: "3:2", ratio: 3 / 2 },
  { label: "16:9", ratio: 16 / 9 }, { label: "9:16", ratio: 9 / 16 },
] as const;

/** How far the shape's body reaches in one direction, which is what a tail length is measured against. */
function bodyReach(shape: ShapeBlock, degrees: number): [number, number] {
  return tailTip({ ...shape, tailLength: 0, tailAngle: degrees });
}


/**
 * The strip-everything view used by hold-to-compare.
 *
 * Node stacks come off, placement and blending stay. "Before" in a compositing tool means before the
 * adjustments, not before the composite -- showing the bottom picture alone would answer a question
 * nobody asked.
 */
function bypassed(document: PhotoDocument): PhotoDocument {
  return {
    ...document,
    nodes: [],
    layers: layersOf(document).map((layer) => ({ ...layer, nodes: [] })),
  };
}

/**
 * The biggest rectangle of the given aspect that fits the straightened frame, centred.
 *
 * Against the framed size rather than the canvas, because straightening changes the shape the crop
 * has to sit inside -- picking a 16:9 crop on a picture rotated four degrees should give 16:9 of
 * what is on screen, not of what the canvas used to be.
 */
function cropForAspect(
  ratio: number, document: PhotoDocument, source: { width: number; height: number } | undefined,
): CropRect {
  if (!source) return { x: 0, y: 0, width: 1, height: 1 };
  const framed = framedSize(source.width, source.height, geometryOf(document));
  const frameRatio = framed.width / framed.height;
  const width = ratio > frameRatio ? 1 : ratio / frameRatio;
  const height = ratio > frameRatio ? frameRatio / ratio : 1;
  return { x: (1 - width) / 2, y: (1 - height) / 2, width, height };
}

function canvasOf(width: number, height: number): HTMLCanvasElement {
  const canvas = window.document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}


/**
 * A canvas-space picture drawn back into a layer's own pixels.
 *
 * The exact inverse of the matrix the layer is placed with, so a mask painted on screen reaches the
 * worker registered to the file it will be run against.
 */
function unplace(
  source: HTMLCanvasElement, layer: PhotoLayer,
  canvas: { width: number; height: number }, target: LayerSource,
): HTMLCanvasElement {
  const out = canvasOf(target.width, target.height);
  const context = out.getContext("2d")!;
  context.fillStyle = "#000";
  context.fillRect(0, 0, target.width, target.height);
  const rect = croppedRect(layer, target);
  const inverse = invert(layerMatrix(layer, canvas.width, canvas.height, target));
  const matrix = compose(translate(rect.left, rect.top), inverse);
  context.save();
  context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f);
  context.drawImage(source, 0, 0);
  context.restore();
  return out;
}

/** A canvas-normalised point expressed in a layer's own pixels, for a worker that sees only the file. */
function toLayerPoint(
  x: number, y: number, layer: PhotoLayer | undefined,
  sources: Map<string, LayerSource>, canvas: { width: number; height: number } | undefined,
): [number, number] {
  const source = layer ? sources.get(layer.sourcePath) : undefined;
  if (!layer || !source || !canvas) return [x, y];
  const rect = croppedRect(layer, source);
  const inverse = invert(layerMatrix(layer, canvas.width, canvas.height, source));
  const [px, py] = applyMatrix(inverse, x * canvas.width, y * canvas.height);
  return [(px + rect.left) / source.width, (py + rect.top) / source.height];
}

/**
 * Draw `source` into `target` through the document's geometry, resizing the target to match.
 *
 * Geometry is applied here rather than baked into the pixel buffer: rotating the working copy on
 * every change would mean re-deriving it constantly, and the spatial passes in the grade are
 * symmetric, so blurring before a quarter-turn and after it give the same picture. The transform is
 * the same matrix the pointer mapping inverts -- see `domain/photoGeometry.ts` for why that matters.
 */
/**
 * The status line a finished render leaves: where each frame's time went, so the next slow render
 * names its bottleneck. Decode runs beside compose and send (pipelined), so a frame's time is about
 * the largest of decode, compose + blocked, and send -- not their sum. Exported for its test.
 */
export function renderTimingLine(info: {
  name: string; frames: number; seconds: number; gpu: boolean;
  spent: { decode: number; wait: number; compose: number; pack?: number; send: number; blocked: number };
  encoder?: string; encoderSeconds?: number; fedSamples: number; resets: number;
  /** How frames travelled (`FrameSender.describe()`), e.g. "HTTP as YUV 4:2:0". */
  transport?: string;
}): string {
  const per = (value: number) => Math.round(value / Math.max(1, info.frames));
  const { spent } = info;
  const encoder = info.encoderSeconds !== undefined ? `, ${info.encoder ?? "ffmpeg"} ${per(info.encoderSeconds * 1000)} ms of it` : "";
  const decoder = info.fedSamples > 0
    ? ` · decoder fed ${info.fedSamples} samples for ${info.frames} frames, ${info.resets} reset${info.resets === 1 ? "" : "s"}` : "";
  return `Rendered ${info.name} · ${info.frames} frames in ${info.seconds.toFixed(1)} s (${(info.frames / Math.max(0.001, info.seconds)).toFixed(1)} fps; `
    + `per frame: decode ${per(spent.decode)} ms (waited ${per(spent.wait)}), ${info.gpu ? "GPU" : "CPU"} compose ${per(spent.compose)} ms, `
    + (spent.pack !== undefined ? `pack ${per(spent.pack)} ms, ` : "")
    + `send ${per(spent.send)} ms${info.transport ? ` over ${info.transport}` : ""} (blocked ${per(spent.blocked)}${encoder})${decoder}) — replace the range or insert it after the clip`;
}

/**
 * Compositor pixels as the bytes a render sends to ffmpeg, which drops alpha. Fully transparent pixels
 * become black, as the canvas round trip this replaced made them (getImageData unpremultiplies a
 * zero-alpha pixel to 0,0,0); every other pixel is passed through unchanged. Exported for its test.
 */
export function encodableRgba(pixels: Uint8ClampedArray): Uint8Array {
  const bytes = new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength);
  for (let i = 3; i < bytes.length; i += 4) {
    if (bytes[i] === 0) { bytes[i - 3] = 0; bytes[i - 2] = 0; bytes[i - 1] = 0; }
  }
  return bytes;
}

function drawOriented(target: HTMLCanvasElement, source: HTMLCanvasElement, geometry: PhotoGeometry): void {
  const size = outputSize(source.width, source.height, geometry);
  target.width = size.width;
  target.height = size.height;
  const matrix = forwardMatrix(source.width, source.height, geometry);
  const context = target.getContext("2d")!;
  context.clearRect(0, 0, target.width, target.height);
  context.save();
  context.setTransform(matrix.a, matrix.b, matrix.c, matrix.d, matrix.e, matrix.f);
  context.drawImage(source, 0, 0);
  context.restore();
}

/**
 * A brush outline, drawn dark-then-light so it reads on any picture.
 *
 * Two strokes rather than one: a white ring vanishes on a white shirt and a black one vanishes in a
 * shadow, and a brush you cannot see is the problem this exists to solve.
 */
function ring(
  context: CanvasRenderingContext2D, x: number, y: number, radius: number,
  line: number, dashed = false,
): void {
  context.save();
  context.setLineDash(dashed ? [radius * 0.35, radius * 0.25] : []);
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.strokeStyle = "rgba(0,0,0,.65)";
  context.lineWidth = line * 2.4;
  context.stroke();
  context.strokeStyle = dashed ? "#ffcf33" : "rgba(255,255,255,.95)";
  context.lineWidth = line;
  context.stroke();
  context.restore();
}

/** The exact point a clone is copying from, which a ring alone does not tell you. */
function crosshair(
  context: CanvasRenderingContext2D, x: number, y: number, arm: number, line: number,
): void {
  context.save();
  context.beginPath();
  context.moveTo(x - arm, y); context.lineTo(x + arm, y);
  context.moveTo(x, y - arm); context.lineTo(x, y + arm);
  context.strokeStyle = "rgba(0,0,0,.65)";
  context.lineWidth = line * 2.4;
  context.stroke();
  context.strokeStyle = "#ffcf33";
  context.lineWidth = line;
  context.stroke();
  context.restore();
}

/** Cheap equality for the skipped map, so an unchanged report does not re-render the panel. */
function sameKeys(left: Map<string, string>, right: Map<string, string>): boolean {
  if (left.size !== right.size) return false;
  for (const [key, value] of left) if (right.get(key) !== value) return false;
  return true;
}

/** A short description of the document, stored on the new asset so its lineage is not a mystery. */
function describe(document: PhotoDocument): string {
  const counts = new Map<string, number>();
  const layers = layersOf(document);
  for (const node of [...document.nodes, ...layers.flatMap((layer) => layer.nodes)]) {
    if (node.enabled) counts.set(node.kind, (counts.get(node.kind) ?? 0) + 1);
  }
  const parts: string[] = [];
  if (layers.length > 1) parts.push(`${layers.length} layers`);
  const captions = layers.filter((layer) => layer.text).length;
  if (captions) parts.push(`${captions} text`);
  parts.push(...[...counts].map(([kind, count]) => `${count} ${kind}`));
  if (document.rotate) parts.push(`${document.rotate}° rotated`);
  return parts.length ? `Photo Edit · ${parts.join(", ")}` : "Photo Edit";
}
