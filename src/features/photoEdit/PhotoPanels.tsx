import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import {
  AlignCenter, AlignLeft, AlignRight, ArrowDown, ArrowUp, Circle, Eraser, Eye, EyeOff,
  Image as ImageIcon, Italic, Layers3, Lightbulb, Lock, LockOpen, Pipette, Plus, RefreshCw, Stamp,
  Palette, Sliders, Trash2, Type, Wand2, Waves, Contrast, KeyRound, FileUp, Scissors,
  Play, Pause, Film, FolderOpen, SkipBack, Repeat,
  Wind, Grid3x3, PenTool,
} from "lucide-react";
import {
  HSL_BAND_NAMES, PHOTO_FILTERS, applyAdjustments, buildCurveLut,
  type CurvePoint, type HslBand, type PhotoAdjustments,
} from "../../domain/photoEdit";
import type {
  GradeNode, MaskShape, PhotoDocument, PhotoLayer, PhotoMask, PhotoNode, RelightNode,
  GradientMapNode, RelightSettings, RemoveNode, RetouchNode, WarpNode,
  LumaKeyNode, MaskEntry, MaskMode, MaskSource,
  AlphaCurveNode, LayerMedia, MatrixRainGenerator,
} from "../../domain/photoDocument";
import { WAN_ALPHA_PRESETS } from "../../domain/elements";
import type { ElementEntry } from "../../services/elementService";
import { MASK_MODES, createMaskEntry } from "../../domain/photoDocument";
import type { WarpMode } from "../../domain/photoWarp";
import {
  GRADIENT_PRESETS, NEUTRAL_GRADIENT, buildGradientLut, normalizeStops, paintColor, rgbToHex,
  solidPaint, toPaint,
  type Gradient, type GradientKind, type GradientStop, type Paint,
} from "../../domain/photoGradient";
import {
  BALLOON_KINDS, SHAPE_PRESETS, type ShapeBlock, type ShapeKind,
} from "../../domain/photoShape";
import { layersOf } from "../../domain/photoDocument";
import { BLEND_MODES } from "../../domain/photoBlend";
import { COMPOSITE_SCOPE } from "../../domain/photoCompose";
import { NEUTRAL_EDGE, type LayerEdge } from "../../domain/photoEdge";
import type { TextAlign, TextBlock } from "../../domain/photoText";
import type { FontGroup } from "../../services/fonts";

/**
 * The card contents for Photo Edit. Presentational on purpose: every one of these takes the
 * document and a way to change it, and none of them knows how a picture is rendered or saved. That
 * keeps the workspace file about pixels and these about controls.
 */

export type Patch = (next: PhotoDocument, options?: { coalesce?: boolean }) => void;

/* ------------------------------------------------------------------ shared */

export function Slider({ label, value, min, max, step = 1, format, disabled, onChange }: {
  label: string; value: number; min: number; max: number; step?: number;
  format: (value: number) => string; disabled?: boolean; onChange: (value: number) => void;
}) {
  return <div className="photo-ctl">
    <span>{label}</span>
    <input type="range" min={min} max={max} step={step} value={value} disabled={disabled}
      onChange={(event) => onChange(Number(event.target.value))} />
    <ValueBox value={value} min={min} max={max} step={step} format={format}
      disabled={disabled} onCommit={onChange} />
  </div>;
}

/**
 * The number beside a slider, typed into directly.
 *
 * A slider is quick and imprecise; some values you know exactly and want to state rather than hunt
 * for. So the readout is the input: it shows the formatted value at rest, swaps to the bare number
 * the moment it has focus so there are no units in the way of typing, and commits on blur or Enter.
 * Escape puts back what was there.
 *
 * Out-of-range entries are clamped rather than refused. Someone typing 400 into a control that stops
 * at 200 means "as far as it goes", and refusing them with no explanation is worse than obliging.
 */
function ValueBox({ value, min, max, step, format, disabled, onCommit }: {
  value: number; min: number; max: number; step: number;
  format: (value: number) => string; disabled?: boolean; onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState<string>();
  const editing = draft !== undefined;

  const commit = (text: string | undefined) => {
    setDraft(undefined);
    if (text === undefined) return;
    const parsed = Number.parseFloat(text.replace(/[^0-9eE.+-]/g, ""));
    if (!Number.isFinite(parsed)) return;
    const clamped = Math.min(max, Math.max(min, parsed));
    // Snap to the slider's own step so the handle lands on a position it can actually return to.
    const snapped = step > 0 ? Math.round(clamped / step) * step : clamped;
    const settled = Number(snapped.toFixed(6));
    if (settled !== value) onCommit(settled);
  };

  return <input
    className={`photo-value${editing ? " editing" : ""}`}
    value={editing ? draft : format(value)}
    disabled={disabled}
    inputMode="decimal"
    spellCheck={false}
    aria-label="Value"
    onFocus={(event) => { setDraft(String(value)); event.currentTarget.select(); }}
    onChange={(event) => setDraft(event.target.value)}
    onBlur={(event) => commit(event.target.value)}
    onKeyDown={(event) => {
      event.stopPropagation();
      if (event.key === "Enter") { event.preventDefault(); commit(event.currentTarget.value); event.currentTarget.blur(); }
      else if (event.key === "Escape") { event.preventDefault(); setDraft(undefined); event.currentTarget.blur(); }
    }} />;
}

const signed = (value: number) => `${value > 0 ? "+" : ""}${value}`;

/* ------------------------------------------------------------------- nodes */

const NODE_ICON: Record<PhotoNode["kind"], ReactNode> = {
  grade: <Sliders size={13} />,
  relight: <Lightbulb size={13} />,
  remove: <Eraser size={13} />,
  retouch: <Stamp size={13} />,
  warp: <Waves size={13} />,
  gradientMap: <Palette size={13} />,
  unmultiply: <Contrast size={13} />,
  lumaKey: <KeyRound size={13} />,
  alphaCurve: <Contrast size={13} />,
  trail: <Wind size={13} />,
  envelope: <Grid3x3 size={13} />,
  clipEffects: <Sliders size={13} />,
  maskVfx: <Wind size={13} />,
};

const nodeLabel = (node: PhotoNode, masks: PhotoMask[]): string => {
  if (node.kind === "grade") return "Grade";
  if (node.kind === "relight") return "Relight";
  if (node.kind === "retouch") {
    return node.strokes.length ? `Retouch · ${node.strokes.length} strokes` : "Retouch";
  }
  if (node.kind === "warp") {
    return node.strokes.length ? `Liquify · ${node.strokes.length} strokes` : "Liquify";
  }
  if (node.kind === "gradientMap") {
    return `Gradient map · ${node.gradient.stops.length} colours`;
  }
  if (node.kind === "unmultiply") return "Unmultiply (black to alpha)";
  if (node.kind === "alphaCurve") return `Alpha curve · gain ${node.gain.toFixed(2)}, gamma ${node.gamma.toFixed(2)}`;
  if (node.kind === "lumaKey") return `Luma key · ${node.low}–${node.high}${node.invert ? " inverted" : ""}`;
  if (node.kind === "trail") return `Trail · ${node.maxLag} frames behind`;
  if (node.kind === "envelope") return `Envelope · ${node.envelope.cols + 1} × ${node.envelope.rows + 1} points`;
  if (node.kind === "clipEffects") return "Clip effects";
  if (node.kind === "maskVfx") return `Mask VFX · ${node.recipe.name}`;
  return `Remove · ${masks.find((mask) => mask.id === node.maskId)?.name ?? "no mask"}`;
};

/**
 * The stack, top of the list being the first thing applied.
 *
 * Shown in document order rather than reversed. A layer list reads bottom-up because layers sit on
 * top of one another; a processing chain reads top-down because each step feeds the next, and
 * showing it the other way round is how people end up thinking their curve runs before their
 * exposure.
 */
export function NodeList({ document, selectedId, skipped, onSelect, patch }: {
  document: PhotoDocument; selectedId?: string;
  skipped: Map<string, string>;
  onSelect: (id: string) => void; patch: Patch;
}) {
  const move = (index: number, delta: number) => {
    const target = index + delta;
    if (target < 0 || target >= document.nodes.length) return;
    const nodes = [...document.nodes];
    [nodes[index], nodes[target]] = [nodes[target], nodes[index]];
    patch({ ...document, nodes }, { coalesce: false });
  };
  return <div className="photo-nodes">
    {document.nodes.map((node, index) => <div key={node.id}
      className={`photo-node${node.id === selectedId ? " selected" : ""}${node.enabled ? "" : " off"}`}>
      <button className="photo-node-main" type="button" onClick={() => onSelect(node.id)}>
        <i>{NODE_ICON[node.kind]}</i>
        <span>
          <strong>{nodeLabel(node, document.masks)}</strong>
          <small>
            {node.maskId ? document.masks.find((mask) => mask.id === node.maskId)?.name ?? "missing mask" : "whole picture"}
            {skipped.has(node.id) ? ` · ${skipped.get(node.id)}` : ""}
          </small>
        </span>
      </button>
      <i className="photo-node-act" title={node.enabled ? "Disable" : "Enable"}
        onClick={() => patch({
          ...document,
          nodes: document.nodes.map((item) => item.id === node.id ? { ...item, enabled: !item.enabled } : item),
        }, { coalesce: false })}>{node.enabled ? <Eye size={13} /> : <EyeOff size={13} />}</i>
      <i className="photo-node-act" title="Move earlier" onClick={() => move(index, -1)}><ArrowUp size={12} /></i>
      <i className="photo-node-act" title="Move later" onClick={() => move(index, 1)}><ArrowDown size={12} /></i>
      <i className="photo-node-act" title="Delete" onClick={() => patch({
        ...document, nodes: document.nodes.filter((item) => item.id !== node.id),
      }, { coalesce: false })}><Trash2 size={12} /></i>
    </div>)}
    {!document.nodes.length && <p className="photo-note">Empty stack. Add a step below.</p>}
  </div>;
}

/** Which mask limits the selected node. */
export function MaskPicker({ document, node, patch }: {
  document: PhotoDocument; node: PhotoNode; patch: Patch;
}) {
  const locked = node.kind === "remove";
  return <label className="studio-field">
    <span>{locked ? "Region to fill" : "Limit to mask"}</span>
    <select value={node.maskId ?? ""} onChange={(event) => patch({
      ...document,
      nodes: document.nodes.map((item) => item.id === node.id
        ? { ...item, maskId: event.target.value || undefined } as PhotoNode : item),
    }, { coalesce: false })}>
      {!locked && <option value="">Whole picture</option>}
      {locked && !node.maskId && <option value="">Pick a mask</option>}
      {document.masks.map((mask) => <option key={mask.id} value={mask.id}>{mask.name}</option>)}
    </select>
  </label>;
}

/* ------------------------------------------------------------------- grade */

export function GradeEditor({ document, node, thumbBase, levels, picking, patch, onPickWhiteBalance }: {
  document: PhotoDocument; node: GradeNode; thumbBase?: ImageData;
  levels?: HistogramData; picking: boolean;
  patch: Patch; onPickWhiteBalance: () => void;
}) {
  const set = <K extends keyof PhotoAdjustments>(key: K, value: PhotoAdjustments[K], coalesce = true) =>
    patch({
      ...document,
      nodes: document.nodes.map((item) => item.id === node.id
        ? { ...item, adjustments: { ...(item as GradeNode).adjustments, [key]: value } } : item),
    }, { coalesce });
  const adjustments = node.adjustments;
  const [band, setBand] = useState(0);
  const bands = adjustments.hsl ?? [];
  const activeBand: HslBand = bands[band] ?? { hue: 0, saturation: 0, luminance: 0 };
  const setBandValue = (key: keyof HslBand, value: number) => set("hsl",
    bands.map((entry, index) => index === band ? { ...entry, [key]: value } : entry));

  return <>
    <Histogram data={levels} />
    <h4>Tone</h4>
    <Slider label="Exposure" value={adjustments.exposure} min={-3} max={3} step={0.01}
      format={(value) => `${value > 0 ? "+" : ""}${value.toFixed(2)} EV`} onChange={(value) => set("exposure", value)} />
    <Slider label="Contrast" value={adjustments.contrast} min={-80} max={80} format={signed}
      onChange={(value) => set("contrast", value)} />
    <Slider label="Highlights" value={adjustments.highlights} min={-100} max={100} format={signed}
      onChange={(value) => set("highlights", value)} />
    <Slider label="Shadows" value={adjustments.shadows} min={-100} max={100} format={signed}
      onChange={(value) => set("shadows", value)} />
    <h4>Levels</h4>
    <Slider label="Black point" value={adjustments.levelsBlack ?? 0} min={0} max={250} format={String}
      onChange={(value) => set("levelsBlack", value)} />
    <Slider label="White point" value={adjustments.levelsWhite ?? 255} min={5} max={255} format={String}
      onChange={(value) => set("levelsWhite", value)} />
    <Slider label="Gamma" value={adjustments.levelsGamma ?? 1} min={0.1} max={4} step={0.01}
      format={(value) => value.toFixed(2)} onChange={(value) => set("levelsGamma", value)} />
    <h4>Colour</h4>
    <button className={`photo-inline wide${picking ? " selected" : ""}`} type="button"
      onClick={onPickWhiteBalance}>
      <Pipette size={12} /> {picking ? "Click something neutral…" : "White balance from a grey"}
    </button>
    <Slider label="Temperature" value={adjustments.temperature} min={-100} max={100} format={signed}
      onChange={(value) => set("temperature", value)} />
    <Slider label="Tint" value={adjustments.tint} min={-100} max={100} format={signed}
      onChange={(value) => set("tint", value)} />
    <Slider label="Saturation" value={adjustments.saturation} min={0} max={200} format={String}
      onChange={(value) => set("saturation", value)} />
    <Slider label="Vibrance" value={adjustments.vibrance} min={-100} max={100} format={signed}
      onChange={(value) => set("vibrance", value)} />
    <h4>Colour bands</h4>
    <div className="photo-bands">
      {HSL_BAND_NAMES.map((name, index) => {
        const entry = bands[index];
        const touched = entry && (entry.hue || entry.saturation || entry.luminance);
        return <button key={name} type="button"
          className={`${index === band ? "selected" : ""}${touched ? " touched" : ""}`}
          onClick={() => setBand(index)}>{name}</button>;
      })}
    </div>
    <Slider label="Hue" value={activeBand.hue} min={-100} max={100} format={signed}
      onChange={(value) => setBandValue("hue", value)} />
    <Slider label="Saturation" value={activeBand.saturation} min={-100} max={100} format={signed}
      onChange={(value) => setBandValue("saturation", value)} />
    <Slider label="Luminance" value={activeBand.luminance} min={-100} max={100} format={signed}
      onChange={(value) => setBandValue("luminance", value)} />
    <h4>Vignette</h4>
    <Slider label="Amount" value={adjustments.vignette ?? 0} min={-100} max={100} format={signed}
      onChange={(value) => set("vignette", value)} />
    <Slider label="Midpoint" value={adjustments.vignetteMidpoint ?? 55} min={0} max={100} format={String}
      onChange={(value) => set("vignetteMidpoint", value)} />
    <Slider label="Feather" value={adjustments.vignetteFeather ?? 60} min={0} max={100} format={String}
      onChange={(value) => set("vignetteFeather", value)} />
    <h4>Noise</h4>
    <Slider label="Luminance" value={adjustments.denoise ?? 0} min={0} max={100} format={String}
      onChange={(value) => set("denoise", value)} />
    {(adjustments.denoise ?? 0) > 0 &&
      <Slider label="Keep detail" value={adjustments.denoiseDetail ?? 40} min={0} max={100} format={String}
        onChange={(value) => set("denoiseDetail", value)} />}
    <Slider label="Colour" value={adjustments.denoiseColor ?? 0} min={0} max={100} format={String}
      onChange={(value) => set("denoiseColor", value)} />
    <p className="photo-note">
      Colour noise sits in wide soft blotches and luminance noise is fine and sharp, so they get
      separate controls — smoothing both at one radius is what makes denoised pictures look waxy.
      Runs before sharpening, never after.
    </p>
    <h4>Detail</h4>
    <Slider label="Sharpen" value={adjustments.sharpen} min={0} max={100} format={String}
      onChange={(value) => set("sharpen", value)} />
    <Slider label="Blur" value={adjustments.blur} min={0} max={100} format={String}
      onChange={(value) => set("blur", value)} />
    <h4>Look</h4>
    <div className="photo-filters">
      {PHOTO_FILTERS.map((filter) => <button key={filter.id} type="button"
        className={filter.id === adjustments.filter ? "selected" : ""}
        onClick={() => set("filter", filter.id, false)}>
        <FilterThumb base={thumbBase} adjustments={adjustments} filterId={filter.id} />
        <small>{filter.name}</small>
      </button>)}
    </div>
    <Slider label="Amount" value={adjustments.filterAmount} min={0} max={100}
      format={(value) => `${value}%`} onChange={(value) => set("filterAmount", value)} />
    <Slider label="Grain" value={adjustments.grain} min={0} max={100} format={String}
      onChange={(value) => set("grain", value)} />
    <h4>Curve</h4>
    <CurveEditor points={adjustments.curve} onChange={(curve) => set("curve", curve)} />
  </>;
}

/**
 * One look previewed on the picture being edited, in the context of the rest of the grade.
 *
 * Showing the look on its own would be cheaper and misleading: what matters is what this button
 * will do to the picture as it stands, so the thumbnail renders the node's current settings with
 * only the filter swapped. The spatial passes are dropped -- a twelve-pixel blur on a
 * seventy-pixel thumbnail says nothing about the look and costs the most to compute.
 */
function FilterThumb({ base, adjustments, filterId }: {
  base?: ImageData; adjustments: PhotoAdjustments; filterId: string;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !base) return;
    canvas.width = base.width;
    canvas.height = base.height;
    const pixels = new Uint8ClampedArray(base.data);
    applyAdjustments(pixels, base.width, base.height,
      { ...adjustments, filter: filterId, filterAmount: 100, blur: 0, sharpen: 0, grain: 0 });
    canvas.getContext("2d")!.putImageData(new ImageData(pixels, base.width, base.height), 0, 0);
  }, [base, filterId, adjustments]);
  return <canvas ref={ref} width={72} height={46} aria-hidden="true" />;
}

export function CurveEditor({ points, onChange }: {
  points: CurvePoint[]; onChange: (points: CurvePoint[]) => void;
}) {
  const ref = useRef<SVGSVGElement>(null);
  const [dragging, setDragging] = useState<number>();
  const lut = useMemo(() => buildCurveLut(points), [points]);
  const path = useMemo(() => {
    let value = `M0 ${(200 - lut[0] * 200).toFixed(1)}`;
    for (let index = 8; index < 256; index += 8) {
      value += `L${(index / 255 * 200).toFixed(1)} ${(200 - lut[index] * 200).toFixed(1)}`;
    }
    return `${value}L200 ${(200 - lut[255] * 200).toFixed(1)}`;
  }, [lut]);

  const move = (event: React.PointerEvent<SVGSVGElement>) => {
    if (dragging === undefined || !ref.current) return;
    const box = ref.current.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    const y = Math.min(1, Math.max(0, 1 - (event.clientY - box.top) / box.height));
    // Points keep their order and never stack, so the curve stays a function of its input.
    const low = dragging === 0 ? 0.04 : points[dragging - 1].x + 0.06;
    const high = dragging === points.length - 1 ? 0.96 : points[dragging + 1].x - 0.06;
    onChange(points.map((point, index) => index === dragging
      ? { x: Math.min(high, Math.max(low, x)), y } : point));
  };

  return <div className="photo-curve">
    <svg ref={ref} viewBox="0 0 200 200" role="img" aria-label="Tone curve"
      onPointerMove={move} onPointerUp={() => setDragging(undefined)}
      onPointerLeave={() => setDragging(undefined)}>
      <rect x="0" y="0" width="200" height="200" />
      {[50, 100, 150].map((at) => <path key={at} className="grid" d={`M${at} 0V200M0 ${at}H200`} />)}
      <path className="diagonal" d="M0 200L200 0" />
      <path className="curve" d={path} />
      {points.map((point, index) => <circle key={index} className="handle"
        cx={point.x * 200} cy={200 - point.y * 200} r={6}
        onPointerDown={(event) => {
          setDragging(index);
          event.currentTarget.setPointerCapture(event.pointerId);
        }} />)}
    </svg>
  </div>;
}

/* ------------------------------------------------------------- mask shapes */

const DEFAULT_LINEAR: MaskShape = { kind: "linear", x1: 0.5, y1: 0, x2: 0.5, y2: 1 };
const DEFAULT_RADIAL: MaskShape = { kind: "radial", cx: 0.5, cy: 0.5, radius: 0.3, aspect: 1, feather: 0.5 };
const DEFAULT_COLOUR: MaskShape = { kind: "colour", r: 128, g: 128, b: 128, tolerance: 18, softness: 0.6 };

/**
 * The generated half of a mask.
 *
 * A linear gradient is stored as two endpoints, which is the right shape for dragging handles on the
 * picture later, but nobody thinks in endpoints -- so the controls are angle, offset and length, and
 * the endpoints are derived from them. Both directions are pure maths, so there is no second copy of
 * the state to fall out of step.
 */
function MaskShapeEditor({ mask, picking, onChange, onPickColour }: {
  mask: PhotoMask; picking: boolean;
  onChange: (shape: MaskShape | undefined, coalesce?: boolean) => void;
  onPickColour: () => void;
}) {
  const shape = mask.shape;
  const kind = shape?.kind ?? "none";

  const choose = (next: "none" | MaskShape["kind"]) => {
    if (next === "none") return onChange(undefined, false);
    if (next === shape?.kind) return;
    onChange(next === "linear" ? { ...DEFAULT_LINEAR }
      : next === "radial" ? { ...DEFAULT_RADIAL } : { ...DEFAULT_COLOUR }, false);
  };

  return <>
    <div className="photo-buttons">
      {(["none", "linear", "radial", "colour"] as const).map((option) => <button key={option} type="button"
        className={kind === option ? "selected" : ""} onClick={() => choose(option)}>
        {option === "none" ? "Brush only" : option === "colour" ? "Colour" : option[0].toUpperCase() + option.slice(1)}
      </button>)}
    </div>

    {shape?.kind === "linear" && (() => {
      const dx = shape.x2 - shape.x1, dy = shape.y2 - shape.y1;
      const length = Math.max(0.05, Math.hypot(dx, dy));
      const angle = Math.round(Math.atan2(dy, dx) * 180 / Math.PI);
      const midX = (shape.x1 + shape.x2) / 2 - 0.5, midY = (shape.y1 + shape.y2) / 2 - 0.5;
      const offset = (midX * dx + midY * dy) / length;
      const rebuild = (nextAngle: number, nextOffset: number, nextLength: number) => {
        const radians = nextAngle * Math.PI / 180;
        const ux = Math.cos(radians), uy = Math.sin(radians);
        const half = nextLength / 2;
        onChange({
          kind: "linear",
          x1: 0.5 + ux * (nextOffset - half), y1: 0.5 + uy * (nextOffset - half),
          x2: 0.5 + ux * (nextOffset + half), y2: 0.5 + uy * (nextOffset + half),
        });
      };
      return <>
        <Slider label="Angle" value={angle} min={-180} max={180} format={(value) => `${value}°`}
          onChange={(value) => rebuild(value, offset, length)} />
        <Slider label="Offset" value={Number(offset.toFixed(3))} min={-0.6} max={0.6} step={0.005}
          format={(value) => value.toFixed(2)} onChange={(value) => rebuild(angle, value, length)} />
        <Slider label="Length" value={Number(length.toFixed(3))} min={0.05} max={1.6} step={0.01}
          format={(value) => value.toFixed(2)} onChange={(value) => rebuild(angle, offset, value)} />
      </>;
    })()}

    {shape?.kind === "radial" && <>
      <Slider label="Centre X" value={shape.cx} min={0} max={1} step={0.005}
        format={(value) => value.toFixed(2)} onChange={(value) => onChange({ ...shape, cx: value })} />
      <Slider label="Centre Y" value={shape.cy} min={0} max={1} step={0.005}
        format={(value) => value.toFixed(2)} onChange={(value) => onChange({ ...shape, cy: value })} />
      <Slider label="Radius" value={shape.radius} min={0.02} max={1.2} step={0.005}
        format={(value) => value.toFixed(2)} onChange={(value) => onChange({ ...shape, radius: value })} />
      <Slider label="Aspect" value={shape.aspect} min={0.25} max={4} step={0.01}
        format={(value) => value.toFixed(2)} onChange={(value) => onChange({ ...shape, aspect: value })} />
      <Slider label="Feather" value={shape.feather} min={0} max={1} step={0.01}
        format={(value) => `${Math.round(value * 100)}`} onChange={(value) => onChange({ ...shape, feather: value })} />
    </>}

    {shape?.kind === "colour" && <>
      <button className={`photo-inline wide${picking ? " selected" : ""}`} type="button" onClick={onPickColour}>
        <Pipette size={12} /> {picking ? "Click the colour…" : "Pick a colour"}
      </button>
      <div className="photo-sample">
        <i style={{ background: `rgb(${shape.r}, ${shape.g}, ${shape.b})` }} />
        <span>rgb({shape.r}, {shape.g}, {shape.b})</span>
      </div>
      <Slider label="Tolerance" value={shape.tolerance} min={1} max={100} format={String}
        onChange={(value) => onChange({ ...shape, tolerance: value })} />
      <Slider label="Softness" value={shape.softness} min={0.05} max={3} step={0.05}
        format={(value) => value.toFixed(2)} onChange={(value) => onChange({ ...shape, softness: value })} />
    </>}
  </>;
}

/* --------------------------------------------------------------- histogram */

export type HistogramData = {
  red: Uint32Array; green: Uint32Array; blue: Uint32Array; luma: Uint32Array; peak: number;
};

/**
 * The distribution of the picture as it currently stands.
 *
 * Read off the finished preview rather than the source, so it shows what the whole stack produced --
 * which is the only version worth judging clipping against. Drawn as three additive channel curves
 * over a luma fill: separate curves show a colour cast that a single luma trace hides.
 */
function Histogram({ data }: { data?: HistogramData }) {
  const path = (counts: Uint32Array, peak: number) => {
    const bins = counts.length;
    let value = "M0 40";
    for (let bin = 0; bin < bins; bin++) {
      // Square root, because a photograph's histogram is dominated by one or two bins and a linear
      // scale flattens everything else into the floor.
      const height = Math.sqrt(counts[bin] / peak) * 40;
      value += `L${(bin / (bins - 1) * 100).toFixed(2)} ${(40 - height).toFixed(2)}`;
    }
    return `${value}L100 40Z`;
  };
  return <div className="photo-histogram">
    {data
      ? <svg viewBox="0 0 100 40" preserveAspectRatio="none" role="img" aria-label="Histogram">
          <path className="luma" d={path(data.luma, data.peak)} />
          <path className="red" d={path(data.red, data.peak)} />
          <path className="green" d={path(data.green, data.peak)} />
          <path className="blue" d={path(data.blue, data.peak)} />
        </svg>
      : <span className="photo-note">No picture loaded</span>}
  </div>;
}

/* ----------------------------------------------------------------- relight */

const LIGHT_COLORS: Array<{ name: string; swatch: string; color: [number, number, number] }> = [
  { name: "Tungsten", swatch: "#ff8c3d", color: [1, 0.55, 0.24] },
  { name: "Warm", swatch: "#ffe0b8", color: [1, 0.92, 0.82] },
  { name: "Daylight", swatch: "#f0f7ff", color: [0.94, 0.97, 1] },
  { name: "Practical", swatch: "#59d9ff", color: [0.35, 0.85, 1] },
];

export function RelightEditor({ document, node, patch, depthReady, depthState, onComputeDepth }: {
  document: PhotoDocument; node: RelightNode; patch: Patch;
  depthReady: boolean; depthState: "missing" | "running" | "ready"; onComputeDepth: () => void;
}) {
  const set = <K extends keyof RelightSettings>(key: K, value: RelightSettings[K], coalesce = true) =>
    patch({
      ...document,
      nodes: document.nodes.map((item) => item.id === node.id
        ? { ...item, relight: { ...(item as RelightNode).relight, [key]: value } } : item),
    }, { coalesce });
  const relight = node.relight;

  return <>
    <div className="srcline">
      <span>depth · {depthState === "ready" ? "cached" : depthState === "running" ? "measuring…" : "not measured"}</span>
      <span className="photo-spacer" />
      <button type="button" className="photo-inline" disabled={!depthReady || depthState === "running"}
        onClick={onComputeDepth}>
        {depthState === "running" ? <RefreshCw className="spin" size={12} /> : <RefreshCw size={12} />}
        {depthState === "ready" ? "Re-measure" : "Measure depth"}
      </button>
    </div>
    {!depthReady && <p className="photo-note warn">
      Relighting needs a depth model. Install Depth Anything V2 into <code>models/depth-anything-v2</code>
      and it will light up — everything else in this workspace works without it.
    </p>}
    <h4>Key light</h4>
    <p className="photo-note">Drag the gold handle on the picture to move it.</p>
    <Slider label="Intensity" value={relight.intensity} min={0} max={3} step={0.01}
      format={(value) => value.toFixed(2)} onChange={(value) => set("intensity", value)} />
    <Slider label="Distance" value={relight.distance} min={0.1} max={2} step={0.01}
      format={(value) => value.toFixed(2)} onChange={(value) => set("distance", value)} />
    <Slider label="Softness" value={relight.softness} min={0} max={1} step={0.01}
      format={(value) => `${Math.round(value * 100)}`} onChange={(value) => set("softness", value)} />
    <div className="photo-swatches">
      {LIGHT_COLORS.map((entry) => <button key={entry.name} type="button"
        className={entry.color.every((value, index) => Math.abs(value - relight.color[index]) < 0.01) ? "selected" : ""}
        onClick={() => set("color", [...entry.color] as [number, number, number], false)}>
        <i style={{ background: entry.swatch }} />{entry.name}
      </button>)}
    </div>
    <h4>Fill and rim</h4>
    <Slider label="Ambient" value={relight.ambient} min={0} max={1.5} step={0.01}
      format={(value) => value.toFixed(2)} onChange={(value) => set("ambient", value)} />
    <Slider label="Rim" value={relight.rim} min={0} max={2} step={0.01}
      format={(value) => value.toFixed(2)} onChange={(value) => set("rim", value)} />
  </>;
}

/* ----------------------------------------------------------------- retouch */

/**
 * Clone stamp and spot heal.
 *
 * Worth having next to LaMa rather than instead of it: LaMa is a model run and a GPU queue slot,
 * which is the wrong price for a dust speck or the seam between two kitbashed pieces. These run in
 * a frame, and because the strokes are stored the node stays as editable as everything else.
 */
export function RetouchEditor({ node, brush, sampling, hasSource, onBrush, onSetSource, onUndo }: {
  node: RetouchNode;
  brush: { radius: number; hardness: number; heal: boolean };
  sampling: boolean; hasSource: boolean;
  onBrush: (brush: { radius: number; hardness: number; heal: boolean }) => void;
  onSetSource: () => void;
  onUndo: () => void;
}) {
  return <>
    <div className="photo-buttons">
      <button type="button" className={brush.heal ? "" : "selected"}
        onClick={() => onBrush({ ...brush, heal: false })}>Clone</button>
      <button type="button" className={brush.heal ? "selected" : ""}
        onClick={() => onBrush({ ...brush, heal: true })}>Heal</button>
      <button type="button" disabled={!node.strokes.length} onClick={onUndo}>Undo stroke</button>
    </div>
    <button className={`photo-inline wide${sampling ? " selected" : ""}`} type="button" onClick={onSetSource}>
      <Stamp size={12} /> {sampling ? "Click where to copy FROM…" : hasSource ? "Move the source point" : "Set the source point"}
    </button>
    <p className="photo-note">
      {brush.heal
        ? "Heal copies from the source and then matches the brightness of whatever surrounds the spot, so a sample from a lighter area does not leave a bright disc."
        : "Clone copies from the source point verbatim, keeping its offset as you paint."}
      {hasSource ? "" : " Set a source point before painting."}
    </p>
    <Slider label="Brush size" value={brush.radius} min={0.005} max={0.3} step={0.005}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onBrush({ ...brush, radius: value })} />
    <Slider label="Hardness" value={brush.hardness} min={0} max={1} step={0.01}
      format={(value) => `${Math.round(value * 100)}`}
      onChange={(value) => onBrush({ ...brush, hardness: value })} />
  </>;
}

/* ------------------------------------------------------------------ remove */

export function RemoveEditor({ document, node, ready, running, filled, onFill }: {
  document: PhotoDocument; node: RemoveNode;
  ready: boolean; running: boolean; filled: boolean; onFill: () => void;
}) {
  const mask = document.masks.find((item) => item.id === node.maskId);
  const empty = !mask || (!mask.strokes.length && !mask.sourcePath);
  return <>
    <p className="photo-note">
      Paint over what should go, then fill. Only the masked pixels change — this is the thing Qwen
      Image Edit cannot do, because it takes no mask and regenerates the whole frame.
    </p>
    {!ready && <p className="photo-note warn">
      Needs LaMa at <code>models/lama/big-lama.pt</code>. Everything else here works without it.
    </p>}
    {empty && <p className="photo-note warn">This node&rsquo;s mask is empty. Paint the area first.</p>}
    <button className="photo-primary" type="button" disabled={!ready || running || empty} onClick={onFill}>
      {running ? <RefreshCw className="spin" size={14} /> : <Wand2 size={14} />}
      {filled ? "Fill again" : "Fill the masked area"}
    </button>
  </>;
}

/* ------------------------------------------------------------------- masks */

export function MaskEditor({
  document, activeMaskId, brush, selectReady, selecting, points, picking, patch,
  onActivate, onCreateMask, onPickColour, onBrush, onStartSelect, onRunSelect, onClearPoints,
  layers, ownLayerId, activeEntryId, onActivateEntry, onImportMatte, trackedMattes = [],
}: {
  document: PhotoDocument; activeMaskId?: string;
  /** VFX Phase 2: layers a `layer` entry can read, the scope's own layer, and the entry the brush paints into. */
  layers: PhotoLayer[]; ownLayerId?: string;
  activeEntryId?: string;
  onActivateEntry: (id: string | undefined) => void;
  /** Pick a PNG and add it as an entry of the active mask (or a new mask). */
  onImportMatte: () => void;
  /** VFX Phase 3: SAM 2 matte sequences cached by Mask VFX, attachable as sequence entries. */
  trackedMattes?: TrackedMatte[];
  brush: { radius: number; hardness: number; erase: boolean };
  selectReady: boolean; selecting: boolean; points: number;
  patch: Patch;
  onActivate: (id: string | undefined) => void;
  onCreateMask: () => string | undefined;
  onPickColour: () => void;
  picking: boolean;
  onBrush: (brush: { radius: number; hardness: number; erase: boolean }) => void;
  onStartSelect: () => void; onRunSelect: () => void; onClearPoints: () => void;
}) {
  const active = document.masks.find((mask) => mask.id === activeMaskId);
  const patchMask = (id: string, changes: Partial<PhotoMask>, coalesce = true) => patch({
    ...document,
    masks: document.masks.map((mask) => mask.id === id ? { ...mask, ...changes } : mask),
  }, { coalesce });

  return <>
    <div className="photo-masklist">
      {document.masks.map((mask) => <button key={mask.id} type="button"
        className={mask.id === activeMaskId ? "selected" : ""}
        onClick={() => onActivate(mask.id === activeMaskId ? undefined : mask.id)}>
        <Circle size={11} />
        <span>{mask.name}</span>
        <small>{mask.sourcePath ? "SAM 2" : ""}{mask.strokes.length ? ` ${mask.strokes.length} strokes` : ""}{mask.stack?.length ? ` +${mask.stack.length}` : ""}</small>
        <i title="Delete mask" onClick={(event) => {
          event.stopPropagation();
          patch({
            ...document,
            masks: document.masks.filter((item) => item.id !== mask.id),
            // A node pointing at a deleted mask would silently apply to the whole picture, which is
            // the opposite of what it was for, so those nodes are switched off instead.
            nodes: document.nodes.map((node) => node.maskId === mask.id
              ? { ...node, maskId: undefined, enabled: false } as PhotoNode : node),
          }, { coalesce: false });
          if (activeMaskId === mask.id) onActivate(undefined);
        }}><Trash2 size={11} /></i>
      </button>)}
      {!document.masks.length && <p className="photo-note">No masks yet.</p>}
    </div>

    {active && <>
      <h4>{active.name}</h4>
      <Slider label="Brush size" value={brush.radius} min={0.005} max={0.4} step={0.005}
        format={(value) => `${Math.round(value * 100)}%`}
        onChange={(value) => onBrush({ ...brush, radius: value })} />
      <Slider label="Hardness" value={brush.hardness} min={0} max={1} step={0.01}
        format={(value) => `${Math.round(value * 100)}`}
        onChange={(value) => onBrush({ ...brush, hardness: value })} />
      <div className="photo-buttons">
        <button type="button" className={brush.erase ? "" : "selected"}
          onClick={() => onBrush({ ...brush, erase: false })}>Paint</button>
        <button type="button" className={brush.erase ? "selected" : ""}
          onClick={() => onBrush({ ...brush, erase: true })}>Erase</button>
        <button type="button" disabled={!active.strokes.length}
          onClick={() => patchMask(active.id, { strokes: active.strokes.slice(0, -1) }, false)}>
          Undo stroke
        </button>
      </div>
      <Slider label="Feather" value={active.feather} min={0} max={48}
        format={(value) => `${value} px`}
        onChange={(value) => patchMask(active.id, { feather: value })} />
      <div className="photo-buttons">
        <button type="button" className={active.invert ? "selected" : ""}
          onClick={() => patchMask(active.id, { invert: !active.invert }, false)}>Invert</button>
        <button type="button" className={active.enabled ? "selected" : ""}
          onClick={() => patchMask(active.id, { enabled: !active.enabled }, false)}>
          {active.enabled ? "Active" : "Bypassed"}
        </button>
      </div>

      <h4>Shape</h4>
      <MaskShapeEditor mask={active} picking={picking} onPickColour={onPickColour}
        onChange={(shape, coalesce) => patchMask(active.id, { shape }, coalesce)} />

      <h4>Stack</h4>
      <MaskStackEditor mask={active} layers={layers} ownLayerId={ownLayerId} trackedMattes={trackedMattes}
        activeEntryId={activeEntryId} onActivateEntry={onActivateEntry} picking={picking}
        onPickColour={onPickColour} onImportMatte={onImportMatte}
        onChange={(stack, coalesce) => patchMask(active.id, { stack }, coalesce)} />

      <h4>Select with SAM 2</h4>
      {!selectReady && <p className="photo-note warn">
        Needs the SAM 2 pack the video tracker uses. Brush masks work without it.
      </p>}
      <p className="photo-note">
        Click the subject to include it, shift-click to carve a piece back out. The result becomes
        the starting point for this mask — strokes still correct it afterwards. On a video layer, Run
        tracks the subject from the frame on screen to the clip's end (SAM 2 video, cached) and adds
        it to this mask as a sequence that follows the subject.
      </p>
      <div className="photo-buttons">
        <button type="button" disabled={!selectReady} className={selecting ? "selected" : ""}
          onClick={onStartSelect}>{selecting ? `Clicking · ${points}` : "Click to select"}</button>
        <button type="button" disabled={!selectReady || points === 0} onClick={onRunSelect}>Run</button>
        <button type="button" disabled={points === 0} onClick={onClearPoints}>Clear</button>
      </div>
    </>}

    {/* Creating and attaching are one action: a mask that limits nothing looks like a broken app. */}
    <div className="photo-buttons">
      <button type="button" onClick={() => onCreateMask()}><Plus size={12} /> New mask</button>
      <button type="button" title="A greyscale or alpha PNG as a mask (added to the selected mask, or as a new one)"
        onClick={onImportMatte}><FileUp size={12} /> Import matte…</button>
    </div>
  </>;
}

/* ------------------------------------------------------------- mask stack */

const MODE_NAMES: Record<MaskMode, string> = { add: "Add", subtract: "Subtract", intersect: "Intersect", difference: "Difference" };

function sourceLabel(source: MaskSource, layers: PhotoLayer[]): string {
  if (source.kind === "paint") return "Brush";
  if (source.kind === "shape") return source.shape.kind === "colour" ? "Colour range" : `${source.shape.kind[0].toUpperCase()}${source.shape.kind.slice(1)}`;
  if (source.kind === "file") return `File · ${source.path.split(/[\\/]/).pop()}`;
  if (source.kind === "sequence") return `Sequence · ${source.frameCount} frames`;
  if (source.kind === "path") return `Path · ${source.path.subpaths.reduce((sum, sub) => sum + sub.nodes.length, 0)} points`;
  const layer = layers.find((item) => item.id === source.layerId);
  if (source.kind === "depth") return `Depth of ${layer ? layer.name : "a missing layer"} · nearer than ${source.plane}`;
  return `${source.channel === "luma" ? "Luma" : "Alpha"} of ${layer ? layer.name : "a missing layer"}`;
}

/**
 * The entries a mask combines after its base (VFX Phase 2).
 *
 * The base row is the mask as it always was -- brush, shape and SAM 2 above -- so a mask made before
 * stacks existed shows as one row and behaves exactly as it did. Selecting a row points the brush
 * (and the colour picker) at it, which is how an entry is corrected by hand.
 */
export type TrackedMatte = { label: string; cachePath: string; frameCount: number; fps: number };

function MaskStackEditor({ mask, layers, ownLayerId, activeEntryId, picking, onActivateEntry, onPickColour, onImportMatte, onChange, trackedMattes }: {
  mask: PhotoMask; layers: PhotoLayer[]; ownLayerId?: string; activeEntryId?: string; picking: boolean;
  trackedMattes: TrackedMatte[];
  onActivateEntry: (id: string | undefined) => void;
  onPickColour: () => void;
  onImportMatte: () => void;
  onChange: (stack: MaskEntry[], coalesce?: boolean) => void;
}) {
  const stack = mask.stack ?? [];
  const active = stack.find((entry) => entry.id === activeEntryId);
  const patchEntry = (id: string, over: Partial<MaskEntry>, coalesce = true) =>
    onChange(stack.map((entry) => entry.id === id ? { ...entry, ...over } : entry), coalesce);
  const add = (source: MaskSource, over: Partial<MaskEntry> = {}) => {
    const entry = createMaskEntry(source, over);
    onChange([...stack, entry], false);
    onActivateEntry(entry.id);
  };
  const move = (index: number, delta: number) => {
    const target = index + delta;
    if (target < 0 || target >= stack.length) return;
    const next = [...stack];
    [next[index], next[target]] = [next[target], next[index]];
    onChange(next, false);
  };
  const others = layers.filter((layer) => layer.id !== ownLayerId);

  return <>
    <div className="photo-nodes">
      <div className={`photo-node${activeEntryId ? "" : " selected"}`}>
        <button className="photo-node-main" type="button" onClick={() => onActivateEntry(undefined)}>
          <i><Circle size={11} /></i>
          <span><strong>Base</strong><small>brush, shape and SAM 2 above</small></span>
        </button>
      </div>
      {stack.map((entry, index) => <div key={entry.id}
        className={`photo-node${entry.id === activeEntryId ? " selected" : ""}${entry.enabled ? "" : " off"}`}>
        <button className="photo-node-main" type="button" onClick={() => onActivateEntry(entry.id)}>
          <i>{entry.source.kind === "layer" || entry.source.kind === "depth" ? <Layers3 size={11} /> : entry.source.kind === "paint" ? <Circle size={11} /> : <ImageIcon size={11} />}</i>
          <span>
            <strong>{MODE_NAMES[entry.mode]} · {sourceLabel(entry.source, layers)}</strong>
            <small>{Math.round(entry.opacity * 100)}%{entry.invert ? " · inverted" : ""}{entry.strokes.length ? ` · ${entry.strokes.length} strokes` : ""}</small>
          </span>
        </button>
        <i className="photo-node-act" title={entry.enabled ? "Disable" : "Enable"}
          onClick={() => patchEntry(entry.id, { enabled: !entry.enabled }, false)}>{entry.enabled ? <Eye size={13} /> : <EyeOff size={13} />}</i>
        <i className="photo-node-act" title="Move earlier" onClick={() => move(index, -1)}><ArrowUp size={12} /></i>
        <i className="photo-node-act" title="Move later" onClick={() => move(index, 1)}><ArrowDown size={12} /></i>
        <i className="photo-node-act" title="Remove from the stack" onClick={() => {
          onChange(stack.filter((item) => item.id !== entry.id), false);
          if (entry.id === activeEntryId) onActivateEntry(undefined);
        }}><Trash2 size={12} /></i>
      </div>)}
    </div>
    <div className="photo-buttons">
      <button type="button" title="Paint a new entry by hand" onClick={() => add({ kind: "paint" })}><Plus size={12} /> Brush</button>
      <button type="button" onClick={() => add({ kind: "shape", shape: { ...DEFAULT_LINEAR } })}><Plus size={12} /> Linear</button>
      <button type="button" onClick={() => add({ kind: "shape", shape: { ...DEFAULT_RADIAL } })}><Plus size={12} /> Radial</button>
      <button type="button" onClick={() => add({ kind: "shape", shape: { ...DEFAULT_COLOUR } })}><Plus size={12} /> Colour</button>
      <button type="button" onClick={onImportMatte}><FileUp size={12} /> File…</button>
    </div>
    {others.length > 0 && <label className="studio-field"><span>Add from a layer</span>
      <select value="" onChange={(event) => {
        const [layerId, channel] = event.target.value.split("|");
        if (layerId) add({ kind: "layer", layerId, channel: channel as "alpha" | "luma" });
      }}>
        <option value="">Pick a layer…</option>
        {others.map((layer) => <optgroup key={layer.id} label={layer.name}>
          <option value={`${layer.id}|alpha`}>{layer.name} · alpha</option>
          <option value={`${layer.id}|luma`}>{layer.name} · luma</option>
        </optgroup>)}
      </select>
    </label>}

    {others.some((layer) => layer.media?.depth) && <label className="studio-field"><span>Add depth occlusion</span>
      <select value="" onChange={(event) => {
        if (event.target.value) add({ kind: "depth", layerId: event.target.value, plane: 55, softness: 6 }, { mode: "subtract" });
      }}>
        <option value="">Hide where a clip is nearer…</option>
        {others.filter((layer) => layer.media?.depth).map((layer) => <option key={layer.id} value={layer.id}>{layer.name}</option>)}
      </select>
    </label>}

    {trackedMattes.length > 0 && <label className="studio-field"><span>Add a tracked matte</span>
      <select value="" onChange={(event) => {
        const picked = trackedMattes[Number(event.target.value)];
        if (picked) add({ kind: "sequence", cachePath: picked.cachePath, frameCount: picked.frameCount, fps: picked.fps });
      }}>
        <option value="">SAM 2 track from Mask VFX…</option>
        {trackedMattes.map((matte, index) => <option key={matte.cachePath} value={index}>{matte.label} · {matte.frameCount} frames</option>)}
      </select>
    </label>}

    {active && <>
      <label className="studio-field"><span>Combine</span>
        <select value={active.mode} onChange={(event) => patchEntry(active.id, { mode: event.target.value as MaskMode }, false)}>
          {MASK_MODES.map((mode) => <option key={mode} value={mode}>{MODE_NAMES[mode]}</option>)}
        </select>
      </label>
      <Slider label="Opacity" value={active.opacity} min={0} max={1} step={0.01}
        format={(value) => `${Math.round(value * 100)}%`} onChange={(value) => patchEntry(active.id, { opacity: value })} />
      <Slider label="Expand / choke" value={active.expand} min={-40} max={40}
        format={(value) => value ? `${value > 0 ? "+" : ""}${value} px` : "0"} onChange={(value) => patchEntry(active.id, { expand: value })} />
      <Slider label="Feather" value={active.feather} min={0} max={48}
        format={(value) => `${value} px`} onChange={(value) => patchEntry(active.id, { feather: value })} />
      <div className="photo-buttons">
        <button type="button" className={active.invert ? "selected" : ""}
          onClick={() => patchEntry(active.id, { invert: !active.invert }, false)}>Invert</button>
        <button type="button" disabled={!active.strokes.length}
          onClick={() => patchEntry(active.id, { strokes: active.strokes.slice(0, -1) }, false)}>Undo stroke</button>
      </div>
      {active.source.kind === "shape" && <MaskShapeEditor
        mask={{ ...mask, shape: active.source.shape }} picking={picking} onPickColour={onPickColour}
        onChange={(shape, coalesce) => shape
          ? patchEntry(active.id, { source: { kind: "shape", shape } }, coalesce)
          : patchEntry(active.id, { source: { kind: "paint" } }, false)} />}
      {active.source.kind === "layer" && (() => {
        const source = active.source;
        return <label className="studio-field"><span>Channel</span>
          <select value={source.channel} onChange={(event) =>
            patchEntry(active.id, { source: { ...source, channel: event.target.value as "alpha" | "luma" } }, false)}>
            <option value="alpha">Alpha</option>
            <option value="luma">Luma</option>
          </select>
        </label>;
      })()}
      {active.source.kind === "depth" && (() => {
        const source = active.source;
        return <>
          <Slider label="Plane" value={source.plane} min={0} max={100} step={0.5} format={(value) => `${value}`}
            onChange={(value) => patchEntry(active.id, { source: { ...source, plane: value } })} />
          <Slider label="Softness" value={source.softness} min={0} max={40} step={0.5} format={(value) => `${value}`}
            onChange={(value) => patchEntry(active.id, { source: { ...source, softness: value } })} />
          <p className="photo-note">
            Covers where the clip's depth (0 far, 100 near, per clip) is nearer than the plane. As a
            Subtract entry it hides this layer behind whatever is in front: a hand passing the smoke.
            Turn on the mask overlay and scrub to set the plane.
          </p>
        </>;
      })()}
      {active.source.kind === "file" && (() => {
        const source = active.source;
        return <label className="studio-field"><span>Read</span>
          <select value={source.channel ?? "auto"} onChange={(event) =>
            patchEntry(active.id, { source: { ...source, channel: event.target.value as "auto" | "alpha" | "luma" } }, false)}>
            <option value="auto">Alpha if it has any, else luma</option>
            <option value="alpha">Alpha</option>
            <option value="luma">Luma (Rec. 709)</option>
          </select>
        </label>;
      })()}
      <p className="photo-note">
        The brush paints into this entry while it is selected. Feather and expand here are canvas
        pixels and scale with the render, so the preview and the saved file agree.
      </p>
    </>}
  </>;
}

/* ------------------------------------------------------------------ layers */

/**
 * The canvas, bottom layer last.
 *
 * Reversed against the node stack on purpose: a processing chain reads top-down because each step
 * feeds the next, and a layer list reads bottom-up because layers sit on top of one another. Showing
 * either one the other way round is how people end up misreading what happens first.
 *
 * The composite row is a scope like any layer, not a separate control. Selecting it points the node
 * stack, the masks and the brush at the finished frame instead of at one picture, which is the only
 * distinction that matters and so is the only one the UI makes.
 */
export function LayerList({ document, activeId, onSelect, onChange }: {
  document: PhotoDocument; activeId: string;
  onSelect: (id: string) => void;
  onChange: (layers: PhotoLayer[], label: string) => void;
}) {
  const layers = layersOf(document);
  const move = (index: number, delta: number) => {
    const target = index + delta;
    if (target < 0 || target >= layers.length) return;
    const next = [...layers];
    [next[index], next[target]] = [next[target], next[index]];
    onChange(next, "Reordered");
  };
  const patchLayer = (id: string, over: Partial<PhotoLayer>, label: string) =>
    onChange(layers.map((layer) => layer.id === id ? { ...layer, ...over } : layer), label);

  return <div className="photo-nodes">
    <div className={`photo-node${activeId === COMPOSITE_SCOPE ? " selected" : ""}`}>
      <button className="photo-node-main" type="button" onClick={() => onSelect(COMPOSITE_SCOPE)}>
        <i><Layers3 size={13} /></i>
        <span>
          <strong>Composite</strong>
          <small>{document.nodes.length ? `${document.nodes.length} steps over everything` : "the finished frame"}</small>
        </span>
      </button>
    </div>
    {layers.map((layer, index) => <div key={layer.id}
      className={`photo-node${layer.id === activeId ? " selected" : ""}${layer.visible ? "" : " off"}`}>
      <button className="photo-node-main" type="button" onClick={() => onSelect(layer.id)}>
        <i>{layer.media ? <Film size={13} /> : layer.generator ? <Wand2 size={13} /> : layer.path ? <PenTool size={13} /> : layer.text ? <Type size={13} />
          : layer.gradient ? <Palette size={13} /> : <ImageIcon size={13} />}</i>
        <span>
          <strong>{layer.text ? (layer.text.content.split(/\r?\n/)[0] || "Text") : layer.name}</strong>
          <small>
            {layer.blendMode === "normal" ? "Normal" : BLEND_MODES.find((mode) => mode.id === layer.blendMode)?.name}
            {` · ${Math.round(layer.opacity * 100)}%`}
            {layer.nodes.length ? ` · ${layer.nodes.length} steps` : ""}
            {layer.media ? ` · ${layer.media.loop ? "loop" : "clip"} from ${layer.media.start}` : ""}
            {layer.clip && index > 0 ? " · clipped" : ""}
            {layer.maskId ? " · masked" : ""}
          </small>
        </span>
      </button>
      <i className="photo-node-act" title={layer.visible ? "Hide" : "Show"}
        onClick={() => patchLayer(layer.id, { visible: !layer.visible }, layer.visible ? "Hidden" : "Shown")}>
        {layer.visible ? <Eye size={13} /> : <EyeOff size={13} />}
      </i>
      <i className="photo-node-act" title="Move up" onClick={() => move(index, 1)}><ArrowUp size={12} /></i>
      <i className="photo-node-act" title="Move down" onClick={() => move(index, -1)}><ArrowDown size={12} /></i>
      <i className={`photo-node-act${layers.length > 1 ? "" : " disabled"}`}
        title={layers.length > 1 ? "Remove from the canvas" : "A canvas needs at least one layer"}
        onClick={() => layers.length > 1
          && onChange(layers.filter((item) => item.id !== layer.id), `Removed ${layer.name}`)}>
        <Trash2 size={12} />
      </i>
    </div>).reverse()}
  </div>;
}

/**
 * Where a layer sits, how it blends, and how it is made to belong there.
 *
 * The edge group is the part that earns its place in a kitbashing tool. A cutout arrives with a rim
 * of whatever it used to stand in front of, no relationship to the light around it, and nothing
 * underneath -- three tells that get fixed here in a frame, instead of by a round trip through an
 * image model that regenerates the whole picture to correct an edge.
 */
export function LayerEditor({ layer, layers, matching, onChange, onMatchColour, onTrackMatte }: {
  layer: PhotoLayer; matching: boolean;
  /** Every layer bottom first, for the clip toggle and the matte picker. */
  layers: PhotoLayer[];
  onChange: (over: Partial<PhotoLayer>, label: string) => void;
  onMatchColour: () => void;
  /** Make a track matte: one action that creates the mask, attaches it and hides the matte layer. */
  onTrackMatte: (matteLayerId: string, channel: "alpha" | "luma", invert: boolean) => void;
}) {
  const edge = layer.edge ?? NEUTRAL_EDGE;
  const setEdge = (over: Partial<LayerEdge>, label: string) => onChange({ edge: { ...edge, ...over } }, label);
  return <>
    <label className="studio-field"><span>Blend mode</span>
      <select value={layer.blendMode} onChange={(event) =>
        onChange({ blendMode: event.target.value as PhotoLayer["blendMode"] }, "Blend mode")}>
        {BLEND_MODES.map((mode) => <option key={mode.id} value={mode.id}>{mode.name}</option>)}
      </select>
    </label>
    <Slider label="Opacity" value={layer.opacity} min={0} max={1} step={0.01}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onChange({ opacity: value }, "Opacity")} />

    <h4>Mask and matte</h4>
    <LayerMaskControls layer={layer} layers={layers} onChange={onChange} onTrackMatte={onTrackMatte} />

    <h4>Placement</h4>
    <Slider label="X" value={layer.x} min={-0.5} max={1.5} step={0.002}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onChange({ x: value }, "Moved")} />
    <Slider label="Y" value={layer.y} min={-0.5} max={1.5} step={0.002}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onChange({ y: value }, "Moved")} />
    <Slider label="Scale" value={layer.scale} min={0.05} max={4} step={0.005}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onChange({ scale: value }, "Scaled")} />
    <Slider label="Rotation" value={layer.rotation} min={-180} max={180} step={0.5}
      format={(value) => `${value.toFixed(1)}°`}
      onChange={(value) => onChange({ rotation: value }, "Rotated")} />
    <div className="photo-buttons">
      <button type="button" onClick={() => onChange({ x: 0.5, y: 0.5, rotation: 0 }, "Centred")}>Centre</button>
      <button type="button" onClick={() => onChange({ scale: 1 }, "Reset scale")}>100%</button>
      <button type="button" className={layer.locked ? "selected" : ""}
        onClick={() => onChange({ locked: !layer.locked }, layer.locked ? "Unlocked" : "Locked")}>
        {layer.locked ? <Lock size={12} /> : <LockOpen size={12} />} {layer.locked ? "Locked" : "Unlocked"}
      </button>
    </div>

    <h4>Trim</h4>
    <Slider label="Left" value={layer.cropLeft} min={0} max={90} step={0.5}
      format={(value) => `${value.toFixed(0)}%`} onChange={(value) => onChange({ cropLeft: value }, "Trimmed")} />
    <Slider label="Right" value={layer.cropRight} min={0} max={90} step={0.5}
      format={(value) => `${value.toFixed(0)}%`} onChange={(value) => onChange({ cropRight: value }, "Trimmed")} />
    <Slider label="Top" value={layer.cropTop} min={0} max={90} step={0.5}
      format={(value) => `${value.toFixed(0)}%`} onChange={(value) => onChange({ cropTop: value }, "Trimmed")} />
    <Slider label="Bottom" value={layer.cropBottom} min={0} max={90} step={0.5}
      format={(value) => `${value.toFixed(0)}%`} onChange={(value) => onChange({ cropBottom: value }, "Trimmed")} />

    <h4>Making it belong</h4>
    <button className="photo-inline wide" type="button" disabled={matching} onClick={onMatchColour}>
      {matching ? <RefreshCw className="spin" size={12} /> : <Pipette size={12} />} Match colour to what is behind
    </button>
    <p className="photo-note">
      Match colour measures this layer against the frame under it and adds a grade step with the
      exposure, white balance and contrast that close the gap. It is a solve, not a generation — the
      numbers land on ordinary sliders you can then disagree with.
    </p>
    <Slider label="Defringe" value={edge.defringe} min={0} max={8} step={0.5}
      format={(value) => value ? `${value} px` : "off"}
      onChange={(value) => setEdge({ defringe: value }, "Defringe")} />
    <Slider label="Light wrap" value={edge.lightWrap} min={0} max={100}
      format={(value) => value ? `${value}%` : "off"}
      onChange={(value) => setEdge({ lightWrap: value }, "Light wrap")} />
    {edge.lightWrap > 0 && <Slider label="Wrap spread" value={edge.lightWrapSpread} min={1} max={80}
      format={(value) => `${value} px`}
      onChange={(value) => setEdge({ lightWrapSpread: value }, "Light wrap")} />}
    <Slider label="Contact shadow" value={edge.contactShadow} min={0} max={100}
      format={(value) => value ? `${value}%` : "off"}
      onChange={(value) => setEdge({ contactShadow: value }, "Contact shadow")} />
    {edge.contactShadow > 0 && <>
      <Slider label="Shadow softness" value={edge.contactShadowSoftness} min={0} max={100}
        format={(value) => `${value}`}
        onChange={(value) => setEdge({ contactShadowSoftness: value }, "Contact shadow")} />
      <Slider label="Shadow offset" value={edge.contactShadowOffset} min={-0.2} max={0.3} step={0.005}
        format={(value) => `${Math.round(value * 100)}%`}
        onChange={(value) => setEdge({ contactShadowOffset: value }, "Contact shadow")} />
      <Slider label="Shadow squash" value={edge.contactShadowSquash} min={0.02} max={0.95} step={0.01}
        format={(value) => value.toFixed(2)}
        onChange={(value) => setEdge({ contactShadowSquash: value }, "Contact shadow")} />
    </>}
    <p className="photo-note">
      Defringe eats the contaminated rim and repaints it from inside the subject. Light wrap spills
      the backdrop around the edge. The shadow is derived from this layer&rsquo;s own alpha and drawn
      underneath it.
    </p>
  </>;
}

/**
 * The layer's alpha cut (VFX Phase 2): which of its masks is the layer mask, clipping to the layer
 * below, and the After Effects-style track matte, which is a layer mask holding one layer entry.
 */
function LayerMaskControls({ layer, layers, onChange, onTrackMatte }: {
  layer: PhotoLayer; layers: PhotoLayer[];
  onChange: (over: Partial<PhotoLayer>, label: string) => void;
  onTrackMatte: (matteLayerId: string, channel: "alpha" | "luma", invert: boolean) => void;
}) {
  const index = layers.findIndex((item) => item.id === layer.id);
  const others = layers.filter((item) => item.id !== layer.id);
  return <>
    <label className="studio-field"><span>Layer mask</span>
      <select value={layer.maskId ?? ""} onChange={(event) =>
        onChange({ maskId: event.target.value || undefined }, event.target.value ? "Layer mask set" : "Layer mask removed")}>
        <option value="">None — the whole layer</option>
        {layer.masks.map((mask) => <option key={mask.id} value={mask.id}>{mask.name}</option>)}
      </select>
    </label>
    <div className="photo-buttons">
      <button type="button" disabled={index <= 0} className={layer.clip ? "selected" : ""}
        title={index <= 0 ? "The bottom layer has nothing below it to clip to"
          : "Show this layer only where the layer below it (the base of the clipping group) is"}
        onClick={() => onChange({ clip: !layer.clip }, layer.clip ? "Unclipped" : "Clipped to the layer below")}>
        <Scissors size={12} /> {layer.clip ? "Clipped to below" : "Clip to below"}
      </button>
    </div>
    {others.length > 0 && <label className="studio-field"><span>Track matte</span>
      <select value="" onChange={(event) => {
        const [id, channel, invert] = event.target.value.split("|");
        if (id) onTrackMatte(id, channel as "alpha" | "luma", invert === "1");
      }}>
        <option value="">Use another layer as a matte…</option>
        {others.map((item) => <optgroup key={item.id} label={item.name}>
          <option value={`${item.id}|alpha|0`}>Alpha</option>
          <option value={`${item.id}|alpha|1`}>Alpha inverted</option>
          <option value={`${item.id}|luma|0`}>Luma</option>
          <option value={`${item.id}|luma|1`}>Luma inverted</option>
        </optgroup>)}
      </select>
    </label>}
    <p className="photo-note">
      A track matte hides the matte layer, as After Effects does; show it again and it draws as well
      as cutting. It is read after its own steps and placement. The mask it makes is an ordinary
      layer mask with one entry, so it can be feathered, choked or combined like any other.
    </p>
  </>;
}

/** Luma key: keep what sits between two brightness levels. For stock footage shot on black. */
export function LumaKeyEditor({ node, onChange }: {
  node: LumaKeyNode;
  onChange: (over: Partial<LumaKeyNode>, label: string, coalesce?: boolean) => void;
}) {
  return <>
    <Slider label="Low" value={node.low} min={0} max={100} format={(value) => `${value}%`}
      onChange={(value) => onChange({ low: Math.min(value, node.high) }, "Luma key", true)} />
    <Slider label="High" value={node.high} min={0} max={100} format={(value) => `${value}%`}
      onChange={(value) => onChange({ high: Math.max(value, node.low) }, "Luma key", true)} />
    <Slider label="Softness" value={node.softness} min={0} max={50} format={(value) => `${value}%`}
      onChange={(value) => onChange({ softness: value }, "Luma key", true)} />
    <div className="photo-buttons">
      <button type="button" className={node.invert ? "selected" : ""}
        onClick={() => onChange({ invert: !node.invert }, node.invert ? "Key normal" : "Key inverted")}>Invert</button>
    </div>
    <p className="photo-note">
      Keeps what lies between Low and High (Rec. 709 luma of the encoded picture), fading over
      Softness. For elements shot on black, Unmultiply usually looks better: it keeps the glow.
    </p>
  </>;
}

/* -------------------------------------------------------------------- time (VFX Phase 3) */

/** Frames as a timecode at the composition rate: 00:04+07. */
export function timecode(frame: number, rate: number): string {
  const whole = Math.max(1, Math.round(rate));
  const seconds = Math.floor(frame / whole);
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}+${String(frame % whole).padStart(2, "0")}`;
}

/** The composition's time bar: shown only when a layer moves. */
export function TimeBar({ frame, duration, rate, playing, useProxy, hasProxy, waiting, onFrame, onPlay, onProxy, onDuration }: {
  frame: number; duration: number; rate: number; playing: boolean; useProxy: boolean; hasProxy: boolean;
  /** Frames still decoding for the current frame. */
  waiting: number;
  onFrame: (frame: number) => void; onPlay: (playing: boolean) => void; onProxy: (on: boolean) => void;
  onDuration: (frames: number) => void;
}) {
  return <div className="photo-timebar">
    <button type="button" title="First frame" onClick={() => onFrame(0)}><SkipBack size={13} /></button>
    <button type="button" className={playing ? "selected" : ""} title={playing ? "Pause" : "Play (waits for frames, never shows a missing one)"}
      onClick={() => onPlay(!playing)}>{playing ? <Pause size={13} /> : <Play size={13} />}</button>
    <input type="range" min={0} max={Math.max(0, duration - 1)} step={1} value={Math.min(frame, duration - 1)}
      onChange={(event) => onFrame(Number(event.target.value))} />
    <span className="photo-timecode" title={`Frame ${frame} of ${duration} at ${rate.toFixed(3).replace(/\.?0+$/, "")} fps`}>
      {timecode(frame, rate)} · {frame}/{duration}{waiting ? ` · loading ${waiting}` : ""}
    </span>
    <label className="photo-timebar-length" title="Composition length in frames">
      <Film size={12} />
      <input type="number" min={1} max={100000} value={duration}
        onChange={(event) => { const value = Number(event.target.value); if (value >= 1) onDuration(Math.round(value)); }} />
    </label>
    <button type="button" disabled={!hasProxy} className={useProxy ? "selected" : ""}
      title={hasProxy ? "Scrub from the lighter proxies. Saving always uses the full-quality frames."
        : "No layer has a proxy"}
      onClick={() => onProxy(!useProxy)}>Proxy</button>
  </div>;
}

/** A media layer's timing: where it starts, which source frames, looping. Units are frames. */
export function MediaEditor({ media, rate, onChange, onSaveLoop }: {
  media: LayerMedia; rate: number;
  onChange: (media: LayerMedia, label: string, coalesce?: boolean) => void;
  /** Elements only: write the loop to the element's manifest as its default. */
  onSaveLoop?: () => void;
}) {
  const set = (over: Partial<LayerMedia>, label: string, coalesce = true) => onChange({ ...media, ...over }, label, coalesce);
  const length = media.outPoint - media.inPoint;
  return <>
    <p className="photo-note">
      {media.kind === "frames" ? "Frame sequence" : "Video"} · {media.width} × {media.height} · {media.frameCount} frames at {Number(media.fps.toFixed(3))} fps
      {media.fps !== rate ? ` (composition runs at ${Number(rate.toFixed(3))})` : ""}
    </p>
    <label className="studio-field"><span>Source fps</span>
      <input type="number" min={1} max={240} step={0.001} value={Number(media.fps.toFixed(3))}
        title="The rate the frames were made at. Wan Alpha is 16; an imported sequence is assumed 24."
        onChange={(event) => { const value = Number(event.target.value); if (value > 0) set({ fps: value }, "Source rate", false); }} />
    </label>
    <Slider label="Starts at frame" value={media.start} min={0} max={Math.max(240, media.start + 120)} step={1}
      format={String} onChange={(value) => set({ start: Math.round(value) }, "Start")} />
    <Slider label="In" value={media.inPoint} min={0} max={media.frameCount - 1} step={1} format={String}
      onChange={(value) => set({ inPoint: Math.min(Math.round(value), media.outPoint - 1) }, "In point")} />
    <Slider label="Out" value={media.outPoint} min={1} max={media.frameCount} step={1} format={String}
      onChange={(value) => set({ outPoint: Math.max(Math.round(value), media.inPoint + 1) }, "Out point")} />
    <div className="photo-buttons">
      <button type="button" className={media.loop ? "selected" : ""}
        onClick={() => set({ loop: !media.loop }, media.loop ? "Plays once" : "Loops", false)}><Repeat size={12} /> Loop</button>
      {onSaveLoop && <button type="button" title="Make this loop the element's default for new layers" onClick={onSaveLoop}>
        Save loop to element</button>}
    </div>
    {media.loop && <Slider label="Crossfade" value={media.crossfade} min={0} max={Math.max(1, Math.floor(length / 2))} step={1}
      format={(value) => value ? `${value} frames` : "hard cut"}
      onChange={(value) => set({ crossfade: Math.round(value) }, "Crossfade")} />}
    <p className="photo-note">
      A loop repeats In to Out. The crossfade blends the last frames into the first, premultiplied,
      so smoke loops without a pop; it shortens the loop by that many frames.
    </p>
  </>;
}

/** Matrix rain's controls. Everything is deterministic in the seed and the frame. */
export function MatrixRainEditor({ generator, onChange }: {
  generator: MatrixRainGenerator;
  onChange: (generator: MatrixRainGenerator, label: string, coalesce?: boolean) => void;
}) {
  const set = (over: Partial<MatrixRainGenerator>, label: string, coalesce = true) => onChange({ ...generator, ...over }, label, coalesce);
  return <>
    <div className="photo-buttons">
      <button type="button" onClick={() => set({ seed: Math.floor(Math.random() * 1e6) }, "New rain", false)}>New seed</button>
      <button type="button" className={generator.mirror ? "selected" : ""}
        onClick={() => set({ mirror: !generator.mirror }, "Mirror", false)}>Mirrored glyphs</button>
    </div>
    <Slider label="Cell size" value={generator.cellSize} min={6} max={80} step={1} format={(value) => `${value} px`}
      onChange={(value) => set({ cellSize: Math.round(value) }, "Cell size")} />
    <Slider label="Density" value={generator.density} min={0} max={1} step={0.01} format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => set({ density: value }, "Density")} />
    <Slider label="Trail" value={generator.trail} min={1} max={60} step={1} format={(value) => `${value} cells`}
      onChange={(value) => set({ trail: Math.round(value) }, "Trail")} />
    <Slider label="Loop length" value={generator.loopFrames} min={12} max={1440} step={1} format={(value) => `${value} frames`}
      onChange={(value) => set({ loopFrames: Math.round(value) }, "Loop length")} />
    <Slider label="Slowest column" value={generator.minPasses} min={1} max={8} step={1} format={(value) => `${value} screens/loop`}
      onChange={(value) => set({ minPasses: Math.round(value), maxPasses: Math.max(generator.maxPasses, Math.round(value)) }, "Speed")} />
    <Slider label="Fastest column" value={generator.maxPasses} min={1} max={8} step={1} format={(value) => `${value} screens/loop`}
      onChange={(value) => set({ maxPasses: Math.round(value), minPasses: Math.min(generator.minPasses, Math.round(value)) }, "Speed")} />
    <Slider label="Glyph changes" value={generator.glyphChanges} min={0} max={120} step={1} format={(value) => `${value} per loop`}
      onChange={(value) => set({ glyphChanges: Math.round(value) }, "Glyph changes")} />
    <label className="studio-field"><span>Rain colour</span>
      <input type="color" value={generator.colour} onChange={(event) => set({ colour: event.target.value }, "Colour", false)} />
    </label>
    <label className="studio-field"><span>Head colour</span>
      <input type="color" value={generator.headColour} onChange={(event) => set({ headColour: event.target.value }, "Colour", false)} />
    </label>
    <p className="photo-note">
      Loops exactly: frame {generator.loopFrames} is frame 0 again. It sits on its layer like a
      gradient, so blend it with Screen, grade it, blur a copy for glow, or mask it with a matte.
    </p>
  </>;
}

export function AlphaCurveEditor({ node, onChange }: {
  node: AlphaCurveNode;
  onChange: (over: Partial<AlphaCurveNode>, label: string, coalesce?: boolean) => void;
}) {
  return <>
    <Slider label="Gain" value={node.gain} min={0} max={4} step={0.01} format={(value) => value.toFixed(2)}
      onChange={(value) => onChange({ gain: value }, "Alpha gain", true)} />
    <Slider label="Gamma" value={node.gamma} min={0.2} max={3} step={0.01} format={(value) => value.toFixed(2)}
      onChange={(value) => onChange({ gamma: value }, "Alpha gamma", true)} />
    <p className="photo-note">
      alpha × gain, shaped by gamma (below 1 thickens the thin parts). Wan Alpha elements come out
      mostly translucent; this makes a fireball's core solid without touching its colour.
    </p>
  </>;
}

/** The element library: add, import, generate with Wan Alpha. */
export function ElementsPanel({ elements, busy, onRefresh, onAdd, onAddEmitter, onImport, onGenerate, onBuildProxy }: {
  elements: ElementEntry[]; busy?: string;
  onRefresh: () => void;
  onAdd: (element: ElementEntry) => void;
  /** VFX Phase 4: add the element as a sprite emitter instead of a layer. */
  onAddEmitter?: (element: ElementEntry) => void;
  onImport: () => void;
  onGenerate: (request: { name: string; prompt: string; frames: number; seed: number }) => void;
  onBuildProxy: (id: string) => void;
}) {
  const [preset, setPreset] = useState(WAN_ALPHA_PRESETS[0].id);
  const [prompt, setPrompt] = useState(WAN_ALPHA_PRESETS[0].prompt);
  const [frames, setFrames] = useState(33);
  const [seed, setSeed] = useState(42);
  const [name, setName] = useState(WAN_ALPHA_PRESETS[0].name);
  return <>
    <div className="photo-nodes">
      {elements.map((element) => <div key={element.manifest.id} className="photo-node">
        <button className="photo-node-main" type="button" title={element.manifest.prompt ?? element.dir}
          disabled={Boolean(busy)} onClick={() => onAdd(element)}>
          <i><Film size={12} /></i>
          <span>
            <strong>{element.manifest.name}</strong>
            <small>
              {element.manifest.frameCount} f · {element.manifest.width}×{element.manifest.height} · {element.manifest.source === "wan-alpha" ? "Wan Alpha" : "imported"}
              {element.manifest.proxy ? "" : typeof element.manifest.proxyError === "string" ? " · no proxy (ffmpeg failed)" : " · no proxy"}
            </small>
          </span>
        </button>
        {onAddEmitter && <i className="photo-node-act" title="Add as a sprite emitter (spawns copies over time, at a track or a point)"
          onClick={() => !busy && onAddEmitter(element)}><Wand2 size={12} /></i>}
        {!element.manifest.proxy && <i className="photo-node-act" title="Build the preview proxy"
          onClick={() => onBuildProxy(element.manifest.id)}><RefreshCw size={12} /></i>}
      </div>)}
      {!elements.length && <p className="photo-note">No elements yet. Generate one below, or import a folder of RGBA PNGs.</p>}
    </div>
    <div className="photo-buttons">
      <button type="button" disabled={Boolean(busy)} onClick={onImport}><FolderOpen size={12} /> Import PNG folder…</button>
      <button type="button" disabled={Boolean(busy)} onClick={onRefresh}><RefreshCw size={12} /> Refresh</button>
    </div>
    <p className="photo-note">Click an element to add it as a layer at the current frame; the wand adds it as an emitter.</p>

    <h4>Generate with Wan Alpha</h4>
    <label className="studio-field"><span>Preset</span>
      <select value={preset} onChange={(event) => {
        const picked = WAN_ALPHA_PRESETS.find((item) => item.id === event.target.value);
        setPreset(event.target.value);
        if (picked) { setPrompt(picked.prompt); setName(picked.name); }
      }}>
        {WAN_ALPHA_PRESETS.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
      </select>
    </label>
    {WAN_ALPHA_PRESETS.find((item) => item.id === preset)?.use &&
      <p className="photo-note">{WAN_ALPHA_PRESETS.find((item) => item.id === preset)?.use}</p>}
    <label className="studio-field"><span>Prompt</span>
      <textarea rows={4} value={prompt} onChange={(event) => setPrompt(event.target.value)} />
    </label>
    <label className="studio-field"><span>Name</span>
      <input value={name} onChange={(event) => setName(event.target.value)} />
    </label>
    <div className="photo-buttons">
      {[33, 81].map((count) => <button key={count} type="button" className={frames === count ? "selected" : ""}
        onClick={() => setFrames(count)}>{count} frames{count === 33 ? " · ~1.5 min" : " · ~4 min"}</button>)}
    </div>
    <label className="studio-field"><span>Seed</span>
      <input type="number" value={seed} onChange={(event) => setSeed(Math.max(0, Math.round(Number(event.target.value) || 0)))} />
    </label>
    <button className="photo-inline wide" type="button" disabled={Boolean(busy) || !prompt.trim()}
      onClick={() => onGenerate({ name: name.trim() || "Element", prompt: prompt.trim(), frames, seed })}>
      {busy ? <><RefreshCw className="spin" size={12} /> {busy}…</> : <><Sparkle /> Generate element</>}
    </button>
    <p className="photo-note">
      832 × 480 at 16 fps, 4 steps. Runs through the resource guard (32 GB commit, 13 GB VRAM free);
      close a local LLM first. The result lands in <code>elements\</code> inside this project.
    </p>
  </>;
}

const Sparkle = () => <Wand2 size={12} />;

/* -------------------------------------------------------------------- text */

const ALIGNMENTS: Array<{ id: TextAlign; icon: ReactNode; title: string }> = [
  { id: "left", icon: <AlignLeft size={13} />, title: "Left" },
  { id: "center", icon: <AlignCenter size={13} />, title: "Centre" },
  { id: "right", icon: <AlignRight size={13} />, title: "Right" },
];

const WEIGHTS = [300, 400, 500, 600, 700, 800, 900];

/**
 * A text layer's own controls.
 *
 * Sizes are in canvas pixels rather than points, and deliberately so: a text layer sits on a picture
 * whose size the user chose, and "18pt" means nothing without a DPI nobody has set. Pixels are the
 * unit the canvas is actually measured in, and the numbers stay true when the document is exported.
 */
export type TextPreviews = {
  fill?: string; stroke?: string; shadow?: string; panel?: string; panelBorder?: string;
};

export function TextEditor({ text, fonts, previews, inShape, onChange }: {
  text: TextBlock; fonts: FontGroup[]; previews?: TextPreviews;
  /** Inside a balloon, where the container leads and the words give. */
  inShape?: boolean;
  onChange: (over: Partial<TextBlock>, label: string, coalesce?: boolean) => void;
}) {
  const known = fonts.some((group) => group.families.includes(text.fontFamily));
  return <>
    <label className="studio-field"><span>Words</span>
      <textarea className="photo-text-input" rows={3} value={text.content}
        onChange={(event) => onChange({ content: event.target.value }, "Text")} />
    </label>

    <label className="studio-field"><span>Typeface</span>
      <select value={known ? text.fontFamily : ""}
        onChange={(event) => event.target.value && onChange({ fontFamily: event.target.value }, "Typeface")}>
        {!known && <option value="">{text.fontFamily} · not on this machine</option>}
        {fonts.map((group) => <optgroup key={group.name} label={group.name}>
          {group.families.map((family) => <option key={family} value={family}
            style={{ fontFamily: `"${family}", sans-serif` }}>{family}</option>)}
        </optgroup>)}
      </select>
    </label>
    <label className="studio-field"><span>…or type a name</span>
      <input value={text.fontFamily} spellCheck={false}
        onChange={(event) => onChange({ fontFamily: event.target.value }, "Typeface")} />
    </label>
    {!known && <p className="photo-note warn">
      Nothing here resolves <code>{text.fontFamily}</code>. It will fall back to the nearest generic
      until the font is installed.
    </p>}

    <div className="photo-buttons">
      {ALIGNMENTS.map((option) => <button key={option.id} type="button" title={option.title}
        className={text.align === option.id ? "selected" : ""}
        onClick={() => onChange({ align: option.id }, "Alignment")}>{option.icon}</button>)}
      <button type="button" className={text.italic ? "selected" : ""} title="Italic"
        onClick={() => onChange({ italic: !text.italic }, "Italic")}><Italic size={13} /></button>
    </div>
    <label className="studio-field"><span>Weight</span>
      <select value={text.weight}
        onChange={(event) => onChange({ weight: Number(event.target.value) }, "Weight")}>
        {WEIGHTS.map((weight) => <option key={weight} value={weight}>{weight}</option>)}
      </select>
    </label>

    <Slider label="Size" value={text.fontSize} min={8} max={600} step={1}
      format={(value) => `${Math.round(value)} px`}
      onChange={(value) => onChange({ fontSize: value }, "Size")} />
    {!inShape && <Slider label="Column width" value={text.boxWidth} min={40} max={8000} step={10}
      format={(value) => `${Math.round(value)} px`}
      onChange={(value) => onChange({ boxWidth: value }, "Width")} />}
    {inShape && <>
      <div className="photo-buttons">
        <button type="button" className={text.autoFit !== false ? "selected" : ""}
          onClick={() => onChange({ autoFit: text.autoFit === false }, "Auto-fit")}>
          Shrink to fit
        </button>
      </div>
      <p className="photo-note">
        With this on, the size above is a ceiling and the words shrink until they fit the balloon —
        which is the right way round for lettering: the art decides where the balloon goes, and the
        words give.
      </p>
    </>}
    <Slider label="Line height" value={text.lineHeight} min={0.7} max={3} step={0.01}
      format={(value) => value.toFixed(2)}
      onChange={(value) => onChange({ lineHeight: value }, "Line height")} />
    <Slider label="Letter spacing" value={text.letterSpacing} min={-0.1} max={0.5} step={0.005}
      format={(value) => `${(value * 100).toFixed(1)}%`}
      onChange={(value) => onChange({ letterSpacing: value }, "Letter spacing")} />
    <Slider label="Padding" value={text.padding} min={0} max={300} step={1}
      format={(value) => `${Math.round(value)} px`}
      onChange={(value) => onChange({ padding: value }, "Padding")} />

    <h4>Ink</h4>
    <PaintField label="Fill" paint={toPaint(text.color, "#ffffff")} preview={previews?.fill}
      onChange={(paint, note, coalesce) => onChange({ color: paint }, note, coalesce)} />
    <Slider label="Outline" value={text.strokeWidth} min={0} max={40} step={0.5}
      format={(value) => value ? `${value} px` : "off"}
      onChange={(value) => onChange({ strokeWidth: value }, "Outline")} />
    {text.strokeWidth > 0 &&
      <PaintField label="Outline" paint={toPaint(text.strokeColor, "#000000")} preview={previews?.stroke}
        onChange={(paint, note, coalesce) => onChange({ strokeColor: paint }, note, coalesce)} />}
    <Slider label="Drop shadow" value={text.shadow} min={0} max={60} step={1}
      format={(value) => value ? `${value} px` : "off"}
      onChange={(value) => onChange({ shadow: value }, "Shadow")} />
    {text.shadow > 0 &&
      <PaintField label="Shadow" paint={toPaint(text.shadowColor, "#000000")} preview={previews?.shadow}
        onChange={(paint, note, coalesce) => onChange({ shadowColor: paint }, note, coalesce)} />}
    <p className="photo-note">
      Every one of these takes a ramp as well as a flat colour — the shadow included, which the
      canvas cannot do on its own, so it is drawn as its own blurred silhouette and painted through
      itself. The outline goes under the fill rather than over it, so half of it hides behind the
      letterforms; stroking on top closes up thin type.
    </p>

    <h4>Panel behind it</h4>
    <div className="photo-buttons">
      {(["none", "box", "rounded"] as const).map((option) => <button key={option} type="button"
        className={text.panel === option ? "selected" : ""}
        onClick={() => onChange({ panel: option }, "Panel")}>{option}</button>)}
    </div>
    {text.panel !== "none" && <>
      <PaintField label="Panel" paint={toPaint(text.panelColor, "#000000")} preview={previews?.panel}
        onChange={(paint, note, coalesce) => onChange({ panelColor: paint }, note, coalesce)} />
      <Slider label="Border" value={text.panelBorder} min={0} max={30} step={0.5}
        format={(value) => value ? `${value} px` : "off"}
        onChange={(value) => onChange({ panelBorder: value }, "Panel")} />
      {text.panelBorder > 0 &&
        <PaintField label="Border" paint={toPaint(text.panelBorderColor, "#ffffff")}
          preview={previews?.panelBorder}
          onChange={(paint, note, coalesce) => onChange({ panelBorderColor: paint }, note, coalesce)} />}
    </>}
    <p className="photo-note">
      A text layer is a layer. It takes a blend mode, an opacity, its own grade and relight steps and
      the edge group, so a caption can be lit and coloured to sit in the picture rather than on it.
    </p>
  </>;
}


/* ----------------------------------------------------------------- liquify */

const WARP_MODES: Array<{ id: WarpMode; name: string; hint: string }> = [
  { id: "push", name: "Push", hint: "drags pixels along with the brush" },
  { id: "bloat", name: "Bloat", hint: "swells outward from the brush centre" },
  { id: "pinch", name: "Pinch", hint: "draws inward toward the brush centre" },
  { id: "twirl", name: "Twirl", hint: "rotates around the brush centre" },
];

/**
 * Liquify's controls.
 *
 * Four modes and two numbers is the whole tool, which is the point: everything it does is one
 * displacement field and one resample, so it never produces a colour the picture did not already
 * have. That also means it cannot fix a warp by painting over it -- the undo is per stroke, and it
 * is listed here rather than buried, because reaching for it is normal rather than a failure.
 */
export function WarpEditor({ node, brush, onBrush, onUndo, onClear }: {
  node: WarpNode;
  brush: { radius: number; strength: number; mode: WarpMode };
  onBrush: (brush: { radius: number; strength: number; mode: WarpMode }) => void;
  onUndo: () => void;
  onClear: () => void;
}) {
  const mode = WARP_MODES.find((option) => option.id === brush.mode) ?? WARP_MODES[0];
  return <>
    <div className="photo-buttons">
      {WARP_MODES.map((option) => <button key={option.id} type="button"
        className={brush.mode === option.id ? "selected" : ""}
        onClick={() => onBrush({ ...brush, mode: option.id })}>{option.name}</button>)}
    </div>
    <p className="photo-note">{mode.name} {mode.hint}. Repeated passes go further — one stroke is
      capped so a fast drag cannot tear the picture in a single dab.</p>
    <Slider label="Brush size" value={brush.radius} min={0.01} max={0.5} step={0.005}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onBrush({ ...brush, radius: value })} />
    <Slider label="Strength" value={brush.strength} min={0.02} max={1} step={0.01}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onBrush({ ...brush, strength: value })} />
    <div className="photo-buttons">
      <button type="button" disabled={!node.strokes.length} onClick={onUndo}>Undo stroke</button>
      <button type="button" disabled={!node.strokes.length} onClick={onClear}>Clear all</button>
    </div>
    <p className="photo-note">
      {node.strokes.length ? `${node.strokes.length} strokes` : "No strokes yet"} — stored as paths,
      so the warp is rebuilt at whatever size the picture is rendered at rather than baked at one.
    </p>
  </>;
}

/* ---------------------------------------------------------------- gradient */

const GRADIENT_KINDS: Array<{ id: GradientKind; name: string; hint: string }> = [
  { id: "linear", name: "Linear", hint: "runs one way across the frame" },
  { id: "reflected", name: "Reflected", hint: "mirrors about the centre line — for a light source inside the frame" },
  { id: "radial", name: "Radial", hint: "out from a point" },
  { id: "angular", name: "Angular", hint: "sweeps around a point" },
  { id: "diamond", name: "Diamond", hint: "square-cornered radial" },
];

/** Eight compass points, so a direction can be set exactly without dragging for it. */
const GRADIENT_ANGLES = [
  { label: "↑", angle: 270 }, { label: "↗", angle: 315 }, { label: "→", angle: 0 },
  { label: "↘", angle: 45 }, { label: "↓", angle: 90 }, { label: "↙", angle: 135 },
  { label: "←", angle: 180 }, { label: "↖", angle: 225 },
];

/**
 * The ramp editor.
 *
 * Two things the user asked for and one they did not know to. The direction is an angle, eight
 * one-click compass points, and a handle on the picture itself. The number of colours is however
 * many stops the list has — add and remove them freely, and each carries its own position, colour
 * and opacity, so a ramp can fade out as well as change hue.
 *
 * The one they did not ask for is the interpolation. Mixing two colours channel by channel is what
 * every naive gradient does and it is why so many of them have a dead grey band through the middle;
 * OKLab is spaced the way seeing is and keeps the crossing live. It is the default, with sRGB
 * available for when the job is matching a ramp made somewhere else.
 */
export function GradientEditor({ gradient, preview, onChange }: {
  gradient: Gradient; preview?: string;
  onChange: (over: Partial<Gradient>, label: string, coalesce?: boolean) => void;
}) {
  const stops = normalizeStops(gradient.stops);
  const [picked, setPicked] = useState(0);
  const active = stops[Math.min(picked, stops.length - 1)];

  const setStops = (next: GradientStop[], label: string, coalesce = false) =>
    onChange({ stops: next }, label, coalesce);
  const patchStop = (index: number, over: Partial<GradientStop>, label: string, coalesce = true) =>
    setStops(stops.map((stop, at) => at === index ? { ...stop, ...over } : stop), label, coalesce);

  const addStop = () => {
    // Dropped into the widest gap rather than at the end, so pressing it repeatedly spreads stops
    // out instead of piling them on the last one.
    let gapAt = 0.5, widest = -1;
    for (let index = 0; index < stops.length - 1; index++) {
      const gap = stops[index + 1].at - stops[index].at;
      if (gap > widest) { widest = gap; gapAt = (stops[index].at + stops[index + 1].at) / 2; }
    }
    const next = normalizeStops([...stops, {
      at: gapAt, color: sampleHex(gradient, gapAt), opacity: 1,
    }]);
    setStops(next, "Colour added");
    setPicked(next.findIndex((stop) => Math.abs(stop.at - gapAt) < 1e-6));
  };

  return <>
    <div className="photo-ramp">
      {/* The ramp is its own element rather than a background on the bar: an inline background-image
          replaces the whole shorthand, which would take the transparency chequerboard with it. */}
      <i className="photo-ramp-fill"
        style={preview ? { backgroundImage: `url("${preview}")` } : undefined} />
      {stops.map((stop, index) => <button key={index} type="button"
        className={`photo-ramp-stop${index === Math.min(picked, stops.length - 1) ? " on" : ""}`}
        title={`${stop.color} at ${Math.round(stop.at * 100)}%`}
        style={{ left: `${stop.at * 100}%`, background: stop.color }}
        onClick={() => setPicked(index)} />)}
    </div>

    <div className="photo-buttons">
      <button type="button" onClick={addStop}><Plus size={12} /> Colour</button>
      <button type="button" disabled={stops.length <= 2}
        title={stops.length > 2 ? "Remove the selected colour" : "A ramp needs at least two colours"}
        onClick={() => {
          setStops(stops.filter((_, at) => at !== picked), "Colour removed");
          setPicked((current) => Math.max(0, current - 1));
        }}><Trash2 size={12} /> Remove</button>
      <button type="button" title="Reverse the order of every colour"
        onClick={() => setStops(
          stops.map((stop) => ({ ...stop, at: 1 - stop.at })).reverse(), "Reversed")}>
        Flip
      </button>
      <button type="button" title="Space the colours evenly"
        onClick={() => setStops(
          stops.map((stop, index) => ({ ...stop, at: index / (stops.length - 1) })), "Spread")}>
        Even
      </button>
    </div>
    <p className="photo-note">{stops.length} colours. Click a marker on the ramp to edit it.</p>

    <label className="studio-field"><span>Colour {picked + 1}</span>
      <input type="color" value={active.color}
        onChange={(event) => patchStop(picked, { color: event.target.value }, "Colour")} />
    </label>
    <Slider label="Position" value={active.at} min={0} max={1} step={0.005}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => patchStop(picked, { at: value }, "Position")} />
    <Slider label="Opacity" value={active.opacity} min={0} max={1} step={0.01}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => patchStop(picked, { opacity: value }, "Opacity")} />

    <h4>Direction</h4>
    <label className="studio-field"><span>Shape</span>
      <select value={gradient.kind}
        onChange={(event) => onChange({ kind: event.target.value as GradientKind }, "Shape")}>
        {GRADIENT_KINDS.map((kind) => <option key={kind.id} value={kind.id}>{kind.name}</option>)}
      </select>
    </label>
    <p className="photo-note">
      {GRADIENT_KINDS.find((kind) => kind.id === gradient.kind)?.hint}
    </p>
    <div className="photo-buttons photo-compass">
      {GRADIENT_ANGLES.map((option) => <button key={option.angle} type="button"
        className={Math.round(gradient.angle) % 360 === option.angle ? "selected" : ""}
        title={`${option.angle}°`}
        onClick={() => onChange({ angle: option.angle }, "Direction")}>{option.label}</button>)}
    </div>
    <Slider label="Angle" value={gradient.angle} min={0} max={360} step={1}
      format={(value) => `${Math.round(value)}°`}
      onChange={(value) => onChange({ angle: value }, "Direction", true)} />
    <Slider label="Spread" value={gradient.scale} min={0.05} max={4} step={0.01}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onChange({ scale: value }, "Spread", true)} />
    <Slider label="Shift" value={gradient.offset} min={-1} max={1} step={0.005}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onChange({ offset: value }, "Shift", true)} />
    <Slider label="Centre across" value={gradient.cx} min={-0.5} max={1.5} step={0.005}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onChange({ cx: value }, "Centre", true)} />
    <Slider label="Centre down" value={gradient.cy} min={-0.5} max={1.5} step={0.005}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onChange({ cy: value }, "Centre", true)} />
    <p className="photo-note">
      Spread of 100% means the ramp exactly crosses the frame in whatever direction it is pointing,
      so the same number looks the same at every angle.
    </p>

    <h4>How it blends</h4>
    <div className="photo-buttons">
      <button type="button" className={gradient.space !== "rgb" ? "selected" : ""}
        onClick={() => onChange({ space: "oklab" }, "OKLab")}>OKLab</button>
      <button type="button" className={gradient.space === "rgb" ? "selected" : ""}
        onClick={() => onChange({ space: "rgb" }, "sRGB")}>sRGB</button>
    </div>
    <p className="photo-note">
      OKLab mixes the way the eye spaces colour, so blue into yellow crosses through a live teal.
      sRGB mixes the numbers, which sends the same pair through a grey-brown — occasionally what you
      want, usually not.
    </p>
    <Slider label="Repeat" value={gradient.repeat} min={1} max={12} step={1}
      format={(value) => value === 1 ? "once" : `${Math.round(value)}×`}
      onChange={(value) => onChange({ repeat: value }, "Repeat")} />
    <div className="photo-buttons">
      <button type="button" className={gradient.mirror ? "selected" : ""}
        title="Ping-pong instead of restarting, so a repeat has no hard seam"
        onClick={() => onChange({ mirror: !gradient.mirror }, "Mirror")}>Mirror</button>
    </div>
    <Slider label="Ease" value={gradient.ease} min={0.25} max={4} step={0.05}
      format={(value) => value.toFixed(2)}
      onChange={(value) => onChange({ ease: value }, "Ease", true)} />
    <Slider label="Dither" value={gradient.dither} min={0} max={1} step={0.05}
      format={(value) => value ? `${Math.round(value * 100)}%` : "off"}
      onChange={(value) => onChange({ dither: value }, "Dither", true)} />
    <p className="photo-note">
      A smooth ramp has more pixels to cross than an eight-bit picture has values, so it bands.
      Half a level of ordered noise removes the stripes and is itself invisible — leave it on.
    </p>

    <h4>Start from</h4>
    <div className="photo-buttons">
      {GRADIENT_PRESETS.map((preset) => <button key={preset.name} type="button"
        onClick={() => onChange({
          ...preset.gradient,
          // Copied, because the presets are module-level objects and a stored reference would let
          // editing one document's ramp change the preset for every other.
          stops: (preset.gradient.stops ?? []).map((stop) => ({ ...stop })),
        }, preset.name)}>{preset.name}</button>)}
    </div>
  </>;
}

/** The colour the ramp already has at a point, so a new stop lands where the eye expects it. */
function sampleHex(gradient: Gradient, at: number): string {
  const lut = buildGradientLut(gradient, 256);
  const index = Math.min(255, Math.max(0, Math.round(at * 255))) * 4;
  return rgbToHex(lut[index], lut[index + 1], lut[index + 2]);
}

/** The gradient map's own card: a ramp, plus how far the picture travels toward it. */
export function GradientMapEditor({ node, preview, onChange }: {
  node: GradientMapNode; preview?: string;
  onChange: (over: Partial<GradientMapNode>, label: string, coalesce?: boolean) => void;
}) {
  return <>
    <p className="photo-note">
      Shadows take the colour at one end of the ramp and highlights the other, with everything
      between following its own brightness. Duotone and split-toning, from the same editor that
      paints a fill.
    </p>
    <Slider label="Amount" value={node.amount} min={0} max={100} step={1}
      format={(value) => `${Math.round(value)}%`}
      onChange={(value) => onChange({ amount: value }, "Amount", true)} />
    <GradientEditor gradient={node.gradient} preview={preview}
      onChange={(over, label, coalesce) =>
        onChange({ gradient: { ...node.gradient, ...over } }, label, coalesce)} />
  </>;
}

/* ------------------------------------------------------------------ paints */

/**
 * A colour that might be a ramp.
 *
 * Folded behind a swatch on purpose. A text block has five paints and a shape has two, and putting
 * seven full gradient editors on one card would bury the two controls anyone actually reaches for.
 * Closed it is a swatch and a pair of buttons; opened it is the whole ramp editor, the same one the
 * gradient layer uses.
 */
export function PaintField({ label, paint, preview, onChange }: {
  label: string; paint: Paint; preview?: string;
  onChange: (paint: Paint, label: string, coalesce?: boolean) => void;
}) {
  const gradient = paint.kind === "gradient" ? paint.gradient : undefined;
  return <div className="photo-paint">
    <div className="photo-paint-head">
      <span>{label}</span>
      {gradient
        ? <i className="photo-paint-swatch ramp"
            style={preview ? { backgroundImage: `url("${preview}")` } : undefined} />
        : <input type="color" value={paint.kind === "solid" ? paint.color : "#ffffff"}
            onChange={(event) => onChange(solidPaint(event.target.value), label, true)} />}
      <button type="button" className={gradient ? "" : "selected"}
        onClick={() => onChange(solidPaint(paintColor(paint)), `${label} · solid`)}>Solid</button>
      <button type="button" className={gradient ? "selected" : ""}
        onClick={() => onChange(gradient
          ? paint
          // Seeded from the colour it had, so switching to a gradient starts somewhere recognisable
          // rather than at an unrelated black-to-white.
          : { kind: "gradient", gradient: {
              ...NEUTRAL_GRADIENT,
              stops: [
                { at: 0, color: paintColor(paint), opacity: 1 },
                { at: 1, color: "#ffffff", opacity: 1 },
              ],
            } }, `${label} · gradient`)}>Ramp</button>
    </div>
    {gradient && <details className="photo-paint-body">
      <summary>Edit the ramp</summary>
      <GradientEditor gradient={gradient} preview={preview}
        onChange={(over, note, coalesce) =>
          onChange({ kind: "gradient", gradient: { ...gradient, ...over } }, note, coalesce)} />
    </details>}
  </div>;
}

/* ------------------------------------------------------------------ shapes */

const SHAPE_KINDS: Array<{ group: string; kinds: Array<{ id: ShapeKind; name: string }> }> = [
  { group: "Balloons", kinds: [
    { id: "speech", name: "Speech" }, { id: "thought", name: "Thought" },
    { id: "shout", name: "Shout" }, { id: "scream", name: "Scream" },
    { id: "electric", name: "Broadcast" }, { id: "icicle", name: "Dripping" },
    { id: "wobble", name: "Wobble" }, { id: "whisper", name: "Whisper" },
    { id: "caption", name: "Caption" },
  ] },
  { group: "Shapes", kinds: [
    { id: "rectangle", name: "Box" }, { id: "rounded", name: "Rounded" },
    { id: "ellipse", name: "Ellipse" }, { id: "polygon", name: "Polygon" },
    { id: "star", name: "Star" }, { id: "burst", name: "Burst" },
    { id: "heart", name: "Heart" }, { id: "lightning", name: "Bolt" },
    { id: "splat", name: "Splat" }, { id: "arrow", name: "Arrow" },
    { id: "banner", name: "Banner" },
  ] },
  { group: "Behind the action", kinds: [
    { id: "impact", name: "Impact" }, { id: "sunburst", name: "Sunburst" },
    { id: "speedlines", name: "Speed lines" },
  ] },
];

/**
 * What the repeating feature round this shape's edge is actually called.
 *
 * One slider underneath, because they are one number; but "Points" on a drip balloon and "Depth" on
 * a set of speed lines are the kind of label that makes a control feel like it belongs to some other
 * tool.
 */
function countLabel(kind: ShapeKind): string {
  if (kind === "icicle") return "Drips";
  if (kind === "wobble") return "Waves";
  if (kind === "thought" || kind === "splat") return "Lobes";
  if (kind === "impact" || kind === "sunburst") return "Rays";
  if (kind === "speedlines") return "Lines";
  if (kind === "polygon") return "Sides";
  return "Points";
}

function depthLabel(kind: ShapeKind): string {
  if (kind === "icicle") return "Drip length";
  if (kind === "wobble") return "Wave depth";
  if (kind === "impact") return "Hole";
  if (kind === "speedlines") return "Thickness";
  if (kind === "splat") return "Roundness";
  return "Depth";
}

/**
 * A shape's own controls.
 *
 * Rectangle, rounded rectangle and ellipse share one roundness slider because they are one family
 * underneath -- a superellipse with a single exponent -- so there is no seam in the middle where a
 * "rounded rectangle" stops and an "ellipse" starts.
 */
export function ShapeEditor({ shape, fillPreview, strokePreview, onChange }: {
  shape: ShapeBlock; fillPreview?: string; strokePreview?: string;
  onChange: (over: Partial<ShapeBlock>, label: string, coalesce?: boolean) => void;
}) {
  const balloon = BALLOON_KINDS.includes(shape.kind);
  // Everything with a repeating feature round its edge shares one count and one depth, whether the
  // feature is a spike, a wave, a drip, a lobe or a ray.
  const REPEATING: ShapeKind[] = [
    "polygon", "star", "burst", "shout", "scream", "electric", "icicle", "wobble", "splat",
    "thought", "impact", "sunburst", "speedlines",
  ];
  const repeating = REPEATING.includes(shape.kind);
  const hasDepth = repeating && shape.kind !== "polygon" && shape.kind !== "thought";
  // Only the shapes whose corners are actually irregular have anything to shuffle.
  const IRREGULAR: ShapeKind[] = [
    "star", "burst", "shout", "scream", "electric", "splat", "impact", "sunburst", "speedlines",
  ];
  const irregular = IRREGULAR.includes(shape.kind);
  const round = shape.kind === "rounded" || shape.kind === "caption" || shape.kind === "whisper"
    || shape.kind === "icicle";
  return <>
    <label className="studio-field"><span>Shape</span>
      <select value={shape.kind}
        onChange={(event) => onChange({ kind: event.target.value as ShapeKind }, "Shape")}>
        {SHAPE_KINDS.map((group) => <optgroup key={group.group} label={group.group}>
          {group.kinds.map((kind) => <option key={kind.id} value={kind.id}>{kind.name}</option>)}
        </optgroup>)}
      </select>
    </label>
    <Slider label="Width" value={shape.width} min={24} max={4000} step={2}
      format={(value) => `${Math.round(value)} px`}
      onChange={(value) => onChange({ width: value }, "Width", true)} />
    <Slider label="Height" value={shape.height} min={24} max={4000} step={2}
      format={(value) => `${Math.round(value)} px`}
      onChange={(value) => onChange({ height: value }, "Height", true)} />
    {round && <Slider label="Roundness" value={shape.radius} min={0} max={1} step={0.01}
      format={(value) => `${Math.round(value * 100)}%`}
      onChange={(value) => onChange({ radius: value }, "Roundness", true)} />}
    {repeating && <>
      <Slider label={countLabel(shape.kind)} value={shape.points} min={3} max={48} step={1}
        format={(value) => String(Math.round(value))}
        onChange={(value) => onChange({ points: value }, "Count")} />
      {hasDepth &&
        <Slider label={depthLabel(shape.kind)} value={shape.innerRadius} min={0.05} max={0.98} step={0.01}
          format={(value) => `${Math.round(value * 100)}%`}
          onChange={(value) => onChange({ innerRadius: value }, "Depth", true)} />}
      <Slider label="Turn points" value={shape.spin ?? 0} min={-180} max={180} step={1}
        format={(value) => `${Math.round(value)}°`}
        onChange={(value) => onChange({ spin: value }, "Turn", true)} />
    </>}
    {irregular && <>
      <Slider label="Irregular" value={shape.jitter ?? 0} min={0} max={1} step={0.01}
        format={(value) => value ? `${Math.round(value * 100)}%` : "even"}
        onChange={(value) => onChange({ jitter: value }, "Irregular", true)} />
      <div className="photo-buttons">
        <button type="button" disabled={!(shape.jitter ?? 0)}
          title="A different set of uneven spikes. The shape is fixed once chosen, so it never crawls."
          onClick={() => onChange({ seed: Math.max(1, Math.round(Math.random() * 9999)) }, "Reshuffled")}>
          <RefreshCw size={12} /> Reshuffle
        </button>
      </div>
      <p className="photo-note">
        A perfectly even starburst reads as a sale sticker. Uneven spikes are most of what makes one
        read as drawn — and the pattern is fixed once picked, so it does not shimmer as you work.
      </p>
    </>}
    {shape.kind === "arrow" &&
      <Slider label="Shaft" value={shape.innerRadius} min={0.05} max={0.9} step={0.01}
        format={(value) => `${Math.round(value * 100)}%`}
        onChange={(value) => onChange({ innerRadius: value }, "Shaft", true)} />}

    <h4>Ink</h4>
    <PaintField label="Fill" paint={toPaint(shape.fill, "#ffffff")} preview={fillPreview}
      onChange={(paint, note, coalesce) => onChange({ fill: paint }, note, coalesce)} />
    <PaintField label="Outline" paint={toPaint(shape.stroke, "#000000")} preview={strokePreview}
      onChange={(paint, note, coalesce) => onChange({ stroke: paint }, note, coalesce)} />
    <Slider label="Outline width" value={shape.strokeWidth} min={0} max={60} step={0.5}
      format={(value) => value ? `${value} px` : "off"}
      onChange={(value) => onChange({ strokeWidth: value }, "Outline", true)} />
    <div className="photo-buttons">
      <button type="button" className={shape.dashed ? "selected" : ""}
        title="A dashed outline is what makes a whisper a whisper"
        onClick={() => onChange({ dashed: !shape.dashed }, "Dashes")}>Dashed</button>
    </div>
    <Slider label="Drop shadow" value={shape.shadow} min={0} max={80} step={1}
      format={(value) => value ? `${value} px` : "off"}
      onChange={(value) => onChange({ shadow: value }, "Shadow", true)} />
    {shape.shadow > 0 && <>
      <label className="studio-field"><span>Shadow colour</span>
        <input type="color" value={shape.shadowColor}
          onChange={(event) => onChange({ shadowColor: event.target.value }, "Shadow", true)} />
      </label>
      <Slider label="Shadow drop" value={shape.shadowOffset} min={-60} max={60} step={1}
        format={(value) => `${Math.round(value)} px`}
        onChange={(value) => onChange({ shadowOffset: value }, "Shadow", true)} />
    </>}

    {balloon && <>
      <h4>Tail</h4>
      <p className="photo-note">
        Drag the tail&rsquo;s tip on the picture to point it at whoever is speaking. The tail and the
        balloon are one outline, so the stroke runs round both without a seam.
      </p>
      <Slider label="Direction" value={shape.tailAngle} min={0} max={360} step={1}
        format={(value) => `${Math.round(value)}°`}
        onChange={(value) => onChange({ tailAngle: value }, "Tail", true)} />
      <Slider label="Length" value={shape.tailLength} min={0} max={2.5} step={0.01}
        format={(value) => value < 0.03 ? "none" : `${Math.round(value * 100)}%`}
        onChange={(value) => onChange({ tailLength: value }, "Tail", true)} />
      {shape.kind !== "thought" && <>
        <Slider label="Width" value={shape.tailWidth} min={3} max={80} step={1}
          format={(value) => `${Math.round(value)}°`}
          onChange={(value) => onChange({ tailWidth: value }, "Tail", true)} />
        <Slider label="Bend" value={shape.tailCurve} min={-1} max={1} step={0.02}
          format={(value) => value.toFixed(2)}
          onChange={(value) => onChange({ tailCurve: value }, "Tail", true)} />
      </>}
    </>}

    <h4>Lettering</h4>
    <Slider label="Padding" value={shape.padding} min={0} max={200} step={1}
      format={(value) => `${Math.round(value)} px`}
      onChange={(value) => onChange({ padding: value }, "Padding", true)} />
    <p className="photo-note">
      Words go in the largest box that fits inside the outline, worked out from the same maths that
      draws it — so they cannot poke through the side of a balloon.
    </p>

    <h4>Start from</h4>
    <div className="photo-buttons">
      {SHAPE_PRESETS.map((preset) => <button key={preset.name} type="button"
        onClick={() => onChange(preset.shape, preset.name)}>{preset.name}</button>)}
    </div>
  </>;
}
