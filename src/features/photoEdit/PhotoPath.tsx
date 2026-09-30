/**
 * VFX Phase 6: the vector tools' panels and on-canvas handles -- the Path card, the Repeat card, the
 * pen tool's anchors and handles, and the envelope's control points (on a path, and as a node on
 * any layer). Vectors are pictures, so all of this shows on both surfaces (Photo Edit and Clip
 * Editor -> VFX); none of it is a video control.
 *
 * The maths lives in the domain (`photoPath`, `photoPathOps`, `photoPathActions`, `photoEnvelope`,
 * `photoRepeat`); these components only map pointer positions and call it.
 */

import { Fragment, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Combine, Grid3x3, PenTool, Repeat, Scissors, Spline, Trash2 } from "lucide-react";
import type { EnvelopeNode, LayerRepeat, PhotoLayer } from "../../domain/photoDocument";
import { createEnvelope, isIdentityEnvelope } from "../../domain/photoEnvelope";
import { applyMatrix, invert } from "../../domain/photoGeometry";
import type { Paint } from "../../domain/photoGradient";
import {
  closeSubpath, deleteAnchor, moveAnchor, pathData, setHandle, shapedPath, toggleSmooth,
  type PathBlock, type PathCap, type PathJoin, type Point,
} from "../../domain/photoPath";
import { pathToCanvas, type BooleanOp } from "../../domain/photoPathOps";
import { Handle } from "./PhotoHandles";
import { PaintField, Slider } from "./PhotoPanels";

type Canvas = { width: number; height: number };

/* ------------------------------------------------------------ mapping */

/** Path coordinates <-> normalised canvas for one path layer. */
export function pathMapping(layer: PhotoLayer, path: PathBlock, canvas: Canvas) {
  const forward = pathToCanvas(layer, path, canvas.width, canvas.height);
  const inverse = invert(forward);
  return {
    toCanvas: (p: Point): Point => { const [x, y] = applyMatrix(forward, p[0], p[1]); return [x / canvas.width, y / canvas.height]; },
    toLocal: (at: Point): Point => applyMatrix(inverse, at[0] * canvas.width, at[1] * canvas.height),
    /** Path units per canvas pixel (for hit radii). */
    unitsPerPixel: 1 / Math.max(1e-6, Math.sqrt(Math.abs(forward.a * forward.d - forward.b * forward.c))),
  };
}

/* ------------------------------------------------------------ Path card */

const JOINS: PathJoin[] = ["miter", "round", "bevel"];
const CAPS: PathCap[] = ["butt", "round", "square"];
const SOLID = (color: string): Paint => ({ kind: "solid", color });

export function PathEditor({ layer, layers, busy, envelopeEditing, onChange, onCombine, onOutlineStroke, onClipAbove, onMaskBelow, onBlend, onEnvelope, onEditEnvelope }: {
  layer: PhotoLayer; layers: PhotoLayer[]; busy?: boolean; envelopeEditing: boolean;
  onChange: (over: Partial<PathBlock>, label: string, coalesce?: boolean) => void;
  onCombine: (op: BooleanOp | "compound") => void;
  onOutlineStroke: () => void;
  onClipAbove: () => void;
  onMaskBelow: () => void;
  onBlend: (otherId: string, steps: number) => void;
  onEnvelope: (on: boolean) => void;
  onEditEnvelope: (on: boolean) => void;
}) {
  const path = layer.path!;
  const index = layers.findIndex((item) => item.id === layer.id);
  const below = layers.slice(0, index).reverse().find((item) => item.path);
  const others = layers.filter((item) => item.path && item.id !== layer.id);
  const [blendWith, setBlendWith] = useState<string>();
  const [steps, setSteps] = useState(5);
  const target = others.find((item) => item.id === blendWith) ?? others[others.length - 1];
  const points = path.subpaths.reduce((sum, subpath) => sum + subpath.nodes.length, 0);
  const last = path.subpaths.length - 1;
  const dashText = (path.dash ?? []).join(" ");
  return <>
    <p className="photo-note">
      {path.subpaths.length} subpath{path.subpaths.length === 1 ? "" : "s"}, {points} points.
      With the <PenTool size={11} /> pen: click for a corner, drag for a smooth point, click the first point to close,
      Esc or Enter to finish. Drag points and handles to edit; alt-click a point to switch corner/smooth,
      Delete removes the last point you touched. Shift-click with the pen adds a subpath to this layer.
    </p>
    <h4>Ink</h4>
    <div className="photo-buttons">
      <button type="button" className={path.fill ? "selected" : ""}
        onClick={() => onChange({ fill: path.fill ? undefined : SOLID("#f2c230") }, path.fill ? "No fill" : "Filled")}>Fill</button>
      <button type="button" className={path.stroke && path.strokeWidth > 0 ? "selected" : ""}
        onClick={() => onChange(path.stroke && path.strokeWidth > 0 ? { stroke: undefined } : { stroke: path.stroke ?? SOLID("#1b1b1b"), strokeWidth: path.strokeWidth || 4 }, "Stroke")}>Stroke</button>
    </div>
    {path.fill && <PaintField label="Fill" paint={path.fill} onChange={(paint, note, coalesce) => onChange({ fill: paint }, note, coalesce)} />}
    {path.fill && <label className="studio-field"><span>Fill rule</span>
      <select value={path.fillRule} onChange={(event) => onChange({ fillRule: event.target.value as PathBlock["fillRule"] }, "Fill rule")}>
        <option value="nonzero">Non-zero (overlaps fill)</option>
        <option value="evenodd">Even-odd (overlaps cut holes)</option>
      </select>
    </label>}
    {path.stroke && <>
      <PaintField label="Stroke" paint={path.stroke} onChange={(paint, note, coalesce) => onChange({ stroke: paint }, note, coalesce)} />
      <Slider label="Width" value={path.strokeWidth} min={0} max={120} step={0.5}
        format={(value) => value ? `${value} px` : "off"} onChange={(value) => onChange({ strokeWidth: value }, "Stroke width", true)} />
      <label className="studio-field"><span>Corners</span>
        <select value={path.join} onChange={(event) => onChange({ join: event.target.value as PathJoin }, "Corners")}>
          {JOINS.map((join) => <option key={join} value={join}>{join}</option>)}
        </select>
      </label>
      {path.join === "miter" && <Slider label="Miter limit" value={path.miterLimit} min={1} max={20} step={0.5}
        format={(value) => value.toFixed(1)} onChange={(value) => onChange({ miterLimit: value }, "Miter limit", true)} />}
      <label className="studio-field"><span>Ends</span>
        <select value={path.cap} onChange={(event) => onChange({ cap: event.target.value as PathCap }, "Ends")}>
          {CAPS.map((cap) => <option key={cap} value={cap}>{cap}</option>)}
        </select>
      </label>
      <label className="studio-field"><span>Dashes (px)</span>
        <input type="text" defaultValue={dashText} key={dashText} placeholder="e.g. 12 6"
          onBlur={(event) => {
            const dash = event.target.value.split(/[\s,]+/).map(Number).filter((value) => Number.isFinite(value) && value >= 0);
            onChange({ dash: dash.some((value) => value > 0) ? dash : undefined }, dash.length ? "Dashes" : "Solid stroke");
          }} />
      </label>
      {path.dash && <Slider label="Dash offset" value={path.dashOffset ?? 0} min={-200} max={200} step={0.5}
        format={(value) => `${value} px`} onChange={(value) => onChange({ dashOffset: value }, "Dash offset", true)} />}
      <div className="photo-buttons">
        <button type="button" className={path.paintOrder === "stroke" ? "selected" : ""}
          title="Draw the stroke under the fill, so only its outer half shows (how text outlines are drawn)"
          onClick={() => onChange({ paintOrder: path.paintOrder === "stroke" ? undefined : "stroke" }, "Stroke order")}>Stroke behind fill</button>
      </div>
    </>}

    <h4>Points</h4>
    <div className="photo-buttons">
      {last >= 0 && <button type="button" onClick={() => onChange({ subpaths: closeSubpath(path, last, !path.subpaths[last].closed).subpaths }, path.subpaths[last].closed ? "Opened" : "Closed")}>
        {path.subpaths[last]?.closed ? "Open last subpath" : "Close last subpath"}</button>}
    </div>

    <h4>Combine with the path below{below ? ` (${below.name})` : ""}</h4>
    <div className="photo-buttons">
      {(["unite", "subtract", "intersect", "exclude"] as const).map((op) =>
        <button key={op} type="button" disabled={!below || busy}
          title={op === "subtract" ? "The layer below minus this one (keeps the lower layer's paint)" : "Keeps this layer's paint"}
          onClick={() => onCombine(op)}><Combine size={12} /> {op[0].toUpperCase() + op.slice(1)}</button>)}
      <button type="button" disabled={!below || busy} title="Both outlines in one path: overlaps become holes by winding"
        onClick={() => onCombine("compound")}>Compound</button>
    </div>
    <div className="photo-buttons">
      <button type="button" disabled={!path.stroke || path.strokeWidth <= 0} onClick={onOutlineStroke}
        title="The stroke becomes its own filled path layer above"><Spline size={12} /> Outline stroke</button>
      <button type="button" disabled={index + 1 >= layers.length} onClick={onClipAbove}
        title="Clip the layer directly above to this path"><Scissors size={12} /> Clip layer above</button>
      <button type="button" disabled={index < 1} onClick={onMaskBelow}
        title="The path's fill becomes a vector mask on the layer directly below; this layer is hidden">Mask layer below</button>
    </div>

    <h4>Envelope</h4>
    <div className="photo-buttons">
      <button type="button" className={path.envelope ? "selected" : ""} onClick={() => onEnvelope(!path.envelope)}>
        <Grid3x3 size={12} /> {path.envelope ? "Remove envelope" : "Add envelope"}</button>
      {path.envelope && <button type="button" className={envelopeEditing ? "selected" : ""}
        onClick={() => onEditEnvelope(!envelopeEditing)}>Edit on canvas</button>}
      {path.envelope && !isIdentityEnvelope(path.envelope) && <button type="button"
        onClick={() => onChange({ envelope: createEnvelope(path.envelope!.rect, path.envelope!.cols, path.envelope!.rows) }, "Envelope reset")}>Reset</button>}
    </div>
    {path.envelope && <p className="photo-note">A 4 × 4 Bézier mesh: drag its points on the canvas. The curves stay curves at any size.</p>}

    <h4>Make Blend</h4>
    {others.length ? <>
      <label className="studio-field"><span>With</span>
        <select value={target?.id} onChange={(event) => setBlendWith(event.target.value)}>
          {others.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select>
      </label>
      <Slider label="Steps" value={steps} min={1} max={60} step={1} format={(value) => String(value)} onChange={setSteps} />
      <div className="photo-buttons">
        <button type="button" disabled={!target || busy} onClick={() => target && onBlend(target.id, steps)}>
          <Repeat size={12} /> Make blend</button>
      </div>
    </> : <p className="photo-note">Needs a second path layer.</p>}
  </>;
}

/* ----------------------------------------------------------- Repeat card */

export function RepeatEditor({ layer, layers, onChange }: {
  layer: PhotoLayer; layers: PhotoLayer[];
  onChange: (repeat: LayerRepeat | undefined, label: string, coalesce?: boolean) => void;
}) {
  const repeat = layer.repeat;
  const guides = layers.filter((item) => item.path && item.id !== layer.id);
  const set = (over: Partial<LayerRepeat>, label: string, coalesce = true) => repeat && onChange({ ...repeat, ...over } as LayerRepeat, label, coalesce);
  return <>
    <label className="studio-field"><span>Repeat</span>
      <select value={repeat?.kind ?? "none"} onChange={(event) => {
        const kind = event.target.value;
        onChange(kind === "grid" ? { kind: "grid", columns: 4, rows: 3, gapX: 120, gapY: 120 }
          : kind === "radial" ? { kind: "radial", count: 8, radius: 150, rotate: true }
            : kind === "path" && guides.length ? { kind: "path", layerId: guides[guides.length - 1].id, count: 10, rotate: true }
              : undefined, kind === "none" ? "No repeat" : `Repeat: ${kind}`, false);
      }}>
        <option value="none">None</option>
        <option value="grid">Grid</option>
        <option value="radial">Around a ring</option>
        <option value="path" disabled={!guides.length}>Along a path{guides.length ? "" : " (needs a path layer)"}</option>
      </select>
    </label>
    {repeat?.kind === "grid" && <>
      <Slider label="Columns" value={repeat.columns} min={1} max={40} step={1} format={String} onChange={(value) => set({ columns: value }, "Columns")} />
      <Slider label="Rows" value={repeat.rows} min={1} max={40} step={1} format={String} onChange={(value) => set({ rows: value }, "Rows")} />
      <Slider label="Across" value={repeat.gapX} min={-2000} max={2000} step={1} format={(value) => `${value} px`} onChange={(value) => set({ gapX: value }, "Spacing")} />
      <Slider label="Down" value={repeat.gapY} min={-2000} max={2000} step={1} format={(value) => `${value} px`} onChange={(value) => set({ gapY: value }, "Spacing")} />
      <Slider label="Brick shift" value={repeat.shift ?? 0} min={-1} max={1} step={0.05} format={(value) => `${Math.round(value * 100)}%`} onChange={(value) => set({ shift: value }, "Brick shift")} />
    </>}
    {repeat?.kind === "radial" && <>
      <Slider label="Copies" value={repeat.count} min={1} max={120} step={1} format={String} onChange={(value) => set({ count: value }, "Copies")} />
      <Slider label="Radius" value={repeat.radius} min={0} max={3000} step={1} format={(value) => `${value} px`} onChange={(value) => set({ radius: value }, "Radius")} />
      <Slider label="Start" value={repeat.startAngle ?? 0} min={-180} max={180} step={1} format={(value) => `${value}°`} onChange={(value) => set({ startAngle: value }, "Start")} />
      <Slider label="Sweep" value={repeat.sweep ?? 360} min={-360} max={360} step={1} format={(value) => `${value}°`} onChange={(value) => set({ sweep: value }, "Sweep")} />
    </>}
    {repeat?.kind === "path" && <>
      <label className="studio-field"><span>Along</span>
        <select value={repeat.layerId} onChange={(event) => set({ layerId: event.target.value }, "Guide path", false)}>
          {guides.map((item) => <option key={item.id} value={item.id}>{item.name}</option>)}
        </select>
      </label>
      <Slider label="Copies" value={repeat.count} min={1} max={200} step={1} format={String} onChange={(value) => set({ count: value }, "Copies")} />
      <Slider label="From" value={repeat.start ?? 0} min={0} max={1} step={0.01} format={(value) => `${Math.round(value * 100)}%`} onChange={(value) => set({ start: value }, "From")} />
      <Slider label="To" value={repeat.end ?? 1} min={0} max={1} step={0.01} format={(value) => `${Math.round(value * 100)}%`} onChange={(value) => set({ end: value }, "To")} />
    </>}
    {repeat && repeat.kind !== "grid" && <div className="photo-buttons">
      <button type="button" className={repeat.rotate ? "selected" : ""} onClick={() => set({ rotate: !repeat.rotate }, "Turn copies", false)}>Turn copies</button>
    </div>}
    {repeat && <>
      <Slider label="Scale each" value={repeat.scaleStep ?? 1} min={0.5} max={1.5} step={0.01} format={(value) => `×${value.toFixed(2)}`} onChange={(value) => set({ scaleStep: value }, "Scale step")} />
      <Slider label="Fade each" value={repeat.opacityStep ?? 1} min={0.5} max={1} step={0.01} format={(value) => `×${value.toFixed(2)}`} onChange={(value) => set({ opacityStep: value }, "Fade step")} />
      <p className="photo-note">Copies are ordinary layers when drawn: same steps, masks and blend. A layer clipped to a repeated one clips to its last copy.</p>
    </>}
  </>;
}

/* ------------------------------------------------------ on-canvas handles */

/**
 * The active path layer's outline, anchors and handles (with the pen tool), or its envelope's
 * control points (while editing the envelope). `atDisplay` maps normalised canvas to the display.
 */
export function PathOverlay({ layer, canvas, atDisplay, toCanvas, envelope, drawing, onChange, onTouch, onCloseDrawing }: {
  layer: PhotoLayer; canvas: Canvas;
  atDisplay: (u: number, v: number) => Point;
  toCanvas: (event: ReactPointerEvent<Element>) => Point | undefined;
  /** Show the envelope's points instead of the anchors. */
  envelope: boolean;
  /** The subpath the pen is appending to, if any. */
  drawing?: number;
  onChange: (path: PathBlock, label: string, coalesce?: boolean) => void;
  onTouch: (at: [number, number] | undefined) => void;
  onCloseDrawing: () => void;
}) {
  const path = layer.path!;
  const map = pathMapping(layer, path, canvas);
  const display = (p: Point) => { const [u, v] = map.toCanvas(p); return atDisplay(u, v); };
  const d = pathData(shapedPath(path), (p) => { const [x, y] = display(p); return [x * 100, y * 100]; });
  const local = (event: ReactPointerEvent<Element>) => { const at = toCanvas(event); return at ? map.toLocal(at) : undefined; };

  if (envelope && path.envelope) {
    const env = path.envelope;
    const grid: string[] = [];
    for (let j = 0; j <= env.rows; j++) grid.push(env.points.slice(j * (env.cols + 1), (j + 1) * (env.cols + 1)).map((p) => display(p).map((v) => v * 100).join(",")).join(" "));
    for (let i = 0; i <= env.cols; i++) grid.push(Array.from({ length: env.rows + 1 }, (_, j) => display(env.points[j * (env.cols + 1) + i]).map((v) => v * 100).join(",")).join(" "));
    return <>
      <svg className="photo-frame" viewBox="0 0 100 100" preserveAspectRatio="none">
        <path d={d} className="photo-path-outline" />
        {grid.map((points, k) => <polyline key={k} points={points} className="photo-path-mesh" />)}
      </svg>
      {env.points.map((p, k) => <Handle key={k} at={display(p)} className="gradient end" title="Drag to bend"
        onStart={() => undefined}
        onMove={(event) => {
          const to = local(event);
          if (to) onChange({ ...path, envelope: { ...env, points: env.points.map((q, i) => i === k ? to : q) } }, "Envelope", true);
        }} />)}
    </>;
  }

  return <>
    <svg className="photo-frame" viewBox="0 0 100 100" preserveAspectRatio="none">
      <path d={d} className="photo-path-outline" />
      {path.subpaths.flatMap((subpath, s) => subpath.nodes.flatMap((node, n) => (["in", "out"] as const)
        .filter((side) => node[side]).map((side) => {
          const a = display([node.x, node.y]), h = display([node.x + node[side]![0], node.y + node[side]![1]]);
          return <line key={`${s}-${n}-${side}`} x1={a[0] * 100} y1={a[1] * 100} x2={h[0] * 100} y2={h[1] * 100} className="photo-path-arm" />;
        })))}
    </svg>
    {path.subpaths.map((subpath, s) => subpath.nodes.map((node, n) => <Fragment key={`a-${s}-${n}`}>
      <Handle at={display([node.x, node.y])} className={`path-anchor${drawing === s && n === 0 ? " start" : ""}`}
        title={drawing === s && n === 0 && subpath.nodes.length > 1 ? "Click to close the path" : "Drag to move · alt-click: corner/smooth"}
        onStart={(event) => {
          onTouch([s, n]);
          if (drawing === s && n === 0 && subpath.nodes.length > 1) {
            onChange(closeSubpath(path, s), "Path closed");
            onCloseDrawing();
          } else if (event.altKey) onChange(toggleSmooth(path, s, n), "Corner / smooth");
        }}
        onMove={(event) => { const to = local(event); if (to) onChange(moveAnchor(path, s, n, to), "Point moved", true); }} />
      {(["in", "out"] as const).filter((side) => node[side]).map((side) =>
        <Handle key={side} at={display([node.x + node[side]![0], node.y + node[side]![1]])} className="path-handle"
          title="Drag to shape the curve · alt-drag moves this side alone"
          onStart={() => onTouch([s, n])}
          onMove={(event) => {
            const to = local(event);
            if (!to) return;
            const both = node.in && node.out;
            onChange(setHandle(path, s, n, side, to, event.altKey || !both ? false : "angle"), "Curve", true);
          }} />)}
    </Fragment>))}
  </>;
}

/** Delete one anchor (the Delete key with the pen). */
export const withoutAnchor = (path: PathBlock, at: [number, number]): PathBlock => deleteAnchor(path, at[0], at[1]);

/* -------------------------------------------------- envelope node (pixels) */

export function EnvelopeNodeEditor({ node, editing, onEdit, onChange }: {
  node: EnvelopeNode; editing: boolean; onEdit: (on: boolean) => void;
  onChange: (node: EnvelopeNode, label: string) => void;
}) {
  const env = node.envelope;
  return <>
    <p className="photo-note">
      Bends this layer&rsquo;s pixels with a {env.cols + 1} × {env.rows + 1} Bézier mesh over the rectangle it was made on.
      Drag its points on the canvas.
    </p>
    <div className="photo-buttons">
      <button type="button" className={editing ? "selected" : ""} onClick={() => onEdit(!editing)}><Grid3x3 size={12} /> Edit on canvas</button>
      <button type="button" disabled={isIdentityEnvelope(env)}
        onClick={() => onChange({ ...node, envelope: createEnvelope(env.rect, env.cols, env.rows) }, "Envelope reset")}>
        <Trash2 size={12} /> Reset</button>
    </div>
  </>;
}

/** The pixel envelope's mesh and points; the node's coordinates are normalised to the canvas. */
export function EnvelopeNodeOverlay({ node, atDisplay, toCanvas, onChange }: {
  node: EnvelopeNode;
  atDisplay: (u: number, v: number) => Point;
  toCanvas: (event: ReactPointerEvent<Element>) => Point | undefined;
  onChange: (node: EnvelopeNode) => void;
}) {
  const env = node.envelope;
  const display = (p: Point) => atDisplay(p[0], p[1]);
  const lines: string[] = [];
  for (let j = 0; j <= env.rows; j++) lines.push(env.points.slice(j * (env.cols + 1), (j + 1) * (env.cols + 1)).map((p) => display(p).map((v) => v * 100).join(",")).join(" "));
  for (let i = 0; i <= env.cols; i++) lines.push(Array.from({ length: env.rows + 1 }, (_, j) => display(env.points[j * (env.cols + 1) + i]).map((v) => v * 100).join(",")).join(" "));
  return <>
    <svg className="photo-frame" viewBox="0 0 100 100" preserveAspectRatio="none">
      {lines.map((points, k) => <polyline key={k} points={points} className="photo-path-mesh" />)}
    </svg>
    {env.points.map((p, k) => <Handle key={k} at={display(p)} className="gradient end" title="Drag to bend"
      onStart={() => undefined}
      onMove={(event) => {
        const to = toCanvas(event);
        if (to) onChange({ ...node, envelope: { ...env, points: env.points.map((q, i) => i === k ? to : q) } });
      }} />)}
  </>;
}
