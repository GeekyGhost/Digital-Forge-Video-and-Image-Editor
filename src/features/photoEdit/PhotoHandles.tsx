import { useRef, type PointerEvent as ReactPointerEvent } from "react";
import type { CropRect } from "../../domain/photoGeometry";

/**
 * Things you grab on the picture.
 *
 * Everything here was a slider first, and sliders are the right control for a number you know and
 * the wrong one for a position you can see. A crop is a rectangle on a photograph; a gradient is a
 * direction across it; a layer is somewhere in the frame. Dragging them is not a convenience, it is
 * how anyone actually decides where they go.
 *
 * Every handle is positioned in DISPLAY fractions and reports back in whatever space the caller
 * asked for, because the mapping between the two is the document's geometry and belongs in exactly
 * one place -- see `domain/photoGeometry.ts` for why this project is careful about that.
 */

export type Point = [number, number];

/** A grabbable dot. Capture is taken on the element, so a fast drag cannot outrun it. */
export function Handle({ at, className, title, onStart, onMove, onEnd }: {
  at: Point; className?: string; title?: string;
  onStart: (event: ReactPointerEvent<HTMLSpanElement>) => void;
  onMove: (event: ReactPointerEvent<HTMLSpanElement>) => void;
  onEnd?: () => void;
}) {
  const held = useRef(false);
  return <span
    className={`photo-handle${className ? ` ${className}` : ""}`}
    title={title}
    style={{ left: `${at[0] * 100}%`, top: `${at[1] * 100}%` }}
    onPointerDown={(event) => {
      event.preventDefault();
      event.stopPropagation();
      event.currentTarget.setPointerCapture(event.pointerId);
      held.current = true;
      onStart(event);
    }}
    onPointerMove={(event) => { if (held.current) onMove(event); }}
    onPointerUp={() => { held.current = false; onEnd?.(); }}
    onPointerCancel={() => { held.current = false; onEnd?.(); }}
  />;
}

const CORNERS = [
  { key: "nw", x: 0, y: 0 }, { key: "ne", x: 1, y: 0 },
  { key: "se", x: 1, y: 1 }, { key: "sw", x: 0, y: 1 },
] as const;
const EDGES = [
  { key: "n", x: 0.5, y: 0 }, { key: "e", x: 1, y: 0.5 },
  { key: "s", x: 0.5, y: 1 }, { key: "w", x: 0, y: 0.5 },
] as const;

/**
 * The crop rectangle, over an uncropped picture.
 *
 * Showing the picture uncropped while the tool is active is the whole trick: a crop drawn over its
 * own result can only ever shrink, because everything it would grow into has already been thrown
 * away on screen. With the full frame showing, the rectangle's fractions and the display's fractions
 * are the same numbers, and the drag maths stays something you can read.
 */
export function CropOverlay({ rect, aspect, frameOf, onChange }: {
  rect: CropRect;
  /** Locked width/height ratio, or undefined for free. */
  aspect?: number;
  frameOf: () => DOMRect | undefined;
  onChange: (rect: CropRect) => void;
}) {
  const from = useRef<{ rect: CropRect; at: Point } | undefined>(undefined);

  const at = (event: ReactPointerEvent<HTMLElement>): Point | undefined => {
    const box = frameOf();
    if (!box || box.width <= 0 || box.height <= 0) return undefined;
    return [(event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height];
  };
  const begin = (event: ReactPointerEvent<HTMLElement>) => {
    const point = at(event);
    if (point) from.current = { rect, at: point };
  };

  const MIN = 0.02;
  const resize = (corner: string) => (event: ReactPointerEvent<HTMLElement>) => {
    const start = from.current;
    const point = at(event);
    if (!start || !point) return;
    let left = start.rect.x;
    let top = start.rect.y;
    let right = start.rect.x + start.rect.width;
    let bottom = start.rect.y + start.rect.height;
    if (corner.includes("w")) left = clamp(point[0], 0, right - MIN);
    if (corner.includes("e")) right = clamp(point[0], left + MIN, 1);
    if (corner.includes("n")) top = clamp(point[1], 0, bottom - MIN);
    if (corner.includes("s")) bottom = clamp(point[1], top + MIN, 1);
    let width = right - left;
    let height = bottom - top;
    if (aspect) {
      // The frame is not square, so a locked ratio is a ratio of PIXELS, not of fractions. Without
      // the frame's own aspect in here, "1:1" would give a square only on a square picture.
      const box = frameOf();
      const frame = box && box.height > 0 ? box.width / box.height : 1;
      const wanted = aspect / frame;
      if (width / height > wanted) width = height * wanted; else height = width / wanted;
      if (corner.includes("w")) left = right - width;
      if (corner.includes("n")) top = bottom - height;
      width = Math.min(width, 1 - left);
      height = Math.min(height, 1 - top);
    }
    onChange({ x: left, y: top, width, height });
  };

  const move = (event: ReactPointerEvent<HTMLElement>) => {
    const start = from.current;
    const point = at(event);
    if (!start || !point) return;
    onChange({
      ...start.rect,
      x: clamp(start.rect.x + point[0] - start.at[0], 0, 1 - start.rect.width),
      y: clamp(start.rect.y + point[1] - start.at[1], 0, 1 - start.rect.height),
    });
  };

  const held = useRef(false);
  return <div className="photo-crop" style={{
    left: `${rect.x * 100}%`, top: `${rect.y * 100}%`,
    width: `${rect.width * 100}%`, height: `${rect.height * 100}%`,
  }}>
    <div className="photo-crop-body" title="Drag to move the crop"
      onPointerDown={(event) => {
        event.preventDefault();
        event.currentTarget.setPointerCapture(event.pointerId);
        held.current = true;
        begin(event);
      }}
      onPointerMove={(event) => { if (held.current) move(event); }}
      onPointerUp={() => { held.current = false; }}
      onPointerCancel={() => { held.current = false; }} />
    {/* Thirds, because a crop is usually a composition decision and this is the grid people use. */}
    <i className="photo-crop-third v" style={{ left: "33.333%" }} />
    <i className="photo-crop-third v" style={{ left: "66.667%" }} />
    <i className="photo-crop-third h" style={{ top: "33.333%" }} />
    <i className="photo-crop-third h" style={{ top: "66.667%" }} />
    {CORNERS.map((corner) => <Handle key={corner.key} at={[corner.x, corner.y]}
      className={`corner ${corner.key}`} title="Drag to resize"
      onStart={begin} onMove={resize(corner.key)} />)}
    {EDGES.map((edge) => <Handle key={edge.key} at={[edge.x, edge.y]}
      className={`edge ${edge.key}`} title="Drag to resize"
      onStart={begin} onMove={resize(edge.key)} />)}
  </div>;
}

const clamp = (value: number, low: number, high: number) =>
  value < low ? low : value > high ? high : value;
