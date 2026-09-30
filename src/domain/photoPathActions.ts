/**
 * VFX Phase 6: the layer-level path actions behind the Path card -- pure functions from a layer list
 * to a layer list, so the editor's buttons are one line each and every one is unit-tested.
 *
 * Paths on different layers live in different coordinates (each around its own layer's centre,
 * under its own scale and rotation), so every action that combines two of them first maps both
 * into canvas pixels (`pathInCanvas`, exact: placement is affine) and maps the result into the
 * coordinates of the layer that keeps it.
 */

import { createMask, createMaskEntry, createPathLayer, type PhotoLayer } from "./photoDocument";
import { createEnvelope } from "./photoEnvelope";
import { mapPath, pathBounds, type PathBlock } from "./photoPath";
import {
  blendSteps, booleanPaths, canvasToLayerPath, centreOnBox, compoundPaths, outlineStroke, pathInCanvas, type BooleanOp,
} from "./photoPathOps";

type Canvas = { width: number; height: number };

const indexOf = (layers: PhotoLayer[], id: string) => layers.findIndex((layer) => layer.id === id);

/** The nearest path layer below `id`, or undefined. */
export function pathLayerBelow(layers: PhotoLayer[], id: string): PhotoLayer | undefined {
  for (let i = indexOf(layers, id) - 1; i >= 0; i--) if (layers[i].path) return layers[i];
  return undefined;
}

/**
 * Combine the path layer `topId` with the nearest path layer below it (Illustrator's Shape Modes):
 * the result replaces the lower layer, in its placement, and the top layer goes. Subtract (minus
 * front) keeps the lower layer's paint; unite, intersect and exclude take the top's, as Illustrator
 * does. `compound` keeps both geometries as one path (holes by winding) in the lower layer's paint.
 */
export function combinePathLayers(layers: PhotoLayer[], topId: string, op: BooleanOp | "compound", canvas: Canvas): PhotoLayer[] | undefined {
  const top = layers.find((layer) => layer.id === topId);
  const bottom = top?.path ? pathLayerBelow(layers, topId) : undefined;
  if (!top?.path || !bottom?.path) return undefined;
  const a = pathInCanvas(bottom, bottom.path, canvas.width, canvas.height);
  const b = pathInCanvas(top, top.path, canvas.width, canvas.height);
  const keepsBottom = op === "subtract" || op === "compound";
  const styled = keepsBottom ? bottom.path : top.path;
  // The style's stroke and dashes in canvas pixels, then in the kept layer's own units.
  const styledInCanvas = keepsBottom ? a : b;
  const unit = Math.max(1e-9, Math.abs(bottom.scale));
  const combined = op === "compound" ? compoundPaths(a, b) : booleanPaths(a, b, op);
  const local = canvasToLayerPath(combined, bottom, bottom.path, canvas.width, canvas.height);
  const path: PathBlock = {
    ...styled,
    subpaths: local.subpaths,
    fillRule: op === "compound" ? bottom.path.fillRule : "nonzero",
    strokeWidth: styledInCanvas.strokeWidth / unit,
    dash: styledInCanvas.dash?.map((value) => value / unit),
    dashOffset: styledInCanvas.dashOffset === undefined ? undefined : styledInCanvas.dashOffset / unit,
    envelope: undefined,
  };
  if (path.dash === undefined) delete path.dash;
  if (path.dashOffset === undefined) delete path.dashOffset;
  return layers.filter((layer) => layer.id !== topId).map((layer) => layer.id === bottom.id ? { ...layer, path } : layer);
}

/** The stroke of `id` as its own filled layer just above it; the original keeps its fill only. */
export function outlineLayerStroke(layers: PhotoLayer[], id: string): PhotoLayer[] | undefined {
  const layer = layers.find((item) => item.id === id);
  const outlined = layer?.path ? outlineStroke(layer.path) : undefined;
  if (!layer?.path || !outlined) return undefined;
  const strokeLayer: PhotoLayer = { ...layer, id: `${layer.id}-stroke-${Date.now().toString(36)}`, name: `${layer.name} stroke`, path: outlined, masks: [], maskId: undefined, repeat: undefined };
  const kept: PhotoLayer = { ...layer, path: { ...layer.path, stroke: undefined, strokeWidth: 0 } };
  const at = indexOf(layers, id);
  return [...layers.slice(0, at), kept, strokeLayer, ...layers.slice(at + 1)];
}

/** Clip the layer directly above `id` to it (Phase 2 clipping: the path layer becomes the base). */
export function clipAboveToPath(layers: PhotoLayer[], id: string): PhotoLayer[] | undefined {
  const at = indexOf(layers, id);
  if (at < 0 || at + 1 >= layers.length) return undefined;
  return layers.map((layer, index) => index === at + 1 ? { ...layer, clip: true } : layer);
}

/**
 * A vector mask: the path's fill as a `path` mask entry (normalised to the canvas) on the layer
 * directly below, made that layer's mask; the path layer itself is hidden (its geometry stays in
 * the mask, the layer stays for editing).
 */
export function maskBelowWithPath(layers: PhotoLayer[], id: string, canvas: Canvas): PhotoLayer[] | undefined {
  const at = indexOf(layers, id);
  const layer = layers[at];
  if (!layer?.path || at < 1) return undefined;
  const inCanvas = pathInCanvas(layer, layer.path, canvas.width, canvas.height);
  const normalised = mapPath({ ...inCanvas, envelope: undefined }, ([x, y]) => [x / canvas.width, y / canvas.height]);
  const mask = { ...createMask(`${layer.name} (vector)`), stack: [createMaskEntry({ kind: "path", path: { ...normalised, stroke: undefined, strokeWidth: 0, dash: undefined } })] };
  return layers.map((item, index) => {
    if (index === at - 1) return { ...item, masks: [...item.masks, mask], maskId: mask.id };
    if (index === at) return { ...item, visible: false };
    return item;
  });
}

/**
 * Make Blend: `steps` in-between path layers from `aId` to `bId`, in canvas pixels (so two layers
 * placed and turned differently still blend where they are), each its own layer centred on its box,
 * inserted just above the lower of the two.
 */
export function blendPathLayers(layers: PhotoLayer[], aId: string, bId: string, steps: number, canvas: Canvas): PhotoLayer[] | undefined {
  const a = layers.find((layer) => layer.id === aId), b = layers.find((layer) => layer.id === bId);
  if (!a?.path || !b?.path || a.id === b.id) return undefined;
  const [lower, upper] = indexOf(layers, a.id) < indexOf(layers, b.id) ? [a, b] : [b, a];
  const from = pathInCanvas(lower, lower.path!, canvas.width, canvas.height);
  const to = pathInCanvas(upper, upper.path!, canvas.width, canvas.height);
  const made = blendSteps({ ...from, envelope: undefined }, { ...to, envelope: undefined }, steps).map((path, index) => {
    const centred = centreOnBox(path, canvas.width, canvas.height);
    return {
      ...createPathLayer(centred.path, centred.x, centred.y, `${lower.name} → ${upper.name} ${index + 1}`),
      opacity: lower.opacity + ((upper.opacity - lower.opacity) * (index + 1)) / (steps + 1),
      blendMode: lower.blendMode,
    };
  });
  const at = indexOf(layers, lower.id);
  return [...layers.slice(0, at + 1), ...made, ...layers.slice(at + 1)];
}

/** An identity 4 x 4 envelope over the path's own box (padding a pixel so edges bend too). */
export function envelopeFor(path: PathBlock): PathBlock {
  const b = pathBounds({ ...path, envelope: undefined });
  const pad = 1;
  return { ...path, envelope: createEnvelope({ x: b.minX - pad, y: b.minY - pad, width: b.maxX - b.minX + 2 * pad, height: b.maxY - b.minY + 2 * pad }, 3, 3) };
}
