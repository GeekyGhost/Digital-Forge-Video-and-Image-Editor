/**
 * VFX Phase 6: Repeat -- copies of a layer in a grid, around a ring, or along another layer's path.
 *
 * Expanded into ordinary layers BEFORE compositing (`expandRepeats`), by both compositors, so a copy
 * is just a layer: it has the source layer's nodes, masks, blend and edge, and the two compositors
 * agree on repeats because they already agree on layers. Deterministic: the copies are a pure
 * function of the document.
 *
 * Copy 0 keeps the layer's id (selection, handles, a track matte naming it all still find it); the
 * others get `<id>~<k>`. Limits worth knowing:
 * - a layer clipped to a repeated base clips to its LAST copy (the nearest layer below it);
 * - a guide path is read where the document puts it, before any attach-to-track offset.
 */

import type { PhotoDocument, PhotoLayer } from "./photoDocument";
import { layersOf } from "./photoDocument";
import { flattenSubpath, pointAlong, polylineLength, shapedPath, type Point } from "./photoPath";
import { pathInCanvas } from "./photoPathOps";

export const MAX_REPEAT_COPIES = 500;

export type RepeatPlacement = { x: number; y: number; rotation: number; scale: number; opacity: number };

const radians = (degrees: number) => (degrees * Math.PI) / 180;

/** Where each copy of `layer` goes (copy 0 first), in the layer's own units (normalised x/y). */
export function repeatPlacements(document: PhotoDocument, layer: PhotoLayer, canvas: { width: number; height: number }): RepeatPlacement[] {
  const repeat = layer.repeat;
  if (!repeat) return [{ x: layer.x, y: layer.y, rotation: layer.rotation, scale: layer.scale, opacity: layer.opacity }];
  const W = Math.max(1, canvas.width), H = Math.max(1, canvas.height);
  const scaleStep = repeat.scaleStep ?? 1, opacityStep = repeat.opacityStep ?? 1;
  const copy = (k: number, x: number, y: number, rotation: number): RepeatPlacement => ({
    x, y, rotation,
    scale: layer.scale * scaleStep ** k,
    opacity: Math.min(1, Math.max(0, layer.opacity * opacityStep ** k)),
  });
  const out: RepeatPlacement[] = [];
  if (repeat.kind === "grid") {
    const columns = Math.max(1, Math.round(repeat.columns)), rows = Math.max(1, Math.round(repeat.rows));
    for (let r = 0; r < rows && out.length < MAX_REPEAT_COPIES; r++) {
      for (let c = 0; c < columns && out.length < MAX_REPEAT_COPIES; c++) {
        const dx = c * repeat.gapX + (r % 2 === 1 ? (repeat.shift ?? 0) * repeat.gapX : 0);
        const dy = r * repeat.gapY;
        out.push(copy(out.length, layer.x + dx / W, layer.y + dy / H, layer.rotation));
      }
    }
  } else if (repeat.kind === "radial") {
    const count = Math.max(1, Math.min(MAX_REPEAT_COPIES, Math.round(repeat.count)));
    const start = repeat.startAngle ?? 0;
    const sweep = repeat.sweep ?? 360;
    const full = Math.abs(sweep) >= 360 - 1e-9;
    const step = count === 1 ? 0 : full ? sweep / count : sweep / (count - 1);
    // The layer sits on the ring at `start`; the ring's centre is `radius` behind it.
    const cx = layer.x * W - repeat.radius * Math.cos(radians(start));
    const cy = layer.y * H - repeat.radius * Math.sin(radians(start));
    for (let k = 0; k < count; k++) {
      const angle = start + k * step;
      out.push(copy(k, (cx + repeat.radius * Math.cos(radians(angle))) / W, (cy + repeat.radius * Math.sin(radians(angle))) / H,
        layer.rotation + (repeat.rotate ? angle - start : 0)));
    }
  } else {
    const guide = layersOf(document).find((item) => item.id === repeat.layerId && item.path);
    const count = Math.max(1, Math.min(MAX_REPEAT_COPIES, Math.round(repeat.count)));
    if (!guide?.path) return [copy(0, layer.x, layer.y, layer.rotation)];
    const inCanvas = shapedPath(pathInCanvas(guide, guide.path, W, H));
    const subpath = inCanvas.subpaths.find((item) => item.nodes.length >= 2);
    if (!subpath) return [copy(0, layer.x, layer.y, layer.rotation)];
    const points: Point[] = flattenSubpath(subpath, 0.1);
    const total = polylineLength(points, subpath.closed);
    const from = Math.min(1, Math.max(0, repeat.start ?? 0)), to = Math.min(1, Math.max(0, repeat.end ?? 1));
    const wraps = subpath.closed && from === 0 && to === 1;
    for (let k = 0; k < count; k++) {
      const t = count === 1 ? from : wraps ? k / count : from + ((to - from) * k) / (count - 1);
      const { at, angle } = pointAlong(points, subpath.closed, t * total);
      out.push(copy(k, at[0] / W, at[1] / H, layer.rotation + (repeat.rotate ? (angle * 180) / Math.PI : 0)));
    }
  }
  return out;
}

/** The document with every repeated layer replaced by its copies (unchanged when nothing repeats). */
export function expandRepeats(document: PhotoDocument, canvas: { width: number; height: number }): PhotoDocument {
  const layers = layersOf(document);
  if (!layers.some((layer) => layer.repeat)) return document;
  const expanded: PhotoLayer[] = [];
  for (const layer of layers) {
    if (!layer.repeat) { expanded.push(layer); continue; }
    const { repeat: _repeat, ...plain } = layer;
    repeatPlacements(document, layer, canvas).forEach((place, k) => {
      expanded.push({
        ...plain,
        id: k === 0 ? layer.id : `${layer.id}~${k}`,
        name: k === 0 ? layer.name : `${layer.name} ${k + 1}`,
        x: place.x, y: place.y, rotation: place.rotation, scale: place.scale, opacity: place.opacity,
      });
    });
  }
  return { ...document, layers: expanded };
}
