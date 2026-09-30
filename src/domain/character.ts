/**
 * VFX Phase 5: Wan Animate 2 character insertion on the Clip Editor's VFX surface.
 *
 * The worker (ai-runtime/aiv_runtime/wan_animate.py) makes a video of a character image performing a
 * driving clip's motion, sized W x H from the clip's aspect, covering exactly `crop` (x, y, w, h in
 * the clip's pixels) of each driving frame, frame for frame. SAM 2 + BiRefNet cut it out into an
 * RGBA element. This module is the pure part the editor needs around that:
 *
 * - `drivingRange`: the plate's source frames, which the worker reads as the driving clip;
 * - `seedPoints`: SAM 2 include points on the OUTPUT's first frame, taken from the plate's actor
 *   matte (the character performs the actor's motion in the actor's place, so the actor's body is
 *   where the character is);
 * - `alignToPlate`: the placement that lays the element pixel-for-pixel over that crop of the plate
 *   as the plate is placed (through the same `layerMatrix` both compositors use);
 * - `characterLayer`: the element as a frames layer starting with the plate, aligned, occluded by the
 *   plate's depth when the plate has a depth sequence.
 * - `cleanPlateRequest` / `insertAbove`: the clean plate patch for a locked-off shot
 *   (src-tauri/src/ai/clean_plate.rs), which records its placement like a character and goes
 *   between the plate and the character, so the actor does not show where the character does not
 *   cover him.
 */

import { applyMatrix } from "./photoGeometry";
import { croppedRect, layerMatrix } from "./photoCompose";
import { createDepthOcclusionMask, createMediaLayer, layersOf, type MaskSource, type PhotoDocument, type PhotoLayer } from "./photoDocument";
import { mediaFromElement, type ElementLike } from "./elements";
import { sequenceFramePath } from "./photoMask";

/** An element library entry as this module reads it (a subset of `ElementEntry`). */
export type CharacterElement = ElementLike & { manifest: ElementLike["manifest"] & { name: string; [key: string]: unknown } };

/** x, y, w, h in the driving clip's pixels. */
export type PlateCrop = [number, number, number, number];

/** What `extra.animate` in an element manifest records (src-tauri/src/ai/wan_animate.rs `register`). */
export type AnimateRecord = {
  crop: PlateCrop;
  outputWidth: number;
  outputHeight: number;
  plateWidth: number;
  plateHeight: number;
  drivingVideo: string;
  firstFrame: number;
  frameCount: number;
};

export function animateRecord(element: CharacterElement): AnimateRecord | undefined {
  const record = element.manifest.animate as Partial<AnimateRecord> | undefined;
  if (!record || !Array.isArray(record.crop) || record.crop.length !== 4) return undefined;
  if (!record.outputWidth || !record.outputHeight) return undefined;
  return record as AnimateRecord;
}

/** The plate's source frames [first, first + count): what drives the character, frame for frame. */
export function drivingRange(plate: PhotoLayer): { first: number; count: number } | undefined {
  const media = plate.media;
  if (!media || media.kind !== "video") return undefined;
  const first = Math.max(0, Math.floor(media.inPoint));
  const count = Math.max(0, Math.min(media.frameCount, Math.floor(media.outPoint)) - first);
  return count > 0 ? { first, count } : undefined;
}

/** The sequence mattes on a layer's masks (SAM 2 tracks), by mask and entry, for the actor picker. */
export function sequenceMattes(layer: PhotoLayer): Array<{ id: string; label: string; source: Extract<MaskSource, { kind: "sequence" }> }> {
  const found: Array<{ id: string; label: string; source: Extract<MaskSource, { kind: "sequence" }> }> = [];
  for (const mask of layer.masks) {
    for (const entry of mask.stack ?? []) {
      if (entry.source.kind === "sequence") {
        found.push({ id: `${mask.id}/${entry.id}`, label: `${mask.name}${entry.name ? ` · ${entry.name}` : ""}`, source: entry.source });
      }
    }
  }
  return found;
}

/** The actor matte's file at a plate source frame (clamped to the frames the track has). */
export const actorMattePath = (source: Extract<MaskSource, { kind: "sequence" }>, sourceFrame: number) =>
  sequenceFramePath(source, sourceFrame);

/**
 * Up to three SAM 2 include points on the output's first frame, from the actor matte (`alpha`, one
 * byte per plate pixel, > 127 is the actor): the middle of the longest run of actor pixels on the rows
 * at 25%, 45% and 65% of the actor's height (head/torso/hips; a centroid can fall between the legs),
 * mapped from plate pixels into the output, normalised. Points outside the crop are dropped.
 */
export function seedPoints(alpha: Uint8Array | Uint8ClampedArray, width: number, height: number, crop: PlateCrop): Array<{ x: number; y: number; include: boolean }> {
  let top = -1, bottom = -1;
  for (let y = 0; y < height && top < 0; y++) for (let x = 0; x < width; x++) if (alpha[y * width + x] > 127) { top = y; break; }
  for (let y = height - 1; y >= 0 && bottom < 0; y--) for (let x = 0; x < width; x++) if (alpha[y * width + x] > 127) { bottom = y; break; }
  if (top < 0) return [];
  const [cx, cy, cw, ch] = crop;
  const points: Array<{ x: number; y: number; include: boolean }> = [];
  for (const fraction of [0.25, 0.45, 0.65]) {
    const y = Math.round(top + (bottom - top) * fraction);
    let best: [number, number] | undefined;
    let runStart = -1;
    for (let x = 0; x <= width; x++) {
      const inside = x < width && alpha[y * width + x] > 127;
      if (inside && runStart < 0) runStart = x;
      if (!inside && runStart >= 0) {
        if (!best || x - runStart > best[1] - best[0]) best = [runStart, x];
        runStart = -1;
      }
    }
    if (!best) continue;
    const u = (best[0] + best[1]) / 2, v = y + 0.5;
    const nx = (u - cx) / cw, ny = (v - cy) / ch;
    if (nx >= 0 && nx <= 1 && ny >= 0 && ny <= 1) points.push({ x: round4(nx), y: round4(ny), include: true });
  }
  return points;
}

const round4 = (value: number) => Math.round(value * 10000) / 10000;

/**
 * The placement (x, y, scale, rotation) that lays a W x H element over `crop` of the plate as the
 * plate is placed on a canvas of `canvas` size: its centre goes where the crop's centre goes through
 * the plate's own matrix, its scale is the plate's times crop / output, its rotation the plate's.
 * The crop has the output's aspect to within a pixel, so one scale serves both axes.
 */
export function alignToPlate(
  plate: PhotoLayer, canvas: { width: number; height: number }, crop: PlateCrop, outputWidth: number, outputHeight: number,
): Pick<PhotoLayer, "x" | "y" | "scale" | "rotation"> {
  const media = plate.media!;
  const size = { width: media.width, height: media.height };
  const rect = croppedRect(plate, size);
  const [cx, cy, cw, ch] = crop;
  const [x, y] = applyMatrix(layerMatrix(plate, canvas.width, canvas.height, size, 1), cx + cw / 2 - rect.left, cy + ch / 2 - rect.top);
  const ratio = (cw / outputWidth + ch / outputHeight) / 2;
  return { x: x / canvas.width, y: y / canvas.height, scale: plate.scale * ratio, rotation: plate.rotation };
}

/**
 * The character element as a layer over the plate: starting where the plate's first driving frame
 * shows, and aligned to the plate. A frames layer has no picture file (`sourcePath` stays empty, as
 * `createMediaLayer` makes it; the loader treats a non-empty one as an image to open -- the first
 * 4090 run put the element folder there and got "could not be decoded as an image").
 *
 * With `occlude`, the layer is also masked by depth occlusion against the plate. Off by default for a
 * character the PLATE drove: the actor it replaces is the nearest thing in the plate, so occlusion
 * would hide the character behind the very person it stands in for. Use it for foreground objects
 * (a lamppost the actor walks behind) and set the plane in Masks.
 */
export function characterLayer(
  document: PhotoDocument, plate: PhotoLayer, element: CharacterElement, canvas: { width: number; height: number },
  options: { occlude?: boolean } = {},
): PhotoLayer | undefined {
  const record = animateRecord(element);
  const media = plate.media;
  if (!record || !media) return undefined;
  // Element frame i is driving (plate source) frame firstFrame + i, which shows at this composition frame.
  const rate = document.frameRate ?? media.fps;
  const start = media.start + (record.firstFrame - media.inPoint) * rate / media.fps;
  const layer = createMediaLayer(element.manifest.name, mediaFromElement(element, start),
    alignToPlate(plate, canvas, record.crop, record.outputWidth, record.outputHeight));
  if (options.occlude && media.depth) {
    const occlusion = createDepthOcclusionMask(plate.id);
    return { ...layer, masks: [occlusion], maskId: occlusion.id };
  }
  return layer;
}

/**
 * The actor's union box over `rasters` (x0, y0, x1, y1), normalised: the neighbourhood the worker
 * drives with, so the character gets most of the output's pixels. Undefined when all are empty.
 */
export function actorRegion(rasters: Array<Uint8Array | Uint8ClampedArray>, width: number, height: number): [number, number, number, number] | undefined {
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  for (const alpha of rasters) {
    for (let y = 0; y < height; y++) {
      const row = y * width;
      for (let x = 0; x < width; x++) {
        if (alpha[row + x] > 127) {
          if (x < x0) x0 = x;
          if (x > x1) x1 = x;
          if (y < y0) y0 = y;
          if (y > y1) y1 = y;
        }
      }
    }
  }
  return x1 < 0 ? undefined : [round4(x0 / width), round4(y0 / height), round4((x1 + 1) / width), round4((y1 + 1) / height)];
}

/**
 * The actor as the worker frames the character by: (cx, feet, standing), normalised to the plate --
 * the matte's centroid x on the first driving frame (on the torso; a box centre drifts toward an
 * outstretched leg), the bottom of its box there, and the tallest box height over `samples` (the
 * first 4090 run's dancer measured 599 px at frame 0 in a stride, 517 crouched later). Undefined when
 * the first frame's matte is empty. `samples` are alpha rasters of the plate's size.
 */
export function actorFrame(
  first: Uint8Array | Uint8ClampedArray, samples: Array<Uint8Array | Uint8ClampedArray>, width: number, height: number,
): [number, number, number] | undefined {
  const measure = (alpha: Uint8Array | Uint8ClampedArray) => {
    let top = -1, bottom = -1, sumX = 0, count = 0;
    for (let y = 0; y < height; y++) {
      const row = y * width;
      let any = false;
      for (let x = 0; x < width; x++) if (alpha[row + x] > 127) { sumX += x + 0.5; count++; any = true; }
      if (any) { if (top < 0) top = y; bottom = y; }
    }
    return count ? { cx: sumX / count, top, bottom: bottom + 1 } : undefined;
  };
  const start = measure(first);
  if (!start) return undefined;
  let standing = start.bottom - start.top;
  for (const sample of samples) {
    const box = measure(sample);
    if (box) standing = Math.max(standing, box.bottom - box.top);
  }
  return [round4(start.cx / width), round4(start.bottom / height), round4(standing / height)];
}

/** True when the element was made from this plate's own footage (so aligning it makes sense). */
export function drivenByPlate(element: CharacterElement, plate: PhotoLayer): boolean {
  const record = animateRecord(element);
  return Boolean(record && plate.media && record.drivingVideo === plate.media.path);
}

/** The plate layer of a clip document, if any (the only locked video layer the clip surface makes). */
export const plateOf = (document: PhotoDocument, plateId: string) => layersOf(document).find((layer) => layer.id === plateId);

/** A clean plate patch element (src-tauri/src/ai/clean_plate.rs), rather than a character. */
export const isCleanPlate = (element: CharacterElement) => element.manifest.source === "clean-plate";

/** What `clean_plate_build` needs: the plate's source range and clip length, and the actor's track. */
export function cleanPlateRequest(
  plate: PhotoLayer, matte: { cachePath: string; first?: number }, name: string,
): { name: string; video: string; firstFrame: number; frameCount: number; totalFrames: number; fps: number;
  plateWidth: number; plateHeight: number; matteCache: string; matteFirst: number } | undefined {
  const range = drivingRange(plate);
  const media = plate.media;
  if (!range || !media) return undefined;
  return {
    name, video: media.path, firstFrame: range.first, frameCount: range.count, totalFrames: media.frameCount,
    fps: media.fps, plateWidth: media.width, plateHeight: media.height,
    matteCache: matte.cachePath, matteFirst: Math.max(0, Math.floor(matte.first ?? 0)),
  };
}

/** `layers` with `layer` just above the one with `anchorId` (on top when it is not there). */
export function insertAbove(layers: PhotoLayer[], anchorId: string, layer: PhotoLayer): PhotoLayer[] {
  const index = layers.findIndex((item) => item.id === anchorId);
  return index < 0 ? [...layers, layer] : [...layers.slice(0, index + 1), layer, ...layers.slice(index + 1)];
}
