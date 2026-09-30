/**
 * Masks, depth and fills for one scope, at one working resolution -- the `resourcesFor` both
 * compositors take.
 *
 * One function, used by Photo Edit and by the parity harness (src/render/parity.ts). Until VFX
 * Phase 1 the harness carried a hand-kept copy of the component's version, which is exactly the
 * two-implementations shape this project keeps paying for.
 *
 * `pixels` must be the scope's UNGRADED input at the working resolution, because a colour-range mask
 * matches against it. Handing it the preview buffer while rendering at full size would silently
 * produce an empty mask: the selection would work on screen and vanish from the saved file.
 *
 * A derived file belonging to a layer is PLACED through that layer's matrix rather than stretched to
 * the canvas: a matte and a fill were both produced in the layer's own pixels, so anything else lands
 * them correctly on an untransformed plate and slides them off a scaled cutout.
 */

import { croppedRect, layerMatrix, pathCoverage, placeLayer, type ComposeScope } from "./photoCompose";
import { compose, scaleBy, translate, type Matrix } from "./photoGeometry";
import type { PathBlock } from "./photoPath";
import type { PhotoDocument, PhotoLayer, PhotoMask } from "./photoDocument";
import {
  coverageFromFile, coverageFromImage, layerCoverage, maskFilePaths, maskReadsPixels, matteRequest, rasterizeMaskStack,
  type MatteRequest,
} from "./photoMask";
import { emptyResources, type RenderResources } from "./photoRender";

/** A loaded picture: the canvas it was drawn into and its size. */
export type LoadedPicture = { canvas: HTMLCanvasElement; width: number; height: number };

export type ResourceInputs = {
  document: PhotoDocument;
  /** Loaded pictures by path; a missing one simply contributes nothing yet. */
  files: (path: string) => LoadedPicture | undefined;
  /**
   * Rasterised coverage kept between frames. Masks are small enough that stringifying one is
   * cheaper than getting a staleness check subtly wrong. Optional: the harness runs without one.
   */
  cache?: Map<string, Uint8Array>;
};

export function buildScopeResources(
  inputs: ResourceInputs, scope: ComposeScope, width: number, height: number, pixels?: Uint8ClampedArray,
): RenderResources {
  const { document, files, cache } = inputs;
  const resources = emptyResources();
  const place = (held: LoadedPicture): Uint8ClampedArray =>
    scope.layer
      ? placeLayer(scope.layer, { image: held.canvas, width: held.width, height: held.height },
        width, height, scope.canvasScale ?? 1)
      : stretched(held, width, height);
  // Where the layer sits only matters to a mask that carries a file, since that file is placed
  // through the layer. Keying every mask on placement would re-rasterise painted strokes on every
  // frame of a drag for no reason.
  const where = scope.layer ? JSON.stringify(placement(scope.layer)) : "canvas";

  /** A file's coverage, placed and decoded, cached on the file, the rule, the placement and the size. */
  const matte = (path: string, key: string, decode: (data: Uint8ClampedArray) => Uint8Array): Uint8Array | undefined => {
    const held = files(path);
    if (!held) return undefined;
    const cacheKey = `matte|${key}|${where}|${width}x${height}`;
    let coverage = cache?.get(cacheKey);
    if (!coverage) {
      coverage = decode(place(held));
      if (cache) {
        if (cache.size > 32) cache.clear();
        cache.set(cacheKey, coverage);
      }
    }
    return coverage;
  };
  /**
   * VFX Phase 4: a `depth` entry's coverage source -- the depth frame of the layer it names, placed
   * through THAT layer (not the scope's), read as alpha-weighted luma. A depth file of another size
   * than the media is stretched to it first, so it lands where the picture does.
   */
  const depthOf = (layerId: string): Uint8Array | undefined => {
    const ref = scope.depthFrame?.(layerId);
    const held = ref ? files(ref.path) : undefined;
    if (!ref || !held) return undefined;
    const cacheKey = `depth|${ref.path}|${JSON.stringify(placement(ref.layer))}|${width}x${height}|${scope.canvasScale ?? 1}`;
    let coverage = cache?.get(cacheKey);
    if (!coverage) {
      const sized = held.width === ref.width && held.height === ref.height ? held.canvas : stretchedCanvas(held, ref.width, ref.height);
      coverage = layerCoverage(placeLayer(ref.layer, { image: sized, width: ref.width, height: ref.height }, width, height, scope.canvasScale ?? 1), "luma", 1);
      if (cache) {
        if (cache.size > 32) cache.clear();
        cache.set(cacheKey, coverage);
      }
    }
    return coverage;
  };
  /** VFX Phase 6: a path entry's fill at this size, cached on the geometry. */
  const pathMatte = (path: PathBlock): Uint8Array => {
    const cacheKey = `${pathMatteKey(path)}|${width}x${height}`;
    let coverage = cache?.get(cacheKey);
    if (!coverage) {
      coverage = pathCoverage(path, width, height);
      if (cache) {
        if (cache.size > 32) cache.clear();
        cache.set(cacheKey, coverage);
      }
    }
    return coverage;
  };
  const legacy = (path: string) => matte(path, legacyMatteKey(path), (data) => coverageFromImage(data, width * height));
  const fileMatte = (request: MatteRequest) =>
    matte(request.path, matteKey(request), (data) => coverageFromFile(data, width * height, request.channel));

  for (const mask of wantedMasks(scope)) {
    if (scope.filesOnly) {
      // The GPU rasterises the stack itself and only needs the files, placed and decoded.
      resources.mattes ??= new Map();
      if (!mask.enabled) continue;
      if (mask.sourcePath) {
        const coverage = legacy(mask.sourcePath);
        if (coverage) resources.mattes.set(legacyMatteKey(mask.sourcePath), coverage);
      }
      for (const entry of mask.stack ?? []) {
        if (entry.enabled && entry.source.kind === "path") {
          resources.mattes.set(pathMatteKey(entry.source.path), pathMatte(entry.source.path));
          continue;
        }
        if (entry.enabled && entry.source.kind === "depth") {
          const coverage = depthOf(entry.source.layerId);
          if (coverage) resources.mattes.set(depthKey(entry.source.layerId), coverage);
          continue;
        }
        const request = entry.enabled ? matteRequest(entry.source, scope.frame) : undefined;
        const coverage = request ? fileMatte(request) : undefined;
        if (request && coverage) resources.mattes.set(matteKey(request), coverage);
      }
      continue;
    }
    // A colour-range mask depends on the pixels it matches against, and a layer entry on another
    // layer's pixels, which change whenever anything under or in them does -- so neither is served
    // from the cache. Before VFX Phase 1 colour masks were keyed like any other mask and a
    // magic-wand selection could go stale after a layer moved.
    const cacheable = cache && !maskReadsPixels(mask);
    // The files a mask reads are part of its key: a sequence entry reads a different file per frame
    // (VFX Phase 4 fix -- before, scrubbing back to a frame whose matte was already loaded could reuse
    // another frame's raster), and a depth entry reads another layer's depth frame, placed.
    const paths = maskFilePaths(mask, scope.frame);
    const depthRefs = (mask.stack ?? []).flatMap((entry) => {
      if (entry.source.kind !== "depth") return [];
      const ref = scope.depthFrame?.(entry.source.layerId);
      return [ref ? `${ref.path}@${JSON.stringify(placement(ref.layer))}@${files(ref.path) ? 1 : 0}` : "-"];
    });
    const key = `${JSON.stringify(mask)}|${paths.join("|")}|${depthRefs.join("|")}|${paths.length ? where : "canvas"}|${width}x${height}|${scope.canvasScale ?? 1}`;
    let coverage = cacheable ? cache.get(key) : undefined;
    if (!coverage) {
      coverage = rasterizeMaskStack(mask, {
        width, height, pixels, canvasScale: scope.canvasScale, frame: scope.frame,
        legacyMatte: mask.enabled && mask.sourcePath ? legacy(mask.sourcePath) : undefined,
        matte: fileMatte,
        layer: (layerId, channel) => {
          const held = scope.layerSource?.(layerId);
          return held ? layerCoverage(held.pixels, channel, held.opacity) : undefined;
        },
        depth: depthOf,
        path: pathMatte,
      });
      if (cacheable) {
        if (cache.size > 32) cache.clear();
        cache.set(key, coverage);
      }
    }
    resources.masks.set(mask.id, coverage);
  }

  // Only placed when something in this scope would actually read it: placing a full-size depth map
  // per layer per frame is real work, and most layers have no relight node at all.
  const wantsDepth = scope.nodes.some((node) => node.kind === "relight" && node.enabled);
  const depth = wantsDepth && document.depthPath ? files(document.depthPath) : undefined;
  if (depth) resources.depth = coverageFromImage(place(depth), width * height);

  for (const node of scope.nodes) {
    if (node.kind !== "remove" || !node.cachePath) continue;
    const fill = files(node.cachePath);
    if (fill) resources.removals.set(node.id, place(fill));
  }

  // VFX Phase 7: a mask VFX node's subject matte, placed like the layer's own picture (the matte is
  // the source's size), and where the layer's source pixels land (the effect measures positions in
  // the source frame). Built in both modes: the GPU runs the node on its CPU path.
  for (const node of scope.nodes) {
    if (node.kind !== "maskVfx" || !node.enabled) continue;
    const held = files(node.mattePath);
    if (!held) continue;
    const size = { width: Math.max(1, node.sourceWidth), height: Math.max(1, node.sourceHeight) };
    const cacheKey = `maskfx|${node.mattePath}|${where}|${width}x${height}|${size.width}x${size.height}|${scope.canvasScale ?? 1}`;
    let coverage = cache?.get(cacheKey);
    if (!coverage) {
      const sized = held.width === size.width && held.height === size.height ? held.canvas : stretchedCanvas(held, size.width, size.height);
      const placed = scope.layer
        ? placeLayer(scope.layer, { image: sized, width: size.width, height: size.height }, width, height, scope.canvasScale ?? 1)
        : stretched({ canvas: sized, width: size.width, height: size.height }, width, height);
      coverage = coverageFromFile(placed, width * height, "auto");
      if (cache) {
        if (cache.size > 32) cache.clear();
        cache.set(cacheKey, coverage);
      }
    }
    resources.maskFx ??= new Map();
    resources.maskFx.set(node.id, { matte: coverage, toWork: sourceToWork(scope.layer, size, width, height, scope.canvasScale ?? 1) });
  }
  return resources;
}

/**
 * Source pixels of a layer -> working pixels: `layerMatrix` measures from the cropped rectangle's
 * corner, so the crop is undone first. Without a layer, the source is stretched over the frame.
 */
export function sourceToWork(
  layer: PhotoLayer | undefined, source: { width: number; height: number }, width: number, height: number, canvasScale = 1,
): Matrix {
  if (!layer) return scaleBy(width / source.width, height / source.height);
  const rect = croppedRect(layer, source);
  return compose(layerMatrix(layer, width, height, source, canvasScale), translate(-rect.left, -rect.top));
}

/**
 * The masks a scope actually reads: those limiting an enabled node, the layer mask, and the one the
 * overlay asked for. Rasterising only these matters since VFX Phase 2, because a stack with a layer
 * entry renders that other layer, and an unused mask should neither cost that nor report a cycle.
 * Scope order, so every compositor rasterises (and so visits other layers) in the same order.
 */
export function wantedMasks(scope: ComposeScope): PhotoMask[] {
  const ids = new Set<string>();
  for (const node of scope.nodes) if (node.enabled && node.maskId) ids.add(node.maskId);
  if (scope.layer?.maskId) ids.add(scope.layer.maskId);
  if (scope.captureMask) ids.add(scope.captureMask);
  return scope.masks.filter((mask) => ids.has(mask.id));
}

export const legacyMatteKey = (path: string) => `legacy|${path}`;
/** VFX Phase 4: a depth entry's placed depth coverage in `RenderResources.mattes`. */
export const depthKey = (layerId: string) => `depth|${layerId}`;
export const matteKey = (request: MatteRequest) => `${request.channel}|${request.path}`;
/** VFX Phase 6: a path entry's coverage in `RenderResources.mattes`, keyed on its geometry. */
export const pathMatteKey = (path: PathBlock) => `path|${path.fillRule}|${JSON.stringify(path.subpaths)}|${JSON.stringify(path.envelope ?? null)}`;

export const isColourMask = (mask: PhotoMask): boolean => mask.enabled && mask.shape?.kind === "colour";

/** Just the fields that move a layer, for keying a cached raster on where it landed. */
export const placement = (layer: PhotoLayer) => [
  layer.x, layer.y, layer.scale, layer.rotation,
  layer.cropLeft, layer.cropTop, layer.cropRight, layer.cropBottom,
  // VFX Phase 7: a stretched layer places its mattes stretched too.
  layer.stretchX ?? 1,
];

/** A picture resized to exactly width x height (a depth file written at another size than its clip). */
function stretchedCanvas(held: LoadedPicture, width: number, height: number): HTMLCanvasElement {
  const canvas = window.document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d")!.drawImage(held.canvas, 0, 0, width, height);
  return canvas;
}

/** Stretch a whole picture across the working size. Used for canvas-space files. */
export function stretched(held: LoadedPicture, width: number, height: number): Uint8ClampedArray {
  const canvas = window.document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.drawImage(held.canvas, 0, 0, width, height);
  return context.getImageData(0, 0, width, height).data;
}
