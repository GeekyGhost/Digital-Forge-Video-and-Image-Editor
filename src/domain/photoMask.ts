/**
 * Turning a mask's strokes into pixels.
 *
 * A mask is stored as normalised strokes plus, optionally, a bitmap from SAM 2. This is where the
 * two become one 8-bit coverage buffer at whatever resolution the caller is working at -- the proxy
 * for the preview, full size for the save -- which is the whole reason strokes are stored instead
 * of a painted bitmap.
 */

import type { MaskEntry, MaskMode, MaskShape, MaskSource, PhotoMask, MaskStroke } from "./photoDocument";

/**
 * VFX Phase 4: depth (disparity byte, 255 = near) to coverage, as a 256-entry table: where the depth
 * is nearer than `plane` (0..100), smoothstepped over `softness` (0..100). A table, so both
 * compositors apply identical bytes and the threshold is exact rather than float-rounded twice.
 */
export function depthThresholdLut(plane: number, softness: number): Uint8Array {
  const lut = new Uint8Array(256);
  const soft = Math.max(0, softness);
  const low = plane - soft / 2, high = plane + soft / 2;
  for (let v = 0; v < 256; v++) {
    const d = v / 255 * 100;
    let t: number;
    if (high - low < 1e-6) t = d >= plane ? 1 : 0;
    else {
      t = Math.min(1, Math.max(0, (d - low) / (high - low)));
      t = t * t * (3 - 2 * t);
    }
    lut[v] = Math.round(t * 255);
  }
  return lut;
}

export function applyLut(values: Uint8Array, lut: Uint8Array): Uint8Array {
  const out = new Uint8Array(values.length);
  for (let index = 0; index < values.length; index++) out[index] = lut[values[index]];
  return out;
}

/**
 * Rasterise one mask.
 *
 * `sam2` is the model's contribution, already scaled to `width` x `height`, as a single byte per
 * pixel. Strokes are stamped over the top so a click-selection can be corrected by hand rather than
 * thrown away when it gets an edge wrong -- an additive stroke fills, an erase stroke cuts.
 *
 * Returns coverage from 0 (untouched) to 255 (fully affected). A disabled mask returns full
 * coverage rather than none: a node whose mask is switched off should apply everywhere, which is
 * what someone toggling it expects to see.
 */
export function rasterizeMask(
  mask: PhotoMask, width: number, height: number, sam2?: Uint8Array, pixels?: Uint8ClampedArray,
  canvasScale = 1,
): Uint8Array {
  const coverage = new Uint8Array(width * height);
  if (!mask.enabled) {
    coverage.fill(255);
    return coverage;
  }
  if (sam2 && sam2.length === coverage.length) coverage.set(sam2);
  // A generated shape and a model matte can both be present; take whichever claims a pixel, so a
  // gradient can be added to a subject selection rather than replacing it.
  if (mask.shape) {
    const generated = shapeCoverage(mask.shape, width, height, pixels);
    for (let index = 0; index < coverage.length; index++) {
      if (generated[index] > coverage[index]) coverage[index] = generated[index];
    }
  }

  const longEdge = Math.max(width, height);
  for (const stroke of mask.strokes) stampStroke(coverage, width, height, stroke, longEdge);
  // Feather is in canvas pixels, scaled to the render (VFX Phase 2 fix). Before, a 10 px feather was
  // 10 px of the 1500 px preview and 10 px of a 6000 px save -- four times sharper in the file than on
  // screen. At full size canvasScale is 1, so saved files are byte-for-byte what they were.
  if (mask.feather > 0) featherCoverage(coverage, width, height, Math.round(mask.feather * canvasScale));
  if (mask.invert) {
    for (let index = 0; index < coverage.length; index++) coverage[index] = 255 - coverage[index];
  }
  return coverage;
}

/**
 * Stamp one stroke.
 *
 * A stroke is a list of sampled pointer positions, which arrive far apart when the pointer moves
 * quickly, so the gaps between them are walked at half-radius steps. Without that a fast drag
 * paints a dotted line rather than a stroke.
 */
function stampStroke(
  coverage: Uint8Array, width: number, height: number, stroke: MaskStroke, longEdge: number,
): void {
  const dab = dabShape(stroke, longEdge);
  forEachDab(stroke, width, height, longEdge, (x, y) => stampDab(coverage, width, height, x, y, dab, stroke.erase));
}

/** A stroke's dab radius and falloff, in working pixels. The GPU stamps with the same numbers. */
export function dabShape(stroke: MaskStroke, longEdge: number): { radius: number; solid: number; fade: number } {
  const radius = Math.max(1, stroke.radius * longEdge);
  // Where the falloff starts. At hardness 1 it starts at the rim, which is a plain disc.
  const solid = radius * Math.min(0.999, stroke.hardness);
  return { radius, solid, fade: Math.max(1e-6, radius - solid) };
}

/**
 * Every dab centre of a stroke, in working pixels, in the order they are stamped. One walk shared by
 * the CPU stamp and the GPU's instanced dabs, so both put the same dabs in the same places.
 */
export function forEachDab(
  stroke: MaskStroke, width: number, height: number, longEdge: number,
  visit: (x: number, y: number) => void,
): void {
  const radius = Math.max(1, stroke.radius * longEdge);
  const step = Math.max(1, radius * 0.5);
  const points = stroke.points;
  if (!points.length) return;

  let [previousX, previousY] = points[0];
  visit(previousX * width, previousY * height);
  for (let index = 1; index < points.length; index++) {
    const [nextX, nextY] = points[index];
    const x0 = previousX * width, y0 = previousY * height;
    const x1 = nextX * width, y1 = nextY * height;
    const distance = Math.hypot(x1 - x0, y1 - y0);
    const steps = Math.ceil(distance / step);
    for (let stepIndex = 1; stepIndex <= steps; stepIndex++) {
      const t = stepIndex / steps;
      visit(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t);
    }
    previousX = nextX;
    previousY = nextY;
  }
}

/** The pixels a dab can touch: inclusive integer bounds, clipped to the buffer. */
export function dabBounds(centreX: number, centreY: number, radius: number, width: number, height: number) {
  return {
    left: Math.max(0, Math.floor(centreX - radius)),
    right: Math.min(width - 1, Math.ceil(centreX + radius)),
    top: Math.max(0, Math.floor(centreY - radius)),
    bottom: Math.min(height - 1, Math.ceil(centreY + radius)),
  };
}

function stampDab(
  coverage: Uint8Array, width: number, height: number,
  centreX: number, centreY: number, dab: { radius: number; solid: number; fade: number }, erase: boolean,
): void {
  const { radius, solid, fade } = dab;
  const { left, right, top, bottom } = dabBounds(centreX, centreY, radius, width, height);

  for (let y = top; y <= bottom; y++) {
    const dy = y - centreY;
    for (let x = left; x <= right; x++) {
      const dx = x - centreX;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance > radius) continue;
      const strength = distance <= solid ? 1 : 1 - (distance - solid) / fade;
      const value = Math.round(strength * 255);
      const index = y * width + x;
      // Strokes accumulate towards full rather than replacing, so overlapping dabs within one
      // stroke do not leave visible seams where the falloffs cross.
      coverage[index] = erase
        ? Math.min(coverage[index], 255 - value)
        : Math.max(coverage[index], value);
    }
  }
}

/**
 * Separable box blur over the coverage buffer, three passes for a near-Gaussian edge.
 *
 * Feathering the mask rather than the picture is what keeps a masked adjustment from showing a
 * hard boundary, and doing it here means every node gets it without knowing about it.
 */
export function featherCoverage(coverage: Uint8Array, width: number, height: number, radius: number): void {
  if (radius < 1) return;
  const scratch = new Uint8Array(coverage.length);
  for (let pass = 0; pass < 3; pass++) {
    boxPass(coverage, scratch, width, height, radius);
    boxPass(scratch, coverage, height, width, radius);
  }
}

/** One axis, written out transposed; running it twice lands back in the original orientation. */
function boxPass(
  source: Uint8Array, target: Uint8Array, width: number, height: number, radius: number,
): void {
  const window = radius * 2 + 1;
  for (let y = 0; y < height; y++) {
    const row = y * width;
    let total = 0;
    for (let x = -radius; x <= radius; x++) {
      total += source[row + Math.min(width - 1, Math.max(0, x))];
    }
    for (let x = 0; x < width; x++) {
      target[x * height + y] = Math.round(total / window);
      total += source[row + Math.min(width - 1, Math.max(0, x + radius + 1))]
        - source[row + Math.min(width - 1, Math.max(0, x - radius))];
    }
  }
}

/**
 * Rasterise a generated shape.
 *
 * Split out from `rasterizeMask` so the three kinds stay readable, and so a caller that only wants
 * a preview of the shape -- while dragging a gradient handle, say -- can ask for it on its own.
 */
export function shapeCoverage(
  shape: MaskShape, width: number, height: number, pixels?: Uint8ClampedArray,
): Uint8Array {
  const coverage = new Uint8Array(width * height);
  if (shape.kind === "linear") {
    // Project each pixel onto the line and use how far along it lands. A zero-length line would
    // divide by nothing, so it degrades to covering everything rather than to NaN.
    const ax = shape.x1 * width, ay = shape.y1 * height;
    const bx = shape.x2 * width, by = shape.y2 * height;
    const dx = bx - ax, dy = by - ay;
    const lengthSquared = dx * dx + dy * dy;
    if (lengthSquared < 1e-6) {
      coverage.fill(255);
      return coverage;
    }
    for (let y = 0, index = 0; y < height; y++) {
      for (let x = 0; x < width; x++, index++) {
        const t = ((x - ax) * dx + (y - ay) * dy) / lengthSquared;
        const clamped = t < 0 ? 0 : t > 1 ? 1 : t;
        coverage[index] = Math.round(clamped * clamped * (3 - 2 * clamped) * 255);
      }
    }
    return coverage;
  }
  if (shape.kind === "radial") {
    const cx = shape.cx * width, cy = shape.cy * height;
    const long = Math.max(width, height);
    const aspect = shape.aspect || 1;
    const radiusX = Math.max(1, shape.radius * long * aspect);
    const radiusY = Math.max(1, shape.radius * long / aspect);
    const solid = 1 - Math.min(0.999, Math.max(0, shape.feather));
    for (let y = 0, index = 0; y < height; y++) {
      for (let x = 0; x < width; x++, index++) {
        const nx = (x - cx) / radiusX, ny = (y - cy) / radiusY;
        const distance = Math.sqrt(nx * nx + ny * ny);
        if (distance >= 1) continue;
        const t = distance <= solid ? 1 : 1 - (distance - solid) / Math.max(1e-6, 1 - solid);
        coverage[index] = Math.round(t * t * (3 - 2 * t) * 255);
      }
    }
    return coverage;
  }
  // Colour range. With no pixels there is nothing to compare against, and covering everything would
  // silently apply the node to the whole picture -- so it covers nothing instead.
  if (!pixels || pixels.length < width * height * 4) return coverage;
  // 442 is the longest distance in RGB, so a tolerance of 100 reaches every colour.
  const tolerance = Math.max(1, shape.tolerance) * 4.42;
  const softness = Math.max(0.01, shape.softness);
  for (let index = 0, at = 0; index < coverage.length; index++, at += 4) {
    const dr = pixels[at] - shape.r, dg = pixels[at + 1] - shape.g, db = pixels[at + 2] - shape.b;
    const distance = Math.sqrt(dr * dr + dg * dg + db * db);
    if (distance >= tolerance) continue;
    const t = 1 - distance / tolerance;
    // Softness bends the falloff: low keeps the selection crisp, high lets it trail off.
    coverage[index] = Math.round(Math.pow(t, 1 / softness) * 255);
  }
  return coverage;
}

/** Pull the coverage out of a greyscale or alpha PNG the model wrote. */
export function coverageFromImage(data: Uint8ClampedArray, pixels: number): Uint8Array {
  const coverage = new Uint8Array(pixels);
  for (let index = 0; index < pixels; index++) {
    const at = index * 4;
    // Prefer alpha when the file carries one, since a cutout's shape lives there; otherwise the
    // file is a plain white-on-black matte and luminance is the mask.
    const alpha = data[at + 3];
    coverage[index] = alpha < 255 ? alpha : (data[at] * 0.299 + data[at + 1] * 0.587 + data[at + 2] * 0.114) | 0;
  }
  return coverage;
}

/* ================================================================ VFX Phase 2: mask stacks */

/**
 * Everything a mask stack can read besides its own parameters, at one working resolution.
 *
 * Files and layers are looked up rather than passed in, so the same stack can be rasterised by the
 * CPU compositor (which has the layers' pixels) and fed to the GPU (which only needs the files).
 */
export type MaskContext = {
  width: number;
  height: number;
  /** The scope's ungraded pixels, for a colour-range shape. */
  pixels?: Uint8ClampedArray;
  /** Working pixels per canvas pixel; entry feather and expand are scaled by it. */
  canvasScale?: number;
  /** The base's SAM 2 file, placed, decoded the legacy way (`coverageFromImage`). */
  legacyMatte?: Uint8Array;
  /** A file entry's coverage, placed and decoded (`coverageFromFile`), by its request. */
  matte?: (request: MatteRequest) => Uint8Array | undefined;
  /** A layer entry's coverage (`layerCoverage`). Undefined for a missing layer or a matte cycle. */
  layer?: (layerId: string, channel: "alpha" | "luma") => Uint8Array | undefined;
  /** Frame index for sequence sources. Photo Edit renders frame 0. */
  frame?: number;
  /**
   * VFX Phase 4: a `depth` entry's layer's depth frame, placed through that layer and read as
   * alpha-weighted luma (`layerCoverage(placed, "luma", 1)`): disparity, 255 near, 0 where the layer
   * is not. Undefined when the layer has no depth, is off screen, or its frame has not loaded.
   */
  depth?: (layerId: string) => Uint8Array | undefined;
  /** VFX Phase 6: a `path` entry's fill coverage at this size (`pathCoverage`, a canvas raster). */
  path?: (path: Extract<MaskSource, { kind: "path" }>["path"]) => Uint8Array | undefined;
};

/** A file an entry reads, and how its pixels become coverage. */
export type MatteRequest = { path: string; channel: "auto" | "alpha" | "luma" };

/** The file behind a file or sequence entry at a frame, or undefined for the other kinds. */
export function matteRequest(source: MaskSource, frame = 0): MatteRequest | undefined {
  if (source.kind === "file") return { path: source.path, channel: source.channel ?? "auto" };
  if (source.kind === "sequence") return { path: sequenceFramePath(source, source.frame ?? frame), channel: "auto" };
  return undefined;
}

/** `<cachePath>/masks/00000.png`, the `mask_cache.py` layout, clamped to the frames that exist. */
export function sequenceFramePath(source: Extract<MaskSource, { kind: "sequence" }>, frame: number): string {
  const last = Math.max(0, (source.frameCount | 0) - 1);
  const index = Math.min(last, Math.max(0, Math.floor(frame) - Math.floor(source.first ?? 0)));
  const separator = source.cachePath.includes("\\") && !source.cachePath.includes("/") ? "\\" : "/";
  const base = source.cachePath.replace(/[\\/]+$/, "");
  return `${base}${separator}masks${separator}${String(index).padStart(5, "0")}.png`;
}

/** Every file a mask needs loaded: the base's SAM 2 file and each file or sequence entry's frame. */
export function maskFilePaths(mask: PhotoMask, frame = 0): string[] {
  const paths: string[] = [];
  if (mask.sourcePath) paths.push(mask.sourcePath);
  for (const entry of mask.stack ?? []) {
    const request = matteRequest(entry.source, frame);
    if (request) paths.push(request.path);
  }
  return paths;
}

/** The layers a mask reads through `layer` entries (enabled or not: a toggle must not reload). */
export function maskLayerIds(mask: PhotoMask): string[] {
  return (mask.stack ?? []).flatMap((entry) => entry.source.kind === "layer" ? [entry.source.layerId] : []);
}

/** True when a mask's coverage depends on pixels that change between frames (colour, layers). */
export function maskReadsPixels(mask: PhotoMask): boolean {
  if (!mask.enabled) return false;
  if (mask.shape?.kind === "colour") return true;
  return (mask.stack ?? []).some((entry) => entry.enabled
    && (entry.source.kind === "layer" || (entry.source.kind === "shape" && entry.source.shape.kind === "colour")));
}

/**
 * Rasterise a whole mask stack: the base (exactly `rasterizeMask`), then each enabled entry combined
 * into it. A mask without `stack` goes through `rasterizeMask` alone, so every mask written before
 * VFX Phase 2 comes out byte for byte as it always did.
 *
 * An entry whose file has not loaded, or whose layer is missing or would read itself (a matte
 * cycle), is skipped: it neither adds nor restricts until it can be drawn.
 */
export function rasterizeMaskStack(mask: PhotoMask, context: MaskContext): Uint8Array {
  const { width, height } = context;
  const coverage = rasterizeMask(mask, width, height, context.legacyMatte, context.pixels, context.canvasScale ?? 1);
  if (!mask.enabled || !mask.stack?.length) return coverage;
  for (const entry of mask.stack) {
    if (!entry.enabled) continue;
    const own = entryCoverage(entry, context);
    if (own) combineCoverage(coverage, own, entry.mode, entry.opacity);
  }
  return coverage;
}

/** One entry's coverage, before combining: source, strokes, expand/choke, feather, invert. */
export function entryCoverage(entry: MaskEntry, context: MaskContext): Uint8Array | undefined {
  const { width, height } = context;
  const source = entry.source;
  let coverage: Uint8Array;
  if (source.kind === "paint") coverage = new Uint8Array(width * height);
  else if (source.kind === "shape") coverage = shapeCoverage(source.shape, width, height, context.pixels);
  else if (source.kind === "layer") {
    const held = context.layer?.(source.layerId, source.channel);
    if (!held || held.length !== width * height) return undefined;
    coverage = new Uint8Array(held);
  } else if (source.kind === "depth") {
    const held = context.depth?.(source.layerId);
    if (!held || held.length !== width * height) return undefined;
    coverage = applyLut(held, depthThresholdLut(source.plane, source.softness));
  } else if (source.kind === "path") {
    const held = context.path?.(source.path);
    if (!held || held.length !== width * height) return undefined;
    coverage = new Uint8Array(held);
  } else {
    const request = matteRequest(source, context.frame);
    const held = request ? context.matte?.(request) : undefined;
    if (!held || held.length !== width * height) return undefined;
    coverage = new Uint8Array(held);
  }
  const longEdge = Math.max(width, height);
  for (const stroke of entry.strokes) stampStroke(coverage, width, height, stroke, longEdge);
  const scale = context.canvasScale ?? 1;
  const expand = Math.round((entry.expand || 0) * scale);
  if (expand !== 0) morphCoverage(coverage, width, height, expand);
  const feather = Math.round((entry.feather || 0) * scale);
  if (feather >= 1) featherCoverage(coverage, width, height, feather);
  if (entry.invert) {
    for (let index = 0; index < coverage.length; index++) coverage[index] = 255 - coverage[index];
  }
  return coverage;
}

/** Index of each mode, as the GPU's combine shader numbers them. */
export const MASK_MODE_INDEX: Record<MaskMode, number> = { add: 0, subtract: 1, intersect: 2, difference: 3 };

/**
 * Combine an entry into the accumulated coverage, in place. Bytes throughout: the entry is scaled by
 * its opacity and rounded first, then the mode is applied and rounded. None of the modes can land
 * exactly on .5 (a*e/255 never does), so Math.round and the GPU's floor(x + 0.5) always agree.
 */
export function combineCoverage(accumulated: Uint8Array, entry: Uint8Array, mode: MaskMode, opacity: number): void {
  const strength = Math.min(1, Math.max(0, opacity));
  for (let index = 0; index < accumulated.length; index++) {
    const a = accumulated[index];
    const e = strength === 1 ? entry[index] : Math.round(entry[index] * strength);
    let value: number;
    if (mode === "subtract") value = a * (255 - e) / 255;
    else if (mode === "intersect") value = a * e / 255;
    else if (mode === "difference") value = Math.abs(a - e);
    else value = a + e - a * e / 255;
    accumulated[index] = Math.round(value);
  }
}

/**
 * Grow (radius > 0, a max filter) or choke (radius < 0, a min filter) coverage by a square of
 * |radius| pixels, separably: rows, then columns, edges repeated. Square rather than round because it
 * is separable (two passes instead of r^2 reads a pixel on the GPU) and exact in bytes; the feather
 * that usually follows rounds the corners.
 */
export function morphCoverage(coverage: Uint8Array, width: number, height: number, radius: number): void {
  if (radius === 0) return;
  const grow = radius > 0;
  const size = Math.abs(radius);
  const scratch = new Uint8Array(coverage.length);
  const pick = grow ? Math.max : Math.min;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = coverage[y * width + x];
      for (let k = -size; k <= size; k++) {
        value = pick(value, coverage[y * width + Math.min(width - 1, Math.max(0, x + k))]);
      }
      scratch[y * width + x] = value;
    }
  }
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let value = scratch[y * width + x];
      for (let k = -size; k <= size; k++) {
        value = pick(value, scratch[Math.min(height - 1, Math.max(0, y + k)) * width + x]);
      }
      coverage[y * width + x] = value;
    }
  }
}

/**
 * Coverage from an imported matte file (VFX Phase 2), placed at the working size.
 *
 * Rec. 709 and rounded, not the Rec. 601 truncation `coverageFromImage` keeps for SAM 2 files: that
 * one maps 65 of the 256 grey levels to one below themselves ((v*0.299 + v*0.587 + v*0.114) | 0 falls
 * just short of v), which is kept for the base so old masks do not move, and not repeated here.
 * "auto" decides per FILE, not per pixel: alpha when any pixel is transparent (a cutout), luma when
 * none is (a white-on-black matte). Per pixel, an opaque pixel inside a cutout would read its colour.
 */
export function coverageFromFile(data: Uint8ClampedArray, pixels: number, channel: "auto" | "alpha" | "luma" = "auto"): Uint8Array {
  const coverage = new Uint8Array(pixels);
  let useAlpha = channel === "alpha";
  if (channel === "auto") {
    for (let index = 0; index < pixels; index++) {
      if (data[index * 4 + 3] < 255) { useAlpha = true; break; }
    }
  }
  for (let index = 0; index < pixels; index++) {
    const at = index * 4;
    coverage[index] = useAlpha ? data[at + 3]
      : Math.round(0.2126 * data[at] + 0.7152 * data[at + 1] + 0.0722 * data[at + 2]);
  }
  return coverage;
}

/**
 * A layer's result as coverage: its alpha, or its Rec. 709 luma weighted by alpha (transparent
 * counts as black, as in After Effects), times the layer's opacity. Rounded once.
 */
export function layerCoverage(pixels: Uint8ClampedArray, channel: "alpha" | "luma", opacity: number): Uint8Array {
  const count = pixels.length / 4;
  const coverage = new Uint8Array(count);
  const strength = Math.min(1, Math.max(0, opacity));
  for (let index = 0, at = 0; index < count; index++, at += 4) {
    const alpha = pixels[at + 3];
    const value = channel === "alpha" ? alpha * strength
      : (0.2126 * pixels[at] + 0.7152 * pixels[at + 1] + 0.0722 * pixels[at + 2]) * (alpha / 255) * strength;
    coverage[index] = Math.round(value);
  }
  return coverage;
}
