/**
 * The GPU compositor, Phase 1: Photo Edit's `composeDocument` rebuilt on WebGL2 with the same output.
 *
 * Same inputs, same result type, same order of operations -- so the Photo Edit call site switches
 * with a flag and nothing else. The document is converted to a `Composition` on the way in (the
 * project file is not migrated), and each stage below names the CPU function it replaces.
 *
 * On the GPU: placement, gradient fills, the grade (pointwise, denoise, blur, sharpen, grain), the
 * gradient map, mask mixing and every blend mode.
 * Still on the CPU, uploaded as textures: text and shape rasterisation, mask rasterisation, relight,
 * retouch, warp and the edge group (defringe, contact shadow, light wrap). Each is recorded in
 * `stats.cpu` so it is visible which frames paid for a readback.
 */

import { applyEnvelopeRaster, isIdentityEnvelope } from "../domain/photoEnvelope";
import { expandRepeats } from "../domain/photoRepeat";
import { applyClipEffects, isNeutralClipEffects } from "../domain/photoClipEffects";
import { BLEND_MODES, compositeOver } from "../domain/photoBlend";
import {
  COMPOSITE_SCOPE, MATTE_CYCLE, canvasSizeOf, clipBases, croppedRect, drawnSource, layerMatrix, placeLayer,
  referencedLayers,
} from "../domain/photoCompose";
import type { ComposeInput, ComposeResult, ComposeScope, LayerSource } from "../domain/photoCompose";
import {
  PHOTO_LAYER_KINDS, drawableLayers, fromPhotoDocument, toPhotoLayer, type CompositionLayer,
} from "../domain/composition";
import type { LayerMedia, MaskEntry, MaskShape, MaskStroke, MatrixRainGenerator, PhotoLayer, PhotoMask, PhotoNode } from "../domain/photoDocument";
import { MASK_MODE_INDEX, dabBounds, dabShape, forEachDab, matteRequest } from "../domain/photoMask";
import { depthKey, legacyMatteKey, matteKey, pathMatteKey, wantedMasks } from "../domain/photoResources";
import { applyLightWrap, buildContactShadow, defringeLayer, isNeutralEdge } from "../domain/photoEdge";
import {
  ENCODE_CEILING, ENCODE_STEPS, HSL_BAND_WIDTH, HSL_CENTRES, buildOutputLuts, isNeutralGrade, photoFilter,
  srgbToLinear, type PhotoAdjustments,
} from "../domain/photoEdit";
import { BAYER, buildGradientLut, type Gradient } from "../domain/photoGradient";
import { invert } from "../domain/photoGeometry";
import { applyRelight } from "../domain/photoRelight";
import type { RenderReport, RenderResources } from "../domain/photoRender";
import { applyRetouch } from "../domain/photoRetouch";
import { applyWarp } from "../domain/photoWarp";
import { GlContext, type Target } from "./gl";
import {
  ALPHA_CUT_FS, BOX_FS, COMPOSITE_FS, COVERAGE_COPY_FS, DAB_FS, DAB_VS, DENOISE_FS, FILL_FS, GRADE_FS,
  GRADIENT_FILL_FS, GRADIENT_MAP_FS, GRAIN_FS, LAYER_COVERAGE_FS, LUMA_KEY_FS, MASK_COMBINE_FS, MASK_INVERT_FS,
  MASK_MIX_FS, MASK_SHAPE_FS, MORPH_FS, PLACE_FS, PLANES_FS, PLANE_BOX_FS, SHARPEN_FS, UNMULTIPLY_FS,
  ALPHA_CURVE_FS, CROSSFADE_FS, MATRIX_RAIN_FS, DEPTH_THRESHOLD_FS, TRAIL_FS,
} from "./shaders";
import {
  attachedLayer, depthFrameOf, emitterParticles, particleKey, particleLayer, type CanvasSize, type Particle,
} from "../domain/photoTrack";
import { depthThresholdLut } from "../domain/photoMask";
import { MASK_FX_NO_MATTE, TRAIL_NO_TRACK, runMaskVfx } from "../domain/photoRender";
import { placementSize, trailFields } from "../domain/photoCompose";
import { buildRainAtlas, hexBytes, rainLayout, type RainAtlas } from "../domain/matrixRain";
import { compositionRate, mediaFrameKey, mediaFramesAt, scopeFrame, type MediaFrame } from "../domain/photoTime";

export type RendererStats = {
  /** What ran on the CPU this frame, one entry per occurrence ("relight", "edge", ...). */
  cpu: string[];
  /** Full-frame readbacks, including the final one. */
  readbacks: number;
  passes: number;
  milliseconds: number;
};

export type GpuComposeResult = ComposeResult & { stats: RendererStats };

export type RendererOptions = {
  /**
   * "gpu" places layers with the shader; "canvas" uses the old `placeLayer` and uploads the result.
   * The second exists for diagnosis: if parity fails, it tells placement apart from everything else.
   */
  placement?: "gpu" | "canvas";
};

/** Returned by a node that would leave its input unchanged (a neutral grade), so the mask logic still runs. */
const UNCHANGED = Symbol("unchanged");

const EMPTY = new Uint8ClampedArray(0);

/** Every pass, by the name `pass()` is called with, so they can all be compiled up front. */
const PROGRAMS: Record<string, string> = {
  fill: FILL_FS, place: PLACE_FS, gradientFill: GRADIENT_FILL_FS, composite: COMPOSITE_FS,
  maskMix: MASK_MIX_FS, gradientMap: GRADIENT_MAP_FS, grade: GRADE_FS, sharpen: SHARPEN_FS,
  grain: GRAIN_FS, box: BOX_FS, planeBox: PLANE_BOX_FS, planes: PLANES_FS, denoise: DENOISE_FS,
  // VFX Phase 2
  coverageCopy: COVERAGE_COPY_FS, maskShape: MASK_SHAPE_FS, morph: MORPH_FS, maskInvert: MASK_INVERT_FS,
  maskCombine: MASK_COMBINE_FS, layerCoverage: LAYER_COVERAGE_FS, alphaCut: ALPHA_CUT_FS,
  unmultiply: UNMULTIPLY_FS, lumaKey: LUMA_KEY_FS,
  // VFX Phase 3
  crossfade: CROSSFADE_FS, alphaCurve: ALPHA_CURVE_FS, matrixRain: MATRIX_RAIN_FS,
  // VFX Phase 4
  trail: TRAIL_FS, depthThreshold: DEPTH_THRESHOLD_FS,
};

/** Programs with their own vertex shader. */
const INSTANCED_PROGRAMS: Record<string, [string, string]> = { dab: [DAB_VS, DAB_FS] };

/** GPU memory the renderer may keep for layer sources between frames. */
const SOURCE_TEXTURE_BUDGET = 1.5 * 1024 ** 3;

export class PhotoRenderer {
  readonly gl: GlContext;
  private readonly sourceTextures = new Map<CanvasImageSource, { texture: WebGLTexture; bytes: number }>();
  private sourceBytes = 0;
  private readonly noise = new Map<string, WebGLTexture>();
  private toLinear?: WebGLTexture;
  private unpremultiply?: WebGLTexture;
  private bayer?: WebGLTexture;
  /** Matrix-rain glyph atlases by the atlas object (which is itself cached by size). */
  private readonly atlases = new Map<RainAtlas, WebGLTexture>();
  /** Textures that live for one `compose` call (masks, LUTs, CPU results). */
  private scratch: WebGLTexture[] = [];
  private stats: RendererStats = { cpu: [], readbacks: 0, passes: 0, milliseconds: 0 };

  constructor(gl?: GlContext) {
    this.gl = gl ?? new GlContext();
    for (const [name, fragment] of Object.entries(PROGRAMS)) this.gl.prepare(name, fragment);
    for (const [name, [vertex, fragment]] of Object.entries(INSTANCED_PROGRAMS)) this.gl.prepare(name, fragment, vertex);
  }

  compose(input: ComposeInput, options: RendererOptions = {}): GpuComposeResult {
    if (this.gl.lost) throw new Error("the WebGL context was lost");
    const started = performance.now();
    this.stats = { cpu: [], readbacks: 0, passes: 0, milliseconds: 0 };
    const { width, height, sources, resourcesFor } = input;
    const limit = this.gl.maxTextureSize;
    if (width > limit || height > limit) throw new Error(`${width}x${height} is larger than this GPU's ${limit}px texture limit`);
    this.gl.trimPool(width, height);
    const full = canvasSizeOf(input.document, sources);
    // VFX Phase 6: repeated layers become their copies first, exactly as in `composeDocument`.
    const document = expandRepeats(input.document, full ?? { width, height });
    const composition = fromPhotoDocument(document);
    const canvasScale = full && full.width > 0 ? width / full.width : 1;
    const report: RenderReport = { skipped: [] };
    const raw = new Map<string, Uint8ClampedArray>();
    const backdrops = new Map<string, Uint8ClampedArray>();
    let maskCoverage: Uint8Array | undefined;

    // VFX Phase 2: a layer's result after placement, nodes and its alpha cut, made at most once and
    // kept while another layer may read it (a clip base or a track matte). Same walk, same order and
    // same reports as `composeDocument`, which is the reference.
    const frame = Math.max(0, Math.floor(input.frame ?? 0));
    const rate = compositionRate(document);
    const canvas: CanvasSize = full ?? { width, height };
    const layers = drawableLayers(composition).filter((layer) => PHOTO_LAYER_KINDS.includes(layer.kind));
    // VFX Phase 4: each layer as placed at this frame (attach-to-track), exactly as `composeDocument`.
    const photoLayers = new Map(layers.map((layer) => [layer.id, attachedLayer(document, toPhotoLayer(layer), frame, canvas, rate)]));
    const byId = new Map(layers.map((layer) => [layer.id, layer]));
    const bases = clipBases([...photoLayers.values()]);
    const referenced = referencedLayers(document);
    const prepared = new Map<string, Target | null>();
    const visiting = new Set<string>();
    const owned: Target[] = [];
    const depthFrame = (layerId: string) => depthFrameOf(photoLayers, layerId, frame, rate);

    const layerTarget = (layerId: string) => {
      const layer = byId.get(layerId);
      const out = layer ? prepare(layer) : undefined;
      return layer && out ? { target: out, opacity: layer.opacity } : undefined;
    };

    const prepare = (layer: CompositionLayer): Target | undefined => {
      const held = prepared.get(layer.id);
      if (held !== undefined) return held ?? undefined;
      if (visiting.has(layer.id)) {
        report.skipped.push({ nodeId: layer.id, reason: MATTE_CYCLE });
        return undefined;
      }
      visiting.add(layer.id);
      const out = prepareOnce(layer);
      visiting.delete(layer.id);
      prepared.set(layer.id, out ?? null);
      return out;
    };

    const prepareOnce = (layer: CompositionLayer): Target | undefined => {
      const photoLayer = photoLayers.get(layer.id)!;
      let placed: Target;
      let shown: MediaFrame[] | undefined;
      if (layer.kind === "video") {
        shown = mediaFramesAt(layer.media, frame, rate);
        if (!shown) {
          placed = this.filledRgba(width, height);
        } else {
          const frames: Target[] = [];
          for (const one of shown) {
            const source = sources.get(mediaFrameKey(layer.media, one.index, input.useProxy));
            if (!source) {
              for (const made of frames) this.gl.release(made);
              report.skipped.push({ nodeId: layer.id, reason: "its picture is still loading" });
              return undefined;
            }
            frames.push(this.place(photoLayer, source, width, height, canvasScale, options));
          }
          if (frames.length === 2) {
            placed = this.pass("crossfade", CROSSFADE_FS, this.gl.target(width, height),
              { frameA: frames[0].texture, frameB: frames[1].texture }, { weight: shown[1].weight });
            this.gl.release(frames[0]);
            this.gl.release(frames[1]);
          } else placed = frames[0];
        }
      } else if (layer.kind === "generator" && layer.generator.kind === "emitter") {
        const particles = emitterParticles(document, layer.generator, frame, rate, canvas);
        const drawn = this.emitter(particles, layer.generator.element, sources, width, height, canvasScale, input.useProxy, options);
        if (!drawn) {
          report.skipped.push({ nodeId: layer.id, reason: "its picture is still loading" });
          return undefined;
        }
        placed = drawn;
      } else if (layer.kind === "generator" && layer.generator.kind === "matrixRain") {
        placed = this.matrixRain(layer.generator, width, height, frame, canvasScale);
      } else if (layer.kind === "gradient") {
        placed = this.gradientFill(layer.gradient, width, height);
      } else {
        const source = layer.kind === "text" || layer.kind === "shape" || layer.kind === "vector" || layer.kind === "graphic"
          ? drawnSource(photoLayer) : sources.get(layer.source.sourcePath);
        if (!source) {
          report.skipped.push({ nodeId: layer.id, reason: "its picture is still loading" });
          return undefined;
        }
        placed = this.place(photoLayer, source, width, height, canvasScale, options);
      }
      if (input.capture === layer.id) raw.set(layer.id, this.read(placed));

      const captureMask = layer.masks.some((mask) => mask.id === input.captureMask) ? input.captureMask : undefined;
      const scope: ComposeScope = {
        nodes: layer.nodes, masks: layer.masks, layer: photoLayer, canvasScale, captureMask,
        frame: scopeFrame(photoLayer, shown, frame), filesOnly: true, depthFrame,
      };
      const resources = resourcesFor(scope, EMPTY);
      resources.trails = trailFields(document, photoLayer, placementSize(photoLayer, sources), frame, rate, canvas, width, height, canvasScale);
      Object.assign(resources, { canvasScale, frame, pictures: (key: string) => sources.get(key) });
      const masks = this.masks(scope, placed, resources, layerTarget);
      if (captureMask && masks.has(captureMask)) maskCoverage = this.readCoverage(masks.get(captureMask)!);
      let graded = this.stack(placed, width, height, layer.nodes, masks, resources, report);

      const mask = layer.maskId ? masks.get(layer.maskId) : undefined;
      const base = bases.get(layer.id);
      const baseTarget = base ? prepare(byId.get(base.id)!) : undefined;
      if (base && !baseTarget) {
        this.releaseAll(masks);
        this.gl.release(graded);
        return undefined;
      }
      if (mask || baseTarget) {
        const cut = this.pass("alphaCut", ALPHA_CUT_FS, this.gl.target(width, height), {
          layer: graded.texture, mask: (mask ?? graded).texture, base: (baseTarget ?? graded).texture,
        }, { useMask: { int: mask ? 1 : 0 }, useBase: { int: baseTarget ? 1 : 0 }, baseOpacity: Math.min(1, Math.max(0, base?.opacity ?? 1)) });
        this.gl.release(graded);
        graded = cut;
      }
      this.releaseAll(masks);
      owned.push(graded);
      return graded;
    };

    try {
      let pixels = this.gl.target(width, height);
      const fill = composition.background === "white" ? 255 : 0;
      const opaque = composition.background === "black" || composition.background === "white";
      this.pass("fill", FILL_FS, pixels, {}, { value: [fill * Number(opaque), fill * Number(opaque), fill * Number(opaque), opaque ? 255 : 0] });

      for (const layer of layers) {
        if (!layer.visible || layer.opacity <= 0) continue;
        if (layer.kind === "video" && !mediaFramesAt(layer.media, frame, rate)) continue;
        const base = bases.get(layer.id);
        if (base && (!base.visible || base.opacity <= 0)) continue;
        const ready = prepare(layer);
        if (!ready) continue;
        let graded = ready;

        const edge = layer.edge;
        if (edge && !isNeutralEdge(edge)) {
          // CPU fallback: the edge group reads and writes both the layer and the backdrop. Into a new
          // target, so a layer another one reads (a clip base, a matte) keeps its prepared result.
          this.stats.cpu.push("edge");
          const gradedPixels = this.read(graded);
          const backdrop = this.read(pixels);
          if (edge.defringe > 0) defringeLayer(gradedPixels, width, height, edge.defringe);
          const shadow = buildContactShadow(gradedPixels, width, height, edge);
          if (shadow) compositeOver(backdrop, shadow, "normal", 1);
          if (edge.lightWrap > 0) applyLightWrap(gradedPixels, backdrop, width, height, edge.lightWrap, edge.lightWrapSpread);
          graded = this.uploaded(gradedPixels, width, height);
          this.gl.uploadInto(pixels, backdrop);
        }

        if (input.capture === layer.id) backdrops.set(layer.id, this.read(pixels));
        const next = this.composite(pixels, graded, layer.blend ?? "normal", layer.opacity);
        this.gl.release(pixels);
        pixels = next;
        if (graded !== ready) this.gl.release(graded);
        if (!referenced.has(layer.id)) {
          owned.splice(owned.indexOf(ready), 1);
          this.gl.release(ready);
        }
      }

      const wantsComposite = input.capture === COMPOSITE_SCOPE;
      if (wantsComposite) raw.set(COMPOSITE_SCOPE, this.read(pixels));
      const layered = composition.photoShape === "layers";
      const compositeCapture = layered && composition.masks.some((mask) => mask.id === input.captureMask)
        ? input.captureMask : undefined;
      if (composition.nodes.length || compositeCapture) {
        const scope: ComposeScope = {
          nodes: composition.nodes, masks: composition.masks, canvasScale, captureMask: compositeCapture, frame, filesOnly: true, depthFrame,
        };
        const resources = resourcesFor(scope, EMPTY);
        resources.trails = trailFields(document, undefined, undefined, frame, rate, canvas, width, height, canvasScale, composition.nodes);
        Object.assign(resources, { canvasScale, frame, pictures: (key: string) => sources.get(key) });
        const masks = this.masks(scope, pixels, resources, layerTarget);
        if (compositeCapture && masks.has(compositeCapture)) maskCoverage = this.readCoverage(masks.get(compositeCapture)!);
        pixels = this.stack(pixels, width, height, composition.nodes, masks, resources, report);
        this.releaseAll(masks);
      }

      const out = this.read(pixels);
      this.gl.release(pixels);
      // Allocation failures (out of GPU memory) do not throw in WebGL; they show up here, and the
      // caller falls back to the CPU compositor rather than using a frame built from garbage.
      this.gl.check("while composing");
      this.stats.milliseconds = performance.now() - started;
      return { pixels: out, report, raw, backdrops, maskCoverage, stats: this.stats };
    } finally {
      for (const target of owned) this.gl.release(target);
      for (const texture of this.scratch) this.gl.deleteTexture(texture);
      this.scratch = [];
    }
  }

  /** Drop the texture made from `image` (a media frame the frame store evicted). */
  releaseSource(image: CanvasImageSource): void {
    const held = this.sourceTextures.get(image);
    if (!held) return;
    this.gl.deleteTexture(held.texture);
    this.sourceBytes -= held.bytes;
    this.sourceTextures.delete(image);
  }

  /** Frees everything on the GPU. The renderer cannot be used afterwards. */
  dispose(): void {
    if (!this.gl.lost) {
      for (const entry of this.sourceTextures.values()) this.gl.deleteTexture(entry.texture);
      for (const texture of this.noise.values()) this.gl.deleteTexture(texture);
      this.gl.deleteTexture(this.toLinear);
      this.gl.deleteTexture(this.unpremultiply);
      this.gl.deleteTexture(this.bayer);
      for (const texture of this.atlases.values()) this.gl.deleteTexture(texture);
      this.gl.destroy();
    }
    this.sourceTextures.clear();
    this.noise.clear();
    this.sourceBytes = 0;
  }

  /* ------------------------------------------------------------ plumbing */

  private pass(name: string, fragment: string, target: Target,
    textures: Record<string, WebGLTexture>, uniforms: Parameters<GlContext["draw"]>[4] = {}): Target {
    this.gl.draw(name, fragment, target, textures, uniforms);
    this.stats.passes++;
    return target;
  }

  private read(target: Target): Uint8ClampedArray {
    this.stats.readbacks++;
    return this.gl.read(target);
  }

  private temporary(texture: WebGLTexture): WebGLTexture {
    this.scratch.push(texture);
    return texture;
  }

  /** CPU pixels as a pooled target, so they can be released like any other stage. */
  private uploaded(pixels: Uint8ClampedArray, width: number, height: number): Target {
    const target = this.gl.target(width, height);
    this.gl.uploadInto(target, pixels);
    return target;
  }

  private sourceTexture(source: LayerSource): WebGLTexture {
    const held = this.sourceTextures.get(source.image);
    if (held) {
      // Most recently used goes to the back, so eviction takes the stalest first.
      this.sourceTextures.delete(source.image);
      this.sourceTextures.set(source.image, held);
      return held.texture;
    }
    // Sources are immutable once loaded (files) or cached by content (drawn text and shapes), so the
    // image object is the key. Bounded by bytes: full-resolution sources are the big allocation here
    // (an 8K RGBA8 picture is 256 MB), and they have to share the card with a resident model.
    const bytes = source.width * source.height * 4;
    while (this.sourceTextures.size && this.sourceBytes + bytes > SOURCE_TEXTURE_BUDGET) {
      const [oldest, entry] = this.sourceTextures.entries().next().value!;
      this.gl.deleteTexture(entry.texture);
      this.sourceBytes -= entry.bytes;
      this.sourceTextures.delete(oldest);
    }
    const texture = this.gl.uploadImagePremultiplied(source.image as TexImageSource);
    this.sourceTextures.set(source.image, { texture, bytes });
    this.sourceBytes += bytes;
    return texture;
  }

  /* ------------------------------------------------------------ layers */

  /** `placeLayer`: the layer's source through `layerMatrix`, into a canvas-sized buffer. */
  private place(
    photoLayer: PhotoLayer, source: LayerSource,
    width: number, height: number, canvasScale: number, options: RendererOptions,
  ): Target {
    if (options.placement === "canvas") {
      this.stats.cpu.push("placement");
      return this.uploaded(placeLayer(photoLayer, source, width, height, canvasScale), width, height);
    }
    const rect = croppedRect(photoLayer, source);
    const matrix = layerMatrix(photoLayer, width, height, source, canvasScale);
    const inverse = invert(matrix);
    const target = this.gl.target(width, height);
    return this.pass("place", PLACE_FS, target, { source: this.sourceTexture(source), unpremultiply: this.unpremultiplyTexture() }, {
      inverseRow0: [inverse.a, inverse.c],
      inverseRow1: [inverse.b, inverse.d],
      inverseOffset: [inverse.e, inverse.f],
      rect: [rect.left, rect.top, rect.width, rect.height],
      deviceScale: Math.abs(photoLayer.scale * canvasScale),
      antialias: 1,
    });
  }

  /** `renderGradient`. */
  private gradientFill(gradient: Gradient, width: number, height: number): Target {
    const lut = buildGradientLut(gradient);
    const steps = lut.length / 4;
    const lutTexture = this.temporary(this.gl.uploadRgba(lut, steps, 1));
    const kinds = ["linear", "reflected", "radial", "angular", "diamond"];
    return this.pass("gradientFill", GRADIENT_FILL_FS, this.gl.target(width, height),
      { lut: lutTexture, bayer: this.bayerTexture() }, {
        kind: { int: Math.max(0, kinds.indexOf(gradient.kind)) },
        angle: gradient.angle * Math.PI / 180,
        centre: [gradient.cx, gradient.cy],
        gradScale: gradient.scale,
        offset: gradient.offset,
        repeatCount: gradient.repeat,
        mirror: gradient.mirror ? 1 : 0,
        ease: gradient.ease || 0,
        dither: Math.min(1, Math.max(0, gradient.dither ?? 0)),
        size: [width, height],
        last: { int: steps - 1 },
      });
  }

  /** `compositeOver(base, layer, mode, opacity)`: a new target, both inputs left for the caller. */
  private composite(base: Target, layer: Target, mode: string, opacity: number): Target {
    const strength = Math.min(1, Math.max(0, opacity));
    const out = this.gl.target(base.width, base.height);
    return this.pass("composite", COMPOSITE_FS, out, { base: base.texture, layer: layer.texture }, {
      mode: { int: Math.max(0, BLEND_MODES.findIndex((entry) => entry.id === mode)) },
      strength,
    });
  }

  /* ------------------------------------------------------------ node stack */

  /** `renderStack`: takes ownership of `start` and returns the target holding the result. */
  private stack(
    start: Target, width: number, height: number, nodes: PhotoNode[],
    masks: Map<string, Target>, resources: RenderResources, report: RenderReport,
  ): Target {
    let current = start;
    for (const node of nodes) {
      if (!node.enabled) continue;
      const result = this.node(node, current, width, height, resources, report);
      if (!result) continue;
      const coverage = node.maskId ? masks.get(node.maskId) : undefined;
      if (node.maskId && !coverage) {
        report.skipped.push({ nodeId: node.id, reason: "its mask has not been drawn yet" });
        if (result !== UNCHANGED) this.gl.release(result);
        continue;
      }
      if (result === UNCHANGED) continue;
      if (!coverage) {
        this.gl.release(current);
        current = result;
        continue;
      }
      const mixed = this.pass("maskMix", MASK_MIX_FS, this.gl.target(width, height),
        { current: current.texture, layer: result.texture, coverage: coverage.texture });
      this.gl.release(current);
      this.gl.release(result);
      current = mixed;
    }
    return current;
  }

  /** `renderNode`: a node's full-frame result as a NEW target, UNCHANGED, or undefined to skip. */
  private node(
    node: PhotoNode, current: Target, width: number, height: number,
    resources: RenderResources, report: RenderReport,
  ): Target | typeof UNCHANGED | undefined {
    if (node.kind === "grade") {
      if (isNeutralGrade(node.adjustments)) return UNCHANGED;
      return this.grade(current, width, height, node.adjustments);
    }
    if (node.kind === "retouch") {
      if (!node.strokes.length) return undefined;
      return this.onCpu("retouch", current, width, height, (data) => applyRetouch(data, width, height, node.strokes));
    }
    if (node.kind === "warp") {
      if (!node.strokes.length) return undefined;
      return this.onCpu("warp", current, width, height, (data) => applyWarp(data, width, height, node.strokes));
    }
    if (node.kind === "clipEffects") {
      // VFX Phase 7: the CPU reference's own function, so the two compositors agree exactly.
      if (isNeutralClipEffects(node.effects)) return undefined;
      return this.onCpu("clipEffects", current, width, height, (data) => applyClipEffects(data, width, height, node.effects, resources.canvasScale ?? 1));
    }
    if (node.kind === "maskVfx") {
      // VFX Phase 7: the CPU reference's own function (`renderMaskFx`), so the two agree exactly.
      if (!resources.maskFx?.get(node.id)) {
        report.skipped.push({ nodeId: node.id, reason: MASK_FX_NO_MATTE });
        return undefined;
      }
      let result: Uint8ClampedArray | undefined;
      const target = this.onCpu("maskVfx", current, width, height, (data) => { result = runMaskVfx(node, data, width, height, resources, report); if (result) data.set(result); });
      if (!result) { this.gl.release(target); return undefined; }
      return target;
    }
    if (node.kind === "envelope") {
      // VFX Phase 6: the CPU reference's own function, so the two compositors agree exactly.
      if (isIdentityEnvelope(node.envelope)) return undefined;
      return this.onCpu("envelope", current, width, height, (data) => applyEnvelopeRaster(data, width, height, node.envelope));
    }
    if (node.kind === "gradientMap") {
      if (node.amount <= 0) return undefined;
      return this.gradientMap(current, node.gradient, node.amount);
    }
    if (node.kind === "unmultiply") {
      return this.pass("unmultiply", UNMULTIPLY_FS, this.gl.target(width, height), { source: current.texture });
    }
    if (node.kind === "alphaCurve") {
      return this.pass("alphaCurve", ALPHA_CURVE_FS, this.gl.target(width, height), { source: current.texture }, {
        gain: Math.max(0, node.gain), gamma: Math.max(0.01, node.gamma),
      });
    }
    if (node.kind === "lumaKey") {
      return this.pass("lumaKey", LUMA_KEY_FS, this.gl.target(width, height), { source: current.texture }, {
        low: node.low / 100 * 255, high: node.high / 100 * 255,
        soft: Math.max(0, node.softness) / 100 * 255, invertKey: { int: node.invert ? 1 : 0 },
      });
    }
    if (node.kind === "trail") {
      const field = resources.trails?.get(node.id);
      if (!field) {
        report.skipped.push({ nodeId: node.id, reason: TRAIL_NO_TRACK });
        return undefined;
      }
      if (field.still) return undefined;
      const tableX: number[] = [], tableY: number[] = [];
      for (let bin = 0; bin < field.table.length / 2; bin++) { tableX.push(field.table[bin * 2]); tableY.push(field.table[bin * 2 + 1]); }
      const m = field.inverse;
      return this.pass("trail", TRAIL_FS, this.gl.target(width, height), { source: current.texture }, {
        "tableX[0]": tableX, "tableY[0]": tableY, inverseRow1: [m.b, m.d], inverseOffsetY: m.f,
        rectHeight: field.rectHeight, size: { ints: [width, height] },
      });
    }
    if (node.kind === "relight") {
      const depth = resources.depth;
      if (!depth) {
        report.skipped.push({ nodeId: node.id, reason: "no depth map yet" });
        return undefined;
      }
      return this.onCpu("relight", current, width, height, (data) => applyRelight(data, width, height, depth, node.relight));
    }
    const removal = resources.removals.get(node.id);
    if (!removal || removal.length !== width * height * 4) {
      report.skipped.push({ nodeId: node.id, reason: "not filled yet" });
      return undefined;
    }
    return this.uploaded(removal, width, height);
  }

  /** A node that has not been ported: read back, run the CPU function on a copy, upload. */
  private onCpu(name: string, current: Target, width: number, height: number,
    run: (data: Uint8ClampedArray) => void): Target {
    this.stats.cpu.push(name);
    const data = this.read(current);
    run(data);
    return this.uploaded(data, width, height);
  }

  /** `applyGradientMap`. */
  private gradientMap(current: Target, gradient: Gradient, amount: number): Target {
    const lut = this.temporary(this.gl.uploadRgba(buildGradientLut(gradient, 256), 256, 1));
    return this.pass("gradientMap", GRADIENT_MAP_FS, this.gl.target(current.width, current.height),
      { source: current.texture, lut }, {
        strength: Math.min(1, Math.max(0, amount / 100)),
        repeatCount: gradient.repeat,
        mirror: gradient.mirror ? 1 : 0,
        ease: gradient.ease || 0,
      });
  }

  /** `applyAdjustments`, non-neutral. Returns a new target; `current` is untouched. */
  private grade(current: Target, width: number, height: number, adjustments: PhotoAdjustments): Target {
    const exposure = Math.pow(2, adjustments.exposure);
    const warm = adjustments.temperature / 100;
    const green = adjustments.tint / 100;
    const filter = photoFilter(adjustments.filter);
    const mix = Math.min(1, Math.max(0, adjustments.filterAmount / 100));
    const [lutR, lutG, lutB] = buildOutputLuts(adjustments);
    const packed = new Uint8Array(ENCODE_STEPS * 4);
    for (let index = 0; index < ENCODE_STEPS; index++) {
      packed[index * 4] = lutR[index];
      packed[index * 4 + 1] = lutG[index];
      packed[index * 4 + 2] = lutB[index];
      packed[index * 4 + 3] = 255;
    }
    const outputLut = this.temporary(this.gl.uploadRgba(packed, ENCODE_STEPS, 1));
    const bands = adjustments.hsl ?? [];
    const band = (pick: (entry: { hue: number; saturation: number; luminance: number }) => number) =>
      HSL_CENTRES.map((_, index) => bands[index] ? pick(bands[index]) : 0);
    const centreX = width / 2, centreY = height / 2;

    let out = this.pass("grade", GRADE_FS, this.gl.target(width, height),
      { source: current.texture, toLinear: this.toLinearTexture(), outputLut }, {
        gain: [exposure * (1 + 0.16 * warm), exposure * (1 - 0.09 * green), exposure * (1 - 0.16 * warm)],
        highlights: adjustments.highlights / 100,
        shadows: adjustments.shadows / 100,
        contrast: 1 + adjustments.contrast / 100,
        saturation: (adjustments.saturation / 100) * (1 + (filter.saturation - 1) * mix),
        vibrance: adjustments.vibrance / 100,
        bandsActive: bands.some((entry) => entry.hue || entry.saturation || entry.luminance) ? 1 : 0,
        "bandHue[0]": band((entry) => entry.hue),
        "bandSaturation[0]": band((entry) => entry.saturation),
        "bandLuminance[0]": band((entry) => entry.luminance),
        "bandCentre[0]": HSL_CENTRES,
        bandWidth: HSL_BAND_WIDTH,
        vignette: (adjustments.vignette ?? 0) / 100,
        centre: [centreX, centreY],
        maxRadius: Math.hypot(centreX, centreY) || 1,
        vignetteStart: (adjustments.vignetteMidpoint ?? 55) / 100,
        vignetteSpread: Math.max(0.02, (adjustments.vignetteFeather ?? 60) / 100),
        scale: (ENCODE_STEPS - 1) / ENCODE_CEILING,
        last: { int: ENCODE_STEPS - 1 },
      });

    const denoise = adjustments.denoise ?? 0;
    const denoiseColour = adjustments.denoiseColor ?? 0;
    if ((denoise > 0 || denoiseColour > 0) && width * height >= 4) {
      out = this.replace(out, this.denoise(out, denoise, adjustments.denoiseDetail ?? 0, denoiseColour));
    }
    if (adjustments.blur > 0) out = this.replace(out, this.boxBlur(out, adjustments.blur / 100 * 12));
    if (adjustments.sharpen > 0) {
      const blurred = this.boxBlur(out, 1);
      const sharpened = this.pass("sharpen", SHARPEN_FS, this.gl.target(width, height),
        { source: out.texture, blurred: blurred.texture }, { strength: adjustments.sharpen / 100 * 1.4 });
      this.gl.release(blurred);
      out = this.replace(out, sharpened);
    }
    if (adjustments.grain > 0) {
      out = this.replace(out, this.pass("grain", GRAIN_FS, this.gl.target(width, height),
        { source: out.texture, noise: this.noiseTexture(width, height) }, { strength: adjustments.grain / 100 * 26 }));
    }
    return out;
  }

  private replace(previous: Target, next: Target): Target {
    this.gl.release(previous);
    return next;
  }

  /** `blurBuffer`: three rounds of horizontal then vertical box passes, 8-bit between each. */
  private boxBlur(source: Target, radius: number): Target {
    const size = Math.max(1, Math.round(radius));
    let current = source;
    for (let round = 0; round < 3; round++) {
      for (const direction of [[1, 0], [0, 1]]) {
        const next = this.pass("box", BOX_FS, this.gl.target(source.width, source.height),
          { source: current.texture }, { direction: { ints: direction }, radius: { int: size } });
        if (current !== source) this.gl.release(current);
        current = next;
      }
    }
    return current;
  }

  /** `blurPlane` over the float planes. */
  private planeBlur(planes: Target, radius: number): Target {
    let current = planes;
    for (let round = 0; round < 3; round++) {
      for (const direction of [[1, 0], [0, 1]]) {
        const next = this.pass("planeBox", PLANE_BOX_FS, this.gl.target(planes.width, planes.height, "rgba32f"),
          { source: current.texture }, { direction: { ints: direction }, radius: { int: radius } });
        if (current !== planes) this.gl.release(current);
        current = next;
      }
    }
    return current;
  }

  /** `denoiseBuffer`. */
  private denoise(source: Target, amount: number, detail: number, colour: number): Target {
    const luma = Math.min(1, Math.max(0, amount / 100));
    const chroma = Math.min(1, Math.max(0, colour / 100));
    const keep = Math.min(1, Math.max(0, detail / 100));
    const { width, height } = source;
    const planes = this.pass("planes", PLANES_FS, this.gl.target(width, height, "rgba32f"), { source: source.texture });
    const lumaBase = luma > 0 ? this.planeBlur(planes, Math.max(1, Math.round(1 + luma * 2))) : planes;
    const chromaBase = chroma > 0 ? this.planeBlur(planes, Math.max(1, Math.round(2 + chroma * 10))) : planes;
    const out = this.pass("denoise", DENOISE_FS, this.gl.target(width, height), {
      source: source.texture, planes: planes.texture, lumaBase: lumaBase.texture, chromaBase: chromaBase.texture,
    }, { luma, chroma, keep });
    if (lumaBase !== planes) this.gl.release(lumaBase);
    if (chromaBase !== planes) this.gl.release(chromaBase);
    this.gl.release(planes);
    return out;
  }

  /* ------------------------------------------------------------ time (VFX Phase 3) */

  private filledRgba(width: number, height: number): Target {
    return this.pass("fill", FILL_FS, this.gl.target(width, height), {}, { value: [0, 0, 0, 0] });
  }

  /** `renderMatrixRain`: the glyph atlas is drawn once per cell size on the CPU and kept as a texture. */
  private matrixRain(generator: MatrixRainGenerator, width: number, height: number, frame: number, canvasScale: number): Target {
    const l = rainLayout(generator, width, height, canvasScale);
    const atlas = buildRainAtlas(l.cell, generator.mirror);
    let texture = this.atlases.get(atlas);
    if (!texture) {
      if (this.atlases.size > 4) { for (const held of this.atlases.values()) this.gl.deleteTexture(held); this.atlases.clear(); }
      texture = this.gl.uploadR8(atlas.coverage, atlas.width, atlas.cell);
      this.atlases.set(atlas, texture);
    }
    return this.pass("matrixRain", MATRIX_RAIN_FS, this.gl.target(width, height), { atlas: texture }, {
      seed: { int: l.seed | 0 }, cell: { int: l.cell }, rows: { int: l.rows }, loopLength: { int: l.loop },
      trail: { int: l.trail }, minPasses: { int: l.minPasses }, maxPasses: { int: l.maxPasses },
      changes: { int: l.changes }, density: { int: l.density },
      inLoop: { int: ((frame % l.loop) + l.loop) % l.loop }, glyphCount: { int: atlas.count },
      colour: { ints: hexBytes(generator.colour) }, headColour: { ints: hexBytes(generator.headColour) },
    });
  }

  /* ------------------------------------------------------------ emitter (VFX Phase 4) */

  /**
   * `renderEmitter`: each particle placed like a layer and composited Normal at its opacity, oldest
   * first, into a transparent target. One place + one composite per particle (correctness first).
   */
  private emitter(
    particles: Particle[], element: LayerMedia, sources: Map<string, LayerSource>,
    width: number, height: number, canvasScale: number, useProxy: boolean | undefined, options: RendererOptions,
  ): Target | undefined {
    let accumulated = this.filledRgba(width, height);
    for (const particle of particles) {
      if (particle.opacity <= 0 || particle.scale <= 0) continue;
      const source = sources.get(particleKey(element, particle, useProxy));
      if (!source) {
        this.gl.release(accumulated);
        return undefined;
      }
      const placed = this.place(particleLayer(particle), source, width, height, canvasScale, options);
      const next = this.composite(accumulated, placed, "normal", particle.opacity);
      this.gl.release(accumulated);
      this.gl.release(placed);
      accumulated = next;
    }
    return accumulated;
  }

  /* ------------------------------------------------------------ masks (VFX Phase 2) */

  /**
   * Every mask the scope reads (`wantedMasks`), rasterised on the GPU: `rasterizeMaskStack`. `scope`
   * is the scope's ungraded pixels, which a colour range samples directly -- no readback. Returned
   * targets belong to the caller (`releaseAll`).
   */
  private masks(
    scope: ComposeScope, scopePixels: Target, resources: RenderResources,
    layerTarget: (layerId: string) => { target: Target; opacity: number } | undefined,
  ): Map<string, Target> {
    const out = new Map<string, Target>();
    for (const mask of wantedMasks(scope)) {
      out.set(mask.id, this.maskStack(mask, scope, scopePixels, resources, layerTarget));
    }
    return out;
  }

  private releaseAll(masks: Map<string, Target>): void {
    for (const target of masks.values()) this.gl.release(target);
    masks.clear();
  }

  private readCoverage(target: Target): Uint8Array {
    const rgba = this.read(target);
    const coverage = new Uint8Array(target.width * target.height);
    for (let index = 0; index < coverage.length; index++) coverage[index] = rgba[index * 4];
    return coverage;
  }

  private filled(width: number, height: number, value: number): Target {
    return this.pass("fill", FILL_FS, this.gl.target(width, height), {}, { value: [value, value, value, 255] });
  }

  /** A placed matte file (bytes) as a coverage target, or undefined when it is not loaded. */
  private matteTarget(coverage: Uint8Array | undefined, width: number, height: number): Target | undefined {
    if (!coverage || coverage.length !== width * height) return undefined;
    const texture = this.temporary(this.gl.uploadR8(coverage, width, height));
    return this.pass("coverageCopy", COVERAGE_COPY_FS, this.gl.target(width, height), { source: texture });
  }

  private maskStack(
    mask: PhotoMask, scope: ComposeScope, scopePixels: Target, resources: RenderResources,
    layerTarget: (layerId: string) => { target: Target; opacity: number } | undefined,
  ): Target {
    const { width, height } = scopePixels;
    if (!mask.enabled) return this.filled(width, height, 255);
    // The base: exactly `rasterizeMask`.
    const legacy = mask.sourcePath ? resources.mattes?.get(legacyMatteKey(mask.sourcePath)) : undefined;
    let accumulated = this.matteTarget(legacy, width, height) ?? this.filled(width, height, 0);
    if (mask.shape) this.shapeInto(accumulated, mask.shape, scopePixels);
    this.strokesInto(accumulated, mask.strokes);
    if (mask.feather > 0) accumulated = this.feather(accumulated, Math.round(mask.feather * (scope.canvasScale ?? 1)));
    if (mask.invert) accumulated = this.invertCoverage(accumulated);

    for (const entry of mask.stack ?? []) {
      if (!entry.enabled) continue;
      const own = this.entry(entry, scope, scopePixels, resources, layerTarget);
      if (!own) continue;
      const combined = this.pass("maskCombine", MASK_COMBINE_FS, this.gl.target(width, height),
        { accumulated: accumulated.texture, entry: own.texture },
        { mode: { int: MASK_MODE_INDEX[entry.mode] ?? 0 }, strength: Math.min(1, Math.max(0, entry.opacity)) });
      this.gl.release(accumulated);
      this.gl.release(own);
      accumulated = combined;
    }
    return accumulated;
  }

  /** `entryCoverage`: source, strokes, expand/choke, feather, invert. */
  private entry(
    entry: MaskEntry, scope: ComposeScope, scopePixels: Target, resources: RenderResources,
    layerTarget: (layerId: string) => { target: Target; opacity: number } | undefined,
  ): Target | undefined {
    const { width, height } = scopePixels;
    const source = entry.source;
    let coverage: Target;
    if (source.kind === "paint") coverage = this.filled(width, height, 0);
    else if (source.kind === "shape") {
      coverage = this.filled(width, height, 0);
      this.shapeInto(coverage, source.shape, scopePixels);
    } else if (source.kind === "layer") {
      const held = layerTarget(source.layerId);
      if (!held) return undefined;
      coverage = this.pass("layerCoverage", LAYER_COVERAGE_FS, this.gl.target(width, height), { source: held.target.texture },
        { luma: { int: source.channel === "luma" ? 1 : 0 }, strength: Math.min(1, Math.max(0, held.opacity)) });
    } else if (source.kind === "path") {
      const held = this.matteTarget(resources.mattes?.get(pathMatteKey(source.path)), width, height);
      if (!held) return undefined;
      coverage = held;
    } else if (source.kind === "depth") {
      const held = this.matteTarget(resources.mattes?.get(depthKey(source.layerId)), width, height);
      if (!held) return undefined;
      const lut = this.temporary(this.gl.uploadR8(depthThresholdLut(source.plane, source.softness), 256, 1));
      coverage = this.pass("depthThreshold", DEPTH_THRESHOLD_FS, this.gl.target(width, height), { source: held.texture, lut });
      this.gl.release(held);
    } else {
      const request = matteRequest(source, scope.frame);
      const held = this.matteTarget(request ? resources.mattes?.get(matteKey(request)) : undefined, width, height);
      if (!held) return undefined;
      coverage = held;
    }
    this.strokesInto(coverage, entry.strokes);
    const scale = scope.canvasScale ?? 1;
    const expand = Math.round((entry.expand || 0) * scale);
    if (expand !== 0) coverage = this.morph(coverage, expand);
    const feather = Math.round((entry.feather || 0) * scale);
    if (feather >= 1) coverage = this.feather(coverage, feather);
    if (entry.invert) coverage = this.invertCoverage(coverage);
    return coverage;
  }

  /** `shapeCoverage`, blended into `target` with MAX. */
  private shapeInto(target: Target, shape: MaskShape, scopePixels: Target): void {
    const { width, height } = target;
    const uniforms: Parameters<GlContext["draw"]>[4] = {};
    if (shape.kind === "linear") {
      const ax = shape.x1 * width, ay = shape.y1 * height;
      const dx = shape.x2 * width - ax, dy = shape.y2 * height - ay;
      const lengthSquared = dx * dx + dy * dy;
      Object.assign(uniforms, {
        kind: { int: 0 }, a: [ax, ay], delta: [dx, dy], lengthSquared: lengthSquared || 1,
        everything: { int: lengthSquared < 1e-6 ? 1 : 0 },
      });
    } else if (shape.kind === "radial") {
      const long = Math.max(width, height);
      const aspect = shape.aspect || 1;
      Object.assign(uniforms, {
        kind: { int: 1 }, a: [shape.cx * width, shape.cy * height],
        radii: [Math.max(1, shape.radius * long * aspect), Math.max(1, shape.radius * long / aspect)],
        solid: 1 - Math.min(0.999, Math.max(0, shape.feather)),
      });
    } else {
      Object.assign(uniforms, {
        kind: { int: 2 }, colour: [shape.r, shape.g, shape.b],
        tolerance: Math.max(1, shape.tolerance) * 4.42, softness: Math.max(0.01, shape.softness),
      });
    }
    this.gl.draw("maskShape", MASK_SHAPE_FS, target, { scope: scopePixels.texture }, uniforms, { blend: "max" });
    this.stats.passes++;
  }

  /** `stampStroke` for every stroke, in order: one instanced draw per stroke, MAX to paint, MIN to erase. */
  private strokesInto(target: Target, strokes: MaskStroke[]): void {
    const { width, height } = target;
    const longEdge = Math.max(width, height);
    for (const stroke of strokes) {
      const dab = dabShape(stroke, longEdge);
      const instances: number[] = [];
      forEachDab(stroke, width, height, longEdge, (x, y) => {
        const { left, right, top, bottom } = dabBounds(x, y, dab.radius, width, height);
        if (right < left || bottom < top) return;
        instances.push(left, top, right - left + 1, bottom - top + 1, x - left, y - top);
      });
      if (!instances.length) continue;
      this.gl.drawInstances("dab", DAB_VS, DAB_FS, target, {
        size: [width, height], radius: dab.radius, solid: dab.solid, fade: dab.fade, erase: { int: stroke.erase ? 1 : 0 },
      }, new Float32Array(instances), stroke.erase ? "min" : "max");
      this.stats.passes++;
    }
  }

  /** `featherCoverage`: three rounds of the 8-bit box, as BOX_FS already does for the grade's blur. */
  private feather(coverage: Target, radius: number): Target {
    if (radius < 1) return coverage;
    const out = this.boxBlur(coverage, radius);
    this.gl.release(coverage);
    return out;
  }

  /** `morphCoverage`: rows then columns. */
  private morph(coverage: Target, radius: number): Target {
    let current = coverage;
    for (const direction of [[1, 0], [0, 1]]) {
      const next = this.pass("morph", MORPH_FS, this.gl.target(coverage.width, coverage.height), { source: current.texture },
        { direction: { ints: direction }, radius: { int: Math.abs(radius) }, grow: { int: radius > 0 ? 1 : 0 } });
      this.gl.release(current);
      current = next;
    }
    return current;
  }

  private invertCoverage(coverage: Target): Target {
    const out = this.pass("maskInvert", MASK_INVERT_FS, this.gl.target(coverage.width, coverage.height), { source: coverage.texture });
    this.gl.release(coverage);
    return out;
  }

  /* ------------------------------------------------------------ constant tables */

  private toLinearTexture(): WebGLTexture {
    this.toLinear ??= this.gl.uploadR32F(srgbToLinear, 256, 1);
    return this.toLinear;
  }

  /**
   * The browser's own premultiplied-to-straight conversion, measured once: every (value, alpha) pair
   * written with putImageData and read back with getImageData. Canvas premultiplies as
   * round(v * a / 255) (which the WebGL upload reproduces exactly) and its unpremultiply rounds in
   * its own way at about 1.5% of pairs; a table is the only way to match it everywhere, including in
   * WebView2, whose Skia may round differently again.
   */
  private unpremultiplyTexture(): WebGLTexture {
    if (this.unpremultiply) return this.unpremultiply;
    const table = new Uint8Array(256 * 256);
    const canvas = window.document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const context = canvas.getContext("2d", { willReadFrequently: true })!;
    const image = context.createImageData(256, 256);
    for (let a = 0; a < 256; a++) {
      for (let v = 0; v < 256; v++) {
        const at = (a * 256 + v) * 4;
        image.data[at] = v;
        image.data[at + 3] = a;
      }
    }
    context.putImageData(image, 0, 0);
    const back = context.getImageData(0, 0, 256, 256).data;
    for (let a = 1; a < 256; a++) {
      for (let v = 0; v < 256; v++) {
        const premultiplied = Math.floor((v * a + 128 + ((v * a + 128) >> 8)) >> 8);
        table[a * 256 + premultiplied] = back[(a * 256 + v) * 4];
      }
    }
    this.unpremultiply = this.gl.uploadR8(table, 256, 256);
    return this.unpremultiply;
  }

  private bayerTexture(): WebGLTexture {
    this.bayer ??= this.gl.uploadR32F(BAYER, 8, 8);
    return this.bayer;
  }

  /** `addGrain`'s field, computed in float64 exactly as the CPU does, once per size. */
  private noiseTexture(width: number, height: number): WebGLTexture {
    const key = `${width}x${height}`;
    const held = this.noise.get(key);
    if (held) return held;
    const field = new Float32Array(width * height);
    for (let pixel = 0; pixel < field.length; pixel++) {
      const x = pixel % width, y = (pixel / width) | 0;
      field[pixel] = (Math.sin(x * 12.9898 + y * 78.233) * 43758.5453) % 1;
    }
    // One size at a time, for the same reason as the target pool: a full-size field is 4 bytes a pixel.
    for (const texture of this.noise.values()) this.gl.deleteTexture(texture);
    this.noise.clear();
    const texture = this.gl.uploadR32F(field, width, height);
    this.noise.set(key, texture);
    return texture;
  }
}

