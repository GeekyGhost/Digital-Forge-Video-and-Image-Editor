/**
 * VFX Phase 3 frames spike: the real frame store (src/render/video/frameStore.ts) on the Phase 0
 * test media, and 1500 px composition speed with element layers.
 *
 * - video: seek exactness (every target read back from the frame-index strip), and sequential
 *   playback fps with decode-ahead;
 * - elements: PNG master load fps, stacked-proxy load fps and its alpha error against the master;
 * - compose: N element layers (+ matrix rain) over a video plate at 1500x1000, frames preloaded,
 *   GPU and CPU ms per frame.
 */
import type { Backend, ClipInfo, MediaSet } from "./backend";
import { readIndexStrip, round } from "./gl";
import { MediaFrameStore, type FramePicture } from "../../src/render/video/frameStore";
import {
  createMatrixRainLayer, createMediaLayer, createPhotoDocument, type LayerMedia, type PhotoDocument, type PhotoLayer,
} from "../../src/domain/photoDocument";
import { mediaFrameKey, mediaFramesAt, compositionRate } from "../../src/domain/photoTime";
import { composeDocument, type LayerSource } from "../../src/domain/photoCompose";
import { emptyResources } from "../../src/domain/photoRender";
import { PhotoRenderer } from "../../src/render/photoRenderer";

function storeFor(backend: Backend, lookahead: number): MediaFrameStore {
  return new MediaFrameStore({
    readBytes: (path) => backend.readFile(path),
    loadImage: async (path) => {
      const bitmap = await createImageBitmap(new Blob([await backend.readFile(path)], { type: "image/png" }), { premultiplyAlpha: "none", colorSpaceConversion: "none" });
      const canvas = document.createElement("canvas");
      canvas.width = bitmap.width;
      canvas.height = bitmap.height;
      canvas.getContext("2d")!.drawImage(bitmap, 0, 0);
      bitmap.close();
      return { canvas, width: canvas.width, height: canvas.height };
    },
    lookahead,
  });
}

const videoMedia = (clip: ClipInfo): LayerMedia => ({
  kind: "video", path: clip.path, frameCount: clip.frames, fps: clip.fps, width: clip.width, height: clip.height,
  start: 0, inPoint: 0, outPoint: clip.frames, loop: false, crossfade: 0,
});

function stripOf(picture: FramePicture): number {
  const data = picture.canvas.getContext("2d", { willReadFrequently: true })!.getImageData(0, 0, 256, 32).data;
  return readIndexStrip(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 256);
}

async function videoClip(backend: Backend, clip: ClipInfo, log: (line: string) => void) {
  const result: Record<string, unknown> = { id: clip.id };
  if (clip.error) return { ...result, error: clip.error };
  const media = videoMedia(clip);
  try {
    const seeks = storeFor(backend, 0);
    const targets = [0, 1, 2, 23, 24, 25, 47, 48, 49, 70, clip.frames - 1, 5, 60, 30, 13, clip.frames - 2].filter((n) => n >= 0 && n < clip.frames);
    let exact = 0;
    const misses: string[] = [];
    const times: number[] = [];
    for (const target of targets) {
      const started = performance.now();
      const picture = await seeks.load(media, target);
      times.push(performance.now() - started);
      const read = stripOf(picture);
      if (read === target % 256) exact++; else misses.push(`${target}->${read}`);
    }
    seeks.dispose();
    result.seek = { targets: targets.length, exact, misses, meanMs: round(times.reduce((a, b) => a + b, 0) / times.length, 1), maxMs: round(Math.max(...times), 1) };
    log(`${clip.id}: seek ${exact}/${targets.length} exact`);

    const play = storeFor(backend, 8);
    const started = performance.now();
    let order = 0;
    for (let frame = 0; frame < clip.frames; frame++) {
      const picture = await play.load(media, frame);
      if (stripOf(picture) === frame % 256) order++;
    }
    const seconds = (performance.now() - started) / 1000;
    result.playback = { frames: clip.frames, inOrder: order, fps: round(clip.frames / seconds, 1), windows: play.stats.windows, decoded: play.stats.decodedFrames };
    log(`${clip.id}: playback ${round(clip.frames / seconds, 1)} fps, ${order}/${clip.frames} in order`);
    play.dispose();
  } catch (error) {
    result.error = String(error);
    log(`${clip.id}: ${error}`);
  }
  return result;
}

export async function runFramesSpike(backend: Backend, set: MediaSet, log: (line: string) => void) {
  const video: Array<Record<string, unknown>> = [];
  for (const clip of set.clips) video.push(await videoClip(backend, clip, log));

  const a = set.alpha;
  const directory = a.pngPaths[0].replace(/[\\/][^\\/]*$/, "");
  const element: LayerMedia = {
    kind: "frames", path: directory, pattern: "frame_%04d.png", frameCount: a.frames, fps: a.fps, width: a.width, height: a.height,
    start: 0, inPoint: 0, outPoint: a.frames, loop: true, crossfade: 8,
    proxyPath: a.stackedPath ?? undefined, proxyStacked: true,
  };
  const elements: Record<string, unknown> = {};
  const store = storeFor(backend, 8);
  try {
    let started = performance.now();
    for (let frame = 0; frame < a.frames; frame++) await store.load(element, frame);
    elements.masterFps = round(a.frames / ((performance.now() - started) / 1000), 1);
    log(`elements: PNG master ${elements.masterFps} fps`);
    if (element.proxyPath) {
      started = performance.now();
      for (let frame = 0; frame < a.frames; frame++) await store.load(element, frame, true);
      elements.proxyFps = round(a.frames / ((performance.now() - started) / 1000), 1);
      let sum = 0, max = 0, count = 0;
      for (const frame of [0, 40]) {
        const master = store.get(mediaFrameKey(element, frame))!.canvas.getContext("2d")!.getImageData(0, 0, a.width, a.height).data;
        const proxy = store.get(mediaFrameKey(element, frame, true))!.canvas.getContext("2d")!.getImageData(0, 0, a.width, a.height).data;
        for (let at = 3; at < master.length; at += 4) { const d = Math.abs(master[at] - proxy[at]); sum += d; max = Math.max(max, d); count++; }
      }
      elements.proxyAlphaMae = round(sum / count, 3);
      elements.proxyAlphaMax = max;
      log(`elements: stacked proxy ${elements.proxyFps} fps, alpha error ${elements.proxyAlphaMae} (max ${max})`);
    }
  } catch (error) {
    elements.error = String(error);
    log(`elements: ${error}`);
  }

  // Compose: a video plate (the first decodable clip) + 3 elements + rain at 1500x1000.
  const compose: Record<string, unknown> = {};
  try {
    const plateClip = set.clips.find((clip) => !clip.error && (video.find((v) => v.id === clip.id) as { error?: string })?.error === undefined);
    const layers: PhotoLayer[] = [];
    if (plateClip) layers.push(createMediaLayer("Plate", videoMedia(plateClip), { scale: 1500 / plateClip.width }));
    for (let n = 0; n < 3; n++) {
      layers.push(createMediaLayer(`Element ${n}`, { ...element, start: n * 7, proxyPath: undefined }, { blendMode: "screen", x: 0.25 + n * 0.25, y: 0.55, scale: 0.8 + n * 0.2 }));
    }
    layers.push(createMatrixRainLayer({ seed: 5 }));
    const document: PhotoDocument = {
      ...createPhotoDocument("spike", ""), nodes: [], masks: [], layers,
      canvasWidth: 1500, canvasHeight: 1000, background: "black", frameRate: plateClip?.fps ?? a.fps,
    };
    const frames = Math.min(48, plateClip?.frames ?? a.frames);
    const play = storeFor(backend, 8);
    const sourcesAt = async (frame: number) => {
      const sources = new Map<string, LayerSource>();
      for (const layer of layers) {
        if (!layer.media) continue;
        for (const shown of mediaFramesAt(layer.media, frame, compositionRate(document)) ?? []) {
          const picture = layer.media.kind === "frames" ? await store.load(layer.media, shown.index) : await play.load(layer.media, shown.index);
          sources.set(mediaFrameKey(layer.media, shown.index), { image: picture.canvas, width: picture.width, height: picture.height });
        }
      }
      return sources;
    };
    const renderer = new PhotoRenderer();
    let fetchMs = 0, gpuMs = 0, cpuMs = 0, cpuFrames = 0;
    for (let frame = 0; frame < frames; frame++) {
      let started = performance.now();
      const sources = await sourcesAt(frame);
      fetchMs += performance.now() - started;
      const input = { document, width: 1500, height: 1000, sources, frame, resourcesFor: () => emptyResources() };
      started = performance.now();
      renderer.compose(input);
      gpuMs += performance.now() - started;
      if (frame % 6 === 0) {
        started = performance.now();
        composeDocument(input);
        cpuMs += performance.now() - started;
        cpuFrames++;
      }
    }
    renderer.dispose();
    play.dispose();
    Object.assign(compose, {
      size: "1500x1000", frames, layers: layers.map((layer) => layer.name),
      fetchMsPerFrame: round(fetchMs / frames, 1), gpuMsPerFrame: round(gpuMs / frames, 1), cpuMsPerFrame: round(cpuMs / Math.max(1, cpuFrames), 1),
      gpuFps: round(1000 / (gpuMs / frames), 1),
    });
    log(`compose 1500x1000, ${layers.length} layers: fetch ${compose.fetchMsPerFrame} ms, GPU ${compose.gpuMsPerFrame} ms, CPU ${compose.cpuMsPerFrame} ms per frame`);
  } catch (error) {
    compose.error = String((error as Error)?.stack ?? error);
    log(`compose: ${error}`);
  }
  store.dispose();
  return { video, elements, compose };
}
