/**
 * Frames for media layers, fetched ahead of a render (VFX Phase 3).
 *
 * The compositors are synchronous and look pictures up by key (`mediaFrameKey`), so decoding happens
 * here, before: `load` resolves once the frame is in the cache as a canvas, sized to the layer's
 * media size so placement is identical whichever file it came from.
 *
 * - "frames" layers (element masters): each PNG through the caller's image loader.
 * - "video" layers: mp4box.js + WebCodecs through `DecodeWindow`, which keeps a window of
 *   `lookahead` frames after the one asked for (plus the frames its ahead-feed produces), and keeps
 *   decoding forward from where it stopped instead of replaying the GOP, so scrubbing forward,
 *   playing and rendering are mostly cache hits and decode each sample about once. Frames are drawn
 *   into canvases on arrival and closed at once.
 * - Proxies: `useProxy` reads `proxyPath` instead. A stacked element proxy (colour left, alpha as
 *   luma right, BT.709 limited range, tagged) is recombined into RGBA here; the browser has already
 *   expanded the range, so the right half's red channel is the alpha.
 *
 * Bounded by bytes (LRU). `onEvict` lets the owner release a GPU texture made from an evicted canvas.
 */

import type { LayerMedia } from "../../domain/photoDocument";
import { mediaFrameKey } from "../../domain/photoTime";
import { DecodeWindow, demux, supportedConfig } from "./demux";

export type FramePicture = { canvas: HTMLCanvasElement; width: number; height: number };

export type FrameStoreOptions = {
  readBytes: (path: string) => Promise<ArrayBuffer>;
  loadImage: (path: string) => Promise<FramePicture>;
  /** Default 768 MB. */
  budgetBytes?: number;
  /** Frames decoded after the one asked for. Default 8. */
  lookahead?: number;
};

export class MediaFrameStore {
  private readonly cache = new Map<string, FramePicture>();
  private bytes = 0;
  private readonly pending = new Map<string, Promise<FramePicture>>();
  private readonly decoders = new Map<string, Promise<DecodeWindow>>();
  private readonly queues = new Map<string, Promise<unknown>>();
  /** Keys that must survive eviction (what is on screen now). */
  private pinned = new Set<string>();
  onEvict?: (picture: FramePicture) => void;
  stats = { decodedFrames: 0, loadedImages: 0, windows: 0 };

  /** Samples handed to the video decoders and decoder resets so far (the render's timing line). */
  decoderStats(): { fedSamples: number; resets: number } {
    const totals = { fedSamples: 0, resets: 0 };
    for (const stream of this.opened) { totals.fedSamples += stream.fedSamples; totals.resets += stream.resets; }
    return totals;
  }

  private readonly opened = new Set<DecodeWindow>();

  constructor(private readonly options: FrameStoreOptions) {}

  get(key: string): FramePicture | undefined {
    const held = this.cache.get(key);
    if (held) { this.cache.delete(key); this.cache.set(key, held); }
    return held;
  }

  /** Everything cached, for building the compositor's `sources` map. */
  entries(): IterableIterator<[string, FramePicture]> { return this.cache.entries(); }

  pin(keys: Iterable<string>): void { this.pinned = new Set(keys); }

  load(media: LayerMedia, index: number, useProxy = false): Promise<FramePicture> {
    const key = mediaFrameKey(media, index, useProxy);
    const held = this.get(key);
    if (held) return Promise.resolve(held);
    const waiting = this.pending.get(key);
    if (waiting) return waiting;
    const proxy = useProxy && media.proxyPath ? media.proxyPath : undefined;
    const job = (media.kind === "frames" && !proxy
      ? this.options.loadImage(key).then((picture) => { this.stats.loadedImages++; return this.put(key, picture); })
      : this.decode(media, index, proxy ?? media.path, Boolean(proxy && media.proxyStacked), useProxy).catch((error) => {
        // A master this WebView cannot decode (HEVC) falls back to its H.264 proxy, stored under the
        // master's keys, so the layer still shows. The proxy is scaled to the media size.
        if (proxy || !media.proxyPath || media.proxyStacked || !/cannot be decoded/.test(String(error))) throw error;
        this.fallbacks.add(media.path);
        return this.decode(media, index, media.proxyPath, false, false);
      })
    ).finally(() => this.pending.delete(key));
    this.pending.set(key, job);
    return job;
  }

  /** Close every decoder and drop every frame. */
  dispose(): void {
    for (const decoder of this.decoders.values()) void decoder.then((held) => held.close(), () => undefined);
    this.opened.clear();
    this.decoders.clear();
    for (const picture of this.cache.values()) this.onEvict?.(picture);
    this.cache.clear();
    this.bytes = 0;
  }

  private put(key: string, picture: FramePicture): FramePicture {
    if (this.cache.has(key)) return this.cache.get(key)!;
    this.cache.set(key, picture);
    this.bytes += picture.width * picture.height * 4;
    const budget = this.options.budgetBytes ?? 768 * 1024 ** 2;
    for (const [oldKey, old] of this.cache) {
      if (this.bytes <= budget) break;
      if (oldKey === key || this.pinned.has(oldKey)) continue;
      this.cache.delete(oldKey);
      this.bytes -= old.width * old.height * 4;
      this.onEvict?.(old);
    }
    return picture;
  }

  private decoder(path: string): Promise<DecodeWindow> {
    let held = this.decoders.get(path);
    if (!held) {
      held = (async () => {
        const media = await demux(await this.options.readBytes(path));
        const config = await supportedConfig(media);
        if (!config) throw new Error(`${media.codec} cannot be decoded here${media.codec.startsWith("hvc") || media.codec.startsWith("hev") ? " (HEVC: use an H.264 proxy)" : ""}`);
        const stream = new DecodeWindow(media, config);
        this.opened.add(stream);
        return stream;
      })();
      held.catch(() => this.decoders.delete(path));
      this.decoders.set(path, held);
    }
    return held;
  }

  /** Masters that could not be decoded and are shown from their proxy. */
  readonly fallbacks = new Set<string>();

  /**
   * One file decodes one window at a time; windows for different files run side by side. Frames are
   * stored under `mediaFrameKey(media, i, keyProxy)`.
   */
  private decode(media: LayerMedia, index: number, path: string, stacked: boolean, useProxy: boolean): Promise<FramePicture> {
    const previous = this.queues.get(path) ?? Promise.resolve();
    const run = previous.catch(() => undefined).then(async () => {
      const key = mediaFrameKey(media, index, useProxy);
      const already = this.cache.get(key);
      if (already) return already;
      const window = await this.decoder(path);
      const lookahead = this.options.lookahead ?? 8;
      this.stats.windows++;
      await window.decode(index, index + lookahead, (at, frame) => {
        const atKey = mediaFrameKey(media, at, useProxy);
        if (this.cache.has(atKey)) return;
        this.stats.decodedFrames++;
        this.put(atKey, toPicture(frame, media.width, media.height, stacked));
      });
      const picture = this.cache.get(key);
      if (!picture) throw new Error(`frame ${index} of ${path} was not produced`);
      return picture;
    });
    this.queues.set(path, run);
    return run;
  }
}

let stackScratch: HTMLCanvasElement | undefined;

/** A decoded frame as a canvas of the layer's size. Called while the frame is open; it is closed after. */
export function toPicture(frame: VideoFrame, width: number, height: number, stacked: boolean): FramePicture {
  const canvas = window.document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  if (!stacked) {
    context.drawImage(frame, 0, 0, width, height);
    return { canvas, width, height };
  }
  // Colour | alpha side by side: draw both halves at the element's size, then move the right half's
  // red into the left half's alpha. The browser has converted from limited range already.
  stackScratch ??= window.document.createElement("canvas");
  if (stackScratch.width !== width * 2 || stackScratch.height !== height) {
    stackScratch.width = width * 2;
    stackScratch.height = height;
  }
  const scratch = stackScratch.getContext("2d", { willReadFrequently: true })!;
  scratch.drawImage(frame, 0, 0, width * 2, height);
  const both = scratch.getImageData(0, 0, width * 2, height).data;
  const out = context.createImageData(width, height);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const to = (y * width + x) * 4, colour = (y * width * 2 + x) * 4, alpha = (y * width * 2 + width + x) * 4;
      out.data[to] = both[colour];
      out.data[to + 1] = both[colour + 1];
      out.data[to + 2] = both[colour + 2];
      out.data[to + 3] = both[alpha];
    }
  }
  context.putImageData(out, 0, 0);
  return { canvas, width, height };
}
