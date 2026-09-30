/**
 * MP4 demux with mp4box.js into WebCodecs chunks, plus a frame-accurate seek.
 *
 * Moved from spikes/src/demux.ts in VFX Phase 3 (the spike re-exports this), unchanged in behaviour:
 * the Phase 0 rules hold -- every non-target frame is closed the moment it arrives (holding them
 * exhausts a hardware decoder's surface pool and flush() never resolves), and every flush or queue
 * wait has a timeout. `DecodeWindow` below streams forward for scrubbing, playback and renders.
 */
import { createFile, DataStream, Endianness, MP4BoxBuffer } from "./vendor/mp4box/mp4box.all.mjs";

export type DemuxedSample = {
  /** Decode order. */
  index: number;
  isSync: boolean;
  /** Presentation timestamp in microseconds (used as the chunk timestamp). */
  timestampUs: number;
  durationUs: number;
  data: Uint8Array;
};

export type Demuxed = {
  codec: string;
  width: number;
  height: number;
  description?: Uint8Array;
  /** Decode order. */
  samples: DemuxedSample[];
  /** presentation[N] = decode-order index of the Nth displayed frame. */
  presentation: number[];
  fps: number;
};

export async function demux(buffer: ArrayBuffer): Promise<Demuxed> {
  const file = createFile();
  let trackId = -1;
  let info: { codec: string; width: number; height: number; timescale: number; nbSamples: number } | undefined;
  const raw: { is_sync: boolean; cts: number; duration: number; timescale: number; data?: Uint8Array }[] = [];
  let failure: string | undefined;
  file.onError = (module: string, message: string) => { failure = `${module}: ${message}`; };
  file.onReady = (movie) => {
    const track = movie.videoTracks[0];
    if (!track) { failure = "no video track"; return; }
    trackId = track.id;
    info = { codec: track.codec, width: track.video?.width ?? track.track_width, height: track.video?.height ?? track.track_height, timescale: track.timescale, nbSamples: track.nb_samples };
    file.setExtractionOptions(track.id, undefined, { nbSamples: Number.MAX_SAFE_INTEGER });
    file.start();
  };
  file.onSamples = (_id, _user, samples) => {
    for (const sample of samples) raw.push({ is_sync: sample.is_sync, cts: sample.cts, duration: sample.duration, timescale: sample.timescale, data: sample.data });
  };
  file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(buffer, 0));
  file.flush();
  if (failure) throw new Error(failure);
  if (!info || trackId < 0) throw new Error("mp4box found no video track");

  const entry = file.getTrackById(trackId).mdia.minf.stbl.stsd.entries[0] as unknown as Record<string, { write(stream: unknown): void } | undefined>;
  const box = entry.avcC ?? entry.hvcC ?? entry.vpcC ?? entry.av1C;
  let description: Uint8Array | undefined;
  if (box) {
    const stream = new DataStream(undefined, 0, Endianness.BIG_ENDIAN);
    box.write(stream);
    description = new Uint8Array(stream.buffer as ArrayBuffer, 8); // drop the box header
  }

  const samples: DemuxedSample[] = raw.map((sample, index) => ({
    index,
    isSync: sample.is_sync,
    timestampUs: Math.round((sample.cts * 1e6) / sample.timescale),
    durationUs: Math.round((sample.duration * 1e6) / sample.timescale),
    data: sample.data!,
  }));
  const presentation = samples.map((s) => s.index).sort((a, b) => samples[a].timestampUs - samples[b].timestampUs);
  const meanDuration = samples.reduce((sum, s) => sum + s.durationUs, 0) / Math.max(1, samples.length);
  return { codec: info.codec, width: info.width, height: info.height, description, samples, presentation, fps: 1e6 / meanDuration };
}

export function decoderConfig(media: Demuxed, hardware: HardwareAcceleration = "prefer-hardware"): VideoDecoderConfig {
  return {
    codec: media.codec,
    codedWidth: media.width,
    codedHeight: media.height,
    description: media.description,
    hardwareAcceleration: hardware,
    optimizeForLatency: true,
  };
}

function chunk(sample: DemuxedSample): EncodedVideoChunk {
  return new EncodedVideoChunk({ type: sample.isSync ? "key" : "delta", timestamp: sample.timestampUs, duration: sample.durationUs, data: sample.data });
}

async function backpressure(decoder: VideoDecoder, limit = 8) {
  const started = performance.now();
  while (decoder.decodeQueueSize > limit) {
    if (performance.now() - started > 20000) throw new Error("decoder stopped draining its queue for 20 s");
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

/** Decodes every frame once, as fast as possible, calling `onFrame` then closing each frame. */
export async function decodeAll(media: Demuxed, config: VideoDecoderConfig, onFrame?: (frame: VideoFrame) => void): Promise<{ frames: number; seconds: number }> {
  let frames = 0;
  let error: unknown;
  const decoder = new VideoDecoder({
    output: (frame) => { frames++; try { onFrame?.(frame); } finally { frame.close(); } },
    error: (e) => { error = e; },
  });
  decoder.configure(config);
  const started = performance.now();
  for (const sample of media.samples) {
    if (error) break;
    await backpressure(decoder);
    decoder.decode(chunk(sample));
  }
  if (!error) await withTimeout(decoder.flush(), 60000, "decode flush");
  const seconds = (performance.now() - started) / 1000;
  decoder.close();
  if (error) throw error;
  return { frames, seconds };
}

/**
 * Frame-accurate seek: decode from the last keyframe at or before frame N (in decode order) up to
 * N itself, flush, and keep the output whose timestamp is N's. Everything else is closed.
 */
export class Seeker {
  private decoder: VideoDecoder;
  private wanted = Number.NaN;
  private found: VideoFrame | undefined;
  private error: unknown;

  constructor(private media: Demuxed, private config: VideoDecoderConfig) {
    // Every frame that is not the target is closed the moment it arrives. Hardware decoders own a
    // small fixed pool of output surfaces; holding frames until flush() (the first version of this
    // spike) exhausts the pool and flush() never resolves -- a deadlock on the first seek past a
    // keyframe. Software decoders have no such pool, which is why the container harness passed.
    this.decoder = new VideoDecoder({
      output: (frame) => {
        if (!this.found && frame.timestamp === this.wanted) this.found = frame;
        else frame.close();
      },
      error: (e) => { this.error = e; },
    });
  }

  async frame(presentationIndex: number): Promise<VideoFrame> {
    const target = this.media.samples[this.media.presentation[presentationIndex]];
    let start = target.index;
    while (start > 0 && !this.media.samples[start].isSync) start--;
    this.found?.close();
    this.found = undefined;
    this.wanted = target.timestampUs;
    this.decoder.reset();
    this.decoder.configure(this.config);
    for (let i = start; i <= target.index; i++) {
      await backpressure(this.decoder);
      this.decoder.decode(chunk(this.media.samples[i]));
    }
    await withTimeout(this.decoder.flush(), 20000, `seek to frame ${presentationIndex}`);
    if (this.error) throw this.error;
    const found = this.found;
    this.found = undefined;
    if (!found) throw new Error(`frame ${presentationIndex} (ts ${target.timestampUs}) was not produced`);
    return found;
  }

  close() {
    this.found?.close();
    this.found = undefined;
    if (this.decoder.state !== "closed") this.decoder.close();
  }
}

/** A stuck decoder becomes an error in the results instead of a frozen page. */
export async function withTimeout<T>(promise: Promise<T>, ms: number, what: string): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      promise,
      new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error(`${what} timed out after ${ms / 1000} s`)), ms); }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/**
 * Samples fed past the last one a window needs, so its frames come out without a flush. 16 is
 * H.264's largest decoded picture buffer: a conformant decoder has output a frame by the time 16
 * later pictures have gone in. Flushing is what the first version did per window, and a flush makes
 * the next chunk a keyframe (WebCodecs' "key chunk required"), so every window re-decoded from the
 * GOP's start.
 */
export const DECODE_AHEAD = 16;

/**
 * Frame-accurate decode of a window [first, last] of presentation indices (VFX Phase 3, streaming
 * since Phase 5).
 *
 * The decoder keeps going forward between windows: when the window starts after every frame already
 * output since the last configure, the samples after the last one fed are appended, with no reset
 * and no flush. Otherwise (a jump back, or past what was fed in another GOP) it resets and decodes
 * from the keyframe at or before `first`. Sequential access -- a render, playback -- decodes each
 * sample about once; the first version reset per window, which on a one-GOP clip made a 124-frame
 * render decode about a thousand frames (measured on the 4090, 2026-09-29: 365 ms of decode a frame).
 *
 * Each frame at or after `first` is handed to `keep` the moment it arrives -- including frames past
 * `last` that the ahead-feed produces, and frames that arrive after `decode` resolved -- and `keep`
 * must copy it synchronously, because it is closed straight after (the Phase 0 surface-pool rule).
 * Frames before `first` are closed on arrival. The stream is flushed only at its end, or when the
 * window's frames have not come out within `settleMs` of feeding (then the next window resets).
 */
export class DecodeWindow {
  private decoder: VideoDecoder;
  private error: unknown;
  private readonly indexOf = new Map<number, number>();
  /** Decode-order index of the last sample fed since configure; -1 when the next must be a keyframe. */
  private fed = -1;
  /** Highest presentation index output since configure. Decoders output in presentation order. */
  private outMax = -1;
  private sink: { first: number; keep: (index: number, frame: VideoFrame) => void } | undefined;
  private waiter: { last: number; resolve: () => void } | undefined;
  /** Samples handed to the decoder, for tests and the render's timing line. */
  fedSamples = 0;
  resets = 0;

  constructor(readonly media: Demuxed, private readonly config: VideoDecoderConfig, private readonly settleMs = 2000) {
    media.presentation.forEach((sample, index) => this.indexOf.set(media.samples[sample].timestampUs, index));
    this.decoder = new VideoDecoder({
      output: (frame) => {
        const index = this.indexOf.get(frame.timestamp);
        try {
          if (index !== undefined) {
            if (index > this.outMax) this.outMax = index;
            if (this.sink && index >= this.sink.first) this.sink.keep(index, frame);
          }
        } finally {
          frame.close();
        }
        const waiter = this.waiter;
        if (waiter && this.outMax >= waiter.last) { this.waiter = undefined; waiter.resolve(); }
      },
      error: (e) => { this.error = e; this.waiter?.resolve(); },
    });
  }

  get frameCount(): number { return this.media.presentation.length; }

  async decode(first: number, last: number, keep: (index: number, frame: VideoFrame) => void): Promise<void> {
    const count = this.media.presentation.length;
    first = Math.max(0, Math.min(count - 1, first));
    last = Math.max(first, Math.min(count - 1, last));
    let start = this.media.presentation[first];
    let end = start;
    for (let index = first; index <= last; index++) {
      const sample = this.media.presentation[index];
      start = Math.min(start, sample);
      end = Math.max(end, sample);
    }
    while (start > 0 && !this.media.samples[start].isSync) start--;
    if (this.decoder.state === "closed") throw this.error ?? new Error("the decoder was closed");
    const continuing = this.fed >= 0 && this.outMax < first && this.fed >= start - 1;
    if (!continuing) {
      this.decoder.reset();
      this.decoder.configure(this.config);
      this.resets++;
      this.fed = start - 1;
      this.outMax = -1;
    }
    this.sink = { first, keep };
    const arrived = new Promise<void>((resolve) => {
      if (this.outMax >= last) resolve();
      else this.waiter = { last, resolve };
    });
    const samples = this.media.samples;
    const feedTo = Math.min(samples.length - 1, end + DECODE_AHEAD);
    try {
      for (let i = this.fed + 1; i <= feedTo && !this.error; i++) {
        await backpressure(this.decoder);
        this.decoder.decode(chunk(samples[i]));
        this.fed = i;
        this.fedSamples++;
      }
      if (!this.error && this.fed >= samples.length - 1) {
        // The end of the stream: only a flush brings the last pictures out.
        await withTimeout(this.decoder.flush(), 30000, `decoding frames ${first}-${last}`);
        this.fed = -1;
      } else if (!this.error && !(await settled(arrived, this.settleMs))) {
        await withTimeout(this.decoder.flush(), 30000, `decoding frames ${first}-${last}`);
        this.fed = -1;
      }
    } finally {
      this.waiter = undefined;
    }
    if (this.error) throw this.error;
  }

  close() {
    this.sink = undefined;
    if (this.decoder.state !== "closed") this.decoder.close();
  }
}

/** True if `promise` resolves within `ms`. */
async function settled(promise: Promise<void>, ms: number): Promise<boolean> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([promise.then(() => true), new Promise<boolean>((resolve) => { timer = setTimeout(() => resolve(false), ms); })]);
  } finally {
    if (timer) clearTimeout(timer);
  }
}

/** The best config this browser supports for the file: hardware, else software, else undefined. */
export async function supportedConfig(media: Demuxed): Promise<VideoDecoderConfig | undefined> {
  for (const preference of ["prefer-hardware", "prefer-software"] as const) {
    const config = decoderConfig(media, preference);
    try {
      if ((await VideoDecoder.isConfigSupported(config)).supported) return config;
    } catch { /* try the next */ }
  }
  return undefined;
}
