/**
 * DecodeWindow against a fake WebCodecs decoder (VFX Phase 5). The fake enforces the rule that made
 * the first version slow -- after configure() or flush() the next chunk must be a keyframe -- and
 * models a decoded picture buffer: a picture comes out once `delay` later chunks have gone in, in
 * presentation order, or at flush. Output is asynchronous, as in a browser.
 */
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { DECODE_AHEAD, DecodeWindow, type Demuxed } from "./demux";

type Chunk = { type: "key" | "delta"; timestamp: number };

class FakeDecoder {
  static delay = 0;
  static decoded = 0;
  state: "unconfigured" | "configured" | "closed" = "unconfigured";
  decodeQueueSize = 0;
  private keyRequired = true;
  private buffer: number[] = [];
  private pending = Promise.resolve();
  /** Outputs queued before a reset() are dropped, as Chromium does. */
  private generation = 0;
  constructor(private readonly init: { output: (frame: { timestamp: number; close(): void }) => void; error: (e: unknown) => void }) {}
  configure() { this.state = "configured"; this.keyRequired = true; this.buffer = []; }
  reset() { this.state = "unconfigured"; this.buffer = []; this.generation++; }
  close() { this.state = "closed"; }
  decode(chunk: Chunk) {
    if (this.state !== "configured") throw new Error("decode() on an unconfigured decoder");
    if (this.keyRequired && chunk.type !== "key") throw new Error("DataError: a key frame is required after configure() or flush()");
    this.keyRequired = false;
    FakeDecoder.decoded++;
    this.buffer.push(chunk.timestamp);
    this.buffer.sort((a, b) => a - b);
    const out: number[] = [];
    while (this.buffer.length > FakeDecoder.delay) out.push(this.buffer.shift()!);
    this.emit(out);
  }
  flush(): Promise<void> {
    const out = this.buffer.splice(0);
    this.keyRequired = true;
    this.emit(out);
    return this.pending;
  }
  private emit(timestamps: number[]) {
    const generation = this.generation;
    this.pending = this.pending.then(() => new Promise<void>((resolve) => setTimeout(() => {
      if (generation === this.generation && this.state !== "closed") {
        for (const timestamp of timestamps) this.init.output({ timestamp, close() {} });
      }
      resolve();
    }, 0)));
  }
}

class FakeChunk implements Chunk {
  type: "key" | "delta";
  timestamp: number;
  constructor(init: Chunk) { this.type = init.type; this.timestamp = init.timestamp; }
}

const globals = globalThis as unknown as Record<string, unknown>;
beforeAll(() => { globals.VideoDecoder = FakeDecoder; globals.EncodedVideoChunk = FakeChunk; });
afterAll(() => { delete globals.VideoDecoder; delete globals.EncodedVideoChunk; });

/**
 * A stream of `count` frames, a keyframe every `gop`. With `bFrames`, each GOP is coded I P B B P B B
 * ... (a P two pictures ahead, then the two B's between), so decode order differs from display.
 */
function stream(count: number, gop: number, bFrames = false): Demuxed {
  const order: number[] = [];
  for (let g = 0; g < count; g += gop) {
    const end = Math.min(count, g + gop);
    order.push(g);
    let previous = g;
    while (previous < end - 1) {
      const anchor = bFrames ? Math.min(previous + 3, end - 1) : previous + 1;
      order.push(anchor);
      for (let b = previous + 1; b < anchor; b++) order.push(b);
      previous = anchor;
    }
  }
  const samples = order.map((shown, index) => ({ index, isSync: shown % gop === 0, timestampUs: shown * 40000, durationUs: 40000, data: new Uint8Array(1) }));
  const presentation = samples.map((sample) => sample.index).sort((a, b) => samples[a].timestampUs - samples[b].timestampUs);
  return { codec: "avc1.640028", width: 16, height: 16, samples, presentation, fps: 25 };
}

/** The frame store's use: ask for the first missing frame with a lookahead of 8. */
async function read(window: DecodeWindow, frames: number[]) {
  const cache = new Map<number, number>();
  for (const at of frames) {
    if (cache.has(at)) continue;
    await window.decode(at, at + 8, (index, frame) => { if (!cache.has(index)) cache.set(index, frame.timestamp); });
    expect(cache.get(at), `frame ${at}`).toBe(at * 40000);
  }
  // Frames the ahead-feed produced after the last window resolved land too.
  await new Promise((resolve) => setTimeout(resolve, 5));
  return cache;
}

const config = { codec: "avc1.640028" } as VideoDecoderConfig;

describe("DecodeWindow streams forward instead of replaying the GOP", () => {
  it("a 124-frame one-GOP clip, read in order, decodes each sample once with one configure", async () => {
    FakeDecoder.delay = 0;
    FakeDecoder.decoded = 0;
    const window = new DecodeWindow(stream(124, 1000), config);
    const all = Array.from({ length: 124 }, (_, i) => i);
    const cache = await read(window, all);
    for (const at of all) expect(cache.get(at)).toBe(at * 40000);
    expect(window.resets).toBe(1);
    expect(FakeDecoder.decoded).toBe(124);
  });

  it("holds exactly with a reorder delay and B-frames, keyframe every 30", async () => {
    FakeDecoder.delay = 2;
    FakeDecoder.decoded = 0;
    const media = stream(95, 30, true);
    expect(media.samples.map((s) => s.timestampUs / 40000).slice(0, 7)).toEqual([0, 3, 1, 2, 6, 4, 5]);
    const window = new DecodeWindow(media, config);
    const all = Array.from({ length: 95 }, (_, i) => i);
    await read(window, all);
    expect(window.resets).toBe(1);
    expect(FakeDecoder.decoded).toBe(95);
  });

  it("a jump back resets from the keyframe; a jump forward inside the GOP continues", async () => {
    FakeDecoder.delay = 1;
    const window = new DecodeWindow(stream(120, 60), config);
    await read(window, [40]);
    expect(window.resets).toBe(1);
    await read(window, [100]);
    expect(window.resets).toBe(1); // 100's keyframe (60) is after what was fed: continuing is cheaper
    await read(window, [10]);
    expect(window.resets).toBe(2);
    await read(window, [5, 15, 70, 3]);
  });

  it("never feeds a delta frame after a flush: the end of the stream, then another read", async () => {
    FakeDecoder.delay = 3;
    const window = new DecodeWindow(stream(40, 1000), config);
    await read(window, [30, 39]);
    await read(window, [39, 20, 0]);
  });

  it("a decoder that holds more than the ahead-feed still delivers, by the settle flush", async () => {
    FakeDecoder.delay = DECODE_AHEAD + 4;
    FakeDecoder.decoded = 0;
    const window = new DecodeWindow(stream(80, 1000), config, 10);
    await read(window, Array.from({ length: 80 }, (_, i) => i));
    expect(window.resets).toBeGreaterThan(1);
  });
});
