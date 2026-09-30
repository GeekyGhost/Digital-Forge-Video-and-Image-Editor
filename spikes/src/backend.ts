/**
 * What the spikes need from outside the WebView. Two implementations:
 * - `TauriBackend`: the real one, inside Digital Forge (Rust side: spikes/rust/spikes.rs).
 * - `HttpBackend`: the container harness (spikes/harness/server.mjs), so the page can be
 *   exercised in headless Chromium before it reaches a real GPU machine.
 */

export type ClipInfo = {
  id: string;
  path: string;
  codec: string;
  width: number;
  height: number;
  frames: number;
  fps: number;
  /** True when frame N carries N as an 8-bit strip in its top-left 256x32 pixels. */
  indexStrip: boolean;
  error?: string;
};

export type AlphaMedia = {
  width: number;
  height: number;
  frames: number;
  fps: number;
  /** RGBA PNG per frame, in order. */
  pngPaths: string[];
  /** H.264, colour on the left half, alpha as luma on the right half. */
  stackedPath?: string;
  /** VP9 with alpha (yuva420p) in WebM. */
  vp9AlphaPath?: string;
  errors: string[];
};

export type MediaSet = {
  directory: string;
  clips: ClipInfo[];
  alpha: AlphaMedia;
  ffmpegVersion: string;
};

export type EncodeSession = { id: string; encoder: string; outputPath: string };

/** The localhost frame channel (spike round 2): Rust listens on 127.0.0.1 and checks the token. */
export type HttpEndpoint = { url: string; token: string };

export type EncodeStats = {
  frames: number;
  bytes: number;
  /** Time Rust spent blocked writing into ffmpeg's stdin. */
  writeSeconds: number;
  /** Time from closing stdin to ffmpeg exiting. */
  drainSeconds: number;
  outputBytes: number;
  /** Frames in the output according to ffprobe, when there is an output. */
  outputFrames: number | null;
  /** Pixel format of the encoded output (NVENC fed RGBA converts on the GPU). */
  outputPixFmt?: string | null;
  stderrTail: string;
};

export type AppUsage = { privateBytes: number; cpuSeconds: number; processes: number };

export type WanAlphaRequest = { label: string; prompt: string; frames: number; seed: number };

export interface Backend {
  readonly name: string;
  prepareMedia(): Promise<MediaSet>;
  readFile(path: string): Promise<ArrayBuffer>;
  /** Frame `index` of `path` as RGBA, converted by ffmpeg with an explicit BT.709 matrix. */
  referenceFrame(path: string, index: number): Promise<Uint8Array>;
  /** `encoder` "null" discards the bytes in Rust (transport-only measurement). `pixFmt` is what the
   * frames are: "rgba", or "yuv420p" packed by the page. */
  encodeStart(width: number, height: number, fps: number, encoder: string, pixFmt: string): Promise<EncodeSession>;
  /** Tauri IPC, raw request body. */
  encodeFrame(id: string, bytes: Uint8Array): Promise<void>;
  httpEndpoint(): Promise<HttpEndpoint>;
  /** Same as encodeFrame, over the localhost HTTP channel. */
  encodeFrameHttp(endpoint: HttpEndpoint, id: string, bytes: Uint8Array): Promise<void>;
  encodeFinish(id: string): Promise<EncodeStats>;
  appUsage(): Promise<AppUsage | null>;
  saveResults(json: string): Promise<string>;
  /** Spike 0.6: one Wan-Alpha generation through the resource guard. App only. */
  wanAlpha(request: WanAlphaRequest): Promise<unknown>;
}

type InvokeFn = (cmd: string, args?: unknown, options?: { headers: Record<string, string> }) => Promise<unknown>;

export class TauriBackend implements Backend {
  readonly name = "tauri";
  constructor(private invoke: InvokeFn) {}

  prepareMedia() { return this.invoke("spike_prepare_media") as Promise<MediaSet>; }

  async readFile(path: string) {
    const bytes = await this.invoke("spike_read_file", { path });
    return toArrayBuffer(bytes);
  }

  async referenceFrame(path: string, index: number) {
    const bytes = await this.invoke("spike_reference_frame", { path, index });
    return new Uint8Array(toArrayBuffer(bytes));
  }

  encodeStart(width: number, height: number, fps: number, encoder: string, pixFmt: string) {
    return this.invoke("spike_encode_start", { width, height, fps, encoder, pixFmt }) as Promise<EncodeSession>;
  }

  httpEndpoint() { return this.invoke("spike_http_start") as Promise<HttpEndpoint>; }
  encodeFrameHttp(endpoint: HttpEndpoint, id: string, bytes: Uint8Array) { return postFrame(endpoint, id, bytes); }

  async encodeFrame(id: string, bytes: Uint8Array) {
    // A Uint8Array argument is sent as the raw request body: no JSON, no base64.
    await this.invoke("spike_encode_frame", bytes, { headers: { "x-spike-session": id } });
  }

  encodeFinish(id: string) { return this.invoke("spike_encode_finish", { id }) as Promise<EncodeStats>; }
  appUsage() { return this.invoke("spike_app_usage") as Promise<AppUsage | null>; }
  saveResults(json: string) { return this.invoke("spike_save_results", { json }) as Promise<string>; }
  wanAlpha(request: WanAlphaRequest) { return this.invoke("spike_wan_alpha", { request }); }
}

async function postFrame(endpoint: HttpEndpoint, id: string, bytes: Uint8Array): Promise<void> {
  const response = await fetch(endpoint.url, {
    method: "POST",
    headers: { "content-type": "application/octet-stream", "x-spike-session": id, "x-spike-token": endpoint.token },
    body: bytes as BodyInit,
  });
  if (!response.ok) throw new Error(`frame POST: ${response.status} ${await response.text()}`);
}

/** Tauri returns `ipc::Response` bodies as ArrayBuffer; tolerate a number[] too. */
function toArrayBuffer(value: unknown): ArrayBuffer {
  if (value instanceof ArrayBuffer) return value;
  if (ArrayBuffer.isView(value)) return value.buffer.slice(value.byteOffset, value.byteOffset + value.byteLength) as ArrayBuffer;
  if (Array.isArray(value)) return new Uint8Array(value as number[]).buffer;
  throw new Error(`expected binary response, got ${typeof value}`);
}

export class HttpBackend implements Backend {
  readonly name = "http";
  constructor(private base = "/api") {}

  private async json<T>(path: string, body?: unknown): Promise<T> {
    const response = await fetch(`${this.base}/${path}`, {
      method: body === undefined ? "GET" : "POST",
      headers: body === undefined ? undefined : { "content-type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    if (!response.ok) throw new Error(`${path}: ${response.status} ${await response.text()}`);
    return response.json() as Promise<T>;
  }

  private async binary(path: string, body: unknown): Promise<ArrayBuffer> {
    const response = await fetch(`${this.base}/${path}`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) });
    if (!response.ok) throw new Error(`${path}: ${response.status} ${await response.text()}`);
    return response.arrayBuffer();
  }

  prepareMedia() { return this.json<MediaSet>("prepare-media", {}); }
  readFile(path: string) { return this.binary("read-file", { path }); }
  async referenceFrame(path: string, index: number) { return new Uint8Array(await this.binary("reference-frame", { path, index })); }
  encodeStart(width: number, height: number, fps: number, encoder: string, pixFmt: string) { return this.json<EncodeSession>("encode-start", { width, height, fps, encoder, pixFmt }); }
  async httpEndpoint(): Promise<HttpEndpoint> { return { url: `${location.origin}/frame`, token: "harness" }; }
  encodeFrameHttp(endpoint: HttpEndpoint, id: string, bytes: Uint8Array) { return postFrame(endpoint, id, bytes); }

  async encodeFrame(id: string, bytes: Uint8Array) {
    const response = await fetch(`${this.base}/encode-frame`, { method: "POST", headers: { "x-spike-session": id }, body: bytes as BodyInit });
    if (!response.ok) throw new Error(`encode-frame: ${response.status}`);
  }

  encodeFinish(id: string) { return this.json<EncodeStats>("encode-finish", { id }); }
  appUsage() { return this.json<AppUsage | null>("app-usage", {}); }
  saveResults(json: string) { return this.json<string>("save-results", { json }); }
  async wanAlpha(): Promise<unknown> { throw new Error("Wan Alpha needs the app (and CUDA); the harness cannot run it"); }
}

export async function detectBackend(): Promise<Backend> {
  const params = new URLSearchParams(location.search);
  const internals = (window as unknown as { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;
  if (internals && params.get("backend") !== "http") {
    const { invoke } = await import("@tauri-apps/api/core");
    return new TauriBackend(invoke as InvokeFn);
  }
  return new HttpBackend();
}
