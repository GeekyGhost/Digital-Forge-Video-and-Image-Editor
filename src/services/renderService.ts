/**
 * VFX Phase 4b: stream composed frames into ffmpeg (src-tauri/src/ai/render.rs). The page composes
 * each frame with the same compositor as its preview and sends it, one request per frame, so nothing
 * is written to disk but the finished MP4.
 *
 * VFX Phase 6: frames go as planar I420 packed on the page (`render/yuv420.ts`, 1.5 bytes per pixel
 * instead of 4) over a per-render localhost HTTP channel (`ai/frame_channel.rs`) instead of Tauri IPC,
 * which Phase 5 measured at ~12 MB/s. IPC stays as the fallback: `FrameSender` switches to it for the
 * rest of the render if the channel can't be reached (it says why in `fallback`).
 */
import { invoke } from "@tauri-apps/api/core";

export type RenderPixelFormat = "rgba" | "yuv420";
export type RenderEndpoint = { url: string; token: string };
export type RenderStarted = {
  id: string; outputPath: string;
  /** Absent from a pre-Phase 6 backend, which only takes RGBA over IPC. */
  pixelFormat?: RenderPixelFormat; frameBytes?: number; http?: RenderEndpoint | null;
};
/** `writeSeconds`: time spent inside ffmpeg's stdin (encoder back-pressure), apart from the transport. */
export type RenderFinished = { outputPath: string; frames: number; writeSeconds?: number; encoder?: string };
export type RenderAudio = { path: string; start: number; duration: number };

export function renderVideoStart(request: {
  projectId: string; name: string; width: number; height: number; fps: number; audio?: RenderAudio;
  pixelFormat?: RenderPixelFormat; http?: boolean;
}) {
  return invoke<RenderStarted>("render_video_start", request);
}

/** `bytes` must be exactly the render's frame size. Resolves to the number of frames written so far. */
export function renderVideoFrame(id: string, bytes: Uint8Array) {
  return invoke<number>("render_video_frame", bytes, { headers: { "x-render-session": id } });
}

export function renderVideoFinish(id: string) {
  return invoke<RenderFinished>("render_video_finish", { id });
}

export function renderVideoCancel(id: string) {
  return invoke<void>("render_video_cancel", { id });
}

type Fetch = (url: string, init: { method: string; body: Uint8Array; headers: Record<string, string> }) =>
  Promise<{ ok: boolean; status: number; text(): Promise<string> }>;

/**
 * Sends one render's frames, in order, one at a time (the caller awaits each before the next).
 * HTTP when the render has a channel; a network-level failure (the channel unreachable, a refused
 * preflight) switches to IPC for the rest of the render. An HTTP error status is a real error (the
 * encoder stopped, a wrong frame size) and is thrown, not retried.
 */
export class FrameSender {
  transport: "http" | "ipc";
  /** Why HTTP was abandoned, when it was. */
  fallback?: string;

  constructor(
    private started: RenderStarted,
    private fetchFn: Fetch = (url, init) => fetch(url, init as RequestInit),
    private ipc: (id: string, bytes: Uint8Array) => Promise<unknown> = renderVideoFrame,
  ) {
    this.transport = started.http ? "http" : "ipc";
  }

  async send(bytes: Uint8Array): Promise<void> {
    const endpoint = this.started.http;
    if (this.transport === "http" && endpoint) {
      let response: Awaited<ReturnType<Fetch>>;
      try {
        response = await this.fetchFn(endpoint.url, { method: "POST", body: bytes, headers: { "x-render-token": endpoint.token } });
      } catch (error) {
        this.transport = "ipc";
        this.fallback = error instanceof Error ? error.message : String(error);
        await this.ipc(this.started.id, bytes);
        return;
      }
      if (!response.ok) throw new Error(`render frame refused (${response.status}): ${await response.text().catch(() => "")}`);
      return;
    }
    await this.ipc(this.started.id, bytes);
  }

  /** What each frame must be: I420 when the backend agreed to it, otherwise RGBA. */
  get format(): RenderPixelFormat {
    return this.started.pixelFormat === "yuv420" ? "yuv420" : "rgba";
  }

  /** "HTTP as YUV 4:2:0", "IPC as RGBA (HTTP failed: …)": for the status line. */
  describe(): string {
    const format = this.started.pixelFormat === "yuv420" ? "YUV 4:2:0" : "RGBA";
    return `${this.transport.toUpperCase()} as ${format}${this.fallback ? ` (HTTP failed: ${this.fallback})` : ""}`;
  }
}
