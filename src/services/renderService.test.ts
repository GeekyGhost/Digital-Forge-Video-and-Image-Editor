import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: vi.fn() }));

import { FrameSender } from "./renderService";

const started = { id: "r1", outputPath: "o.mp4", pixelFormat: "yuv420" as const, frameBytes: 6, http: { url: "http://127.0.0.1:5/frame", token: "t" } };

describe("FrameSender (VFX Phase 6)", () => {
  it("posts frames to the render's channel with its token", async () => {
    const fetchFn = vi.fn(async () => ({ ok: true, status: 204, text: async () => "" }));
    const ipc = vi.fn(async () => 1);
    const sender = new FrameSender(started, fetchFn, ipc);
    const frame = new Uint8Array([1, 2, 3, 4, 5, 6]);
    await sender.send(frame);
    await sender.send(frame);
    expect(fetchFn).toHaveBeenCalledTimes(2);
    expect(fetchFn).toHaveBeenCalledWith("http://127.0.0.1:5/frame", { method: "POST", body: frame, headers: { "x-render-token": "t" } });
    expect(ipc).not.toHaveBeenCalled();
    expect(sender.describe()).toBe("HTTP as YUV 4:2:0");
  });

  it("falls back to IPC for the rest of the render when the channel can't be reached", async () => {
    const fetchFn = vi.fn(async () => { throw new TypeError("Failed to fetch"); });
    const ipc = vi.fn(async () => 1);
    const sender = new FrameSender(started, fetchFn, ipc);
    await sender.send(new Uint8Array(6));
    await sender.send(new Uint8Array(6));
    expect(fetchFn).toHaveBeenCalledTimes(1);
    expect(ipc).toHaveBeenCalledTimes(2);
    expect(ipc).toHaveBeenCalledWith("r1", expect.any(Uint8Array));
    expect(sender.describe()).toBe("IPC as YUV 4:2:0 (HTTP failed: Failed to fetch)");
  });

  it("throws an HTTP error status instead of resending", async () => {
    const fetchFn = vi.fn(async () => ({ ok: false, status: 500, text: async () => "ffmpeg stopped accepting frames" }));
    const ipc = vi.fn(async () => 1);
    const sender = new FrameSender(started, fetchFn, ipc);
    await expect(sender.send(new Uint8Array(6))).rejects.toThrow("render frame refused (500): ffmpeg stopped accepting frames");
    expect(ipc).not.toHaveBeenCalled();
  });

  it("uses IPC with RGBA against a backend that has no channel", async () => {
    const ipc = vi.fn(async () => 1);
    const sender = new FrameSender({ id: "r2", outputPath: "o.mp4" }, vi.fn(), ipc);
    await sender.send(new Uint8Array(8));
    expect(ipc).toHaveBeenCalledTimes(1);
    expect(sender.describe()).toBe("IPC as RGBA");
  });
});
