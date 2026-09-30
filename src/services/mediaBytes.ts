/**
 * A media file's bytes, for the WebCodecs demuxer (VFX Phase 3).
 *
 * The asset protocol first (the same URL the monitor's <video> uses); if the WebView refuses to
 * `fetch` it cross-origin, the `media_read_bytes` command sends the file over IPC as a raw body.
 */

import { convertFileSrc, invoke } from "@tauri-apps/api/core";

export async function readMediaBytes(path: string): Promise<ArrayBuffer> {
  try {
    const response = await fetch(convertFileSrc(path));
    if (response.ok) return await response.arrayBuffer();
  } catch {
    // Fall through to IPC.
  }
  const bytes = await invoke<ArrayBuffer | Uint8Array | number[]>("media_read_bytes", { path });
  if (bytes instanceof ArrayBuffer) return bytes;
  if (bytes instanceof Uint8Array) return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  return new Uint8Array(bytes).buffer;
}
