/**
 * VFX Phase 0 spike page. Opened by the app only when launched with AIV_SPIKES=1
 * (Run-VFX-Spikes.bat), or by the container harness with ?backend=http.
 */
import { runAlphaSpike } from "./alpha";
import { detectBackend, type Backend, type MediaSet } from "./backend";
import { runDecodeSpike } from "./decode";
import { runGpuSpike } from "./gpu";
import { runReadbackSpike } from "./readback";
import { runWanAlphaSpike } from "./wanAlpha";
import { runParity } from "../../src/render/parity";
import { runFramesSpike } from "./frames";

type Results = Record<string, unknown>;

const logEl = document.getElementById("log")!;
const outEl = document.getElementById("results")!;
const statusEl = document.getElementById("status")!;
const results: Results = {};

function log(line: string) {
  const stamp = new Date().toISOString().slice(11, 19);
  logEl.textContent += `${stamp}  ${line}\n`;
  logEl.scrollTop = logEl.scrollHeight;
  console.log(`[spike] ${line}`);
}

function show() { outEl.textContent = JSON.stringify(results, null, 2); }

function environment() {
  const ua = navigator.userAgent;
  const edge = /Edg\/([\d.]+)/.exec(ua)?.[1];
  const chrome = /Chrome\/([\d.]+)/.exec(ua)?.[1];
  return {
    userAgent: ua,
    webview2: edge ?? null,
    chromium: chrome ?? null,
    hardwareConcurrency: navigator.hardwareConcurrency,
    devicePixelRatio: devicePixelRatio,
    webCodecs: typeof VideoDecoder !== "undefined",
    webgpu: "gpu" in navigator,
    crossOriginIsolated: self.crossOriginIsolated,
    timestamp: new Date().toISOString(),
  };
}

let backend: Backend;
let media: MediaSet | undefined;

async function ensureMedia(): Promise<MediaSet> {
  if (media) return media;
  log("preparing test media (first run encodes it with ffmpeg; later runs reuse it)");
  media = await backend.prepareMedia();
  results.media = { directory: media.directory, ffmpeg: media.ffmpegVersion, clips: media.clips.map((c) => ({ id: c.id, error: c.error })), alphaErrors: media.alpha.errors };
  show();
  return media;
}

const spikes: Record<string, () => Promise<void>> = {
  async gpu() {
    log("0.2 GPU in WebView2");
    results.gpu = await runGpuSpike();
  },
  async decode() {
    const set = await ensureMedia();
    log("0.3 video decode");
    results.decode = await runDecodeSpike(backend, set.clips, log);
  },
  async readback() {
    log("0.4 readback -> IPC -> ffmpeg");
    results.readback = await runReadbackSpike(backend, log);
  },
  async alpha() {
    const set = await ensureMedia();
    log("0.5 alpha elements");
    results.alpha = await runAlphaSpike(backend, set.alpha, log);
  },
};

/** Not part of "Run all": it needs Setup-Wan-Alpha.bat and takes a long time. */
const extraSpikes: Record<string, () => Promise<void>> = {
  async wanAlpha() {
    results.wanAlpha = await runWanAlphaSpike(backend, log);
  },
  /** VFX Phase 3: the frame store on the test media, and 1500 px composition speed with elements. */
  async frames() {
    const set = await ensureMedia();
    log("Phase 3 frames: video seek/playback, element master/proxy, compose");
    results.frames = await runFramesSpike(backend, set, log);
  },
  /** VFX compositor parity harness (src/render/parity.ts). Kept when spikes/ is removed: move the button. */
  async parity() {
    // ?filter=text runs only the fixtures whose name contains it (diagnosis), and skips the timing.
    const filter = new URLSearchParams(location.search).get("filter") ?? undefined;
    log(`Compositor parity (VFX Phases 1-4): old compositor vs GPU renderer${filter ? `, fixtures matching "${filter}"` : ""}`);
    results.parity = await runParity(log, { filter, performance: filter ? false : undefined });
  },
};

async function run(names: string[]) {
  document.querySelectorAll("button").forEach((b) => { b.disabled = true; });
  statusEl.textContent = "running";
  results.environment = environment();
  results.backend = backend.name;
  for (const name of names) {
    const started = performance.now();
    try {
      await (spikes[name] ?? extraSpikes[name])();
    } catch (error) {
      results[name] = { error: String(error), stack: (error as Error)?.stack };
      log(`${name} FAILED: ${error}`);
    }
    log(`${name} done in ${((performance.now() - started) / 1000).toFixed(1)} s`);
    show();
  }
  try {
    const path = await backend.saveResults(JSON.stringify(results, null, 2));
    log(`results saved to ${path}`);
  } catch (error) {
    log(`could not save results: ${error}`);
  }
  statusEl.textContent = "done";
  (window as unknown as { __spikeResults?: Results }).__spikeResults = results;
  document.querySelectorAll("button").forEach((b) => { b.disabled = false; });
}

async function main() {
  backend = await detectBackend();
  log(`backend: ${backend.name}`);
  for (const name of Object.keys(spikes)) {
    document.getElementById(`run-${name}`)?.addEventListener("click", () => void run([name]));
  }
  document.getElementById("run-all")!.addEventListener("click", () => void run(Object.keys(spikes)));
  document.getElementById("run-wan-alpha")!.addEventListener("click", () => void run(["wanAlpha"]));
  document.getElementById("run-parity")!.addEventListener("click", () => void run(["parity"]));
  document.getElementById("run-frames")!.addEventListener("click", () => void run(["frames"]));
  document.getElementById("copy")!.addEventListener("click", () => void navigator.clipboard.writeText(outEl.textContent ?? ""));
  const params = new URLSearchParams(location.search);
  const auto = params.get("auto");
  if (auto) void run(auto === "1" ? Object.keys(spikes) : auto.split(","));
}

void main();
