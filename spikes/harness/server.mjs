// Container test harness for the Phase 0 spike page: serves a built dist/ and implements the same
// backend the Tauri app provides (spikes/rust/spikes.rs), using the ffmpeg on PATH.
//
//   node spikes/harness/server.mjs <dist-dir> <work-dir> [port]
//
// Not used on a real install; it exists so the page can be verified in headless Chromium first.
import { spawn, spawnSync } from "node:child_process";
import { createServer } from "node:http";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { extname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = fileURLToPath(new URL(".", import.meta.url));
const recipe = JSON.parse(readFileSync(join(here, "..", "media-recipe.json"), "utf8"));
const [distDir, workDir, portArg] = process.argv.slice(2);
if (!distDir || !workDir) { console.error("usage: server.mjs <dist-dir> <work-dir> [port]"); process.exit(2); }
const mediaDir = resolve(workDir, "media");
mkdirSync(mediaDir, { recursive: true });

// SPIKE_NO_H264=1: the harness Chromium has no H.264 decoder, so the stacked proxy and the
// readback output are encoded as VP9 instead, purely so their checks can run here.
const noH264 = process.env.SPIKE_NO_H264 === "1";
const VP9 = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "12", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1"];
function swapToVp9(args) {
  const i = args.indexOf("libx264");
  if (!noH264 || i < 1) return args;
  const out = [...args];
  out.splice(i - 1, 2, ...VP9);
  const preset = out.indexOf("-preset");
  if (preset >= 0) out.splice(preset, 2);
  return out;
}
const fill = (args, values) => args.map((arg) => arg.replace(/\{(\w+)\}/g, (_, key) => String(values[key] ?? `{${key}}`)));

function ffmpeg(args, input) {
  const result = spawnSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...args], { input, maxBuffer: 1 << 30 });
  if (result.status !== 0) throw new Error(`ffmpeg failed: ${result.stderr.toString().slice(-800)}`);
  return result.stdout;
}

function prepareMedia() {
  const clips = [];
  for (const clip of recipe.videoClips) {
    const out = join(mediaDir, clip.file);
    const info = { id: clip.id, path: out, codec: clip.codec, width: clip.width, height: clip.height, frames: clip.frames, fps: clip.fps, indexStrip: true };
    try {
      if (!existsSync(out)) {
        const values = { W: clip.width, H: clip.height, N: clip.frames, FPS: clip.fps };
        ffmpeg([...fill(recipe.videoInputArgs, values), ...recipe.encoderArgs[clip.codec], ...(clip.tagged ? recipe.tagArgs : []), ...recipe.containerArgs, out]);
      }
    } catch (error) { info.error = String(error); }
    clips.push(info);
  }
  const a = recipe.alpha;
  const dir = join(mediaDir, "alpha-png");
  mkdirSync(dir, { recursive: true });
  const errors = [];
  const values = { W: a.width, H: a.height, N: a.frames, FPS: a.fps, DIR: dir };
  const pngPaths = Array.from({ length: a.frames }, (_, i) => join(dir, `frame_${String(i).padStart(4, "0")}.png`));
  try { if (!existsSync(pngPaths.at(-1))) ffmpeg(fill(a.pngArgs, values)); } catch (error) { errors.push(`png: ${error}`); }
  const stackedPath = join(mediaDir, noH264 ? "alpha-stacked-vp9.mp4" : "alpha-stacked.mp4");
  const vp9AlphaPath = join(mediaDir, "alpha-vp9.webm");
  try { if (!existsSync(stackedPath)) ffmpeg(swapToVp9(fill(a.stackedArgs, { ...values, OUT: stackedPath }))); } catch (error) { errors.push(`stacked: ${error}`); }
  try { if (!existsSync(vp9AlphaPath)) ffmpeg(fill(a.vp9Args, { ...values, OUT: vp9AlphaPath })); } catch (error) { errors.push(`vp9: ${error}`); }
  const version = spawnSync("ffmpeg", ["-version"]).stdout.toString().split("\n")[0];
  return {
    directory: mediaDir, clips, ffmpegVersion: version,
    alpha: { width: a.width, height: a.height, frames: a.frames, fps: a.fps, pngPaths, stackedPath: existsSync(stackedPath) ? stackedPath : undefined, vp9AlphaPath: existsSync(vp9AlphaPath) ? vp9AlphaPath : undefined, errors },
  };
}

function referenceFrame(path, index) {
  const image = [".png", ".jpg", ".jpeg"].includes(extname(path).toLowerCase());
  return ffmpeg(fill(image ? recipe.referenceImageArgs : recipe.referenceVideoArgs, { IN: path, INDEX: index }));
}

let nvencWorks;
function encoderFor(requested) {
  if (requested !== "h264_nvenc") return requested;
  if (noH264) return "libvpx-vp9";
  if (nvencWorks === undefined) {
    nvencWorks = spawnSync("ffmpeg", ["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=s=256x256", "-frames:v", "1", "-c:v", "h264_nvenc", "-f", "null", "-"]).status === 0;
  }
  return nvencWorks ? "h264_nvenc" : "libx264";
}

const sessions = new Map();
let nextId = 1;

function encodeStart({ width, height, fps, encoder, pixFmt = "rgba" }) {
  const id = String(nextId++);
  const outputPath = join(workDir, `readback-${id}-${width}x${height}.mp4`);
  const chosen = encoderFor(encoder);
  const session = { id, frames: 0, bytes: 0, writeSeconds: 0, outputPath, encoder: chosen, child: null, stderr: "" };
  if (chosen !== "null") {
    const codecArgs = chosen === "libx264" ? ["-c:v", "libx264", "-preset", "ultrafast"]
      : chosen === "libvpx-vp9" ? VP9 : ["-c:v", chosen, "-preset", "p4"];
    const tags = pixFmt === "yuv420p" ? ["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"] : [];
    const child = spawn("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", pixFmt, ...tags, "-s", `${width}x${height}`, "-r", String(fps), "-i", "-",
      ...codecArgs, "-pix_fmt", "yuv420p", ...tags, "-g", "48", "-bf", "0", outputPath], { stdio: ["pipe", "ignore", "pipe"] });
    child.stderr.on("data", (d) => { session.stderr = (session.stderr + d).slice(-2000); });
    session.child = child;
  }
  sessions.set(id, session);
  return { id, encoder: chosen, outputPath };
}

async function encodeFrame(id, body) {
  const session = sessions.get(id);
  if (!session) throw new Error("no such session");
  const started = performance.now();
  if (session.child) {
    await new Promise((ok, fail) => session.child.stdin.write(body, (error) => (error ? fail(error) : ok())));
  }
  session.writeSeconds += (performance.now() - started) / 1000;
  session.frames++;
  session.bytes += body.length;
}

async function encodeFinish({ id }) {
  const session = sessions.get(id);
  if (!session) throw new Error("no such session");
  sessions.delete(id);
  const started = performance.now();
  if (session.child) {
    session.child.stdin.end();
    await new Promise((ok) => session.child.on("close", ok));
  }
  const drainSeconds = (performance.now() - started) / 1000;
  const exists = session.child && existsSync(session.outputPath);
  let outputFrames = null;
  if (exists) {
    const probe = spawnSync("ffprobe", ["-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", session.outputPath]);
    outputFrames = Number.parseInt(probe.stdout.toString().trim(), 10) || null;
  }
  return { frames: session.frames, bytes: session.bytes, writeSeconds: session.writeSeconds, drainSeconds, outputBytes: exists ? statSync(session.outputPath).size : 0, outputFrames, stderrTail: session.stderr };
}

const MIME = { ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png" };

async function readBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks);
}

const server = createServer(async (req, res) => {
  const url = new URL(req.url, "http://localhost");
  try {
    if (url.pathname === "/frame") {
      // Same shape as the Rust localhost channel (spikes.rs): token + session headers, raw body.
      if (req.headers["x-spike-token"] !== "harness") { res.writeHead(403); return res.end("bad token"); }
      await encodeFrame(req.headers["x-spike-session"], await readBody(req));
      res.writeHead(204); return res.end();
    }
    if (url.pathname.startsWith("/api/")) {
      const route = url.pathname.slice(5);
      const body = await readBody(req);
      const json = () => JSON.parse(body.toString() || "{}");
      const send = (value) => { res.writeHead(200, { "content-type": "application/json" }); res.end(JSON.stringify(value)); };
      const sendBinary = (buffer) => { res.writeHead(200, { "content-type": "application/octet-stream" }); res.end(buffer); };
      if (route === "prepare-media") return send(prepareMedia());
      if (route === "read-file") return sendBinary(readFileSync(json().path));
      if (route === "reference-frame") { const { path, index } = json(); return sendBinary(referenceFrame(path, index)); }
      if (route === "encode-start") return send(encodeStart(json()));
      if (route === "encode-frame") { await encodeFrame(req.headers["x-spike-session"], body); return send(null); }
      if (route === "encode-finish") return send(await encodeFinish(json()));
      if (route === "app-usage") return send(null);
      if (route === "save-results") { const path = join(workDir, `results-${Date.now()}.json`); writeFileSync(path, json().json); return send(path); }
      res.writeHead(404); return res.end("unknown route");
    }
    let path = join(distDir, decodeURIComponent(url.pathname));
    if (existsSync(path) && statSync(path).isDirectory()) path = join(path, "index.html");
    if (!existsSync(path)) { res.writeHead(404); return res.end("not found"); }
    res.writeHead(200, { "content-type": MIME[extname(path)] ?? "application/octet-stream" });
    res.end(readFileSync(path));
  } catch (error) {
    res.writeHead(500); res.end(String(error?.stack ?? error));
  }
});
const port = Number(portArg ?? 4173);
server.listen(port, () => console.log(`spike harness on http://localhost:${port}/spikes/index.html?backend=http`));
