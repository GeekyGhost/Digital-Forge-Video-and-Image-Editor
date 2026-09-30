/**
 * Spike 0.5: getting RGBA elements (Wan Alpha output) into textures with the alpha intact.
 *
 *   png        RGBA PNG sequence -> createImageBitmap(premultiplyAlpha "none") -> texture.
 *              Candidate master: lossless, one file per frame, trivially frame-accurate.
 *   png-default  the same with browser defaults, to show what premultiplication costs.
 *   stacked    H.264 with colour left and alpha-as-luma right, decoded by WebCodecs and recombined
 *              in a shader. Candidate preview proxy: hardware decode, frame-accurate seek.
 *   vp9-video  VP9 with alpha in WebM, through a <video> element. WebCodecs has no alpha-aware
 *              WebM path, so this is the only way to test VP9 alpha; <video> is not frame-exact.
 * Correctness is measured against ffmpeg's RGBA decode of the PNG frame (ground truth).
 */
import type { AlphaMedia, Backend } from "./backend";
import { decodeAll, decoderConfig, demux, Seeker } from "./demux";
import { compile, createGl, drawFullscreen, framebuffer, FULLSCREEN_VS, mutableTexture, round, texture } from "./gl";
import { uploadBrowser } from "./yuv";

type Accuracy = { alphaMae: number; alphaMax: number; rgbMaeOpaque: number; softAlphaPixels: number; matchedFrame?: number };

export type AlphaPathResult = {
  path: string;
  fps?: number;
  /** For png: time spent reading the files, separate from decode. */
  ioSecondsPerFrame?: number;
  accuracy?: Accuracy[];
  alphaSurvives?: boolean;
  frameFormat?: string | null;
  note?: string;
  error?: string;
};

function accuracy(actual: Uint8Array, reference: Uint8Array): Accuracy {
  let alphaSum = 0;
  let alphaMax = 0;
  let rgbSum = 0;
  let rgbCount = 0;
  let soft = 0;
  for (let i = 0; i < reference.length; i += 4) {
    const da = Math.abs(actual[i + 3] - reference[i + 3]);
    alphaSum += da;
    alphaMax = Math.max(alphaMax, da);
    if (actual[i + 3] > 0 && actual[i + 3] < 255) soft++;
    if (reference[i + 3] >= 250) {
      rgbSum += Math.abs(actual[i] - reference[i]) + Math.abs(actual[i + 1] - reference[i + 1]) + Math.abs(actual[i + 2] - reference[i + 2]);
      rgbCount += 3;
    }
  }
  return { alphaMae: round(alphaSum / (reference.length / 4), 3), alphaMax, rgbMaeOpaque: round(rgbCount ? rgbSum / rgbCount : 0, 3), softAlphaPixels: soft };
}

const CHECK_FRAMES = [0, 40];

async function runPng(backend: Backend, media: AlphaMedia, references: Map<number, Uint8Array>, premultiply: "none" | "default"): Promise<AlphaPathResult> {
  const result: AlphaPathResult = { path: premultiply === "none" ? "png" : "png-default" };
  try {
    const ioStart = performance.now();
    const blobs = await Promise.all(media.pngPaths.map(async (path) => new Blob([await backend.readFile(path)], { type: "image/png" })));
    result.ioSecondsPerFrame = round((performance.now() - ioStart) / 1000 / blobs.length, 4);
    const gl = createGl();
    const tex = mutableTexture(gl);
    const options: ImageBitmapOptions = premultiply === "none" ? { premultiplyAlpha: "none", colorSpaceConversion: "none" } : {};
    const one = new Uint8Array(4);
    const started = performance.now();
    for (const blob of blobs) {
      const bitmap = await createImageBitmap(blob, options);
      uploadBrowser(gl, tex, bitmap);
      bitmap.close();
      const fb = framebuffer(gl, tex);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one);
      gl.deleteFramebuffer(fb);
    }
    result.fps = round(blobs.length / ((performance.now() - started) / 1000), 1);
    result.accuracy = [];
    for (const index of CHECK_FRAMES) {
      const bitmap = await createImageBitmap(blobs[index], options);
      uploadBrowser(gl, tex, bitmap);
      bitmap.close();
      const pixels = new Uint8Array(media.width * media.height * 4);
      const fb = framebuffer(gl, tex);
      gl.readPixels(0, 0, media.width, media.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      gl.deleteFramebuffer(fb);
      result.accuracy.push(accuracy(pixels, references.get(index)!));
    }
    result.alphaSurvives = result.accuracy.every((a) => a.softAlphaPixels > 1000);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  } catch (error) {
    result.error = String(error);
  }
  return result;
}

const COMBINE_FS = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D stacked;
out vec4 color;
void main() {
  // Left half is colour, right half carries alpha as grey. Straight (not premultiplied) alpha.
  vec3 rgb = texture(stacked, vec2(uv.x * 0.5, uv.y)).rgb;
  float a = texture(stacked, vec2(0.5 + uv.x * 0.5, uv.y)).r;
  color = vec4(rgb, a);
}`;

async function runStacked(backend: Backend, media: AlphaMedia, references: Map<number, Uint8Array>): Promise<AlphaPathResult> {
  const result: AlphaPathResult = { path: "stacked-h264" };
  if (!media.stackedPath) return { ...result, error: "no stacked proxy was produced" };
  try {
    const demuxed = await demux(await backend.readFile(media.stackedPath));
    let config = decoderConfig(demuxed, "prefer-hardware");
    if (!(await VideoDecoder.isConfigSupported(config)).supported) config = decoderConfig(demuxed, "prefer-software");
    if (!(await VideoDecoder.isConfigSupported(config)).supported) return { ...result, error: `${demuxed.codec} not supported` };
    result.note = demuxed.codec;
    const gl = createGl();
    const src = mutableTexture(gl, gl.NEAREST);
    const out = texture(gl, gl.RGBA8, media.width, media.height);
    const outFb = framebuffer(gl, out);
    const program = compile(gl, FULLSCREEN_VS, COMBINE_FS);
    const one = new Uint8Array(4);
    const combine = (frame: VideoFrame) => {
      uploadBrowser(gl, src, frame);
      gl.bindFramebuffer(gl.FRAMEBUFFER, outFb);
      gl.viewport(0, 0, media.width, media.height);
      gl.useProgram(program);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, src);
      gl.uniform1i(gl.getUniformLocation(program, "stacked"), 0);
      drawFullscreen(gl);
    };
    const timed = await decodeAll(demuxed, config, (frame) => { combine(frame); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one); });
    result.fps = round(timed.frames / timed.seconds, 1);
    const seeker = new Seeker(demuxed, config);
    result.accuracy = [];
    for (const index of CHECK_FRAMES) {
      const frame = await seeker.frame(index);
      combine(frame);
      frame.close();
      const pixels = new Uint8Array(media.width * media.height * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, outFb);
      gl.readPixels(0, 0, media.width, media.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      result.accuracy.push(accuracy(pixels, references.get(index)!));
    }
    seeker.close();
    result.alphaSurvives = result.accuracy.every((a) => a.softAlphaPixels > 1000);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  } catch (error) {
    result.error = String(error);
  }
  return result;
}

async function runVp9Video(backend: Backend, media: AlphaMedia, references: Map<number, Uint8Array>): Promise<AlphaPathResult> {
  const result: AlphaPathResult = { path: "vp9-alpha-video-element", note: "<video> is not frame-exact; frame chosen by currentTime" };
  if (!media.vp9AlphaPath) return { ...result, error: "no VP9 alpha file was produced" };
  const video = document.createElement("video");
  let url = "";
  try {
    url = URL.createObjectURL(new Blob([await backend.readFile(media.vp9AlphaPath)], { type: "video/webm" }));
    video.muted = true;
    video.preload = "auto";
    video.src = url;
    await new Promise<void>((resolve, reject) => { video.onloadeddata = () => resolve(); video.onerror = () => reject(new Error(`video error ${video.error?.code}: ${video.error?.message}`)); });
    const gl = createGl();
    const tex = mutableTexture(gl);
    result.accuracy = [];
    for (const index of CHECK_FRAMES) {
      video.currentTime = (index + 0.5) / media.fps;
      await new Promise<void>((resolve) => { video.onseeked = () => resolve(); });
      try {
        const frame = new VideoFrame(video);
        result.frameFormat = frame.format;
        frame.close();
      } catch (error) {
        result.frameFormat = `VideoFrame(video) failed: ${error}`;
      }
      uploadBrowser(gl, tex, video);
      const pixels = new Uint8Array(media.width * media.height * 4);
      const fb = framebuffer(gl, tex);
      gl.readPixels(0, 0, media.width, media.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
      gl.deleteFramebuffer(fb);
      // <video> may land on a neighbour: score N-1, N and N+1, keep the closest (alpha + colour),
      // and record all three so an off-by-one is visible rather than hidden.
      let best: Accuracy | undefined;
      const candidates: string[] = [];
      for (const candidate of [index - 1, index, index + 1].filter((n) => n >= 0 && n < media.frames)) {
        const reference = references.get(candidate) ?? await backend.referenceFrame(media.pngPaths[candidate], 0);
        references.set(candidate, reference);
        const scored = { ...accuracy(pixels, reference), matchedFrame: candidate };
        candidates.push(`${candidate}: alpha ${scored.alphaMae}, rgb ${scored.rgbMaeOpaque}`);
        if (!best || scored.alphaMae + scored.rgbMaeOpaque < best.alphaMae + best.rgbMaeOpaque) best = scored;
      }
      result.accuracy.push(best!);
      result.note = `${result.note}; requested ${index} -> ${candidates.join(" | ")}`;
    }
    result.alphaSurvives = result.accuracy.every((a) => a.softAlphaPixels > 1000);
    // Playback rate as a rough throughput figure: frames presented during 2 s at 4x speed.
    let presented = 0;
    const counter = () => { presented++; if (!video.paused) video.requestVideoFrameCallback(counter); };
    video.currentTime = 0;
    video.playbackRate = 4;
    video.loop = true;
    video.requestVideoFrameCallback(counter);
    await video.play();
    await new Promise((resolve) => setTimeout(resolve, 2000));
    video.pause();
    result.fps = round(presented / 2, 1);
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  } catch (error) {
    result.error = String(error);
  } finally {
    video.removeAttribute("src");
    video.load();
    if (url) URL.revokeObjectURL(url);
  }
  return result;
}

export async function runAlphaSpike(backend: Backend, media: AlphaMedia, log: (line: string) => void) {
  const references = new Map<number, Uint8Array>();
  for (const index of CHECK_FRAMES) references.set(index, await backend.referenceFrame(media.pngPaths[index], 0));
  log("alpha: PNG (premultiplyAlpha none)");
  const png = await runPng(backend, media, references, "none");
  log("alpha: PNG (browser defaults)");
  const pngDefault = await runPng(backend, media, references, "default");
  log("alpha: stacked H.264 proxy");
  const stacked = await runStacked(backend, media, references);
  log("alpha: VP9 alpha via <video>");
  const vp9 = await runVp9Video(backend, media, references);
  const exact = png.accuracy?.every((a) => a.alphaMae === 0 && a.rgbMaeOpaque === 0) ?? false;
  return {
    element: `${media.width}x${media.height}x${media.frames}`,
    paths: [png, pngDefault, stacked, vp9],
    pngLossless: exact,
    passBar: "master decodes losslessly with soft alpha intact; a proxy keeps alpha within a few levels and decodes >= 24 fps",
    errors: media.errors,
  };
}

