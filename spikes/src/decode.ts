/**
 * Spike 0.3: frame-accurate WebCodecs decode in the WebView, with colour checked against ffmpeg.
 *
 * Per clip: codec support (hardware and software), decode fps (decode only, and decode + upload),
 * seek accuracy (every seek target read back from the frame-index strip), and colour: the decoded
 * frame through the browser's conversion and through our explicit BT.709 shader, each compared
 * with ffmpeg's frame converted with an explicit BT.709 matrix.
 */
import type { Backend, ClipInfo } from "./backend";
import { decodeAll, decoderConfig, demux, Seeker, type Demuxed } from "./demux";
import { compareRgba, createGl, framebuffer, mutableTexture, readIndexStrip, round } from "./gl";
import { ExplicitConverter, uploadBrowser } from "./yuv";

export type ColourCheck = {
  frame: number;
  frameFormat: string | null;
  colorSpace: VideoColorSpaceInit | null;
  browser: { mae: number; p99: number; max: number } | { error: string };
  explicitLeftSited: { mae: number; p99: number; max: number } | { error: string };
  explicitCentreSited: { mae: number; p99: number; max: number } | { error: string };
};

export type ClipResult = {
  id: string;
  codec: string;
  size: string;
  frames: number;
  supportedHardware: boolean;
  supportedSoftware: boolean;
  error?: string;
  decodeFps?: number;
  decodeUploadFps?: number;
  decodeFpsSoftware?: number;
  seek?: { targets: number; exact: number; misses: string[]; meanMs: number; maxMs: number };
  colour?: ColourCheck[];
  passDecode?: boolean;
  passSeek?: boolean;
  passColour?: boolean;
};

async function supported(config: VideoDecoderConfig): Promise<boolean> {
  try { return !!(await VideoDecoder.isConfigSupported(config)).supported; } catch { return false; }
}

function seekTargets(frames: number): number[] {
  // Keyframes, their neighbours, B-frame positions, the last frame, and backward jumps.
  const picks = [0, 1, 2, 23, 24, 25, 47, 48, 49, 70, frames - 1, 5, 60, 30, 13, frames - 2];
  return picks.filter((n) => n >= 0 && n < frames);
}

export async function runClip(backend: Backend, clip: ClipInfo, log: (line: string) => void): Promise<ClipResult> {
  const result: ClipResult = { id: clip.id, codec: clip.codec, size: `${clip.width}x${clip.height}`, frames: clip.frames, supportedHardware: false, supportedSoftware: false };
  if (clip.error) { result.error = clip.error; return result; }
  let media: Demuxed;
  try {
    media = await demux(await backend.readFile(clip.path));
  } catch (error) {
    result.error = `demux: ${error}`;
    return result;
  }
  result.codec = media.codec;
  const hw = decoderConfig(media, "prefer-hardware");
  const sw = decoderConfig(media, "prefer-software");
  result.supportedHardware = await supported(hw);
  result.supportedSoftware = await supported(sw);
  const config = result.supportedHardware ? hw : result.supportedSoftware ? sw : null;
  if (!config) { result.error = `codec ${media.codec} is not supported by VideoDecoder`; return result; }

  const gl = createGl();
  try {
    log(`${clip.id}: decode throughput`);
    const plain = await decodeAll(media, config);
    result.decodeFps = round(plain.frames / plain.seconds, 1);
    if (result.supportedSoftware && config !== sw) {
      const soft = await decodeAll(media, sw);
      result.decodeFpsSoftware = round(soft.frames / soft.seconds, 1);
    }
    const tex = mutableTexture(gl);
    const one = new Uint8Array(4);
    let uploadFb: WebGLFramebuffer | null = null;
    const uploaded = await decodeAll(media, config, (frame) => {
      uploadBrowser(gl, tex, frame);
      // A 1-pixel read forces the upload to complete inside the timed loop. Same size every
      // frame, so the framebuffer made after the first upload stays valid.
      uploadFb ??= framebuffer(gl, tex);
      gl.bindFramebuffer(gl.FRAMEBUFFER, uploadFb);
      gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, one);
    });
    if (uploadFb) gl.deleteFramebuffer(uploadFb);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    result.decodeUploadFps = round(uploaded.frames / uploaded.seconds, 1);

    log(`${clip.id}: seek accuracy`);
    const seeker = new Seeker(media, config);
    const strip = new Uint8Array(256 * 32 * 4);
    const times: number[] = [];
    const misses: string[] = [];
    let exact = 0;
    const targets = seekTargets(media.presentation.length);
    for (const target of targets) {
      const started = performance.now();
      const frame = await seeker.frame(target);
      times.push(performance.now() - started);
      if (clip.indexStrip) {
        uploadBrowser(gl, tex, frame);
        const fb = framebuffer(gl, tex);
        gl.readPixels(0, 0, 256, 32, gl.RGBA, gl.UNSIGNED_BYTE, strip);
        gl.deleteFramebuffer(fb);
        const read = readIndexStrip(strip, 256);
        if (read === target % 256) exact++; else misses.push(`${target}->${read}`);
      }
      frame.close();
    }
    result.seek = {
      targets: targets.length, exact: clip.indexStrip ? exact : -1, misses,
      meanMs: round(times.reduce((a, b) => a + b, 0) / times.length, 1), maxMs: round(Math.max(...times), 1),
    };

    log(`${clip.id}: colour vs ffmpeg BT.709`);
    const converter = new ExplicitConverter(gl);
    result.colour = [];
    for (const index of [0, Math.floor(media.presentation.length / 2)]) {
      const reference = await backend.referenceFrame(clip.path, index);
      const frame = await seeker.frame(index);
      const width = frame.displayWidth;
      const height = frame.displayHeight;
      const pixels = new Uint8Array(width * height * 4);
      const check: ColourCheck = {
        frame: index,
        frameFormat: frame.format,
        colorSpace: frame.colorSpace?.toJSON?.() ?? null,
        browser: { error: "not run" },
        explicitLeftSited: { error: "not run" },
        explicitCentreSited: { error: "not run" },
      };
      if (reference.length !== pixels.length) {
        check.browser = { error: `reference is ${reference.length} bytes, frame is ${width}x${height}` };
      } else {
        uploadBrowser(gl, tex, frame);
        const fb = framebuffer(gl, tex);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        gl.deleteFramebuffer(fb);
        check.browser = roundCompare(compareRgba(pixels, reference));
        for (const [key, offset] of [["explicitLeftSited", [0.5, 0]], ["explicitCentreSited", [0, 0]]] as const) {
          try {
            await converter.convert(frame, offset as [number, number]);
            check[key] = roundCompare(compareRgba(converter.read(pixels), reference));
          } catch (error) {
            check[key] = { error: String(error) };
          }
        }
      }
      frame.close();
      result.colour.push(check);
    }
    seeker.close();

    result.passDecode = (result.decodeFps ?? 0) >= 30;
    result.passSeek = clip.indexStrip ? result.seek.exact === result.seek.targets : undefined;
    const best = (check: ColourCheck) => Math.min(...[check.browser, check.explicitLeftSited, check.explicitCentreSited]
      .map((value) => ("mae" in value ? value.mae : Infinity)));
    result.passColour = result.colour.every((check) => best(check) <= 1);
  } catch (error) {
    result.error = String(error);
  } finally {
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
  return result;
}

function roundCompare(value: { mae: number; p99: number; max: number }) {
  return { mae: round(value.mae, 3), p99: value.p99, max: value.max };
}

export async function runDecodeSpike(backend: Backend, clips: ClipInfo[], log: (line: string) => void) {
  const results: ClipResult[] = [];
  for (const clip of clips) results.push(await runClip(backend, clip, log));
  return {
    clips: results,
    passBar: "frame-exact seek; mean abs error <= 1/255 vs ffmpeg with explicit BT.709 (best path); >= 30 fps decode at 1080p",
  };
}
