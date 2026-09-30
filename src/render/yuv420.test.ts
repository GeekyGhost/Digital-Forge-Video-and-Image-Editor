import { spawnSync } from "node:child_process";
import { describe, expect, it } from "vitest";
import { packYuv420, yuv420Size } from "./yuv420";

function frame(width: number, height: number, noise: boolean, seed = 1): Uint8Array {
  const rgba = new Uint8Array(width * height * 4);
  let state = seed;
  const random = () => { state = (state * 1664525 + 1013904223) >>> 0; return state / 2 ** 32; };
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      // Gradients on the left, gradients + strong grain on the right half.
      const grain = noise && x >= width / 2 ? () => (random() - 0.5) * 120 : () => 0;
      rgba[i] = Math.max(0, Math.min(255, Math.round((x / (width - 1)) * 255 + grain())));
      rgba[i + 1] = Math.max(0, Math.min(255, Math.round((y / (height - 1)) * 255 + grain())));
      rgba[i + 2] = Math.max(0, Math.min(255, Math.round(((x + 3 * y) % 256) + grain())));
      rgba[i + 3] = 255;
    }
  }
  return rgba;
}

/** ffmpeg's own conversion, exactly as ai/render.rs asks for it on the RGBA path. */
function ffmpegYuv(rgba: Uint8Array, width: number, height: number): Uint8Array | undefined {
  const run = spawnSync("ffmpeg", [
    "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", `${width}x${height}`, "-i", "-",
    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_color_matrix=bt709:out_range=tv:flags=area+accurate_rnd,format=yuv420p",
    "-f", "rawvideo", "-pix_fmt", "yuv420p", "-",
  ], { input: rgba, maxBuffer: 1 << 28 });
  if (run.error || run.status !== 0) return undefined;
  return new Uint8Array(run.stdout.buffer, run.stdout.byteOffset, run.stdout.byteLength);
}

const hasFfmpeg = !spawnSync("ffmpeg", ["-version"]).error;

describe("packYuv420", () => {
  it("converts known colours with the BT.709 limited-range matrix", () => {
    const colours = [[255, 255, 255, 255], [0, 0, 0, 255], [255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [200, 100, 50, 0]];
    const expected = [[235, 128, 128], [16, 128, 128], [63, 102, 240], [173, 42, 26], [32, 240, 118], [16, 128, 128]];
    colours.forEach((colour, index) => {
      const out = packYuv420(new Uint8Array([...colour, ...colour, ...colour, ...colour]), 2, 2);
      expect([out[0], out[4], out[5]], `colour ${colour}`).toEqual(expected[index]);
      expect(out.length).toBe(6);
    });
  });

  it("pads odd sizes with black and averages each 2x2 block", () => {
    expect(yuv420Size(961, 543)).toEqual({ width: 962, height: 544, bytes: (962 * 544 * 3) / 2 });
    const white = new Uint8Array(3 * 3 * 4).fill(255);
    const out = packYuv420(white, 3, 3);
    expect(out.length).toBe(4 * 4 * 1.5);
    expect([...out.subarray(0, 16)]).toEqual([235, 235, 235, 16, 235, 235, 235, 16, 235, 235, 235, 16, 16, 16, 16, 16]);
    // Chroma of white and black is neutral whatever the average.
    expect([...out.subarray(16)]).toEqual([128, 128, 128, 128, 128, 128, 128, 128]);
  });

  it("reuses the buffer it is given", () => {
    const buffer = new Uint8Array(100);
    const out = packYuv420(new Uint8Array(16).fill(255), 2, 2, buffer);
    expect(out.buffer).toBe(buffer.buffer);
    expect(out.length).toBe(6);
  });

  it.skipIf(!hasFfmpeg)("matches ffmpeg's own BT.709 conversion to 1/255 on gradients and grain", () => {
    for (const [width, height] of [[960, 544], [321, 179]]) {
      const rgba = frame(width, height, true);
      // Some fully transparent pixels: the page blacks them (encodableRgba) before either path sees them.
      for (let i = 0; i < 400; i += 4) { rgba[i + 3] = 0; rgba[i] = 0; rgba[i + 1] = 0; rgba[i + 2] = 0; }
      const ours = packYuv420(rgba, width, height);
      const theirs = ffmpegYuv(rgba, width, height)!;
      expect(theirs.length).toBe(ours.length);
      let worst = 0, off = 0;
      for (let i = 0; i < ours.length; i++) {
        const d = Math.abs(ours[i] - theirs[i]);
        if (d > worst) worst = d;
        if (d) off++;
      }
      expect(worst, `${width}x${height}: largest byte difference`).toBeLessThanOrEqual(1);
      expect(off / ours.length, `${width}x${height}: share of bytes off by one`).toBeLessThan(0.01);
    }
  });
});
