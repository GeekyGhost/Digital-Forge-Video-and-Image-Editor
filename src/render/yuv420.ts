/**
 * VFX Phase 6: the render's RGB -> YUV conversion, done on the page instead of in ffmpeg.
 *
 * A render used to send 4 bytes per pixel of RGBA and let ffmpeg's `scale` filter convert. Packing
 * planar I420 here sends 1.5 bytes per pixel (2.7x fewer), and the transport was the measured limit
 * (Phase 5: ~169 ms per 960x544 RGBA frame over Tauri IPC).
 *
 * The conversion is the one the RGBA path asks ffmpeg for, byte for byte (a test holds both to 1/255):
 * - BT.709 matrix, limited range (Y 16..235, Cb/Cr 16..240), on the gamma-encoded bytes;
 * - chroma = the plain average of each 2x2 block (centre-sited), which is what ffmpeg's `scale` does
 *   with `flags=area+accurate_rnd` (ai/render.rs asks for exactly that on the RGBA path);
 * - odd sizes are padded to even with black on the right/bottom, as the RGBA path's `pad` filter does;
 * - a fully transparent pixel counts as black (`encodableRgba`: what the old canvas round trip made
 *   of it); every other pixel's alpha is ignored, as ffmpeg ignores it.
 */

const KR = 0.2126;
const KB = 0.0722;
const KG = 1 - KR - KB;

/** Luma LUTs: 219 * K * v / 255 for each channel, so Y = 16 + lr[r] + lg[g] + lb[b]. */
const LUMA_R = new Float64Array(256);
const LUMA_G = new Float64Array(256);
const LUMA_B = new Float64Array(256);
for (let v = 0; v < 256; v++) {
  LUMA_R[v] = (219 * KR * v) / 255;
  LUMA_G[v] = (219 * KG * v) / 255;
  LUMA_B[v] = (219 * KB * v) / 255;
}

// Cb = 128 + 224 (B - L) / 1.8556, Cr = 128 + 224 (R - L) / 1.5748, on 0..1 values: as coefficients of
// the SUM of a 2x2 block's bytes (so / 4 / 255 are folded in).
const CB_SCALE = 224 / (2 * (1 - KB)) / (4 * 255);
const CR_SCALE = 224 / (2 * (1 - KR)) / (4 * 255);
const CB_R = -KR * CB_SCALE, CB_G = -KG * CB_SCALE, CB_B = (1 - KB) * CB_SCALE;
const CR_R = (1 - KR) * CR_SCALE, CR_G = -KG * CR_SCALE, CR_B = -KB * CR_SCALE;

/** The even size a 4:2:0 frame of `width` x `height` is sent at. */
export function yuv420Size(width: number, height: number): { width: number; height: number; bytes: number } {
  const w = width + (width & 1);
  const h = height + (height & 1);
  return { width: w, height: h, bytes: (w * h * 3) / 2 };
}

/**
 * Packs straight RGBA (`width` x `height`, row 0 at the top) into planar I420 (Y, then Cb, then Cr)
 * at the even size of `yuv420Size`. Writes into `out` when it is large enough, else allocates.
 */
export function packYuv420(rgba: Uint8Array | Uint8ClampedArray, width: number, height: number, out?: Uint8Array): Uint8Array {
  if (rgba.length < width * height * 4) throw new Error(`a ${width}x${height} frame needs ${width * height * 4} bytes of RGBA, got ${rgba.length}`);
  const size = yuv420Size(width, height);
  const W = size.width, H = size.height;
  const target = out && out.length >= size.bytes ? out.subarray(0, size.bytes) : new Uint8Array(size.bytes);
  const cw = W >> 1;
  const uBase = W * H;
  const vBase = uBase + cw * (H >> 1);
  const rowBytes = width * 4;
  const evenW = width & ~1, evenH = height & ~1;
  // The interior (every 2x2 block inside the frame), unrolled; fully transparent pixels are black.
  for (let y0 = 0; y0 < evenH; y0 += 2) {
    let i0 = y0 * rowBytes, i1 = i0 + rowBytes;
    let o0 = y0 * W, o1 = o0 + W;
    let c = (y0 >> 1) * cw;
    for (let x = 0; x < evenW; x += 2, i0 += 8, i1 += 8, o0 += 2, o1 += 2, c++) {
      const a = rgba[i0 + 3] ? 1 : 0, b = rgba[i0 + 7] ? 1 : 0, d = rgba[i1 + 3] ? 1 : 0, e = rgba[i1 + 7] ? 1 : 0;
      const r0 = rgba[i0] * a, g0 = rgba[i0 + 1] * a, b0 = rgba[i0 + 2] * a;
      const r1 = rgba[i0 + 4] * b, g1 = rgba[i0 + 5] * b, b1 = rgba[i0 + 6] * b;
      const r2 = rgba[i1] * d, g2 = rgba[i1 + 1] * d, b2 = rgba[i1 + 2] * d;
      const r3 = rgba[i1 + 4] * e, g3 = rgba[i1 + 5] * e, b3 = rgba[i1 + 6] * e;
      target[o0] = (16.5 + LUMA_R[r0] + LUMA_G[g0] + LUMA_B[b0]) | 0;
      target[o0 + 1] = (16.5 + LUMA_R[r1] + LUMA_G[g1] + LUMA_B[b1]) | 0;
      target[o1] = (16.5 + LUMA_R[r2] + LUMA_G[g2] + LUMA_B[b2]) | 0;
      target[o1 + 1] = (16.5 + LUMA_R[r3] + LUMA_G[g3] + LUMA_B[b3]) | 0;
      const sr = r0 + r1 + r2 + r3, sg = g0 + g1 + g2 + g3, sb = b0 + b1 + b2 + b3;
      target[uBase + c] = (128.5 + CB_R * sr + CB_G * sg + CB_B * sb) | 0;
      target[vBase + c] = (128.5 + CR_R * sr + CR_G * sg + CR_B * sb) | 0;
    }
  }
  // The padded right column and bottom row of an odd size: blocks that reach outside the frame.
  if (width & 1 || height & 1) {
    for (let cy = 0; cy < H >> 1; cy++) {
      for (let cx = 0; cx < cw; cx++) {
        const x0 = cx * 2, y0 = cy * 2;
        if (x0 + 1 < width && y0 + 1 < height) continue;
        let sr = 0, sg = 0, sb = 0;
        for (let k = 0; k < 4; k++) {
          const x = x0 + (k & 1), y = y0 + (k >> 1);
          let r = 0, g = 0, b = 0;
          if (x < width && y < height) {
            const i = y * rowBytes + x * 4;
            if (rgba[i + 3] !== 0) { r = rgba[i]; g = rgba[i + 1]; b = rgba[i + 2]; }
          }
          target[y * W + x] = (16.5 + LUMA_R[r] + LUMA_G[g] + LUMA_B[b]) | 0;
          sr += r; sg += g; sb += b;
        }
        const c = cy * cw + cx;
        target[uBase + c] = (128.5 + CB_R * sr + CB_G * sg + CB_B * sb) | 0;
        target[vBase + c] = (128.5 + CR_R * sr + CR_G * sg + CR_B * sb) | 0;
      }
    }
  }
  return target;
}
