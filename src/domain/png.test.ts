import { describe, expect, it } from "vitest";
import { inflateSync, crc32 } from "node:zlib";
import { encodePng } from "./png";

describe("exact RGBA PNG encoding (VFX Phase 5)", () => {
  it("round-trips every byte, soft alpha included, with valid chunk CRCs", async () => {
    const width = 7, height = 5;
    const rgba = new Uint8ClampedArray(width * height * 4).map((_, index) => (index * 37 + 11) % 256);
    rgba[3] = 1; rgba[7] = 0; rgba[11] = 10;
    const png = await encodePng(rgba, width, height);
    expect(Array.from(png.subarray(0, 8))).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    const view = new DataView(png.buffer, png.byteOffset);
    let at = 8;
    const chunks: Record<string, Uint8Array> = {};
    while (at < png.length) {
      const length = view.getUint32(at);
      const type = String.fromCharCode(...png.subarray(at + 4, at + 8));
      const body = png.subarray(at + 8, at + 8 + length);
      expect(view.getUint32(at + 8 + length)).toBe(crc32(png.subarray(at + 4, at + 8 + length)) >>> 0);
      chunks[type] = body;
      at += 12 + length;
    }
    expect(Object.keys(chunks)).toEqual(["IHDR", "IDAT", "IEND"]);
    expect(Array.from(chunks.IHDR)).toEqual([0, 0, 0, 7, 0, 0, 0, 5, 8, 6, 0, 0, 0]);
    const raw = inflateSync(chunks.IDAT);
    for (let y = 0; y < height; y++) {
      expect(raw[y * (width * 4 + 1)]).toBe(0);
      expect(Array.from(raw.subarray(y * (width * 4 + 1) + 1, (y + 1) * (width * 4 + 1)))).toEqual(Array.from(rgba.subarray(y * width * 4, (y + 1) * width * 4)));
    }
    await expect(encodePng(rgba, 8, 5)).rejects.toThrow("needs 160 bytes");
  });
});
