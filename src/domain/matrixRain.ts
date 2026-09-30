/**
 * Matrix rain (VFX Phase 3): the first generator layer, and the CPU reference for MATRIX_RAIN_FS.
 *
 * Everything is integer arithmetic on a 32-bit hash, so the GPU (uint maths, which wraps the same
 * way) produces the same bytes, and the rain is a pure function of (seed, frame): a preview frame and
 * a saved frame are the same picture, and frame `loopFrames` is frame 0 again, exactly.
 *
 * Per column: whether it rains, how many whole screens it falls per loop (an integer, which is what
 * makes the loop exact), an offset, and a gap so columns do not march in step. Per cell: a glyph that
 * changes `glyphChanges` times per loop, each cell at its own phase. The trail fades linearly from the
 * head, which takes the head colour.
 *
 * The glyphs come from an atlas the CPU draws once per cell size (`buildRainAtlas`); both compositors
 * read the same atlas bytes.
 */

import type { MatrixRainGenerator } from "./photoDocument";

/** Coverage of every glyph, side by side: `count` cells of `cell` x `cell`, one byte per pixel. */
export type RainAtlas = { cell: number; count: number; width: number; coverage: Uint8Array };

/** Half-width katakana, digits and a few marks: the film's set, near enough. */
export const RAIN_GLYPHS = [
  ...Array.from({ length: 0xff9d - 0xff66 + 1 }, (_, i) => String.fromCharCode(0xff66 + i)),
  ..."012345789Z:.=*+-<>|".split(""),
];

export function mix32(value: number): number {
  let x = value | 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
}

/** hash(a, b, c, d) -- the GLSL `hash4` mirrors this bit for bit. */
export function hash4(a: number, b: number, c: number, d: number): number {
  return mix32(a ^ mix32(b ^ mix32(c ^ mix32((d + 0x9e3779b9) | 0))));
}

/** Integer layout shared by both compositors, for one render size. */
export function rainLayout(generator: MatrixRainGenerator, width: number, height: number, canvasScale: number) {
  const cell = Math.max(4, Math.round(generator.cellSize * canvasScale));
  const columns = Math.ceil(width / cell);
  const rows = Math.ceil(height / cell);
  const loop = Math.max(1, Math.round(generator.loopFrames));
  const trail = Math.max(1, Math.round(generator.trail));
  const minPasses = Math.max(1, Math.round(generator.minPasses));
  const maxPasses = Math.max(minPasses, Math.round(generator.maxPasses));
  return {
    cell, columns, rows, loop, trail, minPasses, maxPasses,
    seed: (Math.round(generator.seed) >>> 0),
    density: Math.round(Math.min(1, Math.max(0, generator.density)) * 1000),
    changes: Math.max(0, Math.round(generator.glyphChanges)),
  };
}

export function hexBytes(hex: string): [number, number, number] {
  const value = /^#?([0-9a-f]{6})$/i.exec(hex.trim())?.[1] ?? "ffffff";
  return [parseInt(value.slice(0, 2), 16), parseInt(value.slice(2, 4), 16), parseInt(value.slice(4, 6), 16)];
}

/** Render one frame into `out` (straight RGBA, fully overwritten). */
export function renderMatrixRain(
  out: Uint8ClampedArray, width: number, height: number, generator: MatrixRainGenerator,
  frame: number, canvasScale: number, atlas: RainAtlas,
): void {
  const l = rainLayout(generator, width, height, canvasScale);
  const colour = hexBytes(generator.colour), head = hexBytes(generator.headColour);
  const inLoop = ((Math.floor(frame) % l.loop) + l.loop) % l.loop;
  out.fill(0);
  if (atlas.cell !== l.cell || atlas.count < 1) return;
  for (let column = 0; column < l.columns; column++) {
    const h = hash4(l.seed, column, 0, 0);
    if (h % 1000 >= l.density) continue;
    const cycle = l.rows + l.trail + hash4(l.seed, column, 1, 0) % Math.max(1, l.rows);
    const passes = l.minPasses + hash4(l.seed, column, 2, 0) % (l.maxPasses - l.minPasses + 1);
    const offset = hash4(l.seed, column, 3, 0) % cycle;
    const headRow = (Math.floor((inLoop * passes * cycle) / l.loop) + offset) % cycle;
    for (let row = 0; row < l.rows; row++) {
      const distance = (headRow - row + cycle) % cycle;
      if (distance >= l.trail) continue;
      const level = Math.floor(((l.trail - distance) * 255) / l.trail);
      const phase = hash4(l.seed, column, row, 4) % l.loop;
      const epoch = Math.floor((inLoop * l.changes + phase) / l.loop);
      const glyph = hash4(l.seed ^ 0x51ed, column, row, epoch) % atlas.count;
      const [r, g, b] = distance === 0 ? head : colour;
      const x0 = column * l.cell, y0 = row * l.cell;
      for (let ly = 0; ly < l.cell && y0 + ly < height; ly++) {
        for (let lx = 0; lx < l.cell && x0 + lx < width; lx++) {
          const coverage = atlas.coverage[ly * atlas.width + glyph * l.cell + lx];
          const alpha = Math.floor((coverage * level + 127) / 255);
          if (alpha === 0) continue;
          const at = ((y0 + ly) * width + x0 + lx) * 4;
          out[at] = r; out[at + 1] = g; out[at + 2] = b; out[at + 3] = alpha;
        }
      }
    }
  }
}

const atlases = new Map<string, RainAtlas>();

/** Draw (once per size) the glyph atlas. Needs a DOM canvas; tests pass their own atlas. */
export function buildRainAtlas(cell: number, mirror: boolean): RainAtlas {
  const key = `${cell}|${mirror}`;
  const held = atlases.get(key);
  if (held) return held;
  const count = RAIN_GLYPHS.length;
  const canvas = window.document.createElement("canvas");
  canvas.width = cell * count;
  canvas.height = cell;
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.fillStyle = "#ffffff";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.font = `${Math.max(3, Math.round(cell * 0.86))}px "MS Gothic", "Yu Gothic", "Meiryo", "Noto Sans CJK JP", monospace`;
  RAIN_GLYPHS.forEach((glyph, index) => {
    context.save();
    context.translate(index * cell + cell / 2, cell / 2);
    if (mirror) context.scale(-1, 1);
    context.fillText(glyph, 0, 0);
    context.restore();
  });
  const data = context.getImageData(0, 0, canvas.width, cell).data;
  const coverage = new Uint8Array(canvas.width * cell);
  for (let index = 0; index < coverage.length; index++) coverage[index] = data[index * 4 + 3];
  const atlas = { cell, count, width: canvas.width, coverage };
  if (atlases.size > 8) atlases.clear();
  atlases.set(key, atlas);
  return atlas;
}
