/**
 * VFX Phase 5: spritesheet export (the plan's "separate spritesheet pipeline becomes an export target
 * of the compositor"). Frames rendered by the compositor are packed into one RGBA atlas, with a Godot 4
 * `SpriteFrames` resource and a JSON sidecar. Pure: the page renders, this packs and writes text, Rust
 * saves the files.
 *
 * The `.tres` text follows Godot 4.5's own saver (scene/resources/resource_format_text.cpp and
 * core/variant/variant_parser.cpp at 4.5-stable, read 2026-09-29), not a guess:
 * - header `[gd_resource type="SpriteFrames" load_steps=N format=3]`, where N = saved resources +
 *   external resources, and the saved ones are the sub-resources PLUS the main resource: frames + 2
 *   here (written only when N > 1); format 3 is what every Godot 4.x loads (4.5 writes 4 only for
 *   syntax it needs);
 * - `[ext_resource type="Texture2D" path="..." id="1_..."]`: a RELATIVE path is resolved against the
 *   .tres file's own folder (`local_path.get_base_dir().path_join(path)` in the loader), so the export
 *   folder can be dropped anywhere in a Godot project;
 * - one `[sub_resource type="AtlasTexture" id="AtlasTexture_..."]` per frame with `atlas` and a
 *   `region = Rect2(x, y, w, h)` (Rect2 components written without a forced ".0");
 * - `[resource]` with `animations = [{ ... }]`: dictionary keys sorted ("frames", "loop", "name",
 *   "speed"; "duration", "texture"), floats always with a decimal point, the name a StringName
 *   `&"..."` (SpriteFrames::_get_animations / _set_animations).
 */

export type Rect = { x: number; y: number; width: number; height: number };

export type SheetLayout = {
  columns: number;
  rows: number;
  width: number;
  height: number;
  /** Each frame's region in the atlas, in order. */
  cells: Rect[];
};

/**
 * A grid of `count` cells of cellWidth x cellHeight, `padding` pixels between cells and around the
 * edge (so linear filtering never samples a neighbour). As square as the frame count allows, but never
 * wider than `maxWidth` (a texture-size limit, not a preference).
 */
export function sheetLayout(count: number, cellWidth: number, cellHeight: number, padding = 2, maxWidth = 8192): SheetLayout {
  if (count < 1 || cellWidth < 1 || cellHeight < 1) throw new Error("a spritesheet needs at least one frame with a size");
  const fit = Math.max(1, Math.floor((maxWidth - padding) / (cellWidth + padding)));
  // Near-square in pixels: columns ~ sqrt(count * h / w).
  const columns = Math.max(1, Math.min(count, fit, Math.ceil(Math.sqrt(count * cellHeight / cellWidth))));
  const rows = Math.ceil(count / columns);
  const cells: Rect[] = [];
  for (let index = 0; index < count; index++) {
    const column = index % columns, row = Math.floor(index / columns);
    cells.push({ x: padding + column * (cellWidth + padding), y: padding + row * (cellHeight + padding), width: cellWidth, height: cellHeight });
  }
  return { columns, rows, width: padding + columns * (cellWidth + padding), height: padding + rows * (cellHeight + padding), cells };
}

/**
 * The smallest rectangle holding every non-transparent pixel of every frame: one trim for all frames,
 * so the sprite's pivot stays put from frame to frame. Undefined when every frame is empty.
 */
export function unionAlphaBounds(frames: Array<Uint8Array | Uint8ClampedArray>, width: number, height: number): Rect | undefined {
  let left = width, top = height, right = -1, bottom = -1;
  for (const frame of frames) {
    for (let y = 0; y < height; y++) {
      const row = y * width * 4;
      for (let x = 0; x < width; x++) {
        if (frame[row + x * 4 + 3] !== 0) {
          if (x < left) left = x;
          if (x > right) right = x;
          if (y < top) top = y;
          if (y > bottom) bottom = y;
        }
      }
    }
  }
  return right < 0 ? undefined : { x: left, y: top, width: right - left + 1, height: bottom - top + 1 };
}

/** Copy `source` (the `crop` of each width x height frame) into its cell. Everything else stays transparent. */
export function packAtlas(frames: Array<Uint8Array | Uint8ClampedArray>, width: number, crop: Rect, layout: SheetLayout): Uint8ClampedArray {
  const atlas = new Uint8ClampedArray(layout.width * layout.height * 4);
  frames.forEach((frame, index) => {
    const cell = layout.cells[index];
    for (let y = 0; y < crop.height; y++) {
      const from = ((crop.y + y) * width + crop.x) * 4;
      atlas.set(frame.subarray(from, from + crop.width * 4), ((cell.y + y) * layout.width + cell.x) * 4);
    }
  });
  return atlas;
}

/** Godot's float literal: always with a decimal point (variant_parser.cpp appends ".0"). */
const godotFloat = (value: number) => {
  const text = String(Number(value.toPrecision(9)));
  return /[.eE]/.test(text) || !Number.isFinite(value) ? text : `${text}.0`;
};

/** Godot's string escaping for a quoted literal (c_escape: backslash, quote, newline, tab). */
const godotString = (text: string) => text.replace(/\\/g, "\\\\").replace(/"/g, "\\\"").replace(/\n/g, "\\n").replace(/\t/g, "\\t");

/** A stable id suffix like Godot's generated ones (5 lowercase letters/digits), derived from `seed`. */
function uniqueSuffix(seed: string): string {
  let hash = 2166136261;
  for (const char of seed) hash = Math.imul(hash ^ char.charCodeAt(0), 16777619) >>> 0;
  const alphabet = "abcdefghijklmnopqrstuvwxyz0123456789";
  let out = "";
  for (let index = 0; index < 5; index++) { out += alphabet[hash % 36]; hash = Math.floor(hash / 36) || (hash ^ 0x9e3779b9) >>> 0; }
  return out;
}

/**
 * A Godot 4 SpriteFrames resource: one animation of `regions` (in atlas pixels) of the texture at
 * `texturePath` (relative to the .tres), `fps` frames per second, looping or not.
 */
export function spriteFramesTres(options: { texturePath: string; animation?: string; fps: number; loop: boolean; regions: Rect[] }): string {
  const { regions } = options;
  if (!regions.length) throw new Error("a SpriteFrames resource needs at least one frame");
  const base = uniqueSuffix(`${options.texturePath}|${regions.length}`);
  const textureId = `1_${base}`;
  const lines: string[] = [];
  const steps = regions.length + 1 /* the SpriteFrames itself */ + 1 /* the texture */;
  lines.push(`[gd_resource type="SpriteFrames" load_steps=${steps} format=3]`, "");
  lines.push(`[ext_resource type="Texture2D" path="${godotString(options.texturePath)}" id="${textureId}"]`, "");
  const ids = regions.map((_, index) => `AtlasTexture_${uniqueSuffix(`${base}|${index}`)}${index}`);
  regions.forEach((region, index) => {
    lines.push(`[sub_resource type="AtlasTexture" id="${ids[index]}"]`);
    lines.push(`atlas = ExtResource("${textureId}")`);
    lines.push(`region = Rect2(${region.x}, ${region.y}, ${region.width}, ${region.height})`, "");
  });
  lines.push("[resource]");
  const frames = ids.map((id) => `{\n"duration": 1.0,\n"texture": SubResource("${id}")\n}`).join(", ");
  lines.push(`animations = [{\n"frames": [${frames}],\n"loop": ${options.loop ? "true" : "false"},\n"name": &"${godotString(options.animation ?? "default")}",\n"speed": ${godotFloat(options.fps)}\n}]`);
  return `${lines.join("\n")}\n`;
}

/** A JSON sidecar in the common "frames + meta" shape, for engines other than Godot. */
export function sheetJson(options: {
  image: string; layout: SheetLayout; fps: number; loop: boolean; sourceSize: { width: number; height: number }; trim?: Rect;
  source: string; frames: number[];
}): string {
  const trim = options.trim ?? { x: 0, y: 0, width: options.sourceSize.width, height: options.sourceSize.height };
  return JSON.stringify({
    frames: options.layout.cells.map((cell, index) => ({
      frame: { x: cell.x, y: cell.y, w: cell.width, h: cell.height },
      trimmed: Boolean(options.trim),
      spriteSourceSize: { x: trim.x, y: trim.y, w: trim.width, h: trim.height },
      sourceSize: { w: options.sourceSize.width, h: options.sourceSize.height },
      duration: Math.round(1000 / options.fps),
      compositionFrame: options.frames[index],
    })),
    meta: {
      app: "Digital Forge Video Studio", image: options.image, format: "RGBA8888",
      size: { w: options.layout.width, h: options.layout.height }, fps: options.fps, loop: options.loop,
      columns: options.layout.columns, rows: options.layout.rows, source: options.source,
    },
  }, null, 2);
}

/** Composition frames first..last (inclusive) every `step`. */
export function sheetFrames(first: number, last: number, step = 1): number[] {
  const frames: number[] = [];
  for (let at = Math.max(0, Math.floor(first)); at <= Math.floor(last); at += Math.max(1, Math.floor(step))) frames.push(at);
  return frames;
}

/** A file-system-safe name for the export folder and files. */
export const sheetSlug = (name: string) =>
  name.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 40) || "spritesheet";
