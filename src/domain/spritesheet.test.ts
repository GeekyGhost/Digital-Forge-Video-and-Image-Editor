import { describe, expect, it } from "vitest";
import { packAtlas, sheetFrames, sheetJson, sheetLayout, sheetSlug, spriteFramesTres, unionAlphaBounds } from "./spritesheet";

/** A w x h RGBA frame, transparent except `rect`, filled with (value, index, 0, 255). */
function frame(w: number, h: number, rect: [number, number, number, number], value: number) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let y = rect[1]; y < rect[1] + rect[3]; y++) for (let x = rect[0]; x < rect[0] + rect[2]; x++) data.set([value, x, y, 255], (y * w + x) * 4);
  return data;
}

describe("spritesheet export (VFX Phase 5)", () => {
  it("lays cells out near-square with padding between and around, within the width limit", () => {
    const layout = sheetLayout(10, 64, 32, 2);
    expect(layout.columns).toBe(3); // ceil(sqrt(10 * 32 / 64)) = 3
    expect(layout.rows).toBe(4);
    expect(layout.cells[0]).toEqual({ x: 2, y: 2, width: 64, height: 32 });
    expect(layout.cells[4]).toEqual({ x: 2 + 66, y: 2 + 34, width: 64, height: 32 });
    expect(layout.width).toBe(2 + 3 * 66);
    expect(layout.height).toBe(2 + 4 * 34);
    expect(sheetLayout(100, 1000, 1000, 2, 4096).columns).toBe(4);
    expect(() => sheetLayout(0, 10, 10)).toThrow();
  });

  it("trims every frame to the union of their alpha, so the pivot does not move", () => {
    const frames = [frame(20, 10, [5, 2, 3, 3], 10), frame(20, 10, [8, 4, 6, 5], 20)];
    expect(unionAlphaBounds(frames, 20, 10)).toEqual({ x: 5, y: 2, width: 9, height: 7 });
    expect(unionAlphaBounds([new Uint8ClampedArray(20 * 10 * 4)], 20, 10)).toBeUndefined();
  });

  it("copies each frame's crop into its cell exactly, and nothing else", () => {
    const frames = [frame(20, 10, [5, 2, 3, 3], 10), frame(20, 10, [8, 4, 6, 5], 20), frame(20, 10, [5, 2, 1, 1], 30)];
    const crop = unionAlphaBounds(frames, 20, 10)!;
    const layout = sheetLayout(frames.length, crop.width, crop.height, 1);
    const atlas = packAtlas(frames, 20, crop, layout);
    let opaque = 0;
    for (let index = 3; index < atlas.length; index += 4) if (atlas[index]) opaque++;
    expect(opaque).toBe(9 + 30 + 1);
    frames.forEach((source, index) => {
      const cell = layout.cells[index];
      for (let y = 0; y < crop.height; y++) for (let x = 0; x < crop.width; x++) {
        const from = ((crop.y + y) * 20 + crop.x + x) * 4, to = ((cell.y + y) * layout.width + cell.x + x) * 4;
        expect(Array.from(atlas.subarray(to, to + 4))).toEqual(Array.from(source.subarray(from, from + 4)));
      }
    });
  });

  it("writes the SpriteFrames text Godot 4's own saver writes", () => {
    const text = spriteFramesTres({ texturePath: "hero.png", fps: 24, loop: true, regions: [{ x: 2, y: 2, width: 64, height: 32 }, { x: 68, y: 2, width: 64, height: 32 }] });
    const lines = text.split("\n");
    expect(lines[0]).toBe('[gd_resource type="SpriteFrames" load_steps=4 format=3]');
    expect(lines[1]).toBe("");
    expect(lines[2]).toMatch(/^\[ext_resource type="Texture2D" path="hero\.png" id="1_[a-z0-9]{5}"\]$/);
    const textureId = /id="([^"]+)"/.exec(lines[2])![1];
    const subs = [...text.matchAll(/\[sub_resource type="AtlasTexture" id="([^"]+)"\]\natlas = ExtResource\("([^"]+)"\)\nregion = Rect2\(([^)]*)\)/g)];
    expect(subs.map((match) => [match[2], match[3]])).toEqual([[textureId, "2, 2, 64, 32"], [textureId, "68, 2, 64, 32"]]);
    expect(new Set(subs.map((match) => match[1])).size).toBe(2);
    const resource = text.slice(text.indexOf("[resource]"));
    expect(resource).toBe(`[resource]\nanimations = [{\n"frames": [{\n"duration": 1.0,\n"texture": SubResource("${subs[0][1]}")\n}, {\n"duration": 1.0,\n"texture": SubResource("${subs[1][1]}")\n}],\n"loop": true,\n"name": &"default",\n"speed": 24.0\n}]\n`);
    expect(spriteFramesTres({ texturePath: "a.png", fps: 12.5, loop: false, regions: [{ x: 0, y: 0, width: 1, height: 1 }] }))
      .toContain('"loop": false,\n"name": &"default",\n"speed": 12.5\n');
    expect(spriteFramesTres({ texturePath: "a.png", fps: 12.5, loop: false, regions: [{ x: 0, y: 0, width: 1, height: 1 }] }).startsWith("[gd_resource type=\"SpriteFrames\" load_steps=3 format=3]")).toBe(true);
  });

  it("describes the sheet in JSON and picks frames and names", () => {
    const layout = sheetLayout(2, 8, 8, 1);
    const json = JSON.parse(sheetJson({ image: "hero.png", layout, fps: 24, loop: true, sourceSize: { width: 20, height: 10 }, trim: { x: 5, y: 2, width: 8, height: 8 }, source: "Hero", frames: [0, 2] }));
    expect(json.frames[1]).toMatchObject({ frame: { x: 10, y: 1, w: 8, h: 8 }, trimmed: true, spriteSourceSize: { x: 5, y: 2, w: 8, h: 8 }, compositionFrame: 2, duration: 42 });
    expect(json.meta).toMatchObject({ image: "hero.png", size: { w: layout.width, h: layout.height }, columns: 2 });
    expect(sheetFrames(0, 9, 3)).toEqual([0, 3, 6, 9]);
    expect(sheetSlug("  Hero Knight / walk!! ")).toBe("hero-knight-walk");
    expect(sheetSlug("***")).toBe("spritesheet");
  });
});
