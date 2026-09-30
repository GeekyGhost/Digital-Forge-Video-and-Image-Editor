/**
 * VFX Phase 6: "Convert text to outlines" -- a text layer becomes a path layer that looks the same
 * and sits in the same place, and can then be edited, combined and bent like any path.
 *
 * Positions come from the canvas (`textGlyphRuns`: the same layout the text layer is drawn with);
 * shapes come from the installed font file (`fonts.rs`, via ttf-parser). Each glyph's outline is
 * scaled from font units to the font size, flipped to y-down and set on its baseline at its x.
 * TrueType quadratics become exact cubics. Glyph outlines fill non-zero, as fonts are designed to.
 *
 * What does not carry over, said in the status line rather than silently: the drop shadow and the
 * panel (a text layer's own extras), ligatures the canvas may form (each character becomes its own
 * glyph), and bold or italic that the font does not have and the canvas synthesised.
 */

import type { Paint } from "./photoGradient";
import type { PathBlock, PathNode, Point, Subpath } from "./photoPath";
import { paintsOf, type TextBlock } from "./photoText";

/** As `fonts.rs` serialises it. */
export type FontOutlines = {
  face: { path: string; index: number; family: string; legacyFamily: string; weight: number; italic: boolean };
  unitsPerEm: number;
  ascender: number;
  descender: number;
  glyphs: Array<{ ch: string; advance: number; commands: Array<[string, number[]]>; found: boolean }>;
  substituted: boolean;
  synthetic: boolean;
};

/** One glyph's commands (font units, y up) as subpaths at `x`, `baseline`, `scale` pixels per unit. */
export function glyphSubpaths(commands: Array<[string, number[]]>, scale: number, x: number, baseline: number): Subpath[] {
  const map = (fx: number, fy: number): Point => [x + fx * scale, baseline - fy * scale];
  const subpaths: Subpath[] = [];
  let nodes: PathNode[] = [];
  let pen: Point = [0, 0];
  const finish = (closed: boolean) => {
    if (nodes.length >= 2) {
      // A closing command that returns to the start leaves a duplicate end anchor: fold it in.
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (closed && Math.abs(first.x - last.x) < 1e-6 && Math.abs(first.y - last.y) < 1e-6) {
        if (last.in) first.in = last.in;
        nodes.pop();
      }
      subpaths.push({ closed, nodes });
    }
    nodes = [];
  };
  const to = (p: Point, c1?: Point, c2?: Point) => {
    const previous = nodes[nodes.length - 1];
    if (previous && c1) previous.out = [c1[0] - previous.x, c1[1] - previous.y];
    const node: PathNode = { x: p[0], y: p[1] };
    if (c2) node.in = [c2[0] - p[0], c2[1] - p[1]];
    nodes.push(node);
    pen = p;
  };
  for (const [op, v] of commands) {
    if (op === "M") { finish(false); to(map(v[0], v[1])); }
    else if (op === "L") to(map(v[0], v[1]));
    else if (op === "Q") {
      const q = map(v[0], v[1]), end = map(v[2], v[3]);
      // The exact cubic of a quadratic: control points two thirds of the way to its one control.
      to(end, [pen[0] + (2 / 3) * (q[0] - pen[0]), pen[1] + (2 / 3) * (q[1] - pen[1])], [end[0] + (2 / 3) * (q[0] - end[0]), end[1] + (2 / 3) * (q[1] - end[1])]);
    } else if (op === "C") to(map(v[4], v[5]), map(v[0], v[1]), map(v[2], v[3]));
    else if (op === "Z") finish(true);
  }
  finish(false);
  return subpaths;
}

/**
 * The text's glyphs as one path, around the text bitmap's centre (its layer's origin), painted with
 * the text's fill and outline. The text layer draws its outline 2w wide, centred on the glyph edge,
 * BEFORE the fill, so w shows outside the letters; the path does the same with a 2w stroke and
 * `paintOrder: "stroke"`.
 */
export function textPath(
  text: TextBlock, runs: Array<{ ch: string; x: number; baseline: number }>, size: { width: number; height: number },
  fontSize: number, font: FontOutlines,
): { path: PathBlock; missing: string[] } {
  const scale = fontSize / Math.max(1, font.unitsPerEm);
  const byChar = new Map(font.glyphs.map((glyph) => [glyph.ch, glyph]));
  const subpaths: Subpath[] = [];
  const missing = new Set<string>();
  const cx = size.width / 2, cy = size.height / 2;
  for (const run of runs) {
    const glyph = byChar.get(run.ch);
    if (!glyph || !glyph.found) { if (run.ch.trim()) missing.add(run.ch); continue; }
    subpaths.push(...glyphSubpaths(glyph.commands, scale, run.x - cx, run.baseline - cy));
  }
  const paints = paintsOf(text);
  const stroke: Paint | undefined = text.strokeWidth > 0 ? paints.stroke : undefined;
  return {
    path: {
      subpaths, fillRule: "nonzero", fill: paints.fill, stroke,
      strokeWidth: text.strokeWidth > 0 ? text.strokeWidth * 2 : 0,
      join: "round", cap: "round", miterLimit: 2, paintOrder: "stroke",
    },
    missing: [...missing],
  };
}

/** The first family of a CSS list, unquoted, and the generic the list (or the name) falls back to. */
export function fontRequest(fontFamily: string, genericOf: (family: string) => string): { family: string; generic: string } {
  const parts = fontFamily.split(",").map((part) => part.trim().replace(/^["']|["']$/g, "")).filter(Boolean);
  const generics = ["serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui"];
  const family = parts.find((part) => !generics.includes(part)) ?? parts[0] ?? "Arial";
  const generic = parts.find((part) => generics.includes(part)) ?? genericOf(family);
  return { family, generic };
}
