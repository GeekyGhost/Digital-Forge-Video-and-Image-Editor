/**
 * Text as a layer.
 *
 * A text layer is a picture the app draws instead of one it loads, and that is the whole design.
 * Everything a layer already has -- placement, rotation, opacity, blend mode, the edge group, its
 * own stack of grade and relight nodes -- applies to it unchanged, because by the time the
 * compositor sees it, it is a bitmap of a known size like any other. Nothing here knows about
 * layers; it knows about laying out lines.
 *
 * Layout is separated from drawing and takes a measuring function rather than a canvas, so the part
 * that decides where words break can be tested without a browser. The part that puts ink down lives
 * in `photoCompose.ts` next to everything else that touches a context.
 */

import { solidPaint, toPaint, type Paint } from "./photoGradient";

export type TextAlign = "left" | "center" | "right";
export type TextPanel = "none" | "box" | "rounded";

export type TextBlock = {
  content: string;
  /** A CSS family list. The first name that the machine actually has is what gets used. */
  fontFamily: string;
  /** Cap height in canvas pixels, so text scales with the picture rather than with the window. */
  fontSize: number;
  weight: number;
  italic: boolean;
  /** A flat colour or a whole ramp. Older documents hold a plain string; `paintsOf` reads both. */
  color: Paint | string;
  align: TextAlign;
  /** A multiple of the font size. 1.25 is a comfortable default for display type. */
  lineHeight: number;
  /** A fraction of the font size, positive to open the text up. */
  letterSpacing: number;
  /** The width text wraps inside, in canvas pixels. */
  boxWidth: number;
  padding: number;
  /** Outline drawn behind the fill, in canvas pixels. 0 is off. */
  strokeWidth: number;
  strokeColor: Paint | string;
  shadow: number;
  shadowColor: Paint | string;
  panel: TextPanel;
  panelColor: Paint | string;
  panelBorder: number;
  panelBorderColor: Paint | string;
  /**
   * Shrink the type until it fits the space it was given.
   *
   * The Comic Life behaviour, and the right way round for lettering: a balloon is placed where the
   * art has room for it, and the words are what gives. Off for a free-standing text layer, where
   * the size is the point and the column should grow instead.
   */
  autoFit?: boolean;
  /** How small auto-fit may go before it gives up and lets the text overflow. */
  fitMin?: number;
};

export const NEUTRAL_TEXT: TextBlock = {
  content: "Double-click to edit",
  fontFamily: "Inter",
  fontSize: 72,
  weight: 700,
  italic: false,
  color: solidPaint("#ffffff"),
  align: "center",
  lineHeight: 1.2,
  letterSpacing: 0,
  boxWidth: 900,
  padding: 24,
  strokeWidth: 0,
  strokeColor: solidPaint("#000000"),
  shadow: 0,
  shadowColor: solidPaint("#000000"),
  panel: "none",
  panelColor: solidPaint("#000000"),
  panelBorder: 0,
  panelBorderColor: solidPaint("#ffffff"),
  autoFit: false,
  fitMin: 8,
};

/**
 * The `font` string for a context.
 *
 * A family the machine does not have falls through to the generic at the end rather than to
 * whatever the canvas defaults to, so a document opened on another computer still reads as the kind
 * of type it was set in.
 */
export function fontSpec(text: TextBlock): string {
  const family = text.fontFamily.includes(",")
    ? text.fontFamily
    : `"${text.fontFamily}", ${generic(text.fontFamily)}`;
  return `${text.italic ? "italic " : ""}${Math.round(text.weight)} ${Math.max(1, text.fontSize)}px ${family}`;
}

/** A sensible last resort for a family, guessed from its name. Better than the canvas default. */
export function generic(family: string): string {
  const name = family.toLowerCase();
  if (/mono|code|console|courier|consolas/.test(name)) return "monospace";
  if (/script|hand|brush|comic|marker|casual/.test(name)) return "cursive";
  if (/serif|times|georgia|garamond|book|roman|slab/.test(name) && !/sans/.test(name)) return "serif";
  return "sans-serif";
}

/**
 * Break the content into the lines it will actually be drawn as.
 *
 * Explicit newlines are honoured first and always break, then each paragraph is greedily wrapped to
 * the inner width. A single word wider than the box is left on its own line rather than being cut
 * up: a chopped word is always wrong, and an overflowing one is at least legible and obviously a
 * signal to widen the box.
 */
export function wrapLines(text: TextBlock, measure: (line: string) => number): string[] {
  const inner = Math.max(1, text.boxWidth - text.padding * 2);
  const lines: string[] = [];
  for (const paragraph of text.content.split(/\r?\n/)) {
    const words = paragraph.split(/ +/).filter((word) => word.length > 0);
    if (!words.length) { lines.push(""); continue; }
    let current = words[0];
    for (let index = 1; index < words.length; index++) {
      const candidate = `${current} ${words[index]}`;
      if (measure(candidate) <= inner) current = candidate;
      else { lines.push(current); current = words[index]; }
    }
    lines.push(current);
  }
  return lines;
}

/**
 * The size of the bitmap a text block needs.
 *
 * Width is the box the user set, not the width of the longest line: a text layer is a column you
 * place, and having it silently change width as you type would move it under your hands. Height
 * follows the line count, because that is the one dimension nobody wants to manage by hand.
 */
export function textBoxSize(text: TextBlock, lines: number): { width: number; height: number } {
  const step = Math.max(1, text.fontSize * Math.max(0.5, text.lineHeight));
  return {
    width: Math.max(8, Math.round(text.boxWidth)),
    height: Math.max(8, Math.round(Math.max(1, lines) * step + text.padding * 2)),
  };
}

/** Where a line starts, given the alignment. `width` is the whole bitmap, padding included. */
export function lineOrigin(text: TextBlock, width: number): { x: number; align: CanvasTextAlign } {
  if (text.align === "left") return { x: text.padding, align: "left" };
  if (text.align === "right") return { x: width - text.padding, align: "right" };
  return { x: width / 2, align: "center" };
}


/**
 * Every paint a text block carries, read through whatever is actually stored.
 *
 * One place that knows a colour used to be a string, so nothing downstream has to. Called once per
 * draw rather than per glyph.
 */
export function paintsOf(text: TextBlock): {
  fill: Paint; stroke: Paint; shadow: Paint; panel: Paint; panelBorder: Paint;
} {
  return {
    fill: toPaint(text.color, "#ffffff"),
    stroke: toPaint(text.strokeColor, "#000000"),
    shadow: toPaint(text.shadowColor, "#000000"),
    panel: toPaint(text.panelColor, "#000000"),
    panelBorder: toPaint(text.panelBorderColor, "#ffffff"),
  };
}

/**
 * The largest size at or below the block's own that fits `box`.
 *
 * A search rather than a formula, because wrapping is not continuous: one point smaller can save a
 * whole line, and the height then jumps down rather than easing. Bisection over an integer range
 * settles in about seven measurements, which is nothing next to laying the text out to draw it.
 *
 * `measure` is handed a size and returns what the text would occupy at it, so the caller keeps
 * ownership of the font and this stays testable with arithmetic.
 */
export function fitFontSize(
  text: TextBlock, box: { width: number; height: number },
  measure: (size: number) => { width: number; height: number },
): number {
  const most = Math.max(1, text.fontSize);
  const least = Math.max(1, Math.min(most, text.fitMin ?? 8));
  const fits = (size: number) => {
    const used = measure(size);
    return used.width <= box.width + 0.5 && used.height <= box.height + 0.5;
  };
  if (fits(most)) return most;

  let low = least, high = most;
  // The invariant: `low` is known to fit or is the floor, `high` is known not to. Twenty rounds is
  // far more than a one-pixel range needs and costs nothing if it ends early.
  for (let round = 0; round < 20 && high - low > 0.5; round++) {
    const middle = (low + high) / 2;
    if (fits(middle)) low = middle; else high = middle;
  }
  return Math.max(least, Math.floor(low * 2) / 2);
}
