/**
 * VFX Phase 7: timeline titles, lower thirds, banners, bugs and captions, drawn by the compositor.
 *
 * Before Phase 7 a graphic had two renderers that disagreed: CSS in the monitor (`.graphic-*`
 * classes, px at whatever size the monitor happened to be) and `drawtext` in the export (its own
 * box positions, its own motion formulas, a fixed font file). Now a graphic is one bitmap the size
 * of the frame, drawn here with Canvas2D, and both the monitor and the export place it.
 *
 * The design is the monitor's (it is what people designed against), fixed to FRAME pixels: every
 * CSS px of the old monitor becomes `H / 1080` frame pixels, so a 1080p sequence looks as the
 * monitor did at 1080 tall, and a 4K one is the same picture at twice the pixels. Motion comes from
 * `textMotionFrame` (the monitor's), so the typewriter, reveal, glow and shimmer look as designed.
 */

import { textMotionFrame } from "./textMotion";
import type { GraphicOverlay } from "./types";

export type TimelineGraphicBlock = {
  overlay: GraphicOverlay;
  /** Seconds into the graphic clip, and its length. */
  local: number;
  duration: number;
  canvasWidth: number;
  canvasHeight: number;
};

export function graphicFontStack(family: GraphicOverlay["fontFamily"]): string {
  if (family === "serif") return "Georgia, 'Times New Roman', serif";
  if (family === "mono") return "Consolas, 'Courier New', monospace";
  if (family === "sans") return "Arial, Helvetica, sans-serif";
  return "Outfit, 'Segoe UI', system-ui, sans-serif";
}

/** `#rrggbb` (or `#rrggbbaa`) with its alpha multiplied by `opacity`, as a CSS colour. */
export function colourWithOpacity(color: string, opacity: number): string {
  const raw = color.trim().replace("#", "");
  const hex = /^[0-9a-f]{6}([0-9a-f]{2})?$/i.test(raw) ? raw : "ffffff";
  const embedded = hex.length === 8 ? parseInt(hex.slice(6), 16) / 255 : 1;
  const [r, g, b] = [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16));
  return `rgba(${r}, ${g}, ${b}, ${Math.max(0, Math.min(1, embedded * opacity)).toFixed(4)})`;
}

type Edge = { top: number; right: number; bottom: number; left: number };

/** The box's border per template and backdrop, in CSS px (before the frame scale). */
export function graphicBorder(overlay: GraphicOverlay): Edge {
  const backdrop = overlay.backdrop ?? "box";
  const none: Edge = { top: 0, right: 0, bottom: 0, left: 0 };
  if (backdrop === "none") return none;
  if (backdrop === "outline") {
    const w = overlay.backgroundBorderWidth ?? 3;
    return { top: w, right: w, bottom: w, left: w };
  }
  if (backdrop === "underline") return { ...none, bottom: 4 };
  switch (overlay.template) {
    case "title": return { ...none, bottom: 6 };
    case "banner": return { ...none, top: 6 };
    case "bug": return { ...none, left: 4 };
    case "caption": return { ...none, bottom: 4 };
    default: return { ...none, left: 6 };
  }
}

/**
 * Where the box goes in a W x H frame, given its own size (both in frame pixels): the old monitor's
 * flex layout. `unit` is frame pixels per CSS px. The monitor's container padding is 5% of the WIDTH
 * on every side (CSS percentages of padding are of the width); lower thirds and captions lift off
 * the bottom by 9% and 7% of the width; banners span the frame.
 */
export function graphicBoxPosition(overlay: GraphicOverlay, W: number, H: number, boxW: number, boxH: number): { x: number; y: number } {
  const pad = 0.05 * W;
  switch (overlay.template) {
    case "title": return { x: (W - boxW) / 2, y: (H - boxH) / 2 };
    case "banner": return { x: 0, y: H - boxH };
    case "bug": return { x: W - pad - boxW, y: pad };
    case "caption": return { x: (W - boxW) / 2, y: H - 0.07 * W - boxH };
    default: return { x: pad, y: H - 0.09 * W - boxH };
  }
}

/** Greedy word wrap to `maxWidth` (a single long word keeps its own line). */
export function wrapWords(text: string, maxWidth: number, measure: (line: string) => number): string[] {
  const lines: string[] = [];
  for (const paragraph of text.split("\n")) {
    const words = paragraph.split(/\s+/).filter(Boolean);
    if (!words.length) { lines.push(""); continue; }
    let line = words[0];
    for (const word of words.slice(1)) {
      const next = `${line} ${word}`;
      if (measure(next) <= maxWidth) line = next; else { lines.push(line); line = word; }
    }
    lines.push(line);
  }
  return lines;
}

/** The graphic at its moment, as a frame-sized bitmap (transparent where there is no graphic). */
export function drawTimelineGraphic(block: TimelineGraphicBlock): { image: HTMLCanvasElement; width: number; height: number } {
  const { overlay } = block;
  const W = Math.max(1, Math.round(block.canvasWidth)), H = Math.max(1, Math.round(block.canvasHeight));
  const unit = H / 1080;
  const canvas = window.document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const context = canvas.getContext("2d")!;
  const frame = textMotionFrame(overlay, block.local, block.duration);
  if (frame.opacity <= 0) return { image: canvas, width: W, height: H };

  const backdrop = overlay.backdrop ?? "box";
  const family = graphicFontStack(overlay.fontFamily);
  const weight = overlay.fontWeight ?? 700;
  const size = Math.max(1, overlay.fontSize) * unit;
  const titleFont = `${overlay.italic ? "italic " : ""}${weight} ${size}px ${family}`;
  const subtitleSize = 19 * unit;
  const subtitleFont = `${subtitleSize}px ${family}`;
  // letter-spacing: -.02em plus the tracking motion (CSS px).
  const spacing = -0.02 * size + frame.tracking * unit;
  const setSpacing = (value: number) => {
    try { (context as CanvasRenderingContext2D & { letterSpacing: string }).letterSpacing = `${value}px`; } catch { /* older engines */ }
  };

  const padding = (overlay.backgroundPadding ?? 16) * unit;
  const border = graphicBorder(overlay);
  const b = { top: border.top * unit, right: border.right * unit, bottom: border.bottom * unit, left: border.left * unit };
  const showSubtitle = Boolean(overlay.subtitle) && overlay.template !== "bug" && overlay.template !== "caption";
  // Width available to the box: the container's content box (frame less 5% of W each side, none for
  // a banner), capped at 78% of it; a caption box is 80% wide; a banner is the whole frame.
  const containerW = overlay.template === "banner" ? W : W - 0.1 * W;
  const fixedWidth = overlay.template === "banner" ? W : overlay.template === "caption" ? 0.8 * containerW : undefined;
  const maxBoxW = fixedWidth ?? 0.78 * containerW;
  const innerMax = Math.max(1, maxBoxW - 2 * padding - b.left - b.right);

  context.font = titleFont;
  setSpacing(spacing);
  const titleLines = frame.text ? wrapWords(frame.text, innerMax, (line) => context.measureText(line).width) : [];
  const lineHeight = 1.05 * size;
  const titleW = Math.max(0, ...titleLines.map((line) => context.measureText(line).width));
  context.font = subtitleFont;
  setSpacing(0);
  const subtitleLines = showSubtitle ? wrapWords(overlay.subtitle, innerMax, (line) => context.measureText(line).width) : [];
  const subtitleLineH = subtitleSize * 1.2;
  const subtitleW = Math.max(0, ...subtitleLines.map((line) => context.measureText(line).width));
  const gap = subtitleLines.length ? 4 * unit : 0;
  const contentW = Math.min(innerMax, Math.max(titleW, subtitleW));
  const contentH = titleLines.length * lineHeight + gap + subtitleLines.length * subtitleLineH;
  const boxW = fixedWidth ?? contentW + 2 * padding + b.left + b.right;
  const boxH = contentH + 2 * padding + b.top + b.bottom;
  const at = graphicBoxPosition(overlay, W, H, boxW, boxH);
  const centred = overlay.template === "title" || overlay.template === "caption";

  context.save();
  context.globalAlpha = Math.max(0, Math.min(1, frame.opacity));
  // The motion transform about the box centre, then the reveal (clip from the right).
  const cx = at.x + boxW / 2, cy = at.y + boxH / 2;
  context.translate(cx + frame.translateX * unit, cy + frame.translateY * unit);
  context.rotate((frame.rotation * Math.PI) / 180);
  context.scale(frame.scale, frame.scale);
  context.translate(-boxW / 2, -boxH / 2);
  const filters: string[] = [];
  if (frame.blur > 0.01) filters.push(`blur(${(frame.blur * unit).toFixed(2)}px)`);
  if (Math.abs(frame.hue) > 0.01) filters.push(`hue-rotate(${frame.hue.toFixed(2)}deg)`);
  if (filters.length) context.filter = filters.join(" ");
  if (frame.reveal < 1) {
    context.beginPath();
    context.rect(0, -H, Math.max(0, boxW * frame.reveal), boxH + 2 * H);
    context.clip();
  }

  if (backdrop === "box") {
    context.save();
    context.shadowColor = "rgba(0, 0, 0, 0.4)";
    context.shadowBlur = 24 * unit;
    context.shadowOffsetY = 8 * unit;
    context.fillStyle = colourWithOpacity(overlay.backgroundColor, overlay.backgroundOpacity ?? 1);
    context.fillRect(0, 0, boxW, boxH);
    context.restore();
  }
  if (b.top || b.right || b.bottom || b.left) {
    context.fillStyle = colourWithOpacity(overlay.accentColor, 1);
    if (b.top) context.fillRect(0, 0, boxW, b.top);
    if (b.bottom) context.fillRect(0, boxH - b.bottom, boxW, b.bottom);
    if (b.left) context.fillRect(0, 0, b.left, boxH);
    if (b.right) context.fillRect(boxW - b.right, 0, b.right, boxH);
  }

  const innerX = b.left + padding, innerW = boxW - b.left - b.right - 2 * padding;
  let y = b.top + padding;
  context.textBaseline = "top";
  context.font = titleFont;
  setSpacing(spacing);
  const stroke = (overlay.strokeWidth ?? 0) * unit;
  const glow = frame.glow > 0 ? { blur: (8 + frame.glow * 22) * unit, color: overlay.accentColor } : undefined;
  const drop = (overlay.shadowOffset ?? 0) > 0
    ? { offset: (overlay.shadowOffset ?? 0) * unit, blur: Math.max(2, (overlay.shadowOffset ?? 0) * 1.6) * unit, color: overlay.shadowColor ?? "#000000" }
    : undefined;
  for (const line of titleLines) {
    const width = context.measureText(line).width;
    const x = centred ? innerX + (innerW - width) / 2 : innerX;
    // Shadows as separate passes (CSS text-shadow draws every listed shadow).
    for (const shadow of [glow && { ...glow, offset: 0 }, drop].filter(Boolean) as Array<{ blur: number; color: string; offset: number }>) {
      context.save();
      context.shadowColor = shadow.color;
      context.shadowBlur = shadow.blur;
      context.shadowOffsetX = shadow.offset;
      context.shadowOffsetY = shadow.offset;
      context.fillStyle = overlay.textColor;
      context.fillText(line, x, y);
      context.restore();
    }
    if (overlay.motionLoop === "shimmer") {
      const gradient = context.createLinearGradient(x, y, x + width, y + lineHeight * 0.4);
      // background-size 240%, position (100 - shimmer*200)%: the bright band sweeps across.
      const centre = 0.49 + (frame.shimmer * 2 - 1) * 0.7;
      gradient.addColorStop(0, overlay.textColor);
      gradient.addColorStop(Math.max(0, Math.min(1, centre - 0.15)), overlay.textColor);
      gradient.addColorStop(Math.max(0, Math.min(1, centre)), overlay.accentColor);
      gradient.addColorStop(Math.max(0, Math.min(1, centre + 0.15)), overlay.textColor);
      gradient.addColorStop(1, overlay.textColor);
      context.fillStyle = gradient;
    } else {
      context.fillStyle = overlay.textColor;
    }
    context.fillText(line, x, y);
    if (stroke > 0) {
      // -webkit-text-stroke is centred on the glyph edge, over the fill.
      context.lineWidth = stroke;
      context.lineJoin = "round";
      context.strokeStyle = overlay.strokeColor ?? "#000000";
      context.strokeText(line, x, y);
    }
    y += lineHeight;
  }
  if (subtitleLines.length) {
    y += gap;
    context.font = subtitleFont;
    setSpacing(0);
    context.globalAlpha *= 0.86;
    context.fillStyle = overlay.textColor;
    for (const line of subtitleLines) {
      const width = context.measureText(line).width;
      context.fillText(line, centred ? innerX + (innerW - width) / 2 : innerX, y);
      y += subtitleLineH;
    }
  }
  context.restore();
  return { image: canvas, width: W, height: H };
}
