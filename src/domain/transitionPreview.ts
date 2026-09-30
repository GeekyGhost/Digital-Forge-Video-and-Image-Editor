import type { CSSProperties } from "react";

/**
 * CSS approximations of the FFmpeg xfade transitions used by the renderer, so the program
 * monitor previews the same edit the export will produce. `progress` runs 0 -> 1 across the
 * transition window; 0 is fully outgoing, 1 is fully incoming.
 */
export type TransitionLayers = {
  outgoing: CSSProperties;
  incoming: CSSProperties;
  /** Solid colour dipped through the middle of the transition, if the transition dips. */
  dip?: { color: string; opacity: number };
};

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
const smooth = (value: number) => value * value * (3 - 2 * value);
const pct = (value: number) => `${(value * 100).toFixed(3)}%`;

const inset = (top: number, right: number, bottom: number, left: number): CSSProperties => ({
  clipPath: `inset(${pct(top)} ${pct(right)} ${pct(bottom)} ${pct(left)})`,
});

const translate = (x: number, y: number): CSSProperties => ({
  transform: `translate(${pct(x)}, ${pct(y)})`,
});

const stripeMask = (progress: number, angle: number, stripes = 12, stagger = 0): CSSProperties => {
  const band = 100 / stripes;
  const reach = clamp01(progress * (1 + stagger)) * band;
  const image = `repeating-linear-gradient(${angle}deg, #000 0 ${reach.toFixed(3)}%, transparent ${reach.toFixed(3)}% ${band.toFixed(3)}%)`;
  return { maskImage: image, WebkitMaskImage: image } as CSSProperties;
};

const conicMask = (progress: number): CSSProperties => {
  const degrees = (clamp01(progress) * 360).toFixed(2);
  const image = `conic-gradient(from 0deg, #000 0 ${degrees}deg, transparent ${degrees}deg 360deg)`;
  return { maskImage: image, WebkitMaskImage: image } as CSSProperties;
};

const circle = (radius: number): CSSProperties => ({
  clipPath: `circle(${pct(radius)} at 50% 50%)`,
});

const diagonal = (progress: number, corner: "tl" | "tr" | "bl" | "br"): CSSProperties => {
  // A straight edge sweeping toward the named corner, expressed as a growing polygon.
  const p = clamp01(progress) * 200;
  const polygons: Record<typeof corner, string> = {
    tl: `polygon(0% 0%, ${p}% 0%, 0% ${p}%)`,
    tr: `polygon(100% 0%, ${100 - p}% 0%, 100% ${p}%)`,
    bl: `polygon(0% 100%, ${p}% 100%, 0% ${100 - p}%)`,
    br: `polygon(100% 100%, ${100 - p}% 100%, 100% ${100 - p}%)`,
  };
  return { clipPath: polygons[corner] };
};

const crossfade = (progress: number, extra: CSSProperties = {}): TransitionLayers => ({
  outgoing: {},
  incoming: { opacity: progress, ...extra },
});

const dipThrough = (progress: number, color: string): TransitionLayers => {
  const half = progress < 0.5;
  return {
    outgoing: {},
    incoming: { opacity: half ? 0 : 1 },
    dip: { color, opacity: 1 - Math.abs(progress - 0.5) * 2 },
  };
};

export function transitionLayers(id: string | undefined, rawProgress: number): TransitionLayers {
  const p = clamp01(rawProgress);
  const s = smooth(p);
  switch (id) {
    case undefined:
    case "":
    case "none":
      return { outgoing: {}, incoming: { opacity: p < 0.5 ? 0 : 1 } };

    // Essentials -------------------------------------------------------------
    case "fade":
    case "dissolve":
    case "distance":
      return crossfade(p);
    case "fadefast":
      return crossfade(clamp01(p * 1.6));
    case "fadeslow":
      return crossfade(smooth(p) * p);
    case "fadeblack":
      return dipThrough(p, "#000000");
    case "fadewhite":
      return dipThrough(p, "#ffffff");
    case "fadegrays": {
      const grey = 1 - Math.abs(p - 0.5) * 2;
      return {
        outgoing: { filter: `grayscale(${grey})` },
        incoming: { opacity: p, filter: `grayscale(${grey})` },
      };
    }
    case "hblur":
      return {
        outgoing: { filter: `blur(${(p * 14).toFixed(2)}px)` },
        incoming: { opacity: p, filter: `blur(${((1 - p) * 14).toFixed(2)}px)` },
      };
    case "pixelize": {
      const bite = 1 - Math.abs(p - 0.5) * 2;
      return {
        outgoing: { filter: `blur(${(bite * 9).toFixed(2)}px) contrast(${1 + bite * 0.4})` },
        incoming: { opacity: p, filter: `blur(${(bite * 9).toFixed(2)}px) contrast(${1 + bite * 0.4})` },
      };
    }

    // Wipes ------------------------------------------------------------------
    case "wipeleft":
      return { outgoing: {}, incoming: inset(0, 0, 0, 1 - p) };
    case "wiperight":
      return { outgoing: {}, incoming: inset(0, 1 - p, 0, 0) };
    case "wipeup":
      return { outgoing: {}, incoming: inset(1 - p, 0, 0, 0) };
    case "wipedown":
      return { outgoing: {}, incoming: inset(0, 0, 1 - p, 0) };
    case "wipetl":
    case "diagtl":
      return { outgoing: {}, incoming: diagonal(p, "tl") };
    case "wipetr":
    case "diagtr":
      return { outgoing: {}, incoming: diagonal(p, "tr") };
    case "wipebl":
    case "diagbl":
      return { outgoing: {}, incoming: diagonal(p, "bl") };
    case "wipebr":
    case "diagbr":
      return { outgoing: {}, incoming: diagonal(p, "br") };

    // Slides & motion --------------------------------------------------------
    case "slideleft":
      return { outgoing: translate(-p, 0), incoming: translate(1 - p, 0) };
    case "slideright":
      return { outgoing: translate(p, 0), incoming: translate(p - 1, 0) };
    case "slideup":
      return { outgoing: translate(0, -p), incoming: translate(0, 1 - p) };
    case "slidedown":
      return { outgoing: translate(0, p), incoming: translate(0, p - 1) };
    case "smoothleft":
      return { outgoing: translate(-s, 0), incoming: translate(1 - s, 0) };
    case "smoothright":
      return { outgoing: translate(s, 0), incoming: translate(s - 1, 0) };
    case "smoothup":
      return { outgoing: translate(0, -s), incoming: translate(0, 1 - s) };
    case "smoothdown":
      return { outgoing: translate(0, s), incoming: translate(0, s - 1) };
    case "coverleft":
      return { outgoing: {}, incoming: translate(1 - s, 0) };
    case "coverright":
      return { outgoing: {}, incoming: translate(s - 1, 0) };
    case "coverup":
      return { outgoing: {}, incoming: translate(0, 1 - s) };
    case "coverdown":
      return { outgoing: {}, incoming: translate(0, s - 1) };
    case "revealleft":
      return { outgoing: translate(-s, 0), incoming: {} };
    case "revealright":
      return { outgoing: translate(s, 0), incoming: {} };
    case "revealup":
      return { outgoing: translate(0, -s), incoming: {} };
    case "revealdown":
      return { outgoing: translate(0, s), incoming: {} };

    // Shapes & iris ----------------------------------------------------------
    case "circleopen":
      return { outgoing: {}, incoming: circle(p * 0.75) };
    case "circleclose":
      return { outgoing: circle((1 - p) * 0.75), incoming: {} };
    case "circlecrop":
      return p < 0.5
        ? { outgoing: circle((1 - p * 2) * 0.75), incoming: {}, dip: { color: "#000000", opacity: 1 } }
        : { outgoing: {}, incoming: circle((p - 0.5) * 1.5), dip: { color: "#000000", opacity: 1 } };
    case "rectcrop":
      return { outgoing: {}, incoming: inset((1 - p) / 2, (1 - p) / 2, (1 - p) / 2, (1 - p) / 2) };
    case "radial":
      return { outgoing: {}, incoming: conicMask(p) };
    case "vertopen":
      return { outgoing: {}, incoming: inset(0, (1 - p) / 2, 0, (1 - p) / 2) };
    case "vertclose":
      return { outgoing: inset(0, p / 2, 0, p / 2), incoming: {} };
    case "horzopen":
      return { outgoing: {}, incoming: inset((1 - p) / 2, 0, (1 - p) / 2, 0) };
    case "horzclose":
      return { outgoing: inset(p / 2, 0, p / 2, 0), incoming: {} };

    // Slices & wind ----------------------------------------------------------
    case "hlslice":
      return { outgoing: {}, incoming: stripeMask(p, 90) };
    case "hrslice":
      return { outgoing: {}, incoming: stripeMask(p, 270) };
    case "vuslice":
      return { outgoing: {}, incoming: stripeMask(p, 180) };
    case "vdslice":
      return { outgoing: {}, incoming: stripeMask(p, 0) };
    case "hlwind":
      return { outgoing: {}, incoming: { ...stripeMask(p, 90, 9, 0.45), opacity: clamp01(p * 1.4) } };
    case "hrwind":
      return { outgoing: {}, incoming: { ...stripeMask(p, 270, 9, 0.45), opacity: clamp01(p * 1.4) } };
    case "vuwind":
      return { outgoing: {}, incoming: { ...stripeMask(p, 180, 9, 0.45), opacity: clamp01(p * 1.4) } };
    case "vdwind":
      return { outgoing: {}, incoming: { ...stripeMask(p, 0, 9, 0.45), opacity: clamp01(p * 1.4) } };

    // Stylized ---------------------------------------------------------------
    case "squeezeh":
      return {
        outgoing: { transform: `scaleY(${(1 - s).toFixed(4)})` },
        incoming: { transform: `scaleY(${s.toFixed(4)})` },
      };
    case "zoomin":
      return {
        outgoing: {},
        incoming: { opacity: p, transform: `scale(${(0.4 + s * 0.6).toFixed(4)})` },
      };
    default:
      return crossfade(p);
  }
}

/**
 * Fade from black at the head of a sequence, or to black at the tail, expressed as an
 * overlay opacity. Mirrors the `fade=in`/`fade=out` filters used on export.
 */
export const edgeFadeOpacity = (id: string | undefined, progress: number): number => {
  if (!id || id === "none") return 0;
  return 1 - clamp01(progress);
};
