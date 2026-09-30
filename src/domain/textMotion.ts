import type {
  GraphicOverlay, GraphicTemplate, TextMotionEasing, TextMotionIn, TextMotionLoop, TextMotionOut,
} from "./types";

export const TEXT_MOTION_IN: Array<{ value: TextMotionIn; label: string }> = [
  { value: "none", label: "None" },
  { value: "fade", label: "Fade in" },
  { value: "typewriter", label: "Type in" },
  { value: "word-reveal", label: "Word reveal" },
  { value: "slide-up", label: "Slide up" },
  { value: "slide-down", label: "Slide down" },
  { value: "slide-left", label: "Slide from left" },
  { value: "slide-right", label: "Slide from right" },
  { value: "pop", label: "Pop in" },
  { value: "bounce", label: "Spring bounce" },
  { value: "zoom-blur", label: "Zoom through blur" },
  { value: "tracking", label: "Tracking reveal" },
  { value: "cinematic", label: "Cinematic rise" },
  { value: "magic", label: "Magic reveal" },
  { value: "glitch", label: "Digital glitch" },
];

export const TEXT_MOTION_OUT: Array<{ value: TextMotionOut; label: string }> = [
  { value: "none", label: "None" },
  { value: "fade", label: "Fade out" },
  { value: "typewriter", label: "Type out" },
  { value: "word-reveal", label: "Word dissolve" },
  { value: "slide-up", label: "Slide up" },
  { value: "slide-down", label: "Slide down" },
  { value: "slide-left", label: "Slide left" },
  { value: "slide-right", label: "Slide right" },
  { value: "shrink", label: "Shrink out" },
  { value: "zoom-blur", label: "Zoom into blur" },
  { value: "tracking", label: "Tracking dissolve" },
  { value: "cinematic", label: "Cinematic fall" },
  { value: "magic", label: "Magic dissolve" },
  { value: "glitch", label: "Digital glitch" },
];

export const TEXT_MOTION_LOOPS: Array<{ value: TextMotionLoop; label: string }> = [
  { value: "none", label: "None" },
  { value: "pulse", label: "Gentle pulse" },
  { value: "float", label: "Slow float" },
  { value: "flicker", label: "Neon flicker" },
  { value: "breathe", label: "Soft breathe" },
  { value: "shimmer", label: "Light shimmer" },
  { value: "jitter", label: "Micro jitter" },
];

export const TEXT_MOTION_EASINGS: Array<{ value: TextMotionEasing; label: string }> = [
  { value: "linear", label: "Linear" },
  { value: "smooth", label: "Smooth" },
  { value: "ease-out", label: "Fast out" },
  { value: "back", label: "Overshoot" },
  { value: "bounce", label: "Bounce" },
];

export const TEXT_STYLE_PRESETS: Array<{ id: string; name: string; description: string; values: Partial<GraphicOverlay> }> = [
  { id: "clean", name: "Clean", description: "Transparent, readable title", values: {
    backdrop: "none", fontFamily: "display", fontWeight: 700, strokeWidth: 2, strokeColor: "#000000", shadowOffset: 3,
  } },
  { id: "boxed", name: "Boxed", description: "Modern solid title card", values: {
    backdrop: "box", backgroundOpacity: 0.82, backgroundPadding: 18, backgroundBorderWidth: 0,
    fontFamily: "sans", fontWeight: 700, strokeWidth: 0, shadowOffset: 2,
  } },
  { id: "outlined", name: "Frame", description: "Transparent outlined frame", values: {
    backdrop: "outline", backgroundOpacity: 0, backgroundPadding: 20, backgroundBorderWidth: 3,
    fontFamily: "display", fontWeight: 700, strokeWidth: 1, shadowOffset: 2,
  } },
  { id: "cinema", name: "Cinema", description: "Wide, restrained movie title", values: {
    backdrop: "none", fontFamily: "serif", fontWeight: 400, strokeWidth: 1, shadowOffset: 4,
    motionIn: "cinematic", motionOut: "fade", motionEasing: "smooth",
  } },
  { id: "neon", name: "Neon", description: "Glowing nightlife graphic", values: {
    backdrop: "outline", backgroundOpacity: 0.08, backgroundBorderWidth: 2, fontFamily: "display",
    fontWeight: 900, strokeWidth: 2, strokeColor: "#071018", shadowOffset: 0, motionLoop: "flicker",
  } },
  { id: "social", name: "Social Pop", description: "Fast short-form emphasis", values: {
    backdrop: "box", backgroundOpacity: 0.9, fontFamily: "sans", fontWeight: 900, strokeWidth: 3,
    strokeColor: "#000000", shadowOffset: 4, motionIn: "bounce", motionOut: "shrink", motionEasing: "back",
  } },
];

export function defaultGraphicOverlay(template: GraphicTemplate = "title"): GraphicOverlay {
  return {
    template,
    text: template === "title" ? "Your title" : template === "bug" ? "LIVE" : template === "caption" ? "Caption text" : "Headline",
    subtitle: template === "lower-third" ? "Name · Role or location" : template === "banner" ? "Add supporting information" : "",
    textColor: "#ffffff",
    backgroundColor: template === "caption" ? "#000000" : "#10131a",
    accentColor: "#ff6b4a",
    fontSize: template === "title" ? 72 : template === "bug" ? 28 : template === "caption" ? 48 : 44,
    backdrop: "box",
    backgroundOpacity: template === "caption" ? 0.78 : template === "title" ? 0.72 : 0.9,
    backgroundPadding: template === "title" ? 24 : 16,
    backgroundBorderWidth: 3,
    fontFamily: "display",
    fontWeight: 700,
    italic: false,
    strokeWidth: 0,
    strokeColor: "#000000",
    shadowOffset: 3,
    shadowColor: "#000000",
    lineSpacing: 4,
    motionIn: "none",
    motionOut: "none",
    motionDuration: 0.6,
    motionLoop: "none",
    motionEasing: "smooth",
  };
}

export type TextMotionFrame = {
  text: string;
  opacity: number;
  translateX: number;
  translateY: number;
  scale: number;
  blur: number;
  hue: number;
  glow: number;
  rotation: number;
  tracking: number;
  reveal: number;
  shimmer: number;
};

const clamp01 = (value: number) => Math.max(0, Math.min(1, value));

function ease(value: number, kind: TextMotionEasing): number {
  const x = clamp01(value);
  if (kind === "linear") return x;
  if (kind === "ease-out") return 1 - (1 - x) ** 3;
  if (kind === "back") {
    const c = 1.70158; const y = x - 1;
    return 1 + (c + 1) * y ** 3 + c * y ** 2;
  }
  if (kind === "bounce") {
    const n = 7.5625; const d = 2.75;
    if (x < 1 / d) return n * x * x;
    if (x < 2 / d) { const y = x - 1.5 / d; return n * y * y + 0.75; }
    if (x < 2.5 / d) { const y = x - 2.25 / d; return n * y * y + 0.9375; }
    const y = x - 2.625 / d; return n * y * y + 0.984375;
  }
  return x * x * (3 - 2 * x);
}

/** Resolve animated text at an exact timeline time. Used by Clip Editor and program monitor. */
export function textMotionFrame(graphic: GraphicOverlay, atSeconds: number, durationSeconds: number): TextMotionFrame {
  const duration = Math.max(0.04, durationSeconds);
  const motionDuration = Math.min(Math.max(0.05, graphic.motionDuration ?? 0.6), duration / 2);
  const local = Math.max(0, Math.min(duration, atSeconds));
  const enterRaw = clamp01(local / motionDuration);
  const exitRaw = clamp01((duration - local) / motionDuration);
  const easing = graphic.motionEasing ?? "smooth";
  const enter = ease(enterRaw, easing);
  const exit = ease(exitRaw, easing);
  const entering = local < motionDuration;
  const exiting = local > duration - motionDuration;
  const inKind = graphic.motionIn ?? "none";
  const outKind = graphic.motionOut ?? "none";
  let opacity = 1;
  let translateX = 0;
  let translateY = 0;
  let scale = 1;
  let blur = 0;
  let hue = 0;
  let glow = 0;
  let rotation = 0;
  let tracking = 0;
  let reveal = 1;
  let shimmer = 0;
  let characters = graphic.text.length;

  if (entering) {
    if (["fade", "magic", "zoom-blur", "tracking", "cinematic", "word-reveal"].includes(inKind)) opacity *= enter;
    if (["word-reveal", "cinematic"].includes(inKind)) reveal = enter;
    if (["slide-up", "cinematic"].includes(inKind)) translateY += (1 - enter) * 54;
    if (inKind === "slide-down") translateY -= (1 - enter) * 54;
    if (inKind === "slide-left") translateX -= (1 - enter) * 90;
    if (inKind === "slide-right") translateX += (1 - enter) * 90;
    if (inKind === "pop") scale *= 0.55 + 0.45 * enter;
    if (inKind === "bounce") scale *= 0.42 + 0.58 * ease(enterRaw, "back");
    if (inKind === "zoom-blur") { scale *= 1.45 - 0.45 * enter; blur += (1 - enter) * 14; }
    if (inKind === "tracking") tracking += (1 - enter) * 18;
    if (inKind === "magic") { blur += (1 - enter) * 12; glow += 1 - enter; hue += (1 - enter) * 80; }
    if (inKind === "glitch") { translateX += Math.sin(local * 83) * (1 - enter) * 14; rotation += Math.sin(local * 59) * (1 - enter) * 2; opacity *= 0.65 + enter * 0.35; }
    if (inKind === "typewriter") characters = Math.ceil(graphic.text.length * enter);
  }
  if (exiting) {
    if (["fade", "magic", "zoom-blur", "tracking", "cinematic", "word-reveal"].includes(outKind)) opacity *= exit;
    if (["word-reveal", "cinematic"].includes(outKind)) reveal = Math.min(reveal, exit);
    if (outKind === "slide-up") translateY -= (1 - exit) * 54;
    if (outKind === "slide-down") translateY += (1 - exit) * 42;
    if (outKind === "slide-left") translateX -= (1 - exit) * 90;
    if (outKind === "slide-right") translateX += (1 - exit) * 90;
    if (outKind === "shrink") scale *= 0.55 + 0.45 * exit;
    if (outKind === "zoom-blur") { scale *= 1.45 - 0.45 * exit; blur += (1 - exit) * 14; }
    if (outKind === "tracking") tracking += (1 - exit) * 18;
    if (outKind === "magic") { blur += (1 - exit) * 12; glow += 1 - exit; hue += (1 - exit) * 80; }
    if (outKind === "glitch") { translateX += Math.sin(local * 97) * (1 - exit) * 14; rotation += Math.sin(local * 61) * (1 - exit) * 2; opacity *= 0.65 + exit * 0.35; }
    if (outKind === "typewriter") characters = Math.floor(graphic.text.length * exit);
  }

  switch (graphic.motionLoop ?? "none") {
    case "pulse": scale *= 1 + Math.sin(local * Math.PI * 2) * 0.025; break;
    case "float": translateY += Math.sin(local * Math.PI) * 6; break;
    case "flicker": opacity *= Math.sin(local * 37) > -0.82 ? 1 : 0.58; glow += 0.45; break;
    case "breathe": opacity *= 0.92 + 0.08 * Math.sin(local * Math.PI * 1.4); scale *= 1 + 0.012 * Math.sin(local * Math.PI * 1.4); break;
    case "shimmer": shimmer = (local * 0.65) % 1; glow += 0.18 + Math.max(0, Math.sin(local * Math.PI * 2)) * 0.12; break;
    case "jitter": translateX += Math.sin(local * 53) * 1.6; translateY += Math.sin(local * 71) * 1.2; rotation += Math.sin(local * 43) * 0.35; break;
  }

  return {
    text: graphic.text.slice(0, Math.max(0, characters)), opacity: clamp01(opacity),
    translateX, translateY, scale, blur, hue, glow, rotation, tracking, reveal, shimmer,
  };
}
