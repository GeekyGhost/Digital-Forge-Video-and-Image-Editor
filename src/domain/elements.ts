/**
 * Turning library elements and video assets into layer media (VFX Phase 3), and the Wan Alpha
 * prompt presets. Pure: the library itself is Rust's (src-tauri/src/ai/elements.rs).
 */

import type { LayerMedia } from "./photoDocument";
import type { MediaAsset } from "./types";
import { DEFAULT_FRAME_PATTERN } from "./photoTime";

/** The subset of an element manifest this needs (kept structural so the domain has no service import). */
export type ElementLike = {
  dir: string;
  manifest: {
    id: string; frameCount: number; fps: number; width: number; height: number; pattern: string;
    proxy?: string; loopIn: number; loopOut?: number; crossfade: number;
  };
};

function join(folder: string, name: string): string {
  const separator = folder.includes("\\") && !folder.includes("/") ? "\\" : "/";
  return `${folder.replace(/[\\/]+$/, "")}${separator}${name}`;
}

/** An element placed at composition frame `start`, with the loop defaults its manifest carries. */
export function mediaFromElement(element: ElementLike, start = 0): LayerMedia {
  const m = element.manifest;
  const loops = m.crossfade > 0 || m.loopOut !== undefined;
  return {
    kind: "frames", path: element.dir, pattern: m.pattern || DEFAULT_FRAME_PATTERN,
    frameCount: m.frameCount, fps: m.fps, width: m.width, height: m.height,
    start, inPoint: Math.min(m.loopIn, m.frameCount - 1), outPoint: Math.min(m.loopOut ?? m.frameCount, m.frameCount),
    loop: loops, crossfade: m.crossfade,
    proxyPath: m.proxy ? join(element.dir, m.proxy) : undefined, proxyStacked: m.proxy ? true : undefined,
    elementId: m.id,
  };
}

const rate = (value: { numerator: number; denominator: number } | undefined) =>
  value && value.denominator ? value.numerator / value.denominator : undefined;

/** A video asset as a layer's media, or undefined when it lacks the size or length to place it. */
export function mediaFromVideoAsset(asset: MediaAsset, start = 0): LayerMedia | undefined {
  const fps = rate(asset.frameRate) ?? 24;
  const seconds = rate(asset.duration) ?? 0;
  if (!asset.width || !asset.height || seconds <= 0) return undefined;
  const frameCount = Math.max(1, Math.round(seconds * fps));
  return {
    kind: "video", path: asset.path, frameCount, fps, width: asset.width, height: asset.height,
    start, inPoint: 0, outPoint: frameCount, loop: false, crossfade: 0,
    proxyPath: asset.proxyPath || undefined,
  };
}

/**
 * Wan Alpha prompt presets. Wan-Alpha's own guidance: open with the transparent background, then
 * style, shot type and subject. The three first ones are the Phase 0 prompts that measured well.
 */
export const WAN_ALPHA_PRESETS: Array<{ id: string; name: string; prompt: string; use?: string }> = [
  { id: "smoke", name: "Smoke wisp", prompt: "Transparent background. Realistic style. Close-up. A thin wisp of grey cigarette smoke rises slowly, curling and thinning as it drifts upward, soft and translucent, fading away at the top." },
  { id: "fireball", name: "Fireball", prompt: "Transparent background. Realistic style. Medium shot. A roiling fireball of orange and yellow flame bursts outward, glowing embers scattering, its edges soft with translucent heat haze." },
  { id: "sparks", name: "Magic sparks", prompt: "Transparent background. Fantasy style. Medium shot. Glittering blue and violet magic sparks swirl in a rising spiral, leaving soft glowing trails that fade out." },
  { id: "fire", name: "Flames", prompt: "Transparent background. Realistic style. Medium shot. A tall column of flickering orange flame dances in place, bright yellow at its core, licking upward and breaking into small tongues at the top." },
  { id: "explosion", name: "Explosion", prompt: "Transparent background. Realistic style. Wide shot. A fiery explosion erupts in a dense ball of orange flame and black smoke, debris and sparks flying outward, the smoke billowing and slowly thinning." },
  { id: "dust", name: "Dust burst", prompt: "Transparent background. Realistic style. Medium shot. A puff of fine brown dust bursts outward from the ground and spreads, hanging in the air and slowly settling, soft and translucent at the edges." },
  { id: "energy", name: "Energy aura", prompt: "Transparent background. Fantasy style. Medium shot. A crackling ring of bright cyan electric energy pulses and flickers, thin arcs of lightning jumping around its edge, glowing softly." },
  // VFX Phase 7: the element-like looks of the old mask VFX renderer (its flame, rings and burst
  // mattes and the fireball), as generated elements. Attach them to a tracked subject and blend them
  // with Screen or Add; the matte-driven effects (glow, outline, ...) stay in Clip Editor -> Mask Effects.
  { id: "flame-aura", name: "Flame aura (power-up)", use: "Attach to the subject's track, scale it around the body, blend Screen; a Mask Effects \"behind subject\" layer keeps the subject in front.", prompt: "Transparent background. Anime style. Medium shot. A tall aura of white-hot and electric blue flame surrounds an empty silhouette-shaped space, tongues of fire rising and flickering upward, bright at the edges and fading to transparent at the top." },
  { id: "energy-rings", name: "Energy rings", use: "Attach to the subject's track at the feet; blend Add.", prompt: "Transparent background. Fantasy style. Medium shot. Several thin glowing cyan elliptical rings of energy rise one after another from the ground, expanding and fading as they climb, soft light trailing behind them." },
  { id: "radial-burst", name: "Radial burst", use: "Place at the impact point; blend Screen.", prompt: "Transparent background. Anime style. Medium shot. Bright yellow and white rays of light burst outward from a single point, streaking to the edges and fading quickly, a flash of light at the centre." },
  { id: "fireball-orb", name: "Fireball orb (held)", use: "Attach to a tracked hand; loop it; blend Screen.", prompt: "Transparent background. Realistic style. Close-up. A small swirling orb of orange and yellow fire hovers in place, its surface churning, embers spinning off and fading, glowing softly." },
];
