import type { Gradient } from "./photoGradient";

export const MASK_VFX_RECIPE_VERSION = 1 as const;

export type MaskSource = "sam2" | "birefnet";
export type MaskEffectPlacement = "inside" | "behind" | "front";
export type MaskEffectMatteStyle = "subject" | "flame" | "rings" | "burst" | "clip";
export type MaskEffectTransition =
  | "none" | "fade" | "edges-to-center" | "center-to-edges"
  | "top-to-bottom" | "bottom-to-top" | "explosion" | "power-up";
export type MaskEffectId =
  | "glow" | "outline" | "aura" | "neon-edge" | "electric" | "energy-shield"
  | "spotlight" | "background-blur" | "background-pixelate"
  | "silhouette" | "hologram" | "heat-vision" | "rgb-split" | "subject-echo"
  | "fireball" | "disintegrate" | "comic-pop" | "power-up" | "animated-matte";
export type MaskVfxBlendMode = "normal" | "screen" | "add" | "multiply" | "overlay" | "soft-light";
export type MaskVfxKeyMode = "alpha" | "luma" | "chroma";
export type MaskVfxTransitionOrigin = "subject" | "effect";
export type MaskVfxEasing = "linear" | "smooth" | "ease-in" | "ease-out";

export type MaskVfxAnimatedValues = {
  effectX: number;
  effectY: number;
  effectScale: number;
  effectRotation: number;
  effectOpacity: number;
  intensity: number;
};

export type MaskVfxKeyframe = {
  id: string;
  /** Normalised position inside the layer range. */
  at: number;
  easing: MaskVfxEasing;
  values: MaskVfxAnimatedValues;
};

/** Every visual choice needed to reproduce a look. Presets are full recipes, never partial patches. */
export type MaskVfxRecipeV1 = {
  schemaVersion: typeof MASK_VFX_RECIPE_VERSION;
  presetId: string;
  name: string;
  effect: MaskEffectId;
  matteStyle: MaskEffectMatteStyle;
  placement: MaskEffectPlacement;
  effectClipPath: string;
  effectClipColor: "original" | "gradient";
  blendMode: MaskVfxBlendMode;
  keyMode: MaskVfxKeyMode;
  keyThreshold: number;
  keySoftness: number;
  keyColor: string;
  color: string;
  secondaryColor: string;
  gradientEnabled: boolean;
  gradient: Gradient;
  intensity: number;
  effectOpacity: number;
  effectX: number;
  effectY: number;
  effectScale: number;
  effectRotation: number;
  effectSpeed: number;
  particleAmount: number;
  loop: boolean;
  pulseAmount: number;
  pulseRate: number;
  effectFeather: number;
  effectExpand: number;
  /** Independent from effect growth: this is only the clean foreground holdout edge. */
  occlusionFeather: number;
  occlusionExpand: number;
  temporalSmooth: number;
  cleanup: number;
  invert: boolean;
  transitionIn: MaskEffectTransition;
  transitionOut: MaskEffectTransition;
  transitionInDuration: number;
  transitionOutDuration: number;
  transitionOrigin: MaskVfxTransitionOrigin;
  audioReactive: "off" | "amplitude" | "bass" | "transients";
  reactiveAmount: number;
  reactiveSmoothing: number;
  /** One recipe may contribute behind, on, and in front of the subject. */
  rearAmount: number;
  subjectAmount: number;
  frontAmount: number;
  motionBlur: number;
  lightWrap: number;
  colorSpill: number;
  grain: number;
  displacement: number;
  keyframes: MaskVfxKeyframe[];
};

export type MaskMatteCache = {
  cacheId: string;
  cachePath: string;
  manifestPath: string;
  frameCount: number;
  width: number;
  height: number;
  fps: number;
  model: MaskSource;
  createdAt: string;
};

/** Persistent, non-destructive timeline object. Render files are disposable caches, not the edit. */
export type ClipVfxLayer = {
  id: string;
  name: string;
  enabled: boolean;
  rangeStart: number;
  rangeEnd: number;
  recipe: MaskVfxRecipeV1;
  matteCache?: MaskMatteCache;
  renderCachePath?: string;
  renderCacheKey?: string;
  /**
   * VFX Phase 7 (live rendering in the compositor). The matte after the recipe's temporal smoothing
   * and island cleanup, baked once (`mask_cache.py --stabilize`): a folder in the matte cache's
   * layout (`masks/00000.png`), and the settings it was baked with. Absent: the raw matte is used.
   */
  stableMatte?: { cachePath: string; temporalSmooth: number; cleanup: number };
  /** The per-frame audio envelope (0..1) for the recipe's audio-reactive mode, computed once. */
  envelope?: { mode: MaskVfxRecipeV1["audioReactive"]; smoothing: number; values: number[] };
  /** The imported effect clip's frames, probed once (the "clip" matte style). */
  effectClipMedia?: { frameCount: number; fps: number; width: number; height: number; proxyPath?: string };
  createdAt: string;
  updatedAt: string;
};

export function cloneMaskVfxRecipe(recipe: MaskVfxRecipeV1): MaskVfxRecipeV1 {
  return {
    ...recipe,
    gradient: { ...recipe.gradient, stops: recipe.gradient.stops.map((stop) => ({ ...stop })) },
    keyframes: recipe.keyframes.map((keyframe) => ({ ...keyframe, values: { ...keyframe.values } })),
  };
}

export function maskVfxAnimatedValues(recipe: MaskVfxRecipeV1): MaskVfxAnimatedValues {
  return {
    effectX: recipe.effectX, effectY: recipe.effectY, effectScale: recipe.effectScale,
    effectRotation: recipe.effectRotation, effectOpacity: recipe.effectOpacity, intensity: recipe.intensity,
  };
}

export function sortMaskVfxKeyframes(keyframes: MaskVfxKeyframe[]): MaskVfxKeyframe[] {
  return keyframes.slice().sort((left, right) => left.at - right.at);
}
