import type { Gradient } from "./photoGradient";
import {
  cloneMaskVfxRecipe, MASK_VFX_RECIPE_VERSION, type MaskEffectId, type MaskEffectMatteStyle,
  type MaskEffectPlacement, type MaskEffectTransition, type MaskSource, type MaskVfxRecipeV1,
} from "./maskVfx";
export type {
  ClipVfxLayer, MaskEffectId, MaskEffectMatteStyle, MaskEffectPlacement, MaskEffectTransition,
  MaskMatteCache, MaskSource, MaskVfxBlendMode, MaskVfxKeyMode, MaskVfxKeyframe, MaskVfxRecipeV1,
} from "./maskVfx";

export type AudioReactiveMode = MaskVfxRecipeV1["audioReactive"];

export const MASK_EFFECTS: Array<{ id: MaskEffectId; name: string; group: string; description: string }> = [
  { id: "glow", name: "Subject Glow", group: "Light", description: "Soft colored light around the mask." },
  { id: "outline", name: "Energy Outline", group: "Light", description: "A crisp colored edge around the subject." },
  { id: "aura", name: "Movie Aura", group: "Light", description: "Layered atmospheric light for supernatural shots." },
  { id: "neon-edge", name: "Neon Edge", group: "Light", description: "Studio42-style neon edge with a hot core and wide bloom." },
  { id: "electric", name: "Electric Arc", group: "Movie", description: "Animated lightning crawls around the selected subject." },
  { id: "energy-shield", name: "Energy Shield", group: "Movie", description: "Pulsing concentric shield rings anchored to the mask." },
  { id: "spotlight", name: "Hero Spotlight", group: "Focus", description: "Protect the subject while dimming the scene." },
  { id: "background-blur", name: "Background Blur", group: "Focus", description: "Keep the subject sharp and soften everything else." },
  { id: "background-pixelate", name: "Privacy Pixelate", group: "Focus", description: "Pixelate outside—or invert to pixelate the subject." },
  { id: "silhouette", name: "Color Silhouette", group: "Stylize", description: "Fill the subject with a graphic color and halo." },
  { id: "hologram", name: "Hologram", group: "Movie", description: "Tint, scan lines, and a projected-light edge." },
  { id: "heat-vision", name: "Heat Vision", group: "Movie", description: "False-color thermal treatment inside the mask." },
  { id: "rgb-split", name: "RGB Split", group: "Social", description: "Beat-friendly chromatic channel separation limited to the mask." },
  { id: "subject-echo", name: "Subject Echo", group: "Social", description: "Offset screen-blended subject copies for motion and beat hits." },
  { id: "fireball", name: "Fireball", group: "Movie", description: "Procedural flame orb, bloom and embers anchored to the subject." },
  { id: "disintegrate", name: "Disintegrate", group: "Movie", description: "Animated masked particles peel away from the subject." },
  { id: "comic-pop", name: "Comic Pop", group: "Social", description: "Posterized subject, ink edge and halftone energy." },
  { id: "power-up", name: "Power Up", group: "Anime", description: "Subject-relative flame aura, gathering energy, pulses and a hot outline." },
  { id: "animated-matte", name: "VFX Matte", group: "Build", description: "Color and animate a procedural shape or imported effect clip." },
];

export const MASK_MATTE_STYLES: Array<{ value: MaskEffectMatteStyle; label: string; description: string }> = [
  { value: "subject", label: "Subject silhouette", description: "Use the tracked subject directly." },
  { value: "flame", label: "Flame silhouette", description: "An animated turbulent rim rising around the subject." },
  { value: "rings", label: "Energy rings", description: "Pulsing elliptical bands centered on the subject." },
  { value: "burst", label: "Radial burst", description: "Animated rays expanding from the effect anchor." },
  { value: "clip", label: "Imported VFX clip", description: "Use a looping video as original-color VFX or as a gradient-colored luma matte." },
];

export const MASK_PLACEMENTS: Array<{ value: MaskEffectPlacement; label: string; description: string }> = [
  { value: "behind", label: "Behind subject", description: "Plate → VFX → clean subject cutout." },
  { value: "inside", label: "Inside subject", description: "Restrict the effect to the tracked silhouette." },
  { value: "front", label: "In front", description: "Composite the effect above the complete shot." },
];

export const DEFAULT_VFX_GRADIENT: Gradient = {
  stops: [
    { at: 0, color: "#ffffff", opacity: 1 },
    { at: 0.28, color: "#00f0ff", opacity: 1 },
    { at: 0.72, color: "#3166ff", opacity: 0.82 },
    { at: 1, color: "#1010a0", opacity: 0 },
  ],
  kind: "radial", angle: 270, cx: 0.5, cy: 0.5, scale: 0.72, offset: 0,
  repeat: 1, mirror: false, space: "oklab", ease: 0.8, dither: 0.7,
};

export function cloneVfxGradient(): Gradient {
  return { ...DEFAULT_VFX_GRADIENT, stops: DEFAULT_VFX_GRADIENT.stops.map((stop) => ({ ...stop })) };
}

export const DEFAULT_MASK_VFX_RECIPE: MaskVfxRecipeV1 = {
  schemaVersion: MASK_VFX_RECIPE_VERSION,
  presetId: "custom",
  name: "Custom VFX",
  effect: "neon-edge",
  matteStyle: "subject",
  placement: "behind",
  effectClipPath: "",
  effectClipColor: "original",
  blendMode: "screen",
  keyMode: "luma",
  keyThreshold: 0.08,
  keySoftness: 0.12,
  keyColor: "#00ff00",
  color: "#00f0ff",
  secondaryColor: "#ff6b1a",
  gradientEnabled: true,
  gradient: cloneVfxGradient(),
  intensity: 1,
  effectOpacity: 0.9,
  effectX: 0.5,
  effectY: 0.5,
  effectScale: 100,
  effectRotation: 0,
  effectSpeed: 1,
  particleAmount: 1,
  loop: true,
  pulseAmount: 0.18,
  pulseRate: 1.25,
  effectFeather: 6,
  effectExpand: 4,
  occlusionFeather: 1.5,
  occlusionExpand: 0,
  temporalSmooth: 2,
  cleanup: 300,
  invert: false,
  transitionIn: "fade",
  transitionOut: "fade",
  transitionInDuration: 0.35,
  transitionOutDuration: 0.3,
  transitionOrigin: "subject",
  audioReactive: "off",
  reactiveAmount: 0.75,
  reactiveSmoothing: 0.45,
  rearAmount: 1,
  subjectAmount: 0.28,
  frontAmount: 0.12,
  motionBlur: 0,
  lightWrap: 0.35,
  colorSpill: 0.18,
  grain: 0.04,
  displacement: 0,
  keyframes: [],
};

function recipe(
  presetId: string, name: string, over: Partial<MaskVfxRecipeV1>,
): MaskVfxRecipeV1 {
  return cloneMaskVfxRecipe({ ...DEFAULT_MASK_VFX_RECIPE, ...over, presetId, name,
    gradient: over.gradient ?? DEFAULT_MASK_VFX_RECIPE.gradient,
    keyframes: over.keyframes ?? [],
  });
}

/** Complete, reproducible starting looks. Selecting one replaces every visual parameter. */
export const MASK_VFX_RECIPES: MaskVfxRecipeV1[] = [
  recipe("anime-power-up", "Anime Power Up", {
    effect: "power-up", matteStyle: "flame", placement: "behind",
    color: "#f7ffff", secondaryColor: "#27bfff", intensity: 1.25,
    effectOpacity: 0.94, effectExpand: 8, effectSpeed: 1.35, particleAmount: 1.35,
    pulseAmount: 0.22, pulseRate: 1.8, transitionIn: "power-up", transitionInDuration: 0.7,
    rearAmount: 1, subjectAmount: 0.42, frontAmount: 0.2,
    lightWrap: 0.62, colorSpill: 0.32, motionBlur: 1.5, displacement: 2.5,
    gradient: {
      ...DEFAULT_VFX_GRADIENT, kind: "radial", scale: 0.68, repeat: 2, mirror: true,
      stops: [
        { at: 0, color: "#ffffff", opacity: 1 }, { at: .28, color: "#8ffcff", opacity: 1 },
        { at: .7, color: "#2177ff", opacity: .8 }, { at: 1, color: "#3720b8", opacity: 0 },
      ],
    },
  }),
  recipe("dark-magic", "Dark Magic Aura", {
    effect: "aura", matteStyle: "flame", color: "#ff4ce6", secondaryColor: "#5b16c9",
    effectSpeed: .7, pulseAmount: .12, pulseRate: .65, effectExpand: 12,
    blendMode: "add", rearAmount: 1, subjectAmount: .25, frontAmount: .3,
    lightWrap: .48, colorSpill: .3, displacement: 4,
    gradient: { ...DEFAULT_VFX_GRADIENT, kind: "angular", repeat: 2, mirror: true,
      stops: [{ at: 0, color: "#ff87ef", opacity: .95 }, { at: .48, color: "#711ee8", opacity: .8 }, { at: 1, color: "#08001f", opacity: 0 }] },
  }),
  recipe("energy-shield", "Energy Shield", {
    effect: "energy-shield", matteStyle: "rings", placement: "front",
    color: "#b9ffff", secondaryColor: "#00a9ff", effectOpacity: .82,
    effectScale: 118, effectSpeed: .85, pulseAmount: .08, pulseRate: 1,
    transitionIn: "center-to-edges", transitionOut: "center-to-edges", transitionOrigin: "effect",
    rearAmount: .15, subjectAmount: .25, frontAmount: 1, lightWrap: .4, motionBlur: .8,
  }),
  recipe("comic-surge", "Comic Surge", {
    effect: "comic-pop", matteStyle: "burst", placement: "front",
    color: "#fff25d", secondaryColor: "#ff3385", gradientEnabled: true,
    effectOpacity: .95, particleAmount: 1.4, pulseAmount: .15, pulseRate: 2.2,
    blendMode: "normal", rearAmount: .35, subjectAmount: .8, frontAmount: .75,
    lightWrap: .1, colorSpill: .08, grain: .12,
  }),
  recipe("cinematic-explosion", "Cinematic Explosion Plate", {
    effect: "animated-matte", matteStyle: "clip", placement: "behind",
    gradientEnabled: false, effectClipColor: "original", keyMode: "luma",
    keyThreshold: .06, keySoftness: .16, blendMode: "screen", effectScale: 82,
    transitionIn: "explosion", transitionInDuration: .18, transitionOut: "fade", transitionOutDuration: .65,
    rearAmount: 1, subjectAmount: .38, frontAmount: .22,
    lightWrap: .7, colorSpill: .3, motionBlur: 1.2, grain: .08, displacement: 1.5,
  }),
];

export function maskVfxRecipeById(id: string): MaskVfxRecipeV1 {
  return cloneMaskVfxRecipe(MASK_VFX_RECIPES.find((item) => item.presetId === id) ?? DEFAULT_MASK_VFX_RECIPE);
}

/** Flatten a persistent recipe into the native worker's transport names. */
export function maskVfxRenderFields(recipe: MaskVfxRecipeV1) {
  return {
    effect: recipe.effect, color: recipe.color, secondaryColor: recipe.secondaryColor,
    intensity: recipe.intensity, feather: recipe.effectFeather, expand: recipe.effectExpand,
    temporalSmooth: recipe.temporalSmooth, cleanup: recipe.cleanup,
    effectSpeed: recipe.effectSpeed, particleAmount: recipe.particleAmount,
    transitionIn: recipe.transitionIn, transitionOut: recipe.transitionOut,
    transitionInDuration: recipe.transitionInDuration, transitionOutDuration: recipe.transitionOutDuration,
    invert: recipe.invert, audioReactive: recipe.audioReactive,
    reactiveAmount: recipe.reactiveAmount, reactiveSmoothing: recipe.reactiveSmoothing,
    placement: recipe.placement, matteStyle: recipe.matteStyle,
    effectClipPath: recipe.effectClipPath, effectClipColor: recipe.effectClipColor,
    effectOpacity: recipe.effectOpacity, effectX: recipe.effectX, effectY: recipe.effectY,
    effectScale: recipe.effectScale, effectRotation: recipe.effectRotation,
    loop: recipe.loop, pulseAmount: recipe.pulseAmount, pulseRate: recipe.pulseRate,
    gradientEnabled: recipe.gradientEnabled,
    gradient: { ...recipe.gradient, stops: recipe.gradient.stops.map((stop) => ({ ...stop })) },
    occlusionFeather: recipe.occlusionFeather, occlusionExpand: recipe.occlusionExpand,
    transitionOrigin: recipe.transitionOrigin, blendMode: recipe.blendMode,
    keyMode: recipe.keyMode, keyThreshold: recipe.keyThreshold, keySoftness: recipe.keySoftness,
    keyColor: recipe.keyColor, rearAmount: recipe.rearAmount, subjectAmount: recipe.subjectAmount,
    frontAmount: recipe.frontAmount, motionBlur: recipe.motionBlur, lightWrap: recipe.lightWrap,
    colorSpill: recipe.colorSpill, grain: recipe.grain, displacement: recipe.displacement,
    keyframes: recipe.keyframes.map((keyframe) => ({ ...keyframe, values: { ...keyframe.values } })),
  };
}

export const AUDIO_REACTIVE_MODES: Array<{ value: AudioReactiveMode; label: string; description: string }> = [
  { value: "off", label: "Off", description: "Constant effect strength." },
  { value: "amplitude", label: "Overall loudness", description: "Follow the full soundtrack level." },
  { value: "bass", label: "Bass pulse", description: "Follow energy below roughly 180 Hz." },
  { value: "transients", label: "Hits + transients", description: "React to impacts and sudden attacks." },
];

export const MASK_TRANSITIONS: Array<{ value: MaskEffectTransition; label: string; description: string }> = [
  { value: "none", label: "None · static", description: "Keep the effect fully active for the entire selected range." },
  { value: "fade", label: "Soft fade", description: "Blend the complete effect smoothly." },
  { value: "edges-to-center", label: "Edges → center", description: "Sweep inward from the outside of the frame." },
  { value: "center-to-edges", label: "Center → edges", description: "Expand radially from the center." },
  { value: "top-to-bottom", label: "Top → bottom", description: "Soft vertical reveal from above." },
  { value: "bottom-to-top", label: "Bottom → top", description: "Soft vertical reveal from below." },
  { value: "explosion", label: "Explosive burst", description: "Fast turbulent radial shockwave." },
  { value: "power-up", label: "Power gathering", description: "Build through colored concentric energy waves." },
];

export function maskEffectReady(source: MaskSource, runtime?: { sam2Ready: boolean; birefnetReady: boolean }): boolean {
  return source === "sam2" ? Boolean(runtime?.sam2Ready) : Boolean(runtime?.birefnetReady);
}

export function maskEffectFileSuffix(effect: MaskEffectId): string {
  return effect.replace("background-", "bg-");
}
