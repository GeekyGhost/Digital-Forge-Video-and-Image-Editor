import type { ClipEffects } from "./types";

export const DEFAULT_EFFECTS: ClipEffects = {
  brightness: 0,
  contrast: 1,
  saturation: 1,
  gamma: 1,
  hue: 0,
  blur: 0,
  sharpen: 0,
  vignette: 0,
  noise: 0,
};

export type EffectPreset = {
  id: string;
  name: string;
  category: "Correction" | "Looks" | "Stylize";
  description: string;
  effects: ClipEffects;
};

const preset = (
  id: string,
  name: string,
  category: EffectPreset["category"],
  description: string,
  effects: Partial<ClipEffects>,
): EffectPreset => ({ id, name, category, description, effects: { ...DEFAULT_EFFECTS, ...effects } });

export const EFFECT_PRESETS: EffectPreset[] = [
  preset("reset", "Reset effects", "Correction", "Return every image effect to neutral.", {}),
  preset("auto-contrast", "Clean contrast", "Correction", "A restrained contrast and saturation lift.", { contrast: 1.12, saturation: 1.08, sharpen: 0.3 }),
  preset("brighten", "Exposure lift", "Correction", "Lift a dark clip without crushing color.", { brightness: 0.12, gamma: 1.08 }),
  preset("soften", "Portrait softening", "Correction", "Gentle diffusion with reduced sharpening.", { blur: 1.2, contrast: 0.96, saturation: 0.96 }),
  preset("cinematic", "Cinematic", "Looks", "Cooler, denser color with a subtle vignette.", { brightness: -0.035, contrast: 1.16, saturation: 0.88, hue: -5, vignette: 0.35, sharpen: 0.25 }),
  preset("warm", "Warm daylight", "Looks", "Warm the hue and lift natural color.", { brightness: 0.025, contrast: 1.05, saturation: 1.14, hue: 8 }),
  preset("cool", "Cool steel", "Looks", "A cooler, lower-saturation technical look.", { contrast: 1.1, saturation: 0.78, hue: -12, sharpen: 0.4 }),
  preset("mono", "Monochrome", "Looks", "Neutral black and white with firm contrast.", { contrast: 1.18, saturation: 0, gamma: 0.98, sharpen: 0.25 }),
  preset("dream", "Dream bloom", "Stylize", "Soft, bright diffusion for memory or fantasy shots.", { brightness: 0.08, contrast: 0.9, saturation: 1.08, blur: 2.2, vignette: 0.18 }),
  preset("gritty", "Gritty documentary", "Stylize", "Hard contrast, reduced color, grain and detail.", { contrast: 1.28, saturation: 0.72, sharpen: 0.85, vignette: 0.42, noise: 12 }),
  preset("vintage", "Vintage print", "Stylize", "Faded warmth, low saturation and film grain.", { brightness: 0.035, contrast: 0.9, saturation: 0.7, gamma: 1.08, hue: 10, vignette: 0.3, noise: 8 }),
  preset("surreal", "Surreal color", "Stylize", "Strong hue shift and saturated color.", { contrast: 1.08, saturation: 1.65, hue: 72, vignette: 0.2 }),
];

export const effectsFor = (effects?: Partial<ClipEffects>): ClipEffects => ({ ...DEFAULT_EFFECTS, ...effects });
