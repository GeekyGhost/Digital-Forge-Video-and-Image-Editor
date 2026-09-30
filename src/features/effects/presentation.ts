import { effectsFor } from "../../domain/effects";
import type { BlendMode, TimelineClip } from "../../domain/types";

export const cssBlendMode = (mode: BlendMode | undefined): React.CSSProperties["mixBlendMode"] =>
  mode === "hardlight" ? "hard-light" : (mode ?? "normal") as React.CSSProperties["mixBlendMode"];

export const clipFilter = (clip: TimelineClip | undefined) => {
  const effect = effectsFor(clip?.effects);
  return `brightness(${Math.max(0, 1 + effect.brightness)}) contrast(${effect.contrast}) saturate(${effect.saturation}) hue-rotate(${effect.hue}deg) blur(${effect.blur}px)`;
};
