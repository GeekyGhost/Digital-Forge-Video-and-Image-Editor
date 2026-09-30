import type { CaptionedLoraImage } from "../services/modelService";

export type LoraProfile = "subject" | "character" | "style" | "product" | "concept";

export function suggestedTriggerToken(name: string) {
  const stem = name.toLowerCase().replace(/[^a-z0-9]+/g, "").slice(0, 32);
  return stem ? `${stem}sks` : "";
}

export function recommendedTrainingSteps(imageCount: number, profile: LoraProfile) {
  // The pinned Diffusers Z-Image DreamBooth recipe uses 500 optimizer steps. Subject and designed
  // character training stay close to that known-good baseline; broader visual concepts retain a
  // longer schedule.
  const base = profile === "subject" || profile === "character" ? 500 : profile === "style" ? 1200 : 800;
  if (imageCount < 5) return Math.max(400, base - 100);
  if (imageCount > 50) return base + 600;
  if (imageCount > 30) return base + 300;
  return base;
}

export function usableTrainingImages(images: CaptionedLoraImage[]) {
  return images.filter((image) => image.caption.trim() && !image.duplicateOf);
}

export function trainingReadiness(options: {
  name: string; trigger: string; className: string; images: CaptionedLoraImage[];
}) {
  const problems: string[] = [];
  if (!/[a-z0-9]/i.test(options.name)) problems.push("Enter a LoRA name.");
  if (!options.trigger.trim()) problems.push("Enter a unique trigger token.");
  if (!options.className.trim()) problems.push("Enter a class noun.");
  const usable = usableTrainingImages(options.images);
  if (usable.length < 3) problems.push("Approve at least three unique captioned images.");
  if (usable.some((image) => !image.caption.includes(options.trigger.trim()))) {
    problems.push("Every approved caption must contain the trigger token.");
  }
  return { ready: problems.length === 0, problems, usable };
}
