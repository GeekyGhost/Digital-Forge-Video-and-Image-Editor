import { describe, expect, it } from "vitest";
import { recommendedTrainingSteps, suggestedTriggerToken, trainingReadiness } from "./loraTraining";

const image = (path: string, caption: string, duplicateOf?: string) => ({
  imagePath: path, caption, tags: [], qualityFlags: [], confidence: 0.9,
  width: 768, height: 768, duplicateOf, rawCaption: caption,
});

describe("LoRA training preparation", () => {
  it("requires three unique, trigger-bearing captions", () => {
    const result = trainingReadiness({
      name: "Hero", trigger: "TOK", className: "person",
      images: [image("a.png", "TOK person outside"), image("b.png", "TOK person inside"), image("c.png", "TOK person close-up")],
    });
    expect(result.ready).toBe(true);
    expect(result.usable).toHaveLength(3);
  });

  it("rejects captions that lose the trigger contract", () => {
    const result = trainingReadiness({
      name: "Hero", trigger: "TOK", className: "person",
      images: [image("a.png", "person outside"), image("b.png", "TOK person inside"), image("c.png", "TOK person close-up")],
    });
    expect(result.ready).toBe(false);
  });

  it("scales the style preset for broader datasets", () => {
    expect(recommendedTrainingSteps(60, "style")).toBeGreaterThan(recommendedTrainingSteps(20, "style"));
  });

  it("uses the pinned 500-step identity baseline and derives a per-LoRA trigger", () => {
    expect(recommendedTrainingSteps(8, "character")).toBe(500);
    expect(recommendedTrainingSteps(8, "subject")).toBe(500);
    expect(suggestedTriggerToken("Captain Dicio")).toBe("captaindiciosks");
  });
});
