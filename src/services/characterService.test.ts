import { describe, expect, it, vi } from "vitest";

vi.mock("@tauri-apps/api/core", () => ({ invoke: () => Promise.reject(new Error("no backend in tests")) }));
vi.mock("@tauri-apps/api/event", () => ({ listen: async () => () => undefined }));

import { describeCharacterProgress } from "./characterService";

describe("character progress in the status line (VFX Phase 5)", () => {
  it("says what each stage is doing and how long the steps have left", () => {
    expect(describeCharacterProgress({ event: "plan", frames: 124, width: 672, height: 384 }, "Hero")).toBe("Hero: 124 frames at 672 × 384");
    expect(describeCharacterProgress({ event: "stage-start", stage: "bake" }, "Hero")).toContain("first run only");
    expect(describeCharacterProgress({ event: "stage-start", stage: "sam2" }, "Hero")).toBe("Hero: SAM 2 is tracking the character…");
    expect(describeCharacterProgress({ event: "animate-start", frames: 124, segments: 4 }, "Hero")).toBe("Hero: animating 124 frames in 4 segments…");
    expect(describeCharacterProgress({ event: "step", step: 10, total: 40, seconds: 300 }, "Hero")).toBe("Hero: step 10 of 40 · about 15 min left");
    expect(describeCharacterProgress({ event: "step", step: 39, total: 40, seconds: 390 }, "Hero")).toBe("Hero: step 39 of 40 · about 10 s left");
    expect(describeCharacterProgress({ event: "stage", stage: "animate", ok: false, error: "CUDA out of memory" }, "Hero")).toBe("Hero: animate failed — CUDA out of memory");
    expect(describeCharacterProgress({ event: "stage", stage: "encode", ok: true }, "Hero")).toBeUndefined();
  });
});
