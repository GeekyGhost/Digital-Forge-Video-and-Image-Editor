import { describe, expect, it } from "vitest";
import { looksGenerated, nameFromPrompt, slugify, suggestedFileName } from "./naming";

describe("slugify", () => {
  it("keeps the content words and drops the filler", () => {
    expect(slugify("a rusted lighthouse in the fog at dusk")).toBe("rusted-lighthouse-fog-dusk");
  });

  it("stops at the word limit", () => {
    expect(slugify("one two three four five six seven")).toBe("one-two-three-four-five");
  });

  it("collapses punctuation and non-ASCII rather than emitting it", () => {
    expect(slugify("Café  scene, 35mm -- warm!")).toBe("caf-scene-35mm-warm");
  });

  it("still names a prompt made entirely of filler", () => {
    expect(slugify("the a of in")).toBe("the-a-of-in");
  });

  it("returns empty for a prompt with nothing usable in it", () => {
    expect(slugify("   ...   ")).toBe("");
  });

  it("drops a word that would push it past the length cap", () => {
    const slug = slugify("portrait supercalifragilisticexpialidociously luminous", 5, 20);
    expect(slug).toBe("portrait");
  });

  it("truncates rather than returning nothing when the first word alone is too long", () => {
    expect(slugify("supercalifragilistic", 5, 10)).toBe("supercalif");
  });
});

describe("suggestedFileName", () => {
  it("puts the prefix first and the meaning after it", () => {
    expect(suggestedFileName("H3-image", "a rusted lighthouse at dusk", "mp4"))
      .toBe("H3-image-rusted-lighthouse-dusk.mp4");
  });

  it("falls back to a timestamp when there is no prompt, so runs do not collide", () => {
    expect(suggestedFileName("LTX-image", "", "mp4")).toMatch(/^LTX-image-\d{8}-\d{6}\.mp4$/);
  });
});

describe("nameFromPrompt", () => {
  it("recovers a name for an asset generated before prompts were kept in filenames", () => {
    expect(nameFromPrompt("neon alley after rain", "png")).toBe("neon-alley-after-rain.png");
  });

  it("declines rather than inventing one when no prompt was recorded", () => {
    expect(nameFromPrompt(undefined, "png")).toBeUndefined();
  });
});

describe("looksGenerated", () => {
  it("recognises the old timestamp-and-uuid names", () => {
    expect(looksGenerated("zit-20260903-185500-3f9a2b1c4d5e6f70.png")).toBe(true);
    expect(looksGenerated("qwen-edit-20260903-190112-aabbccddeeff0011.png")).toBe(true);
  });

  it("leaves names a person would have written alone", () => {
    expect(looksGenerated("rusted-lighthouse-dusk.png")).toBe(false);
    expect(looksGenerated("Hero shot v2.png")).toBe(false);
    expect(looksGenerated("beach-1080p.mp4")).toBe(false);
  });
});
