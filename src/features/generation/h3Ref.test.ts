import { describe, expect, it } from "vitest";
import {
  h3ReferenceCapError,
  h3ReferenceLimits,
  H3_MAX_AUDIOS,
  H3_MAX_IMAGES,
  H3_MAX_REFERENCES,
  H3_MAX_VIDEOS,
  allocateH3Labels,
  allocateH3Speakers,
  h3SpeakerCount,
  h3ShotTimingIssues,
  parseH3Timecode,
  allocateH3Subjects,
  buildH3ReferencePrompt,
  h3Marker,
  type H3AssetInfo,
  type H3ReferenceDraft,
  type H3Shot,
} from "./h3Ref";

const ASSETS: Record<string, H3AssetInfo> = {
  img1: { name: "ava.png", hasAudio: false },
  img2: { name: "collar.jpg", hasAudio: false },
  vidLoud: { name: "cafe.mp4", hasAudio: true },
  vidSilent: { name: "walk.mp4", hasAudio: false },
  aud1: { name: "room.wav", hasAudio: true },
};
const info = (id: string) => ASSETS[id];

const ref = (o: Partial<H3ReferenceDraft> & Pick<H3ReferenceDraft, "id" | "assetId" | "kind">):
  H3ReferenceDraft => ({ intents: [], includeEmbeddedAudio: true, subjects: [], ...o });

/**
 * These are the invariants the six Python Ref2VA tests structurally cannot reach: the label
 * numbering lives entirely in TypeScript, and getting it wrong points every subsequent label at
 * the wrong reference with no error at all.
 */
describe("presentation label allocation", () => {
  it("counts each modality independently", () => {
    const l = allocateH3Labels([
      ref({ id: "a", assetId: "img1", kind: "image" }),
      ref({ id: "b", assetId: "img2", kind: "image" }),
      ref({ id: "c", assetId: "vidSilent", kind: "video" }),
    ], info);
    expect([l[0].picture, l[1].picture]).toEqual([1, 2]);
    expect(l[2].video).toBe(1);
    expect(l[2].audio).toBeUndefined();
  });

  it("shares one audio counter between standalone clips and embedded soundtracks", () => {
    const l = allocateH3Labels([
      ref({ id: "a", assetId: "aud1", kind: "audio" }),
      ref({ id: "b", assetId: "vidLoud", kind: "video" }),
    ], info);
    expect(l[0].audio).toBe(1);
    expect(l[1].audio).toBe(2);
  });

  it("renumbers audio when the tray order changes", () => {
    const l = allocateH3Labels([
      ref({ id: "b", assetId: "vidLoud", kind: "video" }),
      ref({ id: "a", assetId: "aud1", kind: "audio" }),
    ], info);
    expect(l[0].audio).toBe(1);
    expect(l[1].audio).toBe(2);
  });

  it("frees the audio slot when the soundtrack is muted", () => {
    const l = allocateH3Labels([
      ref({ id: "b", assetId: "vidLoud", kind: "video", includeEmbeddedAudio: false }),
      ref({ id: "a", assetId: "aud1", kind: "audio" }),
    ], info);
    expect(l[0].audio).toBeUndefined();
    expect(l[1].audio).toBe(1);
  });

  it("never labels audio on a silent source, whatever the toggle says", () => {
    const l = allocateH3Labels(
      [ref({ id: "x", assetId: "vidSilent", kind: "video", includeEmbeddedAudio: true })], info);
    expect(l[0].audio).toBeUndefined();
  });

  // The same mixed set as `ai-runtime/tests/test_h3_presentation_order.py`, which runs the real
  // `MiniMaxH3Ref2VATextEncoderStep._build_presentation` against a stub tokenizer. Both lists are
  // written out in full so the two sides can be compared by eye; change one and change the other.
  it("agrees with the order Diffusers emits for a mixed reference set", () => {
    const l = allocateH3Labels([
      ref({ id: "a", assetId: "img1", kind: "image" }),
      ref({ id: "b", assetId: "vidLoud", kind: "video" }),
      ref({ id: "c", assetId: "aud1", kind: "audio" }),
      ref({ id: "d", assetId: "img2", kind: "image" }),
      ref({ id: "e", assetId: "vidSilent", kind: "video" }),
    ], info);
    expect(l.map((e) => [e.picture, e.audio, e.video])).toEqual([
      [1, undefined, undefined],       // <Picture 1>
      [undefined, 1, 1],               // <Audio 1> then <Video 1> -- audio label comes first
      [undefined, 2, undefined],       // <Audio 2>, label alone; a waveform never reaches the conditioner
      [2, undefined, undefined],       // <Picture 2>
      [undefined, undefined, 2],       // <Video 2>, silent so no audio number
    ]);
  });
});

describe("subjects and speakers", () => {
  it("numbers subjects globally in tray order and skips blanks", () => {
    const l = allocateH3Labels([
      ref({ id: "a", assetId: "img1", kind: "image", subjects: [{ id: "s1", text: "the woman" }] }),
      ref({ id: "b", assetId: "img2", kind: "image",
            subjects: [{ id: "s2", text: "the collar" }, { id: "s3", text: "   " }] }),
    ], info);
    const s = allocateH3Subjects(l);
    expect(s.map((x) => x.n)).toEqual([1, 2]);
    expect(s[1].text).toBe("the collar");
  });

  it("numbers speakers, not lines", () => {
    // This used to expect ["S1", "S2"] for two lines nobody had assigned a speaker to, which is
    // what the code did: it incremented per line. One character saying two sentences was therefore
    // presented to the model as two different voices, and MiniMax's own worked example gives the
    // same character both of his lines as (S1). Unassigned lines are now one speaker.
    const shots: H3Shot[] = [
      { id: "s1", timecode: "", description: "a", dialogue: [{ id: "d1", language: "English", text: "One." }] },
      { id: "s2", timecode: "00:04.500", description: "b", dialogue: [{ id: "d2", language: "English", text: "Two." }] },
    ];
    expect(allocateH3Speakers(shots).map((s) => s.id)).toEqual(["S1", "S1"]);
  });

  it("assigns ids per speaker in order of first vocal event", () => {
    const shots: H3Shot[] = [
      { id: "s1", timecode: "", description: "a", dialogue: [
        { id: "d1", language: "English", text: "One.", speaker: 2 },
        { id: "d2", language: "English", text: "Two.", speaker: 1 },
      ] },
      { id: "s2", timecode: "00:04.500", description: "b", dialogue: [
        { id: "d3", language: "English", text: "Three.", speaker: 2 },
      ] },
    ];
    // <Subject 2> speaks first, so it is S1. Its later line keeps that id rather than taking a new one.
    expect(allocateH3Speakers(shots).map((s) => s.id)).toEqual(["S1", "S2", "S1"]);
    expect(h3SpeakerCount(shots)).toBe(2);
  });
});

describe("shot production preflight", () => {
  it("parses the guide's timecode notation and rejects malformed boundaries", () => {
    expect(parseH3Timecode("01:02.500")).toBe(62.5);
    expect(parseH3Timecode("00:60.000")).toBeUndefined();
    expect(parseH3Timecode("3.5")).toBeUndefined();
  });

  it("blocks non-increasing and out-of-range shot starts", () => {
    const shots: H3Shot[] = [
      { id: "s1", timecode: "", description: "one", dialogue: [] },
      { id: "s2", timecode: "00:04.000", description: "two", dialogue: [] },
      { id: "s3", timecode: "00:03.000", description: "three", dialogue: [] },
      { id: "s4", timecode: "00:06.000", description: "four", dialogue: [] },
    ];
    const issues = h3ShotTimingIssues(shots, 5.17);
    expect(issues.filter((issue) => issue.severity === "block").map((issue) => issue.shotId))
      .toEqual(["s3", "s4"]);
  });

  it("warns rather than rewriting exact dialogue that cannot fit before the cut", () => {
    const shots: H3Shot[] = [
      { id: "s1", timecode: "", description: "one", dialogue: [{
        id: "d1", language: "English", text: "one two three four five six seven eight nine ten",
      }] },
      { id: "s2", timecode: "00:02.000", description: "two", dialogue: [] },
    ];
    expect(h3ShotTimingIssues(shots, 5.17)).toContainEqual(expect.objectContaining({
      severity: "warning", shotId: "s1",
    }));
  });
});

describe("retention markers", () => {
  it("uses the visual vocabulary for visual intents", () => {
    expect(h3Marker("subject", "keep")).toBe("fully_preserved");
    expect(h3Marker("style", "inspire")).toBe("weak_reference");
  });

  it("uses the audio vocabulary for audio intents", () => {
    expect(h3Marker("voice", "keep")).toBe("fully_copy");
    expect(h3Marker("ambience", "adapt")).toBe("partially_copy");
    expect(h3Marker("music", "inspire")).toBe("reference");
  });
});

describe("prompt assembly", () => {
  const shots: H3Shot[] = [
    { id: "s1", timecode: "", description: "She sits.", dialogue: [{ id: "d1", language: "English", text: "One." }] },
    { id: "s2", timecode: "00:04.500", description: "He answers.", dialogue: [] },
  ];

  const prompt = buildH3ReferencePrompt({
    style: "Warm handheld cinematography.",
    summary: "",
    shots,
    soundscape: "Cafe room tone.",
    music: "",
    references: [
      ref({ id: "a", assetId: "img1", kind: "image", intents: ["subject"],
            subjects: [{ id: "s1", text: "the woman from <Picture 1>." }] }),
      ref({ id: "b", assetId: "vidLoud", kind: "video", intents: ["continuation", "voice"] }),
    ],
    assetInfo: info,
  });

  it("emits all six fields in the trained order", () => {
    const order = ["subject_definitions:", "summary:", "retention_analysis:",
                   "detailed_description:", "overall_soundscape:", "non_diegetic_music:"];
    const positions = order.map((f) => prompt.indexOf(f));
    expect(positions.every((p) => p >= 0)).toBe(true);
    expect([...positions].sort((a, b) => a - b)).toEqual(positions);
  });

  it("leaves Shot 1 unstamped and stamps every later shot", () => {
    // What this is actually about is the timestamp: Shot 1 carries none, every later shot does.
    // It deliberately does NOT assert that the description follows the tag immediately -- this
    // fixture includes a continuation reference, which correctly inserts a resume clause first.
    // The original assertion coupled the two and broke when that fix landed.
    expect(prompt).toMatch(/\[Shot 1\](?!\s*At )/);
    expect(prompt).toContain("She sits.");
    expect(prompt).toMatch(/\[Shot 2\] At 00:04\.500,/);
  });

  it("wraps dialogue and attributes a speaker", () => {
    expect(prompt).toContain("<d>[English] One.</d>");
    expect(prompt).toContain("(S1) says");
  });

  it("keeps delivery and overlap direction outside the exact dialogue payload", () => {
    const directed = buildH3ReferencePrompt({
      style: "", summary: "", soundscape: "", music: "", references: [], assetInfo: info,
      shots: [{ id: "s1", timecode: "", description: "A close shot.", dialogue: [{
        id: "d1", language: "English", text: "Stop.", delivery: "hushed and urgent",
        overlapPolicy: "interrupt",
      }] }],
    });
    expect(directed).toContain("says with hushed and urgent delivery and interrupts the preceding voice");
    expect(directed).toContain("<d>[English] Stop.</d>");
  });

  it("prefixes the summary with bracketed task types", () => {
    expect(prompt).toMatch(/summary:\n\[[^\]]+\]/);
  });

  it("writes N/A rather than an empty music field", () => {
    expect(prompt).toMatch(/non_diegetic_music:\nN\/A/);
  });

  it("treats attribute_transfer as a target rather than a strength", () => {
    const transferred = buildH3ReferencePrompt({
      style: "x", summary: "", shots: [{ id: "s", timecode: "", description: "y", dialogue: [] }],
      soundscape: "", music: "",
      references: [
        ref({ id: "a", assetId: "img1", kind: "image", intents: ["subject"],
              subjects: [{ id: "s1", text: "the woman" }] }),
        ref({ id: "b", assetId: "img2", kind: "image", intents: ["style"],
              subjects: [{ id: "s2", text: "the collar" }], transferOnto: 1 }),
      ],
      assetInfo: info,
    });
    expect(transferred).toContain("attribute_transfer");
    expect(transferred).toContain("applied onto <Subject 1>");
  });
});

describe("every allocated label is explained", () => {
  /**
   * Diffusers hands the model every reference whether or not the prompt mentions it. A
   * <Picture n> that appears in the presentation but nowhere in the text is an unexplained
   * input -- the model is left to guess what it is for, which is worse than not passing it.
   */
  const cases: Array<[string, H3ReferenceDraft[]]> = [
    ["image with no subject text", [
      ref({ id: "a", assetId: "img1", kind: "image", intents: ["subject"] }),
      ref({ id: "b", assetId: "vidLoud", kind: "video", intents: ["continuation"] }),
    ]],
    ["image cited by a subject", [
      ref({ id: "a", assetId: "img1", kind: "image", intents: ["subject"],
            subjects: [{ id: "s1", text: "the woman from <Picture 1>." }] }),
    ]],
    ["standalone audio plus a muted video", [
      ref({ id: "a", assetId: "vidLoud", kind: "video", intents: ["motion"], includeEmbeddedAudio: false }),
      ref({ id: "b", assetId: "aud1", kind: "audio", intents: ["ambience"] }),
    ]],
  ];

  it.each(cases)("%s", (_name, references) => {
    const prompt = buildH3ReferencePrompt({
      style: "", summary: "",
      shots: [{ id: "s", timecode: "", description: "A shot.", dialogue: [] }],
      soundscape: "", music: "", references, assetInfo: info,
    });
    for (const entry of allocateH3Labels(references, info)) {
      if (entry.picture !== undefined) expect(prompt).toContain(`<Picture ${entry.picture}>`);
      if (entry.video !== undefined) expect(prompt).toContain(`<Video ${entry.video}>`);
      if (entry.audio !== undefined) expect(prompt).toContain(`<Audio ${entry.audio}>`);
    }
  });

  it("does not also emit a standalone line for a picture a subject already cites", () => {
    const prompt = buildH3ReferencePrompt({
      style: "", summary: "",
      shots: [{ id: "s", timecode: "", description: "A shot.", dialogue: [] }],
      soundscape: "", music: "",
      references: [ref({ id: "a", assetId: "img1", kind: "image", intents: ["subject"],
                         subjects: [{ id: "s1", text: "the woman from <Picture 1>." }] })],
      assetInfo: info,
    });
    expect(prompt).not.toMatch(/<Picture 1> is /);
    // The copula form MiniMax's own example uses. This was `<Subject 1>: ...`, which reads as a
    // form field rather than the sentence the conditioner was trained on.
    expect(prompt).toContain("<Subject 1> is the woman from <Picture 1>.");
  });
});

describe("continuation", () => {
  /**
   * Ref2V has no structural extend mode: continuation is a prompt instruction and nothing else.
   * An earlier revision stated only that the camera trajectory continues, and the model
   * reasonably read the clip as a general reference and restarted from its first frame.
   */
  const prompt = buildH3ReferencePrompt({
    style: "", summary: "",
    shots: [{ id: "s", timecode: "", description: "Two creatures clash.", dialogue: [] }],
    soundscape: "", music: "",
    references: [ref({ id: "a", assetId: "vidSilent", kind: "video", intents: ["continuation"] })],
    assetInfo: info,
  });

  it("names the clip as the preceding shot", () => {
    expect(prompt).toContain("the immediately preceding shot");
  });

  it("says where to start and forbids replaying", () => {
    expect(prompt).toContain("start after its last frame");
    expect(prompt).toContain("do not replay any part of it");
  });

  it("anchors the resume point in the description, not only the summary", () => {
    expect(prompt).toContain("[Shot 1] The shot resumes exactly where <Video 1> ends.");
  });

  it("says none of that when continuation is not requested", () => {
    const plain = buildH3ReferencePrompt({
      style: "", summary: "",
      shots: [{ id: "s", timecode: "", description: "Two creatures clash.", dialogue: [] }],
      soundscape: "", music: "",
      references: [ref({ id: "a", assetId: "vidSilent", kind: "video", intents: ["motion"] })],
      assetInfo: info,
    });
    expect(plain).not.toContain("do not replay");
    expect(plain).not.toContain("resumes exactly where");
  });
});

/**
 * Measured against MiniMax's own worked Ref2VA example in
 * `docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` and the model card. Running our builder over that
 * scenario is what surfaced each of these: the guide's prompt is continuous prose naming labels,
 * ours was a form printout carrying interface copy and file names into the conditioner.
 */
describe("the prompt reads like the guide's example", () => {
  const scenario = (over: Partial<Parameters<typeof buildH3ReferencePrompt>[0]> = {}) =>
    buildH3ReferencePrompt({
      style: "The target video is in realistic photographic style.",
      summary: "The target is an edited version of the source clip.",
      shots: [{
        id: "sh", timecode: "", description: "He stands in a sunlit pasture holding a black lamb.",
        dialogue: [
          { id: "d1", language: "English", text: "Follow the wind, live free.", speaker: 1 },
          { id: "d2", language: "English", text: "Leave worries behind.", speaker: 1 },
        ],
      }],
      soundscape: "Wind over grass.", music: "N/A",
      references: [
        ref({ id: "a", assetId: "vidLoud", kind: "video", intents: ["subject", "motion"],
              strengths: { subject: "keep", motion: "keep" },
              subjects: [{ id: "s1", text: "the young man in the bright pink suit" }] }),
        ref({ id: "b", assetId: "aud1", kind: "audio", intents: ["voice"],
              strengths: { voice: "inspire" }, includeEmbeddedAudio: false }),
      ],
      assetInfo: info,
      ...over,
    });

  it("never puts an asset file name in front of the model", () => {
    const prompt = scenario();
    for (const asset of Object.values(ASSETS)) {
      expect(prompt).not.toContain(asset.name);
    }
  });

  it("never puts interface copy in front of the model", () => {
    const prompt = scenario();
    // These are `H3_REFERENCE_INTENTS[].label`, written for a dropdown, and they used to be
    // emitted verbatim as a reference's role: "<Video 1>: subject / identity, motion / action".
    for (const uiCopy of ["subject / identity", "motion / action", "visual style", "frame anchor for the shot ("]) {
      expect(prompt.toLowerCase()).not.toContain(uiCopy.toLowerCase());
    }
  });

  it("defines every label as a sentence", () => {
    const prompt = scenario();
    expect(prompt).toContain("<Subject 1> is the young man in the bright pink suit.");
    expect(prompt).toMatch(/<Video 1> is the source video for /);
    expect(prompt).toMatch(/<Audio 1> is the synchronized audio track of <Video 1>/);
    expect(prompt).toMatch(/<Audio 2> is the reference for the voice timbre/);
  });

  it("calls a preserved source clip a video edit, not a fresh generation", () => {
    // MiniMax's own example for this exact shape opens `[video editing + audio reference + audio
    // reuse]`. We only ever emitted "reference generation", which asks the model for a different
    // task than the one the description then describes.
    expect(scenario()).toMatch(/^summary:\n\[video editing/m);
  });

  it("gives one character one voice across all of their lines", () => {
    const prompt = scenario();
    expect(prompt).toContain("<Subject 1> (S1) says <d>[English] Follow the wind, live free.</d>");
    expect(prompt).toContain("<Subject 1> (S1) says <d>[English] Leave worries behind.</d>");
    expect(prompt).not.toContain("(S2)");
  });

  it("gives every visible label a retention line with a context", () => {
    const prompt = scenario();
    expect(prompt).toMatch(/<Video 1> \(source video\): fully_preserved - /);
    expect(prompt).toMatch(/<Subject 1> \(appears in \[Shot 1\]\): fully_preserved - /);
  });

  it("still gives a standalone picture its own retention line", () => {
    const prompt = buildH3ReferencePrompt({
      style: "", summary: "",
      shots: [{ id: "s", timecode: "", description: "A shot.", dialogue: [] }],
      soundscape: "", music: "",
      references: [ref({ id: "a", assetId: "img1", kind: "image", intents: ["style"] })],
      assetInfo: info,
    });
    expect(prompt).toMatch(/<Picture 1> \(visual reference\): weak_reference - /);
  });
});

/**
 * The reference contract, mirrored from `src-tauri/src/ai/h3_ref.rs::check_audio_pairing`. Both
 * sides enforce it; the Rust worker is the authority and this exists so the panel refuses before
 * a 20 GB load rather than after it.
 */
describe("reference caps and pairing", () => {
  const limits = (refs: H3ReferenceDraft[]) => h3ReferenceLimits(refs, info);

  it("counts a video's own soundtrack as an audio reference", () => {
    const l = limits([
      ref({ id: "a", assetId: "vidLoud", kind: "video" }),
      ref({ id: "b", assetId: "aud1", kind: "audio" }),
    ]);
    expect(l.audios).toBe(1);
    expect(l.embeddedAudios).toBe(1);
    expect(l.effectiveAudios).toBe(2);
    expect(l.visuals).toBe(1);
  });

  it("does not count a muted soundtrack, or one that was never there", () => {
    const l = limits([
      ref({ id: "a", assetId: "vidLoud", kind: "video", includeEmbeddedAudio: false }),
      ref({ id: "b", assetId: "vidSilent", kind: "video" }),
    ]);
    expect(l.effectiveAudios).toBe(0);
  });

  it("refuses six effective audio references dressed up as three", () => {
    // Three sound-bearing videos plus three audio clips. This used to pass on both sides.
    const l = limits([
      ref({ id: "v1", assetId: "vidLoud", kind: "video" }),
      ref({ id: "v2", assetId: "vidLoud", kind: "video" }),
      ref({ id: "v3", assetId: "vidLoud", kind: "video" }),
      ref({ id: "a1", assetId: "aud1", kind: "audio" }),
      ref({ id: "a2", assetId: "aud1", kind: "audio" }),
      ref({ id: "a3", assetId: "aud1", kind: "audio" }),
    ]);
    expect(l.effectiveAudios).toBe(6);
    expect(h3ReferenceCapError(l)).toMatch(/at most 3 audio references/);
    expect(h3ReferenceCapError(l)).toMatch(/soundtrack/);
  });

  it("requires at least as many visual references as audio ones", () => {
    const l = limits([
      ref({ id: "v1", assetId: "vidLoud", kind: "video" }),
      ref({ id: "a1", assetId: "aud1", kind: "audio" }),
    ]);
    expect(l.effectiveAudios).toBe(2);
    expect(l.visuals).toBe(1);
    expect(h3ReferenceCapError(l)).toMatch(/2 audio against 1 visual/);
  });

  it("accepts a tray that satisfies the rule", () => {
    expect(h3ReferenceCapError(limits([
      ref({ id: "i1", assetId: "img1", kind: "image" }),
      ref({ id: "v1", assetId: "vidLoud", kind: "video" }),
      ref({ id: "a1", assetId: "aud1", kind: "audio" }),
    ]))).toBeUndefined();
    expect(h3ReferenceCapError(limits([
      ref({ id: "i1", assetId: "img1", kind: "image" }),
    ]))).toBeUndefined();
  });

  it("still requires a visual to pair audio with", () => {
    expect(h3ReferenceCapError(limits([
      ref({ id: "a1", assetId: "aud1", kind: "audio" }),
    ]))).toMatch(/Pair the audio reference/);
  });

  it("uses the model card's caps, not the old 16 GB profile", () => {
    expect([H3_MAX_IMAGES, H3_MAX_VIDEOS, H3_MAX_AUDIOS, H3_MAX_REFERENCES]).toEqual([9, 3, 3, 12]);
  });

  it("falls back to the standalone count when assets have not resolved", () => {
    const l = h3ReferenceLimits([ref({ id: "v1", assetId: "vidLoud", kind: "video" })]);
    expect(l.embeddedAudios).toBe(0);
    expect(l.effectiveAudios).toBe(0);
  });
});
