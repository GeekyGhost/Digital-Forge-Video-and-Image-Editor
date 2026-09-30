/**
 * MiniMax-H3 Ref2VA reference model and prompt builder.
 *
 * The prompt is not free text. MiniMax trained Ref2V on a six-field schema
 * (`docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md` in MiniMaxAI/MiniMax-H3):
 *
 *     subject_definitions / summary / retention_analysis /
 *     detailed_description / overall_soundscape / non_diegetic_music
 *
 * Two numbering systems meet here and they are NOT the same thing:
 *
 *   - `<Picture n>` / `<Video n>` / `<Audio n>` are PRESENTATION labels. Diffusers assigns them in
 *     `MiniMaxH3Ref2VATextEncoderStep._build_presentation()`, per modality, in the order the
 *     references are passed. We do not get to choose them -- we only get to match them. Three
 *     rules, all load-bearing:
 *        1. counters are independent per modality;
 *        2. a video's embedded soundtrack increments the SAME audio counter as a standalone
 *           audio reference;
 *        3. for a sound-bearing video the audio label is emitted BEFORE its video label.
 *     Get any of these wrong and every subsequent label in the prompt points at the wrong
 *     reference -- silently, with no error, producing output that looks like a model failure.
 *
 *   - `<Subject n>` is PROMPT-ONLY. We invent it. It binds reusable visible content to whichever
 *     references define it, so the description can refer back without re-describing.
 */
import type {
  H3AssetInfo, H3DialogueLine, H3ReferenceDraft, H3ReferenceIntent, H3ReferenceKind, H3Shot,
  H3Strength,
} from "../../domain/h3Director";

export type {
  H3AssetInfo, H3DialogueLine, H3ReferenceDraft, H3ReferenceIntent, H3ReferenceKind, H3Shot,
  H3Strength,
} from "../../domain/h3Director";

export const H3_REFERENCE_INTENTS: { value: H3ReferenceIntent; label: string; hint: string }[] = [
  { value: "subject", label: "Subject / identity",
    hint: "Who or what this reference is. Bind it to a <Subject> below so the description can name it." },
  { value: "style", label: "Visual style",
    hint: "Palette, grain, lighting and lens character — not the content." },
  { value: "motion", label: "Motion / action",
    hint: "How things move. Keep this as video: separate stills lose the timing." },
  { value: "camera", label: "Camera / composition",
    hint: "Framing and camera movement." },
  { value: "anchor", label: "Frame anchor",
    hint: "This picture IS a frame of the shot — the shot begins from it. Use for a hard start point." },
  { value: "continuation", label: "Continue clip",
    hint: "Treat the clip as the immediately preceding shot: start after its last frame and never replay it. Ref2V has no extend mode, so this is carried by the prompt alone." },
  { value: "voice", label: "Voice / timbre",
    hint: "How a speaker sounds. Matching timbre does not carry the source's words across." },
  { value: "ambience", label: "Ambience / sound",
    hint: "Room tone and physical sound." },
  { value: "music", label: "Music",
    hint: "Score. Goes in the non-diegetic music field, not the soundscape." },
];

export const H3_STRENGTHS: { value: H3Strength; label: string; hint: string }[] = [
  { value: "keep", label: "Keep exactly",
    hint: "Hold it unchanged — fully_preserved for picture, fully_copy for sound." },
  { value: "adapt", label: "Adapt",
    hint: "Keep some of it, let the description change the rest — partially_preserved / partially_copy." },
  { value: "inspire", label: "Inspired by",
    hint: "Broad similarity only — weak_reference for picture, reference for sound." },
];

/** Audio intents use the audio marker vocabulary; everything else uses the visual one. */
const AUDIO_INTENTS: H3ReferenceIntent[] = ["voice", "ambience", "music"];

const VISUAL_MARKER: Record<H3Strength, string> = {
  keep: "fully_preserved",
  adapt: "partially_preserved",
  inspire: "weak_reference",
};

const AUDIO_MARKER: Record<H3Strength, string> = {
  keep: "fully_copy",
  adapt: "partially_copy",
  inspire: "reference",
};

const DEFAULT_STRENGTH: Record<H3ReferenceIntent, H3Strength> = {
  subject: "keep", anchor: "keep", style: "inspire", motion: "adapt",
  camera: "adapt", continuation: "adapt", voice: "inspire",
  ambience: "adapt", music: "inspire",
};

export function isAudioIntent(intent: H3ReferenceIntent): boolean {
  return AUDIO_INTENTS.includes(intent);
}

export function h3Marker(intent: H3ReferenceIntent, strength: H3Strength): string {
  return isAudioIntent(intent) ? AUDIO_MARKER[strength] : VISUAL_MARKER[strength];
}

export function defaultH3ReferenceIntents(kind: H3ReferenceKind): H3ReferenceIntent[] {
  return [kind === "audio" ? "voice" : kind === "video" ? "motion" : "subject"];
}

export function availableH3ReferenceIntents(
  kind: H3ReferenceKind,
  includeAudio: boolean,
): H3ReferenceIntent[] {
  if (kind === "image") return ["subject", "style", "camera", "anchor"];
  if (kind === "audio") return ["voice", "ambience", "music"];
  return [
    "subject", "style", "motion", "camera", "continuation",
    ...(includeAudio ? (["voice", "ambience", "music"] as H3ReferenceIntent[]) : []),
  ];
}

export function strengthOf(reference: H3ReferenceDraft, intent: H3ReferenceIntent): H3Strength {
  return reference.strengths?.[intent] ?? DEFAULT_STRENGTH[intent];
}

// ---------------------------------------------------------------- label allocation

export type H3LabelledReference = {
  reference: H3ReferenceDraft;
  asset: H3AssetInfo;
  /** 1-based presentation labels. Undefined where the modality does not apply. */
  picture?: number;
  video?: number;
  audio?: number;
};

/**
 * Mirror Diffusers' presentation numbering exactly. See the module header for why each rule
 * matters. This function is the single source of truth for labels -- nothing else may count.
 */
export function allocateH3Labels(
  references: H3ReferenceDraft[],
  assetInfo: (assetId: string) => H3AssetInfo,
): H3LabelledReference[] {
  const counts = { image: 0, video: 0, audio: 0 };
  return references.map((reference) => {
    const asset = assetInfo(reference.assetId);
    const entry: H3LabelledReference = { reference, asset };
    if (reference.kind === "image") {
      entry.picture = ++counts.image;
    } else if (reference.kind === "audio") {
      entry.audio = ++counts.audio;
    } else {
      // Audio first: Diffusers emits a sound-bearing video's <Audio n> before its <Video n>.
      if (asset.hasAudio && reference.includeEmbeddedAudio) entry.audio = ++counts.audio;
      entry.video = ++counts.video;
    }
    return entry;
  });
}

export type H3Subject = { n: number; text: string; entry: H3LabelledReference };

/** Subjects numbered globally in tray order, across every reference that defines one. */
export function allocateH3Subjects(labelled: H3LabelledReference[]): H3Subject[] {
  const out: H3Subject[] = [];
  let n = 0;
  for (const entry of labelled) {
    for (const subject of entry.reference.subjects ?? []) {
      if (subject.text.trim()) out.push({ n: ++n, text: subject.text.trim(), entry });
    }
  }
  return out;
}

export type H3Speaker = { line: H3DialogueLine; id: string; shotId: string };

export type H3ShotTimingIssue = {
  severity: "block" | "warning";
  shotId: string;
  message: string;
};

/** Parse the exact MM:SS.mmm notation the H3 writing guide uses. */
export function parseH3Timecode(value: string): number | undefined {
  const match = value.trim().match(/^(\d{1,3}):(\d{2})(?:\.(\d{1,3}))?$/);
  if (!match) return undefined;
  const minutes = Number(match[1]);
  const seconds = Number(match[2]);
  if (seconds >= 60) return undefined;
  const milliseconds = Number((match[3] ?? "").padEnd(3, "0") || 0);
  return minutes * 60 + seconds + milliseconds / 1000;
}

/**
 * Validate shot boundaries before an expensive render and estimate whether authored dialogue can
 * physically fit. The estimate is deliberately advisory; delivery and language change speaking
 * rate, while a malformed or non-increasing timestamp is deterministic and blocks.
 */
export function h3ShotTimingIssues(shots: H3Shot[], durationSeconds: number): H3ShotTimingIssue[] {
  const issues: H3ShotTimingIssue[] = [];
  const starts = shots.map((shot, index) => index === 0 ? 0 : parseH3Timecode(shot.timecode));
  for (let index = 1; index < shots.length; index += 1) {
    const start = starts[index];
    const previous = starts[index - 1];
    if (start === undefined) {
      issues.push({ severity: "block", shotId: shots[index].id,
        message: `Shot ${index + 1} needs a valid MM:SS.mmm timestamp.` });
      continue;
    }
    if (start <= (previous ?? -1)) {
      issues.push({ severity: "block", shotId: shots[index].id,
        message: `Shot ${index + 1} must begin after Shot ${index}.` });
    }
    if (start >= durationSeconds) {
      issues.push({ severity: "block", shotId: shots[index].id,
        message: `Shot ${index + 1} begins at ${shots[index].timecode}, outside the ${durationSeconds.toFixed(2)}s clip.` });
    }
  }
  for (let index = 0; index < shots.length; index += 1) {
    const start = starts[index];
    if (start === undefined) continue;
    const end = starts[index + 1] ?? durationSeconds;
    if (end <= start) continue;
    const words = shots[index].dialogue.reduce((count, line) =>
      count + line.text.trim().split(/\s+/).filter(Boolean).length, 0);
    if (!words) continue;
    // 150 wpm plus a small breath/turn allowance: conservative enough to catch clear overflow.
    const estimatedSeconds = words / 2.5 + shots[index].dialogue.filter((line) => line.text.trim()).length * 0.2;
    if (estimatedSeconds > end - start) {
      issues.push({ severity: "warning", shotId: shots[index].id,
        message: `Shot ${index + 1} has about ${estimatedSeconds.toFixed(1)}s of dialogue in a ${(end - start).toFixed(1)}s window; shorten it or move the next cut.` });
    }
  }
  return issues;
}

/** Speaker IDs assigned once, in order of first vocal event, then reused. */
/**
 * Assign `(Sn)` per SPEAKER, in order of first vocal event -- which is what the writing guide says
 * and what this function's name always claimed, but not what it used to do.
 *
 * Lines that name no speaker are all the same speaker. That is the common case (one character
 * talking) and it is the safe default: collapsing two voices into one is a smaller error than
 * splitting one voice into two, because the model has to invent the second timbre from nothing.
 */
export function allocateH3Speakers(shots: H3Shot[]): H3Speaker[] {
  const out: H3Speaker[] = [];
  const ids = new Map<string, string>();
  for (const shot of shots) {
    for (const line of shot.dialogue) {
      if (!line.text.trim()) continue;
      const key = line.speaker === undefined || line.speaker === "" ? "\u0000unnamed" : String(line.speaker);
      let id = ids.get(key);
      if (!id) {
        id = `S${ids.size + 1}`;
        ids.set(key, id);
      }
      out.push({ line, id, shotId: shot.id });
    }
  }
  return out;
}

/** Distinct speakers in a script, in the order they first speak. */
export function h3SpeakerCount(shots: H3Shot[]): number {
  return new Set(allocateH3Speakers(shots).map((speaker) => speaker.id)).size;
}

// ---------------------------------------------------------------- task types

const TASK_TYPES = [
  "keyframe completion", "reference generation", "video editing",
  "video continuation", "audio reuse", "audio reference",
] as const;
export type H3TaskType = (typeof TASK_TYPES)[number];
export const H3_TASK_TYPES: readonly H3TaskType[] = TASK_TYPES;

export function deriveH3TaskTypes(labelled: H3LabelledReference[]): H3TaskType[] {
  const found = new Set<H3TaskType>();
  for (const entry of labelled) {
    const intents = entry.reference.intents;
    if (intents.includes("continuation")) found.add("video continuation");
    if (intents.includes("anchor")) found.add("keyframe completion");
    // A source video held at `fully_preserved` is the guide's "video editing" task: the target is
    // that clip with something changed in it, not a new shot built from it. MiniMax's own worked
    // example is exactly this case and opens `[video editing + audio reference + audio reuse]`.
    // We only ever emitted "reference generation", which describes a different task to the model.
    if (
      entry.video !== undefined
      && !intents.includes("continuation")
      && strengthOf(entry.reference, intents.find((i) => !isAudioIntent(i)) ?? "motion") === "keep"
    ) {
      found.add("video editing");
    } else if (intents.some((i) => ["subject", "style", "motion", "camera"].includes(i))) {
      found.add("reference generation");
    }
    if (entry.audio !== undefined) {
      const audioIntents = intents.filter(isAudioIntent);
      const copied = audioIntents.some((i) => strengthOf(entry.reference, i) !== "inspire");
      found.add(copied ? "audio reuse" : "audio reference");
    }
  }
  if (!found.size) found.add("reference generation");
  return TASK_TYPES.filter((t) => found.has(t));
}

// ---------------------------------------------------------------- the six fields

export type H3PromptInput = {
  /** One or two sentences of overall look, emitted before [Shot 1]. */
  style: string;
  /** Free-text summary body. The bracketed task type is prepended automatically. */
  summary: string;
  shots: H3Shot[];
  soundscape: string;
  music: string;
  references: H3ReferenceDraft[];
  assetInfo: (assetId: string) => H3AssetInfo;
};

/**
 * What a reference role is called *in the prompt*.
 *
 * Deliberately not `H3_REFERENCE_INTENTS[].label`, which is interface copy: that put strings like
 * "subject / identity" and "motion / action" into the prompt, alongside the asset's file name.
 * MiniMax's worked example contains neither -- every line is ordinary prose ("<Video 1> is the
 * source video for the editing task"), because the conditioner is a language model reading text,
 * not a parser reading a form. A file name is a token the model has never seen carrying meaning.
 */
const PROMPT_ROLE: Record<H3ReferenceIntent, string> = {
  subject: "the subject's appearance and identity",
  style: "the visual style",
  motion: "the motion and action",
  camera: "the camera framing and movement",
  anchor: "a frame anchor for the shot",
  continuation: "the immediately preceding shot",
  voice: "the voice timbre",
  ambience: "the ambience and physical sound",
  music: "the musical score",
};

const roleProse = (intents: H3ReferenceIntent[], fallback: string): string => {
  const names = intents.map((intent) => PROMPT_ROLE[intent]);
  if (!names.length) return fallback;
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
};

function subjectDefinitions(labelled: H3LabelledReference[], subjects: H3Subject[]): string[] {
  // `<Subject 1> is the young man with ...` -- the copula form MiniMax's own example uses. This
  // was `<Subject 1>: ...`, which reads as a form field rather than the sentence it was trained on.
  const lines = subjects.map((s) => `<Subject ${s.n}> is ${s.text.trim().replace(/\.$/, "")}.`);
  for (const entry of labelled) {
    const { reference: ref } = entry;
    // Every allocated label MUST appear here. Diffusers hands the model each reference whether
    // or not we describe it, and an unexplained <Picture n> is worse than no reference at all.
    //
    // The guide prefers a picture that merely establishes appearance to be cited inside a
    // <Subject n> line rather than getting a standalone one -- so emit the standalone line only
    // when it is a real frame anchor, or when nothing else in the prompt mentions it.
    if (entry.picture !== undefined) {
      const citedBySubject = subjects.some((s) =>
        s.entry === entry || s.text.includes(`<Picture ${entry.picture}>`));
      if (ref.intents.includes("anchor")) {
        lines.push(`<Picture ${entry.picture}> is a frame anchor: the shot begins from it.`);
      } else if (!citedBySubject) {
        const roles = ref.intents.filter((i) => !isAudioIntent(i));
        lines.push(`<Picture ${entry.picture}> is the reference for ${roleProse(roles, "the target's visual content")}.`);
      }
    }
    if (entry.video !== undefined) {
      const roles = ref.intents.filter((i) => !isAudioIntent(i));
      if (ref.intents.includes("continuation")) {
        // Ref2V has no structural "extend" mode -- continuation is a prompt instruction and
        // nothing else. Without "start after its final moment" and "do not replay", the model
        // reasonably treats the clip as a general reference and restarts from its first frame.
        lines.push(
          `<Video ${entry.video}> is the immediately preceding shot. It provides the ending scene ` +
          `state, subjects, camera trajectory and motion momentum the target continues from.`,
        );
      } else {
        lines.push(`<Video ${entry.video}> is the source video for ${roleProse(roles, "the target's structure and pacing")}.`);
      }
    }
    if (entry.audio !== undefined) {
      const roles = ref.intents.filter(isAudioIntent);
      lines.push(
        ref.kind === "video"
          ? `<Audio ${entry.audio}> is the synchronized audio track of <Video ${entry.video}>, providing ${roleProse(roles, "its soundtrack")}.`
          : `<Audio ${entry.audio}> is the reference for ${roleProse(roles, "the target's sound")}.`,
      );
    }
  }
  return lines;
}

function retentionAnalysis(labelled: H3LabelledReference[], subjects: H3Subject[]): string[] {
  const lines: string[] = [];
  for (const subject of subjects) {
    const ref = subject.entry.reference;
    if (ref.transferOnto !== undefined) {
      lines.push(
        `<Subject ${subject.n}>: attribute_transfer - applied onto <Subject ${ref.transferOnto}>; ` +
        `the source asset itself does not appear.`,
      );
      continue;
    }
    const intent = ref.intents.find((i) => !isAudioIntent(i)) ?? "subject";
    const strength = strengthOf(ref, intent);
    const because = strength === "keep"
      ? "identity and defining features remain intact."
      : strength === "inspire"
        ? "broad stylistic similarity only."
        : "characteristics partly retained, partly restated by the description.";
    lines.push(`<Subject ${subject.n}> (appears in [Shot 1]): ${h3Marker(intent, strength)} - ${because}`);
  }
  for (const entry of labelled) {
    const ref = entry.reference;
    // A picture that defines a subject is already covered by that subject's line; one that stands
    // on its own would otherwise be the only label in the prompt with no retention marker at all.
    if (entry.picture !== undefined && !subjects.some((s) => s.entry === entry)) {
      const intent = ref.intents.find((i) => !isAudioIntent(i)) ?? "subject";
      const strength = strengthOf(ref, intent);
      const context = ref.intents.includes("anchor") ? "frame anchor" : "visual reference";
      lines.push(
        `<Picture ${entry.picture}> (${context}): ${h3Marker(intent, strength)} - ` +
        `${roleProse(ref.intents.filter((i) => !isAudioIntent(i)), "its visual content")} carries into the target.`,
      );
    }
    if (entry.video !== undefined) {
      const intent = ref.intents.find((i) => !isAudioIntent(i)) ?? "motion";
      const strength = strengthOf(ref, intent);
      const continuation = ref.intents.includes("continuation");
      const because = continuation
        ? "ending camera trajectory, lighting and motion momentum continue across the cut."
        : "structure and pacing guide the target shot.";
      const context = continuation ? "preceding shot" : "source video";
      lines.push(`<Video ${entry.video}> (${context}): ${h3Marker(intent, strength)} - ${because}`);
    }
    if (entry.audio !== undefined) {
      const intent = ref.intents.find(isAudioIntent) ?? "ambience";
      const strength = strengthOf(ref, intent);
      const because = strength === "keep"
        ? "the source track is reused in full as the final audio."
        : strength === "adapt"
          ? "part of the source layer continues underneath new material."
          : "timbre and room character are matched, not copied; source dialogue is not carried over.";
      lines.push(`<Audio ${entry.audio}>: ${h3Marker(intent, strength)} - ${because}`);
    }
  }
  return lines;
}

function detailedDescription(
  input: H3PromptInput, speakers: H3Speaker[], continuedVideo?: number,
): string {
  const blocks: string[] = [];
  if (input.style.trim()) blocks.push(input.style.trim());
  input.shots.forEach((shot, index) => {
    // [Shot 1] takes no timestamp; every later shot is stamped MM:SS.mmm.
    const head = index === 0
      ? "[Shot 1]"
      : `[Shot ${index + 1}] At ${shot.timecode.trim() || "00:00.000"},`;
    // detailed_description is where the model reads the actual action, so the resume point has
    // to be stated here too -- a relationship declared only in summary is too easy to overlook.
    const resume = index === 0 && continuedVideo !== undefined
      ? `The shot resumes exactly where <Video ${continuedVideo}> ends. `
      : "";
    let body = `${head} ${resume}${shot.description.trim()}`.trim();
    for (const line of shot.dialogue) {
      if (!line.text.trim()) continue;
      const speaker = speakers.find((s) => s.line.id === line.id);
      // `<Subject 1> (S1) speaks, <d>...` -- the guide's example names the subject beside the
      // speaker tag, so the voice is bound to a described person rather than to a bare id.
      const who = typeof line.speaker === "number" ? `<Subject ${line.speaker}> ` : "";
      const delivery = line.delivery?.trim() ? ` with ${line.delivery.trim()} delivery` : "";
      const overlap = line.overlapPolicy === "overlap" ? " while overlapping the preceding voice"
        : line.overlapPolicy === "interrupt" ? " and interrupts the preceding voice" : "";
      body += ` ${who}(${speaker?.id ?? "S1"}) says${delivery}${overlap} <d>[${line.language}] ${line.text.trim()}</d>`;
    }
    blocks.push(body);
  });
  return blocks.join("\n\n");
}

/** Assemble the six fields, in the trained order. */
export function buildH3ReferencePrompt(input: H3PromptInput): string {
  const labelled = allocateH3Labels(input.references, input.assetInfo);
  const subjects = allocateH3Subjects(labelled);
  const speakers = allocateH3Speakers(input.shots);
  const tasks = deriveH3TaskTypes(labelled);

  const definitions = subjectDefinitions(labelled, subjects);
  const retention = retentionAnalysis(labelled, subjects);
  const continuedEntry = labelled.find((entry) =>
    entry.video !== undefined && entry.reference.intents.includes("continuation"));
  const description = detailedDescription(input, speakers, continuedEntry?.video);

  const continued = continuedEntry;
  const continuationClause = continued
    ? `Continue <Video ${continued.video}> from its final moment in a single unbroken take: ` +
      `start after its last frame, preserve the ending scene state, and do not replay any part of it. `
    : "";
  const summaryBody = continuationClause + (input.summary.trim()
    || (subjects.length
      ? `${subjects.map((s) => `<Subject ${s.n}>`).join(" and ")} carry into the target shot.`
      : "Generate one coherent shot from the supplied references."));

  return [
    `subject_definitions:\n${definitions.join("\n") || "(none)"}`,
    `summary:\n[${tasks.join(" + ")}] ${summaryBody}`,
    `retention_analysis:\n${retention.join("\n") || "(none)"}`,
    `detailed_description:\n${description || "(none)"}`,
    `overall_soundscape:\n${input.soundscape.trim() || "N/A"}`,
    `non_diegetic_music:\n${input.music.trim() || "N/A"}`,
  ].join("\n\n");
}

// ---------------------------------------------------------------- limits

/**
 * The checkpoint's own limits, from MiniMax's model card. Mirrors `src-tauri/src/ai/h3_ref.rs`;
 * change one and change the other.
 */
export const H3_MAX_IMAGES = 9;
export const H3_MAX_VIDEOS = 3;
export const H3_MAX_AUDIOS = 3;
export const H3_MAX_REFERENCES = 12;

export type H3ReferenceLimits = {
  images: number;
  videos: number;
  /** Standalone audio references. */
  audios: number;
  /** Video references contributing their own soundtrack. */
  embeddedAudios: number;
  /** What the checkpoint actually sees: `audios + embeddedAudios`. */
  effectiveAudios: number;
  visuals: number;
  total: number;
};

/**
 * Count the tray the way the model does.
 *
 * A video's soundtrack IS an audio reference -- it takes its own `<Audio n>` label and its own
 * audio rows -- so it counts against the budget of three. Counting only `kind === "audio"` let
 * three sound-bearing videos plus three audio clips look like three references while conditioning
 * on six.
 *
 * `assetInfo` is optional because the tray summary can be rendered before assets resolve; without
 * it, embedded soundtracks are simply not counted and `effectiveAudios` falls back to the
 * standalone count.
 */
export function h3ReferenceLimits(
  references: H3ReferenceDraft[],
  assetInfo?: (assetId: string) => H3AssetInfo | undefined,
): H3ReferenceLimits {
  const images = references.filter((item) => item.kind === "image").length;
  const videos = references.filter((item) => item.kind === "video").length;
  const audios = references.filter((item) => item.kind === "audio").length;
  const embeddedAudios = assetInfo
    ? references.filter((item) => item.kind === "video"
        && item.includeEmbeddedAudio
        && Boolean(assetInfo(item.assetId)?.hasAudio)).length
    : 0;
  return {
    images, videos, audios, embeddedAudios,
    effectiveAudios: audios + embeddedAudios,
    visuals: images + videos,
    total: references.length,
  };
}

/**
 * The reference contract, as one message or nothing. The Rust worker enforces the same rules and
 * is the authority; this exists so the panel refuses before a 20 GB load rather than after it.
 */
export function h3ReferenceCapError(limits: H3ReferenceLimits): string | undefined {
  if (limits.images > H3_MAX_IMAGES || limits.videos > H3_MAX_VIDEOS || limits.total > H3_MAX_REFERENCES) {
    return `MiniMax-H3 accepts at most ${H3_MAX_IMAGES} images, ${H3_MAX_VIDEOS} videos and ${H3_MAX_REFERENCES} references in total.`;
  }
  if (limits.effectiveAudios > H3_MAX_AUDIOS) {
    return `MiniMax-H3 accepts at most ${H3_MAX_AUDIOS} audio references, and a video's own soundtrack is one of them; this tray has ${limits.effectiveAudios}. Turn off the embedded soundtrack on a video reference, or remove an audio clip.`;
  }
  if (limits.visuals === 0) return "Pair the audio reference with an image or video.";
  if (limits.effectiveAudios > limits.visuals) {
    return `MiniMax-H3 needs at least as many image and video references as audio ones; this tray has ${limits.effectiveAudios} audio against ${limits.visuals} visual.`;
  }
  return undefined;
}

/** Word count of the description body, for the 350-500 guidance in the writing guide. */
export function h3DescriptionWordCount(input: Pick<H3PromptInput, "style" | "shots">): number {
  const text = [
    input.style,
    ...input.shots.flatMap((s) => [s.description, ...s.dialogue.map((d) => d.text)]),
  ].join(" ").trim();
  return text ? text.split(/\s+/).filter(Boolean).length : 0;
}
