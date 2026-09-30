/**
 * Durable MiniMax-H3 Director state.
 *
 * Ref2V used to keep its cast, prompt, settings, and completed takes only in React state. Closing
 * the app or opening another project erased the production record even though the generated files
 * remained on disk. This project-owned document makes a take reproducible: every take carries the
 * exact creative and sampling snapshot that produced it, and the current working session is saved
 * with the ordinary `.aivproj` file.
 */

export type H3ReferenceKind = "image" | "video" | "audio";

export type H3ReferenceIntent =
  | "subject" | "style" | "motion" | "camera" | "anchor" | "continuation"
  | "voice" | "ambience" | "music";

export type H3Strength = "keep" | "adapt" | "inspire";

export type H3SubjectDraft = { id: string; text: string };

export type H3ReferenceDraft = {
  id: string;
  assetId: string;
  kind: H3ReferenceKind;
  intents: H3ReferenceIntent[];
  strengths?: Partial<Record<H3ReferenceIntent, H3Strength>>;
  includeEmbeddedAudio: boolean;
  subjects?: H3SubjectDraft[];
  transferOnto?: number;
  splitAudioFrom?: string;
};

export type H3DialogueLine = {
  id: string;
  language: string;
  text: string;
  speaker?: number | string;
  /** Performance direction that stays outside the exact quoted words. */
  delivery?: string;
  /** Relationship to the preceding vocal event in the same shot. */
  overlapPolicy?: "separate" | "overlap" | "interrupt";
};

export type H3Shot = {
  id: string;
  timecode: string;
  description: string;
  dialogue: H3DialogueLine[];
};

export type H3AssetInfo = { name: string; hasAudio: boolean };

export type H3DirectorSession = {
  references: H3ReferenceDraft[];
  style: string;
  summary: string;
  shots: H3Shot[];
  soundscape: string;
  music: string;
  rawPrompt: string;
  promptTab: "built" | "raw";
  aspect: string;
  tier: "native" | "standard" | "balanced" | "compact" | "custom";
  customWidth: number;
  customHeight: number;
  lockRatio: boolean;
  frames: number;
  seed: number;
  attentionBackend: "native" | "sage" | "sage1" | "sage2_stock";
  samplingMode: "turbo-4" | "base";
  steps: number;
  videoShift: number;
  refImageDetail: "draft" | "balanced" | "match" | "high" | "max";
  refVideoDetail: "safe" | "match" | "native";
  vaeTile: number;
  offload: string;
};

export type H3DirectorTake = {
  id: string;
  index: number;
  videoPath: string;
  audioPath: string;
  seed: number;
  frames: number;
  width: number;
  height: number;
  /** Kept for projects written by the first Ref2V take implementation. */
  at: string;
  createdAt: string;
  log: string;
  onTimeline: boolean;
  videoAssetId?: string;
  audioAssetId?: string;
  /** Exact cast, prompt and sampler state used for this render. */
  session: H3DirectorSession;
  sessionFingerprint: string;
};

export type H3DirectorWorkspace = {
  schemaVersion: 1;
  updatedAt: string;
  session: H3DirectorSession;
  takes: H3DirectorTake[];
  selectedTakeId?: string;
};

const newId = () => crypto.randomUUID();

export function createH3DirectorSession(): H3DirectorSession {
  return {
    references: [],
    style: "",
    summary: "",
    shots: [{ id: newId(), timecode: "", description: "", dialogue: [] }],
    soundscape: "",
    music: "",
    rawPrompt: "",
    promptTab: "built",
    aspect: "16:9",
    tier: "standard",
    customWidth: 960,
    customHeight: 544,
    lockRatio: true,
    frames: 124,
    seed: -1,
    attentionBackend: "native",
    samplingMode: "base",
    steps: 20,
    videoShift: 12,
    refImageDetail: "match",
    refVideoDetail: "safe",
    vaeTile: 256,
    offload: "balanced",
  };
}

export function cloneH3DirectorSession(session: H3DirectorSession): H3DirectorSession {
  return {
    ...session,
    references: session.references.map((reference) => ({
      ...reference,
      intents: [...reference.intents],
      strengths: reference.strengths ? { ...reference.strengths } : undefined,
      subjects: reference.subjects?.map((subject) => ({ ...subject })),
    })),
    shots: session.shots.map((shot) => ({
      ...shot,
      dialogue: shot.dialogue.map((line) => ({ ...line })),
    })),
  };
}

export function createH3DirectorWorkspace(): H3DirectorWorkspace {
  return {
    schemaVersion: 1,
    updatedAt: new Date().toISOString(),
    session: createH3DirectorSession(),
    takes: [],
  };
}

/** A small deterministic identity for stale/restore badges; not a security hash. */
export function h3DirectorSessionFingerprint(session: H3DirectorSession): string {
  const text = JSON.stringify(session);
  let hash = 0x811c9dc5;
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, "0");
}

/**
 * Old projects have no Director document, and early development builds may have written only part
 * of one. Merge with defaults at the boundary so the workspace never turns missing optional JSON
 * into an uncontrolled-input crash.
 */
export function normalizeH3DirectorWorkspace(value?: Partial<H3DirectorWorkspace>): H3DirectorWorkspace {
  const defaults = createH3DirectorWorkspace();
  const session = { ...defaults.session, ...(value?.session ?? {}) };
  session.references = Array.isArray(value?.session?.references) ? value.session.references : [];
  session.shots = Array.isArray(value?.session?.shots) && value.session.shots.length
    ? value.session.shots
    : defaults.session.shots;
  const takes = Array.isArray(value?.takes)
    ? value.takes.filter((take): take is H3DirectorTake => Boolean(
      take && take.id && take.videoPath && take.audioPath && take.session,
    ))
    : [];
  const selectedTakeId = takes.some((take) => take.id === value?.selectedTakeId)
    ? value?.selectedTakeId
    : takes.at(-1)?.id;
  return {
    schemaVersion: 1,
    updatedAt: value?.updatedAt ?? defaults.updatedAt,
    session,
    takes,
    selectedTakeId,
  };
}
