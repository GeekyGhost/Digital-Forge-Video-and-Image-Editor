/**
 * Resource guard (VFX Phase 0, task 0.1): the shapes the Rust pre-flight check sends with the
 * `ai-preflight` event, and the small formatting rules the warning dialog uses.
 *
 * The decision itself is made in Rust (`src-tauri/src/ai/resource_guard.rs`), which is also where
 * the requirement table lives. Nothing here decides anything.
 */

export type ResourceJobId =
  | "ltx" | "ltx-retake" | "ltx-foley" | "minimax-h3" | "minimax-h3-ref" | "moss-sfx" | "kiwi-edit"
  | "sam2-track" | "mask-effect" | "whisper" | "z-image-turbo" | "z-image-base" | "z-image-caption"
  | "z-image-training" | "qwen-edit" | "background-removal" | "photo-model" | "wan-alpha"
  | "point-track" | "video-depth" | "wan-animate" | "prompt-rewrite";

export type ResourceHolderKind = "known-offender" | "this-app" | "other";

export type ResourceHolder = {
  pid: number;
  name: string;
  privateGb: number;
  kind: ResourceHolderKind;
  note: string;
  reclaimable: boolean;
};

export type ResourceVerdict = {
  job: ResourceJobId;
  jobId: ResourceJobId;
  label: string;
  requirementSource: string;
  ok: boolean;
  measured: boolean;
  commitNeededGb: number;
  commitFreeGb: number;
  commitLimitGb: number;
  commitReclaimableGb: number;
  vramNeededGb: number;
  vramFreeGb: number | null;
  vramTotalGb: number | null;
  vramSource: string;
  ramFreeGb: number;
  ramTotalGb: number;
  shortfalls: string[];
  holders: ResourceHolder[];
  hasReleasableAppWorker: boolean;
  measureMs: number;
};

export type PreflightPrompt = { id: string; verdict: ResourceVerdict };

export type PreflightDecision = "cancel" | "continue" | "release-app-workers";

export type ResourceRequirement = {
  job: ResourceJobId;
  id: ResourceJobId;
  label: string;
  commitGb: number;
  vramGb: number;
  source: string;
};

export function formatGb(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "unknown";
  return `${value >= 10 ? value.toFixed(0) : value.toFixed(1)} GB`;
}

const KIND_LABEL: Record<ResourceHolderKind, string> = {
  "known-offender": "Known memory holder",
  "this-app": "This app",
  other: "",
};

export function holderTag(holder: ResourceHolder): string {
  if (holder.reclaimable) return "Replaced by this job";
  return holder.note || KIND_LABEL[holder.kind];
}

/** The line that says, in one breath, how short the job is on commit. */
export function commitLine(verdict: ResourceVerdict): string {
  const available = verdict.commitFreeGb + verdict.commitReclaimableGb;
  return `${formatGb(available)} free of ${formatGb(verdict.commitLimitGb)} commit · needs ${formatGb(verdict.commitNeededGb)}`;
}

export function vramLine(verdict: ResourceVerdict): string {
  if (verdict.vramFreeGb === null) return `VRAM not measured (${verdict.vramSource}) · needs ${formatGb(verdict.vramNeededGb)}`;
  return `${formatGb(verdict.vramFreeGb)} free of ${formatGb(verdict.vramTotalGb)} VRAM · needs ${formatGb(verdict.vramNeededGb)}`;
}

/** True when the job is short on commit (as opposed to only VRAM). */
export function shortOnCommit(verdict: ResourceVerdict): boolean {
  return verdict.commitFreeGb + verdict.commitReclaimableGb < verdict.commitNeededGb;
}
