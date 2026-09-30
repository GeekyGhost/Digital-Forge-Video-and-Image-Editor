import { clipDuration, clipPlaybackRate, fromSeconds, seconds } from "../../domain/time";
import type { MediaAsset, Project, TimelineClip } from "../../domain/types";

export type RangeReplacement = {
  project: Project;
  selectedClipId: string;
  status: string;
};

/** Replace only a source-relative interval while preserving the clip head, tail, and timeline span. */
export function replaceClipRange(
  project: Project,
  sequenceId: string,
  clipId: string,
  requestedStart: number,
  requestedEnd: number,
  asset: MediaAsset,
): RangeReplacement {
  const sequence = project.sequences.find((item) => item.id === sequenceId);
  const source = sequence?.tracks.flatMap((track) => track.clips).find((clip) => clip.id === clipId);
  const track = sequence?.tracks.find((item) => item.id === source?.trackId);
  if (!sequence || !source || !track || source.trackId !== "v1" || track.locked) {
    throw new Error(track?.locked ? `${track.name} is locked` : "Range replacement needs an unlocked V1 clip");
  }

  const total = clipDuration(source.sourceIn, source.sourceOut);
  const rate = clipPlaybackRate(source);
  const sourceStart = seconds(source.sourceIn);
  const startOffset = Math.max(0, Math.min(requestedStart, total));
  const endOffset = Math.max(startOffset, Math.min(requestedEnd, total));
  const minimum = sequence.frameRate.denominator / sequence.frameRate.numerator * rate;
  if (endOffset - startOffset < minimum) throw new Error("The replacement range is shorter than one timeline frame");

  const replacementSourceSpan = Math.min(seconds(asset.duration), endOffset - startOffset);
  if (replacementSourceSpan < minimum) throw new Error("The generated replacement is shorter than one timeline frame");

  const pieces: TimelineClip[] = [];
  if (startOffset >= minimum) {
    pieces.push({
      ...source,
      sourceOut: fromSeconds(sourceStart + startOffset),
      transitionOut: "none",
      transitionOutDuration: 0,
    });
  }

  const replacementId = pieces.length === 0 ? source.id : crypto.randomUUID();
  pieces.push({
    ...source,
    id: replacementId,
    assetId: asset.id,
    name: asset.name,
    start: fromSeconds(seconds(source.start) + startOffset / rate),
    sourceIn: fromSeconds(0),
    sourceOut: fromSeconds(replacementSourceSpan),
    transitionIn: pieces.length ? "none" : source.transitionIn,
    transitionDuration: pieces.length ? 0 : source.transitionDuration,
    transitionOut: total - endOffset >= minimum ? "none" : source.transitionOut,
    transitionOutDuration: total - endOffset >= minimum ? 0 : source.transitionOutDuration,
  });

  if (total - endOffset >= minimum) {
    pieces.push({
      ...source,
      id: crypto.randomUUID(),
      start: fromSeconds(seconds(source.start) + endOffset / rate),
      sourceIn: fromSeconds(sourceStart + endOffset),
      transitionIn: "none",
      transitionDuration: 0,
    });
  }

  const updated: Project = {
    ...project,
    updatedAt: new Date().toISOString(),
    assets: project.assets.some((item) => item.id === asset.id) ? project.assets : [...project.assets, asset],
    sequences: project.sequences.map((item) => item.id === sequence.id ? {
      ...item,
      tracks: item.tracks.map((candidate) => candidate.id === track.id ? {
        ...candidate,
        clips: candidate.clips.flatMap((clip) => clip.id === source.id ? pieces : [clip]),
      } : candidate),
    } : item),
  };
  return { project: updated, selectedClipId: replacementId, status: `Replaced ${(endOffset - startOffset).toFixed(2)}s inside ${source.name}` };
}
