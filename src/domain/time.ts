import type { Rational, TimelineClip } from "./types";

export const rational = (numerator: number, denominator = 1): Rational => ({
  numerator,
  denominator,
});

export const seconds = (value: Rational): number => value.numerator / value.denominator;

export const fromSeconds = (value: number, scale = 1_000_000): Rational =>
  rational(Math.round(value * scale), scale);

export const clipDuration = (sourceIn: Rational, sourceOut: Rational): number =>
  Math.max(0, seconds(sourceOut) - seconds(sourceIn));

export const clipPlaybackRate = (clip: Pick<TimelineClip, "speed">): number =>
  Math.max(0.25, Math.min(4, clip.speed?.[0]?.rate ?? 1));

export const timelineClipDuration = (clip: Pick<TimelineClip, "sourceIn" | "sourceOut" | "speed">): number =>
  clipDuration(clip.sourceIn, clip.sourceOut) / clipPlaybackRate(clip);

export const formatTimecode = (value: number, fps = 24): string => {
  const safe = Math.max(0, value);
  const whole = Math.floor(safe);
  const frames = Math.floor((safe - whole) * fps);
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  return [hours, minutes, secs, frames].map((part) => String(part).padStart(2, "0")).join(":");
};

export const snapTime = (
  candidate: number,
  points: number[],
  thresholdSeconds: number,
): { value: number; snapped: boolean } => {
  let nearest = candidate;
  let distance = thresholdSeconds;
  for (const point of points) {
    const nextDistance = Math.abs(candidate - point);
    if (nextDistance <= distance) {
      nearest = point;
      distance = nextDistance;
    }
  }
  return { value: Math.max(0, nearest), snapped: nearest !== candidate };
};

/**
 * How far a ripple insert at `start` actually has to push the timeline.
 *
 * Room that is already empty is room the insert does not have to make. Shifting by the whole
 * inserted length regardless would move the following clip a gap's worth too far — so "put this
 * between those two" silently became "put this after that one and shove everything right",
 * whenever the two were not already touching.
 *
 * The room is the smallest hole across every track that gets rippled, not just the picture track:
 * pushing by the V1 gap alone would let the insert land on top of something sitting closer on A2.
 */
export function rippleShiftFor(
  tracks: { clips: { start: Rational }[] }[], start: number, span: number,
): number {
  const nextStart = Math.min(
    ...tracks.flatMap((track) => track.clips
      .map((clip) => seconds(clip.start))
      .filter((value) => value >= start - 0.0001)),
  );
  if (!Number.isFinite(nextStart)) return 0;
  return Math.max(0, span - Math.max(0, nextStart - start));
}
