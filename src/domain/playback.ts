/**
 * Whole-file Web Audio decoding is useful for responsive forwards/reverse scrub grains, but its
 * memory cost follows the uncompressed PCM duration rather than the compressed file size. Keep it
 * for short material only; long-form sources still scrub picture and use normal media playback.
 */
export const SCRUB_AUDIO_DECODE_LIMIT_SECONDS = 180;
export const SCRUB_VIDEO_DECODE_LIMIT_SECONDS = 30;

export function canDecodeWholeAssetForScrub(durationSeconds: number, kind: "audio" | "video" = "audio"): boolean {
  const limit = kind === "video" ? SCRUB_VIDEO_DECODE_LIMIT_SECONDS : SCRUB_AUDIO_DECODE_LIMIT_SECONDS;
  return Number.isFinite(durationSeconds)
    && durationSeconds > 0
    && durationSeconds <= limit;
}

export type TimelineFollowRequest = {
  playheadSeconds: number;
  zoom: number;
  scrollLeft: number;
  viewportWidth: number;
  scrollWidth: number;
  following: boolean;
  trackHeaderWidth?: number;
};

export type TimelineFollowResult = { scrollLeft: number; following: boolean };

/**
 * Keep a playing programme clock visible without constantly recentering a playhead the editor has
 * just parked. Follow begins near the right edge (or whenever the playhead is off-screen), then
 * holds it at 68% of the usable lane while playback advances.
 */
export function timelineFollowScroll(request: TimelineFollowRequest): TimelineFollowResult {
  const header = request.trackHeaderWidth ?? 164;
  const viewport = Math.max(1, request.viewportWidth);
  const usable = Math.max(1, viewport - header);
  const playheadPixel = header + Math.max(0, request.playheadSeconds) * Math.max(0.01, request.zoom);
  const screenPixel = playheadPixel - Math.max(0, request.scrollLeft);
  const shouldBegin = screenPixel < header + 8 || screenPixel > header + usable * 0.82;

  if (!request.following && !shouldBegin) {
    return { scrollLeft: request.scrollLeft, following: false };
  }

  const maximum = Math.max(0, request.scrollWidth - viewport);
  const target = Math.min(maximum, Math.max(0, playheadPixel - (header + usable * 0.68)));
  return { scrollLeft: target, following: true };
}

export function boundedMediaTime(target: number, sourceIn: number, sourceOut: number): number {
  const low = Math.max(0, Number.isFinite(sourceIn) ? sourceIn : 0);
  const high = Math.max(low, Number.isFinite(sourceOut) ? sourceOut - 0.02 : low);
  return Math.min(Math.max(Number.isFinite(target) ? target : low, low), high);
}
