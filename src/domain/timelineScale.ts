export type TimelineRulerTick = { time: number; major: boolean };

function niceStep(value: number): number {
  const safe = Math.max(0.001, value);
  const magnitude = 10 ** Math.floor(Math.log10(safe));
  const normalized = safe / magnitude;
  const multiplier = normalized <= 1 ? 1 : normalized <= 2 ? 2 : normalized <= 5 ? 5 : 10;
  return multiplier * magnitude;
}

/**
 * Create only the ruler marks currently near the viewport. Timeline lane width may represent hours,
 * but DOM size remains tied to the screen rather than one element per second of media.
 */
export function visibleTimelineTicks(
  durationSeconds: number, zoom: number, scrollLeft: number, viewportWidth: number, trackHeaderWidth = 164,
): TimelineRulerTick[] {
  const pixelsPerSecond = Math.max(0.01, zoom);
  const visibleStart = Math.max(0, scrollLeft / pixelsPerSecond);
  const visibleEnd = Math.min(durationSeconds, Math.max(visibleStart, (scrollLeft + viewportWidth - trackHeaderWidth) / pixelsPerSecond));
  const span = Math.max(1, visibleEnd - visibleStart);
  const overscan = Math.max(2, span * 0.75);
  const left = Math.max(0, visibleStart - overscan);
  const right = Math.min(durationSeconds, visibleEnd + overscan);
  const minor = niceStep(42 / pixelsPerSecond);
  const major = niceStep(110 / pixelsPerSecond);
  const first = Math.floor(left / minor) * minor;
  const count = Math.min(500, Math.ceil((right - first) / minor) + 1);
  return Array.from({ length: count }, (_, index) => {
    const time = first + index * minor;
    const ratio = time / major;
    return { time, major: Math.abs(ratio - Math.round(ratio)) < 1e-6 };
  }).filter((tick) => tick.time >= 0 && tick.time <= durationSeconds + 1e-6);
}
