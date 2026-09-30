import { clipPlaybackRate, seconds, timelineClipDuration } from "./time";
import type { TimelineClip } from "./types";

/**
 * One place that decides where every clip actually sits once transitions are accounted for.
 *
 * A D-second dissolve needs D seconds of spare material from somewhere:
 *
 *   - **Handles** -- frames beyond the clip's in-point. Using them costs nothing and the timeline
 *     is untouched. This is what Premiere, Resolve, Final Cut and Kdenlive do.
 *   - **Visible frames** -- the clips genuinely overlap and the programme gets shorter. This is
 *     what ffmpeg's `xfade`, CapCut and Shotcut's drag-overlap do, and it is the ONLY option when
 *     a clip starts at frame 0, which every generated LTX clip does.
 *   - **Invented frames** -- freeze the first frame and blend into a still. Premiere alone does
 *     this, and stripes it with a warning because it looks wrong. Not an option here.
 *
 * The important part is that this is computed ONCE and read by the monitor, the timeline and the
 * export alike. Having the export ripple while the timeline stayed butt-joined is precisely how
 * the preview and the exported file came to disagree.
 */
export type JunctionLayout = {
  /** Effective programme start per clip id, after every upstream ripple. */
  start: Map<string, number>;
  /** Timeline seconds this clip borrows from before its in-point for its incoming transition. */
  borrowed: Map<string, number>;
  /** Effective length of this clip's incoming transition, after capping. */
  transition: Map<string, number>;
  /** Seconds this clip's incoming transition took out of visible material. */
  consumed: Map<string, number>;
  /** Programme duration -- what the export will actually be. */
  duration: number;
  /** Total seconds lost to ripple across the whole sequence. */
  rippled: number;
};

const EMPTY: JunctionLayout = {
  start: new Map(), borrowed: new Map(), transition: new Map(), consumed: new Map(),
  duration: 0, rippled: 0,
};

/** `clips` must be the enabled picture clips of one track, ascending by start. */
export function resolveJunctionLayout(clips: TimelineClip[]): JunctionLayout {
  if (!clips.length) return EMPTY;
  const layout: JunctionLayout = {
    start: new Map(), borrowed: new Map(), transition: new Map(), consumed: new Map(),
    duration: 0, rippled: 0,
  };
  let shift = 0;
  clips.forEach((clip, index) => {
    const previous = index > 0 ? clips[index - 1] : undefined;
    let requested = 0;
    if (previous) {
      const outSet = Boolean(previous.transitionOut && previous.transitionOut !== "none");
      const inSet = Boolean(clip.transitionIn && clip.transitionIn !== "none");
      if (outSet || inSet) {
        requested = Math.max(
          outSet ? previous.transitionOutDuration ?? previous.transitionDuration ?? 0.5 : 0,
          inSet ? clip.transitionDuration ?? 0.5 : 0,
        );
      }
    }
    // Never longer than either neighbour can give up. A dissolve longer than the clip is
    // meaningless, and ffmpeg truncates it silently rather than erroring.
    const capped = previous
      ? Math.max(0, Math.min(requested, timelineClipDuration(clip) * 0.9, timelineClipDuration(previous) * 0.9))
      : 0;
    // Head handle in TIMELINE seconds: sourceIn is source time, and playback rate divides it.
    const handle = seconds(clip.sourceIn) / Math.max(0.01, clipPlaybackRate(clip));
    const borrowed = Math.min(capped, handle);
    const consumed = Math.max(0, capped - borrowed);
    shift += consumed;

    layout.start.set(clip.id, Math.max(0, seconds(clip.start) - shift));
    layout.borrowed.set(clip.id, borrowed);
    layout.transition.set(clip.id, capped);
    layout.consumed.set(clip.id, consumed);
    layout.rippled += consumed;
  });
  const last = clips[clips.length - 1];
  layout.duration = (layout.start.get(last.id) ?? 0) + timelineClipDuration(last);
  return layout;
}

/**
 * Map an authored time on any track onto the programme timeline.
 *
 * Applied to graphics, captions and audio so they move with the picture. Without this they keep
 * their absolute times while the picture slides earlier, which is exactly the drift that made
 * lower thirds land early in exports.
 */
export function mapToProgramme(layout: JunctionLayout, clips: TimelineClip[], time: number): number {
  let shift = 0;
  for (const clip of clips) {
    if (time < seconds(clip.start) - 1e-6) break;
    shift = seconds(clip.start) - (layout.start.get(clip.id) ?? seconds(clip.start));
  }
  return Math.max(0, time - shift);
}
