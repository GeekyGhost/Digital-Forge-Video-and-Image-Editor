/**
 * Spike 0.6: Wan Alpha on 16 GB. Each run goes through the app's resource guard
 * (JobKind::WanAlpha), so a shortfall shows the warning in the main window, and the job monitor
 * appends the run's real peaks to .aiv/resource-log.jsonl.
 *
 * Plan: the three handoff prompts at 832x480x81, then the smoke prompt again at 33 frames.
 * Prompts follow Wan-Alpha's guidance: transparent background, style, shot type, subject.
 */
import type { Backend } from "./backend";

export const WAN_ALPHA_PLAN = [
  { label: "smoke", frames: 81, prompt: "Transparent background. Realistic style. Close-up. A thin wisp of grey cigarette smoke rises slowly, curling and thinning as it drifts upward, soft and translucent, fading away at the top." },
  { label: "fireball", frames: 81, prompt: "Transparent background. Realistic style. Medium shot. A roiling fireball of orange and yellow flame bursts outward, glowing embers scattering, its edges soft with translucent heat haze." },
  { label: "sparks", frames: 81, prompt: "Transparent background. Fantasy style. Medium shot. Glittering blue and violet magic sparks swirl in a rising spiral, leaving soft glowing trails that fade out." },
  { label: "smoke", frames: 33, prompt: "Transparent background. Realistic style. Close-up. A thin wisp of grey cigarette smoke rises slowly, curling and thinning as it drifts upward, soft and translucent, fading away at the top." },
];

export async function runWanAlphaSpike(backend: Backend, log: (line: string) => void) {
  const runs: unknown[] = [];
  for (const item of WAN_ALPHA_PLAN) {
    log(`0.6 Wan Alpha: ${item.label} ${item.frames} frames (several minutes)`);
    try {
      runs.push(await backend.wanAlpha({ ...item, seed: 42 }));
    } catch (error) {
      runs.push({ label: item.label, frames: item.frames, error: String(error) });
      // A cancelled resource warning or a missing setup applies to every run; stop here.
      if (/Cancelled|Not started|not set up/.test(String(error))) break;
    }
  }
  return {
    runs,
    passBar: "runs without spilling (no 0xC0000005, free commit never exhausted); soft alpha visibly correct on the checkerboard preview",
    note: "Look at checkerboard.mp4 in each outDir. softAlphaFraction > 0 means alpha is not just a hard mask.",
  };
}
