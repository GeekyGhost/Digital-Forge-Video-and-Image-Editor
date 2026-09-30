export type KiwiStrategy = "stable" | "long";

export const KIWI_WORKING_FPS = 15;
export const KIWI_MAX_FRAMES = 81;
export const KIWI_OVERLAP_FRAMES = 25;

export type KiwiWindowPlan = {
  windows: number;
  frames: number;
  outputSeconds: number;
  overlapFrames: number;
  supported: boolean;
  message?: string;
};

export function planKiwiRange(durationSeconds: number, strategy: KiwiStrategy): KiwiWindowPlan {
  const frames = Math.max(0, Math.round(durationSeconds * KIWI_WORKING_FPS));
  if (frames < 5) {
    return { windows: 0, frames, outputSeconds: frames / KIWI_WORKING_FPS, overlapFrames: 0, supported: false, message: "Range is shorter than five working frames" };
  }
  if (strategy === "stable") {
    const supported = frames <= KIWI_MAX_FRAMES;
    return {
      windows: supported ? 1 : 0,
      frames,
      outputSeconds: frames / KIWI_WORKING_FPS,
      overlapFrames: 0,
      supported,
      message: supported ? undefined : `Stable mode is limited to ${(KIWI_MAX_FRAMES / KIWI_WORKING_FPS).toFixed(1)} seconds at ${KIWI_WORKING_FPS} FPS`,
    };
  }
  const stride = KIWI_MAX_FRAMES - KIWI_OVERLAP_FRAMES;
  const windows = frames <= KIWI_MAX_FRAMES ? 1 : 1 + Math.ceil((frames - KIWI_MAX_FRAMES) / stride);
  return { windows, frames, outputSeconds: frames / KIWI_WORKING_FPS, overlapFrames: windows > 1 ? KIWI_OVERLAP_FRAMES : 0, supported: true };
}
