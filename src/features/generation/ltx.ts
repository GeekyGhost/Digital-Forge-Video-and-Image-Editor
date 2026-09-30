import { LTX_MAX_PIXELS, snapLtxCanvas } from "./ltxGeometry";

export type LtxDimensions = { width: number; height: number; value: string };

/**
 * Preserve source aspect ratio inside the 16 GB LTX envelope.
 *
 * The grid is 32, from `ltxGeometry`, not the 64 this used to round to -- 64 is the TWO-stage
 * rule and there is no stage 2 here. The 256 floor is this panel's own: it is a sensible smallest
 * generation, not a model limit, so it stays here rather than in the shared module.
 */
export function fitLtxDimensions(sourceWidth = 768, sourceHeight = 512): LtxDimensions {
  const { width, height } = snapLtxCanvas(sourceWidth, sourceHeight, LTX_MAX_PIXELS, 256);
  return { width, height, value: `${width}x${height}` };
}
