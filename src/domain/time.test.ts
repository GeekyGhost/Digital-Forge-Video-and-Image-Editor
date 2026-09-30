import { describe, expect, it } from "vitest";
import { rippleShiftFor } from "./time";

const at = (...starts: number[]) => ({
  clips: starts.map((value) => ({ start: { numerator: Math.round(value * 1000), denominator: 1000 } })),
});

/**
 * The timeline half of "insert between these two and push the rest along only if needed".
 *
 * `insertGeneratedAvPairAfterClip` shifted everything downstream by the inserted clip's full
 * length. With the clips touching that is right; with a gap it moved the following clip a gap's
 * worth too far, so a bridge generated to fill a hole left a bigger hole than it started with.
 */
describe("how far a ripple insert has to push", () => {
  it("pushes by the whole length when the next clip is flush against the insert point", () => {
    expect(rippleShiftFor([at(0, 5)], 5, 5.167)).toBeCloseTo(5.167, 3);
  });

  it("uses an existing gap first and pushes only the overflow", () => {
    expect(rippleShiftFor([at(0, 10)], 5.167, 5.167)).toBeCloseTo(0.334, 3);
  });

  it("pushes nothing when the gap is larger than the insert", () => {
    expect(rippleShiftFor([at(0, 30)], 5, 5.167)).toBe(0);
  });

  it("pushes nothing when there is nothing downstream at all", () => {
    expect(rippleShiftFor([at(0)], 5, 5.167)).toBe(0);
    expect(rippleShiftFor([], 5, 5.167)).toBe(0);
  });

  it("measures the smallest hole across every rippled track, not just the picture", () => {
    // V1 has room to spare, but A2 has a clip sitting close behind the insert point. Pushing by
    // the V1 gap alone would drop the new material on top of it.
    const v1 = at(0, 20);
    const a2 = at(0, 6);
    expect(rippleShiftFor([v1, a2], 5, 5.167)).toBeCloseTo(4.167, 3);
  });
});
