import { describe, expect, it } from "vitest";
import { audioWindow } from "./programmeExport";
import { audioGainAt } from "../monitor/ProgrammeMonitor";
import type { Programme, ProgrammeAudio } from "../../domain/programme";

const entry: ProgrammeAudio = { id: "a", path: "a.mp4", start: 2, duration: 4, sourceIn: 10, rate: 2, volumeDb: 0, pan: 0, fadeIn: 1, fadeOut: 1, kind: "picture" };

describe("programme audio", () => {
  it("cuts a window out of the programme's sound", () => {
    expect(audioWindow([entry], 3, 2)).toEqual([{ ...entry, start: 0, duration: 2, sourceIn: 12, fadeIn: 0, fadeOut: 0 }]);
    expect(audioWindow([entry], 0, 10)).toEqual([entry]);
    expect(audioWindow([entry], 7, 1)).toEqual([]);
  });

  it("follows fades and the programme's dips in the monitor", () => {
    const programme = { duration: 10, headFade: { color: "#000", duration: 2 }, tailFade: undefined } as unknown as Programme;
    expect(audioGainAt(entry, programme, 2.5)).toBeCloseTo(0.5);
    expect(audioGainAt({ ...entry, start: 0 }, programme, 1.5)).toBeCloseTo(1 * 0.75);
    expect(audioGainAt(entry, programme, 4)).toBeCloseTo(1);
    expect(audioGainAt(entry, programme, 5.5)).toBeCloseTo(0.5);
    expect(audioGainAt({ ...entry, volumeDb: -60 }, programme, 4)).toBe(0);
  });
});
