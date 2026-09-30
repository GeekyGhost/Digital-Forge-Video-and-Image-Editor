import { describe, expect, it } from "vitest";
import { commitLine, formatGb, holderTag, shortOnCommit, vramLine, type ResourceVerdict } from "./resourceGuard";

const crashDay: ResourceVerdict = {
  job: "ltx", jobId: "ltx", label: "LTX generate / extend / bridge", requirementSource: "handoff", ok: false, measured: true,
  commitNeededGb: 55, commitFreeGb: 24, commitLimitGb: 86, commitReclaimableGb: 0,
  vramNeededGb: 13, vramFreeGb: 15, vramTotalGb: 16, vramSource: "nvml",
  ramFreeGb: 30, ramTotalGb: 65,
  shortfalls: ["needs 55 GB of free commit, 24.0 GB is free (commit limit 86 GB)"],
  holders: [{ pid: 900, name: "llama-server.exe", privateGb: 23, kind: "known-offender", note: "local LLM server", reclaimable: false }],
  hasReleasableAppWorker: false, measureMs: 14,
};

describe("resource guard formatting", () => {
  it("formats gigabytes with one decimal below ten", () => {
    expect(formatGb(23.04)).toBe("23 GB");
    expect(formatGb(6.25)).toBe("6.3 GB");
    expect(formatGb(null)).toBe("unknown");
    expect(formatGb(Number.NaN)).toBe("unknown");
  });

  it("describes the commit shortfall including reclaimable worker memory", () => {
    expect(commitLine(crashDay)).toBe("24 GB free of 86 GB commit · needs 55 GB");
    expect(shortOnCommit(crashDay)).toBe(true);
    const retake = { ...crashDay, commitReclaimableGb: 40 };
    expect(commitLine(retake)).toBe("64 GB free of 86 GB commit · needs 55 GB");
    expect(shortOnCommit(retake)).toBe(false);
  });

  it("says when VRAM could not be measured", () => {
    expect(vramLine(crashDay)).toBe("15 GB free of 16 GB VRAM · needs 13 GB");
    expect(vramLine({ ...crashDay, vramFreeGb: null, vramTotalGb: null, vramSource: "unavailable" }))
      .toBe("VRAM not measured (unavailable) · needs 13 GB");
  });

  it("tags holders", () => {
    expect(holderTag(crashDay.holders[0])).toBe("local LLM server");
    expect(holderTag({ ...crashDay.holders[0], reclaimable: true })).toBe("Replaced by this job");
    expect(holderTag({ ...crashDay.holders[0], kind: "other", note: "" })).toBe("");
  });
});
