import { describe, expect, it } from "vitest";
import {
  cloneH3DirectorSession, createH3DirectorSession, createH3DirectorWorkspace,
  h3DirectorSessionFingerprint, normalizeH3DirectorWorkspace,
} from "./h3Director";

describe("durable H3 Director workspace", () => {
  it("creates a usable first shot and the released Ref2V defaults", () => {
    const workspace = createH3DirectorWorkspace();
    expect(workspace.schemaVersion).toBe(1);
    expect(workspace.session.shots).toHaveLength(1);
    expect(workspace.session.frames).toBe(124);
    expect(workspace.session.samplingMode).toBe("base");
    expect(workspace.session.steps).toBe(20);
  });

  it("deep-clones the creative arrays captured by a take", () => {
    const session = createH3DirectorSession();
    session.references.push({
      id: "r1", assetId: "asset-1", kind: "image", intents: ["subject"],
      includeEmbeddedAudio: false, subjects: [{ id: "subject-1", text: "the pilot" }],
    });
    session.shots[0].description = "The pilot looks up.";
    const captured = cloneH3DirectorSession(session);
    session.references[0].intents.push("style");
    session.references[0].subjects![0].text = "changed";
    session.shots[0].description = "changed";
    expect(captured.references[0].intents).toEqual(["subject"]);
    expect(captured.references[0].subjects![0].text).toBe("the pilot");
    expect(captured.shots[0].description).toBe("The pilot looks up.");
  });

  it("fingerprints creative or sampling changes but not object identity", () => {
    const session = createH3DirectorSession();
    const same = cloneH3DirectorSession(session);
    expect(h3DirectorSessionFingerprint(same)).toBe(h3DirectorSessionFingerprint(session));
    same.steps += 10;
    expect(h3DirectorSessionFingerprint(same)).not.toBe(h3DirectorSessionFingerprint(session));
  });

  it("repairs partial state and rejects take entries without a reproducible session", () => {
    const normalized = normalizeH3DirectorWorkspace({
      schemaVersion: 1,
      updatedAt: "2026-09-11T00:00:00.000Z",
      session: { style: "Noir" } as ReturnType<typeof createH3DirectorSession>,
      takes: [{ id: "old", videoPath: "a.mp4", audioPath: "a.wav" } as never],
    });
    expect(normalized.session.style).toBe("Noir");
    expect(normalized.session.shots).toHaveLength(1);
    expect(normalized.takes).toEqual([]);
  });
});
