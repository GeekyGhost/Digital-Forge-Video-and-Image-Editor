/** The element library (VFX Phase 3): Rust owns `elements/<id>/` (src-tauri/src/ai/elements.rs). */

import { invoke } from "@tauri-apps/api/core";

export type ElementManifest = {
  version: number;
  id: string;
  name: string;
  frameCount: number;
  fps: number;
  width: number;
  height: number;
  pattern: string;
  proxy?: string;
  loopIn: number;
  loopOut?: number;
  crossfade: number;
  source: "wan-alpha" | "import" | string;
  prompt?: string;
  seed?: number;
  createdAt: string;
  /** Anything else the manifest carries, e.g. `proxyError` or the generation's stats. */
  [key: string]: unknown;
};

export type ElementEntry = { dir: string; manifest: ElementManifest };

export const listElements = () => invoke<ElementEntry[]>("element_list");
export const importElement = (folder: string, name: string, fps: number) =>
  invoke<ElementEntry>("element_import", { request: { folder, name, fps } });
export const generateElement = (request: { name: string; prompt: string; frames: number; seed: number }) =>
  invoke<ElementEntry>("element_generate", { request });
export const updateElement = (request: { id: string; name?: string; loopIn?: number; loopOut?: number; crossfade?: number }) =>
  invoke<ElementEntry>("element_update", { request });
export const buildElementProxy = (id: string) => invoke<ElementEntry>("element_build_proxy", { id });
