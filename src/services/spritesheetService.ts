/** VFX Phase 5: spritesheet export files (src-tauri/src/ai/spritesheet.rs), under exports/spritesheets/. */
import { invoke } from "@tauri-apps/api/core";

/** A new export folder; resolves to its name. */
export const spritesheetFolder = (name: string) => invoke<string>("spritesheet_folder", { name });

/** One file of an export (raw body). Resolves to its full path. */
export const spritesheetWrite = (folder: string, file: string, bytes: Uint8Array) =>
  invoke<string>("spritesheet_write", bytes, { headers: { "x-sheet-folder": folder, "x-sheet-file": file } });
