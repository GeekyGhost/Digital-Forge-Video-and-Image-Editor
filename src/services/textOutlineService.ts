/**
 * VFX Phase 6: "Convert text to outlines". The page lays the text out exactly as the text layer
 * draws it (`textGlyphRuns`), Rust reads the glyph outlines from the installed font (`fonts.rs`),
 * and `textPath` puts them together. The result is a path layer in the text layer's placement.
 */
import { invoke } from "@tauri-apps/api/core";
import { textGlyphRuns } from "../domain/photoCompose";
import type { PhotoLayer } from "../domain/photoDocument";
import { generic } from "../domain/photoText";
import { fontRequest, textPath, type FontOutlines } from "../domain/textOutline";

export async function textLayerToPath(layer: PhotoLayer): Promise<{ layer: PhotoLayer; notes: string[] }> {
  const text = layer.text;
  if (!text) throw new Error("not a text layer");
  if (layer.shape) throw new Error("lettering inside a shape cannot be converted yet; convert a plain text layer");
  const { family, generic: fallback } = fontRequest(text.fontFamily, generic);
  const font = await invoke<FontOutlines>("text_outlines", {
    family, generic: fallback, weight: Math.round(text.weight), italic: text.italic, text: text.content,
  });
  const laid = textGlyphRuns(text);
  const { path, missing } = textPath(text, laid.runs, laid, laid.fontSize, font);
  const notes: string[] = [];
  if (font.substituted) notes.push(`"${family}" is not installed; outlines use ${font.face.family}, as the canvas does`);
  if (font.synthetic) notes.push(`the font has no ${text.italic ? "italic " : ""}${text.weight} face; the canvas faked it, the outlines use ${font.face.family} ${font.face.weight}${font.face.italic ? " italic" : ""}`);
  if (missing.length) notes.push(`no glyph in ${font.face.family} for ${missing.map((ch) => `"${ch}"`).join(" ")} (left out)`);
  if (text.shadow > 0) notes.push("the drop shadow is not converted");
  if (text.panel !== "none") notes.push("the panel behind the text is not converted");
  const { text: _text, ...rest } = layer;
  return {
    layer: {
      ...rest, id: crypto.randomUUID(), name: `${layer.name} (outlines)`, path,
      cropLeft: 0, cropTop: 0, cropRight: 0, cropBottom: 0,
    },
    notes,
  };
}
