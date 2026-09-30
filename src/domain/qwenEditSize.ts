/**
 * Output size for a Qwen-Image-Edit-2511 edit.
 *
 * By default the edit follows Image 1 (the canvas image being edited): the worker reproduces the
 * pipeline's own `calculate_dimensions` -- Image 1's aspect ratio at about one megapixel, snapped
 * to 32 (`canvas_output_size` in image_studio/qwen.py). An override asks for an exact size instead,
 * snapped to 16 (what the pipeline accepts: VAE factor 8 times the 2x2 latent packing). Qwen was
 * trained around a megapixel; larger sizes work, but take longer and drift more.
 */

export type EditSize = { width: number; height: number };

/** Image 1's shape at ~1 MP, snapped to 32: the worker's default (`canvas_output_size`). */
export const QWEN_VAE_AREA = 1024 * 1024;
export const EDIT_SIZE_MIN = 256;
export const EDIT_SIZE_MAX = 2048;
/** The largest output area the edit accepts (2048 x 2048). */
export const EDIT_AREA_MAX = EDIT_SIZE_MAX * EDIT_SIZE_MAX;
/** Above this, the UI says the edit will be slower and less faithful. */
export const EDIT_AREA_COMFORT = 1.6 * 1024 * 1024;

export const EDIT_SIZE_PRESETS: { value: string; label: string; size: EditSize }[] = [
  { value: "1024x1024", label: "1024 × 1024 · 1:1", size: { width: 1024, height: 1024 } },
  { value: "1344x768", label: "1344 × 768 · 16:9", size: { width: 1344, height: 768 } },
  { value: "768x1344", label: "768 × 1344 · 9:16", size: { width: 768, height: 1344 } },
  { value: "1152x896", label: "1152 × 896 · 4:3", size: { width: 1152, height: 896 } },
  { value: "896x1152", label: "896 × 1152 · 3:4", size: { width: 896, height: 1152 } },
  { value: "1216x832", label: "1216 × 832 · 3:2", size: { width: 1216, height: 832 } },
  { value: "832x1216", label: "832 × 1216 · 2:3", size: { width: 832, height: 1216 } },
  { value: "1536x640", label: "1536 × 640 · 21:9", size: { width: 1536, height: 640 } },
  { value: "1920x1088", label: "1920 × 1088 · 1080p (2 MP, slower)", size: { width: 1920, height: 1088 } },
];

/** What the worker makes when there is no override (see `canvas_output_size`). */
export function defaultEditSize(image: EditSize | undefined): EditSize | undefined {
  if (!image || image.width <= 0 || image.height <= 0) return undefined;
  const ratio = image.width / image.height;
  const width = Math.sqrt(QWEN_VAE_AREA * ratio);
  return { width: Math.max(32, Math.round(width / 32) * 32), height: Math.max(32, Math.round(width / ratio / 32) * 32) };
}

/** A requested size as the edit will make it: multiples of 16, 256..2048 a side, at most 2048². */
export function snapEditSize(size: EditSize): EditSize {
  let width = Number.isFinite(size.width) ? size.width : 1024;
  let height = Number.isFinite(size.height) ? size.height : 1024;
  // Shrink (never stretch) into the limits, keeping the shape: the area cap and the longest side.
  const k = Math.min(1, Math.sqrt(EDIT_AREA_MAX / Math.max(1, width * height)), EDIT_SIZE_MAX / Math.max(width, height));
  width *= k; height *= k;
  const snap = (value: number) => Math.min(EDIT_SIZE_MAX, Math.max(EDIT_SIZE_MIN, Math.round(value / 16) * 16));
  return { width: snap(width), height: snap(height) };
}

/** The override to send (undefined = follow Image 1) for a dropdown choice and the manual fields. */
export function resolveEditSize(choice: string, manual: EditSize): EditSize | undefined {
  if (choice === "default") return undefined;
  if (choice === "manual") return snapEditSize(manual);
  const preset = EDIT_SIZE_PRESETS.find((item) => item.value === choice);
  return preset ? { ...preset.size } : undefined;
}
