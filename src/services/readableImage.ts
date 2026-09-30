/**
 * Loading a picture the canvas is actually allowed to read back.
 *
 * Everything the compositor does needs `getImageData`, which only works if nothing tainted the
 * canvas. The asset protocol is its own origin, so whether it taints depends on headers this app
 * does not set, and getting it wrong is a SecurityError at the first slider that kills a workspace.
 * Rather than assume either way, this draws one pixel and tries to read it: if that works the fast
 * path is fine, and if it does not the file is fetched through the app and re-decoded from a data
 * URL, which is same-origin and can never taint anything.
 *
 * Shared rather than copied into each workspace: it is exactly the kind of small, load-bearing,
 * easy-to-get-subtly-different helper that this project has been bitten by twice.
 */

import { convertFileSrc } from "@tauri-apps/api/core";
import type { LayerSource } from "../domain/photoCompose";
import { readImageBase64 } from "./modelService";

const MIME_TYPES: Record<string, string> = {
  png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg",
  webp: "image/webp", tif: "image/tiff", tiff: "image/tiff", bmp: "image/bmp",
};

function decodeImage(source: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image();
    // Asks the asset protocol for permission to read the pixels back. Harmless if it says nothing.
    image.crossOrigin = "anonymous";
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("the file could not be decoded as an image"));
    image.src = source;
  });
}

export async function loadReadableImage(path: string): Promise<HTMLImageElement> {
  const direct = await decodeImage(convertFileSrc(path));
  const probe = window.document.createElement("canvas");
  probe.width = 1;
  probe.height = 1;
  const context = probe.getContext("2d")!;
  context.drawImage(direct, 0, 0, 1, 1);
  try {
    context.getImageData(0, 0, 1, 1);
    return direct;
  } catch {
    const extension = path.split(".").pop()?.toLowerCase() ?? "png";
    const base64 = await readImageBase64(path);
    return decodeImage(`data:${MIME_TYPES[extension] ?? "image/png"};base64,${base64}`);
  }
}

/** Decode a set of paths into something the compositor can place. Duplicates are loaded once. */
export async function loadLayerSources(paths: Iterable<string>): Promise<Map<string, LayerSource>> {
  const unique = [...new Set(paths)];
  const loaded = await Promise.all(unique.map(async (path) => {
    const image = await loadReadableImage(path);
    const width = image.naturalWidth || image.width;
    const height = image.naturalHeight || image.height;
    const canvas = window.document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;
    canvas.getContext("2d", { willReadFrequently: true })!.drawImage(image, 0, 0);
    return [path, { image: canvas as CanvasImageSource, width, height }] as const;
  }));
  return new Map(loaded);
}
