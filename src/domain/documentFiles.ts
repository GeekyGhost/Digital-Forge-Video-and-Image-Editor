/**
 * Every file a compositor document reads at one composition frame -- the loader's fetch list.
 *
 * Moved out of Photo Edit in VFX Phase 7 so the programme monitor and the timeline export load
 * exactly what the Clip Editor loads (and mask VFX mattes with it).
 */

import type { ComposeScope } from "./photoCompose";
import { layersOf, type PhotoDocument } from "./photoDocument";
import { maskFilePaths } from "./photoMask";
import { compositionRate, mediaFramesAt, scopeFrame } from "./photoTime";
import { depthFramePaths } from "./photoTrack";

/** Every file the document reads at composition frame `frame`: layer pictures, mattes, depth, fills. */
export function filesWantedAt(document: PhotoDocument, frame: number): Set<string> {
  const wanted = new Set<string>();
  // A text layer names no file -- it is drawn, not loaded -- so an empty path is skipped rather
  // than handed to the decoder, which would fail and blank the workspace over nothing.
  for (const layer of layersOf(document)) if (layer.sourcePath) wanted.add(layer.sourcePath);
  if (document.depthPath) wanted.add(document.depthPath);
  // Mask sequences follow a frame: the source frame on screen for a media layer, the composition
  // frame for anything else (`scopeFrame`), so the loader fetches that frame's matte.
  const rateNow = compositionRate(document);
  const scopes: Array<Pick<ComposeScope, "nodes" | "masks"> & { frame: number }> = [
    { nodes: document.nodes, masks: document.masks, frame },
    ...layersOf(document).map((layer) => ({
      nodes: layer.nodes, masks: layer.masks,
      frame: scopeFrame(layer, layer.media ? mediaFramesAt(layer.media, frame, rateNow) : undefined, frame),
    })),
  ];
  // VFX Phase 4: depth occlusion reads another layer's depth frame at the frame that layer shows.
  for (const path of depthFramePaths(document, frame, rateNow)) wanted.add(path);
  for (const scope of scopes) {
    for (const mask of scope.masks) for (const path of maskFilePaths(mask, scope.frame)) wanted.add(path);
    for (const node of scope.nodes) {
      if (node.kind === "remove" && node.cachePath) wanted.add(node.cachePath);
      // VFX Phase 7: a mask VFX node's subject matte for this frame (already resolved into the node).
      if (node.kind === "maskVfx" && node.enabled && node.mattePath) wanted.add(node.mattePath);
    }
  }
  return wanted;
}
