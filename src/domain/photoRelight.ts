/**
 * Depth-based relighting.
 *
 * A monocular depth map is not geometry, so this is not physics -- it is a look. Given a per-pixel
 * distance estimate you can recover a plausible surface normal, and a plausible normal is enough to
 * put a light somewhere and have the picture respond to it in a way that reads as lighting rather
 * than as a gradient laid on top. It is convincing on a clear subject against a simple background
 * and obviously wrong on hair, glass and fine edges, which is the honest limit of the technique.
 *
 * Deliberately absent: cast shadows. Working out what occludes what needs real geometry, and a
 * shadow in the wrong place is far more damaging to a picture than no shadow at all.
 */

import type { RelightSettings } from "./photoDocument";

const LUMA_R = 0.2126, LUMA_G = 0.7152, LUMA_B = 0.0722;

const srgbToLinear = (() => {
  const table = new Float32Array(256);
  for (let index = 0; index < 256; index++) {
    const value = index / 255;
    table[index] = value <= 0.04045 ? value / 12.92 : Math.pow((value + 0.055) / 1.055, 2.4);
  }
  return table;
})();

const linearToByte = (value: number) => {
  if (value <= 0) return 0;
  if (value >= 1) return 255;
  const encoded = value <= 0.0031308 ? 12.92 * value : 1.055 * Math.pow(value, 1 / 2.4) - 0.055;
  return Math.round(encoded * 255);
};

/**
 * Surface normals from a depth buffer, by central difference.
 *
 * `strength` scales how much a depth step tilts the surface. It is a taste control rather than a
 * measurement: depth maps are relative, so there is no true value, and too much of it turns every
 * compression artefact in the depth into a facet.
 */
export function normalsFromDepth(
  depth: Uint8Array, width: number, height: number, strength = 2.2,
): Float32Array {
  const normals = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    const up = y > 0 ? y - 1 : y;
    const down = y < height - 1 ? y + 1 : y;
    for (let x = 0; x < width; x++) {
      const left = x > 0 ? x - 1 : x;
      const right = x < width - 1 ? x + 1 : x;
      // Depth conventionally rises towards the camera, so a step to the right that gets nearer
      // tilts the surface to the left -- hence the negated gradients.
      const dx = (depth[y * width + right] - depth[y * width + left]) / 255 * strength;
      const dy = (depth[down * width + x] - depth[up * width + x]) / 255 * strength;
      const length = Math.sqrt(dx * dx + dy * dy + 1);
      const at = (y * width + x) * 3;
      normals[at] = -dx / length;
      normals[at + 1] = -dy / length;
      normals[at + 2] = 1 / length;
    }
  }
  return normals;
}

/**
 * Light an RGBA buffer in place using a depth map of the same dimensions.
 *
 * The light is a point in front of the picture: its direction changes across the frame, which is
 * what makes dragging it read as moving a lamp rather than rotating a gradient. Ambient is a floor
 * so nothing goes fully black, and rim is a Fresnel-ish term that catches surfaces turning away.
 */
export function applyRelight(
  data: Uint8ClampedArray, width: number, height: number,
  depth: Uint8Array, settings: RelightSettings,
): void {
  if (depth.length !== width * height) return;
  const normals = normalsFromDepth(depth, width, height);
  const longEdge = Math.max(width, height);
  const lightX = settings.x * width;
  const lightY = settings.y * height;
  const lightZ = Math.max(0.05, settings.distance) * longEdge;
  const [colorR, colorG, colorB] = settings.color;
  const ambient = settings.ambient;
  const intensity = settings.intensity;
  const rim = settings.rim;
  // Soft light spreads the falloff out; hard light concentrates it. Expressed as an exponent on the
  // lambert term because that keeps the bright side from clipping as the control moves.
  const focus = 1 + (1 - Math.min(1, Math.max(0, settings.softness))) * 1.6;

  for (let y = 0, index = 0, at = 0; y < height; y++) {
    for (let x = 0; x < width; x++, index++, at += 4) {
      const toX = lightX - x, toY = lightY - y;
      const distance = Math.sqrt(toX * toX + toY * toY + lightZ * lightZ);
      const dirX = toX / distance, dirY = toY / distance, dirZ = lightZ / distance;
      const normal = index * 3;
      let lambert = normals[normal] * dirX + normals[normal + 1] * dirY + normals[normal + 2] * dirZ;
      lambert = lambert > 0 ? Math.pow(lambert, focus) : 0;
      const falloff = 1 / (1 + Math.pow(distance / (0.95 * longEdge), 2) * 1.8);
      const key = intensity * lambert * falloff;

      // Depth also dims what is far away, which is what stops a relit background from staying as
      // bright as the subject and giving the whole thing away.
      const nearness = 0.55 + 0.45 * (depth[index] / 255);
      const facing = 1 - Math.abs(normals[normal + 2]);
      const rimTerm = rim * Math.pow(facing, 3) * Math.max(0, -(normals[normal] * dirX + normals[normal + 1] * dirY));

      const red = srgbToLinear[data[at]];
      const green = srgbToLinear[data[at + 1]];
      const blue = srgbToLinear[data[at + 2]];
      const luma = red * LUMA_R + green * LUMA_G + blue * LUMA_B;

      data[at] = linearToByte(red * (ambient + key * colorR * nearness) + rimTerm * luma * colorR);
      data[at + 1] = linearToByte(green * (ambient + key * colorG * nearness) + rimTerm * luma * colorG);
      data[at + 2] = linearToByte(blue * (ambient + key * colorB * nearness) + rimTerm * luma * colorB);
    }
  }
}
