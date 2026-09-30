/**
 * Clone stamp and spot heal.
 *
 * Both are pure canvas work, no model, and both matter next to LaMa rather than instead of it: LaMa
 * is a full model run and a GPU queue slot, which is the wrong price for a dust speck or a seam
 * between two kitbashed pieces. These run in a frame.
 *
 * A retouch node samples ITS OWN INPUT -- the picture as the stack has it at that point -- so the
 * result is well defined wherever the node sits, and the strokes stay re-editable because they are
 * stored as paths rather than as painted pixels, exactly like masks.
 */

/** One clone or heal gesture. Coordinates and radius are normalised to the picture. */
export type RetouchStroke = {
  points: Array<[number, number]>;
  radius: number;
  hardness: number;
  /** Where to sample from, relative to where the brush is. Set by alt-clicking a source point. */
  offsetX: number;
  offsetY: number;
  /**
   * Heal matches the copied patch's brightness to what surrounds the destination, so a sample taken
   * from a lighter part of the picture does not leave a bright disc behind. Clone copies verbatim.
   */
  heal: boolean;
};

export function applyRetouch(
  data: Uint8ClampedArray, width: number, height: number, strokes: RetouchStroke[],
): void {
  if (!strokes.length) return;
  const longEdge = Math.max(width, height);
  for (const stroke of strokes) {
    if (!stroke.points.length) continue;
    // Snapshot per stroke, not per dab: within one gesture the source has to stay still, or the
    // brush chases its own output and smears. Between strokes the picture has moved on, which is
    // what lets a second stroke clean up after the first.
    const source = new Uint8ClampedArray(data);
    const radius = Math.max(1, stroke.radius * longEdge);
    const step = Math.max(1, radius * 0.35);
    const offsetX = Math.round(stroke.offsetX * width);
    const offsetY = Math.round(stroke.offsetY * height);
    if (offsetX === 0 && offsetY === 0 && !stroke.heal) continue;

    let [previousX, previousY] = stroke.points[0];
    stamp(data, source, width, height, previousX * width, previousY * height, radius, offsetX, offsetY, stroke);
    for (let index = 1; index < stroke.points.length; index++) {
      const [nextX, nextY] = stroke.points[index];
      const x0 = previousX * width, y0 = previousY * height;
      const x1 = nextX * width, y1 = nextY * height;
      const distance = Math.hypot(x1 - x0, y1 - y0);
      const steps = Math.ceil(distance / step);
      for (let at = 1; at <= steps; at++) {
        const t = at / steps;
        stamp(data, source, width, height,
          x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, radius, offsetX, offsetY, stroke);
      }
      previousX = nextX;
      previousY = nextY;
    }
  }
}

function stamp(
  data: Uint8ClampedArray, source: Uint8ClampedArray, width: number, height: number,
  centreX: number, centreY: number, radius: number,
  offsetX: number, offsetY: number, stroke: RetouchStroke,
): void {
  const left = Math.max(0, Math.floor(centreX - radius));
  const right = Math.min(width - 1, Math.ceil(centreX + radius));
  const top = Math.max(0, Math.floor(centreY - radius));
  const bottom = Math.min(height - 1, Math.ceil(centreY + radius));
  if (right < left || bottom < top) return;
  const solid = radius * Math.min(0.999, stroke.hardness);
  const fade = Math.max(1e-6, radius - solid);

  // Heal shifts the sampled patch so its average matches what already surrounds the spot. Measuring
  // the destination on a ring just outside the brush is what keeps a heal from averaging in the
  // blemish it was called to remove.
  let shiftR = 0, shiftG = 0, shiftB = 0;
  if (stroke.heal) {
    const inner = sampleMean(source, width, height, centreX + offsetX, centreY + offsetY, radius);
    const outer = ringMean(source, width, height, centreX, centreY, radius);
    if (inner.count > 0 && outer.count > 0) {
      shiftR = outer.r - inner.r;
      shiftG = outer.g - inner.g;
      shiftB = outer.b - inner.b;
    }
  }

  for (let y = top; y <= bottom; y++) {
    const dy = y - centreY;
    for (let x = left; x <= right; x++) {
      const dx = x - centreX;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance > radius) continue;
      const sourceX = x + offsetX, sourceY = y + offsetY;
      if (sourceX < 0 || sourceY < 0 || sourceX >= width || sourceY >= height) continue;
      const strength = distance <= solid ? 1 : 1 - (distance - solid) / fade;
      if (strength <= 0) continue;
      const to = (y * width + x) * 4;
      const from = (sourceY * width + sourceX) * 4;
      data[to] += (source[from] + shiftR - data[to]) * strength;
      data[to + 1] += (source[from + 1] + shiftG - data[to + 1]) * strength;
      data[to + 2] += (source[from + 2] + shiftB - data[to + 2]) * strength;
      // Alpha is left alone: retouching a cutout must not eat into its matte.
    }
  }
}

function sampleMean(
  data: Uint8ClampedArray, width: number, height: number,
  centreX: number, centreY: number, radius: number,
): { r: number; g: number; b: number; count: number } {
  let r = 0, g = 0, b = 0, count = 0;
  const left = Math.max(0, Math.floor(centreX - radius)), right = Math.min(width - 1, Math.ceil(centreX + radius));
  const top = Math.max(0, Math.floor(centreY - radius)), bottom = Math.min(height - 1, Math.ceil(centreY + radius));
  for (let y = top; y <= bottom; y++) {
    for (let x = left; x <= right; x++) {
      if (Math.hypot(x - centreX, y - centreY) > radius) continue;
      const at = (y * width + x) * 4;
      r += data[at]; g += data[at + 1]; b += data[at + 2];
      count++;
    }
  }
  return count ? { r: r / count, g: g / count, b: b / count, count } : { r: 0, g: 0, b: 0, count: 0 };
}

/** The average of a band just outside the brush -- what the repair has to blend into. */
function ringMean(
  data: Uint8ClampedArray, width: number, height: number,
  centreX: number, centreY: number, radius: number,
): { r: number; g: number; b: number; count: number } {
  const outer = radius * 1.6;
  let r = 0, g = 0, b = 0, count = 0;
  const left = Math.max(0, Math.floor(centreX - outer)), right = Math.min(width - 1, Math.ceil(centreX + outer));
  const top = Math.max(0, Math.floor(centreY - outer)), bottom = Math.min(height - 1, Math.ceil(centreY + outer));
  for (let y = top; y <= bottom; y++) {
    for (let x = left; x <= right; x++) {
      const distance = Math.hypot(x - centreX, y - centreY);
      if (distance <= radius || distance > outer) continue;
      const at = (y * width + x) * 4;
      r += data[at]; g += data[at + 1]; b += data[at + 2];
      count++;
    }
  }
  return count ? { r: r / count, g: g / count, b: b / count, count } : { r: 0, g: 0, b: 0, count: 0 };
}
