/**
 * Liquify: pushing pixels around without inventing any.
 *
 * Every mode is one idea -- a displacement field, then one resample through it. Nothing here paints;
 * it decides, for each output pixel, where in the input to look. That is why liquify never produces
 * a colour the picture did not already contain, and why it composes cleanly with everything else in
 * the stack: it is a rearrangement, not a generation.
 *
 * The field is built in the working buffer's own pixels from strokes stored in normalised
 * coordinates, so the preview and the exported file describe the same warp at two resolutions --
 * the same reason masks and retouch strokes are stored the way they are.
 */

export type WarpMode = "push" | "bloat" | "pinch" | "twirl";

export type WarpStroke = {
  points: Array<[number, number]>;
  /** Brush radius as a fraction of the picture's long edge. */
  radius: number;
  /** 0..1. How far one pass moves things. */
  strength: number;
  mode: WarpMode;
};

/**
 * Smooth falloff from the middle of the brush to nothing at its edge.
 *
 * Squared rather than linear so the field has no crease where two dabs meet: a linear falloff is
 * continuous but its slope is not, and the eye finds that seam immediately on skin or sky.
 */
const falloff = (distanceSquared: number, radiusSquared: number): number => {
  if (distanceSquared >= radiusSquared) return 0;
  const t = 1 - distanceSquared / radiusSquared;
  return t * t;
};

/** The largest single fetch offset one dab may contribute, as a share of the brush radius. */
const MAX_PUSH = 0.9;

/**
 * Strokes are applied ONE AT A TIME, each over the result of the last.
 *
 * Not accumulated into a single field, which is the cheaper thing and the wrong one: a single field
 * has to be capped somewhere or a long drag folds the picture over itself, and once it is capped a
 * second pass over the same spot does nothing. Warping the warped result is what makes repeated
 * passes go further, which is how everyone expects this tool to behave, and it is the same shape as
 * the retouch brush, which snapshots per stroke for the same reason.
 *
 * The scratch buffer is allocated once and refilled, and only each stroke's own bounding box is
 * resampled, so the cost is the brush area rather than the frame for every stroke.
 */
export function applyWarp(
  data: Uint8ClampedArray, width: number, height: number, strokes: WarpStroke[],
): void {
  if (!strokes.length) return;
  let source: Uint8ClampedArray | undefined;
  for (const stroke of strokes) {
    const field = buildField(width, height, [stroke]);
    if (!field) continue;
    source ??= new Uint8ClampedArray(data.length);
    source.set(data);
    resample(data, source, width, height, field);
  }
}

/**
 * Where each output pixel should read from, as an offset in pixels.
 *
 * Returned separately from the resample so it can be tested on its own, and because a field of two
 * floats per pixel is the honest description of what liquify is.
 */
export type WarpField = {
  x: Float32Array; y: Float32Array;
  /** The rectangle any of the strokes actually reached. Everything outside it is exactly zero. */
  left: number; top: number; right: number; bottom: number;
};

export function buildField(
  width: number, height: number, strokes: WarpStroke[],
): WarpField | undefined {
  const count = width * height;
  if (count < 4) return undefined;
  const long = Math.max(width, height);
  const fieldX = new Float32Array(count);
  const fieldY = new Float32Array(count);
  let touched = false;
  const box = { left: width, top: height, right: -1, bottom: -1 };

  for (const stroke of strokes) {
    if (!stroke.points.length || stroke.strength <= 0) continue;
    const radius = Math.max(2, stroke.radius * long);
    // Dabs are spaced against the radius rather than at a fixed interval, so a big brush costs the
    // same as a small one per unit of stroke instead of quadratically more.
    const step = Math.max(1, radius * 0.25);
    let lastX = stroke.points[0][0] * width, lastY = stroke.points[0][1] * height;
    dab(fieldX, fieldY, width, height, lastX, lastY, 0, 0, radius, stroke, box);
    touched = true;

    for (let index = 1; index < stroke.points.length; index++) {
      const toX = stroke.points[index][0] * width;
      const toY = stroke.points[index][1] * height;
      const spanX = toX - lastX, spanY = toY - lastY;
      const distance = Math.hypot(spanX, spanY);
      const steps = Math.max(1, Math.ceil(distance / step));
      for (let at = 1; at <= steps; at++) {
        const t = at / steps;
        dab(fieldX, fieldY, width, height,
          lastX + spanX * t, lastY + spanY * t, spanX / steps, spanY / steps, radius, stroke, box);
      }
      lastX = toX; lastY = toY;
    }
  }
  if (!touched || box.right < box.left || box.bottom < box.top) return undefined;
  return { x: fieldX, y: fieldY, ...box };
}

function dab(
  fieldX: Float32Array, fieldY: Float32Array, width: number, height: number,
  centreX: number, centreY: number, moveX: number, moveY: number,
  radius: number, stroke: WarpStroke,
  box: { left: number; top: number; right: number; bottom: number },
): void {
  const left = Math.max(0, Math.floor(centreX - radius));
  const right = Math.min(width - 1, Math.ceil(centreX + radius));
  const top = Math.max(0, Math.floor(centreY - radius));
  const bottom = Math.min(height - 1, Math.ceil(centreY + radius));
  if (right < left || bottom < top) return;
  if (left < box.left) box.left = left;
  if (top < box.top) box.top = top;
  if (right > box.right) box.right = right;
  if (bottom > box.bottom) box.bottom = bottom;
  const radiusSquared = radius * radius;
  const strength = Math.min(1, Math.max(0, stroke.strength));
  // A drag longer than the brush would otherwise tear the picture in one dab, so the push a single
  // dab can contribute is capped at a share of the radius and repeated passes do the rest.
  const cap = radius * MAX_PUSH;
  const travel = Math.hypot(moveX, moveY);
  const scale = travel > cap ? cap / travel : 1;

  for (let y = top; y <= bottom; y++) {
    const dy = y - centreY;
    for (let x = left; x <= right; x++) {
      const dx = x - centreX;
      const weight = falloff(dx * dx + dy * dy, radiusSquared);
      if (weight <= 0) continue;
      const index = y * width + x;
      const amount = weight * strength;
      let addX = 0, addY = 0;
      if (stroke.mode === "push") {
        // Backward mapping: to move content along the drag, each pixel reads from behind it.
        addX = -moveX * scale * amount;
        addY = -moveY * scale * amount;
      } else if (stroke.mode === "bloat" || stroke.mode === "pinch") {
        const sign = stroke.mode === "bloat" ? -1 : 1;
        addX = dx * amount * 0.5 * sign;
        addY = dy * amount * 0.5 * sign;
      } else {
        // Twirl: read from a point rotated back around the brush centre.
        const angle = -amount * 1.2;
        const cos = Math.cos(angle), sin = Math.sin(angle);
        addX = dx * cos - dy * sin - dx;
        addY = dx * sin + dy * cos - dy;
      }
      // Held to under one brush radius across the whole stroke. Past that the fetch for one pixel
      // crosses the fetch for its neighbour, which is what a tear is: the picture folded over
      // itself. Going further is what a second stroke is for.
      let nextX = fieldX[index] + addX;
      let nextY = fieldY[index] + addY;
      const reach = Math.hypot(nextX, nextY);
      if (reach > cap) {
        nextX = nextX / reach * cap;
        nextY = nextY / reach * cap;
      }
      fieldX[index] = nextX;
      fieldY[index] = nextY;
    }
  }
}

/** Read the picture through the field, bilinearly, from an untouched copy of itself. */
function resample(
  data: Uint8ClampedArray, source: Uint8ClampedArray,
  width: number, height: number, field: WarpField,
): void {
  for (let y = field.top; y <= field.bottom; y++) {
    for (let x = field.left; x <= field.right; x++) {
      const index = y * width + x;
      const offsetX = field.x[index];
      const offsetY = field.y[index];
      if (offsetX === 0 && offsetY === 0) continue;
      sampleInto(data, index * 4, source, width, height, x + offsetX, y + offsetY);
    }
  }
}

function sampleInto(
  target: Uint8ClampedArray, at: number, source: Uint8ClampedArray,
  width: number, height: number, x: number, y: number,
): void {
  const clampedX = x < 0 ? 0 : x > width - 1 ? width - 1 : x;
  const clampedY = y < 0 ? 0 : y > height - 1 ? height - 1 : y;
  const x0 = Math.floor(clampedX), y0 = Math.floor(clampedY);
  const x1 = Math.min(width - 1, x0 + 1), y1 = Math.min(height - 1, y0 + 1);
  const fx = clampedX - x0, fy = clampedY - y0;
  const topLeft = (y0 * width + x0) * 4, topRight = (y0 * width + x1) * 4;
  const bottomLeft = (y1 * width + x0) * 4, bottomRight = (y1 * width + x1) * 4;
  for (let channel = 0; channel < 4; channel++) {
    const top = source[topLeft + channel] + (source[topRight + channel] - source[topLeft + channel]) * fx;
    const bottom = source[bottomLeft + channel] + (source[bottomRight + channel] - source[bottomLeft + channel]) * fx;
    target[at + channel] = top + (bottom - top) * fy;
  }
}
