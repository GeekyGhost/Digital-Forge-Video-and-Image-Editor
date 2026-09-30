/**
 * Where a pixel ends up.
 *
 * Quarter turns, flips, straightening, free scale and the crop rectangle are all one affine
 * transform, composed once and inverted for anything that has to go the other way. That matters
 * more than it sounds: the renderer needs source-to-screen and the brush needs screen-to-source,
 * and when those were two hand-written switch statements they were one edit away from disagreeing.
 * A disagreement is invisible until the picture is rotated, at which point every stroke lands
 * somewhere else. Here there is one definition and an inverse that cannot drift from it.
 */

export type CropRect = { x: number; y: number; width: number; height: number };

export type PhotoGeometry = {
  rotate: 0 | 90 | 180 | 270;
  flipHorizontal: boolean;
  flipVertical: boolean;
  /** Fine rotation in degrees, on top of the quarter turns. */
  straighten: number;
  /** Free scale, 1 being none. */
  scale: number;
  /** Normalised to the straightened frame. Undefined is the whole picture. */
  crop?: CropRect;
};

export const NEUTRAL_GEOMETRY: PhotoGeometry = {
  rotate: 0, flipHorizontal: false, flipVertical: false, straighten: 0, scale: 1,
};

export const isNeutralGeometry = (geometry: PhotoGeometry): boolean =>
  geometry.rotate === 0 && !geometry.flipHorizontal && !geometry.flipVertical
  && Math.abs(geometry.straighten) < 1e-4 && Math.abs(geometry.scale - 1) < 1e-4
  && !geometry.crop;

/** `x' = a x + c y + e`, `y' = b x + d y + f` -- the same order `setTransform` takes. */
export type Matrix = { a: number; b: number; c: number; d: number; e: number; f: number };

const IDENTITY: Matrix = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 };

/** `outer` applied after `inner`. */
export function compose(outer: Matrix, inner: Matrix): Matrix {
  return {
    a: outer.a * inner.a + outer.c * inner.b,
    b: outer.b * inner.a + outer.d * inner.b,
    c: outer.a * inner.c + outer.c * inner.d,
    d: outer.b * inner.c + outer.d * inner.d,
    e: outer.a * inner.e + outer.c * inner.f + outer.e,
    f: outer.b * inner.e + outer.d * inner.f + outer.f,
  };
}

export function invert(matrix: Matrix): Matrix {
  const determinant = matrix.a * matrix.d - matrix.b * matrix.c;
  if (Math.abs(determinant) < 1e-12) return IDENTITY;
  return {
    a: matrix.d / determinant,
    b: -matrix.b / determinant,
    c: -matrix.c / determinant,
    d: matrix.a / determinant,
    e: (matrix.c * matrix.f - matrix.d * matrix.e) / determinant,
    f: (matrix.b * matrix.e - matrix.a * matrix.f) / determinant,
  };
}

export const applyMatrix = (matrix: Matrix, x: number, y: number): [number, number] =>
  [matrix.a * x + matrix.c * y + matrix.e, matrix.b * x + matrix.d * y + matrix.f];

export const translate = (x: number, y: number): Matrix => ({ a: 1, b: 0, c: 0, d: 1, e: x, f: y });
export const scaleBy = (x: number, y: number): Matrix => ({ a: x, b: 0, c: 0, d: y, e: 0, f: 0 });
export const rotateBy = (degrees: number): Matrix => {
  const radians = degrees * Math.PI / 180;
  const cos = Math.cos(radians), sin = Math.sin(radians);
  return { a: cos, b: sin, c: -sin, d: cos, e: 0, f: 0 };
};

/**
 * The axis-aligned box a rotated, scaled source fills.
 *
 * Straightening a picture makes it bigger, not smaller -- the corners have to go somewhere. Cropping
 * back into it is the user's decision, so this reports the honest size and lets the crop rectangle
 * take it back.
 */
export function framedSize(sourceWidth: number, sourceHeight: number, geometry: PhotoGeometry): {
  width: number; height: number;
} {
  const radians = (geometry.rotate + geometry.straighten) * Math.PI / 180;
  const cos = Math.abs(Math.cos(radians)), sin = Math.abs(Math.sin(radians));
  const scale = geometry.scale || 1;
  return {
    width: (sourceWidth * cos + sourceHeight * sin) * scale,
    height: (sourceWidth * sin + sourceHeight * cos) * scale,
  };
}

/** The finished picture's pixel size, after the crop has taken its bite. */
export function outputSize(sourceWidth: number, sourceHeight: number, geometry: PhotoGeometry): {
  width: number; height: number;
} {
  const framed = framedSize(sourceWidth, sourceHeight, geometry);
  const crop = geometry.crop;
  return {
    width: Math.max(1, Math.round(framed.width * (crop?.width ?? 1))),
    height: Math.max(1, Math.round(framed.height * (crop?.height ?? 1))),
  };
}

/**
 * Source pixels to output pixels.
 *
 * Read bottom-up, because composition applies the last one first: centre the source, flip it, turn
 * it, straighten it, scale it, put it in the middle of its new frame, then slide the crop's corner
 * to the origin.
 */
export function forwardMatrix(
  sourceWidth: number, sourceHeight: number, geometry: PhotoGeometry,
): Matrix {
  const framed = framedSize(sourceWidth, sourceHeight, geometry);
  const crop = geometry.crop;
  let matrix = translate(-sourceWidth / 2, -sourceHeight / 2);
  matrix = compose(scaleBy(geometry.flipHorizontal ? -1 : 1, geometry.flipVertical ? -1 : 1), matrix);
  matrix = compose(rotateBy(geometry.rotate + geometry.straighten), matrix);
  matrix = compose(scaleBy(geometry.scale || 1, geometry.scale || 1), matrix);
  matrix = compose(translate(framed.width / 2, framed.height / 2), matrix);
  if (crop) {
    matrix = compose(translate(-crop.x * framed.width, -crop.y * framed.height), matrix);
  }
  return matrix;
}

/**
 * A point on the shown picture, in 0..1, back to a point on the source, in 0..1.
 *
 * This is what turns a pointer position into a brush stroke, so it has to be the exact inverse of
 * what was drawn rather than an approximation of it.
 */
export function displayToSource(
  px: number, py: number, sourceWidth: number, sourceHeight: number, geometry: PhotoGeometry,
): [number, number] {
  const output = outputSize(sourceWidth, sourceHeight, geometry);
  const inverse = invert(forwardMatrix(sourceWidth, sourceHeight, geometry));
  const [x, y] = applyMatrix(inverse, px * output.width, py * output.height);
  return [x / sourceWidth, y / sourceHeight];
}

/** The other direction, for drawing handles where a stored point actually appears. */
export function sourceToDisplay(
  u: number, v: number, sourceWidth: number, sourceHeight: number, geometry: PhotoGeometry,
): [number, number] {
  const output = outputSize(sourceWidth, sourceHeight, geometry);
  const [x, y] = applyMatrix(
    forwardMatrix(sourceWidth, sourceHeight, geometry), u * sourceWidth, v * sourceHeight);
  return [x / output.width, y / output.height];
}
