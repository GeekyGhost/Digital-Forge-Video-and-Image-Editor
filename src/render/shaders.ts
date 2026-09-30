/**
 * Fragment shaders for the Phase 1 compositor, each a port of one CPU function in `src/domain/`.
 *
 * The rule for every one of them: do the same arithmetic in the same order as the TypeScript it
 * replaces, and quantise at the same points. The CPU code writes into a `Uint8ClampedArray`, which
 * clamps and rounds half-to-even; `store()` below does exactly that before the value reaches the
 * RGBA8 target, so the target holds the byte the CPU would have held. Differences left are float32
 * (GPU) against float64 (JS) rounding, which only matter when a value lands within about 1e-6 of a
 * rounding boundary.
 *
 * The source function each shader mirrors is named above it. Change them together.
 */

const HEADER = `#version 300 es
precision highp float;
precision highp int;
precision highp sampler2D;
out vec4 outColor;
ivec2 here() { return ivec2(gl_FragCoord.xy); }
/* An RGBA8 texel as its bytes, 0..255. Exact: an 8-bit texel is k/255 and this undoes it. */
vec4 bytesAt(sampler2D s, ivec2 p) { return floor(texelFetch(s, p, 0) * 255.0 + 0.5); }
/* What assigning to a Uint8ClampedArray does: NaN to 0, clamp, round half to even. */
float clampByte(float v) { return isnan(v) ? 0.0 : clamp(roundEven(v), 0.0, 255.0); }
vec4 store(vec4 b) { return vec4(clampByte(b.r), clampByte(b.g), clampByte(b.b), clampByte(b.a)) / 255.0; }
`;

/** Copy, or fill with a constant (the document background). */
export const FILL_FS = `${HEADER}
uniform vec4 value;
void main() { outColor = value / 255.0; }`;

/**
 * Layer placement: `placeLayer` in photoCompose.ts, which is canvas `drawImage` through
 * `layerMatrix` with bilinear smoothing into a premultiplied 8-bit canvas, read back unpremultiplied
 * by `getImageData`.
 *
 * `source` is uploaded PREMULTIPLIED (as the canvas holds it; WebGL's premultiplying upload gives the
 * canvas's own bytes). Sampling is exact bilinear on texel centres, clamped to the crop rectangle,
 * and the rectangle's edges are antialiased by coverage in device pixels. The result is quantised to
 * 8-bit premultiplied and unpremultiplied through a table measured from the browser's own
 * `getImageData`, so an untransformed or integer-translated layer comes out byte-identical to the
 * canvas path. Fractional scales and offsets differ where Chrome's software canvas uses Skia's
 * legacy 4-bit bilinear weights; the shader is the exact one (see parity.ts, placementAccuracy).
 */
export const PLACE_FS = `${HEADER}
uniform sampler2D source;
uniform vec2 inverseRow0;   // canvas pixel -> rect-local pixel (a, c, e) of the inverse matrix
uniform vec2 inverseRow1;   // (b, d, f)
uniform vec2 inverseOffset;
uniform vec4 rect;          // left, top, width, height of the cropped source, in source pixels
uniform float deviceScale;  // canvas pixels per source pixel (layer.scale * canvasScale)
uniform float antialias;    // 1 to antialias the rectangle's edges, 0 for a hard pixel-centre test
uniform sampler2D unpremultiply; // 256x256 R8: [premultiplied value, alpha] -> the browser's getImageData byte

vec4 texel(ivec2 p) {
  ivec2 lo = ivec2(floor(rect.xy));
  ivec2 hi = ivec2(ceil(rect.xy + rect.zw)) - 1;
  return texelFetch(source, clamp(p, lo, hi), 0);
}

void main() {
  vec2 p = gl_FragCoord.xy;
  vec2 local = vec2(dot(inverseRow0, p), dot(inverseRow1, p)) + inverseOffset;
  vec2 inside = min(local, rect.zw - local);
  float coverage;
  if (antialias > 0.5) {
    vec2 c = clamp(inside * deviceScale + 0.5, 0.0, 1.0);
    coverage = c.x * c.y;
  } else {
    coverage = (inside.x >= 0.0 && inside.y >= 0.0) ? 1.0 : 0.0;
  }
  if (coverage <= 0.0) { outColor = vec4(0.0); return; }
  vec2 s = rect.xy + local - 0.5;
  vec2 base = floor(s);
  vec2 f = s - base;
  ivec2 i = ivec2(base);
  vec4 top = mix(texel(i), texel(i + ivec2(1, 0)), f.x);
  vec4 bottom = mix(texel(i + ivec2(0, 1)), texel(i + ivec2(1, 1)), f.x);
  vec4 premultiplied = floor(mix(top, bottom, f.y) * coverage * 255.0 + 0.5);
  float a = premultiplied.a;
  if (a <= 0.0) { outColor = vec4(0.0); return; }
  vec3 p8 = min(premultiplied.rgb, vec3(a));
  float r = texelFetch(unpremultiply, ivec2(int(p8.r), int(a)), 0).r;
  float g = texelFetch(unpremultiply, ivec2(int(p8.g), int(a)), 0).r;
  float b = texelFetch(unpremultiply, ivec2(int(p8.b), int(a)), 0).r;
  outColor = vec4(r, g, b, a / 255.0);
}`;

/** `compositeOver` and `blendChannel` in photoBlend.ts. Mode ids follow BLEND_MODES order. */
export const COMPOSITE_FS = `${HEADER}
uniform sampler2D base;
uniform sampler2D layer;
uniform int mode;
uniform float strength;

float blendChannel(int m, float backdrop, float source) {
  if (m == 1) return backdrop * source;
  if (m == 2) return backdrop + source - backdrop * source;
  if (m == 3) { // overlay = hard light with the operands swapped
    float s = backdrop, b = source;
    return s <= 0.5 ? b * (2.0 * s) : b + (2.0 * s - 1.0) - b * (2.0 * s - 1.0);
  }
  if (m == 4) return min(backdrop, source);
  if (m == 5) return max(backdrop, source);
  if (m == 6) {
    if (backdrop <= 0.0) return 0.0;
    if (source >= 1.0) return 1.0;
    return min(1.0, backdrop / (1.0 - source));
  }
  if (m == 7) {
    if (backdrop >= 1.0) return 1.0;
    if (source <= 0.0) return 0.0;
    return 1.0 - min(1.0, (1.0 - backdrop) / source);
  }
  if (m == 8) return source <= 0.5 ? backdrop * (2.0 * source) : backdrop + (2.0 * source - 1.0) - backdrop * (2.0 * source - 1.0);
  if (m == 9) {
    float d = backdrop <= 0.25 ? ((16.0 * backdrop - 12.0) * backdrop + 4.0) * backdrop : sqrt(backdrop);
    return source <= 0.5
      ? backdrop - (1.0 - 2.0 * source) * backdrop * (1.0 - backdrop)
      : backdrop + (2.0 * source - 1.0) * (d - backdrop);
  }
  if (m == 10) return abs(backdrop - source);
  if (m == 11) return backdrop + source - 2.0 * backdrop * source;
  return source;
}

void main() {
  ivec2 p = here();
  vec4 b = bytesAt(base, p);
  vec4 l = bytesAt(layer, p);
  float alpha = (l.a / 255.0) * strength;
  if (alpha <= 0.0) { outColor = b / 255.0; return; }
  float backdropAlpha = b.a / 255.0;
  float outAlpha = alpha + backdropAlpha * (1.0 - alpha);
  if (outAlpha <= 0.0) { outColor = vec4(b.rgb, 0.0) / 255.0; return; }
  vec4 result;
  for (int c = 0; c < 3; c++) {
    float backdrop = b[c] / 255.0;
    float source = l[c] / 255.0;
    float blended = mode == 0 ? source : source + backdropAlpha * (blendChannel(mode, backdrop, source) - source);
    float value = (alpha * blended + backdropAlpha * backdrop * (1.0 - alpha)) / outAlpha;
    result[c] = value * 255.0;
  }
  result.a = outAlpha * 255.0;
  outColor = store(result);
}`;

/** `blend` in photoRender.ts: a node's result mixed back through its mask coverage. */
export const MASK_MIX_FS = `${HEADER}
uniform sampler2D current;
uniform sampler2D layer;
uniform sampler2D coverage;
void main() {
  ivec2 p = here();
  vec4 c = bytesAt(current, p);
  float amount = floor(texelFetch(coverage, p, 0).r * 255.0 + 0.5);
  if (amount == 0.0) { outColor = c / 255.0; return; }
  vec4 l = bytesAt(layer, p);
  if (amount == 255.0) { outColor = l / 255.0; return; }
  float mixAmount = amount / 255.0;
  outColor = store(c + (l - c) * mixAmount);
}`;

/**
 * The pointwise part of `applyAdjustments` in photoEdit.ts: linearise, white balance and exposure,
 * highlights/shadows, contrast, saturation/vibrance, HSL bands, vignette, then the folded output
 * table (`buildOutputLuts`), which is uploaded rather than rebuilt so preview and save share it.
 */
export const GRADE_FS = `${HEADER}
uniform sampler2D source;
uniform sampler2D toLinear;   // 256x1 R32F: srgbToLinear
uniform sampler2D outputLut;  // 4096x1 RGBA8: lutR, lutG, lutB
uniform vec3 gain;
uniform float highlights;
uniform float shadows;
uniform float contrast;
uniform float saturation;
uniform float vibrance;
uniform float bandsActive;
uniform float bandHue[8];
uniform float bandSaturation[8];
uniform float bandLuminance[8];
uniform float bandCentre[8];
uniform float bandWidth;
uniform float vignette;
uniform vec2 centre;
uniform float maxRadius;
uniform float vignetteStart;
uniform float vignetteSpread;
uniform float scale;          // (ENCODE_STEPS - 1) / ENCODE_CEILING
uniform int last;             // ENCODE_STEPS - 1

const float LR = 0.2126, LG = 0.7152, LB = 0.0722;

float lin(float byteValue) { return texelFetch(toLinear, ivec2(int(byteValue), 0), 0).r; }

int encodeIndex(float v) {
  if (v <= 0.0) return 0;
  float scaled = v * scale;
  if (scaled >= float(last)) return last;
  return int(scaled);
}

void main() {
  ivec2 p = here();
  vec4 px = bytesAt(source, p);
  float r = lin(px.r) * gain.r;
  float g = lin(px.g) * gain.g;
  float b = lin(px.b) * gain.b;

  if (highlights != 0.0 || shadows != 0.0) {
    float luma = r * LR + g * LG + b * LB;
    float top = min(1.0, luma * 1.9);
    float factor = 1.0 + highlights * 0.55 * top + shadows * 0.55 * (1.0 - top) * (1.0 - top);
    r *= factor; g *= factor; b *= factor;
  }
  if (contrast != 1.0) {
    r = (r - 0.16) * contrast + 0.16;
    g = (g - 0.16) * contrast + 0.16;
    b = (b - 0.16) * contrast + 0.16;
  }
  if (saturation != 1.0 || vibrance != 0.0) {
    float luma = r * LR + g * LG + b * LB;
    float amount = saturation;
    if (vibrance != 0.0) {
      float peak = max(r, max(g, b));
      float dullness = peak <= 0.0 ? 1.0 : 1.0 - (peak - min(r, min(g, b))) / peak;
      amount += vibrance * dullness;
    }
    r = luma + (r - luma) * amount;
    g = luma + (g - luma) * amount;
    b = luma + (b - luma) * amount;
  }
  if (bandsActive > 0.5) {
    float mx = max(r, max(g, b)), mn = min(r, min(g, b));
    float delta = mx - mn;
    if (delta > 1e-6) {
      float hue;
      if (mx == r) { float q = (g - b) / delta; hue = q - 6.0 * trunc(q / 6.0); }
      else if (mx == g) hue = (b - r) / delta + 2.0;
      else hue = (r - g) / delta + 4.0;
      hue *= 60.0;
      if (hue < 0.0) hue += 360.0;
      float chroma = min(1.0, delta / max(1e-6, mx));
      float weights[8];
      float total = 0.0;
      for (int band = 0; band < 8; band++) {
        float distance = abs(hue - bandCentre[band]);
        if (distance > 180.0) distance = 360.0 - distance;
        float w = distance >= bandWidth ? 0.0 : (1.0 - distance / bandWidth) * chroma;
        weights[band] = w;
        total += w;
      }
      if (total > 0.0) {
        float hueShift = 0.0, saturate = 0.0, lighten = 0.0;
        for (int band = 0; band < 8; band++) {
          float weight = weights[band] / total;
          if (weight == 0.0) continue;
          hueShift += weight * bandHue[band];
          saturate += weight * bandSaturation[band];
          lighten += weight * bandLuminance[band];
        }
        if (hueShift != 0.0) {
          float angle = hueShift * 0.3 * 3.141592653589793 / 180.0;
          float c = cos(angle), s = sin(angle);
          float nr = r * (0.213 + c * 0.787 - s * 0.213) + g * (0.715 - c * 0.715 - s * 0.715) + b * (0.072 - c * 0.072 + s * 0.928);
          float ng = r * (0.213 - c * 0.213 + s * 0.143) + g * (0.715 + c * 0.285 + s * 0.140) + b * (0.072 - c * 0.072 - s * 0.283);
          float nb = r * (0.213 - c * 0.213 - s * 0.787) + g * (0.715 - c * 0.715 + s * 0.715) + b * (0.072 + c * 0.928 + s * 0.072);
          r = nr; g = ng; b = nb;
        }
        if (saturate != 0.0) {
          float luma = r * LR + g * LG + b * LB;
          float amount = 1.0 + saturate / 100.0;
          r = luma + (r - luma) * amount;
          g = luma + (g - luma) * amount;
          b = luma + (b - luma) * amount;
        }
        if (lighten != 0.0) {
          float amount = 1.0 + lighten / 100.0;
          r *= amount; g *= amount; b *= amount;
        }
      }
    }
  }
  if (vignette != 0.0) {
    vec2 xy = vec2(p);
    float distance = length(xy - centre) / maxRadius;
    float t = clamp((distance - vignetteStart) / vignetteSpread, 0.0, 1.0);
    float falloff = t * t * (3.0 - 2.0 * t);
    float factor = 1.0 + vignette * falloff;
    r *= factor; g *= factor; b *= factor;
  }
  float outR = texelFetch(outputLut, ivec2(encodeIndex(r), 0), 0).r;
  float outG = texelFetch(outputLut, ivec2(encodeIndex(g), 0), 0).g;
  float outB = texelFetch(outputLut, ivec2(encodeIndex(b), 0), 0).b;
  outColor = vec4(outR, outG, outB, px.a / 255.0);
}`;

/**
 * One axis of `boxPass` (photoEdit.ts): a box of radius `radius`, edge texels repeated, on RGB bytes,
 * alpha passed through. `blurBuffer` is three horizontal+vertical rounds of this, quantising after
 * every pass exactly as the CPU's 8-bit scratch buffer does.
 */
export const BOX_FS = `${HEADER}
uniform sampler2D source;
uniform ivec2 direction;
uniform int radius;
void main() {
  ivec2 p = here();
  ivec2 size = textureSize(source, 0);
  vec3 total = vec3(0.0);
  for (int k = -radius; k <= radius; k++) {
    ivec2 q = clamp(p + direction * k, ivec2(0), size - 1);
    total += bytesAt(source, q).rgb;
  }
  vec3 value = total / float(radius * 2 + 1);
  outColor = store(vec4(value, bytesAt(source, p).a));
}`;

/** `sharpenBuffer`: the picture plus its difference from a radius-1 blur, times 1.4 x amount. */
export const SHARPEN_FS = `${HEADER}
uniform sampler2D source;
uniform sampler2D blurred;
uniform float strength;
void main() {
  ivec2 p = here();
  vec4 o = bytesAt(source, p);
  vec3 b = bytesAt(blurred, p).rgb;
  outColor = store(vec4(o.rgb + (o.rgb - b) * strength, o.a));
}`;

/** `addGrain`: the noise field comes from the CPU (float64 sin), the add happens here. */
export const GRAIN_FS = `${HEADER}
uniform sampler2D source;
uniform sampler2D noise;
uniform float strength;
void main() {
  ivec2 p = here();
  vec4 o = bytesAt(source, p);
  float shift = (texelFetch(noise, p, 0).r - 0.5) * strength;
  outColor = store(vec4(o.rgb + shift, o.a));
}`;

/** `denoiseBuffer`, step 1: bytes to Rec. 709 luma plus two colour differences, as float planes. */
export const PLANES_FS = `${HEADER}
uniform sampler2D source;
void main() {
  vec4 o = bytesAt(source, here());
  float y = o.r * 0.2126 + o.g * 0.7152 + o.b * 0.0722;
  outColor = vec4(y, o.b - y, o.r - y, 0.0);
}`;

/** `planePass`: one axis of the float box blur, all four channels. */
export const PLANE_BOX_FS = `${HEADER}
uniform sampler2D source;
uniform ivec2 direction;
uniform int radius;
void main() {
  ivec2 p = here();
  ivec2 size = textureSize(source, 0);
  vec4 total = vec4(0.0);
  for (int k = -radius; k <= radius; k++) {
    total += texelFetch(source, clamp(p + direction * k, ivec2(0), size - 1), 0);
  }
  outColor = total / float(radius * 2 + 1);
}`;

/** `denoiseBuffer`, last step: Wiener shrinkage of each plane toward its blur, then back to RGB. */
export const DENOISE_FS = `${HEADER}
uniform sampler2D source;
uniform sampler2D planes;
uniform sampler2D lumaBase;
uniform sampler2D chromaBase;
uniform float luma;
uniform float chroma;
uniform float keep;
float shrink(float value, float smoothValue, float sigma, float keepShare) {
  float detail = value - smoothValue;
  float weight = detail * detail / (detail * detail + sigma * sigma);
  return smoothValue + detail * (weight + (1.0 - weight) * keepShare);
}
void main() {
  ivec2 p = here();
  vec4 plane = texelFetch(planes, p, 0);
  float y = plane.x, cb = plane.y, cr = plane.z;
  if (luma > 0.0) y = shrink(y, texelFetch(lumaBase, p, 0).x, 2.0 + luma * 16.0, keep);
  if (chroma > 0.0) {
    vec4 base = texelFetch(chromaBase, p, 0);
    cb = shrink(cb, base.y, 3.0 + chroma * 30.0, 0.0);
    cr = shrink(cr, base.z, 3.0 + chroma * 30.0, 0.0);
  }
  float red = y + cr;
  float blue = y + cb;
  float green = (y - red * 0.2126 - blue * 0.0722) / 0.7152;
  outColor = store(vec4(red, green, blue, bytesAt(source, p).a));
}`;

/** `applyGradientMap` in photoGradient.ts, with `rampParameter` folded in. */
export const GRADIENT_MAP_FS = `${HEADER}
uniform sampler2D source;
uniform sampler2D lut;   // 256x1 RGBA8 from buildGradientLut(gradient, 256)
uniform float strength;
uniform float repeatCount;
uniform float mirror;
uniform float ease;

float pingPong(float v) { float w = mod(abs(v), 2.0); return w > 1.0 ? 2.0 - w : w; }
float ramp(float position) {
  float r = max(1.0, repeatCount);
  float t = position * r;
  if (r > 1.0) t = mirror > 0.5 ? pingPong(t) : t - floor(t);
  else t = mirror > 0.5 ? pingPong(t) : clamp(t, 0.0, 1.0);
  if (ease != 0.0 && abs(ease - 1.0) > 1e-4) t = pow(clamp(t, 0.0, 1.0), max(0.05, ease));
  return clamp(t, 0.0, 1.0);
}
void main() {
  ivec2 p = here();
  vec4 o = bytesAt(source, p);
  float luma = o.r * 0.2126 + o.g * 0.7152 + o.b * 0.0722;
  float t = ramp(clamp(luma / 255.0, 0.0, 1.0));
  int index = int(clamp(floor(t * 255.0 + 0.5), 0.0, 255.0));
  vec4 entry = floor(texelFetch(lut, ivec2(index, 0), 0) * 255.0 + 0.5);
  float weight = strength * (entry.a / 255.0);
  outColor = store(vec4(o.rgb + (entry.rgb - o.rgb) * weight, o.a));
}`;

/** `renderGradient` with `gradientPosition` and `rampParameter`: a gradient layer's own pixels. */
export const GRADIENT_FILL_FS = `${HEADER}
uniform sampler2D lut;    // 512x1 RGBA8 from buildGradientLut(gradient)
uniform sampler2D bayer;  // 8x8 R32F, the exported BAYER table
uniform int kind;         // 0 linear, 1 reflected, 2 radial, 3 angular, 4 diamond
uniform float angle;      // radians
uniform vec2 centre;      // normalised
uniform float gradScale;
uniform float offset;
uniform float repeatCount;
uniform float mirror;
uniform float ease;
uniform float dither;
uniform vec2 size;
uniform int last;

float pingPong(float v) { float w = mod(abs(v), 2.0); return w > 1.0 ? 2.0 - w : w; }
float ramp(float position) {
  float r = max(1.0, repeatCount);
  float t = position * r;
  if (r > 1.0) t = mirror > 0.5 ? pingPong(t) : t - floor(t);
  else t = mirror > 0.5 ? pingPong(t) : clamp(t, 0.0, 1.0);
  if (ease != 0.0 && abs(ease - 1.0) > 1e-4) t = pow(clamp(t, 0.0, 1.0), max(0.05, ease));
  return clamp(t, 0.0, 1.0);
}
float position(vec2 xy) {
  float c = cos(angle), s = sin(angle);
  vec2 d = xy - centre * size;
  float u = d.x * c + d.y * s;
  float v = -d.x * s + d.y * c;
  float sc = max(0.01, gradScale);
  if (kind == 3) return (atan(v, u) / (3.141592653589793 * 2.0) + 0.5) / sc + offset;
  float along = abs(size.x / 2.0 * c) + abs(size.y / 2.0 * s);
  float across = abs(size.x / 2.0 * s) + abs(size.y / 2.0 * c);
  if (kind == 0) return u / (2.0 * max(1e-6, along) * sc) + 0.5 + offset;
  if (kind == 1) return abs(u) / (max(1e-6, along) * sc) + offset;
  if (kind == 4) return (abs(u) / max(1e-6, along) + abs(v) / max(1e-6, across)) / (2.0 * sc) + offset;
  float radius = length(size / 2.0);
  return length(vec2(u, v)) / (max(1e-6, radius) * sc) + offset;
}
void main() {
  ivec2 p = here();
  float t = ramp(position(vec2(p) + 0.5));
  int index = int(clamp(floor(t * float(last) + 0.5), 0.0, float(last)));
  vec4 entry = floor(texelFetch(lut, ivec2(index, 0), 0) * 255.0 + 0.5);
  if (dither > 0.0) {
    float wobble = (texelFetch(bayer, ivec2(p.x & 7, p.y & 7), 0).r - 0.5) * dither;
    outColor = store(vec4(entry.rgb + wobble, entry.a));
  } else {
    outColor = entry / 255.0;
  }
}`;

/* ================================================================ VFX Phase 2: masks and keys */

/*
 * Mask coverage lives in ordinary RGBA8 targets with the byte in r, g and b and alpha 255, so the
 * pool, BOX_FS (the feather) and MASK_MIX_FS (which reads .r) all work on it unchanged.
 */

/** An R8 upload (a placed matte file) copied into a coverage target. */
export const COVERAGE_COPY_FS = `${HEADER}
uniform sampler2D source;
void main() {
  float v = floor(texelFetch(source, here(), 0).r * 255.0 + 0.5);
  outColor = vec4(v, v, v, 255.0) / 255.0;
}`;

/**
 * `shapeCoverage` in photoMask.ts, drawn with blend equation MAX over the target, which is how
 * `rasterizeMask` takes the larger of the SAM 2 matte and the shape. Pixel coordinates are the
 * integer indices the CPU loops over (gl_FragCoord - 0.5). The colour range samples the scope's own
 * ungraded texture: no readback.
 */
export const MASK_SHAPE_FS = `${HEADER}
uniform int kind;          // 0 linear, 1 radial, 2 colour
uniform vec2 a;            // linear: start; radial: centre
uniform vec2 delta;        // linear: end - start
uniform float lengthSquared;
uniform int everything;    // linear with no length: covers everything
uniform vec2 radii;        // radial
uniform float solid;       // radial
uniform sampler2D scope;   // colour
uniform vec3 colour;
uniform float tolerance;
uniform float softness;
void main() {
  vec2 p = floor(gl_FragCoord.xy);
  float v = 0.0;
  if (kind == 0) {
    if (everything == 1) v = 255.0;
    else {
      float t = ((p.x - a.x) * delta.x + (p.y - a.y) * delta.y) / lengthSquared;
      float c = clamp(t, 0.0, 1.0);
      v = floor(c * c * (3.0 - 2.0 * c) * 255.0 + 0.5);
    }
  } else if (kind == 1) {
    vec2 n = (p - a) / radii;
    float distance = sqrt(n.x * n.x + n.y * n.y);
    if (distance < 1.0) {
      float t = distance <= solid ? 1.0 : 1.0 - (distance - solid) / max(1e-6, 1.0 - solid);
      v = floor(t * t * (3.0 - 2.0 * t) * 255.0 + 0.5);
    }
  } else {
    vec3 d = bytesAt(scope, here()).rgb - colour;
    float distance = sqrt(d.r * d.r + d.g * d.g + d.b * d.b);
    if (distance < tolerance) {
      float t = 1.0 - distance / tolerance;
      v = floor(pow(t, 1.0 / softness) * 255.0 + 0.5);
    }
  }
  outColor = vec4(v, v, v, 255.0) / 255.0;
}`;

/**
 * One brush dab per instance: `stampDab` in photoMask.ts. The quad covers the dab's integer pixel
 * bounds (`dabBounds`, computed on the CPU); the centre travels as an offset from the bounds' origin
 * so float32 keeps it to a millionth of a pixel. Drawn with blend equation MAX (paint) or MIN (erase).
 */
export const DAB_VS = `#version 300 es
layout(location = 0) in vec4 box;    // left, top, width, height in pixels
layout(location = 1) in vec2 offset; // centre minus (left, top)
uniform vec2 size;
flat out vec2 origin;
flat out vec2 centre;
void main() {
  vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
  origin = box.xy;
  centre = offset;
  gl_Position = vec4((box.xy + corner * box.zw) / size * 2.0 - 1.0, 0.0, 1.0);
}`;

export const DAB_FS = `${HEADER}
flat in vec2 origin;
flat in vec2 centre;
uniform float radius;
uniform float solid;
uniform float fade;
uniform int erase;
void main() {
  vec2 d = (floor(gl_FragCoord.xy) - origin) - centre;
  float distance = sqrt(d.x * d.x + d.y * d.y);
  if (distance > radius) discard;
  float strength = distance <= solid ? 1.0 : 1.0 - (distance - solid) / fade;
  float value = floor(strength * 255.0 + 0.5);
  float v = erase == 1 ? 255.0 - value : value;
  outColor = vec4(v, v, v, 255.0) / 255.0;
}`;

/** `morphCoverage`: one axis of a square max (grow) or min (choke) filter, edges repeated. */
export const MORPH_FS = `${HEADER}
uniform sampler2D source;
uniform ivec2 direction;
uniform int radius;
uniform int grow;
void main() {
  ivec2 p = here();
  ivec2 size = textureSize(source, 0);
  float v = bytesAt(source, p).r;
  for (int k = -radius; k <= radius; k++) {
    float q = bytesAt(source, clamp(p + direction * k, ivec2(0), size - 1)).r;
    v = grow == 1 ? max(v, q) : min(v, q);
  }
  outColor = vec4(v, v, v, 255.0) / 255.0;
}`;

/** 255 - coverage. */
export const MASK_INVERT_FS = `${HEADER}
uniform sampler2D source;
void main() {
  float v = 255.0 - bytesAt(source, here()).r;
  outColor = vec4(v, v, v, 255.0) / 255.0;
}`;

/** `combineCoverage`: the entry scaled by its opacity and rounded, then the mode, rounded. */
export const MASK_COMBINE_FS = `${HEADER}
uniform sampler2D accumulated;
uniform sampler2D entry;
uniform int mode;          // MASK_MODE_INDEX: add, subtract, intersect, difference
uniform float strength;
void main() {
  ivec2 p = here();
  float a = bytesAt(accumulated, p).r;
  float e = bytesAt(entry, p).r;
  if (strength != 1.0) e = floor(e * strength + 0.5);
  float value;
  if (mode == 1) value = a * (255.0 - e) / 255.0;
  else if (mode == 2) value = a * e / 255.0;
  else if (mode == 3) value = abs(a - e);
  else value = a + e - a * e / 255.0;
  float v = floor(value + 0.5);
  outColor = vec4(v, v, v, 255.0) / 255.0;
}`;

/** `layerCoverage`: another layer's alpha, or its 709 luma weighted by alpha, times its opacity. */
export const LAYER_COVERAGE_FS = `${HEADER}
uniform sampler2D source;
uniform int luma;
uniform float strength;
void main() {
  vec4 b = bytesAt(source, here());
  float value = luma == 1
    ? (0.2126 * b.r + 0.7152 * b.g + 0.0722 * b.b) * (b.a / 255.0) * strength
    : b.a * strength;
  float v = floor(value + 0.5);
  outColor = vec4(v, v, v, 255.0) / 255.0;
}`;

/** `applyAlphaCut` in photoCompose.ts: a' = a * ((mask / 255) * (baseAlpha / 255 * baseOpacity)). */
export const ALPHA_CUT_FS = `${HEADER}
uniform sampler2D layer;
uniform sampler2D mask;
uniform sampler2D base;
uniform int useMask;
uniform int useBase;
uniform float baseOpacity;
void main() {
  ivec2 p = here();
  vec4 l = bytesAt(layer, p);
  float m = useMask == 1 ? bytesAt(mask, p).r / 255.0 : 1.0;
  float c = useBase == 1 ? (bytesAt(base, p).a / 255.0) * baseOpacity : 1.0;
  outColor = store(vec4(l.rgb, l.a * (m * c)));
}`;

/**
 * `applyUnmultiply` in photoKey.ts. Integer arithmetic with round-half-to-even, because GLSL
 * division is not correctly rounded and c * 255 / m lands exactly on .5 for some pairs (1 / 2).
 */
export const UNMULTIPLY_FS = `${HEADER}
uniform sampler2D source;
int divideRounded(int n, int d) {
  int q = n / d;
  int twice = 2 * (n - q * d);
  if (twice > d || (twice == d && (q & 1) == 1)) q++;
  return q;
}
void main() {
  ivec4 b = ivec4(bytesAt(source, here()));
  int m = max(b.r, max(b.g, b.b));
  if (m == 0) { outColor = vec4(0.0); return; }
  outColor = vec4(float(divideRounded(b.r * 255, m)), float(divideRounded(b.g * 255, m)),
    float(divideRounded(b.b * 255, m)), float(divideRounded(b.a * m, 255))) / 255.0;
}`;

/** `applyLumaKey` / `lumaKeyAmount` in photoKey.ts. Bounds arrive already in bytes. */
export const LUMA_KEY_FS = `${HEADER}
uniform sampler2D source;
uniform float low;
uniform float high;
uniform float soft;
uniform int invertKey;
void main() {
  vec4 b = bytesAt(source, here());
  float luma = 0.2126 * b.r + 0.7152 * b.g + 0.0722 * b.b;
  float keep;
  if (soft <= 0.0) keep = luma >= low && luma <= high ? 1.0 : 0.0;
  else {
    float rise = clamp((luma - (low - soft)) / soft, 0.0, 1.0);
    float fall = clamp(((high + soft) - luma) / soft, 0.0, 1.0);
    keep = min(rise, fall);
  }
  if (invertKey == 1) keep = 1.0 - keep;
  outColor = store(vec4(b.rgb, b.a * keep));
}`;

/* ================================================================ VFX Phase 3: time */

/** `crossfadeFrames` in photoTime.ts: two placed frames of one layer, blended premultiplied. */
export const CROSSFADE_FS = `${HEADER}
uniform sampler2D frameA;
uniform sampler2D frameB;
uniform float weight;
void main() {
  ivec2 p = here();
  vec4 f = bytesAt(frameA, p);
  vec4 t = bytesAt(frameB, p);
  float a0 = f.a / 255.0 * (1.0 - weight);
  float a1 = t.a / 255.0 * weight;
  float alpha = a0 + a1;
  if (alpha <= 0.0) { outColor = vec4(0.0); return; }
  outColor = store(vec4((f.rgb * a0 + t.rgb * a1) / alpha, alpha * 255.0));
}`;

/** `applyAlphaCurve` in photoTime.ts. */
export const ALPHA_CURVE_FS = `${HEADER}
uniform sampler2D source;
uniform float gain;
uniform float gamma;
void main() {
  vec4 b = bytesAt(source, here());
  if (b.a == 0.0) { outColor = b / 255.0; return; }
  outColor = store(vec4(b.rgb, min(1.0, gain * pow(b.a / 255.0, gamma)) * 255.0));
}`;

/**
 * `renderMatrixRain` in matrixRain.ts, one pixel at a time. Integer and uint arithmetic only (uint
 * multiplication wraps mod 2^32 exactly as Math.imul does), so the bytes match the CPU's exactly.
 */
export const MATRIX_RAIN_FS = `${HEADER}
uniform sampler2D atlas;   // R8 coverage, glyphs side by side
uniform int seed;
uniform int cell;
uniform int rows;
uniform int loopLength;
uniform int trail;
uniform int minPasses;
uniform int maxPasses;
uniform int changes;
uniform int density;       // out of 1000
uniform int inLoop;
uniform int glyphCount;
uniform ivec3 colour;
uniform ivec3 headColour;
uint mixu(uint x) {
  x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu; x ^= x >> 16u;
  return x;
}
uint hash4(uint a, uint b, uint c, uint d) { return mixu(a ^ mixu(b ^ mixu(c ^ mixu(d + 0x9e3779b9u)))); }
void main() {
  ivec2 p = here();
  int column = p.x / cell;
  int row = p.y / cell;
  uint s = uint(seed);
  uint c = uint(column);
  if (int(hash4(s, c, 0u, 0u) % 1000u) >= density) { outColor = vec4(0.0); return; }
  int cycle = rows + trail + int(hash4(s, c, 1u, 0u) % uint(max(1, rows)));
  int passes = minPasses + int(hash4(s, c, 2u, 0u) % uint(maxPasses - minPasses + 1));
  int offset = int(hash4(s, c, 3u, 0u) % uint(cycle));
  int headRow = ((inLoop * passes * cycle) / loopLength + offset) % cycle;
  int distance = (headRow - row + cycle) % cycle;
  if (distance >= trail) { outColor = vec4(0.0); return; }
  int level = ((trail - distance) * 255) / trail;
  int phase = int(hash4(s, c, uint(row), 4u) % uint(loopLength));
  int epoch = (inLoop * changes + phase) / loopLength;
  int glyph = int(hash4(s ^ 0x51edu, c, uint(row), uint(epoch)) % uint(glyphCount));
  ivec2 local = p - ivec2(column, row) * cell;
  int coverage = int(floor(texelFetch(atlas, ivec2(glyph * cell + local.x, local.y), 0).r * 255.0 + 0.5));
  int alpha = (coverage * level + 127) / 255;
  if (alpha == 0) { outColor = vec4(0.0); return; }
  ivec3 rgb = distance == 0 ? headColour : colour;
  outColor = vec4(vec3(rgb), float(alpha)) / 255.0;
}`;

/* ================================================================ VFX Phase 4: tracks and depth */

/**
 * `applyTrail` / `trailDisplacement` in photoTrack.ts: resample through a displacement that depends
 * on the pixel's height along the layer's up axis. The table is the SAME Float32 table the CPU
 * reads. Bilinear on premultiplied bytes, outside texels transparent, then unpremultiplied.
 */
export const TRAIL_FS = `${HEADER}
uniform sampler2D source;
uniform float tableX[32];
uniform float tableY[32];
uniform vec2 inverseRow1;     // (b, d) of the inverse layer matrix
uniform float inverseOffsetY; // f
uniform float rectHeight;
uniform ivec2 size;
void main() {
  vec2 p = gl_FragCoord.xy;
  float ly = dot(inverseRow1, p) + inverseOffsetY;
  float h = clamp(1.0 - ly / rectHeight, 0.0, 1.0);
  float b = h * 31.0;
  int i0 = int(floor(b));
  int i1 = min(31, i0 + 1);
  float f = b - float(i0);
  vec2 d = vec2(tableX[i0] * (1.0 - f) + tableX[i1] * f, tableY[i0] * (1.0 - f) + tableY[i1] * f);
  vec2 s = p - d - 0.5;
  vec2 base = floor(s);
  vec2 fr = s - base;
  ivec2 i = ivec2(base);
  vec4 weights = vec4((1.0 - fr.x) * (1.0 - fr.y), fr.x * (1.0 - fr.y), (1.0 - fr.x) * fr.y, fr.x * fr.y);
  ivec2 corners[4] = ivec2[4](i, i + ivec2(1, 0), i + ivec2(0, 1), i + ivec2(1, 1));
  vec3 c = vec3(0.0);
  float a = 0.0;
  for (int k = 0; k < 4; k++) {
    ivec2 q = corners[k];
    if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) continue;
    vec4 t = bytesAt(source, q);
    float w = weights[k] * t.a;
    c += t.rgb * w;
    a += w;
  }
  if (a <= 0.0) { outColor = vec4(0.0); return; }
  outColor = store(vec4(c / a, a));
}`;

/** `depthThresholdLut` applied to placed depth coverage (an entry of a mask stack). */
export const DEPTH_THRESHOLD_FS = `${HEADER}
uniform sampler2D source;
uniform sampler2D lut;
void main() {
  int d = int(bytesAt(source, here()).r);
  float v = floor(texelFetch(lut, ivec2(d, 0), 0).r * 255.0 + 0.5);
  outColor = vec4(v, v, v, 255.0) / 255.0;
}`;
