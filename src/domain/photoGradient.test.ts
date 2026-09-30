import { describe, expect, it } from "vitest";
import {
  GRADIENT_PRESETS, NEUTRAL_GRADIENT, applyGradientMap, buildGradientLut, gradientPosition,
  hexToRgb, normalizeStops, oklabToRgb, rampParameter, renderGradient, rgbToHex, rgbToOklab,
  visibleGradientHandle, type Gradient, type GradientStop,
} from "./photoGradient";

const ramp = (over: Partial<Gradient> = {}): Gradient => ({
  ...NEUTRAL_GRADIENT,
  stops: NEUTRAL_GRADIENT.stops.map((stop) => ({ ...stop })),
  dither: 0,
  ...over,
});

const stop = (at: number, color: string, opacity = 1): GradientStop => ({ at, color, opacity });

/** The ramp colour at a position, read out of the table the renderer uses. */
function at(gradient: Gradient, t: number): [number, number, number, number] {
  const lut = buildGradientLut(gradient, 256);
  const index = Math.round(t * 255) * 4;
  return [lut[index], lut[index + 1], lut[index + 2], lut[index + 3]];
}

/* --------------------------------------------------------------- the stops */

describe("normalizeStops", () => {
  it("sorts stops that were entered out of order", () => {
    const sorted = normalizeStops([stop(0.8, "#ff0000"), stop(0.2, "#00ff00")]);
    expect(sorted.map((entry) => entry.at)).toEqual([0.2, 0.8]);
  });

  it("completes a single colour into a flat ramp rather than rejecting it", () => {
    // One stop is a reasonable thing to have typed on the way to a second.
    const filled = normalizeStops([stop(0.4, "#123456")]);
    expect(filled).toHaveLength(2);
    expect(filled[0].at).toBe(0);
    expect(filled[1].at).toBe(1);
    expect(filled[0].color).toBe("#123456");
  });

  it("falls back to black and white when there is nothing usable", () => {
    expect(normalizeStops([])).toHaveLength(2);
  });

  it("pulls positions and opacities back into range", () => {
    const fixed = normalizeStops([stop(-3, "#000000", 5), stop(9, "#ffffff", -2)]);
    expect(fixed[0].at).toBe(0);
    expect(fixed[0].opacity).toBe(1);
    expect(fixed[1].at).toBe(1);
    expect(fixed[1].opacity).toBe(0);
  });

  it("survives a stop with a nonsense position", () => {
    const fixed = normalizeStops([stop(Number.NaN, "#ff0000"), stop(1, "#0000ff")]);
    expect(fixed.every((entry) => Number.isFinite(entry.at))).toBe(true);
  });
});

/* --------------------------------------------------------------- the ramp */

describe("buildGradientLut", () => {
  it("puts each stop's own colour at its own position", () => {
    const table = ramp({ stops: [stop(0, "#ff0000"), stop(1, "#0000ff")] });
    expect(at(table, 0).slice(0, 3)).toEqual([255, 0, 0]);
    expect(at(table, 1).slice(0, 3)).toEqual([0, 0, 255]);
  });

  it("carries opacity, so a ramp can fade to nothing", () => {
    const table = ramp({ stops: [stop(0, "#000000", 1), stop(1, "#000000", 0)] });
    expect(at(table, 0)[3]).toBe(255);
    expect(at(table, 1)[3]).toBe(0);
    expect(at(table, 0.5)[3]).toBeGreaterThan(100);
    expect(at(table, 0.5)[3]).toBeLessThan(155);
  });

  it("honours more than two colours", () => {
    const table = ramp({ stops: [stop(0, "#ff0000"), stop(0.5, "#00ff00"), stop(1, "#0000ff")] });
    const middle = at(table, 0.5);
    expect(middle[1]).toBeGreaterThan(middle[0]);
    expect(middle[1]).toBeGreaterThan(middle[2]);
  });

  it("makes a hard edge from two stops in the same place, instead of dividing by zero", () => {
    const table = ramp({ stops: [stop(0, "#000000"), stop(0.5, "#000000"), stop(0.5, "#ffffff"), stop(1, "#ffffff")] });
    expect(at(table, 0.3)[0]).toBeLessThan(40);
    expect(at(table, 0.7)[0]).toBeGreaterThan(215);
    expect(Number.isFinite(at(table, 0.5)[0])).toBe(true);
  });

  it("keeps blue-to-yellow out of the grey it goes through in sRGB", () => {
    // The reason OKLab is the default. Straight channel mixing sends this pair through a dead
    // grey-brown; a perceptual space keeps the crossing saturated.
    const stops = [stop(0, "#0040ff"), stop(1, "#ffcc00")];
    const perceptual = at(ramp({ stops, space: "oklab" }), 0.5);
    const numeric = at(ramp({ stops, space: "rgb" }), 0.5);
    const chroma = (color: number[]) => Math.max(...color.slice(0, 3)) - Math.min(...color.slice(0, 3));
    expect(chroma(perceptual)).toBeGreaterThan(chroma(numeric));
  });

  it("still ends on exactly the colours it was given, whichever space it mixed in", () => {
    const stops = [stop(0, "#0040ff"), stop(1, "#ffcc00")];
    for (const space of ["oklab", "rgb"] as const) {
      const ends = at(ramp({ stops, space }), 0);
      expect(ends[0]).toBeCloseTo(0, -1);
      expect(ends[2]).toBeGreaterThan(240);
    }
  });
});

describe("visibleGradientHandle", () => {
  it("leaves reachable handles exactly where the gradient put them", () => {
    expect(visibleGradientHandle([0.35, 0.7])).toEqual({ at: [0.35, 0.7], clipped: false });
  });

  it("offers an inset edge proxy for handles outside the canvas", () => {
    expect(visibleGradientHandle([-0.5, 2])).toEqual({ at: [0.025, 0.975], clipped: true });
  });

  it("recovers safely from invalid stored coordinates", () => {
    expect(visibleGradientHandle([Number.NaN, Number.POSITIVE_INFINITY])).toEqual({
      at: [0.5, 0.5], clipped: true,
    });
  });
});

/* ------------------------------------------------------------ the geometry */

describe("gradientPosition", () => {
  const frame = { width: 200, height: 100 };

  it("runs left to right at zero degrees, centred on a half", () => {
    const g = ramp({ angle: 0 });
    expect(gradientPosition(g, 0, 50, frame.width, frame.height)).toBeCloseTo(0, 5);
    expect(gradientPosition(g, 100, 50, frame.width, frame.height)).toBeCloseTo(0.5, 5);
    expect(gradientPosition(g, 200, 50, frame.width, frame.height)).toBeCloseTo(1, 5);
  });

  it("runs top to bottom at ninety", () => {
    const g = ramp({ angle: 90 });
    expect(gradientPosition(g, 100, 0, frame.width, frame.height)).toBeCloseTo(0, 5);
    expect(gradientPosition(g, 100, 100, frame.width, frame.height)).toBeCloseTo(1, 5);
  });

  it("reverses when the angle turns around", () => {
    const forward = gradientPosition(ramp({ angle: 0 }), 180, 50, frame.width, frame.height);
    const back = gradientPosition(ramp({ angle: 180 }), 180, 50, frame.width, frame.height);
    expect(forward + back).toBeCloseTo(1, 5);
  });

  it("still reaches the corner at forty-five degrees, which a width-based extent would not", () => {
    // The whole reason the extent is the frame's half-diagonal projected onto the axis: measured
    // against the width instead, a diagonal ramp finishes long before the corner and every angle
    // needs its own spread to look the same.
    const corner = gradientPosition(ramp({ angle: 45 }), frame.width, frame.height, frame.width, frame.height);
    expect(corner).toBeGreaterThan(0.99);
    expect(corner).toBeLessThan(1.01);
  });

  it("mirrors about the centre when reflected", () => {
    const g = ramp({ kind: "reflected", angle: 0 });
    const left = gradientPosition(g, 40, 50, frame.width, frame.height);
    const right = gradientPosition(g, 160, 50, frame.width, frame.height);
    expect(left).toBeCloseTo(right, 5);
    expect(gradientPosition(g, 100, 50, frame.width, frame.height)).toBeCloseTo(0, 5);
  });

  it("measures distance from the centre when radial", () => {
    const g = ramp({ kind: "radial" });
    expect(gradientPosition(g, 100, 50, frame.width, frame.height)).toBeCloseTo(0, 5);
    const edge = gradientPosition(g, 200, 100, frame.width, frame.height);
    expect(edge).toBeCloseTo(1, 5);
  });

  it("sweeps around when angular", () => {
    const g = ramp({ kind: "angular", angle: 0 });
    const right = gradientPosition(g, 200, 50, frame.width, frame.height);
    const left = gradientPosition(g, 0, 50, frame.width, frame.height);
    expect(right).toBeCloseTo(0.5, 4);
    expect(Math.abs(left - 0.5)).toBeGreaterThan(0.4);
  });

  it("adds the two axes when diamond, so its contours have corners", () => {
    const g = ramp({ kind: "diamond", angle: 0 });
    const straight = gradientPosition(g, 200, 50, frame.width, frame.height);
    const cornerward = gradientPosition(g, 200, 100, frame.width, frame.height);
    expect(cornerward).toBeGreaterThan(straight);
  });

  it("moves with the centre", () => {
    const g = ramp({ angle: 0, cx: 0.25 });
    expect(gradientPosition(g, 50, 50, frame.width, frame.height)).toBeCloseTo(0.5, 5);
  });

  it("compresses with the spread", () => {
    const wide = gradientPosition(ramp({ angle: 0, scale: 1 }), 150, 50, frame.width, frame.height);
    const tight = gradientPosition(ramp({ angle: 0, scale: 0.5 }), 150, 50, frame.width, frame.height);
    expect(tight - 0.5).toBeCloseTo((wide - 0.5) * 2, 5);
  });
});

describe("rampParameter", () => {
  it("clamps at both ends when the ramp runs once", () => {
    expect(rampParameter(ramp(), -0.4)).toBe(0);
    expect(rampParameter(ramp(), 1.4)).toBe(1);
  });

  it("wraps when it repeats", () => {
    const g = ramp({ repeat: 4 });
    expect(rampParameter(g, 0.25)).toBeCloseTo(0, 5);
    expect(rampParameter(g, 0.375)).toBeCloseTo(0.5, 5);
  });

  it("ping-pongs when mirrored, so a repeat has no seam", () => {
    const g = ramp({ repeat: 2, mirror: true });
    // Just before and just after the turn are the same value, which is what "no seam" means.
    expect(rampParameter(g, 0.5 - 0.001)).toBeCloseTo(rampParameter(g, 0.5 + 0.001), 3);
  });

  it("bends toward one end when eased, without moving the ends themselves", () => {
    const eased = ramp({ ease: 2 });
    expect(rampParameter(eased, 0)).toBe(0);
    expect(rampParameter(eased, 1)).toBe(1);
    expect(rampParameter(eased, 0.5)).toBeCloseTo(0.25, 5);
  });

  it("never leaves the table's range, whatever it is handed", () => {
    for (const value of [-9, -0.001, 0, 0.5, 1, 1.001, 12, 1e9]) {
      const p = rampParameter(ramp({ repeat: 3, mirror: true, ease: 0.4 }), value);
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });
});

/* ---------------------------------------------------------------- the fill */

describe("renderGradient", () => {
  const fill = (gradient: Gradient, width = 32, height = 32) => {
    const data = new Uint8ClampedArray(width * height * 4);
    renderGradient(data, width, height, gradient);
    return data;
  };
  const pixel = (data: Uint8ClampedArray, width: number, x: number, y: number) => {
    const at = (y * width + x) * 4;
    return [data[at], data[at + 1], data[at + 2], data[at + 3]];
  };

  it("paints dark to light across the frame", () => {
    const data = fill(ramp({ angle: 0 }));
    expect(pixel(data, 32, 0, 16)[0]).toBeLessThan(20);
    expect(pixel(data, 32, 31, 16)[0]).toBeGreaterThan(235);
  });

  it("writes alpha, so a fade to nothing really is transparent", () => {
    const data = fill(ramp({ angle: 0, stops: [stop(0, "#000000", 1), stop(1, "#000000", 0)] }));
    expect(pixel(data, 32, 0, 16)[3]).toBeGreaterThan(240);
    expect(pixel(data, 32, 31, 16)[3]).toBeLessThan(15);
  });

  it("is constant along the direction it is not travelling", () => {
    const data = fill(ramp({ angle: 0 }));
    expect(pixel(data, 32, 10, 0)[0]).toBe(pixel(data, 32, 10, 31)[0]);
  });

  it("breaks up banding when dithering is on, and is flat when it is off", () => {
    const width = 64;
    const flat = fill(ramp({ angle: 0, dither: 0 }), width, 8);
    const dithered = fill(ramp({ angle: 0, dither: 1 }), width, 8);
    const columnVaries = (data: Uint8ClampedArray) => {
      let varied = 0;
      for (let x = 0; x < width; x++) {
        const first = pixel(data, width, x, 0)[0];
        for (let y = 1; y < 8; y++) if (pixel(data, width, x, y)[0] !== first) { varied++; break; }
      }
      return varied;
    };
    expect(columnVaries(flat)).toBe(0);
    expect(columnVaries(dithered)).toBeGreaterThan(0);
  });

  it("keeps the dither pattern fixed, so a still picture has no crawling grain", () => {
    const once = fill(ramp({ angle: 0, dither: 1 }));
    const twice = fill(ramp({ angle: 0, dither: 1 }));
    expect([...once]).toEqual([...twice]);
  });

  it("renders every preset without producing a NaN", () => {
    for (const preset of GRADIENT_PRESETS) {
      const data = fill(ramp(preset.gradient));
      expect(data.every((value) => Number.isFinite(value))).toBe(true);
    }
  });
});

/* ----------------------------------------------------------------- the map */

describe("applyGradientMap", () => {
  /** A strip running black to white, which is what a tone map is defined against. */
  const strip = () => {
    const data = new Uint8ClampedArray(256 * 4);
    for (let index = 0, at = 0; index < 256; index++, at += 4) {
      data[at] = data[at + 1] = data[at + 2] = index;
      data[at + 3] = 255;
    }
    return data;
  };
  const read = (data: Uint8ClampedArray, index: number) =>
    [data[index * 4], data[index * 4 + 1], data[index * 4 + 2]];

  it("does nothing at zero", () => {
    const before = strip();
    const after = strip();
    applyGradientMap(after, ramp(), 0);
    expect([...after]).toEqual([...before]);
  });

  it("sends shadows to one end of the ramp and highlights to the other", () => {
    const data = strip();
    applyGradientMap(data, ramp({ stops: [stop(0, "#2040ff"), stop(1, "#ffcc20")] }), 100);
    const dark = read(data, 4);
    const light = read(data, 250);
    expect(dark[2]).toBeGreaterThan(dark[0]);
    expect(light[0]).toBeGreaterThan(light[2]);
  });

  it("moves the picture only part of the way at a partial amount", () => {
    const half = strip();
    const full = strip();
    const duotone = ramp({ stops: [stop(0, "#2040ff"), stop(1, "#ffcc20")] });
    applyGradientMap(half, duotone, 50);
    applyGradientMap(full, duotone, 100);
    const original = 200;
    expect(Math.abs(read(half, 200)[2] - original))
      .toBeLessThan(Math.abs(read(full, 200)[2] - original));
  });

  it("never touches alpha", () => {
    const data = strip();
    for (let at = 3; at < data.length; at += 4) data[at] = 90;
    applyGradientMap(data, ramp(), 100);
    for (let at = 3; at < data.length; at += 4) expect(data[at]).toBe(90);
  });

  it("weights by the stop's own opacity, so a faded ramp fades out of the map too", () => {
    const solid = strip();
    const faded = strip();
    applyGradientMap(solid, ramp({ stops: [stop(0, "#ff0000", 1), stop(1, "#ff0000", 1)] }), 100);
    applyGradientMap(faded, ramp({ stops: [stop(0, "#ff0000", 0.2), stop(1, "#ff0000", 0.2)] }), 100);
    expect(Math.abs(read(faded, 128)[0] - 128)).toBeLessThan(Math.abs(read(solid, 128)[0] - 128));
  });
});

/* -------------------------------------------------------------- the colour */

describe("colour conversion", () => {
  it("reads both hex lengths and refuses to produce NaN", () => {
    expect(hexToRgb("#ff8000")).toEqual([255, 128, 0]);
    expect(hexToRgb("#f80")).toEqual([255, 136, 0]);
    expect(hexToRgb("nonsense")).toEqual([0, 0, 0]);
    expect(hexToRgb("")).toEqual([0, 0, 0]);
  });

  it("writes hex back with both digits", () => {
    expect(rgbToHex(255, 128, 0)).toBe("#ff8000");
    expect(rgbToHex(0, 0, 5)).toBe("#000005");
    expect(rgbToHex(-40, 900, 12.6)).toBe("#00ff0d");
  });

  it("round-trips through OKLab", () => {
    for (const colour of [[0, 0, 0], [255, 255, 255], [255, 128, 0], [18, 52, 86], [0, 64, 255]]) {
      const [L, a, b] = rgbToOklab(colour[0], colour[1], colour[2]);
      const back = oklabToRgb(L, a, b);
      for (let channel = 0; channel < 3; channel++) {
        expect(Math.abs(back[channel] - colour[channel])).toBeLessThan(1);
      }
    }
  });

  it("puts grey on the neutral axis, which is what makes the space usable", () => {
    const [, a, b] = rgbToOklab(128, 128, 128);
    expect(Math.abs(a)).toBeLessThan(1e-3);
    expect(Math.abs(b)).toBeLessThan(1e-3);
  });

  it("orders lightness the way brightness does", () => {
    expect(rgbToOklab(20, 20, 20)[0]).toBeLessThan(rgbToOklab(200, 200, 200)[0]);
  });
});
