/**
 * Spike 0.2: does the GPU API work in WebView2?
 *
 * Both APIs render the same scene into a 3840x2160 RGBA16F target:
 *   1. clear to the backdrop B = (0.2, 0.4, 0.6, 1);
 *   2. draw a full-target quad textured from a 2x2 premultiplied source S with fixed-function
 *      "over" blending (ONE, ONE_MINUS_SRC_ALPHA);
 *   3. a second pass reads that result as a texture and applies a shader-side multiply by M,
 *      which is how the compositor will do non-separable and gamma-space blends.
 * The four quadrants are read back and compared with the arithmetic below. Texel 3 is 2.0 in red:
 * it only survives if the target really is floating point.
 */
import { compile, createGl, drawFullscreen, framebuffer, FULLSCREEN_VS, rendererInfo, round, texture } from "./gl";

const W = 3840;
const H = 2160;
const BACKDROP = [0.2, 0.4, 0.6, 1.0];
/** Premultiplied texels, row-major, row 0 first. */
const SOURCE = [
  [1.0, 0.0, 0.0, 1.0], // opaque red
  [0.0, 0.5, 0.0, 0.5], // half-transparent green
  [0.0, 0.0, 0.0, 0.0], // fully transparent
  [2.0, 1.0, 0.5, 1.0], // HDR, only representable in a float target
];
const MULTIPLY = [0.5, 0.25, 1.0];

function expected(): number[][] {
  return SOURCE.map((s) => {
    const over = [0, 1, 2, 3].map((c) => s[c] + BACKDROP[c] * (1 - s[3]));
    return [over[0] * MULTIPLY[0], over[1] * MULTIPLY[1], over[2] * MULTIPLY[2], over[3]];
  });
}

/** Centre of texel `index`'s quadrant, in "row 0 first" coordinates. */
function quadrantCentre(index: number): [number, number] {
  return [(index % 2) * (W / 2) + W / 4, Math.floor(index / 2) * (H / 2) + H / 4];
}

function maxError(actual: number[][], wanted: number[][]): number {
  let worst = 0;
  actual.forEach((pixel, i) => pixel.forEach((value, c) => { worst = Math.max(worst, Math.abs(value - wanted[i][c])); }));
  return worst;
}

export type GpuApiResult = {
  available: boolean;
  error?: string;
  adapter?: Record<string, unknown>;
  maxTexture2D?: number;
  floatTarget?: boolean;
  floatBlend?: boolean;
  pixels?: number[][];
  expected?: number[][];
  maxAbsError?: number;
  blendCorrect?: boolean;
  hdrPreserved?: boolean;
  /** Rough full-target shader pass rate; only meaningful on real hardware. */
  passesPerSecond4k?: number;
  extensions?: string[];
  features?: string[];
};

export async function runWebGl2(): Promise<GpuApiResult> {
  let gl: WebGL2RenderingContext;
  try { gl = createGl(); } catch (error) { return { available: false, error: String(error) }; }
  const wanted = expected();
  const result: GpuApiResult = {
    available: true,
    adapter: { ...rendererInfo(gl), version: gl.getParameter(gl.VERSION), glsl: gl.getParameter(gl.SHADING_LANGUAGE_VERSION) },
    maxTexture2D: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    extensions: gl.getSupportedExtensions() ?? [],
  };
  const colorBufferFloat = !!gl.getExtension("EXT_color_buffer_float");
  result.floatBlend = !!gl.getExtension("EXT_float_blend") || colorBufferFloat;
  gl.getExtension("EXT_color_buffer_half_float");
  gl.getExtension("OES_texture_float_linear");
  if (!colorBufferFloat) {
    result.floatTarget = false;
    result.error = "EXT_color_buffer_float missing: no renderable float targets";
    return result;
  }
  try {
    const source = texture(gl, gl.RGBA32F, 2, 2);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 2, 2, gl.RGBA, gl.FLOAT, new Float32Array(SOURCE.flat()));
    const over = texture(gl, gl.RGBA16F, W, H);
    const overFb = framebuffer(gl, over);
    result.floatTarget = true;
    gl.viewport(0, 0, W, H);
    gl.clearColor(BACKDROP[0], BACKDROP[1], BACKDROP[2], BACKDROP[3]);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const quad = compile(gl, FULLSCREEN_VS, `#version 300 es
      precision highp float; in vec2 uv; uniform highp sampler2D src; out vec4 color;
      void main() { color = texelFetch(src, ivec2(floor(uv * 2.0)), 0); }`);
    gl.useProgram(quad);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, source);
    gl.uniform1i(gl.getUniformLocation(quad, "src"), 0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    drawFullscreen(gl);
    gl.disable(gl.BLEND);

    const final = texture(gl, gl.RGBA16F, W, H);
    const finalFb = framebuffer(gl, final);
    const multiply = compile(gl, FULLSCREEN_VS, `#version 300 es
      precision highp float; in vec2 uv; uniform highp sampler2D backdrop; uniform vec3 m; out vec4 color;
      void main() { vec4 b = texelFetch(backdrop, ivec2(gl_FragCoord.xy), 0); color = vec4(b.rgb * m, b.a); }`);
    gl.useProgram(multiply);
    gl.bindTexture(gl.TEXTURE_2D, over);
    gl.uniform1i(gl.getUniformLocation(multiply, "backdrop"), 0);
    gl.uniform3fv(gl.getUniformLocation(multiply, "m"), MULTIPLY);
    drawFullscreen(gl);

    const pixels: number[][] = [];
    const one = new Float32Array(4);
    for (let i = 0; i < 4; i++) {
      const [x, y] = quadrantCentre(i);
      gl.readPixels(x, y, 1, 1, gl.RGBA, gl.FLOAT, one);
      pixels.push(Array.from(one, (v) => round(v, 4)));
    }
    result.pixels = pixels;
    result.expected = wanted;
    result.maxAbsError = round(maxError(pixels, wanted), 5);
    result.blendCorrect = result.maxAbsError < 2e-3;
    result.hdrPreserved = pixels[3][0] > 0.99;

    // Throughput: one full-target shader pass per iteration, synchronised by a 1-pixel read.
    gl.bindFramebuffer(gl.FRAMEBUFFER, finalFb);
    gl.useProgram(multiply);
    const started = performance.now();
    let passes = 0;
    while (performance.now() - started < 1000 && passes < 240) {
      drawFullscreen(gl);
      passes++;
      if (passes % 8 === 0) gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, one);
    }
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, one);
    result.passesPerSecond4k = round(passes / ((performance.now() - started) / 1000), 1);
    gl.deleteFramebuffer(overFb);
    gl.deleteFramebuffer(finalFb);
  } catch (error) {
    result.error = String(error);
  }
  gl.getExtension("WEBGL_lose_context")?.loseContext();
  return result;
}

/** IEEE half to float. */
function halfToFloat(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exponent = (h >> 10) & 0x1f;
  const fraction = h & 0x3ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Infinity;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

export async function runWebGpu(): Promise<GpuApiResult> {
  const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
  if (!gpu) return { available: false, error: "navigator.gpu is undefined" };
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) return { available: false, error: "requestAdapter returned null" };
  const info = (adapter as GPUAdapter & { info?: GPUAdapterInfo }).info;
  const result: GpuApiResult = {
    available: true,
    adapter: info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description } : {},
    maxTexture2D: adapter.limits.maxTextureDimension2D,
    features: [...adapter.features].sort(),
  };
  try {
    const device = await adapter.requestDevice({
      requiredLimits: { maxTextureDimension2D: Math.min(adapter.limits.maxTextureDimension2D, 8192) },
    });
    const errors: string[] = [];
    device.pushErrorScope("validation");
    const source = device.createTexture({ size: [2, 2], format: "rgba32float", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture: source }, new Float32Array(SOURCE.flat()), { bytesPerRow: 2 * 16 }, [2, 2]);
    const make = () => device.createTexture({
      size: [W, H], format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
    });
    const over = make();
    const final = make();
    result.floatTarget = true;
    result.floatBlend = true; // rgba16float is blendable in core WebGPU.

    const vs = `@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
      let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
      return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
    }`;
    // WebGPU framebuffer row 0 is the top row, so position.y is already "row 0 first".
    const quadModule = device.createShaderModule({ code: `${vs}
      @group(0) @binding(0) var src: texture_2d<f32>;
      @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let texel = vec2i(floor(pos.xy / vec2f(${W / 2}.0, ${H / 2}.0)));
        return textureLoad(src, texel, 0);
      }` });
    const multiplyModule = device.createShaderModule({ code: `${vs}
      @group(0) @binding(0) var backdrop: texture_2d<f32>;
      @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let b = textureLoad(backdrop, vec2i(pos.xy), 0);
        return vec4f(b.rgb * vec3f(${MULTIPLY.map((v) => v.toFixed(4)).join(", ")}), b.a);
      }` });
    const blend: GPUBlendState = {
      color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
      alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
    };
    const quadPipeline = device.createRenderPipeline({
      layout: "auto", vertex: { module: quadModule, entryPoint: "vs" },
      fragment: { module: quadModule, entryPoint: "fs", targets: [{ format: "rgba16float", blend }] },
    });
    const multiplyPipeline = device.createRenderPipeline({
      layout: "auto", vertex: { module: multiplyModule, entryPoint: "vs" },
      fragment: { module: multiplyModule, entryPoint: "fs", targets: [{ format: "rgba16float" }] },
    });
    const quadGroup = device.createBindGroup({ layout: quadPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: source.createView() }] });
    const multiplyGroup = device.createBindGroup({ layout: multiplyPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: over.createView() }] });

    const encodePasses = (encoder: GPUCommandEncoder) => {
      const first = encoder.beginRenderPass({ colorAttachments: [{
        view: over.createView(), loadOp: "clear", storeOp: "store",
        clearValue: { r: BACKDROP[0], g: BACKDROP[1], b: BACKDROP[2], a: BACKDROP[3] },
      }] });
      first.setPipeline(quadPipeline); first.setBindGroup(0, quadGroup); first.draw(3); first.end();
      const second = encoder.beginRenderPass({ colorAttachments: [{ view: final.createView(), loadOp: "clear", storeOp: "store" }] });
      second.setPipeline(multiplyPipeline); second.setBindGroup(0, multiplyGroup); second.draw(3); second.end();
    };

    // Read each quadrant centre: one 256-byte row per pixel (8 bytes per rgba16float texel).
    const readback = device.createBuffer({ size: 256 * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encodePasses(encoder);
    for (let i = 0; i < 4; i++) {
      const [x, y] = quadrantCentre(i);
      encoder.copyTextureToBuffer({ texture: final, origin: [x, y] }, { buffer: readback, offset: i * 256, bytesPerRow: 256 }, [1, 1]);
    }
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(readback.getMappedRange().slice(0));
    readback.unmap();
    const pixels = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((c) => round(halfToFloat(halves[i * 128 + c]), 4)));
    const validation = await device.popErrorScope();
    if (validation) errors.push(validation.message);
    const wanted = expected();
    result.pixels = pixels;
    result.expected = wanted;
    result.maxAbsError = round(maxError(pixels, wanted), 5);
    result.blendCorrect = result.maxAbsError < 2e-3;
    result.hdrPreserved = pixels[3][0] > 0.99;

    const started = performance.now();
    let passes = 0;
    while (performance.now() - started < 1000 && passes < 240) {
      const batch = device.createCommandEncoder();
      for (let i = 0; i < 8; i++) encodePasses(batch);
      device.queue.submit([batch.finish()]);
      await device.queue.onSubmittedWorkDone();
      passes += 16; // two full-target passes per encodePasses
    }
    result.passesPerSecond4k = round(passes / ((performance.now() - started) / 1000), 1);
    if (errors.length) result.error = errors.join("; ");
    device.destroy();
  } catch (error) {
    result.error = String(error);
  }
  return result;
}

export async function runGpuSpike() {
  const webgl2 = await runWebGl2();
  const webgpu = await runWebGpu().catch((error) => ({ available: false, error: String(error) }) as GpuApiResult);
  return {
    webgl2,
    webgpu,
    pass: !!(webgl2.floatTarget && webgl2.blendCorrect && webgl2.hdrPreserved),
    passBar: "WebGL2 renders to RGBA16F at 3840x2160, over + shader multiply within 2e-3, HDR 2.0 preserved",
  };
}
