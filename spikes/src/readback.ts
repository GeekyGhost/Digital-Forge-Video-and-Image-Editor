/**
 * Spike 0.4: render offscreen, read back, move the bytes to Rust, pipe into ffmpeg -> NVENC.
 *
 * Round 1 (2026-09-27) found the GPU side fast (1080p readback 195 fps WebGL2, 484 fps WebGPU) and
 * the Tauri IPC raw body the bottleneck: ~100 MB/s whatever the API, ~0.09 CPU-s per frame, so
 * 12 fps at 1080p and 3 fps at 4K. Round 2 measures the two remedies, alone and together:
 *   format    rgba (4 B/px), or yuv420 packed by a shader (1.5 B/px, BT.709 limited range --
 *             which also takes the RGB->YUV matrix away from NVENC and makes it ours)
 *   transport ipc (Tauri raw body) or http (a token-protected 127.0.0.1 listener in Rust)
 *   sink      null (Rust discards) or encode (ffmpeg rawvideo on stdin -> NVENC)
 * Every frame carries its index in the top-left strip. After `encode` the output is decoded again:
 * the strip sequence proves nothing was dropped or reordered, and frame 0 is compared with the
 * rendered RGBA to measure the colour error of the whole export path.
 */
import type { AppUsage, Backend } from "./backend";
import { decodeAll, decoderConfig, demux, Seeker } from "./demux";
import { compareRgba, compile, createGl, drawFullscreen, framebuffer, FULLSCREEN_VS, mutableTexture, readIndexStrip, round, texture } from "./gl";
import { uploadBrowser } from "./yuv";

const SCENE_FS = `#version 300 es
precision highp float;
in vec2 uv;
uniform float t;
out vec4 color;
// A deliberately non-trivial HDR scene: gradients, moving soft discs, and a hash grain.
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
void main() {
  vec3 c = vec3(uv.x, uv.y, 0.5 + 0.5 * sin(t));
  for (int i = 0; i < 6; i++) {
    float fi = float(i);
    vec2 centre = vec2(0.5 + 0.35 * sin(t * (0.7 + 0.1 * fi) + fi), 0.5 + 0.35 * cos(t * (0.9 + 0.13 * fi) + fi * 1.7));
    float d = length((uv - centre) * vec2(1.7778, 1.0));
    c += vec3(1.5, 0.8, 0.3) * exp(-d * d * 60.0) * (0.5 + 0.5 * sin(fi + t));
  }
  c += (hash(gl_FragCoord.xy + t) - 0.5) * 0.02;
  color = vec4(c, 1.0);
}`;

const OUTPUT_FS = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D scene;
uniform int frameIndex;
out vec4 color;
void main() {
  vec3 c = texelFetch(scene, ivec2(gl_FragCoord.xy), 0).rgb;
  c = c / (1.0 + c); // simple tone map, HDR -> [0,1)
  c = pow(c, vec3(1.0 / 2.2));
  if (gl_FragCoord.x < 256.0 && gl_FragCoord.y < 32.0) {
    int bit = int(gl_FragCoord.x) / 32;
    c = vec3(float((frameIndex >> bit) & 1));
  }
  color = vec4(c, 1.0);
}`;

/**
 * Packs the RGBA8 output into planar I420 (Y, then U, then V), four bytes per RGBA8 texel, in a
 * (W/4) x (3H/2) target, so one readPixels yields exactly what `ffmpeg -pix_fmt yuv420p` expects.
 * BT.709, limited range; chroma is the 2x2 average (centre-sited, which round 1 measured Chromium's
 * decoder to assume). Needs W % 8 == 0 and H % 4 == 0.
 */
const PACK_YUV420_FS = `#version 300 es
precision highp float;
uniform sampler2D rgb;
uniform int W;
uniform int H;
out vec4 color;
float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float yByte(ivec2 p) { return (16.0 + 219.0 * luma(texelFetch(rgb, p, 0).rgb)) / 255.0; }
vec2 cbcr(ivec2 cp) {
  ivec2 p = cp * 2;
  vec3 c = 0.25 * (texelFetch(rgb, p, 0).rgb + texelFetch(rgb, p + ivec2(1, 0), 0).rgb
                 + texelFetch(rgb, p + ivec2(0, 1), 0).rgb + texelFetch(rgb, p + ivec2(1, 1), 0).rgb);
  float y = luma(c);
  return vec2(128.0 + 224.0 * (c.b - y) / 1.8556, 128.0 + 224.0 * (c.r - y) / 1.5748) / 255.0;
}
void main() {
  ivec2 t = ivec2(gl_FragCoord.xy);
  if (t.y < H) {
    int x = t.x * 4;
    color = vec4(yByte(ivec2(x, t.y)), yByte(ivec2(x + 1, t.y)), yByte(ivec2(x + 2, t.y)), yByte(ivec2(x + 3, t.y)));
    return;
  }
  int quarter = H / 4;
  int plane = (t.y - H) / quarter;           // 0 = U, 1 = V
  int r = (t.y - H) - plane * quarter;        // each target row holds two chroma rows
  int bx = t.x * 4;
  int halfWidth = W / 2;                      // "half" is a reserved word in GLSL ES
  int cy = 2 * r + bx / halfWidth;
  int cx = bx - (bx / halfWidth) * halfWidth;
  vec4 o;
  for (int k = 0; k < 4; k++) {
    vec2 c = cbcr(ivec2(cx + k, cy));
    o[k] = plane == 0 ? c.x : c.y;
  }
  color = o;
}`;

export type PixelFormat = "rgba" | "yuv420";
export type Transport = "none" | "ipc" | "http";
export type Sink = "null" | "encode";

export type ReadbackRun = {
  api: "webgl2" | "webgpu";
  format: PixelFormat;
  transport: Transport;
  sink: Sink;
  width: number;
  height: number;
  frames: number;
  fps: number;
  mbPerSecond: number;
  /** Mean ms per frame waiting on GPU + copying out of the pack buffer. */
  readbackMs: number;
  /** Mean ms per frame for the transport call (includes Rust's write into ffmpeg). */
  transportMs: number;
  bytesPerFrame: number;
  /** Frame 0 decoded back from the output vs the rendered RGBA (whole export path colour). */
  outputColour?: { mae: number; p99: number; max: number } | { error: string };
  encoder?: string;
  outputFrames?: number | null;
  outputPixFmt?: string | null;
  outputOrderCorrect?: boolean;
  outputOrderDetail?: string;
  rustWriteSeconds?: number;
  rustDrainSeconds?: number;
  cpuSecondsPerFrame?: number;
  peakPrivateGb?: number;
  error?: string;
};

interface Renderer {
  api: "webgl2" | "webgpu";
  bytesPerFrame: number;
  /** Synchronous RGBA8 read of the frame just submitted (for the colour reference). */
  snapshotRgba(): Uint8Array;
  /** Starts rendering frame `index` and its readback into ring slot `slot`. */
  submit(index: number, slot: number): void;
  /** Waits for slot `slot` and returns its RGBA8 bytes (valid until the slot is reused). */
  collect(slot: number): Promise<Uint8Array>;
  dispose(): void;
}

class WebGl2Renderer implements Renderer {
  api = "webgl2" as const;
  private gl: WebGL2RenderingContext;
  private scene: WebGLProgram;
  private output: WebGLProgram;
  private sceneFb: WebGLFramebuffer;
  private sceneTex: WebGLTexture;
  private outFb: WebGLFramebuffer;
  private outTex: WebGLTexture;
  private pack: WebGLProgram | null = null;
  private packFb: WebGLFramebuffer | null = null;
  private pbos: WebGLBuffer[] = [];
  private fences: (WebGLSync | null)[] = [];
  private cpu: Uint8Array;
  bytesPerFrame: number;

  constructor(private width: number, private height: number, slots: number, private format: PixelFormat = "rgba") {
    const gl = createGl(16, 16);
    this.gl = gl;
    if (!gl.getExtension("EXT_color_buffer_float")) throw new Error("no float render targets");
    this.scene = compile(gl, FULLSCREEN_VS, SCENE_FS);
    this.output = compile(gl, FULLSCREEN_VS, OUTPUT_FS);
    this.sceneTex = texture(gl, gl.RGBA16F, width, height);
    this.sceneFb = framebuffer(gl, this.sceneTex);
    this.outTex = texture(gl, gl.RGBA8, width, height);
    this.outFb = framebuffer(gl, this.outTex);
    if (format === "yuv420") {
      if (width % 8 || height % 4) throw new Error(`yuv420 packing needs W % 8 == 0 and H % 4 == 0, got ${width}x${height}`);
      this.pack = compile(gl, FULLSCREEN_VS, PACK_YUV420_FS);
      this.packFb = framebuffer(gl, texture(gl, gl.RGBA8, width / 4, (height * 3) / 2));
    }
    const bytes = format === "yuv420" ? (width * height * 3) / 2 : width * height * 4;
    this.bytesPerFrame = bytes;
    for (let i = 0; i < slots; i++) {
      const pbo = gl.createBuffer()!;
      gl.bindBuffer(gl.PIXEL_PACK_BUFFER, pbo);
      gl.bufferData(gl.PIXEL_PACK_BUFFER, bytes, gl.STREAM_READ);
      this.pbos.push(pbo);
      this.fences.push(null);
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.cpu = new Uint8Array(bytes);
  }

  submit(index: number, slot: number) {
    const gl = this.gl;
    gl.viewport(0, 0, this.width, this.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneFb);
    gl.useProgram(this.scene);
    gl.uniform1f(gl.getUniformLocation(this.scene, "t"), index / 24);
    drawFullscreen(gl);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFb);
    gl.useProgram(this.output);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.sceneTex);
    gl.uniform1i(gl.getUniformLocation(this.output, "scene"), 0);
    gl.uniform1i(gl.getUniformLocation(this.output, "frameIndex"), index);
    drawFullscreen(gl);
    let readWidth = this.width;
    let readHeight = this.height;
    if (this.pack && this.packFb) {
      readWidth = this.width / 4;
      readHeight = (this.height * 3) / 2;
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.packFb);
      gl.viewport(0, 0, readWidth, readHeight);
      gl.useProgram(this.pack);
      gl.bindTexture(gl.TEXTURE_2D, this.outTex);
      gl.uniform1i(gl.getUniformLocation(this.pack, "rgb"), 0);
      gl.uniform1i(gl.getUniformLocation(this.pack, "W"), this.width);
      gl.uniform1i(gl.getUniformLocation(this.pack, "H"), this.height);
      drawFullscreen(gl);
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbos[slot]);
    gl.readPixels(0, 0, readWidth, readHeight, gl.RGBA, gl.UNSIGNED_BYTE, 0);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    this.fences[slot] = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
    gl.flush();
  }

  async collect(slot: number): Promise<Uint8Array> {
    const gl = this.gl;
    const fence = this.fences[slot];
    if (fence) {
      for (;;) {
        const state = gl.clientWaitSync(fence, 0, 0);
        if (state === gl.ALREADY_SIGNALED || state === gl.CONDITION_SATISFIED) break;
        if (state === gl.WAIT_FAILED) throw new Error("clientWaitSync failed");
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
      gl.deleteSync(fence);
      this.fences[slot] = null;
    }
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, this.pbos[slot]);
    gl.getBufferSubData(gl.PIXEL_PACK_BUFFER, 0, this.cpu);
    gl.bindBuffer(gl.PIXEL_PACK_BUFFER, null);
    return this.cpu;
  }

  snapshotRgba(): Uint8Array {
    const gl = this.gl;
    const pixels = new Uint8Array(this.width * this.height * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.outFb);
    gl.readPixels(0, 0, this.width, this.height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    return pixels;
  }

  dispose() { this.gl.getExtension("WEBGL_lose_context")?.loseContext(); }
}

class WebGpuRenderer implements Renderer {
  api = "webgpu" as const;
  get bytesPerFrame() { return this.width * this.height * 4; }
  snapshotRgba(): Uint8Array { throw new Error("round 2 colour reference is WebGL2 only"); }
  private buffers: GPUBuffer[] = [];
  private pending: (Promise<void> | null)[] = [];
  private cpu: Uint8Array;
  private scenePipeline: any;
  private outputPipeline: any;
  private sceneTex: GPUTexture;
  private outTex: GPUTexture;
  private uniform: GPUBuffer;
  private sceneGroup: any;
  private outputGroup: any;

  static async create(width: number, height: number, slots: number): Promise<WebGpuRenderer> {
    const gpu = (navigator as Navigator & { gpu?: GPU }).gpu;
    const adapter = await gpu?.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice({ requiredLimits: { maxTextureDimension2D: Math.min(adapter.limits.maxTextureDimension2D, 8192) } });
    return new WebGpuRenderer(device, width, height, slots);
  }

  private constructor(private device: GPUDevice, private width: number, private height: number, slots: number) {
    const vs = `@vertex fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
      let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(p * 2.0 - 1.0, 0.0, 1.0); }`;
    const common = `struct U { t: f32, frameIndex: u32, w: f32, h: f32 }; @group(0) @binding(0) var<uniform> u: U;`;
    const scene = device.createShaderModule({ code: `${vs} ${common}
      fn hash(p: vec2f) -> f32 { return fract(sin(dot(p, vec2f(12.9898, 78.233))) * 43758.5453); }
      @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        let uv = pos.xy / vec2f(u.w, u.h);
        var c = vec3f(uv.x, uv.y, 0.5 + 0.5 * sin(u.t));
        for (var i = 0; i < 6; i++) {
          let fi = f32(i);
          let centre = vec2f(0.5 + 0.35 * sin(u.t * (0.7 + 0.1 * fi) + fi), 0.5 + 0.35 * cos(u.t * (0.9 + 0.13 * fi) + fi * 1.7));
          let d = length((uv - centre) * vec2f(1.7778, 1.0));
          c += vec3f(1.5, 0.8, 0.3) * exp(-d * d * 60.0) * (0.5 + 0.5 * sin(fi + u.t));
        }
        c += (hash(pos.xy + u.t) - 0.5) * 0.02;
        return vec4f(c, 1.0);
      }` });
    const output = device.createShaderModule({ code: `${vs} ${common}
      @group(0) @binding(1) var sceneTex: texture_2d<f32>;
      @fragment fn fs(@builtin(position) pos: vec4f) -> @location(0) vec4f {
        var c = textureLoad(sceneTex, vec2i(pos.xy), 0).rgb;
        c = c / (1.0 + c);
        c = pow(c, vec3f(1.0 / 2.2));
        if (pos.x < 256.0 && pos.y < 32.0) {
          let bit = u32(pos.x) / 32u;
          c = vec3f(f32((u.frameIndex >> bit) & 1u));
        }
        return vec4f(c, 1.0);
      }` });
    this.sceneTex = device.createTexture({ size: [width, height], format: "rgba16float", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING });
    this.outTex = device.createTexture({ size: [width, height], format: "rgba8unorm", usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    this.uniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.scenePipeline = device.createRenderPipeline({ layout: "auto", vertex: { module: scene, entryPoint: "vs" }, fragment: { module: scene, entryPoint: "fs", targets: [{ format: "rgba16float" }] } });
    this.outputPipeline = device.createRenderPipeline({ layout: "auto", vertex: { module: output, entryPoint: "vs" }, fragment: { module: output, entryPoint: "fs", targets: [{ format: "rgba8unorm" }] } });
    this.sceneGroup = device.createBindGroup({ layout: this.scenePipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }] });
    this.outputGroup = device.createBindGroup({ layout: this.outputPipeline.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: this.uniform } }, { binding: 1, resource: this.sceneTex.createView() }] });
    for (let i = 0; i < slots; i++) {
      this.buffers.push(device.createBuffer({ size: width * height * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }));
      this.pending.push(null);
    }
    this.cpu = new Uint8Array(width * height * 4);
  }

  submit(index: number, slot: number) {
    const device = this.device;
    const data = new ArrayBuffer(16);
    new Float32Array(data, 0, 1)[0] = index / 24;
    new Uint32Array(data, 4, 1)[0] = index;
    new Float32Array(data, 8, 2).set([this.width, this.height]);
    device.queue.writeBuffer(this.uniform, 0, data);
    const encoder = device.createCommandEncoder();
    const a = encoder.beginRenderPass({ colorAttachments: [{ view: this.sceneTex.createView(), loadOp: "clear", storeOp: "store" }] });
    a.setPipeline(this.scenePipeline); a.setBindGroup(0, this.sceneGroup); a.draw(3); a.end();
    const b = encoder.beginRenderPass({ colorAttachments: [{ view: this.outTex.createView(), loadOp: "clear", storeOp: "store" }] });
    b.setPipeline(this.outputPipeline); b.setBindGroup(0, this.outputGroup); b.draw(3); b.end();
    encoder.copyTextureToBuffer({ texture: this.outTex }, { buffer: this.buffers[slot], bytesPerRow: this.width * 4 }, [this.width, this.height]);
    device.queue.submit([encoder.finish()]);
    this.pending[slot] = this.buffers[slot].mapAsync(GPUMapMode.READ);
  }

  async collect(slot: number): Promise<Uint8Array> {
    await this.pending[slot];
    this.pending[slot] = null;
    const buffer = this.buffers[slot];
    this.cpu.set(new Uint8Array(buffer.getMappedRange()));
    buffer.unmap();
    return this.cpu;
  }

  dispose() { this.device.destroy(); }
}

const SLOTS = 3;

async function verifyOutput(backend: Backend, path: string, frames: number): Promise<{ ok: boolean; detail: string }> {
  const media = await demux(await backend.readFile(path));
  let config = decoderConfig(media, "prefer-hardware");
  if (!(await VideoDecoder.isConfigSupported(config)).supported) {
    config = decoderConfig(media, "prefer-software");
    if (!(await VideoDecoder.isConfigSupported(config)).supported) return { ok: false, detail: `cannot verify: this WebView cannot decode ${media.codec}` };
  }
  const gl = createGl();
  const tex = mutableTexture(gl);
  const strip = new Uint8Array(256 * 32 * 4);
  const seen: number[] = [];
  // Output order from a decoder is presentation order, which is what we want to check.
  await decodeAll(media, config, (frame) => {
    uploadBrowser(gl, tex, frame);
    const fb = framebuffer(gl, tex);
    gl.readPixels(0, 0, 256, 32, gl.RGBA, gl.UNSIGNED_BYTE, strip);
    gl.deleteFramebuffer(fb);
    seen.push(readIndexStrip(strip, 256));
  });
  gl.getExtension("WEBGL_lose_context")?.loseContext();
  const bad = seen.map((value, i) => [i, value]).filter(([i, value]) => value !== i % 256);
  const ok = seen.length === frames && bad.length === 0;
  return { ok, detail: `${seen.length}/${frames} frames decoded; ${bad.length} out of sequence${bad.length ? ` (first ${bad.slice(0, 3).map(([i, v]) => `${i}->${v}`).join(", ")})` : ""}` };
}

export type ReadbackSpec = { api: "webgl2" | "webgpu"; format: PixelFormat; transport: Transport; sink: Sink; width: number; height: number; frames: number };

async function outputColour(backend: Backend, path: string, reference: Uint8Array, width: number, height: number) {
  const media = await demux(await backend.readFile(path));
  let config = decoderConfig(media, "prefer-hardware");
  if (!(await VideoDecoder.isConfigSupported(config)).supported) config = decoderConfig(media, "prefer-software");
  if (!(await VideoDecoder.isConfigSupported(config)).supported) return { error: `this WebView cannot decode ${media.codec}` };
  const seeker = new Seeker(media, config);
  const gl = createGl();
  try {
    const frame = await seeker.frame(0);
    const tex = mutableTexture(gl);
    uploadBrowser(gl, tex, frame);
    frame.close();
    const pixels = new Uint8Array(width * height * 4);
    const fb = framebuffer(gl, tex);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
    gl.deleteFramebuffer(fb);
    const result = compareRgba(pixels, reference);
    return { mae: round(result.mae, 3), p99: result.p99, max: result.max };
  } finally {
    seeker.close();
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }
}

export async function runReadback(backend: Backend, spec: ReadbackSpec): Promise<ReadbackRun> {
  const { api, format, transport, sink, width, height, frames } = spec;
  const run: ReadbackRun = { api, format, transport, sink, width, height, frames, fps: 0, mbPerSecond: 0, readbackMs: 0, transportMs: 0, bytesPerFrame: 0 };
  let renderer: Renderer | undefined;
  let session: { id: string; encoder: string; outputPath: string } | undefined;
  let peakPrivate = 0;
  let sampler: ReturnType<typeof setInterval> | undefined;
  try {
    renderer = api === "webgl2" ? new WebGl2Renderer(width, height, SLOTS, format) : await WebGpuRenderer.create(width, height, SLOTS);
    run.bytesPerFrame = renderer.bytesPerFrame;
    const endpoint = transport === "http" ? await backend.httpEndpoint() : undefined;
    if (transport !== "none") {
      session = await backend.encodeStart(width, height, 24, sink === "null" ? "null" : "h264_nvenc", format === "yuv420" ? "yuv420p" : "rgba");
      run.encoder = session.encoder;
    }
    const send = (bytes: Uint8Array) => transport === "http" ? backend.encodeFrameHttp(endpoint!, session!.id, bytes) : backend.encodeFrame(session!.id, bytes);
    const before: AppUsage | null = await backend.appUsage();
    sampler = setInterval(() => { void backend.appUsage().then((usage) => { if (usage) peakPrivate = Math.max(peakPrivate, usage.privateBytes); }); }, 1000);
    let reference: Uint8Array | undefined;
    let readbackTotal = 0;
    let transportTotal = 0;
    const started = performance.now();
    // Keep SLOTS-1 frames in flight on the GPU while the oldest is collected and sent.
    for (let i = 0; i < frames + SLOTS - 1; i++) {
      if (i < frames) {
        renderer.submit(i, i % SLOTS);
        if (i === 0 && sink === "encode") reference = renderer.snapshotRgba();
      }
      const j = i - (SLOTS - 1);
      if (j < 0) continue;
      const t0 = performance.now();
      const bytes = await renderer.collect(j % SLOTS);
      const t1 = performance.now();
      readbackTotal += t1 - t0;
      if (session) {
        await send(bytes);
        transportTotal += performance.now() - t1;
      }
    }
    const elapsed = (performance.now() - started) / 1000;
    const after: AppUsage | null = await backend.appUsage();
    run.fps = round(frames / elapsed, 1);
    run.mbPerSecond = round((frames * run.bytesPerFrame) / elapsed / 1e6, 0);
    run.readbackMs = round(readbackTotal / frames, 2);
    run.transportMs = round(transportTotal / frames, 2);
    if (before && after) {
      run.cpuSecondsPerFrame = round((after.cpuSeconds - before.cpuSeconds) / frames, 4);
      peakPrivate = Math.max(peakPrivate, before.privateBytes, after.privateBytes);
      run.peakPrivateGb = round(peakPrivate / 2 ** 30, 2);
    }
    if (session) {
      const outputPath = session.outputPath;
      const stats = await backend.encodeFinish(session.id);
      session = undefined;
      run.rustWriteSeconds = round(stats.writeSeconds, 3);
      run.rustDrainSeconds = round(stats.drainSeconds, 3);
      run.outputFrames = stats.outputFrames;
      run.outputPixFmt = stats.outputPixFmt ?? null;
      if (sink === "encode") {
        if (stats.outputBytes > 0) {
          const check = await verifyOutput(backend, outputPath, frames).catch((error) => ({ ok: false, detail: String(error) }));
          run.outputOrderCorrect = check.ok;
          run.outputOrderDetail = check.detail;
          if (reference) run.outputColour = await outputColour(backend, outputPath, reference, width, height).catch((error) => ({ error: String(error) }));
        } else {
          run.error = `ffmpeg produced no output: ${stats.stderrTail}`;
        }
      }
    }
  } catch (error) {
    run.error = String(error);
  } finally {
    if (sampler) clearInterval(sampler);
    if (session) await backend.encodeFinish(session.id).catch(() => undefined);
    renderer?.dispose();
  }
  return run;
}

export async function runReadbackSpike(backend: Backend, log: (line: string) => void) {
  // ?readbackSizes=1080 limits the run (used by the container harness, where 4K is too slow).
  const only = new URLSearchParams(location.search).get("readbackSizes");
  const plan: Omit<ReadbackSpec, "api" | "width" | "height" | "frames">[][] = [
    // 1080p: baselines, each remedy alone, then together.
    [
      { format: "rgba", transport: "none", sink: "null" },
      { format: "yuv420", transport: "none", sink: "null" },
      { format: "rgba", transport: "http", sink: "null" },
      { format: "yuv420", transport: "http", sink: "null" },
      { format: "rgba", transport: "ipc", sink: "encode" },
      { format: "yuv420", transport: "ipc", sink: "encode" },
      { format: "rgba", transport: "http", sink: "encode" },
      { format: "yuv420", transport: "http", sink: "encode" },
    ],
    // 4K: the combinations that could reach a usable rate.
    [
      { format: "yuv420", transport: "none", sink: "null" },
      { format: "yuv420", transport: "ipc", sink: "encode" },
      { format: "rgba", transport: "http", sink: "encode" },
      { format: "yuv420", transport: "http", sink: "encode" },
    ],
  ];
  const sizes: [number, number, number][] = [[1920, 1080, 120], [3840, 2160, 48]];
  const specs: ReadbackSpec[] = [];
  sizes.forEach(([width, height, frames], index) => {
    if (only && !only.split(",").includes(String(height))) return;
    for (const item of plan[index]) specs.push({ ...item, api: "webgl2", width, height, frames });
  });
  const results: ReadbackRun[] = [];
  for (const spec of specs) {
    log(`readback: ${spec.width}x${spec.height} ${spec.format} via ${spec.transport} -> ${spec.sink}`);
    results.push(await runReadback(backend, spec));
  }
  const encodes1080 = results.filter((r) => r.sink === "encode" && r.width === 1920 && !r.error && r.outputOrderCorrect);
  const best = encodes1080.reduce<ReadbackRun | undefined>((a, b) => (!a || b.fps > a.fps ? b : a), undefined);
  return {
    round: 2,
    note: "WebGL2 only this round: round 1 showed both APIs hit the same IPC ceiling; the GPU side is not the limit.",
    runs: results,
    best1080Encode: best ? { fps: best.fps, format: best.format, transport: best.transport } : null,
    pass: !!best && best.fps >= 15,
    passBar: ">= 15 fps end to end at 1080p (render -> readback -> transport -> ffmpeg), output frame order verified",
  };
}
