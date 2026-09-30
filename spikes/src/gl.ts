/** Small WebGL2 helpers shared by the spikes. Nothing here is meant to survive into Phase 1. */

export function createGl(width = 16, height = 16): WebGL2RenderingContext {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const gl = canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true, preserveDrawingBuffer: false, alpha: true, powerPreference: "high-performance" });
  if (!gl) throw new Error("WebGL2 is not available");
  return gl;
}

export function compile(gl: WebGL2RenderingContext, vertex: string, fragment: string): WebGLProgram {
  const make = (type: number, source: string) => {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(`shader: ${gl.getShaderInfoLog(shader)}`);
    return shader;
  };
  const program = gl.createProgram()!;
  gl.attachShader(program, make(gl.VERTEX_SHADER, vertex));
  gl.attachShader(program, make(gl.FRAGMENT_SHADER, fragment));
  gl.linkProgram(program);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(`link: ${gl.getProgramInfoLog(program)}`);
  return program;
}

/** Full-screen triangle; `uv` has (0,0) at the first texel row / first framebuffer row. */
export const FULLSCREEN_VS = `#version 300 es
out vec2 uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

export function drawFullscreen(gl: WebGL2RenderingContext) {
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

export function texture(gl: WebGL2RenderingContext, internal: number, width: number, height: number, filter: number = gl.NEAREST): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internal, width, height);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

/** A texture whose storage is re-specified on every upload (for VideoFrame / ImageBitmap). */
export function mutableTexture(gl: WebGL2RenderingContext, filter: number = gl.NEAREST): WebGLTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}

export function framebuffer(gl: WebGL2RenderingContext, tex: WebGLTexture): WebGLFramebuffer {
  const fb = gl.createFramebuffer()!;
  gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`framebuffer incomplete: 0x${status.toString(16)}`);
  return fb;
}

/** Reads an RGBA8 texture back, rows in texture order (row 0 = first uploaded image row). */
export function readTextureRgba8(gl: WebGL2RenderingContext, tex: WebGLTexture, width: number, height: number, out?: Uint8Array): Uint8Array {
  const fb = framebuffer(gl, tex);
  const pixels = out ?? new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.deleteFramebuffer(fb);
  return pixels;
}

export function rendererInfo(gl: WebGL2RenderingContext): { vendor: string; renderer: string } {
  const debug = gl.getExtension("WEBGL_debug_renderer_info");
  return {
    vendor: String(debug ? gl.getParameter(debug.UNMASKED_VENDOR_WEBGL) : gl.getParameter(gl.VENDOR)),
    renderer: String(debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER)),
  };
}

/** Waits for the GPU without blocking the main thread. */
export async function gpuFence(gl: WebGL2RenderingContext): Promise<void> {
  const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0)!;
  gl.flush();
  for (;;) {
    const state = gl.clientWaitSync(sync, 0, 0);
    if (state === gl.ALREADY_SIGNALED || state === gl.CONDITION_SATISFIED) break;
    if (state === gl.WAIT_FAILED) throw new Error("clientWaitSync failed");
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  gl.deleteSync(sync);
}

/** Mean and 99th-percentile absolute error over RGB (alpha ignored unless `channels` says so). */
export function compareRgba(a: Uint8Array, b: Uint8Array, channels: number[] = [0, 1, 2]): { mae: number; p99: number; max: number } {
  if (a.length !== b.length) throw new Error(`size mismatch ${a.length} vs ${b.length}`);
  const histogram = new Uint32Array(256);
  let sum = 0;
  let count = 0;
  for (let i = 0; i < a.length; i += 4) {
    for (const c of channels) {
      const d = Math.abs(a[i + c] - b[i + c]);
      histogram[d]++;
      sum += d;
      count++;
    }
  }
  let max = 0;
  for (let d = 255; d >= 0; d--) if (histogram[d]) { max = d; break; }
  let seen = 0;
  let p99 = 0;
  for (let d = 0; d < 256; d++) { seen += histogram[d]; if (seen >= count * 0.99) { p99 = d; break; } }
  return { mae: sum / count, p99, max };
}

/** Reads the 8-bit frame index strip (8 blocks of 32 px, LSB first) from an RGBA image. */
export function readIndexStrip(rgba: Uint8Array, width: number): number {
  let value = 0;
  for (let bit = 0; bit < 8; bit++) {
    const x = bit * 32 + 16;
    const y = 16;
    const i = (y * width + x) * 4;
    const luma = (rgba[i] + rgba[i + 1] + rgba[i + 2]) / 3;
    if (luma > 127) value |= 1 << bit;
  }
  return value;
}

export const now = () => performance.now();

export function round(value: number, digits = 2): number {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}
