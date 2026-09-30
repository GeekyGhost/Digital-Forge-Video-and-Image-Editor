/**
 * Thin WebGL2 plumbing for the compositor: programs, 8-bit render targets, uploads and readback.
 *
 * Every intermediate is an RGBA8 texture holding STRAIGHT (not premultiplied) alpha, row 0 at the
 * top. That is deliberate for Phase 1: the photo compositor this replaces keeps every stage in a
 * `Uint8ClampedArray`, and quantising at the same points is what makes pixel parity reachable. Float
 * intermediates are a later, separate decision (they change the output, so they need their own
 * sign-off).
 *
 * Nothing is flipped anywhere. Uploads put image row 0 at texel row 0, shaders address texels with
 * `texelFetch(ivec2(gl_FragCoord.xy))`, and `readPixels` returns row 0 first -- so buffers go in and
 * out in the same order a `Uint8ClampedArray` uses, with no Y convention to get wrong.
 */

export type Target = { texture: WebGLTexture; framebuffer: WebGLFramebuffer; width: number; height: number };

export class GlContext {
  readonly gl: WebGL2RenderingContext;
  readonly canvas: HTMLCanvasElement | OffscreenCanvas;
  private readonly vao: WebGLVertexArrayObject;
  private instanceVao?: WebGLVertexArrayObject;
  private instanceBuffer?: WebGLBuffer;
  private readonly programs = new Map<string, WebGLProgram>();
  private readonly pending = new Map<string, { program: WebGLProgram; fragment: WebGLShader }>();
  private readonly uniforms = new Map<WebGLProgram, Map<string, WebGLUniformLocation | null>>();
  /** Free render targets by size, reused between frames (a slider drag allocates nothing). */
  private readonly pool = new Map<string, Target[]>();
  private readonly pooled = new Set<Target>();
  readonly floatTargets: boolean;

  constructor(canvas?: HTMLCanvasElement | OffscreenCanvas) {
    this.canvas = canvas ?? (typeof OffscreenCanvas !== "undefined"
      ? new OffscreenCanvas(1, 1) : window.document.createElement("canvas"));
    const gl = this.canvas.getContext("webgl2", {
      alpha: true, antialias: false, depth: false, stencil: false,
      premultipliedAlpha: false, preserveDrawingBuffer: false, powerPreference: "high-performance",
    }) as WebGL2RenderingContext | null;
    if (!gl) throw new Error("WebGL2 is not available");
    this.gl = gl;
    this.floatTargets = Boolean(gl.getExtension("EXT_color_buffer_float"));
    gl.getExtension("KHR_parallel_shader_compile");
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.PACK_ALIGNMENT, 1);
    gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
    gl.disable(gl.BLEND);
    gl.disable(gl.DITHER);
    this.vao = gl.createVertexArray()!;
  }

  get lost(): boolean { return this.gl.isContextLost(); }

  /**
   * Start compiling and linking a program without waiting for it.
   *
   * Drivers compile in the background when nothing asks for the result (KHR_parallel_shader_compile
   * makes that explicit), so preparing every program when the renderer is made moves the 15-30 ms
   * first-use stall on the RTX 4090 out of the first frame that needs each pass.
   */
  prepare(name: string, fragment: string, vertexSource: string = FULLSCREEN_VS): void {
    if (this.programs.has(name) || this.pending.has(name)) return;
    const gl = this.gl;
    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type)!;
      gl.shaderSource(shader, source);
      gl.compileShader(shader);
      return shader;
    };
    const vertex = compile(gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = compile(gl.FRAGMENT_SHADER, fragment);
    const program = gl.createProgram()!;
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    this.pending.set(name, { program, fragment: fragmentShader });
  }

  program(name: string, fragment: string, vertexSource: string = FULLSCREEN_VS): WebGLProgram {
    const held = this.programs.get(name);
    if (held) return held;
    this.prepare(name, fragment, vertexSource);
    const gl = this.gl;
    const { program, fragment: shader } = this.pending.get(name)!;
    this.pending.delete(name);
    // Status is only read here, on first use, so a prepared program is never waited for early.
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getShaderInfoLog(shader) || gl.getProgramInfoLog(program);
      throw new Error(`${name}: ${log}`);
    }
    this.programs.set(name, program);
    this.uniforms.set(program, new Map());
    return program;
  }

  private location(program: WebGLProgram, name: string): WebGLUniformLocation | null {
    const cache = this.uniforms.get(program)!;
    if (!cache.has(name)) cache.set(name, this.gl.getUniformLocation(program, name));
    return cache.get(name)!;
  }

  /**
   * Run `fragment` once per pixel of `target`.
   *
   * `textures` are bound to units in order and their sampler uniforms set by name. Numbers become
   * float uniforms, `{ int }` int uniforms, arrays vecN (by length) or float arrays when longer than 4.
   */
  draw(
    name: string, fragment: string, target: Target,
    textures: Record<string, WebGLTexture>, uniforms: Record<string, UniformValue> = {},
    options: { blend?: "max" | "min" } = {},
  ): void {
    const gl = this.gl;
    const program = this.program(name, fragment);
    this.bindProgram(program, textures, uniforms);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
    gl.viewport(0, 0, target.width, target.height);
    gl.bindVertexArray(this.vao);
    this.withBlend(options.blend, () => gl.drawArrays(gl.TRIANGLES, 0, 3));
  }

  /**
   * Instanced quads (VFX Phase 2: brush dabs). `instances` holds six floats per instance, bound to
   * attribute 0 (vec4) and 1 (vec2) of `vertexSource`; the quad is a 4-vertex strip.
   */
  drawInstances(
    name: string, vertexSource: string, fragment: string, target: Target,
    uniforms: Record<string, UniformValue>, instances: Float32Array, blend: "max" | "min",
  ): void {
    const count = instances.length / 6;
    if (!count) return;
    const gl = this.gl;
    const program = this.program(name, fragment, vertexSource);
    this.bindProgram(program, {}, uniforms);
    if (!this.instanceVao) {
      this.instanceVao = gl.createVertexArray()!;
      this.instanceBuffer = gl.createBuffer()!;
      gl.bindVertexArray(this.instanceVao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 4, gl.FLOAT, false, 24, 0);
      gl.vertexAttribDivisor(0, 1);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 2, gl.FLOAT, false, 24, 16);
      gl.vertexAttribDivisor(1, 1);
    }
    gl.bindVertexArray(this.instanceVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.instanceBuffer!);
    gl.bufferData(gl.ARRAY_BUFFER, instances, gl.STREAM_DRAW);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
    gl.viewport(0, 0, target.width, target.height);
    this.withBlend(blend, () => gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, count));
    gl.bindVertexArray(this.vao);
  }

  private withBlend(blend: "max" | "min" | undefined, run: () => void): void {
    const gl = this.gl;
    if (!blend) { run(); return; }
    gl.enable(gl.BLEND);
    gl.blendEquation(blend === "max" ? gl.MAX : gl.MIN);
    gl.blendFunc(gl.ONE, gl.ONE);
    run();
    gl.disable(gl.BLEND);
    gl.blendEquation(gl.FUNC_ADD);
  }

  private bindProgram(program: WebGLProgram, textures: Record<string, WebGLTexture>, uniforms: Record<string, UniformValue>): void {
    const gl = this.gl;
    gl.useProgram(program);
    let unit = 0;
    for (const [uniform, texture] of Object.entries(textures)) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.uniform1i(this.location(program, uniform), unit);
      unit++;
    }
    for (const [uniform, value] of Object.entries(uniforms)) {
      const at = this.location(program, uniform);
      if (at === null) continue;
      if (typeof value === "number") gl.uniform1f(at, value);
      else if ("int" in value) gl.uniform1i(at, value.int);
      else if ("ints" in value) {
        const ints = Int32Array.from(value.ints);
        if (ints.length === 2) gl.uniform2iv(at, ints);
        else if (ints.length === 3) gl.uniform3iv(at, ints);
        else if (ints.length === 4) gl.uniform4iv(at, ints);
        else gl.uniform1iv(at, ints);
      }
      else if (value.length === 2) gl.uniform2fv(at, value);
      else if (value.length === 3) gl.uniform3fv(at, value);
      else if (value.length === 4) gl.uniform4fv(at, value);
      else gl.uniform1fv(at, value);
    }
  }

  /** An RGBA8 (or RGBA32F) texture with a framebuffer, from the pool when one of that size is free. */
  target(width: number, height: number, format: "rgba8" | "rgba32f" = "rgba8"): Target {
    const key = `${format}:${width}x${height}`;
    const free = this.pool.get(key);
    if (free?.length) {
      const reused = free.pop()!;
      this.pooled.delete(reused);
      return reused;
    }
    if (format === "rgba32f" && !this.floatTargets) throw new Error("float render targets are not supported");
    const gl = this.gl;
    const texture = this.texture();
    if (format === "rgba32f") gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, width, height, 0, gl.RGBA, gl.FLOAT, null);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
    const framebuffer = gl.createFramebuffer()!;
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (status !== gl.FRAMEBUFFER_COMPLETE) throw new Error(`framebuffer incomplete (${format}): 0x${status.toString(16)}`);
    const made: Target = { texture, framebuffer, width, height };
    (made as Target & { key: string }).key = key;
    return made;
  }

  release(target: Target | undefined): void {
    if (!target) return;
    // Releasing twice would hand one texture to two stages at once, which corrupts silently.
    if (this.pooled.has(target)) throw new Error("render target released twice");
    this.pooled.add(target);
    const key = (target as Target & { key: string }).key;
    const list = this.pool.get(key) ?? [];
    list.push(target);
    this.pool.set(key, list);
  }

  /**
   * Free pooled targets of every size but this one.
   *
   * Called at the start of each render. Preview frames all share one size; a full-size save or
   * capture allocates big targets once, and without this they would sit in the pool -- hundreds of
   * megabytes beside whatever model is loaded -- until Photo Edit closed.
   */
  trimPool(width: number, height: number): void {
    const keep = `:${width}x${height}`;
    for (const [key, list] of this.pool) {
      if (key.endsWith(keep)) continue;
      for (const target of list) {
        this.gl.deleteFramebuffer(target.framebuffer);
        this.gl.deleteTexture(target.texture);
        this.pooled.delete(target);
      }
      this.pool.delete(key);
    }
  }

  /** A texture with nearest filtering and clamped edges: every shader here uses `texelFetch`. */
  texture(): WebGLTexture {
    const gl = this.gl;
    const texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return texture;
  }

  uploadRgba(pixels: Uint8Array | Uint8ClampedArray, width: number, height: number, into?: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const texture = into ?? this.texture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE,
      pixels instanceof Uint8Array ? pixels : new Uint8Array(pixels.buffer, pixels.byteOffset, pixels.byteLength));
    return texture;
  }

  /** Into an existing target's texture, so it can be drawn over or read like any other stage. */
  uploadInto(target: Target, pixels: Uint8ClampedArray): void {
    this.uploadRgba(pixels, target.width, target.height, target.texture);
  }

  uploadR8(values: Uint8Array, width: number, height: number, into?: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const texture = into ?? this.texture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, values);
    return texture;
  }

  uploadR32F(values: Float32Array, width: number, height: number, into?: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const texture = into ?? this.texture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, width, height, 0, gl.RED, gl.FLOAT, values);
    return texture;
  }

  /**
   * An image as PREMULTIPLIED RGBA8, the way a 2D canvas stores it.
   *
   * Premultiplied because the canvas `drawImage` this replaces resamples premultiplied values; a
   * straight-alpha texture filtered the same way bleeds the colour of transparent pixels into edges.
   */
  uploadImagePremultiplied(image: TexImageSource, into?: WebGLTexture): WebGLTexture {
    const gl = this.gl;
    const texture = into ?? this.texture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, image);
    gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
    return texture;
  }

  read(target: Target, into?: Uint8ClampedArray): Uint8ClampedArray {
    const gl = this.gl;
    const out = into ?? new Uint8ClampedArray(target.width * target.height * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
    gl.readPixels(0, 0, target.width, target.height, gl.RGBA, gl.UNSIGNED_BYTE,
      new Uint8Array(out.buffer, out.byteOffset, out.byteLength));
    return out;
  }

  get maxTextureSize(): number { return this.gl.getParameter(this.gl.MAX_TEXTURE_SIZE) as number; }

  /** Throws if any GL call since the last check failed (an out-of-memory allocation is silent otherwise). */
  check(where: string): void {
    const error = this.gl.getError();
    if (error !== this.gl.NO_ERROR) throw new Error(`WebGL error 0x${error.toString(16)} ${where}`);
  }

  /** Frees every pooled target and the context itself, so the GPU memory is returned now, not at GC. */
  destroy(): void {
    const gl = this.gl;
    for (const list of this.pool.values()) {
      for (const target of list) {
        gl.deleteFramebuffer(target.framebuffer);
        gl.deleteTexture(target.texture);
      }
    }
    this.pool.clear();
    this.pooled.clear();
    if (this.instanceBuffer) gl.deleteBuffer(this.instanceBuffer);
    if (this.instanceVao) gl.deleteVertexArray(this.instanceVao);
    for (const program of this.programs.values()) gl.deleteProgram(program);
    for (const { program } of this.pending.values()) gl.deleteProgram(program);
    this.programs.clear();
    this.pending.clear();
    gl.getExtension("WEBGL_lose_context")?.loseContext();
  }

  deleteTexture(texture: WebGLTexture | undefined): void {
    if (texture) this.gl.deleteTexture(texture);
  }
}

export type UniformValue = number | number[] | Float32Array | { int: number } | { ints: Int32Array | number[] };

/** One triangle that covers the viewport; no vertex buffer needed. */
const FULLSCREEN_VS = `#version 300 es
void main() {
  vec2 corner = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}`;
