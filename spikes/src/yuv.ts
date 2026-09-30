/**
 * Two ways to turn a decoded VideoFrame into RGBA on the GPU:
 * - `browser`: texImage2D(frame). The browser picks the YUV->RGB matrix from the frame's
 *   colorSpace (or guesses when the stream is untagged). This is the colour-bug risk.
 * - `explicit`: copy the Y/U/V planes out and convert in our own shader with BT.709 limited
 *   range, whatever the stream says. Deterministic, costs a CPU round trip per frame.
 */
import { compile, drawFullscreen, framebuffer, FULLSCREEN_VS, mutableTexture, texture } from "./gl";

export function uploadBrowser(gl: WebGL2RenderingContext, tex: WebGLTexture, source: TexImageSource) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.pixelStorei(gl.UNPACK_COLORSPACE_CONVERSION_WEBGL, gl.NONE);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, source);
}

const FS = `#version 300 es
precision highp float;
in vec2 uv;
uniform sampler2D yPlane;
uniform sampler2D uPlane;
uniform sampler2D vPlane;
uniform sampler2D uvPlane;
uniform bool nv12;
uniform vec2 lumaSize;
uniform vec2 chromaOffset; // in luma pixels: (0.5, 0) = MPEG-2 / H.264 left siting
out vec4 color;
void main() {
  float y = texture(yPlane, uv).r;
  vec2 cuv = uv + chromaOffset / lumaSize;
  vec2 c = nv12 ? texture(uvPlane, cuv).rg : vec2(texture(uPlane, cuv).r, texture(vPlane, cuv).r);
  // BT.709, limited range (16-235 / 16-240) to full-range RGB.
  float Y = (y - 16.0 / 255.0) * (255.0 / 219.0);
  float U = (c.x - 128.0 / 255.0) * (255.0 / 224.0);
  float V = (c.y - 128.0 / 255.0) * (255.0 / 224.0);
  color = vec4(Y + 1.5748 * V, Y - 0.18733 * U - 0.46812 * V, Y + 1.8556 * U, 1.0);
}`;

export class ExplicitConverter {
  private program: WebGLProgram;
  private planes = { y: null as WebGLTexture | null, u: null as WebGLTexture | null, v: null as WebGLTexture | null, uv: null as WebGLTexture | null };
  private buffer = new Uint8Array(0);
  target: WebGLTexture | null = null;
  private fb: WebGLFramebuffer | null = null;
  private size = [0, 0];

  constructor(private gl: WebGL2RenderingContext) {
    this.program = compile(gl, FULLSCREEN_VS, FS);
  }

  /** Converts `frame` into `this.target` (RGBA8, row 0 = top row). Returns the frame format. */
  async convert(frame: VideoFrame, chromaOffset: [number, number]): Promise<string> {
    const gl = this.gl;
    const format = frame.format;
    if (format !== "I420" && format !== "NV12") throw new Error(`explicit path supports I420/NV12, got ${format}`);
    const width = frame.displayWidth;
    const height = frame.displayHeight;
    const size = frame.allocationSize();
    if (this.buffer.length < size) this.buffer = new Uint8Array(size);
    const layout = await frame.copyTo(this.buffer);
    if (this.size[0] !== width || this.size[1] !== height) {
      this.target = texture(gl, gl.RGBA8, width, height);
      this.fb = framebuffer(gl, this.target);
      this.size = [width, height];
    }
    const cw = Math.ceil(width / 2);
    const ch = Math.ceil(height / 2);
    const upload = (key: "y" | "u" | "v" | "uv", plane: PlaneLayout, w: number, h: number, channels: 1 | 2) => {
      this.planes[key] ??= mutableTexture(gl, gl.LINEAR);
      gl.bindTexture(gl.TEXTURE_2D, this.planes[key]);
      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.pixelStorei(gl.UNPACK_ROW_LENGTH, plane.stride / channels);
      const internal = channels === 1 ? gl.R8 : gl.RG8;
      const fmt = channels === 1 ? gl.RED : gl.RG;
      gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, fmt, gl.UNSIGNED_BYTE, this.buffer, plane.offset);
      gl.pixelStorei(gl.UNPACK_ROW_LENGTH, 0);
    };
    upload("y", layout[0], width, height, 1);
    if (format === "NV12") upload("uv", layout[1], cw, ch, 2);
    else { upload("u", layout[1], cw, ch, 1); upload("v", layout[2], cw, ch, 1); }

    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
    gl.viewport(0, 0, width, height);
    gl.useProgram(this.program);
    const bind = (unit: number, name: string, tex: WebGLTexture | null) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(gl.getUniformLocation(this.program, name), unit);
    };
    bind(0, "yPlane", this.planes.y);
    bind(1, "uPlane", this.planes.u ?? this.planes.y);
    bind(2, "vPlane", this.planes.v ?? this.planes.y);
    bind(3, "uvPlane", this.planes.uv ?? this.planes.y);
    gl.activeTexture(gl.TEXTURE0);
    gl.uniform1i(gl.getUniformLocation(this.program, "nv12"), format === "NV12" ? 1 : 0);
    gl.uniform2f(gl.getUniformLocation(this.program, "lumaSize"), width, height);
    gl.uniform2f(gl.getUniformLocation(this.program, "chromaOffset"), chromaOffset[0], chromaOffset[1]);
    drawFullscreen(gl);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return format;
  }

  read(out: Uint8Array): Uint8Array {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fb);
    gl.readPixels(0, 0, this.size[0], this.size[1], gl.RGBA, gl.UNSIGNED_BYTE, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    return out;
  }
}
