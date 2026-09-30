# VFX Phase 0 spikes (throwaway)

Measures, inside the app's own WebView2, the four assumptions the single-GPU-compositor plan rests
on.

| Spike | File | What it does |
|---|---|---|
| 0.2 GPU | `src/gpu.ts` | WebGL2 and WebGPU: adapter, limits, 3840x2160 RGBA16F target, fixed-function "over" plus a shader multiply, HDR retention, checked against arithmetic |
| 0.3 Decode | `src/decode.ts`, `src/demux.ts`, `src/yuv.ts` | mp4box.js + WebCodecs: decode fps (HW and SW), decode + upload fps, frame-exact seek verified by an index strip, colour vs ffmpeg with explicit BT.709 through the browser's conversion and through our own shader |
| 0.4 Readback | `src/readback.ts` | Render offscreen, read back (WebGL2 PBO + fence, WebGPU mapAsync), send as a raw IPC body, Rust pipes to `ffmpeg -f rawvideo` -> NVENC. Modes readback / ipc / encode separate the costs; the output is decoded again and its frame order checked |
| 0.5 Alpha | `src/alpha.ts` | RGBA PNG sequence via createImageBitmap, stacked colour+alpha H.264 via WebCodecs, VP9-alpha WebM via `<video>`; alpha and colour error vs ffmpeg's PNG decode |

Test media is generated from `media-recipe.json` (shared by the Rust side and the harness) into
`scratch/vfx-spikes/media/`. Every clip carries its frame index in the top-left 256x32 pixels.

## Running it (Windows)

1. `Build-AIV-Studio.bat` (the exe must contain the spike commands).
2. `Run-VFX-Spikes.bat` - starts the newest `Releases\aiv-studio-*.exe` with `AIV_SPIKES=1`, which
   opens a second window, "VFX Phase 0 spikes".
3. Press **Run all**. Results appear on the right and are saved to
   `scratch\vfx-spikes\results-<time>.json`.

## Running it in the container (correctness only)

```
npx vite build
SPIKE_NO_H264=1 node spikes/harness/server.mjs dist /tmp/spikework 4173 &
SPIKE_QUERY=readbackSizes=1080 node spikes/harness/run.mjs http://localhost:4173/spikes/index.html
```
Headless Chromium has no H.264/HEVC decoder and renders through SwiftShader, so it checks logic,
not performance. `SPIKE_NO_H264=1` swaps the stacked proxy and readback output to VP9 so their
checks can run there.

## Compositor parity (VFX Phases 1-2)

The **Compositor parity** button (called "Phase 1 parity" until VFX Phase 2) runs
`src/render/parity.ts`: every fixture document rendered by the old CPU compositor and by the GPU
renderer (`src/render/`), compared pixel by pixel, plus -- since Phase 2 -- the combined coverage of
each fixture's mask, a placement accuracy check and two 1500 px slider-drag timings (plain, and
through a masked grade on a track-matted layer). In the container:

```
node spikes/harness/run.mjs http://localhost:4173/spikes/index.html parity
SPIKE_QUERY="filter=track matte" node spikes/harness/run.mjs http://localhost:4173/spikes/index.html parity
```

`?filter=` runs only the fixtures whose name contains the text, and skips the timing.

`src/render/parity.ts` is not spike code and stays when this folder goes; move the button.

## Removing it

Delete `spikes/`, `Run-VFX-Spikes.bat`, the `spikes` input in `vite.config.ts`, "spikes" in
`src-tauri/capabilities/default.json`, and in `src-tauri/src/lib.rs` the `mod spikes` line, the
`.manage(spikes::Sessions::default())` and `.setup(...)` lines and the `spikes::` handler entries.
`scratch/vfx-spikes/` holds only generated output.

mp4box.js 2.4.1 is vendored in `vendor/mp4box/` (BSD-3-Clause, licence included).
