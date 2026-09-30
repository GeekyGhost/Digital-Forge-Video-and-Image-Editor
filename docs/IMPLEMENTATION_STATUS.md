# Implementation status

## Working now

### Editor core

- Digital Forge theme and branding across the shell, timeline, inspector, and dialogs.
- React editor shell with a professional dark NLE workspace layout.
- Media bin with real ffmpeg-generated thumbnails, search, drag-and-drop, and a context menu.
- Multi-track timeline (V2 graphics, V1 picture, A1 audio, C1 captions) with exact rational project time.
- Selection, blade, and track-select tools with keyboard shortcuts (V / B / A).
- Clip drag, trim, split, duplicate, ripple delete, group move, undo/redo, and track lock/mute/visibility.
- Timeline filmstrip thumbnails on picture clips and decoded audio waveforms on clips with sound.
- Toggleable edit-point/playhead snapping.

### Transitions

- 58 qualified native transitions, each with an **incoming** and an **outgoing** side per clip.
- At a cut the two sides fuse: the incoming clip's choice sets the look, and the longer of the two
  durations wins so neither side is truncated. The inspector reports the resolved head and tail cut.
- The first clip's incoming transition renders as a fade **from** black (or white/gray for dip
  transitions); the last clip's outgoing transition renders as a fade **to** that colour.
- Timeline clips draw gold ramps on the edges that carry a transition.
- The program monitor composites the transition live with a second video layer, so dissolves,
  wipes, pushes, covers, reveals, iris/shape, clock, slice/wind, squeeze and zoom all preview
  the same edit the renderer will produce.
- Export builds the matching `xfade` graph with `acrossfade` audio, plus `fade`/`afade` at the
  head and tail of the edit.

### AI

- **AI clip editor** (Kiwi-Edit 5B): a dedicated workspace for rewriting a range of a timeline clip
  from a plain-language instruction, optionally guided by a reference image. Scrub the clip's own
  source, set the range, run locally, then replace the clip's media in place or insert the result as
  a new clip. Ctrl+K, the Clip menu, the clip context menu, and the inspector all open it.
  The adapter targets the pipeline that ships in the pack (`KiwiEditPipeline`), honouring its
  32-pixel and 4n+1 frame constraints and its 832x480 / 81-frame budget for a 16 GB card.
- **Mask Effects clip-editor mode** (`mask_effects.py`): the same selected-range workflow can use
  BiRefNet for automatic soft foreground mattes or SAM 2 include/exclude points for temporal object
  tracking. Both feed one shared compositor with glow, outline, aura, spotlight, background blur or
  pixelation, silhouette, hologram and heat-vision looks. Strength can follow overall loudness,
  bass, or transients; the rendered range keeps source audio and remains reviewable before apply.
- **Text Motion clip-editor mode**: a contextual creator for editable G1/C1 graphics rather than a
  burn-in tool. Type In/Out, fade, slide, pop/shrink, magic, glitch, pulse, float and flicker preview
  against the selected clip, persist on the graphic, remain adjustable in the Inspector, and render
  through the ordinary NLE export graph.
- **LTX generation, six modes**, all on the one-stage distilled path that fits 16 GB:
  **first + last frame** (the original hand-tuned single pass, unchanged), **image → video**,
  **text → video**, **clip extend**, **clip to clip**, and **video → audio**.
- **Clip to clip** (`bridge_clips`): generates the missing motion between two adjacent V1 clips.
  It extracts the outgoing clip's last visible frame and the incoming clip's first visible frame as
  lossless PNGs natively — respecting both clips' trims, so it uses the frames you actually see, not
  the untrimmed source — then runs the ordinary first/last-frame pass with those as conditioning and
  ripple-inserts the bridge between the pair. Both boundary frames are duplicates of frames already
  on the timeline, so the head and tail frame plus their matching audio are trimmed off before
  insertion, exactly as clip extend does at one end.
- **Video-conditioned foley** (`ltx_foley_v2a.py`): a purpose-built stage-1-only pipeline. Upstream
  `DubItPipeline` runs two stages, but its returned audio is decoded from the **stage 1** latent —
  stage 2 marks the audio `frozen=True` and discards its output, existing only to upscale the
  picture. Since the source picture is kept untouched, stage 2 and the spatial upsampler are dead
  weight, so this adapter runs stage 1, decodes the audio, and muxes it onto the original video.
  That removes the upsampler from the VRAM budget entirely. Needs the Foley V2A IC-LoRA, which
  carries the video-conditioning behaviour. The adapter supports the scaled-FP8 checkpoint through
  the project's companion-scale-aware LoRA fusion path. A silent source gets a silent track added
  so the reference-audio encoder has something to read.
- **Classifier-free guidance** is exposed for every LTX mode, separately for video and audio.
  The distilled schedule runs unguided at 1.0; audio only starts responding to the prompt around
  **1.5**, which is the foley default. Above 1.0 the pipeline switches from `SimpleDenoiser` to
  `GuidedDenoiser` with a `MultiModalGuider` per modality, and the negative prompt is encoded to
  supply the unconditional branch — so it costs roughly double the time and VRAM per step. At 1.0
  that second encode is skipped entirely.
- **Foley shaping**: `reference hold` scales how hard the generated audio is pinned to the source
  footage, and `LoRA strength` scales the adapter itself. Raise the first if the sound is plausible
  but ignores the action; lower the second if it sounds over-processed.
- **Automatic OOM recovery** (`aiv_runtime/oom.py`) wraps every LTX run. On a CUDA OOM the ladder
  retries with disk weight streaming first (costs speed, not quality), then with guidance disabled
  (halves per-step work but changes the result, so it is announced). Rungs already in effect are
  skipped so one OOM never burns a useful option, and non-OOM errors propagate untouched rather
  than being masked by pointless degradation. The clip editor keeps its own block-swap fallback.
- **SAM 2 subject tracking** (`sam2_track.py`): arm point picking, click the subject in the program
  monitor, and SAM 2 propagates the selection across the clip's trimmed range. Shift-click adds an
  exclude point. Three deliverables, because they serve different jobs: `matte` (black-and-white
  mask for keying), `cutout` (subject on black) and `removed` (subject blacked out, for clean-plate
  work), with optional edge feathering. Points are stored normalised so the worker scales them to
  the source resolution, and they clear when the selected clip changes.
- LTX Retake for regenerating a source interval in place.
- Whisper large-v3-turbo auto-captioning: transcribes the selected clip and places timed cues on C1
  in timeline time, respecting the clip's trim and speed.
- Frozen-frame + silence pause detection with ripple-closing removal.
- **MiniMax-H3 Director workspace**: structured mixed-media references, cast/subject roles, shot
  timestamps, exact dialogue markers, delivery and overlap direction, output controls, and a raw
  prompt escape hatch. The complete session and take ledger persist in `.aivproj`; each generated
  take stores its exact recipe and resolved seed and can restore that setup later.
- **H3 production preflight** reports all blockers and advisories in one pass: runtime/model state,
  missing or excessive references, timed-media limits, invalid canvas/output choices, shot timestamp
  order and range, estimated dialogue overflow, and high-risk VRAM configurations.

### Native and runtime

- Rust commands for FFprobe, thumbnails, waveforms, atomic project save/load, subtitle import,
  native AI job registration, and FFmpeg export.
- Python JSON-lines worker protocol with a hard 14 GiB lease cap.
- License-aware, free-space-aware model pack installer.
- LTX 2.3 distilled, utility, and Kiwi-Edit pack manifests.

## Known limitations

- **Legibility**: the type scale was raised across the whole interface (smallest text 7px → 9px,
  body copy 8-9px → 11-12px) and the controls, panel widths and row heights grew to match. Labels
  carrying an explanation are marked with a dotted underline and reveal it on hover.
- SAM 2 lives in the isolated Image Studio environment with BiRefNet and the image pipelines. Run
  `scripts/setup-image-studio.ps1`; do not install it into the application or LTX environments.
- **Audio → video is not built.** `A2VidPipelineTwoStage` genuinely needs its second stage (unlike
  DubIt, where the audio came from stage 1), so it cannot reuse the foley trick and needs its own
  VRAM profile with the spatial upsampler resident.
- **Blend modes** composite correctly in the program monitor (including across a transition) but the
  export path is a linear segment concatenation, so it renders every clip as Normal. Making them
  render requires replacing the concat compiler with an overlay-based timeline compositor.
- The clip editor reports precisely what is blocking it — `Pack missing`, `Dependencies missing`,
  `Adapter missing` or `Python missing` — because each needs a different fix. The pack's Python
  libraries (diffusers, transformers, einops, torchvision) are part of the `inference` extra.
- Clip edits longer than one 81-frame window are split into consecutive windows and concatenated,
  so the output matches the source range. Up to three frames (0.125 s) can be lost to the 4n+1
  constraint. The editor shows the pass count and effective output length before you run it.
- The Foley V2A LoRA is a **gated** Hugging Face repository, so it must be fetched with an accepted
  licence and `huggingface-cli login`, or placed by hand. The app searches `models/loras/ltx`,
  `models/loras` and `models/LTX-2.3/loras`, so a manually downloaded file is found wherever it
  reasonably lands. Until it is present the video → audio mode is disabled and says so, rather than
  silently producing unsynchronised audio.
- Foley prefers the scaled-FP8 checkpoint and uses a dedicated fusion shim that preserves and
  writes back each FP8 companion scale. The BF16 checkpoint remains a fallback, not a requirement.
- `KeyframeInterpolationPipeline` remains unwired: it needs the distilled 384 LoRA (now included in
  the `ltx-loras` pack) *and* genuinely uses its second stage, so it does not get the same
  stage-1-only treatment. The existing hand-tuned first/last path covers this ground at 16 GB.
- The bundled Kiwi-Edit pipeline still exposes no mask parameter. Mask Effects is therefore a
  separate tracked-matte compositor: it applies deterministic VFX to the selected subject rather
  than pretending Kiwi can perform prompt-guided masked regeneration.
- The Kiwi-Edit pack vendors its own copy of the Qwen2.5-VL modeling code from the transformers
  4.x era, so `transformers` is pinned to `>=4.49,<5` and `diffusers` to `>=0.33,<0.40`. These
  bounds are load-bearing: transformers 5 removed the `default` entry from `ROPE_INIT_FUNCTIONS`,
  which the vendored code indexes directly, and the failure surfaces as an opaque
  `KeyError: 'default'` deep inside model construction. The adapter and the launcher both check
  the installed versions up front and say so plainly.
- **Neither of accelerate's offload helpers works with this pack**, so memory is managed the way
  ComfyUI manages it. `enable_sequential_cpu_offload` swaps parameters for meta tensors and
  restores them from a hook on `forward`, but the encoder reads `self.ref_queries` directly inside
  its own forward and gets a meta tensor ("Cannot copy out of meta tensor").
  `enable_model_cpu_offload` keeps real tensors but also hooks `Module.forward`, and the pipeline
  calls submodules directly (`transformer.patch_embedding(...)`, `vae.encode(...)`), so the hook
  never fires.
- **Staging** is the default, mirroring how ComfyUI unloads CLIP after conditioning and before
  sampling. Measured weights are transformer 9.3 GiB, VAE 2.6 GiB, MLLM encoder 7.1 GiB
  (Qwen2.5-VL 3B). The encoder owns the GPU for step 3, is evicted, then the transformer and VAE
  take its place, so peak usage is the larger stage (11.9 GiB) rather than the 19.0 GiB total —
  and both stages run at full GPU speed.
- **Block streaming** is available for tighter cards, and is applied automatically if the denoising
  stage still runs out of VRAM. It pages the first 20 of the transformer's 30 blocks (~0.31 GiB
  each) in from system RAM, the technique ComfyUI-WanVideoWrapper exposes as `blocks_to_swap`. It
  works here because `_model_forward` runs `for block in t.blocks:` — an ordinary module call, so
  forward hooks fire, unlike a hook on the bypassed top-level forward.
- diffusers keeps parts of the Wan transformer in fp32 for numerical stability
  (`_keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]`) and
  casts around them inside its own forwards. The pack reimplements the transformer forward and calls
  `condition_embedder.time_embedder` directly with a bfloat16 tensor, so that cast never happens
  ("mat1 and mat2 must have the same dtype"). The adapter reproduces diffusers' cast at the module
  boundary rather than flattening the model to a single dtype, which would discard the stability the
  keep-list is there to provide. The blocks need no help — their forward is diffusers' own.
- The staged handover is driven by a wrapper on the encoder's forward, which also moves the context
  tensor to the accelerator. `_execution_device` is pinned, because that property resolves from an
  unordered set of component names and can otherwise return whichever component is on the CPU.
- Whisper timings come from the chunked pipeline, so long single-speaker takes can drift by a few
  hundred milliseconds against a hand-timed subtitle file.
- H3 generations are durable as project take recipes, but the native worker queue is not yet a
  restart-safe long-form job graph. There is no latent/audio Motion Context continuation, partial
  segment rerender, seam scoring, or remote H3 worker yet.

## Next engineering milestones

1. Replace the concat export compiler with an overlay compositor so blend modes and true multi-layer
   picture tracks render.
2. Add proxy generation for 4K+ sources.
3. Implement slip and slide trims and linked A/V operations.
4. Add GPU-native preview decoding.
5. Add multi-object SAM 2 layers and mid-range correction points to Mask Effects.
6. Qualify the remaining optional editing models under the 14 GiB lease.
7. Package NVIDIA RTX Video SDK only after its SDK agreement is reviewed.
8. Make H3 jobs restart-safe, then add a long-form shot/segment dependency graph with partial
   rerender and resume.
9. Integrate upstream-supported H3 Motion Context continuation with audio-latent continuity and
   measurable seam diagnostics.

The blueprint remains the source of truth for phase order and acceptance criteria.
