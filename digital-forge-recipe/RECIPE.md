# MiniMax-H3 Ref2VA Turbo portable-FP8 recipe

## Result

The canonical conversion is `MiniMax-H3-Ref2VA-DF-Turbo-Portable-FP8.safetensors`.
It is a genuine runtime format, not an INT4 file that gets expanded permanently
at load time:

- all 313 source TorchAO `Float8Tensor` payloads and row scales are copied
  bit-for-bit into ordinary safetensors tensors;
- all 322 non-FP8 source tensors remain BF16;
- all 312 official Ref2V Turbo LoRA A/B pairs remain BF16 and are embedded next
  to their FP8 base projection with the published alpha 8 (rank 128, scale
  0.0625);
- the runtime dequantizes only the active projection for BF16 compute;
- no TorchAO tensor subclass is resident, so Diffusers group offload can safely
  move the ordinary tensors between CPU and CUDA.

This fixes the observed cross-device storage-aliasing exception. It does not
claim that a new W4A8 quantizer has been quality-qualified.

## Inputs and provenance

Place these files in the existing MiniMax-H3 model pack:

| Input | Expected SHA-256 |
|---|---|
| `quantized/MiniMax-H3-Ref2VA-FP8.pt` | `5f49ff7dcc9310c3aeb5470db1151616697acf19c11627793d04bba662d50636` |
| `loras/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors` | `9e642fc8749c74f8da5e2382877ab5c7aa37b9a73b7fd0d6d457bd1b3cb1ae99` |

The converter validates that every LoRA A tensor has a B tensor and that all
312 targets exist in the source checkpoint.

## Build the checkpoint

From the project root:

```powershell
vendor\MiniMax-H3\.venv\Scripts\python.exe digital-forge-recipe\replicate_df_conversion.py
```

Use `--overwrite` only when intentionally replacing an existing output. The
writer first creates a `.partial` file and atomically renames it after a complete
write. Allow at least 25 GiB of free disk space. Source hashes are calculated by
default; `--no-hash` skips that verification.

The portable conversion is CPU serialization and does not require GemLite,
Triton, CUDA, or a Rust build. Torch/TorchAO are needed to read the original
TorchAO checkpoint; safetensors is needed to write the portable file.

The completed artifact from this qualification run is:

| Property | Value |
|---|---|
| Size | 21,553,623,328 bytes (20.0734 GiB) |
| SHA-256 | `4d2c499830c23c88812dd60fdf434bb705b787ccbc50d9cd04ff1bdf0da5cf39` |
| Conversion time | 73.48 seconds |
| Portable source-FP8 projections | 313 |
| Embedded BF16 LoRA pairs | 312 |

The adjacent `.conversion.json` report records counts, hashes, versions, and
elapsed time for auditability.

Verify the saved tensor structure and runtime projection math with:

```powershell
vendor\MiniMax-H3\.venv\Scripts\python.exe digital-forge-recipe\verify_portable_fp8.py
# read and compare every saved FP8/LoRA value against both source files
vendor\MiniMax-H3\.venv\Scripts\python.exe digital-forge-recipe\verify_portable_fp8.py --deep
```

## Runtime integration

The app selects the portable checkpoint in `src-tauri/src/ai/h3_ref.rs`. The
Python loader recognizes the `df_portable_source_fp8_v1` metadata, replaces the
corresponding `nn.Linear` modules with `PortableFP8Linear`, and wraps the 312
LoRA-targeted projections with `EmbeddedLoraLinear`.

The decode pipeline is process-isolated from denoising. Audio and video decode
run separately, only one VAE is resident at a time, video uses 256-pixel tiles
with 64-pixel overlap, and completed temporal clips are moved to CPU. Progress
logs report clip/tile counts, process RSS, resident VRAM, and peak VRAM.

Rust is unrelated to model conversion. Recompile the desktop shell only after
changing Rust integration code:

```powershell
.\Compile-Rust.bat
# or an optimized build
.\Compile-Rust.bat release
```

## Qualification status

A full 960×544, 4-step Ref2VA run completed:

- denoise: 206.52 seconds with SageAttention;
- denoise peak VRAM: 8.42 GiB; host RSS: 23.21 GiB;
- isolated audio/video decode: 63.28 seconds;
- video VAE: 256/64 tiling, 5×3 grid, 37 latent frames, 7 temporal clips;
- video VAE peak VRAM: 10.11 GiB;
- total with cached conditioning: 269.84 seconds.

The structural and memory-path qualification passed. Visual qualification is
still pending: the sample retained subject and motion but showed excessive neon
chroma. Native SDPA reproduced the same chroma while running substantially
slower, so attention-kernel choice is not the demonstrated cause. Alpha 128
produced washed-out noise, confirming that the official alpha 8 must remain.

## Experimental profiles

`w8-a16`, `w4w8-a16`, `experimental-w8a8`, and `experimental-w4a8` are retained
for research only. They require CUDA and GemLite and have not passed visual
qualification. Do not ship their output merely because it loads or completes a
forward pass; the tested variants produced blur or severe color corruption.
