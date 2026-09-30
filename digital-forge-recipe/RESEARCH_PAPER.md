# Engineering validation: portable FP8 for MiniMax-H3 Ref2VA Turbo

## Scope

This is an engineering investigation and reproducibility report, not a claim of
a novel quantization paper. It records what was attempted, what failed, the
evidence used to choose the replacement, and what remains unqualified.

## Failure being solved

The source `MiniMax-H3-Ref2VA-FP8.pt` contains 313 TorchAO `Float8Tensor`
subclasses. Diffusers group offload calls `.to(cuda)` on these objects. In the
tested Torch/TorchAO combination, alias correction then attempts to associate a
CUDA tensor with CPU storage and raises:

```text
RuntimeError: Attempted to set the storage of a tensor on device "cuda:0"
to a storage on different device "cpu".
```

The failure was reproduced in the untouched source-FP8 control at denoise step
zero. A catch-and-retry monkeypatch cannot make a tensor whose transfer already
violated PyTorch's storage invariant safe; it was the wrong repair boundary.

The original VAE symptom was separate: leaf-level or repeatedly streamed VAE
offload creates excessive transfer/synchronization work and can appear hung for
hours. Decode therefore belongs in an isolated process after denoising, with one
complete VAE active at a time.

## Research findings

The official [MiniMax-H3 Turbo inference
script](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/inference_minimax_h3.py)
uses the LoRA as an inference adapter and defaults its alpha to 8. The official
[Diffusers setup and inference
guide](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/DIFFUSERS_SETUP_AND_INFERENCE.md)
provides the upstream pipeline context. The published [LightX2V Ref2V Turbo
adapter](https://huggingface.co/lightx2v/Minimax-h3-Turbo/blob/main/minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors)
has rank 128 tensors, making the effective PEFT scale `8 / 128 = 0.0625`.

[Q-DiT](https://arxiv.org/abs/2406.17343) identifies channel-wise outliers and
timestep-dependent activation distributions as core obstacles in diffusion
transformer quantization. [ViDiT-Q](https://arxiv.org/abs/2406.02540) likewise
uses channel balancing and timestep-aware quantization rather than naïve uniform
PTQ; its [official implementation](https://github.com/thu-nics/ViDiT-Q) exposes
calibration-oriented workflows. Those findings make an uncalibrated W4A8 or
W8A8 conversion a hypothesis to test, not an authoritative recipe.

[GemLite](https://github.com/dropbox/gemlite) is a genuine packed low-bit kernel
backend, and [TorchAO's quantization
API](https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py)
provides supported quantization transforms. Neither backend by itself supplies
the H3-specific calibration data, error objective, or layer-sensitivity policy
needed to assert visual equivalence.

## Experiments and decision

| Variant | Runtime result | Visual result | Decision |
|---|---|---|---|
| Original TorchAO FP8 + group offload | Fails at denoise step 0 with cross-device storage alias | No sample | Reject runtime path |
| GemLite W4A8 | Completes structurally | Severe blur/collapse | Reject |
| GemLite W8A8 | Completes structurally | Severe chroma corruption | Reject |
| GemLite mixed W4/W8 A16 | Completes structurally | Blur/chroma corruption | Reject |
| Portable source FP8 + Sage, alpha 8 | Full denoise and decode complete | Subject/motion retained; excessive neon chroma | Runtime-qualified, visual pending |
| Portable source FP8 + native SDPA, alpha 8 | Full run; slower | Same chroma pattern | Kernel not demonstrated cause |
| Portable source FP8 + alpha 128 | Full run | Washed-out noise | Confirms alpha 8 |

The genuine solution is therefore a format conversion, not further
requantization. It preserves the already supplied source FP8 quantization and
changes only the representation responsible for the offload incompatibility.

## Conversion algorithm

For each source `Float8Tensor` with quantized payload `Q` and row scale `S`, the
converter writes ordinary tensors `qdata = Q`, `scale = S`, and `orig_shape`.
No dequantization or requantization occurs. At inference, only the active linear
projection reconstructs:

```text
W_compute = BF16(FP32(Q) * FP32(S))
y = linear(x, W_compute) + 0.0625 * linear(linear(x, A), B)
```

The FP32 multiply followed by one BF16 cast matches the source dequantization
order. The FP8 payload remains the resident base representation; the temporary
BF16 weight dies after that projection. All saved values are ordinary tensors,
so Diffusers can transfer them without invoking TorchAO subclass dispatch.

The LoRA is embedded as BF16 A/B tensors rather than folded and requantized.
This preserves the adapter residual exactly at BF16 precision and avoids erasing
a small distilled delta inside a second quantization pass. “Baked in” here means
self-contained in one checkpoint, not mathematically destroyed by fusion.

## Measured system result

On the tested Windows RTX 4090 Laptop GPU system, a 960×544, four-step run with
cached conditioning measured 206.52 seconds for SageAttention denoising and
63.28 seconds for isolated audio/video decode, 269.84 seconds total. Peak
denoising VRAM was 8.42 GiB and video decode peak was 10.11 GiB. The video VAE
used 256-pixel tiles, 64-pixel overlap, a 5×3 spatial grid, 37 latent frames, and
7 temporal clips streamed to CPU.

Native SDPA denoising took 351.87 seconds and reproduced the chroma issue. This
supports two limited conclusions: the portable format fixes the TorchAO offload
crash, and swapping SageAttention for SDPA does not fix the observed color. It
does not prove general perceptual equivalence to an upstream BF16 baseline.

## Remaining validation plan

Visual qualification should proceed before any more low-bit optimization:

1. Capture latent checksums, value ranges, and dtypes at conditioner output,
   each scheduler step, and pre-VAE output.
2. Run the same reference, prompt, seed, scheduler shifts, and VAE through the
   official MiniMax/LightX2V pipeline and compare the first diverging boundary.
3. Decode identical saved latents through both VAE paths to separate denoiser
   color error from decoder color/range error.
4. Confirm reference-image normalization, channel order, latent scaling, and
   video color conversion against upstream code.
5. Add fixed-seed image statistics and perceptual comparisons to the release
   gate. Only after a passing baseline should FlashAttention, FP16 storage, or
   alternative kernels be evaluated.

The converted model is structurally and operationally genuine. Its current
release status must remain **runtime-qualified, visual-quality pending**.

