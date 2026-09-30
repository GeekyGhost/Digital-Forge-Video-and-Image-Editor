# Acknowledgements

Digital Forge Video Studio is an editor wrapped around other people's research. The timeline, the
export graph and the adapter plumbing are mine; nearly everything that makes the AI features work
was built, trained, quantized or debugged by someone else and given away.

This file tries to name all of them. If you are listed here and would like the wording, the link or
the attribution changed, please open an issue — I would rather get it right.

---

## Specific debts

A few pieces of work are the direct reason a particular feature exists on a 16 GB laptop GPU at all.
They deserve more than a line in a dependency table.

**[LightX2V](https://huggingface.co/lightx2v)** — the step-distilled adapters. MiniMax-H3 at four or
eight transformer evaluations instead of twenty, and Qwen Image Edit at Lightning speed, are what
make these models usable on consumer hardware rather than merely loadable.
[`Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) ·
[`Qwen-Image-Edit-2511-Lightning`](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning)

**[Unsloth](https://huggingface.co/unsloth)** — the FP8 quantizations. A 20.3 GB checkpoint that
loads on a 16 GB card is the entire reason MiniMax-H3 is in this project.
[`MiniMax-H3-FP8`](https://huggingface.co/unsloth/MiniMax-H3-FP8)

**[woct0rdho](https://github.com/woct0rdho)** — Windows builds of
[SageAttention](https://github.com/woct0rdho/SageAttention) and
[Triton](https://github.com/woct0rdho/triton-windows). Without these wheels the fast attention path
simply would not exist on Windows, and a great many people would be stuck compiling. Thankless
maintenance work that quietly unblocks a whole platform.

**[apolinario](https://github.com/apolinario)** — for
[diffusers#14355](https://github.com/huggingface/diffusers/pull/14355), which added MiniMax-H3 to
Diffusers. Both H3 workers in this project are built directly on that modular pipeline, and this
repository pins that PR by reference.

**[Lightricks](https://github.com/Lightricks)** — for publishing LTX-2 openly, weights and source
together, including the distilled and FP8 variants that actually fit on consumer cards. The
generation, retake and foley features are all built on it.

---

## Models

Every one of these carries its own license, and some restrict commercial use. `scripts/models.py`
requires explicit acceptance for exactly that reason. Please read the terms for whatever you install.

### Video and audio generation

| Model | By | Used for |
|---|---|---|
| [LTX-2 / LTX-2.3](https://github.com/Lightricks/LTX-2) · [weights](https://huggingface.co/Lightricks/LTX-2.3) · [FP8](https://huggingface.co/Lightricks/LTX-2.3-fp8) | [Lightricks](https://www.lightricks.com/) | generation, clip extend, clip-to-clip, retake |
| [LTX-2.3-22b-LoRA-Foley-V2A](https://huggingface.co/Lightricks/LTX-2.3-22b-LoRA-Foley-V2A) | Lightricks | video-conditioned foley |
| [MOSS-SoundEffect v2](https://github.com/OpenMOSS/MOSS-TTS) | [OpenMOSS](https://github.com/OpenMOSS) | prompt-designed sound effects; its inference code, which builds on ModelScope's [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio), is vendored in `ai-runtime/moss_soundeffect_v2/` (Apache-2.0) |
| [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) · [source](https://github.com/MiniMax-AI/MiniMax-H3) | [MiniMax](https://www.minimax.io/) | joint video + audio, FL2VA and Ref2VA |
| [MiniMax-H3-FP8](https://huggingface.co/unsloth/MiniMax-H3-FP8) | [Unsloth](https://unsloth.ai/) | the quantization that makes H3 fit |
| [Minimax-h3-Turbo](https://huggingface.co/lightx2v/Minimax-h3-Turbo) | [LightX2V](https://github.com/ModelTC/LightX2V) | 4- and 8-step distilled adapters |

### Clip editing

| Model | By | Used for |
|---|---|---|
| [Kiwi-Edit](https://github.com/showlab/Kiwi-Edit) · [weights](https://huggingface.co/linyq/kiwi-edit-5b-instruct-reference-diffusers) | [Show Lab, NUS](https://github.com/showlab) · [linyq](https://huggingface.co/linyq) | the AI clip editor (Ctrl+K) |

### Image Studio

| Model | By | Used for |
|---|---|---|
| [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) | [Tongyi-MAI, Alibaba](https://huggingface.co/Tongyi-MAI) | still-frame generation |
| [Z-Image](https://huggingface.co/Tongyi-MAI/Z-Image) | [Tongyi-MAI, Alibaba](https://huggingface.co/Tongyi-MAI) | Creative generation and LoRA training |
| [Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | [Qwen, Alibaba](https://huggingface.co/Qwen) | instruction-guided image editing |
| [Qwen-Image-Edit-2511-Lightning](https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning) | LightX2V | the fast path for the above |
| [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) · [weights](https://huggingface.co/ZhengPeng7/BiRefNet) | [Peng Zheng](https://github.com/ZhengPeng7) | background removal and soft alpha mattes |

### Utilities

| Model | By | Used for |
|---|---|---|
| [SAM 2](https://github.com/facebookresearch/sam2) · [weights](https://huggingface.co/facebook/sam2.1-hiera-large) | [Meta AI](https://ai.meta.com/) | click-to-track subject masking |
| [Whisper large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo) · [source](https://github.com/openai/whisper) | [OpenAI](https://openai.com/) | automatic captions |
| [Gemma 3 12B](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) | [Google DeepMind](https://deepmind.google/) | LTX text conditioning |
| [Demucs](https://github.com/adefossez/demucs) | [Alexandre Défossez](https://github.com/adefossez) | music and dialogue stem separation |

### Tracking and depth (VFX)

| Model | By | Used for |
|---|---|---|
| [CoTracker3](https://github.com/facebookresearch/co-tracker) · [weights](https://huggingface.co/facebook/cotracker3) | [Meta AI](https://ai.meta.com/) | point tracks for attach-to-track, the trail and emitters. **CC-BY-NC 4.0: non-commercial use only.** The built-in Lucas-Kanade tracker needs no model. |
| [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything) · [Small weights](https://huggingface.co/depth-anything/Video-Depth-Anything-Small) | [ByteDance](https://github.com/DepthAnything) | per-frame depth for depth occlusion. Small only (Apache-2.0); the Base and Large checkpoints are not used. |

---

## The machine-learning stack

- **[PyTorch](https://pytorch.org/)** — every adapter in this project runs on it.
- **[Hugging Face](https://huggingface.co/)** — [Diffusers](https://github.com/huggingface/diffusers),
  [Transformers](https://github.com/huggingface/transformers),
  [Accelerate](https://github.com/huggingface/accelerate),
  [PEFT](https://github.com/huggingface/peft),
  [Safetensors](https://github.com/huggingface/safetensors) and
  [huggingface_hub](https://github.com/huggingface/huggingface_hub). Diffusers' modular pipelines and
  its group-offloading hooks are what let a 20 GB transformer stream through a 16 GB card at all, and
  the Hub is how every model here is distributed.
- **[TorchAO](https://github.com/pytorch/ao)** — the FP8 and int8 quantization primitives behind the
  H3 checkpoints and the baked text encoder.
- **[SageAttention](https://github.com/thu-ml/SageAttention)** ([THU-ML](https://github.com/thu-ml)) —
  quantized attention, offered as an optional backend.
- **[einops](https://github.com/arogozhnikov/einops)**, **[NumPy](https://numpy.org/)**,
  **[Pillow](https://python-pillow.org/)**, **[PyAV](https://github.com/PyAV-Org/PyAV)**,
  **[pydantic](https://pydantic.dev/)**, **[psutil](https://github.com/giampaolo/psutil)**,
  **[tqdm](https://github.com/tqdm/tqdm)**.

## The application

- **[FFmpeg](https://ffmpeg.org/)** — every frame this editor imports, previews, thumbnails,
  waveforms or exports passes through it. The single most load-bearing dependency in the project.
- **[Tauri](https://tauri.app/)** — the desktop shell.
- **[mp4box.js](https://github.com/gpac/mp4box.js)** ([GPAC](https://gpac.io/)) — MP4 demuxing for
  the WebCodecs decode path behind the VFX compositor. Vendored, BSD 3-Clause.
- **[React](https://react.dev/)** and **[Vite](https://vite.dev/)** — the interface and its build.
- **[Zustand](https://github.com/pmndrs/zustand)** — editor state.
- **[Zod](https://zod.dev/)** — project-file validation.
- **[Lucide](https://lucide.dev/)** — the icon set.
- **[Rust](https://www.rust-lang.org/)**, and the crates the native core is built from:
  [tokio](https://tokio.rs/), [serde](https://serde.rs/),
  [anyhow](https://github.com/dtolnay/anyhow), [thiserror](https://github.com/dtolnay/thiserror),
  [uuid](https://github.com/uuid-rs/uuid), [chrono](https://github.com/chronotope/chrono),
  [parking_lot](https://github.com/Amanieu/parking_lot),
  [sha2](https://github.com/RustCrypto/hashes) and
  [base64](https://github.com/marshallpierce/rust-base64).
- **[Astral](https://astral.sh/)** — [uv](https://github.com/astral-sh/uv) resolves and builds the
  isolated Python environments, and [Ruff](https://github.com/astral-sh/ruff) lints them.
- **[pytest](https://pytest.org/)** — the Python test suite.

## Licensing

This project is released under the
[Functional Source License 1.1 with an Apache 2.0 future licence](LICENSE.md), a template published
by [Sentry](https://sentry.io/) at [fsl.software](https://fsl.software/). Thanks to them for putting
a readable, time-limited source-available licence into the commons.

**Everything listed above carries its own separate licence**, and several of the model licences are
stricter than this project's. Nothing here grants you rights to any of them. Check each one.
