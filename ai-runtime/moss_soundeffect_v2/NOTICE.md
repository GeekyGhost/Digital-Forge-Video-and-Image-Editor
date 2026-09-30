# MOSS-SoundEffect v2 — vendored source

This folder is third-party code. It is **not** covered by this repository's FSL-1.1-ALv2 licence.

- **Upstream:** [OpenMOSS/MOSS-TTS](https://github.com/OpenMOSS/MOSS-TTS), directory `moss_soundeffect_v2/`
- **Upstream commit:** `934d6826b084c46a0d033402174d5f8ac4ed2519`
- **Copyright:** the OpenMOSS Team
- **Licence:** Apache License 2.0 — see [`LICENSE`](LICENSE) in this folder
- **Includes:** the `diffsynth/` package as shipped by OpenMOSS, a trimmed copy of ModelScope's
  [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) (also Apache License 2.0)

Upstream publishes no NOTICE file for this code.

## Changes made for Digital Forge

Per section 4(b) of the Apache License, these files were modified from the upstream commit above,
and each carries a comment saying so:

| File | Change |
|---|---|
| `pipeline_moss_soundeffect.py` | `save_audio` writes WAV through SoundFile (frames-first, PCM 16) instead of `torchaudio.save`, avoiding torchaudio 2.11's optional TorchCodec dependency |
| `diffsynth/models/qwen3_text_encoder.py` | `from_pretrained` uses `dtype=` instead of the deprecated `torch_dtype=`, and no longer passes `output_hidden_states=True` |
| `pyproject.toml` | Removed `gradio` and `imageio` from the core dependencies, so the app-owned runtime does not install a web UI it never starts |

Every other file in this folder is unchanged from upstream.
