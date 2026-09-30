# F5 — NVIDIA RTX super-resolution

**Status:** integrated and verified on a 4090. Installed by `Install-RTX-Upscale.bat
-AcceptLicense`, selected as the `nvidia` upscale type in the export panel, executed by
`ai-runtime/aiv_runtime/upscale.py`'s `NvvfxBackend`. Licence question settled in revision 2, the
temporal-state question settled below.

**Revision 3, 2026-08-19.** Revision 3 replaces the open temporal-state question with the
measured answer, and corrects the supported-factor claim. Revision 2 said, of revision 1:
revision 1 was wrong in its central claim, and wrong
in a way worth recording: it concluded F5 was "a project, not a plan item" requiring native
DirectX/Vulkan interop and a legal blocker on distribution. Both conclusions came from reading SDK
documentation and EULAs rather than checking what NVIDIA actually publishes to PyPI.

---

## What is actually true

**`pip install nvidia-vfx`.** NVIDIA publishes official Python bindings for the Maxine VFX SDK,
including the `VideoSuperRes` effect. First released 2026-03-09, wrapping VFX SDK 1.2.0.0.

```python
import torch
from nvvfx import VideoSuperRes

vsr = VideoSuperRes(quality=VideoSuperRes.QualityLevel.HIGH)
vsr.output_width, vsr.output_height = 3840, 2160
vsr.load()
out = torch.from_dlpack(vsr.run(frame).image).clone()   # frame: (3, H, W) float32 CUDA, [0,1]
```

That is the entire integration surface. A CUDA tensor in, a CUDA tensor out, zero-copy through
DLPack.

### What this removes from revision 1

| revision 1 said | actually |
|---|---|
| "needs a Rust or C++ shim over a DX or Vulkan surface" | No native code at all. Torch tensors in and out. |
| "SDK headers, redistributable model files" | The VFX libraries are bundled in the wheel. |
| "cannot be verified without building on the target GPU" | One pip install and a 320×180 probe frame. |
| "a project, not a plan item" | A worker backend and a model-table entry. |
| "developer-portal login gate" | Public PyPI, no login. |
| "per-GPU-architecture redistributable installers" | Not applicable — that is the *other* distribution channel. |

Revision 1 was researching the wrong artifact. NVIDIA maintains several parallel distribution
channels for Maxine — an NGC developer kit, per-architecture end-user redistributable `.exe`s, the
login-gated RTX Video SDK zip, and this wheel — and only the wheel matters here.

### What it does not remove

- **`LicenseRef-NvidiaProprietary`.** The wheel is proprietary. It is installed by the user from
  NVIDIA's index; we redistribute nothing. `scripts/setup-rtx-upscale.ps1` gates on explicit
  licence acceptance and prints the terms, matching how every other pack in this project works.
- **Notify NVIDIA before commercial release**, and cloud deployment sits outside their standard
  terms.
- The **driver floor is 570.65** on Windows, and the GPU needs Tensor Cores (Turing or newer).

### The licence question, settled

Revision 1 called NVIDIA's "may not cause it to become subject to an open source software license"
clause a blocker. It is not one here. That clause guards against copyleft reciprocity dragging
NVIDIA's code under a licence like the GPL. **FSL-1.1-ALv2 and its Apache-2.0 future licence are
both non-copyleft and have no such reach.** Nothing is being redistributed by us in any case.

The Third-Party Components section of `LICENSE.md` now states this boundary explicitly.

---

## The open question, answered

**Does `VideoSuperRes` carry temporal state between frames? No.**

`scripts/probe-nvvfx.py`, run against the installed wheel on 2026-08-19, from a 640x360 frame:

```
repeat_same_frame_identical      true    max delta 0.0
unaffected_by_intervening_frame  true    max delta 0.0
fresh_instance_identical         true    max delta 0.0
```

The same frame twice through one instance, the same frame again after an intervening *different*
frame, and the same frame through a brand new instance all produce **bit-identical** output. There
is no state in the stream and none in the object.

So it is a per-frame image model, and the name is aspirational. It does **not** fix Real-ESRGAN's
shimmer, and any UI that implied otherwise would be lying. F5 still lands, with honest labelling: a
second AI model with different strengths — it reaches an arbitrary output size directly rather than
overshooting and resampling back, and it has denoise and deblur modes ESRGAN cannot do at all —
rather than the temporally-coherent option.

The per-frame shimmer warning in `resolveUpscalePlan` used to name Real-ESRGAN whichever model was
selected, which read as a reason to switch to this one. It now names the selected model, because
the warning is true of both.

### The rest of what the probe settled

| question | answer |
|---|---|
| `QualityLevel` members | **17**, not the 28 the setup probe reported — `dir()` on an IntEnum leaks `int`'s own methods. Bicubic/Low/Medium/High/Ultra (0–4), Denoise (8–11), Deblur (12–15), HighBitrate (16–19). |
| supported scale factors | **Anything.** 1.33x, 1.5x, 2x, 3x, 4x, a non-uniform 1600x720 and an odd 1281x721 were all accepted and all returned exactly the requested size. NVIDIA's `rtx-upscale` CLI exposing only `2x/3x/4x` is that CLI's choice, not the effect's. |
| beyond 4x | 5x was accepted too. Only the returned *shape* was checked, and 5x is past anything the model was trained for, so the 4x cap stays — ours, in the model table, deliberately. |
| below the documented 360p minimum | Runs. 426x240 to 852x480 succeeded, so "suggested minimum" is a quality note, not a gate. The worker warns once and continues. |
| denoise at native resolution | Works and changes pixels. A real capability we do not expose yet. |
| does `run()` take a batch? | **No.** Stacks of 2, 4 and 8 all raise `ValueError: too many values to unpack (expected 3)` — the binding unpacks the shape into exactly three values, so `(3, H, W)` is the only accepted form. Frame batching in the worker therefore buys nothing; the speedup there comes from overlapping the PNG decode and encode with the model. |
| one instance, a different input size | Fine. |

---

## Integration shape

`AI_UPSCALE_MODELS` in `src/domain/upscale.ts` is already the right abstraction — an entry needs a
factor and a label. Two things it does not currently carry:

1. **A backend tag.** `"esrgan" | "nvvfx"`, so the worker knows which path to take.
2. **Variable factor.** Every ESRGAN entry has one fixed trained factor. `VideoSuperRes` takes an
   arbitrary `output_width`/`output_height` up to 4×, so it can hit the target exactly and often
   needs no resample stage at all. `resolveUpscalePlan` handles this by treating the entry's factor
   as a *maximum* rather than a fixed value — a small change to one branch, and the existing tests
   pin the surrounding behaviour.

Then, in order:

- `ai-runtime/aiv_runtime/upscale.py` gains an `nvvfx` branch beside the spandrel one. Same CLI,
  same JSON events, same tiled/untiled contract. The frame loop is unchanged — only the model call
  differs.
- `src-tauri/src/ai/upscale.rs` gains `nvvfx_state` / readiness beside `real_esrgan_state`, mirroring
  what is already there.
- The quality levels (0–19: BICUBIC through ULTRA, plus `DENOISE_*`, `DEBLUR_*`, `HIGHBITRATE_*`)
  need exposing. Denoise and deblur at native resolution are worth their own control, since they
  are not upscaling at all — they are the thing to reach for on noisy or soft source footage.

**Estimate: comparable to the Real-ESRGAN work in this pass, minus the model pack.** No manifest,
no weight downloads, no hashes — pip does all of it.

---

## What is already in place

- `scripts/setup-rtx-upscale.ps1` — licence gate, Python and driver preflight, install from
  NVIDIA's index, then a real probe that builds the effect and runs a frame. Installing cleanly and
  failing to load are different outcomes and it reports them differently.
- `LICENSE.md` → Third-Party Components.
- `resolveUpscalePlan`, the AI-first-then-resample ordering rule, the live chain readout, and the
  readiness-and-degrade pattern — all built for Real-ESRGAN, all reusable unchanged.

## Ruled out

- **DLSS / NGX** — driven by motion vectors and depth from a 3D render pipeline. Cannot be pointed
  at decoded video.
- **The consumer RTX Video Super Resolution toggle** — a driver-level playback hook in another
  process. Never touches an export.
- **The login-gated RTX Video SDK zip, and the per-architecture redistributables** — both are other
  routes to the same models. The wheel is strictly easier.

---

## Sources

- [nvidia-vfx on PyPI](https://pypi.org/project/nvidia-vfx/) — verified: `wheel-stub` redirecting to
  `https://pypi.nvidia.com/`, `LicenseRef-NvidiaProprietary`, released 2026-03-09
- [NVIDIA VFX Python Bindings Guide](https://docs.nvidia.com/maxine/vfx-python/latest/index.html)
- [Maxine VFX SDK — Video Super Resolution filter](https://docs.nvidia.com/maxine/vfx/latest/Filters/VideoSuperResolution.html)
- [License Agreement for NVIDIA Software Development Kits](https://developer.download.nvidia.com/gameworks/dlss/NVIDIA_RTX_SDK_EULA_(1Feb2021).pdf)
- [License For Customer Use of NVIDIA Software](https://www.nvidia.com/en-us/drivers/nvidia-license/)
- [NVIDIA Broadcast SDK resources](https://www.nvidia.com/en-gb/geforce/broadcasting/broadcast-sdk/resources/) — the redistributable route, not needed
