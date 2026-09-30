# Qwen-Image-Edit-2511 — multi-image input mechanics

Research note, 2026-09-07. Focus: how multiple input images are accepted, encoded,
and combined with the noise latent. Source of truth is the diffusers pipeline
`QwenImageEditPlusPipeline` (read from diffusers 0.40.0; local `.venv` has 0.39.0,
which already carries the 2511 `zero_cond_t` support) plus the ComfyUI native
implementation. Weights released 2025-12-23.

---

## 1. How many images

| Surface | Limit |
|---|---|
| diffusers `QwenImageEditPlusPipeline` | **No hard cap.** `image` is a `list`; the code loops over it. |
| ComfyUI `TextEncodeQwenImageEditPlus` | **3 slots** — `image1`, `image2`, `image3`, all optional. No slicing, the list is just fixed at 3. |
| Official guidance | "Optimal performance is currently achieved with 1 to 3 input images" (2509 card; Replicate repeats "works best with 1 to 3" for 2511). Trained combinations: person+person, person+product, person+scene. |
| Architectural ceiling | RoPE tables are built from `torch.arange(4096)`, and the per-image index rides the *frame* axis, so ~4096 images before the table runs out. The real limit is attention cost (§5). |

`batch_size > 1` is **rejected** outright:

```python
if batch_size > 1:
    raise ValueError("QwenImageEditPlusPipeline currently only supports batch_size=1, ...")
```

---

## 2. Two independent paths per image

Every input image is consumed **twice**, at two different resolutions, for two
different purposes.

```
                       ┌── resize to  384*384 px area ──► Qwen2.5-VL ──► prompt_embeds   (semantics)
input image ──┤
                       └── resize to 1024*1024 px area ──► VAE encode ──► image_latents   (appearance)
```

Constants at the top of the pipeline:

```python
CONDITION_IMAGE_SIZE = 384 * 384      # what the VL encoder sees
VAE_IMAGE_SIZE      = 1024 * 1024     # what the VAE sees
```

Both are **target areas**, not dimensions. Aspect ratio is preserved:

```python
def calculate_dimensions(target_area, ratio):
    width  = math.sqrt(target_area * ratio)
    height = width / ratio
    width  = round(width  / 32) * 32
    height = round(height / 32) * 32
    return width, height
```

Each image is sized **independently** — a 3:1 panorama and a 3:4 portrait both end
up at ~1M pixels but with completely different token grids.

> ComfyUI uses the same two area targets (`384*384`, `1024*1024`) but rounds to a
> multiple of **8**, not 32, and upscales with `"area"`. Token grids therefore do
> not match diffusers exactly — same model, non-identical results.

### 2a. Semantic path — `_get_qwen_prompt_embeds`

Images are announced to Qwen2.5-VL positionally, then the user prompt follows:

```python
img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
if isinstance(image, list):
    base_img_prompt = ""
    for i, img in enumerate(image):
        base_img_prompt += img_prompt_template.format(i + 1)
```

so a two-image call literally encodes:

```
Picture 1: <|vision_start|><|image_pad|><|vision_end|>Picture 2: <|vision_start|><|image_pad|><|vision_end|>{prompt}
```

wrapped in the system template, with the first `prompt_template_encode_start_idx = 64`
hidden states dropped afterwards. **This is why "the person from the first image"
style prompting works** — the ordinal is a real token the VL model attends to.
ComfyUI builds the identical string.

Text and vision share **one** sequence. Vision cost per condition image is roughly
14×14 = **~196 tokens** for a square (Qwen2.5-VL re-rounds 384 → 392, patch 14 with
2×2 merge). Three images ≈ 590 tokens of the encoder budget before the prompt.

### 2b. Appearance path — `_encode_vae_image` + `prepare_latents`

```python
image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
image_latents = (image_latents - latents_mean) / latents_std
```

`sample_mode="argmax"` takes the posterior **mode**, not a sample — reference
encoding is deterministic and unaffected by the seed.

Then per image: pack into 2×2 patches, and concatenate along the **sequence** axis:

```python
for image in images:
    ...
    image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents,
                                       image_latent_height, image_latent_width)
    all_image_latents.append(image_latents)
image_latents = torch.cat(all_image_latents, dim=1)     # <-- sequence, not channels
```

---

## 3. The "blend" is sequence concatenation, not blending

There is no alpha, no mask, no latent mixing. Once per denoise step:

```python
latent_model_input = latents
if image_latents is not None:
    latent_model_input = torch.cat([latents, image_latents], dim=1)
```

and immediately after the transformer:

```python
noise_pred = noise_pred[:, : latents.size(1)]
```

Consequences worth internalising:

- Reference latents are **constant for the whole schedule**. They are never noised,
  never stepped, never updated.
- The model's prediction over the reference region is **thrown away**.
- References influence the output *only* as extra keys/values in self-attention.
  "Combining two images" is entirely an attention-routing behaviour learned in
  training — there is no explicit fusion operator to tune.
- Therefore: strength of a reference is not a scalar you can dial. The only levers
  are prompt wording, reference resolution (token count), and ordering.

---

## 4. How the model tells the images apart — 3D RoPE frame axis

`img_shapes` is built as **output first, then each reference in order**:

```python
img_shapes = [[
    (1, height      // vae_scale_factor // 2, width     // vae_scale_factor // 2),   # output
    *[(1, vae_height // vae_scale_factor // 2, vae_width // vae_scale_factor // 2)   # each ref
      for vae_width, vae_height in vae_image_sizes],
]] * batch_size
```

In `QwenEmbedRope.forward`, the **list position becomes the frame index**:

```python
for idx, fhw in enumerate(video_fhw):
    frame, height, width = fhw
    video_freq = self._compute_video_freqs(frame, height, width, idx, device)
```

```python
freqs_frame = freqs_pos[0][idx : idx + frame].view(frame, 1, 1, -1).expand(frame, height, width, -1)
if self.scale_rope:
    freqs_height = torch.cat([freqs_neg[1][-(height - height // 2):], freqs_pos[1][: height // 2]], dim=0)
    freqs_width  = torch.cat([freqs_neg[2][-(width  - width  // 2):], freqs_pos[2][: width  // 2]], dim=0)
```

So:

- Output tokens sit at frame **0**, image1 at frame **1**, image2 at frame **2**, …
- `scale_rope=True` centres each image's h/w coordinates on zero **independently**,
  so every image shares the same spatial origin.
- Net effect: the images are stacked **along a time-like axis**, not laid out side by
  side on a canvas. That is what lets the model say "the head from picture 2 at the
  same spatial position" instead of treating the input as one wide collage.

`axes_dims_rope = [16, 56, 56]` → 16 dims encode which picture a token came from,
56+56 encode where in that picture. Head dim 128, 24 heads, 60 layers.

---

## 5. Cost

At 1024×1024, `vae_scale_factor = 8` and 2×2 packing → 64×64 = **4096 tokens per image**.

| Refs | Sequence length | Attention cost vs 1 ref |
|---|---|---|
| 1 | ~8,192 | 1.0× |
| 2 | ~12,288 | ~2.3× |
| 3 | ~16,384 | ~4.0× |

Quadratic, across 60 layers. This is the practical reason for the "1 to 3" advice —
not a modelling limit. Downscaling a reference is a legitimate speed lever, at the
cost of detail transfer.

---

## 6. What actually changed in 2511 — `zero_cond_t`

`transformer/config.json` diff vs 2509 is a **single key**:

```json
{ ... "axes_dims_rope": [16,56,56], "num_layers": 60, "zero_cond_t": true }
```

(2509's config is otherwise byte-identical and has no `zero_cond_t`.)

In `QwenImageTransformer2DModel.forward`:

```python
if self.zero_cond_t:
    timestep = torch.cat([timestep, timestep * 0], dim=0)
    modulate_index = torch.tensor(
        [[0] * prod(sample[0]) + [1] * sum([prod(s) for s in sample[1:]]) for sample in img_shapes],
        device=timestep.device, dtype=torch.int,
    )
else:
    modulate_index = None
```

`modulate_index` marks output tokens `0` and **all** reference tokens `1`; the block's
`_modulate()` then applies AdaLN conditioned on the *real* timestep to output tokens
and on **t = 0** to reference tokens. The references are explicitly told "you are
clean, not noise at level t". This is the change behind the claimed reduction in
image drift and the improved character / multi-person consistency.

### ComfyUI equivalent — and a real trap

ComfyUI implements the same thing as `ref_method`:

```python
ref_method = kwargs.get("ref_latents_method", self.default_ref_method)
index_ref_method    = (ref_method == "index") or (ref_method == "index_timestep_zero")
negative_ref_method =  ref_method == "negative_index"
timestep_zero       =  ref_method == "index_timestep_zero"
```

with three modes:

- `"index"` — default; frame-index separation only (2509 behaviour).
- `"index_timestep_zero"` — 2511 behaviour, adds the zeroed timestep.
- `"negative_index"` — walks the index *backwards* (used by Qwen-Image-Layered).

The mode is selected by sniffing a **marker key in the state dict**:

```python
if "{}__index_timestep_zero__".format(key_prefix) in state_dict_keys:  # 2511
    dit_config["default_ref_method"] = "index_timestep_zero"
```

**A repackaged 2511 checkpoint (GGUF, custom quant, merged LoRA) that loses that
sentinel key will silently load as `"index"` and behave like 2509 — no error, just
drift and weaker identity.** Community GGUF repos had to patch the key back in
after the fact. If we quantize or repack, carry `__index_timestep_zero__` forward.

ComfyUI's ref loop, for comparison with §3/§4:

```python
for ref in ref_latents:
    if index_ref_method:
        index += 1; h_offset = 0; w_offset = 0
    elif negative_ref_method:
        index -= 1; h_offset = 0; w_offset = 0
    else:                       # legacy Kontext-style spatial offset
        index = 1
        if ref.shape[-2] + h > ref.shape[-1] + w: w_offset = w
        else:                                     h_offset = h
        h = max(h, ref.shape[-2] + h_offset); w = max(w, ref.shape[-1] + w_offset)
    kontext, kontext_ids, _ = self.process_img(ref, index=index, h_offset=h_offset, w_offset=w_offset)
    hidden_states = torch.cat([hidden_states, kontext], dim=1)
    img_ids       = torch.cat([img_ids,       kontext_ids], dim=1)
```

Note the `else` branch: the pre-2509 method packed references **spatially** (tiled
beside the canvas) rather than by index. Do not reach for it with 2511.

---

## 7. Gotchas

1. **Default output size comes from the LAST image**, not the first:
   `image_size = image[-1].size if isinstance(image, list) else image.size`.
   Reordering the list changes the output aspect ratio. Pass explicit `height`/`width`
   to make this deterministic.
2. `height`/`width` are floored to a multiple of `vae_scale_factor * 2` = 16.
3. **`max_sequence_length` is dead.** It is accepted by `__call__` (default 512),
   validated (≤1024), passed to `encode_prompt` — and then `encode_prompt` calls
   `self._get_qwen_prompt_embeds(prompt, image, device)` without it. Nothing truncates.
4. `guidance_scale` is ignored unless the transformer is guidance-distilled;
   2511's config has `guidance_embeds: false`, so real CFG comes from `true_cfg_scale`
   (model card uses 4.0) plus a non-empty `negative_prompt`. `true_cfg_scale > 1` with
   no negative prompt just logs a warning and does nothing.
5. CFG here is norm-preserving, not plain lerp:
   `noise_pred = comb_pred * (||cond_pred|| / ||comb_pred||)`.
6. `QwenEmbedRope` warns and silently uses `video_fhw[0]` if a batch has mixed sizes —
   moot while batch is forced to 1, but a landmine if we ever patch batching in.
7. Reference VAE encoding uses the posterior mode, so identical references always
   give identical conditioning regardless of seed.

---

## 8. Minimal reference call

```python
from diffusers import QwenImageEditPlusPipeline
import torch
from PIL import Image

pipe = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511", torch_dtype=torch.bfloat16).to("cuda")

out = pipe(
    image=[Image.open("a.png"), Image.open("b.png")],   # order defines "Picture 1/2"
    prompt="The magician bear is on the left, the alchemist bear is on the right, "
           "facing each other in the central park square.",
    negative_prompt=" ",
    true_cfg_scale=4.0,
    guidance_scale=1.0,
    num_inference_steps=40,
    height=1024, width=1024,          # pin it; otherwise image[-1] decides
    generator=torch.manual_seed(0),
)
out.images[0].save("out.png")
```

---

## Sources

- Qwen/Qwen-Image-Edit-2511 model card + `transformer/config.json` (HF)
- Qwen/Qwen-Image-Edit-2509 model card ("optimal performance … 1 to 3 input images")
- QwenLM/Qwen-Image README ("2025.12.23: We released Qwen-Image-Edit-2511 weights!")
- diffusers 0.40.0: `pipelines/qwenimage/pipeline_qwenimage_edit_plus.py`,
  `models/transformers/transformer_qwenimage.py`
- ComfyUI master: `comfy_extras/nodes_qwen.py`, `comfy/ldm/qwen_image/model.py`,
  `comfy/model_detection.py`
- unsloth/Qwen-Image-Edit-2511-GGUF discussion #4 (`__index_timestep_zero__` metadata)
