# Z-Image LoRA trainer + auto-captioner review — 2026-09-23

Scope: `zimage_training.py`, `zimage_caption_worker.py`, `zimage_train_worker.py`,
`scripts/patches/zimage-aiv-training.patch`, `setup-zimage-trainer.ps1`, `ai/lora.rs`, `ai/image.rs`,
`image_studio/zit.py`, `LoraTraining.tsx`, `ImageStudio.tsx`, and the two finished runs
`models/loras/z-image/cap-dc` and `captain-dicio`.

## What the finished runs show

| | cap-dc / captain-dicio |
|---|---|
| Images | 8 (7 are 1376×768, same render style) |
| Schedule | 500 optimizer steps × grad-accum 4 = 2000 samples (250 passes over 8 images) |
| Adapter | rank 16, alpha 16, **attention only** (`to_q/k/v/out.0`) — 272 tensors |
| Precision | `--do_fp8_training` (torchao) |
| Speed | **8.2 s per micro-step** → ~4.5 h per run |
| Loss | flat ~0.59 (normal for flow matching with random timesteps; not diagnostic) |
| Captions | tag lists, e.g. `capdiciosks person, medium shot, frontal, walking, neutral, desert canyon…` |
| Outlier | `layers.16.attention.to_v` ΔW Frobenius 46 (median 2.1), max |ΔW| 1.8 |

## What was right

- **Training on Z-Image Base, not Turbo.** The Tongyi model card calls Base the fine-tuning target;
  community results agree (train on Base, generate on either).
- **The pinned official Diffusers trainer** plus the AIV patch. Pause/resume is correct: the pause
  checkpoint is taken after an optimizer step, `resume_step` skips the consumed micro-batches, and the
  dataloader order is fixed when captions are cached, so the skip lands on the right batch.
- **Continuation from an existing LoRA** via `set_peft_model_state_dict` with an unexpected-key check.
- **Dataset staging**: `metadata.jsonl` + imagefolder, SHA-256 and perceptual dedupe, aspect buckets,
  no random flip (right for asymmetric designs), cached latents, text-encoder offload, 8-bit Adam.
- **Trigger contract**: the app, not the VLM, inserts the trigger exactly once.
- **Character profile principle**: leave the invariant design out of captions so it binds to the trigger.
- **Inference**: Diffusers `load_lora_weights` with an active-adapter check; Turbo at CFG 0 / 9 steps with
  no negative prompt; Base at 36 steps / CFG 4 (card: 28–50 steps, CFG 3–5).
- **File format**: `transformer.layers.N.attention.to_q.lora_A.weight` loads in ComfyUI unchanged —
  ComfyUI maps `to_q/k/v` onto its fused `qkv` and also maps `feed_forward.w1/w2/w3`.

## What was wrong (by impact) and what changed

1. **Attention-only adapter.** The Diffusers default. AI Toolkit trains every linear layer of the
   Z-Image transformer; the SwiGLU feed-forward (`w1/w2/w3`) is where much of a DiT's visual detail
   lives. → New runs target `to_q,to_k,to_v,to_out.0,feed_forward.w1,feed_forward.w2,feed_forward.w3`
   (file ~2.4× larger).
2. **fp8 saved no memory.** torchao float8 training keeps weights in bf16 (~12.3 GB for the 6B
   transformer) and only casts for the matmul; without `torch.compile` it is slower, not faster. On a
   16 GB laptop GPU, 8.2 s per micro-step is consistent with spilling into shared system memory.
   → Default precision is now **NF4 QLoRA** (`--bnb_quantization_config_path`), ~3.5 GB of weights.
   *Not yet measured on this GPU* — compare the s/it in the job message with 8.2.
3. **Captions.**
   - Tag lists instead of prose; Z-Image's text encoder is Qwen3-4B and it was trained on sentences.
   - The medium ("3D render") was never captioned, so it bound to the trigger.
   - Weapons: captioned in image 8, absent in 4 and 7 — inconsistent binding.
   - `max_new_tokens=192` for an 11-field JSON; a truncated answer fell through as raw JSON text into
     the caption. A non-list `tags` value would fail Rust's `Vec<String>` deserialisation.
   → The VLM now writes a prose `caption` using a `<subject>` placeholder plus a `medium` field. The app
   substitutes `trigger class` for the placeholder. For characters, if the prose contains identity words
   (from a built-in list plus the VLM's own `clothing`/`held_items` fields) it is rebuilt from the scene
   fields, with offending fragments removed, and the image is flagged `caption-rebuilt`. 512 tokens,
   truncated-JSON repair, `caption-unparsed` flag, list coercion.
4. **Timestep shift.** Base ships `shift=6.0`: only 14% of uniformly sampled training steps fall below
   sigma 0.5, where fine detail is learned. → `--aiv_timestep_shift 3.0` (AI Toolkit's Z-Image training
   value, Turbo's inference value): 25%.
5. **Using a Base LoRA on Turbo.** Community reports need ~1.5–2.5 strength; the cap was 2.0. → Cap 3.0,
   hint text updated.
6. **Smaller fixes.** `--guidance_scale` removed (a Flux leftover; Z-Image ignores it). `constant` →
   `constant_with_warmup` (warmup was ignored). Style LoRAs' `triggerPhrase` now `TOK style`, matching
   the `in TOK style` captions (was `TOK image`). EXIF-rotated images report their real size.
   `HF_DEACTIVATE_ASYNC_LOAD=1` in both workers (same transformers v5 fault as Music 3).
   The setup script now restores the pristine trainer before patching, so a revised patch can be applied
   over an older one.

Resume safety: the resolved settings are written back to `training.json`. A job paused before this
change resumes with its original layout (attention-only, fp8, model shift).
The new settings are config keys (`loraLayers`, `precision`, `timestepShift`) and are not in the UI yet.

## Recommendations not implemented

- **Dataset**: 15–30 images. Add close-ups of the face and visor, side and back views, and full-body
  shots. Vary the lighting. Include a few square or portrait crops.
- **Class noun**: `man` rather than `person` for Captain Dicio.
- **Trigger**: the `sks` suffix comes from CLIP-era models. Qwen3 splits any invented word into subword
  pieces, so what matters is that the trigger is unique and not a real word. `capdiciosks` is fine;
  `Captain_Dicio` pulls in the meaning of "captain".
- **Schedule**: 2000 samples is in range (AI Toolkit uses 3000 at batch 1). Compare checkpoints
  200–500 on Base.
- **ComfyUI/CivitAI**: the file stores no `.alpha` tensors, so ComfyUI assumes scale 1. Keep
  alpha = rank for anything you share.
- **Turbo-first alternative**: AI Toolkit's "Z-Image Turbo (w/ Training Adapter)" route, if Turbo is
  the main target.

## Using the LoRA

- Prompt in sentences, with the trigger phrase as the subject: `A full-body shot of capdiciosks man
  on a rooftop at night, neon rain, 3D render.`
- Base: 28–50 steps, CFG 3–5, strength 0.8–1.2, negative prompt allowed.
- Turbo: 8–9 steps, CFG 0, no negative prompt, strength 1.5–2.5.

## Verified / not verified

Verified in the cloud workspace: 15/15 `test_zimage_training.py`. The patch applies to Diffusers
`9f12469`. The patched trainer's argparse accepts the full new command. PEFT target names match the real
`ZImageTransformer2DModel`. An attention-only LoRA continues into the FFN config with no unexpected keys
and zero-initialised FFN B matrices. An FFN LoRA loads through `ZImagePipeline`. The worker ran end to end
against a fake trainer.

Not verified: a GPU run (NF4 speed and memory), `tsc`, and the Rust build. The device shell could not
mount the repo in this session.
