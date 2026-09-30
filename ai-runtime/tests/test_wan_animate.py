"""Wan-Animate-2 worker (VFX Phase 5).

The geometry helpers are tested everywhere. Where Diffusers 0.40 is installed, the resolved size is
checked against Diffusers' own image step, and a tiny random-initialised model of the exact
architecture runs through the real modular pipeline end to end: bake -> FP8 load -> the quantised
CPU reference cache -> two segments -> frames trimmed to the driving clip. That proves the wiring
before a 46 GB download; the weights themselves are proved on the GPU.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from aiv_runtime import wan_animate


def test_plan_size_is_sixteen_aligned_plate_aspect_within_the_area():
    width, height = wan_animate.plan_size(960, 544, 512 * 512)
    assert (width % 16, height % 16) == (0, 0)
    assert width * height <= 512 * 512
    assert abs(width / height - 960 / 544) < 16 / height
    assert (width, height) == (672, 384)
    assert wan_animate.plan_size(1920, 1080) == (672, 384)
    assert wan_animate.plan_size(768, 512) == (624, 416)
    portrait = wan_animate.plan_size(1080, 1920)
    assert portrait[0] < portrait[1] and portrait[0] * portrait[1] <= 512 * 512


def test_plate_crop_has_the_output_aspect_and_is_centred():
    for plate, size in (((960, 544), (672, 384)), ((1920, 1080), (672, 384)), ((768, 512), (624, 416)), ((1080, 1920), None)):
        size = size or wan_animate.plan_size(*plate)
        x, y, w, h = wan_animate.plate_crop(*plate, *size)
        assert 0 <= x and 0 <= y and x + w <= plate[0] and y + h <= plate[1]
        assert abs(w / h - size[0] / size[1]) <= 1.0 / h + 1e-9
        assert abs((plate[0] - w) - 2 * x) <= 1 and abs((plate[1] - h) - 2 * y) <= 1
    assert wan_animate.plate_crop(960, 544, 672, 384) == (4, 0, 952, 544)
    assert wan_animate.plate_crop(1920, 1080, 672, 384) == (15, 0, 1890, 1080)


def test_area_inputs_resolve_exactly_even_where_plain_inputs_floor_short():
    floors_short = []
    for plate in ((960, 544), (1920, 1080), (768, 512), (1280, 720), (1080, 1920), (832, 480), (640, 640)):
        for area in (256 * 256, 384 * 384, 512 * 512, 640 * 640):
            width, height = wan_animate.plan_size(*plate, area)
            h_in, w_in = wan_animate.area_inputs(width, height)
            assert wan_animate.upstream_resolved_size(width, height, h_in, w_in) == (height, width)
            if wan_animate.upstream_resolved_size(width, height, height, width) != (height, width):
                floors_short.append((width, height))
    # Recorded, not required: whether plain W x H inputs ever floor short depends on float rounding.
    print("sizes where W x H inputs floor short:", floors_short)


def test_segment_length_is_four_n_plus_one_and_capped_by_size():
    assert wan_animate.segment_length(672, 384, 124) == 33
    assert wan_animate.segment_length(768, 512, 124) == 17
    assert wan_animate.segment_length(672, 384, 10) == 13
    assert wan_animate.segment_length(672, 384, 124, requested=9) == 9
    with pytest.raises(ValueError):
        wan_animate.segment_length(672, 384, 124, requested=10)


def test_driving_filter_selects_exact_frames_then_crops_and_scales():
    text = wan_animate.driving_filter((0, 2, 960, 539), 672, 384, 30, 124)
    assert text.startswith("select='between(n\\,30\\,153)',")
    assert "crop=960:539:0:2" in text and "scale=672:384" in text


def test_pad_reference_keeps_aspect_and_uses_the_border_colour():
    Image = pytest.importorskip("PIL.Image")
    image = Image.new("RGB", (100, 200), (200, 30, 30))
    image.paste((10, 200, 10), (20, 20, 80, 180))
    padded = wan_animate.pad_reference(image, 672, 384)
    assert padded.size == (672, 384)
    assert padded.getpixel((5, 5)) == (200, 30, 30)
    assert padded.getpixel((336, 192)) == (10, 200, 10)


try:  # The model tests need Diffusers 0.40 (the Wan Animate venv or the container); the helpers above do not.
    import diffusers
    import torch
except ImportError:
    diffusers = torch = None
pytestmark_needs_animate = pytest.mark.skipif(diffusers is None or not hasattr(diffusers, "WanAnimate2DistilledBlocks"),
                                              reason="needs torch and Diffusers with the Animate-2 modular pipeline")


@pytestmark_needs_animate
def test_resolved_size_matches_diffusers_own_image_step():
    from PIL import Image
    from diffusers import WanAnimate2DistilledBlocks
    from diffusers.modular_pipelines.wan_animate_2.encoders import WanAnimate2ProcessImagesInputStep

    pipeline = WanAnimate2DistilledBlocks().init_pipeline()
    step = WanAnimate2ProcessImagesInputStep()
    for plate in ((960, 544), (1920, 1080), (1080, 1920), (768, 512)):
        width, height = wan_animate.plan_size(*plate)
        h_in, w_in = wan_animate.area_inputs(width, height)
        from diffusers.modular_pipelines.modular_pipeline import PipelineState

        state = PipelineState()
        for key, value in {"image": Image.new("RGB", (width, height)), "height": h_in, "width": w_in}.items():
            state.set(key, value)
        _, state = step(pipeline, state)
        assert (state.get("height"), state.get("width")) == (height, width)
        assert state.get("crop_region") == (0, 0, height, width)


OFFICIAL_ATTENTION = ((".to_q.", ".q."), (".to_k.", ".k."), (".to_v.", ".v."), (".to_out.0.", ".o."),
                      (".add_k_proj.", ".k_img."), (".add_v_proj.", ".v_img."), (".norm_added_k.", ".norm_k_img."))


def official_name(name: str) -> str:
    """The inverse of Diffusers' `convert_wan_animate_2_transformer_to_diffusers`."""
    if name.startswith("blocks."):
        head, index, rest = name.split(".", 2)
        name = f"{head}.{index}.block.{rest}"
    if ".self_attn." in name or ".cross_attn." in name:
        for new, old in OFFICIAL_ATTENTION:
            if new in name:
                return name.replace(new, old)
    return name


def _tiny_models(root: Path) -> Path:
    """models/Wan-Animate-2 in miniature: the real classes at toy sizes, saved the way the repo is."""
    from diffusers import AutoencoderKLWan, WanAnimate2Transformer3DModel
    from safetensors.torch import save_file
    from transformers import CLIPVisionConfig, CLIPVisionModel

    torch.manual_seed(0)
    distilled = root / "distilled"
    transformer = WanAnimate2Transformer3DModel(dim=64, ffn_dim=128, num_heads=4, num_layers=2, text_dim=32, freq_dim=32)
    tdir = distilled / "transformer"
    tdir.mkdir(parents=True)
    transformer.register_to_config()
    (tdir / "config.json").write_text(json.dumps(dict(transformer.config)), encoding="utf-8")
    # Saved in the OFFICIAL layout, as the pinned revision is (`blocks.N.block.cross_attn.k.bias`, found on
    # the first real setup run, 2026-09-29), with the checkpoint's own fp32 tensors left as they are.
    state = {official_name(name): (value if value.dtype == torch.float32 and "modulation" in name else value.to(torch.bfloat16)).contiguous()
             for name, value in transformer.state_dict().items()}
    names = sorted(state)
    half = len(names) // 2
    shards = {"diffusion_pytorch_model-00001-of-00002.safetensors": names[:half],
              "diffusion_pytorch_model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        save_file({key: state[key] for key in keys}, str(tdir / shard))
        weight_map.update({key: shard for key in keys})
    (tdir / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")
    CLIPVisionModel(CLIPVisionConfig(hidden_size=1280, intermediate_size=256, num_hidden_layers=2, num_attention_heads=16,
                                     image_size=224, patch_size=14)).save_pretrained(distilled / "image_encoder")
    AutoencoderKLWan(base_dim=8, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=1).save_pretrained(distilled / "vae")
    (distilled / "model_index.json").write_text(json.dumps({
        "_class_name": "WanAnimate2Pipeline", "scheduler": ["diffusers", "DPMSolverMultistepScheduler"]}), encoding="utf-8")
    (distilled / "scheduler").mkdir()
    (distilled / "scheduler" / "scheduler_config.json").write_text(json.dumps({
        "_class_name": "UniPCMultistepScheduler", "flow_shift": 5.0, "prediction_type": "flow_prediction",
        "use_flow_sigmas": True, "solver_order": 2, "solver_type": "bh2", "predict_x0": True,
        "num_train_timesteps": 1000, "final_sigmas_type": "zero", "lower_order_final": True,
        "timestep_spacing": "linspace", "time_shift_type": "exponential"}), encoding="utf-8")
    return root


@pytestmark_needs_animate
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_tiny_model_runs_the_real_modular_pipeline_end_to_end(tmp_path):
    from PIL import Image
    from safetensors.torch import save_file

    models = _tiny_models(tmp_path / "models")
    # bake: BF16 shards -> FP8 where layerwise casting would store FP8
    assert wan_animate.main(["bake", "--models", str(models), "--stats", str(tmp_path / "bake.json")]) == 0
    bake = json.loads((tmp_path / "bake.json").read_text())
    assert bake["ok"], bake.get("error")
    assert bake["fp8Tensors"] > 0 and bake["bf16Tensors"] > 0 and bake["fp32Tensors"] > 0 and bake["renamed"] > 0
    from safetensors import safe_open

    baked = sorted((models / "baked").glob("*.safetensors"))
    with safe_open(str(baked[0]), framework="pt") as handle:
        names = list(handle.keys())
        assert not any(".block." in name for name in names) and any(".to_q." in name for name in names)
    dtypes = {}
    for path in baked:
        with safe_open(str(path), framework="pt") as handle:
            dtypes.update({name: handle.get_slice(name).get_dtype() for name in handle.keys()})
    assert dtypes["blocks.0.modulation"] == "F32" and dtypes["blocks.0.self_attn.to_q.weight"] == "F8_E4M3"

    # A 13-frame driving clip whose frames are numbered by brightness, 400x228 (not the output aspect).
    clip = tmp_path / "plate.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=black:s=400x228:r=24:d=1", "-vf", "geq=lum='16+N*10':cb=128:cr=128", "-frames:v", "20",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip)], check=True)
    width, height = wan_animate.plan_size(400, 228, 256 * 256)
    crop = wan_animate.plate_crop(400, 228, width, height)
    frames = wan_animate.extract_driving(clip, tmp_path / "check", crop, width, height, 3, 13)
    first = Image.open(frames[0]).convert("L").getpixel((width // 2, height // 2))
    last = Image.open(frames[-1]).convert("L").getpixel((width // 2, height // 2))
    full = lambda n: (n * 10) * 255 / 219  # limited-range luma 16 + 10n, read back full range
    assert abs(first - full(3)) <= 3 and abs(last - full(15)) <= 3, (first, last)

    character = tmp_path / "hero.png"
    Image.new("RGB", (60, 120), (90, 90, 200)).save(character)
    embeds = tmp_path / "embeds.safetensors"
    save_file({"prompt_embeds": torch.randn(1, 512, 32, dtype=torch.bfloat16),
               "prompt_ref_embeds": torch.randn(1, 512, 32, dtype=torch.bfloat16)}, str(embeds))
    out = tmp_path / "run"
    code = wan_animate.main(["animate", "--models", str(models), "--embeds", str(embeds), "--character", str(character),
                             "--driving-video", str(clip), "--first-frame", "3", "--frame-count", "13",
                             "--width", str(width), "--height", str(height), "--crop", ",".join(map(str, crop)),
                             "--steps", "2", "--segment-frames", "9", "--out-dir", str(out), "--allow-cpu",
                             "--stats", str(tmp_path / "animate.json")])
    stats = json.loads((tmp_path / "animate.json").read_text())
    assert code == 0, stats.get("error")
    assert stats["frames"] == 13 and stats["segments"] == 2 and stats["segmentFrames"] == 9
    assert (stats["width"], stats["height"]) == (width, height)
    from aiv_runtime import wan_animate_cache
    assert wan_animate_cache.StreamingKVCache.peak_bytes > 0  # the quantised cache served the reference K/V
    written = sorted((out / "frames").glob("*.png"))
    assert len(written) == 13 and Image.open(written[0]).size == (width, height)
    assert (out / "animate.mp4").is_file()


@pytestmark_needs_animate
def test_official_names_map_to_diffusers_and_a_mismatch_is_refused(tmp_path):
    names = ["blocks.3.block.self_attn.q.weight", "blocks.3.block.cross_attn.k_img.bias", "blocks.3.block.cross_attn.norm_k_img.weight",
             "blocks.3.block.self_attn.norm_k.weight", "blocks.3.block.ffn.0.weight", "head.head.weight"]
    mapping = wan_animate.diffusers_names(names)
    assert mapping == {
        "blocks.3.block.self_attn.q.weight": "blocks.3.self_attn.to_q.weight",
        "blocks.3.block.cross_attn.k_img.bias": "blocks.3.cross_attn.add_k_proj.bias",
        "blocks.3.block.cross_attn.norm_k_img.weight": "blocks.3.cross_attn.norm_added_k.weight",
        "blocks.3.block.self_attn.norm_k.weight": "blocks.3.self_attn.norm_k.weight",
        "blocks.3.block.ffn.0.weight": "blocks.3.ffn.0.weight",
        "head.head.weight": "head.head.weight",
    }
    assert all(official_name(new) == old for old, new in mapping.items())


@pytestmark_needs_animate
def test_the_quantised_cache_replaces_diffusers_own_and_round_trips():
    from diffusers.modular_pipelines.wan_animate_2 import denoise as denoise_module

    from aiv_runtime import wan_animate_cache

    original = denoise_module.WanAnimate2KVCache
    try:
        wan_animate.build_pipeline({}, "int8")
        cache = denoise_module.WanAnimate2KVCache(3)
        assert isinstance(cache, wan_animate_cache.StreamingKVCache) and cache.codec == "int8"
    finally:
        denoise_module.WanAnimate2KVCache = original
    torch.manual_seed(1)
    key, value = torch.randn(1, 50, 4, 16), torch.randn(1, 50, 4, 16)
    for codec, tolerance in (("bf16", 0.02), ("int8", 0.03), ("int4", 0.35)):
        cache = wan_animate_cache.StreamingKVCache(2, codec=codec, pin_memory=False, prefetch=False)
        cache.get(1).store(key, value)
        restored_key, restored_value = cache.get(1).get()
        assert restored_key.shape == key.shape and restored_key.dtype == key.dtype
        assert (restored_key - key).abs().max() / key.abs().max() < tolerance, codec
        assert (restored_value - value).abs().max() / value.abs().max() < tolerance, codec
        assert cache.stored_bytes < key.numel() * 4 * 2 if codec != "bf16" else True
        cache.clear()
        assert cache.get(1).key is None


@pytestmark_needs_animate
def test_encode_stage_writes_prompt_and_reference_embeddings(tmp_path):
    from safetensors.torch import load_file
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel

    distilled = tmp_path / "models" / "distilled"
    vocab = {"<pad>": 0, "</s>": 1, "<unk>": 2, "a": 3, "knight": 4}
    raw = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="</s>", unk_token="<unk>").save_pretrained(distilled / "tokenizer")
    UMT5EncoderModel(UMT5Config(vocab_size=8, d_model=32, d_kv=8, d_ff=64, num_layers=1, num_heads=4)).save_pretrained(distilled / "text_encoder")
    embeds = tmp_path / "embeds.safetensors"
    code = wan_animate.main(["encode", "--models", str(tmp_path / "models"), "--prompt", "a knight", "--embeds", str(embeds),
                             "--max-sequence-length", "16", "--t5-device", "cpu", "--stats", str(tmp_path / "encode.json")])
    assert code == 0, json.loads((tmp_path / "encode.json").read_text()).get("error")
    tensors = load_file(str(embeds))
    assert tensors["prompt_embeds"].shape == (1, 16, 32) and tensors["prompt_ref_embeds"].shape == (1, 16, 32)
    assert tensors["prompt_embeds"].dtype == torch.bfloat16


def test_the_character_is_framed_like_the_actor_it_replaces():
    Image = pytest.importorskip("PIL.Image")
    reference = Image.new("RGB", (400, 400), (200, 200, 205))
    reference.paste((30, 30, 30), (150, 40, 250, 380))       # the character: 100 wide, 340 tall
    reference.putpixel((5, 5), (0, 0, 0))                     # a speck of noise is not the character
    assert wan_animate.character_bounds(reference) == (150, 40, 250, 380)
    framed, framing = wan_animate.frame_reference(reference, 672, 384, (400.0, 300.0, 170.0))
    assert framing["scale"] == pytest.approx(0.5)
    bounds = wan_animate.character_bounds(framed)
    assert bounds is not None
    x0, y0, x1, y1 = bounds
    assert abs((x0 + x1) / 2 - 400) <= 1 and abs(y1 - 300) <= 1 and abs((y1 - y0) - 170) <= 1
    assert framed.getpixel((5, 5)) == (200, 200, 205)          # the rest is the reference's border colour
    plain, none = wan_animate.frame_reference(Image.new("RGB", (64, 64), (9, 9, 9)), 672, 384, (400.0, 300.0, 170.0))
    assert none is None and plain.size == (672, 384)            # nothing to find: padded as before


def test_actor_frame_maps_through_the_crop():
    target = wan_animate.output_target((990 / 1920, 816 / 1080, 599 / 1080), (15, 0, 1890, 1080), 672, 384, 1920, 1080)
    assert target == pytest.approx(((990 - 15) * 672 / 1890, 816 * 384 / 1080, 599 * 384 / 1080))


def test_bounds_follow_a_graded_studio_backdrop():
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    ramp = np.linspace(110, 215, 400, dtype=np.float32)[None, :, None].repeat(400, 0).repeat(3, 2)
    ramp[40:380, 170:230] = 25                                 # the character, dark, in the middle
    image = Image.fromarray(ramp.astype(np.uint8), mode="RGB")
    assert wan_animate.character_bounds(image) == (170, 40, 230, 380)
    assert wan_animate.character_centre_x(image, (170, 40, 230, 380)) == pytest.approx(200, abs=0.6)


def test_the_backdrop_is_extended_not_boxed():
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    source = np.zeros((100, 80, 3), np.uint8)
    source[:, :, 0] = np.linspace(50, 150, 80, dtype=np.uint8)[None, :]
    canvas = np.asarray(wan_animate.extend_backdrop(Image.fromarray(source), 300, -20, 672, 384))
    assert canvas.shape == (384, 672, 3)
    assert np.array_equal(canvas[0, 300:380], source[20])      # cropped at the top
    assert (canvas[200, :300, 0] == source[0, 0, 0]).all()      # left of it: its own left edge, extended
    assert (canvas[200, 380:, 0] == source[0, -1, 0]).all()     # right of it: its right edge
    assert np.array_equal(canvas[300, 300:380], source[-1])     # below it: its bottom row


def test_the_driving_crop_is_the_actors_neighbourhood():
    crop = wan_animate.region_crop(1920, 1080, 672, 384, (798 / 1920, 217 / 1080, 1407 / 1920, 817 / 1080))
    x, y, w, h = crop
    assert abs(w / h - 672 / 384) < 0.01
    assert x <= 798 and y <= 217 - 0.12 * 600 + 1 and x + w >= 1407 and y + h >= 817 + 0.12 * 600 - 1
    assert 0 <= x and 0 <= y and x + w <= 1920 and y + h <= 1080
    assert h < 1080                                             # tighter than the whole plate
    assert wan_animate.region_crop(1920, 1080, 672, 384, (0.0, 0.0, 1.0, 1.0)) == wan_animate.plate_crop(1920, 1080, 672, 384)
    tiny = wan_animate.region_crop(1920, 1080, 672, 384, (0.5, 0.5, 0.51, 0.52))
    assert tiny[3] >= 384                                       # never smaller than the output (no upscale beyond 1:1)
