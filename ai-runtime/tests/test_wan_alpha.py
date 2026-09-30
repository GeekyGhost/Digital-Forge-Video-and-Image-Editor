"""Wan-Alpha worker (VFX Phase 0, task 0.6).

The pure helpers are tested everywhere. The bake is tested against upstream's own merge code on a
tiny model, which needs the pinned Wan-Alpha clone (vendor/Wan-Alpha/repo, or AIV_WAN_ALPHA_REPO)
plus torch and diffusers; without them those tests skip.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from aiv_runtime import wan_alpha, wan_alpha_bake  # noqa: E402

REPO = Path(os.environ.get("AIV_WAN_ALPHA_REPO", wan_alpha.DEFAULT_REPO))


def test_latent_geometry_matches_upstream():
    shape = wan_alpha.latent_shape(832, 480, 81)
    assert shape == (16, 21, 60, 104)
    assert wan_alpha.sequence_length(shape) == 32760
    assert wan_alpha.latent_shape(832, 480, 33)[1] == 9
    with pytest.raises(ValueError):
        wan_alpha.latent_shape(832, 480, 80)


def test_dora_key_mapping_is_upstreams():
    keys = [
        "diffusion_model.blocks.0.self_attn.q.lora_B.default.weight",
        "diffusion_model.blocks.0.self_attn.q.lora_A.default.weight",
        "diffusion_model.blocks.0.self_attn.q.lora_magnitude_vector.default.weight",
        "blocks.3.ffn.0.lora_B.weight",
    ]
    targets = wan_alpha_bake.dora_targets(keys)
    assert set(targets) == {"blocks.0.self_attn.q.weight", "blocks.3.ffn.0.weight"}
    assert targets["blocks.0.self_attn.q.weight"]["magnitude"] == keys[2]
    assert targets["blocks.3.ffn.0.weight"]["lora_A"] == "blocks.3.ffn.0.lora_A.weight"


def test_lightx2v_plan_covers_pairs_and_diffs():
    keys = ["diffusion_model.blocks.0.ffn.0.lora_down.weight", "diffusion_model.blocks.0.ffn.0.lora_up.weight",
            "diffusion_model.blocks.0.diff_m", "diffusion_model.blocks.1.norm3.diff", "diffusion_model.blocks.1.self_attn.q.diff_b"]
    plan = wan_alpha_bake.lora_plan(keys)
    assert plan.pairs["blocks.0.ffn.0.weight"] == (keys[0], keys[1])
    assert plan.diffs["blocks.0.modulation"] == keys[2]
    assert plan.diffs["blocks.1.norm3.weight"] == keys[3]
    assert plan.diffs["blocks.1.self_attn.q.bias"] == keys[4]


def test_mask_is_resized_like_upstream_and_resampled_in_time():
    mask = torch.rand(21, 96, 170)
    same = wan_alpha.shaped_mask(mask, 21, 60, 104)
    expected = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(60, 104), mode="bilinear", align_corners=False).squeeze(1)
    assert torch.equal(same, expected)
    short = wan_alpha.shaped_mask(mask, 9, 60, 104)
    assert short.shape == (9, 60, 104)
    # Ends of the resampled time axis are the original first and last frames.
    first = torch.nn.functional.interpolate(mask[:1].unsqueeze(1), size=(60, 104), mode="bilinear", align_corners=False).squeeze(1)
    assert torch.allclose(short[:1], first, atol=1e-6)


def test_rgba_frames_follow_upstream_rounding():
    fgr = torch.tensor([-1.0, 0.0, 1.0, 2.0]).view(1, 4, 1, 1).expand(3, 4, 1, 1)
    pha = torch.stack([torch.full((4, 1, 1), -1.0), torch.full((4, 1, 1), 0.0), torch.full((4, 1, 1), 1.0)])
    frames = wan_alpha.rgba_frames(fgr, pha)
    assert frames.shape == (4, 1, 1, 4)
    assert list(frames[:, 0, 0, 0]) == [0, 127, 255, 255]  # truncation, then clamp
    assert int(frames[0, 0, 0, 3]) == int((0 + 127 + 255) / 3)


def test_checkerboard_composite():
    import numpy as np

    rgba = np.zeros((1, 16, 16, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[0, 0, 0, 3] = 255
    out = wan_alpha.checkerboard_composite(rgba)
    assert tuple(out[0, 0, 0]) == (255, 0, 0)
    assert tuple(out[0, 0, 1]) == (200, 200, 200)
    assert tuple(out[0, 0, 8]) == (255, 255, 255)


def test_sdpa_stand_in_matches_naive_attention():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 12, 4, 16) for _ in range(3))
    out = wan_alpha.sdpa_attention(q, k, v, k_lens=torch.tensor([12]), dtype=torch.float32)
    weights = torch.softmax(torch.einsum("blnc,bmnc->bnlm", q, k) / math.sqrt(16), dim=-1)
    naive = torch.einsum("bnlm,bmnc->blnc", weights, v)
    assert torch.allclose(out, naive, atol=1e-5)
    with pytest.raises(NotImplementedError):
        wan_alpha.sdpa_attention(q, k, v, k_lens=torch.tensor([7]))


# ------------------------------------------------------------------------------------------------
# Bake vs upstream, on a tiny Wan model.
# ------------------------------------------------------------------------------------------------

TINY = dict(model_type="t2v", patch_size=(1, 2, 2), text_len=16, in_dim=16, dim=64, ffn_dim=128, freq_dim=32,
            text_dim=48, out_dim=16, num_heads=4, num_layers=2, window_size=(-1, -1), qk_norm=True,
            cross_attn_norm=True, eps=1e-6)

needs_repo = pytest.mark.skipif(not (REPO / "wan" / "modules" / "model_dora_lightx2v.py").is_file(),
                                reason=f"Wan-Alpha clone not found at {REPO}")


def _upstream():
    # Upstream's model module and the bake need these; the app venv may not have them (the
    # Wan-Alpha venv does), so skip rather than fail there.
    for name in ("diffusers", "accelerate", "einops", "safetensors"):
        pytest.importorskip(name)
    try:
        return wan_alpha.wan_module(REPO, "modules.model_dora_lightx2v")
    except ImportError as error:
        pytest.skip(f"upstream Wan-Alpha modules need {error.name}")


def _fixtures(tmp_path: Path):
    from safetensors.torch import save_file

    module = _upstream()
    torch.manual_seed(1)
    base = module.WanModel_dora_lightx2v(**TINY)
    for param in base.parameters():
        param.data.normal_(0, 0.05)
    state = {k: v.detach().clone().contiguous() for k, v in base.state_dict().items()}
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    names = sorted(state)
    halves = {"diffusion_pytorch_model-00001-of-00002.safetensors": names[: len(names) // 2],
              "diffusion_pytorch_model-00002-of-00002.safetensors": names[len(names) // 2:]}
    weight_map = {}
    for file, keys in halves.items():
        save_file({k: state[k] for k in keys}, str(base_dir / file))
        weight_map.update({k: file for k in keys})
    (base_dir / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    (base_dir / "config.json").write_text(json.dumps({**TINY, "_class_name": "WanModel"}))

    rank = 4
    dora = {}
    for target in ("blocks.0.self_attn.q", "blocks.1.ffn.0", "blocks.1.cross_attn.o"):
        out_features, in_features = state[f"{target}.weight"].shape
        dora[f"diffusion_model.{target}.lora_A.default.weight"] = (torch.randn(rank, in_features) * 0.1).to(torch.bfloat16)
        dora[f"diffusion_model.{target}.lora_B.default.weight"] = (torch.randn(out_features, rank) * 0.1).to(torch.bfloat16)
        dora[f"diffusion_model.{target}.lora_magnitude_vector.default.weight"] = (torch.rand(out_features) + 0.5).to(torch.bfloat16)
    lightx2v = {
        # Overlaps a DoRA target: upstream applies DoRA first, then this.
        "diffusion_model.blocks.0.self_attn.q.lora_down.weight": (torch.randn(rank, 64) * 0.1).to(torch.bfloat16),
        "diffusion_model.blocks.0.self_attn.q.lora_up.weight": (torch.randn(64, rank) * 0.1).to(torch.bfloat16),
        "diffusion_model.blocks.1.ffn.2.lora_A.weight": (torch.randn(rank, 128) * 0.1).to(torch.bfloat16),
        "diffusion_model.blocks.1.ffn.2.lora_B.weight": (torch.randn(64, rank) * 0.1).to(torch.bfloat16),
        "diffusion_model.blocks.0.diff_m": (torch.randn(1, 6, 64) * 0.01).to(torch.bfloat16),
        "diffusion_model.blocks.1.norm3.diff": (torch.randn(64) * 0.01).to(torch.bfloat16),
        "diffusion_model.blocks.1.self_attn.q.diff_b": (torch.randn(64) * 0.01).to(torch.bfloat16),
    }
    save_file(dora, str(tmp_path / "t2v.safetensors"))
    save_file(lightx2v, str(tmp_path / "lightx2v.safetensors"))

    expected = module.WanModel_dora_lightx2v(**TINY)
    expected.load_state_dict(state)
    expected.load_dora(str(tmp_path / "t2v.safetensors"), 1.0)
    expected.load_lightx2v_lora(str(tmp_path / "lightx2v.safetensors"))
    return module, base_dir, expected


@needs_repo
def test_streamed_bake_equals_upstream_merge(tmp_path):
    from accelerate import init_empty_weights

    from aiv_runtime.safetensors_reader import load_state_dict

    module, base_dir, expected = _fixtures(tmp_path)
    with init_empty_weights():
        meta = module.WanModel_dora_lightx2v(**TINY)
    fp8_names = wan_alpha_bake.fp8_parameter_names(meta)
    assert "blocks.0.self_attn.q.weight" in fp8_names
    assert not any("norm" in name or "patch_embedding" in name or "time_embedding" in name or name.startswith("head") for name in fp8_names)
    assert "blocks.1.ffn.2.weight" in fp8_names and "text_embedding.0.weight" not in fp8_names

    out = tmp_path / "baked"
    manifest = wan_alpha_bake.bake(base_dir, tmp_path / "t2v.safetensors", tmp_path / "lightx2v.safetensors", out, fp8_names, log=lambda _: None)
    assert manifest["doraApplied"] == 3
    assert manifest["lightx2vApplied"] == 5
    files = wan_alpha_bake.baked_files(out)
    assert files and len(files) == 2
    baked = {}
    for path in files:
        baked.update(load_state_dict(path))

    reference = expected.state_dict()
    assert set(baked) == set(reference)
    for name, tensor in reference.items():
        if name in fp8_names:
            assert baked[name].dtype == torch.float8_e4m3fn, name
            assert torch.equal(baked[name].view(torch.uint8), tensor.to(torch.float8_e4m3fn).view(torch.uint8)), name
        else:
            assert baked[name].dtype == tensor.dtype, name
            assert torch.equal(baked[name], tensor), name


@needs_repo
def test_baked_model_runs_with_layerwise_casting_close_to_upstream(tmp_path):
    from accelerate import init_empty_weights

    from aiv_runtime.safetensors_reader import load_state_dict

    module, base_dir, expected = _fixtures(tmp_path)
    module.flash_attention = wan_alpha.sdpa_attention
    with init_empty_weights():
        meta = module.WanModel_dora_lightx2v(**TINY)
    fp8_names = wan_alpha_bake.fp8_parameter_names(meta)
    out = tmp_path / "baked"
    wan_alpha_bake.bake(base_dir, tmp_path / "t2v.safetensors", tmp_path / "lightx2v.safetensors", out, fp8_names, log=lambda _: None)

    with init_empty_weights():
        model = module.WanModel_dora_lightx2v(**TINY)
    state = {}
    for path in wan_alpha_bake.baked_files(out):
        state.update(load_state_dict(path))
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    model.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16,
                                   skip_modules_pattern=wan_alpha_bake.SKIP_LAYERWISE_CASTING)
    assert {n for n, p in model.named_parameters() if p.dtype == torch.float8_e4m3fn} == fp8_names

    # Upstream wraps the time embedding in `torch.cuda.amp.autocast(dtype=float32)` and asserts it
    # stays fp32; on CPU that context is inert, so emulate it (fp32 region) for the comparison.
    # This is also why the time/text embedders must stay out of FP8 storage: cast to BF16 compute
    # they would trip that assertion on the GPU.
    import contextlib
    import types

    module.amp = types.SimpleNamespace(autocast=lambda **_: torch.autocast("cpu", enabled=False) if _.get("dtype") == torch.float32 else contextlib.nullcontext())
    torch.manual_seed(3)
    shape = (16, 3, 8, 8)  # 9 frames of 64x64
    seq_len = wan_alpha.sequence_length(shape)
    x = torch.randn(*shape)
    context = torch.randn(10, 48)
    t = torch.tensor([500.0])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        ours = model([x], t=t, context=[context], seq_len=seq_len)[0].float()
        theirs = expected.eval()([x], t=t, context=[context], seq_len=seq_len)[0].float()
    relative = (ours - theirs).norm() / theirs.norm()
    # Measured 0.0087 in the container (2026-09-27); 0.08 leaves room for other torch builds.
    assert relative < 0.08, f"FP8 storage drifted {relative:.3f} from upstream"
