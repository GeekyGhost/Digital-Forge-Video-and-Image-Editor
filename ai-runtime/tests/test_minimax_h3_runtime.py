"""Pins H3 sampling and memory-profile behavior without loading any model weights."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import wave
from argparse import Namespace
from pathlib import Path

import pytest

from aiv_runtime import minimax_h3, minimax_h3_convrot, minimax_h3_ref
from aiv_runtime.minimax_h3 import (
    OFFLOAD_ALIASES,
    OFFLOAD_PROFILES,
    TURBO_AUDIO_SHIFT,
    TURBO_LORA_ALPHA,
    TURBO_NFE,
    TURBO_VIDEO_SHIFT,
    VAE_TILE_OVERLAP,
    VAE_TILE_SIZE,
    VIDEO_VAE_OFFLOAD_TYPE,
    EmbeddedLoraLinear,
    PortableFP8Linear,
    _cast_turbo_lora_parameters_bf16,
    _checkpoint_shards,
    _guard_h3_sage_dispatch,
    _install_curve_adaln,
    _prepare_portable_fp8_transformer,
    decode_video_frames_to_cpu,
    declared_lora_alpha,
    effective_offload_profile,
    ensure_mm_token_type_ids,
    install_qwen3vl_pos_embed_device_shim,
    scheduler_grid_points,
    temporal_decode_plan,
    turbo_lora_alpha,
    turbo_shifts,
)
from aiv_runtime.minimax_h3_ref import (
    H3_AUDIO_SAMPLE_RATE,
    H3_FPS,
    H3_REF_IMAGE_SHORT_EDGE,
    H3_REF_VIDEO_MAX_PIXELS,
    H3_REF_VIDEO_SHORT_EDGE,
    ReferenceDecodeBudget,
    _audio_reference_at_h3_rate,
    load_references,
    qwen_conditioner_attention_context,
    ref_pipeline,
)
from aiv_runtime.minimax_h3_convrot import LORA_A_SUFFIX
from aiv_runtime.minimax_h3_convrot_text_encoder import (
    PUBLISHED_METADATA,
    _validate_metadata,
    native_to_transformers,
)


def test_turbo_adds_the_terminal_scheduler_point():
    args = Namespace(sampling_mode="turbo-8", steps=TURBO_NFE["turbo-8"])
    assert scheduler_grid_points(args) == 9
    assert scheduler_grid_points(args) - 1 == TURBO_NFE["turbo-8"]


def test_both_h3_workers_accept_opt_in_denoise_profile_paths(tmp_path):
    base_args = minimax_h3.build_parser().parse_args(
        [
            "--repo-root", str(tmp_path),
            "--prompt", "test",
            "--output-path", str(tmp_path / "base.mp4"),
            "--profile-path", str(tmp_path / "base-profile.jsonl"),
        ]
    )
    assert base_args.profile_path.endswith("base-profile.jsonl")

    ref_args = minimax_h3_ref.parser().parse_args(
        [
            "--repo-root", str(tmp_path),
            "--transformer-path", str(tmp_path / "model.safetensors"),
            "--text-encoder-path", str(tmp_path / "encoder.safetensors"),
            "--references-json", "[]",
            "--prompt", "test",
            "--output-path", str(tmp_path / "ref.mp4"),
            "--audio-path", str(tmp_path / "ref.wav"),
            "--width", "960",
            "--height", "544",
            "--num-frames", "124",
            "--profile-denoise",
        ]
    )
    assert ref_args.profile_denoise is True


def test_draft_adds_the_terminal_scheduler_point():
    args = Namespace(sampling_mode="turbo-4", steps=TURBO_NFE["turbo-4"])
    assert scheduler_grid_points(args) == 5
    assert scheduler_grid_points(args) - 1 == TURBO_NFE["turbo-4"]


def test_base_sampling_also_adds_the_terminal_scheduler_point():
    assert scheduler_grid_points(Namespace(sampling_mode="base", steps=50)) == 51


def test_four_block_profile_spends_memory_on_transformer_throughput():
    performance = OFFLOAD_PROFILES["performance"]
    assert performance == {
        "blocks_per_group": 4,
        "use_stream": False,
        "record_stream": False,
        "low_cpu_mem_usage": True,
        "vae_resident": False,
    }


def test_sm89_fp8_fallback_disables_streamed_prefetch_only():
    assert effective_offload_profile("balanced", "native")["use_stream"] is True
    fallback = effective_offload_profile("balanced", "weight-only")
    assert fallback["blocks_per_group"] == 1
    assert fallback["use_stream"] is False
    assert fallback["record_stream"] is False
    assert effective_offload_profile("performance", "weight-only")["blocks_per_group"] == 4


def test_portable_fp8_linear_matches_explicit_source_dequantization():
    import torch

    qdata = torch.tensor(
        [[1.0, -2.0, 0.5], [3.0, 1.0, -1.0]], dtype=torch.float8_e4m3fn
    )
    scale = torch.tensor([[0.25], [0.5]], dtype=torch.float32)
    module = PortableFP8Linear(qdata, scale, torch.tensor([2, 3], dtype=torch.int32))
    inputs = torch.tensor([[2.0, -1.0, 4.0]], dtype=torch.float32)

    expected_weight = qdata.float() * scale.float()
    expected = torch.nn.functional.linear(inputs, expected_weight)
    torch.testing.assert_close(module(inputs), expected)
    assert module.weight.dtype == torch.bfloat16
    assert type(module.qdata) is torch.nn.Parameter
    assert type(module.qdata.data) is torch.Tensor


def test_portable_loader_installs_complete_embedded_lora(monkeypatch):
    import torch
    from torch import nn

    monkeypatch.delenv("AIV_TURBO_LORA_ALPHA", raising=False)
    model = nn.Sequential(nn.Linear(3, 2, bias=False))
    state = {
        "0.aiv_fp8.qdata": torch.tensor(
            [[1.0, -2.0, 0.5], [3.0, 1.0, -1.0]], dtype=torch.float8_e4m3fn
        ),
        "0.aiv_fp8.scale": torch.tensor([[0.25], [0.5]], dtype=torch.float32),
        "0.aiv_fp8.orig_shape": torch.tensor([2, 3], dtype=torch.int32),
        "0.aiv_fp8.lora_A": torch.tensor([[1.0, 0.0, -1.0]], dtype=torch.bfloat16),
        "0.aiv_fp8.lora_B": torch.tensor([[2.0], [-1.0]], dtype=torch.bfloat16),
        "0.aiv_fp8.lora_scale": torch.tensor(0.125, dtype=torch.float32),
    }

    count, remaining = _prepare_portable_fp8_transformer(model, state)
    assert count == 1
    assert remaining == {}
    assert isinstance(model[0], EmbeddedLoraLinear)
    assert isinstance(model[0].base, PortableFP8Linear)

    inputs = torch.tensor([[2.0, -1.0, 4.0]], dtype=torch.bfloat16)
    base_weight = (state["0.aiv_fp8.qdata"].float() * state["0.aiv_fp8.scale"].float()).to(
        torch.bfloat16
    )
    expected_base = torch.nn.functional.linear(inputs, base_weight)
    expected_lora = torch.nn.functional.linear(
        torch.nn.functional.linear(inputs, state["0.aiv_fp8.lora_A"]),
        state["0.aiv_fp8.lora_B"],
    )
    torch.testing.assert_close(model(inputs), expected_base + expected_lora * 0.125)


def test_baked_checkpoint_does_not_require_external_lora_file(tmp_path):
    import torch
    from safetensors.torch import save_file

    checkpoint = tmp_path / "portable.safetensors"
    encoder = tmp_path / "conditioner.safetensors"
    save_file(
        {"stub": torch.zeros(1)},
        str(checkpoint),
        metadata={"is_turbo_baked": "True"},
    )
    save_file({"stub": torch.zeros(1)}, str(encoder))
    args = Namespace(
        width=960,
        height=544,
        num_frames=124,
        prompt="test",
        transformer_path=str(checkpoint),
        turbo_lora_path=str(tmp_path / "not-installed.safetensors"),
        text_encoder_path=str(encoder),
        vae_tile=VAE_TILE_SIZE,
        stage="all",
        embeds_path=None,
        latents_path=None,
        references_json='[{"kind":"image"}]',
        # `validate` gained these when Ref2VA got Base sampling, and this test kept the older
        # namespace -- so it raised AttributeError on the line it was written to exercise, which
        # reads as a pass to nobody but is not the assertion it claims to make. Turbo is the mode
        # that matters here: a baked residual is what excuses the missing adapter file.
        sampling_mode="turbo-4",
        steps=4,
    )

    minimax_h3_ref.validate(args)

    # And the same checkpoint under Base, which needs no adapter at all and must not ask for one.
    args.sampling_mode = "base"
    args.steps = 20
    args.turbo_lora_path = None
    minimax_h3_ref.validate(args)


def test_reference_encoder_vae_offloads_every_projection_boundary():
    """Leaf offload remains limited to Stage 1 reference encoding, never output decode."""
    assert VIDEO_VAE_OFFLOAD_TYPE == "leaf_level"


def test_turbo_lora_parameters_are_recast_without_touching_parameter_identity():
    import torch
    from torch import nn

    fp32 = nn.Parameter(torch.ones(2, 2, dtype=torch.float32))
    bf16 = nn.Parameter(torch.ones(2, 2, dtype=torch.bfloat16))
    fp32_identity = id(fp32)

    assert _cast_turbo_lora_parameters_bf16({"a": fp32, "b": bf16}) == 1
    assert id(fp32) == fp32_identity
    assert fp32.dtype == torch.bfloat16
    assert bf16.dtype == torch.bfloat16


def test_h3_sage_guard_falls_back_per_call_without_changing_compatible_calls():
    import torch

    calls = []

    def dispatch(query, key, value, *args, backend=None, **kwargs):
        calls.append((backend, kwargs.get("attn_mask")))
        return "ok"

    guarded = _guard_h3_sage_dispatch(dispatch)
    cuda = Namespace(type="cuda")
    fp32 = Namespace(dtype=torch.float32, device=cuda)
    bf16 = Namespace(dtype=torch.bfloat16, device=cuda)

    assert guarded(fp32, fp32, fp32, backend="sage") == "ok"
    assert calls[-1][0] == "native"
    assert guarded(bf16, bf16, bf16, backend="sage") == "ok"
    assert calls[-1][0] == "sage"
    assert guarded(bf16, bf16, bf16, backend="sage", attn_mask=object()) == "ok"
    assert calls[-1][0] == "native"


def test_baked_encoder_shards_keep_index_order_and_unique_names(tmp_path):
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(json.dumps({"weight_map": {
        "layer.0.weight": "part-02.bin",
        "layer.0.bias": "part-02.bin",
        "embed.weight": "part-01.bin",
    }}), encoding="utf-8")
    shards, keys = _checkpoint_shards(index)
    assert shards == ["part-02.bin", "part-01.bin"]
    assert keys == {"layer.0.weight", "layer.0.bias", "embed.weight"}


def test_transformers_457_processor_gets_upstream_multimodal_type_ids():
    processor = type("Processor457", (), {"image_token_id": 101, "video_token_id": 202})()
    assert ensure_mm_token_type_ids(processor)
    assert processor.create_mm_token_type_ids([[7, 101, 8, 202]]) == [[0, 1, 0, 2]]
    assert not ensure_mm_token_type_ids(processor)


def test_curve_checkpoint_rebuilds_the_pruned_adaln_architecture():
    import torch
    import torch.nn as nn

    class AdaLN(nn.Module):
        def __init__(self, hidden_size, output_multiplier):
            super().__init__()
            self.hidden_size = hidden_size
            self.linear = nn.Linear(5, output_multiplier * hidden_size * 3)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.adaln_proj = AdaLN(2, 6)

    class OutputAdaLN(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.RMSNorm(2)
            self.linear = nn.Linear(5, 4)

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.time_proj = nn.Linear(1, 1)
            self.time_embedder = nn.Linear(1, 5)
            self.transformer_blocks = nn.ModuleList([Block()])
            self.norm_out = OutputAdaLN()

    transformer = Transformer()
    table = torch.tensor([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]])
    state = {
        "time_embedder.table": table,
        "transformer_blocks.0.adaln_proj.linear.weight": torch.empty((36, 2)),
        "norm_out.linear.weight": torch.empty((4, 2)),
    }
    metadata = {
        "family": "minimax-h3",
        "scheme": "fp8",
        "quant_backend": "torchao",
        "transformer_class": "MiniMaxH3Transformer3DModel",
        "adaln_form": "curve",
        "curve_dim": 2,
        "curve_grid": 3,
        "adaln_out_dtype": "bfloat16",
    }
    _install_curve_adaln(
        transformer,
        "unsloth_prequant_transformer_state_dict_v1",
        metadata,
        state,
    )
    transformer.time_embedder.table = table

    assert isinstance(transformer.time_proj, nn.Identity)
    assert transformer.transformer_blocks[0].adaln_proj.linear.weight.shape == (36, 2)
    assert transformer.norm_out.linear.weight.shape == (4, 2)
    assert torch.equal(transformer.time_embedder(torch.tensor([0.25])), torch.tensor([[5.0, 10.0]]))


def test_ref2va_audio_is_normalized_without_torchaudio(tmp_path):
    source = tmp_path / "reference-48k.wav"
    with wave.open(str(source), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(48_000)
        output.writeframes(b"\x00\x00" * 48_000)

    class WaveReference:
        @classmethod
        def from_file(cls, path):
            with wave.open(str(path), "rb") as audio:
                value = cls()
                value.sample_rate = audio.getframerate()
                value.frames = audio.getnframes()
                value.audio = b"decoded"
                return value

    normalized = _audio_reference_at_h3_rate(str(source), WaveReference)
    assert normalized.sample_rate == H3_AUDIO_SAMPLE_RATE
    assert abs(normalized.frames - H3_AUDIO_SAMPLE_RATE) <= 1


def test_ref2va_video_uses_the_official_ordered_frame_loader(monkeypatch):
    import sys
    from types import ModuleType

    calls = []

    class VideoReference:
        @classmethod
        def from_file(cls, path):
            import numpy as np

            calls.append(path)
            return Namespace(
                frames=np.zeros((48, 32, 64, 3), dtype=np.uint8),
                fps=24.0,
                audio=None,
                sample_rate=None,
            )

    minimax_module = ModuleType("diffusers.modular_pipelines.minimax_h3")
    minimax_module.__path__ = []
    references_module = ModuleType("diffusers.modular_pipelines.minimax_h3.references")
    references_module.MiniMaxH3AudioReference = object
    references_module.MiniMaxH3ImageReference = object
    references_module.MiniMaxH3VideoReference = VideoReference
    minimax_module.references = references_module
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3", minimax_module)
    monkeypatch.setitem(sys.modules, "diffusers.modular_pipelines.minimax_h3.references", references_module)
    values = load_references(json.dumps([{
        "kind": "video",
        "path": "motion-reference.mp4",
        "includeEmbeddedAudio": False,
    }]))

    assert calls == ["motion-reference.mp4"]
    assert values[0].frames.shape == (48, 32, 64, 3)
    assert values[0].fps == 24.0


def test_both_h3_workers_share_qwen_position_interpolation_onload_fix():
    import torch
    import torch.nn as nn

    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_grid_per_side = 4
            self.pos_embed = nn.Embedding(16, 6)
            self.config = Namespace(spatial_merge_size=2)

        def fast_pos_embed_interpolate(self, _grid):
            raise AssertionError("the incompatible Transformers method was not replaced")

    model = Namespace(visual=Visual())
    assert install_qwen3vl_pos_embed_device_shim(model, "cpu")
    output = model.visual.fast_pos_embed_interpolate(torch.tensor([[1, 4, 4]]))
    assert output.shape == (16, 6)
    assert output.device.type == "cpu"
    assert not install_qwen3vl_pos_embed_device_shim(model, "cpu")

    base_encode = Path(minimax_h3.__file__).read_text(encoding="utf-8").split(
        "def run_encode("
    )[1].split("\ndef ")[0]
    assert base_encode.index("apply_group_offloading(") < base_encode.index(
        "install_qwen3vl_pos_embed_device_shim("
    )
    ref_encode = Path(minimax_h3_ref.__file__).read_text(encoding="utf-8").split(
        "def run_encode("
    )[1].split("\ndef ")[0]
    assert "base.install_qwen3vl_pos_embed_device_shim(" in ref_encode


def test_qwen_position_interpolation_survives_real_cuda_leaf_offload():
    import torch
    import torch.nn as nn
    from diffusers.hooks import apply_group_offloading

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required to exercise the CPU/CUDA offload boundary")

    class Visual(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_grid_per_side = 4
            self.pos_embed = nn.Embedding(16, 6)
            self.config = Namespace(spatial_merge_size=2)

        def fast_pos_embed_interpolate(self, _grid):
            raise AssertionError("the incompatible Transformers method was not replaced")

    model = Namespace(visual=Visual())
    apply_group_offloading(
        model.visual,
        offload_type="leaf_level",
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
    )
    assert install_qwen3vl_pos_embed_device_shim(model, "cuda")
    output = model.visual.fast_pos_embed_interpolate(torch.tensor([[1, 4, 4]]))
    assert output.shape == (16, 6)
    assert output.device.type == "cuda"
    assert model.visual.pos_embed.weight.device.type == "cpu"


def test_ref2va_16gb_profile_caps_the_shared_reference_video_canvas(monkeypatch):
    class Blocks:
        def __init__(self):
            self.sub_blocks = {
                "before_encode": object(),
                "text_encoder": object(),
                "decode.video": object(),
            }

        def get_workflow(self, name):
            assert name == "ref2va"
            return self

        def init_pipeline(self, repo_root):
            assert repo_root == "model"
            return Namespace(config=Namespace(
                reference_image_short_edge=2048,
                canvas_short_edge=768,
                canvas_max_pixels=768 * 1344,
            ))

    fake_module = Namespace(from_pretrained=lambda _root: Namespace(blocks=Blocks()))
    monkeypatch.setattr("diffusers.ModularPipeline", fake_module)

    pipeline = ref_pipeline(
        Namespace(repo_root="model", width=960, height=544),
        {"before_encode", "text_encoder"},
    )

    # Videos keep the conservative shared canvas; images follow the generated canvas's own short
    # edge (ComfyUI's `match`), which is 544 here -- NOT the video canvas, and not clamped to it.
    assert pipeline.config.reference_image_short_edge == 544
    assert pipeline.config.canvas_short_edge == H3_REF_VIDEO_SHORT_EDGE
    assert pipeline.config.canvas_max_pixels == H3_REF_VIDEO_MAX_PIXELS
    assert H3_REF_IMAGE_SHORT_EDGE is None, "the image short edge is `match` by default"


def test_an_explicit_reference_video_tier_is_not_clamped_back_to_the_canvas(monkeypatch):
    """The panel's reference-video tier has to survive reaching the pipeline.

    The conservative 384 default is held under the generated canvas, which is right for a fallback.
    Applying that same clamp to an explicit request would make the 768 tier inert at 544p -- one
    more control that looks live and changes nothing, which is the fault class this pass removed.
    ComfyUI does not clamp a reference video to the generation at all; it fits it under the same
    768x1344 budget either way.
    """
    class Blocks:
        def __init__(self):
            self.sub_blocks = {"before_encode": object(), "text_encoder": object()}

        def get_workflow(self, name):
            return self

        def init_pipeline(self, repo_root):
            return Namespace(config=Namespace(
                reference_image_short_edge=2048, canvas_short_edge=768, canvas_max_pixels=768 * 1344,
            ))

    monkeypatch.setattr(
        "diffusers.ModularPipeline", Namespace(from_pretrained=lambda _root: Namespace(blocks=Blocks())),
    )
    keep = {"before_encode", "text_encoder"}

    asked = ref_pipeline(
        Namespace(repo_root="model", width=960, height=544, ref_video_short_edge=768), keep,
    )
    assert asked.config.canvas_short_edge == 768, "an explicit tier is honoured above the canvas"
    assert asked.config.canvas_max_pixels == 768 * 1344, "and carries upstream's own area budget"

    default = ref_pipeline(Namespace(repo_root="model", width=960, height=544), keep)
    assert default.config.canvas_short_edge == H3_REF_VIDEO_SHORT_EDGE
    assert default.config.canvas_max_pixels == H3_REF_VIDEO_MAX_PIXELS

    monkeypatch.setenv("AIV_H3_REF_VIDEO_SHORT_EDGE", "544")
    from_env = ref_pipeline(Namespace(repo_root="model", width=960, height=544), keep)
    assert from_env.config.canvas_short_edge == 544, "the environment counts as an explicit request"


def test_ref2va_conditioner_attention_keeps_native_cpu_fallback():
    import contextlib

    fake_torch = Namespace(cuda=Namespace(is_available=lambda: False))
    assert isinstance(qwen_conditioner_attention_context(fake_torch), contextlib.nullcontext)

def test_ref2va_denoise_asks_for_the_terminal_scheduler_point():
    """Ref2VA must resolve its grid through the shared helper, not hardcode the NFE count.

    minimax_h3_ref.run_denoise previously passed num_inference_steps=4 directly, which ran the
    four-evaluation Ref2V LoRA at three transformer evaluations. main() sets sampling_mode and
    steps specifically so scheduler_grid_points can answer this; the regression was that nothing
    called it.
    """
    source = Path(minimax_h3_ref.__file__).read_text(encoding="utf-8")
    assert "num_inference_steps=base.scheduler_grid_points(args)" in source, (
        "Ref2VA denoise must resolve grid points through the shared helper"
    )
    assert "num_inference_steps=4" not in source, "the hardcoded NFE count is back"

    args = Namespace(sampling_mode="turbo-4", steps=4)
    assert scheduler_grid_points(args) == 5


def test_ref2va_passes_its_real_nfe_to_the_lora_loader():
    """apply_turbo_lora's nfe argument exists to be reported; Ref2VA must not pass a literal."""
    source = Path(minimax_h3_ref.__file__).read_text(encoding="utf-8")
    assert "base.apply_turbo_lora(transformer, args.turbo_lora_path, base.TURBO_NFE[args.sampling_mode])" in source


def _setup_step_resample(frames, fps: float, num_frames: int):
    """`before_encoder._normalize_video_condition`'s frame arithmetic, and nothing else.

    Copied deliberately rather than imported: this is the vendored behaviour
    ReferenceDecodeBudget claims to invert, so a test that called the real thing would move with
    it instead of catching it moving.
    """
    import math

    import numpy as np

    frames = np.asarray(frames)
    if fps != H3_FPS:
        scale = H3_FPS / fps
        slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
        frames = np.repeat(
            frames,
            np.diff(slots, append=math.floor(frames.shape[0] * scale + 0.5)),
            axis=0,
        )
    return frames[:num_frames]


def test_reference_frame_budget_keeps_every_frame_the_setup_step_reads():
    """Truncating the decode must be invisible downstream, at every rate a camera produces.

    The budget exists so a 1080p reference does not decode a gigabyte of frames the request
    discards -- but a budget that came up one frame short would silently shorten the reference
    instead, which is a different request with nothing raised about it.
    """
    import numpy as np

    budget_rule = (32, H3_REF_VIDEO_SHORT_EDGE, H3_REF_VIDEO_MAX_PIXELS)
    for num_frames in (22, 124, 141, 226):
        budget = ReferenceDecodeBudget(num_frames, *budget_rule)
        for fps in (12.0, 15.0, 23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0):
            total = int(fps * 15)
            limit = budget.source_frames(fps)
            assert limit < total, f"{fps} fps: the budget did not bound a 15-second source"
            whole = _setup_step_resample(np.arange(total), fps, num_frames)
            bounded = _setup_step_resample(np.arange(limit), fps, num_frames)
            assert len(whole) == num_frames, f"{fps} fps: the fixture is too short to be a test"
            assert np.array_equal(bounded, whole), f"{fps} fps, {num_frames} frames: short by "\
                f"{num_frames - len(bounded)}"


def test_reference_frame_budget_leaves_a_short_source_alone():
    budget = ReferenceDecodeBudget(124, 32, H3_REF_VIDEO_SHORT_EDGE, H3_REF_VIDEO_MAX_PIXELS)
    assert budget.source_frames(30.0) > 60
    try:
        budget.source_frames(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("a zero frame rate must not resolve to a frame budget")


def test_reference_canvas_declines_a_rescale_it_cannot_prove(monkeypatch):
    """Pre-scaling at decode time is only transparent where the canvas is a fixed point.

    The setup step resolves the canvas from whatever geometry it is handed. If it would resolve a
    pre-scaled reference onto something else, the request ends up conditioned on a twice-rescaled
    reference, so `canvas_for` has to decline rather than assume.
    """
    import sys
    from types import ModuleType

    budget = ReferenceDecodeBudget(124, 32, 384, 384 * 672)

    def install(resolve):
        module = ModuleType("diffusers.modular_pipelines.minimax_h3.modular_pipeline")
        module.resolve_canvas_size = resolve
        monkeypatch.setitem(
            sys.modules, "diffusers.modular_pipelines.minimax_h3.modular_pipeline", module
        )

    install(lambda width, height, *_rule: (384, 672))
    assert budget.canvas_for(1920, 1080) == (384, 672)

    # Resolves once, then somewhere else from its own output: not a fixed point.
    install(lambda width, height, *_rule: (384, 672) if width > 700 else (352, 640))
    assert budget.canvas_for(1920, 1080) is None

    def out_of_range(width, height, *_rule):
        raise ValueError("aspect ratio out of range")

    install(out_of_range)
    assert budget.canvas_for(4000, 200) is None


def test_ref2va_decodes_its_references_before_the_denoiser_is_resident():
    """Ref2VA pre-encodes references in Stage 1 so raw media is never resident with the denoiser."""
    source = Path(minimax_h3_ref.__file__).read_text(encoding="utf-8")
    assert "references=load_references(" not in source, (
        "a reference decode is back inside a pipeline call's argument list"
    )
    denoise = source.split("def run_denoise(")[1].split("\ndef ")[0]
    assert "load_references(" not in denoise, (
        "run_denoise must not load raw reference media; references must be pre-encoded in Stage 1"
    )
    assert "condition_latents=condition_latents," in denoise

    encode = source.split("def run_encode(")[1].split("\ndef ")[0]
    assert encode.index("load_references(") < encode.index("base.load_text_encoder("), (
        "run_encode must decode its references before it loads the conditioner"
    )
    assert '"vae_encoder"' in encode


def test_generation_defaults_to_sage_because_native_was_never_a_choice():
    """sageattention and triton are installed in the H3 venv and were never reached.

    `native` was the argparse default, the serde default and the useState default, so every run
    ever made went through PyTorch SDPA while the Sage kernels sat unused on disk. The per-call
    guard falls back to native whenever a call does not meet Sage's constraints, so the downside
    of this default is a log line, not a wrong result.
    """
    parser = minimax_h3.build_parser()
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["attention_backend"] == "sage"


def test_ref2va_defaults_to_native_attention_for_the_correctness_baseline():
    """Ref2VA is the deliberate exception to the Sage default above.

    Its baseline must be established on native SDPA before GATE 4 lands. Changing the kernel
    underneath a path that is not yet qualified would make the next quality regression ambiguous
    between the model and the attention kernel.
    """
    parser = minimax_h3_ref.parser()
    defaults = {action.dest: action.default for action in parser._actions}
    assert defaults["attention_backend"] == "native"
    assert defaults["sampling_mode"] == "base"
    assert defaults["first_block_cache"] is False
    assert defaults["first_block_cache_threshold"] == 0.08




def test_turbo_lora_alpha_is_lightx2v_published_absolute_value():
    """The regression that produced a garbled Ref2VA take on 2026-08-18.

    LightX2V ship no adapter_config.json, and the alpha the bf16 repacks *do* declare in their
    safetensors metadata is exactly this same 8 -- see
    `test_a_declared_alpha_outranks_the_filename_table`. The filename table below is the fallback
    for the releases that declare nothing. MiniMax's own MiniMax-H3-Turbo-Lora Space hardcodes 8
    with the comment
    "rank 128, alpha 8 -> fold scale 0.0625", and notes that fold matches LightX2V's own inference
    at weights=1.0. Deriving alpha from the rank instead applies the rank-128 Ref2VA adapter at
    16x. Every OTHER H3 adapter family (ai-toolkit, fal, larryvrh) really is alpha == rank -- this
    test exists so that fact does not talk anyone into "fixing" this one again.
    """
    assert TURBO_LORA_ALPHA == 8

    source = Path(minimax_h3.__file__).read_text(encoding="utf-8")
    apply = source.split("def apply_turbo_lora(")[1].split("\ndef ")[0]
    assert "alpha = turbo_lora_alpha(lora_path)" in apply, (
        "the alpha must come from the one function every path resolves it through, so that the "
        "AIV_TURBO_LORA_ALPHA override reaches ConvRot too"
    )
    assert "or rank" not in apply, "alpha must not fall back to the adapter's own rank"
    # rank 128 -> 0.0625, and that must NOT be reported as a fault.
    assert "scale > 1.0" in apply, "only over-application is worth a warning"

    fl2va_rank, ref2va_rank = 8, 128
    assert TURBO_LORA_ALPHA / fl2va_rank == 1.0
    assert TURBO_LORA_ALPHA / ref2va_rank == 0.0625
    assert turbo_lora_alpha("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors") == 128


def test_turbo_shifts_follow_the_adapter_not_the_module():
    """The 768p FL2V 4-step adapter was distilled at video shift 6, everything else at 12."""
    assert turbo_shifts("minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors") == (
        TURBO_VIDEO_SHIFT, TURBO_AUDIO_SHIFT
    )
    assert turbo_shifts("minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors") == (12.0, 3.0)
    assert turbo_shifts("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors") == (6.0, 3.0)
    assert turbo_shifts("/some/dir/MINIMAX_H3_FL2V_TURBO_4STEP_V1.0_768P_BF16.SAFETENSORS") == (
        6.0, 3.0
    ), "the filename match must be case-insensitive"


def test_turbo_shifts_and_alpha_are_overridable(monkeypatch):
    monkeypatch.setenv("AIV_TURBO_AUDIO_SHIFT", "5")
    assert turbo_shifts("minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors") == (12.0, 5.0)
    monkeypatch.setenv("AIV_TURBO_VIDEO_SHIFT", "9.5")
    assert turbo_shifts("minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors") == (9.5, 5.0)


def test_the_alpha_override_reaches_the_path_that_actually_ships(monkeypatch):
    """AIV_TURBO_LORA_ALPHA was documented at the constant and unreachable from ConvRot.

    Both readers -- `apply_turbo_lora` and `EmbeddedLoraLinear.__init__` -- are off the live
    ConvRot path, which resolves its alpha by calling `turbo_lora_alpha` and nothing else. Same
    shape as the AIV_TURBO_VIDEO_SHIFT bug `turbo_shifts` carries a note about.
    """
    published = "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    other = "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
    assert turbo_lora_alpha(published) == 128
    assert turbo_lora_alpha(other) == TURBO_LORA_ALPHA

    monkeypatch.setenv("AIV_TURBO_LORA_ALPHA", "16")
    assert turbo_lora_alpha(published) == 16, "the override must outrank the filename table"
    assert turbo_lora_alpha(other) == 16

    # An empty or blank value is not an override; it is an unset variable spelled badly.
    monkeypatch.setenv("AIV_TURBO_LORA_ALPHA", "   ")
    assert turbo_lora_alpha(published) == 128


def test_reference_geometry_overrides_reach_the_pipeline(monkeypatch):
    """`max` mode must be reachable: 2048 is above the generated canvas and must not be clamped."""
    from aiv_runtime.minimax_h3_ref import _geometry_override

    assert _geometry_override("AIV_H3_MISSING_KNOB", 384) == 384
    monkeypatch.setenv("AIV_H3_REF_IMAGE_SHORT_EDGE", "2048")
    assert _geometry_override("AIV_H3_REF_IMAGE_SHORT_EDGE", 544) == 2048
    monkeypatch.setenv("AIV_H3_REF_IMAGE_SHORT_EDGE", "16")
    try:
        _geometry_override("AIV_H3_REF_IMAGE_SHORT_EDGE", 544)
    except ValueError:
        pass
    else:
        raise AssertionError("a sub-32 reference edge must be rejected, not silently accepted")


def test_output_decode_defaults_to_minimax_reference_tiles():
    base_defaults = {action.dest: action.default for action in minimax_h3.build_parser()._actions}
    ref_defaults = {action.dest: action.default for action in minimax_h3_ref.parser()._actions}
    assert base_defaults["vae_tile"] == VAE_TILE_SIZE == 256
    assert ref_defaults["vae_tile"] == VAE_TILE_SIZE
    assert VAE_TILE_OVERLAP == 64
    base_stage = next(action for action in minimax_h3.build_parser()._actions if action.dest == "stage")
    ref_stage = next(action for action in minimax_h3_ref.parser()._actions if action.dest == "stage")
    assert "decode" in base_stage.choices
    assert "decode" in ref_stage.choices


def test_dead_decode_monkeypatch_is_gone_and_denoise_serializes_latents():
    for module in (minimax_h3, minimax_h3_ref):
        source = Path(module.__file__).read_text(encoding="utf-8")
        denoise = source.split("def run_denoise(")[1].split("\ndef ")[0]
        assert "blk.__call__" not in denoise
        assert 'output=["videos", "audio", "sampling_rate"]' not in denoise
        assert "save_denoised_latents(" in denoise


def test_the_keyframe_vae_is_placed_on_a_device_in_the_denoise_stage():
    """The 2026-08-26 crash, pinned at the source.

    `run_encode` keeps only the `text_encoder` block, so a request carrying `image` or `last_image`
    reaches `vae.encode` for the FIRST time in the denoise process.
    `MiniMaxH3KeyframeVaeEncoderStep` moves its pixels to `components._execution_device` -- CUDA,
    because the transformer is there -- so a VAE left sitting on CPU raises at the first Conv3D:

        RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor)
        should be the same

    `781f61f` deleted the placement during the ConvRot rebase and replaced it with a comment
    asserting that a "CPU-resident VAE" was fine for keyframe encoding. Every keyframe mode --
    image, first-last, last-only, and the clip-extend / clip-to-clip editor workflows that rewrite
    themselves into `image` and `first-last` -- failed in Stage 2 until 2026-08-26.

    This is a source check because reaching the fault at run time needs a GPU and 20 GiB of
    weights. It is the same shape as the module-level-import guard above, and for the same reason:
    a test that can only fail on the machine that has the hardware protects nothing on the others.
    """
    denoise = Path(minimax_h3.__file__).read_text(encoding="utf-8")
    denoise = denoise.split("def run_denoise(")[1].split("\ndef ")[0]
    placement = denoise[denoise.index("denoise_vae = getattr(pipeline") :]

    assert 'profile["vae_resident"]' in placement, (
        "`vae_resident` must decide where the keyframe VAE goes. It was declared in every profile, "
        "printed in the offload log line and asserted in a test, while nothing read it -- so the "
        "log said `vae=offloaded` about a VAE that was neither offloaded nor resident."
    )
    assert 'denoise_vae.to("cuda")' in placement, "the resident profile must put it on the GPU"
    assert "apply_group_offloading(" in placement, "every other profile must give it onload hooks"
    assert "offload_type=VIDEO_VAE_OFFLOAD_TYPE" in placement, (
        "block level follows only the repeated block lists named in `_no_split_modules` and leaves "
        "`conv_in`, `quant_conv`, `post_quant_conv` and the decoder projections behind on CPU"
    )
    assert VIDEO_VAE_OFFLOAD_TYPE == "leaf_level"
    # And the thing that made the old bug possible: a name bound in one branch, used in another.
    assert denoise.index('devices = {"onload_device"') < denoise.index("enable_group_offload("), (
        "`devices` must be bound before both placements, not inside the transformer's else branch"
    )


def test_the_video_vae_runs_at_the_released_precision_unless_told_otherwise(monkeypatch):
    """FP32 is the default because it is what the released pipeline does.

    `AutoencoderKLMiniMaxH3._keep_in_fp32_modules` covers `encoder`, `decoder`, `quant_conv` and
    `post_quant_conv` -- effectively everything -- and Diffusers enforces it at load time whatever
    dtype is asked for, so `dtype=torch.float16` on this component was always a silent no-op.

    That pin is a Diffusers-side conservative default rather than a model requirement: ComfyUI
    ships `minimax_h3_video_vae_fp16.safetensors` and selects it in every official H3 template, and
    the one published measurement (ComfyUI #15416) puts fp16, bf16 and fp32 "within 0.1 of each
    other" on a 0-255 scale. So it is a switch, not a decision baked into the loader -- and the
    switch defaults to matching the release.
    """
    import torch

    monkeypatch.delenv("AIV_H3_VAE_DTYPE", raising=False)
    assert minimax_h3.video_vae_dtype() is None, "no environment variable means the released FP32"
    monkeypatch.setenv("AIV_H3_VAE_DTYPE", "fp32")
    assert minimax_h3.video_vae_dtype() is None, "and so does asking for it by name"
    monkeypatch.setenv("AIV_H3_VAE_DTYPE", "BF16")
    assert minimax_h3.video_vae_dtype() is torch.bfloat16, "the name is case-insensitive"
    monkeypatch.setenv("AIV_H3_VAE_DTYPE", "fp16")
    assert minimax_h3.video_vae_dtype() is torch.float16
    monkeypatch.setenv("AIV_H3_VAE_DTYPE", "int8")
    with pytest.raises(ValueError, match="fp32, bf16, fp16"):
        minimax_h3.video_vae_dtype()


def test_casting_the_video_vae_clears_the_pin_that_would_otherwise_fight_it(monkeypatch):
    """A half-applied cast is worse than either whole one.

    Left set, `_keep_in_fp32_modules` makes `.to()` warn about "inconsistent results" and cast
    anyway, and `get_parameter_dtype` keeps skipping those parameters when it reports the model's
    dtype -- so `AutoencoderKLMiniMaxH3.encode`, which casts incoming pixels to the encoder's own
    dtype, would hand FP32 pixels to FP16 weights. Cleared, the cast is total and self-consistent.
    """
    import torch

    def fake_vae():
        module = torch.nn.Conv3d(2, 2, kernel_size=1)
        module._keep_in_fp32_modules = ["encoder", "decoder", "quant_conv", "post_quant_conv"]
        return module

    monkeypatch.delenv("AIV_H3_VAE_DTYPE", raising=False)
    untouched = fake_vae()
    assert minimax_h3.cast_video_vae(untouched) is torch.float32
    assert next(untouched.parameters()).dtype is torch.float32
    assert untouched._keep_in_fp32_modules, "the released path must not disturb the pin at all"

    monkeypatch.setenv("AIV_H3_VAE_DTYPE", "bf16")
    cast = fake_vae()
    assert minimax_h3.cast_video_vae(cast) is torch.bfloat16
    assert next(cast.parameters()).dtype is torch.bfloat16
    assert cast._keep_in_fp32_modules is None, (
        "the pin has to go, or the cast is the worst of both: a warning, a partial conversion and "
        "a `get_parameter_dtype` that still reports float32"
    )

    # Asking for what it already is must not touch the pin either.
    monkeypatch.setenv("AIV_H3_VAE_DTYPE", "fp32")
    idempotent = fake_vae()
    assert minimax_h3.cast_video_vae(idempotent) is torch.float32
    assert idempotent._keep_in_fp32_modules


def test_the_decode_autocast_follows_the_weights():
    """An fp16 autocast over bf16 weights casts them down per operation and throws away exactly
    the exponent range bf16 was chosen for."""
    source = Path(minimax_h3.__file__).read_text(encoding="utf-8")
    decode = source.split("def decode_video_frames_to_cpu(")[1].split("\ndef ")[0]
    assert "autocast_dtype = weight_dtype if weight_dtype in (torch.float16, torch.bfloat16)" in decode
    assert "dtype=autocast_dtype" in decode
    assert "dtype=torch.float16," not in decode, "the autocast dtype must not be hardcoded again"


def test_the_video_vae_is_cast_before_it_is_placed():
    """Every stage that loads this VAE must cast it before moving or hooking it.

    A dtype cast on a group-offloaded module is not what the offload hooks expect, and `.to()`
    on one is refused outright with a warning. Order is the whole contract.
    """
    # Line-anchored, all of them. Bare substrings collide with things that legitimately run
    # earlier and are not this VAE: `audio_pipeline.audio_vae.to("cuda")` contains
    # `vae.to("cuda")`, and `hasattr(pipeline.vae, "enable_tiling")` contains `pipeline.vae,`.
    ordered = (
        (minimax_h3, "run_denoise", "cast_video_vae(denoise_vae)",
         ('\n            denoise_vae.to("cuda")', "\n                denoise_vae,")),
        (minimax_h3, "run_decode", "cast_video_vae(vae)", ('\n    vae.to("cuda")',)),
        (minimax_h3_ref, "run_encode", "base.cast_video_vae(pipeline.vae)",
         ("\n        pipeline.vae,",)),
    )
    for module, stage, cast, placements in ordered:
        source = Path(module.__file__).read_text(encoding="utf-8")
        body = source.split(f"def {stage}(")[1].split("\ndef ")[0]
        assert cast in body, f"{module.__name__}.{stage} never casts the video VAE"
        for placement in placements:
            assert placement in body, f"{module.__name__}.{stage} lost {placement.strip()!r}"
            assert body.index(cast) < body.index(placement), (
                f"{module.__name__}.{stage} reaches `{placement.strip()}` before `{cast}`"
            )


def test_every_offload_profile_declares_where_the_vae_goes():
    """`profile["vae_resident"]` is now read, so a profile missing it would be a KeyError."""
    for name, profile in OFFLOAD_PROFILES.items():
        assert "vae_resident" in profile, f"the {name} profile does not say where the VAE goes"
    assert OFFLOAD_PROFILES["resident"]["vae_resident"] is True
    for name in ("performance", "balanced", "minimal"):
        assert OFFLOAD_PROFILES[name]["vae_resident"] is False, (
            f"{name} is a 16 GB profile; a resident FP32 video VAE is over half the card"
        )
    # Every alias and every accepted `--offload` value has to resolve to one of those.
    for name in (*OFFLOAD_PROFILES, *OFFLOAD_ALIASES):
        assert "vae_resident" in minimax_h3.effective_offload_profile(name)


def test_a_text_request_does_not_load_ten_gigabytes_of_video_vae():
    """`MiniMaxH3KeyframeVaeEncoderStep` reads `components.vae` once per keyframe, so a t2va
    request never touches it -- and the released VAE is FP32 whatever dtype it is asked for."""
    denoise = Path(minimax_h3.__file__).read_text(encoding="utf-8")
    denoise = denoise.split("def run_denoise(")[1].split("\ndef ")[0]
    assert "if keyframes:" in denoise, "the video VAE load must be conditional on having keyframes"
    load_index = denoise.index('names=["vae"]')
    assert denoise.rindex("if keyframes:", 0, load_index) > denoise.index("keyframes = _keyframes("), (
        "the guard has to sit between resolving the keyframes and loading the VAE"
    )
    # And the pipeline call has to reuse the same opened images rather than re-opening them.
    assert "**keyframes," in denoise
    assert "**_keyframes(" not in denoise


def test_the_keyframe_modes_that_actually_carry_a_frame():
    """Which modes reach the VAE at all. The editor's clip workflows rewrite themselves into these
    before the worker ever sees them (`h3.rs::generate_clip_extension` sets `mode = "image"`)."""
    from PIL import Image

    scratch = Path(tempfile.mkdtemp())
    first, last = scratch / "first.png", scratch / "last.png"
    for path in (first, last):
        Image.new("RGB", (64, 64)).save(path)
    try:
        def frames(mode: str) -> set[str]:
            args = Namespace(mode=mode, first_frame=str(first), last_frame=str(last))
            return set(minimax_h3._keyframes(args, 960, 544))

        assert frames("text") == set(), "a text request must not load the VAE at all"
        assert frames("image") == {"image"}
        assert frames("last-only") == {"last_image"}
        assert frames("first-last") == {"image", "last_image"}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_temporal_decoder_retires_completed_chunks_to_cpu():
    import torch
    from torch import nn

    class Config:
        token_drop = 1
        clip_length = 2
        latents_mean = (0.0,)
        latents_std = (1.0,)

    class FakeVAE(nn.Module):
        spatial_compression_ratio = 1
        temporal_compression_ratio = 1
        tokens_chunk_size = 2
        token_overlap = 1
        frame_overlap = 1
        frame_pre_padding = 0
        config = Config()

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
            self.decode_calls = 0

        def enable_tiling(self, **kwargs):
            self.tile_options = kwargs

        @staticmethod
        def _split_tiles(length, tile_size, _overlap):
            return ([0], [length], []) if tile_size >= length else ([0, tile_size], [tile_size] * 2, [64])

        def _decode_clip(self, z):
            self.decode_calls += 1
            values = torch.cat([z, z[:, :, -1:]], dim=2)
            return values.repeat(1, 3, 1, 1, 1)

        @staticmethod
        def _blend(a, b, extent, dim):
            # The production VAE's one-frame blend gives the prior overlap weight 1 at position 0.
            prefix = a.narrow(dim, a.shape[dim] - extent, extent)
            rest = b.narrow(dim, extent, b.shape[dim] - extent)
            return torch.cat([prefix, rest], dim=dim)

    vae = FakeVAE()
    latents = torch.tensor([0.0, 0.1, 0.2, 0.3]).view(1, 1, 4, 1, 1)
    frames = decode_video_frames_to_cpu(vae, latents, VAE_TILE_SIZE)

    assert vae.decode_calls == 2
    assert len(frames) == 5, "one repeated pad frame must be removed after streaming"
    assert all(frame.shape == (1, 1, 3) and frame.dtype.name == "uint8" for frame in frames)
    assert vae.tile_options["tile_sample_min_overlap_height"] == VAE_TILE_OVERLAP


def test_base_worker_launches_encode_denoise_decode_as_separate_processes(monkeypatch, tmp_path):
    paths = iter((tmp_path / "embeds.pt", tmp_path / "latents.pt"))

    def fake_mkstemp(suffix):
        path = next(paths)
        return os.open(path, os.O_CREAT | os.O_RDWR), str(path)

    commands = []

    def fake_run(command, **_kwargs):
        commands.append(command)
        return Namespace(returncode=0, stdout="")

    monkeypatch.setattr(minimax_h3.tempfile, "mkstemp", fake_mkstemp)
    monkeypatch.setattr(minimax_h3.subprocess, "run", fake_run)
    monkeypatch.setattr(minimax_h3.sys, "argv", ["minimax_h3.py", "--repo-root", "model"])

    minimax_h3.run_all(Namespace())

    assert [command[command.index("--stage") + 1] for command in commands] == [
        "encode", "denoise", "decode"
    ]
    assert all("--latents-path" in command for command in commands)


def test_production_h3_denoisers_are_safetensors_only(tmp_path):
    args = Namespace(
        repo_root=str(tmp_path), mode="text", first_frame=None, last_frame=None,
        prompt="test", steps=50, vae_tile=VAE_TILE_SIZE, stage="all", embeds_path=None,
        latents_path=None, sampling_mode="base", turbo_lora_path=None,
        transformer_path=str(tmp_path / "legacy.pt"),
    )
    (tmp_path / "modular_model_index.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="safetensors"):
        minimax_h3.validate(args)


def test_production_h3_conditioner_is_safetensors_only(tmp_path):
    args = Namespace(
        repo_root=str(tmp_path), mode="text", first_frame=None, last_frame=None,
        prompt="test", steps=50, vae_tile=VAE_TILE_SIZE, stage="all", embeds_path=None,
        latents_path=None, sampling_mode="base", turbo_lora_path=None,
        transformer_path=None, text_encoder_path=str(tmp_path / "legacy.bin"),
    )
    (tmp_path / "modular_model_index.json").write_text("{}", encoding="utf-8")
    (tmp_path / "legacy.bin").write_bytes(b"not model weights")
    with pytest.raises(ValueError, match="safetensors"):
        minimax_h3.validate(args)


def test_published_conditioner_name_mapping_and_metadata_contract():
    assert native_to_transformers("model.layers.0.self_attn.q_proj.weight") == (
        "model.language_model.layers.0.self_attn.q_proj.weight"
    )
    assert native_to_transformers("visual.patch_embed.proj.weight") == (
        "model.visual.patch_embed.proj.weight"
    )
    _validate_metadata({"minimax_h3_te": json.dumps(PUBLISHED_METADATA)})
    with pytest.raises(ValueError, match="metadata"):
        _validate_metadata({"minimax_h3_te": json.dumps({"num_hidden_layers": 49})})


def test_a_declared_alpha_outranks_the_filename_table(tmp_path):
    """The installed adapters state their own alpha; a file that describes itself wins.

    Measured 2026-08-25 on the installed releases: `minimax_h3_ref2v_turbo_4step_v0.1_bf16` and
    `minimax_h3_fl2v_turbo_8step_v1.0_bf16` both carry `alpha: '8'` at rank 128, and
    `minimax_h3_fl2v_turbo_4step_v0.1` carries none. So the declaration and the published 8 agree
    today -- the point of preferring it is the day they would not, e.g. a renamed file, or the
    768p v1.0 adapter the filename table can only guess 128 for.
    """
    import torch
    from safetensors.torch import save_file

    weights = {"transformer_blocks.0.attn.to_q" + LORA_A_SUFFIX: torch.zeros(4, 4)}

    declared = tmp_path / "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
    save_file(weights, str(declared), metadata={"alpha": "8", "format": "pt"})
    assert declared_lora_alpha(declared) == 8
    assert turbo_lora_alpha(declared) == 8

    # A name the table would answer 128 for, in a file that says 8. The file wins.
    disagreeing = tmp_path / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    save_file(weights, str(disagreeing), metadata={"alpha": "8", "format": "pt"})
    assert turbo_lora_alpha(disagreeing) == 8, "a declared alpha must beat the filename guess"

    # The same name with nothing declared falls back to the table.
    silent = tmp_path / "quiet" / "minimax_h3_fl2v_turbo_4step_v1.0_768p_bf16.safetensors"
    silent.parent.mkdir()
    save_file(weights, str(silent), metadata={"format": "pt"})
    assert declared_lora_alpha(silent) is None
    assert turbo_lora_alpha(silent) == 128


def test_an_unusable_declared_alpha_falls_back_rather_than_applying_nonsense(tmp_path):
    """A metadata probe must never be able to make the adapter apply at a nonsense scale."""
    import torch
    from safetensors.torch import save_file

    weights = {"transformer_blocks.0.attn.to_q" + LORA_A_SUFFIX: torch.zeros(4, 4)}
    for label, metadata in (
        ("absent", {"format": "pt"}),
        ("empty", {"alpha": "", "format": "pt"}),
        ("words", {"alpha": "eight", "format": "pt"}),
        ("zero", {"alpha": "0", "format": "pt"}),
        ("negative", {"alpha": "-8", "format": "pt"}),
    ):
        path = tmp_path / f"{label}.safetensors"
        save_file(weights, str(path), metadata=metadata)
        assert declared_lora_alpha(path) is None, label
        assert turbo_lora_alpha(path) == TURBO_LORA_ALPHA, label

    # A file that is not a safetensors file at all, and a path that does not exist.
    broken = tmp_path / "broken.safetensors"
    broken.write_bytes(b"not safetensors")
    assert declared_lora_alpha(broken) is None
    assert declared_lora_alpha(tmp_path / "absent-entirely.safetensors") is None


def test_the_convrot_report_states_the_scale_the_residual_is_applied_at():
    """`turbo_pairs` alone cannot distinguish an adapter at 0.0625 from one at 16x.

    The 2026-08-18 fault was invisible in the log: the count was right and the strength was 256x
    off. The ConvRot path adds its residual unfused, so it never passes through
    `apply_turbo_lora`'s warning -- the numbers have to come from its own report.
    """
    source = Path(minimax_h3_convrot.__file__).read_text(encoding="utf-8")
    report = source.split("transformer._aiv_convrot_report = {")[1].split("}")[0]
    for field in ("turbo_rank", "turbo_alpha", "turbo_scale"):
        assert field in report, f"the ConvRot report must state {field}"

    loader = Path(minimax_h3.__file__).read_text(encoding="utf-8")
    convrot_branch = loader.split("if is_convrot:")[1].split("\n    config = ")[0]
    assert "turbo_scale" in convrot_branch, "the resolved scale must reach the worker log"
    assert "WARNING: turbo LoRA scale" in convrot_branch, (
        "over-unity application must warn on the ConvRot path too, not only through PEFT"
    )


def test_the_writer_hands_frames_over_untouched(tmp_path, monkeypatch):
    """The inversion regression, with no encoder needed so it runs in the suite's own venv.

    `imageio` lives in `vendor/MiniMax-H3/.venv`, not in the project `.venv` that runs these tests,
    so the end-to-end round trip below can only `importorskip`. A skipped test is not protection,
    and this is the one regression worth guaranteeing: `export_to_video` rescaled our already-uint8
    frames and NumPy 2 wrapped the result modulo 256, inverting every generation. Here the writer
    is captured rather than exercised, which asserts exactly our half of the contract -- that
    nothing between `_decoded_chunk_to_frames` and the encoder touches a pixel.
    """
    import sys
    import types as _types

    import numpy as np

    captured = []

    class _Writer:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def append_data(self, frame):
            captured.append(frame)

    fake = _types.ModuleType("imageio")
    fake.get_writer = lambda *args, **kwargs: _Writer()
    monkeypatch.setitem(sys.modules, "imageio", fake)

    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8) for _ in range(3)]
    args = Namespace(output_path=str(tmp_path / "out.mp4"))
    minimax_h3.write_video_output(args, frames, len(frames))

    assert len(captured) == len(frames)
    for index, (sent, original) in enumerate(zip(captured, frames)):
        assert sent.dtype == np.uint8, f"frame {index} reached the encoder as {sent.dtype}"
        assert np.array_equal(sent, original), (
            f"frame {index} was modified on the way to the encoder. The bug this guards against "
            f"rescaled uint8 by 255, which NumPy wraps to 256 - x."
        )


def test_uint8_frames_survive_the_video_writer(tmp_path):
    """The 2026-08-25 inversion, pinned end to end.

    `diffusers.utils.export_to_video` rescales numpy frames unconditionally on the assumption they
    are float in [0, 1]. Ours are uint8, and under NumPy 2 `uint8 * 255` wraps modulo 256, so that
    rescale mapped every pixel to `256 - x`. This writes a known ramp through the real writer and
    reads it back: an inverted writer fails on the very first row.
    """
    imageio = pytest.importorskip("imageio", reason="the writer's own backend")
    pytest.importorskip("imageio_ffmpeg", reason="imageio's mp4 encoder")
    import numpy as np

    # A horizontal ramp per channel, offset so no two channels agree. A channel swap, an inversion
    # and a flat frame are each distinguishable from this; a mid-grey test image is not.
    width, height = 256, 64
    ramp = np.arange(width, dtype=np.uint8)
    frame = np.stack(
        [
            np.tile(ramp, (height, 1)),
            np.tile(np.roll(ramp, 64), (height, 1)),
            np.tile(np.roll(ramp, 128), (height, 1)),
        ],
        axis=-1,
    ).astype(np.uint8)

    output = tmp_path / "ramp.mp4"
    args = Namespace(output_path=str(output))
    minimax_h3.write_video_output(args, [frame] * 8, 8)
    assert output.is_file()

    read_back = imageio.v3.imread(str(output), index=4)
    assert read_back.shape == frame.shape

    # Per pixel, not per channel mean. A full ramp is symmetric about 127.5, so inverting it barely
    # moves any channel's mean -- a mean-based assertion passes the very bug this test exists for.
    # Measured on this ramp: real writer 1.1, the rescale bug 127.0, R/B swapped 85.3, flat grey
    # 64.0. H.264 through YUV 4:2:0 is lossy, so exact equality is not on offer; 8 clears the
    # encoder by a wide margin and fails every fault by a wider one.
    drift = float(
        np.abs(read_back.astype(np.int32) - frame.astype(np.int32)).mean()
    )
    assert drift < 8, (
        f"frames changed by {drift:.1f} per pixel through the writer. The rescale bug measures "
        f"about 127 on this ramp, a red/blue swap about 85."
    )


def test_the_audio_writer_asks_for_a_float_wav_not_the_clipping_default(tmp_path):
    """`soundfile.write` defaults WAV to PCM_16, which squares off everything past +/-1.0.

    The H3 audio decoder does not bound its output -- the released `decoders.py` clamps the video
    branch and deliberately leaves the audio branch alone -- so a loud generation arrives with
    samples above unity and every one of them used to be clipped on the way to disk. That is the
    harsh, crackling audio reported upstream, produced by our writer rather than by the model.

    Injecting the module rather than round-tripping a real file keeps this running in an
    environment without `soundfile`, which is where the regression would otherwise go unnoticed.
    """
    import sys
    import types as _types

    import numpy as np

    captured = {}

    fake = _types.ModuleType("soundfile")

    def _write(path, samples, rate, subtype=None, **kwargs):
        captured["path"] = path
        captured["samples"] = np.array(samples, copy=True)
        captured["rate"] = rate
        captured["subtype"] = subtype

    fake.write = _write
    monkeypatch_target = sys.modules.get("soundfile")
    sys.modules["soundfile"] = fake
    try:
        import torch

        # Two channels, deliberately hot: the peak is 1.4, which PCM_16 would flatten to 1.0.
        loud = torch.tensor([[[0.0, 1.4, -1.25, 0.5]], [[0.0, -1.4, 1.25, -0.5]]]).reshape(1, 2, 4)
        args = Namespace(output_path=str(tmp_path / "take.mp4"), audio_path=None)
        minimax_h3.write_audio_output(args, loud, 32_000)
    finally:
        if monkeypatch_target is None:
            del sys.modules["soundfile"]
        else:
            sys.modules["soundfile"] = monkeypatch_target

    assert captured["subtype"] == minimax_h3.AUDIO_WAV_SUBTYPE == "FLOAT", (
        "the writer must name a float subtype; soundfile's WAV default is PCM_16"
    )
    assert captured["rate"] == 32_000
    assert str(captured["path"]).endswith("take.wav")
    peak = float(np.abs(captured["samples"]).max())
    assert peak == pytest.approx(1.4, abs=1e-6), (
        f"the writer changed the waveform on the way out (peak {peak}). It must not scale, clamp "
        f"or normalize -- the peak is reported in the log instead."
    )


def test_a_hot_waveform_survives_the_real_audio_writer(tmp_path):
    """The same contract through the real backend: 1.4 must come back as 1.4, not as 1.0."""
    soundfile = pytest.importorskip("soundfile", reason="the audio writer's own backend")
    import numpy as np
    import torch

    assert soundfile.default_subtype("WAV") == "PCM_16", (
        "if soundfile's WAV default ever becomes a float format this test's premise changes, but "
        "the explicit subtype stays correct either way"
    )

    samples = np.linspace(-1.4, 1.4, 2048, dtype=np.float32)
    waveform = torch.from_numpy(np.stack([samples, -samples])).reshape(1, 2, samples.size)
    output = tmp_path / "hot.wav"
    args = Namespace(output_path=str(tmp_path / "hot.mp4"), audio_path=str(output))
    minimax_h3.write_audio_output(args, waveform, 32_000)

    read_back, rate = soundfile.read(str(output), dtype="float32", always_2d=True)
    assert rate == 32_000
    assert read_back.shape == (samples.size, 2)
    assert float(np.abs(read_back).max()) == pytest.approx(1.4, abs=1e-5), (
        "PCM_16 would have clipped this to 1.0 -- which is exactly the distortion the writer used "
        "to introduce on any generation loud enough to exceed unity."
    )
    assert np.allclose(read_back[:, 0], samples, atol=1e-6)


def test_the_environment_shift_override_outranks_the_command_line(monkeypatch):
    """Precedence: published pair, then the worker's argument, then the environment.

    The Ref2VA command builder used to send `--video-shift` on every request, and the worker
    applied it after `turbo_shifts` had already consulted the environment -- so the documented
    `AIV_TURBO_VIDEO_SHIFT` dial could never be reached from inside the application.
    """
    adapter = "minimax_h3_ref2v_turbo_4step_v0.1_bf16.safetensors"
    assert turbo_shifts(adapter) == (12.0, 3.0), "the published pair with nothing overriding it"
    assert turbo_shifts(adapter, 8.0) == (8.0, 3.0), "an explicit argument beats the table"
    monkeypatch.setenv("AIV_TURBO_VIDEO_SHIFT", "12")
    assert turbo_shifts(adapter, 8.0) == (12.0, 3.0), "and the environment beats the argument"


def test_the_reference_video_area_follows_whichever_short_edge_is_in_force(monkeypatch):
    """Raising the reference-video tier has to raise its area budget too.

    The area was a module constant pinned to the 384 tier, so asking for a 768 px reference video
    left it capped at 384*672 and the setting did nothing. The budget is now derived, and the two
    endpoints land exactly on upstream's numbers.
    """
    from aiv_runtime.minimax_h3_ref import _ref_video_max_pixels

    assert _ref_video_max_pixels(384) == 384 * 672 == H3_REF_VIDEO_MAX_PIXELS
    assert _ref_video_max_pixels(768) == 768 * 1344, "the released reference-video budget"
    assert _ref_video_max_pixels(544) > _ref_video_max_pixels(384)


def test_the_embedding_cache_key_versions_the_geometry_it_was_built_from(monkeypatch):
    """Stage 1's output depends on the canvas, so a key that omits it can replay the wrong one."""
    from aiv_runtime.minimax_h3_ref import _compute_embedding_hash

    def request(**overrides):
        fields = dict(
            prompt="a shot", references_json="[]", width=960, height=544, num_frames=124,
            ref_image_short_edge=544, ref_video_short_edge=384,
        )
        fields.update(overrides)
        return Namespace(**fields)

    baseline = _compute_embedding_hash(request())
    assert _compute_embedding_hash(request()) == baseline, "the key must be stable"
    for field, value in (
        ("prompt", "a different shot"), ("references_json", '["x"]'),
        ("width", 1344), ("height", 768), ("num_frames", 141),
        ("ref_image_short_edge", 2048), ("ref_video_short_edge", 768),
    ):
        assert _compute_embedding_hash(request(**{field: value})) != baseline, (
            f"changing {field} must invalidate the cached reference embeddings"
        )
    monkeypatch.setenv("AIV_H3_REF_IMAGE_SHORT_EDGE", "2048")
    assert _compute_embedding_hash(request()) != baseline, (
        "the geometry environment overrides bypass the argument namespace, so they have to be "
        "keyed directly or an override would silently reuse the old embeddings"
    )


def test_the_writer_refuses_float_frames_rather_than_darkening_them(tmp_path):
    """Float frames in [0, 1] would encode to near-black, which reads as a model failure.

    The inversion at least announced itself. Silently writing a black video would not, so the
    writer's contract is enforced rather than assumed.
    """
    import numpy as np

    args = Namespace(output_path=str(tmp_path / "float.mp4"))
    frames = [np.zeros((16, 16, 3), dtype=np.float32)]
    with pytest.raises(TypeError, match="uint8"):
        minimax_h3.write_video_output(args, frames, 1)


def test_numpy_still_wraps_uint8_multiplication():
    """The reason the writer must not rescale, pinned against a NumPy behaviour change.

    Under NEP 50 `uint8 * 255` stays uint8 and wraps modulo 256, giving `(256 - x) % 256`. If a
    future NumPy promotes to a wider type instead, this test fails and the comment in
    `write_video_output` needs revisiting -- the writer should still not rescale, but the story
    about *why* the output inverted would no longer be true.
    """
    import numpy as np

    values = np.array([0, 1, 2, 50, 100, 128, 200, 254, 255], dtype=np.uint8)
    wrapped = (values * 255).astype(np.uint8)
    assert wrapped.tolist() == [(256 - int(v)) % 256 for v in values]
    # True black folds onto itself rather than onto white: the hard discontinuity that showed up
    # as burned speckle in the darkest parts of the picture.
    assert wrapped[0] == 0


def test_short_clips_still_get_a_temporal_chunk():
    """Diffusers' chunk arithmetic returns zero chunks below seven latent frames; ComfyUI's guards it.

    `token_drop` subtracts a whole chunk that the padding never restores, so `_decode` builds an
    empty list and `torch.cat` raises `expected a non-empty list of Tensors`. ComfyUI pads one
    extra chunk instead (`comfy/ldm/minimax/vae.py::_decode_temporal_chunks`). We copied Diffusers.
    """
    chunk_size, drop = 5, 3

    # The two counts that used to crash. One latent frame is what a single encoded image produces.
    for latent_frames in (1, 2):
        pad, chunks = temporal_decode_plan(latent_frames, chunk_size, drop)
        assert chunks >= 1, f"{latent_frames} latent frames still plans {chunks} chunks"
        assert pad >= chunk_size, "the extra chunk has to be backed by padded latent frames"

    # Every count must plan at least one chunk, and the padding must always complete whole chunks.
    for latent_frames in range(1, 64):
        pad, chunks = temporal_decode_plan(latent_frames, chunk_size, drop)
        assert chunks >= 1
        assert (latent_frames + drop + pad) % chunk_size == 0
        assert (latent_frames + drop + pad) // chunk_size - 1 == chunks


def test_the_guard_does_not_disturb_counts_that_already_worked():
    """Anything from seven latent frames up must plan exactly what it planned before.

    Pinned against two measured runs: a 124-frame generation encodes to 37 latent frames and
    decodes back to 124, and the VAE round trip in `scripts/bisect-h3-colour.py` decoded 7 latent
    frames to 22 pixel frames.
    """
    chunk_size, drop = 5, 3
    for latent_frames in range(7, 64):
        pad, chunks = temporal_decode_plan(latent_frames, chunk_size, drop)
        pseudo = latent_frames + drop
        unguarded_pad = (-pseudo) % chunk_size
        unguarded_chunks = (pseudo + unguarded_pad) // chunk_size - 1
        assert (pad, chunks) == (unguarded_pad, unguarded_chunks), latent_frames

    # The two anchors, as frame counts. Each chunk emits `clip_length` frames and one trailing
    # overlap chunk of `frame_overlap` follows the loop.
    def frames_for(latent_frames: int) -> int:
        pad, chunks = temporal_decode_plan(latent_frames, chunk_size, drop)
        total = chunks * 17 + 5
        if pad:
            intra_tail = 17 % 4
            total -= sum(
                intra_tail if intra_tail and (latent_frames + k) % chunk_size == 0 else 4
                for k in range(pad)
            )
        return total

    assert frames_for(37) == 124, "a 124-frame generation must decode back to 124 frames"
    assert frames_for(7) == 22, "the measured VAE round trip decoded 7 latent frames to 22"


#: Packages that exist in a worker venv but not in the one the suite runs under. A module-level
#: import of any of these makes the module unimportable during collection, which fails the whole
#: test file -- including tests that need none of them. Measured 2026-08-25 by diffing
#: `vendor/MiniMax-H3/.venv` against the project `.venv`; 21 packages differ, these are the ones
#: this codebase actually imports.
#:
#: `diffusers` is here for a different reason: it is present in both, but the test venv carries
#: 0.39 and the MiniMax-H3 classes landed in 0.40. Absent and too-old fail identically at
#: collection, so they belong in one list.
WORKER_ONLY_PACKAGES = frozenset({
    "comfy_kitchen", "diffusers", "gemlite", "hqq", "imageio", "imageio_ffmpeg",
    "peft", "sageattention", "sentencepiece", "soundfile", "torchao", "triton",
})


def _module_level_worker_imports(source: Path) -> list[str]:
    """Worker-only packages this file imports at module level, i.e. at collection time."""
    import ast

    found = []
    for node in ast.parse(source.read_text(encoding="utf-8")).body:  # body == module level only
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in WORKER_ONLY_PACKAGES:
                found.append(f"from {node.module} import " + ", ".join(a.name for a in node.names))
        elif isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names
                      if a.name.split(".")[0] in WORKER_ONLY_PACKAGES]
    return found


def _runtime_modules_reachable_from_tests(runtime: Path, tests: Path) -> set[Path]:
    """Every `aiv_runtime` module collection will import, following intra-package imports.

    Scoped to what the suite actually reaches rather than the whole package. `image_studio/zit.py`
    and `image_studio/qwen.py` do import diffusers at module level and are perfectly correct to:
    they run under Image Studio's own venv and no test imports them. Failing them would be a false
    alarm that teaches people to delete the check. If a test ever does import one, the closure grows
    and the check starts applying -- which is exactly when it should.
    """
    import ast

    def imported_names(source: Path) -> set[str]:
        names = set()
        for node in ast.walk(ast.parse(source.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("aiv_runtime"):
                names.add(node.module)
                # `from aiv_runtime import x, y` and `from aiv_runtime.pkg import mod`
                names.update(f"{node.module}.{alias.name}" for alias in node.names)
            elif isinstance(node, ast.Import):
                names.update(a.name for a in node.names if a.name.startswith("aiv_runtime"))
        return names

    def to_path(dotted: str) -> Path | None:
        parts = dotted.split(".")[1:]
        if not parts:  # a bare `from aiv_runtime import x` names the package itself
            init = runtime / "__init__.py"
            return init if init.is_file() else None
        relative = Path(*parts)
        for candidate in (runtime / relative.with_suffix(".py"), runtime / relative / "__init__.py"):
            if candidate.is_file():
                return candidate
        return None

    pending = set()
    for test_file in tests.glob("test_*.py"):
        pending |= imported_names(test_file)
    reached: set[Path] = set()
    while pending:
        found = to_path(pending.pop())
        if found is None or found in reached:
            continue
        reached.add(found)
        pending |= imported_names(found)
    return reached


def test_no_collected_module_imports_a_worker_only_package():
    """The failure that broke the 2026-08-25 build, generalised past the one package that caused it.

    `minimax_h3_convrot.py` imported `MiniMaxH3Transformer3DModel` at module level. The suite runs
    under the project `.venv` (diffusers 0.39, no H3 classes), so collection raised and took both H3
    test files with it -- including the pure key-mapping tests, which need no diffusers at all.

    Checking the source rather than attempting an import is the point: an import-based check passes
    on any machine where the package happens to be installed, and only fails on someone else's.
    That is exactly how the original reached a build.
    """
    runtime = Path(minimax_h3.__file__).parent
    tests = Path(__file__).parent

    reachable = _runtime_modules_reachable_from_tests(runtime, tests)
    assert reachable, "found no runtime modules at all; the import walk is broken, not clean"
    assert (runtime / "minimax_h3_convrot.py") in reachable, (
        "the module whose hoisted import caused this is no longer reachable from the tests, so "
        "this check is no longer guarding it"
    )

    offenders = {
        source.relative_to(runtime).as_posix(): hoisted
        for source in sorted(reachable)
        if (hoisted := _module_level_worker_imports(source))
    }
    assert not offenders, (
        "these modules are imported during collection and import a package the test venv does not "
        f"have; move the import into the function that uses it: {offenders}"
    )


def test_the_worker_only_list_still_describes_this_machine():
    """A stale hazard list guards nothing. Verify it against the venvs when both are present.

    Skipped rather than failed where the worker venv is absent -- on a machine without it there is
    nothing to compare against, and the list is still the best available description.
    """
    project = Path(minimax_h3.__file__).parents[2]
    worker_site = project / "vendor" / "MiniMax-H3" / ".venv" / "Lib" / "site-packages"
    if not worker_site.is_dir():
        pytest.skip("the MiniMax-H3 venv is not installed on this machine")

    installed = {
        entry.name[:-3] if entry.name.endswith(".py") else entry.name
        for entry in worker_site.iterdir()
        if not entry.name.endswith((".dist-info", ".pth", ".egg-link"))
    }
    missing = sorted(WORKER_ONLY_PACKAGES - installed - {"diffusers"})
    assert not missing, (
        f"WORKER_ONLY_PACKAGES names {missing}, which the worker venv does not have either. "
        "Either the venv changed or the list has drifted; stale entries make the check look "
        "stricter than it is."
    )


def test_the_scheduler_grid_is_reported_rather_than_assumed(monkeypatch, capsys):
    """`steps + 1` grid points must be observed to buy `steps` evaluations, not asserted in a docstring.

    Read against the installed `MiniMaxH3Scheduler` on 2026-09-08: `set_timesteps` builds
    `linspace(1, 0, n)`, shifts it, drops float32 duplicates with `unique_consecutive`, and
    publishes `timesteps = 1 - sigmas[:-1]`. So an n-point grid drives n-1 evaluations and our
    convention is correct -- a turbo-4 run is 4 NFE. The duplicate collapse is the one way that
    identity breaks, so the realized counts are logged on every run.
    """

    class StubScheduler:
        def __init__(self):
            self.sigmas = None
            self.timesteps = None

        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None):
            grid = list(range(num_inference_steps))
            self.sigmas = grid
            self.timesteps = grid[:-1]
            return "untouched"

    class StubPipeline:
        def __init__(self):
            self.scheduler = StubScheduler()
            self.audio_scheduler = StubScheduler()

    pipeline = StubPipeline()
    minimax_h3.instrument_scheduler_grid(pipeline)
    # Wrapping twice must not stack wrappers or double-log.
    minimax_h3.instrument_scheduler_grid(pipeline)

    assert pipeline.scheduler.set_timesteps(num_inference_steps=5) == "untouched"
    assert pipeline.audio_scheduler.set_timesteps(5) == "untouched"

    reported = capsys.readouterr().err
    assert reported.count("scheduler grid") == 2, reported
    assert "requested 5 points" in reported
    assert "realized 5 sigmas" in reported
    assert "4 transformer evaluations" in reported
    assert "audio_scheduler grid" in reported
    # Instrumentation only: the schedule itself is whatever the scheduler built.
    assert pipeline.scheduler.sigmas == [0, 1, 2, 3, 4]
    assert pipeline.scheduler.timesteps == [0, 1, 2, 3]


def test_scheduler_grid_points_buys_one_terminal_point_per_request():
    assert scheduler_grid_points(Namespace(steps=4)) == 5
    assert scheduler_grid_points(Namespace(steps=50)) == 51


def test_the_encode_stage_keeps_the_block_that_builds_the_keyframes():
    """`before_encode` holds `MiniMaxH3ResizeStep`, which is what turns image/last_image into keyframes.

    Popping it left `MiniMaxH3FL2VATextEncoderStep` selected but with nothing to look at -- it
    guards its vision path on `if block_state.keyframes:` -- so every image and first-last request
    built a text-only presentation. Ref2VA has always kept the block; this pins the base worker to
    the same set.
    """
    assert "before_encode" in minimax_h3.ENCODE_STAGE_BLOCKS
    assert "text_encoder" in minimax_h3.ENCODE_STAGE_BLOCKS
    # Everything downstream belongs to the denoise process and must still be dropped.
    for downstream in ("vae_encoder", "denoise", "decode"):
        assert downstream not in minimax_h3.ENCODE_STAGE_BLOCKS


def test_a_keyframe_request_must_produce_a_vision_block(capsys):
    """An all-TEXT tag vector passed the old `embeds is None or tags is None` guard happily."""
    import torch

    class StubPipeline:
        text_tag = 1
        video_tag = 0

    pipeline = StubPipeline()
    text_only = torch.tensor([1, 1, 1, 1], dtype=torch.long)
    grounded = torch.tensor([1, 0, 0, 0, 1, 1], dtype=torch.long)

    # A text request has no keyframe, so an all-text presentation is exactly right.
    minimax_h3.assert_conditioner_saw_the_keyframes(pipeline, text_only, {})
    reported = capsys.readouterr().err
    assert "4 rows" in reported and "0 video" in reported

    # A keyframe request with the same tags is the silent failure this exists to catch.
    with pytest.raises(RuntimeError, match="text-only presentation"):
        minimax_h3.assert_conditioner_saw_the_keyframes(
            pipeline, text_only, {"image": object()}
        )

    minimax_h3.assert_conditioner_saw_the_keyframes(pipeline, grounded, {"image": object()})
    reported = capsys.readouterr().err
    assert "3 video" in reported and "1 keyframe(s) sent" in reported


def test_a_failed_sage_warmup_puts_the_dispatcher_back(monkeypatch, capsys):
    """The safety net used to do the opposite of what it logged.

    `_install_h3_sage_fallback` replaces `transformer_minimax_h3.dispatch_attention_fn` module-wide.
    On a failed warm-up the handler reset the registry, logged "falling back to native PyTorch
    SDPA" and returned "native" -- but left the wrapper installed. Diffusers then calls with
    `backend=None`, the guard resolves `effective_backend = backend_name or mode`, and every
    attention call takes the Sage branch that had just proved unavailable. The job died at the
    first transformer block, about 20 GB of weights into the load.
    """
    import sys
    import types

    import torch

    def original_dispatch(*args, **kwargs):
        return "the module's own dispatcher"

    h3_module = types.SimpleNamespace(dispatch_attention_fn=original_dispatch)
    monkeypatch.setitem(
        sys.modules, "diffusers.models.transformers.transformer_minimax_h3", h3_module
    )

    class Registry:
        active = "native"

        @classmethod
        def get_active_backend(cls):
            return (cls.active, None)

        @classmethod
        def set_active_backend(cls, value):
            cls.active = value

    monkeypatch.setitem(
        sys.modules,
        "diffusers.models.attention_dispatch",
        types.SimpleNamespace(
            _AttentionBackendRegistry=Registry, dispatch_attention_fn=original_dispatch
        ),
    )

    # Fail the warm-up without touching CUDA, and before any kernel is asked for.
    def no_warmup_tensor(*args, **kwargs):
        raise RuntimeError("no Sage kernel on this machine")

    monkeypatch.setattr(torch, "randn", no_warmup_tensor)

    class Transformer:
        def __init__(self):
            self.resets = 0

        def reset_attention_backend(self):
            self.resets += 1

        def set_attention_backend(self, name):
            raise AssertionError("the warm-up failed; the backend must not be selected")

    transformer = Transformer()
    assert minimax_h3.configure_attention_backend(transformer, "sage") == "native"

    # The claim in the log has to be true of the module afterwards.
    assert "falling back to native PyTorch SDPA" in capsys.readouterr().err
    assert h3_module.dispatch_attention_fn is original_dispatch
    assert not hasattr(h3_module.dispatch_attention_fn, "_aiv_h3_sage_fallback")
    assert Registry.active == "native"
    assert transformer.resets >= 1


def test_uninstalling_the_sage_wrapper_is_idempotent(monkeypatch):
    import sys
    import types

    def original_dispatch(*args, **kwargs):
        return "original"

    h3_module = types.SimpleNamespace(dispatch_attention_fn=original_dispatch)
    monkeypatch.setitem(
        sys.modules, "diffusers.models.transformers.transformer_minimax_h3", h3_module
    )

    # Nothing installed: leave the module alone rather than clearing the attribute.
    minimax_h3._uninstall_h3_sage_fallback()
    assert h3_module.dispatch_attention_fn is original_dispatch

    minimax_h3._install_h3_sage_fallback(mode="sage")
    assert h3_module.dispatch_attention_fn._aiv_h3_sage_fallback is True

    minimax_h3._uninstall_h3_sage_fallback()
    assert h3_module.dispatch_attention_fn is original_dispatch
    minimax_h3._uninstall_h3_sage_fallback()
    assert h3_module.dispatch_attention_fn is original_dispatch


def test_the_dead_teacache_is_gone_and_stays_gone():
    """It had zero call sites and was wrong in four ways, so it was never worth wiring up.

    It returned the previous step's absolute block-stack output instead of adding a residual, so a
    skipped step discarded the current latent entirely; `_teacache_skipped < 1` meant it could only
    ever skip alternate steps; it never reset between runs, so a second generation started with the
    first one's cache; and it logged a hardcoded "skipped 48 blocks" on a 50-block model.

    Step skipping is worth having. The replacement is specified separately -- a first-block
    signature, an added residual, state reset per run -- and starts from that specification, not
    from this. Anyone finding this function name in a log or an old branch should not restore it.
    """
    source = Path(minimax_h3.__file__).read_text(encoding="utf-8")
    assert "_install_teacache" not in source
    assert "teacache" not in source.lower()
    assert not hasattr(minimax_h3, "_install_teacache")


def test_bfloat16_deletes_the_conditioning_noise_augmentation():
    """Why the Ref2VA condition round-trip must stay in float32.

    MiniMax-H3 holds a visual anchor at `t = 0.999`, just short of clean, because the released
    model was trained with its anchors very slightly noised -- conditioning at exactly `t = 1.0` is
    off-distribution. `MiniMaxH3Scheduler.scale_noise` computes `t*x0 + (1 - t)*noise` and casts
    `t` to the *sample's* dtype first.

    bfloat16 has 7 mantissa bits. 0.999 rounds to exactly 1.0 there, so `1 - t` is exactly 0.0 and
    the noise term does not shrink -- it disappears. The augmentation is not degraded, it is
    absent, and the anchor lands exactly on the value the model was trained not to see.
    """
    import torch

    assert torch.tensor(0.999, dtype=torch.bfloat16).item() == 1.0
    assert torch.tensor(0.999, dtype=torch.float32).item() != 1.0

    sample = torch.randn(4, 4)
    noise = torch.randn(4, 4)

    def mix(dtype):
        x0 = sample.to(dtype)
        n = noise.to(dtype)
        t = torch.tensor(0.999, dtype=dtype)  # scale_noise casts to the sample's dtype
        return t * x0 + (1.0 - t) * n

    assert torch.equal(mix(torch.bfloat16), sample.to(torch.bfloat16)), (
        "in bfloat16 the mix returns the clean anchor untouched -- no augmentation happened"
    )
    assert not torch.equal(mix(torch.float32), sample), "float32 must actually mix the noise in"


def test_the_ref2va_condition_round_trip_stays_at_the_precision_it_was_encoded_at():
    """`encode_vae_condition` returns float32; casting it down on reload threw that away."""
    source = Path(minimax_h3_ref.__file__).read_text(encoding="utf-8")
    denoise = source.split("def run_denoise(")[1].split("\ndef ")[0]

    # The two reload comprehensions, matched exactly: a device move and nothing else.
    assert 'c.to("cuda") if isinstance(c, torch.Tensor) else c' in denoise
    assert 'a.to("cuda") if isinstance(a, torch.Tensor) else a' in denoise
    for cast in (
        'c.to("cuda", dtype=torch.bfloat16)',
        'a.to("cuda", dtype=torch.bfloat16)',
    ):
        assert cast not in denoise, f"the condition latents are being cast down again: {cast}"

    # The prompt embeds are a different thing -- activations, not an anchor the scheduler mixes
    # noise into -- and they are still handed over as bfloat16 on purpose.
    assert 'saved["prompt_embeds"].to("cuda", dtype=torch.bfloat16)' in denoise


def test_the_shipped_shifts_do_not_collapse_the_sigma_grid():
    """The guard's premise, measured rather than assumed.

    The shift compresses the grid near sigma = 1, and `unique_consecutive` drops the float32
    duplicates that compression creates. Our published pairs are nowhere near it: at 51 grid
    points the first collapse is above shift ~212,000, and at 9 points ~2.4 million.
    """
    for shift in (TURBO_VIDEO_SHIFT, TURBO_AUDIO_SHIFT, 6.0, 12.0, 3.0):
        for points in (5, 9, 51):
            assert minimax_h3.sigma_grid_length(shift, points) == points

    # Every shipped pair, at every grid size a request can ask for.
    for points in (5, 9, 51):
        minimax_h3.assert_schedules_agree(TURBO_VIDEO_SHIFT, TURBO_AUDIO_SHIFT, points)
        minimax_h3.assert_schedules_agree(6.0, 3.0, points)


def test_a_shift_that_would_desynchronise_the_tracks_is_refused():
    """What AIV_TURBO_VIDEO_SHIFT can reach with a stray zero, or `12e5` typed for `12`."""
    assert minimax_h3.sigma_grid_length(400_000.0, 51) < 51

    with pytest.raises(ValueError, match="desynchronised"):
        minimax_h3.assert_schedules_agree(400_000.0, 3.0, 51)

    message = str(
        pytest.raises(
            ValueError, minimax_h3.assert_schedules_agree, 1_000_000.0, 3.0, 51
        ).value
    )
    assert "lower the larger shift" in message


def test_a_collapse_that_hits_both_tracks_equally_is_reported_not_refused(capsys):
    """Both tracks still agree, so nothing desynchronises -- but the run is a step short."""
    minimax_h3.assert_schedules_agree(1_000_000.0, 1_000_000.0, 51)
    assert "collapsed the sigma grid" in capsys.readouterr().err


def test_the_runtime_guard_catches_what_the_shifts_could_not_predict(capsys):
    """A Base run keeps the scheduler's released default shift, which the cheap check cannot see."""

    class StubScheduler:
        def __init__(self, realized):
            self.realized = realized
            self.sigmas = None
            self.timesteps = None

        def set_timesteps(self, num_inference_steps=None, device=None, sigmas=None):
            self.sigmas = list(range(self.realized))
            self.timesteps = self.sigmas[:-1]

    class StubPipeline:
        def __init__(self, video, audio):
            self.scheduler = StubScheduler(video)
            self.audio_scheduler = StubScheduler(audio)

    agreeing = StubPipeline(51, 51)
    minimax_h3.instrument_scheduler_grid(agreeing)
    agreeing.scheduler.set_timesteps(num_inference_steps=51)
    agreeing.audio_scheduler.set_timesteps(num_inference_steps=51)

    disagreeing = StubPipeline(51, 48)
    minimax_h3.instrument_scheduler_grid(disagreeing)
    disagreeing.scheduler.set_timesteps(num_inference_steps=51)
    with pytest.raises(RuntimeError, match="desynchronised"):
        disagreeing.audio_scheduler.set_timesteps(num_inference_steps=51)
