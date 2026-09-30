"""The LTX attention backend, and the architectural fact that made it necessary.

The vendored selector reaches FlashAttention only on sm90 and sm100. This machine is Ada, sm89,
so no FlashAttention install could ever have changed which kernel LTX runs -- the branch is not
taken. These tests pin the parts that can be checked without a GPU: the selector's own source,
the shapes and constraints, and the promise that an unavailable kernel costs a log line rather
than a run.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from aiv_runtime import ltx_attention
from aiv_runtime.ltx_attention import (
    ATTENTION_BACKENDS,
    DEFAULT_ATTENTION_BACKEND,
    describe_attention,
    sage_head_dim_supported,
    validate_backend,
)

PINNED_ATTENTION = (
    Path(__file__).resolve().parents[2]
    / "vendor/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/attention.py"
)


def test_the_default_is_native_because_nothing_is_installed_yet():
    """Unlike H3, where the kernels were on disk and merely unselected."""
    assert DEFAULT_ATTENTION_BACKEND == "native"
    assert ATTENTION_BACKENDS[0] == "native"


@pytest.mark.parametrize("backend", ATTENTION_BACKENDS)
def test_every_offered_backend_validates(backend):
    assert validate_backend(backend) == backend


def test_an_unknown_backend_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="native, sage"):
        validate_backend("flash")


def test_the_head_dims_sage_has_kernels_for():
    """A ceiling, not a set. `sageattention/core.py` pads head_dim to 64 or 128 itself and only
    raises above 128, so refusing 96 or 48 would have cost a fallback for no reason."""
    assert sage_head_dim_supported(128), "LTX-2.3's own head dim"
    assert sage_head_dim_supported(96)
    assert sage_head_dim_supported(64)
    assert sage_head_dim_supported(48)
    assert not sage_head_dim_supported(256)
    assert not sage_head_dim_supported(0)


class _Fake:
    """A tensor-shaped stand-in: enough surface for the refusal check, no torch."""

    def __init__(self, shape, dtype="torch.bfloat16", device="cuda"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype
        self.device = type("D", (), {"type": device})()


def _refusal(q, k, v, heads):
    return ltx_attention.SageAttention(_Recorder())._refusal(q, k, v, heads)


def test_cross_attention_shapes_are_accepted():
    """The regression. LTX text cross-attention: 6144 video queries, 1024 text keys, 32 heads.

    The first version reshaped k and v to the QUERY's length and raised
    "shape '[1, 6144, 32, 128]' is invalid for input of size 4194304" on the first block of the
    first step -- after the four-minute load, every time.
    """
    q = _Fake((1, 6144, 4096))
    kv = _Fake((1, 1024, 4096))
    assert _refusal(q, kv, kv, 32) is None


def test_self_attention_shapes_are_accepted():
    q = _Fake((1, 6144, 4096))
    assert _refusal(q, q, q, 32) is None


def test_keys_and_values_must_agree_with_each_other():
    q = _Fake((1, 6144, 4096))
    k = _Fake((1, 1024, 4096))
    v = _Fake((1, 512, 4096))
    assert "k and v disagree on length" in _refusal(q, k, v, 32)


def test_a_width_that_does_not_divide_into_heads_is_refused():
    q = _Fake((1, 6144, 4095))
    assert "does not divide" in _refusal(q, q, q, 32)


def test_a_context_projected_to_a_different_width_is_refused():
    q = _Fake((1, 6144, 4096))
    kv = _Fake((1, 1024, 2048))
    assert "disagree on width" in _refusal(q, kv, kv, 32)


def test_a_head_dim_over_the_ceiling_is_refused():
    q = _Fake((1, 6144, 4096))
    assert "head dim=256" in _refusal(q, q, q, 16)


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"device": "cpu"}, "device=cpu"),
        ({"dtype": "torch.float32"}, "dtype=torch.float32"),
    ],
)
def test_the_kernels_hard_constraints_are_refused_not_attempted(kwargs, expected):
    q = _Fake((1, 6144, 4096), **kwargs)
    assert expected in _refusal(q, q, q, 32)


def test_the_log_line_says_when_it_did_not_get_what_was_asked_for():
    line = describe_attention("sage", "native", 0, "ImportError: no module named sageattention")
    assert "native" in line and "asked for sage" in line and "sageattention" in line


@pytest.mark.skipif(not PINNED_ATTENTION.is_file(), reason="the pinned LTX tree is not installed")
def test_flash_is_unreachable_on_this_architecture():
    """The reason this module wraps Sage and not Flash, read from the pinned source.

    `torch.cuda.get_device_capability` returns major 8 on Ada (sm89), 9 on Hopper and 10 on
    datacenter Blackwell. Both FlashAttention branches are guarded on major 9 or 10, so on an
    sm89 card `_select_primary_attention` can only ever fall through to SDPA -- whatever is
    installed. If this test fails because the guards changed, revisit whether Flash is now an
    option here before assuming Sage still is.
    """
    tree = ast.parse(PINNED_ATTENTION.read_text(encoding="utf-8"))
    selector = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_select_primary_attention"
    )
    source = ast.unparse(selector)
    assert "FlashAttention3" in source and "FlashAttention4" in source
    majors = {
        node.comparators[0].value
        for node in ast.walk(selector)
        if isinstance(node, ast.Compare)
        and isinstance(node.left, ast.Name)
        and node.left.id == "major"
        and isinstance(node.comparators[0], ast.Constant)
    }
    assert majors == {9, 10}, f"flash is gated on {majors}, not 9 and 10 -- recheck sm89"
    assert 8 not in majors, "Ada is major 8"


def test_asking_for_native_installs_nothing_and_needs_no_import():
    """It must not import the vendored tree, so it works in a bare test environment too."""
    assert ltx_attention.install_attention("native", object()) == ("native", 0, None)


def test_asking_for_an_unknown_backend_is_an_error_not_a_silent_native():
    with pytest.raises(ValueError):
        ltx_attention.install_attention("flash", object())


@pytest.mark.skipif(not PINNED_ATTENTION.is_file(), reason="the pinned LTX tree is not installed")
def test_the_earlier_hook_would_have_been_wired_to_nothing():
    """Why installation swaps built modules instead of replacing the AUTOMATIC selector.

    `DEFAULT_TRANSFORMER_OPS = TransformerOpsConfig()` is built at import time, and its
    `AttentionOps.attention_function` default factory calls `automatic_attention()` right then.
    Every configurator then takes `ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS` as a
    DEFAULT ARGUMENT, which Python binds to that one object when the `def` executes. So the
    callable is chosen and captured before any adapter can run, and replacing the name later
    reports success while changing nothing -- a lever wired to nothing, which is precisely the
    bug class this project keeps finding.
    """
    transformer_source = PINNED_ATTENTION.parent / "transformer.py"
    configurator_source = PINNED_ATTENTION.parent / "model_configurator.py"
    if not (transformer_source.is_file() and configurator_source.is_file()):
        pytest.skip("the pinned transformer sources are not installed")

    assert "DEFAULT_TRANSFORMER_OPS = TransformerOpsConfig()" in transformer_source.read_text(
        encoding="utf-8"
    ), "the import-time singleton is what makes the selector hook useless"
    assert "ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS" in configurator_source.read_text(
        encoding="utf-8"
    ), "the default-argument binding is the second capture"


class _Site:
    """Stands in for an `Attention` module, which holds the callable as a plain attribute."""

    def __init__(self, name: str):
        self.name = name
        self.attention_function = _Recorder()


class _Pipeline:
    """Shaped like RetakePipeline: the denoiser is one attribute among non-module ones."""

    def __init__(self):
        self.device = "cuda"
        self.dtype = "bfloat16"
        self.stage = _Stage()
        self.video_decoder = object()


class _Stage:
    def __init__(self):
        self.blocks = [_Site("attn1"), _Site("attn2")]


class _LazyStage:
    def __init__(self):
        self.calls = 0

    def run(self, transformer, value):
        self.calls += 1
        return transformer, value


def test_installation_reaches_sites_nested_below_non_modules(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "sageattention", object())
    pipeline = _Pipeline()
    active, swapped, reason = ltx_attention.install_attention("sage", pipeline)
    assert (active, swapped, reason) == ("sage", 2, None)
    assert all(
        isinstance(site.attention_function, ltx_attention.SageAttention)
        for site in pipeline.stage.blocks
    )


def test_finding_nothing_reports_native_rather_than_claiming_sage(monkeypatch):
    """The whole point of counting: a hook attached to nothing must not read as success."""
    monkeypatch.setitem(__import__("sys").modules, "sageattention", object())
    active, swapped, reason = ltx_attention.install_attention("sage", _Pipeline().video_decoder)
    assert active == "native"
    assert swapped == 0
    assert "no attention sites" in reason


def test_a_missing_kernel_is_a_reason_not_an_exception(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "sageattention":
            raise ImportError("No module named 'sageattention'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    active, swapped, reason = ltx_attention.install_attention("sage", _Pipeline())
    assert (active, swapped) == ("native", 0)
    assert "sageattention" in reason


def test_installing_twice_does_not_wrap_twice(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "sageattention", object())
    pipeline = _Pipeline()
    ltx_attention.install_attention("sage", pipeline)
    active, swapped, _ = ltx_attention.install_attention("sage", pipeline)
    assert swapped == 0, "already-wrapped sites are not counted again"
    assert active == "native", "and nothing new was swapped, so nothing new is claimed"


def test_uninstall_restores_the_original_callable(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "sageattention", object())
    pipeline = _Pipeline()
    originals = [site.attention_function for site in pipeline.stage.blocks]
    ltx_attention.install_attention("sage", pipeline)
    assert ltx_attention.uninstall_attention(pipeline) == 2
    assert [site.attention_function for site in pipeline.stage.blocks] == originals


def test_the_log_line_carries_the_count_as_evidence():
    assert describe_attention("sage", "sage", 288) == "attention sage - 288 attention sites"
    assert describe_attention("native", "native", 0) == "attention native"
    fallen_back = describe_attention("sage", "native", 0, "no attention sites found to swap")
    assert "asked for sage" in fallen_back and "no attention sites" in fallen_back


def test_lazy_stage_installs_on_the_transformer_when_run(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "sageattention", object())
    stage = _LazyStage()
    messages = []

    active, swapped, reason = ltx_attention.install_stage_attention(
        "sage", stage, log=messages.append
    )
    assert (active, swapped, reason) == ("sage", -1, None)
    assert describe_attention(active, active, swapped) == (
        "attention sage - deferred until transformer build"
    )

    transformer = _Pipeline().stage
    returned_transformer, value = stage.run(transformer, 42)
    assert returned_transformer is transformer
    assert value == 42
    assert stage.calls == 1
    assert all(
        isinstance(site.attention_function, ltx_attention.SageAttention)
        for site in transformer.blocks
    )
    assert messages[-1] == "attention sage - 2 attention sites"


def test_lazy_stage_native_is_not_wrapped():
    stage = _LazyStage()
    original_run = stage.run
    assert ltx_attention.install_stage_attention("native", stage) == ("native", 0, None)
    assert stage.run == original_run


class _Recorder:
    """Stands in for the callable the vendored selector would have returned."""

    def __init__(self):
        self.calls = 0

    def __call__(self, q, k, v, heads):
        self.calls += 1
        return "fell back"


def test_a_call_sage_cannot_serve_falls_back_and_says_why_once():
    """Thousands of calls per run, so the reason is logged per cause, not per call."""
    fallback = _Recorder()
    lines: list[str] = []
    sage = ltx_attention.SageAttention(fallback, log=lines.append)

    for _ in range(5):
        assert sage("not a tensor", None, None, heads=24) == "fell back"

    assert fallback.calls == 5, "every call still produced a result"
    assert len(lines) == 1, "but the reason was said once"
    assert "fallback" in lines[0]


def test_the_label_is_what_the_vendored_logger_reads():
    assert ltx_attention.SageAttention(_Recorder()).label == "Sage2"
