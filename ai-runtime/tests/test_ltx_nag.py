"""Normalized Attention Guidance: the combination itself, and where it attaches.

CFG cannot carry a negative prompt on a guidance-distilled checkpoint -- its contribution is
`(scale - 1) * (cond - uncond)` and distillation drives that difference toward zero. NAG works
inside the text cross-attention instead and does not depend on that gap surviving.
"""

from __future__ import annotations

import sys
import types

import pytest

from aiv_runtime import ltx_nag
from aiv_runtime.ltx_nag import DEFAULT_ALPHA, DEFAULT_TAU, combine


@pytest.fixture(autouse=True)
def _leave_no_state_behind():
    yield
    ltx_nag.deactivate()
    ltx_nag._PATCHED = False


def _tensor(values):
    torch = pytest.importorskip("torch")
    return torch.tensor([[values]], dtype=torch.float32)


def test_a_zero_scale_is_exactly_the_unguided_output():
    """The identity that makes shipping it off safe: nothing about the result changes."""
    torch = pytest.importorskip("torch")
    positive, negative = _tensor([1.0, 1.0]), _tensor([-1.0, -1.0])
    assert torch.allclose(combine(positive, negative, 0.0, DEFAULT_TAU, DEFAULT_ALPHA), positive)


def test_a_negative_prompt_that_agrees_with_the_positive_changes_nothing():
    torch = pytest.importorskip("torch")
    positive = _tensor([0.5, -2.0])
    assert torch.allclose(combine(positive, positive, 5.0, DEFAULT_TAU, DEFAULT_ALPHA), positive)


def test_the_extrapolation_and_the_blend_are_what_they_claim():
    """pos [1,1], neg [0,0], scale 1 -> guided [2,2], ratio 2 <= tau, out 0.5*[2,2]+0.5*[1,1]."""
    torch = pytest.importorskip("torch")
    out = combine(_tensor([1.0, 1.0]), _tensor([0.0, 0.0]), 1.0, DEFAULT_TAU, DEFAULT_ALPHA)
    assert torch.allclose(out, _tensor([1.5, 1.5]))


def test_the_clamp_pulls_a_runaway_back_to_tau():
    """pos [1,1], neg [-1,-1], scale 3 -> guided [7,7], ratio 7 > 3.5, clamped to [3.5,3.5].

    Without this the mechanism degenerates exactly when the negative prompt disagrees most, which
    is when it is being asked to do the most work.
    """
    torch = pytest.importorskip("torch")
    out = combine(_tensor([1.0, 1.0]), _tensor([-1.0, -1.0]), 3.0, tau=3.5, alpha=0.5)
    assert torch.allclose(out, _tensor([2.25, 2.25]))

    # And with alpha at 1.0 the clamped vector is the output, at exactly tau times the positive.
    fully = combine(_tensor([1.0, 1.0]), _tensor([-1.0, -1.0]), 3.0, tau=3.5, alpha=1.0)
    assert torch.allclose(fully, _tensor([3.5, 3.5]))


def test_alpha_bounds_do_what_they_say():
    torch = pytest.importorskip("torch")
    positive, negative = _tensor([1.0, 1.0]), _tensor([0.0, 0.0])
    assert torch.allclose(combine(positive, negative, 1.0, DEFAULT_TAU, 0.0), positive)
    assert torch.allclose(combine(positive, negative, 1.0, DEFAULT_TAU, 1.0), _tensor([2.0, 2.0]))


def test_a_token_below_tau_is_left_exactly_alone():
    """The clamp is per token: one runaway must not rescale its neighbours."""
    torch = pytest.importorskip("torch")
    positive = torch.tensor([[[1.0, 1.0], [1.0, 1.0]]])
    negative = torch.tensor([[[0.5, 0.5], [-1.0, -1.0]]])
    out = combine(positive, negative, 3.0, tau=3.5, alpha=1.0)
    # First token: guided [2.5,2.5], ratio 2.5 <= 3.5, untouched.
    # Second token: guided [7,7], ratio 7 > 3.5, pulled back to [3.5,3.5].
    assert torch.allclose(out, torch.tensor([[[2.5, 2.5], [3.5, 3.5]]]))

    gentle = combine(positive, torch.tensor([[[0.5, 0.5], [0.0, 0.0]]]), 1.0, tau=3.5, alpha=1.0)
    # [1.5,1.5] and [2,2]; neither exceeds tau, so neither is touched.
    assert torch.allclose(gentle, torch.tensor([[[1.5, 1.5], [2.0, 2.0]]]))


# ---------------------------------------------------------------- attachment


def _stub_transformer_module(monkeypatch):
    calls = []

    class LTXTransformerBlock:
        def __init__(self):
            self.attn2 = object()
            self.audio_attn2 = object()

        def _apply_text_cross_attention(self, x_normed, context, attn, *args, **kwargs):
            calls.append((context, attn))
            return context

    module = types.SimpleNamespace(LTXTransformerBlock=LTXTransformerBlock)
    monkeypatch.setitem(
        sys.modules,
        "ltx_core.model.transformer",
        types.SimpleNamespace(transformer=module),
    )
    return module, calls


def test_the_patch_falls_through_when_nothing_is_activated(monkeypatch):
    """Installing is not turning on: an unguided run pays one `is None` test per block."""
    module, calls = _stub_transformer_module(monkeypatch)
    assert ltx_nag.install() is True
    assert ltx_nag.active() is False

    block = module.LTXTransformerBlock()
    assert block._apply_text_cross_attention("x", "positive", block.attn2) == "positive"
    assert calls == [("positive", block.attn2)]


def test_installing_twice_does_not_wrap_the_wrapper(monkeypatch):
    module, _ = _stub_transformer_module(monkeypatch)
    ltx_nag.install()
    wrapped = module.LTXTransformerBlock._apply_text_cross_attention
    ltx_nag.install()
    assert module.LTXTransformerBlock._apply_text_cross_attention is wrapped


def test_it_dispatches_on_the_attention_module_not_on_tensor_identity(monkeypatch):
    """The context is rebuilt per step; `attn2` versus `audio_attn2` is stable and exact."""
    torch = pytest.importorskip("torch")
    module, calls = _stub_transformer_module(monkeypatch)

    ltx_nag.activate(
        scale=1.0,
        negative_video_context=torch.zeros(1, 1, 2),
        negative_audio_context=None,
    )
    block = module.LTXTransformerBlock()

    positive = torch.ones(1, 1, 2)
    video_out = block._apply_text_cross_attention("x", positive, block.attn2)
    # Two calls to the original: positive, then negative.
    assert len(calls) == 2
    assert torch.allclose(video_out, torch.tensor([[[1.5, 1.5]]]))
    assert ltx_nag.state().applied == {"video": 1}

    # Audio has no negative context, so it is left alone -- one call, and the positive returned.
    calls.clear()
    audio_out = block._apply_text_cross_attention("x", positive, block.audio_attn2)
    assert len(calls) == 1
    assert torch.allclose(audio_out, positive)

    # An attention that is neither is not ours to guide.
    calls.clear()
    other = object()
    assert torch.allclose(block._apply_text_cross_attention("x", positive, other), positive)
    assert len(calls) == 1


def test_activate_refuses_settings_that_would_not_mean_anything(monkeypatch):
    _stub_transformer_module(monkeypatch)
    torch = pytest.importorskip("torch")
    negative = torch.zeros(1, 1, 2)

    with pytest.raises(ValueError, match="scale must be positive"):
        ltx_nag.activate(scale=0.0, negative_video_context=negative)
    with pytest.raises(ValueError, match="tau must be at least 1.0"):
        ltx_nag.activate(scale=1.0, negative_video_context=negative, tau=0.5)
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        ltx_nag.activate(scale=1.0, negative_video_context=negative, alpha=1.5)
    with pytest.raises(ValueError, match="negative caption embedding"):
        ltx_nag.activate(scale=1.0)


def test_the_description_says_whether_it_is_on():
    assert "NAG off" in ltx_nag.describe()


def test_it_ships_off_and_the_adapter_says_so():
    from aiv_runtime import ltx_single_stage

    defaults = {action.dest: action.default for action in ltx_single_stage.build_parser()._actions}
    assert defaults["nag_scale"] == 0.0, "NAG must be opt-in until it has been run against a picture"
    assert defaults["nag_tau"] == DEFAULT_TAU == 3.5
    assert defaults["nag_alpha"] == DEFAULT_ALPHA == 0.5

    import argparse

    off = argparse.Namespace(negative_prompt="blurry", nag_scale=0.0)
    on = argparse.Namespace(negative_prompt="blurry", nag_scale=3.0)
    blank = argparse.Namespace(negative_prompt="   ", nag_scale=3.0)
    assert ltx_single_stage.nag_requested(off) is False
    assert ltx_single_stage.nag_requested(on) is True
    assert ltx_single_stage.nag_requested(blank) is False, "nothing to steer away from"
