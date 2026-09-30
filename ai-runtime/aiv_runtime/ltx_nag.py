"""Normalized Attention Guidance for LTX 2.3: a negative prompt that works on a distilled model.

`ltx_retake_16gb.py` explains why classifier-free guidance cannot carry a negative prompt here.
CFG's entire contribution is `(scale - 1) * (cond - uncond)`, and guidance distillation trains the
conditioned prediction to already contain the guidance, which drives that difference toward zero.
Multiplying a near-zero by a larger number changes nothing, which is exactly what "the negative
prompt does nothing" means.

NAG does not depend on that gap. It works inside the **text cross-attention**: attend the same
queries against the positive caption and against the negative one, extrapolate away from the
negative, and normalize the result back to the positive's scale.

    z_guide = z_pos + scale * (z_pos - z_neg)
    ratio   = |z_guide|_1 / |z_pos|_1                  per token
    z_guide = z_guide * (tau / ratio)   where ratio > tau
    out     = alpha * z_guide + (1 - alpha) * z_pos

The clamp is the "normalized" part and is what stops the extrapolation running away: without it,
a negative prompt that disagrees strongly with the positive one produces an enormous vector and
the sample degenerates. Reference values are `tau = 3.5` and `alpha = 0.5`.

Cost is one extra **text** cross-attention per block per modality per step -- not a second pass
over the whole transformer. The caption is a few hundred tokens against thousands of video tokens,
so this is a small fraction of a step, which is the whole reason to prefer it to CFG.

Where it attaches
-----------------
`LTXTransformerBlock._apply_text_cross_attention` is the one seam that carries the caption, and it
is called exactly twice per block: once with `self.attn2` for video and once with
`self.audio_attn2` for audio. The wrapper dispatches on which of those it was handed, which is an
exact test and does not depend on tensor identity surviving a step boundary. It calls the original
method rather than reimplementing it, so the AdaLN-modulated branch is covered for free.

The vendored tree is left untouched, as with `scaled_fp8_lora` and the pread loader.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Reference values. `tau` bounds how far the guided output may exceed the positive one in L1;
#: `alpha` blends the clamped result back toward the positive.
DEFAULT_TAU = 3.5
DEFAULT_ALPHA = 0.5


@dataclass
class NagState:
    """What the patched cross-attention needs, set once before the denoising loop.

    The negative caption embedding is constant across steps, so this is built once and read on
    every block's forward.
    """

    scale: float
    tau: float = DEFAULT_TAU
    alpha: float = DEFAULT_ALPHA
    negative_video_context: object | None = None
    negative_audio_context: object | None = None
    #: Blocks that actually ran with guidance, for the run log. A count of zero means the wrapper
    #: was installed and never fired, which is worth knowing before blaming the mechanism.
    applied: dict[str, int] = field(default_factory=dict)

    def negative_for(self, modality: str):
        return self.negative_video_context if modality == "video" else self.negative_audio_context


#: Module-level because the wrapper is a patched method on a vendored class and has nowhere else
#: to look. `None` means inactive, and inactive is the default.
_STATE: NagState | None = None
_PATCHED = False


def active() -> bool:
    return _STATE is not None


def state() -> NagState | None:
    return _STATE


def deactivate() -> None:
    """Leave the patch installed but inert. The wrapper falls straight through when inactive."""
    global _STATE
    _STATE = None


def activate(
    scale: float,
    negative_video_context=None,
    negative_audio_context=None,
    tau: float = DEFAULT_TAU,
    alpha: float = DEFAULT_ALPHA,
) -> NagState:
    """Turn guidance on for the next run, and install the patch if it is not already there."""
    global _STATE
    if scale <= 0.0:
        raise ValueError(f"nag scale must be positive to activate, got {scale}")
    if tau < 1.0:
        raise ValueError(f"nag tau must be at least 1.0 (no clamp below the positive), got {tau}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"nag alpha must be in [0, 1], got {alpha}")
    if negative_video_context is None and negative_audio_context is None:
        raise ValueError("nag needs a negative caption embedding for at least one modality")
    install()
    _STATE = NagState(
        scale=scale,
        tau=tau,
        alpha=alpha,
        negative_video_context=negative_video_context,
        negative_audio_context=negative_audio_context,
    )
    return _STATE


def combine(positive, negative, scale: float, tau: float, alpha: float):
    """The NAG combination itself, on two cross-attention outputs of shape `(B, T, dim)`.

    Separated from the patch so it can be tested against hand-computed values without a model.
    """
    import torch

    guided = positive + scale * (positive - negative)
    positive_norm = positive.abs().mean(dim=-1, keepdim=True).clamp_min(1e-8)
    guided_norm = guided.abs().mean(dim=-1, keepdim=True)
    ratio = guided_norm / positive_norm
    # Only the tokens that ran away are pulled back; the rest are left exactly as computed.
    guided = torch.where(ratio > tau, guided * (tau / ratio.clamp_min(1e-8)), guided)
    return alpha * guided + (1.0 - alpha) * positive


def install() -> bool:
    """Wrap `LTXTransformerBlock._apply_text_cross_attention`. Idempotent.

    Installing does nothing on its own: the wrapper falls through to the original whenever
    `activate` has not been called, so an installed patch on an unguided run costs one `is None`
    test per block.
    """
    global _PATCHED
    if _PATCHED:
        return True

    from ltx_core.model.transformer import transformer as transformer_module

    block_class = transformer_module.LTXTransformerBlock
    original = block_class._apply_text_cross_attention

    def guided_text_cross_attention(self, x_normed, context, attn, *args, **kwargs):
        current = _STATE
        if current is None:
            return original(self, x_normed, context, attn, *args, **kwargs)

        if attn is getattr(self, "attn2", None):
            modality = "video"
        elif attn is getattr(self, "audio_attn2", None):
            modality = "audio"
        else:
            # Not a text cross-attention we recognise; guidance is not ours to apply.
            return original(self, x_normed, context, attn, *args, **kwargs)

        negative = current.negative_for(modality)
        positive_out = original(self, x_normed, context, attn, *args, **kwargs)
        if negative is None:
            return positive_out

        negative = negative.to(device=context.device, dtype=context.dtype)
        negative_out = original(self, x_normed, negative, attn, *args, **kwargs)
        current.applied[modality] = current.applied.get(modality, 0) + 1
        return combine(positive_out, negative_out, current.scale, current.tau, current.alpha)

    guided_text_cross_attention._aiv_nag_original = original
    block_class._apply_text_cross_attention = guided_text_cross_attention
    _PATCHED = True
    return True


def uninstall() -> None:
    """Put the vendored method back. Used by tests; a worker process simply exits."""
    global _PATCHED
    if not _PATCHED:
        return
    from ltx_core.model.transformer import transformer as transformer_module

    block_class = transformer_module.LTXTransformerBlock
    installed = block_class._apply_text_cross_attention
    original = getattr(installed, "_aiv_nag_original", None)
    if original is not None:
        block_class._apply_text_cross_attention = original
    _PATCHED = False
    deactivate()


def describe() -> str:
    """One line for the run log."""
    current = _STATE
    if current is None:
        return "NAG off · the negative prompt only reaches the model through CFG"
    modalities = ", ".join(
        name for name, value in (
            ("video", current.negative_video_context), ("audio", current.negative_audio_context)
        ) if value is not None
    )
    return (
        f"NAG on · scale {current.scale:g} · tau {current.tau:g} · alpha {current.alpha:g} · "
        f"{modalities} · one extra text cross-attention per block, not a second full pass"
    )
