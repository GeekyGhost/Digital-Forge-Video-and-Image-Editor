"""SageAttention for the LTX transformer, which the vendored pipeline cannot reach on its own.

The vendored `ltx_core.model.transformer.attention` knows exactly three backends: PyTorch SDPA,
FlashAttention 3 (`flash_attn_interface`) and FlashAttention 4 (`flash_attn.cute`). Its selector
reads, in full:

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major == 9:  ...FA3, then FA4
        if major == 10 and flash_attn_4_func is not None: ...FA4
    return _sdpa_full_priority()

`major` is 9 on Hopper and 10 on datacenter Blackwell. This machine is an RTX 4090 Laptop --
Ada, sm89, major 8 -- so both flash branches are unreachable by architecture, not by packaging.
Installing FlashAttention here would change nothing at all: the LTX transformer would still run
on SDPA. That is why this module exists and why it wraps Sage rather than Flash.

Sage is the kernel that does target sm89: woct0rdho's SageAttention 2.2.0 Windows wheels state
support for "RTX 40xx (sm89) and 50xx (sm120) GPUs, and CUDA >= 12.8, therefore PyTorch >= 2.7",
and `sageattention-2.2.0+cu128torch2.7.1-cp310-cp310-win_amd64.whl` matches the LTX venv exactly
-- cu128, torch 2.7.1, CPython 3.10, Windows -- so no torch bump is involved. `scripts\\
install-ltx-attention.ps1` installs that wheel and its matching triton.

Nothing here modifies the vendored packages. The selector is a module-level function that
`AttentionFunction.AUTOMATIC.to_callable()` resolves by name at call time, so replacing that name
before the transformer is built is enough, and putting the original back removes every trace.

Only the UNMASKED path is replaced. The masked path keeps SDPA, because the additive masks LTX
passes there are not something Sage's kernels accept.
"""

from __future__ import annotations

import types

#: The `--attention-backend` choices, in the order the panel should offer them.
ATTENTION_BACKENDS = ("native", "sage")

#: Native until a run proves parity on this machine. Unlike H3, where the kernels were installed
#: and merely unselected, nothing is installed here yet -- so the default cannot assume Sage.
DEFAULT_ATTENTION_BACKEND = "native"

#: The largest head dimension the kernel accepts. Read from the installed
#: `sageattention/core.py`, which pads head_dim up to 64 or 128 itself and raises above 128:
#:
#:     elif head_dim_og > 128:
#:         raise ValueError(f"Unsupported head_dim: {head_dim_og}")
#:
#: So the real constraint is a ceiling, not a set of sizes. LTX-2.3's transformer is 128.
SAGE_MAX_HEAD_DIM = 128


def sage_head_dim_supported(dim_head: int) -> bool:
    """Whether Sage has a kernel for this head dimension."""
    return 0 < dim_head <= SAGE_MAX_HEAD_DIM


def validate_backend(backend: str) -> str:
    """Return `backend` if it names one, else raise with the list."""
    if backend not in ATTENTION_BACKENDS:
        raise ValueError(
            f"unknown attention backend {backend!r}; expected one of {', '.join(ATTENTION_BACKENDS)}"
        )
    return backend


def describe_attention(backend: str, active: str, swapped: int = 0, reason: str | None = None) -> str:
    """One line for the run log: what was asked for, what is running, and the evidence.

    `swapped` is in the line on purpose. "attention sage" is a claim; "attention sage - 288
    attention sites" is a measurement, and it is the only thing that distinguishes a kernel that
    is running from a hook that silently attached to nothing.
    """
    if backend == active and active == "native":
        return "attention native"
    if backend == active and swapped < 0:
        return f"attention {active} - deferred until transformer build"
    if backend == active:
        return f"attention {active} - {swapped} attention sites"
    return f"attention {active} (asked for {backend}: {reason or 'unavailable'})"


class SageAttention:
    """LTX-shaped unmasked attention on SageAttention 2, falling back per call.

    The vendored protocol is `(q, k, v, heads) -> out` with q/k/v shaped
    `[batch, seq, heads * dim_head]`, which is what `PytorchAttention` reshapes internally. This
    keeps the tensors in NHD (`[batch, seq, heads, dim_head]`) rather than transposing to HND,
    because that is both the cheaper reshape and the layout the padding below is written for.

    A call that Sage cannot serve goes to `fallback`, which is whatever the selector would have
    returned -- so the worst case is the previous behaviour plus one line in the log naming the
    reason. The reason is logged once per distinct cause, not once per call: a 48-block model at
    8 steps would otherwise print the same sentence thousands of times.
    """

    def __init__(self, fallback, log=None) -> None:
        self._fallback = fallback
        self._log = log
        self._reported: set[str] = set()

    @property
    def label(self) -> str:
        """`attention_label()` in the vendored module reads this before the class name."""
        return "Sage2"

    @property
    def fallback(self):
        """The callable this one wraps, so `uninstall` can put it back."""
        return self._fallback

    def _report(self, reason: str) -> None:
        if reason in self._reported:
            return
        self._reported.add(reason)
        if self._log is not None:
            self._log(f"attention Sage per-call fallback to SDPA ({reason})")

    def _refusal(self, q, k, v, heads: int) -> str | None:
        """Why Sage cannot serve this call, or None when it can.

        Deliberately duck-typed rather than importing torch: this runs on every attention call
        of every block of every step, and the module stays importable -- and testable -- in an
        environment without torch, the way `ltx_geometry` and `ltx_offload` do.
        """
        tensors = (q, k, v)
        if not all(hasattr(tensor, "dtype") and hasattr(tensor, "device") for tensor in tensors):
            return "not tensors"
        if not all(getattr(tensor.device, "type", None) == "cuda" for tensor in tensors):
            return f"device={getattr(q.device, 'type', 'unknown')}"
        if not (q.dtype == k.dtype == v.dtype):
            return "mixed dtypes"
        if str(q.dtype) not in ("torch.float16", "torch.bfloat16"):
            return f"dtype={q.dtype}"
        if not all(getattr(tensor, "ndim", None) == 3 for tensor in tensors):
            return f"rank={getattr(q, 'ndim', 'unknown')}"
        if heads <= 0 or q.shape[-1] % heads:
            return f"width {q.shape[-1]} does not divide into {heads} heads"
        if k.shape[-1] != q.shape[-1] or v.shape[-1] != q.shape[-1]:
            return "q, k and v disagree on width"
        # Cross-attention hands us keys and values from the text context, so they share a length
        # with each other but not with the queries. Only that pairing has to hold.
        if k.shape[1] != v.shape[1]:
            return "k and v disagree on length"
        dim_head = q.shape[-1] // heads
        if not sage_head_dim_supported(dim_head):
            return f"head dim={dim_head}"
        return None

    def __call__(self, q, k, v, heads: int):
        refusal = self._refusal(q, k, v, heads)
        if refusal is not None:
            self._report(refusal)
            return self._fallback(q, k, v, heads)

        from sageattention import sageattn

        batch, q_len, _ = q.shape
        dim_head = q.shape[-1] // heads
        # Each tensor keeps ITS OWN length. Reshaping all three to the query's length is correct
        # only for self-attention; LTX's text cross-attention passes 6144 video queries against
        # 1024 text keys, and the shared-length version raised
        # "shape '[1, 6144, 32, 128]' is invalid for input of size 4194304" on the first block.
        query = q.reshape(batch, q_len, heads, dim_head)
        key = k.reshape(batch, k.shape[1], heads, dim_head)
        value = v.reshape(batch, v.shape[1], heads, dim_head)

        # No sequence padding. The H3 adapter pads to a 64-token stride and this copied it, but
        # the installed `sageattention/core.py` pads head_dim, never the sequence, and takes any
        # length. Padding keys and values is worse than unnecessary: zero-filled key positions
        # still score against every real query, so softmax hands them weight and they drag the
        # output toward zero. Aligning the query alone would be harmless; aligning k and v would
        # quietly corrupt every result.
        out = sageattn(query, key, value, tensor_layout="NHD", is_causal=False)
        return out.reshape(batch, q_len, heads * dim_head)


def _attention_holders(root, max_depth: int = 6):
    """Yield every object that holds an `attention_function`, from any root.

    Duck-typed on purpose. The root handed in is a pipeline, not a module: `RetakePipeline` keeps
    its denoiser on `.stage` alongside encoders and decoders that are not modules at all, so a
    plain `root.modules()` would raise and a hardcoded `root.stage.model` would break the first
    time the vendored pipeline moved an attribute.
    """
    seen: set[int] = set()

    def walk(node, depth: int):
        if depth < 0 or node is None or id(node) in seen:
            return
        seen.add(id(node))
        if isinstance(node, (list, tuple, set)):
            for item in node:
                yield from walk(item, depth)
            return
        if hasattr(node, "attention_function"):
            yield node
        modules = getattr(node, "modules", None)
        if callable(modules):
            try:
                children = list(modules())
            except Exception:
                children = []
            for child in children:
                if child is not node and hasattr(child, "attention_function") and id(child) not in seen:
                    seen.add(id(child))
                    yield child
            return
        for value in list(getattr(node, "__dict__", {}).values()):
            if isinstance(value, (str, bytes, int, float, bool)):
                continue
            yield from walk(value, depth - 1)

    yield from walk(root, max_depth)


def install_attention(backend: str, root, log=None) -> tuple[str, int, str | None]:
    """Swap Sage into every attention site reachable from `root`. Returns `(active, swapped, reason)`.

    Call this AFTER the pipeline is built, and never before.

    The obvious hook -- replacing the module-level `automatic_attention` that
    `AttentionFunction.AUTOMATIC.to_callable()` consults -- looks right and does nothing. The
    vendored tree builds `DEFAULT_TRANSFORMER_OPS = TransformerOpsConfig()` at import time, whose
    `AttentionOps.attention_function` default factory calls `automatic_attention()` once, right
    then; and each configurator takes `ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS` as a
    default argument, which Python binds to that object when the `def` executes. So by the time
    any adapter could call this, the callable is already chosen and captured twice over, and
    replacing the name afterwards changes nothing while reporting success. That is the shape of
    bug this codebase keeps finding: a lever that is wired to nothing, reported as engaged.

    Swapping the built modules cannot lie in that way, because the count IS the evidence. Zero
    swapped is reported as native, not as sage.

    Never raises for an unavailable kernel: a missing package or an old torch is a reason to run
    on SDPA and say so, not a reason to lose the run.
    """
    validate_backend(backend)
    if backend == "native":
        return "native", 0, None

    try:
        import sageattention  # noqa: F401
    except Exception as error:  # ImportError, or a DLL/triton failure on Windows
        return "native", 0, f"{type(error).__name__}: {error}"

    swapped = 0
    for holder in _attention_holders(root):
        current = getattr(holder, "attention_function", None)
        if current is None or isinstance(current, SageAttention):
            continue
        holder.attention_function = SageAttention(current, log=log)
        swapped += 1

    if swapped == 0:
        return "native", 0, "no attention sites found to swap"
    return "sage", swapped, None


def install_stage_attention(backend: str, stage, log=None) -> tuple[str, int, str | None]:
    """Install attention after a lazy ``DiffusionStage`` builds its transformer.

    ``RetakePipeline`` and the foley adapter keep only a transformer builder on the stage.  The
    attention modules do not exist until ``DiffusionStage.__call__`` enters its model context and
    calls ``run``.  Traversing the pipeline after construction therefore finds zero sites even
    though single-stage generation (which builds explicitly) finds all 288.

    Wrap the stage's instance ``run`` method so the normal, measured installer sees the actual
    transformer immediately after it is built.  A fresh streaming transformer is created for
    every call, so installation is repeated per call and the measured site count is logged each
    time.  The vendored stage and builder remain untouched.
    """

    validate_backend(backend)
    if backend == "native":
        return "native", 0, None

    try:
        import sageattention  # noqa: F401
    except Exception as error:  # ImportError, or a DLL/triton failure on Windows
        return "native", 0, f"{type(error).__name__}: {error}"

    if not callable(getattr(stage, "run", None)):
        return "native", 0, "lazy diffusion stage has no callable run method"
    if getattr(stage, "_aiv_sage_attention_deferred", False):
        return "sage", -1, None

    original_run = stage.run

    def run_with_attention(_stage, transformer, *args, **kwargs):
        active, swapped, reason = install_attention("sage", transformer, log=log)
        if log is not None:
            log(describe_attention("sage", active, swapped, reason))
        return original_run(transformer, *args, **kwargs)

    stage.run = types.MethodType(run_with_attention, stage)
    stage._aiv_sage_attention_deferred = True
    return "sage", -1, None


def uninstall_attention(root) -> int:
    """Put the original callable back everywhere it was replaced. Returns how many were restored."""
    restored = 0
    for holder in _attention_holders(root):
        current = getattr(holder, "attention_function", None)
        if isinstance(current, SageAttention):
            holder.attention_function = current.fallback
            restored += 1
    return restored
