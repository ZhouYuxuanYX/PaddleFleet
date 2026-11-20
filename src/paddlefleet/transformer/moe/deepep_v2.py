# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepEP V2.5 (``EPBuffer``) dispatch/combine for the ``deepep_v2`` MoE backend.

Mirrors the V1 ``deepep`` PyLayers in ``fused_a2a.py`` so that ``_DeepEPV2Manager`` can reuse the
V1 permute/unpermute logic unchanged:

- ``DeepEPV2Dispatch``: forward dispatch, backward combine(grad, handle, topk_weights=grad_probs).
- ``DeepEPV2Combine``: forward combine, backward cached dispatch(grad, handle).
- ``DeepEPV2CombineAsync``: combine overlapped with the shared experts.

Differences from V1 that callers do not see:

- The buffer must know the maximum number of tokens per rank. Set it with
  ``deepep_v2_buffer_configs={"num_max_tokens_per_rank": N}``; otherwise each forward dispatch
  all-reduces the local token count over the EP group so all ranks grow the buffer together
  (buffer construction is collective).
- Async work is synchronized with the ``EventOverlap`` returned by DeepEP rather than through the
  Paddle process group's comm stream.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import paddle
import paddle.distributed as dist
from paddle import framework
from paddle.autograd import PyLayer

from .fp8_utils import FP8_ALIGN
from .fused_a2a import (
    _quantize_for_deepep_dispatch,
    _record_fp8_combine_grad,
    _sonicmoe_quantize,
    barrier_ep,
)
from .moe_utils import manual_backward

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from paddle.distributed.communication.group import Group

try:
    from paddlefleet_ops import (
        is_deep_ep_v2_available as _is_deep_ep_v2_available,
    )
except ImportError:
    HAVE_DEEP_EP_V2 = False
else:
    HAVE_DEEP_EP_V2 = _is_deep_ep_v2_available()
if HAVE_DEEP_EP_V2:
    from paddlefleet_ops import deep_ep_v2
else:
    deep_ep_v2 = None

# Buffers are grown to a multiple of this many tokens to avoid rebuilding for small changes.
_TOKEN_ALIGNMENT = 256

_DEFAULTS = {
    "num_max_tokens_per_rank": None,
    "num_sms": 0,
    "do_expand": False,
    "expert_alignment": FP8_ALIGN,
    "deterministic": False,
}

_config = {
    "num_max_tokens_per_rank": None,
    "num_sms": 0,
    # Expanding layout: dispatch returns expert-contiguous rows (one per (token, local expert)),
    # so the framework's permute/unpermute can be skipped. Off by default; when off, this backend
    # behaves exactly as before.
    "do_expand": False,
    "expert_alignment": FP8_ALIGN,
    # DeepEP V2.5 deterministic routing: received tokens are sorted so the dispatch output (and
    # hence the expert weight-gradient accumulation order) is reproducible. Fixed at buffer build.
    "deterministic": False,
}


@dataclass
class _BufferEntry:
    buffer: object
    num_max_tokens_per_rank: int
    hidden: int
    num_topk: int
    deterministic: bool = False


# One buffer per EP group, grown monotonically.
_buffers: dict[int, _BufferEntry] = {}
# Replaced buffers are kept alive on purpose: saved handles of in-flight layers may still
# use them, and destroying a buffer is collective, so it must not depend on GC timing.
_retired_buffers: list = []


def is_deep_ep_v2_available() -> bool:
    """Return whether paddlefleet-ops loaded the optional DeepEP V2.5 runtime."""
    return HAVE_DEEP_EP_V2


def configure_buffer_v2(
    num_max_tokens_per_rank: int | None = None,
    num_sms: int | None = None,
    do_expand: bool | None = None,
    expert_alignment: int | None = None,
    deterministic: bool | None = None,
):
    """Configure the DeepEP V2.5 backend (``deepep_v2_buffer_configs``).

    Args:
        num_max_tokens_per_rank: static upper bound of tokens per rank per dispatch. When set, the
            per-dispatch token-count all-reduce is skipped and exceeding it raises an error.
        num_sms: SMs used by the communication kernels; 0 lets DeepEP pick from bandwidths.
        do_expand: use the expanding layout so dispatch returns expert-contiguous rows and the
            framework's permute/unpermute is skipped (see ``DeepEPV2ExpandMoePyLayer``).
        expert_alignment: per-expert row alignment in the expanding layout; must stay a multiple of
            4 so the FP8 scale keeps its unpadded TMA-aligned leading dimension. Defaults to
            ``FP8_ALIGN``.
        deterministic: use DeepEP V2.5's deterministic dispatch (reproducible receive order).
    """
    _config.update(_DEFAULTS)
    if num_max_tokens_per_rank is not None:
        if num_max_tokens_per_rank <= 0:
            raise ValueError("num_max_tokens_per_rank must be positive.")
        _config["num_max_tokens_per_rank"] = int(num_max_tokens_per_rank)
    if num_sms is not None:
        if num_sms < 0:
            raise ValueError("num_sms must be non-negative.")
        _config["num_sms"] = int(num_sms)
    if do_expand is not None:
        _config["do_expand"] = bool(do_expand)
    if expert_alignment is not None:
        if expert_alignment <= 0 or expert_alignment % 4 != 0:
            raise ValueError(
                "expert_alignment must be positive and a multiple of 4."
            )
        _config["expert_alignment"] = int(expert_alignment)
    if deterministic is not None:
        _config["deterministic"] = bool(deterministic)


def is_expand_enabled() -> bool:
    """Whether the expanding layout is configured (``deepep_v2_buffer_configs['do_expand']``)."""
    return bool(_config["do_expand"])


@dataclass(frozen=True)
class KernelConfig:
    """The dispatch geometry one model is built with: a snapshot of ``_config``.

    ``configure_buffer_v2`` writes module-level state, so a second model built later in the same
    process (an RL reference model next to the actor) would otherwise silently change the first
    model's dispatch under it. A layer freezes this when it is built and hands it to every
    dispatch; the forward additionally records the layout in ``states`` so its own backward
    reproduces it. The buffer geometry (``deterministic``, ``num_max_tokens_per_rank``) is not
    part of it: there is one buffer per EP group, shared by every model on that group.
    """

    num_sms: int
    do_expand: bool
    expert_alignment: int


def kernel_config_snapshot() -> KernelConfig:
    """Freeze the current module configuration; call it where a layer is constructed."""
    return KernelConfig(
        num_sms=int(_config["num_sms"]),
        do_expand=bool(_config["do_expand"]),
        expert_alignment=int(_config["expert_alignment"]),
    )


def reset_buffer_v2():
    """Drop all cached buffers and restore the default configuration (for tests)."""
    _buffers.clear()
    _config.update(
        num_max_tokens_per_rank=None,
        num_sms=0,
        do_expand=False,
        expert_alignment=FP8_ALIGN,
        deterministic=False,
    )


def _required_num_tokens(group: Group, num_tokens: int) -> int:
    static_max = _config["num_max_tokens_per_rank"]
    if static_max is not None:
        if num_tokens > static_max:
            raise RuntimeError(
                f"DeepEP V2 dispatch got {num_tokens} tokens on this rank, more than "
                f"deepep_v2_buffer_configs['num_max_tokens_per_rank']={static_max}."
            )
        return static_max
    count = paddle.to_tensor([num_tokens], dtype="int64")
    dist.all_reduce(count, op=dist.ReduceOp.MAX, group=group)
    num_tokens = int(count.item())
    return -(-num_tokens // _TOKEN_ALIGNMENT) * _TOKEN_ALIGNMENT


def get_buffer_v2(group: Group, num_tokens: int, hidden: int, num_topk: int):
    """Get the EP group's buffer, growing it (collectively) when it is too small.

    Must be called by every rank of ``group`` at the same point, like any forward dispatch.
    """
    if not HAVE_DEEP_EP_V2:
        raise ImportError(
            "moe_token_dispatcher_type='deepep_v2' requires paddlefleet_ops.deep_ep_v2, "
            "which is built only when the NCCL paddlefleet_ops was built against provides "
            "the device API (see _DEEPEP_V2_MIN_NCCL in build_utils.py)."
        )
    num_max_tokens = _required_num_tokens(group, num_tokens)
    entry = _buffers.get(group.id)
    if (
        entry is not None
        and entry.num_max_tokens_per_rank >= num_max_tokens
        and entry.hidden >= hidden
        and entry.num_topk >= num_topk
        and entry.deterministic == _config["deterministic"]
    ):
        return entry.buffer
    if entry is not None:
        num_max_tokens = max(num_max_tokens, entry.num_max_tokens_per_rank)
        hidden = max(hidden, entry.hidden)
        num_topk = max(num_topk, entry.num_topk)
        _retired_buffers.append(entry.buffer)
    # Sized for BF16, which also covers FP8 payloads plus their scales.
    buffer = deep_ep_v2.EPBuffer(
        group,
        num_max_tokens_per_rank=num_max_tokens,
        hidden=hidden,
        num_topk=num_topk,
        deterministic=_config["deterministic"],
        explicitly_destroy=True,
    )
    _buffers[group.id] = _BufferEntry(
        buffer, num_max_tokens, hidden, num_topk, _config["deterministic"]
    )
    return buffer


def _quantize_for_dispatch(x, use_ue8m0: bool, using_sonic_moe: bool):
    """FP8 quant for a V2.5 dispatch: V1's recipe with the branches V2.5 cannot reach pinned."""
    return _quantize_for_deepep_dispatch(
        x,
        use_ue8m0,
        using_sonic_moe,
        use_w4a8=False,
        sonic_fp8_group_size=32,
    )


def _expand_kwargs(
    fp8_dispatch: bool, do_expand: bool, expert_alignment: int
) -> dict:
    """Extra dispatch kwargs for the expanding layout (empty when it is off)."""
    if not do_expand:
        return {}
    return {
        "do_expand": True,
        # Zero the alignment gaps: the expert GEMM reads the padded rows too.
        "do_zero_padding": True,
        "expert_alignment": expert_alignment,
        # Column-major scale, which is the layout ExpertsGroupGemmContiguousNode builds anyway.
        "use_tma_aligned_col_major_sf": fp8_dispatch,
    }


def _cached_dispatch(
    buffer,
    grad,
    handle,
    async_finish: bool = False,
    fp8: bool = False,
    *,
    do_expand: bool = False,
    expert_alignment: int = FP8_ALIGN,
):
    if isinstance(grad, tuple):
        grad = (grad[0].contiguous(), grad[1].contiguous())
    else:
        grad = grad.contiguous()
    # num_sms must be passed for a cached dispatch: DeepEP V2.5 estimates the SM count from
    # num_experts before it reads num_experts from the handle. The handle does not carry the
    # expanding layout either, so the caller has to hand back the forward's snapshot.
    recv_x, _, _, _, event = buffer.dispatch(
        grad,
        handle=handle,
        num_sms=handle.num_sms,
        async_with_compute_stream=async_finish,
        **_expand_kwargs(fp8, do_expand, expert_alignment),
    )
    return recv_x, event


_logged_first_dispatch = False


def _log_first_dispatch(handle, group, num_experts, num_topk, num_tokens):
    """Log the resolved kernel/buffer geometry once per process."""
    global _logged_first_dispatch
    if _logged_first_dispatch:
        return
    _logged_first_dispatch = True
    try:
        entry = _buffers.get(group.id)
        logger.info(
            "[deepep_v2] first dispatch: num_sms=%s (config=%s, 0=auto) "
            "num_tokens_this_rank=%s num_max_tokens_per_rank=%s (config=%s) "
            "num_experts=%s num_topk=%s deterministic=%s",
            getattr(handle, "num_sms", "?"),
            _config["num_sms"],
            num_tokens,
            getattr(entry, "num_max_tokens_per_rank", "?"),
            _config["num_max_tokens_per_rank"],
            num_experts,
            num_topk,
            _config["deterministic"],
        )
    except (AttributeError, KeyError, TypeError):
        logger.warning(
            "[deepep_v2] first-dispatch diagnostic failed, ignored",
            exc_info=True,
        )


class DeepEPV2Dispatch(PyLayer):
    """DeepEP V2.5 dispatch: forward dispatch, backward combine."""

    @staticmethod
    def forward(
        ctx,
        x,
        token_indices,
        token_probs,
        num_experts,
        group,
        fp8_dispatch: bool = False,
        moe_ep_barrier: bool = True,
        use_ue8m0: bool = False,
        using_sonic_moe: bool = False,
        kernel_config: KernelConfig | None = None,
    ):
        assert token_probs.dtype == paddle.float32
        if token_indices.dtype != deep_ep_v2.topk_idx_t:
            token_indices = token_indices.cast(deep_ep_v2.topk_idx_t)
        # Keep the token count before quantization: _quantize_for_dispatch returns a
        # (x_fp8, scale) tuple, which has no .shape.
        num_tokens_this_rank = x.shape[0]
        buffer = get_buffer_v2(
            group, x.shape[0], x.shape[1], token_indices.shape[1]
        )
        if moe_ep_barrier:
            barrier_ep(group)
        if fp8_dispatch:
            x = _quantize_for_dispatch(x, use_ue8m0, using_sonic_moe)
        cfg = kernel_config or kernel_config_snapshot()
        do_expand = cfg.do_expand
        expert_alignment = cfg.expert_alignment
        recv_x, recv_indices, recv_probs, handle, _ = buffer.dispatch(
            x,
            topk_idx=token_indices.contiguous(),
            topk_weights=token_probs.contiguous(),
            num_experts=num_experts,
            num_sms=cfg.num_sms,
            **_expand_kwargs(fp8_dispatch, do_expand, expert_alignment),
        )
        _log_first_dispatch(
            handle,
            group,
            num_experts,
            token_indices.shape[1],
            num_tokens_this_rank,
        )
        ctx.buffer = buffer
        ctx.handle = handle
        ctx.group = group
        ctx.moe_ep_barrier = moe_ep_barrier
        ctx.set_grad_in_dtype_consistent(False)
        states = {
            # In the expanding layout dispatch already grouped by expert, so there is no
            # per-token expert id to return (DeepEP returns None) and none is needed.
            "dispatched_indices": recv_indices,
            "tokens_per_expert": handle.num_recv_tokens_per_expert_list,
            "handle": handle,
            # The buffer this handle belongs to (it may be replaced later by a larger one).
            "buffer": buffer,
            "do_expand": do_expand,
            "expert_alignment": expert_alignment,
            # Rows of the expanded output == sum of the alignment-padded per-expert counts.
            "num_permuted_tokens": (
                int(recv_x[0].shape[0] if fp8_dispatch else recv_x.shape[0])
                if do_expand
                else None
            ),
        }
        if fp8_dispatch:
            recv_x, scale = recv_x
            return recv_x, recv_probs, states, {"scale": scale}
        return recv_x, recv_probs, states, None

    @staticmethod
    def backward(ctx, grad_output, grad_token_probs):
        if ctx.moe_ep_barrier:
            barrier_ep(ctx.group)
        # Both grads already match the layout dispatch produced: in the expanding layout
        # `grad_output` is [num_expanded, H] and `grad_token_probs` is 1-D, which is what
        # expand-mode combine expects.
        grad_x, grad_probs, _ = ctx.buffer.combine(
            grad_output.contiguous(),
            ctx.handle,
            topk_weights=(
                None
                if grad_token_probs is None
                else grad_token_probs.cast(paddle.float32).contiguous()
            ),
        )
        return grad_x, None, grad_probs


def _combine_buffer(states):
    buffer = states.get("buffer")
    if buffer is None:
        raise RuntimeError(
            "DeepEP V2 combine requires the states returned by deepep_v2_dispatch."
        )
    return buffer


class DeepEPV2Combine(PyLayer):
    """DeepEP V2.5 combine: forward combine, backward cached dispatch."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        states,
        moe_ep_barrier: bool = True,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ):
        buffer = _combine_buffer(states)
        if moe_ep_barrier:
            barrier_ep(group)
        combined_x, _, _ = buffer.combine(x.contiguous(), states["handle"])
        ctx.buffer = buffer
        ctx.handle = states["handle"]
        ctx.group = group
        ctx.moe_ep_barrier = moe_ep_barrier
        ctx.fp8_dispatch = fp8_dispatch
        ctx.combine_grad_handle = combine_grad_handle
        ctx.do_expand = states["do_expand"]
        ctx.expert_alignment = states["expert_alignment"]
        return combined_x

    @staticmethod
    def backward(ctx, grad_output):
        grad_for_comm = grad_output
        if ctx.fp8_dispatch:
            grad_for_comm = _sonicmoe_quantize(grad_output.contiguous())
        if ctx.moe_ep_barrier:
            barrier_ep(ctx.group)
        grad_x, _ = _cached_dispatch(
            ctx.buffer,
            grad_for_comm,
            ctx.handle,
            fp8=ctx.fp8_dispatch,
            do_expand=ctx.do_expand,
            expert_alignment=ctx.expert_alignment,
        )
        if ctx.fp8_dispatch:
            grad_x = _record_fp8_combine_grad(grad_x, ctx.combine_grad_handle)
        return grad_x


class DeepEPV2CombineAsync(PyLayer):
    """DeepEP V2.5 combine overlapped with the shared experts (``fn``)."""

    @staticmethod
    def forward(
        ctx,
        x,
        group,
        states,
        *fn_args,
        fn,
        is_first_fwd=False,
        fp8_dispatch: bool = False,
        combine_grad_handle: dict | None = None,
    ):
        assert fn is not None, "DeepEPV2CombineAsync requires fn."
        buffer = _combine_buffer(states)
        combined_x, _, event = buffer.combine(
            x.contiguous(),
            states["handle"],
            async_with_compute_stream=True,
        )
        ctx.bwf, fn_out = manual_backward(fn, is_first_fwd, *fn_args)
        event.current_stream_wait()
        ctx.buffer = buffer
        ctx.handle = states["handle"]
        ctx.fp8_dispatch = fp8_dispatch
        ctx.combine_grad_handle = combine_grad_handle
        ctx.do_expand = states["do_expand"]
        ctx.expert_alignment = states["expert_alignment"]
        return (combined_x, *fn_out)

    @staticmethod
    def backward(ctx, grad_output, *fn_out_grads):
        grad_for_comm = grad_output
        if ctx.fp8_dispatch:
            grad_for_comm = _sonicmoe_quantize(grad_output.contiguous())
        grad_x, event = _cached_dispatch(
            ctx.buffer,
            grad_for_comm,
            ctx.handle,
            async_finish=True,
            fp8=ctx.fp8_dispatch,
            do_expand=ctx.do_expand,
            expert_alignment=ctx.expert_alignment,
        )
        fn_args_grads = ctx.bwf(*fn_out_grads)
        event.current_stream_wait()
        if ctx.fp8_dispatch:
            grad_x = _record_fp8_combine_grad(grad_x, ctx.combine_grad_handle)
        return (grad_x, *fn_args_grads)


def deepep_v2_dispatch(
    x,
    token_indices,
    token_probs,
    num_experts,
    group: Group,
    fp8_dispatch: bool = False,
    moe_ep_barrier: bool = True,
    use_ue8m0: bool = False,
    using_sonic_moe: bool = False,
    kernel_config: KernelConfig | None = None,
):
    """Dispatch tokens; returns ``(recv_x, recv_probs, states, scale_handle)`` like V1."""
    recv_x, recv_probs, states, scale = DeepEPV2Dispatch.apply(
        x.contiguous(),
        token_indices,
        token_probs,
        num_experts,
        group,
        fp8_dispatch,
        moe_ep_barrier,
        use_ue8m0,
        using_sonic_moe,
        kernel_config,
    )
    return recv_x, recv_probs, states, scale


def deepep_v2_combine(
    x,
    group: Group,
    states: dict,
    *,
    combine_overlap_handle: dict | None = None,
    moe_ep_barrier: bool = True,
    fp8_dispatch: bool = False,
    combine_grad_handle: dict | None = None,
):
    """Combine expert outputs, optionally overlapped with ``combine_overlap_handle['fn']``."""
    if combine_overlap_handle is None:
        return DeepEPV2Combine.apply(
            x,
            group,
            states,
            moe_ep_barrier,
            fp8_dispatch,
            combine_grad_handle,
        )
    if not isinstance(combine_overlap_handle, dict):
        raise TypeError("combine_overlap_handle must be a dict.")
    if (
        "fn" not in combine_overlap_handle
        or "fn_args" not in combine_overlap_handle
    ):
        raise ValueError(
            "combine_overlap_handle must contain 'fn' and 'fn_args' keys."
        )
    if not isinstance(combine_overlap_handle["fn_args"], tuple):
        raise TypeError("combine_overlap_handle['fn_args'] must be a tuple.")
    combined_x, *fn_out = DeepEPV2CombineAsync.apply(
        x,
        group,
        states,
        *(combine_overlap_handle["fn_args"]),
        fn=combine_overlap_handle["fn"],
        is_first_fwd=not framework._dygraph_tracer()._has_grad,
        fp8_dispatch=fp8_dispatch,
        combine_grad_handle=combine_grad_handle,
    )
    combine_overlap_handle["fn_out"] = fn_out
    return combined_x
