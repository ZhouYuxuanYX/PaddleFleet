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

import inspect

import paddle

from . import is_flash_mask_available


def _cutedsl_hdim_ok(head_dim: int, head_dim_v: int) -> bool:
    """Whether the cutedsl kernels serve ``(head_dim, head_dim_v)``.

    FA3 (SM90) and FA4 (SM100) serve the same pairs, so ``fa_version`` is not a
    parameter. Neither is ``deterministic``: no pair in this whitelist is
    deterministic-only. FA3 on cutedsl has an ordered-accumulation backward, and
    FA4's big-head-dim backward gained an ordered (semaphore-serialised)
    reduction in flash-attention ``5007a05``.

    Args:
        head_dim: Query/Key head dim.
        head_dim_v: Value head dim, already defaulted by the caller.

    Returns:
        ``True`` when the pair is served; ``False`` means degrade to FA2.
    """
    return (
        (head_dim <= 128 and head_dim_v <= 128)
        or (head_dim == 192 and head_dim_v == 128)
        or (head_dim == 256 and head_dim_v == 256)
        or (head_dim == 512 and head_dim_v == 512)
        or (head_dim == 576 and head_dim_v == 512)
    )


def _dispatch_fa_version(
    head_dim: int,
    head_dim_v: int | None = None,
    startend_row_indices: paddle.Tensor | None = None,
) -> int:
    """Pick the FlashAttention version for the given head dims.

    Prefer the public ``get_fa_version``, which validates what this returns.

    Dispatch rules:
      * XPU device -> FA2.
      * Otherwise, respect ``FLAGS_flash_attn_version`` by default.
      * FA3 and FA4 both run on the cutedsl kernels. Without them
        (``is_flash_mask_available()`` is False) both degrade to FA2, and so does
        every head-dim pair outside :func:`_cutedsl_hdim_ok`. That whitelist does
        not narrow under ``FLAGS_cudnn_deterministic`` -- see the note there.

    Args:
        head_dim: Query/Key head dim (always equal).
        head_dim_v: Value head dim. Defaults to ``head_dim`` when not provided.
        startend_row_indices: Accepted and ignored, so the flashmask path can
            hand over what it already has. FA4 used to refuse a 4-column
            (global sliding window) mask and this function degraded to FA2 for
            it; since flash-attention ``5007a05`` FA4 serves ``num_vec == 4``,
            so the mask takes no part in dispatch.

    Returns:
        The FlashAttention version to use (2, 3 or 4).
    """
    if "xpu" in paddle.get_device():
        return 2

    fa_version = paddle.base.framework.get_flags(["FLAGS_flash_attn_version"])[
        "FLAGS_flash_attn_version"
    ]

    if fa_version in (3, 4):
        if not is_flash_mask_available():
            return 2
        _head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        if not _cutedsl_hdim_ok(head_dim, _head_dim_v):
            return 2

    return fa_version


def get_fa_version(
    head_dim: int,
    head_dim_v: int | None = None,
    startend_row_indices: paddle.Tensor | None = None,
) -> int:
    """Pick the FlashAttention version, rejecting versions we cannot dispatch.

    Thin validating wrapper over ``_dispatch_fa_version``; see that function for
    the dispatch rules and argument meanings.

    Returns:
        The FlashAttention version to use (2, 3 or 4).

    Raises:
        ValueError: If the resolved version is not 2, 3 or 4 -- in practice this
            means ``FLAGS_flash_attn_version`` was set to something unsupported,
            since every degrade path returns 2.
    """
    fa_version = _dispatch_fa_version(
        head_dim, head_dim_v, startend_row_indices
    )
    if fa_version not in (2, 3, 4):
        raise ValueError(f"Invalid flash attention version: {fa_version}")
    return fa_version


def uses_cutedsl_backend(fa_version: int) -> bool:
    """Whether ``fa_version`` runs on the cutedsl backend.

    FA3 and FA4 are cutedsl-only; FA2 runs on Paddle's built-in kernel.

    Args:
        fa_version: The version returned by :func:`get_fa_version`.

    Returns:
        ``True`` when the call site must use the cutedsl kernels.
    """
    return fa_version in (3, 4)


def needs_value_padding(
    fa_version: int,
    head_dim: int,
    head_dim_v: int,
) -> bool:
    """Whether ``value`` must be zero-padded from ``head_dim_v`` to ``head_dim``.

    Args:
        fa_version: The version returned by :func:`get_fa_version`.
        head_dim: Query/Key head dim.
        head_dim_v: Value head dim.

    Returns:
        ``True`` when ``value`` needs padding.
    """
    native_pair = (fa_version in (3, 4)) and (
        (head_dim == 192 and head_dim_v == 128)
        or (head_dim == 576 and head_dim_v == 512)
    )
    return head_dim != head_dim_v and not native_pair


def flashmask_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    startend_row_indices: paddle.Tensor | None = None,
    *,
    dropout: float = 0.0,
    causal: bool = False,
    window_size: int | tuple | None = None,
    return_softmax_lse: bool = False,
    return_seed_offset: bool = False,
    fixed_seed_offset: paddle.Tensor | None = None,
    rng_name: str = "",
    training: bool = True,
    name: str | None = None,
    softmax_scale: float | None = None,
    block_mask: paddle.Tensor | None = None,
    learnable_sink: paddle.Tensor | None = None,
):
    bsz, q_len, num_heads, q_head_dim = query.shape
    v_head_dim = value.shape[-1]
    fa_version = get_fa_version(q_head_dim, v_head_dim, startend_row_indices)

    if uses_cutedsl_backend(fa_version):
        # The paddle frontend, not the top-level ``flashmask.flashmask_attention``:
        # the latter dispatches on tensor type with a ``(*args, **kwargs)``
        # signature, which would hide ``learnable_sink`` from the probe below.
        from .flashmask.frontend.paddle import (
            flashmask_attention as _flashmask_attention,
        )
    else:
        from paddle.nn.functional.flash_attention import (
            flashmask_attention as _flashmask_attention,
        )

    if learnable_sink is not None:
        if (
            "learnable_sink"
            not in inspect.signature(_flashmask_attention).parameters
        ):
            raise NotImplementedError(
                "learnable_sink (softmax sink) requires FA4 (cute backend); the "
                "installed flash_mask / current device (e.g. H-card fa2/fa3) does "
                "not support it. Disable the attention sink or run on a "
                "FA4-capable device."
            )

    need_value_padding = needs_value_padding(fa_version, q_head_dim, v_head_dim)

    if need_value_padding:
        value_padding = paddle.zeros(
            [*value.shape[:-1], q_head_dim - v_head_dim],
            dtype=value.dtype,
        )
        value = paddle.concat([value, value_padding], axis=-1)

    extra_kwargs = {}
    if learnable_sink is not None:
        extra_kwargs["learnable_sink"] = learnable_sink

    flashmask_attention_func = _flashmask_attention

    outs = flashmask_attention_func(
        query=query,
        key=key,
        value=value,
        startend_row_indices=(
            startend_row_indices.clone()
            if startend_row_indices is not None
            else None
        ),
        dropout=dropout,
        causal=causal,
        window_size=window_size,
        return_softmax_lse=return_softmax_lse,
        return_seed_offset=return_seed_offset,
        fixed_seed_offset=fixed_seed_offset,
        rng_name=rng_name,
        training=training,
        name=name,
        softmax_scale=softmax_scale,
        block_mask=block_mask,
        **extra_kwargs,
    )

    if return_softmax_lse:
        attn_out, lse = outs
        lse = lse.reshape([bsz, q_len])
    else:
        attn_out = outs

    if need_value_padding:
        attn_out = attn_out[..., :v_head_dim]

    attn_out = attn_out.reshape([bsz, q_len, num_heads, v_head_dim])

    if return_softmax_lse:
        return [attn_out, lse]
    else:
        return attn_out


def flash_attention(
    query: paddle.Tensor,
    key: paddle.Tensor,
    value: paddle.Tensor,
    dropout=0.0,
    causal=False,
    return_softmax=False,
    *,
    fixed_seed_offset=None,
    rng_name="",
    training=True,
    name=None,
    softmax_scale=None,
):
    bsz, q_len, num_heads, q_head_dim = query.shape
    v_head_dim = value.shape[-1]

    # startend_row_indices is None
    fa_version = get_fa_version(q_head_dim, v_head_dim)

    need_value_padding = needs_value_padding(fa_version, q_head_dim, v_head_dim)

    if need_value_padding:
        value_padding = paddle.zeros(
            [*value.shape[:-1], q_head_dim - v_head_dim],
            dtype=value.dtype,
        )
        value = paddle.concat([value, value_padding], axis=-1)

    if uses_cutedsl_backend(fa_version):
        # flashmask has no mask-free entry point; without startend_row_indices
        # its flashmask_attention is plain (causal) attention.
        from .flashmask.frontend.paddle import (
            flashmask_attention as _flashmask_attention,
        )

        if training and dropout != 0.0:
            raise NotImplementedError(
                f"flashmask does not support dropout, got dropout={dropout}"
            )
        if return_softmax:
            raise NotImplementedError(
                "flashmask does not return the softmax; pass return_softmax=False"
            )
        attn_output = _flashmask_attention(
            query,
            key,
            value,
            None,
            causal=causal,
            softmax_scale=softmax_scale,
        )
        softmax_result = None
    else:
        from paddle.nn.functional.flash_attention import (
            flash_attention as _flash_attention,
        )

        attn_output, softmax_result = _flash_attention(
            query=query,
            key=key,
            value=value,
            dropout=dropout,
            causal=causal,
            return_softmax=return_softmax,
            fixed_seed_offset=fixed_seed_offset,
            rng_name=rng_name,
            training=training,
            name=name,
            softmax_scale=softmax_scale,
        )

    if need_value_padding:
        attn_output = attn_output[..., :v_head_dim]

    attn_output = attn_output.reshape([bsz, q_len, num_heads, v_head_dim])

    return attn_output, softmax_result


__all__ = [
    "flashmask_attention",
    "flash_attention",
    "get_fa_version",
    "needs_value_padding",
    "uses_cutedsl_backend",
]
