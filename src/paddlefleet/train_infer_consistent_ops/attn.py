# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

"""Attention index and RoPE layouts for train-inference tensor comparison."""

from __future__ import annotations

import paddle
from paddle import Tensor

from paddlefleet.train_infer_consistent_ops.slice_util import (
    last_dim_segment,
    scatter_last_dim_segment,
)


def _reverse_window_and_topk(
    origin_topk_idxs: Tensor, top_k: Tensor | None, window: Tensor
) -> Tensor:
    """Re-lay the slots as ``[compress | window]``, the order inference uses.

    This function is used for train-inference consistency comparison only; it
    is not used during training.

    Training lays the slots out as ``[window, compress]``; inference
    concatenates them in ``[compress, window]`` order. Training is aligned to
    inference so the two ``indices`` line up slot by slot, which is what
    comparing them needs. This avoids the train-infer mismatch that the
    different accumulation order would cause when computing ``attn_scores``
    -- attention softmaxes over a set, so the reorder only moves the
    accumulation order (the last bits), not the value.
    """
    if top_k is not None:
        return paddle.concat([top_k, window], axis=-1)
    return origin_topk_idxs


def _sort_topk_indices_pad_last(topk_indices: Tensor) -> Tensor:
    """Sort top-k indices ascending with ``-1`` padding kept at the row tail.

    Sort ascending to match the inference side, which emits each row in
    increasing key order; cuDNN emits its own (roughly descending) order, so
    the two dumps disagree on layout even when they select the identical set.
    Attention softmaxes over a set, so the reorder only moves the accumulation
    order (the last bits), not the value. ``-1`` padding is mapped to
    ``INT32_MAX`` for the sort so it stays at the tail of the row, then mapped
    back.
    """
    pad = paddle.full_like(topk_indices, 2**31 - 1)
    sorted_idxs = paddle.sort(
        paddle.where(topk_indices >= 0, topk_indices, pad), axis=-1
    )
    return paddle.where(
        sorted_idxs == 2**31 - 1,
        paddle.full_like(sorted_idxs, -1),
        sorted_idxs,
    )


def _sort_order_pad_last(topk_indices: Tensor) -> Tensor:
    pad = paddle.full_like(topk_indices, 2**31 - 1)
    return paddle.argsort(
        paddle.where(topk_indices >= 0, topk_indices, pad), axis=-1
    )


def _unsort_like(canon: Tensor, live: Tensor) -> Tensor:
    """Inverse of `_sort_topk_indices_pad_last`: put `canon` back in `live`'s order."""
    inv = paddle.argsort(_sort_order_pad_last(live), axis=-1)
    return paddle.take_along_axis(canon, inv, axis=-1)


def _fold_order(token_indices: Tensor, topk_width: int, window_width: int):
    width = topk_width + window_width
    col = paddle.arange(width, dtype=token_indices.dtype)
    is_topk_pad = (col < topk_width) & (token_indices < 0)
    key = paddle.where(is_topk_pad, col + width, col)
    return paddle.argsort(key, axis=-1)


def _unfold_like(
    canon: Tensor, live: Tensor, topk_width: int, window_width: int
) -> Tensor:
    """Inverse of `_fold_window_into_topk_pad`, using `live`'s per-row top-k count."""
    inv = paddle.argsort(_fold_order(live, topk_width, window_width), axis=-1)
    return paddle.take_along_axis(canon, inv, axis=-1)


def _fold_window_into_topk_pad(
    token_indices: Tensor, topk_width: int, window_width: int
) -> Tensor:
    """Pull the window block up into the top-k ``-1`` gap, pad the tail with -1.

    ``token_indices`` is laid out as ``[<topk_width top-k cols>, <window_width
    window cols>]``. The top-k block keeps its valid entries followed by ``-1``
    padding, so the window columns sit past that gap. The inference side emits
    the two runs back to back with the padding trailing, so canonicalize to the
    same layout: keep each row's valid top-k entries, splice the whole window
    block right after them (into the first ``-1`` slot), and push the top-k
    ``-1`` padding to the tail.

    Done as a per-row column reorder: give valid top-k columns keys ``[0,
    topk_width)``, window columns keys ``[topk_width, width)`` and top-k padding
    columns keys ``[width, width + topk_width)``. The three ranges are disjoint,
    so an ascending argsort lands the runs in exactly that order regardless of
    sort stability.
    """
    order = _fold_order(token_indices, topk_width, window_width)
    return paddle.take_along_axis(token_indices, order, axis=-1)


def _canon_token_indices(
    token_indices: Tensor, topk_width: int, window_width: int
) -> Tensor:
    """Inference layout of ``[topk, window]``: sorted top-k, then fold.

    The top-k block arrives in cuDNN order (the topk probe's post_load restores
    it), so it must be sorted before folding.
    """
    sorted_topk = _sort_topk_indices_pad_last(token_indices[..., :topk_width])
    return _fold_window_into_topk_pad(
        paddle.concat([sorted_topk, token_indices[..., topk_width:]], axis=-1),
        topk_width,
        window_width,
    )


def _uncanon_token_indices(
    canon: Tensor, live: Tensor, topk_width: int, window_width: int
) -> Tensor:
    """Undo the fold of `_canon_token_indices`; top-k stays sorted.

    The sorted (inference) order is kept on purpose: the set is the same, and
    the sparse attention then accumulates in the inference order. Restoring
    cuDNN order raised ``mla_indexer_core_out`` mean diff 1.4e-6 -> 3.6e-5.
    """
    sorted_live = paddle.concat(
        [
            _sort_topk_indices_pad_last(live[..., :topk_width]),
            live[..., topk_width:],
        ],
        axis=-1,
    )
    return _unfold_like(canon, sorted_live, topk_width, window_width)


def interleave_rope_segment(t, nope_dim):
    t_rope = last_dim_segment(t, nope_dim)
    return to_interleave_rope_layout_(t_rope)


def interleave_rope_segment_inplace(t, nope_dim):
    t_rope = last_dim_segment(t, nope_dim)
    t_rope_interleave = to_interleave_rope_layout_(t_rope)
    return scatter_last_dim_segment(t, t_rope_interleave, nope_dim)


def deinterleave_rope_segment(t, nope_dim):
    """Inverse of `interleave_rope_segment_inplace`: `[nope | interleaved rope]`
    back to `[nope | half-split rope]`, for folding an sglang dump into the live buffer.
    """
    rope = last_dim_segment(t, nope_dim)
    rope_width = rope.shape[-1]
    if rope_width % 2 != 0:
        raise ValueError(
            "deinterleave_rope_segment needs an even-length rope segment to "
            f"unpair channels (2j, 2j+1); got width {rope_width} for "
            f"[{nope_dim}:]."
        )
    half = paddle.concat([rope[..., 0::2], rope[..., 1::2]], axis=-1)
    return scatter_last_dim_segment(t, half, nope_dim)


def to_interleave_rope_layout_(x, start=0, end=None):
    """Rewrite post-RoPE tensors from half-split to interleaved channel layout.

    The training-side fused kernel reads channel pairs ``(2j, 2j+1)`` but writes
    the two results to ``j`` and ``j + rope_dim // 2`` (a **half-split** layout),
    whereas sglang's RoPE writes each result back to the channel it came from (an
    **interleaved** layout). The two layouts differ by a channel permutation
    only::

        interleaved[2j]     == half[j]                    j = 0 .. rope_dim // 2 - 1
        interleaved[2j + 1] == half[j + rope_dim // 2]

    This helper takes the half-split tensor and rebuilds the interleaved one, so
    the training side can be compared against sglang in sglang's own layout.
    Applying it to q_pe and k_pe together leaves ``q . k`` -- and hence the
    attention output -- unchanged, so this is purely a comparison aid.

    ``start`` / ``end`` restrict the permutation to the ``[..., start:end]``
    segment of the last dim, leaving the head ``[..., :start]`` and tail
    ``[..., end:]`` verbatim. This is the ``[nope | rope]`` case: the rope block
    is half-split while the nope block, which RoPE never touches, is passed
    through. The default spans the whole last dim.

    Returns a fresh tensor via ``concat`` rather than writing into ``x``:
    the live buffer is usually still held by the autograd graph, and an
    in-place write bumps its dygraph inplace version and breaks
    backward / recompute.
    """
    width = x.shape[-1]
    stop = width if end is None else end
    seg = x[..., start:stop]
    seg_width = seg.shape[-1]
    if seg_width % 2 != 0:
        raise ValueError(
            "to_interleave_rope_layout_ needs an even-length segment to pair "
            f"channels (2j, 2j+1); got width {seg_width} for [{start}:{stop}]."
        )
    half = seg_width // 2
    seg = paddle.stack([seg[..., :half], seg[..., half:]], axis=-1).reshape(
        seg.shape
    )
    parts = []
    if start > 0:
        parts.append(x[..., :start])
    parts.append(seg)
    if stop < width:
        parts.append(x[..., stop:])
    return paddle.concat(parts, axis=-1) if len(parts) > 1 else parts[0]
