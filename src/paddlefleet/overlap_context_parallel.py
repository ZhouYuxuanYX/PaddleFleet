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

"""FlashMask context parallel with the KV communication overlapped in-kernel.

Selected by the ``_overlap`` suffix of ``cp_balance_mode`` and dispatched from
``flashmask_attention_cp``. K/V are handed to the FA-4 kernel LOCAL together with
the CP group: the kernel all-gathers them on an internal stream and folds the
dK/dV reduce-scatter back to the local length, so this layer performs no explicit
communication and only has to express the mask with its key axis in group-rank
order; FlashMask maps that onto its own communication order.

Kept in a separate module so that the overlap path adds no import cost, and no
merge surface, to the non-overlap one.
"""

import paddle
from paddle.autograd.py_layer import PyLayer
from paddle.distributed import fleet

from paddlefleet.context_parallel_utils import (
    preprocess_index,
    preprocess_index_dual_chunks,
)

OVERLAP_SUPPORTED = False
try:
    if (
        paddle.device.cuda.device_count() > 0
        and paddle.device.cuda.get_device_capability(0)[0] == 10
    ):
        from paddlefleet_ops.flashmask.cute.flashmask_utils import (
            FlashMaskInfoPaddle,
        )
        from paddlefleet_ops.flashmask.cute.interface import (
            _flash_attn_bwd,
            _flash_attn_fwd,
        )
        from paddlefleet_ops.flashmask.overlap import overlap_runtime

        # Only a FlashMask that reorders the group-rank mask itself may be fed
        # one; older builds expect it pre-ordered and would mis-mask silently.
        OVERLAP_SUPPORTED = hasattr(overlap_runtime, "sr_order_mask")
except (ImportError, AttributeError):
    OVERLAP_SUPPORTED = False


BLOCK_ORDER_CACHE = {}


def block_order(cp_size):
    """DualChunkSwap natural blocks in group-rank order, as a cached index tensor.

    Rank r holds the mirrored pair ``(r, 2*cp_size-1-r)``. Cached: rebuilding the
    index would cost a blocking host-to-device copy on every call.
    """
    index = BLOCK_ORDER_CACHE.get(cp_size)
    if index is None:
        order = [b for r in range(cp_size) for b in (r, 2 * cp_size - 1 - r)]
        index = paddle.to_tensor(order, dtype="int64")
        BLOCK_ORDER_CACHE[cp_size] = index
    return index


def localize_mask(startend_row_indices, seqlen_local, group, mode):
    """Fold the global FlashMask row bounds onto the local query rows.

    Only the values change; axis 2 keeps indexing key positions in natural
    sequence order, which ``gathered_kv_order`` then permutes.
    """
    if mode == "contiguous_allgather_overlap":
        # The local query rows are one contiguous run, so a single rebase does it.
        return preprocess_index(
            startend_row_indices,
            chunk_id=group.rank,
            seq_blocksize=seqlen_local,
            max_seqlen_q=seqlen_local,
        )
    if mode == "dualchunk_allgather_overlap":
        # Two non-adjacent halves, each rebased and the second one offset.
        seq_blocksize = seqlen_local // 2
        return preprocess_index_dual_chunks(
            startend_row_indices,
            chunk_id_first=group.rank,
            chunk_id_second=2 * group.world_size - group.rank - 1,
            seq_blocksize=seq_blocksize,
            max_seqlen_q=seq_blocksize,
        )
    raise ValueError(
        f"Unsupported overlapped FlashMask context parallel mode: {mode}"
    )


def gathered_kv_order(mask, group, mode):
    """Permute the mask key axis into the ranks' local KV shards in group-rank order."""
    if mode == "contiguous_allgather_overlap":
        return mask  # rank r holds keys [r*S_local, (r+1)*S_local): already in order
    cp_size = group.world_size
    batch_size, _, seqlen, num_vecs = mask.shape
    n_blocks = 2 * cp_size
    return (
        mask.reshape([batch_size, -1, n_blocks, seqlen // n_blocks, num_vecs])
        .index_select(block_order(cp_size), axis=2)
        .reshape([batch_size, -1, seqlen, num_vecs])
    )


class OverlappedFlashMaskContextParallel(PyLayer):
    """FlashMask CP attention over an in-kernel overlapped KV all-gather.

    Backward returns map positionally onto the forward TENSOR inputs:
    ``query(0)/key(1)/value(2)/startend_row_indices(3)/learnable_sink(4)``. The
    mask is stop_gradient, so slot 3 must be ``None`` whenever slot 4 is filled;
    when the sink needs no gradient the trailing slots are simply omitted.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        startend_row_indices,
        learnable_sink,
        softmax_scale,
        mode,
    ):
        hcg = fleet.get_hybrid_communicate_group()
        group = hcg.get_context_parallel_group()
        mask = gathered_kv_order(
            localize_mask(startend_row_indices, query.shape[1], group, mode),
            group,
            mode,
        )
        # The in-kernel all-gather copies the raw [B, S_local, H, D] storage.
        key, value = key.contiguous(), value.contiguous()

        output, log_sum_exp = _flash_attn_fwd(
            query,
            key,
            value,
            startend_row_indices=mask,
            learnable_sink=learnable_sink,
            causal=False,
            return_lse=True,
            pack_gqa=False,
            softmax_scale=softmax_scale,
            group=group,
        )

        ctx.save_for_backward(query, key, value, output, log_sum_exp, mask)
        ctx.group = group
        ctx.softmax_scale = softmax_scale
        ctx.learnable_sink = learnable_sink
        ctx.sink_requires_grad = (
            learnable_sink is not None and not learnable_sink.stop_gradient
        )
        return output

    @staticmethod
    def backward(ctx, output_grad):
        query, key, value, output, log_sum_exp, mask = ctx.saved_tensor()
        group = ctx.group

        flashmask_info = FlashMaskInfoPaddle(
            startend_row_indices=mask,
            is_causal=False,
        )
        # dK/dV are reduce-scattered inside the kernel, hence already local.
        # dsink is a parameter gradient over this rank's queries, so it stays a
        # partial that the optimizer's own reduction sums, as in the non-overlap
        # all-gather path.
        query_grad, key_grad, value_grad, sink_grad = _flash_attn_bwd(
            query,
            key,
            value,
            output,
            output_grad,
            log_sum_exp,
            flashmask_info,
            learnable_sink=ctx.learnable_sink,
            causal=False,
            softmax_scale=ctx.softmax_scale,
            deterministic=paddle.get_flags(["FLAGS_cudnn_deterministic"])[
                "FLAGS_cudnn_deterministic"
            ],
            group=group,
        )
        if ctx.sink_requires_grad:
            return query_grad, key_grad, value_grad, None, sink_grad
        return query_grad, key_grad, value_grad


def overlap_flashmask_attention_cp(
    query,
    key,
    value,
    startend_row_indices,
    fixed_seed_offset=None,
    dropout=0.0,
    causal=False,
    training=True,
    learnable_sink=None,
    softmax_scale=None,
    mode="dualchunk_allgather_overlap",
):
    """Overlapped counterpart of ``flashmask_attention_cp``, same signature.

    Everything the overlapped kernel does not implement is rejected here, so the
    layer itself can assume a plain non-causal FlashMask.
    """
    if not OVERLAP_SUPPORTED:
        raise RuntimeError(
            "Overlapped FlashMask context parallel requires an SM100 device and a "
            "paddlefleet_ops build that includes the FlashMask overlap bridge "
            "(paddlefleet_ops.flashmask.overlap)."
        )
    if dropout > 0.0:
        raise NotImplementedError(
            "Dropout is not supported in overlapped FlashMask context parallel."
        )
    if causal:
        raise NotImplementedError(
            "Overlapped FlashMask context parallel does not support causal."
        )
    if fixed_seed_offset is not None:
        raise NotImplementedError(
            "Fixed seed offset is not supported in overlapped FlashMask "
            "context parallel."
        )
    assert query.shape[1] % 2 == 0, (
        f"Query sequence length must be divisible by 2: the mask is permuted in "
        f"2 * cp_size blocks. Current query sequence length: {query.shape[1]}"
    )
    return OverlappedFlashMaskContextParallel.apply(
        query,
        key,
        value,
        startend_row_indices,
        learnable_sink,
        softmax_scale,
        mode,
    )
