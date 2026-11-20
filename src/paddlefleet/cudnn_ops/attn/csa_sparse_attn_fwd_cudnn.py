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

"""Sparse forward kernels for the "cudnn" CSA sparse-attn backend.

This is the forward half of the "cudnn" backend (the backward half lives in
``csa_sparse_attn_bwd_cudnn.py``). ``sparse_attn_fwd`` runs the cuDNN DSA
sparse prefill where ``cudnn_dsa_supported()`` holds (SM100 with the vendored
cudnn-frontend importable) and the FlashMLA sparse prefill otherwise; both
return the same KV-only ``lse``.
"""

import functools

import paddle
import paddle.nn.functional as F

from paddlefleet.cudnn_ops.frontend_version import (
    cudnn_frontend_is_recent,
    cudnn_frontend_version,
    require_recent_cudnn_frontend,
)
from paddlefleet.fusions.csa_sparse_attn_utils import local_to_global_flat

try:
    from paddlefleet_ops.flash_mla import (
        flash_mla_sparse_fwd as _flash_mla_sparse_fwd,
    )
except (ImportError, RuntimeError):
    _flash_mla_sparse_fwd = None


def _get_topk_alignment() -> int:
    """Minimum ``TopK`` alignment required by the current GPU architecture.

    * SM90 : dual-warpgroup loop steps by 2 blocks → ``2 * B_TOPK = 128``
    * SM100: single-pipeline loop steps by 1 block → ``B_TOPK`` (64 for
      head64, 128 for head128). DSA uses ``D = 512`` which maps to the
      head64 kernel path → 64.
    """
    sm = paddle.cuda.get_device_capability()
    if sm[0] >= 10:
        return 64
    return 128


def flash_mla_sparse_attn(
    q,
    kv,
    attn_sink,
    topk_idxs,
    sm_scale=None,
    indexer_topk: int = 0,
    d_v=None,
    topk_length=None,
    global_kv_idx_remap_fusion: bool = False,
):
    if _flash_mla_sparse_fwd is None:
        raise RuntimeError("flash_mla is not available")

    b, sq, h, d = q.shape
    _, skv, _ = kv.shape
    topk = topk_idxs.shape[-1]
    # Value dim may be smaller than the query/key dim (absorbed MLA MQA uses
    # d_qk=576 / d_v=512). Default to a symmetric d_v=d.
    if d_v is None:
        d_v = d

    q_flat = q.reshape([b * sq, h, d])
    kv_flat = kv.reshape([b * skv, d])
    global_idxs = local_to_global_flat(
        topk_idxs, skv, fused=global_kv_idx_remap_fusion
    )
    # [b, sq] -> [b * sq]: one valid-prefix length per flattened query row.
    topk_length_flat = (
        None
        if topk_length is None
        else topk_length.reshape([b * sq]).cast("int32")
    )

    topk_align = _get_topk_alignment()
    topk_padded = (topk + topk_align - 1) // topk_align * topk_align
    if topk_padded != topk:
        global_idxs = F.pad(global_idxs, (0, topk_padded - topk), value=-1)

    res = _flash_mla_sparse_fwd(
        q_flat,
        kv_flat.unsqueeze(1),
        global_idxs.unsqueeze(1),
        sm_scale,
        d_v=d_v,
        attn_sink=attn_sink,
        topk_length=topk_length_flat,
        indexer_topk=indexer_topk,
    )
    if indexer_topk > 0:
        out_flat, _max_logits, lse, lse_indexer = res
        lse_indexer = lse_indexer.reshape([b, sq, h])
    else:
        out_flat, _max_logits, lse = res
        lse_indexer = None
    return (
        out_flat.reshape([b, sq, h, d_v]),
        lse.reshape([b, sq, h]),
        lse_indexer,
    )


@functools.cache
def cudnn_dsa_supported() -> bool:
    """Whether the cuDNN DSA sparse kernels can run in this process.

    Needs an SM100-family GPU (cudnn-frontend ships DSA kernels for no other
    architecture) and an importable vendored cudnn-frontend, which
    ``paddlefleet_ops`` only reports on Python >= 3.12, of at least
    ``MIN_CUDNN_FRONTEND`` (an ops wheel built with 1.28 has no sparse
    forward; ``sparse_attn_fwd`` then keeps FlashMLA). Shared by
    ``sparse_attn_fwd`` and ``is_dsa_available``.
    """
    try:
        import paddlefleet_ops

        return (
            paddle.device.cuda.get_device_capability()[0] == 10
            and paddlefleet_ops.is_cudnn_frontend_available()
            and cudnn_frontend_is_recent(cudnn_frontend_version())
        )
    except (ImportError, RuntimeError, AttributeError):
        return False


def cudnn_sparse_attn(
    q,
    kv,
    attn_sink,
    topk_idxs,
    sm_scale=None,
    indexer_topk: int = 0,
    d_v=None,
    topk_length=None,
    global_kv_idx_remap_fusion: bool = False,
):
    """cuDNN DSA sparse prefill (SM100), same contract as ``flash_mla_sparse_attn``.

    Supports ``(h, d_qk)`` in ``{(64, 512), (64, 576), (128, 512)}`` with
    ``d_v == 512``; ``indexer_topk`` in ``{0, 512, 1024, 2048}`` (``h == 128``:
    ``{0, 512, 1024}``). The wrapper pads ``topk_idxs`` to a multiple of 64.
    """
    from paddlefleet_ops import CUDNN_FRONTEND_HINT, is_cudnn_frontend_available

    if not is_cudnn_frontend_available():
        raise ImportError(CUDNN_FRONTEND_HINT)
    require_recent_cudnn_frontend(
        cudnn_frontend_version(), "The cuDNN DSA sparse forward"
    )
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.sparse_attention_forward.api import (
        sparse_attention_forward_wrapper,
    )

    b, sq, h, d = q.shape
    _, skv, _ = kv.shape
    if d_v is None:
        d_v = d
    if d_v != 512:
        raise ValueError(
            f"cuDNN sparse attention forward needs d_v == 512, got {d_v}"
        )

    global_idxs = local_to_global_flat(
        topk_idxs, skv, fused=global_kv_idx_remap_fusion
    )
    topk_length_flat = (
        None
        if topk_length is None
        else topk_length.reshape([b * sq]).cast("int32")
    )
    res = sparse_attention_forward_wrapper(
        q.reshape([b * sq, h, d]),
        kv.reshape([b * skv, d]),
        global_idxs,
        attn_sink=attn_sink,
        topk_length=topk_length_flat,
        softmax_scale=sm_scale,
        indexer_topk=int(indexer_topk),
    )
    lse_indexer = res["lse_indexer"]
    if lse_indexer is not None:
        lse_indexer = lse_indexer.reshape([b, sq, h])
    return (
        res["out"].reshape([b, sq, h, d_v]),
        res["lse"].reshape([b, sq, h]),
        lse_indexer,
    )


def sparse_attn_fwd(
    q,
    kv,
    attn_sink,
    topk_idxs,
    sm_scale=None,
    indexer_topk: int = 0,
    d_v=None,
    topk_length=None,
    global_kv_idx_remap_fusion: bool = False,
):
    """cuDNN sparse forward where ``cudnn_dsa_supported()``, FlashMLA otherwise.

    Without the vendored cudnn-frontend (e.g. Python < 3.12) SM100 keeps the
    FlashMLA forward, so a FlashMLA forward + tilelang backward setup still
    runs there. Arguments are forwarded by name, so a signature drift in either
    backend fails here.
    """
    fwd = cudnn_sparse_attn if cudnn_dsa_supported() else flash_mla_sparse_attn
    return fwd(
        q,
        kv,
        attn_sink,
        topk_idxs,
        sm_scale=sm_scale,
        indexer_topk=indexer_topk,
        d_v=d_v,
        topk_length=topk_length,
        global_kv_idx_remap_fusion=global_kv_idx_remap_fusion,
    )
