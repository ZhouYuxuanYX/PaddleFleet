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

"""Overlapped FlashMask CP (``*_allgather_overlap``) equals the all-gather path.

The same per-rank Q/K/V and global mask go through ``flashmask_attention_cp``
with and without the ``_overlap`` suffix; out and dQ/dK/dV/dsink must agree to
bf16 noise. Prefix-LM documents straddle every rank boundary, so a wrong row
rebase or key order is off by O(1), not by noise. Two local lengths per mode
(one not a power of two) run in one process, as the per-call length change does
in training.

    python -m paddle.distributed.launch --gpus=0,1,2,3 \
        tests/multi_card_tests/transformer/test_flash_mask_cp_overlap.py
"""

import os
import sys
import unittest

_repo_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..")
)
sys.path.insert(0, os.path.join(_repo_root, "src"))

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.context_parallel_utils import (
    flashmask_attention_cp,
    scatter_balance,
    scatter_contiguous,
)
from paddlefleet.overlap_context_parallel import OVERLAP_SUPPORTED

# Overlap vs all-gather differ only by the dK/dV reduction (bf16 transport vs
# fp32 reduce-scatter) and accumulation order: measured <= 2.7e-3 (out, dQ/dK/dV)
# and 5.6e-3 (dsink) over 4 ranks. A mask off by one rank measures >= 0.9.
RTOL = {"out": 1e-2, "dq": 2e-2, "dk": 2e-2, "dv": 2e-2, "dsink": 2e-2}
HQ, HK, D, BATCH = 8, 2, 128, 2

GROUP = None
_SAVED_FA_VERSION = None


def setUpModule():
    if not OVERLAP_SUPPORTED:
        raise unittest.SkipTest(
            "needs SM100 and paddlefleet_ops built from FlashMask"
        )
    # The all-gather reference must run the same FA4 kernels; the flag is
    # process-global, so pin it here and hand it back in tearDownModule.
    global GROUP, _SAVED_FA_VERSION
    _SAVED_FA_VERSION = paddle.get_flags(["FLAGS_flash_attn_version"])
    paddle.set_flags({"FLAGS_flash_attn_version": 4})
    world = dist.get_world_size()
    strategy = fleet.DistributedStrategy()
    strategy.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": world,
        "sep_degree": 1,
        "cp_degree": world,
        "ep_degree": world,
        "moe_sharding_degree": 1,
        "order": [
            "sharding",
            "moe_sharding",
            "pp",
            "sep",
            "cp",
            "dp",
            "ep",
            "mp",
        ],
    }
    fleet.init(is_collective=True, strategy=strategy)
    GROUP = fleet.get_hybrid_communicate_group().get_context_parallel_group()


def tearDownModule():
    if _SAVED_FA_VERSION is not None:
        paddle.set_flags(_SAVED_FA_VERSION)


def _prefix_lm_mask(s_total, rng):
    """[lt_start, ut_end] per key: each document is bidirectional over a random
    prefix and causal after it."""
    cols = np.empty([BATCH, 1, s_total, 2], dtype=np.int32)
    for b in range(BATCH):
        start = 0
        while start < s_total:
            end = min(
                s_total, start + int(rng.integers(s_total // 7, s_total // 3))
            )
            prefix = int(rng.integers(start, end + 1))
            keys = np.arange(start, end)
            cols[b, 0, start:end, 0] = end
            cols[b, 0, start:end, 1] = np.where(keys < prefix, start, keys)
            start = end
    return paddle.to_tensor(cols)


def _rel(a, b):
    a, b = a.astype("float32"), b.astype("float32")
    return float((a - b).norm() / (b.norm() + 1e-30))


class TestFlashMaskCPOverlap(unittest.TestCase):
    def _run(self, mode, s_local, sink_trainable):
        cp = GROUP.nranks
        rng = np.random.default_rng(s_local)
        paddle.seed(s_local)
        full = [
            paddle.randn([BATCH, s_local * cp, h, D]).astype("bfloat16")
            for h in (HQ, HK, HK)
        ]
        scatter = (
            scatter_balance
            if mode == "dualchunk_allgather"
            else scatter_contiguous
        )
        local = [scatter(t, group=GROUP, axis=1) for t in full]
        mask = _prefix_lm_mask(s_local * cp, rng)
        dout = paddle.randn([BATCH, s_local, HQ, D]).astype("bfloat16")
        sink = (paddle.randn([HQ]) + 2.0).astype("bfloat16")

        results = {}
        for m in (mode, mode + "_overlap"):
            q, k, v, s = (t.detach().clone() for t in (*local, sink))
            for t in (q, k, v):
                t.stop_gradient = False
            s.stop_gradient = not sink_trainable
            out = flashmask_attention_cp(
                q, k, v, mask, learnable_sink=s, mode=m
            )
            (out.astype("float32") * dout.astype("float32")).sum().backward()
            results[m] = {"out": out, "dq": q.grad, "dk": k.grad, "dv": v.grad}
            if sink_trainable:
                results[m]["dsink"] = s.grad

        ref, got = results[mode], results[mode + "_overlap"]
        for name, tol in RTOL.items():
            if name not in ref:
                continue
            self.assertEqual(got[name].shape, ref[name].shape, name)
            err = _rel(got[name], ref[name])
            self.assertLess(
                err, tol, f"{mode} s_local={s_local} {name}: rel {err:.3e}"
            )

    def test_dualchunk(self):
        for s_local in (1024, 1504):
            self._run("dualchunk_allgather", s_local, sink_trainable=True)

    def test_contiguous(self):
        for s_local in (1504, 1024):
            self._run("contiguous_allgather", s_local, sink_trainable=False)


if __name__ == "__main__":
    unittest.main()
