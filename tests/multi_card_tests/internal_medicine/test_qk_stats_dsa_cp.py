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

"""qk_stats on MQA-DSA (top-k) layers under contiguous CP, against a CP=1 pass.

A real ``MQALatentAttention`` in its sparse phase runs twice on the same global
batch: CP=1 on the whole sequence, then CP>1 on each rank's query slice with the
all-gathered latent KV. Both carry the monitor's ``_sparse_attn`` patch, which
computes the softmax over the very ``token_indices`` (indexer top-k + window)
the sparse kernel is handed, and the CP pass averages per head over the real
CP group. The per-head entropy and sink each rank records must equal the CP=1
values. The index sets themselves are checked by
``tests/multi_card_tests/transformer/test_mqa_dsa_cp.py``.

Run (2 GPUs), from the repository root::

    PYTHONPATH=.:./src python -m paddle.distributed.launch --devices 0,1 \\
        tests/multi_card_tests/internal_medicine/test_qk_stats_dsa_cp.py
"""

import types
import unittest
from unittest import mock

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.internal_medicine.backends.paddlefleet import (
    qk_monitor as qk_monitor_module,
)

# The multi-card runner puts the repository root on ``PYTHONPATH``.
from tests.single_card_tests.transformer import hybrid_mla_utils as U

CP_SIZE = CP_RANK = CP_GROUP = None
S_GLOBAL = 512
ATOL = 1e-4
_CUTEDSL_PIN = U._cutedsl_pin()


def setUpModule():
    global CP_SIZE, CP_RANK, CP_GROUP
    if dist.get_world_size() < 2:
        raise unittest.SkipTest("needs >= 2 GPUs")
    _CUTEDSL_PIN.__enter__()
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
    CP_GROUP = fleet.get_hybrid_communicate_group().get_context_parallel_group()
    CP_RANK = CP_GROUP.rank
    CP_SIZE = CP_GROUP.nranks


def tearDownModule():
    _CUTEDSL_PIN.__exit__(None, None, None)


def _build(cp_group, sink):
    cfg = U._create_mqa_config(mode="mqa_dsa")
    cfg.dsa_indexer_use_sparse_loss = True
    cfg.cp_balance_mode = "contiguous_allgather"
    cfg.experimental_dataflow = True
    cfg.pad_token_id = 0
    cfg.context_parallel_size = 1 if cp_group is None else cp_group.nranks
    paddle.seed(7)
    module = U._build_module(
        cfg,
        bf16=True,
        sink=sink,
        pg_collection=types.SimpleNamespace(tp=None, cp=cp_group),
    )
    module.train()
    return module


def _inputs():
    rng = np.random.RandomState(1234)

    def t(shape, scale):
        return paddle.to_tensor(
            (rng.standard_normal(shape) * scale).astype("float32")
        ).cast("bfloat16")

    return {
        "query": t([1, S_GLOBAL, U.H, U.DK], 0.5),
        "key": t([1, S_GLOBAL, 1, U.DK], 0.5),
        "w_v": t([U.DV, U.H, U.V_HEAD_DIM], 0.05),
        "x": t([1, S_GLOBAL, U.HIDDEN], 0.5),
        "qr": t([1, S_GLOBAL, U.Q_LORA], 0.5),
    }


def _recorded_stats(module, inputs, row_end):
    """Run ``module`` with the qk_stats DSA patch; return what it records."""
    monitor = qk_monitor_module.PaddleQKStatsMonitor()
    item = types.SimpleNamespace(layer=None, attn_type="mqa_dsa")
    seen = []
    with mock.patch.object(
        qk_monitor_module, "get_dsa_core", return_value=module
    ):
        patch = monitor._patch_dsa_attn(0, item)
    assert patch is not None
    with mock.patch.object(
        monitor,
        "_record_common_stats",
        side_effect=lambda _idx, _type, stats: seen.append(stats),
    ):
        module(
            inputs["query"],
            inputs["key"],
            None,
            None,
            attn_mask_startend_row_indices=row_end.clone(),
            x=inputs["x"],
            qr=inputs["qr"],
            v_b_proj_weight=inputs["w_v"],
        )
    patch.remove()
    assert len(seen) == 1, f"expected one sparse call, got {len(seen)}"
    return seen[0]


class DSAQKStatsCPTest(unittest.TestCase):
    def _check(self, doc_lens, sink=None):
        row_end = U._row_end(doc_lens, S_GLOBAL)
        inp = _inputs()
        ref_module = _build(None, sink)
        cp_module = _build(CP_GROUP, sink)
        cp_module.set_state_dict(ref_module.state_dict())

        ref = _recorded_stats(ref_module, inp, row_end)
        sl = S_GLOBAL // CP_SIZE
        off = CP_RANK * sl
        local = {
            k: (v if k == "w_v" else v[:, off : off + sl])
            for k, v in inp.items()
        }
        got = _recorded_stats(cp_module, local, row_end)
        for name in ("entropy_per_head", "sink_per_head"):
            want = ref[name].astype("float32")
            have = got[name].astype("float32")
            self.assertEqual(have.shape, want.shape, name)
            diff = float((have - want).abs().max())
            self.assertLess(diff, ATOL, f"{doc_lens} {name}: max |diff| {diff}")

    @U._GPU
    def test_single_document(self):
        self._check([S_GLOBAL])

    @U._GPU
    def test_documents_straddling_ranks(self):
        self._check([200, 150, 162])

    @U._GPU
    def test_learned_sink(self):
        self._check([200, 150, 162], sink=np.linspace(-1.0, 1.0, U.H))


if __name__ == "__main__":
    unittest.main()
