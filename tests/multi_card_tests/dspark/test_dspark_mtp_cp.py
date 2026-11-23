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

"""MTP + DSpark trained together under context parallel (global-tensor dataflow).

Run (2 GPUs) from the repository root::

    PYTHONPATH=.:./src python -m paddle.distributed.launch --gpus=0,1 \
        tests/multi_card_tests/dspark/test_dspark_mtp_cp.py

The repository root on ``PYTHONPATH`` is what makes the shared
``tests.single_card_tests.dspark.test_dspark_e2e_model`` config factory
importable; ``ci/multi-card_test.sh`` exports it for the same reason.
"""

import unittest

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import NoPipelineParallel

from paddlefleet.gpt_builders import gpt_builder
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer import parallel_multi_token_prediction as pm
from paddlefleet.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
)
from tests.single_card_tests.dspark.test_dspark_e2e_model import (
    _V,
    _make_config,
)

SEED = 20261029
SEQ, K, BATCH = 64, 1, 1
CP_SIZE = None
STRATEGY = None


def setUpModule():
    global CP_SIZE, STRATEGY
    CP_SIZE = dist.get_world_size()
    STRATEGY = fleet.DistributedStrategy()
    STRATEGY.hybrid_configs = {
        "dp_degree": 1,
        "mp_degree": 1,
        "pp_degree": 1,
        "sharding_degree": CP_SIZE,
        "sep_degree": 1,
        "cp_degree": CP_SIZE,
        "ep_degree": CP_SIZE,
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
    initialize_fleet(STRATEGY)
    paddle.seed(SEED)
    model_parallel_cuda_manual_seed(SEED)


class TestDsparkMtpContextParallel(unittest.TestCase):
    def _run(self, cp_mode, separate_mtp_input):
        paddle.seed(SEED)
        config = _make_config(
            True,
            num_nextn_predict_layers=K,
            csa_compress_ratios=[0] * (3 + K),
            mtp_loss_scaling_factor=0.3,
            add_mtp_loss=True,
            separate_mtp_input=separate_mtp_input,
            context_parallel_size=CP_SIZE,
            cp_balance_mode=cp_mode,
            experimental_dataflow=True,
            moe_token_dispatcher_type="alltoall",
            expert_model_parallel_size=CP_SIZE,
        )
        model = gpt_builder(config, num_stages=1)

        data = paddle.randint(1, _V, shape=(BATCH, SEQ + K + 1)).cuda()
        dist.broadcast(data, src=0)
        input_ids = data[:, :-1].contiguous()
        labels = data[:, 1:].contiguous()
        startend = paddle.full([BATCH, 1, SEQ, 1], SEQ, dtype="int32").cuda()
        mtp_mask = paddle.ones([BATCH, K, SEQ], dtype="int32").cuda()
        mtp_mask[:, :, -2:] = 0
        mtp_startend = paddle.full(
            [BATCH, K, SEQ, 1], SEQ, dtype="int32"
        ).cuda()

        pipe = NoPipelineParallel(model, STRATEGY)
        inputs = (
            {
                "input_ids": [input_ids],
                "labels": [labels.clone()],
                "attn_mask_startend_row_indices": [startend],
                "mtp_startend_row_indices_all": [mtp_startend],
                "mtp_hidden_inputs_mask_all": [mtp_mask],
            },
            labels.clone(),
        )
        loss = pipe.forward_backward_pipeline(inputs)
        self.assertTrue(np.isfinite(float(loss)), f"loss {float(loss)}")
        for layer_type in (MultiTokenPredictionLayer, pm.DsparkPredictionLayer):
            layers = [m for m in model.sublayers() if type(m) is layer_type]
            self.assertEqual(len(layers), 1)
            missing = [
                n
                for n, p in layers[0].named_parameters()
                if p.grad is None and ".experts." not in n
            ]
            self.assertEqual(missing, [], layer_type.__name__)
        if dist.get_rank() == 0:
            print(
                f"[cp={CP_SIZE} {cp_mode} sep={separate_mtp_input}] loss={float(loss)}"
            )

    def test_contiguous_allgather(self):
        self._run("contiguous_allgather", True)

    def test_contiguous_allgather_default_mtp_layout(self):
        self._run("contiguous_allgather", False)


if __name__ == "__main__":
    unittest.main()
