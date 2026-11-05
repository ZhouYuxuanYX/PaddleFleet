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

"""HyperBody encoder-only EP groups on 8 GPUs (decoder EP 8, encoder EP 4).

* the encoder EP / expert-data groups have the expected members and are cached;
* encoder experts are colored with the encoder expert-data group, replicas of
  one expert start from identical values and different experts do not;
* a decoder MoE layer (global EP) and an encoder MoE layer (encoder EP) run
  forward / backward through DeepEP back to back, each on its own buffer.
"""

import random
import unittest

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
from paddle.distributed import fleet

from paddlefleet.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe import fused_a2a
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig
from paddlefleet.transformers.hyperbody import encoder_ep

WORLD, DEC_EP, ENC_EP = 8, 8, 4
E, H, I, K = 16, 256, 128, 2


def _moe_layer(ep, pg_collection):
    cfg = TransformerConfig(
        hidden_size=H,
        num_attention_heads=4,
        n_routed_experts=E,
        use_cpu_initialization=False,
        num_experts_per_tok=K,
        tensor_model_parallel_size=1,
        expert_model_parallel_size=ep,
        sequence_parallel=False,
        bf16=True,
        params_dtype=paddle.bfloat16,
        moe_intermediate_size=I,
        gated_linear_unit=True,
        n_shared_experts=0,
        hidden_act=F.silu,
        moe_expert_fusion=True,
        moe_token_dispatcher_type="deepep",
        bias_activation_fusion=True,
        moe_deep_gemm=False,
    )
    cfg.deepep_buffer_configs = {
        "num_sms": 20,
        "dispatch_config": [6, 256],
        "combine_config": [6, 256],
    }
    spec = get_gpt_layer_local_spec(cfg, num_experts=E)
    return MoELayer(
        cfg, spec.sublayers_spec.mlp.extra_kwargs["sublayers"], pg_collection
    )


class TestHyperBodyEncoderEP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed = 123
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": WORLD,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": DEC_EP,
            "moe_sharding_degree": WORLD // DEC_EP,
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
        initialize_fleet(strategy=strategy)
        model_parallel_cuda_manual_seed(seed)
        cls.rank = dist.get_rank()
        # The layers here use no Muon/Mixed sharding optimizer, so a Paddle
        # without PaddlePaddle/Paddle#79913 is fine.
        cls.enc_pgs = encoder_ep.get_encoder_pg_collection(
            ENC_EP, allow_color_buckets=True
        )

    def test_groups(self):
        r = self.rank
        pgs = self.enc_pgs
        first = r // ENC_EP * ENC_EP
        self.assertEqual(list(pgs.ep.ranks), list(range(first, first + ENC_EP)))
        self.assertEqual(
            list(pgs.expt_dp.ranks), list(range(r % ENC_EP, WORLD, ENC_EP))
        )
        self.assertIs(
            encoder_ep.get_encoder_pg_collection(
                ENC_EP, allow_color_buckets=True
            ),
            pgs,
        )
        self.assertEqual(encoder_ep.get_encoder_ep_degree(), ENC_EP)
        self.assertIsNotNone(pgs.expert_rng_tracker_name)

    def test_expert_color_and_replica_init(self):
        layer = _moe_layer(ENC_EP, self.enc_pgs)
        self.assertEqual(layer.expert_model_parallel_size, ENC_EP)
        w1 = layer.grouped_gemm_experts.weight1
        self.assertIs(w1.color["group"], self.enc_pgs.expt_dp)
        local = w1.astype("float32")
        replicas = []
        dist.all_gather(replicas, local, group=self.enc_pgs.expt_dp)
        for other in replicas[1:]:
            np.testing.assert_array_equal(replicas[0].numpy(), other.numpy())
        peers = []
        dist.all_gather(peers, local, group=self.enc_pgs.ep)
        self.assertFalse(np.array_equal(peers[0].numpy(), peers[1].numpy()))

    def test_encoder_and_decoder_layers_back_to_back(self):
        dec = _moe_layer(
            DEC_EP, ProcessGroupCollection.use_mpu_process_groups()
        )
        enc = _moe_layer(ENC_EP, self.enc_pgs)
        gen = np.random.RandomState(1000 + self.rank)
        for _ in range(2):
            for layer in (dec, enc):
                x = paddle.to_tensor(
                    gen.standard_normal((2, 64, H)).astype("float32")
                ).astype("bfloat16")
                x.stop_gradient = False
                out = layer(x)[0]
                out.astype("float32").sum().backward()
                self.assertTrue(
                    np.isfinite(x.grad.astype("float32").numpy()).all()
                )
        group_ids = {self.enc_pgs.ep.id, dec.moe_group.id}
        self.assertEqual(len(group_ids), 2)
        self.assertTrue(group_ids <= set(fused_a2a._buffers))


if __name__ == "__main__":
    unittest.main()
