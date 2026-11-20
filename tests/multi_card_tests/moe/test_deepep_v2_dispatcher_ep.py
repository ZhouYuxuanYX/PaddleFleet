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
"""Real expert-parallel behaviour of the DeepEP V2.5 backend (``deepep_v2``).

EP = world size, so every dispatch/combine goes through ``deep_ep_v2``'s real
GIN kernels over the real EP group -- no mocked collective, no faked world size.
Three properties:

  * V2.5 reproduces V1 (``deepep``) on a whole ``MoELayer``: forward output,
    input grad, router-gate grad (the probs-grad path back through combine),
    routed-expert and shared-expert weight grads, in BF16 and in FP8;
  * the expanding layout (``deepep_v2_buffer_configs['do_expand']=True``)
    reproduces the default layout;
  * ``get_buffer_v2`` negotiates one token bound for the whole group, aligns it,
    and never shrinks it.

The validation that needs no communicator (``deepep_v2_buffer_configs`` keys,
``configure_buffer_v2`` reset-to-defaults, the static-bound ``RuntimeError``)
lives in ``tests/single_card_tests/moe/test_deepep_v2_config.py``.
"""

import unittest

import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
import paddlefleet_ops
from paddle.distributed import fleet

import paddlefleet.transformer.moe.fused_a2a as fleet_fused_a2a
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.mlp import MLPSublayersSpec
from paddlefleet.transformer.moe import MoELayer, MoESublayers, deepep_v2
from paddlefleet.transformer.moe.token_dispatcher import fused_dispatch
from paddlefleet.transformer.transformer_config import TransformerConfig

SEQLEN = 2048
HIDDEN_SIZE = 1024
LOCAL_EXPERTS = 8  # routed experts scale with the world, so EP stays exact
TOPK = 4

# A combine that sums the ranks' contributions in a different order rounds at the
# scale of its partial sums, not of its (possibly cancelled) final value, so the
# BF16 tolerance is measured in ulps of each row's largest reference value.
MAX_ROW_ULP = 2.0
# FP32 tensors that sum over every received token (expert weight grads, router
# gate grad) see identical summands in a different order: compare by direction.
MIN_COSINE = 0.99999
# Cosine is scale invariant, so it accepts a probs grad that came back doubled or
# halved. Pair it with a magnitude-sensitive bound: relative L2 error, which a
# wrong scale blows up to O(1) while reordered FP32 summation keeps it near 1e-7.
MAX_REL_L2 = 1e-5

TENSOR_NAMES = (
    "out",
    "hs_grad",
    "w1_grad",
    "w2_grad",
    "shared_w1_grad",
    "shared_w2_grad",
    "gate_grad",
)
# Reduced over tokens and ranks downstream of dispatch -> order-dependent.
REORDERED = {"w1_grad", "w2_grad", "gate_grad"}
# Computed from this rank's own tokens only, so the backend must not touch them.
RANK_LOCAL = {"shared_w1_grad", "shared_w2_grad"}


def _bf16_row_ulp(ref, tgt):
    """Max |ref - tgt| per row, in ulps of the BF16 ulp of that row's largest |ref|."""
    ref = ref.astype("float32").reshape([-1, ref.shape[-1]])
    tgt = tgt.astype("float32").reshape([-1, tgt.shape[-1]])
    row_max = paddle.clip(ref.abs().max(axis=1, keepdim=True), min=1e-30)
    exponent = paddle.floor(paddle.log2(row_max)) - 7  # BF16: 7 mantissa bits
    ulp = paddle.pow(paddle.full_like(row_max, 2.0), exponent)
    return float(((ref - tgt).abs() / ulp).max())


def _cosine(ref, tgt):
    return float(
        F.cosine_similarity(
            ref.astype("float32").flatten(),
            tgt.astype("float32").flatten(),
            axis=0,
            eps=0,
        )
    )


def _rel_l2(ref, tgt):
    """``‖ref - tgt‖₂ / ‖ref‖₂``: unlike cosine, it sees a wrong overall scale."""
    ref = ref.astype("float32")
    tgt = tgt.astype("float32")
    denom = float(paddle.linalg.norm(ref))
    return float(paddle.linalg.norm(ref - tgt)) / (denom or 1.0)


@unittest.skipUnless(
    paddlefleet_ops.is_deep_ep_v2_available(),
    "deepep_v2 needs paddlefleet_ops.deep_ep_v2 (NCCL device API) and SM >= 9.0",
)
class TestDeepEPV2ExpertParallel(unittest.TestCase):
    """DeepEP V2.5 over the real EP group, EP = world size."""

    @classmethod
    def setUpClass(cls):
        world_size = dist.get_world_size()
        strategy = fleet.DistributedStrategy()
        # The dense degrees must cover the whole world or EPHybridCommunicateGroup
        # cannot assign data-parallel ids; sharding takes that role and the expert
        # group overlays it, as in the sibling dispatcher tests.
        strategy.hybrid_configs = {
            "sharding_degree": world_size,
            "pp_degree": 1,
            "sep_degree": 1,
            "cp_degree": 1,
            "dp_degree": 1,
            "ep_degree": world_size,
            "mp_degree": 1,
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
        model_parallel_cuda_manual_seed(dist.get_rank())
        cls.group = ProcessGroupCollection.use_mpu_process_groups()
        cls.ep_group = cls.group.ep

    @staticmethod
    def build_config(dispatcher_type, fp8=None, buffer_configs=None):
        return TransformerConfig(
            hidden_size=HIDDEN_SIZE,
            moe_intermediate_size=512,  # routed and shared experts alike
            gated_linear_unit=True,
            moe_latent_size=512,
            latent_moe_use_norm=True,
            n_routed_experts=LOCAL_EXPERTS * dist.get_world_size(),
            n_shared_experts=1,
            num_experts_per_tok=TOPK,
            topk_method="noaux_tc",
            moe_token_dispatcher_type=dispatcher_type,
            fp8=fp8,
            fp8_wgrad=False,
            use_ue8m0=True,
            moe_topk_fusion=True,
            routing_map_fusion=True,
            sigmoid_gate_fusion=True,
            moe_expert_fusion=True,
            moe_shared_expert_overlap=True,
            hidden_act=F.silu,
            # No num_max_tokens_per_rank: let the group negotiate the bound, which
            # is what training does and what exercises the all-reduce path.
            deepep_v2_buffer_configs=buffer_configs,
        )

    def build_layer(self, dispatcher_type, state_dict=None, **kwargs):
        mlp_spec = MLPSublayersSpec(
            up_gate_proj=ColumnParallelLinear,
            down_proj=RowParallelLinear,
        )
        config = self.build_config(dispatcher_type, **kwargs)
        layer = MoELayer(config, MoESublayers(mlp_spec), self.group)
        layer = paddle.amp.decorate(layer, level="O2", dtype="bfloat16")
        if state_dict is not None:
            layer.set_state_dict(state_dict)
        return layer

    @staticmethod
    def _weight_grads(layer):
        return [
            layer.grouped_gemm_experts.weight1.grad,
            layer.grouped_gemm_experts.weight2.grad,
            layer.shared_experts.up_gate_proj.weight.grad,
            layer.shared_experts.down_proj.weight.grad,
            layer.gate.weight.grad,
        ]

    def run_layer(self, layer, hidden_states, out_grad):
        """One warmup step (discarded) and one measured step; returns the tensors to compare."""
        if layer.fp8:
            layer.fp8_quant_weight(batch_mode=True)

        # 真实情况下传入的 hs 是经过 norm 的, 不是 leaf 节点, 这里用 clone 模拟 norm
        hidden_states = hidden_states.detach()
        hidden_states.stop_gradient = False
        with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
            out, _ = layer(hidden_states.clone())
        out.backward(out_grad)

        grads = self._weight_grads(layer)
        for t in grads:
            self.assertIsNotNone(t, "a compared parameter got no gradient")
            t.zero_()
        # 让 test 充分 overlap, 暴露出 overlap 可能存在的问题
        dist.all_reduce(paddle.empty([1]))

        hidden_states = hidden_states.detach()
        hidden_states.stop_gradient = False
        with paddle.amp.auto_cast(enable=True, dtype="bfloat16"):
            out, _ = layer(hidden_states.clone())
        out.backward(out_grad)

        return dict(
            zip(
                TENSOR_NAMES,
                [out, hidden_states.grad, *self._weight_grads(layer)],
            )
        )

    def check(self, ref, tgt, label, rank_local_bitwise=True):
        """Assert every compared tensor matches, by the criterion its reduction allows.

        ``rank_local_bitwise``: the shared-expert grads are computed from this rank's
        own tokens, so a dispatcher swap must leave them bit-identical. The expanding
        layout moves the shared expert into the fusion node, which re-orders that
        addition, so there it is compared like the other reordered sums.
        """
        failures = []
        for name in TENSOR_NAMES:
            a, b = ref[name], tgt[name]
            diff = (a.astype("float32") - b.astype("float32")).abs()
            max_diff = float(diff.max())
            if max_diff == 0.0:
                print(f"[{label}] {name}: bitwise equal")
                continue
            if name in RANK_LOCAL and rank_local_bitwise:
                failures.append(
                    f"{name} is not bitwise equal (max={max_diff:e})"
                )
            elif name in REORDERED or name in RANK_LOCAL:
                cos = _cosine(a, b)
                rel = _rel_l2(a, b)
                print(
                    f"[{label}] {name}: max={max_diff:e} cos={cos:.8f} "
                    f"rel_l2={rel:e}"
                )
                if cos <= MIN_COSINE:
                    failures.append(f"{name} cosine {cos:.8f} <= {MIN_COSINE}")
                # Catches what cosine cannot: a uniformly scaled gradient.
                if rel > MAX_REL_L2:
                    failures.append(f"{name} rel_l2 {rel:e} > {MAX_REL_L2:e}")
            else:
                ulp = _bf16_row_ulp(a, b)
                print(f"[{label}] {name}: max={max_diff:e} row_ulp={ulp:.2f}")
                if ulp > MAX_ROW_ULP:
                    failures.append(f"{name} row_ulp {ulp:.2f} > {MAX_ROW_ULP}")
        # Every rank must agree: a mismatch on one rank only is still a failure.
        gathered = []
        dist.all_gather_object(gathered, failures)
        self.assertEqual(
            [f for r in gathered for f in r], [], f"{label}: {gathered}"
        )

    @staticmethod
    def _inputs():
        hidden_states = paddle.randn([1, SEQLEN, HIDDEN_SIZE], dtype="bfloat16")
        out_grad = (paddle.randn(hidden_states.shape) * 0.02).cast("bfloat16")
        return hidden_states, out_grad

    def _compare_backends(self, fp8):
        """V1 (``deepep``) is the reference; V2.5 must reproduce it on the same weights."""
        hidden_states, out_grad = self._inputs()

        v1_layer = self.build_layer("deepep", fp8=fp8)
        state_dict = v1_layer.state_dict()
        v1 = self.run_layer(v1_layer, hidden_states, out_grad)
        # V1's buffer is sized for V1's SM count; drop it so the two backends do not
        # compete for the same workspace.
        fleet_fused_a2a._buffer = None
        dist.barrier()

        v2_layer = self.build_layer("deepep_v2", state_dict=state_dict, fp8=fp8)
        self.assertEqual(
            type(v2_layer.token_dispatcher._comm_manager).__name__,
            "_DeepEPV2Manager",
            "the layer did not select the V2.5 comm manager",
        )
        v2 = self.run_layer(v2_layer, hidden_states, out_grad)
        dist.barrier()

        self.check(v1, v2, f"deepep vs deepep_v2 fp8={fp8}")

    @unittest.skipIf(fused_dispatch is None, "DeepEP V1 is unavailable")
    def test_deepep_v2_matches_deepep_v1_bf16(self):
        self._compare_backends(fp8=None)

    @unittest.skipIf(fused_dispatch is None, "DeepEP V1 is unavailable")
    def test_deepep_v2_matches_deepep_v1_fp8(self):
        # Exercises the FP8 dispatch payload (quantize + per-block scales on the wire),
        # a different code path from the BF16 one above.
        self._compare_backends(fp8="e4m3")

    def test_expand_layout_matches_default_layout(self):
        """``do_expand=True`` returns expert-contiguous padded rows; the result must not change.

        The two layers are also built before either runs, with different
        ``deepep_v2_buffer_configs``, which is the RL actor/reference situation: each layer
        must keep the layout it was built with even though ``configure_buffer_v2`` writes
        module-level state and the second build resets it.
        """
        hidden_states, out_grad = self._inputs()

        expand_layer = self.build_layer(
            "deepep_v2", buffer_configs={"do_expand": True}
        )
        self.assertTrue(deepep_v2.is_expand_enabled())
        # Built second, so its configure_buffer_v2 restores the default layout globally.
        default_layer = self.build_layer(
            "deepep_v2", state_dict=expand_layer.state_dict()
        )
        self.assertFalse(deepep_v2.is_expand_enabled())

        def layout_of(layer):
            manager = layer.token_dispatcher._comm_manager
            return (
                layer._use_deepep_v2_expand_fusion(),
                manager.kernel_config.do_expand,
            )

        # Neither layer reads the module config at forward time any more.
        self.assertEqual(layout_of(expand_layer), (True, True))
        self.assertEqual(layout_of(default_layer), (False, False))

        expand_out = self.run_layer(expand_layer, hidden_states, out_grad)
        dist.barrier()
        default_out = self.run_layer(default_layer, hidden_states, out_grad)
        dist.barrier()

        self.check(
            default_out, expand_out, "expand layout", rank_local_bitwise=False
        )

    def test_token_bound_is_negotiated_aligned_and_monotonic(self):
        """Without a static bound the group agrees on one aligned bound and never shrinks it."""
        deepep_v2.configure_buffer_v2()  # defaults: no static num_max_tokens_per_rank
        group = self.ep_group
        alignment = deepep_v2._TOKEN_ALIGNMENT

        # Ranks disagree on the local token count; the bound must cover the largest.
        local = 1 + dist.get_rank() * 10
        bound = deepep_v2._required_num_tokens(group, local)
        expected_max = 1 + (group.nranks - 1) * 10
        self.assertEqual(bound, -(-expected_max // alignment) * alignment)
        bounds = []
        dist.all_gather_object(bounds, bound)
        self.assertEqual(
            set(bounds), {bound}, f"ranks negotiated different bounds: {bounds}"
        )

        # The buffer follows that bound, is reused below it, and grows above it.
        first = deepep_v2.get_buffer_v2(group, 4 * alignment, HIDDEN_SIZE, TOPK)
        capacity = deepep_v2._buffers[group.id].num_max_tokens_per_rank
        self.assertGreaterEqual(capacity, 4 * alignment)

        same = deepep_v2.get_buffer_v2(group, alignment, HIDDEN_SIZE, TOPK)
        self.assertIs(same, first, "a smaller request must reuse the buffer")
        self.assertEqual(
            deepep_v2._buffers[group.id].num_max_tokens_per_rank,
            capacity,
            "a smaller request must not shrink the bound",
        )

        grown = deepep_v2.get_buffer_v2(group, capacity + 1, HIDDEN_SIZE, TOPK)
        self.assertIsNot(
            grown, first, "exceeding the bound must rebuild the buffer"
        )
        self.assertGreater(
            deepep_v2._buffers[group.id].num_max_tokens_per_rank, capacity
        )


if __name__ == "__main__":
    unittest.main()
