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

import random
import unittest
from contextlib import ExitStack, contextmanager
from unittest import mock

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn.functional as F
import paddlefleet_ops
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import mix_precision_utils

from paddlefleet.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
)
from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import model_parallel_cuda_manual_seed
from paddlefleet.training.global_vars import unset_global_variables
from paddlefleet.training.initialize import initialize_fleet
from paddlefleet.transformer.moe.moe_layer import MoELayer
from paddlefleet.transformer.transformer_config import TransformerConfig

if paddlefleet_ops.is_sonic_moe_available():
    from paddlefleet_ops.sonicmoe.functional import clear_all_fp8_weight_caches


class SonicMoETopk(paddle.autograd.PyLayer):
    """PyLayer wrapping SonicMoE's high-performance topk operator.

    Supports:
      - softmax_fusion: fuse softmax into the topk kernel (input is
        raw logits; output scores are softmax-normalized).
      - n_group > 1: group-limited greedy topk selection.
    """

    @staticmethod
    def forward(
        ctx,
        scores,
        k,
        n_group,
        topk_group,
        softmax_fusion,
    ):
        """
        Args:
            scores: [T, E] tensor. If softmax_fusion=True, these are raw
                logits (pre-softmax). Otherwise, already-scored gates.
            k: number of experts per token.
            n_group: number of expert groups. 1 means no grouping.
            topk_group: number of groups to select when n_group > 1.
            softmax_fusion: whether to fuse softmax into the topk kernel.

        Returns:
            top_gate: [T, k] float32 scores of selected experts.
            top_idx: [T, k] int64 indices of selected experts.
        """
        from paddlefleet_ops.sonicmoe.functional.forward import (
            _topk_fwd,
        )

        T, E = scores.shape

        if n_group > 1:
            # Group-limited greedy: select top groups first, then topk
            # within those groups.
            group_scores = scores.reshape([T, n_group, -1]).max(axis=-1)
            group_idx = paddle.topk(
                group_scores, k=topk_group, axis=-1, sorted=True
            )[1]
            group_mask = paddle.zeros_like(group_scores).put_along_axis(
                group_idx,
                paddle.to_tensor(1.0, dtype="float32"),
                axis=-1,
            )
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand([T, n_group, E // n_group])
                .reshape([T, -1])
            )
            # Apply mask: zero out non-selected groups.
            # Use a very negative value for softmax_fusion (logits)
            # so softmax gives ~0, or just zero for non-fusion.
            if softmax_fusion:
                masked_scores = scores + (1.0 - score_mask) * (-1e9)
            else:
                masked_scores = scores * score_mask
        else:
            masked_scores = scores

        topk_scores = paddle.empty([T, k], dtype=paddle.float32)
        topk_indices = paddle.empty([T, k], dtype=paddle.int32)
        # print("==== sonicmoe topk fwd ====")
        _topk_fwd(
            masked_scores,
            k,
            topk_scores,
            topk_indices,
            require_softmax_fusion=softmax_fusion,
        )

        ctx.save_for_backward(topk_scores, topk_indices)
        ctx.E = E
        ctx.K = k
        ctx.softmax_fusion = softmax_fusion
        ctx.input_dtype = scores.dtype

        return topk_scores, topk_indices.cast(paddle.int64)

    @staticmethod
    def backward(ctx, dtopk_score, _):
        from paddlefleet_ops.sonicmoe.functional.backward import (
            _softmax_topk_bwd,
        )

        # print("==== sonicmoe topk bwd ====")
        # assert 0
        topk_scores, topk_indices = ctx.saved_tensor()
        T = dtopk_score.shape[0]
        K = ctx.K
        if ctx.softmax_fusion:
            dlogits = paddle.zeros([T, ctx.E], dtype=ctx.input_dtype)
            _softmax_topk_bwd(
                dlogits, None, dtopk_score, topk_scores, topk_indices, K
            )
            return dlogits
        else:
            # No softmax fusion: gradient is simply scattered back
            dscores = paddle.zeros([T, ctx.E], dtype=dtopk_score.dtype)
            dscores = dscores.put_along_axis_(
                topk_indices.cast(paddle.int64),
                dtopk_score,
                axis=1,
            )
            return dscores


@unittest.skipUnless(
    paddlefleet_ops.is_sonic_moe_available(),
    "Sonic-MoE not available (requires Python>=3.12, CUDA>=12.9, SM>=90)",
)
class TestSonicMoEExpertParallelPrecision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 1,
            "pp_degree": 1,
            "sharding_degree": 8,
            "sep_degree": 1,
            "cp_degree": 1,
            "ep_degree": 8,
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
        initialize_fleet(strategy=strategy)
        if dist.get_world_size() != 8:
            raise AssertionError(
                "Sonic MoE EP acceptance requires world_size == 8, got "
                f"{dist.get_world_size()}"
            )
        cls.pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        if cls.pg_collection.ep.nranks != 8:
            raise AssertionError(
                "Sonic MoE EP acceptance requires an 8-rank EP group, got "
                f"{cls.pg_collection.ep.nranks}"
            )

    @classmethod
    def tearDownClass(cls):
        unset_global_variables()

    def setUp(self):
        clear_all_fp8_weight_caches()
        self.seed = 123
        self.hidden_size = 2048
        self.n_routed_experts = 64

        random.seed(self.seed)
        np.random.seed(self.seed)
        paddle.seed(self.seed)
        paddle.manual_seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        self.pg_collection = self.__class__.pg_collection

    @staticmethod
    def calc_diff(x: paddle.Tensor, y: paddle.Tensor):
        x, y = x.double(), y.double()
        denominator = (x * x + y * y).sum()
        if denominator.item() == 0:
            return 0.0
        sim = 2 * (x * y).sum() / denominator
        return (1 - sim).item()

    @staticmethod
    def _small_init_method(tensor):
        """Small uniform init for precision tests (matches pre-update behavior)."""
        paddle.nn.initializer.Uniform(-0.001, 0.001)(tensor)

    def _build_transformer_config(
        self,
        using_sonic_moe=False,
        fp8=None,
        moe_deep_gemm=False,
        fp8_wgrad=True,
        expert_model_parallel_size=8,
        fp8_weight_quant_format="32x32",
    ):
        return TransformerConfig(
            hidden_size=self.hidden_size,
            num_attention_heads=4,
            n_routed_experts=self.n_routed_experts,
            use_cpu_initialization=False,
            num_experts_per_tok=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=expert_model_parallel_size,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=1024,
            gated_linear_unit=True,
            n_shared_experts=0,
            hidden_act=F.silu,
            moe_expert_fusion=True,
            moe_deep_gemm=moe_deep_gemm,
            bias_activation_fusion=True,
            moe_token_dispatcher_type="deepep",
            moe_use_fusion_node=True,
            using_sonic_moe=using_sonic_moe,
            fp8=fp8,
            fp8_wgrad=fp8_wgrad,
            fp8_weight_quant_format=fp8_weight_quant_format,
            use_ue8m0=bool(fp8),
            init_method=self._small_init_method,
            output_layer_init_method=self._small_init_method,
        )

    def _build_moe_layer(
        self,
        using_sonic_moe=False,
        fp8=None,
        moe_deep_gemm=False,
        fp8_wgrad=True,
        expert_model_parallel_size=4,
        pg_collection=None,
        config=None,
        fp8_weight_quant_format="32x32",
    ):
        transformer_config = config or self._build_transformer_config(
            using_sonic_moe=using_sonic_moe,
            fp8=fp8,
            moe_deep_gemm=moe_deep_gemm,
            fp8_wgrad=fp8_wgrad,
            expert_model_parallel_size=expert_model_parallel_size,
            fp8_weight_quant_format=fp8_weight_quant_format,
        )
        transformer_layer_spec = get_gpt_layer_local_spec(
            transformer_config,
            num_experts=self.n_routed_experts,
        )
        moe_layer = MoELayer(
            transformer_config,
            transformer_layer_spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            pg_collection or self.pg_collection,
        )
        gate_weight = moe_layer.gate.weight
        gate_had_main_grad = hasattr(gate_weight, "main_grad")
        if not gate_had_main_grad:
            gate_weight.main_grad = None
        mix_precision_utils.MixPrecisionLayer(moe_layer, dtype="bfloat16")
        if not gate_had_main_grad:
            del gate_weight.main_grad
        expert = getattr(moe_layer, "grouped_gemm_experts", None)
        if expert is not None:
            for param in (expert.weight1, expert.weight2):
                if hasattr(param, "main_grad") and param.main_grad is None:
                    param.main_grad = paddle.zeros_like(
                        param, dtype=paddle.float32
                    )
        return moe_layer

    @staticmethod
    def _expert_slice_for_rank(tensor, ep_rank, ep_size):
        chunk_size = tensor.shape[0] // ep_size
        return tensor[ep_rank * chunk_size : (ep_rank + 1) * chunk_size]

    @classmethod
    def _copy_single_card_weights_to_ep(
        cls, src_layer, dst_layer, ep_rank, ep_size
    ):
        src_params = dict(src_layer.named_parameters())
        for name, dst_param in dst_layer.named_parameters():
            src_param = src_params[name]
            if "grouped_gemm_experts.weight" in name:
                src_param = cls._expert_slice_for_rank(
                    src_param, ep_rank, ep_size
                )
            dst_param.set_value(src_param.clone())

    @classmethod
    def _gather_ep_expert_grads(cls, grads, ep_group, ep_size):
        gathered_grads = {}
        for name, grad in grads.items():
            if "grouped_gemm_experts.weight" not in name:
                gathered_grads[name] = grad
                continue
            parts = []
            dist.all_gather(parts, grad, group=ep_group)
            gathered_grads[name] = paddle.concat(parts, axis=0) / ep_size
        return gathered_grads

    @staticmethod
    def _is_expert_param(name):
        return name in {
            "grouped_gemm_experts.weight1",
            "grouped_gemm_experts.weight2",
        }

    @classmethod
    def _collect_grads(cls, layer):
        grads = {}
        for name, param in layer.named_parameters():
            main_grad = getattr(param, "main_grad", None)
            if cls._is_expert_param(name):
                grad = main_grad if main_grad is not None else param.grad
            else:
                grad = param.grad
            if grad is not None:
                grads[name] = grad.detach().clone()
        return grads

    @staticmethod
    def _clear_grads(layer):
        for _, param in layer.named_parameters():
            if hasattr(param, "main_grad") and param.main_grad is not None:
                param.main_grad.zero_()
            if param.grad is not None:
                param.grad.zero_()

    def _flush_sonic_expert_layout(self, moe_layer):
        expert = getattr(moe_layer, "grouped_gemm_experts", None)
        if not hasattr(expert, "flush_to_grouped_layout"):
            return

        weight_ptrs = (expert.weight1.data_ptr(), expert.weight2.data_ptr())
        grad_ptrs = []
        for param in (expert.weight1, expert.weight2):
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            grad_ptrs.append(None if grad is None else grad.data_ptr())

        expert.flush_to_grouped_layout()

        self.assertEqual(expert.weight1.data_ptr(), weight_ptrs[0])
        self.assertEqual(expert.weight2.data_ptr(), weight_ptrs[1])
        for param, grad_ptr in zip((expert.weight1, expert.weight2), grad_ptrs):
            if grad_ptr is None:
                continue
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            self.assertEqual(grad.data_ptr(), grad_ptr)

    @staticmethod
    def _deterministic_upstream(output, step_idx):
        values = paddle.arange(output.numel(), dtype=paddle.float32).reshape(
            output.shape
        )
        scale = 1.0 / max(output.numel(), 1)
        return ((values % 31) - 14.0 + step_idx * 0.25) * scale

    @staticmethod
    def _sync_initialized_parameters(source, *destinations):
        source_params = dict(source.named_parameters())
        for destination in destinations:
            destination_params = dict(destination.named_parameters())
            if set(destination_params) != set(source_params):
                raise AssertionError(
                    "MoE parameter names differ before synchronization"
                )
            for name, source_param in source_params.items():
                destination_params[name].set_value(
                    source_param.detach().clone()
                )

    @contextmanager
    def _record_sonic_calls(self):
        from paddlefleet_ops.sonicmoe import functional
        from paddlefleet_ops.sonicmoe.quack_utils import sm100_funcs

        records = {"gated": [], "dgated": [], "wgrad": []}

        def record_gemm(kind, original):
            def wrapped(*args, **kwargs):
                a_idx = kwargs.get("A_idx")
                cu_seqlens = kwargs.get("cu_seqlens_m")
                records[kind].append(
                    {
                        "source_rows": int(args[0].shape[0]),
                        "route_rows": int(a_idx.numel()),
                        "route_end": int(cu_seqlens[-1].item()),
                        "a_idx_min": int(a_idx.min().item()),
                        "a_idx_max": int(a_idx.max().item()),
                        "a_idx": a_idx,
                        "cu_seqlens": cu_seqlens,
                    }
                )
                return original(*args, **kwargs)

            return wrapped

        def record_wgrad(name, original):
            def wrapped(*args, **kwargs):
                records["wgrad"].append(
                    {
                        "symbol": name,
                        "source_rows": int(args[0].shape[1]),
                        "total_k": int(kwargs["total_K"]),
                        "route_rows": int(args[3].numel()),
                        "route_end": int(args[2][-1].item()),
                        "a_idx_min": int(args[3].min().item()),
                        "a_idx_max": int(args[3].max().item()),
                        "a_idx": args[3],
                        "cu_seqlens": args[2],
                    }
                )
                return original(*args, **kwargs)

            return wrapped

        with ExitStack() as stack:
            # BF16 path: the QuACK interfaces called from functional.  SM100
            # FP8 path: the QuACK gated interface and the dgated kernel called
            # from sm100_funcs (its gemm_dgated op reads
            # ``quack_gemm_dgated.default_config``, so that name stays unpatched).
            for module, name, kind in (
                (functional, "gemm_gated", "gated"),
                (functional, "gemm_dgated", "dgated"),
                (sm100_funcs, "quack_gemm_gated", "gated"),
                (sm100_funcs, "gemm_dgated_kernel", "dgated"),
            ):
                original = getattr(module, name)
                stack.enter_context(
                    mock.patch.object(module, name, record_gemm(kind, original))
                )
            for name in (
                "bf16_wgrad_gemm_varlen_k",
                "bf16_wgrad_gemm_varlen_k_accumulate",
                "bf16_wgrad_gemm_varlen_k_tma_add",
            ):
                original = getattr(functional, name)
                stack.enter_context(
                    mock.patch.object(
                        functional, name, record_wgrad(name, original)
                    )
                )
            yield records

    def _assert_sonic_calls(self, records, expected_calls):
        local_experts = self.n_routed_experts // self.pg_collection.ep.nranks
        self.assertEqual(len(records["gated"]), expected_calls)
        self.assertEqual(len(records["dgated"]), expected_calls)
        for phase in ("gated", "dgated"):
            for record in records[phase]:
                a_idx = record["a_idx"]
                cu_seqlens = record["cu_seqlens"]
                self.assertIsNotNone(a_idx, f"{phase} must use A_idx")
                self.assertEqual(record["route_rows"], record["route_end"])
                self.assertGreater(record["source_rows"], 0)
                self.assertGreaterEqual(
                    record["route_rows"], record["source_rows"]
                )
                # DeepEP route padding gathers the synthetic zero-score rows from
                # real source row 0; no expanded token tensor or out-of-range
                # sentinel row may be materialized.
                self.assertGreaterEqual(record["a_idx_min"], 0)
                self.assertLess(record["a_idx_max"], record["source_rows"])
                self.assertEqual(str(a_idx.dtype), "paddle.int32")
                self.assertIsNotNone(cu_seqlens)
                self.assertEqual(int(cu_seqlens.numel()), local_experts + 1)
        self.assertEqual(len(records["wgrad"]), 2 * expected_calls)
        for record in records["wgrad"]:
            self.assertEqual(record["total_k"], record["route_rows"])
            self.assertEqual(record["total_k"], record["route_end"])
            self.assertGreater(record["source_rows"], 0)
            self.assertGreaterEqual(record["total_k"], record["source_rows"])
            self.assertGreaterEqual(record["a_idx_min"], 0)
            self.assertLess(record["a_idx_max"], record["source_rows"])
            self.assertEqual(str(record["a_idx"].dtype), "paddle.int32")
            self.assertEqual(str(record["cu_seqlens"].dtype), "paddle.int32")
            self.assertEqual(
                int(record["cu_seqlens"].numel()), local_experts + 1
            )

    def _run_forward_backward(self, moe_layer, input_data):
        moe_layer = paddle.amp.decorate(
            models=moe_layer,
            level="O2",
            dtype="bfloat16",
            master_grad=True,
            master_weight=True,
        )
        # mix_precision_utils.MixPrecisionLayer(moe_layer, dtype="bfloat16")
        hidden_states = input_data.detach().clone()
        hidden_states.stop_gradient = False
        with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
            output = moe_layer(hidden_states)[0]
        upstream = self._deterministic_upstream(output, 0)
        loss_value = (output.cast("float32") * upstream).sum().item()
        output.backward(upstream.cast(output.dtype))
        self._flush_sonic_expert_layout(moe_layer)
        return (
            output.detach().clone(),
            loss_value,
            hidden_states.grad.detach().clone(),
            self._collect_grads(moe_layer),
        )

    def _run_accumulated_forward_backward(self, moe_layer, input_data_list):
        self._clear_grads(moe_layer)
        outputs = []
        input_grads = []
        losses = []
        for step_idx, input_data in enumerate(input_data_list):
            hidden_states = input_data.detach().clone()
            hidden_states.stop_gradient = False
            with paddle.amp.auto_cast(level="O2", dtype="bfloat16"):
                output = moe_layer(hidden_states)[0]
            upstream = self._deterministic_upstream(output, step_idx)
            losses.append((output.cast("float32") * upstream).sum().item())
            output.backward(upstream.cast(output.dtype))
            outputs.append(output.detach().clone())
            input_grads.append(hidden_states.grad.detach().clone())
        self._flush_sonic_expert_layout(moe_layer)
        return (
            outputs[-1],
            losses[-1],
            input_grads[-1],
            self._collect_grads(moe_layer),
        )

    def _assert_loss_close(self, lhs, rhs, rtol, atol, title):
        adiff = abs(lhs - rhs)
        scale = max(abs(lhs), abs(rhs))
        rdiff = adiff / max(scale, 1e-12)
        allowed = atol + rtol * scale
        print(
            f"{title}: loss relative diff = {rdiff:.6e}, "
            f"absolute diff = {adiff:.6e}, allowed = {allowed:.6e}"
        )
        self.assertLessEqual(
            adiff,
            allowed,
            f"{title} loss deviates too much: lhs={lhs}, rhs={rhs}, "
            f"relative={rdiff:.6e}, absolute={adiff:.6e}, "
            f"allowed={allowed:.6e}",
        )

    def _assert_tensor_diff_less(self, lhs, rhs, tol, title):
        diff = self.calc_diff(lhs, rhs)
        max_abs_diff = (
            (lhs.cast("float32") - rhs.cast("float32")).abs().max().item()
        )
        signal_norm = rhs.cast("float32").norm().item()
        print(
            f"{title}: diff = {diff:.6e}, max abs diff = {max_abs_diff:.6e}, "
            f"reference norm = {signal_norm:.6e}"
        )
        if signal_norm < 1e-2:
            self.assertLess(
                max_abs_diff,
                5e-5,
                f"{title} absolute diff too large: max_abs_diff={max_abs_diff:.6e}",
            )
        else:
            self.assertLess(
                diff, tol, f"{title} diff too large: diff={diff:.6e}"
            )

    def _assert_grad_diff_less(
        self,
        lhs_grads,
        rhs_grads,
        tol,
        title,
    ):
        lhs_names = set(lhs_grads)
        rhs_names = set(rhs_grads)
        self.assertEqual(
            lhs_names,
            rhs_names,
            (
                f"Gradient tensors mismatch for {title}: "
                f"lhs_only={sorted(lhs_names - rhs_names)}, "
                f"rhs_only={sorted(rhs_names - lhs_names)}"
            ),
        )
        self.assertTrue(lhs_names, f"No grad tensors found for {title}")
        for name in sorted(lhs_names):
            grad_tol = tol[name] if isinstance(tol, dict) else tol
            self._assert_tensor_diff_less(
                lhs_grads[name],
                rhs_grads[name],
                tol=grad_tol,
                title=f"{title} grad {name}",
            )

    def _assert_bitwise_equal(self, lhs, rhs, title):
        max_abs_diff = (
            (lhs.cast("float32") - rhs.cast("float32")).abs().max().item()
        )
        print(f"{title}: max abs diff = {max_abs_diff:.6e}")
        self.assertEqual(
            max_abs_diff,
            0.0,
            f"{title} not bitwise equal: max_abs_diff={max_abs_diff:.6e}",
        )

    def _assert_required_grads(self, grads, input_grad, title):
        required = {
            "gate.weight",
            "grouped_gemm_experts.weight1",
            "grouped_gemm_experts.weight2",
        }
        self.assertEqual(
            required, set(grads), f"Unexpected gradients for {title}"
        )
        for name in sorted(required):
            self.assertTrue(
                paddle.isfinite(grads[name]).all().item(),
                f"Non-finite gradient for {title} {name}",
            )
            self.assertGreater(
                paddle.count_nonzero(grads[name]).item(),
                0,
                f"Untouched gradient for {title} {name}",
            )
        self.assertIsNotNone(input_grad, f"Missing input gradient for {title}")
        self.assertTrue(
            paddle.isfinite(input_grad).all().item(),
            f"Non-finite input gradient for {title}",
        )

    def run_test_sonic_moe_ep_grad_accumulation(
        self, fp8_weight_quant_format=None
    ):
        clear_all_fp8_weight_caches()
        acc_steps = 2

        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_base = self._build_moe_layer(using_sonic_moe=False)
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_sonic_bf16 = self._build_moe_layer(using_sonic_moe=True)
        moe_layer_sonic_fp8 = None
        moe_layer_sonic_fp8_dispatch = None
        if fp8_weight_quant_format is not None:
            paddle.seed(self.seed)
            model_parallel_cuda_manual_seed(self.seed)
            moe_layer_sonic_fp8 = self._build_moe_layer(
                using_sonic_moe=True,
                fp8="e4m3",
                fp8_weight_quant_format=fp8_weight_quant_format,
            )
            moe_layer_sonic_fp8.fp8_dispatch = False
            moe_layer_sonic_fp8.fp8_dispatch_bwd = False
            paddle.seed(self.seed)
            model_parallel_cuda_manual_seed(self.seed)
            moe_layer_sonic_fp8_dispatch = self._build_moe_layer(
                using_sonic_moe=True,
                fp8="e4m3",
                fp8_weight_quant_format=fp8_weight_quant_format,
            )
            moe_layer_sonic_fp8_dispatch.fp8_dispatch = True
            moe_layer_sonic_fp8_dispatch.fp8_dispatch_bwd = False
        self.assertIsNot(moe_layer_base, moe_layer_sonic_bf16)
        self.assertIsNot(
            moe_layer_base.grouped_gemm_experts,
            moe_layer_sonic_bf16.grouped_gemm_experts,
        )
        destinations = [moe_layer_sonic_bf16]
        if moe_layer_sonic_fp8 is not None:
            self.assertIsNot(moe_layer_sonic_bf16, moe_layer_sonic_fp8)
            destinations.extend(
                [moe_layer_sonic_fp8, moe_layer_sonic_fp8_dispatch]
            )
        self._sync_initialized_parameters(moe_layer_base, *destinations)

        input_data_list = []
        for step_idx in range(acc_steps):
            paddle.seed(self.seed + step_idx)
            input_data_list.append(
                paddle.randn(
                    [4, 256, self.hidden_size],
                    dtype=paddle.bfloat16,
                )
            )

        output_base, loss_base, input_grad_base, grads_base = (
            self._run_accumulated_forward_backward(
                moe_layer_base, input_data_list
            )
        )
        with self._record_sonic_calls() as bf16_records:
            output_bf16, loss_bf16, input_grad_bf16, grads_bf16 = (
                self._run_accumulated_forward_backward(
                    moe_layer_sonic_bf16, input_data_list
                )
            )
        self._assert_sonic_calls(bf16_records, acc_steps)

        for title, grads, input_grad in (
            ("baseline", grads_base, input_grad_base),
            ("Sonic BF16", grads_bf16, input_grad_bf16),
        ):
            self._assert_required_grads(grads, input_grad, title)

        self._assert_loss_close(
            loss_bf16,
            loss_base,
            rtol=1e-2,
            atol=1e-10,
            title="Sonic-MoE BF16 vs Baseline",
        )
        self._assert_tensor_diff_less(
            output_bf16,
            output_base,
            tol=1e-2,
            title="Sonic-MoE BF16 vs Baseline final output",
        )
        self._assert_grad_diff_less(
            grads_bf16,
            grads_base,
            tol=1e-2,
            title="Sonic-MoE BF16 vs Baseline accumulated grad",
        )
        self._assert_tensor_diff_less(
            input_grad_bf16,
            input_grad_base,
            tol=1e-2,
            title="Sonic-MoE BF16 vs Baseline input grad",
        )
        if fp8_weight_quant_format is None:
            return

        moe_layer_sonic_fp8.grouped_gemm_experts.quant_weight()
        with self._record_sonic_calls() as fp8_records:
            output_fp8, loss_fp8, input_grad_fp8, grads_fp8 = (
                self._run_accumulated_forward_backward(
                    moe_layer_sonic_fp8, input_data_list
                )
            )
        # The recorded wrappers (sm100_funcs.quack_gemm_gated /
        # gemm_dgated_kernel, functional.bf16_wgrad_*) are the SM100 QuACK FP8
        # path; native SM90 FP8 (quack_utils/sm90_funcs.py) calls the
        # gemm_sm90_scaled kernels directly and never reaches them.
        if paddle.device.cuda.get_device_capability()[0] >= 10:
            self._assert_sonic_calls(fp8_records, acc_steps)
        moe_layer_sonic_fp8_dispatch.grouped_gemm_experts.quant_weight()
        (
            output_fp8_dispatch,
            loss_fp8_dispatch,
            input_grad_fp8_dispatch,
            grads_fp8_dispatch,
        ) = self._run_accumulated_forward_backward(
            moe_layer_sonic_fp8_dispatch, input_data_list
        )
        clear_all_fp8_weight_caches()
        for title, grads, input_grad in (
            ("Sonic FP8", grads_fp8, input_grad_fp8),
            ("Sonic FP8 dispatch", grads_fp8_dispatch, input_grad_fp8_dispatch),
        ):
            self._assert_required_grads(grads, input_grad, title)

        fp8_tol = 5e-3
        self._assert_loss_close(
            loss_fp8,
            loss_bf16,
            rtol=fp8_tol,
            atol=1e-7,
            title="Sonic-MoE FP8 vs BF16",
        )
        self._assert_tensor_diff_less(
            output_fp8,
            output_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 final output",
        )
        self._assert_grad_diff_less(
            grads_fp8,
            grads_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 accumulated grad",
        )
        self._assert_tensor_diff_less(
            input_grad_fp8,
            input_grad_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 input grad",
        )

        self._assert_loss_close(
            loss_fp8_dispatch,
            loss_bf16,
            rtol=fp8_tol,
            atol=1e-7,
            title="Sonic-MoE FP8-dispatch vs BF16",
        )
        self._assert_tensor_diff_less(
            output_fp8_dispatch,
            output_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8-dispatch vs BF16 final output",
        )
        self._assert_grad_diff_less(
            grads_fp8_dispatch,
            grads_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8-dispatch vs BF16 accumulated grad",
        )
        self._assert_tensor_diff_less(
            input_grad_fp8_dispatch,
            input_grad_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8-dispatch vs BF16 input grad",
        )
        self._assert_tensor_diff_less(
            output_fp8_dispatch,
            output_fp8,
            tol=1e-8,
            title="Sonic-MoE FP8-dispatch vs FP8 final output",
        )
        self._assert_loss_close(
            loss_fp8_dispatch,
            loss_fp8,
            rtol=1e-8,
            atol=1e-12,
            title="Sonic-MoE FP8-dispatch vs FP8",
        )
        self._assert_tensor_diff_less(
            input_grad_fp8_dispatch,
            input_grad_fp8,
            tol=1e-8,
            title="Sonic-MoE FP8-dispatch vs FP8 input grad",
        )
        self._assert_grad_diff_less(
            grads_fp8_dispatch,
            grads_fp8,
            tol=1e-8,
            title="Sonic-MoE FP8-dispatch vs FP8 accumulated grad",
        )

        print("Final output and parameter gradient precision checks passed!")

    def run_test_ep_precision(self):
        clear_all_fp8_weight_caches()
        ep_size = self.pg_collection.ep.nranks
        ep_rank = dist.get_rank(self.pg_collection.ep)
        single_rank_group = dist.new_group([dist.get_rank()])
        single_pg_collection = ProcessGroupCollection(
            ep=single_rank_group,
            expt_dp=single_rank_group,
        )

        moe_layer_single = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
            expert_model_parallel_size=1,
            pg_collection=single_pg_collection,
        )
        moe_layer_ep = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
            expert_model_parallel_size=ep_size,
            pg_collection=self.pg_collection,
        )
        self._copy_single_card_weights_to_ep(
            moe_layer_single,
            moe_layer_ep,
            ep_rank,
            ep_size,
        )

        paddle.seed(self.seed + 2024)
        input_data = paddle.randn(
            [4, 256, self.hidden_size],
            dtype=paddle.bfloat16,
        )

        moe_layer_single.grouped_gemm_experts.quant_weight()
        output_single, loss_single, input_grad_single, grads_single = (
            self._run_forward_backward(moe_layer_single, input_data)
        )
        moe_layer_ep.grouped_gemm_experts.quant_weight()
        output_ep, loss_ep, input_grad_ep, grads_ep = (
            self._run_forward_backward(moe_layer_ep, input_data)
        )
        grads_ep = self._gather_ep_expert_grads(
            grads_ep,
            self.pg_collection.ep,
            ep_size,
        )
        clear_all_fp8_weight_caches()

        self._assert_loss_close(
            loss_ep,
            loss_single,
            rtol=2e-2,
            atol=1e-7,
            title="Sonic-MoE FP8 EP vs single-card",
        )
        self._assert_tensor_diff_less(
            output_ep,
            output_single,
            tol=5e-6,
            title="Sonic-MoE FP8 EP vs single-card output",
        )
        self._assert_tensor_diff_less(
            input_grad_ep,
            input_grad_single,
            tol=5e-6,
            title="Sonic-MoE FP8 EP vs single-card input grad",
        )
        self._assert_grad_diff_less(
            grads_ep,
            grads_single,
            tol=5e-6,
            title="Sonic-MoE FP8 EP vs single-card param grad",
        )

        print("Expert-parallel FP8 precision checks passed!")

    def run_test_bf16_wgrad(self):
        acc_steps = 2

        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_sonic_bf16 = self._build_moe_layer(using_sonic_moe=True)
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        moe_layer_sonic_fp8 = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
            fp8_wgrad=False,
        )

        input_data_list = []
        for step_idx in range(acc_steps):
            paddle.seed(self.seed + step_idx)
            input_data_list.append(
                paddle.randn(
                    [4, 256, self.hidden_size],
                    dtype=paddle.bfloat16,
                )
            )

        output_bf16, _, _, grads_bf16 = self._run_accumulated_forward_backward(
            moe_layer_sonic_bf16, input_data_list
        )
        output_fp8, _, _, grads_fp8 = self._run_accumulated_forward_backward(
            moe_layer_sonic_fp8, input_data_list
        )

        fp8_tol = 5e-3
        self._assert_tensor_diff_less(
            output_fp8,
            output_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 final output",
        )
        self._assert_grad_diff_less(
            grads_fp8,
            grads_bf16,
            tol=fp8_tol,
            title="Sonic-MoE FP8 vs BF16 accumulated grad",
        )

    def run_test_z_bf16_recompute_z(self):
        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        sonic_fp8 = self._build_moe_layer(using_sonic_moe=True, fp8="e4m3")

        recompute_config = self._build_transformer_config(
            using_sonic_moe=True, fp8="e4m3"
        )
        recompute_config.recompute_granularity = "selective"
        recompute_config.recompute_modules = ["moe_gate_up"]

        paddle.seed(self.seed)
        model_parallel_cuda_manual_seed(self.seed)
        sonic_fp8_recompute_z = self._build_moe_layer(
            using_sonic_moe=True,
            fp8="e4m3",
            config=recompute_config,
        )
        self.assertFalse(sonic_fp8.recompute_moe_gate_up)
        self.assertTrue(sonic_fp8_recompute_z.recompute_moe_gate_up)

        input_data_list = [
            paddle.randn([2, 64, self.hidden_size], dtype=paddle.bfloat16)
        ]
        output, _, _, grads = self._run_accumulated_forward_backward(
            sonic_fp8, input_data_list
        )
        output_recompute, _, _, grads_recompute = (
            self._run_accumulated_forward_backward(
                sonic_fp8_recompute_z, input_data_list
            )
        )
        clear_all_fp8_weight_caches()

        self._assert_tensor_diff_less(
            output,
            output_recompute,
            tol=1e-2,
            title="Sonic-MoE FP8 recompute_z output",
        )
        self._assert_grad_diff_less(
            grads,
            grads_recompute,
            tol=1e-2,
            title="Sonic-MoE FP8 recompute_z accumulated grad",
        )

    # Recompute must not change numerics: each recompute variant has to be
    # bitwise equal to the non-recompute variant with the same z format.
    _FP8_RECOMPUTE_EXACT_REF = {
        "recompute_z_fp8": "save_z_fp8+epilogue_quant",
    }

    # (name, sonic FP8 config overrides, recompute_moe_gate_up, env overrides)
    _FP8_CONFIG_VARIANTS = (
        (
            "save_z_fp8",
            {"save_z_fp8": True, "epilogue_quant": False},
            False,
            {},
        ),
        (
            "save_z_fp8+epilogue_quant",
            {"save_z_fp8": True, "epilogue_quant": True},
            False,
            {},
        ),
        (
            "save_z_fp8+fused_zy1_quant",
            {"save_z_fp8": True, "epilogue_quant": False},
            False,
            {"SONIC_MOE_FP8_FUSED_ZY1_QUANT": "1"},
        ),
        ("unfused_y1_quant", {"fuse_y1_quant": False}, False, {}),
        ("recompute_z_bf16", {"save_z_fp8": False}, True, {}),
        (
            "recompute_z_fp8",
            {"save_z_fp8": True, "epilogue_quant": True},
            True,
            {},
        ),
    )

    def run_test_fp8_config_variants(self, fp8_weight_quant_format):
        """Every sonic FP8 config option vs BF16 and vs the default FP8 run."""
        input_data_list = []
        for step_idx in range(2):
            paddle.seed(self.seed + step_idx)
            input_data_list.append(
                paddle.randn([4, 256, self.hidden_size], dtype=paddle.bfloat16)
            )

        def run(using_fp8, overrides=None, recompute=False, env=None):
            config = self._build_transformer_config(
                using_sonic_moe=True,
                fp8="e4m3" if using_fp8 else None,
                expert_model_parallel_size=4,
                fp8_weight_quant_format=fp8_weight_quant_format,
            )
            if recompute:
                config.recompute_granularity = "selective"
                config.recompute_modules = ["moe_gate_up"]
            paddle.seed(self.seed)
            model_parallel_cuda_manual_seed(self.seed)
            layer = self._build_moe_layer(using_sonic_moe=True, config=config)
            self.assertEqual(layer.recompute_moe_gate_up, recompute)
            for key, value in (overrides or {}).items():
                setattr(layer.grouped_gemm_experts.sonic_moe_config, key, value)
            with mock.patch.dict("os.environ", env or {}):
                output, _, input_grad, grads = (
                    self._run_accumulated_forward_backward(
                        layer, input_data_list
                    )
                )
            del layer
            clear_all_fp8_weight_caches()
            return output, input_grad, grads

        output_bf16, input_grad_bf16, grads_bf16 = run(False)
        output_fp8, input_grad_fp8, grads_fp8 = run(True)
        fp8_tol = 5e-3
        exact_refs = set(self._FP8_RECOMPUTE_EXACT_REF.values())
        variant_results = {}
        for name, overrides, recompute, env in self._FP8_CONFIG_VARIANTS:
            print(f"==== FP8 config {name} ({fp8_weight_quant_format}) ====")
            output, input_grad, grads = run(True, overrides, recompute, env)
            title = f"Sonic-MoE FP8 [{name}, {fp8_weight_quant_format}]"
            if name in exact_refs:
                variant_results[name] = (output, input_grad, grads)
            ref_name = self._FP8_RECOMPUTE_EXACT_REF.get(name)
            if ref_name is not None:
                ref_output, ref_input_grad, ref_grads = variant_results[
                    ref_name
                ]
                ref_title = f"{title} vs [{ref_name}]"
                self._assert_bitwise_equal(
                    output, ref_output, f"{ref_title} output"
                )
                self._assert_bitwise_equal(
                    input_grad, ref_input_grad, f"{ref_title} input grad"
                )
                self.assertEqual(set(grads), set(ref_grads))
                for grad_name in sorted(grads):
                    self._assert_bitwise_equal(
                        grads[grad_name],
                        ref_grads[grad_name],
                        f"{ref_title} grad {grad_name}",
                    )
            self._assert_tensor_diff_less(
                output,
                output_bf16,
                tol=fp8_tol,
                title=f"{title} vs BF16 output",
            )
            self._assert_tensor_diff_less(
                input_grad,
                input_grad_bf16,
                tol=fp8_tol,
                title=f"{title} vs BF16 input grad",
            )
            self._assert_grad_diff_less(
                grads, grads_bf16, tol=fp8_tol, title=f"{title} vs BF16 grad"
            )
            self._assert_tensor_diff_less(
                output, output_fp8, tol=fp8_tol, title=f"{title} vs FP8 output"
            )
            self._assert_tensor_diff_less(
                input_grad,
                input_grad_fp8,
                tol=fp8_tol,
                title=f"{title} vs FP8 input grad",
            )
            self._assert_grad_diff_less(
                grads, grads_fp8, tol=fp8_tol, title=f"{title} vs FP8 grad"
            )

    def test_sonic_moe_all(self):
        formats = (
            ("32x32", "128x128")
            if paddle.device.cuda.get_device_capability()[0] == 9
            else ("32x32",)
        )
        for fp8_weight_quant_format in formats:
            # with self.subTest(fp8_weight_quant_format=fp8_weight_quant_format):
            print(
                f"==== fp8_weight_quant_format: {fp8_weight_quant_format} ===="
            )
            self.run_test_sonic_moe_ep_grad_accumulation(
                fp8_weight_quant_format
            )
        print("==== Test EP precision ====")
        self.run_test_ep_precision()
        print("==== Test BF16 wgrad ====")
        self.run_test_bf16_wgrad()
        self.run_test_z_bf16_recompute_z()
        for fp8_weight_quant_format in formats:
            print(
                f"==== Test FP8 config variants: {fp8_weight_quant_format} ===="
            )
            self.setUp()
            self.run_test_fp8_config_variants(fp8_weight_quant_format)


if __name__ == "__main__":
    unittest.main()
