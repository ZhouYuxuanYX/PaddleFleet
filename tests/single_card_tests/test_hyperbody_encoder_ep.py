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

"""HyperBody encoder-only EP (``hyperencoder_expert_model_parallel_size``),
the parts that need no distributed init.

* encoder EP / expert-data rank lists nest inside the decoder EP groups;
* config plumbing: the encoder view gets the encoder EP, the decoder view never
  sees the key;
* checks: checkpoint paths and the Paddle 2D bucket key;
* ZCC ``parallel_config`` records and compares the encoder EP size;
* the encoder expert RNG stream, the reshard group lookup and the per-group
  DeepEP buffer cache;
* EP=1 fusion path: switch, fallbacks, and numerics against the single-card
  path while the MoE input and output grad stay shared with a residual branch.
"""

import dataclasses
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import paddle
import paddle.nn.functional as F

from paddlefleet.trainer.utils import zero_cost_checkpoint as zcc
from paddlefleet.trainer.utils.reshard import common as reshard_common
from paddlefleet.transformer.moe import fused_a2a
from paddlefleet.transformer.moe.moe_layer import (
    MoELayer,
    _SingleCardFusionMap,
)
from paddlefleet.transformers.hyperbody import encoder_ep, modeling as hb

_DECODER = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 1,
    "num_attention_heads": 4,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
}
_ENCODER = {
    "vocab_size": 128,
    "hidden_size": 256,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "moe_intermediate_size": 128,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "hyperencoder_query_length": 16,
}


def _hb_config(enc_ep=None, dec_ep=8):
    from paddlefleet.transformers.hyperbody.configuration import (
        HyperBodyConfig,
    )

    enc = dict(_ENCODER)
    if enc_ep is not None:
        enc["hyperencoder_expert_model_parallel_size"] = enc_ep
    cfg = HyperBodyConfig(
        decoder_config=dict(_DECODER),
        encoder_config=enc,
        expert_model_parallel_size=dec_ep,
    )
    # ERNIEBot keeps the runtime switch flat on the config as well and copies
    # it down into encoder_config.
    cfg.hyperencoder_expert_model_parallel_size = enc_ep
    return cfg


class TestRankLists(unittest.TestCase):
    def test_nested_in_decoder_ep(self):
        dec = [list(range(0, 8)), list(range(8, 16))]
        ep, edp = encoder_ep.build_encoder_ep_rank_lists(dec, 4)
        self.assertEqual(
            ep, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        )
        self.assertEqual(edp[0], [0, 4, 8, 12])
        self.assertEqual(edp[3], [3, 7, 11, 15])
        # Every rank sits in exactly one EP and one expert-data group.
        for lists in (ep, edp):
            self.assertEqual(
                sorted(r for g in lists for r in g), list(range(16))
            )

    def test_ep1_and_equal(self):
        dec = [list(range(8))]
        ep, edp = encoder_ep.build_encoder_ep_rank_lists(dec, 1)
        self.assertEqual(ep, [[r] for r in range(8)])
        self.assertEqual(edp, [list(range(8))])
        ep, edp = encoder_ep.build_encoder_ep_rank_lists(dec, 8)
        self.assertEqual(ep, dec)
        self.assertEqual(edp, [[r] for r in range(8)])

    def test_must_divide(self):
        for bad in (3, 0, 16):
            with self.assertRaisesRegex(ValueError, "must divide"):
                encoder_ep.build_encoder_ep_rank_lists([list(range(8))], bad)


class TestConfigPlumbing(unittest.TestCase):
    def test_encoder_view_gets_encoder_ep(self):
        for enc_ep, want_ep in ((None, 8), (8, 8), (2, 2), (1, 1)):
            with self.subTest(enc_ep=enc_ep):
                cfg = _hb_config(enc_ep)
                view = hb._build_encoder_view(cfg, _DECODER["hidden_size"])
                self.assertEqual(view.expert_model_parallel_size, want_ep)
                self.assertEqual(
                    view.hyperencoder_expert_model_parallel_size, enc_ep
                )

    def test_decoder_view_drops_the_key(self):
        view = hb._build_decoder_view(_hb_config(1))
        self.assertFalse(
            hasattr(view, "hyperencoder_expert_model_parallel_size")
        )
        self.assertEqual(view.expert_model_parallel_size, 8)

    def test_int_coercion_and_validation(self):
        cfg = _hb_config("4")
        self.assertEqual(
            cfg.encoder_config.hyperencoder_expert_model_parallel_size, 4
        )
        for bad in (0, -2):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                _hb_config(bad)

    def test_allow_color_buckets_flag(self):
        self.assertFalse(
            _hb_config(2).encoder_config.hyperencoder_allow_color_buckets
        )


class TestInjectEncoderPgCollection(unittest.TestCase):
    def _specs(self):
        return [SimpleNamespace(extra_kwargs={}) for _ in range(3)]

    def test_only_encoder_specs_get_encoder_groups(self):
        sentinel = object()
        for enc_ep, want in (
            (2, sentinel),
            (1, sentinel),
            (None, None),
            (8, None),
        ):
            with (
                self.subTest(enc_ep=enc_ep),
                mock.patch.object(
                    encoder_ep,
                    "get_encoder_pg_collection",
                    return_value=sentinel,
                ) as get_pgs,
            ):
                cfg = _hb_config(enc_ep)
                view = hb._build_encoder_view(cfg, _DECODER["hidden_size"])
                specs = self._specs()
                hb._inject_encoder_pg_collection(cfg, view, specs)
                if want is None:
                    get_pgs.assert_not_called()
                    self.assertTrue(all(not s.extra_kwargs for s in specs))
                else:
                    get_pgs.assert_called_once_with(
                        enc_ep, allow_color_buckets=False
                    )
                    self.assertTrue(
                        all(
                            s.extra_kwargs["pg_collection"] is want
                            for s in specs
                        )
                    )


def _ckpt_cfg(**kw):
    base = {
        "enable_zero_cost_checkpoint": True,
        "save_checkpoint_format": "flex_checkpoint",
        "load_checkpoint_format": "flex_checkpoint",
        "load_sharded_model": False,
    }
    base.update(kw)
    return SimpleNamespace(**base)


class TestCheckpointCompat(unittest.TestCase):
    def test_reads_training_arguments_fields(self):
        from paddlefleet.trainer.training_args import TrainingArguments

        names = {f.name for f in dataclasses.fields(TrainingArguments)}
        for key in _ckpt_cfg().__dict__:
            self.assertIn(key, names)

    def test_flex_zcc_allowed(self):
        encoder_ep.check_encoder_ep_checkpoint_compat(_ckpt_cfg())

    def test_legacy_zcc_rejected(self):
        for kw in (
            {"save_checkpoint_format": "sharding_io"},
            {"save_checkpoint_format": None},
            {"load_checkpoint_format": "sharding_io"},
        ):
            with (
                self.subTest(**kw),
                self.assertRaisesRegex(
                    NotImplementedError, "enable_zero_cost_checkpoint"
                ),
            ):
                encoder_ep.check_encoder_ep_checkpoint_compat(_ckpt_cfg(**kw))

    def test_zcc_off_any_format(self):
        encoder_ep.check_encoder_ep_checkpoint_compat(
            _ckpt_cfg(
                enable_zero_cost_checkpoint=False, save_checkpoint_format=None
            )
        )

    def test_legacy_sharded_load_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "load_sharded_model"):
            encoder_ep.check_encoder_ep_checkpoint_compat(
                _ckpt_cfg(
                    enable_zero_cost_checkpoint=False,
                    load_sharded_model=True,
                    load_checkpoint_format="sharding_io",
                )
            )


class TestPaddleBucketKeyCheck(unittest.TestCase):
    def _shells(self):
        from paddle.distributed.fleet.meta_optimizers.mixed_sharding_optimizer import (
            MixedShardingOptimizer,
        )
        from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
            MuonShardingOptimizer,
        )

        return MuonShardingOptimizer, MixedShardingOptimizer

    def test_bucket_keyed_paddle_passes(self):
        muon, mixed = self._shells()
        with (
            mock.patch.object(
                muon,
                "_param_bucket_key",
                classmethod(lambda c, p, g: None),
                create=True,
            ),
            mock.patch.object(
                mixed,
                "_param_bucket_key",
                classmethod(lambda c, p, g: None),
                create=True,
            ),
        ):
            encoder_ep.check_paddle_bucket_keys()

    def test_color_keyed_paddle_rejected(self):
        muon, _ = self._shells()
        saved = muon.__dict__.get("_param_bucket_key")
        if saved is not None:
            delattr(muon, "_param_bucket_key")
        try:
            with self.assertRaisesRegex(NotImplementedError, "79913"):
                encoder_ep.check_paddle_bucket_keys()
        finally:
            if saved is not None:
                muon._param_bucket_key = saved


class TestZccParallelConfig(unittest.TestCase):
    def setUp(self):
        self._saved = dict(encoder_ep._ENCODER_PGS)
        encoder_ep._ENCODER_PGS.clear()
        self.v = object.__new__(zcc.DistInfoCollectorValidator)
        self.v.args = SimpleNamespace(use_hybrid_parallel=False)

    def tearDown(self):
        encoder_ep._ENCODER_PGS.clear()
        encoder_ep._ENCODER_PGS.update(self._saved)

    def _set_encoder_ep(self, n):
        encoder_ep._ENCODER_PGS[n] = SimpleNamespace(
            ep=SimpleNamespace(nranks=n)
        )

    def _same(self, old):
        self.v._load_model_meta_impl = lambda _: {"parallel_config": old}
        return self.v.check_same_strategy("ckpt")

    def test_degree(self):
        self.assertIsNone(encoder_ep.get_encoder_ep_degree())
        self._set_encoder_ep(4)
        self.assertEqual(encoder_ep.get_encoder_ep_degree(), 4)

    def test_no_encoder_groups_keeps_old_config(self):
        self.assertNotIn(
            "encoder_ep_degree", self.v._get_distributed_strategy()
        )

    def test_records_encoder_ep(self):
        self._set_encoder_ep(8)
        self.assertEqual(
            self.v._get_distributed_strategy()["encoder_ep_degree"], 8
        )

    def test_equal_to_ep_not_recorded(self):
        self._set_encoder_ep(1)  # ep_degree is 1 without hybrid parallel
        self.assertNotIn(
            "encoder_ep_degree", self.v._get_distributed_strategy()
        )

    def test_check_same_strategy(self):
        base = self.v._get_distributed_strategy()
        self.assertEqual(self._same(dict(base)), (True, None))
        self._set_encoder_ep(8)
        # Old checkpoint without the key: encoder used the global EP (1).
        ok, msg = self._same(dict(base))
        self.assertFalse(ok)
        self.assertIn("encoder_ep_degree", msg)
        self.assertEqual(
            self._same(dict(base, encoder_ep_degree=8)), (True, None)
        )
        self.assertFalse(self._same(dict(base, encoder_ep_degree=4))[0])
        encoder_ep._ENCODER_PGS.clear()
        self.assertFalse(self._same(dict(base, encoder_ep_degree=8))[0])


class TestReshardGroup(unittest.TestCase):
    def setUp(self):
        self._saved = dict(encoder_ep._ENCODER_PGS)
        encoder_ep._ENCODER_PGS.clear()

    def tearDown(self):
        encoder_ep._ENCODER_PGS.clear()
        encoder_ep._ENCODER_PGS.update(self._saved)

    def test_whitelisted_groups_only(self):
        default = SimpleNamespace(ranks=[0, 1], id=1)
        moe_sd = SimpleNamespace(ranks=[0], id=2)
        hcg = SimpleNamespace(
            get_sharding_parallel_group=lambda: default,
            get_moe_sharding_parallel_group=lambda: moe_sd,
        )

        def colored(group):
            return SimpleNamespace(
                color={"color": "moe_expert", "group": group}
            )

        get = reshard_common.get_param_sharding_group
        self.assertIs(get(SimpleNamespace(), hcg), default)
        self.assertIs(get(colored(default), hcg), default)
        self.assertIs(get(colored(moe_sd), hcg), moe_sd)
        enc_group = SimpleNamespace(ranks=[0, 4], id=7)
        # Not built through encoder_ep: rejected.
        with self.assertRaisesRegex(ValueError, "unsupported group"):
            get(colored(enc_group), hcg)
        encoder_ep._ENCODER_PGS[2] = SimpleNamespace(expt_dp=enc_group)
        self.assertIs(get(colored(enc_group), hcg), enc_group)
        for bad in (None, SimpleNamespace(ranks=[0, 4], id=7)):
            with self.assertRaisesRegex(ValueError, "unsupported group"):
                get(colored(bad), hcg)


class _FakeConfig:
    def get_nvl_buffer_size_hint(self, hidden_bytes, world_size):
        return hidden_bytes

    def get_rdma_buffer_size_hint(self, hidden_bytes, world_size):
        return 0


class _FakeBuffer:
    num_sms = 20
    created = 0

    def __init__(self, group, num_nvl_bytes, num_rdma_bytes, num_qps_per_rank):
        type(self).created += 1
        self.group = group
        self.num_nvl_bytes = num_nvl_bytes
        self.num_rdma_bytes = num_rdma_bytes

    @staticmethod
    def get_dispatch_config(world_size):
        return _FakeConfig()

    @staticmethod
    def get_combine_config(world_size):
        return _FakeConfig()


class TestDeepEPBufferPerGroup(unittest.TestCase):
    def setUp(self):
        self._saved = (dict(fused_a2a._buffers), fused_a2a._buffer)
        fused_a2a._buffers.clear()
        _FakeBuffer.created = 0

    def tearDown(self):
        fused_a2a._buffers.clear()
        fused_a2a._buffers.update(self._saved[0])
        fused_a2a._buffer = self._saved[1]

    def test_one_buffer_per_group(self):
        enc = SimpleNamespace(id=11, world_size=2)
        dec = SimpleNamespace(id=12, world_size=8)
        with mock.patch.object(
            fused_a2a,
            "deep_ep",
            SimpleNamespace(Buffer=_FakeBuffer),
            create=True,
        ):
            b_enc = fused_a2a.get_buffer(enc, 100)
            b_dec = fused_a2a.get_buffer(dec, 100)
            # Switching back and forth reuses the buffers.
            self.assertIs(fused_a2a.get_buffer(enc, 100), b_enc)
            self.assertIs(fused_a2a.get_buffer(dec, 50), b_dec)
            self.assertIs(fused_a2a._buffer, b_dec)
            self.assertEqual(_FakeBuffer.created, 2)
            # A larger request on one group regrows only that group's buffer.
            b_enc2 = fused_a2a.get_buffer(enc, 200)
            self.assertIsNot(b_enc2, b_enc)
            self.assertIs(fused_a2a.get_buffer(dec, 100), b_dec)
            self.assertEqual(_FakeBuffer.created, 3)


@unittest.skipUnless(paddle.is_compiled_with_cuda(), "needs CUDA")
class TestEncoderExpertRng(unittest.TestCase):
    def test_stream_seeded_by_encoder_ep_rank(self):
        from paddlefleet.tensor_parallel import random as fl_random

        tracker = fl_random.get_cuda_rng_tracker()
        name = fl_random._ENCODER_EXPERT_PARALLEL_RNG_TRACKER_NAME
        saved_states = dict(tracker.get_states())
        saved_seed = fl_random._BASE_SEED

        def draw():
            with tracker.fork(name):
                return paddle.randn([16]).numpy()

        try:
            fl_random._BASE_SEED = 1234
            out = {}
            for ep_rank in (0, 0, 1):
                tracker.reset()
                self.assertEqual(
                    fl_random.add_encoder_expert_parallel_rng(ep_rank), name
                )
                # Idempotent: a second call keeps the stream where it is.
                fl_random.add_encoder_expert_parallel_rng(ep_rank)
                out.setdefault(ep_rank, []).append(draw())
            np.testing.assert_array_equal(out[0][0], out[0][1])
            self.assertFalse(np.array_equal(out[0][0], out[1][0]))
        finally:
            tracker.reset()
            tracker.set_states(saved_states)
            fl_random._BASE_SEED = saved_seed


class TestEp1FusionSwitch(unittest.TestCase):
    def _layer(self, enc_ep, **attrs):
        base = {
            "config": SimpleNamespace(
                hyperencoder_expert_model_parallel_size=enc_ep
            ),
            "using_sonic_moe": False,
            "use_accuracy_compatible": False,
            "fp8": None,
            "use_w4a8": False,
            "num_experts": 64,
        }
        base.update(attrs)
        return SimpleNamespace(**base)

    def test_only_explicit_encoder_ep1(self):
        on = MoELayer._ep1_fusion_enabled
        self.assertTrue(on(self._layer(1)))
        for enc_ep in (None, 2, 8):
            self.assertFalse(on(self._layer(enc_ep)))
        self.assertFalse(on(SimpleNamespace(config=SimpleNamespace())))

    def test_skip_reasons(self):
        why = MoELayer._ep1_fusion_skip_reason
        idx = paddle.zeros([2, 2], dtype="int64")
        self.assertIsNone(why(self._layer(1), idx, None))
        # No expert-count cap here: moe_permute rejects too many experts itself.
        self.assertIsNone(why(self._layer(1, num_experts=256), idx, None))
        cases = [
            ({}, idx, ("p", "i"), "bit-exact"),
            ({"using_sonic_moe": True}, idx, None, "sonic"),
            ({"use_accuracy_compatible": True}, idx, None, "accuracy"),
            ({"fp8": "e4m3"}, idx, None, "fp8"),
            ({"use_w4a8": True}, idx, None, "w4a8"),
            ({}, None, None, "no topk_indices"),
        ]
        for attrs, topk, pre, needle in cases:
            with self.subTest(needle=needle):
                self.assertIn(needle, why(self._layer(1, **attrs), topk, pre))

    def test_fusion_map(self):
        layer = SimpleNamespace(experts="e", num_experts=4)
        m = _SingleCardFusionMap(layer, [1, 2, 3, 4])
        self.assertFalse(m.owns_dispatch_buffers)
        self.assertEqual(
            m.token_dispatcher._comm_manager.tokens_per_expert, [1, 2, 3, 4]
        )
        self.assertEqual(m.experts, "e")
        # The MoE layer itself keeps the default (EP>1: node owns its buffers).
        self.assertFalse(hasattr(MoELayer, "owns_dispatch_buffers"))


def _sm90_plus():
    if not paddle.is_compiled_with_cuda():
        return False
    try:
        return paddle.device.cuda.get_device_capability()[0] >= 9
    except Exception:
        return False


@unittest.skipUnless(_sm90_plus(), "fusion node needs a SM90+ GPU")
class TestEp1FusionForward(unittest.TestCase):
    """A real single-card MoELayer: the key picks the fusion branch in forward
    and the result matches the single-card grouped-GEMM branch."""

    def _layer(self, key):
        from paddlefleet.models.gpt.gpt_layer_specs import (
            get_gpt_layer_local_spec,
        )
        from paddlefleet.process_groups_config import ProcessGroupCollection
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        cfg = TransformerConfig(
            hidden_size=256,
            num_attention_heads=4,
            n_routed_experts=16,
            num_experts_per_tok=4,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
            sequence_parallel=False,
            bf16=True,
            params_dtype=paddle.bfloat16,
            moe_intermediate_size=128,
            gated_linear_unit=True,
            n_shared_experts=1,
            hidden_act=F.silu,
            moe_expert_fusion=True,
            moe_token_dispatcher_type="deepep",
            bias_activation_fusion=True,
            moe_deep_gemm=False,
        )
        cfg.hyperencoder_expert_model_parallel_size = key
        spec = get_gpt_layer_local_spec(cfg, num_experts=16)
        return MoELayer(
            cfg,
            spec.sublayers_spec.mlp.extra_kwargs["sublayers"],
            ProcessGroupCollection.use_mpu_process_groups(),
        )

    def test_forward_picks_fusion_and_matches(self):
        from paddlefleet.tensor_parallel.random import (
            model_parallel_cuda_manual_seed,
        )

        model_parallel_cuda_manual_seed(1)
        ref = self._layer(None)
        fused = self._layer(1)
        fused.set_state_dict(ref.state_dict())
        self.assertEqual(ref.expert_model_parallel_size, 1)
        x0 = paddle.randn([2, 64, 256]).astype("bfloat16")
        res = []
        for layer in (ref, fused):
            x = x0.detach()
            x.stop_gradient = False
            with mock.patch.object(
                MoELayer,
                "_forward_single_card_fusion",
                autospec=True,
                side_effect=MoELayer._forward_single_card_fusion,
            ) as spy:
                out = layer(x)[0]
            (out.astype("float32") + x.astype("float32") * 0.5).sum().backward()
            res.append(
                (
                    out.astype("float32").numpy(),
                    x.grad.astype("float32").numpy(),
                    spy.call_count,
                )
            )
        self.assertEqual((res[0][2], res[1][2]), (0, 1))
        for i, name in ((0, "output"), (1, "x_grad")):
            scale = np.abs(res[0][i]).max() + 1e-12
            rel = np.abs(res[0][i] - res[1][i]).max() / scale
            self.assertLess(rel, 3e-2, f"{name}: rel diff {rel:.3e}")


@unittest.skipUnless(_sm90_plus(), "fusion node needs a SM90+ GPU")
class TestEp1FusionNumerics(unittest.TestCase):
    """EP=1 fusion node vs the single-card path, on the same weights/router.

    The routed output feeds ``out + 0.5 * x``: the residual reads the MoE input
    after the experts ran and shares their output grad, as the router / shared
    experts do in the real layer, so a node that freed either would break it.
    """

    N, H, I, E, K = 2048, 256, 128, 16, 4

    def _layer(self):
        from paddlefleet.tensor_parallel.random import (
            model_parallel_cuda_manual_seed,
        )
        from paddlefleet.transformer.moe.moe_expert import GroupedMLPExpert
        from paddlefleet.transformer.transformer_config import (
            TransformerConfig,
        )

        model_parallel_cuda_manual_seed(1234)
        cfg = TransformerConfig(
            hidden_size=self.H,
            moe_intermediate_size=self.I,
            gated_linear_unit=True,
            hidden_act=F.silu,
        )
        cfg.hyperencoder_expert_model_parallel_size = 1
        experts = GroupedMLPExpert(
            num_local_experts=self.E, config=cfg, moe_deep_gemm=False
        )
        experts.activation_recompute = False
        with paddle.no_grad():
            for w in (experts.weight1, experts.weight2):
                w.set_value((paddle.randn(w.shape) * 0.02).astype("bfloat16"))
        layer = MoELayer.__new__(MoELayer)
        paddle.nn.Layer.__init__(layer)
        attrs = {
            "grouped_gemm_experts": experts,
            "config": cfg,
            "num_experts": self.E,
            "num_experts_per_tok": self.K,
            "experts": None,
            "layer_number": 0,
            "using_sonic_moe": False,
            "use_accuracy_compatible": False,
            "fp8": None,
            "use_w4a8": False,
            "fp8_wgrad": False,
            "moe_deep_gemm": False,
            "moe_expert_fusion": True,
            "recompute_moe_gate_up": False,
            "recompute_moe_premute": False,
            "use_auto_subbatch": False,
            "auto_subbatch_mode": None,
            "moe_subbatch_token_num_after_dispatch": None,
            "moe_subbatch_diag": False,
            "use_ue8m0": False,
            "defer_expert_up_gate_dw": False,
            "defer_expert_down_dw": False,
            "w4a8_route_factor_post_w2": None,
            "_activation_type": "swiglu",
        }
        for k, v in attrs.items():
            setattr(layer, k, v)
        return paddle.amp.decorate(layer, level="O2", dtype="bfloat16"), experts

    def _run(self, fused):
        paddle.seed(2026)
        layer, experts = self._layer()
        x = paddle.randn([self.N, self.H]).astype("bfloat16")
        x.stop_gradient = False
        logits = paddle.randn([self.N, self.E], "float32")
        logits.stop_gradient = False
        gout = paddle.randn([self.N, self.H]).astype("bfloat16")
        gates = F.softmax(logits, -1)
        top_gate, top_idx = paddle.topk(gates, self.K, axis=-1)
        top_gate = top_gate / top_gate.sum(-1, keepdim=True)
        probs = paddle.zeros_like(gates).put_along_axis(
            top_idx, top_gate, axis=1
        )
        mask = (
            paddle.zeros_like(gates)
            .put_along_axis(top_idx, paddle.ones_like(top_gate), axis=1)
            .astype("bool")
        )
        if fused:
            for w in (experts.weight1, experts.weight2):
                w.main_grad = None
            out = layer._forward_single_card_fusion(x, mask, top_idx, top_gate)
            self.assertTrue(
                x._is_initialized(), "fusion node freed the MoE input"
            )
        else:
            out = layer._forward_single_card_grouped_gemm_moe(
                x, mask, probs, top_idx, top_gate
            )
        out = out + x * 0.5
        paddle.autograd.backward([out], [gout])

        def wgrad(w):
            g = getattr(w, "main_grad", None)
            return (g if g is not None else w.grad).astype("float32").numpy()

        return {
            "out": out.astype("float32").numpy(),
            "x_grad": x.grad.astype("float32").numpy(),
            "logits_grad": logits.grad.numpy(),
            "dW1": wgrad(experts.weight1),
            "dW2": wgrad(experts.weight2),
        }

    def test_matches_single_card_path(self):
        ref, got = self._run(fused=False), self._run(fused=True)
        for k in ref:
            with self.subTest(tensor=k):
                scale = np.abs(ref[k]).max() + 1e-12
                rel = np.abs(ref[k] - got[k]).max() / scale
                self.assertLess(rel, 2e-2, f"{k}: rel diff {rel:.3e}")


if __name__ == "__main__":
    unittest.main()
