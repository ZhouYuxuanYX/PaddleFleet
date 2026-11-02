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
#
# Scope: standard ``SelfAttention`` fuses the checkpoint q/k/v projections
# (each transposed) into the model ``qkv_proj`` via the ``fused_qkv`` macro,
# adds an optional fused ``axis=0`` bias, then recurses every other direct
# child generically (``o_proj`` -> Linear ``^T``). It owns no direct parameter
# or buffer, so there is no own-parameter fallback. This test pins that
# coverage contract and the independent inverse split, and runs the qkv
# statements on real non-square tensors to check the round trip. Head counts
# are read per layer (SWA-aware) off ``self``.
import os
import sys

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
    ),
)

import unittest

import paddle
from paddle.distributed.flex_checkpoint.aoa.generation import AOAContext


def _ctx(*, checkpoint_name_prefix="hf", model_name_prefix="model"):
    return AOAContext(
        config=None,
        checkpoint_name_prefix=checkpoint_name_prefix,
        pp_to_single_mapping={},
        checkpoint_name_mapping={},
        model_name_prefix=model_name_prefix,
    )


def _make_linear_leaf(bias=False):
    from paddlefleet.tensor_parallel.layers import Linear

    class _LinearLeaf(paddle.nn.Layer):
        gen_aoa_statements = Linear.gen_aoa_statements
        gen_inv_aoa_statements = Linear.gen_inv_aoa_statements

        def __init__(self):
            super().__init__()
            self.weight = self.create_parameter(shape=[2, 2])
            self.bias = self.create_parameter(shape=[2]) if bias else None

    return _LinearLeaf


class _AttnConfig:
    gpt_model_use_experimental_version = False


def _make_attn_leaf(qkv_bias=False):
    """Binds the SelfAttention AOA overrides onto a controllable stand-in.

    The real ``SelfAttention`` needs a live TP process group, so we bind the
    two entry points, the qkv-head hooks they dispatch into, and the fused
    qkv-bias helpers those hooks delegate their bias tail to. The stand-in
    carries a ``qkv_proj`` leaf (whose ``.bias`` gates the fused-bias
    statement) and an ``o_proj`` Linear leaf that is recursed generically.
    ``gated_attention = False`` selects the standard qkv branch (the gated
    layout has its own component test).
    """
    from paddlefleet.transformer.attention import SelfAttention

    linear_with_bias = _make_linear_leaf(bias=qkv_bias)
    linear_plain = _make_linear_leaf(bias=False)

    class _AttnLeaf(paddle.nn.Layer):
        gen_aoa_statements = SelfAttention.gen_aoa_statements
        gen_inv_aoa_statements = SelfAttention.gen_inv_aoa_statements
        _gen_qkv_head_aoa_statements = (
            SelfAttention._gen_qkv_head_aoa_statements
        )
        _gen_inv_qkv_head_aoa_statements = (
            SelfAttention._gen_inv_qkv_head_aoa_statements
        )
        _gen_qkv_bias_aoa_statements = (
            SelfAttention._gen_qkv_bias_aoa_statements
        )
        _gen_inv_qkv_bias_aoa_statements = (
            SelfAttention._gen_inv_qkv_bias_aoa_statements
        )
        gated_attention = False
        num_attention_heads = 8
        num_key_value_heads = 2
        head_dim = 4
        v_head_dim = 4
        config = _AttnConfig()

        def __init__(self):
            super().__init__()
            self.qkv_proj = linear_with_bias()
            self.o_proj = linear_plain()

    return _AttnLeaf


_PFX = "model.layers.0.self_attn."
_HF = "hf.layers.0.self_attn."
_MD = "model.layers.0.self_attn."


class TestSelfAttentionClassContract(unittest.TestCase):
    def test_override_present(self):
        from paddlefleet.transformer.attention import SelfAttention

        for name in (
            "gen_aoa_statements",
            "gen_inv_aoa_statements",
            "_gen_qkv_head_aoa_statements",
            "_gen_inv_qkv_head_aoa_statements",
            "_gen_qkv_bias_aoa_statements",
            "_gen_inv_qkv_bias_aoa_statements",
        ):
            self.assertIn(name, SelfAttention.__dict__)


class TestSelfAttentionForward(unittest.TestCase):
    def test_fused_qkv_weight_and_child_recursion(self):
        model = _make_attn_leaf(qkv_bias=False)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        self.assertIn(
            f"{_HF}q_proj.weight^T, {_HF}k_proj.weight^T, "
            f"{_HF}v_proj.weight^T "
            f"-> {_MD}qkv_proj.weight, fused_qkv, "
            f"num_heads=8, num_key_value_groups=2",
            stmts,
        )
        self.assertIn(f"{_HF}o_proj.weight^T -> {_MD}o_proj.weight", stmts)
        # No own parameter/buffer, so exactly: fused qkv weight + o_proj weight.
        self.assertEqual(len(stmts), 2)

    def test_fused_qkv_bias_present_when_qkv_has_bias(self):
        model = _make_attn_leaf(qkv_bias=True)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        self.assertIn(
            f"{_HF}q_proj.bias, {_HF}k_proj.bias, {_HF}v_proj.bias "
            f"-> {_MD}qkv_proj.bias, fused_qkv, "
            f"num_heads=8, num_key_value_groups=2, axis=0",
            stmts,
        )

    def test_no_qkv_bias_statement_when_absent(self):
        model = _make_attn_leaf(qkv_bias=False)()
        stmts = model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
        self.assertFalse(any("qkv_proj.bias" in s for s in stmts))


class TestSelfAttentionInverse(unittest.TestCase):
    def test_fused_qkv_split_and_child_recursion(self):
        model = _make_attn_leaf(qkv_bias=True)()
        stmts = model.gen_inv_aoa_statements(
            _ctx(), structured_name_prefix=_PFX
        )
        # inverse transposes the fused weight, then splits on axis 0
        self.assertIn(
            f"{_MD}qkv_proj.weight^T -> {_HF}q_proj.weight, "
            f"{_HF}k_proj.weight, {_HF}v_proj.weight, "
            f"fused_qkv, num_heads=8, num_key_value_groups=2, axis=0",
            stmts,
        )
        self.assertIn(
            f"{_MD}qkv_proj.bias -> {_HF}q_proj.bias, "
            f"{_HF}k_proj.bias, {_HF}v_proj.bias, "
            f"fused_qkv, num_heads=8, num_key_value_groups=2, axis=0",
            stmts,
        )
        self.assertIn(f"{_MD}o_proj.weight^T -> {_HF}o_proj.weight", stmts)


class TestSelfAttentionQKVRoundTrip(unittest.TestCase):
    """Runs the generated qkv statements on real tensors.

    The fused weight is ``[hidden, fused_out]`` while each checkpoint piece is
    ``[out, hidden]``, so a missing transpose on either side only shows up as
    wrong values (or not at all on a square attention). Shapes here are all
    non-square: hidden=6, q_out=32, kv_out=8, fused_out=48.
    """

    HIDDEN = 6

    def setUp(self):
        self._device = paddle.get_device()
        paddle.set_device("cpu")
        self._env = {}
        for key, value in {
            "PADDLE_TRAINER_ID": "0",
            "PADDLE_TRAINERS_NUM": "1",
            "PADDLE_TRAINER_ENDPOINTS": "127.0.0.1:60001",
            "PADDLE_CURRENT_ENDPOINT": "127.0.0.1:60001",
        }.items():
            self._env[key] = os.environ.get(key)
            os.environ[key] = value

    def tearDown(self):
        paddle.set_device(self._device)
        for key, value in self._env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @staticmethod
    def _load(src, dst_shapes, statements):
        import tempfile

        import paddle.distributed as dist
        from paddle.distributed.flex_checkpoint.dcp.sharded_weight import (
            build_sharded_state_dict,
        )

        with tempfile.TemporaryDirectory() as path:
            dist.save_state_dict(
                build_sharded_state_dict(
                    {n: paddle.to_tensor(a) for n, a in src.items()}
                ),
                path,
            )
            sharded = build_sharded_state_dict(
                {
                    n: paddle.zeros(s, dtype="float32")
                    for n, s in dst_shapes.items()
                }
            )
            dist.load_state_dict(
                sharded, path, aoa_config={"aoa_statements": statements}
            )
            return {n: s.local_tensor.numpy() for n, s in sharded.items()}

    def test_forward_then_inverse_recovers_checkpoint(self):
        import numpy as np

        model = _make_attn_leaf(qkv_bias=False)()
        nh, nkv, d = (
            model.num_attention_heads,
            model.num_key_value_heads,
            model.head_dim,
        )
        rng = np.random.default_rng(0)
        ckpt = {
            f"{_HF}{p}_proj.weight": rng.standard_normal(
                (n * d, self.HIDDEN)
            ).astype("float32")
            for p, n in (("q", nh), ("k", nkv), ("v", nkv))
        }
        fused = f"{_MD}qkv_proj.weight"

        def _qkv_only(stmts):
            return [s for s in stmts if "qkv_proj" in s]

        fwd = self._load(
            ckpt,
            {fused: [self.HIDDEN, (nh + 2 * nkv) * d]},
            _qkv_only(
                model.gen_aoa_statements(_ctx(), structured_name_prefix=_PFX)
            ),
        )
        # Fused layout: per KV group, its Q heads then K then V on the out axis.
        q, k, v = ckpt.values()
        group = nh // nkv
        blocks = []
        for g in range(nkv):
            blocks.append(q[g * group * d : (g + 1) * group * d])
            blocks += [k[g * d : (g + 1) * d], v[g * d : (g + 1) * d]]
        np.testing.assert_array_equal(fwd[fused], np.concatenate(blocks).T)

        inv = self._load(
            fwd,
            {n: list(a.shape) for n, a in ckpt.items()},
            _qkv_only(
                model.gen_inv_aoa_statements(
                    _ctx(), structured_name_prefix=_PFX
                )
            ),
        )
        for name, want in ckpt.items():
            np.testing.assert_array_equal(inv[name], want)


if __name__ == "__main__":
    unittest.main()
