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

"""``d_sink`` of the cuDNN DSA backward on the absorbed-MQA path.

``mqa_sparse_attn`` (d_qk=576, d_v=512) takes ``d_sink`` straight from
``csa_sparse_attn_bwd_cudnn``, fed the KV-only forward ``lse`` and the real
sink. Checked against an fp64 autograd ground-truth (softmax over the gathered
columns plus the sink logit, differentiated w.r.t. the sink), with a distinct
sink per head so a head mix-up cannot cancel out, in both the
non-deterministic and the deterministic kernel, at a size where the reduction
over queries spans many tiles.
"""

import contextlib
import unittest
import unittest.mock

import numpy as np
import paddle

from paddlefleet.cudnn_ops.attn.csa_sparse_attn_bwd_cudnn import (
    csa_sparse_attn_bwd_cudnn,
)
from paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
    sparse_attn_fwd,
)
from paddlefleet.fusions.csa_sparse_attn_utils import local_to_global_flat
from paddlefleet.fusions.mqa_sparse_attn import (
    _DSA_HEADS,
    _NEG_SINK,
    mqa_sparse_attn,
)

from .hybrid_mla_utils import _GPU
from .test_block_sparse_dsa_gradcheck import (
    DK,
    DV,
    SM,
    _causal_indices,
    _rand,
    _relerr,
)

# bf16 inputs; measured 1.7e-3 relative L2 against the fp64 ground-truth.
TOL = 1e-2
S, L = 768, 384


@contextlib.contextmanager
def _deterministic(on):
    flag = "FLAGS_cudnn_deterministic"
    old = paddle.get_flags([flag])[flag]
    paddle.set_flags({flag: on})
    try:
        yield
    finally:
        paddle.set_flags({flag: old})


def _case(h, seed, n_dead=0, s=S, width=L):
    """bf16 ``q [1,s,h,DK]``, ``kv [1,s,DK]``, ``dO [1,s,h,DV]``; fp32 sink
    ``[h]`` with distinct values per head (the last ``n_dead`` set to the
    ``-1e30`` padded-head value); causal ``ti [1,s,width]`` with -1 holes."""
    q = paddle.to_tensor(_rand([1, s, h, DK], 0.3, seed)).cast("bfloat16")
    kv = paddle.to_tensor(_rand([1, s, DK], 0.3, seed + 1)).cast("bfloat16")
    dO = paddle.to_tensor(_rand([1, s, h, DV], 1.0, seed + 2)).cast("bfloat16")
    sink = np.random.RandomState(seed + 3).normal(0.0, 2.0, h).astype("float32")
    if n_dead:
        sink[-n_dead:] = _NEG_SINK
    ti = _causal_indices(s, width, seed=seed + 4)
    return q, kv, dO, sink, ti


def _ground_truth_dsink(q, kv, dO, sink, ti):
    """fp64 autograd d_sink on the same bf16-rounded inputs."""
    s, h = q.shape[1], q.shape[2]
    width = ti.shape[-1]
    q64 = q[0].cast("float64")
    kv64 = kv[0].cast("float64")
    s64 = paddle.to_tensor(sink).cast("float64")
    s64.stop_gradient = False
    idx = ti[0]
    g = paddle.gather(
        kv64, paddle.to_tensor(np.clip(idx, 0, None).reshape([-1]))
    ).reshape([s, width, DK])
    logit = paddle.einsum("shd,sld->shl", q64, g) * SM
    valid = paddle.to_tensor(idx >= 0).unsqueeze(1)
    logit = paddle.where(valid, logit, paddle.full_like(logit, -float("inf")))
    full = paddle.concat(
        [logit, s64.reshape([1, h, 1]).expand([s, h, 1])], axis=-1
    )
    p = paddle.nn.functional.softmax(full, axis=-1)[..., :width]
    o = paddle.einsum("shl,sld->shd", p, g[..., :DV])
    loss = (o * dO[0].cast("float64")).sum()
    (gs,) = paddle.grad(loss, [s64])
    return gs.numpy()


def _op_dsink(q, kv, dO, sink, ti):
    """d_sink_flat exactly as ``mqa_sparse_attn``'s cuDNN backward gets it."""
    s, h = q.shape[1], q.shape[2]
    sink_t = paddle.to_tensor(sink)
    ti_t = paddle.to_tensor(ti)
    out, lse, _ = sparse_attn_fwd(q, kv, sink_t, ti_t, sm_scale=SM, d_v=DV)
    _, _, d_sink = csa_sparse_attn_bwd_cudnn(
        q.reshape([s, h, DK]),
        kv.reshape([s, DK]),
        out.reshape([s, h, DV]),
        dO.reshape([s, h, DV]),
        lse.reshape([s, h]),
        sink_t,
        local_to_global_flat(ti_t, s),
        softmax_scale=SM,
        topk_length=None,
    )
    return d_sink.cast("float32").numpy()


def _layer_dsink(q, kv, dO, sink, ti):
    """d_sink returned by ``mqa_sparse_attn`` for a learnable ``[h]`` sink."""
    sink_t = paddle.to_tensor(sink)
    sink_t.stop_gradient = False
    out = mqa_sparse_attn(q, kv, paddle.to_tensor(ti), float(SM), DV, sink_t)
    (
        out.cast("float32") * dO.reshape(out.shape).cast("float32")
    ).sum().backward()
    return sink_t.grad.cast("float32").numpy()


@_GPU
class TestKernelDSink(unittest.TestCase):
    """``d_sink_flat`` from ``csa_sparse_attn_bwd_cudnn`` (64 heads, 576/512)."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("gpu")
        cls.case = _case(_DSA_HEADS, seed=11)
        cls.gt = _ground_truth_dsink(*cls.case)

    def _check(self, got, gt):
        self.assertLess(_relerr(got, gt), TOL)
        # every head, not only the L2 aggregate
        self.assertLess(
            float(np.abs(got - gt).max()) / float(np.abs(gt).max()), 2 * TOL
        )

    def test_matches_ground_truth(self):
        for det in (False, True):
            with self.subTest(deterministic=det), _deterministic(det):
                self._check(_op_dsink(*self.case), self.gt)

    def test_padded_heads_are_zero(self):
        # The layer pads h < 64 with -1e30 sinks; those heads carry no mass.
        q, kv, dO, sink, ti = _case(_DSA_HEADS, seed=21, n_dead=24)
        gt = _ground_truth_dsink(q, kv, dO, sink, ti)
        for det in (False, True):
            with self.subTest(deterministic=det), _deterministic(det):
                got = _op_dsink(q, kv, dO, sink, ti)
                self.assertTrue(np.all(got[-24:] == 0.0))
                self._check(got[:-24], gt[:-24])


@_GPU
class TestLayerDSink(unittest.TestCase):
    """End to end through ``mqa_sparse_attn`` with h < 64 (padded inside)."""

    @classmethod
    def setUpClass(cls):
        paddle.set_device("gpu")

    def test_layer_matches_ground_truth(self):
        q, kv, dO, sink, ti = _case(20, seed=41)
        gt = _ground_truth_dsink(q, kv, dO, sink, ti)
        for det in (False, True):
            with self.subTest(deterministic=det), _deterministic(det):
                got = _layer_dsink(q, kv, dO, sink, ti)
                self.assertEqual(got.shape, (20,))
                self.assertLess(_relerr(got, gt), TOL)


class TestCudnnBackwardPlumbing(unittest.TestCase):
    """The ``"cudnn"`` backward wiring with both kernels stubbed (any device):
    the KV-only ``lse`` and the padded sink reach ``csa_sparse_attn_bwd_cudnn``
    with ``topk_length=None``, and ``d_sink`` comes back sliced to the real
    heads in the sink's dtype."""

    def test_dsink_and_lse_reach_the_kernel(self):
        s, h, hp = 3, 2, _DSA_HEADS
        q = paddle.zeros([1, s, h, DK], dtype="bfloat16")
        kv = paddle.zeros([1, 4, DK], dtype="bfloat16")
        sink = paddle.to_tensor([0.5, -0.5], dtype="bfloat16")
        for t in (q, kv, sink):
            t.stop_gradient = False
        ti = paddle.to_tensor(np.tile(np.int32([0, 1, -1, 3]), (1, s, 1)))
        lse = paddle.full([1, s, hp], 3.0, dtype="float32")
        out = paddle.zeros([1, s, hp, DV], dtype="bfloat16")
        grads = (
            paddle.zeros([s, hp, DK], dtype="bfloat16"),
            paddle.zeros([4, DK], dtype="bfloat16"),
            paddle.arange(hp, dtype="float32"),
        )
        bwd = unittest.mock.Mock(return_value=grads)
        with (
            unittest.mock.patch(
                "paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn.sparse_attn_fwd",
                return_value=(out, lse, None),
            ),
            unittest.mock.patch(
                "paddlefleet.cudnn_ops.csa_sparse_attn_bwd_cudnn", bwd
            ),
        ):
            y = mqa_sparse_attn(q, kv, ti, 0.25, DV, attn_sink=sink)
            y.cast("float32").sum().backward()
        args, kw = bwd.call_args
        lse_k, sink_k = args[4], args[5]
        self.assertTrue(bool((lse_k == 3.0).all()))  # KV-only lse, untouched
        np.testing.assert_array_equal(sink_k.numpy()[:h], [0.5, -0.5])
        self.assertTrue(bool((sink_k[h:] == _NEG_SINK).all()))  # padded heads
        self.assertEqual((kw["softmax_scale"], kw["topk_length"]), (0.25, None))
        self.assertEqual(sink.grad.dtype, paddle.bfloat16)
        np.testing.assert_array_equal(sink.grad.cast("float32").numpy(), [0, 1])


if __name__ == "__main__":
    unittest.main()
