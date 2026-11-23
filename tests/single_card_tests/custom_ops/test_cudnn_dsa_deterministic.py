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

"""Deterministic mode of the cuDNN DSA sparse attention backward.

* ``sort_topk_ascending_pad_last``: ascending rows, ``-1`` at the tail.
* ``cudnn_deterministic_kwargs``: each branch, with stand-in wrappers; the flag
  is read per call, so toggling it at runtime takes effect. The SM90 relay
  probe falls back size by size and warns when no size can launch.
* ``cudnn_frontend_version`` parses the vendored version; with one older
  than 1.32.3 the ``d_qk != d_v`` backward raises instead of returning the old
  kernel's all-zero ``d_sink``, and so does any backward under
  FLAGS_cudnn_deterministic (the old kernels are not deterministic).
* ``csa_sparse_attn_bwd_cudnn`` on a GPU: with FLAGS_cudnn_deterministic the
  rows are sorted before the wrapper (dKV / d_sink equal to the unsorted
  deterministic run bit for bit, equal to a pre-sorted wrapper call in every
  output), two runs are bitwise identical, and without the flag the rows go
  through unsorted.
"""

import contextlib
import importlib
import sys
import unittest
from unittest import mock

import numpy as np
import paddle

try:
    import paddlefleet_ops

    from paddlefleet.cudnn_ops import (
        deterministic as det,
        frontend_version as fev,
    )

    # the package re-exports a function of the same name; take the module
    bwd = importlib.import_module(
        "paddlefleet.cudnn_ops.attn.csa_sparse_attn_bwd_cudnn"
    )
    from paddlefleet.cudnn_ops.attn.csa_sparse_attn_fwd_cudnn import (
        sparse_attn_fwd,
    )

    _IMPORT_OK = True
    _CUDNN = paddlefleet_ops.is_cudnn_frontend_available()
except (ImportError, RuntimeError, AttributeError):
    _IMPORT_OK = _CUDNN = False

_MODULE = unittest.skipUnless(
    _IMPORT_OK, "paddlefleet.cudnn_ops not importable"
)
_GPU = unittest.skipUnless(
    _CUDNN
    and paddle.is_compiled_with_cuda()
    and paddle.device.cuda.get_device_capability()[0] >= 9,
    "requires SM90+ and the vendored cudnn-frontend",
)

FLAG = "FLAGS_cudnn_deterministic"


@contextlib.contextmanager
def _flag(on):
    old = paddle.get_flags([FLAG])[FLAG]
    paddle.set_flags({FLAG: on})
    try:
        yield
    finally:
        paddle.set_flags({FLAG: old})


def _i32(rows):
    return paddle.to_tensor(np.asarray(rows, dtype=np.int32))


@_MODULE
class TestSortTopk(unittest.TestCase):
    def test_ascending_with_pad_last(self):
        out = det.sort_topk_ascending_pad_last(
            _i32([[5, -1, 0, 9, -1, 2], [-1, -1, -1, -1, -1, -1]])
        )
        np.testing.assert_array_equal(
            out.numpy(), [[0, 2, 5, 9, -1, -1], [-1, -1, -1, -1, -1, -1]]
        )


@_MODULE
class TestDeterministicKwargs(unittest.TestCase):
    @staticmethod
    def _with(
        q, kv, deterministic=False, deterministic_relay_cluster_size=None
    ):
        pass

    @staticmethod
    def _det_only(q, kv, deterministic=False):
        pass

    @staticmethod
    def _without(q, kv):
        pass

    def test_flag_and_option(self):
        # The flag is read per call: an off call must not pin later calls.
        with _flag(False):
            self.assertEqual(det.cudnn_deterministic_kwargs(self._det_only), {})
        with _flag(True):
            self.assertEqual(det.cudnn_deterministic_kwargs(self._without), {})
            self.assertEqual(
                det.cudnn_deterministic_kwargs(self._det_only),
                {"deterministic": True},
            )

    def test_relay_only_on_sm90(self):
        with (
            _flag(True),
            mock.patch.object(det, "_sm90_relay_cluster_size", return_value=8),
            mock.patch.object(
                paddle.device.cuda, "get_device_capability", return_value=(9, 0)
            ),
        ):
            self.assertEqual(
                det.cudnn_deterministic_kwargs(self._with),
                {"deterministic": True, "deterministic_relay_cluster_size": 8},
            )
        with (
            _flag(True),
            mock.patch.object(
                paddle.device.cuda,
                "get_device_capability",
                return_value=(10, 0),
            ),
        ):
            self.assertEqual(
                det.cudnn_deterministic_kwargs(self._with),
                {"deterministic": True},
            )

    @staticmethod
    def _probe(patch):
        det._sm90_relay_cluster_size.cache_clear()
        try:
            with patch:
                return det._sm90_relay_cluster_size()
        finally:
            det._sm90_relay_cluster_size.cache_clear()

    def test_relay_probe(self):
        probe = "paddlefleet_ops.cudnn.frost.occupancy"

        def fake(size):
            if size == 16:
                raise NotImplementedError("does not fit")
            return 10

        # The widest size that launches wins; sizes that do not fit are skipped.
        patch = mock.patch(f"{probe}.max_active_clusters", side_effect=fake)
        self.assertEqual(self._probe(patch), 8)
        # No size launches, or no probe at all: cuDNN's default, with a warning.
        broke = RuntimeError("probe broke")
        for patch, msg in (
            (
                mock.patch(f"{probe}.max_active_clusters", side_effect=broke),
                "probe broke",
            ),
            (mock.patch.dict(sys.modules, {probe: None}), "occupancy probe"),
        ):
            with (
                self.subTest(msg),
                self.assertLogs(det.logger, level="WARNING") as logs,
            ):
                self.assertIsNone(self._probe(patch))
            self.assertIn(msg, "\n".join(logs.output))


@_MODULE
class TestFrontendVersion(unittest.TestCase):
    def setUp(self):
        fev.cudnn_frontend_version.cache_clear()

    def tearDown(self):
        fev.cudnn_frontend_version.cache_clear()

    def test_version_parsed(self):
        for text, want in (("1.28.0", (1, 28, 0)), ("1.32.3", (1, 32, 3))):
            fev.cudnn_frontend_version.cache_clear()
            fake = mock.Mock(__version__=text)
            with mock.patch.dict(sys.modules, {"paddlefleet_ops.cudnn": fake}):
                self.assertEqual(fev.cudnn_frontend_version(), want)
        fev.cudnn_frontend_version.cache_clear()
        with mock.patch.dict(sys.modules, {"paddlefleet_ops.cudnn": None}):
            self.assertIsNone(fev.cudnn_frontend_version())

    def test_old_frontend_rejects_mqa_backward(self):
        q = paddle.zeros([2, 1, 576], dtype="bfloat16")
        out = paddle.zeros([2, 1, 512], dtype="bfloat16")
        for version, found in (
            ((1, 28, 0), "found 1.28.0"),
            (None, "not importable"),
        ):
            with (
                self.subTest(version=version),
                mock.patch.object(
                    bwd, "is_cudnn_frontend_available", return_value=True
                ),
                mock.patch.object(
                    bwd, "cudnn_frontend_version", return_value=version
                ),
                self.assertRaisesRegex(
                    RuntimeError, f"1.32.3.*{found}.*Rebuild"
                ),
            ):
                bwd.csa_sparse_attn_bwd_cudnn(
                    q, None, out, None, None, None, None
                )

    def test_old_frontend_rejects_deterministic_backward(self):
        # d_qk == d_v (CSA / HCA): an old wheel has no deterministic kernel, so
        # FLAGS_cudnn_deterministic must fail loudly instead of running the
        # atomic one.
        q = paddle.zeros([2, 1, 512], dtype="bfloat16")
        with (
            _flag(True),
            mock.patch.object(
                bwd, "is_cudnn_frontend_available", return_value=True
            ),
            mock.patch.object(
                bwd, "cudnn_frontend_version", return_value=(1, 28, 0)
            ),
            self.assertRaisesRegex(
                RuntimeError, "deterministic.*1.32.3.*found 1.28.0"
            ),
        ):
            bwd.csa_sparse_attn_bwd_cudnn(q, None, q, None, None, None, None)


H, D, S, SKV, TOPK = 64, 512, 256, 1024, 128


def _inputs(seed=0):
    rng = np.random.RandomState(seed)
    paddle.seed(seed)
    q = (paddle.randn([1, S, H, D]) * 0.5).cast("bfloat16")
    kv = (paddle.randn([1, SKV, D]) * 0.5).cast("bfloat16")
    sink = paddle.to_tensor(rng.normal(0, 1, H).astype("float32"))
    ti = np.full([1, S, TOPK], -1, np.int32)
    for i in range(S):
        n = TOPK - rng.randint(0, 8)  # a few -1 holes per row
        ti[0, i, :n] = rng.permutation(SKV)[:n]  # shuffled, duplicate-free
    ti = paddle.to_tensor(ti)
    out, lse, _ = sparse_attn_fwd(q, kv, sink, ti, sm_scale=D**-0.5, d_v=D)
    dout = paddle.randn(out.shape).cast("bfloat16")
    flat = (
        q.reshape([S, H, D]),
        kv.reshape([SKV, D]),
        out.reshape([S, H, D]),
        dout.reshape([S, H, D]),
        lse.reshape([S, H]),
        sink,
    )
    return flat, ti.reshape([S, TOPK])


def _wrapper():
    from paddlefleet_ops.cudnn.deepseek_sparse_attention.sparse_attention_backward.api import (
        sparse_attention_backward_wrapper,
    )

    return sparse_attention_backward_wrapper


def _call_wrapper(flat, idx, **kw):
    r = _wrapper()(*flat, idx, softmax_scale=D**-0.5, topk_length=None, **kw)
    return _np(r["dq"], r["dkv"], r["d_sink"])


def _call_bwd(flat, idx):
    return _np(
        *bwd.csa_sparse_attn_bwd_cudnn(*flat, idx, softmax_scale=D**-0.5)
    )


def _np(*ts):
    return [t.cast("float32").numpy() for t in ts]


@_GPU
class TestDeterministicBackward(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paddle.set_device("gpu")
        cls.flat, cls.idx = _inputs()

    def test_det_sorts_rows_and_keeps_dkv_dsink(self):
        with _flag(True):
            kw = det.cudnn_deterministic_kwargs(_wrapper())
            got = _call_bwd(self.flat, self.idx)
            unsorted = _call_wrapper(self.flat, self.idx, **kw)
            presorted = _call_wrapper(
                self.flat, det.sort_topk_ascending_pad_last(self.idx), **kw
            )
        for name, a, b in zip(("dq", "dkv", "d_sink"), got, presorted):
            self.assertTrue(np.array_equal(a, b), f"{name} != pre-sorted call")
        self.assertTrue(np.array_equal(got[1], unsorted[1]), "dkv moved")
        self.assertTrue(np.array_equal(got[2], unsorted[2]), "d_sink moved")
        rel = np.linalg.norm(got[0] - unsorted[0]) / np.linalg.norm(unsorted[0])
        self.assertLess(rel, 1e-3, "dq moved more than a summation reorder")

    def test_det_two_runs_bitwise(self):
        with _flag(True):
            a = _call_bwd(self.flat, self.idx)
            b = _call_bwd(self.flat, self.idx)
        for name, x, y in zip(("dq", "dkv", "d_sink"), a, b):
            self.assertTrue(np.array_equal(x, y), f"{name} not reproducible")

    def test_non_det_does_not_sort(self):
        with (
            _flag(False),
            mock.patch.object(
                bwd,
                "sort_topk_ascending_pad_last",
                wraps=bwd.sort_topk_ascending_pad_last,
            ) as spy,
        ):
            _call_bwd(self.flat, self.idx)
        spy.assert_not_called()


if __name__ == "__main__":
    unittest.main()
