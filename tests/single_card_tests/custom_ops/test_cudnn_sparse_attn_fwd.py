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

"""``sparse_attn_fwd`` dispatch and the cuDNN forward's ``lse_indexer``.

* ``cudnn_dsa_supported``: SM100 *and* an importable vendored cudnn-frontend
  of at least 1.32.3; ``is_dsa_available`` is the same predicate. Without it
  (Python < 3.12, or an ops wheel built with 1.28) SM100 keeps the FlashMLA
  forward instead of raising.
* ``sparse_attn_fwd`` forwards every argument by name to the chosen backend.
* ``cudnn_sparse_attn`` plumbing with the kernel stubbed (any device): guards,
  the flattened inputs the wrapper receives, the reshaped outputs.
* ``cudnn_sparse_attn`` on a GPU: ``lse`` and ``lse_indexer`` against an fp64
  log-sum-exp over the gathered columns (all of them / the first
  ``indexer_topk``), with ``-1`` holes in both parts.
"""

import importlib
import unittest
from unittest import mock

import numpy as np
import paddle

try:
    import paddlefleet_ops

    from paddlefleet.cudnn_ops import block_sparse_mqa_dsa
    from paddlefleet.cudnn_ops.attn import csa_sparse_attn_fwd_cudnn as fwd

    _IMPORT_OK = True
except (ImportError, RuntimeError, AttributeError):
    _IMPORT_OK = False

_MODULE = unittest.skipUnless(
    _IMPORT_OK, "paddlefleet.cudnn_ops not importable"
)


def _clear_caches():
    fwd.cudnn_dsa_supported.cache_clear()
    block_sparse_mqa_dsa.is_dsa_available.cache_clear()


@_MODULE
class TestDispatch(unittest.TestCase):
    def setUp(self):
        _clear_caches()

    def tearDown(self):
        _clear_caches()

    def _supported(self, major, frontend, version=(1, 32, 3)):
        with (
            mock.patch.object(
                fwd, "cudnn_frontend_version", return_value=version
            ),
            mock.patch.object(
                paddle.device.cuda,
                "get_device_capability",
                return_value=(major, 0),
            ),
            mock.patch.object(
                paddlefleet_ops,
                "is_cudnn_frontend_available",
                return_value=frontend,
            ),
        ):
            _clear_caches()
            return (
                fwd.cudnn_dsa_supported(),
                block_sparse_mqa_dsa.is_dsa_available(),
            )

    def test_predicate(self):
        self.assertEqual(self._supported(10, True), (True, True))
        # SM100 without cudnn-frontend: FlashMLA forward, no DSA path
        self.assertEqual(self._supported(10, False), (False, False))
        # an ops wheel with an older cudnn-frontend (1.28 has no sparse
        # forward) or one whose version cannot be read: FlashMLA forward
        for version in ((1, 28, 0), None):
            self.assertEqual(self._supported(10, True, version), (False, False))
        # cuDNN ships DSA kernels for SM100 only
        for major in (9, 11, 12):
            self.assertEqual(self._supported(major, True), (False, False))
        # no usable CUDA device: the probe raises, the predicate says False
        with mock.patch.object(
            paddle.device.cuda,
            "get_device_capability",
            side_effect=RuntimeError("no device"),
        ):
            _clear_caches()
            self.assertFalse(fwd.cudnn_dsa_supported())

    def test_routes_by_predicate_and_forwards_by_name(self):
        args = ("q", "kv", "sink", "idx")
        kw = {
            "sm_scale": 0.5,
            "indexer_topk": 512,
            "d_v": 512,
            "topk_length": "len",
            "global_kv_idx_remap_fusion": True,
        }
        for supported, chosen, other in (
            (True, "cudnn_sparse_attn", "flash_mla_sparse_attn"),
            (False, "flash_mla_sparse_attn", "cudnn_sparse_attn"),
        ):
            with (
                self.subTest(supported=supported),
                mock.patch.object(
                    fwd, "cudnn_dsa_supported", return_value=supported
                ),
                mock.patch.object(fwd, chosen, return_value="out") as hit,
                mock.patch.object(fwd, other) as miss,
            ):
                self.assertEqual(fwd.sparse_attn_fwd(*args, **kw), "out")
                hit.assert_called_once_with(*args, **kw)
                miss.assert_not_called()


_FWD_API = "paddlefleet_ops.cudnn.deepseek_sparse_attention.sparse_attention_forward.api"


@_MODULE
class TestCudnnSparseAttnContract(unittest.TestCase):
    """``cudnn_sparse_attn`` with the kernel wrapper stubbed (any device): the
    guards, the flat-global indices and int32 ``topk_length`` it passes, and the
    reshaped outputs. The kernel itself is covered by ``TestCudnnLseIndexer``.
    """

    b, sq, h, d, skv, topk = 2, 3, 64, 576, 5, 4

    def _inputs(self):
        q = paddle.zeros([self.b, self.sq, self.h, self.d], dtype="bfloat16")
        kv = paddle.zeros([self.b, self.skv, self.d], dtype="bfloat16")
        sink = paddle.zeros([self.h], dtype="float32")
        idx = paddle.to_tensor(
            np.tile(np.array([0, 2, -1, 4], np.int32), (self.b, self.sq, 1))
        )
        return q, kv, sink, idx

    def _fake(self, rows):
        out = paddle.ones([rows, self.h, 512], dtype="bfloat16")
        lse = paddle.full([rows, self.h], 2.0, dtype="float32")
        return mock.Mock(
            return_value={"out": out, "lse": lse, "lse_indexer": lse * 0.5}
        )

    def test_frontend_missing_raises(self):
        with (
            mock.patch.object(
                paddlefleet_ops,
                "is_cudnn_frontend_available",
                return_value=False,
            ),
            self.assertRaises(ImportError),
        ):
            fwd.cudnn_sparse_attn(*self._inputs())

    def test_wrapper_contract(self):
        try:
            importlib.import_module(_FWD_API)
        except Exception as e:
            self.skipTest(f"cuDNN sparse forward API not importable: {e!r}")
        rows = self.b * self.sq
        fake = self._fake(rows)
        with (
            mock.patch.object(
                paddlefleet_ops,
                "is_cudnn_frontend_available",
                return_value=True,
            ),
            mock.patch(f"{_FWD_API}.sparse_attention_forward_wrapper", fake),
        ):
            # d_v defaults to d_qk (576 here), which the kernel cannot serve
            with self.assertRaisesRegex(ValueError, "d_v == 512"):
                fwd.cudnn_sparse_attn(*self._inputs())
            fake.assert_not_called()
            q, kv, sink, idx = self._inputs()
            length = paddle.full([self.b, self.sq], self.topk, dtype="int64")
            out, lse, lse_ix = fwd.cudnn_sparse_attn(
                q,
                kv,
                sink,
                idx,
                sm_scale=0.125,
                indexer_topk=512,
                d_v=512,
                topk_length=length,
            )
        (_, _, fidx), kw = fake.call_args
        # batch-local columns -> flat-global ones, -1 kept
        self.assertEqual(fidx.numpy()[self.sq].tolist(), [5, 7, -1, 9])
        self.assertEqual(kw["topk_length"].dtype, paddle.int32)
        self.assertEqual(list(kw["topk_length"].shape), [rows])
        self.assertEqual(
            (kw["softmax_scale"], kw["indexer_topk"]), (0.125, 512)
        )
        self.assertEqual(list(out.shape), [self.b, self.sq, self.h, 512])
        self.assertEqual(list(lse.shape), [self.b, self.sq, self.h])
        self.assertTrue(bool((lse_ix == 1.0).all()))


H, DQK, DV, S, SKV, TOPK = 64, 576, 512, 64, 2048, 1024


def _lse_ref(q, kv, idx, scale):
    """fp64 natural-log LSE over the gathered columns, ``-1`` skipped."""
    q, kv = q.astype(np.float64), kv.astype(np.float64)
    out = np.empty(q.shape[:2])
    for i in range(q.shape[0]):
        cols = idx[i][idx[i] >= 0]
        s = q[i] @ kv[cols].T * scale  # [H, n]
        m = s.max(-1, keepdims=True)
        out[i] = (m + np.log(np.exp(s - m).sum(-1, keepdims=True)))[:, 0]
    return out


@_MODULE
@unittest.skipUnless(
    _IMPORT_OK and block_sparse_mqa_dsa.is_dsa_available(),
    "requires SM100 and the vendored cudnn-frontend",
)
class TestCudnnLseIndexer(unittest.TestCase):
    def test_lse_and_lse_indexer_match_fp64(self):
        paddle.set_device("gpu")
        rng = np.random.RandomState(0)
        q = (rng.randn(S, H, DQK) * 0.5).astype("float32")
        kv = (rng.randn(SKV, DQK) * 0.5).astype("float32")
        idx = np.stack([rng.permutation(SKV)[:TOPK] for _ in range(S)])
        idx = idx.astype(np.int32)
        idx[rng.rand(S, TOPK) < 0.05] = -1  # holes in indexer and tail parts
        # the bf16 values the kernel sees, so the reference shares them
        qb = paddle.to_tensor(q).cast("bfloat16")
        kvb = paddle.to_tensor(kv).cast("bfloat16")
        q64 = qb.cast("float32").numpy()
        kv64 = kvb.cast("float32").numpy()
        scale = DQK**-0.5
        sink = paddle.full([H], -1e30, dtype="float32")
        for itk in (512, 1024):
            with self.subTest(indexer_topk=itk):
                _, lse, lse_ix = fwd.cudnn_sparse_attn(
                    qb.reshape([1, S, H, DQK]),
                    kvb.reshape([1, SKV, DQK]),
                    sink,
                    paddle.to_tensor(idx).reshape([1, S, TOPK]),
                    sm_scale=scale,
                    indexer_topk=itk,
                    d_v=DV,
                )
                np.testing.assert_allclose(
                    lse.numpy()[0],
                    _lse_ref(q64, kv64, idx, scale),
                    atol=2e-3,
                    rtol=0,
                )
                np.testing.assert_allclose(
                    lse_ix.numpy()[0],
                    _lse_ref(q64, kv64, idx[:, :itk], scale),
                    atol=2e-3,
                    rtol=0,
                )


if __name__ == "__main__":
    unittest.main()
