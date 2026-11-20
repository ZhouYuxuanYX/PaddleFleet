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

"""``dsa_index_topk_backend="flashtopk"`` vs ``"paddle"`` in ``cudnn_indexer_topk_fwd``.

The output layout (shapes, dtypes, ``topk_length``, ``-1``/``-inf`` padding)
matches ``"paddle"``; the selected keys and their scores are checked against an
fp64 reference, with ties allowed within ``REL_TIE`` of the row's largest score.
"""

import sys
import types
import unittest
from unittest import mock

import numpy as np
import paddle
from paddlefleet_ops import is_cudnn_frontend_available

import paddlefleet.cudnn_ops.indexer.csa_indexer_fwd_cudnn as indexer_mod

REL_TIE = 1e-5
REL_SCORE = 1e-4


def _flashtopk_kernel_available():
    if not is_cudnn_frontend_available():
        return False
    try:
        indexer_mod._require_flashtopk()
    except ImportError:
        return False
    return True


_HAS_FLASHTOPK = _flashtopk_kernel_available()


def _docmask_valid_range(doc_lens, ratio, seq_len):
    """``[1, S, 2]`` int32 document-local causal windows of a packed batch.

    Row 5 gets an empty window ``[end, end)``, and the early rows of each
    document keep fewer candidates than ``top_k``.
    """
    rows, col = [], 0
    for n in doc_lens:
        rows += [[col, col + (t + 1) // ratio] for t in range(n)]
        col += n // ratio
    window = np.asarray(rows, dtype=np.int32).reshape(1, seq_len, 2)
    window[0, 5] = [window[0, 5, 1], window[0, 5, 1]]
    return paddle.to_tensor(window)


def _reference_scores(index_q, index_k, weights, scale):
    """fp64 ``[S, S_k]``: ``scale * sum_h w_h relu(q_h . k)``, from the bf16 operands."""
    # bf16 -> fp32 is exact; ``Tensor.numpy()`` of a bf16 tensor returns raw uint16 bits.
    q, k, w = (
        t[0].astype("float32").numpy().astype(np.float64)
        for t in (index_q, index_k, weights)
    )
    dots = np.maximum(np.einsum("shd,kd->shk", q, k, optimize=True), 0.0)
    return np.einsum("sh,shk->sk", w, dots, optimize=True) * scale


def _windows(valid_range, seq_len, sk, ratio, seq_offset):
    """``(start, end)`` int64 ``[S]``: each row's candidate window in key ids."""
    causal = np.minimum((seq_offset + np.arange(seq_len) + 1) // ratio, sk)
    if valid_range is None:
        return np.zeros(seq_len, dtype=np.int64), causal
    window = valid_range.reshape([seq_len, 2]).numpy().astype(np.int64)
    return np.maximum(window[:, 0], 0), np.minimum(window[:, 1], causal)


@unittest.skipIf(not _HAS_FLASHTOPK, "FlashTopK + cuDNN indexer require SM100")
class TestFlashTopKIndexerFwdKernel(unittest.TestCase):
    """``"flashtopk"`` vs ``"paddle"`` on identical random bf16 indexer inputs."""

    S, TOPK = 256, 64

    def setUp(self):
        paddle.set_device("gpu")
        paddle.seed(2026)
        self.scale = 128**-0.5

    def _call(self, inputs, kwargs, backend, with_scores):
        return indexer_mod.cudnn_indexer_topk_fwd(
            *inputs,
            ratio=self.ratio,
            topk_effective=self.TOPK,
            return_topk_scores=with_scores,
            dsa_index_topk_backend=backend,
            **kwargs,
        )

    def _check(self, inputs, kwargs, exact):
        seq_offset = int(kwargs.get("seq_offset", 0))
        start, end = _windows(
            kwargs.get("valid_range"), self.S, self.sk, self.ratio, seq_offset
        )
        ref_ids, ref_len, ref_scores = (
            t.numpy() for t in self._call(inputs, kwargs, "paddle", True)
        )
        ref_ids, ref_scores = ref_ids[0], ref_scores[0]
        # The fp64 oracle must reproduce cuDNN's scores, or the checks below are moot.
        rows, slots = np.nonzero(ref_ids >= 0)
        diff = np.abs(
            ref_scores[rows, slots] - exact[rows, ref_ids[rows, slots]]
        )
        bound = REL_SCORE * np.abs(exact[rows]).max(axis=1)
        self.assertTrue(
            (diff <= bound).all(),
            f"oracle disagrees with cuDNN: max diff {diff.max():.3e}",
        )
        for with_scores in (False, True):
            with self.subTest(scores=with_scores):
                # FlashTopK runs on both calls and the cuDNN path on neither.
                kernel = mock.Mock(wraps=indexer_mod._require_flashtopk())
                with (
                    mock.patch.object(
                        indexer_mod, "_require_flashtopk", return_value=kernel
                    ),
                    mock.patch.object(
                        indexer_mod,
                        "_cudnn_indexer_topk_fwd_impl",
                        side_effect=AssertionError("cuDNN path ran"),
                    ),
                ):
                    out = self._call(inputs, kwargs, "flashtopk", with_scores)
                    again = self._call(inputs, kwargs, "flashtopk", with_scores)
                self.assertEqual(kernel.call_count, 2)
                self.assertEqual(len(out), 3 if with_scores else 2)
                ids, length = out[0], out[1]
                self.assertEqual(ids.dtype, paddle.int32)
                self.assertEqual(list(ids.shape), [1, self.S, self.TOPK])
                self.assertEqual(length.dtype, paddle.int32)
                self.assertEqual(list(length.shape), [1, self.S])
                np.testing.assert_array_equal(length.numpy()[0], ref_len[0])
                np.testing.assert_array_equal(ids.numpy()[0] >= 0, ref_ids >= 0)
                self.assertEqual(
                    (ids.numpy()[0] >= 0).sum(-1).tolist(),
                    length.numpy()[0].tolist(),
                )
                if with_scores:
                    self.assertEqual(out[2].dtype, paddle.float32)
                    self.assertEqual(list(out[2].shape), [1, self.S, self.TOPK])
                    np.testing.assert_array_equal(
                        np.isneginf(out[2].numpy()[0]), np.isneginf(ref_scores)
                    )
                for got, want in zip(
                    out, again
                ):  # same call twice -> same bits
                    np.testing.assert_array_equal(got.numpy(), want.numpy())

                got_ids = ids.numpy()[0]
                got_scores = out[2].numpy()[0] if with_scores else None
                for row in range(self.S):
                    lo, hi = int(start[row]), int(end[row])
                    live = min(self.TOPK, max(hi - lo, 0))
                    if live == 0:
                        self.assertFalse((got_ids[row] >= 0).any())
                        continue
                    picked = got_ids[row][got_ids[row] >= 0]
                    self.assertEqual(picked.size, live)
                    self.assertEqual(len(set(picked.tolist())), live)
                    self.assertTrue(((picked >= lo) & (picked < hi)).all())
                    window = exact[row, lo:hi]
                    theta = np.partition(window, window.size - live)[
                        window.size - live
                    ]
                    # fp32 rounding scales with the row's scores, not with theta.
                    row_scale = np.abs(window).max()
                    tol = REL_TIE * row_scale
                    above = lo + np.flatnonzero(window > theta + tol)
                    self.assertTrue(
                        np.isin(above, picked).all(),
                        f"row {row}: a key above the K-th score was dropped",
                    )
                    self.assertTrue(
                        (exact[row, picked] >= theta - tol).all(),
                        f"row {row}: a key below the K-th score was picked",
                    )
                    if got_scores is None:
                        continue
                    np.testing.assert_allclose(
                        got_scores[row][got_ids[row] >= 0],
                        exact[row, picked],
                        rtol=0,
                        atol=REL_SCORE * row_scale,
                        err_msg=f"row {row}: score not paired with its key",
                    )

    def test_causal_and_docmask(self):
        doc_lens = [96, self.S - 96]
        for heads in (32, 64):
            for ratio in (1, 4):
                self.ratio, self.sk = ratio, self.S // ratio
                inputs = (
                    paddle.randn([1, self.S, heads, 128]).astype("bfloat16"),
                    paddle.randn([1, self.sk, 128]).astype("bfloat16"),
                    paddle.randn([1, self.S, heads]).astype("bfloat16"),
                )
                exact = _reference_scores(*inputs, self.scale)
                window = _docmask_valid_range(doc_lens, ratio, self.S)
                modes = {
                    "causal": {},
                    "causal_offset": {"seq_offset": 64},
                    "docmask": {"valid_range": window},
                    "docmask_offset": {"valid_range": window, "seq_offset": 64},
                    "thd_docmask": {
                        "valid_range": window,
                        "doc_lens": doc_lens,
                    },
                }
                for mode, kwargs in modes.items():
                    with self.subTest(heads=heads, ratio=ratio, mode=mode):
                        self._check(inputs, kwargs, exact)


class TestFlashTopKPlumbing(unittest.TestCase):
    """Dispatch and argument contract, with FlashTopK's kernel replaced by a fake.

    No GPU and no FlashTopK build are required, so these cases run on every
    machine instead of being skipped like the kernel comparison above.
    """

    S, H, D, TOPK = 16, 64, 128, 8

    def setUp(self):
        self.index_q = paddle.zeros(
            [1, self.S, self.H, self.D], dtype="bfloat16"
        )
        self.index_k = paddle.zeros([1, self.S, self.D], dtype="bfloat16")
        self.weights = paddle.zeros([1, self.S, self.H], dtype="bfloat16")
        self.valid_range = paddle.zeros([1, self.S, 2], dtype="int32")

    def _call(self, return_topk_scores=False):
        """Run the public entry point against a recording fake kernel."""
        # FlashTopK's ``indexer_topk`` always returns three values; ``scores``
        # is None when ``return_scores`` is False.
        result = ("indices", "length", "scores" if return_topk_scores else None)
        calls = []

        def kernel(*args, **kwargs):
            calls.append((args, kwargs))
            return result

        with mock.patch.object(
            indexer_mod, "_require_flashtopk", return_value=kernel
        ):
            out = indexer_mod.cudnn_indexer_topk_fwd(
                self.index_q,
                self.index_k,
                self.weights,
                ratio=4,
                topk_effective=self.TOPK,
                indexer_softmax_scale=2.0,
                valid_range=self.valid_range,
                seq_offset=128,
                return_topk_scores=return_topk_scores,
                dsa_index_topk_backend="flashtopk",
            )
        return out, calls

    def test_require_flashtopk_returns_the_library_entry_point(self):
        """The lazy import returns FlashTopK's own function, unwrapped."""
        entry = object()
        module = types.ModuleType("paddlefleet_ops.flashtopk")
        module.flashtopk_indexer_topk = entry
        with mock.patch.dict(
            sys.modules, {"paddlefleet_ops.flashtopk": module}
        ):
            self.assertIs(indexer_mod._require_flashtopk(), entry)

    def test_head_dim_other_than_128_is_rejected(self):
        with self.assertRaisesRegex(ValueError, r"D_i=128, got 64"):
            indexer_mod._validate_indexer_inputs(
                paddle.zeros([1, self.S, self.H, 64], dtype="bfloat16"),
                paddle.zeros([1, self.S, 64], dtype="bfloat16"),
                self.weights,
            )

    def test_forwards_exactly_the_contract_arguments(self):
        """Every kernel argument is forwarded unchanged, including the scale."""
        out, calls = self._call(return_topk_scores=True)
        self.assertEqual(len(calls), 1)
        args, kwargs = calls[0]
        self.assertEqual(len(args), 4)
        for got, want in zip(
            args[:3], (self.index_q, self.index_k, self.weights)
        ):
            self.assertIs(got, want)
        self.assertEqual(args[3], self.TOPK)
        # dim ** -0.5 times the caller's extra scale, computed the same way.
        self.assertEqual(kwargs["softmax_scale"], float(self.D) ** -0.5 * 2.0)
        self.assertEqual(kwargs["ratio"], 4)
        self.assertEqual(kwargs["seq_offset"], 128)
        self.assertIs(kwargs["valid_range"], self.valid_range)
        self.assertIs(kwargs["return_scores"], True)
        self.assertEqual(out, ("indices", "length", "scores"))

    def test_index_only_call_returns_two_values(self):
        out, calls = self._call(return_topk_scores=False)
        self.assertIs(calls[0][1]["return_scores"], False)
        self.assertEqual(out, ("indices", "length"))


if __name__ == "__main__":
    unittest.main()
