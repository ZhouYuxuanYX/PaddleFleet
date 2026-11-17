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

"""``HyperBodyEncoderBridge._forward_packed``: which rows reach the decoder.

The bridge keeps only each segment's suffix (query) rows of the packed encoder
output, normalizes them and out-projects them. These tests pin the three things
that can go wrong silently: the wrong rows are picked, ``final_norm`` still runs
over the whole packed sequence (the memory the change exists to save), or the
gradient flows back into context / pad rows.
"""

import types
import unittest

import numpy as np
import paddle

from paddlefleet.models.hyperencoder.norm import HyperEncoderRMSNorm
from paddlefleet.transformer.prefix_lm_mask import build_flashmask_row_indices
from paddlefleet.transformers.hyperbody.modeling import HyperBodyEncoderBridge

H_ENC, H_DEC = 8, 4


class _Projector(paddle.nn.Layer):
    """``RowParallelLinear`` stand-in: ``(x @ W, bias)``, the caller adds the bias."""

    def __init__(self):
        super().__init__()
        self.w = self.create_parameter([H_ENC, H_DEC])
        self.b = self.create_parameter([H_DEC])

    def forward(self, x):
        return paddle.matmul(x, self.w), self.b


class _Bridge(paddle.nn.Layer):
    """The real ``_forward_packed`` / ``_project`` with toy towers and no CP."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            hyperencoder_cp_shard=False, cp_balance_mode="contiguous_allgather"
        )
        self.final_norm = HyperEncoderRMSNorm(
            types.SimpleNamespace(
                hidden_size=H_ENC, rms_norm_eps=1e-6, params_dtype="float32"
            )
        )
        self.out_projector = _Projector()

    _project = HyperBodyEncoderBridge._project
    _forward_packed = HyperBodyEncoderBridge._forward_packed


def _make():
    paddle.seed(0)
    bridge = _Bridge()
    with paddle.no_grad():
        for p in bridge.parameters():
            p.set_value(paddle.randn(p.shape, dtype="float32") * 0.5)
    return bridge


def _layout(n_contexts, n_queries, pad):
    """The dict the frontend packs: prefix sums, per-segment lengths, query row ids."""
    starts, run = [], 0
    for nc, nq in zip(n_contexts, n_queries):
        starts.append(run)
        run += nc + nq
    _, query_index = build_flashmask_row_indices(
        n_contexts, n_queries, pad, return_query_index=True
    )
    return {
        "segment_starts": tuple(starts),
        "n_contexts": tuple(n_contexts),
        "n_queries": tuple(n_queries),
        "query_index": query_index,
    }


def _norm_then_slice(bridge, h, layout):
    """The pre-change semantics: normalize the whole pack, then keep the suffix rows."""
    normed = bridge.final_norm(h)[0]
    outs = [
        bridge._project(normed[s + nc : s + nc + nq, :])
        for s, nc, nq in zip(
            layout["segment_starts"],
            layout["n_contexts"],
            layout["n_queries"],
        )
    ]
    return paddle.concat(outs, axis=0)


LAYOUTS = [
    ([3], [4], 0),  # single segment
    ([2, 5], [3, 3], 7),  # + pad
    ([0, 4], [2, 2], 0),  # no context in the first segment
    ([1, 0, 6], [5, 5, 5], 2),  # no context in the middle
    ([2, 3, 1], [4, 2, 6], 3),  # uneven query lengths
]


class TestForwardPacked(unittest.TestCase):
    def _run(self, n_contexts, n_queries, pad):
        bridge = _make()
        layout = _layout(n_contexts, n_queries, pad)
        total = sum(n_contexts) + sum(n_queries) + pad
        h = paddle.randn([1, total, H_ENC]) * 3.0
        h.stop_gradient = False
        return bridge, layout, h

    def test_matches_norm_then_slice(self):
        # RMSNorm is row-wise, so gathering first changes no value: the forward is
        # bitwise equal to normalizing the whole pack and slicing.
        for n_contexts, n_queries, pad in LAYOUTS:
            with self.subTest(
                n_contexts=n_contexts, n_queries=n_queries, pad=pad
            ):
                bridge, layout, h = self._run(n_contexts, n_queries, pad)
                got = bridge._forward_packed({"hidden_states": h}, layout)
                want = _norm_then_slice(bridge, h, layout)
                self.assertEqual(got.shape, [sum(n_queries), H_DEC])
                np.testing.assert_array_equal(got.numpy(), want.numpy())

    def test_normalizes_suffix_rows_only(self):
        seen = {}

        class _Record(paddle.nn.Layer):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                seen["shape"] = list(x.shape)
                return self.inner(x)

        for n_contexts, n_queries, pad in LAYOUTS:
            with self.subTest(
                n_contexts=n_contexts, n_queries=n_queries, pad=pad
            ):
                bridge, layout, h = self._run(n_contexts, n_queries, pad)
                bridge.final_norm = _Record(bridge.final_norm)
                bridge._forward_packed({"hidden_states": h}, layout)
                self.assertEqual(seen["shape"], [1, sum(n_queries), H_ENC])

    def test_backward_skips_context_and_pad_rows(self):
        n_contexts, n_queries, pad = [2, 5], [3, 3], 7
        bridge, layout, h = self._run(n_contexts, n_queries, pad)
        out = bridge._forward_packed({"hidden_states": h}, layout)
        (
            out
            * paddle.arange(1, out.shape[0] + 1).astype("float32").unsqueeze(-1)
        ).sum().backward()
        g = h.grad[0]
        keep = sorted(layout["query_index"].numpy().tolist())
        drop = [r for r in range(h.shape[1]) if r not in set(keep)]
        self.assertEqual(float(g[drop].abs().max().item()), 0.0)

        ref = _make()
        for name, p in ref.named_parameters():
            p.set_value(dict(bridge.named_parameters())[name])
        h_ref = h.detach().clone()
        h_ref.stop_gradient = False
        out_ref = _norm_then_slice(ref, h_ref, layout)
        (
            out_ref
            * paddle.arange(1, out_ref.shape[0] + 1)
            .astype("float32")
            .unsqueeze(-1)
        ).sum().backward()
        for name, p in ref.named_parameters():
            other = dict(bridge.named_parameters())[name]
            self.assertIsNotNone(other.grad, name)
            self.assertLess(
                float((p.grad - other.grad).abs().max().item()), 1e-6, name
            )

        # The kept rows must carry gradient as well: a gather whose gradient is
        # detached still passes the checks above (dropped rows are zero anyway).
        np.testing.assert_array_equal(
            h.grad.numpy()[0][keep], h_ref.grad.numpy()[0][keep]
        )
        self.assertGreater(
            float(np.abs(h_ref.grad.numpy()[0][keep]).max()), 0.0
        )

    def test_uses_the_layout_query_index(self):
        # The bridge must follow the packed index, not re-derive rows from the starts.
        n_contexts, n_queries, pad = [2, 5], [3, 3], 7
        bridge, layout, h = self._run(n_contexts, n_queries, pad)
        reversed_layout = dict(layout)
        reversed_layout["query_index"] = layout["query_index"][::-1]
        got = bridge._forward_packed({"hidden_states": h}, reversed_layout)
        base = bridge._forward_packed({"hidden_states": h}, layout)
        np.testing.assert_array_equal(got.numpy()[::-1], base.numpy())


if __name__ == "__main__":
    unittest.main()
