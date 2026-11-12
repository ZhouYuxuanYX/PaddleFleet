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

"""CPU tests for the cutedsl path of ``flash_mask_facade.flash_attention``.

FlashMask has no mask-free entry point, so on FA3/FA4 the facade's
``flash_attention`` calls the paddle frontend's ``flashmask_attention`` with
``startend_row_indices=None``. The frontend is replaced by a recording stub
(installed in ``sys.modules`` so no CuTe kernel is imported), and
``get_fa_version`` is pinned, so the routing, argument relay, value padding and
the unsupported-argument checks are observed without a GPU.
"""

import sys
import types
import unittest
from unittest import mock

try:
    import paddle
    from paddlefleet_ops import flash_mask_facade as facade

    _IMPORT_ERROR = None
except ImportError as exc:  # paddle / paddlefleet_ops absent in this env
    paddle = None
    facade = None
    _IMPORT_ERROR = exc

_SKIP_REASON = f"paddlefleet_ops not importable: {_IMPORT_ERROR}"

_FRONTEND = "paddlefleet_ops.flashmask.frontend.paddle"


class _RecordingFrontend:
    """Stands in for ``flashmask.frontend.paddle.flashmask_attention``."""

    def __init__(self):
        self.calls = []

    def __call__(self, query, key, value, startend_row_indices=None, **kwargs):
        self.calls.append(
            {
                "value_shape": list(value.shape),
                "startend_row_indices": startend_row_indices,
                **kwargs,
            }
        )
        # Output carries the (possibly padded) value head dim, like the kernel.
        return paddle.ones([*query.shape[:-1], value.shape[-1]], query.dtype)


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class TestFlashAttentionCutedslPath(unittest.TestCase):
    def setUp(self):
        paddle.set_device("cpu")
        self.frontend = _RecordingFrontend()
        fake_modules = {
            "paddlefleet_ops.flashmask": types.ModuleType(
                "paddlefleet_ops.flashmask"
            ),
            "paddlefleet_ops.flashmask.frontend": types.ModuleType(
                "paddlefleet_ops.flashmask.frontend"
            ),
            _FRONTEND: types.ModuleType(_FRONTEND),
        }
        fake_modules[_FRONTEND].flashmask_attention = self.frontend
        patcher = mock.patch.dict(sys.modules, fake_modules)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _call(self, fa_version, q_dim=128, v_dim=128, **kwargs):
        q = paddle.zeros([2, 8, 4, q_dim], "float32")
        k = paddle.zeros([2, 8, 4, q_dim], "float32")
        v = paddle.zeros([2, 8, 4, v_dim], "float32")
        with mock.patch.object(
            facade, "get_fa_version", return_value=fa_version
        ):
            return facade.flash_attention(q, k, v, **kwargs)

    def test_routes_to_flashmask_without_a_mask(self):
        for fa_version in (3, 4):
            self.frontend.calls.clear()
            out, softmax = self._call(
                fa_version, causal=True, softmax_scale=0.5
            )
            self.assertIsNone(softmax)
            self.assertEqual(list(out.shape), [2, 8, 4, 128])
            self.assertEqual(len(self.frontend.calls), 1)
            call = self.frontend.calls[0]
            self.assertIsNone(call["startend_row_indices"])
            self.assertIs(call["causal"], True)
            self.assertEqual(call["softmax_scale"], 0.5)

    def test_value_is_padded_and_output_sliced_back(self):
        # (256, 128) is not a native pair, so value goes in padded to 256.
        out, _ = self._call(4, q_dim=256, v_dim=128, causal=True)
        self.assertEqual(self.frontend.calls[0]["value_shape"][-1], 256)
        self.assertEqual(list(out.shape), [2, 8, 4, 128])

    def test_native_pair_is_not_padded(self):
        self._call(3, q_dim=192, v_dim=128, causal=True)
        self.assertEqual(self.frontend.calls[0]["value_shape"][-1], 128)

    def test_dropout_in_training_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "dropout"):
            self._call(4, dropout=0.1, training=True)
        self.assertEqual(self.frontend.calls, [])

    def test_dropout_is_ignored_in_eval(self):
        self._call(4, dropout=0.1, training=False)
        self.assertEqual(len(self.frontend.calls), 1)

    def test_return_softmax_is_rejected(self):
        with self.assertRaisesRegex(NotImplementedError, "softmax"):
            self._call(4, return_softmax=True)


@unittest.skipIf(_IMPORT_ERROR is not None, _SKIP_REASON)
class TestUsesCutedslBackend(unittest.TestCase):
    def test_fa3_and_fa4_are_cutedsl_fa2_is_not(self):
        self.assertFalse(facade.uses_cutedsl_backend(2))
        self.assertTrue(facade.uses_cutedsl_backend(3))
        self.assertTrue(facade.uses_cutedsl_backend(4))


if __name__ == "__main__":
    unittest.main()
