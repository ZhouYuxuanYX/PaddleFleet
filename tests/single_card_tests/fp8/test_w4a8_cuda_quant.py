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

"""Bit-exact parity tests for the fused W4A8 1x32 CUDA custom ops."""

import os
import sys
import unittest

import numpy as np
import paddle

from paddlefleet.transformer.moe.fp8_utils import _mxfp8_weighted_swiglu_quant

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from w4a8_quant_reference import (
    fuse_stack_fp8_quant_python,
    fuse_stack_transpose_fp8_quant_python,
    fuse_weighted_swiglu_fp8_quant_clamp_python,
    fuse_weighted_swiglu_fp8_quant_python,
    fused_act_dequant_python,
    mxfp4_stack_dequant_python,
    quant_blockwize,
)


def _prepare_blackwell():
    if (
        not paddle.is_compiled_with_cuda()
        or paddle.device.cuda.device_count() == 0
    ):
        return False
    try:
        paddle.set_device("gpu:0")
        return paddle.device.cuda.get_device_capability()[0] >= 10
    except (RuntimeError, ValueError):
        return False


IS_BLACKWELL = _prepare_blackwell()

if IS_BLACKWELL:
    from paddlefleet_ops import (
        mxfp4_quantize_1x32,
        mxfp4_stack_fake_quantize_1x32,
        mxfp4_stack_quantize_1x32,
        mxfp4_stack_transpose_dequantize_1x32,
        mxfp8_dequantize_1x32,
        mxfp8_quantize_1x32,
        mxfp8_weighted_swiglu_quantize_1x32,
    )


def _fp8_bits(value):
    return value.view("int8")


def _special_value_weights():
    """[2, 64, 256] weights whose first columns hit the quant edge cases.

    Each column of a 32-row group is one 1x32 block of the transpose=True
    quant (blocks along R).
    """
    midpoints = np.array(
        [0.0, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75, 2.5]
        + [-2.5, 3.5, -3.5, 5.0, -5.0, 6.0]
        + [0.0] * 16,
        dtype=np.float32,
    )
    value = np.random.randn(2, 64, 256).astype(np.float32) * 3.0
    value[:, :32, 0] = midpoints  # strict E2M1 midpoints at scale 1
    value[:, :32, 1] = midpoints * 2.0**-3  # the same at scale 2^-3
    value[:, :32, 2] *= 0.01
    value[:, 7, 2] = 6.5  # amax just above 6 * 2^k: the scale carries
    value[:, :32, 3] = 1.0e-6  # below the 1e-4 amax floor
    value[:, :32, 4] = 0.0
    value[:, :32, 5] = -0.0
    value[:, 3, 6] = 1.0e30
    value[:, 3, 7] = np.inf
    value[:, 5, 8] = -np.inf
    value[:, 9, 9] = np.nan
    return paddle.to_tensor(value).astype("bfloat16")


@unittest.skipUnless(IS_BLACKWELL, "requires a Blackwell CUDA device")
class TestW4A8CudaQuantParity(unittest.TestCase):
    def setUp(self):
        paddle.seed(23)
        np.random.seed(23)

    def assert_tensor_equal(self, actual, expected, message):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(list(actual.shape), list(expected.shape))
        self.assertTrue(bool((actual == expected).all()), message)

    def assert_fp8_equal(self, actual, expected):
        self.assert_tensor_equal(
            _fp8_bits(actual), _fp8_bits(expected), "FP8 payload differs"
        )

    def test_quantize_fp8(self):
        for shape in ([1, 32], [255, 32], [256, 32], [257, 32], [37, 256]):
            with self.subTest(shape=shape):
                value = paddle.randn(shape, dtype="bfloat16") * 3.0
                expected_q, expected_scale = quant_blockwize(
                    value, quant_dtype="fp8"
                )
                actual_q, actual_scale = mxfp8_quantize_1x32(value)
                self.assert_fp8_equal(actual_q, expected_q)
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )

    def test_quantize_misaligned_contiguous_bf16(self):
        storage = paddle.randn([257 * 32 + 1], dtype="bfloat16")
        value = storage[1:].reshape([257, 32])
        self.assertTrue(value.is_contiguous())
        expected_q, expected_scale = quant_blockwize(value, quant_dtype="fp8")
        actual_q, actual_scale = mxfp8_quantize_1x32(value)
        self.assert_fp8_equal(actual_q, expected_q)
        self.assert_tensor_equal(
            actual_scale, expected_scale, "misaligned UE8M0 scale differs"
        )

    def test_quantize_fp4_random_and_midpoints(self):
        random_value = paddle.randn([17, 256], dtype="bfloat16") * 3.0
        midpoints = np.array(
            [
                0.0,
                0.25,
                -0.25,
                0.75,
                -0.75,
                1.25,
                -1.25,
                1.75,
                -1.75,
                2.5,
                -2.5,
                3.5,
                -3.5,
                5.0,
                -5.0,
                6.0,
            ]
            + [0.0] * 16,
            dtype=np.float32,
        )
        midpoint_value = paddle.to_tensor(
            np.tile(midpoints[:32], (1, 8)), dtype="float32"
        )
        for value in (random_value, midpoint_value):
            with self.subTest(shape=list(value.shape)):
                expected_q, expected_scale = quant_blockwize(
                    value, quant_dtype="fp4"
                )
                actual_q, actual_scale = mxfp4_quantize_1x32(value)
                self.assert_tensor_equal(
                    actual_q, expected_q, "packed FP4 differs"
                )
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )

    def test_stack_fp4(self):
        weights = paddle.randn([3, 64, 256], dtype="bfloat16")
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                reference = (
                    fuse_stack_transpose_fp8_quant_python
                    if transpose
                    else fuse_stack_fp8_quant_python
                )
                expected_q, expected_scale = reference(
                    weights, quant_dtype="fp4"
                )
                actual_q, actual_scale = mxfp4_stack_quantize_1x32(
                    weights, transpose
                )
                self.assert_tensor_equal(
                    actual_q, expected_q, "packed FP4 differs"
                )
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )

    def test_stack_fp4_large_bf16_fast_path(self):
        """Large stacked weights exercise the BF16 transpose fast path."""
        weights = paddle.randn([4, 256, 512], dtype="bfloat16")
        for transpose in (False, True):
            with self.subTest(transpose=transpose):
                reference = (
                    fuse_stack_transpose_fp8_quant_python
                    if transpose
                    else fuse_stack_fp8_quant_python
                )
                expected_q, expected_scale = reference(
                    weights, quant_dtype="fp4"
                )
                actual_q, actual_scale = mxfp4_stack_quantize_1x32(
                    weights, transpose
                )
                self.assert_tensor_equal(
                    actual_q, expected_q, "packed FP4 differs"
                )
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )

    def test_weighted_swiglu_fp8(self):
        value = paddle.randn([32, 512], dtype="bfloat16") * 2.0
        probs = paddle.rand([32], dtype="float32")
        for clamp_value in (0.0, 10.0):
            with self.subTest(clamp_value=clamp_value):
                if clamp_value:
                    expected_q, expected_scale = (
                        fuse_weighted_swiglu_fp8_quant_clamp_python(
                            value, probs, clamp_value
                        )
                    )
                else:
                    expected_q, expected_scale = (
                        fuse_weighted_swiglu_fp8_quant_python(value, probs)
                    )
                actual_q, actual_scale = mxfp8_weighted_swiglu_quantize_1x32(
                    value, probs, clamp_value
                )
                self.assert_fp8_equal(actual_q, expected_q)
                self.assert_tensor_equal(
                    actual_scale, expected_scale, "UE8M0 scale differs"
                )
                dispatched_q, dispatched_scale = _mxfp8_weighted_swiglu_quant(
                    value,
                    probs,
                    clamp_value,
                )
                self.assert_fp8_equal(dispatched_q, expected_q)
                self.assert_tensor_equal(
                    dispatched_scale,
                    expected_scale,
                    "runtime-dispatched UE8M0 scale differs",
                )

    def test_dequantize_fp8(self):
        value = paddle.randn([37, 256], dtype="bfloat16")
        quantized, scale = quant_blockwize(value, quant_dtype="fp8")
        expected = fused_act_dequant_python(quantized, scale)
        actual = mxfp8_dequantize_1x32(quantized, scale)
        self.assert_tensor_equal(actual, expected, "BF16 dequant differs")

    def assert_bf16_bits_equal(self, actual, expected, message):
        self.assertEqual(actual.dtype, paddle.bfloat16)
        self.assertEqual(list(actual.shape), list(expected.shape))
        self.assertTrue(
            bool((actual.view("int16") == expected.view("int16")).all()),
            message,
        )

    def test_stack_dequantize_fp4_every_byte(self):
        """Every packed byte and a spread of scales, incl. partial tiles."""
        for experts, cols, rows in ((3, 64, 256), (2, 40, 96)):
            with self.subTest(shape=[experts, rows, cols]):
                packed = paddle.to_tensor(
                    np.random.randint(
                        -128, 128, size=(experts, cols, rows // 2)
                    ).astype(np.int8)
                )
                scale = paddle.to_tensor(
                    np.exp2(
                        np.random.randint(
                            -20, 21, size=(experts, cols, rows // 32)
                        )
                    ).astype(np.float32)
                )
                actual = mxfp4_stack_transpose_dequantize_1x32(packed, scale)
                expected = mxfp4_stack_dequant_python(packed, scale)
                self.assert_bf16_bits_equal(
                    actual, expected, "BF16 FP4 dequant differs"
                )

    def test_stack_fake_quantize_fp4(self):
        """fake quant == dequant(mxfp4_stack_quantize_1x32(transpose=True)).

        Covers the fused bf16 kernel (incl. a partial column tile and edge
        values such as E2M1 midpoints, scale carry, the amax floor and +-Inf /
        NaN), and the float32 and unaligned-bf16 paths that quantize first.
        """
        storage = paddle.randn([2 * 64 * 256 + 1], dtype="bfloat16")
        unaligned = storage[1:].reshape([2, 64, 256])
        self.assertTrue(unaligned.is_contiguous())
        cases = {
            "bf16": paddle.randn([3, 64, 256], dtype="bfloat16"),
            "bf16_partial_tile": paddle.randn([2, 96, 160], dtype="bfloat16"),
            "bf16_large": paddle.randn([4, 256, 512], dtype="bfloat16"),
            "bf16_special_values": _special_value_weights(),
            "float32": paddle.randn([3, 64, 256], dtype="float32"),
            "bf16_unaligned": unaligned,
        }
        for name, weights in cases.items():
            with self.subTest(case=name):
                packed, scale = mxfp4_stack_quantize_1x32(weights, True)
                expected = mxfp4_stack_dequant_python(packed, scale)
                actual = mxfp4_stack_fake_quantize_1x32(weights)
                self.assert_bf16_bits_equal(
                    actual, expected, "BF16 FP4 fake quant differs"
                )
                self.assert_bf16_bits_equal(
                    mxfp4_stack_transpose_dequantize_1x32(packed, scale),
                    expected,
                    "BF16 FP4 dequant differs",
                )

    def test_stack_fp4_dqdgrad_ops_reject_bad_shapes(self):
        # PD_CHECK failures surface as OSError
        with self.assertRaisesRegex(OSError, "divisible by 32"):
            mxfp4_stack_fake_quantize_1x32(
                paddle.randn([2, 48, 64], dtype="bfloat16")
            )
        packed = paddle.zeros([2, 64, 16], dtype="int8")
        with self.assertRaisesRegex(OSError, r"scale shape must be"):
            mxfp4_stack_transpose_dequantize_1x32(
                packed, paddle.ones([2, 64, 2], dtype="float32")
            )


if __name__ == "__main__":
    unittest.main()
