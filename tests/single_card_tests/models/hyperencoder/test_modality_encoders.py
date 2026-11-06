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

"""Unit tests for the HyperEncoder image / audio modality towers."""

import os
import unittest
from unittest import mock

import paddle

# The accuracy-compatible conv path routes conv2d to a kernel registered only
# for GPU; disable it so the towers exercise the standard kernel.
paddle.set_flags({"FLAGS_use_accuracy_compatible_kernel": False})

from paddlefleet.models.hyperencoder.modality_encoders import (
    AudioEncoderConv,
    ImageEncoderConv,
    MlpProjector,
    PatchEmbed,
    _abs_pos_use_legacy_interp,
    _aligned_bilinear_2d,
    get_abs_pos_1d,
    get_abs_pos_2d,
    image_to_patch_rows,
)


class TestMlpProjector(unittest.TestCase):
    def test_identity(self):
        proj = MlpProjector(
            {"projector_type": "identity", "input_dim": 4, "n_embed": 4}
        )
        x = paddle.randn([2, 4])
        self.assertTrue(paddle.allclose(proj(x), x))

    def test_linear_forward_and_backward(self):
        proj = MlpProjector(
            {"projector_type": "linear", "input_dim": 4, "n_embed": 6}
        )
        x = paddle.randn([3, 4])
        x.stop_gradient = False
        out = proj(x)
        self.assertEqual(out.shape, [3, 6])
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(proj.layers.weight.grad)
        self.assertIsNotNone(proj.layers.bias.grad)

    def test_mlp_gelu(self):
        proj = MlpProjector(
            {
                "projector_type": "mlp_gelu",
                "input_dim": 4,
                "n_embed": 5,
                "depth": 2,
            }
        )
        x = paddle.randn([2, 4])
        out = proj(x)
        self.assertEqual(out.shape, [2, 5])

    def test_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            MlpProjector(
                {"projector_type": "nope", "input_dim": 4, "n_embed": 4}
            )

    def test_linear_no_bias(self):
        proj = MlpProjector(
            {"projector_type": "linear", "input_dim": 4, "n_embed": 6}
        )
        proj.layers.bias = None
        x = paddle.randn([3, 4])
        out = proj(x)
        self.assertEqual(out.shape, [3, 6])


class TestPatchEmbed(unittest.TestCase):
    def test_output_is_channel_last(self):
        pe = PatchEmbed(
            kernel_size=(2, 2), stride=(2, 2), in_chans=3, embed_dim=8
        )
        x = paddle.randn([1, 3, 4, 4])
        out = pe(x)
        # [B, H//2, W//2, embed_dim]
        self.assertEqual(out.shape, [1, 2, 2, 8])


class TestAbsPos2d(unittest.TestCase):
    def test_short_circuit_when_sizes_match(self):
        abs_pos = paddle.randn([1, 4, 4, 8])
        out = get_abs_pos_2d(abs_pos, (4, 4))
        self.assertTrue(out is abs_pos)

    def test_interpolates_when_sizes_differ(self):
        abs_pos = paddle.randn([1, 4, 4, 8])
        out = get_abs_pos_2d(abs_pos, (2, 2))
        self.assertEqual(out.shape, [1, 2, 2, 8])


class TestAbsPos2dLegacySwitch(unittest.TestCase):
    """The ``HYPERBODY_ABS_POS_LEGACY_INTERP`` env switch and its two paths."""

    def test_switch_defaults_to_new_path(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HYPERBODY_ABS_POS_LEGACY_INTERP", None)
            self.assertFalse(_abs_pos_use_legacy_interp())

    def test_switch_falsy_values(self):
        for val in ("", "0", "false", "False"):
            with mock.patch.dict(
                os.environ, {"HYPERBODY_ABS_POS_LEGACY_INTERP": val}
            ):
                self.assertFalse(_abs_pos_use_legacy_interp())

    def test_switch_truthy_values(self):
        for val in ("1", "true", "yes", "on"):
            with mock.patch.dict(
                os.environ, {"HYPERBODY_ABS_POS_LEGACY_INTERP": val}
            ):
                self.assertTrue(_abs_pos_use_legacy_interp())

    def test_legacy_branch_matches_shape(self):
        abs_pos = paddle.randn([1, 4, 4, 8])
        with mock.patch.dict(
            os.environ, {"HYPERBODY_ABS_POS_LEGACY_INTERP": "1"}
        ):
            out = get_abs_pos_2d(abs_pos, (2, 2))
        self.assertEqual(out.shape, [1, 2, 2, 8])

    def test_default_and_legacy_agree_on_non_degenerate_grid(self):
        # For a normal (non-degenerate) target grid both paths implement the
        # same align_corners=False bilinear, so results should be very close.
        abs_pos = paddle.randn([1, 6, 6, 8])
        with mock.patch.dict(
            os.environ, {"HYPERBODY_ABS_POS_LEGACY_INTERP": "0"}
        ):
            new_path = get_abs_pos_2d(abs_pos, (3, 3))
        with mock.patch.dict(
            os.environ, {"HYPERBODY_ABS_POS_LEGACY_INTERP": "1"}
        ):
            legacy_path = get_abs_pos_2d(abs_pos, (3, 3))
        self.assertTrue(
            paddle.allclose(new_path, legacy_path, atol=1e-5, rtol=1e-5)
        )

    def test_default_path_handles_degenerate_output_dim(self):
        # A target grid with a size-1 spatial dim is exactly what makes the
        # built-in bilinear kernel raise cudaErrorInvalidValue. The default
        # hand-decomposed path must handle it without crashing.
        abs_pos = paddle.randn([1, 4, 4, 8])
        with mock.patch.dict(
            os.environ, {"HYPERBODY_ABS_POS_LEGACY_INTERP": "0"}
        ):
            out = get_abs_pos_2d(abs_pos, (1, 3))
        self.assertEqual(out.shape, [1, 1, 3, 8])

    def test_aligned_bilinear_2d_short_circuits_on_size_1_axis(self):
        # _aligned_interp_1d_axis / _aligned_bilinear_2d must produce a size-1
        # output axis without dividing by zero or launching a bad kernel.
        old = paddle.randn([1, 8, 4, 4])  # [B, C, H, W]
        out = _aligned_bilinear_2d(old, out_h=1, out_w=2)
        self.assertEqual(out.shape, [1, 8, 1, 2])


class TestAbsPos1d(unittest.TestCase):
    def test_short_circuit_when_sizes_match(self):
        abs_pos = paddle.randn([1, 10, 8])
        out = get_abs_pos_1d(abs_pos, 10)
        self.assertTrue(out is abs_pos)

    def test_interpolates_when_sizes_differ(self):
        abs_pos = paddle.randn([1, 10, 8])
        out = get_abs_pos_1d(abs_pos, 5)
        self.assertEqual(out.shape, [1, 5, 8])


class TestImageEncoderConv(unittest.TestCase):
    def test_forward_and_backward(self):
        enc = ImageEncoderConv(
            img_size=8, patch_size=2, in_chans=3, embed_dim=8, out_chans=16
        )
        x = paddle.randn([1, 3, 8, 8])
        x.stop_gradient = False
        out = enc(x)
        # grid = 8 // 2 = 4
        self.assertEqual(out.shape, [1, 4, 4, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)

    def test_pos_embed_interpolated_when_grid_differs(self):
        enc = ImageEncoderConv(
            img_size=8, patch_size=2, in_chans=3, embed_dim=8, out_chans=16
        )
        # A 12x12 image -> 6x6 grid, forcing the pos-embed interpolation branch.
        x = paddle.randn([1, 3, 12, 12])
        out = enc(x)
        self.assertEqual(out.shape, [1, 6, 6, 8])


class TestImageEncoderConvForwardPatches(unittest.TestCase):
    def test_image_to_patch_rows_layout(self):
        img = paddle.arange(3 * 5 * 7, dtype="float32").reshape([3, 5, 7])
        rows = image_to_patch_rows(img, 2)
        self.assertEqual(rows.shape, [2, 3, 12])
        # Block (1, 2) covers pixels [:, 2:4, 4:6], flattened in (c, u, v).
        ref = img[:, 2:4, 4:6].reshape([-1])
        self.assertTrue(bool((rows[1, 2] == ref).all()))
        # numpy input gives the same layout.
        rows_np = image_to_patch_rows(img.numpy(), 2)
        self.assertTrue(bool((paddle.to_tensor(rows_np) == rows).all()))

    def test_matches_per_image_forward(self):
        paddle.seed(0)
        enc = ImageEncoderConv(
            img_size=8, patch_size=2, in_chans=3, embed_dim=8, out_chans=16
        )
        with paddle.no_grad():
            enc.pos_embed.set_value(paddle.randn(enc.pos_embed.shape))
        enc.to(dtype="float64")
        imgs = [
            paddle.randn([3, h, w], dtype="float64")
            for h, w in [(8, 6), (9, 7), (4, 10), (8, 6)]
        ]
        rows = [image_to_patch_rows(im, 2) for im in imgs]
        grids = [tuple(r.shape[:2]) for r in rows]
        out = enc.forward_patches(
            paddle.concat([r.reshape([-1, 12]) for r in rows], axis=0), grids
        )
        ref = paddle.concat(
            [enc(im.unsqueeze(0)).reshape([-1, 8]) for im in imgs], axis=0
        )
        self.assertEqual(out.shape, ref.shape)
        self.assertLess(float((out - ref).abs().max()), 1e-6)


class TestPatchEmbedForwardPatches(unittest.TestCase):
    def test_rejects_overlapping_or_padded_conv(self):
        x = paddle.zeros([4, 3 * 2 * 2])
        for kw in (
            {"kernel_size": (2, 2), "stride": (1, 1)},
            {"kernel_size": (2, 2), "stride": (2, 2), "padding": (1, 1)},
        ):
            with self.assertRaises(ValueError):
                PatchEmbed(in_chans=3, embed_dim=8, **kw).forward_patches(x)

    def test_empty_input(self):
        enc = ImageEncoderConv(img_size=8, patch_size=2, embed_dim=8)
        out = enc.forward_patches(paddle.zeros([0, 12]), [])
        self.assertEqual(out.shape, [0, 8])


class TestAudioEncoderConvForwardBatch(unittest.TestCase):
    def _enc(self):
        paddle.seed(0)
        enc = AudioEncoderConv(
            num_mel_bins=4, embed_dim=8, max_position_embeddings=64
        )
        with paddle.no_grad():
            for p in enc.parameters():
                p.set_value(paddle.randn(p.shape))
        enc.to(dtype="float64")
        return enc

    def test_matches_per_item_forward(self):
        enc = self._enc()
        # Even / odd lengths, length 1 and 2, and an odd item followed by an
        # even one (the gap then needs two zero frames).
        for lengths in ([6, 7, 1, 40, 13, 6, 2, 33], [1], [7, 2], [2, 1, 3]):
            mels = [paddle.randn([4, t], dtype="float64") for t in lengths]
            ref = paddle.concat(
                [enc(m.unsqueeze(0))[0, 0] for m in mels], axis=0
            )
            out, out_lens = enc.forward_batch(mels)
            self.assertEqual(out_lens, [(t + 1) // 2 for t in lengths])
            self.assertEqual(out.shape, ref.shape)
            self.assertLess(float((out - ref).abs().max()), 1e-10, lengths)

    def test_grads_match_per_item_forward(self):
        lengths = [5, 12, 9]
        mels = [paddle.randn([4, t], dtype="float64") for t in lengths]
        grads = []
        for batched in (False, True):
            enc = self._enc()
            if batched:
                out, _ = enc.forward_batch(mels)
            else:
                out = paddle.concat([enc(m.unsqueeze(0))[0, 0] for m in mels])
            (
                out
                * paddle.arange(out.numel(), dtype="float64").reshape(out.shape)
            ).sum().backward()
            grads.append({n: p.grad.numpy() for n, p in enc.named_parameters()})
        for n in grads[0]:
            scale = float(abs(grads[0][n]).max()) + 1e-30
            self.assertLess(
                float(abs(grads[0][n] - grads[1][n]).max()) / scale, 1e-6, n
            )

    def test_empty_input(self):
        out, out_lens = self._enc().forward_batch([])
        self.assertEqual(out.shape, [0, 8])
        self.assertEqual(out.dtype, paddle.float64)  # the tower's dtype
        self.assertEqual(out_lens, [])


class TestAudioEncoderConv(unittest.TestCase):
    def test_forward_and_backward(self):
        enc = AudioEncoderConv(
            num_mel_bins=4, embed_dim=8, max_position_embeddings=64
        )
        x = paddle.randn([1, 4, 20])
        x.stop_gradient = False
        out = enc(x)
        # T' = (20 + 1) // 2 = 10 ; shape [B, 1, T', embed_dim]
        self.assertEqual(out.shape, [1, 1, 10, 8])
        out.sum().backward()
        self.assertIsNotNone(x.grad)


class TestMuonSliceSpecs(unittest.TestCase):
    """These modules intentionally do not expose a Muon slice-spec hook.

    ``muon_slice_specs`` is only needed for weights that pack several logically
    independent matrices into one tensor (fused QKV / gate-up / stacked experts).
    None of these modules does: the projector is a single ``nn.Linear``, the
    towers are ``Conv`` kernels plus non-matrix positional tables. Muon handles
    them with its default per-weight update, so the hook is deliberately absent;
    this assertion locks that decision in.
    """

    def test_no_slice_hook(self):
        modules = [
            MlpProjector(
                {"projector_type": "linear", "input_dim": 4, "n_embed": 6}
            ),
            PatchEmbed(
                kernel_size=(2, 2), stride=(2, 2), in_chans=3, embed_dim=8
            ),
            ImageEncoderConv(
                img_size=8, patch_size=2, in_chans=3, embed_dim=8, out_chans=16
            ),
            AudioEncoderConv(
                num_mel_bins=4, embed_dim=8, max_position_embeddings=64
            ),
        ]
        for m in modules:
            self.assertFalse(
                hasattr(m, "muon_slice_specs"),
                f"{type(m).__name__} unexpectedly declares muon_slice_specs; "
                "if a fused weight was added, provide a real spec here.",
            )


if __name__ == "__main__":
    unittest.main()
