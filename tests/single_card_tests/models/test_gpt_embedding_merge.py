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

"""``GPTEmbedding._merge_multimodal``: replacing placeholder rows with features.

The decoder embedding overwrites whole rows of ``decoder_input`` at image / video
placeholder positions instead of blending element-wise. These tests pin the rows
it overwrites, the ``[B, S]`` token mask it returns, the deepstack join order, and
that gradient reaches the features in placeholder order and nowhere else.
"""

import types
import unittest

import numpy as np
import paddle

from paddlefleet.models.gpt.gpt_embedding import GPTEmbedding

IMG, VID, TXT = 5, 6, 1
H = 4


class _Embedding(paddle.nn.Layer):
    """The real merge / mask / scatter methods with a toy token table."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(
            image_token_id=IMG, video_token_id=VID
        )
        self.embedding = paddle.nn.Embedding(8, H)

    _merge_multimodal = GPTEmbedding._merge_multimodal
    _placeholder_token_mask = GPTEmbedding._placeholder_token_mask
    _scatter_rows = staticmethod(GPTEmbedding._scatter_rows)


def _make(seed=0):
    paddle.seed(seed)
    emb = _Embedding()
    with paddle.no_grad():
        for p in emb.parameters():
            p.set_value(paddle.randn(p.shape, dtype="float32"))
    return emb


def _inputs(ids, dtype="float32"):
    ids = paddle.to_tensor([ids], dtype="int64")
    x = paddle.randn([1, ids.shape[1], H]).astype(dtype)
    x.stop_gradient = False
    return ids, x


class TestImageMerge(unittest.TestCase):
    def test_image_rows_replaced_in_order(self):
        for dtype in ("float32", "bfloat16"):
            with self.subTest(dtype=dtype):
                emb = _make()
                ids, x = _inputs([TXT, IMG, TXT, IMG, IMG, TXT], dtype)
                x0 = x.astype("float32").numpy()
                feat = paddle.randn([3, H]).astype(dtype)
                feat.stop_gradient = False
                out, mask, deepstack = emb._merge_multimodal(
                    {"image_embeds": feat}, ids, x, None, None
                )
                rows = feat.astype("float32").numpy()
                got = out.astype("float32").numpy()[0]
                # placeholders, in position order
                for k, pos in enumerate([1, 3, 4]):
                    self.assertEqual(got[pos].tolist(), rows[k].tolist())
                for pos in (0, 2, 5):
                    self.assertEqual(got[pos].tolist(), x0[0][pos].tolist())
                # token-level mask, not an expanded [B, S, H] one
                self.assertEqual(mask.shape, [1, 6])
                self.assertEqual(
                    mask.numpy().tolist(),
                    [[False, True, False, True, True, False]],
                )
                self.assertIsNone(deepstack)

                g = paddle.randn([1, 6, H])
                (out.astype("float32") * g).sum().backward()
                gx = x.grad.astype("float32").numpy()[0]
                gimg = feat.grad.astype("float32").numpy()
                # the incoming grad is rounded to the tensor dtype, so compare against that
                gd = g.astype(dtype).astype("float32").numpy()[0]
                for pos in (0, 2, 5):
                    self.assertEqual(gx[pos].tolist(), gd[pos].tolist())
                for pos in (1, 3, 4):
                    self.assertEqual(gx[pos].tolist(), [0.0] * H)
                for k, pos in enumerate([1, 3, 4]):
                    self.assertEqual(gimg[k].tolist(), gd[pos].tolist())


class TestImageAndVideoMerge(unittest.TestCase):
    def test_both_modalities_and_deepstack_join(self):
        emb = _make()
        ids, x = _inputs([VID, IMG, TXT, IMG, VID, VID])
        img = paddle.randn([2, H])
        img.stop_gradient = False
        vid = paddle.randn([3, H])
        vid.stop_gradient = False
        # deepstack features are per visual position and masked per modality, so both
        # tensors carry all N_visual rows
        deep_img = [paddle.randn([5, H])]
        deep_vid = [paddle.randn([5, H])]
        for t in deep_img + deep_vid:
            t.stop_gradient = False
        out, mask, deepstack = emb._merge_multimodal(
            {"image_embeds": img, "video_embeds": vid},
            ids,
            x,
            deep_img,
            deep_vid,
        )
        got = out.numpy()[0]
        for k, pos in enumerate([1, 3]):
            self.assertEqual(got[pos].tolist(), img.numpy()[k].tolist())
        for k, pos in enumerate([0, 4, 5]):
            self.assertEqual(got[pos].tolist(), vid.numpy()[k].tolist())
        self.assertEqual(
            mask.numpy().tolist(), [[True, True, False, True, True, True]]
        )
        # deepstack: one row per visual position, image and video merged by position
        joint = deepstack[0].numpy()
        self.assertEqual(joint.shape, (5, H))
        # visual slots 1 and 2 are the two image positions, the rest are video
        for k in range(5):
            want = (deep_img if k in (1, 2) else deep_vid)[0].numpy()[k]
            self.assertEqual(joint[k].tolist(), want.tolist(), k)

        up = paddle.randn([1, 6, H])
        loss = (out * up).sum()
        for joint_out in deepstack:
            loss = loss + joint_out.sum()
        loss.backward()
        # Each feature row gets the upstream grad of its own placeholder position.
        # A second scatter that cleared the image rows, or a row-order mix-up,
        # both survive "is not None".
        for k, pos in enumerate([1, 3]):
            np.testing.assert_array_equal(
                img.grad.numpy()[k], up.numpy()[0][pos]
            )
        for k, pos in enumerate([0, 4, 5]):
            np.testing.assert_array_equal(
                vid.grad.numpy()[k], up.numpy()[0][pos]
            )
        # Deepstack inputs are per visual slot: 1 where the slot is this
        # modality, 0 elsewhere (both joint outputs were summed into the loss).
        expect = np.zeros([5, H], dtype="float32")
        expect[[1, 2]] = 1.0
        np.testing.assert_array_equal(deep_img[0].grad.numpy(), expect)
        expect = np.ones([5, H], dtype="float32")
        expect[[1, 2]] = 0.0
        np.testing.assert_array_equal(deep_vid[0].grad.numpy(), expect)


class TestMergeChecks(unittest.TestCase):
    def test_feature_count_mismatch_raises(self):
        emb = _make()
        ids, x = _inputs([TXT, IMG, TXT])
        with self.assertRaises(ValueError):
            emb._merge_multimodal(
                {"image_embeds": paddle.randn([2, H])}, ids, x, None, None
            )

    def test_input_ids_none_matches_by_embedding(self):
        emb = _make()
        ids, x = _inputs([TXT, IMG, TXT], "float32")
        with paddle.no_grad():
            x[0, 1] = emb.embedding(paddle.to_tensor([IMG], dtype="int64"))[0]
        feat = paddle.randn([1, H])
        out, mask, _ = emb._merge_multimodal(
            {"image_embeds": feat}, None, x, None, None
        )
        self.assertEqual(mask.numpy().tolist(), [[False, True, False]])
        self.assertEqual(out.numpy()[0][1].tolist(), feat.numpy()[0].tolist())

    def test_keepalive_is_added_and_carries_gradient(self):
        emb = _make()
        ids, x = _inputs([TXT, TXT], "float32")
        keep = paddle.full([], 0.0, dtype="float32")
        keep.stop_gradient = False
        out, mask, deepstack = emb._merge_multimodal(
            {"multimodal_keepalive": keep}, ids, x, None, None
        )
        self.assertEqual(out.numpy().tolist(), x.numpy().tolist())
        self.assertIsNone(mask)
        self.assertIsNone(deepstack)
        out.sum().backward()
        self.assertIsNotNone(keep.grad)


if __name__ == "__main__":
    unittest.main()
