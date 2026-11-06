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

"""``hyperencoder_batched_patch_embed`` in the HyperBody frontend.

With the switch on, the data side ships every image as patch rows
(``image_to_patch_rows`` -> ``pack_hyperbody_mm``) and the frontend embeds them
with one GEMM (``_build_context_embeds_from_patches``). This checks that path
against the per-segment conv path (switch off) on a mix of image / audio /
media-less segments, in float64 so a layout bug shows up as an O(1) error,
and that the switch is wired from ``encoder_config`` through ``forward``.
"""

import types
import unittest
from unittest import mock

import paddle

from paddlefleet.models.hyperencoder.modality_encoders import (
    AudioEncoderConv,
    ImageEncoderConv,
    MlpProjector,
    image_to_patch_rows,
)
from paddlefleet.transformers.hyperbody.configuration import (
    AUDIO_PATCH_TOKEN,
    IM_PATCH_TOKEN,
)
from paddlefleet.transformers.hyperbody.mm_pack import (
    pack_hyperbody_mm,
    unpack_hyperbody_mm,
)
from paddlefleet.transformers.hyperbody.modeling import (
    HyperBodyEncoderFrontEnd,
    _build_encoder_view,
)

FE = HyperBodyEncoderFrontEnd
EMBED, HIDDEN, PATCH = 8, 4, 2


class _Frontend(paddle.nn.Layer):
    """The four towers the context-embed builders touch, at toy sizes."""

    packed = True

    def __init__(self, batched=False):
        super().__init__()
        self.config = types.SimpleNamespace(
            hyperencoder_batched_patch_embed=batched
        )
        vocab = max(IM_PATCH_TOKEN, AUDIO_PATCH_TOKEN) + 1
        self.embed_tokens = paddle.nn.Embedding(vocab, HIDDEN)
        self.image_encoder = ImageEncoderConv(
            img_size=8, patch_size=PATCH, in_chans=3, embed_dim=EMBED
        )
        self.audio_encoder = AudioEncoderConv(num_mel_bins=128, embed_dim=EMBED)
        self.projector = MlpProjector(
            {"projector_type": "linear", "input_dim": EMBED, "n_embed": HIDDEN}
        )

    _image_keepalive = FE._image_keepalive
    _audio_keepalive = FE._audio_keepalive
    _splice = staticmethod(FE._splice)
    _assemble_segment = FE.__dict__["_assemble_segment"]
    _first_placeholder = staticmethod(FE._first_placeholder)
    _packed_media_layout = staticmethod(FE._packed_media_layout)
    _embed_packed_audio = FE._embed_packed_audio
    _segment_media = FE._segment_media
    _embed_one_context = FE._embed_one_context
    _embed_patch_rows = FE._embed_patch_rows
    forward = FE.forward
    _build_context_embeds = FE._build_context_embeds
    _build_context_embeds_from_patches = FE._build_context_embeds_from_patches


def _make():
    paddle.seed(0)
    m = _Frontend()
    with paddle.no_grad():
        for p in m.parameters():
            p.set_value(paddle.randn(p.shape, dtype="float32"))
    m.to(dtype="float64")
    return m


IMG, AUD, TXT = IM_PATCH_TOKEN, AUDIO_PATCH_TOKEN, 7

# (segment ids, image size, audio frames) per context segment.
PLANS = {
    # Images with the same grid from different raw sizes, smaller and larger
    # than the pos-embed grid; audio of even / odd / repeated lengths; image
    # and audio segments with several placeholders and trailing tokens (both
    # paths truncate at the first placeholder); a media-less segment.
    "mixed": [
        ([1, IMG], (8, 6), None),
        ([1, AUD], None, 6),
        ([1, 2, IMG, IMG, 9], (9, 7), None),
        ([1, TXT], None, None),
        ([1, AUD, AUD, 5], None, 7),
        ([1, IMG], (4, 10), None),
        ([1, AUD], None, 40),
        ([1, AUD], None, 13),
        ([1, IMG], (12, 12), None),
        ([1, AUD], None, 6),
        ([1, AUD], None, 3),
    ],
    # Each tower absent in turn: only its keepalive keeps it in the graph.
    "image_only": [([1, IMG], (8, 6), None), ([1, 2, IMG], (6, 6), None)],
    "audio_only": [([1, AUD], None, 7), ([1, AUD, 3], None, 12)],
    "media_less": [([1, TXT], None, None), ([1, 2, TXT], None, None)],
}


def _batch(plan="mixed"):
    paddle.seed(1)
    ids, imgs, auds, cu = [], [], [], [0]
    for seg, hw, frames in PLANS[plan]:
        ids += seg
        cu.append(len(ids))
        imgs.append(paddle.randn([3, *hw], dtype="float64") if hw else None)
        auds.append(
            paddle.randn([128, frames], dtype="float64") if frames else None
        )
    cu = paddle.to_tensor(cu, dtype="int32")
    return paddle.to_tensor([ids], dtype="int64"), imgs, auds, cu


def _packed_on(batch):
    """``batch`` packed as the switch-on data side would: ``(ids, patches, audio, cu)``."""
    ids, imgs, auds, cu = batch
    rows = [
        None if im is None else image_to_patch_rows(im, PATCH) for im in imgs
    ]
    args = dict(context_ids=ids, **pack_hyperbody_mm(rows, auds))
    unpack_hyperbody_mm(args, image_as_patches=True)
    return ids, args.get("image_patches"), args["audio"], cu


def _backward(m, out):
    sum((o * (i + 1)).sum() for i, o in enumerate(out)).backward()
    grads = {
        n: (None if p.grad is None else p.grad.numpy())
        for n, p in m.named_parameters()
    }
    return [o.numpy() for o in out], grads


def _run_off(m, batch):
    ids, imgs, auds, cu = batch
    args = dict(context_ids=ids, **pack_hyperbody_mm(imgs, auds))
    unpack_hyperbody_mm(args)
    out = m._build_context_embeds(
        ids, image=args["image"], audio=args["audio"], cu_seqlens_context=cu
    )
    return _backward(m, out)


def _run_on(m, batch):
    ids, patches, audio, cu = _packed_on(batch)
    out = m._build_context_embeds_from_patches(ids, patches, audio, cu)
    return _backward(m, out)


class TestBatchedPatchEmbed(unittest.TestCase):
    def test_patch_rows_match_per_segment_conv(self):
        for plan in PLANS:
            with self.subTest(plan=plan):
                self._check_matches(plan)

    def _check_matches(self, plan):
        batch = _batch(plan)
        ref_out, ref_g = _run_off(_make(), batch)
        out, g = _run_on(_make(), batch)
        self.assertEqual(len(out), len(ref_out))
        for a, b in zip(ref_out, out):
            self.assertEqual(a.shape, b.shape)
            self.assertLess(float(abs(a - b).max()), 1e-10)
        for n in ref_g:
            # Both towers must stay in the backward graph on both paths.
            self.assertEqual(ref_g[n] is None, g[n] is None, n)
            if ref_g[n] is None:
                continue
            # get_abs_pos_2d resamples in fp32, so pos_embed agrees only to
            # fp32 noise (one resample per grid instead of one per image).
            tol = 1e-6 if n.endswith("pos_embed") else 1e-10
            scale = float(abs(ref_g[n]).max()) + 1e-30
            self.assertLess(float(abs(ref_g[n] - g[n]).max()) / scale, tol, n)
        # A tower with no real input is kept alive by its shared keepalive:
        # every parameter gets a gradient, and that gradient is exactly zero.
        absent = []
        if all(hw is None for _, hw, _ in PLANS[plan]):
            absent.append("image_encoder.")
        if all(fr is None for _, _, fr in PLANS[plan]):
            absent.append("audio_encoder.")
        for n in g:
            if n.startswith(tuple(absent)):
                self.assertIsNotNone(g[n], n)
                self.assertEqual(float(abs(g[n]).max()), 0.0, n)

    def test_off_never_uses_forward_patches(self):
        with mock.patch.object(
            ImageEncoderConv, "forward_patches", side_effect=AssertionError
        ):
            _run_off(_make(), _batch())

    def test_off_never_uses_forward_batch(self):
        with mock.patch.object(
            AudioEncoderConv, "forward_batch", side_effect=AssertionError
        ):
            _run_off(_make(), _batch())

    def test_on_runs_each_tower_once(self):
        calls = {"img": [], "aud": [], "aud_batch": 0}
        img_fwd, aud_fwd = ImageEncoderConv.forward, AudioEncoderConv.forward
        aud_batch = AudioEncoderConv.forward_batch

        def img_spy(self_, x):
            calls["img"].append(tuple(x.shape))
            return img_fwd(self_, x)

        def aud_spy(self_, x):
            calls["aud"].append(tuple(x.shape))
            return aud_fwd(self_, x)

        def batch_spy(self_, audios, *a, **kw):
            calls["aud_batch"] += 1
            return aud_batch(self_, audios, *a, **kw)

        with (
            mock.patch.object(ImageEncoderConv, "forward", img_spy),
            mock.patch.object(AudioEncoderConv, "forward", aud_spy),
            mock.patch.object(AudioEncoderConv, "forward_batch", batch_spy),
        ):
            _run_on(_make(), _batch())
        # Real images / audio go through forward_patches / forward_batch once;
        # the only per-call tower passes are one shared keepalive each.
        self.assertEqual(calls["img"], [(1, 3, 128, 128)])
        self.assertEqual(calls["aud"], [(1, 128, 10)])
        self.assertEqual(calls["aud_batch"], 1)

    def test_rejects_image_rows_with_wrong_width(self):
        ids, imgs, auds, cu = _batch()
        rows = [None if im is None else im.transpose([1, 2, 0]) for im in imgs]
        args = dict(context_ids=ids, **pack_hyperbody_mm(rows, auds))
        unpack_hyperbody_mm(args, image_as_patches=True)
        with self.assertRaisesRegex(ValueError, "patch rows"):
            _make()._build_context_embeds_from_patches(
                ids, args["image_patches"], auds, cu
            )

    def test_rejects_seg_index_out_of_range(self):
        ids, imgs, auds, cu = _batch()
        rows = [
            None if im is None else image_to_patch_rows(im, PATCH)
            for im in imgs
        ]
        args = dict(context_ids=ids, **pack_hyperbody_mm(rows, auds))
        unpack_hyperbody_mm(args, image_as_patches=True)
        values, shapes, seg_index = args["image_patches"]
        with self.assertRaisesRegex(ValueError, "seg_index"):
            _make()._build_context_embeds_from_patches(
                ids, (values, shapes[:-1], seg_index), auds, cu
            )

    def test_rejects_audio_list_of_wrong_length(self):
        ids, patches, audio, cu = _packed_on(_batch())
        with self.assertRaisesRegex(ValueError, "audio entries"):
            _make()._build_context_embeds_from_patches(
                ids, patches, audio[:-1], cu
            )

    def test_rejects_segment_with_image_and_audio(self):
        ids, patches, audio, cu = _packed_on(_batch())
        audio = list(audio)
        audio[0] = paddle.randn([128, 6], dtype="float64")  # an image segment
        with self.assertRaisesRegex(ValueError, "both an image and audio"):
            _make()._build_context_embeds_from_patches(ids, patches, audio, cu)

    def test_rejects_audio_segment_without_audio_placeholder(self):
        # Segment 3 is media-less ([1, TXT]); give it a mel but no placeholder.
        ids, patches, audio, cu = _packed_on(_batch())
        audio = list(audio)
        audio[3] = paddle.randn([128, 6], dtype="float64")
        with self.assertRaisesRegex(ValueError, "not an audio-only segment"):
            _make()._build_context_embeds_from_patches(ids, patches, audio, cu)

    def test_rejects_audio_segment_with_image_placeholder(self):
        # Segment 1 is [1, AUD]; turn its first token into an image placeholder.
        ids, patches, audio, cu = _packed_on(_batch())
        ids = ids.clone()
        ids[0, 2] = IMG
        with self.assertRaisesRegex(ValueError, "not an audio-only segment"):
            _make()._build_context_embeds_from_patches(ids, patches, audio, cu)


class _Stop(Exception):
    pass


class TestSwitchWiring(unittest.TestCase):
    def test_forward_dispatches_on_switch(self):
        ids, imgs, auds, cu = _batch()
        for batched in (False, True):
            rows = imgs
            if batched:
                rows = [
                    None if im is None else image_to_patch_rows(im, PATCH)
                    for im in imgs
                ]
            m = _Frontend(batched)
            args = dict(
                context_ids=ids,
                cu_seqlens_context=cu,
                **pack_hyperbody_mm(rows, auds),
            )
            with (
                mock.patch.object(
                    _Frontend, "_build_context_embeds", side_effect=_Stop
                ) as off,
                mock.patch.object(
                    _Frontend,
                    "_build_context_embeds_from_patches",
                    side_effect=_Stop,
                ) as on,
                self.assertRaises(_Stop),
            ):
                m.forward(args)
            self.assertEqual((off.called, on.called), (not batched, batched))
            called = on if batched else off
            images = called.call_args.args[1]
            if batched:
                self.assertEqual(
                    len(images[1]), sum(i is not None for i in imgs)
                )
            else:
                self.assertEqual(len(images), len(imgs))

    def test_switch_reaches_the_encoder_view(self):
        from paddlefleet.transformers.hyperbody.configuration import (
            HyperBodyConfig,
        )

        for extra, want in (
            ({"hyperencoder_batched_patch_embed": True}, True),
            ({"hyperencoder_batched_patch_embed": False}, False),
            ({}, False),
        ):
            cfg = HyperBodyConfig(
                decoder_config={**_DECODER_GEOMETRY},
                encoder_config={**_ENCODER_GEOMETRY, **extra},
            )
            view = _build_encoder_view(cfg, _DECODER_GEOMETRY["hidden_size"])
            self.assertIs(view.hyperencoder_batched_patch_embed, want)


# Smallest geometry the HyperBody configs accept (as in
# test_hyperbody_textonly_keepalive.py).
_DECODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "n_routed_experts": 4,
    "moe_intermediate_size": 32,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
}
_ENCODER_GEOMETRY = {
    "vocab_size": 128,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "moe_intermediate_size": 32,
    "n_routed_experts": 4,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "first_k_dense_replace": 1,
    "hyperencoder_query_length": 16,
}


if __name__ == "__main__":
    unittest.main()
