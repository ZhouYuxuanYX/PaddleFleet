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

"""Unit coverage for the DSpark wiring of ``HyperBodyUnifiedModel``.

Covers, on lightweight stand-ins (no GPU / distributed init):

  * ``get_layer_desc_list``: with DSpark the decoder embedding and LM head
    become ``SharedLayerDesc`` and the embedding / LM head pass-throughs, the
    draft and the DSpark loss sit between the MTP layers and the LM head;
    without DSpark the layout is unchanged;
  * ``_setup_dspark_capture``: only decoder (``model.*``) target layers are
    marked, the final norm is flagged, the decoder embedding hook hands the
    host metadata to the draft, and incomplete wiring raises;
  * ``_iter_transformer_layers`` walks the draft's blocks.
"""

import types
import unittest
from unittest.mock import MagicMock, patch

from paddlefleet.transformer.paddle_norm import WrappedPaddleNormPipe
from paddlefleet.transformers.hyperbody import modeling
from paddlefleet.transformers.hyperbody.modeling import (
    DsparkPredictionLayer,
    GPTEmbedding,
    HyperBodySublayersSpec,
    HyperBodyUnifiedModel,
    TransformerLayer,
)


class _LD:
    def __init__(self, layer_func, *args, **kwargs):
        self.layer_func = layer_func


class _SLD:
    def __init__(self, key, layer_func, forward_func=None, **kwargs):
        self.key = key
        self.layer_func = layer_func
        self.forward_func = forward_func
        self.kwargs = kwargs


class _FakeUnifiedModel:
    """Borrows the methods under test without building a PipelineLayer."""

    add_sequential_layer = staticmethod(
        HyperBodyUnifiedModel.add_sequential_layer
    )
    get_layer_desc_list = HyperBodyUnifiedModel.get_layer_desc_list
    _add_decoder_embedding = HyperBodyUnifiedModel._add_decoder_embedding
    _add_mtp_layers = HyperBodyUnifiedModel._add_mtp_layers
    _add_dspark_layers = HyperBodyUnifiedModel._add_dspark_layers
    _add_lm_head = HyperBodyUnifiedModel._add_lm_head
    _setup_dspark_capture = HyperBodyUnifiedModel._setup_dspark_capture
    _iter_decoder_layers = HyperBodyUnifiedModel._iter_decoder_layers
    _iter_transformer_layers = HyperBodyUnifiedModel._iter_transformer_layers

    def __init__(self, **config):
        self.config = types.SimpleNamespace(**config)


def _spec(dspark=None, mtp=None):
    return HyperBodySublayersSpec(
        encoder_frontend="frontend",
        encoder_layers=["enc0"],
        bridge="bridge",
        decoder_embedding="emb",
        decoder_layers=["dec0", "dec1"],
        mtp=mtp,
        layer_norm="norm",
        lm_head="head",
        dspark=dspark,
        dspark_loss="dspark_loss" if dspark else None,
    )


def _layout(layers):
    out = []
    for x in layers:
        d = x["layer"]
        out.append(
            (
                x["name_prefix"],
                getattr(d, "key", None),
                d.layer_func,
                getattr(getattr(d, "forward_func", None), "__name__", None),
            )
        )
    return out


@patch.object(modeling, "SharedLayerDesc", _SLD)
@patch.object(modeling, "LayerDesc", _LD)
class TestDsparkLayerDescList(unittest.TestCase):
    def test_without_dspark_layout_is_unchanged(self):
        layers = _FakeUnifiedModel().get_layer_desc_list(_spec())
        self.assertEqual(
            _layout(layers),
            [
                ("encoder", None, "frontend", None),
                ("encoder.layers.0", None, "enc0", None),
                ("encoder.bridge", None, "bridge", None),
                ("model", None, "emb", None),
                ("model.layers.0", None, "dec0", None),
                ("model.layers.1", None, "dec1", None),
                ("model", None, "norm", None),
                ("model.lm_head", None, "head", None),
            ],
        )

    def test_dspark_shares_embedding_and_head(self):
        model = _FakeUnifiedModel(mtp_shared_last_layer=False)
        layers = model.get_layer_desc_list(
            _spec(dspark=["draft0", "draft1"], mtp=["mtp0"])
        )
        self.assertEqual(
            _layout(layers)[3:],
            [
                ("model", "embed", "emb", None),
                ("model.layers.0", None, "dec0", None),
                ("model.layers.1", None, "dec1", None),
                ("model", None, "norm", None),
                ("model.layers.2", None, "mtp0", None),
                ("model", "embed", "emb", "dspark_share_embedding"),
                (
                    "model.lm_head",
                    "dspark_lm_head",
                    "head",
                    "dspark_share_lm_head",
                ),
                ("model.layers.3", None, "draft0", None),
                ("model.layers.4", None, "draft1", None),
                ("model.dspark_loss", None, "dspark_loss", None),
                ("model.lm_head", "dspark_lm_head", "head", None),
            ],
        )

    def test_mtp_shared_weights_key(self):
        model = _FakeUnifiedModel(
            mtp_shared_last_layer=False, mtp_shared_weights=True
        )
        layers = model.get_layer_desc_list(_spec(mtp=["mtp0", "mtp1"]))
        mtp = [x["layer"] for x in layers if x["layer"].layer_func[:3] == "mtp"]
        self.assertEqual([d.key for d in mtp], ["mtp_shared_all"] * 2)


class _TL(TransformerLayer):
    def __init__(self, layer_number):
        self.layer_number = layer_number


class _Emb(GPTEmbedding):
    def __init__(self):
        self.hooks = []

    def register_forward_pre_hook(self, hook, with_kwargs=False):
        self.hooks.append(hook)


def _bare(cls):
    return cls.__new__(cls)


def _capture_model(target_ids, with_draft=True):
    """Encoder layer 40 shares its number with decoder layer 40."""
    enc, emb, dec40, dec41 = _TL(40), _Emb(), _TL(40), _TL(41)
    norm, draft = _bare(WrappedPaddleNormPipe), _bare(DsparkPredictionLayer)
    slots = [
        ("encoder.layers.40", enc),
        ("model", emb),
        ("model.layers.40", dec40),
        ("model.layers.41", dec41),
        ("model", norm),
    ]
    if with_draft:
        slots.append(("model.layers.42", draft))
    model = _FakeUnifiedModel(dspark_target_layer_ids=target_ids)
    model._sequential_layers = [{"name_prefix": p} for p, _ in slots]
    model.run_function = [layer for _, layer in slots]
    return model, enc, emb, dec40, dec41, norm, draft


@patch.object(modeling, "dspark_layers_active", lambda config: True)
class TestSetupDsparkCapture(unittest.TestCase):
    def test_marks_decoder_targets_and_final_norm(self):
        model, enc, emb, dec40, dec41, norm, _ = _capture_model([40, 41])
        model._setup_dspark_capture()
        self.assertFalse(hasattr(enc, "_dspark_target_id"))
        self.assertEqual(dec40._dspark_target_id, 40)
        self.assertEqual(dec41._dspark_target_id, 41)
        self.assertTrue(norm._dspark_is_final_norm)
        self.assertEqual(len(emb.hooks), 1)

    def test_hook_hands_host_meta_to_draft(self):
        model, _, emb, _, _, _, draft = _capture_model([41])
        model._setup_dspark_capture()
        hook = emb.hooks[0]
        hook(
            emb, ({"input_ids": 1, "labels": 2, "startend_row_indices": 3},), {}
        )
        self.assertEqual(
            draft._host_meta,
            {"input_ids": 1, "labels": 2, "attn_mask_startend_row_indices": 3},
        )
        hook(
            emb,
            (),
            {"dict_args": {"attn_mask_startend_row_indices": 4}},
        )
        self.assertEqual(draft._host_meta["attn_mask_startend_row_indices"], 4)
        self.assertIsNone(draft._host_meta["input_ids"])
        draft._host_meta = "untouched"
        hook(emb, (), {})
        self.assertEqual(draft._host_meta, "untouched")

    def test_missing_target_layer_raises(self):
        model = _capture_model([40, 42])[0]
        with self.assertRaisesRegex(
            RuntimeError, r"missing_target_layers=\[42\]"
        ):
            model._setup_dspark_capture()

    def test_missing_draft_raises(self):
        model = _capture_model([40], with_draft=False)[0]
        with self.assertRaisesRegex(RuntimeError, "draft=False"):
            model._setup_dspark_capture()


class TestSetupDsparkCaptureInactive(unittest.TestCase):
    def test_noop_without_dspark(self):
        model, _, emb, dec40, _, norm, _ = _capture_model([40])
        with patch.object(modeling, "dspark_layers_active", lambda c: False):
            model._setup_dspark_capture()
        self.assertFalse(hasattr(dec40, "_dspark_target_id"))
        self.assertEqual(emb.hooks, [])


class TestIterTransformerLayersDraft(unittest.TestCase):
    def test_yields_draft_blocks(self):
        dec = MagicMock(spec=TransformerLayer)
        draft = MagicMock(spec=DsparkPredictionLayer)
        draft.layers = [MagicMock(), MagicMock()]
        model = _FakeUnifiedModel()
        model.run_function = [MagicMock(), dec, draft, MagicMock()]
        self.assertEqual(
            list(model._iter_transformer_layers()), [dec, *draft.layers]
        )


if __name__ == "__main__":
    unittest.main()
