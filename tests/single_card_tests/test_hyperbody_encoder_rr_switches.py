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

"""Encoder-scoped RR switches (no GPU, no dist init).

* ``_build_encoder_view`` maps ``hyperencoder_recompute_dispatch`` /
  ``..._combine`` onto ``recompute_granularity`` + ``recompute_modules``, and
  the switches win over the top-level granularity;
* ``HyperEncoderProvider.__post_init__`` accepts only the one supported
  recompute mode and refuses module lists that would be silently dead.
"""

import unittest
from unittest import mock

from paddlefleet.transformers.hyperbody import modeling as hb
from paddlefleet.transformers.hyperbody.providers import HyperEncoderProvider


class _Loose:
    """Namespace answering any attribute, so a test pins only what it asserts.

    ``_build_encoder_view`` forwards ~40 geometry/runtime fields; this keeps the
    fixture to the recompute ones instead of restating the whole config.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def __getattr__(self, name):
        return None


def _mapped(dispatch=False, combine=False, flat_granularity=None):
    """Return the recompute kwargs ``_build_encoder_view`` hands the provider."""
    config = _Loose(
        recompute_granularity=flat_granularity,
        encoder_config=_Loose(
            hyperencoder_recompute_dispatch=dispatch,
            hyperencoder_recompute_combine=combine,
            # The only forwarded field that is coerced, so it cannot be None.
            router_aux_loss_coef=0.001,
        ),
    )
    captured = {}

    def fake_cfg(**kwargs):
        captured.update(kwargs)
        return _Loose(**kwargs)

    with (
        mock.patch.object(hb, "HyperEncoderConfig", fake_cfg),
        mock.patch.object(
            hb.HyperEncoderProvider,
            "from_config",
            staticmethod(lambda cfg: _Loose()),
        ),
        mock.patch.object(hb, "_configure_encoder_cp"),
        mock.patch(
            "paddlefleet.models.hyperencoder.attn_backend."
            "use_flashmask_encoder_attn",
            return_value=False,
        ),
    ):
        hb._build_encoder_view(config, decoder_hidden=8)
    return (
        captured["recompute_granularity"],
        captured["recompute_modules"],
    )


class TestEncoderRecomputeMapping(unittest.TestCase):
    def test_no_switch_mirrors_the_flat_granularity(self):
        self.assertEqual(_mapped(), (None, None))
        self.assertEqual(_mapped(flat_granularity="full"), ("full", None))
        # The encoder has no "selective" mode: intent maps onto its only one.
        self.assertEqual(_mapped(flat_granularity="selective"), ("full", None))

    def test_each_switch_adds_only_its_own_module(self):
        self.assertEqual(_mapped(dispatch=True), ("full", ["moe_dispatch"]))
        self.assertEqual(_mapped(combine=True), ("full", ["moe_combine"]))

    def test_both_switches_are_independent(self):
        self.assertEqual(
            _mapped(dispatch=True, combine=True),
            ("full", ["moe_dispatch", "moe_combine"]),
        )

    def test_switches_win_over_recompute_being_off(self):
        """RR can be exercised with the top-level recompute off."""
        self.assertEqual(
            _mapped(dispatch=True, flat_granularity=None),
            ("full", ["moe_dispatch"]),
        )


def _post_init(granularity, modules, **over):
    """Drive only the provider's recompute validation."""
    obj = HyperEncoderProvider.__new__(HyperEncoderProvider)
    obj.recompute_granularity = granularity
    obj.recompute_modules = modules
    obj.recompute_method = over.pop("recompute_method", None)
    obj.recompute_num_layers = over.pop("recompute_num_layers", None)
    assert not over, f"unused kwargs {over}"
    # Stop right after the recompute block: everything below it needs the full
    # geometry, which these cases do not exercise.
    with mock.patch.object(
        HyperEncoderProvider, "_check_divisibility", side_effect=_Stop
    ):
        try:
            HyperEncoderProvider.__post_init__(obj)
        except _Stop:
            pass
    return obj


class _Stop(Exception):
    pass


class TestEncoderProviderRecomputeValidation(unittest.TestCase):
    def test_only_full_or_off_is_accepted(self):
        for granularity in ("selective", "core_attn", "nonsense"):
            with (
                self.subTest(granularity=granularity),
                self.assertRaises(ValueError) as cm,
            ):
                _post_init(granularity, None)
            self.assertIn("recompute_granularity", str(cm.exception))

    def test_method_and_num_layers_are_pinned(self):
        with self.assertRaises(ValueError) as cm:
            _post_init("full", None, recompute_method="first_n")
        self.assertIn("recompute_method", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            _post_init("full", None, recompute_num_layers=2)
        self.assertIn("recompute_num_layers", str(cm.exception))

    def test_modules_without_recompute_raise(self):
        with self.assertRaises(ValueError) as cm:
            _post_init(None, ["moe_dispatch"])
        self.assertIn("needs recompute to be on", str(cm.exception))

    def test_plain_selective_entries_are_refused(self):
        """They would be gated off and silently do nothing under "full"."""
        with self.assertRaises(ValueError) as cm:
            _post_init("full", ["moe_dispatch", "moe_gate_up"])
        self.assertIn("moe_gate_up", str(cm.exception))

    def test_rr_entries_pass_and_derive_uniform_1(self):
        obj = _post_init("full", ["moe_dispatch", "moe_combine"])
        # ``_check_divisibility`` aborted before super().__post_init__, which is
        # where the pinned values are restored, so only the accept/reject
        # decision is asserted here.
        self.assertEqual(obj.recompute_modules, ["moe_dispatch", "moe_combine"])


if __name__ == "__main__":
    unittest.main()
