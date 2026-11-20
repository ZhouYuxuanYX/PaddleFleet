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
"""Config-level tests for the DeepEP V2.5 backend (no GPU, no process group).

Covers what can be checked without a communicator:
  * ``deepep_v2_buffer_configs`` key/value/cross-field validation in
    ``TransformerConfig.__post_init__``,
  * ``configure_buffer_v2`` argument validation and its reset-to-defaults behaviour,
  * ``_required_num_tokens`` bound enforcement and alignment.

The dispatch/combine numerics need real ranks and live in
``tests/multi_card_tests/moe/test_deepep_v2_dispatcher_ep.py``.
"""

import dataclasses
import unittest

from paddlefleet.transformer.moe import deepep_v2
from paddlefleet.transformer.transformer_config import TransformerConfig

_BASE = {
    "hidden_size": 256,
    "num_attention_heads": 8,
    "num_hidden_layers": 2,
}


class TestDeepEPV2BufferConfigValidation(unittest.TestCase):
    """``deepep_v2_buffer_configs`` is user facing, so every key is validated eagerly."""

    def _config(self, buffer_configs, **kwargs):
        return TransformerConfig(
            **_BASE, deepep_v2_buffer_configs=buffer_configs, **kwargs
        )

    def test_accepts_every_documented_key(self):
        cfg = self._config(
            {
                "num_max_tokens_per_rank": 16384,
                "num_sms": 32,
                "do_expand": False,
                "expert_alignment": 128,
                "deterministic": True,
            }
        )
        self.assertEqual(
            cfg.deepep_v2_buffer_configs["num_max_tokens_per_rank"], 16384
        )

    def test_accepts_none_and_empty(self):
        self.assertIsNone(self._config(None).deepep_v2_buffer_configs)
        self.assertEqual(self._config({}).deepep_v2_buffer_configs, {})

    def test_rejects_non_dict(self):
        with self.assertRaisesRegex(ValueError, "must be a dict"):
            self._config([("num_sms", 8)])

    def test_rejects_unknown_key(self):
        # A typo must not be silently ignored: the backend only reads the keys it knows.
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            self._config({"num_max_tokens_pre_rank": 16384})

    def test_rejects_non_positive_token_bound(self):
        for bad in (0, -1):
            with (
                self.subTest(bound=bad),
                self.assertRaisesRegex(ValueError, "num_max_tokens_per_rank"),
            ):
                self._config({"num_max_tokens_per_rank": bad})

    def test_rejects_negative_num_sms(self):
        with self.assertRaisesRegex(ValueError, "num_sms"):
            self._config({"num_sms": -1})

    def test_zero_num_sms_is_the_auto_sentinel(self):
        self.assertEqual(
            self._config({"num_sms": 0}).deepep_v2_buffer_configs["num_sms"], 0
        )

    def test_rejects_expert_alignment_not_multiple_of_four(self):
        for bad in (0, -4, 6, 130):
            with (
                self.subTest(alignment=bad),
                self.assertRaisesRegex(ValueError, "expert_alignment"),
            ):
                self._config({"expert_alignment": bad})

    def test_do_expand_rejects_sonic_moe(self):
        with self.assertRaisesRegex(ValueError, "using_sonic_moe"):
            self._config({"do_expand": True}, using_sonic_moe=True)

    def test_do_expand_requires_fusion_node(self):
        with self.assertRaisesRegex(ValueError, "moe_use_fusion_node"):
            self._config({"do_expand": True}, moe_use_fusion_node=False)

    def test_do_expand_ok_with_fusion_node_and_no_sonic(self):
        cfg = self._config(
            {"do_expand": True}, moe_use_fusion_node=True, using_sonic_moe=False
        )
        self.assertTrue(cfg.deepep_v2_buffer_configs["do_expand"])


class TestConfigureBufferV2(unittest.TestCase):
    """``configure_buffer_v2`` is the only writer of the backend's module-level config."""

    def setUp(self):
        self._saved = dict(deepep_v2._config)

    def tearDown(self):
        deepep_v2._config.clear()
        deepep_v2._config.update(self._saved)

    def test_rejects_invalid_values(self):
        for kwargs, token in (
            ({"num_max_tokens_per_rank": 0}, "num_max_tokens_per_rank"),
            ({"num_max_tokens_per_rank": -8}, "num_max_tokens_per_rank"),
            ({"num_sms": -2}, "num_sms"),
            ({"expert_alignment": 6}, "expert_alignment"),
            ({"expert_alignment": 0}, "expert_alignment"),
        ):
            with (
                self.subTest(**kwargs),
                self.assertRaisesRegex(ValueError, token),
            ):
                deepep_v2.configure_buffer_v2(**kwargs)

    def test_resets_unnamed_keys_to_defaults(self):
        # A layer that configures only one key must not inherit another layer's layout:
        # the expanding layout and determinism change the on-wire format, and a stale value
        # would make a later layer's backward disagree with its forward.
        deepep_v2.configure_buffer_v2(do_expand=True, deterministic=True)
        self.assertTrue(deepep_v2.is_expand_enabled())

        deepep_v2.configure_buffer_v2(num_sms=16)
        self.assertEqual(deepep_v2._config["num_sms"], 16)
        self.assertFalse(deepep_v2.is_expand_enabled())
        self.assertFalse(deepep_v2._config["deterministic"])

    def test_empty_call_restores_every_default(self):
        deepep_v2.configure_buffer_v2(
            num_max_tokens_per_rank=1024, num_sms=8, do_expand=True
        )
        deepep_v2.configure_buffer_v2()
        self.assertEqual(deepep_v2._config, deepep_v2._DEFAULTS)

    def test_kernel_config_snapshot_is_frozen(self):
        # What isolates two models built in one process (RL actor and reference): each
        # layer freezes the dispatch geometry when it is built, so a later
        # configure_buffer_v2 cannot move the earlier layer's layout.
        deepep_v2.configure_buffer_v2(
            num_sms=16, do_expand=True, expert_alignment=8
        )
        snapshot = deepep_v2.kernel_config_snapshot()
        self.assertEqual((snapshot.num_sms, snapshot.do_expand), (16, True))
        self.assertEqual(snapshot.expert_alignment, 8)

        deepep_v2.configure_buffer_v2(num_sms=32)
        self.assertEqual((snapshot.num_sms, snapshot.do_expand), (16, True))
        self.assertEqual(snapshot.expert_alignment, 8)
        self.assertEqual(
            deepep_v2.kernel_config_snapshot().num_sms,
            32,
            "a new snapshot follows the current config",
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.num_sms = 1  # nobody can edit a layer's layout in place


class TestRequiredNumTokens(unittest.TestCase):
    """Bound enforcement and alignment, exercised without a process group."""

    def setUp(self):
        self._saved = dict(deepep_v2._config)

    def tearDown(self):
        deepep_v2._config.clear()
        deepep_v2._config.update(self._saved)

    def test_static_bound_is_returned_as_is(self):
        deepep_v2.configure_buffer_v2(num_max_tokens_per_rank=16640)
        # group is unused on this path: a configured bound skips the all-reduce entirely.
        self.assertEqual(deepep_v2._required_num_tokens(None, 16384), 16640)

    def test_exceeding_the_static_bound_raises_with_both_numbers(self):
        deepep_v2.configure_buffer_v2(num_max_tokens_per_rank=8192)
        with self.assertRaises(RuntimeError) as cm:
            deepep_v2._required_num_tokens(None, 18608)
        message = str(cm.exception)
        self.assertIn("18608", message)
        self.assertIn("8192", message)


if __name__ == "__main__":
    unittest.main()
