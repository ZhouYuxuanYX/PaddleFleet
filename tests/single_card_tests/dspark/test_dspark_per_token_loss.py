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

"""Per-token gradient carrier and logging of ``DSparkLanguageLoss``.

Runs ``forward`` on a stand-in with ``compute_dspark_loss`` and the
communication stubbed, so the scale handed to ``DSparkLossAutoScaler``, the
normalization group and the logged value are checked without GPUs:

  * mean mode injects ``factor * loss`` and logs the same value;
  * per-token mode normalizes over the sharding group, injects
    ``factor * loss * T_global`` and logs the loss rescaled to the local
    denominators (the mean mode's normalization).
"""

import types
import unittest
from unittest.mock import MagicMock, patch

import paddle

from paddlefleet.models.common.language_loss import language_loss
from paddlefleet.models.common.language_loss.language_loss import (
    DSparkLanguageLoss,
)
from paddlefleet.transformer import parallel_multi_token_prediction as pmtp
from paddlefleet.utils.dspark import dspark_losses, dspark_utils

_SHARDING_GROUP = object()
_SHARDING_RANKS = 2


class _FakeLoss:
    """Borrows the methods under test without building a FleetLayer."""

    forward = DSparkLanguageLoss.forward
    _global_main_token_count = DSparkLanguageLoss._global_main_token_count
    _per_cp_group_dspark_loss = staticmethod(
        DSparkLanguageLoss._per_cp_group_dspark_loss
    )

    def __init__(self, per_token, factor=2.0):
        self.calculate_per_token_loss = per_token
        self.ignored_index = -100
        self.config = types.SimpleNamespace(
            add_dspark_loss=True, dspark_loss_scaling_factor=factor
        )


def _f(x):
    return paddle.to_tensor(x, dtype="float32")


def _terms():
    # total = 1.0 (non-selector) + 0.5 * 0.4 (selector) = 1.2
    return types.SimpleNamespace(
        loss=_f(1.2),
        terms={
            "loss": _f(1.2),
            "selector_loss": _f(0.4),
            "selector_loss_alpha": _f(0.5),
            "weight_denominator": _f(10.0),
            "normalized_weight_denominator": _f(20.0),
            "selector_weight_denominator": _f(4.0),
            "normalized_selector_weight_denominator": _f(8.0),
        },
    )


def _all_reduce(tensor, group=None):
    # Every sharding rank holds the same count.
    assert group is _SHARDING_GROUP
    tensor.set_value(tensor * _SHARDING_RANKS)


class TestDsparkPerTokenLoss(unittest.TestCase):
    def _run(self, per_token, labels=None):
        seen = {}

        def _compute(output, profile, return_terms, normalization_group):
            seen["group"] = normalization_group
            return _terms()

        def _inject(hidden, scale):
            seen["inject"] = float(scale)
            return hidden

        hcg = MagicMock()
        hcg.get_sharding_parallel_group.return_value = _SHARDING_GROUP
        logged = []
        dict_args = {
            "dspark_out": {
                "target_ids": None,
                "eval_mask": None,
                "block_keep_mask": None,
            },
            "hidden_states": _f(0.0),
        }
        if labels is not None:
            dict_args["labels"] = labels
        with (
            patch.object(dspark_losses, "compute_dspark_loss", _compute),
            patch.object(dspark_utils, "DrafterOutput", types.SimpleNamespace),
            patch.object(pmtp.DSparkLossAutoScaler, "apply", _inject),
            patch.object(
                pmtp.DSparkLossLoggingHelper,
                "save_loss_to_tracker",
                lambda v: logged.append(float(v)),
            ),
            patch.object(language_loss, "_publish_dspark_loss", MagicMock()),
            patch.object(language_loss, "_publish_dspark_terms", MagicMock()),
            patch.object(
                language_loss.fleet,
                "get_hybrid_communicate_group",
                lambda: hcg,
            ),
            patch.object(language_loss.dist, "all_reduce", _all_reduce),
            patch.object(
                language_loss, "get_context_parallel_world_size", lambda: 1
            ),
            patch.object(language_loss, "mtp_layers_active", lambda c: False),
            patch.object(language_loss, "cp_shard_source", lambda c: "model"),
        ):
            out = _FakeLoss(per_token).forward(dict_args)
        self.assertNotIn("dspark_out", out)
        self.assertEqual(logged, [float(out["dspark_loss"])])
        return seen, float(out["dspark_loss"])

    def test_mean_mode(self):
        seen, logged = self._run(per_token=False)
        self.assertIsNone(seen["group"])
        self.assertAlmostEqual(seen["inject"], 2.0 * 1.2, places=5)
        self.assertAlmostEqual(logged, 2.0 * 1.2, places=5)

    def test_per_token_mode(self):
        labels = paddle.to_tensor([[1, 2, -100, 3, -100]], dtype="int64")
        seen, logged = self._run(per_token=True, labels=labels)
        self.assertIs(seen["group"], _SHARDING_GROUP)
        # 3 valid tokens per rank, summed over the sharding group.
        t_global = 3 * _SHARDING_RANKS
        self.assertAlmostEqual(seen["inject"], 2.0 * 1.2 * t_global, places=4)
        # (1.2 - 0.2) * 20 / 10 + 0.2 * 8 / 4
        self.assertAlmostEqual(logged, 2.0 * (2.0 + 0.4), places=4)

    def test_per_token_needs_labels(self):
        with self.assertRaisesRegex(ValueError, "needs labels"):
            self._run(per_token=True)


if __name__ == "__main__":
    unittest.main()
