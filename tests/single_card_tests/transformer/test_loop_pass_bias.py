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

"""Per-pass correction bias of a looped decoder (``transformer/loop_pass.py``).

A looped decoder replays the same layers; with ``add_loop_pass_bias_states`` the extra
passes keep their own aux-loss-free correction bias and expert usage, so each pass is
balanced on its own.

CPU only. The fused top-k path (``moe_topk_fusion``) needs the Triton kernel and is not
covered here; it reads the same ``_pass_tensor`` as the eager path.

Run:  python -m unittest tests/single_card_tests/transformer/test_loop_pass_bias.py
"""

import unittest
from unittest.mock import patch

import numpy as np
import paddle

from paddlefleet.transformer.loop_pass import (
    LOOP_PASS_KEY,
    check_loop_pass,
    check_num_passes,
    loop_pass_kwargs,
)
from paddlefleet.transformer.moe.moe_router import TopKRouter

from .test_router import MockTransformerConfig

PAD = 0
INPUT_IDS = [[5, 6, 7, 8], [9, 10, 11, 0]]


def _config(**overrides):
    cfg = MockTransformerConfig()
    cfg.pad_token_id = PAD
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _router(layer=1, **overrides):
    router = TopKRouter(_config(**overrides))
    router.set_layer_number(layer)
    return router


def _input(router, tokens=8):
    return paddle.to_tensor(
        np.random.normal(size=[tokens, router.config.hidden_size]).astype("float32")
    )


class _CPPatch(unittest.TestCase):
    def setUp(self):
        patcher = patch(
            "paddlefleet.transformer.moe.moe_router.get_context_parallel_world_size",
            return_value=1,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        paddle.seed(7)
        np.random.seed(7)


class TestLoopPassHelpers(unittest.TestCase):
    def test_kwargs_only_for_modules_with_state(self):
        router = _router()
        self.assertEqual(loop_pass_kwargs(router, 2), {})
        router.add_loop_pass_bias_states(2)
        self.assertEqual(loop_pass_kwargs(router, 2), {LOOP_PASS_KEY: 2})
        self.assertEqual(loop_pass_kwargs(router, None), {})
        self.assertEqual(loop_pass_kwargs(paddle.nn.Linear(2, 2), 2), {})

    def test_pass_number_is_validated(self):
        self.assertEqual(check_loop_pass(None, 2, "x"), 1)
        self.assertEqual(check_loop_pass(2, 2, "x"), 2)
        for bad in (0, 3):
            with self.subTest(loop_pass=bad), self.assertRaises(ValueError):
                check_loop_pass(bad, 2, "x")
        self.assertEqual(check_num_passes(3, "x"), 3)
        with self.assertRaises(ValueError):
            check_num_passes(1, "x")


class TestRouterPerPassBias(_CPPatch):
    def test_buffers_start_as_pass_ones_and_are_persistable(self):
        router = _router()
        with paddle.no_grad():
            router.e_score_correction_bias.add_(
                paddle.to_tensor(
                    np.random.normal(size=router.e_score_correction_bias.shape).astype("float32")
                )
            )
        pairs = router.add_loop_pass_bias_states(3)
        self.assertEqual(router.loop_bias_passes, 3)
        self.assertTrue(router.has_loop_pass_state)
        self.assertEqual(router.loop_pass_suffixes(), ["", "_pass2", "_pass3"])
        state = router.state_dict()
        for p in (2, 3):
            name = f"e_score_correction_bias_pass{p}"
            # Starts equal to pass 1's: a warm start's first step routes as the shared state did.
            np.testing.assert_array_equal(
                getattr(router, name).numpy(), router.e_score_correction_bias.numpy()
            )
            self.assertIn(name, state)
            self.assertEqual(
                getattr(router, f"expert_usage_pass{p}").numpy().tolist(),
                [0] * router.num_experts,
            )
        self.assertEqual([tuple(c.shape) for c, _ in pairs],
                         [tuple(router.e_score_correction_bias.shape)] * 2)
        with self.assertRaises(RuntimeError):
            router.add_loop_pass_bias_states(2)

    def test_each_pass_selects_with_its_own_bias(self):
        router = _router()
        router.add_loop_pass_bias_states(2)
        favored = router.num_experts - 1
        with paddle.no_grad():
            bias = router.e_score_correction_bias_pass2
            update = np.zeros(bias.shape, dtype="float32")
            update[..., favored] = 1e4
            bias.add_(paddle.to_tensor(update))
        hidden = _input(router)
        ids = paddle.to_tensor(np.array(INPUT_IDS, dtype="int64"))
        shared = router(hidden, input_ids=ids)[1]
        own = router(hidden, input_ids=ids, loop_pass=2)[1]
        self.assertNotIn(favored, shared.numpy().reshape(-1).tolist())
        # Pass 2's bias only shifts the selection, so the favored expert is in every row.
        self.assertTrue((own.numpy() == favored).any(axis=-1).all())

    def test_the_pass_it_ran_is_read_back(self):
        router = _router()
        hidden, ids = _input(router), paddle.to_tensor(np.array(INPUT_IDS, dtype="int64"))
        router(hidden, input_ids=ids, loop_pass=None)
        # Without the copies the router does not record anything: it is unchanged.
        self.assertFalse(hasattr(router, "last_loop_pass"))
        router.add_loop_pass_bias_states(2)
        router(hidden, input_ids=ids)
        self.assertEqual(router.last_loop_pass, 1)
        router(hidden, input_ids=ids, loop_pass=2)
        self.assertEqual(router.last_loop_pass, 2)
        with self.assertRaises(ValueError):
            router(hidden, input_ids=ids, loop_pass=3)

    def test_usage_counts_into_the_pass_that_routed(self):
        router = _router()
        router.add_loop_pass_bias_states(2)
        hidden, ids = _input(router), paddle.to_tensor(np.array(INPUT_IDS, dtype="int64"))
        hidden.stop_gradient = False
        router(hidden, input_ids=ids, loop_pass=2)
        self.assertEqual(router.expert_usage.numpy().sum(), 0)
        self.assertGreater(router.expert_usage_pass2.numpy().sum(), 0)
        router(hidden, input_ids=ids)
        self.assertGreater(router.expert_usage.numpy().sum(), 0)

    def test_other_routings_refuse_the_pass_number(self):
        router = _router(topk_method="greedy")
        with self.assertRaises(ValueError):
            router.add_loop_pass_bias_states(2)


if __name__ == "__main__":
    unittest.main()
