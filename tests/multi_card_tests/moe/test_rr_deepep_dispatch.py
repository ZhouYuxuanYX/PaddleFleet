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

"""
Multi-card integration test for DeepEP Dispatch Refined Recompute.

Verifies gradient consistency between:
- Recompute WITHOUT RR (re-issues the dispatch a2a in the second forward)
- Recompute WITH RR (caches the first forward's rows + states and replays them)

and that an inference pass in between leaves no RR frame behind: the pass is
told apart only by ``tracer._has_grad``, which cannot distinguish the first
recompute pass from plain inference, so the call site must fall back to the
plain path when no backward will follow.

Run with: paddle.distributed.launch --gpus=0,1,2,3,4,5,6,7 test_rr_deepep_dispatch.py
"""

import os
import random
import sys
import unittest

# Enable coverage in subprocess when WITH_COVERAGE is set
if os.environ.get("WITH_COVERAGE") == "ON":
    import coverage

    cov = coverage.Coverage(
        data_file=os.environ.get("COVERAGE_FILE", ".coverage"),
        config_file=os.environ.get("COVERAGE_RCFILE", True),
    )
    cov.start()

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.distributed import fleet
from paddle.distributed.fleet.utils import recompute

try:
    from paddlefleet.refined_recompute.queue_check import global_rr_queue_log
    from paddlefleet.transformer.moe.fused_a2a import (
        DeepEPDispatchRefinedRecompute,
        fused_combine,
        fused_dispatch,
    )
except ImportError:
    print(
        "Failed to import from paddlefleet.transformer.moe.fused_a2a.",
        file=sys.stderr,
    )
    fused_combine = None
    fused_dispatch = None
    DeepEPDispatchRefinedRecompute = None

# ----- MoE Test Parameters -----
batch_size = 32
hidden_size = 128
num_experts = 64
topk = 2
# ---------------------------


class TestDispatchLayer(paddle.nn.Layer):
    """One MoE communication cycle: dispatch -> per-row compute -> combine.

    ``use_rr`` only swaps how the dispatch is issued; everything downstream,
    including the handle the combine and both backwards key off, is whatever
    the dispatch returned.
    """

    def __init__(self, use_rr=False):
        super().__init__()
        self.use_rr = use_rr
        self._rr_fuseddispatch = (
            DeepEPDispatchRefinedRecompute() if use_rr else None
        )
        self.group = (
            fleet.get_hybrid_communicate_group().get_model_parallel_group()
        )
        self.linear = paddle.nn.Linear(
            hidden_size, hidden_size, bias_attr=False
        )
        self.expert = paddle.nn.Linear(
            hidden_size, hidden_size, bias_attr=False
        )
        self.deepep_dtype = paddle.bfloat16

    def _dispatch(self, x, token_indices, token_probs):
        # Mirrors MoELayer's call site: RR needs a backward to consume the
        # frame it queues, so inference takes the plain path.
        if self.use_rr and self.training:
            return self._rr_fuseddispatch(
                x,
                token_indices,
                token_probs,
                num_experts,
                self.group,
            )
        return fused_dispatch(
            x=x,
            token_indices=token_indices,
            token_probs=token_probs,
            num_experts=num_experts,
            group=self.group,
        )

    def forward(self, x, mock_token_indices, mock_token_probs):
        processed_x = self.linear(x)
        origin_dtype = processed_x.dtype

        dispatched_x, dispatched_probs, states, _ = self._dispatch(
            processed_x.astype(self.deepep_dtype),
            mock_token_indices,
            mock_token_probs,
        )

        # Touch both dispatch outputs so each one's backward is exercised.
        expert_out = self.expert(dispatched_x.astype(origin_dtype))
        expert_out = expert_out * dispatched_probs.sum(
            axis=-1, keepdim=True
        ).astype(origin_dtype)

        combined = fused_combine(
            x=expert_out.astype(self.deepep_dtype),
            group=self.group,
            handle=states["handle"],
        )
        return combined.astype(origin_dtype)


class TestRecomputeLayer(paddle.nn.Layer):
    """Wrapper that applies recompute to the core TestDispatchLayer."""

    def __init__(self, use_rr=False):
        super().__init__()
        self.dispatch_layer = TestDispatchLayer(use_rr=use_rr)

    def forward(self, x, mock_token_indices, mock_token_probs):
        return recompute(
            self.dispatch_layer, x, mock_token_indices, mock_token_probs
        )


class TestRRDeepEPDispatch(unittest.TestCase):
    """Recompute without RR vs with RR: identical results and gradients.

    RR only caches and replays the first forward's dispatch rows, so nothing
    about the computation changes.
    """

    @classmethod
    def setUpClass(cls):
        if fused_dispatch is None or fused_combine is None:
            raise unittest.SkipTest(
                "DeepEP fused_dispatch/fused_combine not available."
            )

        cls.original_dtype = paddle.get_default_dtype()
        paddle.set_default_dtype("float32")

        strategy = fleet.DistributedStrategy()
        strategy.hybrid_configs = {
            "dp_degree": 1,
            "mp_degree": 8,
            "pp_degree": 1,
            "sharding_degree": 1,
        }
        fleet.init(is_collective=True, strategy=strategy)

        seed = 12345
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

        cls.x = paddle.randn([batch_size, hidden_size], dtype="float32")

        num_tokens = paddle.shape(cls.x)[0]
        cls.mock_token_indices = paddle.randint(
            0, num_experts, shape=[num_tokens, topk]
        )
        cls.mock_token_indices.stop_gradient = True

        cls.mock_token_probs_raw = paddle.rand(
            [num_tokens, topk], dtype="float32"
        )
        cls.mock_token_probs_raw.stop_gradient = False

    @classmethod
    def tearDownClass(cls):
        paddle.set_default_dtype(cls.original_dtype)

    def _pair(self):
        """Two layers with identical weights, one plain and one with RR."""
        seed = 42
        paddle.seed(seed)
        without_rr = TestRecomputeLayer(use_rr=False)
        paddle.seed(seed)
        with_rr = TestRecomputeLayer(use_rr=True)
        with_rr.set_state_dict(without_rr.state_dict())
        return without_rr, with_rr

    def run_test_case(self, layer):
        """Run forward + backward, return the output and every gradient."""
        x_copy = self.x.clone()
        x_copy.stop_gradient = False

        self.mock_token_probs_raw.clear_gradient()
        layer.clear_gradients()

        probs = F.softmax(self.mock_token_probs_raw, axis=-1)
        out = layer(x_copy, self.mock_token_indices, probs)
        out.sum().backward()

        return (
            out.detach().numpy(),
            x_copy.grad.detach().numpy(),
            layer.dispatch_layer.linear.weight.grad.detach().numpy(),
            layer.dispatch_layer.expert.weight.grad.detach().numpy(),
            self.mock_token_probs_raw.grad.detach().numpy(),
        )

    def _assert_same(self, baseline, actual):
        names = ("output", "x grad", "linear grad", "expert grad", "probs grad")
        for name, want, got in zip(names, baseline, actual):
            np.testing.assert_array_equal(want, got, err_msg=name)

    def test_rr_deepep_dispatch(self):
        """Gradient consistency: recompute without RR vs recompute with RR."""
        without_rr, with_rr = self._pair()
        baseline = self.run_test_case(without_rr)
        self._assert_same(baseline, self.run_test_case(with_rr))

        # Every frame the first forwards queued was consumed by a replay.
        self.assertEqual(with_rr.dispatch_layer._rr_fuseddispatch.pending, 0)
        global_rr_queue_log.check()

    def test_inference_leaves_no_frame_behind(self):
        """eval() passes must not queue a frame, and must not corrupt training.

        Without the ``training`` guard at the call site every eval step would
        queue a frame (aliasing the whole dispatched row buffer) that no replay
        pops, and the next training step would pop one of those stale frames --
        whose handle and rows belong to a different micro-batch -- instead of
        its own: wrong gradients, no error.
        """
        without_rr, with_rr = self._pair()
        baseline = self.run_test_case(without_rr)

        rr = with_rr.dispatch_layer._rr_fuseddispatch
        with_rr.eval()
        with paddle.no_grad():
            for _ in range(3):
                with_rr(
                    self.x,
                    self.mock_token_indices,
                    F.softmax(self.mock_token_probs_raw, axis=-1),
                )
        self.assertEqual(
            rr.pending,
            0,
            "inference queued an RR frame that no backward will consume",
        )

        # Training after inference must still pair its own frames.
        with_rr.train()
        self._assert_same(baseline, self.run_test_case(with_rr))
        self.assertEqual(rr.pending, 0)
        global_rr_queue_log.check()


if __name__ == "__main__":
    filename = os.path.basename(__file__)
    result = unittest.main(exit=False)
    failed_flag = f"{filename}.failed"
    if not result.result.wasSuccessful():
        with open(failed_flag, "w") as f:
            f.write(f"{filename} unittest failed")

    # Save coverage data for subprocess
    if os.environ.get("WITH_COVERAGE") == "ON":
        cov.stop()
        cov.save()
