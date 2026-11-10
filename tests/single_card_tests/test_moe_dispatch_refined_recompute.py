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

"""``moe_dispatch`` refined recompute: config gating (no GPU, no dist init).

The RR runtime needs DeepEP and a live process group, so this covers the part
that decides *whether* it engages: ``MoeLayer._rr_dispatch_update``'s
preconditions, and the queue-pairing contract of
``DeepEPDispatchRefinedRecompute``.
"""

import types
import unittest

from paddlefleet.transformer.moe.moe_layer import MoELayer


def _fake_layer(modules, **over):
    """Bare stand-in carrying only what ``_rr_dispatch_update`` reads."""
    cfg = types.SimpleNamespace(
        recompute_modules=modules,
        recompute_granularity=over.pop("recompute_granularity", "full"),
        recompute_method=over.pop("recompute_method", "uniform"),
        virtual_pipeline_model_parallel_size=over.pop("vpp", None),
        recompute_num_layers=over.pop("recompute_num_layers", 1),
    )
    layer = types.SimpleNamespace(
        config=cfg,
        fp8=over.pop("fp8", None),
        fp8_dispatch=over.pop("fp8_dispatch", False),
        use_w4a8=over.pop("use_w4a8", False),
        using_sonic_moe=over.pop("using_sonic_moe", False),
        moe_token_dispatcher_type=over.pop(
            "moe_token_dispatcher_type", "deepep"
        ),
        layer_number=over.pop("layer_number", 3),
        is_mtp_layer=False,
        use_rr_deepep_dispatch=False,
    )
    assert not over, f"unused kwargs {over}"
    return layer


def _run(layer, in_full=True, in_mlp=False):
    MoELayer._rr_dispatch_update(layer, in_full, in_mlp)
    return layer.use_rr_deepep_dispatch


class TestMoeDispatchRRGating(unittest.TestCase):
    def test_absent_is_inert(self):
        for modules in (None, [], ["moe_gate_up"], {"moe_combine": "all"}):
            with self.subTest(modules=modules):
                self.assertFalse(_run(_fake_layer(modules)))

    def test_list_mode_enables_every_layer(self):
        """List mode carries no layer info, so RR is on everywhere."""
        self.assertTrue(_run(_fake_layer(["moe_dispatch"])))
        self.assertTrue(
            _run(_fake_layer(["moe_dispatch"], layer_number=0)),
        )

    def test_fp8_raises_not_warns(self):
        """A misconfiguration must fail, not silently drop the optimization.

        All three quantized dispatch paths return a ``(fp8, scale)`` pair the
        RR cache cannot represent, so all three are rejected here rather than
        later in ``_DeepEPManager.dispatch``.
        """
        for over, expected in (
            ({"fp8_dispatch": True, "fp8": "e4m3"}, "fp8_dispatch=True"),
            ({"use_w4a8": True}, "use_w4a8=True"),
            ({"using_sonic_moe": True}, "using_sonic_moe=True"),
        ):
            with self.subTest(**over), self.assertRaises(ValueError) as cm:
                _run(_fake_layer(["moe_dispatch"], **over))
            self.assertIn("moe_dispatch", str(cm.exception))
            # Pin the value, not just the key: the message always prints all
            # three keys, so matching a bare key would pass for any subcase.
            self.assertIn(expected, str(cm.exception))

    def test_non_deepep_raises(self):
        with self.assertRaises(ValueError):
            _run(
                _fake_layer(
                    ["moe_dispatch"], moe_token_dispatcher_type="alltoall"
                )
            )

    def test_recompute_off_raises(self):
        with self.assertRaises(ValueError):
            _run(_fake_layer(["moe_dispatch"], recompute_granularity=None))

    def test_virtual_pipeline_raises(self):
        """VPP reorders replay against the first forward -> frames mispair."""
        with self.assertRaises(ValueError) as cm:
            _run(_fake_layer(["moe_dispatch"], vpp=2))
        self.assertIn("virtual pipeline", str(cm.exception))
        # vpp=1 is the normalized "no VPP" value and must pass.
        self.assertTrue(_run(_fake_layer(["moe_dispatch"], vpp=1)))

    def test_no_recompute_region_raises(self):
        with self.assertRaises(ValueError):
            _run(_fake_layer(["moe_dispatch"]), in_full=False, in_mlp=False)
        # mlp-only recompute is a valid host for the RR frame.
        self.assertTrue(
            _run(_fake_layer(["moe_dispatch"]), in_full=False, in_mlp=True)
        )

    def test_dict_mode_requires_first_n_for_a_count(self):
        with self.assertRaises(ValueError):
            _run(
                _fake_layer(
                    {"moe_dispatch": 2},
                    recompute_method="uniform",
                )
            )

    def test_dict_mode_requires_layer_number(self):
        """Dict mode resolves per layer, so the id must already be set."""
        layer = _fake_layer({"moe_dispatch": [0, 1]})
        del layer.layer_number
        with self.assertRaises(ValueError) as cm:
            _run(layer)
        self.assertIn("layer_number", str(cm.exception))

    def test_dict_mode_selector_is_inverted(self):
        """RR runs on the layers the selector does NOT pick."""
        self.assertFalse(_run(_fake_layer({"moe_dispatch": "all"})))
        self.assertFalse(_run(_fake_layer({"moe_dispatch": None})))
        self.assertTrue(
            _run(_fake_layer({"moe_dispatch": [0, 1]}, layer_number=3))
        )
        self.assertFalse(
            _run(_fake_layer({"moe_dispatch": [0, 1]}, layer_number=1))
        )


class TestDispatchRRQueue(unittest.TestCase):
    def test_replay_without_a_cached_frame_raises(self):
        """An unpaired recompute pass must not read a stale/absent frame."""
        import paddle
        from paddle.base import framework

        from paddlefleet.transformer.moe.fused_a2a import (
            DeepEPDispatchRefinedRecompute,
        )

        rr = DeepEPDispatchRefinedRecompute()
        self.assertEqual(rr.pending, 0)
        # Grad on => the recompute branch, which has nothing queued.
        self.assertTrue(framework._dygraph_tracer()._has_grad)
        with self.assertRaises(RuntimeError) as cm:
            rr.forward(paddle.zeros([1, 1]), None, None, 1, None)
        self.assertIn("Queue is empty", str(cm.exception))


class TestDispatchRowBufferIsNotReleased(unittest.TestCase):
    """Static audit: the cached dispatch row buffer must never be released.

    ``moe_dispatch`` RR aliases ``hs_2d_dispatched`` (too big to copy) in its
    cache, so every release of it in ``MlpNode`` -- direct
    ``_clear_to_zero_allocation()`` or ``self._free_dispatch_buffer(...)`` --
    has to sit behind the ``rr_holds_dispatched`` guard. A missed one
    surfaces only at runtime, deep in the replay, as a bare
    "Tensor's dimension is out of bound ... memory's size is 0" -- hence this
    check rather than a comment.
    """

    def test_every_row_buffer_release_is_guarded(self):
        import ast
        import inspect

        from paddlefleet.transformer.moe import fusion_layer_utils

        tree = ast.parse(inspect.getsource(fusion_layer_utils))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "MlpNode"
        )
        parent = {}
        for node in ast.walk(cls):
            for child in ast.iter_child_nodes(node):
                parent[child] = node

        def negates_guard(test):
            """``not self.rr_holds_dispatched``, alone or as an ``and`` operand.

            Structural, not a substring: an inverted guard
            (``if self.rr_holds_dispatched:``) has no ``Not`` and is rejected,
            and so is an ``or``, which would not actually protect the release.
            """
            if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
                return any(negates_guard(v) for v in test.values)
            return (
                isinstance(test, ast.UnaryOp)
                and isinstance(test.op, ast.Not)
                and ast.unparse(test.operand) == "self.rr_holds_dispatched"
            )

        def guarded(call):
            """True only for a call reached through the guard's true branch."""
            child = call
            node = parent.get(child)
            while node is not None:
                if isinstance(node, ast.If) and negates_guard(node.test):
                    # The ``else`` branch is the RR-on case: not a guard.
                    if any(
                        child is stmt or child in ast.walk(stmt)
                        for stmt in node.body
                    ):
                        return True
                child, node = node, parent.get(node)
            return False

        def releases_row_buffer(n):
            """A call freeing ``hs_2d_dispatched``, directly or via the helper."""
            if not (
                isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            ):
                return False
            if n.func.attr == "_clear_to_zero_allocation":
                return ast.unparse(n.func.value) == "hs_2d_dispatched"
            if n.func.attr == "_free_dispatch_buffer":
                return bool(n.args) and (
                    ast.unparse(n.args[0]) == "hs_2d_dispatched"
                )
            return False

        releases = [n for n in ast.walk(cls) if releases_row_buffer(n)]
        # Guard against a vacuous pass: if the release path is renamed again,
        # this audit would match nothing and silently stop checking anything.
        self.assertTrue(
            releases,
            "found no hs_2d_dispatched release in MlpNode; update this audit "
            "to the current release path",
        )
        unguarded = [ast.unparse(n) for n in releases if not guarded(n)]
        self.assertEqual(
            unguarded,
            [],
            "unguarded hs_2d_dispatched release(s) in MlpNode: wrap them in "
            "`if not self.rr_holds_dispatched:` or moe_dispatch RR will read "
            "freed memory during the recompute pass",
        )

    def test_small_metadata_is_copied_and_rows_are_aliased(self):
        """The metadata the first pass consumes destructively must be copied.

        ``MlpNode._prepare_forward`` casts ``dispatched_indices`` to int32 and
        then releases the int64 original, so the cache cannot alias it. The row
        buffer is the opposite: copying it would double ~0.5 GiB/layer, so it
        must stay an alias (kept alive by ``rr_holds_dispatched`` instead).

        Driven through ``_first_fwd`` with the a2a stubbed out, so it asserts
        the behaviour rather than the shape of the source text.
        """
        from unittest import mock

        import paddle

        from paddlefleet.transformer.moe import fused_a2a

        recv_x = paddle.ones([4, 8], dtype="bfloat16")
        recv_probs = paddle.ones([4, 2], dtype="float32")
        indices = paddle.ones([4, 2], dtype="int64")
        per_expert = paddle.ones([2], dtype="int32")
        states = {
            "dispatched_indices": indices,
            "tokens_per_expert": per_expert,
            "handle": object(),
        }

        rr = fused_a2a.DeepEPDispatchRefinedRecompute()
        with (
            mock.patch.object(
                fused_a2a,
                "fused_dispatch_forward_func",
                return_value=(recv_x, recv_probs, states, None),
            ),
            mock.patch.object(fused_a2a, "wait_for_deepep"),
        ):
            out_x, out_probs, out_states, _ = rr._first_fwd(
                paddle.zeros([4, 8], dtype="bfloat16"),
                indices,
                recv_probs,
                2,
                types.SimpleNamespace(id=0),
                True,
            )

        self.assertEqual(rr.pending, 1)
        # The first pass still gets the originals: it may consume them.
        self.assertIs(out_states, states)
        frame = rr._hold_tensors_queue.queue[0]

        # Rows: aliased, not copied.
        self.assertEqual(frame["recv_x"].data_ptr(), out_x.data_ptr())

        # Metadata: independent copies, so releasing the originals (what
        # MlpNode does) cannot reach the cached frame.
        self.assertNotEqual(
            frame["states"]["dispatched_indices"].data_ptr(),
            indices.data_ptr(),
        )
        self.assertNotEqual(
            frame["states"]["tokens_per_expert"].data_ptr(),
            per_expert.data_ptr(),
        )
        self.assertNotEqual(
            frame["recv_token_probs"].data_ptr(), out_probs.data_ptr()
        )
        indices.zero_()
        per_expert.zero_()
        recv_probs.zero_()
        self.assertTrue(
            bool((frame["states"]["dispatched_indices"] == 1).all())
        )
        self.assertTrue(bool((frame["states"]["tokens_per_expert"] == 1).all()))
        self.assertTrue(bool((frame["recv_token_probs"] == 1).all()))

        # The handle rides through untouched: backward keys off the first
        # pass's layout.
        self.assertIs(frame["states"]["handle"], states["handle"])

    def test_fp8_dispatch_pair_is_rejected(self):
        """A ``(fp8, scale)`` return cannot be cached as one row tensor."""
        from unittest import mock

        import paddle

        from paddlefleet.transformer.moe import fused_a2a

        rr = fused_a2a.DeepEPDispatchRefinedRecompute()
        with (
            mock.patch.object(
                fused_a2a,
                "fused_dispatch_forward_func",
                return_value=(
                    (paddle.ones([2, 4]), paddle.ones([2, 1])),
                    paddle.ones([2, 2], dtype="float32"),
                    {},
                    None,
                ),
            ),
            mock.patch.object(fused_a2a, "wait_for_deepep"),
            self.assertRaises(RuntimeError) as cm,
        ):
            rr._first_fwd(
                paddle.zeros([2, 4]),
                None,
                None,
                2,
                types.SimpleNamespace(id=0),
                True,
            )
        self.assertIn("fp8", str(cm.exception))
        self.assertEqual(rr.pending, 0)


def _bare_moe_layer(training, use_rr=True):
    """Bare ``MoELayer`` carrying only what ``dispatch`` touches.

    A real instance (not a namespace) so the ``rr_dispatch_active`` property
    resolves; ``object.__setattr__`` bypasses ``nn.Layer``'s attribute hooks,
    which ``dispatch`` never needs.
    """
    captured = {}

    def token_dispatch(hidden_states, fp8_dispatch, **kwargs):
        captured.update(kwargs)
        return hidden_states, None

    layer = MoELayer.__new__(MoELayer)
    for name, value in (
        (
            "token_dispatcher",
            types.SimpleNamespace(
                dispatch_preprocess=lambda x, *a, **k: x,
                token_dispatch=token_dispatch,
            ),
        ),
        ("fp8_dispatch", False),
        ("use_ue8m0", False),
        ("using_sonic_moe", False),
        ("use_rr_deepep_dispatch", use_rr),
        ("training", training),
    ):
        object.__setattr__(layer, name, value)
    return layer, captured


class TestRRIsOffDuringInference(unittest.TestCase):
    """RR must fall back to plain execution when no backward will follow.

    Inference runs the recompute region with ``_has_grad`` False as well, so an
    unguarded call site queues a frame per layer per step that nothing pops:
    the dispatched row buffer leaks, and the next training step pops a stale
    frame whose handle and rows belong to another micro-batch -- wrong
    gradients, no error. ``auto_refined_recompute``'s module docstring makes
    this the call site's responsibility.
    """

    def _layer(self, training):
        return types.SimpleNamespace(
            use_rr_deepep_dispatch=True,
            use_rr_deepep_combine=True,
            training=training,
        )

    def test_eval_disables_both(self):
        layer = self._layer(training=False)
        self.assertFalse(MoELayer.rr_dispatch_active.fget(layer))
        self.assertFalse(MoELayer.rr_combine_active.fget(layer))

    def test_train_enables_both(self):
        layer = self._layer(training=True)
        self.assertTrue(MoELayer.rr_dispatch_active.fget(layer))
        self.assertTrue(MoELayer.rr_combine_active.fget(layer))

    def test_gating_still_wins_in_training(self):
        layer = self._layer(training=True)
        layer.use_rr_deepep_dispatch = False
        self.assertFalse(MoELayer.rr_dispatch_active.fget(layer))

    def test_row_buffer_retention_tracks_the_active_flag(self):
        """``dispatch()`` must hand the token dispatcher the active flag.

        Driven through ``MoELayer.dispatch`` with the dispatcher stubbed out,
        so it asserts the value that actually leaves the call site rather than
        the shape of the source text. Keying it off ``use_rr_deepep_dispatch``
        instead would queue a frame during inference that no replay pops.
        """
        for training, use_rr, expected in (
            (False, True, False),  # inference: configured but must stay off
            (True, True, True),
            (True, False, False),  # gating still wins
        ):
            with self.subTest(training=training, use_rr=use_rr):
                layer, captured = _bare_moe_layer(training, use_rr)
                MoELayer.dispatch(layer, "x", None, None)
                self.assertIs(captured["use_rr_deepep_dispatch"], expected)


class TestDeepEPManagerDispatchRR(unittest.TestCase):
    """``_DeepEPManager.dispatch``'s ``use_rr_deepep_dispatch`` branch."""

    def _manager(self):
        from unittest import mock

        from paddlefleet.transformer.moe.token_dispatcher import (
            _DeepEPManager,
        )

        with mock.patch(
            "paddlefleet.transformer.moe.token_dispatcher.fused_dispatch",
            new=mock.MagicMock(),
        ):
            manager = _DeepEPManager(
                group=mock.MagicMock(),
                router_topk=2,
                num_experts=8,
                num_local_experts=4,
                moe_ep_barrier=False,
            )
        manager.token_indices = mock.MagicMock()
        manager.token_probs = mock.MagicMock()
        return manager

    def test_rr_instance_is_created_on_first_use(self):
        from unittest import mock

        import paddle

        from paddlefleet.transformer.moe import fused_a2a

        manager = self._manager()
        self.assertIsNone(manager._rr_fuseddispatch)
        handle = object()
        states = {
            "handle": handle,
            "dispatched_indices": paddle.ones([2, 2], dtype="int64"),
            "tokens_per_expert": paddle.ones([2], dtype="int32"),
        }
        with (
            mock.patch.object(
                fused_a2a,
                "fused_dispatch_forward_func",
                return_value=(
                    paddle.ones([2, 4], dtype="bfloat16"),
                    paddle.ones([2, 2], dtype="float32"),
                    states,
                    None,
                ),
            ),
            mock.patch.object(fused_a2a, "wait_for_deepep"),
            paddle.no_grad(),  # the first-forward branch of the RR object
        ):
            manager.dispatch(
                paddle.zeros([2, 4], dtype="bfloat16"),
                use_rr_deepep_dispatch=True,
            )
        self.assertIsInstance(
            manager._rr_fuseddispatch,
            fused_a2a.DeepEPDispatchRefinedRecompute,
        )
        # The manager must publish the layout the cached rows belong to.
        self.assertIs(manager.handle, handle)

    def test_rr_type_mismatch_raises(self):
        import paddle

        manager = self._manager()
        manager._rr_fuseddispatch = "not_the_right_type"
        with self.assertRaises(RuntimeError) as cm:
            manager.dispatch(paddle.zeros([2, 4]), use_rr_deepep_dispatch=True)
        self.assertIn("_rr_fuseddispatch type mismatch", str(cm.exception))

    def test_rr_requires_bf16_dispatch(self):
        """A direct manager call must not slip a quantized dispatch past RR."""
        import paddle

        for kwargs, attrs in (
            ({"fp8_dispatch": True}, {}),
            ({"using_sonic_moe": True}, {}),
            ({}, {"_use_w4a8": True}),
        ):
            with self.subTest(**(kwargs or attrs)):
                manager = self._manager()
                for name, value in attrs.items():
                    setattr(manager, name, value)
                with self.assertRaises(ValueError) as cm:
                    manager.dispatch(
                        paddle.zeros([2, 4]),
                        use_rr_deepep_dispatch=True,
                        **kwargs,
                    )
                self.assertIn("requires a bf16 dispatch", str(cm.exception))


class TestOverlapSchedulerRejectsDispatchRR(unittest.TestCase):
    """``moe_dispatch`` RR cannot ride the forward/backward overlap scheduler.

    That path dispatches via ``dispatch_overlap``, which has no RR branch, and
    its ``recompute_forward`` replays only attn / mlp / post_process, so the
    dispatch a2a is never re-issued. Accepting the switch would cost peak
    memory for a speedup that never happens.
    """

    def _overlap_layer(self, use_rr):
        from unittest import mock

        from paddlefleet.transformer.transformer_layer import (
            TransformerLayer,
            TransformerLayerWithOverlap,
        )

        mlp = MoELayer.__new__(MoELayer)
        for name, value in (
            ("gate", types.SimpleNamespace(norm_topk_prob=False)),
            ("expert_model_parallel_size", 2),
            ("moe_token_dispatcher_type", "deepep"),
            ("use_rr_deepep_dispatch", use_rr),
        ):
            object.__setattr__(mlp, name, value)

        layer = TransformerLayerWithOverlap.__new__(TransformerLayerWithOverlap)
        for name, value in (
            ("recompute_mlp", False),
            ("recompute_input_layernorm", False),
            ("recompute_post_attention_layernorm", False),
            ("mlp", mlp),
        ):
            object.__setattr__(layer, name, value)

        # Only the subclass's own validation is under test.
        with mock.patch.object(
            TransformerLayer, "__init__", lambda self, *a, **k: None
        ):
            TransformerLayerWithOverlap.__init__(layer)

    def test_dispatch_rr_is_rejected(self):
        with self.assertRaises(ValueError) as cm:
            self._overlap_layer(use_rr=True)
        self.assertIn("moe_dispatch RR", str(cm.exception))

    def test_without_the_switch_the_check_is_inert(self):
        self._overlap_layer(use_rr=False)


def _run_both(layer, in_full=True, in_mlp=False):
    """Drive the full entry point, which gates dispatch and combine in turn."""
    MoELayer.rr_recompute_update(layer, in_full, in_mlp)
    return layer.use_rr_deepep_dispatch, layer.use_rr_deepep_combine


def _fake_both_layer(modules, **over):
    layer = _fake_layer(modules, **over)
    layer.moe_shared_expert_overlap = True
    layer.use_rr_deepep_combine = False
    # ``rr_recompute_update`` delegates to the dispatch half through ``self``,
    # which a bare SimpleNamespace does not resolve.
    layer._rr_dispatch_update = types.MethodType(
        MoELayer._rr_dispatch_update, layer
    )
    return layer


class TestMoeCombineRRGating(unittest.TestCase):
    """``moe_combine`` RR gates independently of ``moe_dispatch``."""

    def test_both_entries_enable_both(self):
        self.assertEqual(
            _run_both(_fake_both_layer(["moe_dispatch", "moe_combine"])),
            (True, True),
        )

    def test_each_entry_is_independent(self):
        self.assertEqual(
            _run_both(_fake_both_layer(["moe_combine"])), (False, True)
        )
        self.assertEqual(
            _run_both(_fake_both_layer(["moe_dispatch"])), (True, False)
        )

    def test_requires_shared_expert_overlap(self):
        layer = _fake_both_layer(["moe_combine"])
        layer.moe_shared_expert_overlap = False
        with self.assertRaises(ValueError) as cm:
            _run_both(layer)
        self.assertIn("moe_shared_expert_overlap", str(cm.exception))

    def test_non_deepep_raises(self):
        with self.assertRaises(ValueError):
            _run_both(
                _fake_both_layer(
                    ["moe_combine"], moe_token_dispatcher_type="alltoall"
                )
            )

    def test_recompute_off_raises(self):
        with self.assertRaises(ValueError):
            _run_both(
                _fake_both_layer(["moe_combine"], recompute_granularity=None)
            )

    def test_virtual_pipeline_raises(self):
        """Same mispairing hazard as dispatch RR: must fail, not compute wrong."""
        with self.assertRaises(ValueError) as cm:
            _run_both(_fake_both_layer(["moe_combine"], vpp=2))
        self.assertIn("virtual pipeline", str(cm.exception))
        # vpp=1 is the normalized "no VPP" value and must pass.
        self.assertEqual(
            _run_both(_fake_both_layer(["moe_combine"], vpp=1)), (False, True)
        )

    def test_no_recompute_region_raises(self):
        """RR caches a frame that only a replay pass consumes."""
        with self.assertRaises(ValueError):
            _run_both(
                _fake_both_layer(["moe_combine"]), in_full=False, in_mlp=False
            )

    def test_fp8_is_allowed_unlike_dispatch(self):
        """combine forward is always bf16, so fp8 is not a blocker here."""
        self.assertEqual(
            _run_both(
                _fake_both_layer(["moe_combine"], fp8="e4m3", fp8_dispatch=True)
            ),
            (False, True),
        )


if __name__ == "__main__":
    unittest.main()
