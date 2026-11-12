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

"""Behavior tests for ``paddlefleet.overlap_context_parallel``.

The overlapped FlashMask context-parallel layer rebases the mask rows onto the
local queries and lays the mask key axis out in group-rank shard order, which
FlashMask then maps onto its own communication order. Those pieces are ordinary
CPU logic and are what these tests exercise with independently derived
expectations:

  - ``block_order``: the DualChunkSwap block permutation into shard order;
  - ``gathered_kv_order``: the mask permutation itself, against a slice-and-
    concat oracle, and with ``localize_mask`` against dense visibility;
  - ``localize_mask``: mode routing, the exact chunk ids it derives, and its
    rejection of unsupported modes;
  - ``overlap_flashmask_attention_cp``: the feature-rejection guards that run
    before the kernel is ever reached.

The FA-4 kernel and the CP process group are NOT exercised here; their real
numerics need an SM100 device plus a FlashMask-built paddlefleet_ops and are
covered by ``multi_card_tests/transformer/test_flash_mask_cp_overlap.py``.
Importing the module still requires paddle, so every test honestly skips when
paddle is absent rather than reporting a false pass.
"""

import os
import sys
import unittest
from unittest import mock

import numpy as np

# Test the in-tree dev version: src/ ahead of any installed package.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "src"))

try:
    import paddle

    from paddlefleet import (
        context_parallel_utils as cpu,
        overlap_context_parallel as ocp,
    )

    _IMPORT_ERROR = None
except ImportError as exc:  # honest skip reason, not a swallowed failure
    paddle = None
    cpu = None
    ocp = None
    _IMPORT_ERROR = exc

_HAS_DEPS = _IMPORT_ERROR is None
_SKIP_REASON = (
    "" if _HAS_DEPS else f"paddle/paddlefleet unavailable: {_IMPORT_ERROR!r}"
)

DUALCHUNK = "dualchunk_allgather_overlap"
CONTIGUOUS = "contiguous_allgather_overlap"


class _Group:
    """Minimal CP-group stand-in; the layer reads only these attributes."""

    def __init__(self, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size


def independent_block_order(cp_size):
    """Natural blocks in group-rank shard order: rank r owns ``(r, 2*cp-1-r)``."""
    order = []
    for owner in range(cp_size):
        order += [owner, 2 * cp_size - 1 - owner]
    return order


def owned_rows(rank, cp_size, s_total, mode):
    """Global sequence positions rank ``rank`` holds, in its local order."""
    if mode == CONTIGUOUS:
        s_local = s_total // cp_size
        return np.arange(rank * s_local, (rank + 1) * s_local)
    block = s_total // (2 * cp_size)
    mirror = 2 * cp_size - 1 - rank
    return np.concatenate(
        [
            np.arange(rank * block, (rank + 1) * block),
            np.arange(mirror * block, (mirror + 1) * block),
        ]
    )


def visible(lt_start, ut_end, rows):
    """2-col non-causal FlashMask: row i sees key j iff ut_end[j] <= i < lt_start[j]."""
    return (rows[:, None] >= ut_end[None, :]) & (
        rows[:, None] < lt_start[None, :]
    )


def make_block_labeled_mask(cp_size, block_len=3, num_vecs=2):
    """A mask whose ``b``-th natural key block carries a unique per-vec value."""
    n_blocks = 2 * cp_size
    seqlen = n_blocks * block_len
    values = paddle.zeros([1, 1, seqlen, num_vecs], dtype="int32")
    rows = []
    for block in range(n_blocks):
        for _ in range(block_len):
            rows.append([block + 1000 * vec for vec in range(num_vecs)])
    values = paddle.to_tensor([rows], dtype="int32").reshape(
        [1, 1, seqlen, num_vecs]
    )
    return values, block_len


def slice_concat_oracle(mask, order, block_len):
    """Reassemble ``mask`` by concatenating whole key blocks in ``order``.

    A slice-and-concat oracle independent of the reshape/index_select the layer
    uses, so a wrong axis or a scrambled block would diverge here.
    """
    blocks = [mask[:, :, b * block_len : (b + 1) * block_len, :] for b in order]
    return paddle.concat(blocks, axis=2)


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestBlockOrder(unittest.TestCase):
    """The DualChunkSwap natural-block permutation into shard order."""

    def setUp(self):
        # block_order caches per cp_size in a module global; isolate every test.
        ocp.BLOCK_ORDER_CACHE.clear()
        self.addCleanup(ocp.BLOCK_ORDER_CACHE.clear)

    def test_hand_enumerated_small_cases(self):
        cases = {
            1: [0, 1],
            2: [0, 3, 1, 2],
            4: [0, 7, 1, 6, 2, 5, 3, 4],
        }
        for cp_size, expected in cases.items():
            self.assertEqual(ocp.block_order(cp_size).tolist(), expected)

    def test_matches_independent_oracle(self):
        for cp_size in (1, 2, 4, 8, 16, 64):
            self.assertEqual(
                ocp.block_order(cp_size).tolist(),
                independent_block_order(cp_size),
            )

    def test_cached_per_cp_size(self):
        first = ocp.block_order(4)
        self.assertIs(first, ocp.block_order(4))
        self.assertIsNot(first, ocp.block_order(8))


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestGatheredKvOrder(unittest.TestCase):
    """The mask key-axis permutation, against slice-and-concat and dense oracles."""

    def setUp(self):
        ocp.BLOCK_ORDER_CACHE.clear()
        self.addCleanup(ocp.BLOCK_ORDER_CACHE.clear)

    def test_dualchunk_matches_oracle(self):
        for cp_size in (1, 2, 4):
            mask, block_len = make_block_labeled_mask(cp_size)
            for rank in range(cp_size):
                got = ocp.gathered_kv_order(
                    mask, _Group(rank=rank, world_size=cp_size), DUALCHUNK
                )
                expected = slice_concat_oracle(
                    mask, independent_block_order(cp_size), block_len
                )
                self.assertTrue(
                    bool((got == expected).all()), f"cp={cp_size} rank={rank}"
                )

    def test_contiguous_is_already_in_shard_order(self):
        mask, _ = make_block_labeled_mask(4)
        self.assertIs(
            ocp.gathered_kv_order(
                mask, _Group(rank=1, world_size=4), CONTIGUOUS
            ),
            mask,
        )

    def test_local_rows_see_the_global_keys(self):
        # localize_mask + gathered_kv_order end to end: local row i sees mask key
        # j exactly when its global row sees the key held at shard position j.
        rng = np.random.default_rng(0)
        s_local = 12
        for mode in (DUALCHUNK, CONTIGUOUS):
            for cp_size in (2, 4, 8):
                s_total = s_local * cp_size
                ut_end = rng.integers(0, s_total + 1, size=s_total)
                lt_start = np.minimum(
                    ut_end + rng.integers(0, s_total + 1, size=s_total), s_total
                )
                glob = paddle.to_tensor(
                    np.stack([lt_start, ut_end], -1)[None, None].astype("int32")
                )
                owned = [
                    owned_rows(r, cp_size, s_total, mode)
                    for r in range(cp_size)
                ]
                keys = np.concatenate(owned)
                for rank in range(cp_size):
                    group = _Group(rank=rank, world_size=cp_size)
                    local = ocp.gathered_kv_order(
                        ocp.localize_mask(glob, s_local, group, mode),
                        group,
                        mode,
                    ).numpy()[0, 0]
                    np.testing.assert_array_equal(
                        visible(local[:, 0], local[:, 1], np.arange(s_local)),
                        visible(lt_start[keys], ut_end[keys], owned[rank]),
                        err_msg=f"{mode} cp={cp_size} rank={rank}",
                    )


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestLocalizeMask(unittest.TestCase):
    """Mode routing and the exact chunk ids localize_mask derives."""

    def test_dualchunk_derives_mirrored_chunk_ids(self):
        captured = {}

        def spy(startend_row_indices, **kwargs):
            captured["mask"] = startend_row_indices
            captured.update(kwargs)
            return "dual-marker"

        mask = paddle.zeros([1, 1, 16, 2], dtype="int32")
        group = _Group(rank=1, world_size=4)
        with mock.patch.object(ocp, "preprocess_index_dual_chunks", spy):
            out = ocp.localize_mask(mask, 16, group, DUALCHUNK)
        self.assertEqual(out, "dual-marker")
        self.assertIs(captured["mask"], mask)
        self.assertEqual(captured["chunk_id_first"], 1)
        # second chunk id = 2 * world_size - rank - 1 = 2*4 - 1 - 1 = 6
        self.assertEqual(captured["chunk_id_second"], 6)
        self.assertEqual(captured["seq_blocksize"], 8)
        self.assertEqual(captured["max_seqlen_q"], 8)

    def test_contiguous_rebases_on_local_chunk(self):
        captured = {}

        def spy(startend_row_indices, **kwargs):
            captured["mask"] = startend_row_indices
            captured.update(kwargs)
            return "contig-marker"

        mask = paddle.zeros([1, 1, 16, 2], dtype="int32")
        group = _Group(rank=3, world_size=4)
        with mock.patch.object(ocp, "preprocess_index", spy):
            out = ocp.localize_mask(mask, 16, group, CONTIGUOUS)
        self.assertEqual(out, "contig-marker")
        self.assertEqual(captured["chunk_id"], 3)
        self.assertEqual(captured["seq_blocksize"], 16)
        self.assertEqual(captured["max_seqlen_q"], 16)

    def test_unsupported_mode_rejected(self):
        mask = paddle.zeros([1, 1, 8, 2], dtype="int32")
        with self.assertRaises(ValueError):
            ocp.localize_mask(mask, 4, _Group(0, 2), "contiguous_a2a_overlap")


@unittest.skipUnless(_HAS_DEPS, _SKIP_REASON)
class TestOverlapEntryGuards(unittest.TestCase):
    """Feature-rejection guards that run before the FA-4 kernel is reached.

    OVERLAP_SUPPORTED is a capability flag, not code under test; patching it True
    lets the guard branches run without an SM100 device or an ops build. The
    kernel apply is never reached by any of these cases.
    """

    @staticmethod
    def _query(seqlen=8):
        return paddle.zeros([1, seqlen, 1, 8], dtype="bfloat16")

    def _call(self, seqlen=8, **kwargs):
        q = self._query(seqlen)
        mask = paddle.zeros([1, 1, seqlen, 2], dtype="int32")
        return ocp.overlap_flashmask_attention_cp(q, q, q, mask, **kwargs)

    def test_requires_capable_build(self):
        with mock.patch.object(ocp, "OVERLAP_SUPPORTED", False):  # noqa: SIM117
            with self.assertRaises(RuntimeError):
                self._call()

    def test_dropout_not_implemented(self):
        with mock.patch.object(ocp, "OVERLAP_SUPPORTED", True):  # noqa: SIM117
            with self.assertRaises(NotImplementedError):
                self._call(dropout=0.1)

    def test_causal_not_implemented(self):
        with mock.patch.object(ocp, "OVERLAP_SUPPORTED", True):  # noqa: SIM117
            with self.assertRaises(NotImplementedError):
                self._call(causal=True)

    def test_fixed_seed_offset_not_implemented(self):
        with mock.patch.object(ocp, "OVERLAP_SUPPORTED", True):  # noqa: SIM117
            with self.assertRaises(NotImplementedError):
                self._call(fixed_seed_offset=paddle.zeros([1], dtype="int64"))

    def test_odd_local_sequence_length_rejected(self):
        with mock.patch.object(ocp, "OVERLAP_SUPPORTED", True):  # noqa: SIM117
            with self.assertRaises(AssertionError):
                self._call(seqlen=7)


if __name__ == "__main__":
    unittest.main()
