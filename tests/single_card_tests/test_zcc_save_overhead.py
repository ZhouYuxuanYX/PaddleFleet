# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
"""Unit tests for the ZCC save-overhead reductions.

Covers the trainer_state slimming sent to the worker and its restoration,
the deferred first OFFLOAD, chunked D2H, pinned CPU buffers, concurrent
flash writes and numpy-aware TrainerState.save_to_json. The GPU cases run the
real fusion storage helpers: chunked vs unchunked offload, model-state save and
EMA accumulation on pinned buffers.
"""

import json
import os
import pickle
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import paddle

from paddlefleet.trainer.trainer_callback import TrainerState
from paddlefleet.trainer.utils import zero_cost_checkpoint as zcc

N = 64


class _Box:
    """Stand-in for multiprocessing.Value."""

    def __init__(self, value=0):
        self.value = value


class _Queue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


def _manager(**kw):
    m = zcc.ZeroCostCheckpointManager.__new__(zcc.ZeroCostCheckpointManager)
    worker = SimpleNamespace(
        worker_id=0,
        task_queue=_Queue(),
        status=_Box(zcc.ZCCWorkerStatus.IDLE.value),
        global_step=_Box(kw.pop("worker_step", 5)),
    )
    m.workers = [worker]
    m.processes = []
    m.current_worker = worker
    m.global_step = 5
    m.pipeline_hooks_steps = kw.pop("hooks", 1)
    m.current_pipeline_hook_step = 0
    m.ready_to_save = True
    m.defer_next_offload = m.offload_deferred = False
    for k, v in kw.items():
        setattr(m, k, v)
    return m, worker


def _offload_tasks(worker):
    return [
        t for t in worker.task_queue.items if t[0] == zcc.ZCCTaskType.OFFLOAD
    ]


class TestDeferredOffload(unittest.TestCase):
    def test_first_offload_is_deferred_until_flush(self):
        m, worker = _manager()
        m.defer_next_offload = True
        m.zcc_pipeline_hook(0)
        self.assertEqual(_offload_tasks(worker), [])
        self.assertTrue(m.offload_deferred)
        m.flush_deferred_offload()
        m.flush_deferred_offload()
        self.assertEqual(_offload_tasks(worker), [(zcc.ZCCTaskType.OFFLOAD, 5)])
        self.assertEqual(m.current_pipeline_hook_step, 1)

    def test_hook_without_defer_request_dispatches(self):
        m, worker = _manager(hooks=2)
        m.zcc_pipeline_hook(0)
        m.zcc_pipeline_hook(1)
        m.zcc_pipeline_hook(2)  # budget exhausted
        self.assertEqual(len(_offload_tasks(worker)), 2)

    def test_defer_request_is_dropped_without_worker(self):
        m, _ = _manager(current_worker=None)
        m.defer_next_offload = True
        m.zcc_pipeline_hook(0)
        self.assertFalse(m.offload_deferred)
        self.assertFalse(m.defer_next_offload)

    def test_step_begin_sync_issues_pending_offload(self):
        m, worker = _manager()
        m.offload_deferred = True
        m.maybe_sync_offload_status()
        self.assertEqual(len(_offload_tasks(worker)), 1)
        self.assertIsNone(m.current_worker)

    def test_blocking_sync_issues_pending_offload(self):
        m, worker = _manager()
        m.offload_deferred = True
        m.sync_offload_status()
        self.assertEqual(len(_offload_tasks(worker)), 1)

    def test_blocking_sync_raises_when_worker_fails_while_waiting(self):
        m, worker = _manager(worker_step=4)
        polls = []

        def fail_worker(_):
            polls.append(1)
            worker.status.value = zcc.ZCCWorkerStatus.ERROR.value

        with (
            mock.patch.object(zcc.time, "sleep", fail_worker),
            self.assertRaises(RuntimeError),
        ):
            m.sync_offload_status()
        self.assertEqual(
            len(polls), 1
        )  # raised on the next poll, not waiting forever

    def test_finalize_issues_pending_offload(self):
        m, worker = _manager(hooks=2)
        m.offload_deferred = True
        m.finalize()
        self.assertEqual(len(_offload_tasks(worker)), 2)
        self.assertEqual(worker.task_queue.items[-1][0], zcc.ZCCTaskType.FINISH)

    def test_on_step_end_requests_deferral(self):
        cb = zcc.ZeroCostCheckpointCallback.__new__(
            zcc.ZeroCostCheckpointCallback
        )
        m, worker = _manager()
        m.get_idle_worker_for_saving = lambda payload: None
        cb.manager = m
        cb.runtime_timer = SimpleNamespace(
            start=lambda *a: None, stop=lambda: None
        )
        cb.maybe_update_zcc_worker = lambda *a: None
        cb._get_save_infos_based_on_steps = lambda *a: (None, None)
        cb.get_rng_states = lambda args: None
        cb.on_step_end(
            SimpleNamespace(
                zcc_save_ema_coef=None, pipeline_model_parallel_size=1
            ),
            SimpleNamespace(global_step=5),
            SimpleNamespace(should_save=True),
            model=object(),
            lr_scheduler=SimpleNamespace(state_dict=lambda: {}),
            optimizer=object(),
        )
        self.assertTrue(m.offload_deferred)
        self.assertEqual(_offload_tasks(worker), [])


def _state(saved, last_start, base=None, diff=None):
    tp = {
        "data_status": np.zeros(N, np.int64),
        "saved_data_status": saved,
        "last_start_data_status": last_start,
        "consumed_samples": int(saved.sum()),
    }
    if base is not None:
        tp["_zcc_saved_base"] = base
        tp["_zcc_saved_diff"] = diff
    return SimpleNamespace(global_step=1, trial_params=tp)


def _callback(n_workers=1, device_id=0):
    cb = zcc.ZeroCostCheckpointCallback.__new__(zcc.ZeroCostCheckpointCallback)
    cb.manager = SimpleNamespace(
        workers=[object()] * n_workers, device_id=device_id
    )
    return cb


def _worker():
    return zcc.ZeroCostCheckpointWorker.__new__(zcc.ZeroCostCheckpointWorker)


def _send(cb, worker, state, dumping=True):
    snap = cb._state_for_worker(state, dumping=dumping)
    worker.trainer_state = pickle.loads(pickle.dumps(snap))
    worker._restore_trainer_state()
    return snap.trial_params, worker.trainer_state.trial_params


class TestTrainerStateSlimming(unittest.TestCase):
    def test_single_worker_base_plus_diff(self):
        rng = np.random.default_rng(0)
        cb, worker = _callback(), _worker()
        base = rng.integers(0, 3, N)
        last_start = rng.integers(0, 3, N)
        saved, acc = base.copy(), {}
        for rnd in range(3):
            ids = rng.integers(0, N, 5)
            np.add.at(saved, ids, 1)
            for i in ids.tolist():
                acc[i] = acc.get(i, 0) + 1
            diff = (
                np.array(list(acc), np.int64),
                np.array(list(acc.values()), np.int64),
            )
            sent, got = _send(cb, worker, _state(saved, last_start, base, diff))
            self.assertNotIn("data_status", sent)
            self.assertNotIn("_zcc_saved_diff", got)
            np.testing.assert_array_equal(got["saved_data_status"], saved)
            np.testing.assert_array_equal(
                got["last_start_data_status"], last_start
            )
            # the base and last_start travel only once
            self.assertEqual(
                isinstance(sent["saved_data_status"][1], np.ndarray), rnd == 0
            )
            self.assertEqual(
                isinstance(sent["last_start_data_status"], np.ndarray), rnd == 0
            )

    def test_single_worker_without_diff_copies_saved(self):
        cb, worker = _callback(), _worker()
        saved = np.arange(N, dtype=np.int64)
        sent, got = _send(
            cb,
            worker,
            _state(saved, saved.copy(), base=saved.copy(), diff=None),
        )
        self.assertFalse(np.shares_memory(sent["saved_data_status"], saved))
        np.testing.assert_array_equal(got["saved_data_status"], saved)

    def test_multiple_workers_get_dense_copies(self):
        cb = _callback(n_workers=2)
        saved = np.arange(N, dtype=np.int64)
        diff = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        for w in (_worker(), _worker()):
            sent, got = _send(
                cb, w, _state(saved, saved.copy(), saved.copy(), diff)
            )
            self.assertIsInstance(sent["saved_data_status"], np.ndarray)
            self.assertFalse(np.shares_memory(sent["saved_data_status"], saved))
            np.testing.assert_array_equal(got["saved_data_status"], saved)

    def test_non_writing_rank_and_ema_step_send_no_arrays(self):
        saved = np.ones(N, np.int64)
        for cb, dumping in (
            (_callback(device_id=1), True),
            (_callback(), False),
        ):
            sent, got = _send(
                cb, _worker(), _state(saved, saved.copy()), dumping
            )
            for key in zcc._ZCC_DATA_STATUS_KEYS:
                self.assertNotIn(key, sent)
            self.assertEqual(got["consumed_samples"], N)

    def test_state_without_data_status_is_passed_through(self):
        cb = _callback()
        state = SimpleNamespace(global_step=1, trial_params={"x": 1})
        self.assertIs(cb._state_for_worker(state), state)
        worker = _worker()
        worker.trainer_state = SimpleNamespace(trial_params=None)
        worker._restore_trainer_state()  # no trial_params: nothing to do

    def test_placeholders_without_cache_raise(self):
        cb = _callback()
        saved = np.ones(N, np.int64)
        last_start = saved.copy()
        base = saved.copy()
        diff = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        # the first send fills the "sent" records
        cb._state_for_worker(_state(saved, last_start, base, diff))
        snap = cb._state_for_worker(_state(saved, last_start, base, diff))
        self.assertEqual(
            snap.trial_params["last_start_data_status"],
            zcc._LAST_START_UNCHANGED,
        )
        self.assertEqual(
            snap.trial_params["saved_data_status"][1],
            zcc._SAVED_BASE_UNCHANGED,
        )
        fresh = _worker()
        fresh.trainer_state = pickle.loads(pickle.dumps(snap))
        # saved_data_status base placeholder without a cached base
        with self.assertRaises(RuntimeError):
            fresh._restore_trainer_state()
        fresh = _worker()
        fresh._cached_saved_base = base
        fresh.trainer_state = pickle.loads(pickle.dumps(snap))
        # last_start_data_status placeholder without a cached value
        with self.assertRaises(RuntimeError):
            fresh._restore_trainer_state()

    def test_sent_arrays_are_read_only(self):
        cb = _callback()
        saved = np.ones(N, np.int64)
        last_start, base = saved.copy(), saved.copy()
        diff = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        cb._state_for_worker(_state(saved, last_start, base, diff))
        # an in-place update would leave the worker cache stale, so it must fail
        for arr in (last_start, base):
            with self.assertRaises(ValueError):
                arr[0] = 7
        self.assertTrue(saved.flags.writeable)

    def test_replaced_array_is_sent_again(self):
        cb, worker = _callback(), _worker()
        saved = np.ones(N, np.int64)
        diff = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        _send(cb, worker, _state(saved, saved.copy(), saved.copy(), diff))
        new_last, new_base = saved * 2, saved * 3
        sent, got = _send(cb, worker, _state(saved, new_last, new_base, diff))
        self.assertIs(sent["last_start_data_status"], new_last)
        self.assertIs(sent["saved_data_status"][1], new_base)
        np.testing.assert_array_equal(got["last_start_data_status"], new_last)
        np.testing.assert_array_equal(got["saved_data_status"], new_base)

    def test_list_values_are_always_sent(self):
        cb, worker = _callback(), _worker()
        saved = np.ones(N, np.int64)
        last_start = [0] * N
        for _ in range(2):
            sent, got = _send(cb, worker, _state(saved, last_start))
            self.assertEqual(sent["last_start_data_status"], last_start)
        last_start[0] = (
            5  # a list cannot be frozen; sending it in full is still correct
        )
        sent, got = _send(cb, worker, _state(saved, last_start))
        self.assertEqual(got["last_start_data_status"][0], 5)

    def test_worker_error_is_reported_to_main_process(self):
        worker = _worker()
        worker.worker_id, worker.device_id = 0, 0
        worker.status = _Box()
        stale = _callback()._state_for_worker(
            _state(np.ones(N, np.int64), np.ones(N, np.int64))
        )
        stale.trial_params["last_start_data_status"] = zcc._LAST_START_UNCHANGED
        worker.task_queue = mock.Mock()
        worker.task_queue.get.return_value = (
            zcc.ZCCTaskType.PREPARE,
            (("/flash/x", None), (None, stale, None)),
        )
        with (
            mock.patch.object(zcc, "get_env_device", return_value="gpu"),
            mock.patch.object(zcc.core, "set_cuda_current_device_id"),
            mock.patch.object(zcc.paddle, "set_device"),
            self.assertRaises(RuntimeError),
        ):
            worker.run()
        # a dead worker must not look busy, or the main process waits forever
        self.assertEqual(worker.status.value, zcc.ZCCWorkerStatus.ERROR.value)

    def test_prepare_task_restores_state(self):
        worker = _worker()
        worker.status = _Box()
        saved = np.arange(N, dtype=np.int64)
        base = saved.copy()
        diff = (np.array([3], np.int64), np.array([2], np.int64))
        np.add.at(saved, [3], [2])
        snap = _callback()._state_for_worker(
            _state(saved, saved.copy(), base, diff)
        )
        worker.process_prepare_task((("/flash/x", None), (None, snap, None)))
        np.testing.assert_array_equal(
            worker.trainer_state.trial_params["saved_data_status"], saved
        )
        self.assertEqual(
            worker.status.value, zcc.ZCCWorkerStatus.OFFLOADING.value
        )


def _mix_state(W, w_base=None, w_diff=None):
    state = _state(np.ones(N, np.int64), np.zeros(N, np.int64))
    tp = state.trial_params
    tp["mix_sample_pos"] = {
        "W": [int(x) for x in W],
        "E": {"3": [7]},
        "cursors": {"0": {"c": 0, "j": 5}},
        "meta": {"seed": 1},
    }
    if w_base is not None:
        tp["_zcc_mix_w_base"] = w_base
        tp["_zcc_mix_w_diff"] = w_diff
    return state


class TestMixPosSlimming(unittest.TestCase):
    def test_non_writing_rank_and_ema_step_send_no_mix_pos(self):
        W = np.arange(N, dtype=np.int64)
        base = W.copy()
        diff = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        for cb, dumping in (
            (_callback(device_id=1), True),
            (_callback(), False),
        ):
            sent, got = _send(cb, _worker(), _mix_state(W, base, diff), dumping)
            for key in ("mix_sample_pos", "_zcc_mix_w_base", "_zcc_mix_w_diff"):
                self.assertNotIn(key, sent)
                self.assertNotIn(key, got)

    def test_mix_pos_alone_is_still_slimmed(self):
        state = SimpleNamespace(
            global_step=1, trial_params={"mix_sample_pos": {"W": [1]}}
        )
        snap = _callback(device_id=1)._state_for_worker(state)
        self.assertNotIn("mix_sample_pos", snap.trial_params)

    def test_single_worker_w_base_plus_diff(self):
        rng = np.random.default_rng(0)
        cb, worker = _callback(), _worker()
        W = rng.integers(0, 5, N)
        base = W.copy()
        for rnd in range(3):
            W = W.copy()
            W[rng.integers(0, N, 4)] += 1
            ids = np.flatnonzero(W != base).astype(np.int64)
            state = _mix_state(W, base, (ids, W[ids]))
            main_pos = state.trial_params["mix_sample_pos"]
            sent, got = _send(cb, worker, state)
            # the worker writes the same list the main process holds
            self.assertEqual(got["mix_sample_pos"], main_pos)
            self.assertIsInstance(got["mix_sample_pos"]["W"], list)
            self.assertNotIn("_zcc_mix_w_base", got)
            self.assertNotIn("_zcc_mix_w_diff", got)
            # the main process's own mix_sample_pos is not replaced
            self.assertIsInstance(main_pos["W"], list)
            # the base travels only once
            self.assertEqual(
                isinstance(sent["mix_sample_pos"]["W"][1], np.ndarray), rnd == 0
            )

    def test_rebased_w_is_sent_again(self):
        cb, worker = _callback(), _worker()
        W = np.arange(N, dtype=np.int64)
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        _send(cb, worker, _mix_state(W, W.copy(), empty))
        W2 = W + 1
        new_base = W2.copy()
        sent, got = _send(cb, worker, _mix_state(W2, new_base, empty))
        self.assertIs(sent["mix_sample_pos"]["W"][1], new_base)
        self.assertEqual(got["mix_sample_pos"]["W"], W2.tolist())

    def test_w_placeholder_without_cache_raises(self):
        cb = _callback()
        W = np.arange(N, dtype=np.int64)
        base = W.copy()
        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64))
        cb._state_for_worker(_mix_state(W, base, empty))
        snap = cb._state_for_worker(_mix_state(W, base, empty))
        self.assertEqual(
            snap.trial_params["mix_sample_pos"]["W"][1],
            zcc._MIX_W_BASE_UNCHANGED,
        )
        fresh = _worker()
        fresh._cached_saved_base = None
        fresh.trainer_state = pickle.loads(pickle.dumps(snap))
        fresh._cached_last_start_data_status = np.zeros(N, np.int64)
        with self.assertRaises(RuntimeError):
            fresh._restore_trainer_state()

    def test_without_w_diff_or_with_several_workers_w_is_sent_in_full(self):
        W = np.arange(N, dtype=np.int64)
        for cb, state in (
            (_callback(), _mix_state(W)),
            (
                _callback(n_workers=2),
                _mix_state(W, W.copy(), (np.zeros(0, np.int64),) * 2),
            ),
        ):
            sent, got = _send(cb, _worker(), state)
            self.assertEqual(sent["mix_sample_pos"]["W"], W.tolist())
            self.assertEqual(got["mix_sample_pos"]["W"], W.tolist())


class _Task:
    def __init__(self, log):
        self.log = log

    def cpu_wait(self):
        self.log.append("wait")


class _Helper:
    def __init__(self, length, log, name, elem=4):
        self.buffer_length = self.all_param_numel = length
        self.buffer = SimpleNamespace(element_size=lambda: elem)
        self.inited_buffers = {"0": (self.buffer, None)}
        self.model_weights_metas = {
            "w": {"buffer_index": "0", "start": 0, "end": length}
        }
        self.tasks = []
        self.log, self.name = log, name

    def sync_partial_param(self, start=None, end=None, numel_to_sync=None):
        n = end - start if numel_to_sync is None else numel_to_sync
        self.log.append((self.name, n))
        self.tasks.append(_Task(self.log))

    def wait_all(self):
        self.log.append((self.name, "wait_all"))
        self.tasks = []


def _offload_worker(opt_len, param_len, chunk_mb, chunk_numel, param_elem=4):
    w = _worker()
    log = []
    w.optimizer_fusion_storage_helper = _Helper(opt_len, log, "opt")
    w.param_fusion_storage_helper = _Helper(param_len, log, "param", param_elem)
    w.all_numel = opt_len + param_len
    w.chunk_size_in_numel = chunk_numel
    w.offloaded_numels = 0
    w.training_args_content = SimpleNamespace(zcc_d2h_chunk_mb=chunk_mb)
    w.status = _Box()
    w.global_step = _Box(0)
    w.ema_coef = None
    return w, log


class TestOffload(unittest.TestCase):
    def test_chunked_offload_copies_chunk_by_chunk(self):
        mb = 1024 * 1024 // 4  # numel per MB of fp32
        w, log = _offload_worker(3 * mb, 2 * mb, chunk_mb=2, chunk_numel=5 * mb)
        self.assertTrue(w.process_offload_task(dump=False, global_step=9))
        copies = [x for x in log if isinstance(x, tuple) and x[1] != "wait_all"]
        self.assertEqual(sum(n for _, n in copies), 5 * mb)
        self.assertTrue(all(n <= 2 * mb for _, n in copies))
        self.assertIn("wait", log)  # earlier chunks drained before the next one
        self.assertEqual(w.global_step.value, 9)
        self.assertEqual(w.status.value, zcc.ZCCWorkerStatus.IDLE.value)

    def test_chunk_size_follows_each_segment_dtype(self):
        mb = 1024 * 1024
        w, log = _offload_worker(
            mb // 2, mb, chunk_mb=1, chunk_numel=3 * mb // 2, param_elem=2
        )
        w.process_offload_task(dump=False, global_step=1)
        copies = [x for x in log if isinstance(x, tuple) and x[1] != "wait_all"]
        # 1 MB is mb/4 fp32 optimizer elements but mb/2 bf16 parameter elements
        self.assertEqual(
            copies, [("opt", mb // 4)] * 2 + [("param", mb // 2)] * 2
        )

    def test_negative_chunk_size_is_rejected(self):
        from paddlefleet.trainer.training_args import TrainingArguments

        args = SimpleNamespace(
            fuse_optimizer_states=True,
            flash_device_save_steps=0,
            save_steps=10,
            zcc_ema_interval=1,
            zcc_d2h_chunk_mb=0,
        )
        TrainingArguments._post_init_zero_cost_checkpoint(args)
        args.zcc_d2h_chunk_mb = -1
        with self.assertRaises(ValueError):
            TrainingArguments._post_init_zero_cost_checkpoint(args)

    def test_chunked_offload_over_several_hooks(self):
        # 12-byte chunks are 3 fp32 elements; 8 elements per hook cross the boundary in the 2nd call
        w, log = _offload_worker(10, 10, chunk_mb=12 / 2**20, chunk_numel=8)
        done = [
            w.process_offload_task(dump=False, global_step=1) for _ in range(3)
        ]
        self.assertEqual(done, [False, False, True])
        copies = [x for x in log if isinstance(x, tuple) and x[1] != "wait_all"]
        expected = [
            *[("opt", 3), ("opt", 3), ("opt", 2)],  # hook 1: [0, 8)
            *[("opt", 2), ("param", 3), ("param", 3)],  # hook 2: [8, 16)
            *[("param", 3), ("param", 1)],  # hook 3: [16, 20)
        ]
        self.assertEqual(copies, expected)

    def test_unchunked_offload_scenes(self):
        # chunk_mb=0: optimizer-only, optimizer+param, then param-only slices
        w, log = _offload_worker(10, 10, chunk_mb=0, chunk_numel=8)
        self.assertFalse(w.process_offload_task(dump=False, global_step=1))
        self.assertFalse(w.process_offload_task(dump=False, global_step=1))
        self.assertTrue(w.process_offload_task(dump=False, global_step=1))
        copies = [x for x in log if isinstance(x, tuple) and x[1] != "wait_all"]
        self.assertEqual(
            copies, [("opt", 8), ("opt", 2), ("param", 6), ("param", 4)]
        )


class TestFlashWrites(unittest.TestCase):
    def _worker(self):
        w = zcc.ZeroCostCheckpointWorkerFcBased.__new__(
            zcc.ZeroCostCheckpointWorkerFcBased
        )
        w._save_model_state = lambda d: w._queue_write(
            {"m": 1}, os.path.join(d, "m")
        )
        w._save_opt_state = lambda d: w._queue_write(
            {"o": 1}, os.path.join(d, "o")
        )
        w._save_ema_state = lambda d: w._queue_write(
            {"e": 1}, os.path.join(d, "e")
        )
        return w

    def _dump(self, output_dir):
        threads, paths = set(), []

        def fake_save(obj, path):
            threads.add(threading.current_thread().name)
            paths.append(path)

        with mock.patch.object(zcc.paddle, "save", fake_save):
            self._worker()._dump_states(output_dir)
        return threads, sorted(os.path.basename(p) for p in paths)

    def test_flash_dir_writes_from_thread_pool(self):
        threads, files = self._dump(
            os.path.join(zcc.FLASH_DEVICE, "checkpoint-1")
        )
        self.assertEqual(files, ["e", "m", "o"])
        self.assertTrue(all(t.startswith("zcc-dump") for t in threads))

    def test_persistent_dir_writes_serially(self):
        threads, files = self._dump("/tmp/persistent/checkpoint-1")
        self.assertEqual(files, ["e", "m", "o"])
        self.assertEqual(threads, {threading.current_thread().name})

    def test_failed_dump_stops_queuing(self):
        w = self._worker()

        def boom(d):
            raise OSError("disk full")

        w._save_opt_state = boom
        with self.assertRaises(OSError):
            w._dump_states("/tmp/persistent/checkpoint-1")
        self.assertIsNone(w._pending_writes)

    def test_queue_write_outside_dump_saves_immediately(self):
        saved = []
        with mock.patch.object(
            zcc.paddle, "save", lambda obj, path: saved.append(path)
        ):
            self._worker()._queue_write({}, "/tmp/x")
        self.assertEqual(saved, ["/tmp/x"])

    def test_flash_dir_detection(self):
        self.assertTrue(
            zcc._zcc_is_flash_dir(os.path.join(zcc.FLASH_DEVICE, "c"))
        )
        self.assertFalse(zcc._zcc_is_flash_dir("/tmp/c"))


class TestPinnedBuffer(unittest.TestCase):
    def _init(self, flatten):
        cuda = mock.MagicMock(shape=[8])
        helper = zcc.ParamFusionStorageHelper.__new__(
            zcc.ParamFusionStorageHelper
        )
        with (
            mock.patch.object(
                zcc.paddle, "is_compiled_with_xpu", return_value=False
            ),
            mock.patch.object(
                zcc.paddle.base.core.LoDTensor, "_new_shared_cuda", create=True
            ),
            mock.patch.object(zcc.paddle, "to_tensor", return_value=cuda),
        ):
            helper.init_buffer({}, flatten=flatten)
        return cuda

    def test_flat_buffers_are_pinned(self):
        cuda = self._init(flatten=True)
        cuda.pin_memory.assert_called_once()
        cuda.cpu.assert_not_called()

    def test_unshard_buffers_stay_pageable(self):
        cuda = self._init(flatten=False)
        cuda.cpu.assert_called_once()
        cuda.pin_memory.assert_not_called()


_HAS_GPU = (
    paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
)
requires_gpu = unittest.skipUnless(_HAS_GPU, "needs a CUDA device")


def _gpu_memory():
    paddle.device.synchronize()
    return paddle.device.cuda.memory_allocated()


def _is_view_of(t, base):
    lo = base.get_tensor()._ptr()
    p = t.get_tensor()._ptr()
    return lo <= p < lo + base._numel() * base.element_size()


@requires_gpu
class TestPin2CpuZeroCopy(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")

    def test_alias_is_cpu_view_for_each_dtype(self):
        for dtype in ("float32", "bfloat16", "float16"):
            src = paddle.arange(16, dtype="float32").astype(dtype)
            pinned = src.pin_memory()
            alias = zcc.pin2cpu_zero_copy(pinned)
            self.assertTrue(alias.place.is_cpu_place(), dtype)
            self.assertEqual(alias.dtype, pinned.dtype)
            self.assertEqual(
                alias.get_tensor()._ptr(), pinned.get_tensor()._ptr()
            )
            with zcc.device_guard("cpu"):
                part = alias._slice(4, 8)
            self.assertTrue(_is_view_of(part, pinned), dtype)
            np.testing.assert_array_equal(
                part.astype("float32").numpy(),
                np.arange(4, 8, dtype=np.float32),
            )

    def test_non_pinned_tensor_is_returned_as_is(self):
        gpu = paddle.ones([4])
        self.assertIs(zcc.pin2cpu_zero_copy(gpu), gpu)
        with zcc.device_guard("cpu"):
            cpu = paddle.ones([4])
        self.assertIs(zcc.pin2cpu_zero_copy(cpu), cpu)
        int_pinned = paddle.ones([4], dtype="int64").pin_memory()
        self.assertIs(zcc.pin2cpu_zero_copy(int_pinned), int_pinned)


def _real_helpers():
    """Real Opt/Param fusion storage helpers: 3000 fp32 optimizer elements,
    a 3000-element bf16 weight and a 1000-element fp32 gate.

    CUDA IPC cannot open a handle in the process that exported it, so only
    `_new_shared_cuda` is replaced; pinning, D2H and slicing are the real ones.
    """
    rng = np.random.default_rng(0)

    def randn(n):
        return paddle.to_tensor(rng.standard_normal(n).astype("float32"))

    gpu = {
        "opt": randn(3000),
        "0": randn(3000).astype("bfloat16"),
        "1": randn(1000),
    }
    acc_meta = {
        "moment1": {
            "p0": {
                "name": "p0_moment1",
                "shape": [1500],
                "start": 0,
                "end": 1500,
            }
        }
    }
    master_meta = {
        "p0": {"name": "p0_master", "shape": [1500], "start": 1500, "end": 3000}
    }
    model_metas = {
        "w0": {
            "buffer_index": "0",
            "name": "w0",
            "shape": [50, 60],
            "start": 0,
            "end": 3000,
        },
        "gate": {
            "buffer_index": "1",
            "name": "gate",
            "shape": [10, 100],
            "start": 0,
            "end": 1000,
        },
    }
    with mock.patch.object(
        paddle.base.core.DenseTensor,
        "_new_shared_cuda",
        side_effect=lambda meta: gpu[meta[0]].get_tensor(),
        create=True,
    ):
        opt = zcc.OptFusionStorageHelper(
            acc_meta, master_meta, None, ("opt",) + (None,) * 6
        )
        param = zcc.ParamFusionStorageHelper(
            model_metas, {"0": ("0",), "1": ("1",)}
        )
    return opt, param


def _real_worker(opt, param, chunk_mb):
    w = _worker()
    w.optimizer_fusion_storage_helper = opt
    w.param_fusion_storage_helper = param
    w.all_numel = w.chunk_size_in_numel = (
        opt.buffer_length + param.all_param_numel
    )
    w.offloaded_numels = 0
    w.training_args_content = SimpleNamespace(zcc_d2h_chunk_mb=chunk_mb)
    w.status = _Box()
    w.global_step = _Box(0)
    w.ema_coef = None
    return w


def _f32(t):
    return t.astype("float32").numpy()


@requires_gpu
class TestRealOffloadAndSave(unittest.TestCase):
    def setUp(self):
        paddle.set_device("gpu")

    def _offload(self, chunk_mb):
        opt, param = _real_helpers()
        w = _real_worker(opt, param, chunk_mb)
        sizes, issue = [], w._issue_offload_range
        w._issue_offload_range = lambda s, n: (sizes.append(n), issue(s, n))
        self.assertTrue(w.process_offload_task(dump=False, global_step=3))
        self.assertEqual(w.global_step.value, 3)
        return opt, param, sizes

    def _states(self, opt, param):
        with zcc.device_guard("cpu"):
            opt_sd = opt.state_dict()
            states = {**opt_sd.pop("master_weights"), **opt_sd}
            states.update(param.state_dict())
        for k, v in states.items():
            self.assertTrue(v.place.is_cpu_place(), k)
        return {k: _f32(v) for k, v in states.items()}

    def test_chunked_offload_matches_unchunked_and_source(self):
        # 4 KB chunks: 1024 fp32 optimizer elements; parameters average 2.5 bytes
        opt, param, sizes = self._offload(4096 / 2**20)
        self.assertEqual(sizes, [1024, 1024, 952, 1638, 1638, 724])
        chunked = self._states(opt, param)
        opt0, param0, sizes0 = self._offload(0)
        self.assertEqual(sizes0, [7000])  # one copy over both segments
        unchunked = self._states(opt0, param0)
        src = _f32(opt.buffer)
        expected = {
            "p0_moment1": src[:1500],
            "p0": src[1500:],
            "w0": _f32(param.inited_buffers["0"][0]).reshape([50, 60]),
            "gate": _f32(param.inited_buffers["1"][0]).reshape([10, 100]),
        }
        self.assertEqual(sorted(chunked), sorted(expected))
        for k, v in expected.items():
            np.testing.assert_array_equal(chunked[k], v, err_msg=k)
            np.testing.assert_array_equal(unchunked[k], v, err_msg=k)

    def test_model_state_is_saved_from_pinned_views(self):
        opt, param, _ = self._offload(0)
        before = _gpu_memory()
        with zcc.device_guard("cpu"):
            state = param.state_dict()
        for k, v in state.items():
            buf = param.inited_buffers[
                param.model_weights_metas[k]["buffer_index"]
            ][1]
            self.assertTrue(buf.place.is_cuda_pinned_place())
            self.assertTrue(_is_view_of(v, buf), k)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model_state.pdparams")
            zcc.paddle.save(state, path)
            self.assertEqual(_gpu_memory(), before)
            loaded = paddle.load(path, return_numpy=True)
        for k, v in state.items():
            self.assertEqual(list(loaded[k].shape), v.shape)
            # bf16 round-trips as its uint16 bit pattern on both sides
            np.testing.assert_array_equal(np.asarray(loaded[k]), v.numpy())

    def test_ema_with_pinned_buffers_stays_on_cpu(self):
        opt, param, _ = self._offload(0)
        ema = zcc.ZeroCostCheckpointEMAProcessor(opt, param, 0.9)
        self.assertEqual(
            sorted(ema.ema_buffer_model_params), ["1"]
        )  # fp32 only
        before = _gpu_memory()
        ema.ema_accumulate(1, 0.0, None)
        ema.ema_accumulate(2, 0.0, None)
        self.assertEqual(_gpu_memory(), before)
        gate = ema.ema_buffer_model_params["1"]
        for t in (ema.ema_buffer, gate):
            self.assertTrue(t.place.is_cpu_place())
        # two steps from zero with the same weights: 0.1 * 0.9 * x + 0.1 * x
        np.testing.assert_allclose(
            ema.ema_buffer.numpy(), 0.19 * _f32(opt.buffer)[1500:], rtol=1e-6
        )
        np.testing.assert_allclose(
            gate.numpy(), 0.19 * _f32(param.inited_buffers["1"][0]), rtol=1e-6
        )
        with zcc.device_guard("cpu"):
            ema_sd = ema.ema_state_dict()
        self.assertTrue(ema_sd["gate"].place.is_cpu_place())
        self.assertTrue(ema_sd["master_weights"]["p0"].place.is_cpu_place())


class TestTrainerStateJson(unittest.TestCase):
    def test_numpy_trial_params_round_trip(self):
        state = TrainerState()
        state.trial_params = {
            "saved_data_status": np.arange(4, dtype=np.int64),
            "n": np.int64(3),
        }
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trainer_state.json")
            state.save_to_json(path)
            data = json.load(open(path))
        self.assertEqual(
            data["trial_params"], {"saved_data_status": [0, 1, 2, 3], "n": 3}
        )

    def test_unknown_object_still_fails(self):
        state = TrainerState()
        state.trial_params = {"x": object()}
        with tempfile.TemporaryDirectory() as d, self.assertRaises(TypeError):
            state.save_to_json(os.path.join(d, "s.json"))


if __name__ == "__main__":
    unittest.main()
