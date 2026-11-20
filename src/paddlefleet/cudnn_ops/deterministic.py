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

"""Deterministic-mode helpers shared by the cuDNN DSA attention and indexer ops."""

import functools
import inspect
import logging

import paddle

logger = logging.getLogger(__name__)

# Deterministic SM90 relay widths cuDNN accepts, widest (fastest) first. 16 is
# a non-portable cluster size: it launches on a full Hopper (132 SM) but not on
# H20 (cudaErrorInvalidClusterSize); 8 is the portable Hopper maximum.
_SM90_RELAY_CLUSTER_SIZES = (16, 8, 4)


@functools.cache
def _wrapper_params(wrapper):
    """Keyword names of a vendored cuDNN wrapper (signature probing only)."""
    return frozenset(inspect.signature(wrapper).parameters)


@functools.cache
def _sm90_relay_cluster_size():
    """Widest SM90 deterministic relay cluster this GPU can launch, or None.

    Asks cuDNN's occupancy probe (1 CTA per SM at the full opt-in shared
    memory budget, the shape of the DSA backward) instead of keying on the
    GPU model. If no size can be confirmed, cuDNN keeps its own default (16),
    which fails to launch on GPUs with fewer SMs (H20); a warning carrying the
    last probe error is logged once (the result is cached) so that failure can
    be traced back here.
    """
    try:
        from paddlefleet_ops.cudnn.frost.occupancy import max_active_clusters
    except ImportError as e:
        logger.warning(
            "SM90 deterministic relay: cannot import cuDNN's occupancy probe "
            "(%r); leaving deterministic_relay_cluster_size to cuDNN's default.",
            e,
        )
        return None
    last_error = None
    for size in _SM90_RELAY_CLUSTER_SIZES:
        try:
            if max_active_clusters(size) > 0:
                return size
        except (NotImplementedError, RuntimeError, ValueError) as e:
            last_error = e
    logger.warning(
        "SM90 deterministic relay: none of the cluster sizes %s can launch here "
        "(last probe error: %r); leaving deterministic_relay_cluster_size to "
        "cuDNN's default, which may fail with cudaErrorInvalidClusterSize.",
        _SM90_RELAY_CLUSTER_SIZES,
        last_error,
    )
    return None


def cudnn_deterministic_kwargs(wrapper):
    """Extra kwargs that make a vendored cuDNN wrapper deterministic.

    ``{"deterministic": True}`` when FLAGS_cudnn_deterministic is on and the
    wrapper has the option (cudnn-frontend >= 1.29 for the sparse attention
    backward, >= 1.32.1 for the indexer backward); ``{}`` otherwise. On SM90 a
    wrapper that takes ``deterministic_relay_cluster_size`` also gets the
    widest relay the GPU can launch (cuDNN's default 16 fails on H20). The
    flag is read on every call (it can be toggled at runtime); only the
    signature probe and the relay probe are cached.
    """
    flag = "FLAGS_cudnn_deterministic"
    if not paddle.get_flags([flag])[flag]:
        return {}
    params = _wrapper_params(wrapper)
    if "deterministic" not in params:
        return {}
    kwargs = {"deterministic": True}
    if (
        "deterministic_relay_cluster_size" in params
        and paddle.device.cuda.get_device_capability()[0] == 9
    ):
        relay = _sm90_relay_cluster_size()
        if relay is not None:
            kwargs["deterministic_relay_cluster_size"] = relay
    return kwargs


def sort_topk_ascending_pad_last(topk_idxs):
    """Sort each top-k row ascending with ``-1`` padding kept at the row tail.

    ``-1`` maps to INT32_MAX for the sort and back afterwards, so the set of
    selected columns is unchanged and a dense-prefix layout (``topk_length``)
    stays a dense prefix. Attention softmaxes over a set, so the reorder only
    moves the accumulation order, not the value.
    """
    big = paddle.full_like(topk_idxs, 2**31 - 1)
    key = paddle.sort(paddle.where(topk_idxs >= 0, topk_idxs, big), axis=-1)
    return paddle.where(key == big, paddle.full_like(key, -1), key)
