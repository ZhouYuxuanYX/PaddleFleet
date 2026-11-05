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
"""Encoder-only expert-parallel groups for HyperBody (experimental).

The decoder keeps the global HCG EP / moe_sharding groups. The encoder MoE gets
its own ``ProcessGroupCollection`` whose ``ep`` / ``expt_dp`` are:

* encoder EP group: a contiguous slice of one decoder EP group, so encoder EP
  nests inside decoder EP (decoder EP must be a multiple of encoder EP);
* encoder expert-data group: the ranks holding the same encoder experts, i.e.
  the same in-slice offset across every slice of every decoder EP group.

``paddle.distributed.new_group`` is collective, so every rank creates every
group in the same order. Groups are cached so all encoder layers share one
Group object (one comm buffer per group in the sharding optimizer).
"""

from __future__ import annotations

import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from paddlefleet.process_groups_config import ProcessGroupCollection
from paddlefleet.tensor_parallel.random import add_encoder_expert_parallel_rng
from paddlefleet.utils.log import logger

_ENCODER_PGS: dict[int, ProcessGroupCollection] = {}


def get_encoder_ep_degree():
    """EP size of the encoder-only expert groups built in this process.

    None when no encoder-only groups exist (encoder experts use the global EP
    layout). Checkpoint metadata records this to detect encoder EP changes.
    """
    sizes = {pgs.ep.nranks for pgs in _ENCODER_PGS.values()}
    if not sizes:
        return None
    if len(sizes) != 1:
        raise RuntimeError(f"several encoder EP sizes in one process: {sizes}")
    return sizes.pop()


def check_encoder_ep_checkpoint_compat(args) -> None:
    """Reject checkpoint paths that assume a single (global) expert layout.

    ``args`` is the ``TrainingArguments``; the Trainer calls this once the
    model has built encoder-only groups. flex_checkpoint save / load / reshard
    and HF export are metadata driven and support encoder-only expert groups.
    ZCC on flex_checkpoint (the FcBased callback/worker) saves the same
    sharded_state_dict descriptors and resumes through the normal flex loader,
    so it is allowed. The paths below read the global EP / moe_sharding groups
    or degrees only, so with encoder-only groups they would silently write or
    read wrong encoder expert shards.
    """
    unsupported = []
    save_format = getattr(args, "save_checkpoint_format", None)
    load_format = getattr(args, "load_checkpoint_format", None)
    if getattr(args, "enable_zero_cost_checkpoint", False) and (
        save_format != "flex_checkpoint" or load_format != "flex_checkpoint"
    ):
        unsupported.append(
            "enable_zero_cost_checkpoint with save_checkpoint_format="
            f"{save_format!r} / load_checkpoint_format={load_format!r} "
            "(legacy ZCC sharding meta records only the global EP layout; "
            "set both formats to 'flex_checkpoint')"
        )
    if getattr(args, "load_sharded_model", False) and load_format != (
        "flex_checkpoint"
    ):
        unsupported.append(
            f"load_sharded_model with load_checkpoint_format={load_format!r} "
            "(legacy ShardingIO reshard only compares global degrees)"
        )
    if unsupported:
        raise NotImplementedError(
            "hyperencoder_expert_model_parallel_size builds encoder-only expert "
            "groups, which these checkpoint paths do not support: "
            + "; ".join(unsupported)
            + ". Use flex_checkpoint (or load_from_hf) instead."
        )


def check_paddle_bucket_keys() -> None:
    """Fail fast on a Paddle whose Muon/Mixed sharding keys 2D buckets by color.

    Encoder and decoder experts both use the color 'moe_expert'. Before
    PaddlePaddle/Paddle#79913 the keep-whole (2D) buckets are keyed by the
    color name only, so decoder Muon experts would be partitioned, averaged and
    broadcast on the encoder expert-data group: silently wrong. Only needed
    when encoder experts are colored, i.e. encoder EP > 1; skipped when
    ``encoder_config.hyperencoder_allow_color_buckets`` is set.
    """
    from paddle.distributed.fleet.meta_optimizers.mixed_sharding_optimizer import (
        MixedShardingOptimizer,
    )
    from paddle.distributed.fleet.meta_optimizers.muon_sharding_optimizer import (
        MuonShardingOptimizer,
    )

    old = [
        c.__name__
        for c in (MuonShardingOptimizer, MixedShardingOptimizer)
        if not hasattr(c, "_param_bucket_key")
    ]
    if old:
        raise NotImplementedError(
            "hyperencoder_expert_model_parallel_size > 1 builds encoder-only "
            f"expert groups, but this Paddle's {', '.join(old)} keys 2D buckets "
            "by color name only (needs PaddlePaddle/Paddle#79913). Use a Paddle "
            "with that fix, or set encoder_config.hyperencoder_allow_color_buckets "
            "if the optimizer is not Muon/Mixed sharding."
        )


def build_encoder_ep_rank_lists(dec_ep_lists, encoder_ep):
    """Pure helper: (encoder EP rank lists, encoder expert-data rank lists)."""
    dec_ep = len(dec_ep_lists[0])
    if encoder_ep <= 0 or dec_ep % encoder_ep != 0:
        raise ValueError(
            f"hyperencoder_expert_model_parallel_size={encoder_ep} must divide "
            f"the decoder expert_model_parallel_size={dec_ep}"
        )
    chunks = dec_ep // encoder_ep
    ep_lists = [
        list(ep_list[b * encoder_ep : (b + 1) * encoder_ep])
        for ep_list in dec_ep_lists
        for b in range(chunks)
    ]
    edp_lists = [sorted(ep[j] for ep in ep_lists) for j in range(encoder_ep)]
    return ep_lists, edp_lists


def get_encoder_pg_collection(
    encoder_ep: int, allow_color_buckets: bool = False
) -> ProcessGroupCollection:
    """Return (creating once) the encoder ProcessGroupCollection.

    ``allow_color_buckets`` skips :func:`check_paddle_bucket_keys`.
    """
    if encoder_ep in _ENCODER_PGS:
        pgs = _ENCODER_PGS[encoder_ep]
        # Idempotent; re-adds the stream if the RNG tracker was reset since.
        add_encoder_expert_parallel_rng(pgs.ep.rank)
        return pgs

    if encoder_ep > 1 and not allow_color_buckets:
        check_paddle_bucket_keys()
    hcg = fleet.get_hybrid_communicate_group()
    dec_ep_lists = hcg._moe_topo.get_comm_list("expert")
    ep_lists, edp_lists = build_encoder_ep_rank_lists(dec_ep_lists, encoder_ep)

    rank = dist.get_rank()
    ep_group = edp_group = None
    for ranks in ep_lists:
        g = dist.new_group(ranks)
        if rank in ranks:
            ep_group = g
    for ranks in edp_lists:
        g = dist.new_group(ranks)
        if rank in ranks:
            edp_group = g
    if ep_group is None or edp_group is None:
        raise RuntimeError(
            f"rank {rank} not covered by encoder EP groups {ep_lists}"
        )
    # Communicators may be created lazily; touch both groups once here so the
    # first MoE dispatch / optimizer reduce does not pay (or race on) it.
    for g in (ep_group, edp_group):
        dist.all_reduce(paddle.zeros([1], dtype="float32"), group=g)

    pgs = ProcessGroupCollection.use_mpu_process_groups()
    pgs.ep = ep_group
    pgs.expt_dp = edp_group
    # Encoder experts get their own init RNG stream, seeded by the encoder EP
    # rank (see add_encoder_expert_parallel_rng).
    pgs.expert_rng_tracker_name = add_encoder_expert_parallel_rng(ep_group.rank)
    _ENCODER_PGS[encoder_ep] = pgs
    logger.info(
        f"[HB-ENC-EP] rank={rank} encoder_ep={encoder_ep} "
        f"decoder_ep={len(dec_ep_lists[0])} ep_ranks={list(ep_group.ranks)} "
        f"expt_dp_ranks={list(edp_group.ranks)}"
    )
    return pgs
