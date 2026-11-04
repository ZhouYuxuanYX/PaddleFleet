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

"""Determinism switches injected by ``--test-accuracy``.

Some kernels and features are numerically fine for throughput runs but make an
accuracy comparison unreproducible (aadiff): the same config replayed on the
same data yields slightly different losses.  Accuracy adaptation therefore
pins them to their deterministic variant.

Adding a switch = one more row in :data:`PRECISION_SWITCHES`.  Each row says
which document owns the key by default; when the key already exists in the
YAML *and* in ``model_config.json`` both copies are pinned, because either one
can feed the framework's transformer config.
"""

from __future__ import annotations

from collections import namedtuple

PrecisionSwitch = namedtuple(
    "PrecisionSwitch", ["target", "key", "value", "reason"]
)

PRECISION_SWITCHES = (
    PrecisionSwitch(
        target="yaml",
        key="csa_sparse_attn_backend",
        value="tilelang",
        reason=(
            "精度对齐：HCA/CSA 稀疏注意力的 cudnn 后端"
            "（FlashMLA 前向 + cuDNN 反向）与 tilelang 结果不一致，"
            "精度测试统一走 tilelang"
        ),
    ),
    PrecisionSwitch(
        target="yaml",
        key="csa_indexer_backend",
        value="tilelang",
        reason=(
            "精度对齐：indexer 的 cudnn top-k 与 tilelang 存在差异，"
            "与 sparse attention 后端取齐为 tilelang"
        ),
    ),
    PrecisionSwitch(
        target="yaml",
        key="mqa_sparse_attn_backward_backend",
        value="tilelang",
        reason=(
            "精度对齐：absorbed-MQA（csa_compress_ratios=-2）层的 dKV 反向"
            "默认走 cuDNN，其原子累加不可逐位复现，"
            "与 CSA 后端一起统一切到确定性的 tilelang"
        ),
    ),
    PrecisionSwitch(
        target="json",
        key="multimax_modules",
        value=None,
        reason=(
            "精度对齐：multimax 的可学习 range/ts 会引入非确定性，"
            "精度测试关闭（null 等价于不启用）"
        ),
    ),
)


def normalize_skip_keys(skip_keys):
    """Return ``skip_keys`` as a frozenset; unknown keys raise ``ValueError``.

    A bare string is one key, not a sequence of characters.
    """
    if isinstance(skip_keys, str):
        skip_keys = (skip_keys,)
    keys = frozenset(skip_keys)
    known = [switch.key for switch in PRECISION_SWITCHES]
    unknown = keys.difference(known)
    if unknown:
        raise ValueError(
            f"未知的精度开关：{', '.join(sorted(unknown))}"
            f"（可选：{', '.join(known)}）"
        )
    return keys


def _resolve_targets(switch, yaml_config, model_config):
    """Documents ``switch`` must be written to; empty when none is writable."""
    targets = []
    if yaml_config is not None and switch.key in yaml_config:
        targets.append("yaml")
    if model_config is not None and switch.key in model_config:
        targets.append("json")
    if targets:
        return targets
    if switch.target == "json" and model_config is None:
        return []
    return [switch.target]


def plan_precision_switches(yaml_config, model_config=None, skip_keys=()):
    """Decide where each determinism switch must be written.

    Returns ``(applied, skipped)``:

    * ``applied`` -- list of ``(target, key, value, reason)`` where ``target``
      is ``"yaml"`` or ``"json"``;
    * ``skipped`` -- human-readable notes for switches that had nowhere to go
      (e.g. a ``json`` switch while no ``model_config.json`` is being written)
      or that the caller excluded via ``skip_keys``.

    ``skip_keys`` goes through :func:`normalize_skip_keys`, so an unknown key
    raises ``ValueError`` instead of silently matching nothing.
    """
    skip_keys = normalize_skip_keys(skip_keys)
    applied = []
    skipped = []

    for switch in PRECISION_SWITCHES:
        if switch.key in skip_keys:
            skipped.append(
                f"{switch.key}：通过 --skip-precision-switch 保留原值，未注入"
            )
            continue

        targets = _resolve_targets(switch, yaml_config, model_config)
        if not targets:
            skipped.append(
                f"{switch.key}：yaml 与 model_config.json 均未声明该字段，"
                f"且本次没有可写的 model_config.json，已跳过"
            )
            continue

        for target in targets:
            applied.append((target, switch.key, switch.value, switch.reason))

    return applied, skipped
