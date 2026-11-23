# !/usr/bin/env python3
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

"""Custom lr schedule

Spec: ``cosine[:<num_decay_steps>]``; ``num_decay_steps`` may also be a group key.
"""

import math

from .base import BaseLRScheduler, parse_nonneg_int, register_lr_scheduler


class CosineLR(BaseLRScheduler):
    """Linear warmup, then half-cosine decay from the peak to ``min_lr``.

    ``num_decay_steps`` is the (absolute) step at which the cosine reaches
    ``min_lr``; the lr then stays at ``min_lr`` until training ends. Unset =
    ``num_training_steps`` (the decay spans warmup..max_steps, as before).
    Up to ``num_decay_steps`` the values are identical to a cosine whose
    ``max_steps`` equals it, so a run can be extended past the end of the
    decay without changing the lr of steps already trained.
    """

    def __init__(
        self,
        learning_rate: float,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        num_cycles: float = 0.5,
        num_decay_steps=None,
    ):
        self.num_cycles = num_cycles
        if num_decay_steps is not None and num_decay_steps <= num_warmup_steps:
            raise ValueError(
                f"cosine num_decay_steps={num_decay_steps} must be > "
                f"warmup_steps={num_warmup_steps}."
            )
        # After the decay the lr is held at the cosine's end value, which is
        # min_lr only for a half cycle.
        if num_decay_steps is not None and not math.isclose(num_cycles, 0.5):
            raise ValueError(
                f"cosine num_decay_steps requires num_cycles=0.5, "
                f"got num_cycles={num_cycles}."
            )
        self.num_decay_steps = (
            num_training_steps if num_decay_steps is None else num_decay_steps
        )
        super().__init__(
            learning_rate,
            num_warmup_steps,
            num_training_steps,
            min_lr,
            last_epoch,
        )

    @classmethod
    def get_param(cls):
        return {"num_decay_steps": parse_nonneg_int}

    def schedule(self, current_step):
        num_warmup_steps = self.num_warmup_steps
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, self.num_decay_steps - num_warmup_steps)
        )
        # Past the end of the decay hold at min_lr (otherwise the cosine
        # would turn back up).
        progress = min(progress, 1.0)
        ratio = max(
            0.0,
            0.5
            * (
                1.0
                + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)
            ),
        )
        min_lr, learning_rate = self.min_lr, self.learning_rate
        return ratio * (1 - min_lr / learning_rate) + min_lr / learning_rate


register_lr_scheduler("cosine", CosineLR)


def get_cosine_schedule_with_warmup(
    learning_rate: float,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    min_lr: float = 0.0,
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        learning_rate (float)
            The initial learning rate. It is a python float number.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
    Return:
        `CosineLR` (a `paddle.optimizer.lr.LambdaDecay`) with the appropriate schedule.
    """
    return CosineLR(
        learning_rate,
        num_warmup_steps,
        num_training_steps,
        min_lr=min_lr,
        last_epoch=last_epoch,
        num_cycles=num_cycles,
    )
