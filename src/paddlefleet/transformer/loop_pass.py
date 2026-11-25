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

"""Per-pass state of the layers a looped decoder replays.

A looped decoder runs the same decoder layers more than once per forward.
A module can keep a separate copy of part of its state for every pass; here
``TopKRouter`` its load-balancing state -- the aux-loss-free correction bias
(``add_loop_pass_bias_states``, ``topk_method='noaux_tc'``). The loop puts the pass number into the layers'
``dict_args`` under :data:`LOOP_PASS_KEY` while it replays pass 2, 3, ..; the
layers hand it on as the ``loop_pass`` argument to every module that has
per-pass state (:func:`loop_pass_kwargs`). No key means pass 1, which uses the
module's own state, so a model without the copies is unchanged.

The pass number travels as an argument rather than as module state: a
recompute replays a module with the arguments of its forward, so it reads the
pass it ran in.
"""

LOOP_PASS_KEY = "loop_pass"


def loop_pass_kwargs(module, loop_pass):
    """``{"loop_pass": loop_pass}`` for a module with per-pass state, else ``{}``.

    Only modules that added per-pass state take the argument, so callers can
    pass it unconditionally and every other module keeps its signature.
    """
    if loop_pass is None or not getattr(module, "has_loop_pass_state", False):
        return {}
    return {LOOP_PASS_KEY: loop_pass}


def check_num_passes(num_passes, owner):
    num_passes = int(num_passes)
    if num_passes < 2:
        raise ValueError(
            f"{owner}: per-pass state needs at least 2 passes, got {num_passes}."
        )
    return num_passes


def check_loop_pass(loop_pass, num_passes, owner):
    """Validate ``loop_pass``; returns it as an int, 1 for None."""
    if loop_pass is None:
        return 1
    loop_pass = int(loop_pass)
    if not 1 <= loop_pass <= num_passes:
        raise ValueError(
            f"{owner}: loop_pass {loop_pass} is outside the {num_passes} passes "
            "it has state for."
        )
    return loop_pass
