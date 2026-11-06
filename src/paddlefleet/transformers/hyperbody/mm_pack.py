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

"""HyperBody multimodal inputs <-> pure-Tensor packing (fleet-pipeline friendly).

Background
----------
The fleet pipeline micro-batch loader (``paddle/.../pipeline_parallel.py``) only
accepts **Tensor/None** fields (it ``.detach()`` / slices each field), it cannot
carry python lists. HyperBody's packed multimodal input is per-segment lists::

    image = [Tensor(C, H, W) | None, ...]   # one per context segment (varying size)
    audio = [Tensor(M, L)    | None, ...]

This module packs them into "pure Tensors + metadata" and losslessly restores the
original lists after the loader. Query selection needs no per-segment data:
every context segment uses the single ``query`` table.

Metadata design (ragged/jagged tensor, self-describing, modality-agnostic)
--------------------------------------------------------------------------
A list of "per-segment optional tensor" is represented by 3 Tensors::

    {prefix}_values    : 1-D, original dtype, all present tensors flattened + concat
    {prefix}_shapes    : int64 [K, R], shape of each present tensor (R = modality rank)
    {prefix}_seg_index : int32 [S], segment s -> index k in the present list, else -1

K = number of present tensors, S = number of segments; per-segment byte offset is
the running sum of prod(shapes), not stored separately (minimal redundancy).
Empty list: values=[0] float32, shapes=[0,0], seg_index all -1.

Image payloads
--------------
The ``hb_image_*`` keys carry one of two layouts, chosen by the encoder switch
``hyperencoder_batched_patch_embed`` (data side and model side read the same
switch):

* off (default): each image is raw pixels ``[C, H, W]``.
* on: each image is patch rows ``[H/p, W/p, C*p*p]`` from
  ``modality_encoders.image_to_patch_rows``, so ``hb_image_values`` reshapes to
  the whole ``[sum(h*w), C*p*p]`` patch matrix of the micro-batch.

Call sites
----------
* Data side (erniebot ``_hyperbody_hack_inputs`` / a future real collator):
  ``batch.update(pack_hyperbody_mm(image, audio))`` then drop the original list
  keys, so the batch is Tensor-only and can enter the fleet pipeline. With the
  switch on, ``image`` holds patch rows.
* Model side (top of ``HyperBodyEncoderFrontEnd.forward``):
  ``unpack_hyperbody_mm(dict_args, image_as_patches=<switch>)`` restores the
  inputs in place and cleans up the packing keys.
"""

from __future__ import annotations

import paddle

__all__ = [
    "pack_optional_tensor_list",
    "unpack_optional_tensor_list",
    "pack_hyperbody_mm",
    "unpack_hyperbody_mm",
    "HB_IMAGE_PREFIX",
    "HB_AUDIO_PREFIX",
]

HB_IMAGE_PREFIX = "hb_image"
HB_AUDIO_PREFIX = "hb_audio"


def pack_optional_tensor_list(items, prefix):
    """list[Tensor|None] -> the 3 Tensors {prefix}_values/_shapes/_seg_index."""
    seg_index = [-1] * len(items)
    flats, shapes, dtype, k = [], [], None, 0
    for s, t in enumerate(items):
        if t is None:
            continue
        if dtype is None:
            dtype = t.dtype
        seg_index[s] = k
        shapes.append([int(d) for d in t.shape])
        flats.append(t.reshape([-1]).astype(dtype))
        k += 1
    if k == 0:
        values = paddle.zeros([0], dtype="float32")
        shapes_t = paddle.zeros([0, 0], dtype="int64")
    else:
        values = paddle.concat(flats, axis=0)
        shapes_t = paddle.to_tensor(shapes, dtype="int64")
    return {
        f"{prefix}_values": values,
        f"{prefix}_shapes": shapes_t,
        f"{prefix}_seg_index": paddle.to_tensor(seg_index, dtype="int32"),
    }


def unpack_optional_tensor_list(packed, prefix):
    """Restore the output of pack_optional_tensor_list -> list[Tensor|None] (length S)."""
    values = packed[f"{prefix}_values"]
    shapes_t = packed[f"{prefix}_shapes"]
    seg_idx = packed[
        f"{prefix}_seg_index"
    ].tolist()  # small int tensor, host-side control flow
    present, off = [], 0
    if int(shapes_t.shape[0]) > 0:
        for shp in shapes_t.tolist():
            shp = [int(x) for x in shp]
            n = 1
            for d in shp:
                n *= d
            present.append(values[off : off + n].reshape(shp))
            off += n
    return [present[i] if i >= 0 else None for i in seg_idx]


def pack_hyperbody_mm(image, audio):
    """Pack HyperBody image/audio lists into a pure-Tensor dict."""
    packed = {}
    packed.update(pack_optional_tensor_list(image, HB_IMAGE_PREFIX))
    packed.update(pack_optional_tensor_list(audio, HB_AUDIO_PREFIX))
    return packed


def unpack_hyperbody_mm(dict_args, image_as_patches=False):
    """Restore image/audio in place and drop the packing keys; no-op if absent.

    ``audio`` becomes a per-segment list. ``image`` does too by default; with
    ``image_as_patches`` (the ``hyperencoder_batched_patch_embed`` layout) the
    image keys are instead kept whole as
    ``dict_args["image_patches"] = (values, shapes, seg_index)``: ``values`` is
    the flat concat of all patch rows, ``shapes`` a host-side ``[K, 3]`` list,
    ``seg_index`` a host-side ``[S]`` list.
    """
    if f"{HB_IMAGE_PREFIX}_seg_index" not in dict_args:
        return dict_args
    if image_as_patches:
        dict_args["image_patches"] = (
            dict_args[f"{HB_IMAGE_PREFIX}_values"],
            dict_args[f"{HB_IMAGE_PREFIX}_shapes"].tolist(),
            dict_args[f"{HB_IMAGE_PREFIX}_seg_index"].tolist(),
        )
    else:
        dict_args["image"] = unpack_optional_tensor_list(
            dict_args, HB_IMAGE_PREFIX
        )
    dict_args["audio"] = unpack_optional_tensor_list(dict_args, HB_AUDIO_PREFIX)
    for key in (
        f"{HB_IMAGE_PREFIX}_values",
        f"{HB_IMAGE_PREFIX}_shapes",
        f"{HB_IMAGE_PREFIX}_seg_index",
        f"{HB_AUDIO_PREFIX}_values",
        f"{HB_AUDIO_PREFIX}_shapes",
        f"{HB_AUDIO_PREFIX}_seg_index",
    ):
        dict_args.pop(key, None)
    return dict_args
