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

"""Image / audio modality towers for HyperEncoder.

This module defines the small vision and audio front-ends that turn raw
patches / mel-spectrogram frames into a shared embedding space, plus the
projector that maps those embeddings into the language model width.

Design notes:

* ``MlpProjector`` exposes its submodule as the attribute ``layers`` because
  the checkpoint weight keys are ``projector.layers.weight``. This name is kept
  consistent across all projector types (including the ``nn.Sequential`` case).
* ``ImageEncoderConv`` accepts an ``out_chans`` argument that it does not use.
  It is retained only to keep the constructor signature stable.
* ``AudioEncoderConv`` returns a 4-D channel-last tensor ``[B, 1, T', 768]``
  (``h'=1, w'=T'``) so it is structurally identical to the image tower output
  ``[B, h', w', 768]``. Downstream code relies on this shape equivalence to
  reuse a single ``permute(0,3,1,2).flatten(2).transpose(1,2)`` path for both
  modalities.

Numerical details:

* ``F.interpolate(align_corners=False)`` is used for position-embedding
  resampling. Both interpolation helpers short-circuit when the source and
  target sizes already match, returning the input unchanged.
* ``F.gelu`` is called with ``approximate=False`` explicitly so it uses the
  exact erf formulation rather than the tanh approximation.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import paddle
import paddle.nn.functional as F
from paddle import nn

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "MlpProjector",
    "PatchEmbed",
    "ImageEncoderConv",
    "AudioEncoderConv",
    "get_abs_pos_2d",
    "get_abs_pos_1d",
    "image_to_patch_rows",
]


class MlpProjector(nn.Layer):
    """Modality-level projector: 768 -> 1280. Shared by the image and audio towers.

    Only ``projector_type="linear"`` is used in practice; the ``identity`` and
    ``mlp_gelu`` variants are kept for completeness. All variants store their
    submodule under ``self.layers`` so the checkpoint weight keys stay stable.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        projector_type = cfg["projector_type"]
        input_dim = cfg["input_dim"]
        n_embed = cfg["n_embed"]
        depth = cfg.get("depth", 1)

        if projector_type == "identity":
            modules: nn.Layer = nn.Identity()
        elif projector_type == "linear":
            modules = nn.Linear(input_dim, n_embed)
        elif projector_type == "mlp_gelu":
            layers = [nn.Linear(input_dim, n_embed)]
            for _ in range(1, depth):
                layers.append(nn.GELU())
                layers.append(nn.Linear(n_embed, n_embed))
            modules = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")

        # Attribute name must be ``layers``; weight key is ``projector.layers.weight``.
        self.layers = modules

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply ``self.layers(x)``."""
        return self.layers(x)


class PatchEmbed(nn.Layer):
    """Image -> patch embedding via a single Conv2d."""

    def __init__(
        self,
        kernel_size: tuple[int, int] = (14, 14),
        stride: tuple[int, int] = (14, 14),
        padding: tuple[int, int] = (0, 0),
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.kernel_size = tuple(kernel_size)
        self.stride = tuple(stride)
        self.padding = tuple(padding)
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.proj(x)
        # B C H W -> B H W C
        return x.transpose([0, 2, 3, 1])

    def forward_patches(self, patches: paddle.Tensor) -> paddle.Tensor:
        """Patch rows ``[N, C*kh*kw]`` -> ``[N, embed_dim]``, one GEMM.

        Each row is one non-overlapping ``kh x kw`` block flattened in
        ``(c, u, v)`` order (``image_to_patch_rows``). With
        ``kernel == stride`` and no padding the conv is exactly this matmul
        with ``weight.reshape([embed_dim, -1])``; other geometries raise.
        """
        if self.kernel_size != self.stride or any(self.padding):
            raise ValueError(
                "PatchEmbed.forward_patches needs kernel_size == stride and "
                f"zero padding, got kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}"
            )
        weight = self.proj.weight.reshape([self.proj.weight.shape[0], -1])
        return F.linear(patches, weight.t(), self.proj.bias)


def _aligned_interp_1d_axis(
    x: paddle.Tensor, axis: int, dst_size: int
) -> paddle.Tensor:
    """1-D ``align_corners=False`` linear interpolation along ``axis``.

    Hand-decomposed as ``v0 + (v1 - v0) * lambda`` in fp32 (matching PyTorch's
    ``_upsample_linear``): ``src(d) = max((d+0.5)*src/dst - 0.5, 0)``,
    ``i0 = floor(src)``, ``i1 = min(i0+1, src-1)``, ``lambda = src - i0``.
    Uses ``index_select`` + elementwise ops so it never launches the built-in
    bilinear kernel (which builds an invalid CUDA launch config for output
    dim == 1); its gradient is provided by autograd.
    """
    src_size = int(x.shape[axis])
    scale = src_size / dst_size
    d = paddle.arange(dst_size, dtype="float32")
    src = paddle.clip((d + 0.5) * scale - 0.5, min=0.0)
    i0f = paddle.floor(src)
    lam = src - i0f
    i0 = i0f.astype("int64")
    i1 = paddle.clip(i0 + 1, max=src_size - 1)
    v0 = paddle.index_select(x, i0, axis=axis)
    v1 = paddle.index_select(x, i1, axis=axis)
    shp = [1] * x.ndim
    shp[axis] = dst_size
    lam_b = lam.reshape(shp)
    return v0 + (v1 - v0) * lam_b


def _aligned_bilinear_2d(
    old: paddle.Tensor, out_h: int, out_w: int
) -> paddle.Tensor:
    """``[1,C,Hs,Ws]`` fp32 -> ``[1,C,out_h,out_w]``, separable W then H."""
    x = _aligned_interp_1d_axis(old, axis=3, dst_size=out_w)  # W
    x = _aligned_interp_1d_axis(x, axis=2, dst_size=out_h)  # H
    return x


def _abs_pos_use_legacy_interp() -> bool:
    """Whether to resample the 2-D position embedding with the *original*
    ``F.interpolate`` bilinear kernel instead of the default hand-decomposed
    path.

    Default (env unset / ``0``) is ``_aligned_bilinear_2d``, which avoids the
    ``cudaErrorInvalidValue`` that ``F.interpolate`` raises when an output
    spatial dim is 1 (e.g. a ``(1, 3)`` grid from a tiny image). Set
    ``HYPERBODY_ABS_POS_LEGACY_INTERP=1`` to restore the original
    ``F.interpolate(mode="bilinear", align_corners=False)`` call.
    """
    return os.environ.get("HYPERBODY_ABS_POS_LEGACY_INTERP", "0") not in (
        "",
        "0",
        "false",
        "False",
    )


def get_abs_pos_2d(
    abs_pos: paddle.Tensor, tgt_size: tuple[int, int]
) -> paddle.Tensor:
    """Resample a 2-D positional embedding to the actual patch grid.

    When the source grid size already equals ``tgt_size`` the input is returned
    unchanged (short-circuit). Otherwise the embedding is transposed to
    channel-first, cast to fp32, resized with a hand-decomposed bilinear
    (``_aligned_bilinear_2d``, ``align_corners=False``) by default, cast back
    and transposed to channel-last. The manual decomposition replaces
    ``F.interpolate``, which raises ``cudaErrorInvalidValue`` when an output
    spatial dim is 1 (e.g. a (1,3) grid from a tiny image). Setting
    ``HYPERBODY_ABS_POS_LEGACY_INTERP=1`` switches back to the original
    ``F.interpolate`` bilinear kernel.
    """
    src_size = (int(abs_pos.shape[1]), int(abs_pos.shape[2]))
    if src_size != tuple(int(v) for v in tgt_size):
        old = abs_pos.transpose([0, 3, 1, 2]).astype("float32")
        if _abs_pos_use_legacy_interp():
            new = F.interpolate(
                old,
                size=[int(tgt_size[0]), int(tgt_size[1])],
                mode="bilinear",
                align_corners=False,
            ).astype(abs_pos.dtype)
        else:
            new = _aligned_bilinear_2d(
                old, int(tgt_size[0]), int(tgt_size[1])
            ).astype(abs_pos.dtype)
        return new.transpose([0, 2, 3, 1])
    return abs_pos


def get_abs_pos_1d(abs_pos: paddle.Tensor, tgt_size: int) -> paddle.Tensor:
    """Resample a 1-D positional embedding. Short-circuits when sizes match."""
    src_size = int(abs_pos.shape[1])
    if src_size != int(tgt_size):
        old = abs_pos.transpose([0, 2, 1]).astype("float32")
        new = F.interpolate(
            old, size=[int(tgt_size)], mode="linear", align_corners=False
        )
        return new.astype(abs_pos.dtype).transpose([0, 2, 1])
    return abs_pos


def image_to_patch_rows(image, patch_size: int = 14):
    """``[C, H, W]`` -> ``[H//p, W//p, C*p*p]``, the layout of ``forward_patches``.

    Each non-overlapping ``p x p`` block is flattened in ``(c, u, v)`` order,
    blocks row-major over the patch grid; trailing pixels are dropped like a
    ``Conv2D(stride=p)`` without padding. Pure data movement, so it can run on
    the data side. Works on numpy arrays and paddle Tensors alike (returns the
    same kind as ``image``).
    """
    c, h, w = (int(d) for d in image.shape)
    p = int(patch_size)
    gh, gw = h // p, w // p
    x = image[:, : gh * p, : gw * p].reshape([c, gh, p, gw, p])
    return x.transpose([1, 3, 0, 2, 4]).reshape([gh, gw, c * p * p])


class ImageEncoderConv(nn.Layer):
    """Patch embedding plus a learnable 2-D positional embedding.

    Default constructor arguments:
    ``img_size=728, patch_size=14, in_chans=3, embed_dim=768, out_chans=256``.

    ``out_chans`` is accepted but never used; it is kept to preserve a stable
    constructor signature.
    """

    def __init__(
        self,
        img_size: int = 728,
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 768,
        out_chans: int = 256,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        grid = img_size // patch_size  # 728 // 14 = 52
        # Learnable parameter (not a buffer), initialized to all zeros.
        self.pos_embed = self.create_parameter(
            shape=[1, grid, grid, embed_dim],
            default_initializer=nn.initializer.Constant(0.0),
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """``[B, 3, H, W]`` -> ``[B, H//p, W//p, embed_dim]``."""
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + get_abs_pos_2d(self.pos_embed, (x.shape[1], x.shape[2]))
        return x

    def forward_patches(
        self, patches: paddle.Tensor, grids: Sequence[tuple[int, int]]
    ) -> paddle.Tensor:
        """Patch rows of many images -> ``[sum(h*w), embed_dim]``, one GEMM.

        ``patches`` is ``[sum(h*w), C*p*p]`` (``p = patch_size``): the data
        side cut every image into its non-overlapping ``p x p`` blocks with
        ``image_to_patch_rows`` and stacked them image after image. ``grids``
        lists each image's patch grid ``(h, w)`` in the same order. The patch
        embed is ``PatchEmbed.forward_patches``; the pos embed is resampled
        once per distinct grid.
        """
        x = self.patch_embed.forward_patches(patches)
        if self.pos_embed is not None and grids:
            pos = {}
            for g in grids:
                if g not in pos:
                    pos[g] = get_abs_pos_2d(self.pos_embed, g).reshape(
                        [g[0] * g[1], -1]
                    )
            x = x + paddle.concat([pos[g] for g in grids], axis=0)
        return x


class AudioEncoderConv(nn.Layer):
    """Mel-spectrogram convolutions plus a learnable 1-D positional embedding.

    Default constructor arguments: ``num_mel_bins=128, embed_dim=768``.

    ``conv2`` uses ``stride=2``, which is why the output length is
    ``T' = (T+1)//2``. ``max_position_embeddings=1500`` sets the positional
    embedding length.
    """

    def __init__(
        self,
        num_mel_bins: int = 128,
        embed_dim: int = 768,
        max_position_embeddings: int = 1500,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1D(
            num_mel_bins, embed_dim, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1D(
            embed_dim, embed_dim, kernel_size=3, stride=2, padding=1
        )
        self.pos_embed = self.create_parameter(
            shape=[1, max_position_embeddings, embed_dim],
            default_initializer=nn.initializer.Constant(0.0),
        )

    def forward(self, input_features: paddle.Tensor) -> paddle.Tensor:
        """``[B, 128, T]`` -> ``[B, 1, T', 768]``, with ``T' = (T+1)//2``.

        The output is deliberately 4-D channel-last (``h'=1, w'=T'``) so it
        matches the image tower's output shape.

        Note: the conv layers must accumulate in fp32 for numerical stability.
        """
        # approximate=False forces the exact erf GELU rather than the tanh form.
        x = F.gelu(self.conv1(input_features), approximate=False)
        x = F.gelu(self.conv2(x), approximate=False)
        x = x.transpose([0, 2, 1])  # [B, T', embed_dim]
        x = x + get_abs_pos_1d(self.pos_embed, x.shape[1])
        return x.unsqueeze(1)  # [B, 1, T', embed_dim]

    def forward_batch(
        self, audios: Sequence[paddle.Tensor]
    ) -> tuple[paddle.Tensor, list[int]]:
        """Many ``[M, T_i]`` mels -> ``([sum(T'_i), D], [T'_i])``.

        Returns the stacked features and each item's row count
        ``T'_i = (T_i+1)//2``; row block ``i`` equals
        ``forward(audios[i][None])[0, 0]``. All items
        are laid end to end on one time axis, each starting at an even frame
        and followed by at least one zero frame, and the convs run once over
        that ``[1, M, L]`` sequence (``L`` exceeds ``sum(T_i)`` by at most two
        frames per item):

        * ``conv1`` (kernel 3, stride 1) reads one frame past each end of an
          item: the zero gap, i.e. what the per-item path sees as padding.
        * The gap frames of the ``conv1`` output are zeroed again (the conv
          adds its bias there) before ``conv2``.
        * ``conv2`` (kernel 3, stride 2) output ``j`` reads ``conv1`` frames
          ``2j-1 .. 2j+1``. With an even start ``s`` the item's outputs are
          ``s/2 .. s/2+T'-1``; they read the gap before ``s`` and, for odd
          ``T``, the gap at ``s+T``, again matching the per-item padding.

        The pos embed is resampled once per distinct ``T'_i``.
        """
        lengths = [int(a.shape[1]) for a in audios]
        if not lengths:
            empty = paddle.zeros([0, self.pos_embed.shape[-1]])
            return empty.astype(self.pos_embed.dtype), []
        starts, total = [], 0
        for t in lengths:
            starts.append(total)
            total += t + 1  # at least one zero frame after each item
            total += total % 2  # next item starts at an even frame
        # Gather source per packed frame; gaps point at one appended zero column.
        zero_col = sum(lengths)
        flat = paddle.concat(
            [*audios, paddle.zeros([audios[0].shape[0], 1], audios[0].dtype)],
            axis=1,
        )
        src = np.full(total, zero_col, dtype=np.int64)
        keep = np.zeros(total, dtype=bool)
        offset = 0
        for s, t in zip(starts, lengths):
            src[s : s + t] = np.arange(offset, offset + t)
            keep[s : s + t] = True
            offset += t
        x = paddle.index_select(flat, paddle.to_tensor(src), axis=1)
        x = F.gelu(self.conv1(x.unsqueeze(0)), approximate=False)
        x = x * paddle.to_tensor(keep[None, None, :]).astype(x.dtype)
        x = F.gelu(self.conv2(x), approximate=False)
        x = x[0].transpose([1, 0])  # [(total+1)//2, D]
        out_lens = [(t + 1) // 2 for t in lengths]
        rows = np.concatenate(
            [s // 2 + np.arange(t) for s, t in zip(starts, out_lens)]
        )
        x = paddle.index_select(x, paddle.to_tensor(rows), axis=0)
        pos = {}
        for t in out_lens:
            if t not in pos:
                pos[t] = get_abs_pos_1d(self.pos_embed, t)[0]
        return x + paddle.concat([pos[t] for t in out_lens], axis=0), out_lens
