// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Fused 1x32 quantization primitives for SM100 W4A8 training.  The numerical
// contract intentionally matches paddlefleet.transformer.moe.fp8_utils:
//   * amax is clamped to 1e-4 before division;
//   * scale is rounded upward to an exact UE8M0 power of two;
//   * E2M1 uses strict midpoint comparisons and packs even K in the low nibble.
//
// 01-kernel adds two vectorized fast paths for mxfp4_stack_quantize_1x32
// (StackContig/StackTranspose Bf16Fp4Quantize1x32VecKernel), used only for
// 16-byte-aligned bfloat16 above kPairKernelMinGroups. They only widen the
// loads and reorder the per-group amax reduction (commutative/associative
// fmaxf), so the FP4 and UE8M0 scale bytes are bitwise identical to the
// original kernels; every other shape/dtype falls through unchanged.
//
// mxfp4_stack_fake_quantize_1x32 and mxfp4_stack_transpose_dequantize_1x32
// return the forward weight DQ(Q1(w)), Q1 = mxfp4_stack_quantize_1x32(w,
// transpose=True), as bf16 [E, R, C] for the W4A8 expert dgrad (a bf16 grouped
// GEMM, see _mxfp4_forward_weight_dequant in fp8_utils). An E2M1 value times a
// UE8M0 scale is exact in fp32 and in bf16, so both reproduce the forward
// weight bit for bit. The dequantize op only reads the transpose=True layout,
// as its name says.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "paddle/extension.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float8_e4m3fn.h"

namespace {

constexpr int kQuantBlock = 32;
constexpr float kAmaxFloor = 1.0e-4f;
constexpr float kFp8Max = 448.0f;
constexpr float kFp4Max = 6.0f;
constexpr int kMaxGridX = 1024 * 1024;
constexpr int kGroupWarpsPerBlock = 8;
constexpr int kGroupThreadsPerBlock = kGroupWarpsPerBlock * 32;
constexpr int64_t kPairKernelMinGroups = 256;

enum class QuantKind : int64_t { kFp8 = 0, kFp4 = 1 };

template <typename T>
__device__ __forceinline__ float LoadAsFloat(const T* ptr, int64_t index) {
  return static_cast<float>(ptr[index]);
}

template <>
__device__ __forceinline__ float LoadAsFloat<__nv_bfloat16>(
    const __nv_bfloat16* ptr, int64_t index) {
  return __bfloat162float(ptr[index]);
}

__device__ __forceinline__ float WarpMax(float value) {
  constexpr unsigned kMask = 0xffffffffu;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(kMask, value, offset));
  }
  return __shfl_sync(kMask, value, 0);
}

__device__ __forceinline__ float HalfWarpMax(float value, unsigned mask) {
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(mask, value, offset, 16));
  }
  return __shfl_sync(mask, value, 0, 16);
}

__device__ __forceinline__ float CeilUe8m0(float value) {
  const uint32_t bits = __float_as_uint(fabsf(value));
  uint32_t exponent = (bits >> 23) & 0xffu;
  exponent += (bits & 0x7fffffu) != 0u;
  exponent = max(1u, min(254u, exponent));
  return __uint_as_float(exponent << 23);
}

__device__ __forceinline__ uint8_t EncodeE2m1Strict(float value) {
  const float absolute = fabsf(value);
  uint8_t code = static_cast<uint8_t>((absolute > 0.25f) + (absolute > 0.75f) +
                                      (absolute > 1.25f) + (absolute > 1.75f) +
                                      (absolute > 2.5f) + (absolute > 3.5f) +
                                      (absolute > 5.0f));
  if (value < 0.0f && code != 0u) {
    code |= 0x8u;
  }
  return code;
}

// Inverse of EncodeE2m1Strict; only the low 4 bits of code are read. The
// magnitudes 0, 0.5 | 1, 1.5, 2, 3, 4, 6 are m / 2 for m < 2 and
// 2^((m >> 1) - 1) * (1 + (m & 1) / 2) above.
__device__ __forceinline__ float DecodeE2m1(uint32_t code) {
  const uint32_t mag = code & 0x7u;
  uint32_t bits = mag < 2u ? mag * 0x3f000000u
                           : (((mag >> 1) + 126u) << 23) | ((mag & 1u) << 22);
  bits |= (code & 0x8u) << 28;
  return __uint_as_float(bits);
}

// Exact 1 / scale for a CeilUe8m0 result. Exponent 254 (only an Inf amax gets
// there) has the subnormal reciprocal 2^-127.
__device__ __forceinline__ float ReciprocalUe8m0(float scale) {
  const uint32_t exponent = __float_as_uint(scale) >> 23;
  return __uint_as_float(exponent < 254u ? (254u - exponent) << 23
                                         : 0x00400000u);
}

inline int GridFor(int64_t work_items) {
  return static_cast<int>(std::min<int64_t>(work_items, kMaxGridX));
}

inline void CheckCudaLaunch(const char* op_name) {
  const cudaError_t error = cudaGetLastError();
  PD_CHECK(error == cudaSuccess,
           op_name,
           " CUDA launch failed: ",
           cudaGetErrorString(error));
}

template <typename T, QuantKind Kind>
__global__ void Quantize1x32Kernel(const T* __restrict__ input,
                                   void* __restrict__ output,
                                   float* __restrict__ scale,
                                   int64_t total_groups) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int64_t group =
           static_cast<int64_t>(blockIdx.x) * kGroupWarpsPerBlock + warp;
       group < total_groups;
       group += static_cast<int64_t>(gridDim.x) * kGroupWarpsPerBlock) {
    const int64_t input_base = group * kQuantBlock;
    float local_amax;

    if constexpr (Kind == QuantKind::kFp8) {
      const float value = LoadAsFloat(input, input_base + lane);
      local_amax = fabsf(value);
      const float amax = WarpMax(local_amax);
      const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp8Max);
      if (lane == 0) {
        scale[group] = group_scale;
      }
      auto* fp8_output = reinterpret_cast<__nv_fp8_e4m3*>(output);
      fp8_output[input_base + lane] =
          static_cast<__nv_fp8_e4m3>(value / group_scale);
    } else {
      float first = 0.0f;
      float second = 0.0f;
      if (lane < 16) {
        first = LoadAsFloat(input, input_base + lane * 2);
        second = LoadAsFloat(input, input_base + lane * 2 + 1);
      }
      local_amax = lane < 16 ? fmaxf(fabsf(first), fabsf(second)) : 0.0f;
      const float amax = WarpMax(local_amax);
      const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp4Max);
      if (lane == 0) {
        scale[group] = group_scale;
      }
      if (lane < 16) {
        const uint8_t low = EncodeE2m1Strict(first / group_scale);
        const uint8_t high = EncodeE2m1Strict(second / group_scale);
        reinterpret_cast<int8_t*>(output)[group * 16 + lane] =
            static_cast<int8_t>(low | (high << 4));
      }
    }
  }
}

template <QuantKind Kind>
__global__ void QuantizeBf16Pair1x32Kernel(
    const __nv_bfloat16* __restrict__ input,
    void* __restrict__ output,
    float* __restrict__ scale,
    int64_t total_groups) {
  const int lane = threadIdx.x & 31;
  const int subwarp = lane >> 4;
  const int sublane = lane & 15;
  const int warp = threadIdx.x >> 5;
  constexpr int kGroupsPerBlock = kGroupWarpsPerBlock * 2;
  for (int64_t group = static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock +
                       warp * 2 + subwarp;
       group < total_groups;
       group += static_cast<int64_t>(gridDim.x) * kGroupsPerBlock) {
    const int64_t input_base = group * kQuantBlock;
    const auto pair = reinterpret_cast<const __nv_bfloat162*>(
        input)[group * (kQuantBlock / 2) + sublane];
    const float first = __bfloat162float(pair.x);
    const float second = __bfloat162float(pair.y);
    const unsigned subwarp_mask = subwarp == 0 ? 0x0000ffffu : 0xffff0000u;
    const float amax =
        HalfWarpMax(fmaxf(fabsf(first), fabsf(second)), subwarp_mask);
    const float quant_max = Kind == QuantKind::kFp8 ? kFp8Max : kFp4Max;
    const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / quant_max);
    if (sublane == 0) {
      scale[group] = group_scale;
    }
    if constexpr (Kind == QuantKind::kFp8) {
      auto* fp8_output = reinterpret_cast<__nv_fp8_e4m3*>(output);
      fp8_output[input_base + sublane * 2] =
          static_cast<__nv_fp8_e4m3>(first / group_scale);
      fp8_output[input_base + sublane * 2 + 1] =
          static_cast<__nv_fp8_e4m3>(second / group_scale);
    } else {
      const uint8_t low = EncodeE2m1Strict(first / group_scale);
      const uint8_t high = EncodeE2m1Strict(second / group_scale);
      reinterpret_cast<int8_t*>(output)[group * 16 + sublane] =
          static_cast<int8_t>(low | (high << 4));
    }
  }
}

template <typename T>
__global__ void StackTransposeFp4Quantize1x32Kernel(const T* __restrict__ input,
                                                    int8_t* __restrict__ output,
                                                    float* __restrict__ scale,
                                                    int64_t rows,
                                                    int64_t cols) {
  // input [E, rows, cols], output [E, cols, rows/2].  Padding one shared
  // column avoids bank conflicts while each column computes its own scale.
  __shared__ float tile[kQuantBlock][kQuantBlock + 1];
  __shared__ float tile_scale[kQuantBlock];

  const int64_t expert = blockIdx.z;
  const int64_t row_group = blockIdx.y;
  const int64_t col_group = blockIdx.x;
  const int thread = threadIdx.x;
  const int64_t input_expert_base = expert * rows * cols;

#pragma unroll
  for (int linear = thread; linear < kQuantBlock * kQuantBlock;
       linear += blockDim.x) {
    const int local_row = linear / kQuantBlock;
    const int local_col = linear % kQuantBlock;
    const int64_t global_row = row_group * kQuantBlock + local_row;
    const int64_t global_col = col_group * kQuantBlock + local_col;
    tile[local_row][local_col] =
        LoadAsFloat(input, input_expert_base + global_row * cols + global_col);
  }
  __syncthreads();

  if (thread < kQuantBlock) {
    float amax = 0.0f;
#pragma unroll
    for (int local_row = 0; local_row < kQuantBlock; ++local_row) {
      amax = fmaxf(amax, fabsf(tile[local_row][thread]));
    }
    const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp4Max);
    tile_scale[thread] = group_scale;
    const int64_t global_col = col_group * kQuantBlock + thread;
    scale[(expert * cols + global_col) * (rows / kQuantBlock) + row_group] =
        group_scale;
  }
  __syncthreads();

  // A 32x32 input tile becomes a 32x16 packed output tile.  The flattened
  // mapping gives each warp two adjacent 16-byte row segments.
#pragma unroll
  for (int linear = thread; linear < kQuantBlock * (kQuantBlock / 2);
       linear += blockDim.x) {
    const int local_col = linear / (kQuantBlock / 2);
    const int local_pair = linear % (kQuantBlock / 2);
    const float group_scale = tile_scale[local_col];
    const uint8_t low =
        EncodeE2m1Strict(tile[local_pair * 2][local_col] / group_scale);
    const uint8_t high =
        EncodeE2m1Strict(tile[local_pair * 2 + 1][local_col] / group_scale);
    const int64_t global_col = col_group * kQuantBlock + local_col;
    const int64_t global_pair = row_group * (kQuantBlock / 2) + local_pair;
    output[(expert * cols + global_col) * (rows / 2) + global_pair] =
        static_cast<int8_t>(low | (high << 4));
  }
}

// BF16 transpose fast path. The legacy transpose kernel computes one column
// maximum serially on only 32 threads. This variant assigns four columns to
// each warp and uses eight-lane reductions, while keeping the same tile and
// output layout so the quantization contract remains unchanged.
__global__ void StackTransposeBf16Fp4Quantize1x32Kernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale,
    int64_t rows,
    int64_t cols) {
  __shared__ __nv_bfloat16 tile[kQuantBlock][kQuantBlock + 1];
  __shared__ float tile_scale[kQuantBlock];

  const int64_t expert = blockIdx.z;
  const int64_t row_group = blockIdx.y;
  const int64_t col_group = blockIdx.x;
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int64_t input_expert_base = expert * rows * cols;

#pragma unroll
  for (int linear = thread; linear < kQuantBlock * kQuantBlock;
       linear += blockDim.x) {
    const int local_row = linear / kQuantBlock;
    const int local_col = linear % kQuantBlock;
    const int64_t global_row = row_group * kQuantBlock + local_row;
    const int64_t global_col = col_group * kQuantBlock + local_col;
    tile[local_row][local_col] =
        input[input_expert_base + global_row * cols + global_col];
  }
  __syncthreads();

  // Each warp handles four columns. Each eight-lane subgroup scans four
  // rows, then reduces within that subgroup. Every column therefore uses all
  // rows but no longer relies on a single thread's serial loop.
  const int column = warp * 4 + (lane >> 3);
  const int subgroup_lane = lane & 7;
  if (column < kQuantBlock) {
    float amax = 0.0f;
#pragma unroll
    for (int row = subgroup_lane; row < kQuantBlock; row += 8) {
      amax = fmaxf(amax, fabsf(__bfloat162float(tile[row][column])));
    }
    const unsigned mask = 0xffu << ((lane >> 3) * 8);
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
      amax = fmaxf(__shfl_down_sync(mask, amax, offset, 8), amax);
    }
    if (subgroup_lane == 0) {
      const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp4Max);
      tile_scale[column] = group_scale;
      const int64_t global_col = col_group * kQuantBlock + column;
      scale[(expert * cols + global_col) * (rows / kQuantBlock) + row_group] =
          group_scale;
    }
  }
  __syncthreads();

#pragma unroll
  for (int linear = thread; linear < kQuantBlock * (kQuantBlock / 2);
       linear += blockDim.x) {
    const int local_col = linear / (kQuantBlock / 2);
    const int local_pair = linear % (kQuantBlock / 2);
    const float group_scale = tile_scale[local_col];
    const uint8_t low = EncodeE2m1Strict(
        __bfloat162float(tile[local_pair * 2][local_col]) / group_scale);
    const uint8_t high = EncodeE2m1Strict(
        __bfloat162float(tile[local_pair * 2 + 1][local_col]) / group_scale);
    const int64_t global_col = col_group * kQuantBlock + local_col;
    const int64_t global_pair = row_group * (kQuantBlock / 2) + local_pair;
    output[(expert * cols + global_col) * (rows / 2) + global_pair] =
        static_cast<int8_t>(low | (high << 4));
  }
}

// -----------------------------------------------------------------------------
// Vectorized FP4 stack-quant fast paths (bitwise identical to the kernels
// above).
// -----------------------------------------------------------------------------

// transpose == false.  Contiguous [E, R, C] -> [E, R, C/2] with a scale per
// 32-element run along C.  Four warp lanes own one group; each lane loads 8
// contiguous bf16 with a single 128-bit int4 load, reduces amax across the 4
// lanes (XOR butterfly, which stays inside the aligned 4-lane block), and emits
// its 8 values as 4 packed FP4 bytes via one 32-bit store.  Numerically this is
// the same per-group amax -> CeilUe8m0 -> EncodeE2m1Strict pipeline as
// QuantizeBf16Pair1x32Kernel<kFp4>; only the load/store width and the amax
// reduction order differ (both bitwise-safe).
__global__ void StackContigBf16Fp4Quantize1x32VecKernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale,
    int64_t total_groups) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int group_in_warp = lane >> 2;  // 0..7, shared by a group's 4 lanes
  const int part = lane & 3;            // 0..3, which 8-element quarter
  const int warps_per_block = blockDim.x >> 5;
  const int64_t groups_per_block = static_cast<int64_t>(warps_per_block) * 8;
  for (int64_t base = static_cast<int64_t>(blockIdx.x) * groups_per_block;
       base < total_groups;
       base += static_cast<int64_t>(gridDim.x) * groups_per_block) {
    const int64_t group = base + warp * 8 + group_in_warp;
    const bool valid = group < total_groups;
    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      v[j] = 0.0f;
    }
    if (valid) {
      const int4 packed = *reinterpret_cast<const int4*>(
          input + group * kQuantBlock + part * 8);
      const __nv_bfloat16* elems =
          reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        v[j] = __bfloat162float(elems[j]);
      }
    }
    float amax = 0.0f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      amax = fmaxf(amax, fabsf(v[j]));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp4Max);
    if (valid) {
      if (part == 0) {
        scale[group] = group_scale;
      }
      uint32_t word = 0;
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const uint8_t low = EncodeE2m1Strict(v[b * 2] / group_scale);
        const uint8_t high = EncodeE2m1Strict(v[b * 2 + 1] / group_scale);
        word |= static_cast<uint32_t>(static_cast<uint8_t>(low | (high << 4)))
                << (b * 8);
      }
      reinterpret_cast<uint32_t*>(output)[group * 4 + part] = word;
    }
  }
}

// transpose == true.  Same shared-memory tile + per-column 32-row reduction as
// StackTransposeBf16Fp4Quantize1x32Kernel, but with 128-bit tile loads and the
// scale kept in registers (no tile_scale[] and no second __syncthreads): each
// column's 8 owning lanes XOR-reduce the amax so every lane holds it, then
// write that column's 16 bytes directly.  The grid, tile layout, amax rows,
// scale/output index math and encode are otherwise identical, so the result is
// bitwise identical.
__global__ void StackTransposeBf16Fp4Quantize1x32VecKernel(
    const __nv_bfloat16* __restrict__ input,
    int8_t* __restrict__ output,
    float* __restrict__ scale,
    int64_t rows,
    int64_t cols) {
  __shared__ __nv_bfloat16 tile[kQuantBlock][kQuantBlock + 1];

  const int64_t expert = blockIdx.z;
  const int64_t row_group = blockIdx.y;
  const int64_t col_group = blockIdx.x;
  const int thread = threadIdx.x;
  const int warp = thread >> 5;
  const int lane = thread & 31;
  const int64_t input_expert_base = expert * rows * cols;

  // Load the 32x32 tile with 128-bit (int4 = 8 bf16) global loads.  The tile is
  // 1024 bf16 = 128 int4; the first 128 threads each issue one load.
  constexpr int kInt4PerTile = (kQuantBlock * kQuantBlock) / 8;
  for (int v = thread; v < kInt4PerTile; v += blockDim.x) {
    const int local_row = v >> 2;        // 4 int4 per 32-column row
    const int local_col0 = (v & 3) * 8;  // 8-column block
    const int64_t global_row = row_group * kQuantBlock + local_row;
    const int64_t global_col0 = col_group * kQuantBlock + local_col0;
    const int4 packed = *reinterpret_cast<const int4*>(
        input + input_expert_base + global_row * cols + global_col0);
    const __nv_bfloat16* elems =
        reinterpret_cast<const __nv_bfloat16*>(&packed);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      tile[local_row][local_col0 + j] = elems[j];
    }
  }
  __syncthreads();

  // Each warp owns four columns; each column is reduced by an 8-lane subgroup.
  const int column = warp * 4 + (lane >> 3);
  const int subgroup_lane = lane & 7;
  float amax = 0.0f;
#pragma unroll
  for (int row = subgroup_lane; row < kQuantBlock; row += 8) {
    amax = fmaxf(amax, fabsf(__bfloat162float(tile[row][column])));
  }
  // XOR butterfly over the aligned 8-lane subgroup -> every lane holds the max.
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
  amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
  const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp4Max);
  const int64_t global_col = col_group * kQuantBlock + column;
  if (subgroup_lane == 0) {
    scale[(expert * cols + global_col) * (rows / kQuantBlock) + row_group] =
        group_scale;
  }
  // The 8 owning lanes emit the column's 16 packed bytes, two adjacent bytes
  // each, matching the (expert, col, row-pair) output layout above.
  int8_t* out_col =
      output + (expert * cols + global_col) * (rows / 2) + row_group * 16;
#pragma unroll
  for (int q = 0; q < 2; ++q) {
    const int local_pair = subgroup_lane * 2 + q;
    const uint8_t low = EncodeE2m1Strict(
        __bfloat162float(tile[local_pair * 2][column]) / group_scale);
    const uint8_t high = EncodeE2m1Strict(
        __bfloat162float(tile[local_pair * 2 + 1][column]) / group_scale);
    out_col[local_pair] = static_cast<int8_t>(low | (high << 4));
  }
}

template <bool HasClamp>
__global__ void WeightedSwigluFp8Quantize1x32Kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ probs,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int64_t rows,
    int64_t hidden,
    float clamp_value,
    int64_t total_groups) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  for (int64_t group_index =
           static_cast<int64_t>(blockIdx.x) * kGroupWarpsPerBlock + warp;
       group_index < total_groups;
       group_index += static_cast<int64_t>(gridDim.x) * kGroupWarpsPerBlock) {
    const int64_t groups_per_row = hidden / kQuantBlock;
    const int64_t row = group_index / groups_per_row;
    const int64_t group = group_index - row * groups_per_row;
    const int64_t col = group * kQuantBlock + lane;
    const int64_t input_base = row * hidden * 2;
    float gate = __bfloat162float(input[input_base + col]);
    float up = __bfloat162float(input[input_base + hidden + col]);
    if constexpr (HasClamp) {
      gate = fminf(gate, clamp_value);
      up = fmaxf(fminf(up, clamp_value), -clamp_value);
    }
    const float sigmoid = HasClamp ? 1.0f / (1.0f + __expf(-gate))
                                   : __frcp_rn(1.0f + __expf(-gate));
    const float value = gate * sigmoid * up * probs[row];
    const float amax = WarpMax(fabsf(value));
    const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp8Max);
    if (lane == 0) {
      scale[group_index] = group_scale;
    }
    output[row * hidden + col] =
        static_cast<__nv_fp8_e4m3>(value / group_scale);
  }
}

// The weighted SwiGLU quantizer has the same 32-element group structure as the
// ordinary BF16 quantizer. Two 16-lane subwarps process two groups per warp,
// reducing the amount of idle work in the activation and scale calculation.
template <bool HasClamp>
__global__ void WeightedSwigluBf16PairQuantize1x32Kernel(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ probs,
    __nv_fp8_e4m3* __restrict__ output,
    float* __restrict__ scale,
    int64_t rows,
    int64_t hidden,
    float clamp_value,
    int64_t total_groups) {
  const int lane = threadIdx.x & 31;
  const int subwarp = lane >> 4;
  const int sublane = lane & 15;
  const int warp = threadIdx.x >> 5;
  constexpr int kGroupsPerBlock = kGroupWarpsPerBlock * 2;
  const unsigned subwarp_mask = subwarp == 0 ? 0x0000ffffu : 0xffff0000u;

  for (int64_t group_index =
           static_cast<int64_t>(blockIdx.x) * kGroupsPerBlock + warp * 2 +
           subwarp;
       group_index < total_groups;
       group_index += static_cast<int64_t>(gridDim.x) * kGroupsPerBlock) {
    const int64_t groups_per_row = hidden / kQuantBlock;
    const int64_t row = group_index / groups_per_row;
    const int64_t group = group_index - row * groups_per_row;
    const int64_t col = group * kQuantBlock + sublane * 2;

    // The input is [row, gate( hidden ), up( hidden )]. In BF16-pair units,
    // the two halves start at row*hidden and row*hidden + hidden/2.
    const int64_t pair_base =
        row * hidden + group * (kQuantBlock / 2) + sublane;
    const auto gate_pair =
        reinterpret_cast<const __nv_bfloat162*>(input)[pair_base];
    const auto up_pair = reinterpret_cast<const __nv_bfloat162*>(
        input)[row * hidden + hidden / 2 + group * (kQuantBlock / 2) + sublane];
    float gate0 = __bfloat162float(gate_pair.x);
    float gate1 = __bfloat162float(gate_pair.y);
    float up0 = __bfloat162float(up_pair.x);
    float up1 = __bfloat162float(up_pair.y);
    if constexpr (HasClamp) {
      gate0 = fminf(gate0, clamp_value);
      gate1 = fminf(gate1, clamp_value);
      up0 = fmaxf(fminf(up0, clamp_value), -clamp_value);
      up1 = fmaxf(fminf(up1, clamp_value), -clamp_value);
    }
    const float prob = probs[row];
    const float sigmoid0 = HasClamp ? 1.0f / (1.0f + __expf(-gate0))
                                    : __frcp_rn(1.0f + __expf(-gate0));
    const float sigmoid1 = HasClamp ? 1.0f / (1.0f + __expf(-gate1))
                                    : __frcp_rn(1.0f + __expf(-gate1));
    const float value0 = gate0 * sigmoid0 * up0 * prob;
    const float value1 = gate1 * sigmoid1 * up1 * prob;
    const float amax =
        HalfWarpMax(fmaxf(fabsf(value0), fabsf(value1)), subwarp_mask);
    const float group_scale = CeilUe8m0(fmaxf(amax, kAmaxFloor) / kFp8Max);
    if (sublane == 0) {
      scale[group_index] = group_scale;
    }
    output[row * hidden + col] =
        static_cast<__nv_fp8_e4m3>(value0 / group_scale);
    output[row * hidden + col + 1] =
        static_cast<__nv_fp8_e4m3>(value1 / group_scale);
  }
}

__global__ void DequantizeFp8OneBy32Kernel(
    const __nv_fp8_e4m3* __restrict__ input,
    const float* __restrict__ scale,
    __nv_bfloat16* __restrict__ output,
    int64_t numel) {
  for (int64_t index =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < numel;
       index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float value =
        static_cast<float>(input[index]) * scale[index / kQuantBlock];
    output[index] = __float2bfloat16_rn(value);
  }
}

// -----------------------------------------------------------------------------
// DQ(Q1(w)) for the W4A8 expert dgrad, bf16 [E, R, C].
// -----------------------------------------------------------------------------

__device__ __forceinline__ float Bf16LowToFloat(uint32_t pair) {
  return __uint_as_float(pair << 16);
}

__device__ __forceinline__ float Bf16HighToFloat(uint32_t pair) {
  return __uint_as_float(pair & 0xffff0000u);
}

__device__ __forceinline__ uint32_t PackBf16x2(float low, float high) {
  const __nv_bfloat162 pair = __floats2bfloat162_rn(low, high);
  return *reinterpret_cast<const uint32_t*>(&pair);
}

// A fake-quant block owns one 32-row group x 256 columns. Each thread keeps 4
// rows x 8 columns in registers (one 16-byte load per row), so w is read once
// and written once, both coalesced: a warp covers one 512-byte row segment.
constexpr int kFakeQuantVec = 8;  // bf16 per 16-byte access
constexpr int kFakeQuantColThreads = 32;
constexpr int kFakeQuantThreads = 256;
constexpr int kFakeQuantRowThreads = kFakeQuantThreads / kFakeQuantColThreads;
constexpr int kFakeQuantRowsPerThread = kQuantBlock / kFakeQuantRowThreads;

// bf16 w [E, R, C] -> DQ(Q1(w)) in one pass; grid.x = E * R / 32 row groups,
// grid.y = column tiles. Same amax -> CeilUe8m0 -> EncodeE2m1Strict pipeline
// as StackTransposeBf16Fp4Quantize1x32VecKernel. x * ReciprocalUe8m0(scale)
// equals the quant kernel's x / scale: both are the correctly rounded x * 2^-k.
// Keeping more rows per thread in registers spills and runs 2-4x slower.
__global__ void __launch_bounds__(kFakeQuantThreads)
    StackBf16Fp4FakeQuantize1x32Kernel(const uint4* __restrict__ input,
                                       uint4* __restrict__ output,
                                       int64_t col_vecs) {
  // partial[row thread][column in the 8-wide vector][column vector]: lanes
  // touch consecutive words in both phases, so neither has bank conflicts.
  __shared__ float partial[kFakeQuantRowThreads][kFakeQuantVec]
                          [kFakeQuantColThreads];
  __shared__ float tile_scale[kFakeQuantVec][kFakeQuantColThreads];

  const int lane_c = threadIdx.x % kFakeQuantColThreads;
  const int lane_r = threadIdx.x / kFakeQuantColThreads;
  const int64_t vec_col =
      static_cast<int64_t>(blockIdx.y) * kFakeQuantColThreads + lane_c;
  const bool active = vec_col < col_vecs;
  const int64_t group_base =
      static_cast<int64_t>(blockIdx.x) * kQuantBlock * col_vecs + vec_col;

  uint4 data[kFakeQuantRowsPerThread];
  float amax[kFakeQuantVec];
#pragma unroll
  for (int k = 0; k < kFakeQuantVec; ++k) {
    amax[k] = 0.0f;
  }
#pragma unroll
  for (int i = 0; i < kFakeQuantRowsPerThread; ++i) {
    const int row = lane_r + i * kFakeQuantRowThreads;
    data[i] = active ? __ldg(input + group_base + row * col_vecs)
                     : make_uint4(0u, 0u, 0u, 0u);
    const uint32_t* words = reinterpret_cast<const uint32_t*>(&data[i]);
#pragma unroll
    for (int w = 0; w < kFakeQuantVec / 2; ++w) {
      amax[2 * w] = fmaxf(amax[2 * w], fabsf(Bf16LowToFloat(words[w])));
      amax[2 * w + 1] =
          fmaxf(amax[2 * w + 1], fabsf(Bf16HighToFloat(words[w])));
    }
  }
#pragma unroll
  for (int k = 0; k < kFakeQuantVec; ++k) {
    partial[lane_r][k][lane_c] = amax[k];
  }
  __syncthreads();

  // One thread per column finishes the 32-row amax.
  for (int index = threadIdx.x; index < kFakeQuantVec * kFakeQuantColThreads;
       index += kFakeQuantThreads) {
    const int k = index / kFakeQuantColThreads;
    const int c = index % kFakeQuantColThreads;
    float column_amax = 0.0f;
#pragma unroll
    for (int r = 0; r < kFakeQuantRowThreads; ++r) {
      column_amax = fmaxf(column_amax, partial[r][k][c]);
    }
    tile_scale[k][c] = CeilUe8m0(fmaxf(column_amax, kAmaxFloor) / kFp4Max);
  }
  __syncthreads();
  if (!active) {
    return;
  }

  float scale[kFakeQuantVec];
  float inv_scale[kFakeQuantVec];
#pragma unroll
  for (int k = 0; k < kFakeQuantVec; ++k) {
    scale[k] = tile_scale[k][lane_c];
    inv_scale[k] = ReciprocalUe8m0(scale[k]);
  }
#pragma unroll
  for (int i = 0; i < kFakeQuantRowsPerThread; ++i) {
    const int row = lane_r + i * kFakeQuantRowThreads;
    uint4 value = data[i];
    uint32_t* words = reinterpret_cast<uint32_t*>(&value);
#pragma unroll
    for (int w = 0; w < kFakeQuantVec / 2; ++w) {
      const float low = Bf16LowToFloat(words[w]) * inv_scale[2 * w];
      const float high = Bf16HighToFloat(words[w]) * inv_scale[2 * w + 1];
      words[w] =
          PackBf16x2(DecodeE2m1(EncodeE2m1Strict(low)) * scale[2 * w],
                     DecodeE2m1(EncodeE2m1Strict(high)) * scale[2 * w + 1]);
    }
    output[group_base + row * col_vecs] = value;
  }
}

// packed [E, C, R/2] + scale [E, C, R/32] (the transpose=True quant layout) ->
// bf16 [E, R, C]. A 64(c) x 64(r) tile is decoded into shared memory, then
// written back along C, so both the packed loads and the stores coalesce.
constexpr int kDequantTile = 64;
constexpr int kDequantThreads = 256;

__global__ void __launch_bounds__(kDequantThreads)
    StackTransposeFp4Dequantize1x32Kernel(const uint32_t* __restrict__ packed,
                                          const float* __restrict__ scale,
                                          __nv_bfloat16* __restrict__ output,
                                          int64_t rows,
                                          int64_t cols) {
  __shared__ __nv_bfloat16 tile[kDequantTile][kDequantTile + 2];  // [r][c]
  const int64_t expert = blockIdx.z;
  const int64_t row_base = static_cast<int64_t>(blockIdx.y) * kDequantTile;
  const int64_t col_base = static_cast<int64_t>(blockIdx.x) * kDequantTile;
  const int64_t words_per_col = rows / 8;  // one 4-byte word = 8 rows
  const int64_t groups_per_col = rows / kQuantBlock;

  // Load: 64 columns x 8 words (64 rows). Nibble j of a word is row j.
  for (int linear = threadIdx.x; linear < kDequantTile * (kDequantTile / 8);
       linear += kDequantThreads) {
    const int local_col = linear / (kDequantTile / 8);
    const int word = linear % (kDequantTile / 8);
    const int64_t col = col_base + local_col;
    const int64_t row = row_base + word * 8;
    if (col < cols && row < rows) {
      const int64_t expert_col = expert * cols + col;
      const uint32_t bits =
          __ldg(packed + expert_col * words_per_col + row / 8);
      const float group_scale =
          __ldg(scale + expert_col * groups_per_col + row / kQuantBlock);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        tile[word * 8 + j][local_col] =
            __float2bfloat16_rn(DecodeE2m1(bits >> (4 * j)) * group_scale);
      }
    }
  }
  __syncthreads();

  // Store: 64 rows x 8 vectors of 8 bf16 along C.
  for (int linear = threadIdx.x; linear < kDequantTile * (kDequantTile / 8);
       linear += kDequantThreads) {
    const int local_row = linear / (kDequantTile / 8);
    const int vec = linear % (kDequantTile / 8);
    const int64_t row = row_base + local_row;
    const int64_t col = col_base + vec * 8;
    if (row < rows && col < cols) {
      const uint32_t* words =
          reinterpret_cast<const uint32_t*>(&tile[local_row][vec * 8]);
      *reinterpret_cast<uint4*>(output + (expert * rows + row) * cols + col) =
          make_uint4(words[0], words[1], words[2], words[3]);
    }
  }
}

void CheckGpuContiguous(const paddle::Tensor& input, const char* op_name) {
  PD_CHECK(input.is_gpu(), op_name, " requires a GPU tensor");
  PD_CHECK(input.is_contiguous(), op_name, " requires a contiguous tensor");
}

inline bool IsBfloat16PairAligned(const paddle::Tensor& input) {
  auto& mutable_input = const_cast<paddle::Tensor&>(input);
  const auto address =
      reinterpret_cast<uintptr_t>(mutable_input.data<phi::bfloat16>());
  return address % alignof(__nv_bfloat162) == 0;
}

// 16-byte (int4 / 128-bit load) alignment check for the vectorized fast paths.
inline bool IsBfloat16Aligned16(const paddle::Tensor& input) {
  auto& mutable_input = const_cast<paddle::Tensor&>(input);
  const auto address =
      reinterpret_cast<uintptr_t>(mutable_input.data<phi::bfloat16>());
  return address % 16 == 0;
}

template <typename Fn>
void DispatchFloatInput(const paddle::Tensor& input, Fn&& fn) {
  auto& mutable_input = const_cast<paddle::Tensor&>(input);
  switch (input.dtype()) {
    case paddle::DataType::BFLOAT16:
      fn(reinterpret_cast<const __nv_bfloat16*>(
          mutable_input.data<phi::bfloat16>()));
      break;
    case paddle::DataType::FLOAT32:
      fn(mutable_input.data<float>());
      break;
    default:
      PD_THROW("MX 1x32 quantization supports only bfloat16 and float32 input");
  }
}

// packed int8 [E, C, R/2] + scale [E, C, R/32] -> output bf16 [E, R, C].
void LaunchStackTransposeFp4Dequantize(const paddle::Tensor& packed,
                                       const paddle::Tensor& scale,
                                       paddle::Tensor* output,
                                       const char* op_name) {
  const int64_t experts = packed.shape()[0];
  const int64_t cols = packed.shape()[1];
  const int64_t rows = packed.shape()[2] * 2;
  const int64_t row_tiles = (rows + kDequantTile - 1) / kDequantTile;
  PD_CHECK(row_tiles <= 65535 && experts <= 65535,
           op_name,
           " tensor too large for the dequantize launch grid");
  auto& mutable_packed = const_cast<paddle::Tensor&>(packed);
  auto& mutable_scale = const_cast<paddle::Tensor&>(scale);
  const dim3 grid((cols + kDequantTile - 1) / kDequantTile,
                  static_cast<unsigned int>(row_tiles),
                  static_cast<unsigned int>(experts));
  StackTransposeFp4Dequantize1x32Kernel<<<grid,
                                          kDequantThreads,
                                          0,
                                          packed.stream()>>>(
      reinterpret_cast<const uint32_t*>(mutable_packed.data<int8_t>()),
      mutable_scale.data<float>(),
      reinterpret_cast<__nv_bfloat16*>(output->data<phi::bfloat16>()),
      rows,
      cols);
}

}  // namespace

std::vector<paddle::Tensor> MxQuantize1x32Impl(const paddle::Tensor& input,
                                               int64_t quant_kind) {
  // Shared implementation behind the two thin PD ops mxfp8_quantize_1x32
  // (kind=0) and mxfp4_quantize_1x32 (kind=1); the op name is picked per kind
  // so PD_CHECK diagnostics name the op the caller actually invoked.
  const char* kOpName = quant_kind == static_cast<int64_t>(QuantKind::kFp8)
                            ? "mxfp8_quantize_1x32"
                            : "mxfp4_quantize_1x32";
  CheckGpuContiguous(input, kOpName);
  PD_CHECK(input.shape().size() == 2, kOpName, " expects rank-2 input");
  PD_CHECK(input.shape()[1] % kQuantBlock == 0,
           kOpName,
           " last dimension must be divisible by 32");
  PD_CHECK(quant_kind == static_cast<int64_t>(QuantKind::kFp8) ||
               quant_kind == static_cast<int64_t>(QuantKind::kFp4),
           kOpName,
           " quant_kind must be 0 (FP8) or 1 (FP4)");

  const int64_t rows = input.shape()[0];
  const int64_t cols = input.shape()[1];
  const bool is_fp8 = quant_kind == static_cast<int64_t>(QuantKind::kFp8);
  auto output = paddle::empty(
      {rows, is_fp8 ? cols : cols / 2},
      is_fp8 ? paddle::DataType::FLOAT8_E4M3FN : paddle::DataType::INT8,
      input.place());
  auto scale = paddle::empty(
      {rows, cols / kQuantBlock}, paddle::DataType::FLOAT32, input.place());
  const int64_t groups = rows * (cols / kQuantBlock);
  if (groups == 0) {
    return {output, scale};
  }

  DispatchFloatInput(input, [&](auto* input_ptr) {
    using InputT = std::remove_cv_t<std::remove_pointer_t<decltype(input_ptr)>>;
    if constexpr (std::is_same_v<InputT, __nv_bfloat16>) {
      constexpr int kGroupsPerBlock = kGroupWarpsPerBlock * 2;
      if (groups < kPairKernelMinGroups || !IsBfloat16PairAligned(input)) {
        if (is_fp8) {
          Quantize1x32Kernel<InputT, QuantKind::kFp8>
              <<<GridFor((groups + kGroupWarpsPerBlock - 1) /
                         kGroupWarpsPerBlock),
                 kGroupThreadsPerBlock,
                 0,
                 input.stream()>>>(
                  input_ptr, output.data(), scale.data<float>(), groups);
        } else {
          Quantize1x32Kernel<InputT, QuantKind::kFp4>
              <<<GridFor((groups + kGroupWarpsPerBlock - 1) /
                         kGroupWarpsPerBlock),
                 kGroupThreadsPerBlock,
                 0,
                 input.stream()>>>(
                  input_ptr, output.data(), scale.data<float>(), groups);
        }
      } else if (is_fp8) {
        QuantizeBf16Pair1x32Kernel<QuantKind::kFp8>
            <<<GridFor((groups + kGroupsPerBlock - 1) / kGroupsPerBlock),
               kGroupThreadsPerBlock,
               0,
               input.stream()>>>(
                input_ptr, output.data(), scale.data<float>(), groups);
      } else {
        QuantizeBf16Pair1x32Kernel<QuantKind::kFp4>
            <<<GridFor((groups + kGroupsPerBlock - 1) / kGroupsPerBlock),
               kGroupThreadsPerBlock,
               0,
               input.stream()>>>(
                input_ptr, output.data(), scale.data<float>(), groups);
      }
    } else if (is_fp8) {
      Quantize1x32Kernel<InputT, QuantKind::kFp8>
          <<<GridFor((groups + kGroupWarpsPerBlock - 1) / kGroupWarpsPerBlock),
             kGroupThreadsPerBlock,
             0,
             input.stream()>>>(
              input_ptr, output.data(), scale.data<float>(), groups);
    } else {
      Quantize1x32Kernel<InputT, QuantKind::kFp4>
          <<<GridFor((groups + kGroupWarpsPerBlock - 1) / kGroupWarpsPerBlock),
             kGroupThreadsPerBlock,
             0,
             input.stream()>>>(
              input_ptr, output.data(), scale.data<float>(), groups);
    }
  });
  CheckCudaLaunch(kOpName);
  return {output, scale};
}

// Thin PD wrappers with the quant kind hard-wired; each gets its own
// PD_BUILD_OP / InferShape / InferDtype below.
std::vector<paddle::Tensor> MxFp8Quantize1x32(const paddle::Tensor& input) {
  return MxQuantize1x32Impl(input, static_cast<int64_t>(QuantKind::kFp8));
}

std::vector<paddle::Tensor> MxFp4Quantize1x32(const paddle::Tensor& input) {
  return MxQuantize1x32Impl(input, static_cast<int64_t>(QuantKind::kFp4));
}

std::vector<paddle::Tensor> MxFp4StackQuantize1x32(const paddle::Tensor& input,
                                                   bool transpose) {
  constexpr const char* kOpName = "mxfp4_stack_quantize_1x32";
  CheckGpuContiguous(input, kOpName);
  PD_CHECK(input.shape().size() == 3, kOpName, " expects [E, R, C]");
  const int64_t experts = input.shape()[0];
  const int64_t rows = input.shape()[1];
  const int64_t cols = input.shape()[2];
  const int64_t quant_dim = transpose ? rows : cols;
  PD_CHECK(quant_dim % kQuantBlock == 0,
           kOpName,
           " quantized dimension must be divisible by 32");
  if (transpose) {
    PD_CHECK(cols % kQuantBlock == 0,
             kOpName,
             " transpose output rows must be divisible by 32");
  }

  const std::vector<int64_t> output_shape =
      transpose ? std::vector<int64_t>{experts, cols, rows / 2}
                : std::vector<int64_t>{experts, rows, cols / 2};
  const std::vector<int64_t> scale_shape =
      transpose ? std::vector<int64_t>{experts, cols, rows / kQuantBlock}
                : std::vector<int64_t>{experts, rows, cols / kQuantBlock};
  auto output =
      paddle::empty(output_shape, paddle::DataType::INT8, input.place());
  auto scale =
      paddle::empty(scale_shape, paddle::DataType::FLOAT32, input.place());
  if (input.numel() == 0) {
    return {output, scale};
  }

  DispatchFloatInput(input, [&](auto* input_ptr) {
    using InputT = std::remove_cv_t<std::remove_pointer_t<decltype(input_ptr)>>;
    if (transpose) {
      const dim3 grid(cols / kQuantBlock,
                      rows / kQuantBlock,
                      static_cast<unsigned int>(experts));
      const int64_t tiles =
          experts * (rows / kQuantBlock) * (cols / kQuantBlock);
      if constexpr (std::is_same_v<InputT, __nv_bfloat16>) {
        if (tiles >= kPairKernelMinGroups && IsBfloat16Aligned16(input)) {
          // Vectorized fast path (128-bit tile loads, single barrier).
          StackTransposeBf16Fp4Quantize1x32VecKernel<<<grid,
                                                       256,
                                                       0,
                                                       input.stream()>>>(
              input_ptr,
              output.data<int8_t>(),
              scale.data<float>(),
              rows,
              cols);
        } else if (tiles >= kPairKernelMinGroups &&
                   IsBfloat16PairAligned(input)) {
          StackTransposeBf16Fp4Quantize1x32Kernel<<<grid,
                                                    256,
                                                    0,
                                                    input.stream()>>>(
              input_ptr,
              output.data<int8_t>(),
              scale.data<float>(),
              rows,
              cols);
        } else {
          StackTransposeFp4Quantize1x32Kernel<InputT>
              <<<grid, 256, 0, input.stream()>>>(input_ptr,
                                                 output.data<int8_t>(),
                                                 scale.data<float>(),
                                                 rows,
                                                 cols);
        }
      } else {
        StackTransposeFp4Quantize1x32Kernel<InputT>
            <<<grid, 256, 0, input.stream()>>>(input_ptr,
                                               output.data<int8_t>(),
                                               scale.data<float>(),
                                               rows,
                                               cols);
      }
    } else {
      const int64_t groups = experts * rows * (cols / kQuantBlock);
      if constexpr (std::is_same_v<InputT, __nv_bfloat16>) {
        constexpr int kGroupsPerBlock = kGroupWarpsPerBlock * 2;
        constexpr int64_t kVecGroupsPerBlock =
            static_cast<int64_t>(kGroupThreadsPerBlock / 32) * 8;
        if (groups >= kPairKernelMinGroups && IsBfloat16Aligned16(input)) {
          // Vectorized fast path: 4 lanes own a group, one 128-bit load each.
          StackContigBf16Fp4Quantize1x32VecKernel<<<
              GridFor((groups + kVecGroupsPerBlock - 1) / kVecGroupsPerBlock),
              kGroupThreadsPerBlock,
              0,
              input.stream()>>>(
              input_ptr, output.data<int8_t>(), scale.data<float>(), groups);
        } else if (groups < kPairKernelMinGroups ||
                   !IsBfloat16PairAligned(input)) {
          Quantize1x32Kernel<InputT, QuantKind::kFp4>
              <<<GridFor((groups + kGroupWarpsPerBlock - 1) /
                         kGroupWarpsPerBlock),
                 kGroupThreadsPerBlock,
                 0,
                 input.stream()>>>(
                  input_ptr, output.data(), scale.data<float>(), groups);
        } else {
          QuantizeBf16Pair1x32Kernel<QuantKind::kFp4>
              <<<GridFor((groups + kGroupsPerBlock - 1) / kGroupsPerBlock),
                 kGroupThreadsPerBlock,
                 0,
                 input.stream()>>>(
                  input_ptr, output.data(), scale.data<float>(), groups);
        }
      } else {
        Quantize1x32Kernel<InputT, QuantKind::kFp4>
            <<<GridFor((groups + kGroupWarpsPerBlock - 1) /
                       kGroupWarpsPerBlock),
               kGroupThreadsPerBlock,
               0,
               input.stream()>>>(
                input_ptr, output.data(), scale.data<float>(), groups);
      }
    }
  });
  CheckCudaLaunch(kOpName);
  return {output, scale};
}

std::vector<paddle::Tensor> MxFp8WeightedSwigluQuantize1x32(
    const paddle::Tensor& input,
    const paddle::Tensor& probs,
    double clamp_value) {
  constexpr const char* kOpName = "mxfp8_weighted_swiglu_quantize_1x32";
  CheckGpuContiguous(input, kOpName);
  CheckGpuContiguous(probs, kOpName);
  PD_CHECK(input.dtype() == paddle::DataType::BFLOAT16,
           kOpName,
           " input must be bfloat16");
  PD_CHECK(probs.dtype() == paddle::DataType::FLOAT32,
           kOpName,
           " probs must be float32");
  PD_CHECK(input.shape().size() == 2 && input.shape()[1] % 2 == 0,
           kOpName,
           " input must have shape [M, 2H]");
  const int64_t rows = input.shape()[0];
  const int64_t hidden = input.shape()[1] / 2;
  PD_CHECK(hidden % kQuantBlock == 0, kOpName, " H must be divisible by 32");
  PD_CHECK(
      probs.numel() == rows, kOpName, " probs must contain one float per row");

  auto output = paddle::empty(
      {rows, hidden}, paddle::DataType::FLOAT8_E4M3FN, input.place());
  auto scale = paddle::empty(
      {rows, hidden / kQuantBlock}, paddle::DataType::FLOAT32, input.place());
  const int64_t groups = rows * (hidden / kQuantBlock);
  if (groups == 0) {
    return {output, scale};
  }

  auto& mutable_input = const_cast<paddle::Tensor&>(input);
  auto& mutable_probs = const_cast<paddle::Tensor&>(probs);
  const auto* input_ptr = reinterpret_cast<const __nv_bfloat16*>(
      mutable_input.data<phi::bfloat16>());
  auto* output_ptr =
      reinterpret_cast<__nv_fp8_e4m3*>(output.data<phi::float8_e4m3fn>());
  const bool use_pair_kernel =
      groups >= kPairKernelMinGroups && IsBfloat16PairAligned(input);
  constexpr int kGroupsPerBlock = kGroupWarpsPerBlock * 2;
  if (use_pair_kernel) {
    if (clamp_value > 0.0) {
      WeightedSwigluBf16PairQuantize1x32Kernel<true>
          <<<GridFor((groups + kGroupsPerBlock - 1) / kGroupsPerBlock),
             kGroupThreadsPerBlock,
             0,
             input.stream()>>>(input_ptr,
                               mutable_probs.data<float>(),
                               output_ptr,
                               scale.data<float>(),
                               rows,
                               hidden,
                               static_cast<float>(clamp_value),
                               groups);
    } else {
      WeightedSwigluBf16PairQuantize1x32Kernel<false>
          <<<GridFor((groups + kGroupsPerBlock - 1) / kGroupsPerBlock),
             kGroupThreadsPerBlock,
             0,
             input.stream()>>>(input_ptr,
                               mutable_probs.data<float>(),
                               output_ptr,
                               scale.data<float>(),
                               rows,
                               hidden,
                               0.0f,
                               groups);
    }
  } else if (clamp_value > 0.0) {
    WeightedSwigluFp8Quantize1x32Kernel<true>
        <<<GridFor((groups + kGroupWarpsPerBlock - 1) / kGroupWarpsPerBlock),
           kGroupThreadsPerBlock,
           0,
           input.stream()>>>(input_ptr,
                             mutable_probs.data<float>(),
                             output_ptr,
                             scale.data<float>(),
                             rows,
                             hidden,
                             static_cast<float>(clamp_value),
                             groups);
  } else {
    WeightedSwigluFp8Quantize1x32Kernel<false>
        <<<GridFor((groups + kGroupWarpsPerBlock - 1) / kGroupWarpsPerBlock),
           kGroupThreadsPerBlock,
           0,
           input.stream()>>>(input_ptr,
                             mutable_probs.data<float>(),
                             output_ptr,
                             scale.data<float>(),
                             rows,
                             hidden,
                             0.0f,
                             groups);
  }
  CheckCudaLaunch(kOpName);
  return {output, scale};
}

std::vector<paddle::Tensor> MxFp8Dequantize1x32(const paddle::Tensor& input,
                                                const paddle::Tensor& scale) {
  constexpr const char* kOpName = "mxfp8_dequantize_1x32";
  CheckGpuContiguous(input, kOpName);
  CheckGpuContiguous(scale, kOpName);
  PD_CHECK(input.dtype() == paddle::DataType::FLOAT8_E4M3FN,
           kOpName,
           " input must be float8_e4m3fn");
  PD_CHECK(scale.dtype() == paddle::DataType::FLOAT32,
           kOpName,
           " scale must be float32");
  PD_CHECK(input.shape().size() == 2 && input.shape()[1] % kQuantBlock == 0,
           kOpName,
           " input must be [M, K] with K divisible by 32");
  const std::vector<int64_t> expected_scale_shape = {
      input.shape()[0], input.shape()[1] / kQuantBlock};
  PD_CHECK(scale.shape() == expected_scale_shape,
           kOpName,
           " scale shape must be [M, K/32]");

  auto output =
      paddle::empty(input.shape(), paddle::DataType::BFLOAT16, input.place());
  if (input.numel() == 0) {
    return {output};
  }
  auto& mutable_input = const_cast<paddle::Tensor&>(input);
  auto& mutable_scale = const_cast<paddle::Tensor&>(scale);
  const int threads = 256;
  const int blocks = GridFor((input.numel() + threads - 1) / threads);
  DequantizeFp8OneBy32Kernel<<<blocks, threads, 0, input.stream()>>>(
      reinterpret_cast<const __nv_fp8_e4m3*>(
          mutable_input.data<phi::float8_e4m3fn>()),
      mutable_scale.data<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data<phi::bfloat16>()),
      input.numel());
  CheckCudaLaunch(kOpName);
  return {output};
}

// bf16 [E, R, C] = DQ(mxfp4_stack_quantize_1x32(input, transpose=True)).
std::vector<paddle::Tensor> MxFp4StackFakeQuantize1x32(
    const paddle::Tensor& input) {
  constexpr const char* kOpName = "mxfp4_stack_fake_quantize_1x32";
  CheckGpuContiguous(input, kOpName);
  PD_CHECK(input.shape().size() == 3, kOpName, " expects [E, R, C]");
  const int64_t experts = input.shape()[0];
  const int64_t rows = input.shape()[1];
  const int64_t cols = input.shape()[2];
  PD_CHECK(rows % kQuantBlock == 0 && cols % kQuantBlock == 0,
           kOpName,
           " R and C must be divisible by 32");

  auto output =
      paddle::empty(input.shape(), paddle::DataType::BFLOAT16, input.place());
  if (input.numel() == 0) {
    return {output};
  }
  if (input.dtype() == paddle::DataType::BFLOAT16 &&
      IsBfloat16Aligned16(input)) {
    const int64_t row_groups = experts * rows / kQuantBlock;
    const int64_t col_vecs = cols / kFakeQuantVec;
    const int64_t col_tiles =
        (col_vecs + kFakeQuantColThreads - 1) / kFakeQuantColThreads;
    PD_CHECK(row_groups <= INT32_MAX && col_tiles <= 65535,
             kOpName,
             " tensor too large for the launch grid");
    auto& mutable_input = const_cast<paddle::Tensor&>(input);
    StackBf16Fp4FakeQuantize1x32Kernel<<<
        dim3(static_cast<unsigned int>(row_groups),
             static_cast<unsigned int>(col_tiles)),
        kFakeQuantThreads,
        0,
        input.stream()>>>(
        reinterpret_cast<const uint4*>(mutable_input.data<phi::bfloat16>()),
        reinterpret_cast<uint4*>(output.data<phi::bfloat16>()),
        col_vecs);
  } else {
    // float32, or a bf16 view that is not 16-byte aligned: quantize with the
    // forward op, then unpack.
    const auto quantized = MxFp4StackQuantize1x32(input, /*transpose=*/true);
    LaunchStackTransposeFp4Dequantize(
        quantized[0], quantized[1], &output, kOpName);
  }
  CheckCudaLaunch(kOpName);
  return {output};
}

// Inverse of mxfp4_stack_quantize_1x32(w, transpose=True): input int8
// [E, C, R/2] (row 2i in the low nibble), scale [E, C, R/32] -> bf16 [E, R, C].
std::vector<paddle::Tensor> MxFp4StackTransposeDequantize1x32(
    const paddle::Tensor& input, const paddle::Tensor& scale) {
  constexpr const char* kOpName = "mxfp4_stack_transpose_dequantize_1x32";
  CheckGpuContiguous(input, kOpName);
  CheckGpuContiguous(scale, kOpName);
  PD_CHECK(input.dtype() == paddle::DataType::INT8,
           kOpName,
           " input must be int8 packed FP4");
  PD_CHECK(scale.dtype() == paddle::DataType::FLOAT32,
           kOpName,
           " scale must be float32");
  PD_CHECK(input.shape().size() == 3, kOpName, " input must be [E, C, R/2]");
  const int64_t experts = input.shape()[0];
  const int64_t cols = input.shape()[1];
  const int64_t rows = input.shape()[2] * 2;
  PD_CHECK(rows % kQuantBlock == 0 && cols % 8 == 0,
           kOpName,
           " R must be divisible by 32 and C by 8");
  const std::vector<int64_t> expected_scale_shape = {
      experts, cols, rows / kQuantBlock};
  PD_CHECK(scale.shape() == expected_scale_shape,
           kOpName,
           " scale shape must be [E, C, R/32]");

  auto output = paddle::empty(
      {experts, rows, cols}, paddle::DataType::BFLOAT16, input.place());
  if (input.numel() == 0) {
    return {output};
  }
  auto& mutable_input = const_cast<paddle::Tensor&>(input);
  PD_CHECK(reinterpret_cast<uintptr_t>(mutable_input.data<int8_t>()) % 4 == 0,
           kOpName,
           " input must be 4-byte aligned");
  LaunchStackTransposeFp4Dequantize(input, scale, &output, kOpName);
  CheckCudaLaunch(kOpName);
  return {output};
}

std::vector<std::vector<int64_t>> MxFp8QuantInferShape(
    std::vector<int64_t> input_shape) {
  return {{input_shape[0], input_shape[1]},
          {input_shape[0], input_shape[1] / kQuantBlock}};
}

std::vector<paddle::DataType> MxFp8QuantInferDtype(
    paddle::DataType input_dtype) {
  return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> MxFp4QuantInferShape(
    std::vector<int64_t> input_shape) {
  return {{input_shape[0], input_shape[1] / 2},
          {input_shape[0], input_shape[1] / kQuantBlock}};
}

std::vector<paddle::DataType> MxFp4QuantInferDtype(
    paddle::DataType input_dtype) {
  return {paddle::DataType::INT8, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> StackInferShape(
    std::vector<int64_t> input_shape, bool transpose) {
  const int64_t e = input_shape[0];
  const int64_t r = input_shape[1];
  const int64_t c = input_shape[2];
  return transpose ? std::vector<std::vector<int64_t>>{{e, c, r / 2},
                                                       {e, c, r / kQuantBlock}}
                   : std::vector<std::vector<int64_t>>{{e, r, c / 2},
                                                       {e, r, c / kQuantBlock}};
}

std::vector<paddle::DataType> StackInferDtype(paddle::DataType input_dtype,
                                              bool transpose) {
  return {paddle::DataType::INT8, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> WeightedSwigluInferShape(
    std::vector<int64_t> input_shape,
    std::vector<int64_t> probs_shape,
    double clamp_value) {
  const int64_t hidden = input_shape[1] / 2;
  return {{input_shape[0], hidden}, {input_shape[0], hidden / kQuantBlock}};
}

std::vector<paddle::DataType> WeightedSwigluInferDtype(
    paddle::DataType input_dtype, paddle::DataType probs_dtype) {
  return {paddle::DataType::FLOAT8_E4M3FN, paddle::DataType::FLOAT32};
}

std::vector<std::vector<int64_t>> DequantInferShape(
    std::vector<int64_t> input_shape, std::vector<int64_t> scale_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> DequantInferDtype(paddle::DataType input_dtype,
                                                paddle::DataType scale_dtype) {
  return {paddle::DataType::BFLOAT16};
}

std::vector<std::vector<int64_t>> StackFakeQuantInferShape(
    std::vector<int64_t> input_shape) {
  return {input_shape};
}

std::vector<paddle::DataType> StackFakeQuantInferDtype(
    paddle::DataType input_dtype) {
  return {paddle::DataType::BFLOAT16};
}

std::vector<std::vector<int64_t>> StackTransposeDequantInferShape(
    std::vector<int64_t> input_shape, std::vector<int64_t> scale_shape) {
  return {{input_shape[0], input_shape[2] * 2, input_shape[1]}};
}

PD_BUILD_OP(mxfp8_quantize_1x32)
    .Inputs({"input"})
    .Outputs({"output", "scale"})
    .SetKernelFn(PD_KERNEL(MxFp8Quantize1x32))
    .SetInferShapeFn(PD_INFER_SHAPE(MxFp8QuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MxFp8QuantInferDtype));

PD_BUILD_OP(mxfp4_quantize_1x32)
    .Inputs({"input"})
    .Outputs({"output", "scale"})
    .SetKernelFn(PD_KERNEL(MxFp4Quantize1x32))
    .SetInferShapeFn(PD_INFER_SHAPE(MxFp4QuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(MxFp4QuantInferDtype));

PD_BUILD_OP(mxfp4_stack_quantize_1x32)
    .Inputs({"input"})
    .Outputs({"output", "scale"})
    .Attrs({"transpose: bool"})
    .SetKernelFn(PD_KERNEL(MxFp4StackQuantize1x32))
    .SetInferShapeFn(PD_INFER_SHAPE(StackInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(StackInferDtype));

PD_BUILD_OP(mxfp8_weighted_swiglu_quantize_1x32)
    .Inputs({"input", "probs"})
    .Outputs({"output", "scale"})
    .Attrs({"clamp_value: double"})
    .SetKernelFn(PD_KERNEL(MxFp8WeightedSwigluQuantize1x32))
    .SetInferShapeFn(PD_INFER_SHAPE(WeightedSwigluInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(WeightedSwigluInferDtype));

PD_BUILD_OP(mxfp8_dequantize_1x32)
    .Inputs({"input", "scale"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(MxFp8Dequantize1x32))
    .SetInferShapeFn(PD_INFER_SHAPE(DequantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DequantInferDtype));

PD_BUILD_OP(mxfp4_stack_fake_quantize_1x32)
    .Inputs({"input"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(MxFp4StackFakeQuantize1x32))
    .SetInferShapeFn(PD_INFER_SHAPE(StackFakeQuantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(StackFakeQuantInferDtype));

PD_BUILD_OP(mxfp4_stack_transpose_dequantize_1x32)
    .Inputs({"input", "scale"})
    .Outputs({"output"})
    .SetKernelFn(PD_KERNEL(MxFp4StackTransposeDequantize1x32))
    .SetInferShapeFn(PD_INFER_SHAPE(StackTransposeDequantInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(DequantInferDtype));
