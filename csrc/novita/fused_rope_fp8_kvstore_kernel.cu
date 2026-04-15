// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

/*
 * Fused RoPE + FP8-Cast + KV-Cache-Store kernels (novita)
 *
 * Two entry points sharing the same RoPE + FP8 + KV-store pipeline:
 *
 *   fused_rope_fp8_kvstore          — minimax_m2 path (primary).
 *                                     Takes pre-normed q, k, v as separate
 *                                     BF16 tensors (QK norm is done upstream
 *                                     via TP allreduce). Applies RoPE, casts
 *                                     to FP8 E4M3, scatter-writes K/V to the
 *                                     paged KV cache.
 *
 *   fused_qk_norm_rope_fp8_kvstore  — full pipeline variant (glm4_moe).
 *                                     Accepts packed qkv + norm weights and
 *                                     additionally fuses per-head RMSNorm
 *                                     before RoPE.
 *
 * Adapted from sgl-kernel/csrc/moe/fused_qknorm_rope_kernel.cu
 */

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <cmath>

#define NOVITA_CHECK_TYPE(x, st) \
  TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), \
              ", while ", st, " is expected")
#define NOVITA_CHECK_TH_CUDA(x) \
  TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define NOVITA_CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define NOVITA_CHECK_INPUT(x, st) \
  NOVITA_CHECK_TH_CUDA(x);        \
  NOVITA_CHECK_CONTIGUOUS(x);     \
  NOVITA_CHECK_TYPE(x, st)

#define FINAL_MASK 0xffffffff

// ============================================================================
// Utility helpers
// ============================================================================

namespace novita_helpers {

template <typename T, int num>
struct packed_as;
template <>
struct packed_as<uint, 1> {
  using type = uint;
};
template <>
struct packed_as<uint, 2> {
  using type = uint2;
};
template <>
struct packed_as<uint, 4> {
  using type = uint4;
};

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1)
    val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
  return val;
}

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

}  // namespace novita_helpers

// ============================================================================
// Kernel 1: Full pipeline — QK Norm + RoPE + FP8 + KV Store
// (used by glm4_moe)
// ============================================================================
//
// Each warp processes one (token, head) pair.
//   - Q heads: RMSNorm -> RoPE -> scale -> FP8 cast -> write to q_output
//   - K heads: RMSNorm -> RoPE -> scale -> FP8 cast -> scatter write to k_cache
//   - V heads: (no norm/rope) -> scale -> FP8 cast -> scatter write to v_cache
//
// Template parameters:
//   head_dim   - dimension of each head (must be multiple of 64)
//   interleave - true for interleaved RoPE, false for NeoX style
template <int head_dim, bool interleave>
__global__ void __launch_bounds__(128, 16) fusedQKNormRopeFP8KVStoreKernel(
    __nv_bfloat16 const* __restrict__ qkv, int const num_heads_q,
    int const num_heads_k, int const num_heads_v, float const eps,
    __nv_bfloat16 const* __restrict__ q_weight,
    __nv_bfloat16 const* __restrict__ k_weight,
    int64_t const* __restrict__ position_ids, int const num_tokens,
    int const rotary_dim,
    __nv_bfloat16 const* __restrict__ cos_sin_cache,  // [max_pos, rotary_dim] BF16
    __nv_fp8_e4m3* q_output, float const* __restrict__ q_scale_ptr,
    int64_t const q_output_stride, __nv_fp8_e4m3* k_cache,
    __nv_fp8_e4m3* v_cache,
    int64_t const* __restrict__ slot_mapping,  // INT64 for vLLM compatibility
    float const* __restrict__ k_scale_ptr, float const* __restrict__ v_scale_ptr,
    int64_t const block_size, int64_t const block_stride,
    int64_t const page_stride, int64_t const head_stride) {
  int const warpsPerBlock = blockDim.x / 32;
  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;

  int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;
  int const total_heads = num_heads_q + num_heads_k + num_heads_v;
  int const tokenIdx = globalWarpIdx / total_heads;
  int const localHeadIdx = globalWarpIdx % total_heads;

  if (tokenIdx >= num_tokens) return;

  enum HeadType { Q_HEAD, K_HEAD, V_HEAD };
  HeadType headType;
  int headIdx;
  if (localHeadIdx < num_heads_q) {
    headType = Q_HEAD;
    headIdx = localHeadIdx;
  } else if (localHeadIdx < num_heads_q + num_heads_k) {
    headType = K_HEAD;
    headIdx = localHeadIdx - num_heads_q;
  } else {
    headType = V_HEAD;
    headIdx = localHeadIdx - num_heads_q - num_heads_k;
  }

  static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64");
  constexpr int numElemsPerThread = head_dim / 32;
  float elements[numElemsPerThread];
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0);
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename novita_helpers::packed_as<uint, vecSize>::type;

  int const num_all_heads = num_heads_q + num_heads_k + num_heads_v;
  int64_t const tokenOff =
      static_cast<int64_t>(tokenIdx) * num_all_heads * head_dim;
  int64_t offsetWarp;
  if (headType == Q_HEAD) {
    offsetWarp = tokenOff + static_cast<int64_t>(headIdx) * head_dim;
  } else if (headType == K_HEAD) {
    offsetWarp = tokenOff + static_cast<int64_t>(num_heads_q) * head_dim +
                 static_cast<int64_t>(headIdx) * head_dim;
  } else {
    offsetWarp = tokenOff +
                 static_cast<int64_t>(num_heads_q + num_heads_k) * head_dim +
                 static_cast<int64_t>(headIdx) * head_dim;
  }
  int64_t offsetThread = offsetWarp + laneId * numElemsPerThread;

  // ---- Load from QKV buffer ----
  {
    vec_T vec =
        *reinterpret_cast<vec_T const*>(&qkv[offsetThread]);
#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      float2 vals = __bfloat1622float2(
          *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  // ---- V heads: no norm/rope, just FP8 cast + store ----
  if (headType == V_HEAD) {
    int64_t const cacheSlot = slot_mapping[tokenIdx];
    if (cacheSlot < 0) return;

    int64_t const blk_idx = cacheSlot / block_size;
    int64_t const blk_off = cacheSlot % block_size;
    int64_t const cacheOffset = blk_idx * block_stride +
                                blk_off * page_stride +
                                static_cast<int64_t>(headIdx) * head_stride +
                                laneId * numElemsPerThread;

    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] / *v_scale_ptr);
      packed |= (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8))
                 << (i * 8));
    }
    *reinterpret_cast<uint32_t*>(&v_cache[cacheOffset]) = packed;
    return;
  }

  // ---- Q and K heads: RMSNorm ----
  float sumOfSquares = 0.0f;
#pragma unroll
  for (int i = 0; i < numElemsPerThread; i++) {
    sumOfSquares += elements[i] * elements[i];
  }
  sumOfSquares = novita_helpers::warpReduceSum(sumOfSquares);
  float rms_rcp = rsqrtf(sumOfSquares / static_cast<float>(head_dim) + eps);

  bool const isQ = (headType == Q_HEAD);
  {
    __nv_bfloat16 const* wptr = isQ ? q_weight : k_weight;
    vec_T wvec =
        *reinterpret_cast<vec_T const*>(&wptr[laneId * numElemsPerThread]);
#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      float2 wvals = __bfloat1622float2(
          *reinterpret_cast<__nv_bfloat162 const*>(
              reinterpret_cast<uint const*>(&wvec) + i));
      elements[2 * i] *= rms_rcp * wvals.x;
      elements[2 * i + 1] *= rms_rcp * wvals.y;
    }
  }

  // ---- Q and K heads: RoPE + FP8 store ----
  int const rotary_lanes = rotary_dim / numElemsPerThread;
  bool const applyRotary = (laneId < rotary_lanes);

  float const* scale_ptr;
  __nv_fp8_e4m3* out_ptr;
  int64_t out_offset;
  if (headType == Q_HEAD) {
    scale_ptr = q_scale_ptr;
    out_ptr = q_output;
    out_offset = static_cast<int64_t>(tokenIdx) * q_output_stride +
                 static_cast<int64_t>(headIdx) * head_dim +
                 laneId * numElemsPerThread;
  } else {
    int64_t const cacheSlot = slot_mapping[tokenIdx];
    if (cacheSlot < 0) return;

    int64_t const blk_idx = cacheSlot / block_size;
    int64_t const blk_off = cacheSlot % block_size;
    scale_ptr = k_scale_ptr;
    out_ptr = k_cache;
    out_offset = blk_idx * block_stride +
                 blk_off * page_stride +
                 static_cast<int64_t>(headIdx) * head_stride +
                 laneId * numElemsPerThread;
  }

  if (applyRotary) {
    int64_t const pos = position_ids[tokenIdx];
    int const half_rotary = rotary_dim / 2;
    __nv_bfloat16 const* cache_row =
        cos_sin_cache + static_cast<int64_t>(pos) * rotary_dim;

    if constexpr (interleave) {
#pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        float e2 = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];
        int half_dim = (laneId * numElemsPerThread + i) / 2;
        float cos_val = __bfloat162float(cache_row[half_dim]);
        float sin_val = __bfloat162float(cache_row[half_rotary + half_dim]);
        elements[i] = (elements[i] * cos_val + e2 * sin_val);
      }
    } else {
      __syncwarp();
      int const half_rotary_lanes = rotary_lanes / 2;
      unsigned int active_mask = (1u << rotary_lanes) - 1;
      int base_half = (laneId * numElemsPerThread) % half_rotary;

      __nv_bfloat162 cos_p0 =
          *reinterpret_cast<__nv_bfloat162 const*>(&cache_row[base_half]);
      __nv_bfloat162 cos_p1 =
          *reinterpret_cast<__nv_bfloat162 const*>(&cache_row[base_half + 2]);
      float2 cf0 = __bfloat1622float2(cos_p0);
      float2 cf1 = __bfloat1622float2(cos_p1);
      float cos_arr[4] = {cf0.x, cf0.y, cf1.x, cf1.y};

      __nv_bfloat162 sin_p0 = *reinterpret_cast<__nv_bfloat162 const*>(
          &cache_row[half_rotary + base_half]);
      __nv_bfloat162 sin_p1 = *reinterpret_cast<__nv_bfloat162 const*>(
          &cache_row[half_rotary + base_half + 2]);
      float2 sf0 = __bfloat1622float2(sin_p0);
      float2 sf1 = __bfloat1622float2(sin_p1);
      float sin_arr[4] = {sf0.x, sf0.y, sf1.x, sf1.y};

#pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        float e2 = __shfl_xor_sync(active_mask, elements[i], half_rotary_lanes);
        if (laneId < half_rotary_lanes) {
          e2 = -e2;
        }
        elements[i] = (elements[i] * cos_arr[i] + e2 * sin_arr[i]);
      }
      __syncwarp();
    }
  }

  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < numElemsPerThread; i++) {
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] / *scale_ptr);
    packed |=
        (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8)) << (i * 8));
  }
  *reinterpret_cast<uint32_t*>(&out_ptr[out_offset]) = packed;
}

// ============================================================================
// Kernel 2: Subset pipeline — RoPE + FP8 + KV Store (no QK Norm)
// (used by minimax_m2, which performs TP-aware QK norm before this kernel)
// ============================================================================
//
// Each warp processes one (token, head) pair.
//   - Q heads: RoPE -> scale -> FP8 cast -> write to q_output
//   - K heads: RoPE -> scale -> FP8 cast -> scatter write to k_cache
//   - V heads: scale -> FP8 cast -> scatter write to v_cache
//
// Inputs q, k, v are separate BF16 tensors (already split + normed).
template <int head_dim, bool interleave>
__global__ void __launch_bounds__(128, 16) fusedRopeFP8KVStoreKernel(
    __nv_bfloat16 const* __restrict__ q,  // [num_tokens, num_heads_q, head_dim]
    __nv_bfloat16 const* __restrict__ k,  // [num_tokens, num_heads_k, head_dim]
    __nv_bfloat16 const* __restrict__ v,  // [num_tokens, num_heads_v, head_dim]
    int const num_heads_q, int const num_heads_k, int const num_heads_v,
    int64_t const* __restrict__ position_ids, int const num_tokens,
    int const rotary_dim,
    __nv_bfloat16 const* __restrict__ cos_sin_cache,  // [max_pos, rotary_dim] BF16
    __nv_fp8_e4m3* q_output, float const* __restrict__ q_scale_ptr,
    int64_t const q_output_stride, __nv_fp8_e4m3* k_cache,
    __nv_fp8_e4m3* v_cache,
    int64_t const* __restrict__ slot_mapping,  // INT64 for vLLM compatibility
    float const* __restrict__ k_scale_ptr, float const* __restrict__ v_scale_ptr,
    int64_t const block_size, int64_t const block_stride,
    int64_t const page_stride, int64_t const head_stride) {
  int const warpsPerBlock = blockDim.x / 32;
  int const warpId = threadIdx.x / 32;
  int const laneId = threadIdx.x % 32;

  int const globalWarpIdx = blockIdx.x * warpsPerBlock + warpId;
  int const total_heads = num_heads_q + num_heads_k + num_heads_v;
  int const tokenIdx = globalWarpIdx / total_heads;
  int const localHeadIdx = globalWarpIdx % total_heads;

  if (tokenIdx >= num_tokens) return;

  enum HeadType { Q_HEAD, K_HEAD, V_HEAD };
  HeadType headType;
  int headIdx;
  if (localHeadIdx < num_heads_q) {
    headType = Q_HEAD;
    headIdx = localHeadIdx;
  } else if (localHeadIdx < num_heads_q + num_heads_k) {
    headType = K_HEAD;
    headIdx = localHeadIdx - num_heads_q;
  } else {
    headType = V_HEAD;
    headIdx = localHeadIdx - num_heads_q - num_heads_k;
  }

  static_assert(head_dim % (32 * 2) == 0, "head_dim must be divisible by 64");
  constexpr int numElemsPerThread = head_dim / 32;
  float elements[numElemsPerThread];
  constexpr int elemSizeBytes = numElemsPerThread * sizeof(__nv_bfloat16);
  static_assert(elemSizeBytes % 4 == 0);
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename novita_helpers::packed_as<uint, vecSize>::type;

  // Select the appropriate input pointer and compute per-head offset
  __nv_bfloat16 const* src_ptr;
  int64_t offsetThread;
  if (headType == Q_HEAD) {
    src_ptr = q;
    offsetThread = (static_cast<int64_t>(tokenIdx) * num_heads_q + headIdx) *
                       head_dim +
                   laneId * numElemsPerThread;
  } else if (headType == K_HEAD) {
    src_ptr = k;
    offsetThread = (static_cast<int64_t>(tokenIdx) * num_heads_k + headIdx) *
                       head_dim +
                   laneId * numElemsPerThread;
  } else {
    src_ptr = v;
    offsetThread = (static_cast<int64_t>(tokenIdx) * num_heads_v + headIdx) *
                       head_dim +
                   laneId * numElemsPerThread;
  }

  // ---- Load elements ----
  {
    vec_T vec = *reinterpret_cast<vec_T const*>(&src_ptr[offsetThread]);
#pragma unroll
    for (int i = 0; i < vecSize; i++) {
      float2 vals = __bfloat1622float2(
          *reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<uint*>(&vec) + i));
      elements[2 * i] = vals.x;
      elements[2 * i + 1] = vals.y;
    }
  }

  // ---- V heads: FP8 cast + scatter write (no RoPE) ----
  if (headType == V_HEAD) {
    int64_t const cacheSlot = slot_mapping[tokenIdx];
    if (cacheSlot < 0) return;

    int64_t const blk_idx = cacheSlot / block_size;
    int64_t const blk_off = cacheSlot % block_size;
    int64_t const cacheOffset = blk_idx * block_stride +
                                blk_off * page_stride +
                                static_cast<int64_t>(headIdx) * head_stride +
                                laneId * numElemsPerThread;

    uint32_t packed = 0;
#pragma unroll
    for (int i = 0; i < numElemsPerThread; i++) {
      __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] / *v_scale_ptr);
      packed |= (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8))
                 << (i * 8));
    }
    *reinterpret_cast<uint32_t*>(&v_cache[cacheOffset]) = packed;
    return;
  }

  // ---- Q and K heads: RoPE ----
  int const rotary_lanes = rotary_dim / numElemsPerThread;
  bool const applyRotary = (laneId < rotary_lanes);

  float const* scale_ptr;
  __nv_fp8_e4m3* out_ptr;
  int64_t out_offset;
  if (headType == Q_HEAD) {
    scale_ptr = q_scale_ptr;
    out_ptr = q_output;
    out_offset = static_cast<int64_t>(tokenIdx) * q_output_stride +
                 static_cast<int64_t>(headIdx) * head_dim +
                 laneId * numElemsPerThread;
  } else {
    int64_t const cacheSlot = slot_mapping[tokenIdx];
    if (cacheSlot < 0) return;

    int64_t const blk_idx = cacheSlot / block_size;
    int64_t const blk_off = cacheSlot % block_size;
    scale_ptr = k_scale_ptr;
    out_ptr = k_cache;
    out_offset = blk_idx * block_stride +
                 blk_off * page_stride +
                 static_cast<int64_t>(headIdx) * head_stride +
                 laneId * numElemsPerThread;
  }

  if (applyRotary) {
    int64_t const pos = position_ids[tokenIdx];
    int const half_rotary = rotary_dim / 2;
    __nv_bfloat16 const* cache_row =
        cos_sin_cache + static_cast<int64_t>(pos) * rotary_dim;

    if constexpr (interleave) {
#pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        float e2 = (i % 2 == 0) ? -elements[i + 1] : elements[i - 1];
        int half_dim = (laneId * numElemsPerThread + i) / 2;
        float cos_val = __bfloat162float(cache_row[half_dim]);
        float sin_val = __bfloat162float(cache_row[half_rotary + half_dim]);
        elements[i] = (elements[i] * cos_val + e2 * sin_val);
      }
    } else {
      // NeoX: shfl_xor pairs lane L with lane L+half_rotary_lanes.
      // active_mask covers only the rotary lanes; __shfl_xor_sync provides
      // its own synchronisation — no __syncwarp() needed (all data is in
      // registers, not shared memory), and calling __syncwarp() here with the
      // default all-lane mask would be UB on Volta+ since lanes >= rotary_lanes
      // never reach this point.
      int const half_rotary_lanes = rotary_lanes / 2;
      unsigned int active_mask = (1u << rotary_lanes) - 1;
      int base_half = (laneId * numElemsPerThread) % half_rotary;

      __nv_bfloat162 cos_p0 =
          *reinterpret_cast<__nv_bfloat162 const*>(&cache_row[base_half]);
      __nv_bfloat162 cos_p1 =
          *reinterpret_cast<__nv_bfloat162 const*>(&cache_row[base_half + 2]);
      float2 cf0 = __bfloat1622float2(cos_p0);
      float2 cf1 = __bfloat1622float2(cos_p1);
      float cos_arr[4] = {cf0.x, cf0.y, cf1.x, cf1.y};

      __nv_bfloat162 sin_p0 = *reinterpret_cast<__nv_bfloat162 const*>(
          &cache_row[half_rotary + base_half]);
      __nv_bfloat162 sin_p1 = *reinterpret_cast<__nv_bfloat162 const*>(
          &cache_row[half_rotary + base_half + 2]);
      float2 sf0 = __bfloat1622float2(sin_p0);
      float2 sf1 = __bfloat1622float2(sin_p1);
      float sin_arr[4] = {sf0.x, sf0.y, sf1.x, sf1.y};

#pragma unroll
      for (int i = 0; i < numElemsPerThread; i++) {
        float e2 = __shfl_xor_sync(active_mask, elements[i], half_rotary_lanes);
        if (laneId < half_rotary_lanes) {
          e2 = -e2;
        }
        elements[i] = (elements[i] * cos_arr[i] + e2 * sin_arr[i]);
      }
    }
  }

  // ---- FP8 cast + store ----
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < numElemsPerThread; i++) {
    __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(elements[i] / *scale_ptr);
    packed |=
        (static_cast<uint32_t>(*reinterpret_cast<uint8_t*>(&fp8)) << (i * 8));
  }
  *reinterpret_cast<uint32_t*>(&out_ptr[out_offset]) = packed;
}

// ============================================================================
// Dispatch helpers
// ============================================================================

#define NOVITA_DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                              \
    const bool INTERLEAVE = true;                                \
    __VA_ARGS__                                                  \
  } else {                                                       \
    const bool INTERLEAVE = false;                               \
    __VA_ARGS__                                                  \
  }

// ============================================================================
// Launcher 1: full pipeline (QK Norm + RoPE + FP8 + KV Store)
// ============================================================================

static void launchFusedQKNormRopeFP8KVStore(
    void const* qkv, int const num_tokens, int const num_heads_q,
    int const num_heads_k, int const num_heads_v, int const head_dim,
    float const eps, void const* q_weight, void const* k_weight,
    bool const interleave, int64_t const* position_ids, int const rotary_dim,
    __nv_bfloat16 const* cos_sin_cache, void* q_output, float const* q_scale,
    int64_t const q_output_stride, void* k_cache, void* v_cache,
    int64_t const* slot_mapping, float const* k_scale, float const* v_scale,
    int64_t const block_size_kv, int64_t const block_stride,
    int64_t const page_stride, int64_t const head_stride,
    cudaStream_t stream) {
  constexpr int blockSize = 128;
  int const warpsPerBlock = blockSize / 32;
  int const totalWarps = num_tokens * (num_heads_q + num_heads_k + num_heads_v);
  int const gridSize = novita_helpers::divUp(totalWarps, warpsPerBlock);

#define NOVITA_LAUNCH_FULL_KERNEL(HD)                                               \
  NOVITA_DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                              \
    fusedQKNormRopeFP8KVStoreKernel<HD, INTERLEAVE>                                 \
        <<<gridSize, blockSize, 0, stream>>>(                                        \
            reinterpret_cast<__nv_bfloat16 const*>(qkv), num_heads_q, num_heads_k, \
            num_heads_v, eps,                                                        \
            reinterpret_cast<__nv_bfloat16 const*>(q_weight),                       \
            reinterpret_cast<__nv_bfloat16 const*>(k_weight), position_ids,         \
            num_tokens, rotary_dim, cos_sin_cache,                                   \
            reinterpret_cast<__nv_fp8_e4m3*>(q_output), q_scale, q_output_stride,  \
            reinterpret_cast<__nv_fp8_e4m3*>(k_cache),                              \
            reinterpret_cast<__nv_fp8_e4m3*>(v_cache), slot_mapping, k_scale,       \
            v_scale, block_size_kv, block_stride, page_stride, head_stride);        \
  });

  switch (head_dim) {
    case 64:
      NOVITA_LAUNCH_FULL_KERNEL(64);
      break;
    case 128:
      NOVITA_LAUNCH_FULL_KERNEL(128);
      break;
    case 256:
      NOVITA_LAUNCH_FULL_KERNEL(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension: ", head_dim);
  }
#undef NOVITA_LAUNCH_FULL_KERNEL
}

// ============================================================================
// Launcher 2: subset pipeline (RoPE + FP8 + KV Store, no QK Norm)
// ============================================================================

static void launchFusedRopeFP8KVStore(
    void const* q, void const* k, void const* v, int const num_tokens,
    int const num_heads_q, int const num_heads_k, int const num_heads_v,
    int const head_dim, bool const interleave, int64_t const* position_ids,
    int const rotary_dim, __nv_bfloat16 const* cos_sin_cache, void* q_output,
    float const* q_scale, int64_t const q_output_stride, void* k_cache,
    void* v_cache, int64_t const* slot_mapping, float const* k_scale,
    float const* v_scale, int64_t const block_size_kv, int64_t const block_stride,
    int64_t const page_stride, int64_t const head_stride,
    cudaStream_t stream) {
  constexpr int blockSize = 128;
  int const warpsPerBlock = blockSize / 32;
  int const totalWarps = num_tokens * (num_heads_q + num_heads_k + num_heads_v);
  int const gridSize = novita_helpers::divUp(totalWarps, warpsPerBlock);

#define NOVITA_LAUNCH_SUBSET_KERNEL(HD)                                               \
  NOVITA_DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {                                \
    fusedRopeFP8KVStoreKernel<HD, INTERLEAVE>                                         \
        <<<gridSize, blockSize, 0, stream>>>(                                          \
            reinterpret_cast<__nv_bfloat16 const*>(q),                                \
            reinterpret_cast<__nv_bfloat16 const*>(k),                                \
            reinterpret_cast<__nv_bfloat16 const*>(v), num_heads_q, num_heads_k,      \
            num_heads_v, position_ids, num_tokens, rotary_dim, cos_sin_cache,          \
            reinterpret_cast<__nv_fp8_e4m3*>(q_output), q_scale, q_output_stride,    \
            reinterpret_cast<__nv_fp8_e4m3*>(k_cache),                                \
            reinterpret_cast<__nv_fp8_e4m3*>(v_cache), slot_mapping, k_scale, v_scale, \
            block_size_kv, block_stride, page_stride, head_stride);                    \
  });

  switch (head_dim) {
    case 64:
      NOVITA_LAUNCH_SUBSET_KERNEL(64);
      break;
    case 128:
      NOVITA_LAUNCH_SUBSET_KERNEL(128);
      break;
    case 256:
      NOVITA_LAUNCH_SUBSET_KERNEL(256);
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension: ", head_dim);
  }
#undef NOVITA_LAUNCH_SUBSET_KERNEL
}

// ============================================================================
// Torch C++ entry point 1: full pipeline (for glm4_moe)
// ============================================================================

void fused_qk_norm_rope_fp8_kvstore(
    torch::Tensor& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, double eps,
    torch::Tensor& q_weight, torch::Tensor& k_weight, bool is_neox,
    torch::Tensor& position_ids, int64_t rotary_dim,
    torch::Tensor& cos_sin_cache, torch::Tensor& q_output,
    torch::Tensor& q_scale, torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& slot_mapping, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  NOVITA_CHECK_INPUT(qkv, torch::kBFloat16);
  NOVITA_CHECK_INPUT(position_ids, torch::kInt64);
  NOVITA_CHECK_INPUT(q_weight, torch::kBFloat16);
  NOVITA_CHECK_INPUT(k_weight, torch::kBFloat16);
  NOVITA_CHECK_INPUT(slot_mapping, torch::kInt64);
  NOVITA_CHECK_INPUT(cos_sin_cache, torch::kBFloat16);
  NOVITA_CHECK_TH_CUDA(q_output);
  NOVITA_CHECK_CONTIGUOUS(q_output);
  NOVITA_CHECK_TH_CUDA(k_cache);
  NOVITA_CHECK_TH_CUDA(v_cache);
  TORCH_CHECK(q_scale.numel() == 1, "q_scale must be a single-element tensor");
  TORCH_CHECK(k_scale.numel() == 1, "k_scale must be a single-element tensor");
  TORCH_CHECK(v_scale.numel() == 1, "v_scale must be a single-element tensor");
  NOVITA_CHECK_INPUT(q_scale, torch::kFloat32);
  NOVITA_CHECK_INPUT(k_scale, torch::kFloat32);
  NOVITA_CHECK_INPUT(v_scale, torch::kFloat32);

  int64_t num_tokens = qkv.size(0);
  int64_t q_output_stride = num_heads_q * head_dim;
  // Extract KV cache strides for layout-agnostic addressing (NHD or HND).
  // k_cache logical shape: [num_blocks, block_size, num_kv_heads, head_dim]
  int64_t block_size_kv = k_cache.size(1);
  int64_t block_stride = k_cache.stride(0);
  int64_t page_stride = k_cache.stride(1);
  int64_t head_stride_kv = k_cache.stride(2);
  auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

  launchFusedQKNormRopeFP8KVStore(
      qkv.data_ptr(), static_cast<int>(num_tokens), static_cast<int>(num_heads_q),
      static_cast<int>(num_heads_k), static_cast<int>(num_heads_v),
      static_cast<int>(head_dim), static_cast<float>(eps), q_weight.data_ptr(),
      k_weight.data_ptr(), !is_neox,
      reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
      static_cast<int>(rotary_dim),
      reinterpret_cast<__nv_bfloat16 const*>(cos_sin_cache.data_ptr()),
      q_output.data_ptr(),
      reinterpret_cast<float const*>(q_scale.data_ptr()), q_output_stride,
      k_cache.data_ptr(), v_cache.data_ptr(),
      reinterpret_cast<int64_t const*>(slot_mapping.data_ptr()),
      reinterpret_cast<float const*>(k_scale.data_ptr()),
      reinterpret_cast<float const*>(v_scale.data_ptr()),
      block_size_kv, block_stride, page_stride, head_stride_kv,
      stream);
}

// ============================================================================
// Torch C++ entry point 2: subset pipeline (for minimax_m2)
// ============================================================================

void fused_rope_fp8_kvstore(
    torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, bool is_neox,
    torch::Tensor& position_ids, int64_t rotary_dim,
    torch::Tensor& cos_sin_cache, torch::Tensor& q_output,
    torch::Tensor& q_scale, torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& slot_mapping, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  NOVITA_CHECK_INPUT(q, torch::kBFloat16);
  NOVITA_CHECK_INPUT(k, torch::kBFloat16);
  NOVITA_CHECK_INPUT(v, torch::kBFloat16);
  NOVITA_CHECK_INPUT(position_ids, torch::kInt64);
  NOVITA_CHECK_INPUT(slot_mapping, torch::kInt64);
  NOVITA_CHECK_INPUT(cos_sin_cache, torch::kBFloat16);
  NOVITA_CHECK_TH_CUDA(q_output);
  NOVITA_CHECK_CONTIGUOUS(q_output);
  NOVITA_CHECK_TH_CUDA(k_cache);
  NOVITA_CHECK_TH_CUDA(v_cache);
  TORCH_CHECK(q_scale.numel() == 1, "q_scale must be a single-element tensor");
  TORCH_CHECK(k_scale.numel() == 1, "k_scale must be a single-element tensor");
  TORCH_CHECK(v_scale.numel() == 1, "v_scale must be a single-element tensor");
  NOVITA_CHECK_INPUT(q_scale, torch::kFloat32);
  NOVITA_CHECK_INPUT(k_scale, torch::kFloat32);
  NOVITA_CHECK_INPUT(v_scale, torch::kFloat32);

  // q: [num_tokens, num_heads_q * head_dim], derive heads/dim from shape
  int64_t num_tokens = q.size(0);
  TORCH_CHECK(q.dim() == 2, "q must be 2D [num_tokens, num_heads_q*head_dim]");
  TORCH_CHECK(k.dim() == 2, "k must be 2D [num_tokens, num_heads_k*head_dim]");
  TORCH_CHECK(v.dim() == 2, "v must be 2D [num_tokens, num_heads_v*head_dim]");

  int64_t kv_heads_times_dim = k.size(1);
  int64_t q_heads_times_dim = q.size(1);
  int64_t v_heads_times_dim = v.size(1);

  int64_t q_output_stride = q_heads_times_dim;

  TORCH_CHECK(kv_heads_times_dim == v_heads_times_dim,
              "k and v must have same last dimension");

  // k_cache logical shape: [num_blocks, block_size, num_kv_heads, head_dim]
  // Strides encode the physical layout (NHD or HND).
  int64_t head_dim = k_cache.size(-1);
  TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
              "head_dim must be 64, 128, or 256; got ", head_dim);

  int64_t block_size_kv = k_cache.size(1);
  int64_t block_stride = k_cache.stride(0);
  int64_t page_stride = k_cache.stride(1);
  int64_t head_stride_kv = k_cache.stride(2);

  int64_t num_heads_q = q_heads_times_dim / head_dim;
  int64_t num_heads_k = kv_heads_times_dim / head_dim;
  int64_t num_heads_v = v_heads_times_dim / head_dim;

  auto stream = at::cuda::getCurrentCUDAStream(q.get_device());

  launchFusedRopeFP8KVStore(
      q.data_ptr(), k.data_ptr(), v.data_ptr(), static_cast<int>(num_tokens),
      static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
      static_cast<int>(num_heads_v), static_cast<int>(head_dim), !is_neox,
      reinterpret_cast<int64_t const*>(position_ids.data_ptr()),
      static_cast<int>(rotary_dim),
      reinterpret_cast<__nv_bfloat16 const*>(cos_sin_cache.data_ptr()),
      q_output.data_ptr(),
      reinterpret_cast<float const*>(q_scale.data_ptr()), q_output_stride,
      k_cache.data_ptr(), v_cache.data_ptr(),
      reinterpret_cast<int64_t const*>(slot_mapping.data_ptr()),
      reinterpret_cast<float const*>(k_scale.data_ptr()),
      reinterpret_cast<float const*>(v_scale.data_ptr()),
      block_size_kv, block_stride, page_stride, head_stride_kv,
      stream);
}
