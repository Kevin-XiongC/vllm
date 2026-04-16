// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused RoPE + FP8 cast + KV cache write (q/k/v are pre-normed upstream)
//
// v4 scheduling:
//   Grid: 2D (num_tokens, num_heads_kv) — one block per (token, kv_head)
//   Block: dynamic warp count = min(2 + gqa_ratio, 5) * 32
//   Scheduling: V/K/Q ops assigned round-robin across warps
//
//   NeoX path (!interleave): contiguous-within-half mapping
//   GPT-J path (interleave): contiguous mapping

#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>

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

template <int N>
struct fp8_store_type;
template <>
struct fp8_store_type<2> {
  using type = uint16_t;
};
template <>
struct fp8_store_type<4> {
  using type = uint32_t;
};
template <>
struct fp8_store_type<8> {
  using type = uint2;
};

template <typename T>
inline __device__ __host__ T divUp(T m, T n) {
  return (m + n - 1) / n;
}

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
  return __bfloat162float(x);
}

__device__ __forceinline__ uint8_t float_to_fp8_e4m3(float val) {
  __nv_fp8_e4m3 fp8 = __nv_fp8_e4m3(val);
  return *reinterpret_cast<uint8_t*>(&fp8);
}

template <int head_dim>
__device__ __forceinline__ void load_head_neox(
    __nv_bfloat16 const* src, float* lo, float* hi, int lane) {
  using T2 = __nv_bfloat162;
  constexpr int HALF = head_dim / 2;
  constexpr int PPT = HALF / 32;
  int const thr_off = PPT * lane;

  if constexpr (PPT == 1) {
    lo[0] = bf16_to_float(src[thr_off]);
    hi[0] = bf16_to_float(src[HALF + thr_off]);
  } else {
    constexpr int halfBytes = PPT * sizeof(__nv_bfloat16);
    constexpr int vecSize = halfBytes / 4;
    using vec_T = typename packed_as<uint, vecSize>::type;
    constexpr int num_packed = halfBytes / sizeof(T2);
    vec_T v_lo = *reinterpret_cast<vec_T const*>(&src[thr_off]);
#pragma unroll
    for (int i = 0; i < num_packed; ++i) {
      float2 vals =
          __bfloat1622float2(*(reinterpret_cast<T2 const*>(&v_lo) + i));
      lo[2 * i] = vals.x;
      lo[2 * i + 1] = vals.y;
    }
    vec_T v_hi = *reinterpret_cast<vec_T const*>(&src[HALF + thr_off]);
#pragma unroll
    for (int i = 0; i < num_packed; ++i) {
      float2 vals =
          __bfloat1622float2(*(reinterpret_cast<T2 const*>(&v_hi) + i));
      hi[2 * i] = vals.x;
      hi[2 * i + 1] = vals.y;
    }
  }
}

template <int head_dim, bool IS_FP8>
__device__ __forceinline__ void write_cache_neox(
    __nv_fp8_e4m3* cache, int64_t offset, float const* lo, float const* hi,
    float scale, int lane) {
  constexpr int HALF = head_dim / 2;
  constexpr int PPT = HALF / 32;

  static_assert(IS_FP8, "novita path expects fp8 cache writes");
  int const thr_off = PPT * lane;
  if constexpr (PPT == 1) {
    reinterpret_cast<uint8_t*>(cache)[offset + thr_off] =
        float_to_fp8_e4m3(lo[0] / scale);
    reinterpret_cast<uint8_t*>(cache)[offset + HALF + thr_off] =
        float_to_fp8_e4m3(hi[0] / scale);
  } else {
    uint8_t fp8_lo[PPT];
    uint8_t fp8_hi[PPT];
#pragma unroll
    for (int i = 0; i < PPT; ++i) {
      fp8_lo[i] = float_to_fp8_e4m3(lo[i] / scale);
      fp8_hi[i] = float_to_fp8_e4m3(hi[i] / scale);
    }
    using fp8_vec_t = typename fp8_store_type<PPT>::type;
    *reinterpret_cast<fp8_vec_t*>(&reinterpret_cast<uint8_t*>(cache)[offset +
                                                                     thr_off]) =
        *reinterpret_cast<fp8_vec_t const*>(fp8_lo);
    *reinterpret_cast<fp8_vec_t*>(
        &reinterpret_cast<uint8_t*>(cache)[offset + HALF + thr_off]) =
        *reinterpret_cast<fp8_vec_t const*>(fp8_hi);
  }
}

template <int head_dim>
__device__ __forceinline__ void rope_neox(
    float* lo, float* hi, __nv_bfloat16 const* cos_ptr,
    __nv_bfloat16 const* sin_ptr, int lane, int embed_dim) {
  constexpr int PPT = (head_dim / 2) / 32;
  int const base = PPT * lane;
#pragma unroll
  for (int p = 0; p < PPT; ++p) {
    int const dim = base + p;
    if (dim < embed_dim) {
      float const c = bf16_to_float(cos_ptr[dim]);
      float const s = bf16_to_float(sin_ptr[dim]);
      float const lo_v = lo[p];
      float const hi_v = hi[p];
      lo[p] = lo_v * c - hi_v * s;
      hi[p] = hi_v * c + lo_v * s;
    }
  }
}

template <int head_dim>
__device__ __forceinline__ void load_head_gptj(
    __nv_bfloat16 const* src, int thr_off, float* elems) {
  using T2 = __nv_bfloat162;
  constexpr int EPT = head_dim / 32;
  constexpr int elemSizeBytes = EPT * sizeof(__nv_bfloat16);
  constexpr int vecSize = elemSizeBytes / 4;
  using vec_T = typename packed_as<uint, vecSize>::type;
  constexpr int num_packed = elemSizeBytes / sizeof(T2);

  vec_T v = *reinterpret_cast<vec_T const*>(&src[thr_off]);
#pragma unroll
  for (int i = 0; i < num_packed; ++i) {
    float2 vals = __bfloat1622float2(*(reinterpret_cast<T2 const*>(&v) + i));
    elems[2 * i] = vals.x;
    elems[2 * i + 1] = vals.y;
  }
}

template <int head_dim, bool IS_FP8>
__device__ __forceinline__ void write_fp8_gptj(
    __nv_fp8_e4m3* dst, int64_t offset, int thr_off, float const* elems,
    float scale) {
  constexpr int EPT = head_dim / 32;
  static_assert(IS_FP8, "novita path expects fp8 outputs");

  uint8_t fp8_vals[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i) {
    fp8_vals[i] = float_to_fp8_e4m3(elems[i] / scale);
  }
  using fp8_vec_t = typename fp8_store_type<EPT>::type;
  *reinterpret_cast<fp8_vec_t*>(&reinterpret_cast<uint8_t*>(dst)[offset +
                                                                 thr_off]) =
      *reinterpret_cast<fp8_vec_t const*>(fp8_vals);
}

template <int head_dim>
__device__ __forceinline__ void rope_gptj(
    float* elems, __nv_bfloat16 const* cos_ptr, __nv_bfloat16 const* sin_ptr,
    int lane, int rotary_dim) {
  constexpr int EPT = head_dim / 32;
  int const dim_base = lane * EPT;
  if (dim_base >= rotary_dim) return;

#pragma unroll
  for (int i = 0; i < EPT / 2; ++i) {
    int const idx0 = 2 * i;
    int const idx1 = 2 * i + 1;
    int const dim0 = dim_base + idx0;
    int const dim1 = dim_base + idx1;
    if (dim1 < rotary_dim) {
      int const half_dim = dim0 / 2;
      float const c = bf16_to_float(cos_ptr[half_dim]);
      float const s = bf16_to_float(sin_ptr[half_dim]);
      float const v0 = elems[idx0];
      float const v1 = elems[idx1];
      elems[idx0] = v0 * c - v1 * s;
      elems[idx1] = v0 * s + v1 * c;
    }
  }
}

template <int head_dim>
__device__ __forceinline__ void rope_neox_shuffle(
    float* elems, __nv_bfloat16 const* cos_ptr,
    __nv_bfloat16 const* sin_ptr, int lane, int embed_dim) {
  constexpr int EPT = head_dim / 32;
  int const dim_base = lane * EPT;
  int const rotary_dim = 2 * embed_dim;
  int const partner_offset = embed_dim / EPT;
  bool const is_lo = (dim_base < embed_dim);
  bool const is_hi = (dim_base >= embed_dim && dim_base < rotary_dim);

  int partner_lane = lane;
  if (is_lo) partner_lane = lane + partner_offset;
  else if (is_hi) partner_lane = lane - partner_offset;

  float partner[EPT];
#pragma unroll
  for (int i = 0; i < EPT; ++i)
    partner[i] = __shfl_sync(0xffffffff, elems[i], partner_lane);

  if (is_lo) {
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      float c = bf16_to_float(cos_ptr[dim_base + i]);
      float s = bf16_to_float(sin_ptr[dim_base + i]);
      elems[i] = elems[i] * c - partner[i] * s;
    }
  } else if (is_hi) {
    int const cos_base = dim_base - embed_dim;
#pragma unroll
    for (int i = 0; i < EPT; ++i) {
      float c = bf16_to_float(cos_ptr[cos_base + i]);
      float s = bf16_to_float(sin_ptr[cos_base + i]);
      elems[i] = elems[i] * c + partner[i] * s;
    }
  }
}

}  // namespace novita_helpers

template <int head_dim, bool interleave, bool IS_FP8>
__global__ void fusedRopeFP8KVStoreKernelV4(
    __nv_bfloat16 const* __restrict__ q, __nv_bfloat16 const* __restrict__ k,
    __nv_bfloat16 const* __restrict__ v, int const num_heads_q,
    int const num_heads_k, int const num_heads_v,
    int64_t const* __restrict__ position_ids, int const num_tokens,
    int const rotary_dim, __nv_bfloat16 const* __restrict__ cos_sin_cache,
    __nv_fp8_e4m3* q_output, float const* __restrict__ q_scale_ptr,
    int64_t const q_output_stride, __nv_fp8_e4m3* k_cache,
    __nv_fp8_e4m3* v_cache, int64_t const* __restrict__ slot_mapping,
    float const* __restrict__ k_scale_ptr, float const* __restrict__ v_scale_ptr,
    int64_t const num_blocks, int64_t const block_size,
    int64_t const block_stride, int64_t const page_stride,
    int64_t const head_stride, int64_t const max_position) {
  static_assert(IS_FP8, "novita fused path writes fp8 outputs only");
  int const token_idx = blockIdx.x;
  int const kv_head = blockIdx.y;
  if (token_idx >= num_tokens || kv_head >= num_heads_k) return;

  int const warp_id = threadIdx.x / 32;
  int const lane = threadIdx.x & 31;
  int const num_warps = blockDim.x / 32;

  int const gqa_ratio = num_heads_q / num_heads_k;
  int const total_ops = 2 + gqa_ratio;  // V + K + all Q in this KV group
  int const q_start = kv_head * gqa_ratio;

  int const q_stride = num_heads_q * head_dim;
  int const kv_stride = num_heads_k * head_dim;

  __nv_bfloat16 const* q_in = q + static_cast<int64_t>(token_idx) * q_stride;
  __nv_bfloat16 const* k_in = k + static_cast<int64_t>(token_idx) * kv_stride;
  __nv_bfloat16 const* v_in = v + static_cast<int64_t>(token_idx) * kv_stride;

  int64_t const pos = position_ids[token_idx];
  // Clamp position to valid cos_sin_cache range to prevent OOB access.
  int64_t const safe_pos = (pos >= 0 && pos < max_position) ? pos : 0;
  int const embed_dim = rotary_dim / 2;
  __nv_bfloat16 const* cos_ptr = cos_sin_cache + safe_pos * rotary_dim;
  __nv_bfloat16 const* sin_ptr = cos_ptr + embed_dim;

  int64_t const slot_idx = slot_mapping[token_idx];
  // Only disable KV writes for OOB slots — never return early, because
  // Q RoPE + FP8 write must always run.  An early return would leave
  // q_output as torch.empty garbage, which FA3 TMA then reads → IMA.
  bool write_kv = (slot_idx >= 0);
  int64_t cache_head_offset = 0;
  if (write_kv) {
    int64_t const blk_idx = slot_idx / block_size;
    int64_t const blk_off = slot_idx % block_size;
    if (blk_idx >= num_blocks) {
      write_kv = false;
    } else {
      cache_head_offset = blk_idx * block_stride + blk_off * page_stride +
                          static_cast<int64_t>(kv_head) * head_stride;
    }
  }
  float const q_scale = *q_scale_ptr;
  float const k_scale = *k_scale_ptr;
  float const v_scale = *v_scale_ptr;

  if constexpr (!interleave) {
    static_assert(head_dim % 64 == 0,
                  "head_dim must be divisible by 64 for NeoX path");
    constexpr int EPT = head_dim / 32;
    int const thr_off = lane * EPT;

    for (int op = warp_id; op < total_ops; op += num_warps) {
      if (op == 0) {
        if (write_kv) {
          float elems[EPT];
          novita_helpers::load_head_gptj<head_dim>(
              v_in + kv_head * head_dim, thr_off, elems);
          novita_helpers::write_fp8_gptj<head_dim, IS_FP8>(
              v_cache, cache_head_offset, thr_off, elems, v_scale);
        }
      } else if (op == 1) {
        if (write_kv) {
          float elems[EPT];
          novita_helpers::load_head_gptj<head_dim>(
              k_in + kv_head * head_dim, thr_off, elems);
          novita_helpers::rope_neox_shuffle<head_dim>(
              elems, cos_ptr, sin_ptr, lane, embed_dim);
          novita_helpers::write_fp8_gptj<head_dim, IS_FP8>(
              k_cache, cache_head_offset, thr_off, elems, k_scale);
        }
      } else {
        int const q_head = q_start + (op - 2);
        float elems[EPT];
        novita_helpers::load_head_gptj<head_dim>(
            q_in + q_head * head_dim, thr_off, elems);
        novita_helpers::rope_neox_shuffle<head_dim>(
            elems, cos_ptr, sin_ptr, lane, embed_dim);

        int64_t const q_offset =
            static_cast<int64_t>(token_idx) * q_output_stride +
            static_cast<int64_t>(q_head) * head_dim;
        novita_helpers::write_fp8_gptj<head_dim, IS_FP8>(
            q_output, q_offset, thr_off, elems, q_scale);
      }
    }
  } else {
    static_assert(head_dim % 64 == 0,
                  "head_dim must be divisible by 64 for GPT-J path");
    constexpr int EPT = head_dim / 32;
    int const thr_off = lane * EPT;

    for (int op = warp_id; op < total_ops; op += num_warps) {
      if (op == 0) {
        if (write_kv) {
          float elems[EPT];
          novita_helpers::load_head_gptj<head_dim>(
              v_in + kv_head * head_dim, thr_off, elems);
          novita_helpers::write_fp8_gptj<head_dim, IS_FP8>(
              v_cache, cache_head_offset, thr_off, elems, v_scale);
        }
      } else if (op == 1) {
        if (write_kv) {
          float elems[EPT];
          novita_helpers::load_head_gptj<head_dim>(
              k_in + kv_head * head_dim, thr_off, elems);
          novita_helpers::rope_gptj<head_dim>(
              elems, cos_ptr, sin_ptr, lane, rotary_dim);
          novita_helpers::write_fp8_gptj<head_dim, IS_FP8>(
              k_cache, cache_head_offset, thr_off, elems, k_scale);
        }
      } else {
        int const q_head = q_start + (op - 2);
        float elems[EPT];
        novita_helpers::load_head_gptj<head_dim>(
            q_in + q_head * head_dim, thr_off, elems);
        novita_helpers::rope_gptj<head_dim>(
            elems, cos_ptr, sin_ptr, lane, rotary_dim);
        int64_t const q_offset =
            static_cast<int64_t>(token_idx) * q_output_stride +
            static_cast<int64_t>(q_head) * head_dim;
        novita_helpers::write_fp8_gptj<head_dim, IS_FP8>(
            q_output, q_offset, thr_off, elems, q_scale);
      }
    }
  }
}

#define NOVITA_DISPATCH_INTERLEAVE(interleave, INTERLEAVE, ...) \
  if (interleave) {                                              \
    const bool INTERLEAVE = true;                                \
    __VA_ARGS__                                                  \
  } else {                                                       \
    const bool INTERLEAVE = false;                               \
    __VA_ARGS__                                                  \
  }

static void launchFusedRopeFP8KVStore(
    void const* q, void const* k, void const* v, int const num_tokens,
    int const num_heads_q, int const num_heads_k, int const num_heads_v,
    int const head_dim, bool const interleave, int64_t const* position_ids,
    int const rotary_dim, __nv_bfloat16 const* cos_sin_cache, void* q_output,
    float const* q_scale, int64_t const q_output_stride, void* k_cache,
    void* v_cache, int64_t const* slot_mapping, float const* k_scale,
    float const* v_scale, int64_t const num_blocks, int64_t const block_size_kv,
    int64_t const block_stride, int64_t const page_stride,
    int64_t const head_stride, int64_t const max_position,
    cudaStream_t stream) {
  dim3 const grid(num_tokens, num_heads_k);

  int const gqa_ratio = num_heads_q / num_heads_k;
  int const total_ops = 2 + gqa_ratio;
  constexpr int MAX_WARPS = 5;
  int const warps_per_block = total_ops < MAX_WARPS ? total_ops : MAX_WARPS;
  int const blockSize = warps_per_block * 32;

#define NOVITA_LAUNCH_KERNEL(HD, INTERLEAVE)                                     \
  fusedRopeFP8KVStoreKernelV4<HD, INTERLEAVE, true>                              \
      <<<grid, blockSize, 0, stream>>>(                                          \
          reinterpret_cast<__nv_bfloat16 const*>(q),                             \
          reinterpret_cast<__nv_bfloat16 const*>(k),                             \
          reinterpret_cast<__nv_bfloat16 const*>(v), num_heads_q, num_heads_k,   \
          num_heads_v, position_ids, num_tokens, rotary_dim, cos_sin_cache,      \
          reinterpret_cast<__nv_fp8_e4m3*>(q_output), q_scale, q_output_stride,   \
          reinterpret_cast<__nv_fp8_e4m3*>(k_cache),                             \
          reinterpret_cast<__nv_fp8_e4m3*>(v_cache), slot_mapping, k_scale,      \
          v_scale, num_blocks, block_size_kv, block_stride, page_stride,         \
          head_stride, max_position)

  switch (head_dim) {
    case 64:
      NOVITA_DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        NOVITA_LAUNCH_KERNEL(64, INTERLEAVE);
      });
      break;
    case 128:
      NOVITA_DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        NOVITA_LAUNCH_KERNEL(128, INTERLEAVE);
      });
      break;
    case 256:
      NOVITA_DISPATCH_INTERLEAVE(interleave, INTERLEAVE, {
        NOVITA_LAUNCH_KERNEL(256, INTERLEAVE);
      });
      break;
    default:
      TORCH_CHECK(false, "Unsupported head dimension: ", head_dim);
  }

#undef NOVITA_LAUNCH_KERNEL
}

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

  TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be 1D");
  TORCH_CHECK(q.size(0) >= slot_mapping.size(0),
              "q first dim must be >= slot_mapping size");
  TORCH_CHECK(k.size(0) >= slot_mapping.size(0),
              "k first dim must be >= slot_mapping size");
  TORCH_CHECK(v.size(0) >= slot_mapping.size(0),
              "v first dim must be >= slot_mapping size");
  TORCH_CHECK(position_ids.size(0) >= slot_mapping.size(0),
              "position_ids first dim must be >= slot_mapping size");

  int64_t const num_tokens = slot_mapping.size(0);
  TORCH_CHECK(q_output.size(0) >= num_tokens,
              "q_output first dim must be >= actual token count");
  TORCH_CHECK(q.dim() == 2, "q must be 2D [num_tokens, num_heads_q*head_dim]");
  TORCH_CHECK(k.dim() == 2, "k must be 2D [num_tokens, num_heads_k*head_dim]");
  TORCH_CHECK(v.dim() == 2, "v must be 2D [num_tokens, num_heads_v*head_dim]");

  int64_t const kv_heads_times_dim = k.size(1);
  int64_t const q_heads_times_dim = q.size(1);
  int64_t const v_heads_times_dim = v.size(1);
  int64_t const q_output_stride = q_heads_times_dim;

  TORCH_CHECK(kv_heads_times_dim == v_heads_times_dim,
              "k and v must have same last dimension");

  int64_t const head_dim = k_cache.size(-1);
  TORCH_CHECK(head_dim == 64 || head_dim == 128 || head_dim == 256,
              "head_dim must be 64, 128, or 256; got ", head_dim);
  TORCH_CHECK(rotary_dim > 0 && rotary_dim <= head_dim,
              "rotary_dim must be in (0, head_dim]");
  TORCH_CHECK(q_heads_times_dim % head_dim == 0,
              "q last dimension must be divisible by head_dim");
  TORCH_CHECK(kv_heads_times_dim % head_dim == 0,
              "k last dimension must be divisible by head_dim");
  TORCH_CHECK(q_heads_times_dim / head_dim > 0,
              "num_heads_q must be greater than 0");
  TORCH_CHECK(kv_heads_times_dim / head_dim > 0,
              "num_heads_k must be greater than 0");
  TORCH_CHECK((q_heads_times_dim / head_dim) % (kv_heads_times_dim / head_dim) == 0,
              "num_heads_q must be divisible by num_heads_k");

  int64_t const num_blocks = k_cache.size(0);
  int64_t const block_size_kv = k_cache.size(1);
  int64_t const block_stride = k_cache.stride(0);
  int64_t const page_stride = k_cache.stride(1);
  int64_t const head_stride_kv = k_cache.stride(2);
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(),
              "k_cache and v_cache must have identical shapes");
  TORCH_CHECK(k_cache.strides() == v_cache.strides(),
              "k_cache and v_cache must have identical strides");

  int64_t const num_heads_q = q_heads_times_dim / head_dim;
  int64_t const num_heads_k = kv_heads_times_dim / head_dim;
  int64_t const num_heads_v = v_heads_times_dim / head_dim;
  TORCH_CHECK(num_heads_k == num_heads_v, "num_heads_k must equal num_heads_v");

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
      num_blocks, block_size_kv, block_stride, page_stride, head_stride_kv,
      cos_sin_cache.size(0), stream);
}
