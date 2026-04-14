// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>

#include "core/registration.h"

// Fused QK-Norm + RoPE + FP8 quantization + KV cache store (glm4_moe)
void fused_qk_norm_rope_fp8_kvstore(
    torch::Tensor& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, double eps,
    torch::Tensor& q_weight, torch::Tensor& k_weight, bool is_neox,
    torch::Tensor& position_ids, int64_t rotary_dim,
    torch::Tensor& cos_sin_cache, torch::Tensor& q_output,
    torch::Tensor& q_scale, torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& slot_mapping, torch::Tensor& k_scale,
    torch::Tensor& v_scale);

// Fused RoPE + FP8 quantization + KV cache store (minimax_m2, no QK Norm)
void fused_rope_fp8_kvstore(
    torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, bool is_neox,
    torch::Tensor& position_ids, int64_t rotary_dim,
    torch::Tensor& cos_sin_cache, torch::Tensor& q_output,
    torch::Tensor& q_scale, torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& slot_mapping, torch::Tensor& k_scale,
    torch::Tensor& v_scale);

// Fused AllReduce + RMSNorm + Residual Add (+ optional FP8/FP4 quantization)
void novita_allreduce_fusion(
    torch::Tensor& allreduce_in, torch::Tensor& residual_in,
    torch::Tensor& residual_out, torch::Tensor& workspace_ptrs,
    int64_t world_size, int64_t world_rank, int64_t token_num,
    int64_t hidden_dim, double rms_eps, torch::Tensor& rms_gamma,
    bool launch_with_pdl, bool fp32_acc, bool use_oneshot,
    int64_t pattern_code, torch::Tensor& norm_out, torch::Tensor& quant_out,
    torch::Tensor& scale_out, torch::Tensor& scale_factor);

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "fused_qk_norm_rope_fp8_kvstore(Tensor! qkv, int num_heads_q, "
      "int num_heads_k, int num_heads_v, int head_dim, float eps, "
      "Tensor q_weight, Tensor k_weight, bool is_neox, "
      "Tensor position_ids, int rotary_dim, Tensor cos_sin_cache, "
      "Tensor! q_output, Tensor q_scale, "
      "Tensor! k_cache, Tensor! v_cache, "
      "Tensor slot_mapping, Tensor k_scale, Tensor v_scale) -> ()");
  ops.impl("fused_qk_norm_rope_fp8_kvstore", torch::kCUDA,
           &fused_qk_norm_rope_fp8_kvstore);

  ops.def(
      "fused_rope_fp8_kvstore(Tensor! q, Tensor! k, Tensor! v, bool is_neox, "
      "Tensor position_ids, int rotary_dim, Tensor cos_sin_cache, "
      "Tensor! q_output, Tensor q_scale, "
      "Tensor! k_cache, Tensor! v_cache, "
      "Tensor slot_mapping, Tensor k_scale, Tensor v_scale) -> ()");
  ops.impl("fused_rope_fp8_kvstore", torch::kCUDA, &fused_rope_fp8_kvstore);

  ops.def(
      "allreduce_fusion(Tensor! allreduce_in, Tensor! residual_in, "
      "Tensor! residual_out, Tensor workspace_ptrs, "
      "int world_size, int world_rank, "
      "int token_num, int hidden_dim, float rms_eps, "
      "Tensor rms_gamma, bool launch_with_pdl, bool fp32_acc, "
      "bool use_oneshot, int pattern_code, "
      "Tensor! norm_out, Tensor! quant_out, "
      "Tensor! scale_out, Tensor scale_factor) -> ()");
  ops.impl("allreduce_fusion", torch::kCUDA, &novita_allreduce_fusion);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
