// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>

#include "core/registration.h"

// Fused RoPE + FP8 quantization + KV cache store (minimax_m2; q/k normed upstream)
void fused_rope_fp8_kvstore(
    torch::Tensor& q, torch::Tensor& k, torch::Tensor& v, bool is_neox,
    torch::Tensor& position_ids, int64_t rotary_dim,
    torch::Tensor& cos_sin_cache, torch::Tensor& q_output,
    torch::Tensor& q_scale, torch::Tensor& k_cache, torch::Tensor& v_cache,
    torch::Tensor& slot_mapping, torch::Tensor& k_scale,
    torch::Tensor& v_scale);

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def(
      "fused_rope_fp8_kvstore(Tensor! q, Tensor! k, Tensor! v, bool is_neox, "
      "Tensor position_ids, int rotary_dim, Tensor cos_sin_cache, "
      "Tensor! q_output, Tensor q_scale, "
      "Tensor! k_cache, Tensor! v_cache, "
      "Tensor slot_mapping, Tensor k_scale, Tensor v_scale) -> ()");
  ops.impl("fused_rope_fp8_kvstore", torch::kCUDA, &fused_rope_fp8_kvstore);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
