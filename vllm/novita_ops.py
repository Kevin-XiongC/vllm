# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Novita fused kernels (vllm._novita_C):
  Fused RoPE + FP8 KV-store for MiniMax M2.

Isolated in a separate .so from the main vLLM C extensions.
"""

from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_novita_available = False
if current_platform.is_cuda():
    try:
        import vllm._novita_C  # noqa: F401

        _novita_available = True
    except ImportError:
        pass


def is_novita_available() -> bool:
    return _novita_available


def register_novita_ops() -> None:
    """Register the novita custom op with vLLM's torch library.

    Must be called after vllm._novita_C is imported.
    """
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="novita_fused_rope_fp8_kvstore",
        op_func=novita_fused_rope_fp8_kvstore,
        mutates_args=["output", "q_output"],
        fake_impl=novita_fused_rope_fp8_kvstore_fake,
    )


# ---------------------------------------------------------------------------
# Custom op: novita_fused_rope_fp8_kvstore
#
# Encapsulates get_attention_context + fused CUDA kernel + attention-only
# forward into a single opaque op so torch.compile / cudagraph can trace
# through MiniMaxM2Attention._forward_fused without graph breaks.
# Registered as a splitting op (see compilation.py).
# ---------------------------------------------------------------------------

_logged_fused_rope_layers: set[str] = set()


def novita_fused_rope_fp8_kvstore(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    q_output: torch.Tensor,
    layer_name: str,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    q_size: int,
    kv_size: int,
    rotary_dim: int,
) -> None:
    """Fused RoPE + FP8 cast + KV cache store + attention for minimax_m2.

    Q and K are already QK-normed (TP allreduce done separately).
    During memory profiling (kv_cache empty), falls back to unfused ops.
    During normal inference, runs the novita fused CUDA kernel then calls
    unified_attention_with_output.

    Both ``output`` (bf16 attention result) and ``q_output`` (fp8 Q after
    RoPE) must be pre-allocated by the caller so that their addresses are
    stable across CUDA-graph replays, avoiding D2D copies at compiled-graph
    boundaries.
    """
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
        unified_attention_with_output,
        unified_kv_cache_update,
    )

    num_heads_v = num_heads_k
    _, _, kv_cache, slot_mapping = get_attention_context(layer_name)

    if kv_cache.numel() == 0 or slot_mapping is None:
        # ---- Profiling fallback: unfused ops ----
        torch.ops._C.rotary_embedding(positions, q, k, head_dim, cos_sin_cache, True)

        q_3d = q.view(-1, num_heads_q, head_dim)
        k_3d = k.view(-1, num_heads_k, head_dim)
        v_3d = v.view(-1, num_heads_v, head_dim)
        output_view = output.view(-1, num_heads_q, head_dim)

        kv_dep = unified_kv_cache_update(k_3d, v_3d, layer_name)
        unified_attention_with_output(
            q_3d,
            k_3d,
            v_3d,
            output_view,
            layer_name,
            kv_cache_dummy_dep=kv_dep,
        )
        return

    # ---- Normal inference: fused kernel + attention-only ----
    key_cache, value_cache = kv_cache.unbind(0)

    if layer_name not in _logged_fused_rope_layers:
        logger.info(
            "novita fused_rope_fp8_kvstore ACTIVE for layer %s (kv_cache=%s)",
            layer_name,
            kv_cache.shape,
        )
        _logged_fused_rope_layers.add(layer_name)

    # The fused kernel only writes the first slot_mapping.shape[0] rows of
    # q_output (one row per actual token).  However, we must pass the
    # full padded q_output to unified_attention_with_output so that
    # FlashAttention can slice it with its own num_actual_tokens — matching
    # the standard (non-fused) attention path.  Pre-slicing here would
    # create a tensor smaller than what the TMA descriptor expects in
    # CUDA-graph replay, causing a TMA out-of-bounds error.
    torch.ops._novita_C.fused_rope_fp8_kvstore(
        q,
        k,
        v,
        True,  # is_neox
        positions,
        rotary_dim,
        cos_sin_cache,
        q_output,
        q_scale,
        key_cache,
        value_cache,
        slot_mapping,
        k_scale,
        v_scale,
    )

    q_view = q_output.view(-1, num_heads_q, head_dim)
    output_view = output.view(-1, num_heads_q, head_dim)
    unified_attention_with_output(q_view, q_view, q_view, output_view, layer_name)


def novita_fused_rope_fp8_kvstore_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    output: torch.Tensor,
    q_output: torch.Tensor,
    layer_name: str,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    q_size: int,
    kv_size: int,
    rotary_dim: int,
) -> None:
    return


if _novita_available:
    register_novita_ops()
