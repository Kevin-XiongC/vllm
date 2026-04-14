# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Novita fused kernels (vllm._novita_C):
  AllReduce + RMSNorm + Residual Add (+ optional FP8/FP4 quantization)

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
        op_name="novita_fused_allreduce_norm",
        op_func=call_novita_fused_allreduce_norm,
        mutates_args=[
            "allreduce_in",
            "residual",
            "norm_out",
            "quant_out",
            "scale_out",
        ],
        fake_impl=call_novita_fused_allreduce_norm_fake,
    )

    direct_register_custom_op(
        op_name="novita_fused_rope_fp8_kvstore",
        op_func=novita_fused_rope_fp8_kvstore,
        mutates_args=["output"],
        fake_impl=novita_fused_rope_fp8_kvstore_fake,
    )



# ---------------------------------------------------------------------------
# Custom op: novita_fused_allreduce_norm
#
# Drop-in replacement for flashinfer_trtllm_fused_allreduce_norm.
# Same signature so the graph pattern matching pass can swap them.
# Uses flashinfer workspace for IPC memory, but our optimized CUDA kernel.
# ---------------------------------------------------------------------------

_NOVITA_ONE_SHOT_MAX_SIZES_MB: dict[int, dict[int, float]] = {
    90: {2: 32, 4: 2, 8: 0.5},
    100: {2: 32, 4: 4, 8: 1},
}

MiB = 1024 * 1024

# Token-count range where novita kernel is slower than flashinfer.
# Benchmarked on H200 / tp=8 / hidden=7168:
#   tokens ≤ 128  → novita wins  (up to 2.9×)
#   tokens 160-768 → flashinfer wins
#   tokens ≥ 896  → novita wins again
NOVITA_AR_FALLBACK_MIN_TOKENS = 160
NOVITA_AR_FALLBACK_MAX_TOKENS = 768

_logged_fallback = False


def _flashinfer_allreduce_fallback(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    workspace: object,
    pattern_code: int,
    launch_with_pdl: bool,
    fp32_acc: bool,
    use_oneshot: bool,
    norm_out: torch.Tensor | None,
    quant_out: torch.Tensor | None,
    scale_out: torch.Tensor | None,
    scale_factor: torch.Tensor | None,
) -> None:
    """Call the native flashinfer allreduce_fusion kernel directly."""
    import flashinfer.comm as fi_comm

    if norm_out is None:
        actual_norm_out = allreduce_in
        residual_out = residual
    else:
        actual_norm_out = norm_out
        residual_out = allreduce_in

    layout_code = None
    if workspace.backend == "trtllm":  # type: ignore[union-attr]
        layout_code = fi_comm.QuantizationSFLayout.SWIZZLED_128x4

    fi_comm.allreduce_fusion(
        input=allreduce_in,
        workspace=workspace,
        pattern=pattern_code,
        launch_with_pdl=launch_with_pdl,
        output=None,
        residual_out=residual_out,
        norm_out=actual_norm_out,
        quant_out=quant_out,
        scale_out=scale_out,
        residual_in=residual,
        rms_gamma=rms_gamma,
        rms_eps=rms_eps,
        scale_factor=scale_factor,
        layout_code=layout_code,
        use_oneshot=use_oneshot,
        fp32_acc=fp32_acc,
    )


def call_novita_fused_allreduce_norm(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    world_size: int,
    launch_with_pdl: bool,
    fp32_acc: bool,
    max_token_num: int,
    pattern_code: int,
    norm_out: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    scale_factor: torch.Tensor | None = None,
) -> None:
    """Fused allreduce + RMSNorm + residual add using novita kernel.

    Mirrors the flashinfer_trtllm_fused_allreduce_norm signature exactly.
    Uses flashinfer workspace for IPC shared memory management.

    When token count falls in the novita-kernel disadvantage zone
    (NOVITA_AR_FALLBACK_MIN_TOKENS .. NOVITA_AR_FALLBACK_MAX_TOKENS),
    falls back to the native flashinfer kernel automatically.
    """
    from vllm.distributed import get_tp_group
    from vllm.distributed.device_communicators.flashinfer_all_reduce import (
        get_fi_ar_quant_workspace,
        get_fi_ar_workspace,
    )
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
    )

    num_tokens, hidden_size = allreduce_in.shape
    element_size = allreduce_in.element_size()
    current_tensor_size = num_tokens * hidden_size * element_size

    curr_device = current_platform.get_device_capability()
    device_capability = (
        curr_device.to_int() if curr_device is not None else None
    )
    max_one_shot_size = _NOVITA_ONE_SHOT_MAX_SIZES_MB.get(
        device_capability, {}  # type: ignore[arg-type]
    ).get(world_size, None)
    use_oneshot = (
        max_one_shot_size is None
        or current_tensor_size <= max_one_shot_size * MiB
    )

    rank = get_tensor_model_parallel_rank()
    group = get_tp_group().device_group
    workspace_kwargs = dict(
        world_size=world_size,
        rank=rank,
        max_token_num=max_token_num,
        hidden_dim=hidden_size,
        dtype=allreduce_in.dtype,
        group=group,
    )

    try:
        import flashinfer.comm as _fi_comm

        ar_patterns = _fi_comm.AllReduceFusionPattern
        is_quant = pattern_code in (
            ar_patterns.kARResidualRMSNormFP8Quant,
            ar_patterns.kARResidualRMSNormFP4Quant,
        )
    except ImportError:
        is_quant = False

    workspace = (
        get_fi_ar_quant_workspace(**workspace_kwargs)
        if is_quant
        else get_fi_ar_workspace(**workspace_kwargs)
    )
    assert workspace is not None, (
        "Flashinfer workspace must be initialized for novita allreduce fusion"
    )

    # --- token-count based dispatch ---
    if NOVITA_AR_FALLBACK_MIN_TOKENS <= num_tokens <= NOVITA_AR_FALLBACK_MAX_TOKENS:
        global _logged_fallback
        if not _logged_fallback:
            logger.info(
                "novita AR fallback to flashinfer for %d tokens "
                "(disadvantage zone %d-%d)",
                num_tokens,
                NOVITA_AR_FALLBACK_MIN_TOKENS,
                NOVITA_AR_FALLBACK_MAX_TOKENS,
            )
            _logged_fallback = True
        _flashinfer_allreduce_fallback(
            allreduce_in=allreduce_in,
            residual=residual,
            rms_gamma=rms_gamma,
            rms_eps=rms_eps,
            workspace=workspace,
            pattern_code=pattern_code,
            launch_with_pdl=launch_with_pdl,
            fp32_acc=fp32_acc,
            use_oneshot=use_oneshot,
            norm_out=norm_out,
            quant_out=quant_out,
            scale_out=scale_out,
            scale_factor=scale_factor,
        )
        return

    # --- novita fused kernel path ---
    if norm_out is None:
        actual_norm_out = allreduce_in
        residual_out = residual
    else:
        actual_norm_out = norm_out
        residual_out = allreduce_in

    def _flatten(t: torch.Tensor) -> torch.Tensor:
        assert t.is_contiguous(), f"Tensor must be contiguous, got {t.shape}"
        return t.view(-1)

    empty = torch.empty(0, device=allreduce_in.device, dtype=torch.float32)

    torch.ops._novita_C.allreduce_fusion(
        _flatten(allreduce_in),
        _flatten(residual),
        _flatten(residual_out),
        workspace.workspace_tensor,
        world_size,
        rank,
        num_tokens,
        hidden_size,
        rms_eps,
        rms_gamma,
        launch_with_pdl,
        fp32_acc,
        use_oneshot,
        pattern_code,
        _flatten(actual_norm_out) if actual_norm_out.numel() > 0 else empty,
        _flatten(quant_out) if quant_out is not None else empty,
        _flatten(scale_out) if scale_out is not None else empty,
        scale_factor if scale_factor is not None else empty,
    )


def call_novita_fused_allreduce_norm_fake(
    allreduce_in: torch.Tensor,
    residual: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    world_size: int,
    launch_with_pdl: bool,
    fp32_acc: bool,
    max_token_num: int,
    pattern_code: int,
    norm_out: torch.Tensor | None = None,
    quant_out: torch.Tensor | None = None,
    scale_out: torch.Tensor | None = None,
    scale_factor: torch.Tensor | None = None,
) -> None:
    pass


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
    """
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
        unified_attention_with_output,
        unified_kv_cache_update,
    )

    num_tokens = q.shape[0]
    num_heads_v = num_heads_k
    _, _, kv_cache, slot_mapping = get_attention_context(layer_name)

    if kv_cache.numel() == 0 or slot_mapping is None:
        # ---- Profiling fallback: unfused ops ----
        torch.ops._C.rotary_embedding(positions, q, k, head_dim,
                                      cos_sin_cache, True)

        q_3d = q.view(-1, num_heads_q, head_dim)
        k_3d = k.view(-1, num_heads_k, head_dim)
        v_3d = v.view(-1, num_heads_v, head_dim)
        output_view = output.view(-1, num_heads_q, head_dim)

        kv_dep = unified_kv_cache_update(k_3d, v_3d, layer_name)
        unified_attention_with_output(
            q_3d, k_3d, v_3d, output_view, layer_name,
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

    q_output = torch.empty(
        num_tokens, q_size, dtype=torch.float8_e4m3fn, device=q.device
    )

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
    unified_attention_with_output(q_view, q_view, q_view, output_view,
                                 layer_name)


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
    layer_name: str,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    q_size: int,
    kv_size: int,
    rotary_dim: int,
) -> None:
    return


# ---------------------------------------------------------------------------
# Fused QK RMS Norm with TP variance all-reduce
# ---------------------------------------------------------------------------

_qk_norm_epoch = 0


def novita_fused_qk_rmsnorm_tp(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    workspace_ptrs: torch.Tensor,
    world_size: int,
    rank: int,
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused QK RMS Norm with TP variance all-reduce via NVLink.

    Replaces MiniMaxText01RMSNormTP.forward_qk with a single CUDA kernel
    that fuses cast + variance + cross-GPU exchange + normalize.

    Args:
        q: [num_tokens, D_q_local] input query tensor (bf16/fp16)
        k: [num_tokens, D_k_local] input key tensor (bf16/fp16)
        q_weight: [D_q_local] RMS norm weight for q
        k_weight: [D_k_local] RMS norm weight for k
        q_eps: epsilon for q RMS norm
        k_eps: epsilon for k RMS norm
        workspace_ptrs: [world_size] int64 tensor of NVLink-accessible buffer pointers
        world_size: tensor parallel world size
        rank: current rank
        max_tokens: maximum tokens the workspace was allocated for

    Returns:
        (q_out, k_out) normalized tensors with same shape and dtype as inputs
    """
    global _qk_norm_epoch
    _qk_norm_epoch += 1

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    torch.ops._novita_C.qk_rmsnorm_tp(
        q, k, q_out, k_out,
        q_weight, k_weight,
        q_eps, k_eps,
        workspace_ptrs,
        world_size, rank,
        max_tokens, _qk_norm_epoch,
    )

    return q_out, k_out


def novita_clear_qk_norm_workspace(
    workspace_ptrs: torch.Tensor,
    world_size: int,
    rank: int,
    max_tokens: int,
) -> None:
    """Clear the QK norm workspace (call once before first use)."""
    torch.ops._novita_C.clear_qk_norm_workspace(
        workspace_ptrs, world_size, rank, max_tokens,
    )


if _novita_available:
    register_novita_ops()
