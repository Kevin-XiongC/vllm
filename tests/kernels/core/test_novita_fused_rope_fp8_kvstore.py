# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _has_novita_kernel() -> bool:
    if not current_platform.is_cuda():
        return False
    try:
        import vllm._novita_C  # noqa: F401

        return hasattr(torch.ops, "_novita_C") and hasattr(
            torch.ops._novita_C, "fused_rope_fp8_kvstore"
        )
    except ImportError:
        return False


def _make_cos_sin_cache(
    max_pos: int, rotary_dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    inv_freq = 1.0 / (
        10000.0
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=torch.float32)
            / rotary_dim
        )
    )
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat([cos, sin], dim=-1).to(dtype)


def _apply_rope_ref(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
    is_neox: bool,
) -> torch.Tensor:
    # x: [T, H, D], cos/sin: [T, rotary_dim/2]
    out = x.float().clone()
    t, h, d = out.shape
    embed = rotary_dim // 2
    c = cos[:, None, :].float()  # [T, 1, embed]
    s = sin[:, None, :].float()

    if is_neox:
        x1 = out[:, :, :embed].clone()
        x2 = out[:, :, embed : 2 * embed].clone()
        out[:, :, :embed] = x1 * c - x2 * s
        out[:, :, embed : 2 * embed] = x2 * c + x1 * s
    else:
        for i in range(embed):
            d0, d1 = 2 * i, 2 * i + 1
            v0 = out[:, :, d0].clone()
            v1 = out[:, :, d1].clone()
            out[:, :, d0] = v0 * c[:, :, i] - v1 * s[:, :, i]
            out[:, :, d1] = v0 * s[:, :, i] + v1 * c[:, :, i]
    return out.to(x.dtype)


def _to_fp8_bytes(x: torch.Tensor, scale: float) -> torch.Tensor:
    return (x.float() / scale).to(torch.float8_e4m3fn).view(torch.uint8)


def _reference_impl(
    q: torch.Tensor,  # [T, nq * hd]
    k: torch.Tensor,  # [T, nk * hd]
    v: torch.Tensor,  # [T, nk * hd]
    positions: torch.Tensor,  # [T]
    cos_sin_cache: torch.Tensor,  # [max_pos, rotary_dim]
    slot_mapping: torch.Tensor,  # [T]
    q_scale: float,
    k_scale: float,
    v_scale: float,
    num_heads_q: int,
    num_heads_k: int,
    head_dim: int,
    block_size: int,
    num_blocks: int,
    rotary_dim: int,
    is_neox: bool,
):
    t = q.shape[0]
    q_h = q.view(t, num_heads_q, head_dim)
    k_h = k.view(t, num_heads_k, head_dim)
    v_h = v.view(t, num_heads_k, head_dim)

    embed = rotary_dim // 2
    cos = cos_sin_cache[positions, :embed]
    sin = cos_sin_cache[positions, embed:rotary_dim]
    q_rope = _apply_rope_ref(q_h, cos, sin, rotary_dim, is_neox)
    k_rope = _apply_rope_ref(k_h, cos, sin, rotary_dim, is_neox)

    q_out_bytes = _to_fp8_bytes(q_rope.reshape(t, num_heads_q * head_dim), q_scale)
    k_cache_ref = torch.zeros(
        num_blocks,
        block_size,
        num_heads_k,
        head_dim,
        dtype=torch.uint8,
        device=q.device,
    )
    v_cache_ref = torch.zeros_like(k_cache_ref)

    for i in range(t):
        slot = int(slot_mapping[i].item())
        if slot < 0:
            continue
        blk = slot // block_size
        off = slot % block_size
        k_cache_ref[blk, off] = _to_fp8_bytes(k_rope[i], k_scale)
        v_cache_ref[blk, off] = _to_fp8_bytes(v_h[i], v_scale)

    return q_out_bytes, k_cache_ref, v_cache_ref


@pytest.mark.skipif(
    not _has_novita_kernel(),
    reason="novita fused_rope_fp8_kvstore requires CUDA + vllm._novita_C",
)
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize(
    "head_dim,rotary_dim",
    [
        (64, 64),  # full rotary
        (128, 128),  # full rotary
        (128, 64),  # partial rotary (MiniMax M2: head_dim=128, rotary_dim=64)
    ],
)
@torch.inference_mode()
def test_novita_fused_rope_fp8_kvstore_matches_reference(
    default_vllm_config,  # noqa: ARG001
    is_neox: bool,
    head_dim: int,
    rotary_dim: int,
):
    set_random_seed(13)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_tokens = 8
    num_heads_q = 8
    num_heads_k = 2
    block_size = 16
    num_blocks = (num_tokens + block_size - 1) // block_size
    max_pos = 4096

    q = torch.randn(num_tokens, num_heads_q * head_dim, dtype=dtype, device=device)
    k = torch.randn(num_tokens, num_heads_k * head_dim, dtype=dtype, device=device)
    v = torch.randn(num_tokens, num_heads_k * head_dim, dtype=dtype, device=device)
    positions = torch.randint(
        0, max_pos, (num_tokens,), dtype=torch.long, device=device
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)
    slot_mapping[-1] = -1

    cos_sin_cache = _make_cos_sin_cache(max_pos, rotary_dim, dtype, device)

    q_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    k_scale = torch.tensor([0.75], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.25], dtype=torch.float32, device=device)

    q_output = torch.zeros(
        num_tokens + 3, num_heads_q * head_dim, dtype=torch.uint8, device=device
    )
    k_cache = torch.zeros(
        num_blocks, block_size, num_heads_k, head_dim, dtype=torch.uint8, device=device
    )
    v_cache = torch.zeros_like(k_cache)

    q_ref, k_ref, v_ref = _reference_impl(
        q=q,
        k=k,
        v=v,
        positions=positions,
        cos_sin_cache=cos_sin_cache,
        slot_mapping=slot_mapping,
        q_scale=float(q_scale.item()),
        k_scale=float(k_scale.item()),
        v_scale=float(v_scale.item()),
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
        rotary_dim=rotary_dim,
        is_neox=is_neox,
    )

    torch.ops._novita_C.fused_rope_fp8_kvstore(
        q,
        k,
        v,
        is_neox,
        positions,
        rotary_dim,
        cos_sin_cache,
        q_output,
        q_scale,
        k_cache,
        v_cache,
        slot_mapping,
        k_scale,
        v_scale,
    )

    q_test_f = q_output[:num_tokens].view(torch.float8_e4m3fn).float()
    q_ref_f = q_ref.view(torch.float8_e4m3fn).float()
    k_test_f = k_cache.view(torch.float8_e4m3fn).float()
    k_ref_f = k_ref.view(torch.float8_e4m3fn).float()
    v_test_f = v_cache.view(torch.float8_e4m3fn).float()
    v_ref_f = v_ref.view(torch.float8_e4m3fn).float()

    torch.testing.assert_close(q_test_f, q_ref_f, atol=2.0, rtol=0.1)
    torch.testing.assert_close(k_test_f, k_ref_f, atol=2.0, rtol=0.1)
    torch.testing.assert_close(v_test_f, v_ref_f, atol=2.0, rtol=0.1)

    assert torch.count_nonzero(q_output[num_tokens:]) == 0


@pytest.mark.skipif(
    not _has_novita_kernel(),
    reason="novita fused_rope_fp8_kvstore requires CUDA + vllm._novita_C",
)
@pytest.mark.parametrize("is_neox", [True, False])
@pytest.mark.parametrize(
    "head_dim,rotary_dim",
    [
        (64, 64),
        (128, 128),
        (128, 64),
    ],
)
@torch.inference_mode()
def test_cudagraph_padding(
    default_vllm_config,  # noqa: ARG001
    is_neox: bool,
    head_dim: int,
    rotary_dim: int,
):
    """Simulate CUDA graph padding: q/k/v/positions are padded beyond the
    real token count.  Padded positions are 0 and padded slot_mapping is -1.
    """
    set_random_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    num_real_tokens = 5
    num_padded_tokens = 8
    num_heads_q = 8
    num_heads_k = 2
    block_size = 16
    num_blocks = (num_real_tokens + block_size - 1) // block_size
    max_pos = 4096

    q = torch.randn(
        num_padded_tokens, num_heads_q * head_dim, dtype=dtype, device=device
    )
    k = torch.randn(
        num_padded_tokens, num_heads_k * head_dim, dtype=dtype, device=device
    )
    v = torch.randn(
        num_padded_tokens, num_heads_k * head_dim, dtype=dtype, device=device
    )

    positions = torch.zeros(num_padded_tokens, dtype=torch.long, device=device)
    positions[:num_real_tokens] = torch.randint(
        0, max_pos, (num_real_tokens,), dtype=torch.long, device=device
    )

    slot_mapping = torch.full((num_padded_tokens,), -1, dtype=torch.long, device=device)
    slot_mapping[:num_real_tokens] = torch.arange(
        num_real_tokens, dtype=torch.long, device=device
    )

    cos_sin_cache = _make_cos_sin_cache(max_pos, rotary_dim, dtype, device)

    q_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    k_scale = torch.tensor([0.75], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.25], dtype=torch.float32, device=device)

    q_output = torch.empty(
        num_padded_tokens, num_heads_q * head_dim, dtype=torch.uint8, device=device
    )
    k_cache = torch.zeros(
        num_blocks, block_size, num_heads_k, head_dim, dtype=torch.uint8, device=device
    )
    v_cache = torch.zeros_like(k_cache)

    q_ref, k_ref, v_ref = _reference_impl(
        q=q[:num_real_tokens],
        k=k[:num_real_tokens],
        v=v[:num_real_tokens],
        positions=positions[:num_real_tokens],
        cos_sin_cache=cos_sin_cache,
        slot_mapping=slot_mapping[:num_real_tokens],
        q_scale=float(q_scale.item()),
        k_scale=float(k_scale.item()),
        v_scale=float(v_scale.item()),
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_dim=head_dim,
        block_size=block_size,
        num_blocks=num_blocks,
        rotary_dim=rotary_dim,
        is_neox=is_neox,
    )

    torch.ops._novita_C.fused_rope_fp8_kvstore(
        q,
        k,
        v,
        is_neox,
        positions,
        rotary_dim,
        cos_sin_cache,
        q_output,
        q_scale,
        k_cache,
        v_cache,
        slot_mapping,
        k_scale,
        v_scale,
    )

    q_test_f = q_output[:num_real_tokens].view(torch.float8_e4m3fn).float()
    q_ref_f = q_ref.view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(q_test_f, q_ref_f, atol=2.0, rtol=0.1)

    k_test_f = k_cache.view(torch.float8_e4m3fn).float()
    k_ref_f = k_ref.view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(k_test_f, k_ref_f, atol=2.0, rtol=0.1)

    v_test_f = v_cache.view(torch.float8_e4m3fn).float()
    v_ref_f = v_ref.view(torch.float8_e4m3fn).float()
    torch.testing.assert_close(v_test_f, v_ref_f, atol=2.0, rtol=0.1)

    padded_q = q_output[num_real_tokens:].view(torch.float8_e4m3fn).float()
    assert padded_q.isfinite().all(), "padded q_output contains non-finite values"
