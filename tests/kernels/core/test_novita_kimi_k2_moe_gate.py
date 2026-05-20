# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed


def _has_novita_kimi_gate() -> bool:
    if not current_platform.is_cuda():
        return False
    try:
        import vllm._novita_C  # noqa: F401

        return hasattr(torch.ops, "_novita_C") and hasattr(
            torch.ops._novita_C, "kimi_k2_moe_fused_gate"
        )
    except ImportError:
        return False


def _reference_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    apply_routed_scaling_factor_on_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.sigmoid(input.float())
    _, topk_ids = torch.topk(scores + bias.float(), topk, dim=-1)
    topk_weights = scores.gather(1, topk_ids)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        if apply_routed_scaling_factor_on_output:
            topk_weights = topk_weights * routed_scaling_factor

    return topk_weights, topk_ids.to(torch.int32)


@pytest.mark.skipif(
    not _has_novita_kimi_gate(),
    reason="novita Kimi K2 MoE gate requires CUDA + vllm._novita_C",
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("num_tokens", [17, 777])
@pytest.mark.parametrize(
    ("renormalize", "apply_routed_scaling_factor_on_output"),
    [(False, False), (True, False), (True, True)],
)
@torch.inference_mode()
def test_novita_kimi_k2_moe_fused_gate_matches_reference(
    default_vllm_config,  # noqa: ARG001
    dtype: torch.dtype,
    num_tokens: int,
    renormalize: bool,
    apply_routed_scaling_factor_on_output: bool,
):
    set_random_seed(17)
    device = torch.device("cuda")
    num_experts = 384
    topk = 8
    routed_scaling_factor = 2.827

    input = torch.randn(num_tokens, num_experts, dtype=dtype, device=device)
    bias = torch.randn(num_experts, dtype=torch.float32, device=device) * 0.01

    actual_weights, actual_ids = torch.ops._novita_C.kimi_k2_moe_fused_gate(
        input,
        bias,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )
    expected_weights, expected_ids = _reference_gate(
        input,
        bias,
        topk,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )

    assert actual_weights.dtype == torch.float32
    assert actual_ids.dtype == torch.int32
    torch.testing.assert_close(actual_weights, expected_weights, atol=1e-6, rtol=1e-6)
    assert torch.equal(actual_ids, expected_ids)
