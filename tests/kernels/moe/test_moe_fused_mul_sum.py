# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import (
    moe_fused_mul_sum,
)
from vllm.platforms import current_platform


@pytest.mark.skipif(
    not current_platform.is_cuda(), reason="This test is skipped on non-CUDA platform."
)
def test_moe_fused_mul_sum_ignores_negative_topk_ids():
    inputs = torch.tensor(
        [
            [[1.0, 2.0], [10.0, 20.0], [100.0, 200.0]],
            [[3.0, 4.0], [30.0, 40.0], [300.0, 400.0]],
        ],
        device="cuda",
        dtype=torch.float16,
    )
    topk_weights = torch.tensor(
        [[0.5, 0.25, 1.0], [1.0, 0.5, 0.25]],
        device="cuda",
        dtype=torch.float16,
    )
    topk_ids = torch.tensor(
        [[0, -1, 2], [-1, 1, 3]],
        device="cuda",
        dtype=torch.int32,
    )
    expert_map = torch.tensor([0, -1, 1, 2], device="cuda", dtype=torch.int32)

    output = moe_fused_mul_sum(
        inputs=inputs,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        expert_map=expert_map,
    )

    expected = torch.tensor(
        [
            [100.5, 201.0],
            [75.0, 100.0],
        ],
        device="cuda",
        dtype=torch.float16,
    )
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
