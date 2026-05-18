# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build only the novita .so extension (without full vllm rebuild).

Target architectures are resolved the same way as vllm's cmake build:
  1. TORCH_CUDA_ARCH_LIST env var (e.g. "8.0 8.9 9.0")
  2. Fall back to torch's default arch list for the installed torch version
"""

import os

import torch
import torch.utils.cpp_extension

torch.utils.cpp_extension._check_cuda_version = lambda *a, **kw: None
from setuptools import setup  # noqa: E402
from torch.utils.cpp_extension import BuildExtension, CUDAExtension  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))


def _gencode_flags() -> list[str]:
    """Return nvcc -gencode flags matching TORCH_CUDA_ARCH_LIST / cmake."""
    arch_list_env = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if arch_list_env:
        archs = arch_list_env.replace(",", " ").split()
    else:
        # Mirror what torch's cmake uses when TORCH_CUDA_ARCH_LIST is unset.
        archs = torch.cuda.get_arch_list()

    flags = []
    for arch in archs:
        arch = arch.strip()
        if not arch:
            continue
        # Normalise "8.0" / "8.0+PTX" → "80"
        ptx = arch.endswith("+PTX")
        base = arch.replace("+PTX", "").replace(".", "")
        flags.append(f"-gencode=arch=compute_{base},code=sm_{base}")
        if ptx:
            flags.append(f"-gencode=arch=compute_{base},code=compute_{base}")
    return flags


setup(
    name="_novita_C",
    ext_modules=[
        CUDAExtension(
            name="vllm._novita_C",
            sources=[
                os.path.join(ROOT, "csrc/novita/fused_rope_fp8_kvstore_kernel.cu"),
                os.path.join(ROOT, "csrc/novita/kimi_k2_moe_gate.cu"),
                os.path.join(ROOT, "csrc/novita/torch_bindings.cpp"),
            ],
            include_dirs=[
                os.path.join(ROOT, "csrc"),
            ],
            extra_compile_args={
                "cxx": ["-O2"],
                "nvcc": [
                    "-O2",
                    "--use_fast_math",
                    *_gencode_flags(),
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
