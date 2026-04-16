"""Build only the novita .so extension (without full vllm rebuild)."""
import os
import torch
import torch.utils.cpp_extension
torch.utils.cpp_extension._check_cuda_version = lambda *a, **kw: None
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
from setuptools import setup

ROOT = os.path.dirname(os.path.abspath(__file__))

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    _arch = f"{major}{minor}"
else:
    _arch = "90"

setup(
    name="_novita_C",
    ext_modules=[
        CUDAExtension(
            name="vllm._novita_C",
            sources=[
                os.path.join(ROOT, "csrc/novita/fused_rope_fp8_kvstore_kernel.cu"),
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
                    f"-gencode=arch=compute_{_arch},code=sm_{_arch}",
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
