# Novita fused kernels:
#   RoPE + FP8 cast + KV cache store (+ optional QK Norm variant)

message(STATUS "Building Novita kernels for archs: ${CUDA_ARCHS}")

set(NOVITA_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita/fused_rope_fp8_kvstore_kernel.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita/torch_bindings.cpp")

set(NOVITA_GPU_FLAGS ${VLLM_GPU_FLAGS})
list(APPEND NOVITA_GPU_FLAGS "--use_fast_math")

set_gencode_flags_for_srcs(
  SRCS "${NOVITA_SRCS}"
  CUDA_ARCHS "${CUDA_ARCHS}")

define_extension_target(
    _novita_C
    DESTINATION vllm
    LANGUAGE ${VLLM_GPU_LANG}
    SOURCES ${NOVITA_SRCS}
    COMPILE_FLAGS ${NOVITA_GPU_FLAGS}
    ARCHITECTURES ${VLLM_GPU_ARCHES}
    INCLUDE_DIRECTORIES
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita
    USE_SABI 3
    WITH_SOABI)
