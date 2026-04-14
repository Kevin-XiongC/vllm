# Integrating Fused CUDA Kernels into vLLM

This document describes the standard pattern for integrating large fused CUDA
kernels (e.g. fused RoPE + quantization + KV cache store, fused attention
variants) into vLLM's compilation and CUDA graph pipeline. It is based on the
patterns established by `novita_fused_attn` (glm4_moe) and
`novita_fused_rope_fp8_kvstore` (minimax_m2).

## Architecture Overview

Fused kernels that replace attention-related ops (RoPE, KV cache write,
attention itself) must interact with vLLM's `ForwardContext` to access
per-batch `slot_mapping` and per-layer `kv_cache`. These are not available as
compile-time tensor inputs — they are managed by the model runner and exposed
through `get_attention_context(layer_name)` at runtime.

The correct integration pattern is a **three-layer architecture**:

```
Layer 1: CUDA Kernel (.cu)          — pure GPU computation
Layer 2: Python Custom Op           — wraps kernel + attention context + attention call
Layer 3: Model _forward_fused()     — thin caller, just invokes the custom op
```

The custom op is registered as a **splitting op**, meaning torch.compile treats
it as a graph break point. This is compatible with CUDA graphs:

- **Piecewise**: compiled subgraphs before/after the op are captured as CUDA
  graph segments.
- **Full CUDA graph**: the entire forward (including GPU work from the splitting
  op) is captured in one full graph for replay.

The splitting op runs as Python during warmup/capture, giving it access to
`ForwardContext`. During CUDA graph replay, only the recorded GPU kernel calls
execute — no Python overhead.

## Layer 1: CUDA Kernel

Location: `csrc/novita/<kernel_name>.cu`

The kernel is a pure CUDA function. All inputs are raw tensor pointers and
scalar parameters. No vLLM runtime dependencies.

```
csrc/novita/my_fused_kernel.cu
├── myFusedKernel<head_dim, ...>()     // __global__ kernel
├── launchMyFusedKernel(...)           // host-side launcher with head_dim dispatch
└── my_fused_kernel(...)               // torch C++ entry point with NOVITA_CHECK_INPUT
```

Register the C++ entry point in `csrc/novita/torch_bindings.cpp`:

```cpp
void my_fused_kernel(torch::Tensor& q, ...);

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // ... existing ops ...
  ops.def("my_fused_kernel(...) -> ()");
  ops.impl("my_fused_kernel", torch::kCUDA, &my_fused_kernel);
}
```

Add the `.cu` file to `cmake/external_projects/novita_kernels.cmake`:

```cmake
set(NOVITA_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita/allreduce_fusion_wrapper.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita/my_fused_kernel.cu"       # <-- add
    "${CMAKE_CURRENT_SOURCE_DIR}/csrc/novita/torch_bindings.cpp")
```

Also update the standalone `build_novita.py` script (used for fast iteration on
remote test machines).

## Layer 2: Python Custom Op

Location: `vllm/novita_ops.py`

The custom op wraps three things into one opaque unit:

1. `get_attention_context(layer_name)` — to obtain `kv_cache` and `slot_mapping`
2. The C kernel call — `torch.ops._novita_C.my_fused_kernel(...)`
3. `unified_attention_with_output(...)` — the standard vLLM attention call

### Implementation function

```python
def my_fused_op(
    q: torch.Tensor,
    # ... all tensor args needed by the kernel and attention ...
    output: torch.Tensor,
    layer_name: str,
    # ... scalar args (int, float) ...
) -> None:
    from vllm.model_executor.layers.attention.attention import (
        get_attention_context,
        unified_attention_with_output,
        unified_kv_cache_update,
    )

    _, _, kv_cache, slot_mapping = get_attention_context(layer_name)

    if kv_cache.numel() == 0 or slot_mapping is None:
        # ---- Profiling fallback (kv_cache not allocated yet) ----
        # Use standard unfused ops so memory profiling can proceed.
        # Example: rotary_embedding + unified_kv_cache_update + attention
        torch.ops._C.rotary_embedding(positions, q, k, head_dim, cos_sin_cache, True)
        q_3d = q.view(-1, num_heads_q, head_dim)
        k_3d = k.view(-1, num_heads_k, head_dim)
        v_3d = v.view(-1, num_heads_v, head_dim)
        output_view = output.view(-1, num_heads_q, head_dim)
        kv_dep = unified_kv_cache_update(k_3d, v_3d, layer_name)
        unified_attention_with_output(
            q_3d, k_3d, v_3d, output_view, layer_name,
            kv_cache_dummy_dep=kv_dep)
        return

    # ---- Normal inference: fused kernel + attention ----
    key_cache, value_cache = kv_cache.unbind(0)

    torch.ops._novita_C.my_fused_kernel(
        q, k, v, ..., key_cache, value_cache, slot_mapping, ...)

    # Run attention (KV already written to cache by the kernel).
    # Pass q as dummy k/v since FlashAttention reads from the cache.
    q_view = q_fp8.view(-1, num_heads_q, head_dim)
    output_view = output.view(-1, num_heads_q, head_dim)
    unified_attention_with_output(q_view, q_view, q_view, output_view, layer_name)
```

### Fake implementation (for torch.compile tracing)

```python
def my_fused_op_fake(
    q: torch.Tensor,
    # ... same signature as above ...
) -> None:
    return
```

### Registration

In `register_novita_ops()`:

```python
direct_register_custom_op(
    op_name="my_fused_op",
    op_func=my_fused_op,
    mutates_args=["output"],
    fake_impl=my_fused_op_fake,
)
```

## Layer 3: Model Integration

Location: `vllm/model_executor/models/<model>.py`

### `__init__`: read the feature flag

```python
from vllm.config import get_current_vllm_config

vllm_config = get_current_vllm_config()
self._use_fused = (
    vllm_config.compilation_config.pass_config.enable_my_fusion
)
if self._use_fused:
    from vllm.novita_ops import is_novita_available
    assert is_novita_available(), "_novita_C not available"
    self._rotary_dim = ...  # any precomputed values needed
```

### `_forward_fused`: thin wrapper

The fused forward only calls the custom op + output projection. All context
lookup and kernel invocation is inside the custom op (Layer 2).

```python
def _forward_fused(self, positions, q, k, v):
    output = torch.empty(num_tokens, self.q_size, dtype=q.dtype, device=q.device)

    torch.ops.vllm.my_fused_op(
        q, k, v, positions, cos_sin_cache,
        self.attn._q_scale, self.attn._k_scale, self.attn._v_scale,
        output, self.attn.layer_name,
        self.num_heads, self.num_kv_heads, self.head_dim,
        self.q_size, self.kv_size, self._rotary_dim,
    )

    output, _ = self.o_proj(output)
    return output
```

### `forward`: branch on feature flag

```python
def forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    # ... any pre-processing (e.g. QK norm) ...
    if self._use_fused:
        return self._forward_fused(positions, q, k, v)
    # ... standard unfused path ...
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output
```

## Configuration

### PassConfig flag

In `vllm/config/compilation.py`, class `PassConfig`:

```python
enable_my_fusion: bool = False
"""Enable my fused kernel. Requires _novita_C to be built."""
```

### Splitting ops registration

In `CompilationConfig.__post_init__` (same file), inside the splitting_ops
setup block:

```python
if self.pass_config.enable_my_fusion:
    self.splitting_ops.append("vllm::my_fused_op")
```

This MUST be inside the existing `if self.splitting_ops is None:` block, at the
same level as the `unified_kv_cache_update` registration.

### User activation

```bash
vllm serve <model> \
  --compilation-config '{"pass_config": {"enable_my_fusion": true}}'
```

## How Splitting Ops Work with CUDA Graphs

```
┌────────────────────────────────────────────────────┐
│               FULL CUDA GRAPH (replay)             │
│                                                    │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────┐ │
│  │  Piecewise   │  │Splitting │  │  Piecewise   │ │
│  │  Subgraph 1  │→ │   Op     │→ │  Subgraph 2  │ │
│  │(qkv+norm etc)│  │(fused k  │  │  (o_proj)    │ │
│  │              │  │+attention)│  │              │ │
│  └──────────────┘  └──────────┘  └──────────────┘ │
│                                                    │
│  torch.compile      Python runs    torch.compile   │
│  captures this      during warmup, captures this   │
│  as CUDA graph      GPU calls are  as CUDA graph   │
│  segment            recorded       segment         │
│                                                    │
│  During full CUDA graph replay, ALL GPU work       │
│  (including splitting op's kernels) replays with   │
│  zero Python overhead.                             │
└────────────────────────────────────────────────────┘
```

- **Piecewise capture**: each compiled subgraph is captured as an independent
  CUDA graph segment.
- **Splitting op**: runs as Python during warmup. Has full access to
  `ForwardContext` (kv_cache, slot_mapping). The GPU kernel calls it makes are
  recorded.
- **Full capture**: `FULL_AND_PIECEWISE` mode captures the entire forward
  (including splitting op GPU work) into one full CUDA graph.
- **Replay**: the full graph replays all GPU work with zero Python overhead. The
  splitting op's Python code does NOT run during replay.

## KV Cache Layout

vLLM FlashAttention (FA3) uses NHD layout:

```
kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_dim]
                 │
                 ├── [0] = key_cache
                 └── [1] = value_cache

After unbind(0):
  key_cache: [num_blocks, block_size, num_kv_heads, head_dim]

Slot stride (between consecutive slots): num_kv_heads * head_dim
  NOT key_cache.stride(0) which is block_size * num_kv_heads * head_dim
```

## Checklist for Adding a New Fused Kernel

1. [ ] **CUDA kernel**: `csrc/novita/<name>_kernel.cu` with launch dispatcher
2. [ ] **Torch bindings**: forward declaration + `ops.def/impl` in `torch_bindings.cpp`
3. [ ] **CMake**: add `.cu` to `novita_kernels.cmake`
4. [ ] **Build script**: update `build_novita.py` sources list
5. [ ] **Python custom op**: implementation + fake_impl + registration in `novita_ops.py`
6. [ ] **PassConfig flag**: `enable_xxx: bool = False` in `compilation.py`
7. [ ] **Splitting ops**: append `"vllm::xxx"` in `CompilationConfig.__post_init__`
8. [ ] **Model**: `__init__` reads flag, `_forward_fused` calls custom op, `forward` branches
9. [ ] **Profiling fallback**: unfused path when `kv_cache.numel() == 0`
10. [ ] **Test eager mode**: `--enforce-eager` to verify kernel correctness without compile/graph
11. [ ] **Test CUDA graph mode**: clear compile cache, verify "ACTIVE" log appears during warmup
12. [ ] **Test E2E**: send a chat completion request and verify coherent output
