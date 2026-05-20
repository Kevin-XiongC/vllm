# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for NIXL speculative-draft KV filtering.

NIXL should transfer target-model KV between prefill and decode. Under
speculative decoding, EAGLE/MTP draft KV groups and appended draft-layer
caches stay local to the decode-side drafter. These tests cover the
shared target-group selector plus scheduler and worker filters.

These tests avoid importing the heavier ``test_nixl_connector`` module
(which pulls in optional deps like ``ray`` at collection time) so the
filter logic can be exercised in a minimal CPU environment.
"""

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.scheduler import (
    NixlConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_nixl_target_kv_group_indices,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
)


def _make_worker(target_layer_names: list[str]) -> NixlConnectorWorker:
    """Build a minimal NixlConnectorWorker stub for filter testing.

    Bypasses the heavy constructor: only ``self._layer_specs`` (the
    canonical target-layer name set used by the filter) needs to be
    populated.
    """
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    # _layer_specs maps target-layer names to per-layer KV cache specs in
    # production; the filter only checks key membership, so a value of None
    # is sufficient here.
    worker._layer_specs = {name: None for name in target_layer_names}
    worker.vllm_config = None
    return worker


def _full_spec() -> FullAttentionSpec:
    return FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=16,
        dtype=torch.float16,
    )


def test_filter_drops_draft_layers_pp_rank_zero():
    """PP rank 0 with target layers 0..3 and draft layers 4..5.

    Without filtering, the draft layers (names ``model.layers.4..5``) would
    be registered with NIXL, polluting handshake metadata. The filter must
    keep only entries whose names appear in ``_layer_specs``.
    """
    target_names = [f"model.layers.{i}.attn" for i in range(4)]
    worker = _make_worker(target_layer_names=target_names)
    kv_caches = {
        **{name: torch.zeros(1) for name in target_names},
        "model.layers.4.attn": torch.zeros(1),  # draft
        "model.layers.5.attn": torch.zeros(1),  # draft
    }
    filtered = worker._filter_kv_caches_to_target_layers(kv_caches)
    assert set(filtered.keys()) == set(target_names)


def test_filter_keeps_target_layers_under_pipeline_parallel_rank_one():
    """PP rank > 0: target absolute indices do NOT start at 0.

    Regression guard for the earlier per-rank-count arithmetic, which
    misclassified target layers on non-zero PP ranks as draft. Here the
    rank-local target set is layers 16..31 (target model has 32 layers
    total, PP=2); draft tacks on layers 32+.
    """
    target_names = [f"model.layers.{i}.attn" for i in range(16, 32)]
    worker = _make_worker(target_layer_names=target_names)
    kv_caches = {
        **{name: torch.zeros(1) for name in target_names},
        "model.layers.32.attn": torch.zeros(1),  # draft
        "model.layers.33.attn": torch.zeros(1),  # draft
    }
    filtered = worker._filter_kv_caches_to_target_layers(kv_caches)
    assert set(filtered.keys()) == set(target_names)


def test_filter_drops_draft_layers_with_unconventional_naming():
    """Draft model uses a distinct prefix, not contiguous numbering.

    Some Eagle/MTP variants register draft layers under names that do not
    continue the target's integer sequence (e.g. ``draft.layers.0``).
    Filtering by ``_layer_specs`` membership is correct here; a filter that
    parsed integer indices would silently accept these.
    """
    target_names = [f"model.layers.{i}.attn" for i in range(4)]
    worker = _make_worker(target_layer_names=target_names)
    kv_caches = {
        **{name: torch.zeros(1) for name in target_names},
        "draft.layers.0.attn": torch.zeros(1),
        "draft.layers.1.attn": torch.zeros(1),
    }
    filtered = worker._filter_kv_caches_to_target_layers(kv_caches)
    assert set(filtered.keys()) == set(target_names)


def test_filter_noop_when_only_target_layers_present():
    """Filter is a no-op when ``kv_caches`` already matches ``_layer_specs``.

    This is the steady-state main-line case (no spec decode); the filter
    must not lose anything the caller passed in.
    """
    target_names = [f"model.layers.{i}.attn" for i in range(4)]
    worker = _make_worker(target_layer_names=target_names)
    kv_caches = {name: torch.zeros(1) for name in target_names}
    filtered = worker._filter_kv_caches_to_target_layers(kv_caches)
    assert filtered.keys() == kv_caches.keys()


def test_filter_drops_arbitrary_unknown_keys_uniformly():
    """Any cache key absent from ``_layer_specs`` is dropped.

    The filter applies the same rule whether or not speculative decoding
    is configured, so an accidental non-target tensor (auxiliary buffer,
    synthetic key) cannot silently slip through in one mode while being
    rejected in another.
    """
    target_names = [f"model.layers.{i}.attn" for i in range(2)]
    worker = _make_worker(target_layer_names=target_names)
    kv_caches = {
        target_names[0]: torch.zeros(1),
        "auxiliary_buffer": torch.zeros(1),
        "model.layers.42.attn": torch.zeros(1),  # draft
    }
    filtered = worker._filter_kv_caches_to_target_layers(kv_caches)
    assert set(filtered.keys()) == {target_names[0]}


def test_filter_returns_empty_when_no_target_keys_match():
    """Defensive: completely-foreign input yields an empty dict, not a crash.

    Production callers should never hit this path, but a guard test
    documents the contract: the filter never raises on an unrecognized
    key, it just drops it.
    """
    worker = _make_worker(target_layer_names=["model.layers.0.attn"])
    kv_caches = {
        "model.layers.42.attn": torch.zeros(1),
        "draft.layers.0.attn": torch.zeros(1),
    }
    filtered = worker._filter_kv_caches_to_target_layers(kv_caches)
    assert filtered == {}


def test_target_group_indices_exclude_eagle_group():
    groups = [
        KVCacheGroupSpec(["model.layers.0.attn"], _full_spec()),
        KVCacheGroupSpec(["model.layers.1.attn"], _full_spec(), is_eagle_group=True),
    ]

    assert get_nixl_target_kv_group_indices(groups) == (0,)


def test_filter_drops_appended_draft_layer_even_if_layer_specs_contains_it():
    worker = _make_worker(
        target_layer_names=[
            "model.layers.0.self_attn.attn",
            "model.layers.1.self_attn.attn",
            "model.layers.2.self_attn.attn",
        ]
    )

    class _SpecConfig:
        def uses_draft_model(self):
            return True

    class _VllmConfig:
        speculative_config = _SpecConfig()

    class _ModelConfig:
        def get_total_num_hidden_layers(self):
            return 2

    worker.vllm_config = _VllmConfig()
    worker.model_config = _ModelConfig()

    kv_caches = {
        "model.layers.0.self_attn.attn": torch.zeros(1),
        "model.layers.1.self_attn.attn": torch.zeros(1),
        "model.layers.2.self_attn.attn": torch.zeros(1),
    }

    filtered = worker._filter_kv_caches_to_target_layers(kv_caches)

    assert set(filtered) == {
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    }


def test_filter_prefers_spec_target_layer_count_for_appended_draft_layer():
    worker = _make_worker(
        target_layer_names=[
            "model.layers.60.self_attn.attn",
            "model.layers.61.self_attn.attn",
        ]
    )

    class _HFConfig:
        num_hidden_layers = 61

    class _TargetModelConfig:
        hf_text_config = _HFConfig()

    class _SpecConfig:
        target_model_config = _TargetModelConfig()

        def uses_draft_model(self):
            return True

    class _VllmConfig:
        speculative_config = _SpecConfig()

    class _ModelConfig:
        def get_total_num_hidden_layers(self):
            return 62

    worker.vllm_config = _VllmConfig()
    worker.model_config = _ModelConfig()

    kv_caches = {
        "model.layers.60.self_attn.attn": torch.zeros(1),
        "model.layers.61.self_attn.attn": torch.zeros(1),
    }

    filtered = worker._filter_kv_caches_to_target_layers(kv_caches)

    assert set(filtered) == {"model.layers.60.self_attn.attn"}


def test_geometry_filter_drops_spec_draft_cache_with_non_target_block_axis():
    worker = _make_worker(
        target_layer_names=[
            "model.layers.60.self_attn.attn",
            "model.layers.61.self_attn.attn",
        ]
    )

    class _SpecConfig:
        def uses_draft_model(self):
            return True

    class _VllmConfig:
        speculative_config = _SpecConfig()

    class _TransferTopology:
        split_k_and_v = False

        def get_transfer_cache_regions(self, cache, layer_spec):
            return [cache]

    worker.vllm_config = _VllmConfig()
    worker.num_blocks = 4877
    worker._logical_num_blocks = 4877
    worker.transfer_topo = _TransferTopology()

    kv_caches = {
        "model.layers.60.self_attn.attn": torch.zeros(4877, 64, 64, 128),
        "model.layers.61.self_attn.attn": torch.zeros(2, 4877, 64, 64, 128),
    }

    filtered = worker._filter_kv_caches_to_target_geometry(kv_caches)

    assert set(filtered) == {"model.layers.60.self_attn.attn"}


def test_geometry_filter_keeps_non_mla_kv_tensor_after_kv_split():
    worker = _make_worker(target_layer_names=["model.layers.0.attn"])

    class _SpecConfig:
        def uses_draft_model(self):
            return True

    class _VllmConfig:
        speculative_config = _SpecConfig()

    class _TransferTopology:
        split_k_and_v = True

        def get_transfer_cache_regions(self, cache, layer_spec):
            return cache

    worker.vllm_config = _VllmConfig()
    worker.num_blocks = 128
    worker._logical_num_blocks = 128
    worker.transfer_topo = _TransferTopology()

    kv_caches = {"model.layers.0.attn": torch.zeros(2, 128, 4, 16)}

    filtered = worker._filter_kv_caches_to_target_geometry(kv_caches)

    assert filtered.keys() == kv_caches.keys()


def test_scheduler_filters_eagle_block_id_group_before_nixl_metadata():
    scheduler = NixlConnectorScheduler.__new__(NixlConnectorScheduler)
    groups = [
        KVCacheGroupSpec(["model.layers.0.attn"], _full_spec()),
        KVCacheGroupSpec(["model.layers.1.attn"], _full_spec(), is_eagle_group=True),
    ]
    scheduler.kv_cache_config = type(
        "KVCacheConfigStub", (), {"kv_cache_groups": groups}
    )()
    scheduler._nixl_kv_group_indices = (0,)
    scheduler._nixl_kv_cache_groups = [groups[0]]
    scheduler._is_hma_required = False

    assert scheduler.get_sw_clipped_blocks(([1, 2], [99])) == ([1, 2],)


def test_cross_layers_block_size_uses_kv_cache_tensor_count():
    layer_spec = _full_spec()
    worker = NixlConnectorWorker.__new__(NixlConnectorWorker)
    worker.kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[
            KVCacheTensor(size=1, shared_by=["model.layers.0.attn"]),
            KVCacheTensor(size=1, shared_by=["model.layers.1.attn"]),
            KVCacheTensor(size=1, shared_by=["model.layers.2.attn"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(
                [
                    "model.layers.0.attn",
                    "model.layers.1.attn",
                    "model.layers.2.attn",
                ],
                layer_spec,
            )
        ],
    )
    worker._nixl_kv_cache_groups = [worker.kv_cache_config.kv_cache_groups[0]]

    assert len(worker._nixl_kv_cache_groups) == 1
    assert worker._num_cross_layer_slices() == 3
