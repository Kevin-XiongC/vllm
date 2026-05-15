# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``NixlConnectorWorker._filter_kv_caches_to_target_layers``.

The filter uses ``self._layer_specs`` (built from
``kv_cache_config.kv_cache_groups``) as the authoritative target-layer
set, so it is naturally correct under pipeline parallelism and under
draft models that do not follow the ``model.layers.<idx>`` continuation
convention. It also applies uniformly regardless of speculative-decode
config, so unexpected non-target keys surface the same way in every
code path.

These tests avoid importing the heavier ``test_nixl_connector`` module
(which pulls in optional deps like ``ray`` at collection time) so the
filter logic can be exercised in a minimal CPU environment.
"""

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlConnectorWorker


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
    return worker


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
