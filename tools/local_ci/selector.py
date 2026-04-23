# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decide which steps to run based on local runner configuration."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass

from .config import RunnerConfig, Step, TestArea


@dataclass
class Selection:
    step: Step
    area: TestArea
    should_run: bool
    skip_reason: str | None = None


# Known AMD-only device prefixes (used to skip mirror-only steps).
_AMD_DEVICE_PREFIXES = ("mi", "rocm")


def _is_amd_only_device(device: str | None) -> bool:
    if not device:
        return False
    return device.lower().startswith(_AMD_DEVICE_PREFIXES)


def _matches_any(patterns: list[str], text: str) -> bool:
    return any(fnmatch.fnmatch(text, p) for p in patterns)


def select(
    areas: list[TestArea],
    cfg: RunnerConfig,
    include_optional: bool = False,
    area_filter: str | None = None,
    step_filter: str | None = None,
) -> list[Selection]:
    """Return a Selection for every step, with should_run / skip_reason set."""

    selections: list[Selection] = []
    for area in areas:
        area_included = (
            not cfg.include_areas
            or area.name in cfg.include_areas
            or area.source_path.stem in cfg.include_areas
        )
        area_excluded = (
            area.name in cfg.exclude_areas or area.source_path.stem in cfg.exclude_areas
        )

        for step in area.steps:
            reason = _skip_reason(
                step,
                area,
                cfg,
                area_included=area_included,
                area_excluded=area_excluded,
                include_optional=include_optional,
                area_filter=area_filter,
                step_filter=step_filter,
            )
            selections.append(
                Selection(
                    step=step,
                    area=area,
                    should_run=reason is None,
                    skip_reason=reason,
                )
            )
    return selections


def _skip_reason(
    step: Step,
    area: TestArea,
    cfg: RunnerConfig,
    *,
    area_included: bool,
    area_excluded: bool,
    include_optional: bool,
    area_filter: str | None,
    step_filter: str | None,
) -> str | None:
    # Command-line filters take priority.
    if area_filter and area_filter not in (area.name, area.source_path.stem):
        return f"area filter: {area_filter!r}"
    if step_filter and not fnmatch.fnmatch(step.label, step_filter):
        return f"step filter: {step_filter!r}"

    # Area include/exclude.
    if not area_included:
        return "area not in include_areas"
    if area_excluded:
        return "area in exclude_areas"

    # Skip patterns (match against step label).
    if _matches_any(cfg.skip_patterns, step.label):
        return "matched skip_patterns"

    # Optional steps.
    if step.optional and not include_optional:
        return "optional (use --include-optional)"

    # AMD-only steps — skip on NVIDIA.
    if _is_amd_only_device(step.device):
        return f"AMD-only device ({step.device})"

    # Device match: if local_devices is set and the step specifies a device,
    # skip when there's no match. If the step has no device, accept it.
    if cfg.local_devices and step.device and step.device not in cfg.local_devices:
        return f"device {step.device!r} not in local_devices {cfg.local_devices}"

    # GPU count.
    if cfg.gpu_count and step.num_devices > cfg.gpu_count:
        return f"needs {step.num_devices} GPUs, have {cfg.gpu_count}"

    # No commands means it's a phantom step (shouldn't happen, but guard).
    if not step.commands:
        return "no commands"

    return None
