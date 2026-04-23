# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Parse .buildkite/test_areas/*.yaml and local runner config."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Step:
    label: str
    commands: list[str]
    area_name: str = ""
    device: str | None = None
    num_devices: int = 1
    timeout_in_minutes: int = 30
    working_dir: str | None = None
    source_file_dependencies: list[str] = field(default_factory=list)
    parallelism: int = 1
    optional: bool = False
    torch_nightly: bool = False
    mirror: dict | None = None

    @classmethod
    def from_dict(cls, d: dict, area_name: str = "") -> Step:
        num_devices = d.get("num_devices") or d.get("num_gpus") or 1
        return cls(
            label=d.get("label", "unnamed"),
            commands=d.get("commands", []),
            area_name=area_name,
            device=d.get("device"),
            num_devices=int(num_devices),
            timeout_in_minutes=int(d.get("timeout_in_minutes", 30)),
            working_dir=d.get("working_dir"),
            source_file_dependencies=d.get("source_file_dependencies", []),
            parallelism=int(d.get("parallelism", 1)),
            optional=bool(d.get("optional", False)),
            torch_nightly=bool(d.get("torch_nightly", False)),
            mirror=d.get("mirror"),
        )


@dataclass
class TestArea:
    name: str
    depends_on: list[str]
    steps: list[Step]
    source_path: Path

    @classmethod
    def from_yaml(cls, path: Path) -> TestArea:
        data = yaml.safe_load(path.read_text())
        name = data.get("group", path.stem)
        depends_on = data.get("depends_on") or []
        steps = [Step.from_dict(s, area_name=name) for s in data.get("steps", [])]
        return cls(name=name, depends_on=depends_on, steps=steps, source_path=path)


@dataclass
class RunnerConfig:
    """Local runner configuration."""

    repo_root: Path
    test_areas_dir: str = ".buildkite/test_areas"
    default_working_dir: str = "tests"
    local_devices: list[str] = field(default_factory=lambda: [])
    gpu_count: int = 0
    skip_optional: bool = True
    skip_patterns: list[str] = field(default_factory=list)
    include_areas: list[str] = field(default_factory=list)
    exclude_areas: list[str] = field(default_factory=list)
    results_dir: str = "ci_results"
    path_remap: dict[str, str] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> RunnerConfig:
        data = yaml.safe_load(path.read_text()) or {}
        repo_root = Path(data.get("repo_root", ".")).expanduser().resolve()
        return cls(
            repo_root=repo_root,
            test_areas_dir=data.get("test_areas_dir", ".buildkite/test_areas"),
            default_working_dir=data.get("default_working_dir", "tests"),
            local_devices=data.get("local_devices", []),
            gpu_count=int(data.get("gpu_count", 0)),
            skip_optional=bool(data.get("skip_optional", True)),
            skip_patterns=data.get("skip_patterns", []),
            include_areas=data.get("include_areas", []),
            exclude_areas=data.get("exclude_areas", []),
            results_dir=data.get("results_dir", "ci_results"),
            path_remap=data.get("path_remap", {}),
            env=data.get("env", {}),
        )

    @classmethod
    def defaults(cls, repo_root: Path) -> RunnerConfig:
        return cls(repo_root=repo_root)


def detect_gpu_count() -> int:
    """Detect available NVIDIA GPUs via nvidia-smi."""
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return len([l for l in result.stdout.strip().splitlines() if l.strip()])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return 0


def load_test_areas(cfg: RunnerConfig) -> list[TestArea]:
    """Load all test area YAML files."""
    areas_dir = cfg.repo_root / cfg.test_areas_dir
    if not areas_dir.is_dir():
        raise FileNotFoundError(f"Test areas directory not found: {areas_dir}")

    areas = []
    for path in sorted(areas_dir.glob("*.yaml")):
        try:
            areas.append(TestArea.from_yaml(path))
        except Exception as e:
            print(f"Warning: failed to parse {path.name}: {e}")
    return areas
