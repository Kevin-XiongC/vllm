# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Local CI runner for vLLM — consumes .buildkite/test_areas/*.yaml.

Usage:
    python -m tools.local_ci --help
    python -m tools.local_ci --dry-run
    python -m tools.local_ci --area basic_correctness
    python -m tools.local_ci --mode list
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from .config import RunnerConfig, detect_gpu_count, load_test_areas
from .executor import run_step_with_sharding
from .reporter import (
    print_step_result,
    print_summary,
    write_json_report,
)
from .selector import select


def _auto_detect_repo_root() -> Path:
    """Walk up from CWD looking for .buildkite/test_areas/."""
    candidate = Path.cwd()
    for _ in range(10):
        if (candidate / ".buildkite" / "test_areas").is_dir():
            return candidate
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    return Path.cwd()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="local_ci",
        description="Local CI runner for vLLM — reads .buildkite/test_areas/*.yaml",
    )
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to runner_config.yaml (optional; auto-detects if absent)",
    )
    p.add_argument(
        "--mode",
        "-m",
        choices=["nightly", "list"],
        default="nightly",
        help="nightly: run tests; list: show what would run",
    )
    p.add_argument(
        "--area",
        type=str,
        default=None,
        help="Run only this test area (group name or yaml stem)",
    )
    p.add_argument(
        "--step",
        type=str,
        default=None,
        help="Run only steps whose label matches this glob pattern",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    p.add_argument(
        "--include-optional",
        action="store_true",
        help="Include steps marked optional",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Directory for logs and reports (default: ci_results/<date>)",
    )
    p.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="Override GPU count (default: auto-detect)",
    )
    p.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Device labels this machine matches (e.g., h100 h200_18gb)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Load or create runner config.
    if args.config and args.config.exists():
        cfg = RunnerConfig.from_yaml(args.config)
    else:
        cfg = RunnerConfig.defaults(_auto_detect_repo_root())

    # CLI overrides.
    if args.gpu_count is not None:
        cfg.gpu_count = args.gpu_count
    if args.devices is not None:
        cfg.local_devices = args.devices
    if args.output_dir:
        cfg.results_dir = str(args.output_dir)

    # Auto-detect GPUs if not set.
    if not cfg.gpu_count:
        cfg.gpu_count = detect_gpu_count()

    # Set up default path_remap if not configured.
    if not cfg.path_remap:
        cfg.path_remap = {
            "/vllm-workspace/tests": "tests",
            "/vllm-workspace": ".",
        }

    # Load test areas.
    try:
        areas = load_test_areas(cfg)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(
        f"Loaded {len(areas)} test areas with "
        f"{sum(len(a.steps) for a in areas)} total steps"
    )
    print(f"GPUs: {cfg.gpu_count}  Devices: {cfg.local_devices or ['any']}")
    print()

    # Select.
    selections = select(
        areas,
        cfg,
        include_optional=args.include_optional,
        area_filter=args.area,
        step_filter=args.step,
    )

    runnable = [s for s in selections if s.should_run]
    skipped = [s for s in selections if not s.should_run]

    # List mode: show selections and exit.
    if args.mode == "list":
        _print_list(selections)
        return 0

    print(f"Will run {len(runnable)} steps, skip {len(skipped)}")
    if skipped and not args.dry_run:
        # Group skip reasons.
        reasons: dict[str, int] = {}
        for s in skipped:
            r = s.skip_reason or "unknown"
            reasons[r] = reasons.get(r, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  skip ({count}): {reason}")
    print()

    # Determine output directory.
    if args.output_dir:
        output_dir = args.output_dir
    else:
        datestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        output_dir = cfg.repo_root / cfg.results_dir / datestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run.
    all_results = []
    current_area = ""
    for sel in selections:
        if not sel.should_run:
            continue

        if sel.area.name != current_area:
            current_area = sel.area.name
            print(f"\n--- {current_area} ---")

        results = run_step_with_sharding(
            sel.step,
            cfg,
            output_dir,
            dry_run=args.dry_run,
        )
        for r in results:
            print_step_result(r)
        all_results.extend(results)

    if not args.dry_run and all_results:
        print_summary(all_results)
        report_path = write_json_report(all_results, output_dir)
        print(f"\nReport: {report_path}")

    # Exit code: non-zero if any step failed or timed out.
    if any(r.status in ("failed", "timeout") for r in all_results):
        return 1
    return 0


def _print_list(selections: list) -> None:
    current_area = ""
    for sel in selections:
        if sel.area.name != current_area:
            current_area = sel.area.name
            print(f"\n  {current_area}")
            print(f"  {'─' * len(current_area)}")
        status = "RUN " if sel.should_run else "SKIP"
        device = sel.step.device or "any"
        gpus = sel.step.num_devices
        opt = " [optional]" if sel.step.optional else ""
        reason = f"  ({sel.skip_reason})" if sel.skip_reason else ""
        print(
            f"    [{status}] {sel.step.label}  device={device} gpus={gpus}{opt}{reason}"
        )


if __name__ == "__main__":
    sys.exit(main())
