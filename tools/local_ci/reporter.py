# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Report and summarize test results."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .executor import StepResult

# ANSI colors for terminal output.
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_RESET = "\033[0m"

_STATUS_STYLE = {
    "passed": (_GREEN, "PASS"),
    "failed": (_RED, "FAIL"),
    "timeout": (_RED, "TIME"),
    "skipped": (_DIM, "SKIP"),
}


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes}m{secs:02d}s"


def print_step_result(result: StepResult) -> None:
    color, tag = _STATUS_STYLE.get(result.status, (_RESET, result.status.upper()))
    duration = _fmt_duration(result.duration_s) if result.duration_s > 0 else ""
    extra = ""
    if result.skip_reason:
        extra = f"  ({result.skip_reason})"
    elif result.status == "failed" and result.log_path:
        extra = f"  -> {result.log_path}"
    print(f"  {color}[{tag}]{_RESET} {result.label}  {_DIM}{duration}{_RESET}{extra}")


def print_summary(results: list[StepResult]) -> None:
    passed = [r for r in results if r.status == "passed"]
    failed = [r for r in results if r.status == "failed"]
    timed_out = [r for r in results if r.status == "timeout"]
    skipped = [r for r in results if r.status == "skipped"]
    total_time = sum(r.duration_s for r in results)

    print()
    print(f"{_CYAN}{'=' * 60}{_RESET}")
    print(
        f"{_CYAN}Summary{_RESET}  "
        f"{_GREEN}{len(passed)} passed{_RESET}  "
        f"{_RED}{len(failed)} failed{_RESET}  "
        f"{_RED}{len(timed_out)} timed out{_RESET}  "
        f"{_DIM}{len(skipped)} skipped{_RESET}  "
        f"total: {_fmt_duration(total_time)}"
    )
    print(f"{_CYAN}{'=' * 60}{_RESET}")

    if failed:
        print(f"\n{_RED}Failed steps:{_RESET}")
        for r in failed:
            print(f"  - {r.label}  (rc={r.returncode})")
            if r.log_path:
                print(f"    log: {r.log_path}")
    if timed_out:
        print(f"\n{_RED}Timed out steps:{_RESET}")
        for r in timed_out:
            print(f"  - {r.label}  (limit={_fmt_duration(r.duration_s)})")
            if r.log_path:
                print(f"    log: {r.log_path}")


def write_json_report(results: list[StepResult], output_dir: Path) -> Path:
    report_path = output_dir / "report.json"
    records = []
    for r in results:
        records.append(
            {
                "label": r.label,
                "area": r.area_name,
                "status": r.status,
                "duration_s": round(r.duration_s, 2),
                "returncode": r.returncode,
                "shard": list(r.shard) if r.shard else None,
                "skip_reason": r.skip_reason,
                "log": str(r.log_path) if r.log_path else None,
                "junit": str(r.junit_path) if r.junit_path else None,
            }
        )
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total": len(records),
        "passed": sum(1 for r in records if r["status"] == "passed"),
        "failed": sum(1 for r in records if r["status"] == "failed"),
        "timed_out": sum(1 for r in records if r["status"] == "timeout"),
        "skipped": sum(1 for r in records if r["status"] == "skipped"),
        "steps": records,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report_path
