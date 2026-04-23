# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Execute test steps via subprocess."""

from __future__ import annotations

import os
import regex as re
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import RunnerConfig, Step


@dataclass
class StepResult:
    label: str
    area_name: str
    status: str  # "passed" | "failed" | "timeout" | "skipped"
    duration_s: float = 0.0
    returncode: int | None = None
    shard: tuple[int, int] | None = None  # (index, total)
    skip_reason: str | None = None
    log_path: Path | None = None
    junit_path: Path | None = None


def _remap_path(path: str, remap: dict[str, str], repo_root: Path) -> str:
    """Apply path_remap substitutions. Longest prefix wins."""
    if not path:
        return path
    for src in sorted(remap, key=len, reverse=True):
        if path == src or path.startswith(src + "/"):
            dst = remap[src]
            new = dst + path[len(src) :]
            # If still relative, resolve against repo_root.
            if not os.path.isabs(new):
                new = str((repo_root / new).resolve())
            return new
    return path


def _resolve_cwd(step: Step, cfg: RunnerConfig) -> Path:
    wd = step.working_dir or cfg.default_working_dir
    remapped = _remap_path(wd, cfg.path_remap, cfg.repo_root)
    if os.path.isabs(remapped):
        return Path(remapped)
    return (cfg.repo_root / remapped).resolve()


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _inject_junit(cmd: str, junit_path: Path) -> str:
    """Append --junit-xml=... to pytest commands that don't already have it."""
    stripped = cmd.strip()
    is_pytest = bool(re.match(r"(?:[A-Z_]+=\S+\s+)*pytest\b", stripped))
    if not is_pytest:
        return cmd
    if "--junit-xml" in stripped or "--junitxml" in stripped:
        return cmd
    return f"{cmd} --junit-xml={shlex.quote(str(junit_path))}"


def run_step(
    step: Step,
    cfg: RunnerConfig,
    output_dir: Path,
    shard_id: int = 0,
    shard_count: int = 1,
    dry_run: bool = False,
) -> StepResult:
    label_for_shard = step.label.replace("%N", str(shard_id))
    shard = (shard_id, shard_count) if shard_count > 1 else None

    cwd = _resolve_cwd(step, cfg)
    slug = _slugify(label_for_shard)
    if shard:
        slug = f"{slug}_shard{shard_id}of{shard_count}"

    log_path = output_dir / f"{slug}.log"
    junit_path = output_dir / f"{slug}.xml"

    env = os.environ.copy()
    env.update(cfg.env)
    env["BUILDKITE_PARALLEL_JOB"] = str(shard_id)
    env["BUILDKITE_PARALLEL_JOB_COUNT"] = str(shard_count)
    env["BUILDKITE"] = "true"  # some tests gate on this

    # Chain commands under `set -e` so a failure stops the step; but the
    # last line of pytest should still run with junit injection.
    injected_cmds = []
    pytest_count = sum(
        1 for c in step.commands if re.match(r"(?:[A-Z_]+=\S+\s+)*pytest\b", c.strip())
    )
    pytest_seen = 0
    for c in step.commands:
        # Buildkite uses $$VAR to escape env expansion into the runner
        # context. In bash, $$ means PID — we want plain $VAR.
        c = c.replace("$$", "$")
        if re.match(r"(?:[A-Z_]+=\S+\s+)*pytest\b", c.strip()):
            pytest_seen += 1
            # Only inject on the last pytest invocation to get a single
            # JUnit file per step (multiple writes would overwrite).
            if pytest_seen == pytest_count:
                c = _inject_junit(c, junit_path)
        injected_cmds.append(c)

    script = "set -eo pipefail\n" + "\n".join(injected_cmds) + "\n"

    if dry_run:
        print(f"[DRY-RUN] {label_for_shard}")
        print(f"  cwd: {cwd}")
        if shard:
            print(f"  shard: {shard_id}/{shard_count}")
        for line in script.splitlines():
            print(f"    $ {line}")
        return StepResult(
            label=label_for_shard,
            area_name=step.area_name,
            status="skipped",
            shard=shard,
            skip_reason="dry-run",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if not cwd.is_dir():
        return StepResult(
            label=label_for_shard,
            area_name=step.area_name,
            status="failed",
            shard=shard,
            skip_reason=f"working_dir does not exist: {cwd}",
        )

    timeout_s = step.timeout_in_minutes * 60
    start = time.monotonic()
    with open(log_path, "wb") as log_f:
        # start_new_session=True makes bash the leader of a new process group
        # so that os.killpg on timeout reaches all grandchildren (pytest
        # workers, vLLM EngineCore processes) and prevents GPU-memory leaks
        # from orphaned processes accumulating across a long run.
        proc = subprocess.Popen(
            ["bash", "-c", script],
            cwd=str(cwd),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(5)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass  # process group already exited after SIGTERM
            proc.wait()  # reap to avoid zombie
            duration = time.monotonic() - start
            return StepResult(
                label=label_for_shard,
                area_name=step.area_name,
                status="timeout",
                duration_s=duration,
                shard=shard,
                log_path=log_path,
            )

    duration = time.monotonic() - start
    status = "passed" if proc.returncode == 0 else "failed"
    return StepResult(
        label=label_for_shard,
        area_name=step.area_name,
        status=status,
        duration_s=duration,
        returncode=proc.returncode,
        shard=shard,
        log_path=log_path,
        junit_path=junit_path if junit_path.exists() else None,
    )


def run_step_with_sharding(
    step: Step,
    cfg: RunnerConfig,
    output_dir: Path,
    dry_run: bool = False,
) -> list[StepResult]:
    if step.parallelism <= 1:
        return [run_step(step, cfg, output_dir, 0, 1, dry_run=dry_run)]
    results = []
    for shard_id in range(step.parallelism):
        results.append(
            run_step(
                step,
                cfg,
                output_dir,
                shard_id=shard_id,
                shard_count=step.parallelism,
                dry_run=dry_run,
            )
        )
    return results
