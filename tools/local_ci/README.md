# local_ci

A local CI runner for vLLM that reads `.buildkite/test_areas/*.yaml` and runs
the steps on your own machine. Useful when you maintain a vLLM fork but don't
have access to Buildkite.

It stays in sync with upstream: as you pull new changes, your runner picks
up any new or modified test areas automatically.

## Installation

The runner only needs `pyyaml` (already a vLLM dependency). No extra install.

## Quickstart

```bash
cd /path/to/vllm

# See which steps would run on your machine (auto-detects GPU count)
python -m tools.local_ci --mode list --devices h100

# Dry-run: print the commands without executing
python -m tools.local_ci --dry-run --devices h100

# Run everything that matches your machine
python -m tools.local_ci --devices h100 --gpu-count 8
```

## Configuration

Most knobs can be set via CLI flags, but for nightly runs a config file is
cleaner. Copy the example and edit:

```bash
cp tools/local_ci/runner_config.example.yaml runner_config.yaml
# edit runner_config.yaml
python -m tools.local_ci --config runner_config.yaml
```

Key fields:

| Field | Purpose |
| --- | --- |
| `repo_root` | Absolute path to the vLLM repo |
| `local_devices` | Buildkite device labels this machine claims (e.g., `[h100]`) |
| `gpu_count` | GPU count; `0` = auto-detect via `nvidia-smi -L` |
| `skip_optional` | Skip steps with `optional: true` (default `true`) |
| `skip_patterns` | Glob patterns against step labels to always skip |
| `include_areas` / `exclude_areas` | Area-level filtering |
| `path_remap` | Translate Docker paths (e.g. `/vllm-workspace/tests` → `tests`) |
| `env` | Extra env vars injected into every step |

## CLI reference

```text
python -m tools.local_ci [options]

  --config PATH           Path to runner_config.yaml
  --mode {nightly,list}   nightly: execute; list: show plan
  --area NAME             Run only this area (group name or YAML stem)
  --step PATTERN          Run only steps whose label matches this glob
  --dry-run               Print commands without executing
  --include-optional      Include steps marked optional
  --output-dir PATH       Override results directory
  --gpu-count N           Override detected GPU count
  --devices LABEL [...]   Override local device labels
```

## How it works

### Step selection

For each step in each test area YAML, the runner applies these filters in order:

1. **CLI filters** — `--area` and `--step` (if set) must match
2. **Area include/exclude** — from config file
3. **Skip patterns** — glob match against step label
4. **Optional gate** — skipped unless `--include-optional` is passed
5. **AMD-only device check** — steps with `device: mi*` are skipped on NVIDIA
6. **Device match** — if `local_devices` is set and the step specifies a
   device, the step's device must be in the list. Steps with no `device`
   field always match
7. **GPU count** — `num_devices` / `num_gpus` must be ≤ local GPU count

Anything that passes all seven filters runs.

### Execution

Each step is run as a single bash script under `set -eo pipefail` — all
commands share one shell, so `export FOO=bar` on line 1 affects later lines
as expected. Each step gets:

- A fresh subprocess with a copy of the current env plus:
    - `BUILDKITE_PARALLEL_JOB` / `BUILDKITE_PARALLEL_JOB_COUNT` (for sharding)
    - `BUILDKITE=true` (some tests gate on this)
    - Anything in your config's `env:` map
- The configured working directory (after `path_remap` substitution)
- A timeout equal to `timeout_in_minutes` from the YAML
- A dedicated log file under `ci_results/<timestamp>/<step-slug>.log`
- A JUnit XML file (`--junit-xml=...` auto-injected into the last pytest
  invocation in the step)

Buildkite's `$$VAR` escape is converted to `$VAR` before the script runs,
so the env we inject is actually read.

### Sharding

When a step sets `parallelism: N`, the runner executes it N times
sequentially with different `BUILDKITE_PARALLEL_JOB` values. pytest
commands in vLLM already consume this via `--num-shards=$BUILDKITE_PARALLEL_JOB_COUNT --shard-id=$BUILDKITE_PARALLEL_JOB`.

Sequential rather than parallel because vLLM uses sharding to fit time
budgets on CI runners — on a single local machine, running shards in
parallel would just fight for the same GPUs.

### AMD / alternate hardware

Steps whose only variant is `mirror.amd.device` are not followed — the
runner only executes the primary step definition. Steps whose primary
`device` is AMD (e.g., `mi325_1`) are always skipped.

## Output

Each run produces a timestamped directory under `ci_results/`:

```text
ci_results/2026-04-16_030000/
├── Basic_Correctness.log
├── Basic_Correctness.xml
├── Kernels_Core_Operation_Test.log
├── Kernels_Core_Operation_Test.xml
├── Language_Models_Tests_Extra_Standard_0_shard0of2.log
├── Language_Models_Tests_Extra_Standard_0_shard0of2.xml
├── ...
└── report.json
```

`report.json` has per-step status, duration, returncode, and paths to the
log / JUnit files. It's easy to consume from a Slack bot or email script.

## Nightly setup

Cron is the simplest option:

```bash
# crontab -e
0 3 * * * cd /path/to/vllm && git pull && \
  python -m tools.local_ci --config runner_config.yaml \
  >> nightly.log 2>&1
```

If you want notifications, wrap it in a script that parses `report.json`
and posts to Slack/email on failure.

## Exit codes

- `0` — all executed steps passed (skipped steps don't affect exit code)
- `1` — one or more steps failed or timed out, or the runner failed to
  load the test areas

## Limitations

- **No incremental mode yet.** Every invocation runs the full selection.
  `source_file_dependencies` is parsed but not used; adding a
  `--mode incremental --base origin/main` would be a small follow-up.
- **No dependency ordering.** `depends_on: [image-build]` is parsed but
  ignored — image-build jobs don't run locally, so this is a no-op. If
  test areas ever depend on each other, this would need work.
- **No AMD mirror execution.** AMD-only variants are skipped entirely.
- **`uv pip install` inside step commands** will mutate your environment.
  For full isolation, run the runner inside a container. For speed, accept
  the env drift.

## Troubleshooting

**"working_dir does not exist"** — your `path_remap` isn't catching a
Docker path. Add an entry mapping it to a local path.

**Step hangs forever** — the `timeout_in_minutes` from the YAML is the
hard limit. If a step gets stuck before that, the runner will kill it on
timeout. Check the `.log` file for the last output.

**"No module named yaml"** — install `pyyaml` in the environment you're
running from (`uv pip install pyyaml`).

**GPU not detected** — the auto-detect runs `nvidia-smi -L`. If that
doesn't work, pass `--gpu-count N` explicitly or set `gpu_count` in the
config.
