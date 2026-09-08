# DPG-k 0.3.0 execution plan

Branch: `DPG-k-0.3.0`

This plan is the coordination contract for Codex and Claude Code. Work packages
touch separate files where possible; each agent must run the focused tests and
record results in the status table before handing off.

## Hardware profile

Detected 2026-09-08: Intel Core i9-10980XE, 18 physical cores / 36 logical CPUs,
62 GiB RAM, one NUMA node. The benchmark launcher defaults to 36 processes,
with one BLAS/OpenMP thread per process to avoid thread oversubscription.
Override with `DPG_WORKERS=N` when memory pressure or interactive work requires it.

## Work packages

| ID | Owner | Scope | Acceptance evidence | Status |
|---|---|---|---|---|
| A | Codex | Core routing fidelity, `context_order`, sink invariant, accessors | focused tests + `pytest -q` report | implemented; audit pending |
| B | Claude Code | Independent review of context-order mathematics and edge cases | review notes; added tests only in `tests/test_context_order_review.py` | available |
| C | Codex | Benchmark execution on full hardware and CSV integrity | `benchmark.csv`, log, failed-row audit | complete; 1125/1125 ok |
| D | Claude Code | README, quickstart, changelog, API/docstring review | documentation diff + link check | available |
| E | Codex + Claude Code | Integration/release audit | no unresolved blocker; final branch report | queued |

## Coordination rules

1. Never rewrite or reset another agent's changes. Use small commits and name
   them `030-A-*`, `030-B-*`, etc.
2. Before editing, run `git status --short --branch`. Before handoff, record
   the commit hash, tests, and files changed in the agent's status note.
3. Do not commit Telegram credentials. They are supplied only in the shell
   environment as `TELEGRAM_CHAT_ID` and `TELEGRAM_BOT_TOKEN`.
4. Benchmark rows are append/audit artifacts: failures must remain rows with
   `status=error` and an `error` field; do not silently drop them.
5. Do not run two benchmark launchers against the same output CSV. Use a unique
   output path for parallel agents or coordinate the run in this file.

## Experiment phases

The launcher supports the procedure's comparable variants:

- `aggregated`: legacy pooled transitions, k=1;
- `execution_trace_k1`: raw execution traces, k=1;
- `dpg_k_auto`: execution traces, auto-resolved context order and auto precision.

The default grid is 3 sklearn datasets × 5 ensemble families × 5 learner
counts × 5 seeds × 3 variants. Every row records commit, timing, graph size,
edge mass, resolved k, violation history, precision, warnings, and errors.

E0/E1 validation uses 675 cells: the same three datasets, five classifier
families, learner counts 5/10/25, seeds 0/1/2, and decimal thresholds
1/2/4/6/auto. It writes `e0_e1_validation.csv` and reports identity, sink,
edge-mass, misroute, and local-context checks.

E2 uses 375 cells across the same datasets and model families, learner counts
5/10/25/50/100, and seeds 0–4. It compares k=1 with auto-k for phantom path
rate/mass, graph size, construction time, and enumeration exactness.

## Long-running command

Run from the repository root and disconnect safely:

```bash
export TELEGRAM_CHAT_ID="..."
export TELEGRAM_BOT_TOKEN="..."
nohup scripts/launch_dpg030_experiments.sh \
  > experiments/dpg_0_3_0/logs/benchmark.nohup.log 2>&1 < /dev/null &
echo $! > experiments/dpg_0_3_0/logs/benchmark.pid
```

Monitor with `tail -f` on the log and inspect the CSV while it grows. Telegram
updates are sent at start, every 25 completed tasks, and finish; notification
failure never aborts the benchmark.

## Release gates

- k=1 node/edge/weight identity remains unchanged for the legacy path.
- auto context order has zero local violations at its resolved order.
- one sink exists per classifier class and edge mass is auditable.
- routing tests cover numeric precision values and `decimal_threshold="auto"`.
- benchmark errors are investigated or documented; no failed cell is omitted.
- docs state that `context_order > 1` requires `execution_trace` and that
  `get_trace_consistent_lrc()` is deprecated for k>1.
