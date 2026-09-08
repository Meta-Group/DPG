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
| A | Codex | Core routing fidelity, `context_order`, sink invariant, accessors | focused tests + `pytest -q` report | implemented; E0/E1 audit clean |
| B | Claude Code | Independent review of context-order mathematics and edge cases | review notes; added tests only in `tests/test_context_order_review.py` | available |
| C | Codex | Benchmark execution on full hardware and CSV integrity | `benchmark.csv`, log, failed-row audit | complete; 1125/1125 ok |
| D | Claude Code | README, quickstart, changelog, API/docstring review | documentation diff + link check | available |
| E | Codex + Claude Code | Integration/release audit | no unresolved blocker; final branch report | queued; E4 complete, E5 running |

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

E3 uses 300 cells over controlled synthetic S1–S4 datasets, varying only
irrelevant-feature count and label noise, to test whether auto-k grows with
irrelevant features.

E4 uses 16 resource-intensive cells on S5 (100k×50) and S6 (10k×500), with
Random Forest and Extra Trees at 10/25 learners and seeds 0/1. It records k=1
and auto-k construction time, graph size, node ratio, and violation history.
The completed audit contains 16/16 successful unique cells, including the
isolated S5 Extra Trees retry cells in `e4_s5_extratrees.csv`.

E5 (`scripts/run_dpg030_e5_lrc.py`, `scripts/lrc_aggregation.py`) compares
execution-trace k=1 against auto-k on iris/wine/breast_cancer x 5 ensemble
families x {10, 25, 50} learners x 5 seeds (225 cells), evaluating predicate
LRC against `feature_importances_` via Spearman correlation and top-k ranking
overlap, alongside runtime, resolved k, and graph size. Three aggregations
are compared from the same node-level LRC values: `sum` (the shipped
production default in `DecisionPredicateGraph.get_predicate_lrc()`, per
CHANGELOG 0.3.0), and `max`/`weighted_sum` (analysis-only, implemented only
in `scripts/lrc_aggregation.py`, never added to the public API). This is a
verification comparison, not an optimization: the aggregation choice must
not be picked by which correlates best, or the comparison becomes circular.
`BaggingClassifier` has no `feature_importances_`; its cells are recorded
with `status=skipped_no_feature_importances` and graph-construction metrics
still populated, never omitted. Status: runner implemented, unit- and
cell-level tested, smoke-tested end to end on a 2-cell grid; the full
225-cell grid is running on a separate output path
(`experiments/dpg_0_3_0/results/e5_lrc_alignment.csv`) now that E4 has
released the machine, with two workers to limit memory pressure.

E6 (regression scope) found that regression leaves (`"Pred <value>"`) are
already treated as terminal sinks by `_context_node`, so DPG-k mechanically
builds a graph for regressors at any `context_order` without raising.
However "one sink per output" is not a defined invariant there the way it is
for classification: a regression sink is only as unique as its 2-decimal
rounded value, an artifact of label rounding rather than a modeled output
count. Investigating this surfaced a real, pre-existing, context_order
-independent bug: `GraphMetrics.extract_communities` assumed at least one
classifier "Class " sink to anchor its absorbing Markov chain, so any
regressor made the underlying solve singular and raised an opaque
`numpy.linalg.LinAlgError`. Decision: regression sink/community semantics
are explicitly out of scope for 0.3.0 (documented in CHANGELOG/README); the
`extract_communities` crash is fixed defensively (raises a clear
`ValueError` naming the requirement) since it is a bug independent of any
DPG-k semantic decision. No regression numeric output or classifier
behavior changed.

E7 (downstream compatibility) exercised every consumer named above --
`plot_dpg`, class-boundary extraction, communities, `DPGExplainer`
global/local explanations, and faithfulness evaluation -- at
`context_order` 2 and `"auto"`, for RandomForest and GradientBoosting on
iris. All pass unmodified; no production code changes were needed for
context_order compatibility itself. No DPG-IF/DPG-CF consumers exist in
this codebase. Two issues were found and deliberately left unfixed as
outside E7's scope, because both reproduce identically at the k=1 default
and are unrelated to context_order: `evaluate_faithfulness()` on
`GradientBoostingClassifier` fails inside sklearn's own internals because
`SklearnEnsembleNormalizer` flattens `estimators_` for DPG's traversal and
`evaluate_faithfulness` calls `.predict()` on that same flattened copy
(tracked via a strict `xfail` test so it is not silently lost); and
`plot_dpg` does not visually disambiguate two nodes that share a predicate
label but differ in context in static PNG/PDF renders (the DOT `tooltip`
carries context but is invisible outside interactive viewers --
`get_node_context(node)` remains the documented way to recover it).

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
