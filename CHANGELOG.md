# Changelog

## 0.3.0

### Added

- Added context-aware DPG construction through `context_order` in execution
  trace mode. `context_order=1` preserves legacy predicate identity; integer
  orders and `"auto"` split nodes by recent execution history.
- Class outcomes are shared terminal sinks at every context order, and node
  metadata exposes `predicate`, `context`, and `context_order`.
- Added enumeration-free local context-order resolution and a reproducible
  multi-process benchmark launcher under `scripts/`.
- Added exact sklearn `decision_path` routing. Threshold rounding now formats
  predicate labels without changing the branch selected by the model.
- Added `decimal_threshold="auto"`, which derives precision from the data and
  warns when a tree threshold is off the derived grid.

### Compatibility and limitations

- The default remains `context_order=1`; DPG-k is opt-in so existing consumers
  keep their graph shape. `context_order > 1` requires `execution_trace` mode.
- `get_trace_consistent_lrc()` remains available for k=1 and is deprecated for
  contextual graphs; k>1 aggregates ordinary unweighted node LRC by predicate.
- The routing correction can change graph weights and labels at floating-point
  boundaries. Residual off-grid behavior is reported by the auto-precision
  warning rather than hidden.
- **Regression sink semantics are out of scope for 0.3.0.** `context_order`
  mechanically builds a graph for regressors (regression leaves are treated
  as terminal sinks, like class leaves), but there is no "one sink per
  output" guarantee: a regression sink is only as unique as the 2-decimal
  rounded leaf value, so two leaves collide into one sink by coincidence of
  rounding, not by any modeled notion of "output". A principled regression
  sink policy is deferred to a future release. `class_boundaries` and
  `communities` remain classifier-only features; calling
  `DPGExplainer.explain_global(communities=True)` on a regressor now raises a
  clear `ValueError` instead of an internal `numpy.linalg.LinAlgError`.

## 0.2.0

### Added

- **Execution-trace artefacts** (opt-in, `graph_construction.mode = "execution_trace"`).
  The pooled DPG graph aggregates transitions across every sample and tree, so a
  multi-hop path can appear in the pooled graph even if no single tree execution
  ever produced it. `DecisionPredicateGraph` now optionally builds artefacts
  derived exclusively from single, observed sample-tree executions:
  - `get_trace_consistent_lrc()` — a local-reaching-centrality-like score per
    predicate, computed only from same-trace reach.
  - `get_trace_consistent_trc()` — observed downstream predicate labels per
    predicate, as sorted tuples.
  - `get_trace_signatures()` — aggregated, deduplicated predicate sequences with
    observed occurrence counts (`TraceSignature`).
  - These getters are empty before `fit()`, reset on every refit, and remain
    empty outside `execution_trace` mode.
  - `perc_var` filtering continues to apply only to the pooled graph's edges; it
    never removes a trace artefact — see `docs/quickstart.md`.
- `metrics.nodes.NodeMetrics.extract_node_metrics` accepts an optional
  `trace_lrc_by_label` mapping; existing callers that omit it are unaffected.
- `DPGExplainer` forwards trace-consistent LRC into node metrics automatically
  when `graph_construction_mode == "execution_trace"`.

### Compatibility

- The default `aggregated_transitions` mode and its output are unchanged.
- All additions above are opt-in and additive.
