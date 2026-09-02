# Changelog

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
