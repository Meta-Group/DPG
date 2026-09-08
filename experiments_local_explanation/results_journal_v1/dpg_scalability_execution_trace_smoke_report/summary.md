# DPG Depth And Ensemble Scalability

This focused stress experiment evaluates whether DPG-local remains usable beyond the shallow depth-4 random forests used in the main ECML submission.

The goal is not to open a new broad benchmark, but to bound the journal claim about random forests with deeper trees and larger ensembles.

## Summary By Depth

| graph_construction_mode | method              | max_depth_label | n_datasets | n_runs | local_matches_model_rate | avg_explanation_confidence | avg_support_margin | avg_competitor_exposure | avg_edge_recall | avg_recombination_rate | avg_num_active_nodes | avg_runtime_ms |
| ----------------------- | ------------------- | --------------- | ---------- | ------ | ------------------------ | -------------------------- | ------------------ | ----------------------- | --------------- | ---------------------- | -------------------- | -------------- |
| execution_trace         | dpg_execution_trace | 8               | 1          | 2      | 0.9                      | 0.5428                     | 0.5805             | 0.2226                  | 0.9864          | 0                      | 138.4                | 5392           |

## Summary By Depth And Ensemble Size

| graph_construction_mode | method              | n_estimators | max_depth_label | n_datasets | n_runs | local_matches_model_rate | avg_explanation_confidence | avg_support_margin | avg_competitor_exposure | avg_edge_recall | avg_recombination_rate | avg_num_active_nodes | avg_runtime_ms |
| ----------------------- | ------------------- | ------------ | --------------- | ---------- | ------ | ------------------------ | -------------------------- | ------------------ | ----------------------- | --------------- | ---------------------- | -------------------- | -------------- |
| execution_trace         | dpg_execution_trace | 20           | 8               | 1          | 1      | 0.9                      | 0.5836                     | 0.5802             | 0.2269                  | 0.9929          | 0                      | 97.4                 | 2555           |
| execution_trace         | dpg_execution_trace | 50           | 8               | 1          | 1      | 0.9                      | 0.5019                     | 0.5807             | 0.2184                  | 0.9799          | 0                      | 179.5                | 8230           |

## Lowest-Match / Highest-Cost Runs

| dataset | method              | graph_construction_mode | n_estimators | max_depth_label | seed | model_accuracy | local_matches_model_rate | avg_explanation_confidence | avg_support_margin | avg_competitor_exposure | avg_num_active_nodes | avg_num_active_edges_filtered | avg_runtime_ms | local_failure_rate |
| ------- | ------------------- | ----------------------- | ------------ | --------------- | ---- | -------------- | ------------------------ | -------------------------- | ------------------ | ----------------------- | -------------------- | ----------------------------- | -------------- | ------------------ |
| vehicle | dpg_execution_trace | execution_trace         | 50           | 8               | 27   | 0.7294         | 0.9                      | 0.5019                     | 0.5807             | 0.2184                  | 179.5                | 300.2                         | 8230           | 0                  |
| vehicle | dpg_execution_trace | execution_trace         | 20           | 8               | 27   | 0.7471         | 0.9                      | 0.5836                     | 0.5802             | 0.2269                  | 97.4                 | 128.6                         | 2555           | 0                  |

## Writing Guidance

- Use this experiment to answer the reviewer concern about shallow random forests.
- If graph size or runtime grows sharply, narrow the claim to small-to-moderate random forests or report pruning as a deployment requirement.
- If trace recovery and diagnostic scores remain stable, report this as robustness evidence, not as a new main contribution.
- Keep output-fidelity metrics as context; the primary scalability quantities are graph size, runtime, edge recall, and recombination.
