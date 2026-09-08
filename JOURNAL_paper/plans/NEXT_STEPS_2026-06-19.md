# Next Steps For Tomorrow

Date saved: 2026-06-18

## Current Running Job

Focused deeper-RF scalability experiment is running under `nohup`.

- PID: `1944379`
- Log: `experiments_local_explanation/results_journal_v1/logs/dpg_scalability_depth_focused_20260618_180303.log`
- Output root: `experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_by_dataset`
- Report target: `experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_report`

Experiment scope:

- Datasets: `iris`, `banknote-authentication`, `diabetes`, `vehicle`, `madelon`, `isolet`
- Graph mode: `execution_trace`
- `n_estimators`: `20,50`
- `max_depth`: `4,8,12`
- Seed: `27`
- `max_test_samples`: `20`

Status when saved:

- Job was still running.
- At least 2 dataset `summary.csv` files had been written.
- The log showed active progress on `madelon`.

## First Check Tomorrow

Check whether the job is still running:

```bash
ps -p 1944379 -o pid,ppid,sid,stat,etime,cmd
```

Check how many dataset summaries exist:

```bash
find experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_by_dataset -maxdepth 2 -name summary.csv | sort | wc -l
```

Inspect the latest log:

```bash
tail -n 80 experiments_local_explanation/results_journal_v1/logs/dpg_scalability_depth_focused_20260618_180303.log
```

Expected final count is 6 dataset-level `summary.csv` files, one per selected dataset.

## If The Job Finished Successfully

Generate the scalability report:

```bash
./.venv/bin/python experiments_local_explanation/analyze_scalability_depth_experiment.py \
  --run_root experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_by_dataset \
  --out_dir experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_report
```

Then read:

```bash
sed -n '1,220p' experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_report/summary.md
```

Key question for interpretation:

- Does DPG-local remain trace-faithful at deeper depths?
- How much do runtime and graph size grow from depth 4 to depth 8/12 and from 20 to 50 trees?
- Are `isolet` and `madelon` clear failure or cost regimes?

## If The Job Is Still Running

Let it continue unless the log shows repeated failures.

If it is too slow on `isolet` or another large dataset, preserve partial results and narrow the scalability claim. Do not delete partial outputs. The partial run is still useful for runtime/size evidence.

## If The Job Failed

Check the last failure in the log:

```bash
tail -n 160 experiments_local_explanation/results_journal_v1/logs/dpg_scalability_depth_focused_20260618_180303.log
```

Then generate a report from any completed dataset summaries anyway. The report script can summarize partial outputs as long as at least one `summary.csv` exists.

## Writing Decision After Report

If runtime and graph size are acceptable:

> A focused stress test on six representative datasets with deeper forests confirmed that DPG-local preserves trace-faithful construction under larger tree depths, although runtime and graph size increase with ensemble size and depth.

If runtime or graph size grows sharply:

> A focused stress test shows that DPG-local is best framed as a diagnostic method for small-to-moderate random forests unless pruning or sampling is used. We therefore narrow the journal claim and report scalability as a practical limitation.

## Paper Work After Scalability

1. Update the Results section with a short scalability paragraph.
2. Add one compact scalability table, probably in appendix or supplementary material.
3. Update Limitations to mention the observed runtime/graph-size boundary.
4. Keep ICE in appendix only.
5. Keep critical nodes as conditional diagnostics, not a main contribution.
6. Use confidence, support margin, competitor exposure, path controls, and scalability as the main journal revision evidence.

## Update: Confidence Sensitivity And Model-Family Work

Completed confidence-score sensitivity analysis:

- Script: `experiments_local_explanation/analyze_confidence_sensitivity.py`
- Output: `experiments_local_explanation/results_journal_v1/confidence_sensitivity/summary.md`
- Main result: reported low-confidence remains strong for local-disagreement detection (`dpg` AUROC 0.9001; `dpg_execution_trace` AUROC 0.9046), while low vote agreement is even stronger (`dpg` AUROC 0.9237; `dpg_execution_trace` AUROC 0.9808).
- Writing implication: present confidence as a robust diagnostic ranking index, but emphasize the broader DPG diagnostic family rather than the aggregate formula alone.

Implemented model-family support in the DPG runner:

- New file: `experiments_local_explanation/model_factory.py`
- Updated runner: `experiments_local_explanation/run_local_experiments.py`
- New option: `--model_families`
- Smoke output: `experiments_local_explanation/results_journal_v1/model_family_smoke/`
- Smoke passed for `random_forest`, `extra_trees`, `adaboost`, and `bagging`.
- `GradientBoostingClassifier` needs an explicit adapter before journal use because the current local code does not handle its 2D `estimators_` representation and its additive class-contribution semantics.

Focused model-family experiment is running under `nohup`.

- PID: `1957655`
- Log: `experiments_local_explanation/results_journal_v1/logs/model_family_dpg_supported_subset_20260619_091604.log`
- PID file: `experiments_local_explanation/results_journal_v1/logs/model_family_dpg_supported_subset_20260619_091604.pid`
- Output root: `experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset`

Experiment scope:

- Datasets: `banknote-authentication`, `iris`, `vehicle`, `madelon`, `isolet`
- Model families: `random_forest`, `extra_trees`, `adaboost`, `bagging`
- `n_estimators`: `20`
- `max_depth`: `4`
- Seed: `27`
- Graph mode: `execution_trace`
- Max test samples: `50`

Check status:

```bash
ps -p 1957655 -o pid,ppid,sid,stat,etime,cmd
tail -n 80 experiments_local_explanation/results_journal_v1/logs/model_family_dpg_supported_subset_20260619_091604.log
```

When finished, inspect:

```bash
sed -n '1,80p' experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset/summary.csv
```

Then create a compact model-family report from `summary.csv`, focusing on model accuracy, local match rate, confidence, edge recall, recombination, graph size, and runtime by `model_family`.
