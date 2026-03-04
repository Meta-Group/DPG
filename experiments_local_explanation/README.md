# DPG Local Explanation Experiments

This folder provides the experiment pipeline used for the DPG fidelity and DPG 2.0 graph-construction studies on numeric tabular classification datasets.

## Scope

Current implementation supports:
- Dataset preparation (`prepare_datasets.py`)
- DPG local explanation sweeps (`run_local_experiments.py`)
- Per-dataset orchestration (`run_per_dataset.py`)
- Baseline comparison runs for `shap`, `lime`, and `ice`
- DPG 2.0 graph-construction experiments (`aggregated_transitions` vs `execution_trace`)
- DPG 2.0 next-phase metrics for path faithfulness, explanation confidence, and misclassification cohorts
- CSV outputs for per-run and per-sample analysis

## Folder Structure

- `prepare_datasets.py`: downloads/prepares numeric datasets and saves train/test splits
- `run_local_experiments.py`: runs model + DPG local explanation sweeps
- `data_numeric/`: prepared datasets (generated)
- `results/`: experiment outputs (generated)

## 1. Prepare Datasets

From `experiments_local_explanation/`:

```bash
python3 prepare_datasets.py --out_dir data_numeric --standardize --max_datasets 20
```

Optional row cap for large datasets:

```bash
python3 prepare_datasets.py --out_dir data_numeric --standardize --max_datasets 20 --max_rows 20000
```

Expected output for each dataset:

- `data_numeric/<dataset>/X_train.npy`
- `data_numeric/<dataset>/y_train.npy`
- `data_numeric/<dataset>/X_test.npy`
- `data_numeric/<dataset>/y_test.npy`
- `data_numeric/<dataset>/feature_names.json`
- `data_numeric/<dataset>/meta.json`

## 2. Run Local Explanation Experiments

From repository root:

```bash
PYTHONPATH=. python3 experiments_local_explanation/run_local_experiments.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results \
  --datasets iris,wine,breast_cancer,digits \
  --n_estimators 10,20 \
  --max_depth 2,4,None \
  --perc_var 0.0,0.001 \
  --decimal_threshold 4,6 \
  --seeds 27,42 \
  --max_test_samples 150
```

Minimal smoke run:

```bash
PYTHONPATH=. python3 experiments_local_explanation/run_local_experiments.py \
  --datasets iris \
  --n_estimators 5 \
  --max_depth 2 \
  --perc_var 0.0 \
  --decimal_threshold 6 \
  --seeds 27 \
  --max_test_samples 10
```

Resume behavior:

- Checkpoints are saved after each `(dataset, config, seed)` run in `results/checkpoints/`.
- Re-running the same command resumes automatically and skips completed runs.
- Use `--no-resume` to force ignoring checkpoints for the current launch.
- Use `--overwrite` to recompute all runs and refresh checkpoints.

## 3. Outputs

The script writes:

- `results/summary.csv`: one row per `(dataset, config, seed)`
- `results/per_sample.csv`: one row per explained sample
- `results/dataset_overview.csv`: compact per-dataset overview

Useful columns in `summary.csv`:

- `model_accuracy`
- `local_matches_model_rate` (local fidelity to model prediction)
- `local_accuracy` (local prediction vs true class)
- `avg_evidence_margin_pred_vs_competitor`
- `avg_num_paths`
- `avg_num_active_nodes`
- `avg_node_recall`, `avg_edge_precision`, `avg_recombination_rate`
- `avg_path_purity`, `avg_competitor_exposure`
- `avg_explanation_confidence`, `avg_critical_split_depth`

## 4. Research Questions Supported

With current outputs, you can already study:

1. Fidelity: how often the local DPG prediction matches the model prediction.
2. Accuracy: how often the local DPG prediction matches the ground truth.
3. Graph-construction effects: `aggregated_transitions` vs `execution_trace`.
4. Complexity tradeoff: path/node counts across DPG settings (`perc_var`, `decimal_threshold`).
5. Trace coverage diagnostics for DPG 2.0 execution-trace explanations.
6. Path-level faithfulness via node/edge recall and precision.
7. Recombination artifacts via `recombination_rate`.
8. Misclassification/disagreement cohorts via `cohort_label`.

## 5. Reproducibility Notes

- Keep the same seeds across runs when comparing settings.
- Use the same `max_test_samples` when comparing datasets/configurations.
- If OpenML availability changes, rely on `RUN_REPORT.txt` in the dataset output folder.

## 6. Troubleshooting

- If `python` is not found, use `python3`.
- If OpenML download fails, re-run later or reduce target datasets.
- If plotting dependencies are missing, note that this experiment script only writes CSV files and does not require graph rendering.

## 7. How to Run

### 7.1 Prepare datasets (numeric-only)

```bash
python prepare_datasets.py --out_dir data_numeric --standardize --max_datasets 20
```

Optional (avoid huge datasets during development):

```bash
python prepare_datasets.py --out_dir data_numeric --standardize --max_datasets 20 --max_rows 20000
```

### 7.2 Run DPG 2.0 graph-construction experiments

```bash
bash experiments_local_explanation/run_selected_datasets_dpg2_graph_construction.sh
```

### 7.2b Run DPG 2.0 next-phase experiments on the curated 15 datasets

```bash
bash experiments_local_explanation/run_selected_datasets_dpg2_next_phase.sh
```

Aggregate the per-dataset outputs and build best-config cohort summaries:

```bash
python3 experiments_local_explanation/analyze_dpg2_next_phase.py \
  --out_root experiments_local_explanation/experiment_dpg2_next_phase \
  --analysis_dir experiments_local_explanation/experiment_dpg2_next_phase/_analysis
```

One dataset per CPU with `tmux`:

```bash
bash experiments_local_explanation/run_selected_datasets_dpg2_graph_construction_tmux_per_cpu.sh
```

### 7.3 Run one process per dataset

This launcher executes `run_local_experiments.py` once per dataset and stores outputs in separate folders.

Sequential:

```bash
python3 experiments_local_explanation/run_per_dataset.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_root experiments_local_explanation/experiment_dpg2_graph_construction \
  --parallel 1
```

Parallel (example with 4 dataset processes):

```bash
python3 experiments_local_explanation/run_per_dataset.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_root experiments_local_explanation/experiment_dpg2_graph_construction \
  --parallel 4
```

### 7.4 Run baseline comparisons (SHAP, LIME, ICE)

Single process:

```bash
PYTHONPATH=. python3 experiments_local_explanation/run_baseline_experiments.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results_baselines \
  --methods shap,lime,ice
```

One process per dataset (parallel):

```bash
PYTHONPATH=. python3 experiments_local_explanation/run_per_dataset_baselines.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_root experiments_local_explanation/results_baselines_by_dataset \
  --methods shap,lime,ice \
  --parallel 4
```

Note:
- `shap` and `lime` Python packages must be installed for those methods.
- `ice` here is implemented as a per-sample response-curve diagnostic (feature perturbation curve), not route/path competition evidence.

### 7.5 Telegram Progress Notifications

You can monitor the DPG 2.0 and baseline runs and send progress updates to Telegram.

Set credentials:

```bash
export TELEGRAM_BOT_TOKEN="<your_bot_token>"
export TELEGRAM_CHAT_ID="<your_chat_id>"
```

One-shot message:

```bash
python3 experiments_local_explanation/monitor_progress_telegram.py --one_shot
```

Continuous monitoring (every 5 minutes, send only on progress change):

```bash
python3 experiments_local_explanation/monitor_progress_telegram.py \
  --interval_sec 300 \
  --baseline_methods shap,lime,ice \
  --notify_on_change_only
```

## 8. Methodological Notes and Reproducibility

- Keep the DPG 2.0 experiment fixed on the same local evidence rule when comparing graph-construction modes.
- Compare `aggregated_transitions` and `execution_trace` on the same datasets, seeds, RF hyperparameters, and `perc_var`.
- Fix global seeds for dataset split and explainer randomness where applicable.
- CKI neighbor sampling
- Use paired evaluation: the same test points are explained by all methods.
- Prefer bootstrapped confidence intervals and paired tests (Wilcoxon signed-rank) for metric comparisons.
- Report all budgets (perturbation samples, KernelSHAP background size, etc.) explicitly.

## 9. Expected Outputs

Per dataset x model x explainer:

- Faithfulness: deletion/insertion AUC, AOPC
- Stability: rank correlation and top-k overlap across runs
- Complexity: size of explanations and DPG local-subgraph stats
- Contrastiveness: evidence scores and margins + ambiguity correlation
- Runtime: time/explanation + model calls

Artifacts:

- `results/tables/metrics.csv` (long-format table)
- `results/plots/` (curves, boxplots, scatter correlation)
- `results/logs/` (config snapshots, seeds, versions)

## 10. Interpreting Results (Quick Guide)

- If DPG evidence shows better deletion AUC and high stability, it is a strong sign of faithful and reliable local explanations.
- If DPG evidence has high `E_{ŷ}` but also high competitor evidence (small margin), label the prediction as ambiguous.
- On errors, if `E_y > E_{ŷ}` frequently, the method provides useful conflict diagnostics (model is internally leaning toward the true class).
- If KernelSHAP outperforms on faithfulness but DPG is more contrastive/diagnostic, you can argue complementary strengths.

## 11. Citation Notes (for the paper)

Attraction-repulsion evidence is a project-specific composite metric inspired by:

- margin/contrast ideas in ensembles
- rule/path contribution aggregation
- centrality-based influence (LRC)

It is not a verbatim metric from one single paper.

## License / Usage

Internal experimental branch for DPG development and evaluation.

If needed, planned follow-up files are:

- `experiments/config.yaml` template (budgets, kNN params, explainer params)
- `experiments/run_all.py` skeleton (iterates datasets/models/explainers and writes one `metrics.csv`)
- `experiments/cki.py` minimal CKI implementation aligned with this README
