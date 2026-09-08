# JOURNAL_v1 Run Manifest

This directory stores artifacts for the journal revision protocol.

## Protocol Files

- Config: `experiments_local_explanation/configs/journal_v1.yaml`
- Split generator: `experiments_local_explanation/create_journal_splits.py`
- Hardware profiler: `experiments_local_explanation/profile_journal_hardware.py`
- Journal parallel launcher: `experiments_local_explanation/run_journal_parallel.py`
- Split registry: `experiments_local_explanation/results_journal_v1/splits/split_registry.json`
- Split summary: `experiments_local_explanation/results_journal_v1/splits/split_summary.csv`
- Hardware profile: `experiments_local_explanation/results_journal_v1/hardware/hardware_profile.json`
- Hardware recommendation: `experiments_local_explanation/results_journal_v1/hardware/hardware_recommendation.md`

## Split Policy

The existing prepared test arrays are preserved as final test data. The split
registry partitions each dataset's existing `X_train.npy` / `y_train.npy` rows
into:

- `train_core`: model fitting for configuration selection runs.
- `validation`: configuration selection only.
- `test`: the existing prepared `X_test.npy` / `y_test.npy`, untouched until
  final reporting.

## Initial Split Command

```bash
python3 experiments_local_explanation/create_journal_splits.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results_journal_v1/splits \
  --validation_size 0.20 \
  --seed 20260617
```

## Next Protocol Tasks

1. Add runner support for `split_registry.json` so validation selection and
   final test evaluation are separated in code.
2. Add a common explainer-output mapping table for DPG-local, DPG-global,
   TreeSHAP, LIME, Anchors, tree-path, ICE, and path controls.
3. Run a smoke protocol on `iris`, `vehicle`, and `spambase`.
4. Launch full validation-grid runs only after the smoke protocol is clean.

## Completed Smoke Checks

### 2026-06-17: Hardware Profile

Command:

```bash
python3 experiments_local_explanation/profile_journal_hardware.py \
  --out_dir experiments_local_explanation/results_journal_v1/hardware
```

Result:

- CPU: Intel Core i9-10980XE, 18 physical cores / 36 logical CPUs.
- Python affinity: 36 CPUs.
- Memory: about 46 GiB available at profiling time.
- Disk: about 320 GiB free on the project filesystem at profiling time.
- Native thread pools: OpenBLAS/OpenMP default to 36 threads, so multi-process runs must cap native threads.

Recommended launch policy:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

- DPG-only validation/final runs: use up to 12 workers with `--rf_n_jobs 1`.
- Baseline runs including SHAP/LIME/Anchors: use up to 6 workers with `--rf_n_jobs 1`.
- Re-profile before full runs if load average or available memory changes substantially.

### 2026-06-17: Split-Aware Runner Smoke

DPG validation smoke:

```bash
python3 experiments_local_explanation/run_local_experiments.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results_journal_v1/smoke_dpg_validation \
  --datasets iris \
  --n_estimators 5 \
  --max_depth 2 \
  --perc_var 0.0001 \
  --decimal_threshold 6 \
  --seeds 27 \
  --graph_construction_modes execution_trace \
  --max_test_samples 5 \
  --progress_every 0 \
  --split_registry experiments_local_explanation/results_journal_v1/splits/split_registry.json \
  --train_split train_core \
  --eval_split validation \
  --rf_n_jobs 1
```

Baseline validation smoke:

```bash
python3 experiments_local_explanation/run_baseline_experiments.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results_journal_v1/smoke_baselines_validation \
  --datasets iris \
  --methods tree_path \
  --n_estimators 5 \
  --max_depth 2 \
  --perc_var 0.0001 \
  --decimal_threshold 6 \
  --seeds 27 \
  --max_test_samples 5 \
  --progress_every 0 \
  --split_registry experiments_local_explanation/results_journal_v1/splits/split_registry.json \
  --train_split train_core \
  --eval_split validation \
  --rf_n_jobs 1
```

Result:

- Both commands completed successfully.
- Output rows include `train_split=train_core`, `eval_split=validation`, and the resolved split registry path.
- Iris validation size is reported as `n_test_total=24`, matching `split_summary.csv`.
- Split-aware checkpoint filenames include `trtrain_core__evvalidation`.

Selection smoke:

```bash
python3 experiments_local_explanation/select_journal_configs.py \
  --dpg_summaries experiments_local_explanation/results_journal_v1/smoke_dpg_validation/summary.csv \
  --baseline_summaries experiments_local_explanation/results_journal_v1/smoke_baselines_validation/summary_baselines.csv \
  --out_dir experiments_local_explanation/results_journal_v1/smoke_selection
```

Result:

- Selection completed successfully.
- `smoke_selection/selected_configs.csv` contains 2 rows:
  one DPG-local row and one `tree_path` baseline row.
- Method-specific outputs are also written:
  `selected_dpg_configs.csv` and `selected_baseline_configs.csv`.
- Selection metadata includes `selection_kind`, `selection_metric_primary`,
  `selection_tie_breakers`, and `selection_rank`.

### 2026-06-17: Multi-CPU Journal Launcher Smoke

DPG parallel validation smoke:

```bash
python3 experiments_local_explanation/run_journal_parallel.py \
  --kind dpg \
  --phase validation \
  --dataset_set smoke \
  --parallel 2 \
  --n_estimators 5 \
  --max_depth 2 \
  --perc_var 0.0001 \
  --seeds 27 \
  --graph_construction_modes execution_trace \
  --max_test_samples 3 \
  --progress_every 0 \
  --out_root experiments_local_explanation/results_journal_v1/smoke_parallel_dpg_validation
```

Baseline parallel validation smoke:

```bash
python3 experiments_local_explanation/run_journal_parallel.py \
  --kind baselines \
  --phase validation \
  --dataset_set smoke \
  --parallel 2 \
  --methods tree_path \
  --n_estimators 5 \
  --max_depth 2 \
  --perc_var 0.0001 \
  --seeds 27 \
  --max_test_samples 3 \
  --progress_every 0 \
  --out_root experiments_local_explanation/results_journal_v1/smoke_parallel_baselines_validation
```

Selection over parallel smoke outputs:

```bash
python3 experiments_local_explanation/select_journal_configs.py \
  --dpg_summaries \
    experiments_local_explanation/results_journal_v1/smoke_parallel_dpg_validation/iris/summary.csv \
    experiments_local_explanation/results_journal_v1/smoke_parallel_dpg_validation/vehicle/summary.csv \
    experiments_local_explanation/results_journal_v1/smoke_parallel_dpg_validation/spambase/summary.csv \
  --baseline_summaries \
    experiments_local_explanation/results_journal_v1/smoke_parallel_baselines_validation/iris/summary_baselines.csv \
    experiments_local_explanation/results_journal_v1/smoke_parallel_baselines_validation/vehicle/summary_baselines.csv \
    experiments_local_explanation/results_journal_v1/smoke_parallel_baselines_validation/spambase/summary_baselines.csv \
  --out_dir experiments_local_explanation/results_journal_v1/smoke_parallel_selection
```

Result:

- DPG and baseline parallel smoke runs completed successfully on `iris`, `vehicle`, and `spambase`.
- Worker processes used `train_split=train_core`, `eval_split=validation`, `rf_n_jobs=1`, and native thread caps.
- `smoke_parallel_selection/selected_configs.csv` contains 6 rows: DPG plus `tree_path` for each smoke dataset.

### 2026-06-17: Moderate DPG Validation Grid

Hardware-aware DPG validation run over the curated 15-dataset panel:

```bash
python3 experiments_local_explanation/run_journal_parallel.py \
  --kind dpg \
  --phase validation \
  --dataset_set curated_15 \
  --parallel 12 \
  --n_estimators 20 \
  --max_depth 4 \
  --perc_var 0.0001,0.00001 \
  --decimal_threshold 6 \
  --seeds 27,42 \
  --graph_construction_modes aggregated_transitions,execution_trace \
  --max_test_samples 0 \
  --progress_every 100 \
  --out_root experiments_local_explanation/results_journal_v1/dpg_validation_moderate_by_dataset
```

Selection:

```bash
python3 experiments_local_explanation/select_journal_configs.py \
  --dpg_summaries experiments_local_explanation/results_journal_v1/dpg_validation_moderate_by_dataset/*/summary.csv \
  --out_dir experiments_local_explanation/results_journal_v1/dpg_validation_moderate_selection
```

Result:

- Run completed successfully using `parallel=12`, `rf_n_jobs=1`, and native thread caps from the hardware profile.
- Output directory: `dpg_validation_moderate_by_dataset/`.
- Integrity check: 15 dataset summaries, 120 validation rows total, 8 rows per dataset.
- Each dataset includes both `aggregated_transitions` and `execution_trace` graph construction modes.
- Selection output: `dpg_validation_moderate_selection/selected_dpg_configs.csv`.
- Selection contains 30 rows: one DPG validation choice per dataset and graph construction mode.

### 2026-06-18: Moderate Baseline Validation Grid

Hardware-aware baseline validation run over the curated 15-dataset panel:

```bash
python3 experiments_local_explanation/run_journal_parallel.py \
  --kind baselines \
  --phase validation \
  --dataset_set curated_15 \
  --parallel 6 \
  --methods shap,lime,anchors,tree_path,ice \
  --n_estimators 20 \
  --max_depth 4 \
  --perc_var 0.0001,0.00001 \
  --decimal_threshold 6 \
  --seeds 27,42 \
  --max_test_samples 0 \
  --progress_every 100 \
  --out_root experiments_local_explanation/results_journal_v1/baselines_validation_moderate_by_dataset
```

The final resume was launched as a detached process with:

```bash
experiments_local_explanation/launch_nohup.sh \
  --name baselines_validation_moderate_resume \
  -- python3 experiments_local_explanation/run_journal_parallel.py ...
```

Result:

- Run completed successfully using `parallel=6`, `rf_n_jobs=1`, and native thread caps.
- Output directory: `baselines_validation_moderate_by_dataset/`.
- Integrity check: 15 dataset summaries, 300 validation rows total, 20 rows per dataset.
- Baseline methods completed: `shap`, `lime`, `anchors`, `tree_path`, `ice`.
- `ice` is retained as a diagnostic/appendix artifact, not as a main comparator.

Main validation selection, excluding `ice`:

```bash
python3 experiments_local_explanation/select_journal_configs.py \
  --dpg_summaries experiments_local_explanation/results_journal_v1/dpg_validation_moderate_by_dataset/*/summary.csv \
  --baseline_summaries experiments_local_explanation/results_journal_v1/baselines_validation_moderate_by_dataset/*/summary_baselines.csv \
  --exclude_baseline_methods ice \
  --out_dir experiments_local_explanation/results_journal_v1/validation_selection_main_noice
```

Selection result:

- `validation_selection_main_noice/selected_configs.csv` contains 90 rows.
- DPG selections: 30 rows, one per dataset and graph construction mode.
- Baseline selections: 60 rows, one per dataset for each of `shap`, `lime`, `anchors`, and `tree_path`.

### 2026-06-18: LORE-Style Baseline Addition

Implemented a lightweight `lore` baseline in `run_baseline_experiments.py`.
The method generates a local synthetic neighborhood around each evaluation
sample, labels that neighborhood with the trained random forest, fits a shallow
local decision-tree surrogate, reports the surrogate path rule, and records a
nearest opposite-label counterfactual summary.

Smoke validation:

```bash
python3 experiments_local_explanation/run_baseline_experiments.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results_journal_v1/smoke_lore_validation \
  --datasets iris \
  --methods lore \
  --n_estimators 5 \
  --rf_n_jobs 1 \
  --max_depth 2 \
  --perc_var 0.0001 \
  --seeds 27 \
  --max_test_samples 5 \
  --split_registry experiments_local_explanation/results_journal_v1/splits/split_registry.json \
  --train_split train_core \
  --eval_split validation \
  --progress_every 1 \
  --overwrite
```

Smoke result:

- Completed successfully with zero local failures.
- Per-sample outputs include `lore_rule`, `lore_rule_length`,
  `lore_fidelity_neighborhood`, `lore_counterfactual_distance`, and
  `lore_counterfactual_features`.

Curated 15-dataset LORE validation launched as detached job:

```bash
experiments_local_explanation/launch_nohup.sh \
  --name lore_validation_moderate \
  -- python3 experiments_local_explanation/run_journal_parallel.py \
    --kind baselines \
    --phase validation \
    --dataset_set curated_15 \
    --parallel 6 \
    --methods lore \
    --n_estimators 20 \
    --max_depth 4 \
    --perc_var 0.0001,0.00001 \
    --decimal_threshold 6 \
    --seeds 27,42 \
    --max_test_samples 0 \
    --progress_every 100 \
    --out_root experiments_local_explanation/results_journal_v1/lore_validation_moderate_by_dataset
```

Tracking:

- PID file: `logs/lore_validation_moderate_20260618_084303.pid`.
- Log file: `logs/lore_validation_moderate_20260618_084303.log`.
- Status at launch check: running as detached PID `1902888`; early dataset summaries were already being written.

### 2026-06-18: Final Validation Selection With LORE

Combined validation selection including `lore` and excluding diagnostic `ice`:

```bash
python3 experiments_local_explanation/select_journal_configs.py \
  --dpg_summaries experiments_local_explanation/results_journal_v1/dpg_validation_moderate_by_dataset/*/summary.csv \
  --baseline_summaries \
    experiments_local_explanation/results_journal_v1/baselines_validation_moderate_by_dataset/*/summary_baselines.csv \
    experiments_local_explanation/results_journal_v1/lore_validation_moderate_by_dataset/*/summary_baselines.csv \
  --exclude_baseline_methods ice \
  --out_dir experiments_local_explanation/results_journal_v1/validation_selection_main_with_lore_noice
```

Result:

- `selected_configs.csv` contains 105 rows.
- DPG selections: 30 rows, one per dataset and graph construction mode.
- Baseline selections: 75 rows, one per dataset for each of `shap`, `lime`,
  `anchors`, `tree_path`, and `lore`.

### 2026-06-18: Locked Final Test Launch

Added `run_selected_journal_test.py` to evaluate only validation-selected
configurations on the locked final test split:

- Training split: `train_core_validation`.
- Evaluation split: `test`.
- Selection source:
  `validation_selection_main_with_lore_noice/selected_configs.csv`.

Smoke tests:

- DPG smoke: 3 selected DPG configurations, 2 test samples each, completed.
- Baseline smoke: selected `shap` and `lore` configurations, 2 test samples
  each, completed.

Detached final test command:

```bash
experiments_local_explanation/launch_nohup.sh \
  --name final_test_main_with_lore_noice \
  -- python3 experiments_local_explanation/run_selected_journal_test.py \
    --selection experiments_local_explanation/results_journal_v1/validation_selection_main_with_lore_noice/selected_configs.csv \
    --out_root experiments_local_explanation/results_journal_v1/final_test_main_with_lore_noice \
    --parallel 6 \
    --max_test_samples 0 \
    --progress_every 100
```

Tracking:

- PID file: `logs/final_test_main_with_lore_noice_20260618_092136.pid`.
- Log file: `logs/final_test_main_with_lore_noice_20260618_092136.log`.
- Status at launch check: running as detached PID `1906745`; early DPG test
  runs were already completing and writing summaries.
