# Journal Reproducibility Package

Date: 2026-06-19

This file is the compact audit map for the journal revision experiments. The older `RUN_MANIFEST.md` keeps chronological launch notes; this file points to the manuscript-ready evidence and the commands needed to regenerate it.

## Hardware Policy

Hardware profile:

- `experiments_local_explanation/results_journal_v1/hardware/hardware_profile.json`
- `experiments_local_explanation/results_journal_v1/hardware/hardware_recommendation.md`

Profiled server:

- CPU: Intel Core i9-10980XE.
- Available logical CPUs: 36.
- Recommended DPG workers: 12.
- Recommended baseline workers: 6.
- Recommended `--rf_n_jobs`: 1.

Before launching multi-process jobs:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

## Final-Test Evidence

Main selected final-test outputs:

- `experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/summary_selected_test.csv`
- `experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/per_sample_selected_test.csv`

Main journal report:

- `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/summary.md`
- `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/method_summary_ci.csv`
- `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/size_runtime_summary_ci.csv`
- `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/paired_tests_holm.csv`
- `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/diagnostic_value_summary.csv`
- `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/common_explainer_interface.csv`
- `experiments_local_explanation/results_journal_v1/journal_report_final_with_path_controls_and_critical/failure_focus_dataset_method.csv`

Interpretation:

- Use DPG as a structure-aware local diagnostic object, not as a universal replacement for SHAP/LIME/Anchors/LORE.
- Use ICE only as appendix/diagnostic material, not as a main comparator.
- Treat path controls as the main structural baseline: they can be output-faithful but are larger and less semantically organized.

## Additional Journal Analyses

Confidence sensitivity:

- Script: `experiments_local_explanation/analyze_confidence_sensitivity.py`
- Report: `experiments_local_explanation/results_journal_v1/confidence_sensitivity/summary.md`

Critical-node validation:

- Report: `experiments_local_explanation/results_journal_v1/critical_node_validation_final_fixed/summary.md`
- Dataset-property analysis: `experiments_local_explanation/results_journal_v1/critical_node_property_analysis/summary.md`

Scalability:

- Script: `experiments_local_explanation/analyze_scalability_depth_experiment.py`
- Report: `experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_report/summary.md`

DPG-supported model families:

- Implementation: `experiments_local_explanation/model_factory.py`
- Runner: `experiments_local_explanation/run_local_experiments.py`
- Report script: `experiments_local_explanation/analyze_model_family_experiment.py`
- Report: `experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset_report/summary.md`

Dataset/model-regime synthesis:

- Script: `experiments_local_explanation/analyze_dataset_model_regimes.py`
- Report: `experiments_local_explanation/results_journal_v1/dataset_model_regime_synthesis/summary.md`
- Table: `experiments_local_explanation/results_journal_v1/dataset_model_regime_synthesis/dataset_model_regime_table.csv`

## Rerun Commands

Regenerate confidence sensitivity:

```bash
./.venv/bin/python experiments_local_explanation/analyze_confidence_sensitivity.py \
  --input experiments_local_explanation/results_journal_v1/final_test_main_with_lore_path_controls/per_sample_selected_test.csv \
  --out_dir experiments_local_explanation/results_journal_v1/confidence_sensitivity
```

Regenerate model-family report:

```bash
./.venv/bin/python experiments_local_explanation/analyze_model_family_experiment.py \
  --summary_csv experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset/summary.csv \
  --out_dir experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset_report
```

Generate the full 15-dataset model-family report after the detached run completes:

```bash
./.venv/bin/python experiments_local_explanation/analyze_model_family_experiment.py \
  --input_root experiments_local_explanation/results_journal_v1/model_family_all15_test \
  --out_dir experiments_local_explanation/results_journal_v1/model_family_all15_test_report
```

Regenerate dataset/model-regime synthesis:

```bash
./.venv/bin/python experiments_local_explanation/analyze_dataset_model_regimes.py \
  --out_dir experiments_local_explanation/results_journal_v1/dataset_model_regime_synthesis
```

Rerun DPG-supported model-family subset with server-aware parallelism:

```bash
experiments_local_explanation/launch_nohup.sh \
  --name model_family_dpg_supported_subset \
  -- ./.venv/bin/python experiments_local_explanation/run_local_experiments.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results_journal_v1/model_family_dpg_supported_subset \
  --datasets banknote-authentication,iris,vehicle,madelon,isolet \
  --model_families random_forest,extra_trees,adaboost,bagging \
  --n_estimators 20 \
  --max_depth 4 \
  --perc_var 0.0001 \
  --decimal_threshold 6 \
  --seeds 27 \
  --graph_construction_modes execution_trace \
  --max_test_samples 50 \
  --rf_n_jobs 1 \
  --progress_every 10
```

Rerun full 15-dataset DPG-supported model-family final-test experiment:

```bash
experiments_local_explanation/launch_nohup.sh \
  --name model_family_all15_test \
  -- ./.venv/bin/python experiments_local_explanation/run_journal_parallel.py \
  --kind dpg \
  --phase test \
  --dataset_set curated_15 \
  --parallel 12 \
  --model_families random_forest,extra_trees,adaboost,bagging \
  --n_estimators 20 \
  --max_depth 4 \
  --perc_var 0.0001 \
  --decimal_threshold 6 \
  --seeds 27 \
  --graph_construction_modes execution_trace \
  --max_test_samples 0 \
  --progress_every 250 \
  --out_root experiments_local_explanation/results_journal_v1/model_family_all15_test
```

## Manuscript Mapping

Recommended result placement:

- Main results: final-test report, path controls, confidence/disagreement AUROC, size/runtime table.
- Ablation/sensitivity: confidence sensitivity, path controls, model-family subset.
- Scalability: depth/ensemble stress report.
- Failure modes: dataset/model-regime synthesis, with `isolet`, `digits`, `madelon`, and `vehicle` highlighted.
- Appendix: ICE, common explainer interface table, critical-node exploratory analysis, full dataset tables.

## Claim Boundaries

Use:

- DPGs enable local diagnostic analysis of tree-ensemble decisions.
- Execution-trace DPGs recover decision traces while exposing support margin, competitor exposure, confidence, graph size, and runtime diagnostics.
- DPG diagnostic utility is regime-dependent.

Avoid:

- Claiming DPG is universally better than SHAP, LIME, Anchors, or LORE.
- Claiming critical nodes are causal.
- Claiming GradientBoosting is validated in this repository before an adapter is implemented.
- Treating construction checks alone as explanatory value.
