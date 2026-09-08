# DPG-Supported Classification Model Family Experiment

Date: 2026-06-19

## Motivation

The DPG documentation lists multiple supported sklearn classification ensemble families. This gives a better journal-scope argument than only using RandomForest:

> DPG-based local diagnostics can be evaluated across the classification model families supported by the DPG framework.

Documentation checked:

- https://dpg.readthedocs.io/en/latest/supported_models.html

Documented classification models:

- `RandomForestClassifier`
- `GradientBoostingClassifier`
- `ExtraTreesClassifier`
- `AdaBoostClassifier`
- `BaggingClassifier`

## Implementation Added

Model factory:

- `experiments_local_explanation/model_factory.py`

Runner support:

- `experiments_local_explanation/run_local_experiments.py`
- New CLI option: `--model_families`
- Default remains `random_forest`, so old commands remain compatible.
- Output CSVs now include `model_family`.

Supported aliases include:

- `random_forest`, `rf`
- `extra_trees`, `extratrees`
- `gradient_boosting`, `gb`, `gbc`
- `adaboost`, `ada`
- `bagging`, `bag`

## Smoke Test Completed

Command:

```bash
./.venv/bin/python experiments_local_explanation/run_local_experiments.py \
  --data_dir experiments_local_explanation/data_numeric \
  --out_dir experiments_local_explanation/results_journal_v1/model_family_smoke \
  --datasets iris \
  --model_families random_forest,extra_trees,adaboost,bagging \
  --n_estimators 3 \
  --max_depth 2 \
  --perc_var 0.0001 \
  --decimal_threshold 6 \
  --seeds 27 \
  --graph_construction_modes execution_trace \
  --max_test_samples 3 \
  --rf_n_jobs 1 \
  --progress_every 1
```

Outputs:

- `experiments_local_explanation/results_journal_v1/model_family_smoke/summary.csv`
- `experiments_local_explanation/results_journal_v1/model_family_smoke/per_sample.csv`
- `experiments_local_explanation/results_journal_v1/model_family_smoke/dataset_overview.csv`

Smoke result:

- `random_forest`: passed.
- `extra_trees`: passed.
- `adaboost`: passed.
- `bagging`: passed.

All four working smoke runs had:

- local failures: 0.
- edge recall: approximately 1.0.
- recombination rate: 0.0 under execution-trace mode.

## GradientBoosting Status

`GradientBoostingClassifier` is documented online as supported, but the local code in this checkout currently fails in DPG extraction because `GradientBoostingClassifier.estimators_` is a 2D array and the extraction code iterates it as if each element were directly a tree estimator.

Observed failure:

```text
AttributeError: 'numpy.ndarray' object has no attribute 'tree_'
```

There is also a semantic issue: boosting estimators are additive class-contribution trees, not majority-vote trees. Therefore, even after flattening `estimators_`, the local class-support interpretation needs a specific adapter. We should not silently include GradientBoosting in the journal experiment until this adapter is implemented and validated.

## Recommended Next Experiment

After confidence sensitivity and dataset-regime analysis, run a focused model-family experiment on the four currently smoke-tested families:

- `random_forest`
- `extra_trees`
- `adaboost`
- `bagging`

Representative datasets:

- `banknote-authentication`
- `iris`
- `vehicle`
- `madelon`
- `isolet`

Suggested command:

```bash
./.venv/bin/python experiments_local_explanation/run_local_experiments.py \
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

## Writing Guidance

Use the model-family result to support this claim:

> DPG-local is not only a RandomForest explanation object; it is a DPG-enabled local diagnostic analysis that can be applied to several DPG-compatible sklearn ensemble classifiers.

Avoid:

- "all tree ensembles"
- "all DPG-documented models" until GradientBoosting adapter support is fixed.
- direct equivalence between bagging/forest vote semantics and boosting additive semantics.

Preferred wording before GradientBoosting is fixed:

> We evaluate DPG-local across four DPG-compatible sklearn classification ensemble families with direct tree-estimator semantics: RandomForest, ExtraTrees, AdaBoost with tree base learners, and Bagging with tree base learners. GradientBoosting requires a separate class-contribution adapter and is left for future extension.
