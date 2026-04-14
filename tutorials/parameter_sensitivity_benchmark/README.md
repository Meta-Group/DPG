# DPG Parameter Sensitivity Benchmark

This tutorial folder packages a cross-dataset benchmark focused on the two main DPG tuning parameters:

- `perc_var`
- `decimal_threshold`

The goal is to help DPGExplainer practitioners understand how these parameters affect:

- graph size
- pruning behavior
- invalid regions where the graph collapses
- stability of feature-level LRC signals

## What Is Included

- [dpg_all_datasets_benchmark.ipynb](/Users/barbon/Python/DPG/tutorials/parameter_sensitivity_benchmark/dpg_all_datasets_benchmark.ipynb)
  Analysis notebook that loads the benchmark CSVs and visualizes the main patterns.
- [results/dataset_inventory.csv](/Users/barbon/Python/DPG/tutorials/parameter_sensitivity_benchmark/results/dataset_inventory.csv)
  Dataset metadata used in the benchmark.
- [results/benchmark_summary.csv](/Users/barbon/Python/DPG/tutorials/parameter_sensitivity_benchmark/results/benchmark_summary.csv)
  One row per `(dataset, perc_var, decimal_threshold)` run.
- [results/benchmark_feature_lrc.csv](/Users/barbon/Python/DPG/tutorials/parameter_sensitivity_benchmark/results/benchmark_feature_lrc.csv)
  Aggregated feature-level LRC values for each successful run.
- [results/benchmark_stability_vs_baseline.csv](/Users/barbon/Python/DPG/tutorials/parameter_sensitivity_benchmark/results/benchmark_stability_vs_baseline.csv)
  Top-10 feature overlap against the baseline configuration.

## Benchmark Scope

The benchmark was run across 18 prepared numeric datasets:

- `banknote-authentication`
- `breast_cancer`
- `diabetes`
- `digits`
- `ionosphere`
- `iris`
- `isolet`
- `madelon`
- `optdigits`
- `phoneme`
- `qsar-biodeg`
- `satimage`
- `segment`
- `spambase`
- `vehicle`
- `wdbc`
- `wine`
- `wine_quality`

Parameter grid:

- `perc_var in [1e-9, 1e-2, 3e-2, 5e-2]`
- `decimal_threshold in [1, 2, 3, 4]`

Model setup used for feasibility across all datasets:

- `n_estimators = 8`
- `max_depth = 5`
- `max_train_rows = 1200` for large datasets
- fixed `random_state = 27`

## High-Level Findings

### 1. `perc_var` is the main graph-pruning control

Across datasets, increasing `perc_var` reduced graph size much more aggressively than changing `decimal_threshold`.

Mean predicate counts across successful runs:

- `perc_var=1e-9, dt=1`: about `238`
- `perc_var=1e-2, dt=1`: about `78`
- `perc_var=3e-2, dt=1`: about `23`

Interpretation:

- use `perc_var` to control how much rare path behavior survives
- aggressive values can simplify the graph quickly
- overly aggressive values may remove all usable path variants

### 2. `decimal_threshold` usually reaches a plateau early, but it depends on variable precision

For many datasets, predicate count stops changing materially at `dt=3` or `dt=4`.

Examples from the benchmark:

- `iris`: `53 -> 60 -> 60 -> 60`
- `wine`: `117 -> 134 -> 134 -> 134`
- `satimage`: `319 -> 342 -> 342 -> 342`

Interpretation:

- `dt=2` often captures most of the useful structural detail
- `dt=3` is a reasonable upper bound when you want a more exact threshold view
- `dt=4` rarely adds much more than `dt=3`

Important caveat:

- `decimal_threshold` should not be treated as a universal constant
- it depends strongly on the precision and measurement granularity of the variables
- if features are naturally coarse or already rounded, a large `decimal_threshold` adds visual complexity without adding much meaning
- if features carry meaningful fine-grained decimal structure, a larger `decimal_threshold` may be justified

Practical interpretation:

- choose `decimal_threshold` to match the meaningful precision of the dataset, not just the model output
- use `2` as a general starting point, not as a fixed rule for every dataset

### 3. Invalid regions are part of the story

The failures in `benchmark_summary.csv` are informative, not infrastructure errors.

All failed runs raised:

- `There is no paths with the current value of perc_var and decimal_threshold!`

Observed pattern:

- many datasets fail at `perc_var=0.05`
- `digits` and `optdigits` already fail at `perc_var=0.03`

Interpretation:

- not every parameter pair is safe
- practitioners should treat very large `perc_var` values as dataset-dependent and risky

### 4. Feature-level stability is often better than graph-level stability

Using the baseline configuration:

- `perc_var = 1e-9`
- `decimal_threshold = 2`

the mean top-10 feature overlap remained high for several datasets:

- `banknote-authentication`: `1.000`
- `diabetes`: `1.000`
- `phoneme`: `1.000`
- `wine`: `0.967`
- `satimage`: `0.933`

More sensitive datasets included:

- `isolet`: `0.592`
- `qsar-biodeg`: `0.583`

Interpretation:

- graph size may change substantially while core features remain stable
- stability should be checked per dataset, especially in high-dimensional settings

## Practical Guidance

Recommended starting point for practitioners:

- use `perc_var = 0.01` as the practical default
- use `decimal_threshold = 2` as the initial default only when it is consistent with the variables' meaningful decimal precision
- lower `decimal_threshold` when features are coarse, discretized, or already rounded
- increase `decimal_threshold` only when exact threshold resolution matters and the variables support that precision
- use `perc_var = 1e-9` as a higher-fidelity alternative when you want to preserve rare path behavior
- increase `perc_var` only when the graph is too dense to inspect

Suggested rule of thumb:

- `perc_var=0.01` is the recommended practitioner default
- `decimal_threshold=2` is a strong starting point for many continuous datasets, but it must be validated against variable precision
- `decimal_threshold=3` is useful for threshold audits on finer-grained variables
- `perc_var>=0.03` should be treated cautiously
- `perc_var=0.05` is often too aggressive

## How To Use The Notebook

Open the notebook:

- [dpg_all_datasets_benchmark.ipynb](/Users/barbon/Python/DPG/tutorials/parameter_sensitivity_benchmark/dpg_all_datasets_benchmark.ipynb)

The notebook is analysis-only in its committed form:

- it reads the CSV outputs stored in `results/`
- it does not need the temporary workspace used to generate them
- it does not modify the repository

This makes it safe to open and reuse as a practitioner-facing reference artifact.
