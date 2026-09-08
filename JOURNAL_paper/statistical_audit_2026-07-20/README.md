# Statistical Audit Exports, 2026-07-20

This directory contains the raw tables needed to compute seed-level variance, confidence intervals, and properly corrected paired statistical tests for the ECML-to-journal revision.

## Source Files

- DPG source: `experiments_local_explanation/experiment_dpg2_next_phase/`
- DPG selected configurations: `experiments_local_explanation/experiment_dpg2_next_phase/_analysis/best_configs.csv`
- Baseline source: `experiments_local_explanation/results_baselines_by_dataset/<dataset>/summary_baselines.csv`
- Reproducibility logic checked against: `experiments_local_explanation/RQ_dpg2_paper_reproducibility.ipynb`
- Export script: `experiments_local_explanation/export_seed_ci_and_corrected_tests.py`

## Exported Tables

- `dpg_all_seed_config_metrics.csv`: all DPG summary rows for the 15 journal datasets, before best-configuration selection.
- `baseline_all_seed_config_metrics.csv`: all baseline summary rows for the same 15 datasets, before best-configuration selection.
- `selected_per_seed_metrics.csv`: seed-level rows for the selected configuration family. The selected configuration is matched after removing the seed suffix, so both seeds 27 and 42 are recovered for each dataset and method.
- `selected_seed_ci_by_dataset_method.csv`: mean, standard deviation, and 95% t confidence-interval half-width across seeds 27 and 42 for each dataset and method.
- `paired_quant_selected_long.csv`: exact notebook-style paired table used for Wilcoxon tests, with one selected row per dataset and method.
- `paired_quant_fidelity_pivot.csv`, `paired_quant_local_acc_pivot.csv`, `paired_quant_margin_pivot.csv`: one row per dataset and one column per method.
- `paired_wilcoxon_corrected.csv`: paired Wilcoxon tests comparing `execution_trace` against each comparator, with raw, Holm-corrected, and Benjamini-Hochberg-corrected p-values computed within each metric family.
- `friedman_tests.csv`: Friedman tests across the complete method family for each shared metric.

## Important Interpretation Note

The seed-level CI uses only two seeds, 27 and 42. This is useful for transparency and for addressing the reviewer concern about variance, but it should not be presented as a high-precision estimate of experimental uncertainty. In the manuscript, it is safer to report these as seed-level variability checks and to avoid overstating narrow statistical confidence.

## Verification

The export was generated with:

```bash
python3 experiments_local_explanation/export_seed_ci_and_corrected_tests.py
python3 -m py_compile experiments_local_explanation/export_seed_ci_and_corrected_tests.py
```

The resulting counts are:

- 15 datasets
- 240 DPG all-seed/all-config rows
- 2280 baseline all-seed/all-config rows
- 210 selected per-seed rows
- 105 paired selected rows

The selected per-seed table contains exactly two seeds for every dataset-method pair.
