# Focused Depth/Ensemble Scalability Experiment

## Purpose

This experiment addresses the reviewer concern that the ECML submission used shallow random forests. It is not intended as a new broad benchmark; it is a bounded stress test for DPG-local under deeper trees and larger forests.

## Scope

Run DPG-local in execution-trace mode only, because this is the journal paper's main local construction.

Representative datasets:

- `iris`: small multiclass sanity case.
- `banknote-authentication`: low-dimensional high-accuracy binary case.
- `diabetes`: lower-accuracy binary case.
- `vehicle`: moderate multiclass case with known contested examples.
- `madelon`: high-dimensional difficult binary case.
- `isolet`: high-dimensional many-class case.

Grid:

- `n_estimators`: 20, 50.
- `max_depth`: 4, 8, 12.
- `seed`: 27.
- `perc_var`: 0.0001.
- `decimal_threshold`: 6.
- `max_test_samples`: 20 per dataset.

Outputs:

- Run root: `experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_by_dataset`
- Report: `experiments_local_explanation/results_journal_v1/dpg_scalability_depth_focused_report`

## Why This Is Enough

The main reviewer concern is not that every benchmark should be rerun, but that the claim should be bounded beyond depth-4 random forests. This focused run checks whether graph size, runtime, trace recovery, recombination, and diagnostic scores remain usable under deeper trees and larger forests.

## Manuscript Use

If results are stable:

> A focused stress test on six representative datasets with deeper forests confirmed that DPG-local preserves trace-faithful construction under larger tree depths, although runtime and graph size increase with ensemble size and depth.

If runtime/size grows sharply:

> A focused stress test shows that DPG-local is best framed as a diagnostic method for small-to-moderate random forests unless pruning or sampling is used. We therefore narrow the journal claim and report scalability as a practical limitation.
