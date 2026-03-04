#!/usr/bin/env bash
set -euo pipefail

# Run DPG 2.0 next-phase experiments on the curated 15-dataset list.
# This logs path-faithfulness, explanation-confidence, and misclassification metrics
# for both graph construction modes.
#
# Usage examples:
#   bash experiments_local_explanation/run_selected_datasets_dpg2_next_phase.sh
#   PARALLEL=2 RF_N_JOBS=4 \
#     OUT_ROOT=experiments_local_explanation/experiment_dpg2_next_phase \
#     bash experiments_local_explanation/run_selected_datasets_dpg2_next_phase.sh

DATA_DIR="${DATA_DIR:-experiments_local_explanation/data_numeric}"
OUT_ROOT="${OUT_ROOT:-experiments_local_explanation/experiment_dpg2_next_phase}"

PARALLEL="${PARALLEL:-2}"
RF_N_JOBS="${RF_N_JOBS:-4}"
GRAPH_CONSTRUCTION_MODES="${GRAPH_CONSTRUCTION_MODES:-aggregated_transitions,execution_trace}"
MAX_DEPTHS="${MAX_DEPTHS:-4}"
DECIMAL_THRESHOLDS="${DECIMAL_THRESHOLDS:-6}"

DATASETS="banknote-authentication,breast_cancer,diabetes,digits,ionosphere,iris,isolet,madelon,phoneme,qsar-biodeg,segment,spambase,vehicle,wdbc,wine"

PYTHONPATH=. python3 experiments_local_explanation/run_per_dataset.py \
  --data_dir "${DATA_DIR}" \
  --out_root "${OUT_ROOT}" \
  --datasets "${DATASETS}" \
  --parallel "${PARALLEL}" \
  --n_estimators "10,20" \
  --rf_n_jobs "${RF_N_JOBS}" \
  --max_depth "${MAX_DEPTHS}" \
  --perc_var "0.0001,0.00001" \
  --decimal_threshold "${DECIMAL_THRESHOLDS}" \
  --seeds "27,42" \
  --graph_construction_modes "${GRAPH_CONSTRUCTION_MODES}" \
  --max_test_samples 0 \
  --progress_every 25
