#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULT_DIR="${ROOT_DIR}/experiments/dpg_0_3_0"
mkdir -p "${RESULT_DIR}/logs"

WORKERS="${DPG_WORKERS:-$(nproc)}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

exec "${ROOT_DIR}/.venv/bin/python" "${ROOT_DIR}/scripts/run_dpg030_experiments.py" \
  --workers "${WORKERS}" \
  --output "${RESULT_DIR}/results/benchmark.csv" \
  "$@"
