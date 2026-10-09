#!/usr/bin/env bash
# Run the KBS local-DPG experiments in stages.
#
#   bash launch_server.sh smoke        # minutes; writes to results_smoke/, never touches results/
#   nohup bash launch_server.sh all > kbs_all.log 2>&1 &
#   bash launch_server.sh select|dpg|scaling|baselines|analyze
#
# Every job writes its own file and is skipped when that file exists, so a stage
# can be re-launched after an interruption. Failed jobs leave <job>.error.txt and
# no CSV; re-running the stage retries them.
#
# Worker counts default to the 18 physical cores of the i9-10980XE. Override with
# W_SELECT, W_DPG, W_SCALE, W_BASE. Baseline cost knobs: BASE_SEEDS, BASE_SAMPLES, LORE_GEN.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="${PY:-$ROOT/.venv-kbs/bin/python}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

W_SELECT="${W_SELECT:-18}"
W_DPG="${W_DPG:-18}"
W_SCALE="${W_SCALE:-8}"      # fewer workers: this stage reports runtimes
W_BASE="${W_BASE:-12}"
BASE_SEEDS="${BASE_SEEDS:-27,42,100}"
BASE_SAMPLES="${BASE_SAMPLES:-50}"
LORE_GEN="${LORE_GEN:-genetic}"

cd "$HERE"

run_stage() {
  local name="$1"; shift
  local log_dir="${KBS_RESULTS:-$HERE/results}/logs"
  mkdir -p "$log_dir"
  echo "[$(date -Is)] start $name"
  "$@" > "$log_dir/$name.log" 2>&1
  echo "[$(date -Is)] done  $name ($(grep -c ' ok\| cached' "$log_dir/$name.log" || true) ok/cached, $(grep -c 'FAILED' "$log_dir/$name.log" || true) failed)"
}

select_stage()    { run_stage select    "$PY" select_blackbox.py --workers "$W_SELECT" "$@"; }
dpg_stage()       { run_stage dpg_local "$PY" run_dpg_local.py   --workers "$W_DPG" "$@"; }
scaling_stage()   { run_stage scaling   "$PY" run_scaling.py     --workers "$W_SCALE" "$@"; }
baseline_stage()  { run_stage baselines "$PY" run_baselines.py   --workers "$W_BASE" --seeds "$BASE_SEEDS" --max_samples "$BASE_SAMPLES" --lore_generator "$LORE_GEN" "$@"; }
analyze_stage()   { run_stage analyze   "$PY" analyze.py; }

case "${1:-}" in
  smoke)
    export KBS_RESULTS="$HERE/results_smoke"
    rm -rf "$KBS_RESULTS"
    select_stage   --datasets iris,vehicle --seeds 27
    dpg_stage      --datasets iris,vehicle --seeds 27 --max_samples 5
    scaling_stage  --datasets iris --max_samples 3
    baseline_stage --datasets iris,vehicle --seeds 27 --max_samples 3 --lore_generator random
    analyze_stage
    echo "Smoke report: $KBS_RESULTS/report/summary.md"
    ls "$KBS_RESULTS"/*/*.error.txt 2>/dev/null && { echo "Smoke run had failures (see files above)"; exit 1; } || true
    ;;
  select)    select_stage ;;
  dpg)       dpg_stage ;;
  scaling)   scaling_stage ;;
  baselines) baseline_stage ;;
  analyze)   analyze_stage ;;
  all)
    select_stage
    dpg_stage
    scaling_stage
    analyze_stage      # main tables are available before the slow baselines finish
    baseline_stage
    analyze_stage
    ;;
  *)
    echo "usage: $0 {smoke|select|dpg|scaling|baselines|analyze|all}" >&2
    exit 2
    ;;
esac
