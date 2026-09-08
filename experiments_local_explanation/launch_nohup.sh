#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  experiments_local_explanation/launch_nohup.sh [--name NAME] [--log-dir DIR] -- COMMAND [ARGS...]

Runs COMMAND under nohup, writing stdout/stderr to a timestamped log and the
background process id to a matching .pid file.

Examples:
  experiments_local_explanation/launch_nohup.sh \
    --name baselines_validation_noice \
    -- python3 experiments_local_explanation/run_journal_parallel.py --kind baselines ...

  tail -f experiments_local_explanation/results_journal_v1/logs/baselines_validation_noice_*.log
USAGE
}

name="journal_run"
log_dir="experiments_local_explanation/results_journal_v1/logs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name)
      name="${2:?--name requires a value}"
      shift 2
      ;;
    --log-dir)
      log_dir="${2:?--log-dir requires a value}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  echo "Missing command after --" >&2
  usage >&2
  exit 2
fi

mkdir -p "$log_dir"
timestamp="$(date +%Y%m%d_%H%M%S)"
log_file="${log_dir}/${name}_${timestamp}.log"
pid_file="${log_dir}/${name}_${timestamp}.pid"

if command -v setsid >/dev/null 2>&1; then
  setsid nohup "$@" >"$log_file" 2>&1 </dev/null &
else
  nohup "$@" >"$log_file" 2>&1 </dev/null &
fi
pid="$!"
printf '%s\n' "$pid" >"$pid_file"

echo "Started PID: $pid"
echo "Log: $log_file"
echo "PID file: $pid_file"
echo "Follow: tail -f $log_file"
