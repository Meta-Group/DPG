#!/usr/bin/env bash
# Create the KBS experiment environment and run the fast local-DPG unit tests.
# Usage: bash setup_server.sh            (PYTHON=python3.13 VENV=/path/to/venv to override)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-$ROOT/.venv-kbs}"

"$PYTHON" -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$HERE/requirements.txt"

# The scripts import dpg from this checkout (common.py puts the repo root on sys.path),
# so no package install is needed. Verify the local-DPG guarantees on this machine:
(cd "$ROOT" && "$VENV/bin/python" -m pytest -q tests/test_local_dpg.py)
"$VENV/bin/python" -c "import shap, lime, anchor, lore_sa, sklearn; print('explainers OK, sklearn', sklearn.__version__)"
echo "Environment ready: $VENV"
