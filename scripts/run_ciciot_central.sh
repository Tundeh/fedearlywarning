#!/usr/bin/env bash
# Run from anywhere; paths are resolved relative to the release root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# CICIoT2023 central comparison: 4 methods x 10 seeds, rho=0.30 (6/20 exposed).
# Outputs: outputs/ciciot_central/<method>/seed_<k>/
# Options (env): GPUS=0,1,2,3  METHODS=fedavg,fedccfa  SEEDS=0,1
exec "$PYTHON_BIN" scripts/launch_runs.py ciciot_central "$@"
