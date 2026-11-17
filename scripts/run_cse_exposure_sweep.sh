#!/usr/bin/env bash
# Run from anywhere; paths are resolved relative to the release root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# CSE spatial-exposure sweep: FedAvg x rho in {0.10,0.30,0.50} x 10 seeds,
# same 3,000 new emerging records per active round.
# Outputs: outputs/cse_exposure_sweep/rho<NN>/seed_<k>/
# Options (env): GPUS=0,1,2,3  METHODS=rho10,rho50  SEEDS=0,1
exec "$PYTHON_BIN" scripts/launch_runs.py cse_exposure_sweep "$@"
