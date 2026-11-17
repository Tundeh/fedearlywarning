#!/usr/bin/env bash
# Run from anywhere; paths are resolved relative to the release root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Rebuilds data/processed/cicids2018.npz (+ .json) exactly as used in the paper:
# all rows of the 10 official CSE-CIC-IDS2018 "TrafficForML_CICFlowMeter" CSVs,
# natural clients auto-detected from "Dst IP" (present only in the 20-02-2018
# file; rows of other files get key "" -> client 0), 20 logical clients, scaler fitted on a 20%
# sample excluding the botnet family, seed 0.
INPUT_ROOT="${CSE_INPUT_ROOT:-data/CSE-CIC-IDS2018}"
OUTPUT="${CSE_OUTPUT:-data/processed/cicids2018.npz}"
mkdir -p "$(dirname "$OUTPUT")"
"$PYTHON_BIN" -u utils/prepare_cicids2018.py \
  --input-root "$INPUT_ROOT" \
  --output "$OUTPUT" \
  --num-logical-clients 20 \
  --max-rows-per-file 0 \
  --scaler-fraction 0.20 \
  --exclude-from-scaler botnet \
  --seed 0
"$PYTHON_BIN" scripts/inspect_cyber_data.py --data "$OUTPUT" || true
echo "Expected SHA-256 (see provenance/processed_data_sha256.txt):"
grep cicids2018.npz provenance/processed_data_sha256.txt
sha256sum "$OUTPUT"
