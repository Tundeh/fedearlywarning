#!/usr/bin/env bash
# Run from anywhere; paths are resolved relative to the release root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Rebuilds data/processed/ciciot2023.npz (+ .json) exactly as used in the paper:
# the first 50,000 rows of each of the 309 CICIoT2023 per-class CSV files
# (labels taken from the folder/file name), 20 deterministic logical clients
# from chronological file blocks, scaler fitted on a 20% sample excluding the
# botnet family, seed 0.
INPUT_ROOT="${CICIOT_INPUT_ROOT:-data/CICIoT2023}"
OUTPUT="${CICIOT_OUTPUT:-data/processed/ciciot2023.npz}"
mkdir -p "$(dirname "$OUTPUT")"
"$PYTHON_BIN" -u utils/prepare_ciciot2023.py \
  --input-root "$INPUT_ROOT" \
  --output "$OUTPUT" \
  --num-logical-clients 20 \
  --max-rows-per-file 50000 \
  --scaler-fraction 0.20 \
  --exclude-from-scaler botnet \
  --seed 0
"$PYTHON_BIN" scripts/inspect_cyber_data.py --data "$OUTPUT" || true
echo "Expected SHA-256 (see provenance/processed_data_sha256.txt):"
grep ciciot2023.npz provenance/processed_data_sha256.txt
sha256sum "$OUTPUT"
