#!/usr/bin/env bash
# Run from anywhere; paths are resolved relative to the release root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Regenerates the paper figures (Fig. 1 spatial exposure, Fig. 2 calibrated
# target-alert trajectories, Fig. 3 acquisition vs dissemination, Fig. S1 target
# FPR trajectories) from saved round-level CSVs. Figures are produced by the same
# analysis pipeline as the tables; this wrapper runs it and copies figures out.
#   bash scripts/make_figures.sh            # -> figures/
#   RESULTS_ROOT=outputs bash scripts/make_figures.sh
DEST="${1:-figures}"
TMP_OUT="$(mktemp -d)/analysis"
bash scripts/analyze_all.sh "$TMP_OUT"
mkdir -p "$DEST"
cp "$TMP_OUT"/figures/* "$DEST"/
rm -rf "$(dirname "$TMP_OUT")"
ls -1 "$DEST"
