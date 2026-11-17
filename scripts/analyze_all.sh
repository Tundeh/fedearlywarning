#!/usr/bin/env bash
# Run from anywhere; paths are resolved relative to the release root.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Recomputes Tables I-III, supplementary tables S1-S5, and Figures 1-3 + S1
# from saved round-level CSVs. No training.
#
#   bash scripts/analyze_all.sh                  # packaged reference_results/
#   RESULTS_ROOT=outputs bash scripts/analyze_all.sh   # your own reruns
#
# RESULTS_ROOT must contain cse_central/, ciciot_central/, cse_exposure_sweep/.
# The analysis code (analysis/paper_analysis.py) expects the original layout
#   <central>/{cse,ciciot}/<method>/seed_<k>  and  <sweep>/cse/fedavg/rho<NN>/seed_<k>,
# so a symlink view with that layout is built in a temporary directory.
RESULTS_ROOT="${RESULTS_ROOT:-reference_results}"
OUT="${1:-analysis_output/paper_$(date +%Y%m%d_%H%M%S)}"
for d in cse_central ciciot_central cse_exposure_sweep; do
  [[ -d "$RESULTS_ROOT/$d" ]] || { echo "Missing $RESULTS_ROOT/$d" >&2; exit 1; }
done
RR="$(cd "$RESULTS_ROOT" && pwd)"
VIEW="$(mktemp -d)"; trap 'rm -rf "$VIEW"' EXIT
mkdir -p "$VIEW/central" "$VIEW/sweep/cse"
ln -s "$RR/cse_central"        "$VIEW/central/cse"
ln -s "$RR/ciciot_central"     "$VIEW/central/ciciot"
ln -s "$RR/cse_exposure_sweep" "$VIEW/sweep/cse/fedavg"
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" analysis/paper_analysis.py "$VIEW/central" "$VIEW/sweep" "$OUT"
echo "Tables:  $OUT/tables   Figures: $OUT/figures   Report: $OUT/RESULTS_REPORT.md"
