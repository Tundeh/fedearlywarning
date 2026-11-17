#!/usr/bin/env python3
"""Compare freshly produced run outputs against the packaged reference runs.

For every run directory present under --fresh (default outputs/), the matching
reference directory under --reference (default reference_results/) is located
by its relative path, and the *_rounds.csv files are compared column by column.

Reports, per run: whether the files are byte-identical, the largest absolute
difference across numeric columns, and whether the run's target-alert endpoint
columns agree exactly. Exits non-zero if any compared run exceeds --atol.

    python scripts/compare_runs.py                       # outputs/ vs reference_results/
    python scripts/compare_runs.py --fresh outputs --atol 1e-6

Only the Python standard library is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Columns whose values carry wall-clock timing and are expected to differ.
TIMING_HINTS = ("time", "seconds", "elapsed", "duration")


def load(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def as_float(x: str):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def compare(fresh: Path, ref: Path):
    a, b = load(fresh), load(ref)
    if len(a) != len(b):
        return {"error": f"row count {len(a)} != {len(b)}"}
    cols = [c for c in b[0].keys()]
    missing = [c for c in cols if c not in a[0]]
    max_diff, worst, text_mismatch = 0.0, None, []
    for c in cols:
        if c in missing or any(h in c.lower() for h in TIMING_HINTS):
            continue
        for ra, rb in zip(a, b):
            fa, fb = as_float(ra[c]), as_float(rb[c])
            if fa is None or fb is None:
                if ra[c] != rb[c]:
                    text_mismatch.append(c)
                    break
                continue
            if math.isnan(fa) and math.isnan(fb):
                continue
            d = abs(fa - fb) if not (math.isnan(fa) or math.isnan(fb)) else math.inf
            if d > max_diff:
                max_diff, worst = d, c
    return {"max_abs_diff": max_diff, "worst_column": worst,
            "text_mismatch": sorted(set(text_mismatch)), "missing_columns": missing}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", default=str(ROOT / "outputs"))
    ap.add_argument("--reference", default=str(ROOT / "reference_results"))
    ap.add_argument("--atol", type=float, default=1e-9)
    a = ap.parse_args()
    fresh_root, ref_root = Path(a.fresh).resolve(), Path(a.reference).resolve()
    runs = sorted(fresh_root.rglob("*_rounds.csv"))
    if not runs:
        print(f"No *_rounds.csv under {fresh_root}")
        return 1
    bad = 0
    for f in runs:
        rel = f.relative_to(fresh_root)
        r = ref_root / rel
        if not r.exists():
            print(f"MISSING-REF  {rel}")
            bad += 1
            continue
        if sha(f) == sha(r):
            print(f"IDENTICAL    {rel}")
            continue
        res = compare(f, r)
        if "error" in res:
            print(f"ERROR        {rel}: {res['error']}")
            bad += 1
            continue
        ok = res["max_abs_diff"] <= a.atol and not res["text_mismatch"] and not res["missing_columns"]
        bad += 0 if ok else 1
        print(f"{'MATCH' if ok else 'DIFFERS':<12} {rel}: max|diff|={res['max_abs_diff']:.3g}"
              f" (worst column: {res['worst_column']})"
              + (f" text-mismatch={res['text_mismatch']}" if res["text_mismatch"] else "")
              + (f" missing={res['missing_columns']}" if res["missing_columns"] else ""))
    print(f"\n{len(runs) - bad}/{len(runs)} runs match the reference within atol={a.atol}.")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
