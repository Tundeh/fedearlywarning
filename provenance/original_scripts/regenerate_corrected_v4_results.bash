#!/usr/bin/env bash
# Corrected v4 result analysis, using saved runs only.
# Usage: bash regenerate_corrected_v4_results.bash [CENTRAL_ROOT] [SPATIAL_ROOT] [OUTPUT_DIR]
# Requires: Python 3.10+, numpy, pandas, scipy, matplotlib, pyyaml.
set -Eeuo pipefail
CENTRAL_ROOT=/LLM/FedLearning/FedCCFA/experiments/target_alert_v4_10seed
SPATIAL_ROOT=/LLM/FedLearning/FedCCFA/experiments/cse_spatial_exposure_target_alert_v4_10seed
OUTPUT_DIR="/LLM/FedLearning/FedCCFA/paper_artifacts/corrected_v4_$(date +%Y%m%d_%H%M%S)"
if (( $# > 0 )); then CENTRAL_ROOT="$1"; fi
if (( $# > 1 )); then SPATIAL_ROOT="$2"; fi
if (( $# > 2 )); then OUTPUT_DIR="$3"; fi
if (( $# > 3 )); then echo "Expected at most three directory arguments" >&2; exit 2; fi
PYTHON_BIN="$(printenv PYTHON_BIN || printf python)"
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" - "$CENTRAL_ROOT" "$SPATIAL_ROOT" "$OUTPUT_DIR" <<'FEDCCFA_PYTHON'
#!/usr/bin/env python3
"""Complete the inferential and utility analysis for target-alert v4.

This script reads the 80 completed round files. It does not train models or
modify run outputs. New analysis artifacts are written below ``analysis/``.
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest


METHODS = (
    "fedavg",
    "feddrift",
    "fedccfa",
    "fedccfa_global_classifier",
)
DATASETS = ("cse", "ciciot")
PAIRS = (
    ("fedccfa_global_classifier", "fedavg"),
    ("fedccfa_global_classifier", "fedccfa"),
    ("fedccfa_global_classifier", "feddrift"),
    ("fedccfa", "fedavg"),
    ("fedccfa", "feddrift"),
    ("fedavg", "feddrift"),
)


TARGET_CONTINUOUS_DIRECTIONS = {
    "empirical_restricted_delay": "lower",
    "confidence_bound_restricted_delay": "lower",
    "all_site_restricted_delay": "lower",
    "final_recall": "higher",
    "final_pattern_balanced_recall": "higher",
    "final_target_vs_rest_fpr": "lower",
    "final_target_vs_rest_fpr_upper_mean": "lower",
    "final_target_vs_rest_fpr_upper_max": "lower",
    "final_precision": "higher",
    "final_auroc": "higher",
    "final_auprc": "higher",
    "final_cal025_recall": "higher",
    "final_cal025_fpr": "lower",
    "final_cal050_recall": "higher",
    "final_cal050_fpr": "lower",
    "final_cal100_recall": "higher",
    "final_cal100_fpr": "lower",
}
TARGET_BINARY_DIRECTIONS = {
    "empirical_attained": "higher",
    "confidence_bound_attained": "higher",
    "all_site_attained": "higher",
    "final_empirical_sustained": "higher",
    "final_confidence_bound_sustained": "higher",
    "final_all_site_sustained": "higher",
}
UTILITY_DIRECTIONS = {
    "post_balanced_accuracy_mean": "higher",
    "post_benign_fpr_mean": "lower",
    "post_attack_recall_mean": "higher",
    "post_accuracy_mean": "higher",
    "post_macro_f1_mean": "higher",
    "final_balanced_accuracy": "higher",
    "final_benign_fpr": "lower",
    "final_attack_recall": "higher",
    "final_accuracy": "higher",
    "final_macro_f1": "higher",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--drift-round", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--recall-target", type=float, default=0.90)
    parser.add_argument("--fpr-limit", type=float, default=0.01)
    parser.add_argument("--sustained-rounds", type=int, default=3)
    return parser.parse_args()


def identity(path: Path, root: Path) -> tuple[str, str, int]:
    rel = path.relative_to(root)
    if len(rel.parts) < 4:
        raise ValueError(f"Unexpected run path: {path}")
    dataset, method, seed_dir = rel.parts[:3]
    if dataset not in DATASETS or method not in METHODS:
        raise ValueError(f"Unexpected dataset/method in {path}")
    if not seed_dir.startswith("seed_"):
        raise ValueError(f"Unexpected seed directory: {seed_dir}")
    return dataset, method, int(seed_dir.split("_", 1)[1])


def discover_round_files(root: Path) -> list[Path]:
    files = sorted(root.glob("*/*/seed_*/*_rounds.csv"))
    if not files:
        raise FileNotFoundError(f"No round files below {root}")
    identities = [identity(path, root) for path in files]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate dataset/method/seed run identities detected")
    seeds = sorted({seed for _, _, seed in identities})
    expected = {
        (dataset, method, seed)
        for dataset in DATASETS
        for method in METHODS
        for seed in seeds
    }
    missing = sorted(expected - set(identities))
    extra = sorted(set(identities) - expected)
    if missing or extra:
        raise ValueError(f"Incomplete matrix: missing={missing}, extra={extra}")
    print(
        f"Found {len(files)} runs: {len(DATASETS)} datasets x "
        f"{len(METHODS)} methods x {len(seeds)} seeds"
    )
    return files


def resolve_column(
    frame: pd.DataFrame,
    *candidates: str,
    required: bool = True,
) -> str | None:
    """Resolve exact or case-insensitive aliases without guessing semantics."""
    for candidate in candidates:
        if candidate in frame:
            return candidate
    lowercase = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in candidates:
        match = lowercase.get(candidate.lower())
        if match is not None:
            return match
    if required:
        raise ValueError(
            "Missing required column; accepted aliases: "
            + ", ".join(candidates)
        )
    return None


def numeric(frame: pd.DataFrame, *candidates: str) -> pd.Series:
    column = resolve_column(frame, *candidates)
    assert column is not None
    return pd.to_numeric(frame[column], errors="coerce")


def sustained_attainment(
    frame: pd.DataFrame,
    statistic_column: str,
    drift_round: int,
    recall_target: float,
    fpr_limit: float,
    required_rounds: int,
) -> tuple[float, int]:
    rounds = numeric(frame, "round").astype(int).to_numpy()
    recalls = numeric(frame, "target_alert_recall_unexposed").to_numpy(float)
    statistics = numeric(frame, statistic_column).to_numpy(float)
    streak = 0
    first_confirmation = np.nan
    for round_id, recall, statistic in zip(rounds, recalls, statistics):
        if round_id < drift_round:
            continue
        qualified = (
            np.isfinite(recall)
            and np.isfinite(statistic)
            and recall >= recall_target
            and statistic <= fpr_limit
        )
        streak = streak + 1 if qualified else 0
        if not np.isfinite(first_confirmation) and streak >= required_rounds:
            first_confirmation = float(round_id)
    return first_confirmation, int(streak)


def restricted_delay(round_id: float, drift_round: int, horizon: int) -> float:
    return (
        float(round_id - drift_round)
        if np.isfinite(round_id)
        else float(horizon)
    )


def build_target_per_run(
    files: Iterable[Path],
    root: Path,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: list[dict] = []
    warned_missing_max = False
    for path in files:
        dataset, method, seed = identity(path, root)
        frame = pd.read_csv(path).sort_values("round")
        if numeric(frame, "round").astype(int).tolist() != list(range(len(frame))):
            raise ValueError(f"Incomplete or unordered rounds: {path}")
        final = frame.iloc[-1]
        recall_column = resolve_column(
            frame, "target_alert_recall_unexposed"
        )
        empirical_fpr_column = resolve_column(
            frame, "target_alert_target_vs_rest_fpr_unexposed"
        )
        mean_upper_column = resolve_column(
            frame,
            "target_alert_target_vs_rest_fpr_upper_unexposed",
            "target_alert_target_vs_rest_fpr_upper_mean_unexposed",
        )
        max_upper_column = resolve_column(
            frame,
            "target_alert_target_vs_rest_fpr_upper_max_unexposed",
            required=False,
        )
        empirical_round, empirical_streak = sustained_attainment(
            frame,
            empirical_fpr_column,
            args.drift_round,
            args.recall_target,
            args.fpr_limit,
            args.sustained_rounds,
        )
        confidence_round, confidence_streak = sustained_attainment(
            frame,
            mean_upper_column,
            args.drift_round,
            args.recall_target,
            args.fpr_limit,
            args.sustained_rounds,
        )
        if max_upper_column is not None:
            all_site_round, all_site_streak = sustained_attainment(
                frame,
                max_upper_column,
                args.drift_round,
                args.recall_target,
                args.fpr_limit,
                args.sustained_rounds,
            )
            final_max_upper = float(final[max_upper_column])
            all_site_attained: float = float(np.isfinite(all_site_round))
            all_site_delay = restricted_delay(
                all_site_round, args.drift_round, args.horizon
            )
            final_all_site_sustained: float = float(
                all_site_streak >= args.sustained_rounds
            )
        else:
            all_site_round, all_site_streak = np.nan, 0
            final_max_upper = np.nan
            all_site_attained = np.nan
            all_site_delay = np.nan
            final_all_site_sustained = np.nan
            if not warned_missing_max:
                print(
                    "WARNING: per-round maximum client FPR upper bound is not "
                    "logged; all-site sensitivity fields will be unavailable."
                )
                warned_missing_max = True
        rows.append({
            "dataset_key": dataset,
            "method": method,
            "seed": seed,
            "empirical_attained": int(np.isfinite(empirical_round)),
            "confidence_bound_attained": int(np.isfinite(confidence_round)),
            "all_site_attained": all_site_attained,
            "empirical_target_round": empirical_round,
            "confidence_bound_target_round": confidence_round,
            "all_site_target_round": all_site_round,
            "empirical_restricted_delay": restricted_delay(
                empirical_round, args.drift_round, args.horizon
            ),
            "confidence_bound_restricted_delay": restricted_delay(
                confidence_round, args.drift_round, args.horizon
            ),
            "all_site_restricted_delay": all_site_delay,
            "final_empirical_sustained": int(
                empirical_streak >= args.sustained_rounds
            ),
            "final_confidence_bound_sustained": int(
                confidence_streak >= args.sustained_rounds
            ),
            "final_all_site_sustained": final_all_site_sustained,
            "final_recall": float(final[recall_column]),
            "final_pattern_balanced_recall": float(
                final["target_alert_pattern_balanced_recall_unexposed"]
            ),
            "final_target_vs_rest_fpr": float(final[empirical_fpr_column]),
            "final_target_vs_rest_fpr_upper_mean": float(final[mean_upper_column]),
            "final_target_vs_rest_fpr_upper_max": final_max_upper,
            "final_precision": float(
                final["target_alert_precision_unexposed"]
            ),
            "final_auroc": float(final["target_alert_auroc_unexposed"]),
            "final_auprc": float(final["target_alert_auprc_unexposed"]),
            "final_cal025_recall": float(
                final["target_alert_calibration_025bp_recall_unexposed"]
            ),
            "final_cal025_fpr": float(
                final[
                    "target_alert_calibration_025bp_target_vs_rest_fpr_unexposed"
                ]
            ),
            "final_cal050_recall": float(
                final["target_alert_calibration_050bp_recall_unexposed"]
            ),
            "final_cal050_fpr": float(
                final[
                    "target_alert_calibration_050bp_target_vs_rest_fpr_unexposed"
                ]
            ),
            "final_cal100_recall": float(
                final["target_alert_calibration_100bp_recall_unexposed"]
            ),
            "final_cal100_fpr": float(
                final[
                    "target_alert_calibration_100bp_target_vs_rest_fpr_unexposed"
                ]
            ),
        })
    return pd.DataFrame(rows).sort_values(
        ["dataset_key", "method", "seed"]
    ).reset_index(drop=True)


def build_utility_per_run(
    files: Iterable[Path], root: Path, drift_round: int
) -> tuple[pd.DataFrame, dict[str, str]]:
    sources = {
        "balanced_accuracy": ("post_balanced_accuracy",),
        "benign_fpr": ("post_benign_fpr", "post_benign_FPR"),
        "attack_recall": ("post_attack_recall",),
        "accuracy": ("post_accuracy",),
        "macro_f1": ("post_macro_f1",),
    }
    rows: list[dict] = []
    available: set[str] | None = None
    for path in files:
        dataset, method, seed = identity(path, root)
        frame = pd.read_csv(path).sort_values("round")
        resolved = {
            name: resolve_column(frame, *aliases, required=False)
            for name, aliases in sources.items()
        }
        present = {name for name, column in resolved.items() if column is not None}
        available = present if available is None else available & present
        post = frame[numeric(frame, "round") >= drift_round]
        row: dict[str, float | int | str] = {
            "dataset_key": dataset,
            "method": method,
            "seed": seed,
        }
        for name in present:
            column = resolved[name]
            assert column is not None
            values = pd.to_numeric(post[column], errors="coerce")
            row[f"post_{name}_mean"] = float(values.mean())
            row[f"final_{name}"] = float(
                pd.to_numeric(frame[column], errors="coerce").iloc[-1]
            )
        rows.append(row)
    available = available or set()
    directions = {
        metric: direction
        for metric, direction in UTILITY_DIRECTIONS.items()
        if metric.replace("post_", "").replace("_mean", "") in available
        or metric.replace("final_", "") in available
    }
    return (
        pd.DataFrame(rows).sort_values(
            ["dataset_key", "method", "seed"]
        ).reset_index(drop=True),
        directions,
    )


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan
    if np.allclose(values, 0.0, rtol=0.0, atol=1e-15):
        return 1.0
    if len(values) > 20:
        raise ValueError("Exact sign-flip enumeration is limited to n <= 20")
    observed = abs(float(np.mean(values)))
    extreme = 0
    total = 2 ** len(values)
    for signs in product((-1.0, 1.0), repeat=len(values)):
        statistic = abs(float(np.mean(values * np.asarray(signs))))
        if statistic >= observed - 1e-15:
            extreme += 1
    return float(extreme / total)


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=np.float64)
    adjusted = np.full(len(values), np.nan, dtype=np.float64)
    finite_idx = np.where(np.isfinite(values))[0]
    if not len(finite_idx):
        return adjusted
    order = finite_idx[np.argsort(values[finite_idx])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        candidate = min(1.0, float((m - rank) * values[idx]))
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def paired_tests(
    per_run: pd.DataFrame,
    directions: dict[str, str],
    binary_metrics: set[str] | None = None,
) -> pd.DataFrame:
    binary_metrics = binary_metrics or set()
    results: list[dict] = []
    for dataset in DATASETS:
        dataset_frame = per_run[per_run["dataset_key"] == dataset]
        for metric, direction in directions.items():
            if metric not in dataset_frame:
                continue
            for focal, comparator in PAIRS:
                left = dataset_frame[dataset_frame["method"] == focal][
                    ["seed", metric]
                ].rename(columns={metric: "focal"})
                right = dataset_frame[dataset_frame["method"] == comparator][
                    ["seed", metric]
                ].rename(columns={metric: "comparator"})
                paired = left.merge(right, on="seed", validate="one_to_one")
                paired = paired.replace([np.inf, -np.inf], np.nan).dropna()
                focal_values = paired["focal"].to_numpy(float)
                comparator_values = paired["comparator"].to_numpy(float)
                favorable = (
                    focal_values - comparator_values
                    if direction == "higher"
                    else comparator_values - focal_values
                )
                if metric in binary_metrics:
                    focal_binary = focal_values.astype(int)
                    comparator_binary = comparator_values.astype(int)
                    favorable_only = int(
                        np.sum((focal_binary == 1) & (comparator_binary == 0))
                    )
                    comparator_only = int(
                        np.sum((focal_binary == 0) & (comparator_binary == 1))
                    )
                    discordant = favorable_only + comparator_only
                    p_value = (
                        float(
                            binomtest(
                                favorable_only,
                                discordant,
                                p=0.5,
                                alternative="two-sided",
                            ).pvalue
                        )
                        if discordant else 1.0
                    )
                    test = "exact_mcnemar"
                else:
                    favorable_only = np.nan
                    comparator_only = np.nan
                    p_value = exact_sign_flip_pvalue(favorable)
                    test = "exact_sign_flip"
                results.append({
                    "dataset_key": dataset,
                    "metric": metric,
                    "direction": direction,
                    "test": test,
                    "focal_method": focal,
                    "comparator_method": comparator,
                    "paired_n": int(len(paired)),
                    "focal_mean": float(np.mean(focal_values)),
                    "comparator_mean": float(np.mean(comparator_values)),
                    "favorable_mean_difference": float(np.mean(favorable)),
                    "favorable_median_difference": float(np.median(favorable)),
                    "focal_only_successes": favorable_only,
                    "comparator_only_successes": comparator_only,
                    "p_value_exact": p_value,
                })
    result = pd.DataFrame(results)
    result["p_value_holm"] = np.nan
    families = result.groupby(
        ["dataset_key", "metric", "test"], sort=False
    ).groups
    for indices in families.values():
        idx = list(indices)
        result.loc[idx, "p_value_holm"] = holm_adjust(
            result.loc[idx, "p_value_exact"]
        )
    result["significant_holm_0_05"] = result["p_value_holm"] < 0.05
    return result.sort_values(
        ["dataset_key", "metric", "focal_method", "comparator_method"]
    ).reset_index(drop=True)


def group_summary(
    frame: pd.DataFrame,
    metrics: Iterable[str],
    binary_metrics: set[str] | None = None,
) -> pd.DataFrame:
    binary_metrics = binary_metrics or set()
    aggregations: dict[str, tuple[str, str]] = {
        "seeds": ("seed", "count")
    }
    for metric in metrics:
        if metric not in frame:
            continue
        aggregations[f"{metric}_mean"] = (metric, "mean")
        aggregations[f"{metric}_sd"] = (metric, "std")
        if metric in binary_metrics:
            aggregations[f"{metric}_count"] = (metric, "sum")
    return frame.groupby(["dataset_key", "method"], as_index=False).agg(
        **aggregations
    )


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    rendered = frame.copy()
    for column in rendered.select_dtypes(include=["float"]).columns:
        if str(column).endswith("_count"):
            rendered[column] = rendered[column].map(
                lambda value: "--" if pd.isna(value) else str(int(value))
            )
        else:
            rendered[column] = rendered[column].map(
                lambda value: "--" if pd.isna(value) else f"{value:.4f}"
            )
    headers = [str(column) for column in rendered.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for values in rendered.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in values) + " |")
    return "\n".join(lines)



# Standalone driver appended to the analysis routines above.
import hashlib
import itertools
import json
import re
import tempfile
from types import SimpleNamespace
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = dict(zip(METHODS, ("FedAvg", "FedDrift", "FedCCFA", "FedCCFA-GC")))
DS = {"cse": "CSE-CIC-IDS2018", "ciciot": "CICIoT2023"}
EP = ("empirical", "confidence_bound", "all_site")
EP_LABEL = dict(zip(EP, ("Empirical", "Mean-UCB", "All-site")))
FP_COL = dict(zip(EP, (
    "target_alert_target_vs_rest_fpr_unexposed",
    "target_alert_target_vs_rest_fpr_upper_unexposed",
    "target_alert_target_vs_rest_fpr_upper_max_unexposed")))
ARGS = SimpleNamespace(drift_round=10, horizon=20, recall_target=.9,
                       fpr_limit=.01, required_rounds=3, sustained_rounds=3)
WARNINGS, PROVENANCE, SPLIT_ROWS = [], [], []
ALIASES = {
    FP_COL["empirical"]: ("target_vs_rest_fpr_unexposed", "target_alert_fpr_unexposed"),
    FP_COL["confidence_bound"]: ("target_alert_target_vs_rest_fpr_upper_mean_unexposed",
                                "target_alert_fpr_upper_mean_unexposed"),
    FP_COL["all_site"]: ("target_alert_target_vs_rest_fpr_upper_maximum_unexposed",
                        "target_alert_fpr_upper_max_unexposed"),
}


def load_run(directory, tag, rho):
    """Merge main/sidecar round files by round; reject conflicts, never guess a group."""
    columns, files = {}, []
    for path in sorted(directory.glob("*round*.csv")):
        f = pd.read_csv(path)
        if "round" not in f or not any("target_alert" in c or c.startswith("post_") for c in f):
            continue
        f = f.sort_values("round").reset_index(drop=True)
        if f["round"].tolist() != list(range(30)):
            raise ValueError(f"{path}: expected unique rounds 0..29")
        for c in f:
            if c in columns:
                try:
                    equal = np.allclose(pd.to_numeric(f[c]), pd.to_numeric(columns[c]),
                                        rtol=0, atol=1e-12, equal_nan=True)
                except (ValueError, TypeError):
                    equal = f[c].equals(columns[c])
                if not equal:
                    raise ValueError(f"{directory}: conflicting column {c} across CSVs")
            columns[c] = f[c]
        files.append(path)
    if not files:
        raise FileNotFoundError(f"No round CSV in {directory}")
    f = pd.DataFrame(columns)
    for target, aliases in ALIASES.items():
        for alias in aliases:
            if alias in f:
                if target in f and not np.allclose(f[target], f[alias], equal_nan=True, rtol=0, atol=1e-12):
                    raise ValueError(f"{directory}: conflicting aliases {target}, {alias}")
                f[target] = f[alias]
    essential = ["target_alert_recall_unexposed", "target_alert_pattern_balanced_recall_unexposed",
                 *FP_COL.values(), "cumulative_selected_distinct_emerging_records",
                 "cumulative_selected_distinct_emerging_feature_patterns"]
    absent = [c for c in essential if c not in f]
    if absent:
        raise ValueError(f"{directory}: missing {absent}. Available columns: {list(f)}")
    for c in essential:
        f[c] = pd.to_numeric(f[c], errors="raise")
        if not np.isfinite(f[c].iloc[10:]).all():
            raise ValueError(f"{directory}: nonfinite post-emergence values in {c}")
    for c in essential[:5]:
        if not f[c].iloc[10:].between(0, 1).all():
            raise ValueError(f"{directory}: probability outside [0,1] in {c}")
    if (f[FP_COL["all_site"]] + 1e-12 < f[FP_COL["confidence_bound"]]).iloc[10:].any():
        raise ValueError(f"{directory}: max UCB below mean UCB")
    if (f[FP_COL["confidence_bound"]] + 1e-12 < f[FP_COL["empirical"]]).iloc[10:].any():
        raise ValueError(f"{directory}: mean UCB below observed FPR")
    if not np.array_equal(f[essential[-2]].iloc[10:], np.arange(1, 21)*3000):
        raise ValueError(f"{directory}: expected 3000 distinct incoming records per round")
    if (f[essential[-1]] > f[essential[-2]]).any() or (f[essential[-1]].diff().dropna() < 0).any():
        raise ValueError(f"{directory}: invalid cumulative pattern counts")
    # Optional diagnostics are explicitly unavailable rather than fabricated.
    for c in ("target_alert_precision_unexposed", "target_alert_auroc_unexposed", "target_alert_auprc_unexposed"):
        if c not in f:
            f[c] = np.nan
            WARNINGS.append(f"{tag}: missing optional metric {c}")
    for budget in ("025", "050", "100"):
        for metric in ("recall", "target_vs_rest_fpr"):
            c = f"target_alert_calibration_{budget}bp_{metric}_unexposed"
            if c not in f:
                f[c] = np.nan
                WARNINGS.append(f"{tag}: missing calibration sensitivity {c}")
    metadata = []
    for p in sorted(directory.glob("*metadata*.json")):
        value = json.loads(p.read_text())
        files.append(p)
        if "emerging_split_audit" in value or "target_alert_split_audit" in value:
            metadata.append(value)
    if len(metadata) > 1 and any(x != metadata[0] for x in metadata[1:]):
        raise ValueError(f"{directory}: conflicting metadata")
    meta = metadata[0] if metadata else {}
    config = meta.get("config", {})
    cp = directory / "config.yaml"
    if cp.exists():
        saved = yaml.safe_load(cp.read_text()) or {}
        # Use effective metadata config where available, otherwise saved overlay.
        if not config:
            config = saved
        files.append(cp)
    checks = (
        ("evaluation.target_alert_primary_calibration_fpr_limit", .005, "target_alert_primary_calibration_fpr_limit"),
        ("evaluation.target_alert_test_fpr_limit", .01, "target_alert_test_fpr_limit"),
        ("evaluation.target_alert_fpr_confidence", .95, "target_alert_fpr_confidence"),
        ("evaluation.target_alert_sustained_rounds", 3, None),
        ("stream.attack_guard_fraction", .05, None), ("stream.attack_test_fraction", .30, None),
        ("initial_model_seed", 424242, None), ("rounds", 30, None),
        ("participation.normal_prob", 1., None), ("participation.exposed_prob", 1., None))
    for key, expected, col in checks:
        value = config
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        observed = [] if value is None else [float(value)]
        if col and col in f:
            observed += pd.to_numeric(f[col], errors="raise").tolist()
        if observed and not np.allclose(observed, expected, atol=1e-12, rtol=0):
            raise ValueError(f"{directory}: {key} disagrees with expected {expected}")
        if not observed:
            WARNINGS.append(f"{tag}: unverified setting {key}")
    if "n_exposed" in f and not (f.n_exposed.iloc[10:] == round(20*rho)).all():
        raise ValueError(f"{directory}: n_exposed disagrees with rho={rho}")
    if "n_selected" in f and not (f.n_selected.iloc[10:] == 20).all():
        raise ValueError(f"{directory}: not full participation")
    if "target_alert_calibration_fpr_unexposed" in f:
        if (f["target_alert_calibration_fpr_unexposed"] > .005 + 1e-12).any():
            raise ValueError(f"{directory}: calibration FPR exceeds budget")
    row = {"run": tag, "run_dir": str(directory)}
    required = {
        "emerging_split_audit": ("n_index_overlap_train_final_challenge", "n_feature_overlap_rows_after_purge",
                                "n_unique_duplicate_patterns_crossing_after_purge"),
        "target_alert_split_audit": ("n_index_overlap_training_evaluation", "n_index_overlap_calibration_test",
                                    "n_feature_overlap_training_calibration", "n_feature_overlap_training_test",
                                    "n_feature_overlap_calibration_test", "n_feature_overlap_positive_calibration",
                                    "n_feature_overlap_positive_test")}
    missing = []
    for section, keys in required.items():
        obj = meta.get(section, {})
        for k, v in obj.items():
            row[section+"."+k] = json.dumps(v) if isinstance(v, (dict,list)) else v
        for k in keys:
            if k not in obj:
                missing.append(section+"."+k)
            elif float(obj[k]) != 0:
                raise ValueError(f"{directory}: split audit failed: {k}={obj[k]}")
    row["status"] = "UNVERIFIED" if missing else "PASS_REPORTED_COUNTERS"
    row["missing"] = ";".join(missing)
    if missing:
        WARNINGS.append(f"{tag}: incomplete split audit metadata")
    SPLIT_ROWS.append(row)
    PROVENANCE.append({"run": tag, "files": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                       "config": config})
    return f


def enrich(frame):
    out = {}
    for ep in EP:
        t, streak = sustained_attainment(frame, FP_COL[ep], 10, .9, .01, 3)
        out[ep+"_attained"] = int(np.isfinite(t))
        out[ep+"_target_round"] = t
        out[ep+"_restricted_delay"] = restricted_delay(t, 10, 20)
        out["final_"+ep+"_sustained"] = int(streak >= 3)
        for name, col in (("records", "cumulative_selected_distinct_emerging_records"),
                          ("patterns", "cumulative_selected_distinct_emerging_feature_patterns")):
            out[ep+"_"+name+"_at_confirmation"] = float(frame.loc[int(t), col]) if np.isfinite(t) else np.nan
        p = frame.copy()
        p["target_alert_recall_unexposed"] = p["target_alert_pattern_balanced_recall_unexposed"]
        pt, ps = sustained_attainment(p, FP_COL[ep], 10, .9, .01, 3)
        out["pattern_"+ep+"_attained"] = int(np.isfinite(pt))
        out["pattern_"+ep+"_restricted_delay"] = restricted_delay(pt, 10, 20)
        logged = {"empirical":"target_alert_empirical_target_round",
                  "confidence_bound":"target_alert_certified_target_round"}.get(ep)
        if logged and logged in frame:
            actual = float(frame[logged].iloc[-1])
            if not ((not np.isfinite(t) and not np.isfinite(actual)) or t == actual):
                raise ValueError(f"Recomputed {ep} confirmation {t} differs from logged {actual}")
    out["final_recall"] = frame.target_alert_recall_unexposed.iloc[-1]
    out["final_target_vs_rest_fpr"] = frame[FP_COL["empirical"]].iloc[-1]
    out["final_pattern_balanced_recall"] = frame.target_alert_pattern_balanced_recall_unexposed.iloc[-1]
    return out


def ci(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20260918)
    return np.quantile(values[rng.integers(0,len(values), (100_000,len(values)))].mean(axis=1), [.025,.975])


def add_intervals(tests, per_run, grouping="method"):
    tests = tests.copy()
    for i, row in tests.iterrows():
        subset = per_run[per_run.dataset_key == row.dataset_key]
        a = subset[subset[grouping] == row.focal_method].set_index("seed")[row.metric].sort_index()
        b = subset[subset[grouping] == row.comparator_method].set_index("seed")[row.metric].sort_index()
        if list(a.index) != list(range(10)) or list(b.index) != list(range(10)):
            raise ValueError("Paired inference requires seeds 0..9 without dropped runs")
        diff = a.to_numpy(float)-b.to_numpy(float)
        tests.loc[i,"difference_focal_minus_reference"] = diff.mean()
        if row.test == "exact_sign_flip":
            lo, hi = ci(diff)
            tests.loc[i,"bootstrap_ci_low"], tests.loc[i,"bootstrap_ci_high"] = lo, hi
    return tests


def mean_sd(group, col, digits=3):
    a = pd.to_numeric(group[col], errors="raise")
    if a.isna().any():
        return "--"
    return f"{a.mean():.{digits}f} ± {a.std(ddof=1):.{digits}f}"


def esc(value):
    return "".join({"_":r"\_", "%":r"\%", "&":r"\&", "#":r"\#", "±":r"$\pm$"}.get(c,c) for c in str(value))


def make_table(out, name, data, caption, label):
    data.to_csv(out/"tables"/f"{name}.csv", index=False)
    (out/"tables"/f"{name}.md").write_text(caption+"\n\n"+markdown_table(data)+"\n")
    breaks = {
        "Target-family recall": r"\shortstack{Target-family\\recall}",
        "Broad benign FPR": r"\shortstack{Broad benign\\FPR}",
        "Mean-UCB success": r"\shortstack{Mean-UCB\\success}",
        "Restricted delay": r"\shortstack{Restricted delay\\(rounds)}",
        "Records at confirmation": r"\shortstack{Records at\\confirmation}",
        "Patterns at confirmation": r"\shortstack{Patterns at\\confirmation}",
    }
    rows = [" & ".join(breaks.get(c,esc(c)) for c in data.columns)+r" \\", r"\hline"]
    rows += [" & ".join(esc(v) for v in r)+r" \\" for r in data.itertuples(index=False,name=None)]
    text = "\n".join([r"\begin{table*}[t]",r"\centering",r"\small",
        r"\setlength{\tabcolsep}{3pt}",r"\caption{"+caption+"}",r"\label{"+label+"}",
        r"\begin{tabular}{"+"l"*len(data.columns)+"}",r"\hline",*rows,r"\hline",
        r"\end{tabular}",r"\end{table*}"])
    (out/"tables"/f"{name}.tex").write_text(text+"\n")


def plot_all(out, frames, spatial):
    plt.rcParams.update({"font.family":"serif","font.size":8,"axes.labelsize":8,"legend.fontsize":7,
                         "pdf.fonttype":42, "savefig.pad_inches":.06})
    colors = dict(zip(METHODS, ("#277da1","#e58f26","#438c4c","#c63845")))
    def save(fig, name):
        for ext in ("png","pdf"):
            fig.savefig(out/"figures"/f"{name}.{ext}",bbox_inches="tight",dpi=220)
        plt.close(fig)
    def curve(ax, arrays, name, color):
        a=np.asarray(arrays,float)
        mu,sd=a.mean(0),a.std(0,ddof=1)
        ax.plot(range(30),mu,label=name,c=color)
        ax.fill_between(range(30),np.clip(mu-sd,0,1),np.clip(mu+sd,0,1),color=color,alpha=.12)
    fig,ax=plt.subplots(figsize=(3.65,3.05))
    for j,ep in enumerate(EP):
        grouped=spatial.groupby("rho")[ep+"_restricted_delay"].agg(["mean","std"])
        x=grouped.index.to_numpy()*100
        ax.errorbar(x,grouped["mean"],yerr=grouped["std"],marker=("o","s","^")[j],
                    capsize=3,label=EP_LABEL[ep],alpha=.8)
        for rho, r in grouped.iterrows():
            n=int(spatial.loc[spatial.rho==rho,ep+"_attained"].sum())
            ax.annotate(f"{n}/10",(100*rho,r["mean"]),xytext=(0,5+9*j),
                        textcoords="offset points",ha="center",fontsize=6)
    ax.set(xlabel="Exposed clients (%)",ylabel="Restricted confirmation\ndelay (rounds)",xticks=[10,30,50])
    low,high=ax.get_ylim()
    ax.set_ylim(0,high+4)
    ax.legend(loc="upper right")
    ax.grid(alpha=.2)
    fig.tight_layout()
    save(fig,"figure_1_calibrated_spatial_exposure_cse")
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.05),sharey=True)
    for ax,d in zip(axes,DATASETS):
        for m in METHODS:
            curve(ax,[frames[d,m,s].target_alert_recall_unexposed for s in range(10)],LABELS[m],colors[m])
        ax.axhline(.9,ls="--",c="gray",lw=.8)
        ax.axvline(10,ls="--",c="gray",lw=.8)
        ax.set(title=DS[d],xlabel="Communication round",ylim=(-.02,1.04))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Unexposed-client\ntarget-alert recall")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",ncol=4)
    fig.tight_layout(rect=(0,0,1,.89))
    save(fig,"figure_2_calibrated_target_alert_trajectories")
    fig,ax=plt.subplots(figsize=(3.65,3.1))
    for m,col,name,color in (
        ("fedccfa","emerging_recall_exposed","FedCCFA: exposed",colors["fedavg"]),
        ("fedccfa","emerging_recall_unexposed","FedCCFA: unexposed",colors["feddrift"]),
        ("fedccfa_global_classifier","emerging_recall_unexposed","GC: unexposed",colors["fedccfa"])):
        if any(col not in frames["cse",m,s] for s in range(10)):
            raise ValueError(f"Figure 3 requires multiclass diagnostic column {col}")
        a=np.asarray([frames["cse",m,s][col].to_numpy(float) for s in range(10)])
        if col=="emerging_recall_exposed":
            a[:,:10]=np.nan
        curve(ax,a,name,color)
    ax.axvline(10,ls="--",c="gray",lw=.8)
    ax.axhline(.9,ls="--",c="gray",lw=.8)
    ax.set(xlabel="Communication round",ylabel="Multiclass target-family\nrecall",ylim=(-.02,1.04))
    fig.legend(*ax.get_legend_handles_labels(),loc="upper center",ncol=1)
    fig.tight_layout(rect=(0,0,1,.80))
    save(fig,"figure_3_acquisition_vs_dissemination_original_ews")
    fig,axes=plt.subplots(3,2,figsize=(7.1,6.2))
    for i,ep in enumerate(EP):
        for ax,d in zip(axes[i],DATASETS):
            for m in METHODS:
                curve(ax,[frames[d,m,s][FP_COL[ep]] for s in range(10)],LABELS[m],colors[m])
            ax.axhline(.01,ls="--",c="black",lw=.8)
            ax.axvline(10,ls="--",c="gray",lw=.8)
            ax.grid(alpha=.2)
            if i==0: ax.set_title(DS[d])
            if i==2: ax.set_xlabel("Communication round")
        axes[i,0].set_ylabel(EP_LABEL[ep]+"\nFPR statistic")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="upper center",ncol=4)
    fig.tight_layout(rect=(0,0,1,.95))
    save(fig,"figure_S1_target_fpr_trajectories")


def updated_main():
    import sys
    central, sweep, out = [Path(x).expanduser().resolve() for x in sys.argv[1:4]]
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output folder is not empty: {out}. Choose a new folder.")
    for root in (central,sweep):
        if not root.is_dir(): raise FileNotFoundError(root)
    print(f"central_root={central}\nspatial_root={sweep}\noutput={out}",flush=True)
    out.mkdir(parents=True,exist_ok=True)
    for d in ("analysis","figures","tables"): (out/d).mkdir()
    frames, spatial_frames = {},{}
    with tempfile.TemporaryDirectory(prefix="fedccfa_analysis_") as tmp:
        normalized_root=Path(tmp)
        expected=set(itertools.product(DATASETS,METHODS,range(10)))
        found=set()
        for folder in central.glob("*/*/seed_*"):
            if folder.is_dir():
                d,m,s=folder.relative_to(central).parts
                found.add((d,m,int(s.removeprefix("seed_"))))
        if found!=expected: raise ValueError(f"Central matrix mismatch: missing={expected-found}, extra={found-expected}")
        files=[]
        for i,(d,m,s) in enumerate(sorted(expected),1):
            folder=central/d/m/f"seed_{s}"
            frame=load_run(folder,f"central/{d}/{m}/seed_{s}",.3)
            frames[d,m,s]=frame
            p=normalized_root/d/m/f"seed_{s}"/"normalized_rounds.csv"
            p.parent.mkdir(parents=True,exist_ok=True)
            frame.to_csv(p,index=False)
            files.append(p)
            if i%10==0: print(f"Validated central: {i}/80",flush=True)
        targets=build_target_per_run(files,normalized_root,ARGS)
        for i,r in targets.iterrows():
            for k,v in enrich(frames[r.dataset_key,r.method,int(r.seed)]).items():
                targets.loc[i,k]=v
        utility,directions=build_utility_per_run(files,normalized_root,10)
        if len(directions)!=10: raise ValueError("Expected all five final and five temporal-average utility metrics")
        if not np.isfinite(utility[list(directions)].to_numpy(float)).all():
            raise ValueError("Missing/nonfinite general utility observations")
        print("Computing exact paired tests and 100,000-resample confidence intervals...",flush=True)
        primary={e+"_restricted_delay":"lower" for e in EP}
        primary.update({e+"_attained":"higher" for e in EP})
        tests=paired_tests(targets,primary,binary_metrics={e+"_attained" for e in EP})
        tests=add_intervals(tests,targets)
        utests=add_intervals(paired_tests(utility,directions),utility)
    spatial_rows=[]
    expected_spatial={(rho,s) for rho in (.1,.3,.5) for s in range(10)}
    found_spatial=set()
    for p in sweep.glob("cse/fedavg/rho*/seed_*"):
        if p.is_dir(): found_spatial.add((int(p.parent.name.removeprefix("rho"))/100,int(p.name.removeprefix("seed_"))))
    if expected_spatial!=found_spatial: raise ValueError(f"Spatial matrix mismatch: missing={expected_spatial-found_spatial}, extra={found_spatial-expected_spatial}")
    for i,(rho,seed) in enumerate(sorted(expected_spatial),1):
        directory=sweep/"cse"/"fedavg"/f"rho{int(rho*100)}"/f"seed_{seed}"
        f=load_run(directory,f"spatial/rho{int(rho*100)}/seed_{seed}",rho)
        spatial_frames[rho,seed]=f
        spatial_rows.append({"rho":rho,"seed":seed,**enrich(f)})
        if i%10==0: print(f"Validated spatial: {i}/30",flush=True)
    spatial=pd.DataFrame(spatial_rows)
    spatial_tests=[]
    for ep in EP:
        for a,b in ((.3,.1),(.5,.1),(.5,.3)):
            aa=spatial[spatial.rho==a].set_index("seed").sort_index()
            bb=spatial[spatial.rho==b].set_index("seed").sort_index()
            delta=(aa[ep+"_restricted_delay"]-bb[ep+"_restricted_delay"]).to_numpy()
            lo,hi=ci(delta)
            spatial_tests.append({"criterion":ep,"metric":"restricted_delay","focal_rho":a,"reference_rho":b,
                "paired_n":10,"difference_focal_minus_reference":delta.mean(),"ci_low":lo,"ci_high":hi,
                "p_exact":exact_sign_flip_pvalue(delta)})
            wins=int(((aa[ep+"_attained"]==1)&(bb[ep+"_attained"]==0)).sum())
            losses=int(((aa[ep+"_attained"]==0)&(bb[ep+"_attained"]==1)).sum())
            spatial_tests.append({"criterion":ep,"metric":"attainment","focal_rho":a,"reference_rho":b,
                "paired_n":10,"difference_focal_minus_reference":float((aa[ep+"_attained"]-bb[ep+"_attained"]).mean()),
                "p_exact":float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.})
    st=pd.DataFrame(spatial_tests)
    for _,idx in st.groupby(["criterion","metric"]).groups.items():
        st.loc[idx,"p_holm"]=holm_adjust(st.loc[idx,"p_exact"])
    consistency=[]
    for seed in range(10):
        row={"seed":seed}
        for name in ("target_alert_recall_unexposed",*FP_COL.values(),
                     "cumulative_selected_distinct_emerging_records",
                     "cumulative_selected_distinct_emerging_feature_patterns"):
            a=frames["cse","fedavg",seed][name].to_numpy(float)
            b=spatial_frames[.3,seed][name].to_numpy(float)
            row[name+"_max_abs_difference"]=float(np.max(np.abs(a-b)))
        row["identical_at_1e_10"]=all(np.isfinite(v) and abs(v)<=1e-10 for k,v in row.items() if k!="seed")
        consistency.append(row)
    consistency=pd.DataFrame(consistency)
    if not consistency.identical_at_1e_10.all():
        WARNINGS.append("Spatial rho=0.30 differs from central CSE FedAvg. Inspect consistency CSV and run configurations; no values were adjusted to force agreement.")
    for name,df in (("central_per_run",targets),("central_paired_tests",tests),("utility_per_run",utility),
                    ("utility_paired_tests",utests),("spatial_per_run",spatial),("spatial_paired_tests",st),
                    ("rho30_central_consistency",consistency),("split_audit",pd.DataFrame(SPLIT_ROWS))):
        df.to_csv(out/"analysis"/f"{name}.csv",index=False)
    for prefix,stem in (("final_","utility_final"),("post_","utility_post_drift_mean")):
        cols=["dataset_key","method","seed"]+[c for c in directions if c.startswith(prefix)]
        utility[cols].rename(columns=lambda c:c.replace("attack_recall","target_family_recall").replace("benign_fpr","benign_to_any_attack_fpr")).to_csv(out/"analysis"/f"{stem}_per_run.csv",index=False)
        utests[utests.metric.str.startswith(prefix)].to_csv(out/"analysis"/f"{stem}_paired_tests.csv",index=False)
    summary_metrics=[c for c in targets.select_dtypes(include="number") if c!="seed"]
    tsum=group_summary(targets,summary_metrics,binary_metrics={c for c in targets if c.endswith(("_attained","_sustained"))})
    tsum.to_csv(out/"analysis"/"central_summary.csv",index=False)
    group_summary(utility,list(directions)).to_csv(out/"analysis"/"utility_summary.csv",index=False)
    spatial.groupby("rho").agg({c:["mean","std"] for c in spatial.select_dtypes("number") if c not in ("rho","seed")}).to_csv(out/"analysis"/"spatial_summary.csv")
    mainrows,allrows,urows,postrows,calrows,pbrows=[],[],[],[],[],[]
    for d,m in itertools.product(DATASETS,METHODS):
        g=targets[(targets.dataset_key==d)&(targets.method==m)]
        u=utility[(utility.dataset_key==d)&(utility.method==m)]
        base={"Dataset":DS[d],"Method":LABELS[m]}
        mainrows.append({**base,"Mean-UCB success":f"{int(g.confidence_bound_attained.sum())}/10",
            "Restricted delay":mean_sd(g,"confidence_bound_restricted_delay",2),
            "Final recall":mean_sd(g,"final_recall",4),"Final target FPR":mean_sd(g,"final_target_vs_rest_fpr",4)})
        for ep in EP:
            allrows.append({**base,"Criterion":EP_LABEL[ep],"Success":f"{int(g[ep+'_attained'].sum())}/10",
                "Restricted delay":mean_sd(g,ep+"_restricted_delay",2),
                "Final sustained":f"{int(g['final_'+ep+'_sustained'].sum())}/10"})
            pbrows.append({**base,"Criterion":EP_LABEL[ep],"Row success":f"{int(g[ep+'_attained'].sum())}/10",
                "Pattern success":f"{int(g['pattern_'+ep+'_attained'].sum())}/10",
                "Pattern delay":mean_sd(g,"pattern_"+ep+"_restricted_delay",2)})
        names={"accuracy":"Accuracy","balanced_accuracy":"BAcc","macro_f1":"Macro-F1",
               "attack_recall":"Target-family recall","benign_fpr":"Broad benign FPR"}
        urows.append({**base,**{label:mean_sd(u,"final_"+col,4) for col,label in names.items()}})
        postrows.append({**base,**{label:mean_sd(u,"post_"+col+"_mean",4) for col,label in names.items()}})
        for tag,budget in (("025","0.25%"),("050","0.5%"),("100","1%")):
            calrows.append({**base,"Budget":budget,"Final recall":mean_sd(g,"final_cal"+tag+"_recall",4),
                            "Final FPR":mean_sd(g,"final_cal"+tag+"_fpr",4)})
    delaytests=tests[(tests.focal_method=="fedccfa_global_classifier")&
                     tests.comparator_method.isin(("fedavg","fedccfa"))&
                     tests.metric.str.endswith("_restricted_delay")]
    prows=[]
    for _,r in delaytests.iterrows():
        prows.append({"Dataset":DS[r.dataset_key],"Comparison":"GC vs "+LABELS[r.comparator_method],
                      "Criterion":EP_LABEL[r.metric.removesuffix("_restricted_delay")],
                      "Difference":f"{r.difference_focal_minus_reference:+.2f}",
                      "95% CI":f"[{r.bootstrap_ci_low:+.2f}, {r.bootstrap_ci_high:+.2f}]",
                      "Exact p":f"{r.p_value_exact:.4f}","Holm p":f"{r.p_value_holm:.4f}"})
    srows,reductions=[],[]
    for rho,g in spatial.groupby("rho"):
        for ep in EP:
            success=g[g[ep+"_attained"]==1]
            med=lambda c:"--" if success.empty else f"{success[c].median():.0f}"
            srows.append({"Exposure":f"{rho:.0%}","Criterion":EP_LABEL[ep],"Success":f"{len(success)}/10",
                "Restricted delay":mean_sd(g,ep+"_restricted_delay",2),
                "Records at confirmation":med(ep+"_records_at_confirmation"),
                "Patterns at confirmation":med(ep+"_patterns_at_confirmation")})
    for ep in EP:
        a=spatial.loc[spatial.rho==.1,ep+"_restricted_delay"].mean()
        b=spatial.loc[spatial.rho==.5,ep+"_restricted_delay"].mean()
        reductions.append({"Criterion":EP_LABEL[ep],"rho10_delay":a,"rho50_delay":b,
                           "difference":b-a,"reduction_percent":100*(a-b)/a})
    pd.DataFrame(reductions).to_csv(out/"analysis"/"spatial_corrected_reductions.csv",index=False)
    descriptions=[
        ("table_I_central_mean_ucb",mainrows,r"Central target alerts. Both success and delay use the mean-client FPR-bound criterion. Mean unexposed-client recall $\geq0.90$ and mean client 95\% FPR upper bound $\leq0.01$ must hold for three rounds. Delay is measured at confirmation; nonattainment contributes 20 rounds. Final observed metrics are at round 29. Mean $\pm$ sample SD over ten seeds.","tab:central_target_alert_main"),
        ("table_II_final_general_utility",urows,r"Final general multiclass utility at round 29. Target-family recall measures DDoS on CSE and botnet on CICIoT. Broad benign FPR counts assignment to any attack class. Mean $\pm$ sample SD over ten seeds.","tab:final_general_utility"),
        ("table_III_paired_calibrated_dissemination",prows,r"Paired restricted-delay differences: global sharing minus reference. Marginal percentile-bootstrap 95\% intervals use 100,000 resamples. Exact sign-flip tests are Holm-adjusted across six comparisons within each dataset--metric family.","tab:paired_calibrated_dissemination"),
        ("table_S1_all_endpoint_outcomes",allrows,"All endpoint outcomes. All-site applies the maximum to client FPR bounds; recall remains a client average. Bounds do not provide simultaneous certification.","tab:all_endpoint_outcomes"),
        ("table_S2_spatial_confirmation",srows,r"Spatial confirmation delays, mean $\pm$ sample SD. Evidence values are medians among attained seeds only. Confirmation occurs at the third qualifying round.","tab:spatial_confirmation"),
        ("table_S3_calibration_sensitivity",calrows,"Final recall and target FPR across calibration budgets. Group means do not establish per-seed attainment.","tab:calibration_sensitivity"),
        ("table_S4_pattern_balanced",pbrows,"Pattern-balanced sustained attainment uses the same negative FPR criterion and one representative per positive feature pattern.","tab:pattern_balanced"),
        ("table_S5_post_drift_mean_utility",postrows,"Temporal mean general utility over rounds 10--29, summarized over seeds. These are not final-round metrics.","tab:post_drift_utility")]
    for name,rows,caption,label in descriptions:
        make_table(out,name,pd.DataFrame(rows),caption,label)
    mechanism=[]
    for (d,m,s),f in frames.items():
        if m!="feddrift":continue
        needed=("feddrift_new_models_created","feddrift_n_global_models")
        if not all(c in f for c in needed):
            WARNINGS.append(f"{d}/FedDrift/seed_{s}: model-count diagnostics unavailable")
            continue
        created=pd.to_numeric(f[needed[0]],errors="raise")
        mechanism.append({"dataset":d,"seed":s,"pre_drift_created":created.iloc[:10].sum(),
            "post_drift_created":created.iloc[10:].sum(),"maximum_models":f[needed[1]].max(),
            "creation_rounds":json.dumps(f.loc[created>0,"round"].tolist())})
    pd.DataFrame(mechanism).to_csv(out/"analysis"/"feddrift_mechanism.csv",index=False)
    print("Generating figures and paper tables...",flush=True)
    plot_all(out,frames,spatial)
    report=["# Corrected v4 analysis","",f"Central: {central}",f"Spatial: {sweep}","",
        "80 central runs and 30 spatial runs; seeds 0--9. Confirmation is the third qualifying round; horizon 20.",
        "Calibration budget 0.5%; held-out FPR limit 1%; recall target 0.90.",
        "Split audit counters are read from metadata; raw-feature separation is not recomputed.",
        "Capture independence and train-only preprocessing provenance remain unverified.",
        "Figure 3 uses multiclass columns from these v4 runs, not a separate legacy run matrix.","",
        "## Central mean-UCB results","",markdown_table(pd.DataFrame(mainrows)),
        "","## Spatial corrected results","",markdown_table(pd.DataFrame(srows)),
        "","## Spatial corrected reductions","",markdown_table(pd.DataFrame(reductions)),
        "","## Paired dissemination comparisons","",markdown_table(pd.DataFrame(prows)),
        "","## Interpretation","",
        "Final utility and temporal-average utility are separately tested. Attack recall means target-family recall.",
        "All-site is a pointwise FPR-bound criterion; recall is averaged across unexposed clients.",
        "Evidence-at-confirmation summaries exclude nonattaining seeds; delay summaries include them at 20.",
        "Repeated client-model decisions on fixed examples are not independent experimental replicates.",
        "Holm families: six method comparisons per dataset/metric; three exposure comparisons per spatial criterion/metric.",
        "Bootstrap intervals are marginal. Precision and AUPRC depend on the constructed test mixture.","",
        "## Warnings and remaining provenance checks","",
        *["- "+w for w in WARNINGS],
        "- Verify that feature selection, imputation, and normalization were fitted without evaluation data.",
        "- Verify threat-screening test-set independence separately.",
        "- Read analysis/rho30_central_consistency.csv before treating spatial rho=0.30 as an exact central replication."]
    (out/"RESULTS_REPORT.md").write_text("\n".join(report)+"\n")
    (out/"analysis"/"input_provenance.json").write_text(json.dumps(PROVENANCE,indent=2,default=str))
    (out/"analysis"/"warnings.json").write_text(json.dumps(WARNINGS,indent=2))
    (out/"README.txt").write_text(
        "Main figures: figures/figure_1*, figure_2*, figure_3*. Supplement: figure_S1 FPR curves.\n"
        "Main tables: tables/table_I*, table_II*, table_III*. Supplements: tables/table_S1* through S5*.\n"
        "CSV, Markdown and LaTeX tables are supplied. PDF and PNG figures are supplied.\n"
        "Raw run inputs are unchanged. Confirmation timing is recomputed from raw round CSVs.\n"
        "See RESULTS_REPORT.md and analysis/warnings.json before using results in the manuscript.\n")
    print("\n"+pd.DataFrame(mainrows).to_string(index=False),flush=True)
    print("\nCorrected spatial reductions:\n"+pd.DataFrame(reductions).to_string(index=False),flush=True)
    print(f"\nDONE: {out}\nReport: {out/'RESULTS_REPORT.md'}\nWarnings: {len(WARNINGS)}",flush=True)


if __name__ == "__main__":
    try:
        updated_main()
    except Exception as exc:
        print(f"\nERROR: {exc}",file=__import__("sys").stderr,flush=True)
        raise SystemExit(1)

FEDCCFA_PYTHON
