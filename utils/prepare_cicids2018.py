#!/usr/bin/env python3
"""Preprocess CSE-CIC-IDS2018-style flow CSV files for FedCCFA.

The script is intentionally tolerant of common CICFlowMeter column-name
variants. It produces:

    <output>.npz   : x, y, client_ids, times
    <output>.json  : feature/class schema and preprocessing metadata

Example
-------
python utils/prepare_cicids2018.py \
    --input-root data/CSE-CIC-IDS2018 \
    --output data/processed/cicids2018.npz \
    --num-logical-clients 20 \
    --max-rows-per-file 250000

For a natural site/client split, pass e.g. ``--client-column "Dst IP"`` when
that field is present in the CSV export. If no suitable identity field is
available, the script creates deterministic logical clients and records this
fact in the schema; do not describe those as physical departments in a paper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


LABEL_CANDIDATES = ["Label", "label", "Class", "class"]
TIME_CANDIDATES = ["Timestamp", "timestamp", "Time", "time"]
CLIENT_CANDIDATES = [
    "Dst IP", "Destination IP", "DestinationIP", "dst_ip",
    "Src IP", "Source IP", "SourceIP", "src_ip",
]
IDENTIFIER_PATTERNS = (
    "flow id", "src ip", "source ip", "dst ip", "destination ip",
    "timestamp", "simillarhttp", "similarhttp",
)

CANONICAL_ORDER = [
    "benign",
    "botnet",
    "brute_force",
    "dos",
    "ddos",
    "web",
    "infiltration",
    "other_attack",
]


def _clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def canonical_attack_family(raw_label: object) -> str:
    s = _clean_text(raw_label).replace("–", "-").replace("—", "-")
    if s in {"benign", "normal", "0"}:
        return "benign"
    if "bot" in s or "mirai" in s:
        return "botnet"
    if any(k in s for k in ("brute", "ftp", "ssh")):
        return "brute_force"
    if "ddos" in s:
        return "ddos"
    if any(k in s for k in ("dos", "hulk", "slowloris", "slowhttp", "goldeneye")):
        return "dos"
    if any(k in s for k in ("xss", "sql", "web attack", "webattack")):
        return "web"
    if "infil" in s:
        return "infiltration"
    return "other_attack"


def find_column(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    columns = list(columns)
    exact = {str(c).strip().lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in exact:
            return exact[candidate.lower()]
    return None


def parse_timestamp(series: pd.Series, file_rank: int) -> np.ndarray:
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    # Nanoseconds. Pandas uses int64 min for NaT, so replace invalid entries.
    values = parsed.array.asi8.copy()
    invalid = values == np.iinfo(np.int64).min
    if invalid.any():
        # Preserve within-file order and file ordering for unparseable values.
        fallback_base = int(file_rank) * 10**15
        fallback = fallback_base + np.arange(len(values), dtype=np.int64)
        values[invalid] = fallback[invalid]
    return values


def should_drop_identifier(column: str) -> bool:
    low = _clean_text(column)
    return any(pattern in low for pattern in IDENTIFIER_PATTERNS)


def read_csvs(
    csv_paths: List[Path],
    max_rows_per_file: int,
    client_column: Optional[str],
) -> Tuple[pd.DataFrame, str, Optional[str], Optional[str]]:
    frames: List[pd.DataFrame] = []
    chosen_label = None
    chosen_time = None
    chosen_client = None

    for rank, path in enumerate(csv_paths):
        nrows = max_rows_per_file if max_rows_per_file > 0 else None
        df = pd.read_csv(path, low_memory=False, nrows=nrows)
        df.columns = [str(c).strip() for c in df.columns]

        label_col = find_column(df.columns, LABEL_CANDIDATES)
        if label_col is None:
            print(f"[skip] {path}: no label column")
            continue

        time_col = find_column(df.columns, TIME_CANDIDATES)
        requested_client = None
        if client_column:
            requested_client = find_column(df.columns, [client_column])
            if requested_client is None:
                raise ValueError(f"Client column {client_column!r} not found in {path}")
        else:
            requested_client = find_column(df.columns, CLIENT_CANDIDATES)

        df["__source_file__"] = str(path)
        df["__file_rank__"] = rank
        if time_col is not None:
            df["__event_time__"] = parse_timestamp(df[time_col], rank)
        else:
            df["__event_time__"] = rank * 10**15 + np.arange(len(df), dtype=np.int64)

        if requested_client is not None:
            df["__client_key__"] = df[requested_client].fillna("UNKNOWN").astype(str)
        else:
            # Defer deterministic logical assignment until all rows are merged.
            df["__client_key__"] = ""

        frames.append(df)
        chosen_label = chosen_label or label_col
        chosen_time = chosen_time or time_col
        chosen_client = chosen_client or requested_client
        print(f"[read] {path.name}: {len(df):,} rows")

    if not frames:
        raise RuntimeError("No usable CIC CSV files were found")
    return pd.concat(frames, ignore_index=True), chosen_label, chosen_time, chosen_client


def build_logical_clients(
    frame: pd.DataFrame,
    natural_client_col: Optional[str],
    num_logical_clients: int,
) -> Tuple[np.ndarray, Dict[str, int], str]:
    if natural_client_col is not None:
        keys = frame["__client_key__"].astype(str).to_numpy()
        unique = sorted(set(keys.tolist()))
        source = f"natural:{natural_client_col}"
        # If there are many hosts, map hosts (not individual rows) to a fixed
        # number of logical sites so the same host always stays at one site.
        mapping = {key: idx % num_logical_clients for idx, key in enumerate(unique)}
        ids = np.asarray([mapping[k] for k in keys], dtype=np.int64)
        return ids, mapping, source

    # No natural identity field was available. Allocate chronological blocks
    # from each source file across logical clients. This is deterministic and
    # avoids using labels to create the federation.
    ids = np.empty(len(frame), dtype=np.int64)
    mapping: Dict[str, int] = {}
    for file_name, group_index in frame.groupby("__source_file__", sort=True).groups.items():
        idx = np.asarray(list(group_index), dtype=np.int64)
        idx = idx[np.argsort(frame.loc[idx, "__event_time__"].to_numpy())]
        chunks = np.array_split(idx, num_logical_clients)
        for cid, chunk in enumerate(chunks):
            ids[chunk] = cid
        mapping[str(file_name)] = -1
    return ids, mapping, "synthetic:chronological_file_blocks"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-logical-clients", type=int, default=20)
    p.add_argument("--client-column", default=None)
    p.add_argument("--max-rows-per-file", type=int, default=0,
                   help="0 = all rows; useful to cap memory while debugging")
    p.add_argument("--scaler-fraction", type=float, default=0.20)
    p.add_argument("--exclude-from-scaler", default="botnet",
                   help="Canonical class excluded from scaler fitting; use 'none' to disable")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    input_root = Path(args.input_root)
    csv_paths = sorted(input_root.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files below {input_root}")

    frame, label_col, time_col, natural_client_col = read_csvs(
        csv_paths,
        max_rows_per_file=args.max_rows_per_file,
        client_column=args.client_column,
    )

    # Canonicalize labels before selecting numeric features.
    frame["__raw_variant__"] = frame[label_col].map(_clean_text)
    variant_names = sorted(frame["__raw_variant__"].unique().tolist())
    variant_to_id = {name: idx for idx, name in enumerate(variant_names)}
    variant_ids = frame["__raw_variant__"].map(variant_to_id).to_numpy(dtype=np.int64)

    frame["__family__"] = frame[label_col].map(canonical_attack_family)
    present = set(frame["__family__"].unique().tolist())
    class_names = [name for name in CANONICAL_ORDER if name in present]
    label_to_id = {name: idx for idx, name in enumerate(class_names)}
    y = frame["__family__"].map(label_to_id).to_numpy(dtype=np.int64)

    client_ids, client_mapping, client_source = build_logical_clients(
        frame,
        natural_client_col=natural_client_col,
        num_logical_clients=args.num_logical_clients,
    )
    times = frame["__event_time__"].to_numpy(dtype=np.int64)

    protected = {
        label_col, time_col, natural_client_col,
        "__source_file__", "__file_rank__", "__event_time__",
        "__client_key__", "__family__", "__raw_variant__",
    }
    feature_names: List[str] = []
    feature_arrays: List[np.ndarray] = []
    for col in frame.columns:
        if col in protected or col is None or should_drop_identifier(col):
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        arr = numeric.to_numpy(dtype=np.float64).copy()
        arr[~np.isfinite(arr)] = np.nan
        finite_ratio = np.mean(np.isfinite(arr))
        if finite_ratio < 0.95:
            continue
        median = np.nanmedian(arr)
        arr[np.isnan(arr)] = median
        if np.nanstd(arr) < 1e-12:
            continue
        feature_names.append(col)
        feature_arrays.append(arr)

    if not feature_arrays:
        raise RuntimeError("No numeric flow features survived preprocessing")

    x = np.column_stack(feature_arrays).astype(np.float32)

    # Fit normalization only on an early chronological reference period.
    order = np.argsort(times)
    n_fit = max(1, int(len(order) * args.scaler_fraction))
    fit_idx = order[:n_fit]
    excluded = args.exclude_from_scaler.strip().lower()
    if excluded != "none" and excluded in label_to_id:
        fit_idx = fit_idx[y[fit_idx] != label_to_id[excluded]]
        if len(fit_idx) == 0:
            raise RuntimeError("Scaler reference set became empty")

    mean = x[fit_idx].mean(axis=0, dtype=np.float64)
    std = x[fit_idx].std(axis=0, dtype=np.float64)
    std[std < 1e-6] = 1.0
    x = ((x - mean) / std).astype(np.float32)
    x = np.clip(x, -20.0, 20.0)

    # Global chronological order makes deterministic downstream slicing easier.
    global_order = np.argsort(times, kind="stable")
    x = x[global_order]
    y = y[global_order]
    client_ids = client_ids[global_order]
    times = times[global_order]
    variant_ids = variant_ids[global_order]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=x, y=y, client_ids=client_ids, times=times, variant_ids=variant_ids)

    schema = {
        "class_names": class_names,
        "label_to_id": label_to_id,
        "variant_names": variant_names,
        "variant_to_id": variant_to_id,
        "feature_names": feature_names,
        "client_mapping": client_mapping,
        "metadata": {
            "dataset": "CSE-CIC-IDS2018",
            "source_root": str(input_root),
            "csv_files": [str(p) for p in csv_paths],
            "rows": int(len(y)),
            "input_dim": int(x.shape[1]),
            "num_classes": int(len(class_names)),
            "num_logical_clients": int(args.num_logical_clients),
            "client_source": client_source,
            "label_column": label_col,
            "time_column": time_col,
            "scaler_fraction": float(args.scaler_fraction),
            "exclude_from_scaler": args.exclude_from_scaler,
            "scaler_mean": mean.tolist(),
            "scaler_std": std.tolist(),
            "seed": int(args.seed),
        },
    }
    output.with_suffix(".json").write_text(json.dumps(schema, indent=2))

    print("\nPrepared dataset")
    print(f"  rows       : {len(y):,}")
    print(f"  features   : {x.shape[1]}")
    print(f"  classes    : {class_names}")
    print(f"  variants   : {len(variant_names)} raw CIC labels")
    print(f"  clients    : {len(np.unique(client_ids))}")
    print(f"  client mode: {client_source}")
    print(f"  output     : {output}")


if __name__ == "__main__":
    main()
