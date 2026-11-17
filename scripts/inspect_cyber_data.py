#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.cyber_dataset import load_prepared_cyber_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    args = p.parse_args()
    d = load_prepared_cyber_data(args.data)
    print(f"rows={len(d.y):,}, input_dim={d.input_dim}, clients={len(set(d.client_ids.tolist()))}")
    print("\nCanonical classes:")
    for i, name in enumerate(d.class_names):
        print(f"  {i:2d}: {name}")
    print("\nRaw variants (use exact text in threat.emerging_variant):")
    for i, name in enumerate(d.variant_names):
        print(f"  {i:2d}: {name}")


if __name__ == "__main__":
    main()
