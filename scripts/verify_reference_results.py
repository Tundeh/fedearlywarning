#!/usr/bin/env python3
"""Verify that packaged (or regenerated) results reproduce the accepted paper.

Checks, failing loudly (non-zero exit) on any mismatch:

1. Coverage: 10 matched seeds (0-9) for every central dataset x method and for
   every exposure level of the CSE sweep, each with rounds CSV + metadata.
2. Protocol invariants, read from every run's metadata (resolved config and
   split audits): clients, rounds, emergence round, exposure fraction,
   participation, evidence mode, records/round, initial model seed, target-alert
   budgets, 65/5/30 purged split, challenge cap, dataset dims and variants.
3. Table I fingerprints (mean-UCB success and restricted delay), recomputed from
   analysis/central_per_run.csv and cross-checked against the formatted table.
4. Table II fingerprints (BAcc, Macro-F1, broad benign FPR), recomputed from
   analysis/utility_per_run.csv.
5. Table III fingerprints (mean-UCB paired delay differences, bootstrap CI,
   exact and Holm-adjusted p), read from analysis/central_paired_tests.csv.

Usage:
    python scripts/verify_reference_results.py
    python scripts/verify_reference_results.py --analysis-dir analysis_output/paper_X \
        --results-root outputs

Only the Python standard library is required.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("fedavg", "feddrift", "fedccfa", "fedccfa_global_classifier")
LABEL = {"fedavg": "FedAvg", "feddrift": "FedDrift", "fedccfa": "FedCCFA",
         "fedccfa_global_classifier": "FedCCFA-GC"}
DS_LABEL = {"cse": "CSE-CIC-IDS2018", "ciciot": "CICIoT2023"}
GROUP = {"cse": "cse_central", "ciciot": "ciciot_central"}

# ---- Accepted-paper fingerprints (specification Section 4) -----------------
TABLE_I = {  # (dataset, method): (successes out of 10, delay mean, delay sd)
    ("cse", "fedavg"): (10, 6.00, 0.00),
    ("cse", "feddrift"): (0, 20.00, 0.00),
    ("cse", "fedccfa"): (8, 13.70, 3.77),
    ("cse", "fedccfa_global_classifier"): (10, 3.10, 0.32),
    ("ciciot", "fedavg"): (10, 5.30, 3.77),
    ("ciciot", "feddrift"): (0, 20.00, 0.00),
    ("ciciot", "fedccfa"): (10, 3.50, 0.53),
    ("ciciot", "fedccfa_global_classifier"): (10, 3.10, 0.32),
}
TABLE_II = {  # (dataset, method): {metric: value at 3 d.p.}
    ("cse", "fedavg"): {"final_balanced_accuracy": 0.176, "final_macro_f1": 0.145, "final_benign_fpr": 0.000},
    ("cse", "fedccfa_global_classifier"): {"final_balanced_accuracy": 0.176, "final_macro_f1": 0.145, "final_benign_fpr": 0.000},
    ("ciciot", "fedavg"): {"final_balanced_accuracy": 0.557, "final_macro_f1": 0.504, "final_benign_fpr": 0.441},
    ("ciciot", "fedccfa_global_classifier"): {"final_balanced_accuracy": 0.533, "final_macro_f1": 0.462, "final_benign_fpr": 0.441},
}
TABLE_III = {  # (dataset, comparator): (diff, ci_low, ci_high, exact p, holm p)
    ("cse", "fedavg"): (-2.90, -3.00, -2.70, 0.0020, 0.0117),
    ("cse", "fedccfa"): (-10.60, -13.00, -8.40, 0.0020, 0.0117),
    ("ciciot", "fedavg"): (-2.20, -4.40, -1.00, 0.0020, 0.0117),
    ("ciciot", "fedccfa"): (-0.40, -0.80, 0.00, 0.2188, 0.2188),
}
DATASET_INVARIANTS = {
    "cse": {"input_dim": 70, "num_classes": 8, "variant": "ddos attack-hoic", "family": "ddos"},
    "ciciot": {"input_dim": 39, "num_classes": 7, "variant": "mirai-udpplain", "family": "botnet"},
}

FAILURES: list[str] = []
N_CHECKS = 0


def check(cond: bool, msg: str) -> None:
    global N_CHECKS
    N_CHECKS += 1
    if not cond:
        FAILURES.append(msg)
        print(f"  FAIL  {msg}")


def close(observed: float, expected: float, tol: float, what: str) -> None:
    check(abs(float(observed) - float(expected)) <= tol,
          f"{what}: observed {observed!r} expected {expected!r} (tol {tol})")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"Missing {path}")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def get(d: dict, dotted: str):
    for part in dotted.split("."):
        d = d.get(part) if isinstance(d, dict) else None
    return d


# ---------------------------------------------------------------------------
def verify_coverage_and_protocol(results: Path) -> None:
    print("[1-2] Seed coverage and protocol invariants")
    runs = []
    for ds in ("cse", "ciciot"):
        for m in METHODS:
            for s in range(10):
                runs.append((ds, m, 0.30, results / GROUP[ds] / m / f"seed_{s}"))
    for rho in (10, 30, 50):
        for s in range(10):
            runs.append(("cse", "fedavg", rho / 100, results / "cse_exposure_sweep" / f"rho{rho}" / f"seed_{s}"))
    for ds, m, rho, d in runs:
        rounds = list(d.glob("*_rounds.csv"))
        metas = list(d.glob("*_metadata.json"))
        check(len(rounds) == 1 and len(metas) == 1, f"{d}: expected one rounds CSV and one metadata JSON")
        if not metas:
            continue
        meta = json.loads(metas[0].read_text())
        c = meta["config"]
        inv = DATASET_INVARIANTS[ds]
        tag = str(d.relative_to(results))
        expect = {
            "client_num": 20, "rounds": 30, "initial_model_seed": 424242, "batch_size": 128,
            "threat.evidence_mode": "fixed_unique", "threat.fixed_total_attack_per_round": 3000,
            "threat.milestones": [[0, 0.0], [10, rho]],
            "threat.emerging_variant": inv["variant"], "threat.attack_family": inv["family"],
            "participation.normal_prob": 1.0, "participation.exposed_prob": 1.0,
            "stream.window_size": 2048,
            "stream.attack_guard_fraction": 0.05, "stream.attack_test_fraction": 0.30,
            "evaluation.target_alert_primary_calibration_fpr_limit": 0.005,
            "evaluation.target_alert_test_fpr_limit": 0.01,
            "evaluation.target_alert_fpr_confidence": 0.95,
            "evaluation.target_alert_sustained_rounds": 3,
            "input_dim": inv["input_dim"], "num_classes": inv["num_classes"],
        }
        if m == "fedccfa_global_classifier":
            expect["fedccfa_sharing_mode"] = "global_classifier"
        elif m == "fedccfa":
            expect["fedccfa_sharing_mode"] = "clustered"
        for key, val in expect.items():
            obs = get(c, key)
            if isinstance(val, float):
                check(obs is not None and abs(float(obs) - val) < 1e-12, f"{tag}: {key}={obs!r} != {val!r}")
            else:
                check(obs == val, f"{tag}: {key}={obs!r} != {val!r}")
        check(int(c.get("seed")) == int(d.name.removeprefix("seed_")), f"{tag}: seed mismatch")
        audit = meta.get("emerging_split_audit", {})
        check(abs(audit.get("train_fraction_configured", -1) - 0.65) < 1e-12, f"{tag}: train fraction != 0.65")
        check(abs(audit.get("guard_fraction_configured", -1) - 0.05) < 1e-12, f"{tag}: guard fraction != 0.05")
        check(abs(audit.get("candidate_challenge_fraction_configured", -1) - 0.30) < 1e-12, f"{tag}: challenge fraction != 0.30")
        check(audit.get("n_feature_overlap_rows_after_purge") == 0, f"{tag}: feature overlap after purge")
        check(audit.get("n_index_overlap_train_final_challenge") == 0, f"{tag}: train/challenge index overlap")
        check(0 < int(meta.get("n_attack_challenge", 0)) <= 5000, f"{tag}: challenge rows outside (0, 5000]")
        check(meta.get("evidence_mode") == "fixed_unique", f"{tag}: metadata evidence_mode")
        check(int(meta.get("first_drift_round", -1)) == 10, f"{tag}: first_drift_round != 10")
    print(f"      {len(runs)} runs checked (80 central + 30 sweep)")


def verify_table_i(analysis: Path) -> None:
    print("[3] Table I: mean-UCB restricted confirmation delay and success")
    rows = read_csv(analysis / "analysis" / "central_per_run.csv")
    for (ds, m), (succ, mean, sd) in TABLE_I.items():
        g = [r for r in rows if r["dataset_key"] == ds and r["method"] == m]
        seeds = sorted(int(float(r["seed"])) for r in g)
        check(seeds == list(range(10)), f"Table I {ds}/{m}: seeds {seeds} != 0..9")
        delays = [float(r["confidence_bound_restricted_delay"]) for r in g]
        attained = sum(int(float(r["confidence_bound_attained"])) for r in g)
        check(attained == succ, f"Table I {ds}/{m}: success {attained}/10 != {succ}/10")
        close(statistics.mean(delays), mean, 0.005, f"Table I {ds}/{m} delay mean")
        close(statistics.stdev(delays), sd, 0.005, f"Table I {ds}/{m} delay SD")
        check(all(0 <= x <= 20 for x in delays), f"Table I {ds}/{m}: delay outside [0, 20]")
    # Cross-check the formatted paper table.
    table = read_csv(analysis / "tables" / "table_I_central_mean_ucb.csv")
    for (ds, m), (succ, mean, sd) in TABLE_I.items():
        r = [t for t in table if t["Dataset"] == DS_LABEL[ds] and t["Method"] == LABEL[m]]
        check(len(r) == 1, f"Table I CSV row missing for {ds}/{m}")
        if r:
            check(r[0]["Mean-UCB success"] == f"{succ}/10", f"Table I CSV {ds}/{m} success cell {r[0]['Mean-UCB success']}")
            dm, dsd = (float(x) for x in r[0]["Restricted delay"].split("±"))
            close(dm, mean, 0.005, f"Table I CSV {ds}/{m} delay")
            close(dsd, sd, 0.005, f"Table I CSV {ds}/{m} SD")


def verify_table_ii(analysis: Path) -> None:
    print("[4] Table II: final general-detection utility")
    rows = read_csv(analysis / "analysis" / "utility_per_run.csv")
    for (ds, m), metrics in TABLE_II.items():
        g = [r for r in rows if r["dataset_key"] == ds and r["method"] == m]
        check(len(g) == 10, f"Table II {ds}/{m}: {len(g)} seeds != 10")
        for col, val in metrics.items():
            # Paper values are shown at 3 d.p.: allow half a display unit (+eps).
            close(statistics.mean(float(r[col]) for r in g), val, 0.0005 + 1e-9,
                  f"Table II {ds}/{m} {col}")


def verify_table_iii(analysis: Path) -> None:
    print("[5] Table III: paired mean-UCB delay comparisons (GC minus reference)")
    rows = read_csv(analysis / "analysis" / "central_paired_tests.csv")
    for (ds, comp), (diff, lo, hi, p, holm) in TABLE_III.items():
        r = [x for x in rows if x["dataset_key"] == ds and x["metric"] == "confidence_bound_restricted_delay"
             and x["focal_method"] == "fedccfa_global_classifier" and x["comparator_method"] == comp]
        check(len(r) == 1, f"Table III row missing for {ds} GC vs {comp}")
        if not r:
            continue
        r = r[0]
        tag = f"Table III {ds} GC vs {LABEL[comp]}"
        check(int(float(r["paired_n"])) == 10, f"{tag}: paired_n != 10")
        close(float(r["difference_focal_minus_reference"]), diff, 0.005, f"{tag} difference")
        close(float(r["bootstrap_ci_low"]), lo, 0.005, f"{tag} CI low")
        close(float(r["bootstrap_ci_high"]), hi, 0.005, f"{tag} CI high")
        close(float(r["p_value_exact"]), p, 0.00005, f"{tag} exact p")
        close(float(r["p_value_holm"]), holm, 0.00005, f"{tag} Holm p")
    # Holm family size: six method comparisons per dataset-metric family.
    for ds in ("cse", "ciciot"):
        fam = [x for x in rows if x["dataset_key"] == ds and x["metric"] == "confidence_bound_restricted_delay"]
        check(len(fam) == 6, f"Table III {ds}: Holm family has {len(fam)} comparisons, expected 6")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=str(ROOT / "reference_results" / "paper_analysis"))
    ap.add_argument("--results-root", default=str(ROOT / "reference_results"))
    a = ap.parse_args()
    analysis, results = Path(a.analysis_dir).resolve(), Path(a.results_root).resolve()
    print(f"analysis_dir={analysis}\nresults_root={results}")
    verify_coverage_and_protocol(results)
    verify_table_i(analysis)
    verify_table_ii(analysis)
    verify_table_iii(analysis)
    print()
    if FAILURES:
        print(f"VERIFICATION FAILED: {len(FAILURES)} of {N_CHECKS} checks failed.")
        return 1
    print(f"VERIFICATION PASSED: all {N_CHECKS} checks reproduce the accepted-paper fingerprints.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
