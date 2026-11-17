#!/usr/bin/env python3
"""Launch and validate the corrected two-dataset central experiment matrix.

The launcher clones the previously validated per-method/per-seed experiment
overlays, writes all outputs to a new root, and schedules one process per GPU.
It never writes into the contaminated experiment directories.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from typing import Optional

import yaml


REPO = Path(__file__).resolve().parents[1]
PROTOCOL_REVISION = (
    "purged_chronological_feature_disjoint_with_pattern_metrics_"
    "v4_multibudget_sustained_target_alert"
)

DATASETS = {
    "cse": {
        "name": "CSE-CIC-IDS2018",
        "template_root": Path(
            "experiments/paper_cicids_central_20260831_220443"
        ),
        "expected_row_challenge": 4439,
        "expected_pattern_challenge": 3112,
    },
    "ciciot": {
        "name": "CICIoT2023",
        "template_root": Path(
            "experiments/ciciot2023_mirai_central_20260831_210113"
        ),
        "expected_row_challenge": 5000,
        "expected_pattern_challenge": 5000,
    },
}

METHODS = {
    "fedavg": {"algorithm": "FedAvg", "sharing_mode": None},
    "feddrift": {"algorithm": "FedDrift", "sharing_mode": None},
    "fedccfa": {"algorithm": "FedCCFA", "sharing_mode": "clustered"},
    "fedccfa_global_classifier": {
        "algorithm": "FedCCFA",
        "sharing_mode": "global_classifier",
    },
}

DEFAULT_NUM_SEEDS = 10
REQUIRED_ROUND_COLUMNS = {
    "round",
    "emerging_recall_unexposed",
    "emerging_pattern_balanced_recall_unexposed",
    "cumulative_selected_distinct_emerging_records",
    "cumulative_selected_distinct_emerging_feature_patterns",
    "early_warning_threshold_cumulative_distinct_records",
    "early_warning_threshold_cumulative_distinct_feature_patterns",
    "target_alert_recall_unexposed",
    "target_alert_pattern_balanced_recall_unexposed",
    "target_alert_target_vs_rest_fpr_unexposed",
    "target_alert_benign_to_target_fpr_unexposed",
    "target_alert_nontarget_attack_to_target_fpr_unexposed",
    "target_alert_precision_unexposed",
    "target_alert_auroc_unexposed",
    "target_alert_auprc_unexposed",
    "target_alert_target_vs_rest_fpr_upper_unexposed",
    "target_alert_primary_calibration_fpr_limit",
    "target_alert_test_fpr_limit",
    "target_alert_calibration_fpr_unexposed",
    "target_alert_target_vs_rest_fp_sum_unexposed",
    "target_alert_target_vs_rest_n_sum_unexposed",
    "target_alert_calibration_025bp_recall_unexposed",
    "target_alert_calibration_050bp_recall_unexposed",
    "target_alert_calibration_100bp_recall_unexposed",
    "target_alert_empirical_target_round",
    "target_alert_empirical_current_streak",
    "target_alert_certified_target_round",
    "target_alert_certified_current_streak",
}


@dataclass(frozen=True)
class Job:
    dataset_key: str
    method_tag: str
    algorithm: str
    sharing_mode: Optional[str]
    seed: int
    config_path: Path
    run_dir: Path
    run_name: str
    rounds: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO / path


def validate_protocol(config: dict, dataset_key: str, method_tag: str, seed: int) -> None:
    errors = []
    for key in ("seed", "stream_seed", "schedule_seed", "participation_seed"):
        if int(config.get(key, -1)) != int(seed):
            errors.append(f"{key}={config.get(key)!r}, expected {seed}")
    if int(config.get("initial_model_seed", -1)) != 424242:
        errors.append("initial_model_seed must be 424242")
    if int(config.get("rounds", -1)) != 30:
        errors.append("rounds must be 30")

    threat = config.get("threat", {})
    if threat.get("evidence_mode") != "fixed_unique":
        errors.append("evidence_mode must be fixed_unique")
    if int(threat.get("fixed_total_attack_per_round", -1)) != 3000:
        errors.append("fixed_total_attack_per_round must be 3000")
    if threat.get("milestones") != [[0, 0.0], [10, 0.30]]:
        errors.append(f"unexpected milestones: {threat.get('milestones')!r}")

    participation = config.get("participation", {})
    if float(participation.get("normal_prob", -1)) != 1.0:
        errors.append("normal_prob must be 1.0")
    if float(participation.get("exposed_prob", -1)) != 1.0:
        errors.append("exposed_prob must be 1.0")

    if dataset_key == "cse":
        expected = ("ddos", "ddos attack-hoic")
    else:
        expected = ("botnet", "mirai-udpplain")
    observed = (threat.get("attack_family"), threat.get("emerging_variant"))
    if observed != expected:
        errors.append(f"threat={observed!r}, expected {expected!r}")

    expected_sharing = METHODS[method_tag]["sharing_mode"]
    if expected_sharing is not None:
        if config.get("fedccfa_sharing_mode") != expected_sharing:
            errors.append(
                f"fedccfa_sharing_mode={config.get('fedccfa_sharing_mode')!r}, "
                f"expected {expected_sharing!r}"
            )

    if errors:
        raise ValueError("; ".join(errors))


def build_jobs(
    exp_root: Path, seeds: tuple[int, ...], write_configs: bool = True
) -> list[Job]:
    jobs = []
    for dataset_key, dataset_spec in DATASETS.items():
        template_root = REPO / dataset_spec["template_root"]
        for method_tag, method_spec in METHODS.items():
            for seed in seeds:
                source = template_root / method_tag / f"seed_{seed}" / "config.yaml"
                source_seed = seed
                if not source.exists():
                    source_seed = 0
                    source = template_root / method_tag / "seed_0" / "config.yaml"
                if not source.exists():
                    raise FileNotFoundError(source)
                config = yaml.safe_load(source.read_text()) or {}
                validate_protocol(config, dataset_key, method_tag, source_seed)

                for key in (
                    "seed",
                    "stream_seed",
                    "schedule_seed",
                    "participation_seed",
                ):
                    config[key] = seed

                run_dir = exp_root / dataset_key / method_tag / f"seed_{seed}"
                source_run_name = str(config["run_name"])
                if re.search(r"seed\d+$", source_run_name) is None:
                    raise ValueError(
                        f"run_name must end in seed<number>: {source_run_name!r}"
                    )
                seeded_run_name = re.sub(
                    r"seed\d+$", f"seed{seed}", source_run_name
                )
                run_name = f"{seeded_run_name}_target_alert_v4"
                config["run_name"] = run_name
                config["result_dir"] = str(run_dir.resolve())
                config["rounds"] = 30
                config["flush_every"] = 1
                config.setdefault("stream", {})
                config["stream"]["attack_guard_fraction"] = 0.05
                config["stream"]["attack_test_fraction"] = 0.30
                config["stream"]["max_target_alert_negative_samples"] = 10000
                config["stream"]["target_alert_calibration_fraction"] = 0.50
                config.setdefault("evaluation", {})
                config["evaluation"]["target_alert_calibration_fpr_limits"] = [
                    0.0025, 0.005, 0.01
                ]
                config["evaluation"][
                    "target_alert_primary_calibration_fpr_limit"
                ] = 0.005
                config["evaluation"]["target_alert_test_fpr_limit"] = 0.01
                config["evaluation"]["target_alert_fpr_confidence"] = 0.95
                config["evaluation"]["target_alert_sustained_rounds"] = 3
                config["protocol_revision"] = PROTOCOL_REVISION
                config["config_template_seed"] = source_seed
                config["config_template_source"] = str(source.relative_to(REPO))
                validate_protocol(config, dataset_key, method_tag, seed)

                config_path = run_dir / "config.yaml"
                if write_configs:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    config_path.write_text(yaml.safe_dump(config, sort_keys=False))

                jobs.append(
                    Job(
                        dataset_key=dataset_key,
                        method_tag=method_tag,
                        algorithm=str(method_spec["algorithm"]),
                        sharing_mode=method_spec["sharing_mode"],
                        seed=seed,
                        config_path=config_path,
                        run_dir=run_dir,
                        run_name=run_name,
                        rounds=30,
                    )
                )
    return jobs


def output_paths(job: Job) -> dict[str, Path]:
    return {
        "rounds": job.run_dir / f"{job.run_name}_rounds.csv",
        "metadata": job.run_dir / f"{job.run_name}_metadata.json",
        "summary": job.run_dir / f"{job.run_name}_early_warning_summary.json",
    }


def validate_output(job: Job) -> tuple[bool, str]:
    paths = output_paths(job)
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        return False, "missing outputs: " + ", ".join(missing)

    with paths["rounds"].open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != job.rounds:
        return False, f"round rows={len(rows)}, expected {job.rounds}"
    missing_columns = REQUIRED_ROUND_COLUMNS - set(rows[0])
    if missing_columns:
        return False, f"missing columns: {sorted(missing_columns)}"
    observed_rounds = [int(float(row["round"])) for row in rows]
    if observed_rounds != list(range(job.rounds)):
        return False, "round sequence is incomplete or unordered"
    for row in rows:
        calibration_limit = float(
            row["target_alert_primary_calibration_fpr_limit"]
        )
        calibration_fpr = float(
            row["target_alert_calibration_fpr_unexposed"]
        )
        test_fpr = float(row["target_alert_target_vs_rest_fpr_unexposed"])
        test_upper = float(
            row["target_alert_target_vs_rest_fpr_upper_unexposed"]
        )
        test_n = int(float(row["target_alert_target_vs_rest_n_sum_unexposed"]))
        if calibration_fpr > calibration_limit + 1e-12:
            return False, "primary calibration FPR exceeds its budget"
        if test_n <= 0:
            return False, "target-alert test denominator is non-positive"
        if test_upper + 1e-12 < test_fpr:
            return False, "target-alert FPR upper bound is below its estimate"

    metadata = json.loads(paths["metadata"].read_text())
    dataset_spec = DATASETS[job.dataset_key]
    if int(metadata.get("n_attack_challenge", -1)) != int(
        dataset_spec["expected_row_challenge"]
    ):
        return False, "unexpected row-weighted challenge size"
    if int(metadata.get("n_attack_pattern_balanced_challenge", -1)) != int(
        dataset_spec["expected_pattern_challenge"]
    ):
        return False, "unexpected pattern-balanced challenge size"
    audit = metadata.get("emerging_split_audit", {})
    for key in (
        "n_index_overlap_train_final_challenge",
        "n_feature_overlap_rows_after_purge",
        "n_unique_duplicate_patterns_crossing_after_purge",
    ):
        if int(audit.get(key, -1)) != 0:
            return False, f"split audit failed: {key}={audit.get(key)!r}"

    target_audit = metadata.get("target_alert_split_audit", {})
    for key in (
        "n_index_overlap_training_evaluation",
        "n_index_overlap_calibration_test",
        "n_feature_overlap_training_calibration",
        "n_feature_overlap_training_test",
        "n_feature_overlap_calibration_test",
        "n_feature_overlap_positive_calibration",
        "n_feature_overlap_positive_test",
    ):
        if int(target_audit.get(key, -1)) != 0:
            return False, (
                f"target-alert split audit failed: {key}="
                f"{target_audit.get(key)!r}"
            )
    if int(metadata.get("n_target_alert_calibration_negatives", 0)) < 10:
        return False, "too few target-alert calibration negatives"
    if int(metadata.get("n_target_alert_test_negatives", 0)) < 10:
        return False, "too few target-alert test negatives"

    summary = json.loads(paths["summary"].read_text())
    for key in (
        "cumulative_selected_distinct_emerging_records",
        "cumulative_selected_distinct_emerging_feature_patterns",
        "early_warning_threshold_cumulative_distinct_records",
        "early_warning_threshold_cumulative_distinct_feature_patterns",
        "target_alert_empirical_recall_target",
        "target_alert_empirical_fpr_limit",
        "target_alert_empirical_target_round",
        "time_to_target_alert_empirical",
        "target_alert_certified_target_round",
        "time_to_target_alert_certified",
        "final_target_alert_recall_unexposed",
        "final_target_alert_target_vs_rest_fpr_unexposed",
        "final_target_alert_target_vs_rest_fpr_upper_unexposed",
    ):
        if key not in summary:
            return False, f"summary missing {key}"

    if job.method_tag == "feddrift":
        for key in (
            "adapter_path",
            "feddrift_n_global_models",
            "feddrift_cluster_assignments",
            "feddrift_models_before_merge",
            "feddrift_models_after_merge",
            "feddrift_models_after_creation",
            "feddrift_new_models_created",
            "feddrift_drifted_client_ids",
        ):
            if key not in rows[-1]:
                return False, f"FedDrift diagnostics missing {key}"

    return True, "validated"


def write_manifest(
    exp_root: Path,
    jobs: list[Job],
    gpus: list[int],
    seeds: tuple[int, ...],
) -> None:
    code_files = [
        REPO / "utils/cyber_stream.py",
        REPO / "utils/cyber_metrics.py",
        REPO / "methods/run_cyber.py",
        REPO / "utils/cyber_adapters.py",
        REPO / "entities/FedDrift.py",
        REPO / "entities/FedDrift_Cyber.py",
    ]
    payload = {
        "created_at_utc": utc_now(),
        "protocol_revision": PROTOCOL_REVISION,
        "repo": str(REPO),
        "experiment_root": str(exp_root),
        "gpus": gpus,
        "job_count": len(jobs),
        "code_sha256": {
            str(path.relative_to(REPO)): file_sha256(path) for path in code_files
        },
        "datasets": DATASETS,
        "methods": METHODS,
        "seeds": list(seeds),
    }
    (exp_root / "manifest.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n"
    )


def write_completion(exp_root: Path, jobs: list[Job]) -> None:
    path = exp_root / "completion.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dataset", "method", "seed", "status", "validation"])
        for job in jobs:
            status_path = job.run_dir / "status.txt"
            status = status_path.read_text().strip() if status_path.exists() else "PENDING"
            valid, detail = validate_output(job)
            writer.writerow(
                [
                    job.dataset_key,
                    job.method_tag,
                    job.seed,
                    status,
                    detail if valid else f"NOT_VALID: {detail}",
                ]
            )


def log_event(exp_root: Path, lock: threading.Lock, message: str) -> None:
    line = f"[{utc_now()}] {message}"
    with lock:
        print(line, flush=True)
        with (exp_root / "launcher.log").open("a") as stream:
            stream.write(line + "\n")


def run_job(job: Job, gpu: int, exp_root: Path, lock: threading.Lock) -> bool:
    valid, _ = validate_output(job)
    status_path = job.run_dir / "status.txt"
    if valid and status_path.exists() and status_path.read_text().startswith("COMPLETED"):
        log_event(exp_root, lock, f"GPU={gpu} SKIP validated {job.run_name}")
        return True

    status_path.write_text(f"RUNNING started_at={utc_now()} gpu={gpu}\n")
    log_event(exp_root, lock, f"GPU={gpu} START {job.run_name}")
    start = time.time()
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "PYTHONHASHSEED": str(job.seed),
            "PYTHONPATH": str(REPO),
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    command = [
        sys.executable,
        "-u",
        "methods/run_cyber.py",
        "--algorithm",
        job.algorithm,
        "--repo-root",
        str(REPO),
        "--experiment-config",
        str(job.config_path),
    ]
    with (job.run_dir / "run.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )

    elapsed = int(time.time() - start)
    valid, detail = validate_output(job)
    success = result.returncode == 0 and valid
    if success:
        status_path.write_text(
            f"COMPLETED exit_code=0 elapsed_seconds={elapsed} validated=true\n"
        )
        log_event(exp_root, lock, f"GPU={gpu} DONE {job.run_name} ({elapsed}s)")
    else:
        status_path.write_text(
            f"FAILED exit_code={result.returncode} elapsed_seconds={elapsed} "
            f"validation={detail}\n"
        )
        log_event(
            exp_root,
            lock,
            f"GPU={gpu} FAILED {job.run_name} rc={result.returncode} {detail}",
        )
    return success


def run_matrix(jobs: list[Job], gpus: list[int], exp_root: Path) -> bool:
    job_queue: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        job_queue.put(job)
    lock = threading.Lock()
    outcomes: list[bool] = []

    def worker(gpu: int) -> None:
        while True:
            try:
                job = job_queue.get_nowait()
            except queue.Empty:
                return
            try:
                outcome = run_job(job, gpu, exp_root, lock)
                with lock:
                    outcomes.append(outcome)
                    write_completion(exp_root, jobs)
            finally:
                job_queue.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return len(outcomes) == len(jobs) and all(outcomes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-root",
        default=None,
        help="New or resumable output root; defaults to a timestamped directory.",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=DEFAULT_NUM_SEEDS,
        help="Number of independent seeds, numbered from 0 (default: 10).",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only and args.validate_only:
        parser.error("--prepare-only and --validate-only are mutually exclusive")
    if args.num_seeds < 1:
        parser.error("--num-seeds must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    if args.exp_root:
        exp_root = resolve_path(Path(args.exp_root)).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_root = REPO / "experiments" / f"corrected_central_{stamp}"
    gpus = [int(item) for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("At least one GPU must be specified")
    seeds = tuple(range(args.num_seeds))

    exp_root.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(exp_root, seeds, write_configs=not args.validate_only)
    if not args.validate_only:
        write_manifest(exp_root, jobs, gpus, seeds)
    write_completion(exp_root, jobs)

    print(f"experiment_root={exp_root}")
    print(f"jobs={len(jobs)} seeds={list(seeds)} gpus={gpus}")
    if args.prepare_only:
        print("Prepared configs and manifest; no training started.")
        return 0
    if args.validate_only:
        failures = []
        for job in jobs:
            valid, detail = validate_output(job)
            if not valid:
                failures.append((job.run_name, detail))
        write_completion(exp_root, jobs)
        if failures:
            for run_name, detail in failures:
                print(f"INVALID {run_name}: {detail}")
            return 1
        print(f"All {len(jobs)} runs validated.")
        return 0

    success = run_matrix(jobs, gpus, exp_root)
    write_completion(exp_root, jobs)
    print(f"completion_csv={exp_root / 'completion.csv'}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
