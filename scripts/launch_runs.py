#!/usr/bin/env python3
"""Launch frozen paper runs from configs/runs/ through the shared runner.

This is a thin job pool around ``methods/run_cyber.py``. It does not implement
any algorithm. Each job is::

    python -u methods/run_cyber.py --algorithm <ALG> --repo-root . \
        --experiment-config configs/runs/<group>/<...>/seed_<k>.yaml

The process environment mirrors the original launchers that produced the
accepted-paper runs:

* central (scripts/run_corrected_central_4gpu.py): CUDA_VISIBLE_DEVICES,
  CUDA_DEVICE_ORDER=PCI_BUS_ID, OMP_NUM_THREADS=4, MKL_NUM_THREADS=4,
  PYTHONHASHSEED=<seed>, PYTHONPATH=<repo>.
* sweep (scripts/run_cse_spatial_exposure_v4_10seed.bash): CUDA_VISIBLE_DEVICES
  and PYTHONPATH=<repo> only.

Both original launchers are preserved under provenance/original_scripts/.
"""
from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]

# Method directory name -> --algorithm passed to the shared runner. FedCCFA and
# FedCCFA-GC share one algorithm; the run overlay sets fedccfa_sharing_mode.
ALGORITHM = {
    "fedavg": "FedAvg",
    "feddrift": "FedDrift",
    "fedccfa": "FedCCFA",
    "fedccfa_global_classifier": "FedCCFA",
}
GROUPS = ("cse_central", "ciciot_central", "cse_exposure_sweep")


def discover(group: str, methods: list[str] | None, seeds: list[int] | None):
    root = REPO / "configs" / "runs" / group
    if not root.is_dir():
        raise SystemExit(f"Missing {root}")
    jobs = []
    for cfg in sorted(root.rglob("seed_*.yaml")):
        seed = int(cfg.stem.removeprefix("seed_"))
        if seeds is not None and seed not in seeds:
            continue
        if group == "cse_exposure_sweep":
            method = "fedavg"  # sweep layout: <rho>/seed_k.yaml
        else:
            method = cfg.parent.name
        if methods is not None and method not in methods and cfg.parent.name not in methods:
            continue
        overlay = yaml.safe_load(cfg.read_text())
        jobs.append(
            {
                "config": cfg.relative_to(REPO),
                "algorithm": ALGORITHM[method],
                "seed": seed,
                "run_name": overlay["run_name"],
                "result_dir": REPO / overlay["result_dir"],
            }
        )
    return jobs


def worker(gpu: str, q: "queue.Queue", profile: str, python: str, failures: list, lock):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        rd: Path = job["result_dir"]
        rd.mkdir(parents=True, exist_ok=True)
        done = rd / "status.txt"
        if done.exists() and done.read_text().startswith("COMPLETED"):
            with lock:
                print(f"[GPU={gpu}] SKIP {job['run_name']} (already completed)", flush=True)
            q.task_done()
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONPATH"] = str(REPO)
        if profile == "central":
            env.update(
                {
                    "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                    "OMP_NUM_THREADS": "4",
                    "MKL_NUM_THREADS": "4",
                    "PYTHONHASHSEED": str(job["seed"]),
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
        cmd = [
            python, "-u", "methods/run_cyber.py",
            "--algorithm", job["algorithm"],
            "--repo-root", ".",
            "--experiment-config", str(job["config"]),
        ]
        with lock:
            print(f"[GPU={gpu}] START {job['run_name']}", flush=True)
        t0 = time.time()
        with (rd / "run.log").open("w") as log:
            rc = subprocess.run(cmd, cwd=REPO, env=env, stdout=log,
                                stderr=subprocess.STDOUT).returncode
        dt = int(time.time() - t0)
        ok = rc == 0 and (rd / f"{job['run_name']}_rounds.csv").exists()
        if ok:
            done.write_text(f"COMPLETED exit_code=0 elapsed_seconds={dt}\n")
        else:
            done.write_text(f"FAILED exit_code={rc} elapsed_seconds={dt}\n")
            failures.append(job["run_name"])
        with lock:
            print(f"[GPU={gpu}] {'DONE' if ok else 'FAILED'} {job['run_name']} ({dt}s)", flush=True)
        q.task_done()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("group", choices=GROUPS)
    p.add_argument("--gpus", default=os.environ.get("GPUS", "0"),
                   help="Comma-separated GPU ids; one concurrent job per id (default: $GPUS or 0)")
    p.add_argument("--methods", default=os.environ.get("METHODS"),
                   help="Comma-separated method dirs (central) or rho dirs (sweep), e.g. fedavg,fedccfa or rho10")
    p.add_argument("--seeds", default=os.environ.get("SEEDS"),
                   help="Comma-separated seeds (default: all 0-9)")
    p.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    a = p.parse_args()

    methods = a.methods.split(",") if a.methods else None
    seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else None
    jobs = discover(a.group, methods, seeds)
    if not jobs:
        raise SystemExit("No jobs matched")
    profile = "sweep" if a.group == "cse_exposure_sweep" else "central"
    print(f"group={a.group} jobs={len(jobs)} gpus={a.gpus} env_profile={profile}", flush=True)

    q: "queue.Queue" = queue.Queue()
    for j in jobs:
        q.put(j)
    failures: list = []
    lock = threading.Lock()
    threads = [threading.Thread(target=worker, args=(g.strip(), q, profile, a.python, failures, lock))
               for g in a.gpus.split(",") if g.strip()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if failures:
        print(f"FAILED runs: {failures}", file=sys.stderr)
        return 1
    print("All runs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
