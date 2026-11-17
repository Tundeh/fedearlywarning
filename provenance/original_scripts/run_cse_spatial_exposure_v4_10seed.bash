#!/usr/bin/env bash
# Rerun the CSE-CIC-IDS2018 FedAvg spatial-exposure sweep needed for Figure 1.
#
# Matrix: rho in {0.10, 0.30, 0.50} x seeds {0,...,9} = 30 runs.
# The launcher clones the completed target-alert-v4 CSE/FedAvg configs, then
# changes only the seed, exposure milestone, run name, and result directory.
# Thus it preserves the v4 target-alert calibration/test construction.

set -Eeuo pipefail

REPO_ROOT="${1:-/LLM/FedLearning/FedCCFA}"
TEMPLATE_ROOT="${2:-${REPO_ROOT}/experiments/target_alert_v4_10seed}"
OUTPUT_ROOT="${3:-${REPO_ROOT}/experiments/cse_spatial_exposure_target_alert_v4_10seed}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_CSV="${GPUS:-0,1,2,3}"
SEED_CSV="${SEEDS:-0,1,2,3,4,5,6,7,8,9}"
RHO_CSV="${RHOS:-0.10,0.30,0.50}"
LIVE_LOGS="${LIVE_LOGS:-0}"
FORCE="${FORCE:-0}"
DRY_RUN="${DRY_RUN:-0}"
HEARTBEAT_SECONDS="${HEARTBEAT_SECONDS:-30}"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "ERROR: repository not found: ${REPO_ROOT}" >&2
  exit 2
fi
if [[ ! -d "${TEMPLATE_ROOT}" ]]; then
  echo "ERROR: completed v4 template root not found: ${TEMPLATE_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${REPO_ROOT}/methods/run_cyber.py" ]]; then
  echo "ERROR: methods/run_cyber.py not found under ${REPO_ROOT}" >&2
  exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
  exit 2
fi

IFS=',' read -r -a GPU_IDS <<< "${GPU_CSV}"
if (( ${#GPU_IDS[@]} == 0 )); then
  echo "ERROR: no GPUs configured in GPUS=${GPU_CSV}" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
MANIFEST="${OUTPUT_ROOT}/jobs.tsv"

echo "repository=${REPO_ROOT}"
echo "v4_template_root=${TEMPLATE_ROOT}"
echo "experiment_root=${OUTPUT_ROOT}"
echo "gpus=[${GPU_CSV}] seeds=[${SEED_CSV}] rhos=[${RHO_CSV}]"
echo "Preparing and validating v4-derived experiment configs..."

"${PYTHON_BIN}" - "${REPO_ROOT}" "${TEMPLATE_ROOT}" "${OUTPUT_ROOT}" \
  "${MANIFEST}" "${SEED_CSV}" "${RHO_CSV}" <<'PY'
from __future__ import annotations

import copy
from pathlib import Path
import shlex
import sys

import yaml

repo = Path(sys.argv[1]).resolve()
template_root = Path(sys.argv[2]).resolve()
output_root = Path(sys.argv[3]).resolve()
manifest = Path(sys.argv[4]).resolve()
seeds = tuple(int(x) for x in sys.argv[5].split(",") if x.strip())
rhos = tuple(float(x) for x in sys.argv[6].split(",") if x.strip())

if seeds != tuple(range(10)):
    raise SystemExit(f"SEEDS must be exactly 0..9 for the paper sweep; got {seeds}")
if rhos != (0.10, 0.30, 0.50):
    raise SystemExit(f"RHOS must be exactly 0.10,0.30,0.50; got {rhos}")


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten(item, path)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix.lower(), value


def approximately(value, expected):
    try:
        return abs(float(value) - float(expected)) <= 1e-12
    except (TypeError, ValueError):
        return False


def validate_v4_template(config, source):
    leaves = list(flatten(config))
    has_calibration = any(
        "calib" in path and approximately(value, 0.005)
        for path, value in leaves
    )
    has_test_limit = any(
        "fpr" in path
        and any(token in path for token in ("limit", "test", "threshold"))
        and approximately(value, 0.01)
        for path, value in leaves
    )
    has_three_rounds = any(
        any(token in path for token in ("sustain", "consecutive"))
        and approximately(value, 3)
        for path, value in leaves
    )
    missing = []
    if not has_calibration:
        missing.append("0.5% calibration budget")
    if not has_test_limit:
        missing.append("1% held-out FPR limit")
    if not has_three_rounds:
        missing.append("three-round sustained criterion")
    if missing:
        relevant = {
            path: value
            for path, value in leaves
            if any(token in path for token in ("target", "alert", "calib", "fpr", "sustain", "consecutive"))
        }
        raise SystemExit(
            f"Template {source} is not recognizably a complete v4 config; "
            f"missing {', '.join(missing)}. Relevant settings: {relevant}"
        )


def locate_template(seed):
    direct = template_root / "cse" / "fedavg" / f"seed_{seed}" / "config.yaml"
    if direct.is_file():
        return direct
    candidates = []
    seed_token = f"seed_{seed}"
    for path in template_root.rglob("config.yaml"):
        parts = [part.lower() for part in path.parts]
        joined = "/".join(parts)
        if "cse" in joined and "fedavg" in joined and seed_token in parts:
            candidates.append(path)
    if len(candidates) != 1:
        raise SystemExit(
            f"Expected one CSE/FedAvg seed-{seed} v4 config, found "
            f"{len(candidates)}: {candidates}"
        )
    return candidates[0]


jobs = []
job_index = 0
for rho in rhos:
    rho_percent = int(round(100 * rho))
    rho_tag = f"rho{rho_percent:02d}"
    for seed in seeds:
        source = locate_template(seed)
        template = yaml.safe_load(source.read_text()) or {}
        validate_v4_template(template, source)
        prepared_data = str(template.get("prepared_data", "")).lower()
        if "cicids2018" not in prepared_data:
            raise SystemExit(
                f"Template {source} does not reference the CSE-CIC-IDS2018 "
                f"prepared data: prepared_data={prepared_data!r}"
            )
        config = copy.deepcopy(template)

        for key in ("seed", "stream_seed", "schedule_seed", "participation_seed"):
            config[key] = seed
        config["initial_model_seed"] = 424242
        config["rounds"] = 30
        config["flush_every"] = 1

        threat = config.setdefault("threat", {})
        threat["attack_family"] = "ddos"
        threat["emerging_variant"] = "ddos attack-hoic"
        threat["evidence_mode"] = "fixed_unique"
        threat["fixed_total_attack_per_round"] = 3000
        threat["milestones"] = [[0, 0.0], [10, float(rho)]]

        # Full client participation is essential: rho changes spatial exposure,
        # not client-selection probability or total evidence per round.
        participation = config.setdefault("participation", {})
        participation["normal_prob"] = 1.0
        participation["exposed_prob"] = 1.0
        participation["min_selected"] = int(participation.get("min_selected", 3))

        stream = config.setdefault("stream", {})
        stream["attack_guard_fraction"] = 0.05
        stream["attack_test_fraction"] = 0.30

        run_name = (
            f"fedavg_cse_hoic_{rho_tag}_N3000_p100_seed{seed}_"
            "target_alert_v4_spatial"
        )
        run_dir = output_root / "cse" / "fedavg" / rho_tag / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path = run_dir / "config.yaml"
        config["run_name"] = run_name
        config["result_dir"] = str(run_dir)
        config["protocol_revision"] = (
            "target_alert_v4_purged_chronological_feature_disjoint_"
            "spatial_exposure_10seed"
        )
        config["spatial_exposure_fraction"] = float(rho)
        config["config_template_source"] = str(source)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))

        command = (
            f"CUDA_VISIBLE_DEVICES=<GPU> python -u methods/run_cyber.py "
            f"--algorithm FedAvg --repo-root {shlex.quote(str(repo))} "
            f"--experiment-config {shlex.quote(str(config_path))}\n"
        )
        (run_dir / "command.sh").write_text(command)
        jobs.append(
            (
                job_index,
                rho_tag,
                f"{rho:.2f}",
                seed,
                str(config_path),
                str(run_dir),
                run_name,
            )
        )
        job_index += 1

manifest.write_text(
    "job_index\trho_tag\trho\tseed\tconfig_path\trun_dir\trun_name\n"
    + "".join("\t".join(map(str, row)) + "\n" for row in jobs)
)
print(f"Prepared {len(jobs)} jobs in {manifest}")
PY

TOTAL_JOBS=$(( $(wc -l < "${MANIFEST}") - 1 ))
if (( TOTAL_JOBS != 30 )); then
  echo "ERROR: expected 30 jobs, found ${TOTAL_JOBS}" >&2
  exit 2
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN=1: prepared and validated 30 configs; no GPU jobs launched."
  echo "manifest=${MANIFEST}"
  exit 0
fi

run_worker() {
  local worker_index="$1"
  local gpu_id="$2"
  local line_index=0

  while IFS=$'\t' read -r job_index rho_tag rho seed config_path run_dir run_name; do
    if [[ "${job_index}" == "job_index" ]]; then
      continue
    fi
    if (( line_index % ${#GPU_IDS[@]} != worker_index )); then
      line_index=$((line_index + 1))
      continue
    fi
    line_index=$((line_index + 1))

    local done_marker="${run_dir}/.process_done"
    local fail_marker="${run_dir}/.process_failed"
    local run_log="${run_dir}/run.log"
    rm -f "${fail_marker}"

    if [[ "${FORCE}" != "1" && -f "${done_marker}" ]]; then
      echo "[$(date --iso-8601=seconds)] GPU=${gpu_id} SKIP ${run_name} (process already completed)"
      continue
    fi

    rm -f "${done_marker}"
    echo "[$(date --iso-8601=seconds)] GPU=${gpu_id} START ${run_name} rho=${rho} seed=${seed}"
    local started=$SECONDS
    local rc=0

    if [[ "${LIVE_LOGS}" == "1" ]]; then
      set +e
      (
        cd "${REPO_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u methods/run_cyber.py \
          --algorithm FedAvg \
          --repo-root "${REPO_ROOT}" \
          --experiment-config "${config_path}"
      ) 2>&1 | awk -v prefix="[GPU=${gpu_id} ${rho_tag} seed=${seed}]" \
        '{ print prefix, $0; fflush(); }' | tee "${run_log}"
      rc=${PIPESTATUS[0]}
      set -e
    else
      set +e
      (
        cd "${REPO_ROOT}"
        CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" -u methods/run_cyber.py \
          --algorithm FedAvg \
          --repo-root "${REPO_ROOT}" \
          --experiment-config "${config_path}"
      ) >"${run_log}" 2>&1
      rc=$?
      set -e
    fi

    local elapsed=$((SECONDS - started))
    if (( rc == 0 )); then
      : > "${done_marker}"
      echo "[$(date --iso-8601=seconds)] GPU=${gpu_id} DONE ${run_name} (${elapsed}s)"
    else
      printf 'exit_status=%s\n' "${rc}" > "${fail_marker}"
      echo "[$(date --iso-8601=seconds)] GPU=${gpu_id} FAIL ${run_name} rc=${rc}; log=${run_log}" >&2
    fi
  done < "${MANIFEST}"
}

echo "Launching ${TOTAL_JOBS} jobs across ${#GPU_IDS[@]} GPUs..."
WORKER_PIDS=()
for worker_index in "${!GPU_IDS[@]}"; do
  run_worker "${worker_index}" "${GPU_IDS[$worker_index]}" &
  WORKER_PIDS+=("$!")
done

while :; do
  alive=0
  for pid in "${WORKER_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  done_count=$(find "${OUTPUT_ROOT}/cse/fedavg" -type f -name .process_done 2>/dev/null | wc -l)
  failed_count=$(find "${OUTPUT_ROOT}/cse/fedavg" -type f -name .process_failed 2>/dev/null | wc -l)
  echo "[$(date --iso-8601=seconds)] progress=${done_count}/${TOTAL_JOBS} failed=${failed_count} active_workers=${alive}"
  if (( alive == 0 )); then
    break
  fi
  sleep "${HEARTBEAT_SECONDS}"
done

worker_failure=0
for pid in "${WORKER_PIDS[@]}"; do
  if ! wait "${pid}"; then
    worker_failure=1
  fi
done

echo "Validating round files, target-alert columns, and split audits..."
"${PYTHON_BIN}" - "${MANIFEST}" "${OUTPUT_ROOT}/completion.csv" <<'PY'
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import yaml

manifest = Path(sys.argv[1])
completion = Path(sys.argv[2])


def all_key_values(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path.lower(), item
            yield from all_key_values(item, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from all_key_values(item, f"{prefix}[{index}]")


def validate(row):
    run_dir = Path(row["run_dir"])
    errors = []
    if not (run_dir / ".process_done").is_file():
        errors.append("process did not complete successfully")

    round_files = sorted(run_dir.glob("*_rounds.csv"))
    rounds = []
    selected_round_file = None
    # Some v4 installations append target-alert columns to the main round
    # file; others write a separate target-alert round file. Accept either.
    for round_file in round_files:
        with round_file.open(newline="") as stream:
            candidate = list(csv.DictReader(stream))
        columns = {name.lower() for name in candidate[0]} if candidate else set()
        has_recall = any(
            "recall" in name and any(token in name for token in ("target", "alert"))
            for name in columns
        )
        has_fpr = any("fpr" in name for name in columns)
        if len(candidate) == 30 and has_recall and has_fpr:
            selected_round_file = round_file
            rounds = candidate
            break
    if selected_round_file is None:
        errors.append(
            f"no 30-row target-alert round file found among {len(round_files)} candidates"
        )
    else:
        if len(rounds) != 30:
            errors.append(f"round rows={len(rounds)}, expected 30")
        elif [int(float(item["round"])) for item in rounds] != list(range(30)):
            errors.append("round sequence is incomplete or unordered")
        if rounds:
            columns = {name.lower() for name in rounds[0]}
            if not any("target" in name and "recall" in name for name in columns):
                errors.append("no target-alert recall column")
            if not any("fpr" in name for name in columns):
                errors.append("no target-alert FPR column")

    metadata_candidates = sorted(run_dir.glob("*_metadata.json"))
    metadata_with_audit = None
    for path in metadata_candidates:
        try:
            candidate = json.loads(path.read_text())
        except Exception:
            continue
        if "emerging_split_audit" in candidate:
            metadata_with_audit = candidate
            break
    if metadata_with_audit is None:
        errors.append("metadata with emerging_split_audit not found")
    else:
        audit = metadata_with_audit.get("emerging_split_audit", {})
        required_zero = (
            "n_index_overlap_train_final_challenge",
            "n_feature_overlap_rows_after_purge",
            "n_unique_duplicate_patterns_crossing_after_purge",
        )
        for key in required_zero:
            if int(audit.get(key, -1)) != 0:
                errors.append(f"split audit failed: {key}={audit.get(key)!r}")

        # Also check any target-alert calibration/test overlap counters exposed
        # by v4 metadata, without depending on their exact nesting path.
        for path, value in all_key_values(metadata_with_audit):
            if "overlap" in path and isinstance(value, (int, float)) and value != 0:
                errors.append(f"nonzero overlap counter: {path}={value}")

    config_path = Path(row["config_path"])
    config = yaml.safe_load(config_path.read_text()) or {}
    milestones = config.get("threat", {}).get("milestones")
    expected_rho = float(row["rho"])
    if milestones != [[0, 0.0], [10, expected_rho]]:
        errors.append(f"unexpected threat milestones: {milestones!r}")
    participation = config.get("participation", {})
    if float(participation.get("normal_prob", -1)) != 1.0:
        errors.append("normal participation is not 1.0")
    if float(participation.get("exposed_prob", -1)) != 1.0:
        errors.append("exposed participation is not 1.0")

    return errors


with manifest.open(newline="") as stream:
    jobs = list(csv.DictReader(stream, delimiter="\t"))

records = []
failures = 0
for job in jobs:
    errors = validate(job)
    status = "VALID" if not errors else "INVALID"
    failures += int(bool(errors))
    if errors:
        # Invalid output must be rerun rather than silently skipped on resume.
        (Path(job["run_dir"]) / ".process_done").unlink(missing_ok=True)
    records.append(
        {
            "job_index": job["job_index"],
            "rho": job["rho"],
            "seed": job["seed"],
            "run_name": job["run_name"],
            "run_dir": job["run_dir"],
            "status": status,
            "message": "; ".join(errors),
        }
    )

with completion.open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

if failures:
    print(f"Validation failed for {failures}/{len(records)} runs.")
    for record in records:
        if record["status"] != "VALID":
            print(f"  {record['run_name']}: {record['message']}")
    raise SystemExit(1)

print(f"All {len(records)} spatial-exposure runs validated.")
print(f"completion_csv={completion}")
PY

if (( worker_failure != 0 )); then
  echo "ERROR: at least one GPU worker exited unsuccessfully." >&2
  exit 1
fi

echo "Spatial-exposure sweep complete: ${OUTPUT_ROOT}"
