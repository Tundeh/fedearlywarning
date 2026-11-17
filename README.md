# Federated Early Warning for Emerging Cyber Threats

Code, frozen configurations and reference outputs for the IEEE BigData 2026 paper

> **Federated Early Warning for Emerging Cyber Threats: Measuring Knowledge
> Dissemination in Distributed Intrusion Detection**

This repository reproduces every experiment in the paper:

| Experiment | Methods | Exposure ρ | Seeds | Paper output |
|---|---|---|---|---|
| CSE-CIC-IDS2018 central | FedAvg, FedDrift, FedCCFA, FedCCFA-GC | 0.30 (6/20 clients) | 0–9 | Tables I–III, Figs. 2–3, S1 |
| CICIoT2023 central | FedAvg, FedDrift, FedCCFA, FedCCFA-GC | 0.30 (6/20 clients) | 0–9 | Tables I–III, Figs. 2–3, S1 |
| CSE-CIC-IDS2018 spatial-exposure sweep | FedAvg | 0.10, 0.30, 0.50 | 0–9 | Fig. 1, Table S2 |

It is built on [FedCCFA](https://github.com/Chen-Junbao/FedCCFA) (MIT). The
upstream algorithms are wrapped, not re-implemented; see [NOTICE.md](NOTICE.md).

You can check the paper's tables **without a GPU or the datasets**: the
packaged round-level outputs of all 110 runs are enough to recompute every
table and figure (see [Quick check](#quick-check-no-gpu-no-data)).

---

## Contents

```
methods/run_cyber.py        shared prequential runner (all methods, both datasets)
entities/                   upstream FedAvg/FedDrift/FedCCFA clients+servers and *_Cyber wrappers
methods/*_Cyber.py          convenience launchers that call run_cyber.py
utils/cyber_*.py            stream, schedules, CyberMLP, adapters, target-alert metrics
utils/prepare_*.py          dataset preprocessing
configs/
  <Alg>.yaml                upstream hyperparameters (loaded first)
  Cyber_Common.yaml         shared cyber settings (overlay 1)
  <Alg>_Cyber.yaml          per-algorithm cyber settings (overlay 2)
  runs/                     frozen per-run overlays for all 110 paper runs (overlay 3)
  resolved/                 fully resolved config of every paper run (from run metadata)
  PAPER_PROTOCOL.yaml       human-readable protocol summary
scripts/                    prepare / run / analyze / figure / verify entry points
analysis/paper_analysis.py  tables, paired tests and figures from saved round CSVs
reference_results/          outputs of the 110 paper runs + the accepted analysis bundle
data/README.md              dataset download and preprocessing instructions
data_schema/                class / variant / feature schema of the processed data
tests/                      unit tests (stream, leakage purge, metrics, FedDrift adapter)
provenance/, environment/   git state, code/data checksums, original launchers, pip freeze
RELEASE_AUDIT.md            how this release was assembled and validated
```

## Environment

Validated with Python 3.12.7, PyTorch 2.10.0 + CUDA 12.8 on NVIDIA A40 GPUs
(Docker image `my-conda-docker`; full list in `environment/`).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

All scripts resolve paths relative to the repository root and set
`PYTHONPATH` themselves. Set `PYTHON_BIN` to use a specific interpreter.

## Quick check (no GPU, no data)

```bash
python scripts/verify_reference_results.py    # stdlib only; checks the packaged analysis
bash scripts/analyze_all.sh                   # recompute all tables/figures from round CSVs (~15 s)
python scripts/verify_reference_results.py --analysis-dir analysis_output/paper_<timestamp>
```

`verify_reference_results.py` exits non-zero if anything below is not reproduced:
10 matched seeds for every run group; protocol invariants in every run's
metadata; Table I success counts and delays; Table II utility fingerprints;
Table III paired differences, bootstrap CIs, exact and Holm-adjusted p-values.

## Full reproduction

### 1. Data

Download CSE-CIC-IDS2018 and CICIoT2023 (not redistributed here) as described
in [data/README.md](data/README.md), then:

```bash
bash scripts/prepare_cse.sh      # -> data/processed/cicids2018.{npz,json}  (~65 GB peak RAM, ~10 min)
bash scripts/prepare_ciciot.sh   # -> data/processed/ciciot2023.{npz,json}
```

Each script prints the expected and obtained SHA-256. The paper used:

```
f6b69380acf29d55890059839b1af0083fa825e78f3d898509698e79c7205a9b  cicids2018.npz
68ee57091a85c7fa4098856774fd0a0ed934831f70fbcba5144acab1a425a2d8  ciciot2023.npz
```

### 2. Training runs

```bash
GPUS=0,1,2,3 bash scripts/run_cse_central.sh        # 40 runs -> outputs/cse_central/
GPUS=0,1,2,3 bash scripts/run_ciciot_central.sh     # 40 runs -> outputs/ciciot_central/
GPUS=0,1,2,3 bash scripts/run_cse_exposure_sweep.sh # 30 runs -> outputs/cse_exposure_sweep/
```

`GPUS` runs one job per listed GPU at a time. Subsets: `SEEDS=0,1`,
`METHODS=fedavg,fedccfa_global_classifier` (central) or `METHODS=rho10`
(sweep). Completed runs are skipped on restart. Each job is exactly

```bash
python -u methods/run_cyber.py --algorithm <FedAvg|FedDrift|FedCCFA> --repo-root . \
    --experiment-config configs/runs/<group>/<method>/seed_<k>.yaml
```

FedCCFA and FedCCFA-GC both use `--algorithm FedCCFA`; the run overlay sets
`fedccfa_sharing_mode: clustered` or `global_classifier`.

If you run inside a container as a UID that has no `/etc/passwd` entry
(`docker exec -u <uid>`), PyTorch fails at start-up with
`KeyError: 'getpwuid(): uid not found'`. Set `USER=<name>` (and optionally
`TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor`) in the environment.

**Runtime** (one A40 per job): 8–10 min per run; the 80 central runs took
12.3 GPU-hours, the sweep about 4 GPU-hours (30 FedAvg runs). Each job loads the full processed
dataset (CSE ≈ 1.3 GB) into host memory.

### 3. Analysis and figures

```bash
RESULTS_ROOT=outputs bash scripts/analyze_all.sh analysis_output/mine
python scripts/verify_reference_results.py --analysis-dir analysis_output/mine --results-root outputs
python scripts/compare_runs.py --fresh outputs           # per-run comparison with reference_results/
RESULTS_ROOT=outputs bash scripts/make_figures.sh figures
```

`analyze_all.sh` writes `tables/` (CSV, Markdown, LaTeX), `figures/` (PDF,
PNG), `analysis/` (per-run values, paired tests) and `RESULTS_REPORT.md`.
Without `RESULTS_ROOT` it uses the packaged `reference_results/`.

| Paper item | File |
|---|---|
| Table I | `tables/table_I_central_mean_ucb.*` |
| Table II | `tables/table_II_final_general_utility.*` |
| Table III | `tables/table_III_paired_calibrated_dissemination.*` |
| Figure 1 | `figures/figure_1_calibrated_spatial_exposure_cse.*` |
| Figure 2 | `figures/figure_2_calibrated_target_alert_trajectories.*` |
| Figure 3 | `figures/figure_3_acquisition_vs_dissemination_original_ews.*` |
| Figure S1 | `figures/figure_S1_target_fpr_trajectories.*` |
| Tables S1–S5 | `tables/table_S*` |

## Protocol summary

Full values: [configs/PAPER_PROTOCOL.yaml](configs/PAPER_PROTOCOL.yaml).

- 20 clients, 30 rounds, full participation of eligible clients. The emerging
  variant appears at round 10 at 6 of 20 clients (ρ = 0.30) in the central
  experiments.
- Evidence mode `fixed_unique`: 3,000 new, previously unseen emerging-variant
  records per active round, split across exposed clients. Unexposed clients
  never receive emerging-variant training examples; the stream raises an
  `AssertionError` if one would (`utils/cyber_stream.py`, `apply_round`).
- Emerging-variant pool split chronologically: 65% training, 5% guard, 30%
  candidate challenge. Candidate challenge rows whose processed features
  exactly match any training row are removed before sampling; the challenge
  set is capped at 5,000 rows.
- Shared CyberMLP: d → 256 → 128 → C, ReLU, dropout 0.2, linear classifier.
  SGD (momentum 0.9, weight decay 1e-5), batch 128, background windows of up to
  2,048 flows per client. Initial model seed 424242 for every method.
- Datasets: CSE-CIC-IDS2018 (d = 70, C = 8, emerging variant DDoS Attack-HOIC,
  target family DDoS); CICIoT2023 (d = 39, C = 7, emerging variant
  mirai-udpplain, target family botnet).

### Target-alert criteria

Each unexposed client's alert score is its model's softmax probability for the
target family (one-vs-rest). The client's threshold is set on held-out
target-negative calibration rows to a 0.5% false-positive budget (β = 0.005)
and then frozen for testing. Each round, **recall** on the emerging-variant challenge set
is **averaged across unexposed clients** and compared with γ = 0.90. Test FPR is
compared with α = 0.01 under one of three criteria:

| Criterion | FPR statistic that must be ≤ 0.01 |
|---|---|
| Empirical | mean of the unexposed clients' observed test FPRs |
| Mean-UCB (primary) | mean of the clients' one-sided 95% Clopper–Pearson upper bounds (δ = 0.05) |
| All-site | **maximum** client upper bound, so FPR is controlled at every client |

A threat is **confirmed** at the third consecutive qualifying round (L = 3).
The **restricted delay** is the confirmation round minus 10. Runs that never
confirm within the H = 20-round post-emergence horizon count as 20, and
attainment (success out of 10) is reported separately.

**Inference.** Paired comparisons over the 10 matched seeds use exact two-sided
sign-flip tests (exact McNemar for binary attainment) with Holm correction
across the six method comparisons within each dataset–metric family. Confidence
intervals are percentile bootstraps with 100,000 paired resamples (RNG seed
20260918).

## Expected results (Table I, mean-UCB)

| Dataset | Method | Success | Restricted delay |
|---|---|---|---|
| CSE-CIC-IDS2018 | FedAvg | 10/10 | 6.00 ± 0.00 |
| CSE-CIC-IDS2018 | FedDrift | 0/10 | 20.00 ± 0.00 |
| CSE-CIC-IDS2018 | FedCCFA | 8/10 | 13.70 ± 3.77 |
| CSE-CIC-IDS2018 | FedCCFA-GC | 10/10 | 3.10 ± 0.32 |
| CICIoT2023 | FedAvg | 10/10 | 5.30 ± 3.77 |
| CICIoT2023 | FedDrift | 0/10 | 20.00 ± 0.00 |
| CICIoT2023 | FedCCFA | 10/10 | 3.50 ± 0.53 |
| CICIoT2023 | FedCCFA-GC | 10/10 | 3.10 ± 0.32 |

Tables II–III and the supplementary tables are in
`reference_results/paper_analysis/`.

## Known limitations

- **CICIoT2023 ordering.** The CICIoT2023 stream order and client assignment
  are a deterministic experimental construction from per-class files
  (first 50,000 rows of each file, chronological file blocks). They are not a
  verified common real-world event timeline.
- **Capture independence.** The challenge set has no exact processed-feature
  overlap with training data, but it can still come from the same capture
  sessions. Independence at the capture level is not verified
  (`capture_independence_verified: false` in run metadata).
- **Bounds.** Clopper–Pearson bounds are per client and per round. They are
  not simultaneous guarantees, and repeated decisions on fixed examples are not
  independent replicates.
- **Bit-exact reruns** depend on matching the library/CUDA stack and GPU type.
  See `RELEASE_AUDIT.md` for the fresh-rerun comparison done for this release.
- Only the algorithms used in the paper are included. Upstream FedCCFA also
  ships AdaptiveFedAvg, Ditto, FedBABU, FedFM, FedPAC, FedProx, FedRep, Flash,
  IFCA, SCAFFOLD and pFedMe, plus image-classification runners and a
  Fashion-MNIST CNN; these were removed (see `RELEASE_AUDIT.md`). The upstream
  `CifarCNN` in `utils/models.py` is kept: the upstream entities construct it as
  a placeholder before the cyber wrappers replace it with CyberMLP, and that
  construction is part of the seeded random stream.

## License

MIT; see [LICENSE](LICENSE) (upstream FedCCFA copyright) and
[NOTICE.md](NOTICE.md) for what this release adds. The datasets are subject to
their own CIC terms.
