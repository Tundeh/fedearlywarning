# Federated Early Warning for Emerging Cyber Threats

Code and data to reproduce the IEEE BigData 2026 paper **"Federated Early
Warning for Emerging Cyber Threats: Measuring Knowledge Dissemination in
Distributed Intrusion Detection"**.

It compares FedAvg, FedDrift, FedCCFA and FedCCFA with a global classifier
(FedCCFA-GC) on CSE-CIC-IDS2018 and CICIoT2023, and runs a spatial-exposure
sweep on CSE-CIC-IDS2018. Built on [FedCCFA](https://github.com/Chen-Junbao/FedCCFA)
(MIT); see [NOTICE.md](NOTICE.md).

## 1. Install

Python 3.12 and a CUDA GPU (tested with PyTorch 2.10.0 + CUDA 12.8 on NVIDIA A40).

```bash
git clone https://github.com/Tundeh/fedearlywarning.git
cd fedearlywarning
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

## 2. Check the paper results (no GPU or dataset needed)

The repository includes the outputs of all 110 paper runs. These commands
recompute every table and figure from them (about 15 seconds):

```bash
bash scripts/analyze_all.sh analysis_output/check
python scripts/verify_reference_results.py --analysis-dir analysis_output/check
```

The second command ends with `VERIFICATION PASSED` if Tables I–III match the paper.

## 3. Reproduce from scratch

### Download and prepare the data

The datasets are not included. Download them from the Canadian Institute for
Cybersecurity and place them in `data/` as described in
[data/README.md](data/README.md):

- CSE-CIC-IDS2018: <https://www.unb.ca/cic/datasets/ids-2018.html> → `data/CSE-CIC-IDS2018/`
- CICIoT2023: <https://www.unb.ca/cic/datasets/iotdataset-2023.html> → `data/CICIoT2023/`

```bash
bash scripts/prepare_cse.sh      # ~10 min, ~65 GB RAM
bash scripts/prepare_ciciot.sh
```

Each script prints the expected and the obtained SHA-256 checksum; they should match.

### Train

```bash
GPUS=0,1,2,3 bash scripts/run_cse_central.sh         # 40 runs
GPUS=0,1,2,3 bash scripts/run_ciciot_central.sh      # 40 runs
GPUS=0,1,2,3 bash scripts/run_cse_exposure_sweep.sh  # 30 runs
```

`GPUS` lists the GPUs to use, one run per GPU at a time. Each run takes about
10 minutes on an A40, and all 110 runs take about 16 GPU-hours. Results go to
`outputs/`. Completed runs are skipped if you restart.

To run fewer, set `SEEDS` and/or `METHODS`, for example:

```bash
SEEDS=0 METHODS=fedavg,fedccfa_global_classifier bash scripts/run_cse_central.sh
```

### Analyze

```bash
RESULTS_ROOT=outputs bash scripts/analyze_all.sh analysis_output/mine
python scripts/verify_reference_results.py --analysis-dir analysis_output/mine --results-root outputs
python scripts/compare_runs.py --fresh outputs   # compare each run with the reference outputs
```

## Where the paper's tables and figures are

`analyze_all.sh` writes these into the output folder you give it:

| Paper | File |
|---|---|
| Table I | `tables/table_I_central_mean_ucb.*` |
| Table II | `tables/table_II_final_general_utility.*` |
| Table III | `tables/table_III_paired_calibrated_dissemination.*` |
| Figure 1 | `figures/figure_1_calibrated_spatial_exposure_cse.*` |
| Figure 2 | `figures/figure_2_calibrated_target_alert_trajectories.*` |
| Figure 3 | `figures/figure_3_acquisition_vs_dissemination_original_ews.*` |
| Figure S1, Tables S1–S5 | `figures/figure_S1_*`, `tables/table_S*` |

The tables and figures in the paper are also in
`reference_results/paper_analysis/`.

## Expected results (Table I)

Success = runs (out of 10) that confirm the threat; delay = rounds after
emergence, with 20 for runs that never confirm.

| Dataset | Method | Success | Delay (rounds) |
|---|---|---|---|
| CSE-CIC-IDS2018 | FedAvg | 10/10 | 6.00 ± 0.00 |
| CSE-CIC-IDS2018 | FedDrift | 0/10 | 20.00 ± 0.00 |
| CSE-CIC-IDS2018 | FedCCFA | 8/10 | 13.70 ± 3.77 |
| CSE-CIC-IDS2018 | FedCCFA-GC | 10/10 | 3.10 ± 0.32 |
| CICIoT2023 | FedAvg | 10/10 | 5.30 ± 3.77 |
| CICIoT2023 | FedDrift | 0/10 | 20.00 ± 0.00 |
| CICIoT2023 | FedCCFA | 10/10 | 3.50 ± 0.53 |
| CICIoT2023 | FedCCFA-GC | 10/10 | 3.10 ± 0.32 |

## Setup in brief

20 clients and 30 rounds. A new attack variant appears at round 10 at 6 of
the 20 clients (DDoS Attack-HOIC on CSE-CIC-IDS2018, mirai-udpplain on
CICIoT2023), which receive 3,000 new records per round. The other 14 clients
never see it. All settings are in
[configs/PAPER_PROTOCOL.yaml](configs/PAPER_PROTOCOL.yaml).

A threat is confirmed when, for 3 consecutive rounds, recall on the new
variant, **averaged over the unexposed clients**, is at least 0.90 and the
false-positive rate is at most 0.01. The paper reports three versions of the
FPR test:

- **Empirical:** the mean observed FPR across those clients.
- **Mean-UCB (main result):** the mean of the clients' 95% Clopper–Pearson upper bounds.
- **All-site:** the largest client upper bound, so the FPR limit holds at every client.

## Limitations

- The CICIoT2023 event order and client split are a fixed experimental
  construction, not a real-world timeline.
- Test records share no exact features with training records, but may come
  from the same capture sessions.
- Bit-identical reruns need the same library, CUDA and GPU versions.
  [RELEASE_AUDIT.md](RELEASE_AUDIT.md) shows how this release was validated.

## License

MIT; see [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md). The datasets are
subject to the Canadian Institute for Cybersecurity's terms.
