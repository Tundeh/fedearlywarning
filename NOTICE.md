# Notice

This repository is derived from **FedCCFA** by Junbao Chen
(<https://github.com/Chen-Junbao/FedCCFA>, upstream commit
`647d8b097509e05278b7a68f116d2e84e106c72c`), released under the MIT License.
The upstream copyright notice and license text are reproduced unchanged in
[`LICENSE`](LICENSE) and apply to the upstream files and to this derivative work.

## What comes from upstream

The following files are upstream FedCCFA code, included unmodified in content
(only file permission bits differ from the upstream commit):

- `entities/base.py`, `entities/FedCCFA.py`, `entities/FedDrift.py`, `entities/__init__.py`
- `utils/gen_dataset.py`, `utils/metric.py`, `utils/__init__.py`
- `configs/FedAvg.yaml`, `configs/FedDrift.yaml`, `configs/FedCCFA.yaml`

`utils/models.py` is upstream code with the unused `FashionMNISTCNN` class
removed; `CifarCNN` and `get_model` are otherwise unchanged. Upstream
algorithms and runners not used in the paper (AdaptiveFedAvg, Ditto, FedBABU,
FedFM, FedPAC, FedProx, FedRep, Flash, IFCA, SCAFFOLD, pFedMe, the
image-classification `methods/*.py` runners and `utils/drift.py`) are not
included; they remain available in the upstream repository.

## What this release adds

The cybersecurity / emerging-threat extensions written for the paper
*Federated Early Warning for Emerging Cyber Threats: Measuring Knowledge
Dissemination in Distributed Intrusion Detection* (IEEE BigData 2026):

- `methods/run_cyber.py` — shared prequential runner for the cyber stream
- `methods/*_Cyber.py`, `entities/*_Cyber.py` — cyber launchers/wrappers for FedAvg, FedDrift and FedCCFA
  (including the FedCCFA global-classifier sharing mode, "FedCCFA-GC")
- `utils/cyber_*.py` — data loading, purged emerging-threat stream, schedules,
  CyberMLP model bridge, adapters, and calibrated target-alert metrics
- `utils/prepare_cicids2018.py`, `utils/prepare_ciciot2023.py` — dataset preprocessing
- `configs/Cyber_Common.yaml`, `configs/*_Cyber.yaml`, `configs/runs/`, `configs/resolved/`
- `analysis/`, `scripts/`, `tests/`, `reference_results/`, `provenance/`, `environment/`

The upstream algorithms are wrapped, not re-implemented; their upstream
hyperparameters are loaded first and overridden only where listed in
`configs/Cyber_Common.yaml`, `configs/*_Cyber.yaml` and `configs/runs/`.

## Datasets

CSE-CIC-IDS2018 and CICIoT2023 are third-party datasets from the Canadian
Institute for Cybersecurity and are **not** redistributed here. See
[`data/README.md`](data/README.md) for their sources and terms.
