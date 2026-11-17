# Release Audit

Assembled 2026-11-17 following *AI Agent Task Specification — Build the Clean
Reproducibility Code Release for the Accepted IEEE BigData 2026 Paper*.

## Source provenance
- Source repo: `~/LLM/FL_Dup/FedLearning/FedCCFA`. This is a copy of
  `~/LLM/FedLearning/FedCCFA`, which is mounted as `/LLM/FedLearning/FedCCFA`
  in the Docker container where the runs were executed. All packaged code files
  were confirmed identical between the two copies.
- Git commit: `647d8b097509e05278b7a68f116d2e84e106c72c` ("Update ReadME.md", upstream FedCCFA)
- Git branch: `master` (remote `https://github.com/Chen-Junbao/FedCCFA.git`)
- Working tree dirty: yes
  - Tracked files: permission-bit changes only. `git -c core.fileMode=false diff` is empty
    (`provenance/working_tree_content_only.patch`), so upstream file contents are unmodified.
  - The entire paper implementation is **untracked** (cyber runner, wrappers, stream,
    metrics, configs, scripts); see `provenance/git_status_ignoring_filemode.txt`.
- Working-tree patch included: yes (`provenance/working_tree.patch`, mode-only). The untracked
  implementation is included as files. `provenance/source_file_map.txt` gives a SHA-256 for every
  packaged code/config file, and all 70 are identical to the source.
- Code identity with the runs: the six files hashed in the central run manifest
  (`run_cyber.py`, `cyber_stream.py`, `cyber_metrics.py`, `cyber_adapters.py`,
  `FedDrift.py`, `FedDrift_Cyber.py`) match the packaged files exactly.
- Config identity: all 110 per-run overlays in `configs/runs/` resolve, through the packaged
  YAML layers, to configs identical to those recorded in the run metadata.
  `result_dir` and `config_template_source` are excluded from that comparison: they are path-only keys.

## Final experiment roots
- CSE central: `experiments/target_alert_v4_10seed/cse` (4 methods × 10 seeds, ρ=0.30, 40/40 validated)
- CICIoT central: `experiments/target_alert_v4_10seed/ciciot` (4 methods × 10 seeds, ρ=0.30, 40/40 validated)
- CSE exposure sweep: `experiments/cse_spatial_exposure_target_alert_v4_10seed` (FedAvg × ρ∈{0.10,0.30,0.50} × 10 seeds, 30/30 valid)
- Accepted analysis bundle: `paper_artifacts/corrected_v4_20260926_195133` → `reference_results/paper_analysis/`
- Selection reasons and rejected candidates: `provenance/final_experiment_roots.txt`

## Packaged implementation
- Core runner: `methods/run_cyber.py` plus its full import closure: `utils/cyber_{adapters,config,dataset,factory,metrics,model_bridge,models,schedule,stream}.py`, `entities/{base,FedCCFA,FedDrift}.py`, `entities/{FedAvg,FedDrift,FedCCFA}_Cyber.py`, and upstream `utils/{models,metric,gen_dataset}.py`. Code for algorithms and models not used in the paper was removed after validation (see "Post-validation trimming").
- Configs: `configs/{FedAvg,FedDrift,FedCCFA}.yaml` (upstream), `Cyber_Common.yaml`, `{FedAvg,FedDrift,FedCCFA}_Cyber.yaml`, `runs/` (110 overlays), `resolved/` (110 resolved configs from metadata), `PAPER_PROTOCOL.yaml`
- Data preparation scripts: `utils/prepare_cicids2018.py`, `utils/prepare_ciciot2023.py` (folder/file-name labels), wrapped by `scripts/prepare_cse.sh`, `scripts/prepare_ciciot.sh`, plus `scripts/inspect_cyber_data.py`
- Run wrappers: `scripts/run_cse_central.sh`, `scripts/run_ciciot_central.sh`, `scripts/run_cse_exposure_sweep.sh`. All three call `scripts/launch_runs.py`, a thin job pool around `methods/run_cyber.py` that reproduces each original launcher's environment. The original launchers are in `provenance/original_scripts/`.
- Analysis scripts: `analysis/paper_analysis.py`, the embedded Python of `scripts/regenerate_corrected_v4_results.bash` extracted verbatim (lines 16–1099). It covers the empirical, mean-UCB and all-site criteria, restricted delay with the 20-round horizon, exact sign-flip and McNemar tests, 100,000-resample bootstrap, and Holm correction. It is wrapped by `scripts/analyze_all.sh`.
- Plotting scripts: the same `analysis/paper_analysis.py` (`plot_all`) produces Figures 1, 2, 3 and S1; `scripts/make_figures.sh` runs it.
- Verification: `scripts/verify_reference_results.py`, `scripts/compare_runs.py`, `tests/` (4 files, 15 tests)

## Excluded material
- Old hotfixes: `fedccfa_*_hotfix_v*`, `cyber_stream_global_eval_hotfix_v3`, `fedprox_flash_hotfix_v1`, `prepaper_validation_*`, `ciciot2023_*_bundle_v1`, `ciciot2023_stream_interleave_hotfix_v4`, `fix_post_metrics_and_diagnose_v2`, the `*.patch` files, `apply_target_alert_v3/v4.bash`, `.target_alert_v*_backups`, and every `*.bak*`, `*.backup*`, `*.before_*` and `*_old.py` file. The final code imports none of these; their changes are already integrated into the active files, as the hash matches with the run manifest confirm.
- Superseded variants: `utils/prepare_cicids2018_fixed_v2.py` (byte-identical duplicate of `prepare_cicids2018.py`); `utils/prepare_cicids2018_fixed.py` (pandas-API edit dated 2026-10-27, after the runs; not the generating script); `utils/prepare_ciciot2023_old.py` (skips files without a label column); `plot_paper_figures.py` (earlier uncalibrated-EWS plots); 40+ exploratory analysis and launch scripts in `scripts/`.
- Stale test: `tests/test_algorithm_adapters.py`. Its string assertions target a pre-final adapter API (`'aggregate_by_updates'` vs. the final `'aggregate_by_updates(reconstructed_delta)'`), so 2 of its tests fail against the run-producing code.
- Smoke tests: `experiments/*smoke*`, `target_alert_smoke_v3/v4_seed0`, `corrected_smoke`
- Failed runs: none in the final roots (110/110 validated). Run logs (`run.log`) are excluded.
- Old 5-seed / uncalibrated analyses: `paper_cicids_central_*`, `ciciot2023_mirai_central_*`, `corrected_central_feature_disjoint_v1/v2`, `early_warning_*`, `visibility_*`, `threat_visibility_*`, `paper_figures`, `paper_analysis`, earlier bundles `paper_artifacts/{target_alert_v4,final_target_alert_v4}`; later `mechanism_*` runs
- Large/raw data: raw CSE-CIC-IDS2018 (6.7 GB) and CICIoT2023 CSVs, and `data/processed/*.npz` (1.3 GB + 256 MB) are not included. Checksums are in `provenance/processed_data_sha256.txt`; the original schema JSONs are in `provenance/processed_schema/`.
- Environment: two unrelated editable packages from the shared container (`-e /LLM/rg_rag_*`) removed from `environment/pip_freeze.txt`.

## Reproducibility checks
- Python compile: PASS. All 66 `.py` files compiled before trimming and all 35 after, and all `scripts/*.sh` pass `bash -n`.
- Unit tests: PASS (15/15 in the run container).
- 10-seed coverage: PASS (80 central + 30 sweep runs, seeds 0–9; protocol invariants checked in every run's metadata).
- Table I fingerprints: PASS (all 8 rows: success counts, delay mean and SD).
- Table II fingerprints: PASS (BAcc, Macro-F1, benign FPR for the 4 specified rows).
- Table III fingerprints: PASS (4 mean-UCB comparisons: Δ, 95% CI, exact p, Holm p; Holm family size 6).
- Re-analysis from packaged round CSVs: PASS. `scripts/analyze_all.sh` regenerates every table file byte-identically to the accepted bundle. In `analysis/`, only `run_dir` paths and the column order of `utility_per_run.csv` differ (values identical). The SHA-256 of all 330 input files matches the accepted bundle's `input_provenance.json`.
- One-seed fresh CSE rerun: PASS. Seed 0 of all 4 methods, run from the release tree via `scripts/run_cse_central.sh`, is identical to the reference in every metric column; only `*_seconds` timing columns differ.
- One-seed fresh CICIoT rerun: PASS (same comparison: seed 0, all 4 methods, identical metrics). See `provenance/fresh_rerun_comparison.txt`.
- Preprocessing regeneration: PASS. `prepare_ciciot2023.py` and `prepare_cicids2018.py`, run with the arguments in `scripts/prepare_*.sh`, regenerate `ciciot2023.npz` and `cicids2018.npz` byte-identically (SHA-256 matches). CSE: 65 GB peak RAM, 10 min.
- Hardcoded private paths scan: PASS, with documented exceptions. There are no `/LLM/` or `/home/jamiu` paths outside `reference_results/` and `provenance/`. The only remaining pattern hits are the relative `config_template_source: experiments/..._2026..._/...` labels in `configs/runs/` and `configs/resolved/`. That key is a provenance label that no code reads (`configs/resolved/README.md`). `RELEASE_AUDIT.md` itself names the source and container paths (`~/LLM/...`, `/LLM/...`) as provenance, which is intended.
- Credential scan: PASS (no hits). No files over 50 MB; the release is about 33 MB.

Validation environment: Docker container `RG_conda_lates` (image `my-conda-docker`), Python 3.12.7,
torch 2.10.0+cu128, cuDNN 9.10.02, 4× NVIDIA A40 (`environment/`). The original runs date from 2026-09-18;
the package versions in effect then were not recorded separately. The bit-identical seed-0 reruns show
that the current stack reproduces them.

## Post-validation trimming
Done at the author's request after all checks above had passed. Code for algorithms and models not used in the paper was removed:
- Deleted from `methods/` and `entities/`: AdaptiveFedAvg, Ditto, FedBABU, FedFM, FedPAC, FedProx, FedRep, Flash, IFCA, SCAFFOLD, pFedMe, and the FedProx/Flash cyber wrappers. Also deleted: the upstream image-classification runners `methods/{FedAvg,FedCCFA,FedDrift}.py` and `utils/drift.py`, which only those runners use.
- Edited:
  - `utils/models.py`: removed `FashionMNISTCNN`.
  - `utils/cyber_adapters.py`: removed `FedProxAdapter` and `FlashAdapter`.
  - `utils/cyber_factory.py`: removed the FedProx/Flash branches.
  - `methods/run_cyber.py`: `--algorithm` choices are now FedAvg, FedDrift, FedCCFA.

  The exact diff is in `provenance/post_validation_trimming.patch`.
- Kept deliberately: upstream `CifarCNN`. The upstream entities construct it as a placeholder (`upstream_stub_model: CNN`, `upstream_stub_dataset: CIFAR10`) before the cyber wrappers replace it with CyberMLP. Its construction is part of the seeded random stream, so removing it could change results.
- Re-validation of the trimmed tree: 35 Python files compile; 15/15 unit tests pass. A second fresh seed-0 rerun of all 4 methods on both datasets is identical to the reference in every metric column (max |diff| = 0; `provenance/fresh_rerun_comparison.txt`). The verifier passes all 3,458 checks.

## Deviations
NONE in results. Notes:
- The specification's paths (`/LLM/FedLearning/FedCCFA` → `/LLM/FedLearning/FedEarlyWarning_BigData2026`) were replaced by the requested source (`FL_Dup/FedLearning/FedCCFA`) and target repository (`FL_Dup/fedearlywarning`).
- The paper has Figures 1–3 plus supplementary Figure S1. These are the four figures the analysis produces; there is no separate "Figure 4" generator.
- Only seed 0 (all methods, both datasets) was rerun fresh, not all ten seeds or the sweep.

## Final status
READY FOR RELEASE
