# Corrected v4 analysis

Central: /LLM/FedLearning/FedCCFA/experiments/target_alert_v4_10seed
Spatial: /LLM/FedLearning/FedCCFA/experiments/cse_spatial_exposure_target_alert_v4_10seed

80 central runs and 30 spatial runs; seeds 0--9. Confirmation is the third qualifying round; horizon 20.
Calibration budget 0.5%; held-out FPR limit 1%; recall target 0.90.
Split audit counters are read from metadata; raw-feature separation is not recomputed.
Capture independence and train-only preprocessing provenance remain unverified.
Figure 3 uses multiclass columns from these v4 runs, not a separate legacy run matrix.

## Central mean-UCB results

| Dataset | Method | Mean-UCB success | Restricted delay | Final recall | Final target FPR |
| --- | --- | --- | --- | --- | --- |
| CSE-CIC-IDS2018 | FedAvg | 10/10 | 6.00 ± 0.00 | 1.0000 ± 0.0000 | 0.0043 ± 0.0012 |
| CSE-CIC-IDS2018 | FedDrift | 0/10 | 20.00 ± 0.00 | 0.0071 ± 0.0226 | 0.0057 ± 0.0016 |
| CSE-CIC-IDS2018 | FedCCFA | 8/10 | 13.70 ± 3.77 | 0.9523 ± 0.0365 | 0.0056 ± 0.0014 |
| CSE-CIC-IDS2018 | FedCCFA-GC | 10/10 | 3.10 ± 0.32 | 0.9999 ± 0.0002 | 0.0044 ± 0.0012 |
| CICIoT2023 | FedAvg | 10/10 | 5.30 ± 3.77 | 0.9979 ± 0.0007 | 0.0044 ± 0.0013 |
| CICIoT2023 | FedDrift | 0/10 | 20.00 ± 0.00 | 0.0953 ± 0.0723 | 0.0046 ± 0.0014 |
| CICIoT2023 | FedCCFA | 10/10 | 3.50 ± 0.53 | 0.9981 ± 0.0008 | 0.0045 ± 0.0009 |
| CICIoT2023 | FedCCFA-GC | 10/10 | 3.10 ± 0.32 | 0.9982 ± 0.0007 | 0.0045 ± 0.0012 |

## Spatial corrected results

| Exposure | Criterion | Success | Restricted delay | Records at confirmation | Patterns at confirmation |
| --- | --- | --- | --- | --- | --- |
| 10% | Empirical | 10/10 | 8.30 ± 1.25 | 30000 | 23417 |
| 10% | Mean-UCB | 10/10 | 8.30 ± 1.25 | 30000 | 23417 |
| 10% | All-site | 10/10 | 8.30 ± 1.25 | 30000 | 23417 |
| 30% | Empirical | 10/10 | 6.00 ± 0.00 | 21000 | 17420 |
| 30% | Mean-UCB | 10/10 | 6.00 ± 0.00 | 21000 | 17420 |
| 30% | All-site | 10/10 | 6.00 ± 0.00 | 21000 | 17420 |
| 50% | Empirical | 10/10 | 5.20 ± 0.42 | 18000 | 15307 |
| 50% | Mean-UCB | 10/10 | 5.20 ± 0.42 | 18000 | 15307 |
| 50% | All-site | 10/10 | 5.20 ± 0.42 | 18000 | 15307 |

## Spatial corrected reductions

| Criterion | rho10_delay | rho50_delay | difference | reduction_percent |
| --- | --- | --- | --- | --- |
| Empirical | 8.3000 | 5.2000 | -3.1000 | 37.3494 |
| Mean-UCB | 8.3000 | 5.2000 | -3.1000 | 37.3494 |
| All-site | 8.3000 | 5.2000 | -3.1000 | 37.3494 |

## Paired dissemination comparisons

| Dataset | Comparison | Criterion | Difference | 95% CI | Exact p | Holm p |
| --- | --- | --- | --- | --- | --- | --- |
| CICIoT2023 | GC vs FedAvg | All-site | -2.20 | [-4.40, -1.00] | 0.0020 | 0.0117 |
| CICIoT2023 | GC vs FedCCFA | All-site | -0.40 | [-0.80, +0.00] | 0.2188 | 0.2188 |
| CICIoT2023 | GC vs FedAvg | Mean-UCB | -2.20 | [-4.40, -1.00] | 0.0020 | 0.0117 |
| CICIoT2023 | GC vs FedCCFA | Mean-UCB | -0.40 | [-0.80, +0.00] | 0.2188 | 0.2188 |
| CICIoT2023 | GC vs FedAvg | Empirical | -1.10 | [-1.30, -1.00] | 0.0020 | 0.0117 |
| CICIoT2023 | GC vs FedCCFA | Empirical | -0.50 | [-0.80, -0.20] | 0.0625 | 0.0625 |
| CSE-CIC-IDS2018 | GC vs FedAvg | All-site | -2.90 | [-3.00, -2.70] | 0.0020 | 0.0117 |
| CSE-CIC-IDS2018 | GC vs FedCCFA | All-site | -11.30 | [-13.90, -8.80] | 0.0020 | 0.0117 |
| CSE-CIC-IDS2018 | GC vs FedAvg | Mean-UCB | -2.90 | [-3.00, -2.70] | 0.0020 | 0.0117 |
| CSE-CIC-IDS2018 | GC vs FedCCFA | Mean-UCB | -10.60 | [-13.00, -8.40] | 0.0020 | 0.0117 |
| CSE-CIC-IDS2018 | GC vs FedAvg | Empirical | -2.90 | [-3.00, -2.70] | 0.0020 | 0.0117 |
| CSE-CIC-IDS2018 | GC vs FedCCFA | Empirical | -9.00 | [-10.20, -7.80] | 0.0020 | 0.0117 |

## Interpretation

Final utility and temporal-average utility are separately tested. Attack recall means target-family recall.
All-site is a pointwise FPR-bound criterion; recall is averaged across unexposed clients.
Evidence-at-confirmation summaries exclude nonattaining seeds; delay summaries include them at 20.
Repeated client-model decisions on fixed examples are not independent experimental replicates.
Holm families: six method comparisons per dataset/metric; three exposure comparisons per spatial criterion/metric.
Bootstrap intervals are marginal. Precision and AUPRC depend on the constructed test mixture.

## Warnings and remaining provenance checks

- Verify that feature selection, imputation, and normalization were fitted without evaluation data.
- Verify threat-screening test-set independence separately.
- Read analysis/rho30_central_consistency.csv before treating spatial rho=0.30 as an exact central replication.
