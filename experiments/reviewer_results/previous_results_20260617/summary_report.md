# Reviewer Experiment Results (Curated)

This folder records the already completed reviewer-response experiments for the AIFLID rebuttal/resubmission package.

## GitHub Experiment Address

Repository path:

`experiments/reviewer_results/previous_results_20260617/summary_report.md`

Direct GitHub URL:

https://github.com/shenmu1129-cell/LT_team3/blob/codex/reviewer-results-20260617/experiments/reviewer_results/previous_results_20260617/summary_report.md

The full curated archive remains on the experiment server at:

`/home/sutongtong/wwt/code/fedmkt-based_bak/experiments/reviewer_runs/curated_previous_results_20260617_124151`

The failed trial `attack_success_20260614_222355` is excluded. The included usable runs are:

- `seed_variance_20260613_214615`
- `non_iid_vs_malicious_20260614_220748`
- `malicious_ratio_20260614_221502`
- `poison_strength_20260614_221523`
- `attack_success_20260614_222449`

## Available Full Files On Server

- `summary_all/summary_table.md`: merged Markdown summary table
- `*/summary/case_group_mean_std.csv`: per-profile mean/std tables
- `*/summary/case_summary.csv`: per-case final/average metrics, communication, and timing
- `*/summary/client_diagnostics.csv`: client-level VFE, KL, entropy, CE, malicious flag, and weights
- `*/metrics_json/`: raw `metrics_*.json` files copied for traceability

## Key Results

### Seed Variance

| case | n | final acc mean | final acc std | final f1 mean | avg VFE | suppression | VFE AUROC | avg comm MB | avg forward ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| seed_variance | 5 | 0.3500 | 0.2236 | 0.2000 | 1.8706 | 1.8480 | 0.8400 | 0.0075 | 821.37 |

Interpretation: across five seeds, VFE retains useful malicious-client separability (AUROC about 0.84) and active-inference weighting suppresses malicious clients on average.

### Benign Non-IID vs Malicious Poisoning

| case | n | final acc mean | final f1 mean | avg VFE | suppression | VFE AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iid_benign | 3 | 0.1667 | 0.0000 | 1.7840 | 0.0000 | - |
| iid_malicious | 3 | 0.1667 | 0.2500 | 1.8803 | 1.9488 | 0.8667 |
| noniid_benign | 3 | 0.2500 | 0.0000 | 1.7778 | 0.0000 | - |
| noniid_malicious | 3 | 0.2500 | 0.1667 | 1.8478 | 1.3253 | 0.8000 |

Interpretation: benign non-IID does not by itself inflate VFE substantially relative to benign IID, while malicious logit poisoning increases VFE and is down-weighted. This directly addresses the reviewer concern that VFE might simply confuse heterogeneity with attacks.

### Malicious Ratio Boundary

| case | n | final acc mean | final f1 mean | avg VFE | suppression | VFE AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| malratio_0p0 | 3 | 0.2500 | 0.0000 | 1.7778 | 0.0000 | - |
| malratio_0p1 | 3 | 0.2500 | 0.0000 | 1.7778 | 0.0000 | - |
| malratio_0p2 | 3 | 0.2500 | 0.1667 | 1.8478 | 1.3253 | 0.8000 |
| malratio_0p3 | 3 | 0.2500 | 0.1667 | 1.8478 | 1.3253 | 0.8000 |
| malratio_0p4 | 3 | 0.3333 | 0.3333 | 1.8656 | 0.8886 | 0.7667 |
| malratio_0p5 | 3 | 0.3333 | 0.3333 | 1.8656 | 0.8886 | 0.7667 |

Interpretation: ratio 0.2-0.3 shows clear malicious separability and down-weighting; at 0.4-0.5 the suppression weakens, giving an initial robustness-boundary signal. Because these are small-scale four-client runs, ratio 0.1 rounds to zero malicious clients and should not be used as a final paper-scale boundary point.

### Poison Strength Boundary

| case | n | final acc mean | final f1 mean | avg VFE | suppression | VFE AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| poison_strength_0p0 | 3 | 0.2500 | 0.1667 | 1.7709 | 0.9742 | 0.4489 |
| poison_strength_2p0 | 3 | 0.2500 | 0.1667 | 1.7816 | 0.9698 | 0.5778 |
| poison_strength_5p0 | 3 | 0.2500 | 0.1667 | 1.8478 | 1.3253 | 0.8000 |
| poison_strength_8p0 | 3 | 0.2500 | 0.1667 | 1.9789 | 2.5522 | 0.8000 |
| poison_strength_12p0 | 3 | 0.2500 | 0.1667 | 2.2191 | 8.3612 | 0.8000 |

Interpretation: VFE and weight suppression increase with logit-poisoning strength, giving a usable attack-strength/failure-mode trend.

### Attack Success Grid (Selected Cells)

| case | n | avg VFE | suppression | VFE AUROC | final f1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| success_alpha0p1_ratio0p3_strength5p0 | 1 | 2.0037 | 1.6043 | 1.0000 | 0.2500 |
| success_alpha0p1_ratio0p3_strength8p0 | 1 | 2.1840 | 3.3238 | 1.0000 | 0.2500 |
| success_alpha0p1_ratio0p5_strength5p0 | 1 | 2.1131 | 1.4462 | 1.0000 | 0.5000 |
| success_alpha0p1_ratio0p5_strength8p0 | 1 | 2.4614 | 2.8626 | 1.0000 | 0.5000 |
| success_alpha1p0_ratio0p3_strength8p0 | 1 | 2.2143 | 3.8302 | 1.0000 | 0.2500 |
| success_alpha1p0_ratio0p5_strength8p0 | 1 | 2.3551 | 3.2788 | 0.9000 | 0.5000 |

Interpretation: attack-success cells with higher malicious ratio and stronger poisoning show larger VFE and stronger suppression. These are single-seed exploratory results and should be expanded before being used as final statistical claims.

## Notes For Paper Use

- Current runs are intentionally small and reproducibility-oriented; many use `NUM_CLIENTS=4` and `NUM_ROUNDS=5`.
- For paper-scale robustness-boundary tables, rerun key profiles with `NUM_CLIENTS=20` so ratios such as 0.1 map to actual malicious clients.
- Communication fields here count serialized logits traffic; timing columns include local forward latency and aggregation/weight computation where available.
- No model checkpoints are included in this GitHub result summary.
