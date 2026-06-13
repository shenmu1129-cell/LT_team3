# Reviewer Experiment Runner

All commands are intended to run from:

```bash
cd /home/sutongtong/wwt/code/fedmkt-based_bak
```

The scripts use the project-local conda environment `wwt310` and the local
InternVL3-2B checkpoint:

```bash
/home/sutongtong/wwt/model/InternVL3-2B
```

The default setting is conservative: `NUM_CLIENTS=4`, `NUM_ROUNDS=5`,
`BATCH_SIZE=1`. This avoids loading too many 2B models at once. For final
paper-scale runs, override the variables, for example `NUM_CLIENTS=20`.

## Background Commands

Smoke test:

```bash
bash experiments/run_reviewer_job.sh smoke 0
```

Multi-seed variance, for reporting mean +/- std:

```bash
bash experiments/run_reviewer_job.sh seed_variance 0
```

Benign IID / benign non-IID / malicious IID / malicious non-IID:

```bash
bash experiments/run_reviewer_job.sh non_iid_vs_malicious 0
```

Malicious-client-ratio robustness boundary:

```bash
bash experiments/run_reviewer_job.sh malicious_ratio 0
```

Logit-poisoning strength boundary:

```bash
bash experiments/run_reviewer_job.sh poison_strength 0
```

Attack success condition grid:

```bash
bash experiments/run_reviewer_job.sh attack_success 0
```

FedAvg / FedProx / active-inference aggregation comparison:

```bash
bash experiments/run_reviewer_job.sh baseline_methods 0
```

## Useful Overrides

Use another GPU:

```bash
bash experiments/run_reviewer_job.sh malicious_ratio 3
```

Run paper-scale client count:

```bash
NUM_CLIENTS=20 NUM_ROUNDS=10 bash experiments/run_reviewer_job.sh malicious_ratio 0
```

Run fewer seeds for a quick check:

```bash
SEEDS=1 bash experiments/run_reviewer_job.sh non_iid_vs_malicious 0
```

Customize malicious ratios:

```bash
RATIOS=0,0.1,0.2,0.3,0.4,0.5 bash experiments/run_reviewer_job.sh malicious_ratio 0
```

Customize poisoning strengths:

```bash
STRENGTHS=0,2,5,8,12 bash experiments/run_reviewer_job.sh poison_strength 0
```

Attack-success heatmap with a smaller grid:

```bash
SEEDS=1 RATIOS=0.1,0.3,0.5 STRENGTHS=2,5,8 DIRICHLET_ALPHAS=1.0,0.1,0.05 bash experiments/run_reviewer_job.sh attack_success 0
```

## Monitor / Stop

After a job starts, the script prints `RUN_ROOT`. Use:

```bash
tail -f <RUN_ROOT>/console.log
cat <RUN_ROOT>/pid.txt
kill $(cat <RUN_ROOT>/pid.txt)
```

Each job automatically writes:

```text
<RUN_ROOT>/summary/case_summary.csv
<RUN_ROOT>/summary/case_group_mean_std.csv
<RUN_ROOT>/summary/client_diagnostics.csv
```

Send the whole `<RUN_ROOT>` path, or these three CSV files, for analysis.
