# LT_team3

## Reviewer Experiment Results Entry

This repository includes a curated Markdown summary of the previously completed reviewer-response experiments for the AIFLID project.

### GitHub Experiment Address

Main readable report:

[experiments/reviewer_results/previous_results_20260617/summary_report.md](experiments/reviewer_results/previous_results_20260617/summary_report.md)

Direct GitHub URL:

https://github.com/shenmu1129-cell/LT_team3/blob/codex/reviewer-results-20260617/experiments/reviewer_results/previous_results_20260617/summary_report.md

### What The Report Contains

The report summarizes these completed experiment profiles:

- `seed_variance`: random-seed variance and mean/std statistics
- `non_iid_vs_malicious`: benign IID, benign non-IID, malicious IID, malicious non-IID comparison
- `malicious_ratio`: malicious-client-ratio robustness boundary
- `poison_strength`: logit-poisoning strength boundary
- `attack_success`: attack-success condition grid, selected cells

### Full Server-Side Result Archive

The full CSV/diagnostics archive is stored on the experiment server at:

`/home/sutongtong/wwt/code/fedmkt-based_bak/experiments/reviewer_runs/curated_previous_results_20260617_124151`

It contains:

- `summary_all/summary_table.md`
- `*/summary/case_group_mean_std.csv`
- `*/summary/case_summary.csv`
- `*/summary/client_diagnostics.csv`
- raw `metrics_*.json` backups

The failed trial `attack_success_20260614_222355` is intentionally excluded.
