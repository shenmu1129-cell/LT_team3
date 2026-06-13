#!/usr/bin/env python3
"""
Summarize reviewer experiment metrics into CSV files.
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def auroc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return ""
    rank_sum = 0.0
    for rank, (_, label) in enumerate(pairs, start=1):
        if label:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def load_metrics(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def case_name_from_metrics(path):
    # .../<case>/logs_federated/metrics_xxx.json
    return path.parents[1].name


def flatten(root):
    rows = []
    diagnostics_rows = []
    for metrics_path in sorted(root.rglob("metrics_*.json")):
        case = case_name_from_metrics(metrics_path)
        data = load_metrics(metrics_path)
        summary = data.get("summary", {})
        rounds = data.get("rounds", [])
        final_round = rounds[-1] if rounds else {}
        extra = final_round.get("extra_metrics", {})
        suppression = extra.get("weight_suppression", {})

        rows.append({
            "case": case,
            "metrics_file": str(metrics_path),
            "total_rounds": summary.get("total_rounds", ""),
            "final_accuracy": summary.get("final_round_accuracy", ""),
            "final_f1": summary.get("final_round_f1", ""),
            "avg_accuracy": summary.get("avg_accuracy", ""),
            "avg_f1": summary.get("avg_f1_score", ""),
            "avg_free_energy": summary.get("avg_free_energy", ""),
            "benign_mean_weight": suppression.get("benign_mean_weight", ""),
            "malicious_mean_weight": suppression.get("malicious_mean_weight", ""),
            "weight_suppression_ratio": suppression.get("benign_to_malicious_ratio", ""),
        })

        labels = []
        scores = []
        for round_data in rounds:
            round_id = round_data.get("round_id")
            extra = round_data.get("extra_metrics", {})
            weights = round_data.get("weights", [])
            for diag in extra.get("client_diagnostics", []):
                comp = diag.get("free_energy_components", {})
                is_malicious = int(bool(diag.get("is_malicious", False)))
                score = comp.get("free_energy", "")
                if score != "":
                    labels.append(is_malicious)
                    scores.append(float(score))
                cid = diag.get("client_id", "")
                diagnostics_rows.append({
                    "case": case,
                    "round_id": round_id,
                    "client_id": cid,
                    "role": diag.get("role", ""),
                    "is_malicious": is_malicious,
                    "attack_ratio": diag.get("attack_ratio", ""),
                    "weight": weights[cid] if isinstance(cid, int) and cid < len(weights) else "",
                    "free_energy": comp.get("free_energy", ""),
                    "kl_divergence": comp.get("kl_divergence", ""),
                    "entropy": comp.get("entropy", ""),
                    "cross_entropy": comp.get("cross_entropy", ""),
                    "logit_poisoned": diag.get("logit_poisoned", ""),
                    "forward_ms": diag.get("forward_ms", ""),
                })

        rows[-1]["vfe_malicious_auroc"] = auroc(labels, scores)
    return rows, diagnostics_rows


def write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_grouped(path, rows):
    groups = {}
    for row in rows:
        prefix = row["case"].rsplit("_seed", 1)[0]
        groups.setdefault(prefix, []).append(row)

    out = []
    numeric_fields = [
        "final_accuracy", "final_f1", "avg_accuracy", "avg_f1",
        "avg_free_energy", "weight_suppression_ratio", "vfe_malicious_auroc"
    ]
    for case, case_rows in sorted(groups.items()):
        record = {"case_group": case, "n": len(case_rows)}
        for field in numeric_fields:
            vals = []
            for row in case_rows:
                try:
                    if row.get(field) != "":
                        vals.append(float(row[field]))
                except (TypeError, ValueError):
                    pass
            if vals:
                record[f"{field}_mean"] = mean(vals)
                record[f"{field}_std"] = stdev(vals) if len(vals) > 1 else 0.0
            else:
                record[f"{field}_mean"] = ""
                record[f"{field}_std"] = ""
        out.append(record)
    write_csv(path, out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    root = Path(args.run_root)
    out_dir = Path(args.out_dir) if args.out_dir else root / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, diagnostics_rows = flatten(root)
    write_csv(out_dir / "case_summary.csv", rows)
    write_csv(out_dir / "client_diagnostics.csv", diagnostics_rows)
    write_grouped(out_dir / "case_group_mean_std.csv", rows)
    print(f"Wrote summaries to {out_dir}")


if __name__ == "__main__":
    main()
