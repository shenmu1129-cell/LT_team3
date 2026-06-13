#!/usr/bin/env python3
"""
Run reviewer-response experiment grids.

Examples:
  python experiments/run_reviewer_experiments.py --profile smoke --dry-run
  python experiments/run_reviewer_experiments.py --profile non_iid_vs_malicious --gpu 0
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = "/home/sutongtong/wwt/model/InternVL3-2B"
DEFAULT_DATAROOT = "/home/sutongtong/LanTu_team3/dataset/nuScenes/train"


def base_args(args, case_name, seed):
    out_root = Path(args.run_root)
    case_log_dir = out_root / case_name / "logs_federated"
    case_save_dir = out_root / case_name / "checkpoints_federated"
    return [
        args.python,
        "run_federated_qwenvl.py",
        "--seed", str(seed),
        "--num_clients", str(args.num_clients),
        "--num_rounds", str(args.num_rounds),
        "--local_epochs", str(args.local_epochs),
        "--batch_size", str(args.batch_size),
        "--num_synthetic_samples", str(args.num_synthetic_samples),
        "--model_type", args.model_type,
        "--model_path", args.model_path,
        "--dataroot", args.dataroot,
        "--version", args.version,
        "--partition_mode", args.partition_mode,
        "--dirichlet_alpha", str(args.dirichlet_alpha),
        "--free_energy_mode", args.free_energy_mode,
        "--aggregation_method", args.aggregation_method,
        "--device", args.device,
        "--log_dir", str(case_log_dir),
        "--save_dir", str(case_save_dir),
        "--no_save_model",
    ]


def build_cases(args):
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    cases = []

    if args.profile == "smoke":
        for seed in seeds[:1]:
            cmd = base_args(args, f"smoke_seed{seed}", seed)
            cmd += [
                "--use_synthetic_data",
                "--client_attack_mode", "malicious",
                "--malicious_client_ratio", "0.5",
                "--benign_attack_ratio", "0.0",
                "--malicious_attack_ratio", "0.8",
                "--enable_logit_poisoning",
                "--logit_poisoning_strength", "5.0",
            ]
            cases.append(("smoke", cmd))

    elif args.profile == "seed_variance":
        for seed in seeds:
            cmd = base_args(args, f"seed_variance_seed{seed}", seed)
            cmd += [
                "--client_attack_mode", "malicious",
                "--malicious_client_ratio", str(args.malicious_ratio),
                "--benign_attack_ratio", "0.0",
                "--malicious_attack_ratio", str(args.malicious_attack_ratio),
                "--enable_logit_poisoning",
                "--logit_poisoning_strength", str(args.logit_poisoning_strength),
            ]
            cases.append((f"seed_variance_seed{seed}", cmd))

    elif args.profile == "non_iid_vs_malicious":
        settings = [
            ("iid_benign", "iid", 1.0, 0.0, False),
            ("noniid_benign", "non-iid-dirichlet", 0.1, 0.0, False),
            ("iid_malicious", "iid", 1.0, args.malicious_ratio, True),
            ("noniid_malicious", "non-iid-dirichlet", 0.1, args.malicious_ratio, True),
        ]
        for seed in seeds:
            for name, partition, alpha, mr, poison in settings:
                cmd = base_args(args, f"{name}_seed{seed}", seed)
                cmd[cmd.index("--partition_mode") + 1] = partition
                cmd[cmd.index("--dirichlet_alpha") + 1] = str(alpha)
                cmd += [
                    "--client_attack_mode", "malicious",
                    "--malicious_client_ratio", str(mr),
                    "--benign_attack_ratio", "0.0",
                    "--malicious_attack_ratio", str(args.malicious_attack_ratio),
                ]
                if poison:
                    cmd += [
                        "--enable_logit_poisoning",
                        "--logit_poisoning_strength", str(args.logit_poisoning_strength),
                    ]
                cases.append((f"{name}_seed{seed}", cmd))

    elif args.profile == "malicious_ratio":
        ratios = [float(v) for v in args.ratios.split(",") if v.strip()]
        for seed in seeds:
            for ratio in ratios:
                tag = str(ratio).replace(".", "p")
                cmd = base_args(args, f"malratio_{tag}_seed{seed}", seed)
                cmd += [
                    "--client_attack_mode", "malicious",
                    "--malicious_client_ratio", str(ratio),
                    "--benign_attack_ratio", "0.0",
                    "--malicious_attack_ratio", str(args.malicious_attack_ratio),
                ]
                if ratio > 0:
                    cmd += [
                        "--enable_logit_poisoning",
                        "--logit_poisoning_strength", str(args.logit_poisoning_strength),
                    ]
                cases.append((f"malratio_{ratio}_seed{seed}", cmd))

    elif args.profile == "poison_strength":
        strengths = [float(v) for v in args.strengths.split(",") if v.strip()]
        for seed in seeds:
            for strength in strengths:
                tag = str(strength).replace(".", "p")
                cmd = base_args(args, f"poison_strength_{tag}_seed{seed}", seed)
                cmd += [
                    "--client_attack_mode", "malicious",
                    "--malicious_client_ratio", str(args.malicious_ratio),
                    "--benign_attack_ratio", "0.0",
                    "--malicious_attack_ratio", str(args.malicious_attack_ratio),
                ]
                if strength > 0:
                    cmd += [
                        "--enable_logit_poisoning",
                        "--logit_poisoning_strength", str(strength),
                    ]
                cases.append((f"poison_strength_{strength}_seed{seed}", cmd))

    elif args.profile == "attack_success":
        ratios = [float(v) for v in args.ratios.split(",") if v.strip()]
        strengths = [float(v) for v in args.strengths.split(",") if v.strip()]
        alphas = [float(v) for v in args.dirichlet_alphas.split(",") if v.strip()]
        for seed in seeds:
            for alpha in alphas:
                for ratio in ratios:
                    for strength in strengths:
                        ratio_tag = str(ratio).replace(".", "p")
                        strength_tag = str(strength).replace(".", "p")
                        alpha_tag = str(alpha).replace(".", "p")
                        cmd = base_args(
                            args,
                            f"success_alpha{alpha_tag}_ratio{ratio_tag}_strength{strength_tag}_seed{seed}",
                            seed,
                        )
                        if alpha >= 1.0:
                            cmd[cmd.index("--partition_mode") + 1] = "iid"
                            cmd[cmd.index("--dirichlet_alpha") + 1] = str(alpha)
                        else:
                            cmd[cmd.index("--partition_mode") + 1] = "non-iid-dirichlet"
                            cmd[cmd.index("--dirichlet_alpha") + 1] = str(alpha)
                        cmd += [
                            "--client_attack_mode", "malicious",
                            "--malicious_client_ratio", str(ratio),
                            "--benign_attack_ratio", "0.0",
                            "--malicious_attack_ratio", str(args.malicious_attack_ratio),
                        ]
                        if ratio > 0 and strength > 0:
                            cmd += [
                                "--enable_logit_poisoning",
                                "--logit_poisoning_strength", str(strength),
                            ]
                        cases.append((f"attack_success_seed{seed}", cmd))

    elif args.profile == "baseline_methods":
        methods = [v.strip() for v in args.methods.split(",") if v.strip()]
        for seed in seeds:
            for method in methods:
                cmd = base_args(args, f"baseline_{method}_seed{seed}", seed)
                cmd[cmd.index("--aggregation_method") + 1] = method
                cmd += [
                    "--client_attack_mode", "malicious",
                    "--malicious_client_ratio", str(args.malicious_ratio),
                    "--benign_attack_ratio", "0.0",
                    "--malicious_attack_ratio", str(args.malicious_attack_ratio),
                    "--enable_logit_poisoning",
                    "--logit_poisoning_strength", str(args.logit_poisoning_strength),
                ]
                cases.append((f"baseline_{method}_seed{seed}", cmd))

    else:
        raise ValueError(f"Unknown profile: {args.profile}")

    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=[
        "smoke", "seed_variance", "non_iid_vs_malicious", "malicious_ratio",
        "poison_strength", "attack_success", "baseline_methods"
    ], default="smoke")
    parser.add_argument("--python", default="python")
    parser.add_argument("--output_root", default="experiments/reviewer_runs")
    parser.add_argument("--run_root", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", default="")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--ratios", default="0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--strengths", default="0,2,5,8,12")
    parser.add_argument("--dirichlet_alphas", default="1.0,0.1,0.05")
    parser.add_argument("--methods", default="active_inference,fedavg,fedprox")
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--num_rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_synthetic_samples", type=int, default=1000)
    parser.add_argument("--model_type", default="internvl")
    parser.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--dataroot", default=DEFAULT_DATAROOT)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--partition_mode", default="non-iid-dirichlet")
    parser.add_argument("--dirichlet_alpha", type=float, default=0.1)
    parser.add_argument("--free_energy_mode", default="ce_entropy")
    parser.add_argument("--aggregation_method", default="active_inference")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--malicious_ratio", type=float, default=0.2)
    parser.add_argument("--malicious_attack_ratio", type=float, default=0.8)
    parser.add_argument("--logit_poisoning_strength", type=float, default=5.0)
    args = parser.parse_args()
    args.run_root = Path(args.run_root) if args.run_root else (
        Path(args.output_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    cases = build_cases(args)
    print(json.dumps({
        "profile": args.profile,
        "num_cases": len(cases),
        "run_root": str(args.run_root)
    }, indent=2))

    env = os.environ.copy()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    for name, cmd in cases:
        print("\n==", name)
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, check=True)


if __name__ == "__main__":
    main()
