"""Run the fixed 20-task BC vs BC-PPO V2 evaluation for one seed.

Examples from the project root:
    python experiments/32_validate_bc_residual_ppo_v2_seed.py --seed 123
    python experiments/32_validate_bc_residual_ppo_v2_seed.py --seed 2026

This launcher selects the matching seed-specific PPO checkpoint and output
folder, then invokes the existing paired benchmark evaluator:
    experiments/31_evaluate_bc_residual_ppo_v2.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EVALUATOR = ROOT / "experiments" / "31_evaluate_bc_residual_ppo_v2.py"
BC_CHECKPOINT = ROOT / "models" / "behavior_cloning" / "bc_best.pth"
TASKS = ROOT / "logs" / "p5_evaluation" / "benchmark_tasks.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate one independently trained BC-PPO V2 seed."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Must match the training run name. Default: seed_<seed>.",
    )
    parser.add_argument("--max-steps", type=int, default=2500)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed < 0:
        raise ValueError("--seed must be a non-negative integer.")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    run_name = args.run_name or f"seed_{args.seed}"
    if not run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--run-name may contain only letters, digits, _ and -.")

    model_dir = ROOT / "models" / f"bc_residual_ppo_v2_{run_name}"
    model_path = model_dir / "bc_residual_ppo_v2_final.zip"
    vecnormalize_path = model_dir / "vecnormalize.pkl"
    output_dir = ROOT / "logs" / "p6_evaluation" / run_name

    required = {
        "benchmark task file": TASKS,
        "BC checkpoint": BC_CHECKPOINT,
        "paired benchmark evaluator": EVALUATOR,
        "seed-specific PPO model": model_path,
        "seed-specific VecNormalize statistics": vecnormalize_path,
    }

    missing = [
        f"{label}: {path}"
        for label, path in required.items()
        if not path.is_file()
    ]
    if missing:
        details = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(
            "Cannot start validation because required files are missing:\n"
            f"{details}\n"
            "Train this seed first and ensure the previous evaluator "
            "31_evaluate_bc_residual_ppo_v2.py is present."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(EVALUATOR),
        "--tasks", str(TASKS),
        "--model", str(model_path),
        "--vecnormalize", str(vecnormalize_path),
        "--bc-checkpoint", str(BC_CHECKPOINT),
        "--output-dir", str(output_dir),
        "--max-steps", str(args.max_steps),
        "--seed", str(args.seed),
    ]

    print("========== BC-PPO V2 SEED VALIDATION ==========")
    print("Seed:", args.seed)
    print("Run name:", run_name)
    print("Benchmark:", TASKS)
    print("PPO model:", model_path)
    print("VecNormalize:", vecnormalize_path)
    print("Output:", output_dir)
    print("\nStarting paired BC vs BC-PPO V2 evaluation...\n")

    completed = subprocess.run(command, cwd=ROOT, check=False)

    if completed.returncode != 0:
        raise SystemExit(
            f"Evaluation failed with exit code {completed.returncode}."
        )

    print("\n========== SEED VALIDATION COMPLETED ==========")
    print("Seed:", args.seed)
    print("Results:", output_dir)


if __name__ == "__main__":
    main()
