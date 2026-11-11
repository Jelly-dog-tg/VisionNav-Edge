"""Evaluate all 400k BC-PPO V2 models and compare against their 200k runs.

Run from the project root:
    python experiments/33_evaluate_bc_residual_ppo_v2_400k.py

This script calls the existing paired evaluator:
    experiments/31_evaluate_bc_residual_ppo_v2.py

It then compares each seed's 200k and 400k results and writes a
cross-seed CSV/JSON summary.
"""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / "experiments" / "31_evaluate_bc_residual_ppo_v2.py"
TASKS = ROOT / "logs" / "p5_evaluation" / "benchmark_tasks.json"
BC_CHECKPOINT = ROOT / "models" / "behavior_cloning" / "bc_best.pth"
SEEDS_DEFAULT = [42, 123, 2026]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare the 200k and 400k BC-PPO V2 runs."
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=SEEDS_DEFAULT,
        choices=[42, 123, 2026],
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Do not rerun the benchmark; aggregate existing 400k summaries only.",
    )
    parser.add_argument("--max-steps", type=int, default=2500)
    return parser.parse_args()


def source_200k_dir(seed):
    if seed == 42:
        return ROOT / "models" / "bc_residual_ppo_v2"
    return ROOT / "models" / f"bc_residual_ppo_v2_seed_{seed}"


def summary_200k_path(seed):
    if seed == 42:
        candidates = [
            ROOT / "logs" / "p6_evaluation" / "bc_residual_ppo_v2_summary.json",
            ROOT / "logs" / "p6_evaluation" / "seed_42" / "bc_residual_ppo_v2_summary.json",
        ]
    else:
        candidates = [
            ROOT / "logs" / "p6_evaluation" / f"seed_{seed}" / "bc_residual_ppo_v2_summary.json"
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def summary_400k_path(seed):
    return (
        ROOT
        / "logs"
        / "p6_evaluation"
        / f"seed_{seed}_400k"
        / "bc_residual_ppo_v2_summary.json"
    )


def run_evaluation(seed, max_steps):
    model_dir = ROOT / "models" / f"bc_residual_ppo_v2_seed_{seed}_400k"
    model_path = model_dir / "bc_residual_ppo_v2_final.zip"
    norm_path = model_dir / "vecnormalize.pkl"
    output_dir = ROOT / "logs" / "p6_evaluation" / f"seed_{seed}_400k"

    required = [EVALUATOR, TASKS, BC_CHECKPOINT, model_path, norm_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing required files for seed "
            f"{seed}:\n  - " + "\n  - ".join(missing)
        )

    command = [
        sys.executable,
        str(EVALUATOR),
        "--tasks", str(TASKS),
        "--model", str(model_path),
        "--vecnormalize", str(norm_path),
        "--bc-checkpoint", str(BC_CHECKPOINT),
        "--output-dir", str(output_dir),
        "--max-steps", str(max_steps),
        "--seed", str(seed),
    ]

    print(f"\n========== EVALUATING SEED {seed} / 400K ==========")
    subprocess.run(command, cwd=ROOT, check=True)


def load_summary(path, label):
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} summary not found: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "controllers" not in payload:
        raise ValueError(f"Invalid summary format in {path}")
    return payload


def get_controller(summary, key):
    controllers = summary["controllers"]
    if key not in controllers:
        raise KeyError(
            f"Controller '{key}' missing. Available: {list(controllers)}"
        )
    return controllers[key]


def numeric(metrics, key):
    value = metrics.get(key)
    return float(value) if value is not None else None


def make_comparison_row(seed, old_summary, new_summary):
    old_ppo = get_controller(old_summary, "BC + PPO V2")
    new_ppo = get_controller(new_summary, "BC + PPO V2")
    old_bc = get_controller(old_summary, "BC only")
    new_bc = get_controller(new_summary, "BC only")

    old_success = numeric(old_ppo, "success_rate")
    new_success = numeric(new_ppo, "success_rate")
    old_steps = numeric(old_ppo, "average_steps")
    new_steps = numeric(new_ppo, "average_steps")
    old_distance = numeric(old_ppo, "average_actual_distance_m")
    new_distance = numeric(new_ppo, "average_actual_distance_m")
    old_error = numeric(old_ppo, "average_final_error_m")
    new_error = numeric(new_ppo, "average_final_error_m")

    return {
        "seed": seed,
        "bc_success_rate_200k": numeric(old_bc, "success_rate"),
        "bc_success_rate_400k_eval": numeric(new_bc, "success_rate"),
        "ppo_success_rate_200k": old_success,
        "ppo_success_rate_400k": new_success,
        "success_rate_change_percentage_points": (
            (new_success - old_success) * 100
            if old_success is not None and new_success is not None else None
        ),
        "ppo_average_steps_200k": old_steps,
        "ppo_average_steps_400k": new_steps,
        "average_steps_change": (
            new_steps - old_steps
            if old_steps is not None and new_steps is not None else None
        ),
        "ppo_actual_distance_m_200k": old_distance,
        "ppo_actual_distance_m_400k": new_distance,
        "actual_distance_change_m": (
            new_distance - old_distance
            if old_distance is not None and new_distance is not None else None
        ),
        "ppo_final_error_m_200k": old_error,
        "ppo_final_error_m_400k": new_error,
        "final_error_change_m": (
            new_error - old_error
            if old_error is not None and new_error is not None else None
        ),
        "ppo_residual_active_fraction_200k": numeric(
            old_ppo, "average_residual_active_fraction"
        ),
        "ppo_residual_active_fraction_400k": numeric(
            new_ppo, "average_residual_active_fraction"
        ),
    }


def add_aggregate_rows(rows):
    numeric_fields = [
        key for key in rows[0]
        if key != "seed" and all(
            row.get(key) is None or isinstance(row.get(key), (int, float))
            for row in rows
        )
    ]

    mean_row = {"seed": "MEAN"}
    std_row = {"seed": "SAMPLE_STD"}
    for key in numeric_fields:
        values = [
            float(row[key]) for row in rows if row.get(key) is not None
        ]
        mean_row[key] = statistics.mean(values) if values else None
        std_row[key] = (
            statistics.stdev(values) if len(values) >= 2 else None
        )
    return mean_row, std_row


def write_csv(path, rows):
    if not rows:
        raise ValueError("No comparison rows to write.")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    if not args.skip_evaluation:
        for seed in args.seeds:
            run_evaluation(seed, args.max_steps)

    comparison_rows = []
    source_paths = {}

    for seed in args.seeds:
        old_path = summary_200k_path(seed)
        new_path = summary_400k_path(seed)

        old_summary = load_summary(old_path, f"Seed {seed} 200k")
        new_summary = load_summary(new_path, f"Seed {seed} 400k")

        if int(old_summary.get("seed", seed)) != seed:
            raise ValueError(f"Seed mismatch in 200k summary: {old_path}")
        if int(new_summary.get("seed", seed)) != seed:
            raise ValueError(f"Seed mismatch in 400k summary: {new_path}")

        comparison_rows.append(
            make_comparison_row(seed, old_summary, new_summary)
        )
        source_paths[str(seed)] = {
            "200k_summary": str(old_path),
            "400k_summary": str(new_path),
        }

    mean_row, std_row = add_aggregate_rows(comparison_rows)
    output_dir = ROOT / "logs" / "p6_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ppo_v2_200k_vs_400k_comparison.csv"
    json_path = output_dir / "ppo_v2_200k_vs_400k_comparison.json"

    all_csv_rows = comparison_rows + [mean_row, std_row]
    write_csv(csv_path, all_csv_rows)

    report = {
        "experiment": "BC-PPO V2 200k vs 400k continuation comparison",
        "seeds": args.seeds,
        "task_count_per_seed": 20,
        "max_steps_per_task": args.max_steps,
        "interpretation": {
            "success_rate_change": "percentage points; positive means 400k improved",
            "average_steps_change": "400k minus 200k; negative means fewer steps",
            "actual_distance_change_m": "400k minus 200k; negative means shorter distance",
            "final_error_change_m": "400k minus 200k; negative means smaller final error",
            "sample_std": "sample standard deviation across the selected seeds",
        },
        "source_files": source_paths,
        "per_seed": comparison_rows,
        "mean": mean_row,
        "sample_std": std_row,
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== PPO V2 200K vs 400K COMPARISON ==========")
    for row in comparison_rows:
        print(
            f"Seed {row['seed']}: "
            f"success {row['ppo_success_rate_200k']} -> "
            f"{row['ppo_success_rate_400k']}; "
            f"steps {row['ppo_average_steps_200k']} -> "
            f"{row['ppo_average_steps_400k']}; "
            f"distance {row['ppo_actual_distance_m_200k']} -> "
            f"{row['ppo_actual_distance_m_400k']}"
        )

    print("\nMean:")
    print(json.dumps(mean_row, ensure_ascii=False, indent=2))
    print("\nSample standard deviation:")
    print(json.dumps(std_row, ensure_ascii=False, indent=2))
    print("\nSaved:", csv_path)
    print("Saved:", json_path)
    print("\n400K EVALUATION AND COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
