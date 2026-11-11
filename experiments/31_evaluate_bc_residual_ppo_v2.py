"""Evaluate BC-only and BC + PPO V2 on the fixed held-out benchmark.

Run from the project root:
    python experiments/31_evaluate_bc_residual_ppo_v2.py

The BC-only baseline uses zero PPO residuals in the same V2 environment.
The V2 policy uses the trained PPO model and its saved VecNormalize statistics.
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionnav.learning.bc_residual_navigation_env_v2 import (  # noqa: E402
    BCResidualNavigationEnvV2,
)


BC_CHECKPOINT = ROOT / "models" / "behavior_cloning" / "bc_best.pth"
MODEL_PATH = (
    ROOT / "models" / "bc_residual_ppo_v2" / "bc_residual_ppo_v2_final.zip"
)
VECNORMALIZE_PATH = (
    ROOT / "models" / "bc_residual_ppo_v2" / "vecnormalize.pkl"
)
TASKS_PATH = ROOT / "logs" / "p5_evaluation" / "benchmark_tasks.json"
OUTPUT_DIR = ROOT / "logs" / "p6_evaluation"

SEED = 42
DEFAULT_MAX_STEPS = 2500


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate BC-only and BC-PPO V2 on fixed benchmark tasks."
    )
    parser.add_argument("--tasks", type=Path, default=TASKS_PATH)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--vecnormalize", type=Path, default=VECNORMALIZE_PATH)
    parser.add_argument("--bc-checkpoint", type=Path, default=BC_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def extract_tasks(payload):
    """Accept the task-list layouts used by the benchmark generator."""
    if isinstance(payload, list):
        tasks = payload
    elif isinstance(payload, dict):
        tasks = None
        for key in ("tasks", "benchmark_tasks", "items", "episodes"):
            value = payload.get(key)
            if isinstance(value, list):
                tasks = value
                break
        if tasks is None:
            raise ValueError(
                "benchmark_tasks.json must contain a task list under "
                "'tasks' (or benchmark_tasks/items/episodes)."
            )
    else:
        raise ValueError("Unsupported benchmark_tasks.json structure.")

    if not tasks:
        raise ValueError("The benchmark task list is empty.")

    validated = []
    required = ("start_xy", "goal_xy", "waypoints_world", "start_yaw_rad")

    for index, original in enumerate(tasks, start=1):
        if not isinstance(original, dict):
            raise ValueError(f"Task {index} is not a JSON object.")

        missing = [key for key in required if key not in original]
        if missing:
            raise ValueError(
                f"Task {index} is missing required fields: {missing}"
            )

        task = dict(original)
        start = np.asarray(task["start_xy"], dtype=np.float32)
        goal = np.asarray(task["goal_xy"], dtype=np.float32)
        waypoints = np.asarray(task["waypoints_world"], dtype=np.float32)
        yaw = float(task["start_yaw_rad"])

        if start.shape != (2,) or goal.shape != (2,):
            raise ValueError(f"Task {index}: start_xy and goal_xy must be 2-D.")
        if (
            waypoints.ndim != 2
            or waypoints.shape[1] != 2
            or len(waypoints) == 0
        ):
            raise ValueError(f"Task {index}: waypoints_world must have shape (N, 2).")
        if not (
            np.isfinite(start).all()
            and np.isfinite(goal).all()
            and np.isfinite(waypoints).all()
            and np.isfinite(yaw)
        ):
            raise ValueError(f"Task {index} contains NaN or Inf.")

        task["start_xy"] = start.tolist()
        task["goal_xy"] = goal.tolist()
        task["waypoints_world"] = waypoints.tolist()
        task["start_yaw_rad"] = yaw
        task.setdefault("task_id", task.get("episode_id", index))
        validated.append(task)

    return validated


def route_length(waypoints):
    points = np.asarray(waypoints, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def make_eval_env(bc_checkpoint):
    return BCResidualNavigationEnvV2(
        checkpoint_path=bc_checkpoint,
        delta_v_max=0.05,
        delta_w_max=0.4,
        lookahead_distance_m=0.5,
    )


def evaluate_controller(
    vec_env,
    tasks,
    controller_name,
    model,
    max_steps,
):
    """Evaluate one controller; model=None means BC-only (zero residual)."""
    rows = []

    for task_index, task in enumerate(tasks, start=1):
        task_id = task.get("task_id", task_index)

        # Set the fixed task for the next vector-environment reset.
        vec_env.venv.set_options({"task": task})
        observation = vec_env.reset()

        previous_position = np.asarray(task["start_xy"], dtype=np.float64)
        actual_distance = 0.0
        episode_return = 0.0
        residual_values = []
        correction_values = []
        path_deviations = []
        final_info = {}
        steps = 0
        done = False
        started = time.perf_counter()

        for step in range(1, max_steps + 1):
            if model is None:
                action = np.zeros((1, 2), dtype=np.float32)
            else:
                action, _ = model.predict(
                    observation,
                    deterministic=True,
                )

            observation, rewards, dones, infos = vec_env.step(action)

            steps = step
            episode_return += float(rewards[0])
            info = infos[0]
            final_info = info

            position_value = info.get("position_xy")
            if position_value is not None:
                position = np.asarray(position_value, dtype=np.float64)
                if position.shape == (2,) and np.isfinite(position).all():
                    actual_distance += float(
                        np.linalg.norm(position - previous_position)
                    )
                    previous_position = position

            residual = info.get("residual_action", [0.0, 0.0])
            correction = info.get("residual_correction", [0.0, 0.0])
            residual_values.append(np.asarray(residual, dtype=np.float64))
            correction_values.append(np.asarray(correction, dtype=np.float64))

            deviation = info.get("path_deviation_m")
            if deviation is not None and np.isfinite(float(deviation)):
                path_deviations.append(float(deviation))

            if bool(dones[0]):
                done = True
                break

        elapsed = time.perf_counter() - started
        success = bool(final_info.get("success", False))
        collision = bool(final_info.get("collision", False))
        timeout = not success and not collision

        residual_array = (
            np.asarray(residual_values, dtype=np.float64)
            if residual_values
            else np.zeros((0, 2), dtype=np.float64)
        )
        correction_array = (
            np.asarray(correction_values, dtype=np.float64)
            if correction_values
            else np.zeros((0, 2), dtype=np.float64)
        )

        row = {
            "controller": controller_name,
            "task_id": task_id,
            "success": success,
            "collision": collision,
            "timeout": timeout,
            "steps": steps,
            "path_length_m": route_length(task["waypoints_world"]),
            "actual_distance_m": actual_distance,
            "final_goal_error_m": final_info.get("goal_distance_m"),
            "final_speed_mps": final_info.get("linear_speed_mps"),
            "final_angular_speed_radps": final_info.get("angular_speed_radps"),
            "elapsed_time_s": elapsed,
            "episode_return": episode_return,
            "mean_abs_residual_v": (
                float(np.mean(np.abs(residual_array[:, 0])))
                if len(residual_array) else 0.0
            ),
            "mean_abs_residual_w": (
                float(np.mean(np.abs(residual_array[:, 1])))
                if len(residual_array) else 0.0
            ),
            "mean_abs_delta_v_mps": (
                float(np.mean(np.abs(correction_array[:, 0])))
                if len(correction_array) else 0.0
            ),
            "mean_abs_delta_w_radps": (
                float(np.mean(np.abs(correction_array[:, 1])))
                if len(correction_array) else 0.0
            ),
            "residual_active_fraction": (
                float(np.mean(np.linalg.norm(residual_array, axis=1) > 0.05))
                if len(residual_array) else 0.0
            ),
            "mean_path_deviation_m": (
                float(mean(path_deviations)) if path_deviations else None
            ),
            "failure_reason": (
                "SUCCESS" if success
                else "COLLISION" if collision
                else "TIMEOUT" if timeout
                else "UNKNOWN"
            ),
            "terminated_or_truncated": done,
        }
        rows.append(row)

        print(
            f"[{controller_name}] Task {task_index:02d}/{len(tasks)} | "
            f"success={success} | collision={collision} | "
            f"steps={steps} | error={row['final_goal_error_m']}"
        )

    return rows


def summarize(rows):
    total = len(rows)
    successful = sum(bool(row["success"]) for row in rows)
    collisions = sum(bool(row["collision"]) for row in rows)
    timeouts = sum(bool(row["timeout"]) for row in rows)

    def avg(key):
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None and np.isfinite(float(row[key]))
        ]
        return float(mean(values)) if values else None

    return {
        "total_tasks": total,
        "successful_tasks": successful,
        "failed_tasks": total - successful,
        "collisions": collisions,
        "timeouts": timeouts,
        "success_rate": successful / total if total else 0.0,
        "collision_rate": collisions / total if total else 0.0,
        "average_steps": avg("steps"),
        "average_actual_distance_m": avg("actual_distance_m"),
        "average_final_error_m": avg("final_goal_error_m"),
        "average_elapsed_time_s": avg("elapsed_time_s"),
        "average_reward": avg("episode_return"),
        "average_mean_abs_delta_v_mps": avg("mean_abs_delta_v_mps"),
        "average_mean_abs_delta_w_radps": avg("mean_abs_delta_w_radps"),
        "average_residual_active_fraction": avg("residual_active_fraction"),
        "average_path_deviation_m": avg("mean_path_deviation_m"),
    }


def make_paired_rows(bc_rows, ppo_rows):
    bc_by_id = {str(row["task_id"]): row for row in bc_rows}
    ppo_by_id = {str(row["task_id"]): row for row in ppo_rows}
    paired = []

    for task_id in bc_by_id:
        if task_id not in ppo_by_id:
            continue
        bc = bc_by_id[task_id]
        ppo = ppo_by_id[task_id]

        bc_error = bc.get("final_goal_error_m")
        ppo_error = ppo.get("final_goal_error_m")

        paired.append({
            "task_id": bc["task_id"],
            "bc_success": bc["success"],
            "ppo_v2_success": ppo["success"],
            "bc_collision": bc["collision"],
            "ppo_v2_collision": ppo["collision"],
            "bc_steps": bc["steps"],
            "ppo_v2_steps": ppo["steps"],
            "steps_delta_ppo_minus_bc": ppo["steps"] - bc["steps"],
            "bc_final_error_m": bc_error,
            "ppo_v2_final_error_m": ppo_error,
            "final_error_delta_ppo_minus_bc_m": (
                float(ppo_error) - float(bc_error)
                if bc_error is not None and ppo_error is not None
                else None
            ),
            "bc_actual_distance_m": bc["actual_distance_m"],
            "ppo_v2_actual_distance_m": ppo["actual_distance_m"],
            "actual_distance_delta_ppo_minus_bc_m": (
                ppo["actual_distance_m"] - bc["actual_distance_m"]
            ),
        })

    return paired


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    for path in (
        args.tasks,
        args.model,
        args.vecnormalize,
        args.bc_checkpoint,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")

    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(args.tasks.read_text(encoding="utf-8"))
    tasks = extract_tasks(payload)

    print("========== BC vs BC-PPO V2 BENCHMARK ==========")
    print("Tasks:", len(tasks))
    print("Model:", args.model)
    print("VecNormalize:", args.vecnormalize)
    print("Benchmark:", args.tasks)
    print("Maximum steps per task:", args.max_steps)

    set_random_seed(args.seed)

    base_vec_env = DummyVecEnv([
        lambda: make_eval_env(args.bc_checkpoint)
    ])
    vec_env = VecNormalize.load(str(args.vecnormalize), base_vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    try:
        model = PPO.load(
            str(args.model),
            env=vec_env,
            device="cpu",
        )

        # Both controllers use exactly the same held-out tasks and V2
        # environment. The BC-only baseline sends [0, 0] residuals.
        bc_rows = evaluate_controller(
            vec_env=vec_env,
            tasks=tasks,
            controller_name="BC only",
            model=None,
            max_steps=args.max_steps,
        )

        ppo_rows = evaluate_controller(
            vec_env=vec_env,
            tasks=tasks,
            controller_name="BC + PPO V2",
            model=model,
            max_steps=args.max_steps,
        )

    finally:
        vec_env.close()

    all_rows = bc_rows + ppo_rows
    paired_rows = make_paired_rows(bc_rows, ppo_rows)

    summary = {
        "experiment": "BC vs BC-PPO V2 fixed benchmark",
        "benchmark_file": str(args.tasks),
        "task_count": len(tasks),
        "max_steps_per_task": args.max_steps,
        "seed": args.seed,
        "model": str(args.model),
        "vecnormalize": str(args.vecnormalize),
        "controllers": {
            "BC only": summarize(bc_rows),
            "BC + PPO V2": summarize(ppo_rows),
        },
        "paired_comparison": {
            "both_successful": sum(
                row["bc_success"] and row["ppo_v2_success"]
                for row in paired_rows
            ),
            "bc_only_successful": sum(
                row["bc_success"] and not row["ppo_v2_success"]
                for row in paired_rows
            ),
            "ppo_v2_only_successful": sum(
                row["ppo_v2_success"] and not row["bc_success"]
                for row in paired_rows
            ),
            "both_failed": sum(
                not row["bc_success"] and not row["ppo_v2_success"]
                for row in paired_rows
            ),
            "mean_final_error_delta_ppo_minus_bc_m": (
                float(mean([
                    row["final_error_delta_ppo_minus_bc_m"]
                    for row in paired_rows
                    if row["final_error_delta_ppo_minus_bc_m"] is not None
                ]))
                if any(
                    row["final_error_delta_ppo_minus_bc_m"] is not None
                    for row in paired_rows
                )
                else None
            ),
        },
    }

    results_path = (
        args.output_dir / "bc_residual_ppo_v2_benchmark_results.csv"
    )
    paired_path = (
        args.output_dir / "bc_vs_ppo_v2_paired_comparison.csv"
    )
    summary_path = (
        args.output_dir / "bc_residual_ppo_v2_summary.json"
    )

    write_csv(results_path, all_rows)
    write_csv(paired_path, paired_rows)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n========== BC vs BC-PPO V2 SUMMARY ==========")
    for controller, metrics in summary["controllers"].items():
        print(f"\n{controller}")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    print("\nPaired comparison:")
    print(json.dumps(
        summary["paired_comparison"],
        ensure_ascii=False,
        indent=2,
    ))

    print("\nSaved:", results_path)
    print("Saved:", paired_path)
    print("Saved:", summary_path)
    print("\nBC-PPO V2 BENCHMARK EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
