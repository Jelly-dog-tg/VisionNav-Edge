"""Evaluate a trained BC-PPO V3 checkpoint against BC and PPO V2.

Run from the project root:
    python experiments/40_evaluate_bc_residual_ppo_v3.py --seed 42

Default benchmark:
    logs/p6_evaluation/benchmark_tasks_v3_validation.json

Controllers:
    1) BC Only: zero PPO residual, residual scale 0.0
    2) BC + PPO V2: V2 checkpoint, residual scale 0.25
    3) BC + PPO V3: V3 checkpoint, residual scale 0.25

Each PPO model is evaluated with its own matching VecNormalize statistics.
Residual scaling is applied exactly once by ResidualScaleWrapper.
This script performs inference only; it does not train or modify checkpoints.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionnav.learning.bc_residual_navigation_env_v2 import (  # noqa: E402
    BCResidualNavigationEnvV2,
)
from visionnav.learning.residual_scale_wrapper import (  # noqa: E402
    ResidualScaleWrapper,
)


SEEDS = (42, 123, 2026)
DEFAULT_MAX_STEPS = 2500
RESIDUAL_SCALE = 0.25
DELTA_V_MAX = 0.05
DELTA_W_MAX = 0.4
LOOKAHEAD_DISTANCE_M = 0.5

BC_CHECKPOINT = ROOT / "models" / "behavior_cloning" / "bc_best.pth"
DEFAULT_TASKS = (
    ROOT / "logs" / "p6_evaluation" / "benchmark_tasks_v3_validation.json"
)

V2_MODEL_DIRS = {
    42: ROOT / "models" / "bc_residual_ppo_v2",
    123: ROOT / "models" / "bc_residual_ppo_v2_seed_123",
    2026: ROOT / "models" / "bc_residual_ppo_v2_seed_2026",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare BC Only, BC-PPO V2 and trained BC-PPO V3."
    )
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--bc-checkpoint", type=Path, default=BC_CHECKPOINT)
    parser.add_argument("--v2-model", type=Path, default=None)
    parser.add_argument("--v2-vecnormalize", type=Path, default=None)
    parser.add_argument("--v3-model", type=Path, default=None)
    parser.add_argument("--v3-vecnormalize", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(args):
    v2_dir = V2_MODEL_DIRS[args.seed]
    v3_dir = ROOT / "models" / f"bc_residual_ppo_v3_seed_{args.seed}"

    paths = {
        "bc": args.bc_checkpoint,
        "v2_model": args.v2_model or (
            v2_dir / "bc_residual_ppo_v2_final.zip"
        ),
        "v2_vecnormalize": args.v2_vecnormalize or (
            v2_dir / "vecnormalize.pkl"
        ),
        "v3_model": args.v3_model or (
            v3_dir / "bc_residual_ppo_v3_final.zip"
        ),
        "v3_vecnormalize": args.v3_vecnormalize or (
            v3_dir / "vecnormalize.pkl"
        ),
        "tasks": args.tasks,
    }
    return paths


def extract_tasks(payload):
    """Read the benchmark JSON layouts used by this project."""
    if isinstance(payload, list):
        source_tasks = payload
    elif isinstance(payload, dict):
        source_tasks = None
        for key in ("tasks", "benchmark_tasks", "items", "episodes"):
            value = payload.get(key)
            if isinstance(value, list):
                source_tasks = value
                break
        if source_tasks is None:
            raise ValueError(
                "Benchmark JSON must contain a task list under "
                "'tasks', 'benchmark_tasks', 'items' or 'episodes'."
            )
    else:
        raise ValueError("Unsupported benchmark JSON structure.")

    if not source_tasks:
        raise ValueError("Benchmark task list is empty.")

    required = (
        "start_xy",
        "goal_xy",
        "waypoints_world",
        "start_yaw_rad",
    )
    tasks = []
    seen_ids = set()

    for index, source in enumerate(source_tasks, start=1):
        if not isinstance(source, dict):
            raise ValueError(f"Task {index} must be a JSON object.")

        missing = [key for key in required if key not in source]
        if missing:
            raise ValueError(f"Task {index} missing fields: {missing}")

        task = dict(source)
        start = np.asarray(task["start_xy"], dtype=np.float64)
        goal = np.asarray(task["goal_xy"], dtype=np.float64)
        waypoints = np.asarray(task["waypoints_world"], dtype=np.float64)
        yaw = float(task["start_yaw_rad"])

        if start.shape != (2,) or goal.shape != (2,):
            raise ValueError(f"Task {index}: start_xy/goal_xy must have shape (2,).")
        if waypoints.ndim != 2 or waypoints.shape[1] != 2 or len(waypoints) < 1:
            raise ValueError(
                f"Task {index}: waypoints_world must have shape (N, 2), N >= 1."
            )
        if not (
            np.isfinite(start).all()
            and np.isfinite(goal).all()
            and np.isfinite(waypoints).all()
            and np.isfinite(yaw)
        ):
            raise ValueError(f"Task {index} contains NaN or Inf.")

        task_id = int(task.get("task_id", task.get("episode_id", index)))
        if task_id in seen_ids:
            raise ValueError(f"Duplicate task_id: {task_id}")
        seen_ids.add(task_id)

        task["task_id"] = task_id
        task["start_xy"] = start.tolist()
        task["goal_xy"] = goal.tolist()
        task["waypoints_world"] = waypoints.tolist()
        task["start_yaw_rad"] = yaw
        tasks.append(task)

    return tasks


def route_length(waypoints):
    points = np.asarray(waypoints, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def make_vector_env(seed, residual_scale, vecnormalize_path):
    """Create V2 -> residual scaling -> Monitor -> VecNormalize."""
    def factory():
        raw_env = BCResidualNavigationEnvV2(
            checkpoint_path=BC_CHECKPOINT,
            delta_v_max=DELTA_V_MAX,
            delta_w_max=DELTA_W_MAX,
            lookahead_distance_m=LOOKAHEAD_DISTANCE_M,
        )
        scaled_env = ResidualScaleWrapper(
            raw_env,
            residual_scale=residual_scale,
        )
        return Monitor(
            scaled_env,
            info_keywords=("success", "collision"),
        )

    base_env = DummyVecEnv([factory])
    base_env.seed(seed)

    try:
        vec_env = VecNormalize.load(str(vecnormalize_path), base_env)
        vec_env.training = False
        vec_env.norm_reward = False
        return vec_env
    except Exception:
        base_env.close()
        raise


def evaluate_controller(
    controller_name,
    model_path,
    vecnormalize_path,
    residual_scale,
    tasks,
    seed,
    max_steps,
):
    """Evaluate one controller with its own matching normalization statistics."""
    set_random_seed(seed)
    vec_env = make_vector_env(seed, residual_scale, vecnormalize_path)
    model = None

    try:
        if model_path is not None:
            model = PPO.load(str(model_path), env=vec_env, device="cpu")
            if model.observation_space.shape != (44,):
                raise RuntimeError(
                    f"{controller_name}: expected PPO observation (44,), "
                    f"got {model.observation_space.shape}"
                )
            if model.action_space.shape != (2,):
                raise RuntimeError(
                    f"{controller_name}: expected PPO action (2,), "
                    f"got {model.action_space.shape}"
                )

        rows = []

        for task_index, task in enumerate(tasks, start=1):
            vec_env.venv.set_options({"task": task})
            observation = vec_env.reset()

            previous_position = np.asarray(
                task["start_xy"], dtype=np.float64
            )
            actual_distance = 0.0
            episode_return = 0.0
            path_deviations = []
            raw_actions = []
            scaled_actions = []
            corrections = []
            residual_active = []
            clipped_actions = []

            final_info = {}
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

                action = np.asarray(action, dtype=np.float32)
                if action.shape != (1, 2) or not np.isfinite(action).all():
                    raise ValueError(
                        f"{controller_name}, task {task['task_id']}: "
                        "invalid PPO action; stopping evaluation."
                    )

                observation, rewards, dones, infos = vec_env.step(action)
                info = final_info = infos[0]

                if (
                    observation.shape != (1, 44)
                    or not np.isfinite(observation).all()
                    or not np.isfinite(rewards).all()
                ):
                    raise RuntimeError(
                        f"{controller_name}, task {task['task_id']}: "
                        "non-finite observation or reward."
                    )

                diagnostic = info.get("v3_residual")
                if diagnostic is None:
                    raise KeyError(
                        "Missing info['v3_residual']; check that "
                        "ResidualScaleWrapper is installed."
                    )

                expected_scaled = (
                    np.clip(action[0], -1.0, 1.0) * residual_scale
                ).astype(np.float32)

                actual_scaled = np.asarray(
                    info.get("residual_action"), dtype=np.float32
                )
                if actual_scaled.shape != (2,) or not np.allclose(
                    actual_scaled, expected_scaled, atol=1e-6
                ):
                    raise RuntimeError(
                        "Residual scaling mismatch; possible double scaling."
                    )

                delta_v = diagnostic.get("delta_v_mps")
                delta_w = diagnostic.get("delta_w_radps")
                if delta_v is None or delta_w is None:
                    raise RuntimeError(
                        "V2 did not report physical residual corrections."
                    )

                correction = np.asarray(
                    [delta_v, delta_w], dtype=np.float64
                )
                expected_correction = (
                    expected_scaled.astype(np.float64)
                    * np.asarray([DELTA_V_MAX, DELTA_W_MAX])
                )
                if not np.allclose(
                    correction, expected_correction, atol=1e-6
                ):
                    raise RuntimeError(
                        "Physical residual correction does not match "
                        "the configured V2 limits."
                    )

                episode_return += float(rewards[0])

                position = np.asarray(
                    info["position_xy"], dtype=np.float64
                )
                actual_distance += float(
                    np.linalg.norm(position - previous_position)
                )
                previous_position = position

                deviation = info.get("path_deviation_m")
                if deviation is not None:
                    path_deviations.append(float(deviation))

                raw_actions.append(action[0].copy())
                scaled_actions.append(
                    np.asarray(diagnostic["scaled_action"], dtype=np.float64)
                )
                corrections.append(correction)
                residual_active.append(
                    bool(diagnostic["residual_active"])
                )
                clipped_actions.append(
                    bool(diagnostic["action_clipped"])
                )

                if bool(dones[0]):
                    done = True
                    break

            elapsed = time.perf_counter() - started
            success = bool(final_info.get("success", False))
            collision = bool(final_info.get("collision", False))
            timeout = not success and not collision

            raw_array = np.asarray(raw_actions, dtype=np.float64)
            scaled_array = np.asarray(scaled_actions, dtype=np.float64)
            correction_array = np.asarray(corrections, dtype=np.float64)

            row = {
                "controller": controller_name,
                "seed": seed,
                "task_id": task["task_id"],
                "success": success,
                "collision": collision,
                "timeout": timeout,
                "steps": step,
                "path_length_m": route_length(task["waypoints_world"]),
                "actual_distance_m": actual_distance,
                "final_goal_error_m": float(final_info["goal_distance_m"]),
                "final_speed_mps": float(final_info["linear_speed_mps"]),
                "final_angular_speed_radps": float(
                    final_info["angular_speed_radps"]
                ),
                "elapsed_time_s": elapsed,
                "episode_return": episode_return,
                "mean_path_deviation_m": (
                    float(np.mean(path_deviations))
                    if path_deviations else None
                ),
                "mean_abs_policy_action_v": float(
                    np.abs(raw_array[:, 0]).mean()
                ),
                "mean_abs_policy_action_w": float(
                    np.abs(raw_array[:, 1]).mean()
                ),
                "mean_abs_scaled_residual_v": float(
                    np.abs(scaled_array[:, 0]).mean()
                ),
                "mean_abs_scaled_residual_w": float(
                    np.abs(scaled_array[:, 1]).mean()
                ),
                "mean_abs_delta_v_mps": float(
                    np.abs(correction_array[:, 0]).mean()
                ),
                "mean_abs_delta_w_radps": float(
                    np.abs(correction_array[:, 1]).mean()
                ),
                "max_abs_delta_v_mps": float(
                    np.abs(correction_array[:, 0]).max()
                ),
                "max_abs_delta_w_radps": float(
                    np.abs(correction_array[:, 1]).max()
                ),
                "residual_active_fraction": float(
                    np.mean(residual_active)
                ),
                "action_clip_fraction": float(
                    np.mean(clipped_actions)
                ),
                "invalid_action_count": 0,
                "failure_reason": (
                    "SUCCESS" if success
                    else "COLLISION" if collision
                    else "TIMEOUT"
                ),
                "terminated_or_truncated": done,
            }
            rows.append(row)

            print(
                f"[{controller_name}] {task_index:02d}/{len(tasks)} "
                f"task={task['task_id']} success={success} "
                f"collision={collision} steps={step} "
                f"error={row['final_goal_error_m']:.4f}m",
                flush=True,
            )

        return rows

    finally:
        vec_env.close()


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty evaluation.")

    total = len(rows)
    successes = sum(row["success"] for row in rows)
    collisions = sum(row["collision"] for row in rows)
    timeouts = sum(row["timeout"] for row in rows)

    def avg(key):
        values = [
            float(row[key])
            for row in rows
            if row.get(key) is not None
            and np.isfinite(float(row[key]))
        ]
        return float(mean(values)) if values else None

    return {
        "total_tasks": total,
        "successful_tasks": successes,
        "failed_tasks": total - successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "success_rate": successes / total,
        "collision_rate": collisions / total,
        "timeout_rate": timeouts / total,
        "average_steps": avg("steps"),
        "average_path_length_m": avg("path_length_m"),
        "average_actual_distance_m": avg("actual_distance_m"),
        "average_final_error_m": avg("final_goal_error_m"),
        "average_elapsed_time_s": avg("elapsed_time_s"),
        "average_reward": avg("episode_return"),
        "average_path_deviation_m": avg("mean_path_deviation_m"),
        "average_mean_abs_delta_v_mps": avg("mean_abs_delta_v_mps"),
        "average_mean_abs_delta_w_radps": avg("mean_abs_delta_w_radps"),
        "average_residual_active_fraction": avg(
            "residual_active_fraction"
        ),
        "average_action_clip_fraction": avg("action_clip_fraction"),
        "invalid_action_count": sum(
            row["invalid_action_count"] for row in rows
        ),
    }


def paired_compare(rows_by_controller):
    bc = {row["task_id"]: row for row in rows_by_controller["BC Only"]}
    v2 = {row["task_id"]: row for row in rows_by_controller["BC + PPO V2"]}
    v3 = {row["task_id"]: row for row in rows_by_controller["BC + PPO V3"]}

    paired = []
    for task_id in sorted(bc):
        b, p2, p3 = bc[task_id], v2[task_id], v3[task_id]
        row = {
            "task_id": task_id,
            "bc_success": b["success"],
            "v2_success": p2["success"],
            "v3_success": p3["success"],
            "bc_collision": b["collision"],
            "v2_collision": p2["collision"],
            "v3_collision": p3["collision"],
            "bc_steps": b["steps"],
            "v2_steps": p2["steps"],
            "v3_steps": p3["steps"],
            "bc_final_error_m": b["final_goal_error_m"],
            "v2_final_error_m": p2["final_goal_error_m"],
            "v3_final_error_m": p3["final_goal_error_m"],
            "v3_minus_v2_steps": p3["steps"] - p2["steps"],
            "v3_minus_v2_final_error_m": (
                p3["final_goal_error_m"] - p2["final_goal_error_m"]
            ),
            "v3_minus_bc_final_error_m": (
                p3["final_goal_error_m"] - b["final_goal_error_m"]
            ),
        }
        paired.append(row)

    def transition(a, b):
        return {
            "both_successful": sum(
                a[k]["success"] and b[k]["success"] for k in a
            ),
            "baseline_only_successful": sum(
                a[k]["success"] and not b[k]["success"] for k in a
            ),
            "candidate_only_successful": sum(
                not a[k]["success"] and b[k]["success"] for k in a
            ),
            "both_failed": sum(
                not a[k]["success"] and not b[k]["success"] for k in a
            ),
        }

    return paired, {
        "BC vs PPO V2": transition(bc, v2),
        "BC vs PPO V3": transition(bc, v3),
        "PPO V2 vs PPO V3": transition(v2, v3),
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive.")

    paths = resolve_paths(args)
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Required {label} file not found:\n{path}"
            )

    payload = json.loads(paths["tasks"].read_text(encoding="utf-8"))
    tasks = extract_tasks(payload)

    if args.output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            ROOT / "logs" / "p6_evaluation" / "ppo_v3_evaluation"
            / f"seed_{args.seed}_{stamp}"
        )
    else:
        output_dir = args.output_dir
        if not output_dir.is_absolute():
            output_dir = ROOT / output_dir

    output_dir.mkdir(parents=True, exist_ok=False)

    print("=" * 68)
    print("BC vs PPO V2 vs PPO V3 EVALUATION")
    print("=" * 68)
    print("Seed:", args.seed)
    print("Tasks:", len(tasks))
    print("Maximum steps:", args.max_steps)
    print("Benchmark:", paths["tasks"])
    print("PPO V2 model:", paths["v2_model"])
    print("PPO V3 model:", paths["v3_model"])
    print("Residual scale: 0.25 for PPO V2 and PPO V3")
    print("Inference: deterministic")
    print("Output:", output_dir)

    controllers = [
        (
            "BC Only",
            None,
            paths["v2_vecnormalize"],
            0.0,
        ),
        (
            "BC + PPO V2",
            paths["v2_model"],
            paths["v2_vecnormalize"],
            RESIDUAL_SCALE,
        ),
        (
            "BC + PPO V3",
            paths["v3_model"],
            paths["v3_vecnormalize"],
            RESIDUAL_SCALE,
        ),
    ]

    rows_by_controller = {}
    all_rows = []

    for name, model_path, norm_path, scale in controllers:
        print(f"\n========== EVALUATING {name} ==========", flush=True)
        rows = evaluate_controller(
            controller_name=name,
            model_path=model_path,
            vecnormalize_path=norm_path,
            residual_scale=scale,
            tasks=tasks,
            seed=args.seed,
            max_steps=args.max_steps,
        )
        rows_by_controller[name] = rows
        all_rows.extend(rows)

    paired_rows, paired_summary = paired_compare(rows_by_controller)
    summary = {
        "experiment": "BC vs PPO V2 vs PPO V3",
        "seed": args.seed,
        "task_file": str(paths["tasks"]),
        "task_count": len(tasks),
        "max_steps": args.max_steps,
        "deterministic_inference": True,
        "residual_scale": RESIDUAL_SCALE,
        "controllers": {
            name: summarize(rows)
            for name, rows in rows_by_controller.items()
        },
        "paired_comparison": paired_summary,
        "model_files": {
            "bc_checkpoint": str(paths["bc"]),
            "v2_model": str(paths["v2_model"]),
            "v2_vecnormalize": str(paths["v2_vecnormalize"]),
            "v3_model": str(paths["v3_model"]),
            "v3_vecnormalize": str(paths["v3_vecnormalize"]),
        },
        "sha256": {
            label: sha256(path)
            for label, path in paths.items()
        },
        "note": (
            "This is an evaluation on the selected benchmark task file. "
            "If V3 training used an online random task sampler without "
            "explicit endpoint exclusion, this 40-task validation set is "
            "not guaranteed to be an untouched final test set."
        ),
    }

    write_csv(output_dir / "ppo_v3_episode_results.csv", all_rows)
    write_csv(output_dir / "ppo_v3_paired_comparison.csv", paired_rows)
    (output_dir / "ppo_v3_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    print("\n========== PPO V3 EVALUATION SUMMARY ==========")
    for name, metrics in summary["controllers"].items():
        print(f"\n{name}")
        for key, value in metrics.items():
            print(f"{key}: {value}")

    print("\n========== PAIRED COMPARISON ==========")
    print(json.dumps(paired_summary, ensure_ascii=False, indent=2))
    print("\nSaved:", output_dir / "ppo_v3_episode_results.csv")
    print("Saved:", output_dir / "ppo_v3_paired_comparison.csv")
    print("Saved:", output_dir / "ppo_v3_summary.json")
    print("\nPPO V3 EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
