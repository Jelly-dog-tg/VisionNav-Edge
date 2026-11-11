"""P8.4: Evaluate ACT on frozen P5/P6 PyBullet benchmark tasks.

From project root:
  python experiments/p8_evaluate_act.py --max-tasks 3
  python experiments/p8_evaluate_act.py --max-tasks 0

DIRECT PyBullet, identical RobotNavigationEnv / benchmark goal and collisions.
This evaluates the ACT policy alone, not BC+PPO or VLM active-search.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionnav.learning.act_controller import ACTChunkController
from visionnav.learning.rl_navigation_env import RobotNavigationEnv

# Some project versions do not include the P5 benchmark file. Prefer the
# already saved V3 validation task set when it exists; do not invent tasks.
BENCHMARK_CANDIDATES = [
    ROOT / "logs/p6_evaluation/benchmark_tasks_v3_validation.json",
    ROOT / "logs/p5_evaluation/benchmark_tasks.json",
    ROOT / "logs/p6_evaluation/benchmark_tasks.json",
]
DEFAULT_MODEL = ROOT / "models/act/act_best.pth"
LOG_ROOT = ROOT / "logs/p8_act_evaluation"


def resolve_benchmark(explicit_path=None):
    """Resolve a real, existing benchmark without modifying the task definitions."""
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(
                f"Specified benchmark file not found: {path}"
            )
        load_tasks(path)  # Fail early for malformed data.
        return path

    candidates = list(BENCHMARK_CANDIDATES)
    # Support other historical naming conventions with a deterministic order.
    log_dir = ROOT / "logs"
    if log_dir.is_dir():
        candidates.extend(sorted(log_dir.rglob("*benchmark*tasks*.json")))
        candidates.extend(sorted(log_dir.rglob("*tasks*validation*.json")))

    seen = set()
    for path in candidates:
        path = Path(path)
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            continue
        try:
            load_tasks(path)
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        return path

    raise FileNotFoundError(
        "No valid saved benchmark task file found under logs/. "
        "Use --benchmark PATH to specify an existing task JSON. "
        "The evaluator will not fabricate a new benchmark."
    )


def load_tasks(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        tasks = next((data[k] for k in ("tasks", "benchmark_tasks", "items", "episodes")
                      if isinstance(data.get(k), list)), None)
    else:
        tasks = data
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Benchmark JSON must contain a nonempty task list")
    required = {"start_xy", "goal_xy", "waypoints_world", "start_yaw_rad"}
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or not required.issubset(task):
            raise ValueError(f"Benchmark task {index} missing required fields")
    return tasks


def run_one(env, controller, task, seed, max_steps, task_index):
    controller.reset()
    observation, initial = env.reset(seed=seed, options={"task": task})
    pose = np.asarray(env.robot.get_pose()[:2], dtype=np.float64)
    total_distance = 0.0
    total_reward = 0.0
    info = {}
    terminated = truncated = False
    t0 = time.perf_counter()
    for step in range(max_steps):
        action, diagnostics = controller.next_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        new_pose = np.asarray(env.robot.get_pose()[:2], dtype=np.float64)
        total_distance += float(np.linalg.norm(new_pose - pose))
        pose = new_pose
        if terminated or truncated:
            break
    elapsed = time.perf_counter() - t0
    success = bool(info.get("success", False)) if (terminated or truncated) else False
    collision = bool(info.get("collision", False))
    return {
        "task_index": task_index,
        "task_success": success,
        "collision": collision,
        "timeout": not success and not collision,
        "steps": int(step + 1),
        "goal_distance_m": float(info.get("goal_distance_m", float("nan"))),
        "actual_path_length_m": total_distance,
        "total_reward": total_reward,
        "elapsed_wall_s": elapsed,
        "replans": int(controller.replans),
        "action_clip_count": int(controller.clipped_commands),
        "warm_start_padding_steps": min(controller.history_len - 1, step + 1),
        "benchmark_goal_xy": initial["goal_xy"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=None,
                        help="Use an explicit saved benchmark JSON; otherwise auto-discover")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-tasks", type=int, default=3,
                        help="Number of tasks (0 = all)")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.max_tasks < 0 or args.start_index < 0 or args.max_steps <= 0:
        parser.error("Invalid benchmark run limits")
    benchmark_path = resolve_benchmark(args.benchmark)
    tasks = load_tasks(benchmark_path)
    print(f"Benchmark file: {benchmark_path}")
    if "validation" in benchmark_path.name.lower():
        print("NOTICE: These are validation tasks, NOT a held-out final test set.")
    end = len(tasks) if args.max_tasks == 0 else min(len(tasks), args.start_index + args.max_tasks)
    selection = list(enumerate(tasks))[args.start_index:end]
    if not selection:
        parser.error("No benchmark tasks selected")

    controller = ACTChunkController(args.model, device=args.device)
    log_dir = LOG_ROOT / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S_%f")
    log_dir.mkdir(parents=True, exist_ok=False)
    print(f"P8.4 ACT | {len(selection)} PyBullet benchmark tasks | model={args.model}")
    results = []
    env = RobotNavigationEnv()
    try:
        for index, task in selection:
            try:
                result = run_one(env, controller, task, args.seed, args.max_steps, index)
            except Exception as exc:
                # Preserve evidence for a technical runtime error; do not
                # mislabel it as a normal navigation failure.
                result = {"task_index": index, "runtime_error": repr(exc),
                          "task_success": False, "collision": None}
            results.append(result)
            (log_dir / f"task_{index:03d}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"task={index:02d} success={result.get('task_success')} "
                  f"collision={result.get('collision')} "
                  f"steps={result.get('steps', 'ERR')} "
                  f"error={result.get('runtime_error', 'none')}")
    finally:
        env.close()

    completed = [x for x in results if "runtime_error" not in x]
    summary = {
        "stage": "P8.4",
        "controller": "ACT standalone, deterministic z=0, action chunking 16/execute 4",
        "note": "Real PyBullet DIRECT closed-loop, NOT VLM/semantic navigation",
        "benchmark": str(benchmark_path),
        "benchmark_role": (
            "development_validation" if "validation" in benchmark_path.name.lower()
            else "existing_fixed_benchmark"
        ),
        "model": str(args.model),
        "seed": args.seed,
        "task_count_requested": len(selection),
        "task_count_completed": len(completed),
        "runtime_error_count": len(results) - len(completed),
        "successful_tasks": sum(bool(x["task_success"]) for x in completed),
        "collisions": sum(bool(x["collision"]) for x in completed),
        "timeouts": sum(bool(x["timeout"]) for x in completed),
        "success_rate": (
            sum(bool(x["task_success"]) for x in completed) / len(completed)
            if completed else None
        ),
        "results": results,
    }
    output = log_dir / "evaluation_summary.json"
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== P8.4 ACT EVALUATION ===")
    print("Success:", summary["successful_tasks"], "/", len(completed))
    print("Collisions:", summary["collisions"])
    print("Runtime errors:", summary["runtime_error_count"])
    print("Report:", output)


if __name__ == "__main__":
    main()
