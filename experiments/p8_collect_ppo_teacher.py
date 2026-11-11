"""P8.6-A | Collect successful BC+PPO V3 teacher rollouts for ACT distillation.

Physical actions [v m/s, w rad/s] and raw 40D observations are stored.
The frozen teacher is BC+PPO V3. An explicit fixed near-goal slow/brake
supervisor (NOT the teacher network) is used to satisfy stable-stop tests.
No fallback Pure Pursuit, no old checkpoint overwrites, no failed episodes saved.

From project root:
  python experiments/p8_collect_ppo_teacher.py --target-successes 24 --max-attempts 80
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionnav.learning.bc_ppo_v3_controller import BCPPOV3Controller
from visionnav.learning.rl_navigation_env import (
    RobotNavigationEnv, MAX_LINEAR_V, MAX_ANGULAR_W,
)

DEFAULT_DIR = ROOT / "logs" / "p8_ppo_teacher_distillation"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def normalize_command(v: float, w: float) -> np.ndarray:
    """Convert teacher physical commands to existing RobotNavigationEnv interface."""
    return np.array([
        2.0 * v / MAX_LINEAR_V - 1.0,
        w / MAX_ANGULAR_W,
    ], dtype=np.float32)


def physical_command(action: np.ndarray) -> np.ndarray:
    """Exact action executed by RobotNavigationEnv after its [-1,1] clipping."""
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return np.array([
        (float(a[0]) + 1.0) * 0.5 * MAX_LINEAR_V,
        float(a[1]) * MAX_ANGULAR_W,
    ], dtype=np.float32)


def extract_action(teacher, env, obs, quiet=True):
    pose = np.asarray(env.robot.get_pose(), dtype=np.float32)
    lidar = np.asarray(env.robot.get_lidar_scan(), dtype=np.float32)
    waypoint = np.asarray(env.waypoints[env.waypoint_index], dtype=np.float32)
    # Sync the teacher's waypoint cursor with the public navigation environment.
    teacher.waypoint_index = int(env.waypoint_index)
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            v, w, _ = teacher.compute_control(pose, waypoint, lidar)
    else:
        v, w, _ = teacher.compute_control(pose, waypoint, lidar)
    raw_action = np.asarray([v, w], dtype=np.float32)
    if raw_action.shape != (2,) or not np.isfinite(raw_action).all():
        raise RuntimeError("Teacher generated invalid action")
    if np.asarray(obs).shape != (40,) or not np.isfinite(obs).all():
        raise RuntimeError("Invalid 40D student observation")
    return raw_action


def run_episode(env, teacher, task_seed, max_steps, brake_distance, release_distance, slow_distance,
                min_length=23, quiet_teacher=True):
    obs, init_info = env.reset(seed=task_seed)
    if teacher is None:
        teacher = BCPPOV3Controller(
            robot=env.robot,
            waypoints=env.waypoints,
            final_goal=env.goal_xy,
            seed=42,
            residual_scale=0.25,
        )
    else:
        teacher.robot = env.robot
        teacher.set_path(env.waypoints, env.goal_xy)

    original_pose = np.asarray(env.robot.get_pose(), dtype=np.float64)
    task = {
        "start_xy": [float(x) for x in original_pose[:2]],
        "start_yaw_rad": float(original_pose[2]),
        "goal_xy": np.asarray(env.goal_xy, dtype=float).tolist(),
        "waypoints_world": np.asarray(env.waypoints, dtype=float).tolist(),
    }
    data_obs = []
    data_actions = []
    data_mode = []
    collisions = 0
    teacher_steps = brake_steps = slow_steps = 0
    success = False
    brake_active = False
    info = {}

    for step in range(max_steps):
        teacher_physical = extract_action(teacher, env, obs, quiet=quiet_teacher)
        action = normalize_command(*teacher_physical.tolist())
        mode = 0  # BC+PPO V3 policy
        dist = float(np.linalg.norm(
            np.asarray(env.robot.get_pose()[:2], dtype=np.float64)
            - np.asarray(env.goal_xy, dtype=np.float64)
        ))
        # Hysteresis matches the P8.4 terminal-stop evaluation supervisor.
        if brake_active and dist >= release_distance:
            brake_active = False
        if not brake_active and dist <= brake_distance:
            brake_active = True
        # Supervisor is separately labeled. It is NOT part of the learned teacher.
        if brake_active:
            action = np.array([-1.0, 0.0], dtype=np.float32)
            mode = 2
            brake_steps += 1
        elif dist <= slow_distance:
            cap = min(MAX_LINEAR_V, 0.04 + 0.30 * dist)
            requested_v = (float(action[0]) + 1.0) * .5 * MAX_LINEAR_V
            if requested_v > cap:
                action[0] = 2.0 * cap / MAX_LINEAR_V - 1.0
                mode = 1
                slow_steps += 1
        if mode == 0:
            teacher_steps += 1
        executed = physical_command(action)
        data_obs.append(np.asarray(obs, dtype=np.float32).copy())
        data_actions.append(executed)
        data_mode.append(mode)
        obs, reward, terminated, truncated, info = env.step(action)
        if info.get("collision", False):
            collisions += 1
        if terminated or truncated:
            success = bool(info.get("success", False))
            break

    count = len(data_obs)
    accepted = success and collisions == 0 and count >= min_length
    episode_stats = {
        "seed": int(task_seed),
        "success": bool(success),
        "accepted": bool(accepted),
        "collision": bool(collisions > 0),
        "steps": int(count),
        "teacher_steps": teacher_steps,
        "terminal_slow_steps": slow_steps,
        "terminal_brake_steps": brake_steps,
        "final_goal_distance_m": float(info.get("goal_distance_m", float("nan"))),
    }
    data = None
    if accepted:
        data = {
            "observations": np.stack(data_obs).astype(np.float32),
            "actions": np.stack(data_actions).astype(np.float32),
            "modes": np.asarray(data_mode, dtype=np.uint8),
        }
    return teacher, data, episode_stats, task


def split_successes(success_ids: list[int], seed: int) -> dict[str, list[int]]:
    if len(success_ids) < 10:
        raise ValueError("Collect at least 10 successful episodes for an episode-level split")
    rng = np.random.default_rng(seed)
    shuffled = [int(x) for x in rng.permutation(success_ids)]
    n = len(shuffled)
    n_test = max(1, int(round(n * .10)))
    n_val = max(1, int(round(n * .10)))
    return {
        "train_episode_ids": sorted(shuffled[:n-n_val-n_test]),
        "val_episode_ids": sorted(shuffled[n-n_val-n_test:n-n_test]),
        "test_episode_ids": sorted(shuffled[n-n_test:]),
    }


def main():
    p = argparse.ArgumentParser(description="P8.6 BC+PPO->ACT distillation teacher collector")
    p.add_argument("--target-successes", type=int, default=24)
    p.add_argument("--max-attempts", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=2500)
    p.add_argument("--base-seed", type=int, default=20261112)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--brake-distance", type=float, default=0.09)
    p.add_argument("--release-distance", type=float, default=0.13)
    p.add_argument("--slow-distance", type=float, default=0.30)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    p.add_argument("--verbose-teacher", action="store_true")
    args = p.parse_args()
    if args.target_successes < 10 or args.max_attempts < args.target_successes:
        p.error("target-successes must be >= 10 and max-attempts >= target-successes")
    if not (0 < args.brake_distance < .12 < args.release_distance < args.slow_distance):
        p.error("Require 0 < brake-distance < 0.12 < release-distance < slow-distance")
    if args.max_steps < 23:
        p.error("max-steps must be >= 23")
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_files = ["teacher_observations.npy", "teacher_actions.npy", "teacher_episode_ids.npy",
                    "teacher_action_modes.npy", "episode_splits.json", "rollout_manifest.json",
                    "collection_summary.json", "heldout_teacher_tasks.json"]
    existing = [x for x in output_files if (output / x).exists()]
    if existing:
        raise FileExistsError(f"Collection output already exists ({existing}); choose a new --output-dir")

    bc = ROOT / "models" / "behavior_cloning" / "bc_best.pth"
    ppo_dir = ROOT / "models" / "bc_residual_ppo_v3_seed_42"
    ppo = ppo_dir / "bc_residual_ppo_v3_final.zip"
    vecnorm = ppo_dir / "vecnormalize.pkl"
    for f in [bc, ppo, vecnorm]:
        if not f.exists():
            raise FileNotFoundError(f"Teacher checkpoint required: {f}")

    records = []
    attempts = []
    obs_chunks, action_chunks, mode_chunks, episode_chunks = [], [], [], []
    teacher = None
    env = RobotNavigationEnv()
    start = time.perf_counter()
    try:
        for attempt in range(args.max_attempts):
            if len(records) >= args.target_successes:
                break
            seed = args.base_seed + attempt
            teacher, data, stat, task = run_episode(
                env, teacher, seed, args.max_steps, args.brake_distance,
                args.release_distance, args.slow_distance, quiet_teacher=not args.verbose_teacher,
            )
            stat["attempt_index"] = attempt
            attempts.append(stat)
            if data is not None:
                eid = attempt
                stat["episode_id"] = eid
                records.append({"episode_id": eid, "task": task, "stats": stat})
                obs_chunks.append(data["observations"])
                action_chunks.append(data["actions"])
                mode_chunks.append(data["modes"])
                episode_chunks.append(np.full(len(data["actions"]), eid, dtype=np.int64))
            print(f"attempt {attempt+1}/{args.max_attempts} seed={seed} "
                  f"success={stat['accepted']} steps={stat['steps']} "
                  f"collected={len(records)}/{args.target_successes}", flush=True)
    finally:
        if teacher is not None:
            teacher.close()
        env.close()

    if len(records) < args.target_successes:
        print(f"WARNING: collected {len(records)}/{args.target_successes} successful trajectories")
    if len(records) < 10:
        raise RuntimeError("Too few successful trajectories. No dataset written. "
                           "Do not distill from predominantly failed teacher rollouts.")
    ids = [r["episode_id"] for r in records]
    splits = split_successes(ids, args.split_seed)
    raw_obs = np.concatenate(obs_chunks, axis=0)
    physical_actions = np.concatenate(action_chunks, axis=0)
    modes = np.concatenate(mode_chunks, axis=0)
    episode_ids = np.concatenate(episode_chunks, axis=0)
    if not (len(raw_obs) == len(physical_actions) == len(modes) == len(episode_ids)):
        raise AssertionError("Rollout data length mismatch")
    if raw_obs.shape[1:] != (40,) or physical_actions.shape[1:] != (2,):
        raise AssertionError("Incorrect distillation observation/action dimensions")
    if not (np.isfinite(raw_obs).all() and np.isfinite(physical_actions).all()):
        raise AssertionError("NaN or Inf in collected teacher data")

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "teacher_observations.npy", raw_obs)
    np.save(output / "teacher_actions.npy", physical_actions)
    np.save(output / "teacher_episode_ids.npy", episode_ids)
    np.save(output / "teacher_action_modes.npy", modes)
    (output / "episode_splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    (output / "rollout_manifest.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    test_ids = set(splits["test_episode_ids"])
    heldout_tasks = [r["task"] | {"task_id": int(r["episode_id"])} for r in records
                     if r["episode_id"] in test_ids]
    (output / "heldout_teacher_tasks.json").write_text(
        json.dumps({"tasks": heldout_tasks}, indent=2), encoding="utf-8")
    summary = {
        "stage": "P8.6-teacher-collection", "created_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_type": "frozen BC+PPO V3, seed=42, residual_scale=0.25",
        "supervisor": "deterministic terminal slow/brake; NOT teacher network",
        "data_semantics": "40D raw observation and 2D physical EXECUTED action",
        "teacher_checkpoints_sha256": {str(x.relative_to(ROOT)): sha256(x) for x in [bc, ppo, vecnorm]},
        "control_dt_s": 10/240, "attempts": len(attempts), "successful_episodes": len(records),
        "rejected_attempts": len(attempts) - len(records), "total_steps": len(raw_obs),
        "teacher_steps": int(np.sum(modes == 0)),
        "terminal_slow_steps": int(np.sum(modes == 1)),
        "terminal_brake_steps": int(np.sum(modes == 2)),
        "episode_splits": splits, "attempts_report": attempts,
        "elapsed_wall_s": round(time.perf_counter()-start, 3), "options": vars(args) | {"output_dir": str(output)},
    }
    (output / "collection_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== P8.6 TEACHER COLLECTION COMPLETE ===")
    print("Successful episodes:", len(records), "/", len(attempts))
    print("Timesteps:", len(raw_obs))
    print("Teacher actions:", summary["teacher_steps"],
          "Terminal supervised:", summary["terminal_slow_steps"] + summary["terminal_brake_steps"])
    print("Split episodes:", {key: len(value) for key, value in splits.items()})
    print("Saved:", output)


if __name__ == "__main__":
    main()
