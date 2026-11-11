
"""P5-7.2: Evaluate the trained BC policy on 20 fixed tasks."""

import csv
import json
import sys
import time
import importlib.util
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diff_drive_robot import DiffDriveRobot
from visionnav.learning import collect_expert as expert
from visionnav.learning.expert_controller import ExpertController


# --------------------------------------------------
# Configuration
# --------------------------------------------------

TASK_FILE = ROOT / "logs/p5_evaluation/benchmark_tasks.json"
OUTPUT_DIR = ROOT / "logs/p5_evaluation"

PHYSICS_DT = 1.0 / 240.0
PHYSICS_STEPS = 10
CONTROL_DT = PHYSICS_DT * PHYSICS_STEPS

INITIAL_SETTLE_STEPS = 120
MAX_CONTROL_STEPS = 2500

GOAL_TOLERANCE = 0.12
SPEED_TOLERANCE = 0.05
REQUIRED_STABLE_CYCLES = 10

MAX_LINEAR_V = 0.2613
MAX_ANGULAR_W = 2.0


# --------------------------------------------------
# Load environment builder
# --------------------------------------------------

def load_environment_builder():

    path = ROOT / "scripts/05_environment_builder.py"

    spec = importlib.util.spec_from_file_location(
        "warehouse_builder",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.EnvironmentBuilder


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def get_speed_components(velocity):

    velocity = np.asarray(velocity, dtype=float).reshape(-1)

    if velocity.size >= 3:
        linear_speed = float(np.linalg.norm(velocity[:2]))
        angular_speed = abs(float(velocity[2]))

    elif velocity.size == 2:
        linear_speed = abs(float(velocity[0]))
        angular_speed = abs(float(velocity[1]))

    else:
        raise ValueError("Unexpected robot velocity format")

    return linear_speed, angular_speed


def calculate_distance(a, b):

    return float(
        np.linalg.norm(
            np.asarray(a, dtype=float)[:2]
            - np.asarray(b, dtype=float)[:2]
        )
    )


# --------------------------------------------------
# Run one BC navigation task
# --------------------------------------------------

def run_task(task, policy, EnvironmentBuilder):

    p.resetSimulation()

    p.setAdditionalSearchPath(
        pybullet_data.getDataPath()
    )

    p.setGravity(0, 0, -9.81)
    p.setTimeStep(PHYSICS_DT)
    p.setRealTimeSimulation(0)

    ground_id = p.loadURDF("plane.urdf")

    EnvironmentBuilder(
        p.getConnectionInfo()["connectionMethod"]
    ).create_warehouse_environment(
        arena_size=10.0
    )

    start_xy = np.asarray(task["start_xy"], dtype=float)
    goal_xy = np.asarray(task["goal_xy"], dtype=float)

    waypoints = np.asarray(
        task["waypoints_world"],
        dtype=float,
    )

    robot = DiffDriveRobot(
        str(ROOT / "urdf/diff_drive_robot.urdf"),
        start_pos=[float(start_xy[0]), float(start_xy[1]), 0.1],
        start_yaw=float(task["start_yaw_rad"]),
    )

    for _ in range(INITIAL_SETTLE_STEPS):
        p.stepSimulation()

    waypoint_index = 0
    stable_cycles = 0

    success = False
    collision = False

    actual_distance = 0.0
    previous_position = np.asarray(
        robot.get_pose()[:2],
        dtype=float,
    )

    start_time = time.perf_counter()

    for step in range(1, MAX_CONTROL_STEPS + 1):

        # 1. Read robot state.
        pose = robot.get_pose()
        velocity = robot.get_velocity()

        position = np.asarray(pose[:2], dtype=float)

        # 2. Advance to the next waypoint.
        while (
            waypoint_index < len(waypoints) - 1
            and calculate_distance(
                position,
                waypoints[waypoint_index],
            ) < 0.12
        ):
            waypoint_index += 1

        waypoint = waypoints[waypoint_index]

        # 3. Read raw LiDAR.
        lidar = np.asarray(
            robot.get_lidar_scan(),
            dtype=np.float32,
        )

        # 4. Construct the training-compatible observation.
        observation = expert.make_observation(
            pose,
            waypoint,
            lidar,
            velocity,
        )

        # PID expert controller predicts the velocity commands.
        control = policy.compute_control(
            pose,
            waypoint,
            lidar,
        )

        v = float(
            np.clip(control[0], 0.0, MAX_LINEAR_V)
        )

        w = float(
            np.clip(control[1], -MAX_ANGULAR_W, MAX_ANGULAR_W)
        )

        # 6. Execute the action.
        robot.set_velocity(v, w)

        for _ in range(PHYSICS_STEPS):

            p.stepSimulation()

            if robot.check_collision():
                collision = True
                break

        if collision:
            break

        # 7. Measure actual movement.
        new_pose = robot.get_pose()
        new_velocity = robot.get_velocity()

        new_position = np.asarray(
            new_pose[:2],
            dtype=float,
        )

        actual_distance += float(
            np.linalg.norm(new_position - previous_position)
        )

        previous_position = new_position.copy()

        goal_error = calculate_distance(
            new_position,
            goal_xy,
        )

        linear_speed, angular_speed = get_speed_components(
            new_velocity
        )

        # 8. Verify stable arrival.
        stable = (
            waypoint_index == len(waypoints) - 1
            and goal_error <= GOAL_TOLERANCE
            and linear_speed <= SPEED_TOLERANCE
            and angular_speed <= SPEED_TOLERANCE
        )

        stable_cycles = (
            stable_cycles + 1 if stable else 0
        )

        if stable_cycles >= REQUIRED_STABLE_CYCLES:
            success = True
            break

    elapsed_time = time.perf_counter() - start_time

    # 9. Stop the robot.
    robot.set_velocity(0.0, 0.0)

    for _ in range(10):
        p.stepSimulation()

    final_pose = robot.get_pose()
    final_velocity = robot.get_velocity()

    final_error = calculate_distance(
        final_pose[:2],
        goal_xy,
    )

    final_linear_speed, _ = get_speed_components(
        final_velocity
    )

    return {
        "episode_id": int(task["episode_id"]),
        "success": bool(success and not collision),
        "collision": bool(collision),
        "timeout": bool(
            not success
            and not collision
            and step >= MAX_CONTROL_STEPS
        ),
        "steps": int(step),
        "elapsed_time_s": float(elapsed_time),
        "planned_path_length_m": float(
            task["planned_path_length_m"]
        ),
        "actual_travel_distance_m": float(
            actual_distance
        ),
        "final_goal_error_m": float(final_error),
        "final_speed_mps": float(final_linear_speed),
    }


# --------------------------------------------------
# Main evaluation
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TASK_FILE.exists():
        raise FileNotFoundError(TASK_FILE)

    benchmark = json.loads(
        TASK_FILE.read_text(encoding="utf-8")
    )

    tasks = benchmark["tasks"]

    print("Loading BC benchmark...")
    print("Tasks:", len(tasks))

    policy = ExpertController(
        goal_threshold=0.12
    )
    EnvironmentBuilder = load_environment_builder()

    client = p.connect(p.DIRECT)

    if client < 0:
        raise RuntimeError("PyBullet connection failed")

    results = []

    try:

        for index, task in enumerate(tasks, start=1):

            print(
                f"\n========== BC TASK "
                f"{index}/{len(tasks)} =========="
            )

            result = run_task(
                task,
                policy,
                EnvironmentBuilder,
            )

            results.append(result)

            print(
                f"Success: {result['success']} | "
                f"Collision: {result['collision']} | "
                f"Steps: {result['steps']} | "
                f"Error: {result['final_goal_error_m']:.3f} m"
            )

    finally:
        p.disconnect(client)

    # Save individual results.
    csv_path = OUTPUT_DIR / "pid_results.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys()),
        )

        writer.writeheader()
        writer.writerows(results)

    # Calculate aggregate metrics.
    total = len(results)

    successes = sum(
        r["success"] for r in results
    )

    collisions = sum(
        r["collision"] for r in results
    )

    timeouts = sum(
        r["timeout"] for r in results
    )

    summary = {
        "controller": "PID Expert",
        "total_tasks": total,
        "successful_tasks": successes,
        "failed_tasks": total - successes,
        "collisions": collisions,
        "timeouts": timeouts,
        "success_rate": successes / total,
        "collision_rate": collisions / total,
        "average_steps": float(
            np.mean([r["steps"] for r in results])
        ),
        "average_actual_distance_m": float(
            np.mean([
                r["actual_travel_distance_m"]
                for r in results
            ])
        ),
        "average_final_error_m": float(
            np.mean([
                r["final_goal_error_m"]
                for r in results
            ])
        ),
        "average_elapsed_time_s": float(
            np.mean([
                r["elapsed_time_s"]
                for r in results
            ])
        ),
    }

    summary_path = OUTPUT_DIR / "pid_summary.json"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== PID EVALUATION SUMMARY ==========")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nSaved:", csv_path)
    print("Saved:", summary_path)

    print("\nBC BATCH EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
