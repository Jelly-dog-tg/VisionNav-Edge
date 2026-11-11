"""VisionNav-Edge: Follow simplified A* waypoints in PyBullet."""

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.diff_drive_robot import DiffDriveRobot

WAYPOINT_FILE = ROOT / "experiments/results/paths/warehouse_waypoints.json"
URDF_FILE = ROOT / "urdf/diff_drive_robot.urdf"
BUILDER_FILE = ROOT / "scripts/05_environment_builder.py"

MAX_STEPS = 3000
GOAL_TOLERANCE = 0.25
WAYPOINT_TOLERANCE = 0.12


def load_environment_builder():
    spec = importlib.util.spec_from_file_location("environment_builder", BUILDER_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.EnvironmentBuilder


def normalize_angle(angle):
    """Normalize angle to [-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def compute_control(pose, target):
    """Turn toward the waypoint and drive forward."""
    dx = target[0] - pose[0]
    dy = target[1] - pose[1]

    distance = math.hypot(dx, dy)
    error = normalize_angle(math.atan2(dy, dx) - pose[2])

    w = float(np.clip(1.5 * error, -0.8, 0.8))

    # Turn in place if the direction error is large
    v = 0.0 if abs(error) > 0.17 else min(0.25, 0.7 * distance)

    return v, w, distance


def detect_collision(robot_id, ground_id):
    """Detect contacts with obstacles, excluding ground."""
    return any(contact[2] != ground_id
               for contact in p.getContactPoints(bodyA=robot_id))


def run_navigation(robot, waypoints, ground_id):
    """Drive along the waypoints. Returns (reached, collision, trajectory)."""
    current_idx = 1
    trajectory = []
    collision = False
    reached = False

    try:
        for step in range(MAX_STEPS):
            pose = robot.get_pose()
            trajectory.append(pose[:2].tolist())

            final_distance = float(np.linalg.norm(np.asarray(pose[:2]) - waypoints[-1]))
            if final_distance < GOAL_TOLERANCE:
                reached = True
                print(f"Goal reached at step {step}")
                break

            if detect_collision(robot.robot_id, ground_id):
                collision = True
                print(f"Collision detected at step {step}")
                break

            # Advance through completed waypoints
            while current_idx < len(waypoints) - 1:
                distance = np.linalg.norm(np.asarray(pose[:2]) - waypoints[current_idx])
                if distance >= WAYPOINT_TOLERANCE:
                    break
                current_idx += 1
                print(f"Waypoint {current_idx}/{len(waypoints) - 1}")

            v, w, _ = compute_control(pose, waypoints[current_idx])
            robot.set_velocity(v, w)

            for _ in range(10):
                p.stepSimulation()

            if step % 100 == 0:
                print(f"Step {step}: "
                      f"pos=({pose[0]:.2f}, {pose[1]:.2f}), "
                      f"waypoint={current_idx}, "
                      f"goal_distance={final_distance:.2f}m, "
                      f"v={v:.2f}, w={w:.2f}")

            time.sleep(1 / 60)

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        robot.set_velocity(0, 0)

    return reached, collision, trajectory


def main():
    if not WAYPOINT_FILE.exists():
        raise FileNotFoundError(WAYPOINT_FILE)

    with WAYPOINT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not data.get("collision_free_on_grid", False):
        raise ValueError("Waypoint path has not passed grid validation")

    waypoints = np.asarray(data["waypoints_world"], dtype=float)
    if waypoints.ndim != 2 or waypoints.shape[1] != 2:
        raise ValueError("Invalid waypoint data")
    if len(waypoints) < 2:
        raise ValueError("At least two waypoints are required")

    client = p.connect(p.GUI)
    if client < 0:
        raise RuntimeError("Cannot open PyBullet GUI")

    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        p.resetDebugVisualizerCamera(
            cameraDistance=13, cameraYaw=45, cameraPitch=-55,
            cameraTargetPosition=[3, 3, 0],
        )

        ground_id = p.loadURDF("plane.urdf")

        EnvironmentBuilder = load_environment_builder()
        builder = EnvironmentBuilder(client)
        builder.create_warehouse_environment()

        robot = DiffDriveRobot(str(URDF_FILE), start_pos=[0, 0, 0.1])

        # Draw the planned waypoint route
        for i in range(len(waypoints) - 1):
            p.addUserDebugLine([*waypoints[i], 0.15], [*waypoints[i + 1], 0.15],
                               lineColorRGB=[0, 0, 1], lineWidth=3)

        print("=" * 50)
        print("VisionNav-Edge A* Waypoint Navigation")
        print("=" * 50)
        print("Waypoints:", len(waypoints))
        print("Goal:", waypoints[-1].tolist())

        reached, collision, trajectory = run_navigation(robot, waypoints, ground_id)

        final_pose = robot.get_pose()
        final_distance = float(np.linalg.norm(np.asarray(final_pose[:2]) - waypoints[-1]))

        print("\n" + "=" * 50)
        print("Navigation Summary")
        print("=" * 50)
        print("Final position:", final_pose[:2])
        print("Final goal distance:", round(final_distance, 3), "m")
        print("Collision detected:", collision)
        print("Goal reached:", reached)
        print("Trajectory samples:", len(trajectory))

        if reached and not collision:
            print("A* NAVIGATION PASSED")
        else:
            print("A* NAVIGATION FAILED OR INCOMPLETE")

    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
