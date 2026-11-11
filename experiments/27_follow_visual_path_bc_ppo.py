
"""P4-5: Active semantic search and approach using BC+PPO V3 navigation."""

import json
import os
import argparse
import re
from dataclasses import asdict
import sys
import time
import importlib.util
from pathlib import Path

# NumPy MKL and PyTorch ship different OpenMP runtimes in this Windows env.
# Sequential MKL avoids loading a second runtime; do not permit unsafe duplicates.
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import numpy as np
import pybullet as p
import pybullet_data
import sys
from pathlib import Path

# Add project root to Python module search path.
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionnav.learning.bc_ppo_v3_controller import (
    BCPPOV3Controller
)

from scripts.diff_drive_robot import DiffDriveRobot

from visionnav.simulation.semantic_scene import (
    add_semantic_targets,
    add_static_detour_obstacle,
    add_occlusion_obstacle,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(
        name, str(path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EnvironmentBuilder = load_module(
    "env_builder",
    ROOT / "scripts/05_environment_builder.py",
).EnvironmentBuilder

# Reuse the physical search replay and own-footprint measurements from 25.
visual_plan = load_module("visual_goal_plan_runtime", ROOT / "experiments/25_visual_goal_plan.py")


# Navigation/stop tolerances are deliberately phase-specific.
# Search viewpoints enter stable stop within the same 12 cm acceptance radius;
# the main approach enters within 10 cm, then terminal docking refines to 1 cm.
# Physical gap, heading, speed, and collision remain independent hard requirements.
FINAL_POSITION_TOLERANCE_M = 0.120
SEARCH_GOAL_TOLERANCE_M = FINAL_POSITION_TOLERANCE_M
APPROACH_GOAL_TOLERANCE_M = 0.100
STOP_LINEAR_SPEED_TOLERANCE_MPS = 0.05
STOP_ANGULAR_SPEED_TOLERANCE_RADPS = 0.05
STOP_HEADING_TOLERANCE_RAD = 0.04
REQUIRED_SETTLE_STEPS = 10

def approach_metrics(pose, target_xyz, geometry, center_z=None):
    """Approximate front-segment gap to a VISUALLY predicted surface point."""
    delta = np.asarray(target_xyz[:2], dtype=float) - np.asarray(pose[:2])
    heading = np.array([np.cos(pose[2]), np.sin(pose[2])])
    forward = float(delta @ heading)
    lateral = float(delta @ np.array([-heading[1], heading[0]]))
    signed_gap = forward - geometry["front_m"]
    outside_width = max(0.0, abs(lateral) - geometry["half_width_m"])
    front_distance = float(np.hypot(signed_gap, outside_width))
    facing = abs(float(np.arctan2(lateral, forward))) <= 0.12
    # A negative projection is not a safe gap; a side/rear approach cannot pass.
    close = bool(facing and 0.08 <= signed_gap <= 0.15 and outside_width <= 0.02)
    metrics = {"center_distance_xy_m": float(np.linalg.norm(delta)),
            "front_gap_projected_m": signed_gap,
            "front_segment_distance_xy_m": front_distance,
            "target_lateral_offset_m": lateral, "lateral_outside_distance_m": outside_width,
            "front_heading_error_rad": abs(float(np.arctan2(lateral, forward))), "facing_target": facing,
            "close_approach_passed": close}
    if center_z is not None:
        metrics["center_distance_3d_m"] = float(np.linalg.norm(
            [delta[0], delta[1], target_xyz[2] - center_z]))
        metrics["front_segment_distance_3d_m"] = float(np.hypot(
            front_distance, target_xyz[2] - center_z))
    return metrics


TERMINAL_PARAMETERS = {"max_linear_mps": 0.04, "max_angular_radps": 0.25,
    "max_distance_m": 0.20, "max_steps": 600, "position_tolerance_m": 0.010,
    "stable_steps": 10, "no_progress_steps": 120}


def safe_terminal_metrics(metrics, elevated=False):
    """Only visual geometry enters online acceptance; thresholds are unchanged."""
    return bool(metrics["close_approach_passed"] and
                (not elevated or metrics["front_segment_distance_3d_m"] <= 0.15))


def terminal_path(grid, current_xy, goal, resolution, arena_size):
    """Plan anew on the existing inflated map; never reuse a previous path."""
    start = visual_plan.planner.world_to_grid(current_xy, resolution, arena_size, grid)
    end = visual_plan.planner.world_to_grid(goal, resolution, arena_size, grid)
    if start is None or end is None or grid[start] != 0 or grid[end] != 0:
        raise RuntimeError("NO_SAFE_TERMINAL_PATH")
    path, _ = visual_plan.planner.astar(grid, start, end, resolution)
    if not path:
        raise RuntimeError("NO_SAFE_TERMINAL_PATH")
    # Current pose and exact safe goal are in their respective free cells.
    return [np.asarray(current_xy).tolist()] + [
        visual_plan.planner.grid_to_world(c, resolution, arena_size) for c in path[1:-1]
    ] + [np.asarray(goal).tolist()]


def has_collision(robot_id, ground_id, barrier_id):
    """Detect contact with the barrier or other scene obstacles."""

    # Conservative: any contact with the test barrier is unsafe.
    if p.getContactPoints(
        bodyA=robot_id,
        bodyB=barrier_id,
    ):
        return True

    for contact in p.getContactPoints(bodyA=robot_id):
        other_body = contact[2]
        normal_force = contact[9]

        # Normal wheel-ground contact is allowed.
        if other_body == ground_id:
            continue

        if normal_force > 0.1:
            return True

    return False


def main():
    plan_path = ROOT / "logs/p4_visual_goal_plan.json"

    if not plan_path.is_file():
        raise FileNotFoundError(
            "Navigation plan not found. Run script 25 first."
        )

    plan = json.loads(
        plan_path.read_text(encoding="utf-8")
    )

    if (plan.get("status") != "READY" or plan.get("plan_version") != 2
            or plan.get("perception_status") != "TARGET_FOUND"):
        raise ValueError("No validated current plan: rerun 25_visual_goal_plan.py")

    if plan.get("scene_id") != "warehouse_test_detour":
        raise ValueError(
            "Unsupported planning scene. "
            "Rerun 25_visual_goal_plan.py first."
        )

    if plan.get("layout") != "test":
        raise ValueError("Unsupported layout: rerun 25_visual_goal_plan.py")

    straight_path_blocked = plan.get("straight_path_blocked")
    if not isinstance(straight_path_blocked, bool):
        raise ValueError("Plan is missing the direct-path diagnostic")

    waypoints = np.asarray(
        plan["waypoints_world"],
        dtype=float,
    )

    start = np.asarray(
        plan["start_xy"],
        dtype=float,
    )

    stop_goal = np.asarray(
        plan["stop_goal_xy"],
        dtype=float,
    )

    target_surface = np.asarray(
        plan["target_surface_xy"],
        dtype=float,
    )

    if (
        waypoints.ndim != 2
        or waypoints.shape[1] != 2
        or len(waypoints) < 1
        or start.shape != (2,)
        or stop_goal.shape != (2,)
        or target_surface.shape != (2,)
    ):
        raise ValueError("Invalid navigation plan coordinates")

    if not all(
        np.all(np.isfinite(a))
        for a in (waypoints, start, stop_goal, target_surface)
    ):
        raise ValueError("Navigation plan contains NaN or Inf")

    start_pose = np.asarray(plan["start_pose"], dtype=float)
    target_xyz = np.asarray(plan["target_surface_xyz"], dtype=float)
    goal_yaw = float(plan["goal_yaw_rad"])
    if (start_pose.shape != (3,) or target_xyz.shape != (3,)
            or not np.all(np.isfinite(start_pose))
            or not np.all(np.isfinite(target_xyz)) or not np.isfinite(goal_yaw)
            or not np.allclose(start_pose[:2], start)
            or not np.allclose(target_xyz[:2], target_surface)):
        raise ValueError("Invalid start pose or visual target XYZ")

    # The straight line from start to stop defines
    # the baseline used to measure actual detour.
    route_direction = stop_goal - start
    route_length = float(np.linalg.norm(route_direction))

    route_unit = route_direction / route_length if route_length > 1e-6 else np.array([1., 0.])

    client = p.connect(p.GUI)

    if client < 0:
        raise RuntimeError("PyBullet GUI connection failed")

    GOAL_TOLERANCE = APPROACH_GOAL_TOLERANCE_M
    FINAL_TOLERANCE = FINAL_POSITION_TOLERANCE_M
    SPEED_TOLERANCE = STOP_LINEAR_SPEED_TOLERANCE_MPS
    DETOUR_TOLERANCE = 0.40  # Diagnostic only; not a success requirement.
    CAMERA_REFRESH_STEPS = 15
    MAX_STEPS = 1500
    PHYSICS_STEPS = 10

    success = False
    collided = False
    steps = 0
    settle_count = 0
    waypoint_index = 0
    max_lateral_deviation = 0.0

    robot = None
    controller = None

    try:
        for preview_flag in (
            p.COV_ENABLE_RGB_BUFFER_PREVIEW,
            p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
            p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
        ):
            p.configureDebugVisualizer(preview_flag, 1)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)

        ground_id = p.loadURDF("plane.urdf")

        builder = EnvironmentBuilder(client)

        builder.create_warehouse_environment(
            arena_size=10.0
        )

        # Reconstruct the same scene used in script 25.
        add_semantic_targets(
            layout=plan.get("layout", "test")
        )

        barrier_id = add_static_detour_obstacle()

        robot = DiffDriveRobot(
            str(ROOT / "urdf/diff_drive_robot.urdf"),
            start_pos=[0, 0, 0.1],
            start_yaw=0.0,
        )

        for _ in range(20):
            p.stepSimulation()

        geometry = visual_plan.robot_geometry(robot)
        if any(abs(geometry[key] - float(plan["robot_geometry"][key])) > 0.005
               for key in ("front_m", "rear_m", "half_width_m")):
            raise RuntimeError("Robot geometry differs from planning")
        # Replay wheel commands in the identical scene; never teleport to the
        # post-search observation pose. Check collisions during every step.
        commands = plan["search_commands"]
        if not isinstance(commands, list) or len(commands) > 1300:
            raise ValueError("Invalid search replay")
        for speed, physics_steps in commands:
            if (not np.isfinite(speed) or abs(speed) > 0.8
                    or physics_steps != 10):
                raise ValueError("Invalid search motion command")
            visual_plan.search_action(robot, ground_id, speed, physics_steps)
        robot.set_velocity(0.0, 0.0)
        robot.get_camera_image()

        initial_pose = robot.get_pose()
        initial_xy = np.asarray(
            initial_pose[:2], dtype=float
        )

        yaw_error = np.arctan2(np.sin(initial_pose[2] - start_pose[2]),
                               np.cos(initial_pose[2] - start_pose[2]))
        if np.linalg.norm(initial_xy - start) > 0.03 or abs(yaw_error) > 0.03:
            raise RuntimeError(
                "Robot post-search position/heading differs from planning"
            )

        controller = BCPPOV3Controller(
            robot=robot,
            waypoints=waypoints,
            final_goal=stop_goal,
            seed=42,
            residual_scale=0.25,
            goal_threshold=GOAL_TOLERANCE,
        )

        p.resetDebugVisualizerCamera(
            cameraDistance=5,
            cameraYaw=45,
            cameraPitch=-50,
            cameraTargetPosition=[1.0, 0.0, 0.0],
        )

        # Draw A* path in cyan.
        for a, b in zip(waypoints[:-1], waypoints[1:]):
            p.addUserDebugLine(
                [float(a[0]), float(a[1]), 0.10],
                [float(b[0]), float(b[1]), 0.10],
                lineColorRGB=[0.0, 0.8, 1.0],
                lineWidth=3,
            )

        # Mark the safe stopping position.
        gx, gy = stop_goal

        p.addUserDebugLine(
            [gx - 0.10, gy, 0.12],
            [gx + 0.10, gy, 0.12],
            [1, 1, 0],
            lineWidth=3,
        )
        p.addUserDebugLine(
            [gx, gy - 0.10, 0.12],
            [gx, gy + 0.10, 0.12],
            [1, 1, 0],
            lineWidth=3,
        )

        print("\nP4-4 SEMANTIC NAVIGATION")
        print(f"Instruction: {plan['instruction']}")
        print(f"VLM target: {plan['target']}")
        print(f"Target surface: {target_surface}")
        print(f"Safe stop goal: {stop_goal}")
        print(f"Path waypoints: {len(waypoints)}")
        print(f"Straight path blocked: {straight_path_blocked}")
        print("\nStarting navigation...\n")

        for steps in range(1, MAX_STEPS + 1):
            # Sensor preview only: no VLM inference during execution.
            if steps % CAMERA_REFRESH_STEPS == 0:
                robot.get_camera_image()

            pose = robot.get_pose()

            # Always use the real robot pose.
            current_xy = np.asarray(
                pose[:2], dtype=float
            )

            final_distance = float(
                np.linalg.norm(current_xy - stop_goal)
            )

            # Perpendicular distance to start-goal line.
            offset = current_xy - start

            lateral_deviation = abs(
                route_unit[0] * offset[1]
                - route_unit[1] * offset[0]
            )

            max_lateral_deviation = max(
                max_lateral_deviation,
                float(lateral_deviation),
            )

            # Reuse existing PID controller parameters.
            if final_distance < 0.6:
                controller.safety_distance = 0.22
                controller.critical_distance = 0.16
            else:
                controller.safety_distance = 0.6
                controller.critical_distance = 0.4

            # Final approach and stable stopping.
            if final_distance <= GOAL_TOLERANCE:
                heading_error = float(np.arctan2(np.sin(goal_yaw - pose[2]),
                                                 np.cos(goal_yaw - pose[2])))
                # Final orientation uses wheel rotation; PID path following is unchanged.
                angular_command = (
                    float(np.clip(1.5 * heading_error, -0.5, 0.5))
                    if abs(heading_error) > STOP_HEADING_TOLERANCE_RAD
                    else 0.0
                )
                robot.set_velocity(0.0, angular_command)

                for _ in range(PHYSICS_STEPS):
                    p.stepSimulation()

                if has_collision(
                    robot.robot_id,
                    ground_id,
                    barrier_id,
                ):
                    collided = True
                    print("Collision during stopping.")
                    break

                stopped_pose = robot.get_pose()
                stopped_xy = np.asarray(stopped_pose[:2], dtype=float)
                stopped_distance = float(np.linalg.norm(stopped_xy - stop_goal))
                stopped_velocity = np.asarray(robot.get_velocity(), dtype=float)
                stopped_speed = float(np.linalg.norm(stopped_velocity[:2]))
                stopped_angular_speed = abs(float(stopped_velocity[2]))
                stopped_heading_error = float(np.arctan2(
                    np.sin(goal_yaw - stopped_pose[2]),
                    np.cos(goal_yaw - stopped_pose[2]),
                ))

                if (
                    stopped_distance <= FINAL_TOLERANCE
                    and stopped_speed <= SPEED_TOLERANCE
                    and stopped_angular_speed <= STOP_ANGULAR_SPEED_TOLERANCE_RADPS
                    and abs(stopped_heading_error) <= STOP_HEADING_TOLERANCE_RAD
                ):
                    settle_count += 1
                else:
                    settle_count = 0

                if settle_count >= REQUIRED_SETTLE_STEPS:
                    success = True
                    break

                continue

            settle_count = 0

            # Advance along the A* waypoints.
            while waypoint_index < len(waypoints) - 1:
                waypoint_distance = float(
                    np.linalg.norm(
                        current_xy - waypoints[waypoint_index]
                    )
                )

                if waypoint_distance >= 0.08:
                    break

                waypoint_index += 1

            waypoint = waypoints[waypoint_index]

            lidar = robot.get_lidar_scan()

            v, w, _ = controller.compute_control(
                pose,
                waypoint,
                lidar,
            )

            robot.set_velocity(
                float(v),
                float(w),
            )

            for _ in range(PHYSICS_STEPS):
                p.stepSimulation()

            if has_collision(
                robot.robot_id,
                ground_id,
                barrier_id,
            ):
                collided = True
                print(
                    f"Collision detected at step {steps}."
                )
                break

            if steps % 30 == 0:
                print(
                    f"Step {steps} | "
                    f"pos=({pose[0]:.2f}, {pose[1]:.2f}) | "
                    f"goal_dist={final_distance:.3f}m | "
                    f"lateral={lateral_deviation:.3f}m | "
                    f"waypoint={waypoint_index + 1}/"
                    f"{len(waypoints)}"
                )

            time.sleep(1 / 60)

        # Stop the robot after exiting the loop.
        robot.set_velocity(0.0, 0.0)

        for _ in range(10):
            p.stepSimulation()

        if has_collision(
            robot.robot_id,
            ground_id,
            barrier_id,
        ):
            collided = True

        final_pose = robot.get_pose()

        final_xy = np.asarray(
            final_pose[:2], dtype=float
        )

        final_distance = float(
            np.linalg.norm(final_xy - stop_goal)
        )

        final_velocity = np.asarray(robot.get_velocity(), dtype=float)
        final_speed = float(np.linalg.norm(final_velocity[:2]))
        final_angular_speed = abs(float(final_velocity[2]))

        target_distance = float(
            np.linalg.norm(final_xy - target_surface)
        )

        final_offset = final_xy - start

        max_lateral_deviation = max(
            max_lateral_deviation,
            abs(
                route_unit[0] * final_offset[1]
                - route_unit[1] * final_offset[0]
            ),
        )

        detour_verified = (
            max_lateral_deviation >= DETOUR_TOLERANCE
        )

        own_base_z = p.getBasePositionAndOrientation(robot.robot_id)[0][2]
        metrics = approach_metrics(final_pose, target_xyz, geometry,
                                   own_base_z + geometry["center_z_offset_m"])
        reached_plan = (
            success
            and not collided
            and final_distance <= FINAL_TOLERANCE
            and final_speed <= SPEED_TOLERANCE
            and final_angular_speed <= STOP_ANGULAR_SPEED_TOLERANCE_RADPS
        )
        ground_plane_close = metrics["close_approach_passed"]
        # A horizontal projection alone cannot certify physical proximity to
        # an elevated object. Report its ground observation stop separately.
        close_approach = ground_plane_close and (
            not plan["elevated_target"]
            or metrics["front_segment_distance_3d_m"] <= 0.15)
        metrics["ground_plane_close"] = ground_plane_close
        metrics["close_approach_passed"] = bool(close_approach)
        passed = reached_plan and close_approach
        print(f"Parking reason: {plan['parking_reason']}")
        print(f"Reached planned stop: {reached_plan}")
        print(f"Ground-plane front-gap requirement passed: {ground_plane_close}")
        print(f"Close approach passed: {close_approach}")
        print(f"Robot geometry: length={geometry['length_m']:.3f}m, "
              f"width={geometry['width_m']:.3f}m")
        print(f"Center-to-predicted-surface XY: {metrics['center_distance_xy_m']:.3f}m")
        print(f"Center-to-predicted-surface 3D: {metrics['center_distance_3d_m']:.3f}m")
        print(f"Signed projected front gap: {metrics['front_gap_projected_m']:.3f}m")
        print(f"Front-segment-to-predicted-surface XY: "
              f"{metrics['front_segment_distance_xy_m']:.3f}m")
        print(f"Facing target: {metrics['facing_target']}")
        if plan["elevated_target"]:
            print(f"Ground observation stop reached: {reached_plan}")
            print(f"Front-to-predicted-surface 3D distance: "
                  f"{metrics['front_segment_distance_3d_m']:.3f}m")
            print("Elevated target: gaps are ground-plane projections; "
                  "this is a ground observation stop, not physical access to the object.")
        if reached_plan and not close_approach:
            print("TARGET_TOO_FAR_OR_UNSAFE_ALIGNMENT: yellow-cross arrival does not pass close approach")
        visual_plan.save_json(ROOT / "logs/p4_navigation_result.json", {
            "run_id": plan["run_id"], "target": plan["target"],
            "reached_planned_stop": reached_plan, "success": passed,
            "collision": collided, "final_pose": final_pose.tolist(),
            "final_error_m": final_distance, "final_speed_mps": final_speed,
            "max_lateral_deviation_m": float(max_lateral_deviation),
            "detour_verified": bool(detour_verified),
            "straight_path_blocked": straight_path_blocked,
            "elevated_target": plan["elevated_target"],
            "ground_observation_stop_reached": bool(reached_plan and plan["elevated_target"]),
            "geometry": geometry, "metrics": metrics,
            "parking_reason": plan["parking_reason"]})

        print("\n" + "=" * 50)
        print("P4-4 SEMANTIC NAVIGATION RESULT")
        print("=" * 50)

        print(f"Target: {plan['target']}")
        print(f"Steps: {steps}")

        print(
            f"Final position: "
            f"({final_pose[0]:.3f}, {final_pose[1]:.3f})"
        )

        print(
            f"Distance to stop goal: "
            f"{final_distance:.3f} m"
        )

        print(
            f"Distance to target surface: "
            f"{target_distance:.3f} m"
        )

        print(f"Final speed: {final_speed:.3f} m/s")
        print(f"Final angular speed: {final_angular_speed:.4f} rad/s")
        print(f"Collision: {collided}")

        print(
            f"Max lateral deviation: "
            f"{max_lateral_deviation:.3f} m"
        )

        print(f"Detour verified: {detour_verified}")
        print(f"Success: {passed}")

        if passed:
            print("\nP4-4 SEMANTIC NAVIGATION PASSED")
        else:
            print("\nP4-4 SEMANTIC NAVIGATION FAILED")

        # Keep the final pose visible briefly in the GUI.
        time.sleep(3)

    finally:
        if controller is not None:
            controller.close()

        if robot is not None:
            controller = None
            try:
                robot.set_velocity(0.0, 0.0)
            except p.error:
                pass

        if p.isConnected(client):
            p.disconnect(client)


def geometric_visibility(grid, xy, yaw, resolution, arena_size, distance=3.5):
    """Approximate horizontal camera coverage using ONLY occupied/free cells."""
    cells = set()
    for angle in np.linspace(yaw - 0.64, yaw + 0.64, 27):
        direction = np.array([np.cos(angle), np.sin(angle)])
        for radius in np.arange(0.0, distance, resolution):
            cell = visual_plan.planner.world_to_grid(
                xy + direction * radius, resolution, arena_size, grid)
            if cell is None:
                break
            cells.add(cell)
            if grid[cell] != 0:
                break
    return cells


def choose_viewpoint(grid, covered, current_xy, visited, resolution, arena_size, search_origin):
    """Finite geometric gain/path-cost ranking; no target inputs or semantic IDs."""
    free = (grid == 0).astype(np.uint8)
    clearance = visual_plan.cv2.distanceTransform(free, visual_plan.cv2.DIST_L2, 5) * resolution
    start = visual_plan.planner.world_to_grid(current_xy, resolution, arena_size, grid)
    if start is None or grid[start] != 0:
        raise RuntimeError("SEARCH_START_BLOCKED")
    # A finite local search region is centered on the initial ROBOT pose.
    # It bounds the six-stop experiment without favoring any semantic location.
    rows, cols = np.indices(grid.shape)
    world_x = -arena_size + (cols + .5) * resolution
    world_y = -arena_size + (rows + .5) * resolution
    local = (world_x-search_origin[0])**2 + (world_y-search_origin[1])**2 <= 3.5**2
    # Unknown geometric boundaries are more useful than distant empty floor.
    weights = np.where(clearance < .8, 2.0, .2)
    candidates = []
    # A coarse lattice is tied to the map, never to a semantic target location.
    for row in range(4, grid.shape[0], 8):
        for col in range(4, grid.shape[1], 8):
            if grid[row, col] or clearance[row, col] < 0.15 or not local[row, col]:
                continue
            point = np.array(visual_plan.planner.grid_to_world((row, col), resolution, arena_size))
            travel = float(np.linalg.norm(point - current_xy))
            if not 0.8 <= travel <= 3.0:
                continue
            if any(np.linalg.norm(point - np.asarray(v)) < 0.75 for v in visited):
                continue
            candidates.append((travel, (row, col), point))
    ranked = []
    # Bounded candidate count and six possible look directions at each point.
    for travel, cell, point in sorted(candidates, key=lambda c: c[0])[:48]:
        union, best_gain, best_yaw = set(), -1, 0.0
        for yaw in np.arange(6) * np.pi / 3:
            visible = geometric_visibility(grid, point, yaw, resolution, arena_size)
            gain = sum(weights[c] for c in visible if local[c] and not covered[c])
            union.update(visible)
            if gain > best_gain:
                best_gain, best_yaw = gain, float(yaw)
        gain = sum(1 for c in union if local[c] and not covered[c])
        weighted_gain = sum(weights[c] for c in union if local[c] and not covered[c])
        if gain:
            ranked.append((float(weighted_gain) / 100.0 - 0.7 * travel,
                           cell, point, best_yaw, gain, float(weighted_gain)))
    best = None
    for _, cell, point, yaw, gain, weighted_gain in sorted(ranked, key=lambda c: c[0], reverse=True)[:12]:
        path, expanded = visual_plan.planner.astar(grid, start, cell, resolution)
        if not path:
            continue
        length = visual_plan.planner.calculate_path_length(path, resolution)
        score = weighted_gain / 100.0 - 0.7 * length
        if best is None or score > best[0]:
            best = (score, point, yaw, path, gain, length, expanded)
    return best


class ActiveSearchMotion:
    """Wheel-driven active-search motion using one reusable BC+PPO V3 controller."""

    def __init__(
        self,
        robot,
        ground_id,
        barrier_id,
        log,
        seed=42,
        residual_scale=0.25,
        goal_threshold=0.120,
    ):
        self.robot = robot
        self.ground_id = ground_id
        self.barrier_id = barrier_id
        self.log = log
        self.control_steps = 0
        self.physics_steps = 20
        self.previous_xy = robot.get_pose()[:2].copy()

        self.seed = int(seed)
        self.residual_scale = float(residual_scale)
        self.goal_threshold = float(goal_threshold)
        self.controller = None

        self.log["controller"] = {
            "name": "BC + PPO V3 Residual",
            "seed": self.seed,
            "residual_scale": self.residual_scale,
            "goal_threshold_m": self.goal_threshold,
            "model_load_count": 0,
            "phase_runs": [],
        }

    def _prepare_controller(self, points):
        """Load BC/PPO once, then only replace the active path between legs."""
        points = np.asarray(points, dtype=np.float32)

        if self.controller is None:
            self.controller = BCPPOV3Controller(
                robot=self.robot,
                waypoints=points,
                final_goal=points[-1],
                seed=self.seed,
                residual_scale=self.residual_scale,
                goal_threshold=self.goal_threshold,
            )
            self.log["controller"]["model_load_count"] = 1
        else:
            # BCPPOV3Controller keeps its loaded BC model, PPO model and
            # VecNormalize statistics. Only the path-dependent state changes.
            if hasattr(self.controller, "set_path"):
                self.controller.set_path(points, final_goal=points[-1])
            else:
                self.controller.waypoints = points.copy()
                self.controller.final_goal = points[-1].copy()
                self.controller.waypoint_index = 0
                self.controller.step_count = 0
                self.controller.last_diagnostics = {}

        return self.controller

    def _record_phase_metrics(self, stats):
        steps = int(stats["steps"])
        record = {
            "phase": stats["phase"],
            "controller": "BC + PPO V3 Residual",
            "steps": steps,
            "mean_abs_delta_v_mps": (
                stats["sum_abs_delta_v_mps"] / steps if steps else 0.0
            ),
            "mean_abs_delta_w_radps": (
                stats["sum_abs_delta_w_radps"] / steps if steps else 0.0
            ),
            "residual_active_fraction": (
                stats["residual_active_steps"] / steps if steps else 0.0
            ),
            "mean_path_deviation_m": (
                stats["sum_path_deviation_m"] / steps if steps else 0.0
            ),
            "goal_tolerance_m": float(stats["goal_tolerance_m"]),
            "settle_steps": int(stats["settle_steps"]),
            "settled_consecutive_steps": int(stats["settled_consecutive_steps"]),
            "stable_stop_steps_required": REQUIRED_SETTLE_STEPS,
            "final_goal_error_m": stats["final_goal_error_m"],
            "final_speed_mps": stats["final_speed_mps"],
            "final_angular_speed_radps": stats["final_angular_speed_radps"],
            "final_heading_error_rad": stats["final_heading_error_rad"],
            "stable_stop": bool(stats["completed"]),
            "completed": bool(stats["completed"]),
        }
        for key in ("controller_used", "fallback_triggered", "fallback_steps", "fallback_reason", "recovery_path"):
            if key in stats:
                record[key] = stats[key]
        record["fallback_completion"] = bool(stats.get("fallback_triggered") and stats["completed"])
        record["collision"] = bool(self.log.get("collision",False))
        self.log["fallback_steps"] = self.log.get("fallback_steps",0)+stats.get("fallback_steps",0)
        self.log["fallback_completion"] = all(
            item.get("completed",False) for item in self.log["controller"]["phase_runs"]+[record]
            if item.get("fallback_triggered")) if self.log.get("fallback_triggered") else False
        self.log["final_goal_error_m"] = stats["final_goal_error_m"]
        self.log["final_heading_error_rad"] = stats["final_heading_error_rad"]
        self.log["controller"]["phase_runs"].append(record)

    def step(self, v, w):
        self.robot.set_velocity(float(v), float(w))
        for _ in range(10):
            p.stepSimulation()
            self.physics_steps += 1
            xy = self.robot.get_pose()[:2]
            self.log["actual_distance_m"] += float(
                np.linalg.norm(xy - self.previous_xy)
            )
            self.previous_xy = xy.copy()
            if has_collision(
                self.robot.robot_id,
                self.ground_id,
                self.barrier_id,
            ):
                self.log["collision"] = True
                self.robot.set_velocity(0.0, 0.0)
                raise RuntimeError("NAVIGATION_COLLISION")

        self.control_steps += 1
        if self.control_steps % 15 == 0:
            self.robot.get_camera_image()
        time.sleep(1 / 120)

    def turn(self, yaw):
        """Rotate in place for observation/alignment; keep this non-PPO action."""
        original_xy = self.robot.get_pose()[:2].copy()
        for _ in range(240):
            pose = self.robot.get_pose()
            error = float(
                np.arctan2(
                    np.sin(yaw - pose[2]),
                    np.cos(yaw - pose[2]),
                )
            )
            if abs(error) < 0.025:
                break
            self.step(0.0, np.clip(2.0 * error, -0.8, 0.8))
        else:
            raise RuntimeError("SEARCH_ROTATION_TIMEOUT")

        for _ in range(12):
            self.step(0.0, 0.0)

        if np.linalg.norm(self.robot.get_pose()[:2] - original_xy) > 0.10:
            raise RuntimeError("SEARCH_ROTATION_TRANSLATION_EXCEEDED")

    @staticmethod
    def path_progress(points, xy):
        """Geometric arc-length projection; independent of policy waypoint state."""
        best = (float("inf"), 0.)
        arc = 0.
        for a, b in zip(points[:-1], points[1:]):
            segment = b-a
            length = float(np.linalg.norm(segment))
            ratio = float(np.clip((xy-a) @ segment / max(length*length, 1e-12), 0., 1.))
            distance = float(np.linalg.norm(xy-(a+ratio*segment)))
            if distance < best[0]:
                best = (distance, arc+ratio*length)
            arc += length
        if len(points) == 1:
            return 0., float(np.linalg.norm(xy-points[0]))
        return best[1], best[0]

    def map_segment_free(self, a, b):
        grid, resolution, arena = self.navigation_map
        for xy in np.linspace(a, b, max(2, int(np.linalg.norm(b-a)/(.2*resolution))+2)):
            cell = visual_plan.planner.world_to_grid(xy, resolution, arena, grid)
            if cell is None or grid[cell] != 0:
                return False
        return True

    def command_map_safe(self, pose, v, w):
        for dt in np.linspace(0., .20, 10):
            if abs(w) < 1e-8:
                xy = pose[:2] + v*dt*np.array([np.cos(pose[2]),np.sin(pose[2])])
            else:
                xy = pose[:2] + v/w*np.array([np.sin(pose[2]+w*dt)-np.sin(pose[2]),
                                             -np.cos(pose[2]+w*dt)+np.cos(pose[2])])
            if not self.map_segment_free(pose[:2], xy):
                return False
        return True

    def activate_recovery(self, points, pose, stats, reason):
        stats.update(fallback_triggered=True, controller_used="BC+PPO V3 + Pure Pursuit",
                     fallback_reason=reason, recovery_start_pose=pose.tolist())
        self.log["fallback_triggered"] = True
        self.log["controller_used"] = "BC+PPO V3 + Pure Pursuit + Terminal Docking"
        grid, resolution, arena = self.navigation_map
        try:
            path = np.asarray(terminal_path(grid, pose[:2], points[-1], resolution, arena))
        except RuntimeError as exc:
            raise RuntimeError("RECOVERY_NO_SAFE_ASTAR_PATH") from exc
        stats["recovery_path"] = path.tolist()
        print(f"Pure Pursuit recovery: {reason}", flush=True)
        return path

    def pure_pursuit_control(self, pose, points, index):
        """Low-speed differential-drive recovery on a freshly planned inflated path."""
        # A curved tracker may pass a cell center without entering a tiny radius.
        # Project onto the remaining polyline so old corner nodes cannot pull it back.
        nearest = (float("inf"), index)
        for segment_index in range(max(0,index-1),len(points)-1):
            a,b = points[segment_index:segment_index+2]
            segment = b-a
            ratio = float(np.clip((pose[:2]-a)@segment/max(float(segment@segment),1e-12),0.,1.))
            distance = float(np.linalg.norm(pose[:2]-(a+ratio*segment)))
            if distance < nearest[0]:
                nearest = (distance,segment_index+1)
        index = max(index,nearest[1])
        while index < len(points)-1 and np.linalg.norm(points[index]-pose[:2]) < .045:
            index += 1
        # Look ahead along the polyline, but never through an occupied corner.
        target = points[index].copy()
        remaining = .20
        cursor = pose[:2].copy()
        for endpoint in points[index:]:
            delta = endpoint-cursor
            length = float(np.linalg.norm(delta))
            candidate = cursor + delta * min(1., remaining/max(length, 1e-12))
            if not self.map_segment_free(pose[:2], candidate):
                break
            target = candidate
            remaining -= length
            cursor = endpoint
            if remaining <= 0:
                break
        if not self.map_segment_free(pose[:2], target):
            raise RuntimeError("RECOVERY_NO_SAFE_TRACKING_SEGMENT")
        delta = target-pose[:2]
        angle = float(np.arctan2(delta[1],delta[0])-pose[2])
        angle = float(np.arctan2(np.sin(angle),np.cos(angle)))
        if abs(angle) > .35:
            return 0., float(np.clip(2*angle,-.5,.5)), index
        error = float(np.linalg.norm(points[-1]-pose[:2]))
        v = min(.10, .65*error)
        w = float(np.clip(2*v*np.sin(angle)/max(np.linalg.norm(delta),.04),-.5,.5))
        # Check the actual curved forward command on the same inflated grid.
        for dt in np.linspace(0.,.20,10):
            yaw=pose[2]+w*dt
            xy=pose[:2]+v*dt*np.array([np.cos((pose[2]+yaw)/2),np.sin((pose[2]+yaw)/2)])
            if not self.map_segment_free(pose[:2],xy):
                return 0.,float(np.clip(2*angle,-.5,.5)),index
        wheel_peak = max(abs((2*v+w*self.robot.wheel_base)/(2*self.robot.wheel_radius)),
                         abs((2*v-w*self.robot.wheel_base)/(2*self.robot.wheel_radius)))
        ratio = min(1.,self.robot.max_wheel_vel/max(wheel_peak,1e-12))
        return v*ratio,w*ratio,index

    def follow(
        self,
        waypoints,
        goal_yaw,
        color,
        phase="navigation",
        goal_tolerance_m=None,
    ):
        """Follow one A* leg with BC+PPO V3 and phase-specific stop tolerance."""
        points = np.asarray(waypoints, dtype=np.float32)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not len(points)
            or not np.isfinite(points).all()
        ):
            raise ValueError("INVALID_LIVE_PATH")

        active_goal_tolerance = (
            self.goal_threshold
            if goal_tolerance_m is None
            else float(goal_tolerance_m)
        )
        if (
            not np.isfinite(active_goal_tolerance)
            or not 0.005 <= active_goal_tolerance <= 0.15
        ):
            raise ValueError(
                "goal_tolerance_m must be finite and between 0.005m and 0.15m"
            )

        for a, b in zip(points[:-1], points[1:]):
            p.addUserDebugLine(
                [float(a[0]), float(a[1]), 0.10],
                [float(b[0]), float(b[1]), 0.10],
                color,
                lineWidth=3,
            )

        controller = self._prepare_controller(points)
        # Keep the controller's reached flag synchronized with this phase.
        controller.goal_threshold = active_goal_tolerance

        phase_stats = {
            "phase": str(phase),
            "steps": 0,  # Number of BC+PPO policy inference steps.
            "sum_abs_delta_v_mps": 0.0,
            "sum_abs_delta_w_radps": 0.0,
            "residual_active_steps": 0,
            "sum_path_deviation_m": 0.0,
            "goal_tolerance_m": active_goal_tolerance,
            "settle_steps": 0,
            "settled_consecutive_steps": 0,
            "final_goal_error_m": None,
            "final_speed_mps": None,
            "final_angular_speed_radps": None,
            "final_heading_error_rad": None,
            "completed": False,
            "controller_used": "BC+PPO V3", "fallback_triggered": False,
            "fallback_steps": 0, "fallback_completion": False,
        }

        progress_window = []
        last_index, unchanged, deviated = -1, 0, 0
        recovery_points, recovery_index = None, 1
        settled = 0
        settling_mode = False
        try:
            for _ in range(1800):
                pose = self.robot.get_pose()
                error = float(np.linalg.norm(pose[:2] - points[-1]))

                # Enter stop control at the phase threshold. Once entered,
                # remain there while inside the final 12 cm acceptance radius.
                if error <= active_goal_tolerance:
                    settling_mode = True
                elif settling_mode and error > FINAL_POSITION_TOLERANCE_M:
                    settling_mode = False
                    settled = 0

                if settling_mode:
                    heading_error = float(
                        np.arctan2(
                            np.sin(goal_yaw - pose[2]),
                            np.cos(goal_yaw - pose[2]),
                        )
                    )
                    angular_command = (
                        float(np.clip(1.5 * heading_error, -0.5, 0.5))
                        if abs(heading_error) > STOP_HEADING_TOLERANCE_RAD
                        else 0.0
                    )
                    self.step(0.0, angular_command)
                    phase_stats["settle_steps"] += 1

                    stopped_pose = self.robot.get_pose()
                    velocity = np.asarray(self.robot.get_velocity(), dtype=float)
                    stopped_distance = float(
                        np.linalg.norm(stopped_pose[:2] - points[-1])
                    )
                    stopped_speed = float(np.linalg.norm(velocity[:2]))
                    stopped_angular_speed = abs(float(velocity[2]))
                    stopped_heading_error = float(
                        np.arctan2(
                            np.sin(goal_yaw - stopped_pose[2]),
                            np.cos(goal_yaw - stopped_pose[2]),
                        )
                    )

                    stable = (
                        stopped_distance <= FINAL_POSITION_TOLERANCE_M
                        and stopped_speed <= STOP_LINEAR_SPEED_TOLERANCE_MPS
                        and stopped_angular_speed
                        <= STOP_ANGULAR_SPEED_TOLERANCE_RADPS
                        and abs(stopped_heading_error) <= STOP_HEADING_TOLERANCE_RAD
                        and not self.log.get("collision", False)
                    )
                    settled = settled + 1 if stable else 0
                    phase_stats["settled_consecutive_steps"] = settled
                    phase_stats["final_goal_error_m"] = stopped_distance
                    phase_stats["final_speed_mps"] = stopped_speed
                    phase_stats["final_angular_speed_radps"] = stopped_angular_speed
                    phase_stats["final_heading_error_rad"] = abs(
                        stopped_heading_error
                    )

                    if settled >= REQUIRED_SETTLE_STEPS:
                        phase_stats["completed"] = True
                        return
                    continue

                settled = 0
                phase_stats["settled_consecutive_steps"] = 0
                progress, deviation = self.path_progress(points, pose[:2])
                progress_window.append(progress)
                progress_window = progress_window[-80:]
                unchanged = unchanged+1 if controller.waypoint_index == last_index else 0
                last_index = controller.waypoint_index
                deviated = deviated+1 if deviation > .25 else 0
                stalled = (len(progress_window) == 80 and progress_window[-1]-progress_window[0] < .05)
                if recovery_points is None and (stalled or unchanged >= 80 or deviated >= 20):
                    reason = ("PATH_PROGRESS_STALLED" if stalled else
                              "WAYPOINT_STALLED" if unchanged >= 80 else "PATH_DEVIATION")
                    recovery_points = self.activate_recovery(points, pose, phase_stats, reason)
                if recovery_points is not None:
                    v,w,recovery_index = self.pure_pursuit_control(pose,recovery_points,recovery_index)
                    phase_stats["fallback_steps"] += 1
                    if phase_stats["fallback_steps"] % 100 == 0:
                        print(f"[Pure Pursuit] steps={phase_stats['fallback_steps']} "
                              f"goal={error:.3f}m waypoint={recovery_index}",flush=True)
                    self.step(v,w)
                    continue
                waypoint_index = min(
                    controller.waypoint_index,
                    len(controller.waypoints) - 1,
                )
                waypoint = controller.waypoints[waypoint_index]
                lidar = self.robot.get_lidar_scan()

                v, w, _ = controller.compute_control(
                    pose,
                    waypoint,
                    lidar,
                )

                diagnostics = controller.last_diagnostics
                delta_v = abs(float(diagnostics.get("delta_v_mps", 0.0)))
                delta_w = abs(float(diagnostics.get("delta_w_radps", 0.0)))
                path_deviation = float(
                    diagnostics.get("path_deviation_m", 0.0)
                )

                phase_stats["steps"] += 1
                phase_stats["sum_abs_delta_v_mps"] += delta_v
                phase_stats["sum_abs_delta_w_radps"] += delta_w
                phase_stats["sum_path_deviation_m"] += path_deviation
                if delta_v > 1e-7 or delta_w > 1e-7:
                    phase_stats["residual_active_steps"] += 1

                if not self.command_map_safe(pose,v,w):
                    recovery_points = self.activate_recovery(points,pose,phase_stats,"MAP_INFLATION_GUARD")
                    continue
                self.step(v, w)

            raise RuntimeError(
                "NAVIGATION_TIMEOUT: BC+PPO V3 did not stably reach the requested stop"
            )
        finally:
            if not phase_stats["completed"]:
                self.robot.set_velocity(0.0, 0.0)

            # Persist measured terminal state even on timeout/collision.
            final_pose = self.robot.get_pose()
            final_velocity = np.asarray(self.robot.get_velocity(), dtype=float)
            phase_stats["final_goal_error_m"] = float(
                np.linalg.norm(final_pose[:2] - points[-1])
            )
            phase_stats["final_speed_mps"] = float(
                np.linalg.norm(final_velocity[:2])
            )
            phase_stats["final_angular_speed_radps"] = abs(
                float(final_velocity[2])
            )
            phase_stats["final_heading_error_rad"] = abs(float(np.arctan2(
                np.sin(goal_yaw - final_pose[2]),
                np.cos(goal_yaw - final_pose[2]),
            )))
            phase_stats["settled_consecutive_steps"] = settled
            self._record_phase_metrics(phase_stats)

    def terminal_refine_to_goal(self, points, target_xyz, geometry, local_context=None,
                                refresh_observation=None):
        """A: safe pre-stop path; B: face the visual target, then straight docking."""
        cfg=TERMINAL_PARAMETERS
        points=np.asarray(points,dtype=float);pre_goal=points[-1].copy()
        target_xyz=np.asarray(target_xyz,dtype=float)
        self.log.update(terminal_refinement_attempted=True,terminal_refinement_steps=0,
            terminal_refinement_collision=False,terminal_stable_stop=False,
            terminal_alignment_completed=False,terminal_approach_completed=False,
            post_docking_target_verified=False,final_target_verified=False,
            terminal_refinement_distance_m=0.,terminal_phase="PRE_DOCKING_PATH")
        start_distance=self.log["actual_distance_m"]
        phase="A";waypoint=1;stable=0;refreshed=False;midpoint=False
        best=float("inf");stale=0;approach_start=None
        goal=pre_goal.copy()
        try:
            for iteration in range(cfg["max_steps"]):
                pose=self.robot.get_pose();velocity=np.asarray(self.robot.get_velocity(),dtype=float)
                moved=self.log["actual_distance_m"]-start_distance
                if moved>=cfg["max_distance_m"]:
                    raise RuntimeError("TERMINAL_REFINEMENT_DISTANCE_EXCEEDED")
                target_delta=target_xyz[:2]-pose[:2]
                target_yaw=float(np.arctan2(target_delta[1],target_delta[0]))
                target_error=float(np.arctan2(np.sin(target_yaw-pose[2]),np.cos(target_yaw-pose[2])))
                v=0.;w=0.
                if phase=="A":
                    error=float(np.linalg.norm(pre_goal-pose[:2]))
                    if error<=cfg["position_tolerance_m"]:
                        phase="B";best=float("inf");stale=0
                        self.log["terminal_phase"]="TARGET_ALIGNMENT"
                        continue
                    while waypoint<len(points)-1 and np.linalg.norm(points[waypoint]-pose[:2])<.02:
                        waypoint+=1
                    delta=points[waypoint]-pose[:2]
                    heading=float(np.arctan2(delta[1],delta[0])-pose[2])
                    heading=float(np.arctan2(np.sin(heading),np.cos(heading)))
                    w=float(np.clip(2*heading,-cfg["max_angular_radps"],cfg["max_angular_radps"]))
                    v=0. if abs(heading)>.12 else min(cfg["max_linear_mps"],.8*np.linalg.norm(delta))
                    # No target-facing projection here: map and swept physical footprint.
                    progress=error
                else:
                    w=float(np.clip(2*target_error,-cfg["max_angular_radps"],cfg["max_angular_radps"]))
                    aligned=abs(target_error)<=STOP_HEADING_TOLERANCE_RAD
                    self.log["terminal_alignment_completed"]=bool(aligned)
                    if aligned and not refreshed:
                        if refresh_observation is not None:
                            self.step(0.,0.)
                            found,observation=refresh_observation()
                            if found is None:raise RuntimeError("PRE_DOCKING_TARGET_VERIFICATION_FAILED")
                            updated,elevated=found
                            if np.linalg.norm(updated-target_xyz)>.45:
                                raise RuntimeError("PRE_DOCKING_TARGET_ESTIMATE_SHIFT")
                            target_xyz=np.asarray(updated,dtype=float)
                            local_context=visual_plan.build_visual_docking_context(
                                observation,self.log["detections"][-1],geometry,
                                self.navigation_map[0],self.navigation_map[2],previous_context=local_context)
                            self.log["pre_docking_target_verified"]=True
                            self.log["final_target_estimate_xyz"]=target_xyz.tolist()
                        if local_context is None:raise RuntimeError("NO_VERIFIED_LOCAL_DOCKING_GEOMETRY")
                        goal,_=visual_plan.select_local_docking(pose[:2],target_xyz,local_context)
                        self.log["stop_goal_xy"]=goal.tolist()
                        approach_start=pose[:2].copy();refreshed=True;best=float("inf");stale=0
                        self.log["terminal_phase"]="STRAIGHT_APPROACH"
                        continue
                    if refreshed and not midpoint and approach_start is not None and np.linalg.norm(pose[:2]-approach_start)>.045:
                        midpoint=True
                        # One bounded fresh RGB-D update, while stopped, not Qwen per control step.
                        if refresh_observation is not None:
                            self.step(0.,0.)
                            found,observation=refresh_observation()
                            if found is None:raise RuntimeError("TERMINAL_TARGET_UPDATE_FAILED")
                            updated,_=found
                            if np.linalg.norm(updated-target_xyz)>.45:raise RuntimeError("TERMINAL_TARGET_ESTIMATE_SHIFT")
                            target_xyz=np.asarray(updated,dtype=float)
                            local_context=visual_plan.build_visual_docking_context(observation,
                                self.log["detections"][-1],geometry,self.navigation_map[0],self.navigation_map[2],previous_context=local_context)
                            goal,_=visual_plan.select_local_docking(pose[:2],target_xyz,local_context)
                            self.log["stop_goal_xy"]=goal.tolist()
                            self.log["final_target_estimate_xyz"]=target_xyz.tolist()
                            best=float("inf");stale=0
                            continue
                    metrics=approach_metrics(pose,target_xyz,geometry)
                    gap=metrics["front_gap_projected_m"]
                    gap_error=gap-.12
                    self.log["terminal_heading_error_rad"]=target_error
                    self.log["terminal_current_gap_m"]=gap
                    stopped=(refreshed and aligned and abs(gap_error)<=.005
                        and metrics["lateral_outside_distance_m"]<=.02
                        and np.linalg.norm(velocity[:2])<=STOP_LINEAR_SPEED_TOLERANCE_MPS
                        and abs(velocity[2])<=STOP_ANGULAR_SPEED_TOLERANCE_RADPS)
                    stable=stable+1 if stopped else 0
                    self.log["terminal_stable_steps"]=stable
                    if stable>=10:
                        self.log.update(terminal_approach_completed=True,terminal_stable_stop=True,
                            terminal_phase="STABLE_STOP",terminal_target_estimate_xyz=target_xyz.tolist())
                        return
                    if aligned and refreshed:
                        if gap<.08:raise RuntimeError("FINAL_GAP_TOO_SMALL")
                        if gap_error<-.005:raise RuntimeError("TERMINAL_TARGET_TOO_CLOSE_FOR_FORWARD_APPROACH")
                        v=0. if abs(gap_error)<=.005 else min(cfg["max_linear_mps"],.8*gap_error)
                        # Predict in the now meaningful target direction; slow before minimum.
                        allowance=max(0.,gap-.08-.003)/(10/240)
                        v=min(v,allowance)
                    progress=abs(gap_error) if aligned and refreshed else abs(target_error)+1.
                predicted=pose.copy()
                predicted[:2]+=v*(10/240)*np.array([np.cos(pose[2]),np.sin(pose[2])])
                predicted[2]+=w*(10/240)
                if moved+abs(v)*(10/240)>cfg["max_distance_m"]:
                    raise RuntimeError("TERMINAL_REFINEMENT_DISTANCE_EXCEEDED")
                if local_context is not None:
                    for alpha in (0.,.5,1.):
                        if not visual_plan.visual_footprint_clear(pose+alpha*(predicted-pose),local_context):
                            raise RuntimeError("UNSAFE_TERMINAL_SWEPT_FOOTPRINT")
                elif not self.command_map_safe(pose,v,w):
                    raise RuntimeError("NO_SAFE_TERMINAL_PATH")
                if phase=="B" and v>0 and abs(target_error)<=STOP_HEADING_TOLERANCE_RAD:
                    if approach_metrics(predicted,target_xyz,geometry)["front_gap_projected_m"]<.08:
                        raise RuntimeError("FINAL_GAP_TOO_SMALL")
                if progress<best-.0003:best,stale=progress,0
                else:stale+=1
                if stale>=cfg["no_progress_steps"]:raise RuntimeError("TERMINAL_REFINEMENT_NO_PROGRESS")
                self.log["terminal_refinement_steps"]=iteration+1
                self.step(v,w)
            raise RuntimeError("TERMINAL_REFINEMENT_TIMEOUT")
        except RuntimeError as exc:
            if self.log.get("collision"):
                self.log["terminal_refinement_collision"]=True
                raise RuntimeError("TERMINAL_REFINEMENT_COLLISION") from exc
            raise
        finally:
            self.robot.set_velocity(0.,0.)
            self.log["terminal_refinement_distance_m"]=self.log["actual_distance_m"]-start_distance
            self.log["terminal_refinement_final_error_m"]=float(np.linalg.norm(self.robot.get_pose()[:2]-goal))

    def close(self):
        """Release VecNormalize resources; PPO/BC are loaded only once per run."""
        if self.controller is not None:
            self.controller.close()
            self.controller = None



# Language/image attributes only; no semantic scene coordinates or instance IDs.
ACTIVE_TARGETS = {
    "blue cylinder": {"color": "blue", "class": "cylinder"},
    "green sphere": {"color": "green", "class": "sphere"},
    "red box": {"color": "red", "class": "box"},
}


def resolve_active_target(instruction):
    """Normalize Qwen's text description while preserving spatial qualifiers."""
    description = visual_plan.qwen_bbox.canonical_target_description(
        visual_plan.understand_instruction(instruction), instruction).lower()
    description = re.sub(r"\b(?:cube|block|square)\b", "box", description)
    description = re.sub(r"\bball\b", "sphere", description)
    matches = [name for name, attrs in ACTIVE_TARGETS.items()
               if re.search(r"\b" + attrs["color"] + r"\b", description)
               and re.search(r"\b" + attrs["class"] + r"\b", description)]
    if len(matches) != 1:
        raise ValueError("TARGET_AMBIGUOUS_OR_UNSUPPORTED: specify blue cylinder, green sphere or red box")
    # Check the explicit Chinese request against the text model's answer.
    requested = []
    for name, color_words, class_words in (
        ("blue cylinder", ("蓝色", "蓝"), ("圆柱",)),
        ("green sphere", ("绿色", "绿"), ("球",)),
        ("red box", ("红色", "红"), ("方块", "箱子", "方盒", "立方体")),
    ):
        if any(w in instruction for w in color_words) and any(w in instruction for w in class_words):
            requested.append(name)
    if len(requested) > 1 or (requested and matches != requested):
        raise ValueError("TARGET_DESCRIPTION_MISMATCH: Qwen description disagrees with instruction")
    return description, matches[0]


def target_color_fraction(image, bbox, color):
    """Same RGB sanity check for each requested color; never generates a bbox."""
    if image is None:
        raise ValueError("RGB_CAPTURE_UNREADABLE")
    x1,y1,x2,y2 = bbox
    region = image[int(y1):int(np.ceil(y2)), int(x1):int(np.ceil(x2))]
    if not region.size:
        raise ValueError("TARGET_BBOX_INVALID: empty RGB region")
    hsv = visual_plan.cv2.cvtColor(region, visual_plan.cv2.COLOR_BGR2HSV)
    hue = hsv[:,:,0]
    if color == "blue":
        mask = (hue >= 95) & (hue <= 130)
    elif color == "green":
        mask = (hue >= 35) & (hue <= 85)
    elif color == "red":
        mask = (hue <= 10) | (hue >= 170)
    else:
        raise ValueError(f"Unsupported target color: {color}")
    mask &= (hsv[:,:,1] >= 80) & (hsv[:,:,2] >= 40)
    return float(np.mean(mask))



def verify_candidate_color(image, bbox, color, record):
    """Separate HSV evidence from Qwen/RGB-D status; unchanged 0.08 threshold."""
    fraction = target_color_fraction(image, bbox, color)
    record["color_check"] = color
    record["color_pixel_fraction"] = fraction
    record["hsv_verification_status"] = "PASSED" if fraction >= .08 else "FAILED"
    if fraction < .08:
        reason = f"HSV_COLOR_CHECK_FAILED: {fraction:.6f} < 0.08 for {color}"
        record["hsv_rejection_reason"] = reason
        raise ValueError(reason)


def evaluate_completed_active_run(log, robot):
    """Always emit a complete independent evaluation, without changing task failure."""
    failure=log.get("failure_reason")
    defaults={"status":"UNAVAILABLE","gt_usage":"post_run_only",
        "actual_target_position_xyz_m":None,"rgbd_estimated_surface_xyz_m":log.get("target_estimate_xyz"),
        "xy_localization_error_m":None,"initial_target_visible":None,"initial_visible_pixels":None,
        "final_distance_to_actual_target_center_xy_m":None,"final_distance_to_stop_goal_m":log.get("final_error_m"),
        "target_found":bool(log.get("target_found",False)),
        "reached_planned_stop":bool(log.get("reached_planned_stop",False)),
        "collision":bool(log.get("collision",False)),"task_success":bool(log.get("task_success",False)),
        "navigation_failure_reason":failure}
    log["post_run_evaluation"]=dict(defaults)
    try:
        _evaluate_completed_active_run(log,robot)
    except Exception as exc:
        log["post_run_evaluation"].update(status="PARTIAL",reason=str(exc),
            reached_planned_stop=bool(log.get("reached_planned_stop",False)),
            final_distance_to_stop_goal_m=log.get("final_error_m"))
    finally:
        log["post_run_evaluation"]={**defaults,**log.get("post_run_evaluation",{})}
        log["post_run_evaluation"].update(target_found=bool(log.get("target_found",False)),
            collision=bool(log.get("collision",False)),task_success=bool(log.get("task_success",False)))
        log["failure_reason"]=failure


def _evaluate_completed_active_run(log, robot):
    """POST-RUN ONLY: independent GT metrics; never returns a planning input.

    Called only in finally, after navigation/search has finished. Target truth
    is identified from the final physics scene, not read from LAYOUTS. No target
    handle or truth position is retained by the perception/search code.
    """
    paths = log.get("search_paths", [])
    def length(points):
        return float(sum(np.linalg.norm(np.asarray(b)-np.asarray(a))
                         for a,b in zip(points[:-1],points[1:])))
    search_length = sum(length(path) for path in paths)
    approach_length = length(log.get("final_path", []))
    log["astar_path_lengths_m"] = {"search": search_length,
        "approach": approach_length, "total": search_length + approach_length}
    log["target_found"] = "TARGET_VERIFIED" in log.get("stages", [])
    if robot is None or not p.isConnected():
        log["post_run_evaluation"] = {"status": "UNAVAILABLE", "reason": "No live completed scene"}
        return
    pose, velocity = robot.get_pose(), np.asarray(robot.get_velocity(), dtype=float)
    log["final_pose"] = pose.tolist()
    log["final_speed_mps"] = float(np.linalg.norm(velocity[:2]))
    log["final_angular_speed_radps"] = abs(float(velocity[2]))
    goal = log.get("stop_goal_xy")
    if goal is not None:
        log["final_error_m"] = float(np.linalg.norm(pose[:2]-np.asarray(goal)))
        log["post_run_reached_legacy_stop"] = bool(
            log["final_error_m"] <= FINAL_POSITION_TOLERANCE_M
            and log["final_speed_mps"] <= STOP_LINEAR_SPEED_TOLERANCE_MPS
            and log["final_angular_speed_radps"] <= STOP_ANGULAR_SPEED_TOLERANCE_RADPS
            and not log["collision"]
        )
    else:
        log["post_run_reached_legacy_stop"] = False
    # One canonical planned-stop meaning for post-run position/velocity diagnostics.
    log["reached_planned_stop"] = bool(log.get("post_run_reached_legacy_stop",False))
    log.setdefault("post_run_evaluation",{})
    attrs = log["target_attributes"]
    shape = {"box": p.GEOM_BOX, "cylinder": p.GEOM_CYLINDER, "sphere": p.GEOM_SPHERE}[attrs["class"]]
    matching_bodies = set()
    for i in range(p.getNumBodies()):
        body = p.getBodyUniqueId(i)
        if body == robot.robot_id:
            continue
        for visual in p.getVisualShapeData(body):
            if visual[1] != -1 or visual[2] != shape:
                continue
            r,g,b,_ = visual[7]
            matches_color = {"red": r > 1.5*g and r > 1.5*b,
                "blue": b > 1.5*r and b > 1.5*g,
                "green": g > 1.5*r and g > 1.5*b}[attrs["color"]]
            if matches_color and p.getCollisionShapeData(body, -1):
                matching_bodies.add(body)
    if len(matching_bodies) != 1:
        raise ValueError("Post-run evaluation requires one unique physical target")
    body = matching_bodies.pop()
    actual = np.asarray(p.getBasePositionAndOrientation(body)[0], dtype=float)
    estimated = log.get("target_estimate_xyz")
    initial_pixels = None
    if log.get("detections"):
        mask_path = log["detections"][0].get("segmentation_path")
        if mask_path:
            # Segmentation is read ONLY after task termination, for visibility
            # evaluation. It is never used by VLM, RGB-D or viewpoint selection.
            mask = np.load(mask_path, allow_pickle=False)
            initial_pixels = int(np.count_nonzero((mask >= 0) & ((mask & ((1 << 24)-1)) == body)))
    log["post_run_evaluation"] = {"status": "RECORDED", "gt_usage": "post_run_only",
        "actual_target_position_xyz_m": actual.tolist(),
        "rgbd_estimated_surface_xyz_m": estimated,
        "xy_localization_error_m": (float(np.linalg.norm(np.asarray(estimated)[:2]-actual[:2]))
                                      if estimated is not None else None),
        "localization_reference": "predicted visible surface versus true body center",
        "initial_target_visible": (initial_pixels > 0 if initial_pixels is not None else None),
        "initial_visible_pixels": initial_pixels,
        "final_distance_to_actual_target_center_xy_m": float(np.linalg.norm(pose[:2]-actual[:2])),
        "final_distance_to_stop_goal_m": log.get("final_error_m"),
        "target_found": log["target_found"],
        "reached_planned_stop": bool(log.get("reached_planned_stop",False)),
        "collision": log["collision"], "task_success": log["task_success"]}


def active_search(instruction, scenario=None, layout="test", seed=42):
    """P4-5: one GUI world, bounded geometric search, then visual approach."""
    summary_path = ROOT / "logs/p4_active_search_summary.json"
    log = {"run_id": str(time.time_ns()), "mode": "active-search", "scenario": scenario,
           "instruction": instruction, "layout": layout, "stage": "INITIALIZING", "stages": [],
           "visited_viewpoint_count": 0, "visited_viewpoints": [], "detections": [],
           "search_paths": [], "final_path": [], "target_estimate_xyz": None,
           "stop_goal_xy": None, "actual_distance_m": 0.0,
           "collision": False, "task_success": False, "failure_reason": None,
           "controller": {"name": "BC + PPO V3 Residual", "seed": int(seed),
                          "residual_scale": 0.25, "model_load_count": 0,
                          "phase_runs": []},
           "control_tolerances": {
               "search_goal_tolerance_m": SEARCH_GOAL_TOLERANCE_M,
               "approach_goal_tolerance_m": APPROACH_GOAL_TOLERANCE_M,
               "final_position_tolerance_m": FINAL_POSITION_TOLERANCE_M,
               "linear_speed_tolerance_mps": STOP_LINEAR_SPEED_TOLERANCE_MPS,
               "angular_speed_tolerance_radps": STOP_ANGULAR_SPEED_TOLERANCE_RADPS,
               "heading_tolerance_rad": STOP_HEADING_TOLERANCE_RAD,
               "required_stable_steps": REQUIRED_SETTLE_STEPS,
           }}
    evidence_dir = ROOT / "logs/p4_5_1_evidence" / f"run_{log['run_id']}"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    latest_summary_path = summary_path
    summary_path = evidence_dir / "summary.json"
    log["evidence_directory"] = str(evidence_dir)
    log["terminal_parameters"] = dict(TERMINAL_PARAMETERS)
    log.update(terminal_refinement_attempted=False, terminal_refinement_steps=0,
               terminal_refinement_distance_m=0., terminal_refinement_final_error_m=None,
               terminal_refinement_collision=False, final_target_verified=False,
               final_close_approach_passed=False, terminal_stable_stop=False,
               pre_docking_target_verified=False,terminal_alignment_completed=False,
               terminal_approach_completed=False,post_docking_target_verified=False,
               reached_planned_stop=False,
               approach_metrics_before_refinement=None, approach_metrics_after_refinement=None)
    visual_plan.save_json(evidence_dir / "control_parameters.json",
                          {"terminal": TERMINAL_PARAMETERS, "navigation": log["control_tolerances"],
                           "seed": seed, "residual_scale": .25})
    def persist():
        visual_plan.save_json(summary_path, log)
        if summary_path != latest_summary_path:
            visual_plan.save_json(latest_summary_path, log)
    def stage(name):
        log["stage"] = name
        log["stages"].append(name)
        persist()
        print(name, flush=True)
    client, robot, motion = -1, None, None
    try:
        target, canonical_target = resolve_active_target(instruction)
        target_attributes = ACTIVE_TARGETS[canonical_target]
        scenario = scenario or f"occluded-{target_attributes['color']}"
        if scenario not in ("occluded-blue", "occluded-green", "occluded-red"):
            raise ValueError(f"Unsupported occlusion scenario: {scenario}")
        if layout not in ("train", "val", "test", "relocated"):
            raise ValueError(f"Unsupported target layout: {layout}")
        output_dir = evidence_dir
        log["summary_path"] = str(summary_path)
        log["scenario"] = scenario
        log["target_description"] = target
        log["canonical_target"] = canonical_target
        log["target_attributes"] = dict(target_attributes)
        client = p.connect(p.GUI)
        if client < 0:
            raise RuntimeError("GUI_CONNECTION_FAILED")
        log["physics_client_id"] = client
        log["connection_method"] = int(p.getConnectionInfo(client)["connectionMethod"])
        for flag in (p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                     p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW):
            p.configureDebugVisualizer(flag, 1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)
        ground_id = p.loadURDF("plane.urdf")
        EnvironmentBuilder(client).create_warehouse_environment(arena_size=10.)
        # Deliberately discard target metadata. The search layer never receives it.
        add_semantic_targets(layout=layout)
        add_static_detour_obstacle()
        barrier_id = add_occlusion_obstacle(scenario)
        robot = DiffDriveRobot(str(ROOT / "urdf/diff_drive_robot.urdf"), start_pos=[0,0,.1], start_yaw=0.)
        for _ in range(20):
            p.stepSimulation()
        geometry = visual_plan.robot_geometry(robot)
        log["robot_geometry"] = geometry
        robot.get_camera_image()
        p.resetDebugVisualizerCamera(cameraDistance=6, cameraYaw=45, cameraPitch=-55,
                                     cameraTargetPosition=[0.8, 0.3, 0.0])
        # Build the geometric map without deleting or moving the live robot.
        # Obstacle IDs are used only here as geometric map sources, never as goals.
        obstacles = []
        for i in range(p.getNumBodies()):
            body = p.getBodyUniqueId(i)
            if body in (ground_id, robot.robot_id):
                continue
            lo, hi = p.getAABB(body)
            obstacles.append((lo[0], lo[1], hi[0], hi[1]))
        resolution, arena_size = visual_plan.occ.RESOLUTION, 10.
        inflation = max(visual_plan.occ.SAFETY_MARGIN, geometry["radius_m"] + resolution / np.sqrt(2))
        extra = inflation - visual_plan.occ.SAFETY_MARGIN
        grid = visual_plan.occ.create_grid([(a-extra,b-extra,c+extra,d+extra)
                                           for a,b,c,d in obstacles], arena_size)
        log["map"] = {"resolution_m": resolution, "inflation_m": inflation}
        np.savez_compressed(evidence_dir/"navigation_grid.npz",grid=grid,resolution=resolution,arena_size=arena_size)
        motion = ActiveSearchMotion(
            robot,
            ground_id,
            barrier_id,
            log,
            seed=seed,
            residual_scale=0.25,
            goal_threshold=SEARCH_GOAL_TOLERANCE_M,
        )
        motion.navigation_map = (grid, resolution, arena_size)
        log.update(controller_used="BC+PPO V3 + Terminal Docking", fallback_triggered=False,
                   fallback_steps=0, fallback_completion=False)
        adapter = visual_plan.PyBulletObservationAdapter(robot, output_dir, width=640, height=480)
        covered = np.zeros_like(grid, dtype=bool)
        visited = []
        search_origin = robot.get_pose()[:2].copy()
        log["search_budget"] = {"max_viewpoints": 6, "max_views_per_stop": 6,
                                "local_radius_m": 3.5, "origin_xy": search_origin.tolist()}

        def observe():
            # Separate frame numbers from script 25; overwrite bounded active-run
            # files on subsequent runs instead of accumulating duplicate captures.
            index = 1000 + len(log["detections"])
            obs = adapter.capture(timestamp_s=motion.physics_steps / 240,
                scene_id=f"warehouse_{scenario.replace('-', '_')}_{layout}", task_id="active_search", frame_index=index)
            record = {"frame": index, "pose": robot.get_pose().tolist(),
                      "rgb_path": obs.rgb_path, "depth_path": obs.depth_path,
                      "segmentation_path": obs.segmentation_path,
                      "camera": asdict(obs.camera), "validated": False, "status": "OBSERVING",
                      "detector_status": "NOT_RUN", "semantic_verification_status": "NOT_RUN",
                      "rgbd_verification_status": "NOT_RUN", "hsv_verification_status": "NOT_RUN",
                      "candidate_accepted": False}
            log["detections"].append(record)
            T = np.asarray(obs.camera.T_world_camera).reshape(4,4)
            yaw = float(np.arctan2(T[1,2], T[0,2]))
            for cell in geometric_visibility(grid, np.array([obs.state.x_m, obs.state.y_m]),
                                               yaw, resolution, arena_size):
                covered[cell] = True
            result = None
            try:
                result = visual_plan.evaluate_observation(obs, obs.rgb_path, obs.depth_path,
                    target, instruction, geometry, record)
                if result is not None:
                    # Reject a hallucinated box on a screen using the ACTUAL
                    # requested color, after semantic class and RGB-D validation.
                    record["target"] = target
                    record["canonical_target"] = canonical_target
                    verify_candidate_color(visual_plan.cv2.imread(str(obs.rgb_path)),
                        record["bbox_pixels"], target_attributes["color"], record)
                if result is not None:
                    record["candidate_accepted"] = True
                    record["status"] = "TARGET_VERIFIED"
            except json.JSONDecodeError as exc:
                record["validated"] = False
                record["status"] = "CANDIDATE_REJECTED"
                record["json_error"] = str(exc)
                record["rejection_reason"] = "JSONDecodeError: " + str(exc)
                result = None
            except (ValueError, TypeError, KeyError) as exc:
                record["validated"] = False
                record["status"] = "CANDIDATE_REJECTED"
                record["rejection_reason"] = str(exc)
                result = None
            except Exception as exc:
                record["status"] = "DETECTION_ERROR"
                record["error"] = str(exc)
                raise
            finally:
                preview = visual_plan.cv2.imread(str(obs.rgb_path))
                if preview is None:
                    raise RuntimeError("RGB_CAPTURE_UNREADABLE")
                box = record.get("bbox_pixels")
                if box is not None:
                    visual_plan.cv2.rectangle(preview, tuple(map(int,box[:2])), tuple(map(int,box[2:])), (0,255,255), 2)
                visual_plan.cv2.putText(preview, record["status"], (8,24),
                    visual_plan.cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,255), 1)
                preview_path = output_dir / f"active_detected_{index}.png"
                if not visual_plan.cv2.imwrite(str(preview_path), preview):
                    raise RuntimeError("DETECTION_PREVIEW_WRITE_FAILED")
                record["preview_path"] = str(preview_path)
                persist()
            print(f"Detection {index}: {record['status']}", flush=True)
            return result, obs

        stage("INITIAL_OBSERVATION")
        found, observation = observe()
        log["initial_target_verified"] = found is not None
        if found is None:
            stage("TARGET_NOT_VISIBLE")
        for _ in range(6):
            if found is not None:
                break
            stage("SELECT_VIEWPOINT")
            current_xy = robot.get_pose()[:2].copy()
            choice = choose_viewpoint(grid, covered, current_xy, visited, resolution, arena_size, search_origin)
            if choice is None:
                break
            score, point, heading, path, gain, length, expanded = choice
            world_path = [visual_plan.planner.grid_to_world(c,resolution,arena_size) for c in path]
            log["search_paths"].append(world_path)
            entry = {"selected_xy": point.tolist(), "start_pose": robot.get_pose().tolist(),
                     "expected_new_cells": gain, "score": score, "path_length_m": length}
            log["visited_viewpoints"].append(entry)
            stage("SEARCH_NAVIGATION")
            motion.follow(
                world_path,
                heading,
                [1.0, 1.0, 0.0],
                phase="search_navigation",
            )
            visited.append(point.tolist())
            entry["arrived_pose"] = robot.get_pose().tolist()
            log["visited_viewpoint_count"] += 1
            stage("REOBSERVE")
            # Six overlapping directions, using wheel control; no position reset.
            base_yaw = float(robot.get_pose()[2])
            for view in range(6):
                if view:
                    motion.turn(base_yaw + view*np.pi/3)
                found, observation = observe()
                if found is not None:
                    break
            if found is not None:
                break
        if found is None:
            stage("SEARCH_EXHAUSTED")
            log["failure_reason"] = "No verified target within six reachable observation points; another search budget/observation area is needed"
            return log
        xyz, elevated = found
        log["target_estimate_xyz"] = xyz.tolist()
        log["verification_pose"] = robot.get_pose().tolist()
        stage("TARGET_VERIFIED")
        stage("APPROACH_PLANNING")
        start = robot.get_pose()[:2].copy()
        start_cell = visual_plan.planner.world_to_grid(start,resolution,arena_size,grid)
        if start_cell is None or grid[start_cell] != 0:
            raise RuntimeError("APPROACH_START_BLOCKED")
        goal, path, expanded, standoff, yaw, gap = visual_plan.select_parking(
            grid,start_cell,start,xyz[:2],np.array([observation.state.x_m,observation.state.y_m]),geometry,arena_size)
        log["stop_goal_xy"] = goal.tolist()
        log["planned_front_gap_m"] = gap
        log["parking_reason"] = ("Preferred gap reachable" if .08 <= gap <= .15 else
            "Nearest reachable safe cell on observed face; preserved map inflation/footprint limits approach")
        world_path = [visual_plan.planner.grid_to_world(c,resolution,arena_size) for c in path]
        log["final_path"] = world_path
        visual_plan.save_json(evidence_dir / "original_path.json", {"path": world_path})
        stage("APPROACH_NAVIGATION")
        motion.follow(
            world_path,
            yaw,
            [0.0, 0.8, 1.0],
            phase="approach_navigation",
            goal_tolerance_m=APPROACH_GOAL_TOLERANCE_M,
        )
        pose = robot.get_pose()
        metrics = approach_metrics(pose, xyz, geometry,
            p.getBasePositionAndOrientation(robot.robot_id)[0][2] + geometry["center_z_offset_m"])
        distance = float(np.linalg.norm(pose[:2]-goal))
        final_velocity = np.asarray(robot.get_velocity(), dtype=float)
        speed = float(np.linalg.norm(final_velocity[:2]))
        angular_speed = abs(float(final_velocity[2]))
        log["final_pose"] = pose.tolist()
        log["final_error_m"], log["final_speed_mps"] = distance, speed
        log["final_angular_speed_radps"] = angular_speed
        log["approach_metrics_before_reobservation"] = metrics
        # The oblique discovery view estimates a visible surface point, not
        # necessarily the surface now facing the robot. Reobserve ONCE after
        # stable parking; never call Qwen inside the control loop or move using GT.
        final_found, final_observation = observe()
        log["detections"][-1]["purpose"] = "FINAL_APPROACH_VERIFICATION"
        log["discovery_estimate_xyz"] = xyz.tolist()
        if final_found is None:
            raise RuntimeError("FINAL_TARGET_VERIFICATION_FAILED")
        final_xyz, elevated = final_found
        log["final_target_estimate_xyz"] = final_xyz.tolist()
        log["estimate_shift_m"] = float(np.linalg.norm(final_xyz-xyz))
        if log["estimate_shift_m"] > .45:
            raise RuntimeError("FINAL_TARGET_VERIFICATION_FAILED: estimate shift exceeds 0.45m")
        log["pre_docking_target_verified"] = True
        center_z = p.getBasePositionAndOrientation(robot.robot_id)[0][2] + geometry["center_z_offset_m"]
        log["approach_metrics_before_refinement"] = approach_metrics(pose, final_xyz, geometry, center_z)
        original_goal = goal.copy()
        goal_yaw = float(np.arctan2(*(final_xyz[:2]-goal)[::-1]))
        planned = approach_metrics(np.array([*goal, goal_yaw]), final_xyz, geometry, center_z)
        log["original_stop_revalidation"] = planned
        context=visual_plan.build_visual_docking_context(final_observation,
            log["detections"][-1],geometry,grid,arena_size)
        # Prefer docking directly from the current global safe pre-stop. If its
        # straight visual corridor is blocked, use the original A* pre-stop path.
        pre_xy=robot.get_pose()[:2].copy()
        try:
            local_goal,local_yaw=visual_plan.select_local_docking(pre_xy,final_xyz,context)
            correction_path=[pre_xy.tolist(),pre_xy.tolist()]
        except RuntimeError:
            local_goal,local_yaw=visual_plan.select_local_docking(original_goal,final_xyz,context)
            correction_path=terminal_path(grid,pre_xy,original_goal,resolution,arena_size)
        log["original_stop_goal_xy"]=original_goal.tolist()
        log["pre_docking_goal_xy"]=correction_path[-1]
        log["stop_goal_xy"]=local_goal.tolist()
        log["terminal_path"]=correction_path
        log["local_docking_corridor"]=[correction_path[-1],local_goal.tolist()]
        log["visual_local_geometry"]={"source":"validated_RGB_bbox_Depth", "target_sample_count":context["target_sample_count"]}
        visual_plan.save_json(evidence_dir/"terminal_path.json",{
            "pre_stop_path":correction_path,"visual_corridor":log["local_docking_corridor"]})
        stage("TERMINAL_DOCKING")
        motion.terminal_refine_to_goal(correction_path,final_xyz,geometry,context,observe)
        goal=np.asarray(log["stop_goal_xy"],dtype=float)
        final_xyz=np.asarray(log["terminal_target_estimate_xyz"],dtype=float)
        stage("TERMINAL_VISUAL_VERIFICATION")
        terminal_found, _ = observe()
        log["detections"][-1]["purpose"] = "POST_TERMINAL_VERIFICATION"
        log["final_target_verified"] = False
        if terminal_found is None:
            raise RuntimeError("FINAL_TARGET_VERIFICATION_FAILED")
        checked_xyz, elevated = terminal_found
        log["terminal_estimate_shift_m"] = float(np.linalg.norm(checked_xyz-final_xyz))
        if log["terminal_estimate_shift_m"] > .45:
            raise RuntimeError("FINAL_TARGET_VERIFICATION_FAILED: terminal estimate shift")
        log["post_docking_target_verified"] = True
        log["final_target_verified"] = True
        log["final_target_estimate_xyz"] = checked_xyz.tolist()
        log["target_estimate_xyz"] = checked_xyz.tolist()
        pose = robot.get_pose()
        metrics = approach_metrics(pose, checked_xyz, geometry,
            p.getBasePositionAndOrientation(robot.robot_id)[0][2]+geometry["center_z_offset_m"])
        velocity = np.asarray(robot.get_velocity(), dtype=float)
        distance = float(np.linalg.norm(pose[:2]-goal))
        speed, angular_speed = float(np.linalg.norm(velocity[:2])), abs(float(velocity[2]))
        log.update(final_pose=pose.tolist(), final_error_m=distance, final_speed_mps=speed,
                   final_angular_speed_radps=angular_speed, approach_metrics=metrics,
                   approach_metrics_after_refinement=metrics,
                   final_goal_error_m=distance,
                   final_heading_error_rad=metrics["front_heading_error_rad"])
        reached = (distance <= TERMINAL_PARAMETERS["position_tolerance_m"]
            and speed <= STOP_LINEAR_SPEED_TOLERANCE_MPS
            and angular_speed <= STOP_ANGULAR_SPEED_TOLERANCE_RADPS
            and log["terminal_stable_stop"] and not log["collision"])
        close = safe_terminal_metrics(metrics, elevated)
        log["reached_planned_stop"] = bool(reached)
        log["final_close_approach_passed"] = bool(close)
        log["task_success"] = bool(reached and close and log["post_docking_target_verified"]
            and log["terminal_alignment_completed"] and log["terminal_approach_completed"])
        if not close:
            gap = metrics["front_gap_projected_m"]
            reason = ("FINAL_GAP_TOO_LARGE" if gap > .15 else "FINAL_GAP_TOO_SMALL" if gap < .08
                      else "FINAL_ALIGNMENT_FAILED" if not metrics["facing_target"]
                      else "FINAL_LATERAL_OR_3D_REQUIREMENT_FAILED")
            raise RuntimeError(reason)
        if not reached:
            raise RuntimeError("TERMINAL_STABLE_STOP_FAILED")
        stage("TASK_SUCCESS")
        return log
    except Exception as exc:
        log["task_success"] = False
        log["failed_stage"] = log["stage"]
        log["failure_reason"] = str(exc)
        stage("TASK_FAILED")
        return log
    finally:
        if robot is not None and client >= 0 and p.isConnected(client):
            robot.set_velocity(0.0, 0.0)
        if motion is not None:
            try:
                motion.close()
            except Exception as exc:
                log["controller"]["close_error"] = str(exc)
        try:
            evaluate_completed_active_run(log, robot)
        except Exception as exc:
            # Evaluation cannot change a finished navigation decision or feed GT back.
            log["post_run_evaluation"] = {"status": "FAILED", "reason": str(exc)}
        visual_plan.save_json(evidence_dir / "terminal_docking_metrics.json", {
            key: value for key, value in log.items() if key.startswith("terminal_")
            or key.startswith("approach_metrics") or key in (
                "final_target_verified", "final_close_approach_passed", "task_success", "failure_reason")})
        persist()
        print(f"Task success: {log['task_success']}; reason: {log['failure_reason']}; log: {summary_path}",flush=True)
        if client >= 0 and p.isConnected(client):
            p.disconnect(client)


def cli():
    parser = argparse.ArgumentParser(description="P4 path execution or P4-5 active visual search")
    parser.add_argument("--mode", choices=("follow-path", "active-search"), default="follow-path")
    parser.add_argument("--scenario", choices=("occluded-blue", "occluded-green", "occluded-red"),
                        help="Occluder configuration; defaults to the target color from the instruction")
    parser.add_argument("--layout", choices=("train", "val", "test", "relocated"), default="test",
                        help="Semantic target placement; existing test layout remains the default")
    parser.add_argument("--instruction", help="Active-search instruction; otherwise prompt interactively")
    parser.add_argument(
        "--seed",
        type=int,
        choices=(42, 123, 2026),
        default=42,
        help="BC+PPO V3 checkpoint seed (default: 42)",
    )
    args = parser.parse_args()
    if args.mode == "follow-path":
        main()
    else:
        instruction = args.instruction
        if instruction is None:
            instruction = input("请输入导航指令（蓝色圆柱体、绿色球体或红色方块）：").strip()
        if not instruction.strip():
            parser.error("导航指令不能为空")
        result = active_search(
            instruction.strip(),
            args.scenario,
            args.layout,
            seed=args.seed,
        )
        if not result["task_success"]:
            raise SystemExit(1)


if __name__ == "__main__":
    cli()
