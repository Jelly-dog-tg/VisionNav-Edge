
"""P8.8: Active semantic search and approach using distilled ACT navigation."""

import json
import os
import argparse
import re
from dataclasses import asdict
import sys
import time
import importlib.util
from pathlib import Path

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

from visionnav.learning.act_semantic_controller import (
    ACTSemanticController, DEFAULT_ACT_CHECKPOINT
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


_recovery_module = load_module("p8_existing_geometric_recovery",
    ROOT / "experiments/27_follow_visual_path_bc_ppo.py")
_recovery_motion = _recovery_module.ActiveSearchMotion


# Navigation/stop tolerances are deliberately phase-specific.
# Search viewpoints use 5 cm precision; only the final
# approach may enter the stable-stop phase within 10 cm. Physical close-gap,
# heading, speed, and collision checks remain independent hard requirements.
SEARCH_GOAL_TOLERANCE_M = 0.050
APPROACH_GOAL_TOLERANCE_M = 0.100
FINAL_POSITION_TOLERANCE_M = 0.120
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
    "max_distance_m": 0.20, "max_steps": 600, "position_tolerance_m": 0.025,
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


def main(checkpoint_path=DEFAULT_ACT_CHECKPOINT):
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

        controller = ACTSemanticController(
            robot=robot,
            waypoints=waypoints,
            final_goal=stop_goal,
            checkpoint_path=checkpoint_path,
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

        print("\nP8.8 ACT SEMANTIC NAVIGATION")
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
        print("P8.8 ACT SEMANTIC NAVIGATION RESULT")
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
            print("\nP8.8 ACT SEMANTIC NAVIGATION PASSED")
        else:
            print("\nP8.8 ACT SEMANTIC NAVIGATION FAILED")

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
    """Wheel-driven active-search motion using one reusable distilled ACT controller."""

    def __init__(
        self,
        robot,
        ground_id,
        barrier_id,
        log,
        seed=42,
        checkpoint_path=DEFAULT_ACT_CHECKPOINT,
        goal_threshold=0.010,
    ):
        self.robot = robot
        self.ground_id = ground_id
        self.barrier_id = barrier_id
        self.log = log
        self.control_steps = 0
        self.physics_steps = 20
        self.previous_xy = robot.get_pose()[:2].copy()

        self.seed = int(seed)
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.goal_threshold = float(goal_threshold)
        self.controller = None

        self.log["controller"] = {
            "name": "Distilled ACT (BC+PPO teacher)",
            "seed": self.seed,
            "checkpoint": str(self.checkpoint_path),
            "goal_threshold_m": self.goal_threshold,
            "history_len": 8,
            "action_chunk_size": 16,
            "execution_steps": 4,
            "model_load_count": 0,
            "history_reset_count": 0,
            "phase_runs": [],
        }

    path_progress = staticmethod(_recovery_motion.path_progress)
    map_segment_free = _recovery_motion.map_segment_free
    command_map_safe = _recovery_motion.command_map_safe
    pure_pursuit_control = _recovery_motion.pure_pursuit_control

    def activate_recovery(self, points, pose, stats, reason):
        stats.update(fallback_triggered=True, fallback_reason=reason,
                     recovery_started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
                     recovery_started_control_step=self.control_steps)
        grid,resolution,arena=self.navigation_map
        path=np.asarray(terminal_path(grid,pose[:2],points[-1],resolution,arena))
        stats["recovery_path"]=path.tolist()
        return path

    def _prepare_controller(self, points, phase="navigation"):
        """Load ACT once; reset its temporal history whenever the A* leg changes."""
        points = np.asarray(points, dtype=np.float32)

        if self.controller is None:
            self.controller = ACTSemanticController(
                robot=self.robot,
                waypoints=points,
                final_goal=points[-1],
                checkpoint_path=self.checkpoint_path,
                goal_threshold=self.goal_threshold,
            )
            self.log["controller"]["model_load_count"] = 1
        self.controller.set_path(points, final_goal=points[-1],reset_reason=f"{phase}:new_independent_astar_path")

        self.log["controller"]["history_reset_count"] = self.controller.history_reset_count
        self.log["controller"]["history_reset_events"] = list(self.controller.history_reset_events)
        return self.controller

    def _record_phase_metrics(self, stats):
        steps = int(stats["steps"])
        record = {
            "phase": stats["phase"],
            "controller": "Distilled ACT (BC+PPO teacher)",
            "steps": steps,
            "mean_abs_action_v_mps": (
                stats["sum_abs_action_v_mps"] / steps if steps else 0.0
            ),
            "mean_abs_action_w_radps": (
                stats["sum_abs_action_w_radps"] / steps if steps else 0.0
            ),
            "action_clipped_fraction": (
                stats["action_clipped_steps"] / steps if steps else 0.0
            ),
            "chunk_replans": int(stats["chunk_replans"]),
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
        for key in ("history_reset_count_at_start","history_frames_at_start","pending_cleared_at_start",
                    "fallback_triggered","fallback_steps","fallback_reason","recovery_started_at_utc",
                    "recovery_started_control_step","recovery_path"):
            if key in stats:record[key]=stats[key]
        if stats.get("fallback_triggered"):
            record["recovery_ended_at_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
            record["recovery_ended_control_step"]=self.control_steps
        record["fallback_completion"]=bool(stats.get("fallback_triggered") and stats["completed"])
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

    def follow(
        self,
        waypoints,
        goal_yaw,
        color,
        phase="navigation",
        goal_tolerance_m=None,
    ):
        """Follow one A* leg with distilled ACT and phase-specific stop tolerance."""
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

        controller = self._prepare_controller(points,phase)
        # Keep the controller's reached flag synchronized with this phase.
        controller.goal_threshold = active_goal_tolerance

        phase_stats = {
            "phase": str(phase),
            "steps": 0,  # Number of ACT policy inference steps.
            "sum_abs_action_v_mps": 0.0,
            "sum_abs_action_w_radps": 0.0,
            "action_clipped_steps": 0,
            "chunk_replans": 0,
            "sum_path_deviation_m": 0.0,
            "goal_tolerance_m": active_goal_tolerance,
            "settle_steps": 0,
            "settled_consecutive_steps": 0,
            "final_goal_error_m": None,
            "final_speed_mps": None,
            "final_angular_speed_radps": None,
            "final_heading_error_rad": None,
            "completed": False,
            "history_reset_count_at_start":controller.history_reset_count,
            "history_frames_at_start":len(controller.act.history),
            "pending_cleared_at_start":controller.act.pending is None,
            "fallback_triggered":False,"fallback_steps":0,"fallback_completion":False,
        }

        progress_window=[];deviated=0;recovery_points=None;recovery_index=1
        phase_final_tolerance = active_goal_tolerance if str(phase).startswith("search") else FINAL_POSITION_TOLERANCE_M
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
                elif settling_mode and error > phase_final_tolerance:
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
                        stopped_distance <= phase_final_tolerance
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
                progress,deviation=self.path_progress(points,pose[:2])
                progress_window.append(progress);progress_window=progress_window[-80:]
                deviated=deviated+1 if deviation>.25 else 0
                stalled=len(progress_window)==80 and progress_window[-1]-progress_window[0]<.05
                if recovery_points is None and (stalled or deviated>=20):
                    recovery_points=self.activate_recovery(points,pose,phase_stats,
                        "PATH_PROGRESS_STALLED" if stalled else "PATH_DEVIATION")
                if recovery_points is not None:
                    v,w,recovery_index=self.pure_pursuit_control(pose,recovery_points,recovery_index)
                    phase_stats["fallback_steps"]+=1
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
                action_v = abs(float(diagnostics.get("v_mps", v)))
                action_w = abs(float(diagnostics.get("w_radps", w)))
                path_deviation = float(diagnostics.get("path_deviation_m", 0.0))

                phase_stats["steps"] += 1
                phase_stats["sum_abs_action_v_mps"] += action_v
                phase_stats["sum_abs_action_w_radps"] += action_w
                phase_stats["sum_path_deviation_m"] += path_deviation
                if diagnostics.get("action_clipped", False):
                    phase_stats["action_clipped_steps"] += 1
                if diagnostics.get("chunk_replanned", False):
                    phase_stats["chunk_replans"] += 1

                if not self.command_map_safe(pose,v,w):
                    recovery_points=self.activate_recovery(points,pose,phase_stats,"MAP_INFLATION_GUARD")
                    continue
                self.step(v, w)

            raise RuntimeError(
                "NAVIGATION_TIMEOUT: distilled ACT did not stably reach the requested stop"
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

    def terminal_refine_to_goal(self, points, target_xyz, geometry):
        """One bounded geometric docking leg, excluded from policy statistics."""
        cfg = TERMINAL_PARAMETERS
        goal = np.asarray(points[-1], dtype=float)
        points = np.asarray(points, dtype=float)
        stats = {"terminal_refinement_attempted": True, "terminal_refinement_steps": 0,
                 "terminal_refinement_distance_m": 0.0,
                 "terminal_refinement_final_error_m": None,
                 "terminal_refinement_collision": False, "terminal_stable_stop": False}
        self.log.update(stats)
        distance_start = self.log["actual_distance_m"]
        stable, waypoint, stale = 0, 1, 0
        best = float("inf")
        try:
            if np.linalg.norm(np.diff(points, axis=0), axis=1).sum() > cfg["max_distance_m"]:
                raise RuntimeError("TERMINAL_REFINEMENT_DISTANCE_EXCEEDED")
            for iteration in range(cfg["max_steps"]):
                pose = self.robot.get_pose()
                error = float(np.linalg.norm(pose[:2] - goal))
                self.log["terminal_refinement_final_error_m"] = error
                moved = self.log["actual_distance_m"] - distance_start
                if moved >= cfg["max_distance_m"]:
                    raise RuntimeError("TERMINAL_REFINEMENT_DISTANCE_EXCEEDED")
                while waypoint < len(points)-1 and np.linalg.norm(points[waypoint]-pose[:2]) < .015:
                    waypoint += 1
                # Position tolerance alone must not reproduce premature stopping.
                target_delta = target_xyz[:2] - pose[:2]
                target_yaw = float(np.arctan2(target_delta[1], target_delta[0]))
                facing_pose = np.array([pose[0], pose[1], target_yaw])
                facing_gap = approach_metrics(facing_pose, target_xyz, geometry)["front_gap_projected_m"]
                aligning = error <= cfg["position_tolerance_m"] and .08 <= facing_gap <= .15
                destination = target_xyz[:2] if aligning else points[waypoint]
                delta = destination - pose[:2]
                desired = float(np.arctan2(delta[1], delta[0]))
                heading_error = float(np.arctan2(np.sin(desired-pose[2]), np.cos(desired-pose[2])))
                velocity = np.asarray(self.robot.get_velocity(), dtype=float)
                settled = (aligning and abs(heading_error) <= STOP_HEADING_TOLERANCE_RAD
                    and np.linalg.norm(velocity[:2]) <= STOP_LINEAR_SPEED_TOLERANCE_MPS
                    and abs(velocity[2]) <= STOP_ANGULAR_SPEED_TOLERANCE_RADPS)
                stable = stable + 1 if settled else 0
                self.log["terminal_heading_error_rad"] = heading_error
                self.log["terminal_stable_steps"] = stable
                if stable >= cfg["stable_steps"]:
                    self.log["terminal_stable_stop"] = True
                    return
                w = float(np.clip(2*heading_error, -cfg["max_angular_radps"], cfg["max_angular_radps"]))
                v = 0.0 if aligning or abs(heading_error) > .12 else min(
                    cfg["max_linear_mps"], .8*float(np.linalg.norm(delta)))
                # Predict the next physical control interval, including a conservative
                # current-speed allowance. Never advance across the minimum gap.
                predicted = pose.copy()
                advance = max(v, float(np.linalg.norm(velocity[:2]))) * (10/240)
                predicted[:2] += advance*np.array([np.cos(pose[2]), np.sin(pose[2])])
                if moved + advance > cfg["max_distance_m"]:
                    raise RuntimeError("TERMINAL_REFINEMENT_DISTANCE_EXCEEDED")
                if v > 0 and approach_metrics(predicted, target_xyz, geometry)["front_gap_projected_m"] < .08:
                    raise RuntimeError("FINAL_GAP_TOO_SMALL")
                progress = error if not aligning else abs(heading_error)
                if progress < best - .0005:
                    best, stale = progress, 0
                else:
                    stale += 1
                if stale >= cfg["no_progress_steps"]:
                    raise RuntimeError("TERMINAL_REFINEMENT_NO_PROGRESS")
                if aligning and not getattr(self, "_terminal_aligning", False):
                    best, stale = float("inf"), 0
                self._terminal_aligning = aligning
                self.log["terminal_refinement_steps"] = iteration + 1
                self.step(v, w)
            raise RuntimeError("TERMINAL_REFINEMENT_TIMEOUT")
        except RuntimeError as exc:
            if self.log["collision"]:
                self.log["terminal_refinement_collision"] = True
                raise RuntimeError("TERMINAL_REFINEMENT_COLLISION") from exc
            raise
        finally:
            self.robot.set_velocity(0., 0.)
            self.log["terminal_refinement_distance_m"] = self.log["actual_distance_m"] - distance_start
            self.log["terminal_refinement_final_error_m"] = float(
                np.linalg.norm(self.robot.get_pose()[:2]-goal))

    def close(self):
        """Release the ACT controller after all semantic navigation phases."""
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
    description = visual_plan.understand_instruction(instruction).lower()
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



def evaluate_completed_active_run(log, robot):
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
        log["reached_planned_stop"] = False
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
        "reached_planned_stop": log["reached_planned_stop"],
        "collision": log["collision"], "task_success": log["task_success"]}


def active_search(instruction, scenario=None, layout="test", seed=42, checkpoint_path=DEFAULT_ACT_CHECKPOINT):
    """P4-5: one GUI world, bounded geometric search, then visual approach."""
    summary_path = ROOT / "logs/p8_8_act_semantic_summary.json"
    log = {"run_id": str(time.time_ns()), "mode": "active-search-act", "scenario": scenario,
           "instruction": instruction, "layout": layout, "stage": "INITIALIZING", "stages": [],
           "visited_viewpoint_count": 0, "visited_viewpoints": [], "detections": [],
           "search_paths": [], "final_path": [], "target_estimate_xyz": None,
           "stop_goal_xy": None, "actual_distance_m": 0.0,
           "collision": False, "task_success": False, "failure_reason": None,
           "controller": {"name": "Distilled ACT (BC+PPO teacher)", "seed": int(seed),
                          "checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
                          "history_len": 8, "action_chunk_size": 16,
                          "execution_steps": 4, "model_load_count": 0,
                          "history_reset_count": 0, "phase_runs": []},
           "control_tolerances": {
               "search_goal_tolerance_m": SEARCH_GOAL_TOLERANCE_M,
               "approach_goal_tolerance_m": APPROACH_GOAL_TOLERANCE_M,
               "final_position_tolerance_m": FINAL_POSITION_TOLERANCE_M,
               "linear_speed_tolerance_mps": STOP_LINEAR_SPEED_TOLERANCE_MPS,
               "angular_speed_tolerance_radps": STOP_ANGULAR_SPEED_TOLERANCE_RADPS,
               "heading_tolerance_rad": STOP_HEADING_TOLERANCE_RAD,
               "required_stable_steps": REQUIRED_SETTLE_STEPS,
           }}
    evidence_dir = ROOT / "logs/p8_8_act_semantic_evidence" / f"run_{log['run_id']}"
    evidence_dir.mkdir(parents=True, exist_ok=False)
    latest_summary_path = summary_path
    summary_path = evidence_dir / "summary.json"
    log["evidence_directory"] = str(evidence_dir)
    log["terminal_parameters"] = dict(TERMINAL_PARAMETERS)
    log.update(terminal_refinement_attempted=False, terminal_refinement_steps=0,
               terminal_refinement_distance_m=0., terminal_refinement_final_error_m=None,
               terminal_refinement_collision=False, final_target_verified=False,
               final_close_approach_passed=False, terminal_stable_stop=False,
               approach_metrics_before_refinement=None, approach_metrics_after_refinement=None)
    visual_plan.save_json(evidence_dir / "control_parameters.json",
                          {"terminal": TERMINAL_PARAMETERS, "navigation": log["control_tolerances"],
                           "seed": seed, "controller": "distilled_act",
                           "act_checkpoint": str(Path(checkpoint_path).expanduser().resolve()),
                           "history_len": 8, "action_chunk_size": 16,
                           "execution_steps": 4})
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
        log["controller"]["checkpoint"] = str(Path(checkpoint_path).expanduser().resolve())
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
        motion = ActiveSearchMotion(
            robot,
            ground_id,
            barrier_id,
            log,
            seed=seed,
            checkpoint_path=checkpoint_path,
            goal_threshold=SEARCH_GOAL_TOLERANCE_M,
        )
        motion.navigation_map=(grid,resolution,arena_size)
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
                      "camera": asdict(obs.camera), "validated": False, "status": "OBSERVING"}
            log["detections"].append(record)
            T = np.asarray(obs.camera.T_world_camera).reshape(4,4)
            yaw = float(np.arctan2(T[1,2], T[0,2]))
            for cell in geometric_visibility(grid, np.array([obs.state.x_m, obs.state.y_m]),
                                               yaw, resolution, arena_size):
                covered[cell] = True
            result = None
            try:
                result = visual_plan.evaluate_observation(obs, obs.rgb_path, obs.depth_path,
                    target, instruction, geometry, record, allow_verifier_bbox_fallback=True)
                if result is not None:
                    # Reject a hallucinated box on a screen using the ACTUAL
                    # requested color, after semantic class and RGB-D validation.
                    record["target"] = target
                    record["canonical_target"] = canonical_target
                    _recovery_module.verify_candidate_color(visual_plan.cv2.imread(str(obs.rgb_path)),
                        record["bbox_pixels"],target_attributes["color"],record)
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
                visual_plan.save_json(evidence_dir/"detection_records.json",log["detections"])
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
        log["final_target_verified"] = True
        center_z = p.getBasePositionAndOrientation(robot.robot_id)[0][2] + geometry["center_z_offset_m"]
        log["approach_metrics_before_refinement"] = approach_metrics(pose, final_xyz, geometry, center_z)
        original_goal = goal.copy()
        goal_yaw = float(np.arctan2(*(final_xyz[:2]-goal)[::-1]))
        planned = approach_metrics(np.array([*goal, goal_yaw]), final_xyz, geometry, center_z)
        log["original_stop_revalidation"] = planned
        if not safe_terminal_metrics(planned, elevated):
            start = robot.get_pose()[:2].copy()
            start_cell = visual_plan.planner.world_to_grid(start, resolution, arena_size, grid)
            if start_cell is None or grid[start_cell] != 0:
                raise RuntimeError("NO_SAFE_TERMINAL_PATH")
            try:
                goal, _, _, _, _, _ = visual_plan.select_parking(
                    grid, start_cell, start, final_xyz[:2],
                    np.array([final_observation.state.x_m, final_observation.state.y_m]), geometry, arena_size)
            except RuntimeError as exc:
                raise RuntimeError("NO_SAFE_TERMINAL_PATH") from exc
            goal_yaw = float(np.arctan2(*(final_xyz[:2]-goal)[::-1]))
            planned = approach_metrics(np.array([*goal, goal_yaw]), final_xyz, geometry, center_z)
            if not safe_terminal_metrics(planned, elevated):
                raise RuntimeError("NO_SAFE_TERMINAL_PATH: no parking point meets strict gap/3D requirements")
        log["original_stop_goal_xy"] = original_goal.tolist()
        log["stop_goal_xy"] = goal.tolist()
        correction_path = terminal_path(grid, robot.get_pose()[:2], goal, resolution, arena_size)
        log["terminal_path"] = correction_path
        visual_plan.save_json(evidence_dir / "original_path.json", {"path": world_path})
        visual_plan.save_json(evidence_dir / "terminal_path.json", {"path": correction_path})
        stage("TERMINAL_DOCKING")
        motion.terminal_refine_to_goal(correction_path, final_xyz, geometry)
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
                   approach_metrics_after_refinement=metrics)
        reached = (distance <= TERMINAL_PARAMETERS["position_tolerance_m"]
            and speed <= STOP_LINEAR_SPEED_TOLERANCE_MPS
            and angular_speed <= STOP_ANGULAR_SPEED_TOLERANCE_RADPS
            and log["terminal_stable_stop"] and not log["collision"])
        close = safe_terminal_metrics(metrics, elevated)
        log["reached_planned_stop"] = bool(reached)
        log["final_close_approach_passed"] = bool(close)
        log["task_success"] = bool(reached and close and log["final_target_verified"])
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
        visual_plan.save_json(evidence_dir/"controller_statistics.json",log["controller"])
        persist()
        print(f"Task success: {log['task_success']}; reason: {log['failure_reason']}; log: {summary_path}",flush=True)
        if client >= 0 and p.isConnected(client):
            p.disconnect(client)


def cli():
    parser = argparse.ArgumentParser(description="P8.8 distilled ACT semantic navigation")
    parser.add_argument("--mode", choices=("follow-path", "active-search"), default="follow-path")
    parser.add_argument("--scenario", choices=("occluded-blue", "occluded-green", "occluded-red"),
                        help="Occluder configuration; defaults to the target color from the instruction")
    parser.add_argument("--layout", choices=("train", "val", "test", "relocated"), default="test",
                        help="Semantic target placement; existing test layout remains the default")
    parser.add_argument("--instruction", help="Active-search instruction; otherwise prompt interactively")
    parser.add_argument("--model", type=Path, default=DEFAULT_ACT_CHECKPOINT,
                        help="Distilled ACT checkpoint (default: models/act_distilled/act_best.pth)")
    parser.add_argument(
        "--seed",
        type=int,
        choices=(42, 123, 2026),
        default=42,
        help="Run seed retained for repeatability metadata (default: 42)",
    )
    args = parser.parse_args()
    if args.mode == "follow-path":
        main(args.model)
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
            checkpoint_path=args.model,
        )
        if not result["task_success"]:
            raise SystemExit(1)


if __name__ == "__main__":
    cli()
