
"""Run a closed-loop Behavior Cloning navigation episode."""

import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data

from scripts.diff_drive_robot import DiffDriveRobot
from visionnav.learning.bc_policy import BehaviorCloningPolicy
from visionnav.learning import collect_expert as expert_data


ROOT = Path(__file__).resolve().parents[2]

ARENA_SIZE = expert_data.ARENA_SIZE
PHYSICS_DT = expert_data.PHYSICS_DT
PHYSICS_STEPS = expert_data.PHYSICS_STEPS

MAX_CONTROL_STEPS = 2500

FINAL_TOLERANCE = 0.12
SPEED_TOLERANCE = 0.05
REQUIRED_STABLE_CYCLES = 10

# Match the action range in the expert demonstrations.
MAX_LINEAR_V = 0.2613
MAX_ANGULAR_W = 2.0

SEED = 2026


def draw_path(waypoints):
    """Draw the A* global path in the PyBullet GUI."""

    for start, end in zip(waypoints[:-1], waypoints[1:]):
        p.addUserDebugLine(
            [float(start[0]), float(start[1]), 0.04],
            [float(end[0]), float(end[1]), 0.04],
            lineColorRGB=[0, 1, 1],
            lineWidth=3,
        )


def main():

    if p.isConnected():
        raise RuntimeError("An existing PyBullet connection was detected")

    client = p.connect(p.GUI)

    if client != 0:
        raise RuntimeError("Failed to create default GUI connection")

    robot = None

    try:
        # -------------------------------------------------
        # 1. Build the warehouse environment
        # -------------------------------------------------

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(PHYSICS_DT)
        p.setRealTimeSimulation(0)

        p.resetDebugVisualizerCamera(
            cameraDistance=16,
            cameraYaw=45,
            cameraPitch=-50,
            cameraTargetPosition=[0, 0, 0],
        )

        ground_id = p.loadURDF("plane.urdf")

        expert_data.EnvironmentBuilder(
            client
        ).create_warehouse_environment(
            arena_size=ARENA_SIZE
        )

        print("\nWarehouse environment ready.")

        # -------------------------------------------------
        # 2. Build the occupancy grid and plan a path
        # -------------------------------------------------

        obstacles = expert_data.occupancy.collect_obstacle_aabbs(
            ground_id
        )

        grid = expert_data.occupancy.create_grid(
            obstacles,
            ARENA_SIZE,
        )

        rng = np.random.default_rng(SEED)

        start, goal, path, yaw, attempts, expanded = (
            expert_data.sample_task(
                grid,
                rng,
                set(),
            )
        )

        start_xy = np.asarray(
            expert_data.planner.grid_to_world(
                start,
                expert_data.occupancy.RESOLUTION,
                ARENA_SIZE,
            ),
            dtype=float,
        )

        goal_xy = np.asarray(
            expert_data.planner.grid_to_world(
                goal,
                expert_data.occupancy.RESOLUTION,
                ARENA_SIZE,
            ),
            dtype=float,
        )

        waypoints = np.asarray(
            [
                expert_data.planner.grid_to_world(
                    node,
                    expert_data.occupancy.RESOLUTION,
                    ARENA_SIZE,
                )
                for node in path
            ],
            dtype=float,
        )

        planned_length = (
            expert_data.planner.calculate_path_length(
                path,
                expert_data.occupancy.RESOLUTION,
            )
        )

        draw_path(waypoints)

        p.addUserDebugText(
            "START",
            [start_xy[0], start_xy[1], 0.3],
            textColorRGB=[0, 1, 0],
            textSize=1.5,
        )

        p.addUserDebugText(
            "GOAL",
            [goal_xy[0], goal_xy[1], 0.3],
            textColorRGB=[1, 0, 0],
            textSize=1.5,
        )

        print("\nA* PATH PLANNED")
        print("Start:", start_xy)
        print("Goal:", goal_xy)
        print("Path length:", round(planned_length, 3), "m")
        print("Expanded nodes:", expanded)

        # -------------------------------------------------
        # 3. Create the robot
        # -------------------------------------------------

        robot = DiffDriveRobot(
            str(ROOT / "urdf" / "diff_drive_robot.urdf"),
            start_pos=[*start_xy, 0.1],
            start_yaw=yaw,
        )

        robot.set_velocity(0.0, 0.0)

        for _ in range(expert_data.INITIAL_SETTLE_STEPS):
            p.stepSimulation()

        print("\nRobot initialized.")

        # -------------------------------------------------
        # 4. Load the trained BC policy
        # -------------------------------------------------

        policy = BehaviorCloningPolicy()

        print("\nBC NAVIGATION STARTED")
        print("---------------------")

        # -------------------------------------------------
        # 5. Closed-loop navigation using the neural network
        # -------------------------------------------------

        waypoint_index = 0
        stable_cycles = 0

        collision = False
        success = False

        total_distance = 0.0
        previous_xy = robot.get_pose()[:2].copy()

        for step in range(1, MAX_CONTROL_STEPS + 1):

            # A. Read the current robot state.
            pose = robot.get_pose()
            velocity = robot.get_velocity()

            # B. Advance to the next waypoint when close enough.
            while (
                waypoint_index < len(waypoints) - 1
                and np.linalg.norm(
                    pose[:2] - waypoints[waypoint_index]
                ) < 0.12
            ):
                waypoint_index += 1

            waypoint = waypoints[waypoint_index]

            # C. Obtain the current sensor measurements.
            lidar = np.asarray(
                robot.get_lidar_scan(),
                dtype=np.float32,
            )

            # D. Construct the same 40D observation used in training.
            observation = expert_data.make_observation(
                pose,
                waypoint,
                lidar,
                velocity,
            )

            # E. Let the neural network predict the action.
            predicted_action = policy.predict(observation)

            raw_v = float(predicted_action[0])
            raw_w = float(predicted_action[1])

            # F. Apply safety limits matching the demonstration range.
            v = float(np.clip(raw_v, 0.0, MAX_LINEAR_V))
            w = float(np.clip(raw_w, -MAX_ANGULAR_W, MAX_ANGULAR_W))

            # G. Execute the predicted action.
            robot.set_velocity(v, w)

            for _ in range(PHYSICS_STEPS):
                p.stepSimulation()

                if expert_data.has_collision(
                    robot.robot_id,
                    ground_id,
                ):
                    collision = True
                    break

            if collision:
                print("\nCOLLISION DETECTED")
                break

            # H. Evaluate the actual robot movement.
            current_pose = robot.get_pose()
            current_velocity = robot.get_velocity()

            total_distance += float(
                np.linalg.norm(
                    current_pose[:2] - previous_xy
                )
            )

            previous_xy = current_pose[:2].copy()

            goal_error = float(
                np.linalg.norm(
                    current_pose[:2] - goal_xy
                )
            )

            stable = (
                waypoint_index == len(waypoints) - 1
                and goal_error <= FINAL_TOLERANCE
                and np.linalg.norm(current_velocity[:2])
                <= SPEED_TOLERANCE
                and abs(current_velocity[2])
                <= SPEED_TOLERANCE
            )

            stable_cycles = (
                stable_cycles + 1 if stable else 0
            )

            # I. Print navigation progress.
            if step % 100 == 0:
                print(
                    f"Step {step:4d} | "
                    f"Waypoint {waypoint_index + 1}/{len(waypoints)} | "
                    f"Goal error: {goal_error:.3f} m | "
                    f"v={v:+.3f} | w={w:+.3f}"
                )

            # J. Check whether the robot has reached the goal.
            if stable_cycles >= REQUIRED_STABLE_CYCLES:
                success = True
                break

            # Maintain approximately real-time GUI playback.
            time.sleep(PHYSICS_STEPS * PHYSICS_DT)

        # -------------------------------------------------
        # 6. Stop and report the navigation result
        # -------------------------------------------------

        robot.set_velocity(0.0, 0.0)

        for _ in range(10):
            p.stepSimulation()

        final_pose = robot.get_pose()
        final_velocity = robot.get_velocity()

        final_error = float(
            np.linalg.norm(final_pose[:2] - goal_xy)
        )

        print("\nBC NAVIGATION RESULT")
        print("--------------------")
        print("Steps:", step)
        print("Success:", success)
        print("Collision:", collision)
        print("Final position:", np.round(final_pose[:2], 3))
        print("Goal position:", np.round(goal_xy, 3))
        print("Final goal error:", round(final_error, 4), "m")
        print("Travelled distance:", round(total_distance, 3), "m")
        print("Final speed:", round(
            float(np.linalg.norm(final_velocity[:2])), 4
        ), "m/s")

        if success:
            print("\nBC NAVIGATION PASSED")
        else:
            print("\nBC NAVIGATION NOT PASSED")

        input("\nPress Enter to close the simulation...")

    finally:
        if robot is not None and p.isConnected():
            robot.set_velocity(0.0, 0.0)

        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()
