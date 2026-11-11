
"""P6-1: Gymnasium environment for PPO robot navigation."""

import importlib.util
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from scripts.diff_drive_robot import DiffDriveRobot
from visionnav.learning import collect_expert as expert


ROOT = Path(__file__).resolve().parents[2]

MAX_LINEAR_V = 0.2613
MAX_ANGULAR_W = 2.0

PHYSICS_STEPS = 10
PHYSICS_DT = 1.0 / 240.0

GOAL_TOLERANCE = 0.12
SPEED_TOLERANCE = 0.05

MAX_STEPS = 2500
INITIAL_SETTLE_STEPS = 120

WAYPOINT_TOLERANCE = 0.08


def load_environment_builder():

    path = ROOT / "scripts" / "05_environment_builder.py"

    spec = importlib.util.spec_from_file_location(
        "rl_warehouse_builder",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.EnvironmentBuilder


EnvironmentBuilder = load_environment_builder()


class RobotNavigationEnv(gym.Env):
    """
    Continuous-action mobile robot navigation environment.

    Observation:
        36 LiDAR values
        2 relative waypoint coordinates
        1 linear velocity
        1 angular velocity

    Action:
        1 normalized linear velocity
        1 normalized angular velocity
    """

    metadata = {"render_modes": []}

    def __init__(self):

        super().__init__()

        # Observation: 40 dimensions.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(40,),
            dtype=np.float32,
        )

        # Action: normalized continuous commands.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        self.client_id = p.connect(p.DIRECT)

        if self.client_id < 0:
            raise RuntimeError(
                "Failed to connect to PyBullet."
            )

        self.robot = None
        self.grid = None
        self.waypoints = None

        self.goal_xy = None
        self.waypoint_index = 0

        self.step_count = 0
        self.stable_cycles = 0
        self.previous_goal_distance = None

        self.closed = False

    def _build_scene(self):

        p.resetSimulation()

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )

        p.setGravity(0, 0, -9.81)
        p.setTimeStep(PHYSICS_DT)
        p.setRealTimeSimulation(0)

        ground_id = p.loadURDF("plane.urdf")

        builder = EnvironmentBuilder(self.client_id)

        builder.create_warehouse_environment(
            arena_size=expert.ARENA_SIZE
        )

        obstacles = expert.occupancy.collect_obstacle_aabbs(
            ground_id
        )

        self.grid = expert.occupancy.create_grid(
            obstacles,
            expert.ARENA_SIZE,
        )

    def _get_velocity_components(self, velocity):

        velocity = np.asarray(
            velocity,
            dtype=float,
        ).reshape(-1)

        if velocity.size >= 3:

            linear = float(
                np.linalg.norm(velocity[:2])
            )

            angular = abs(float(velocity[2]))

        elif velocity.size == 2:

            linear = abs(float(velocity[0]))
            angular = abs(float(velocity[1]))

        else:

            raise ValueError(
                "Unexpected velocity format."
            )

        return linear, angular

    def _get_observation(self):

        pose = self.robot.get_pose()

        velocity = self.robot.get_velocity()

        lidar = np.asarray(
            self.robot.get_lidar_scan(),
            dtype=np.float32,
        )

        waypoint = self.waypoints[
            self.waypoint_index
        ]

        observation = expert.make_observation(
            pose,
            waypoint,
            lidar,
            velocity,
        )

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation.shape != (40,):
            raise RuntimeError(
                f"Invalid observation shape: {observation.shape}"
            )

        if not np.isfinite(observation).all():
            raise RuntimeError(
                "Observation contains NaN or Inf."
            )

        return observation

    def _update_waypoint(self, position):

        while self.waypoint_index < len(self.waypoints) - 1:

            distance = np.linalg.norm(
                position - self.waypoints[self.waypoint_index]
            )

            if distance >= WAYPOINT_TOLERANCE:
                break

            self.waypoint_index += 1

    
    def _remaining_path_distance(self, position):
        """Calculate remaining distance along the A* path."""

        position = np.asarray(position, dtype=float)

        index = self.waypoint_index

        # Distance from current position to current waypoint.
        remaining = float(
            np.linalg.norm(
                position - self.waypoints[index]
            )
        )

        # Add all remaining path segments.
        if index < len(self.waypoints) - 1:

            segments = np.diff(
                self.waypoints[index:],
                axis=0,
            )

            remaining += float(
                np.linalg.norm(
                    segments,
                    axis=1,
                ).sum()
            )

        return remaining

    
    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self._build_scene()

        # ------------------------------------------
        # 1. Load fixed benchmark task or sample task
        # ------------------------------------------

        if options is not None and "task" in options:

            task = options["task"]

            start_xy = np.asarray(
                task["start_xy"],
                dtype=np.float32,
            )

            self.goal_xy = np.asarray(
                task["goal_xy"],
                dtype=np.float32,
            )

            self.waypoints = np.asarray(
                task["waypoints_world"],
                dtype=np.float32,
            )

            start_yaw = float(task["start_yaw_rad"])

            attempts = int(
                task.get("sampling_attempts", 0)
            )

            expanded = int(
                task.get("expanded_nodes", 0)
            )

        else:

            (
                start_cell,
                goal_cell,
                path,
                start_yaw,
                attempts,
                expanded,
            ) = expert.sample_task(
                self.grid,
                self.np_random,
                set(),
            )

            start_xy = expert.planner.grid_to_world(
                start_cell,
                expert.occupancy.RESOLUTION,
                expert.ARENA_SIZE,
            )

            self.goal_xy = np.asarray(
                expert.planner.grid_to_world(
                    goal_cell,
                    expert.occupancy.RESOLUTION,
                    expert.ARENA_SIZE,
                ),
                dtype=np.float32,
            )

            self.waypoints = np.asarray(
                [
                    expert.planner.grid_to_world(
                        node,
                        expert.occupancy.RESOLUTION,
                        expert.ARENA_SIZE,
                    )
                    for node in path
                ],
                dtype=np.float32,
            )

        # ------------------------------------------
        # 2. Validate task data
        # ------------------------------------------

        start_xy = np.asarray(
            start_xy,
            dtype=np.float32,
        )

        if start_xy.shape != (2,):
            raise ValueError("Invalid start position")

        if self.goal_xy.shape != (2,):
            raise ValueError("Invalid goal position")

        if (
            self.waypoints.ndim != 2
            or self.waypoints.shape[1] != 2
            or len(self.waypoints) == 0
        ):
            raise ValueError("Invalid waypoint array")

        if not (
            np.isfinite(start_xy).all()
            and np.isfinite(self.goal_xy).all()
            and np.isfinite(self.waypoints).all()
        ):
            raise ValueError("Task contains NaN or Inf")

        # ------------------------------------------
        # 3. Reset navigation state
        # ------------------------------------------

        self.waypoint_index = 0
        self.step_count = 0
        self.stable_cycles = 0

        # ------------------------------------------
        # 4. Create robot
        # ------------------------------------------

        self.robot = DiffDriveRobot(
            str(ROOT / "urdf" / "diff_drive_robot.urdf"),
            start_pos=[
                float(start_xy[0]),
                float(start_xy[1]),
                0.1,
            ],
            start_yaw=float(start_yaw),
        )

        for _ in range(INITIAL_SETTLE_STEPS):
            p.stepSimulation()

        pose = self.robot.get_pose()

        position = np.asarray(
            pose[:2],
            dtype=float,
        )

        self._update_waypoint(position)

        self.previous_waypoint_distance = float(
            np.linalg.norm(
                position - self.waypoints[self.waypoint_index]
            )
        )

        # ------------------------------------------
        # 5. Generate initial observation
        # ------------------------------------------

        observation = self._get_observation()

        info = {
            "start_xy": start_xy.tolist(),
            "goal_xy": self.goal_xy.tolist(),
            "path_nodes": len(self.waypoints),
            "expanded_nodes": int(expanded),
            "sampling_attempts": int(attempts),
        }

        return observation, info


    
    def step(self, action):

        action = np.asarray(
            action,
            dtype=np.float32,
        )

        action = np.clip(
            action,
            -1.0,
            1.0,
        )

        # 1. Convert normalized actions to physical velocities.
        v = (
            (float(action[0]) + 1.0)
            * 0.5
            * MAX_LINEAR_V
        )

        w = (
            float(action[1])
            * MAX_ANGULAR_W
        )

        # 2. Execute robot movement.
        self.robot.set_velocity(v, w)

        collision = False

        for _ in range(PHYSICS_STEPS):

            p.stepSimulation()

            if self.robot.check_collision():
                collision = True
                break

        self.step_count += 1

        # 3. Read the new robot state.
        pose = self.robot.get_pose()

        position = np.asarray(
            pose[:2],
            dtype=float,
        )

        velocity = self.robot.get_velocity()

        linear_speed, angular_speed = (
            self._get_velocity_components(velocity)
        )

        goal_distance = float(
            np.linalg.norm(
                position - self.goal_xy
            )
        )

        # 4. Calculate progress toward current waypoint.
        waypoint = self.waypoints[
            self.waypoint_index
        ]

        current_waypoint_distance = float(
            np.linalg.norm(
                position - waypoint
            )
        )

        progress = (
            self.previous_waypoint_distance
            - current_waypoint_distance
        )

        reward = 20.0 * float(
            np.clip(progress, -0.1, 0.1)
        )

        # Small time penalty.
        reward -= 0.01

        # 5. Advance waypoint only after reaching it.
        if current_waypoint_distance < WAYPOINT_TOLERANCE:

            if self.waypoint_index < len(self.waypoints) - 1:

                reward += 1.0

                self.waypoint_index += 1

                next_waypoint = self.waypoints[
                    self.waypoint_index
                ]

                self.previous_waypoint_distance = float(
                    np.linalg.norm(
                        position - next_waypoint
                    )
                )

            else:

                self.previous_waypoint_distance = (
                    current_waypoint_distance
                )

        else:

            self.previous_waypoint_distance = (
                current_waypoint_distance
            )

        # 6. Obstacle avoidance penalty.
        lidar = np.asarray(
            self.robot.get_lidar_scan(),
            dtype=np.float32,
        )

        min_lidar = float(
            np.min(lidar)
        )

        if min_lidar < 0.35:

            reward -= (
                0.2
                * (0.35 - min_lidar)
                / 0.35
            )

        # 7. Penalize excessive angular actions.
        reward -= 0.001 * float(
            action[1] ** 2
        )

        # 8. Handle collision.
        if collision:

            reward -= 50.0

            terminated = True
            truncated = False

        else:

            # 9. Check stable arrival at final goal.
            stable = (
                goal_distance <= GOAL_TOLERANCE
                and linear_speed <= SPEED_TOLERANCE
                and angular_speed <= SPEED_TOLERANCE
            )

            if stable:
                self.stable_cycles += 1
            else:
                self.stable_cycles = 0

            success = (
                self.stable_cycles >= 10
            )

            if success:
                reward += 50.0

            terminated = bool(success)

            truncated = (
                not terminated
                and self.step_count >= MAX_STEPS
            )

        # 10. Generate next observation.
        observation = self._get_observation()

        info = {
            "success": bool(
                terminated and not collision
            ),
            "collision": bool(collision),
            "goal_distance_m": goal_distance,
            "linear_speed_mps": linear_speed,
            "angular_speed_radps": angular_speed,
            "waypoint_index": self.waypoint_index,
            "steps": self.step_count,
            "reward": float(reward),
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )


    def close(self):

        if not self.closed:

            if p.isConnected(self.client_id):
                p.disconnect(self.client_id)

            self.closed = True
