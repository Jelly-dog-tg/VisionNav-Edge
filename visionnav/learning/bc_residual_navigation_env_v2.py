"""P6-RL V2: BC baseline with PPO residual control and lookahead features.

BC receives the original 40-dimensional observation.
PPO receives a 44-dimensional observation:
    0:40  original observation
    40    lookahead target x in robot frame (m)
    41    lookahead target y in robot frame (m)
    42    distance to the remaining A* path (m)
    43    distance to the final goal (m)

The original RobotNavigationEnv, BC checkpoint, and V1 environment remain
unchanged. This module is intended for a new V2 PPO model.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from visionnav.learning.bc_policy import BehaviorCloningPolicy
from visionnav.learning.rl_navigation_env import (
    RobotNavigationEnv,
    MAX_LINEAR_V,
    MAX_ANGULAR_W,
)


class BCResidualNavigationEnvV2(gym.Env):
    """BC + PPO residual environment with a 44-D PPO observation."""

    metadata = {"render_modes": []}

    RAW_OBSERVATION_DIM = 40
    PPO_OBSERVATION_DIM = 44

    def __init__(
        self,
        checkpoint_path=None,
        delta_v_max=0.05,
        delta_w_max=0.4,
        lookahead_distance_m=0.5,
    ):
        super().__init__()

        self.base_env = RobotNavigationEnv()

        if checkpoint_path is None:
            self.bc_policy = BehaviorCloningPolicy()
        else:
            self.bc_policy = BehaviorCloningPolicy(checkpoint_path)

        self.delta_v_max = float(delta_v_max)
        self.delta_w_max = float(delta_w_max)
        self.lookahead_distance_m = float(lookahead_distance_m)

        if self.delta_v_max <= 0.0 or self.delta_w_max <= 0.0:
            raise ValueError("Residual action limits must be positive.")

        if self.lookahead_distance_m <= 0.0:
            raise ValueError("lookahead_distance_m must be positive.")

        # PPO action: normalized linear and angular residual corrections.
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # PPO sees 40 original values + 4 path/goal features.
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.PPO_OBSERVATION_DIM,),
            dtype=np.float32,
        )

        # BC must continue to receive the original 40-D observation.
        self.current_observation = None
        self.current_ppo_observation = None

        self.previous_goal_distance = None
        self.previous_remaining_path_distance = None

    # ------------------------------------------------------------------
    # Path geometry
    # ------------------------------------------------------------------

    def _get_remaining_waypoints(self):
        """Return the uncompleted part of the A* path in world coordinates."""
        waypoints = np.asarray(
            self.base_env.waypoints,
            dtype=np.float32,
        )

        if (
            waypoints.ndim != 2
            or waypoints.shape[1] != 2
            or len(waypoints) == 0
            or not np.isfinite(waypoints).all()
        ):
            raise ValueError("The base environment has an invalid A* path.")

        index = int(
            np.clip(
                self.base_env.waypoint_index,
                0,
                len(waypoints) - 1,
            )
        )
        return waypoints[index:]

    def _project_onto_remaining_path(self, position):
        """Project a world-frame position onto the nearest remaining path segment.

        Returns:
            projection_xy, segment_index, distance_m
        """
        position = np.asarray(position, dtype=np.float32)
        if position.shape != (2,) or not np.isfinite(position).all():
            raise ValueError("position must be a finite 2-D vector.")

        path = self._get_remaining_waypoints()

        if len(path) == 1:
            projection = path[0].copy()
            distance = float(np.linalg.norm(position - projection))
            return projection, 0, distance

        best_projection = path[0].copy()
        best_segment_index = 0
        best_distance = float("inf")

        for index, (start, end) in enumerate(zip(path[:-1], path[1:])):
            segment = end - start
            denominator = float(np.dot(segment, segment))

            if denominator < 1e-12:
                ratio = 0.0
            else:
                ratio = float(
                    np.dot(position - start, segment) / denominator
                )
                ratio = float(np.clip(ratio, 0.0, 1.0))

            projection = start + ratio * segment
            distance = float(np.linalg.norm(position - projection))

            if distance < best_distance:
                best_distance = distance
                best_projection = projection.astype(np.float32)
                best_segment_index = index

        return best_projection, best_segment_index, best_distance

    def _select_lookahead_point(self, position):
        """Select a point lookahead_distance_m ahead along the remaining A* path."""
        path = self._get_remaining_waypoints()

        if len(path) == 1:
            return path[0].copy()

        projection, segment_index, _ = (
            self._project_onto_remaining_path(position)
        )

        remaining_distance = self.lookahead_distance_m
        cursor = projection.copy()

        # Start at the nearest projected point, then walk forward on the polyline.
        for endpoint_index in range(segment_index + 1, len(path)):
            endpoint = path[endpoint_index]
            segment = endpoint - cursor
            segment_length = float(np.linalg.norm(segment))

            if segment_length < 1e-8:
                cursor = endpoint.copy()
                continue

            if segment_length >= remaining_distance:
                ratio = remaining_distance / segment_length
                return (cursor + ratio * segment).astype(np.float32)

            remaining_distance -= segment_length
            cursor = endpoint.copy()

        # The remaining path is shorter than the requested lookahead distance.
        return path[-1].copy()

    def _get_path_metrics(self, position, yaw):
        """Calculate lookahead, path deviation, goal distance, and heading error."""
        position = np.asarray(position, dtype=np.float32)
        yaw = float(yaw)

        lookahead_point = self._select_lookahead_point(position)
        target_vector_world = lookahead_point - position

        # World-frame vector -> robot frame (x forward, y left).
        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))

        lookahead_x_robot = (
            cos_yaw * float(target_vector_world[0])
            + sin_yaw * float(target_vector_world[1])
        )
        lookahead_y_robot = (
            -sin_yaw * float(target_vector_world[0])
            + cos_yaw * float(target_vector_world[1])
        )

        target_distance = float(np.linalg.norm(target_vector_world))
        if target_distance < 1e-8:
            heading_error = 0.0
        else:
            target_heading = float(
                np.arctan2(
                    target_vector_world[1],
                    target_vector_world[0],
                )
            )
            heading_error = float(
                np.arctan2(
                    np.sin(target_heading - yaw),
                    np.cos(target_heading - yaw),
                )
            )

        _, _, path_deviation = self._project_onto_remaining_path(position)

        goal_distance = float(
            np.linalg.norm(position - self.base_env.goal_xy)
        )

        remaining_path_distance = float(
            self.base_env._remaining_path_distance(position)
        )

        return {
            "lookahead_point": lookahead_point,
            "lookahead_local": np.asarray(
                [lookahead_x_robot, lookahead_y_robot],
                dtype=np.float32,
            ),
            "path_deviation": path_deviation,
            "goal_distance": goal_distance,
            "remaining_path_distance": remaining_path_distance,
            "heading_error": heading_error,
        }

    def _augment_ppo_observation(self, raw_observation, metrics):
        """Append four geometric features to the original 40-D observation."""
        raw_observation = np.asarray(
            raw_observation,
            dtype=np.float32,
        )

        if raw_observation.shape != (self.RAW_OBSERVATION_DIM,):
            raise ValueError(
                "Expected raw BC observation shape "
                f"({self.RAW_OBSERVATION_DIM},), got {raw_observation.shape}."
            )

        if not np.isfinite(raw_observation).all():
            raise ValueError("Raw observation contains NaN or Inf.")

        extra_features = np.asarray(
            [
                metrics["lookahead_local"][0],
                metrics["lookahead_local"][1],
                metrics["path_deviation"],
                metrics["goal_distance"],
            ],
            dtype=np.float32,
        )

        ppo_observation = np.concatenate(
            [raw_observation, extra_features]
        ).astype(np.float32)

        if ppo_observation.shape != (self.PPO_OBSERVATION_DIM,):
            raise RuntimeError(
                f"Invalid PPO observation shape: {ppo_observation.shape}"
            )

        if not np.isfinite(ppo_observation).all():
            raise RuntimeError("PPO observation contains NaN or Inf.")

        return ppo_observation

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        raw_observation, info = self.base_env.reset(
            seed=seed,
            options=options,
        )

        raw_observation = np.asarray(
            raw_observation,
            dtype=np.float32,
        )

        if raw_observation.shape != (self.RAW_OBSERVATION_DIM,):
            raise RuntimeError(
                f"Base environment must return 40-D observations, "
                f"got {raw_observation.shape}."
            )

        self.current_observation = raw_observation.copy()

        pose = self.base_env.robot.get_pose()
        position = np.asarray(pose[:2], dtype=np.float32)
        yaw = float(pose[2])

        metrics = self._get_path_metrics(position, yaw)

        self.previous_goal_distance = metrics["goal_distance"]
        self.previous_remaining_path_distance = (
            metrics["remaining_path_distance"]
        )

        self.current_ppo_observation = (
            self._augment_ppo_observation(raw_observation, metrics)
        )

        info = dict(info)
        info.update(
            {
                "lookahead_point_xy": metrics["lookahead_point"].tolist(),
                "lookahead_local_xy": metrics["lookahead_local"].tolist(),
                "path_deviation_m": float(metrics["path_deviation"]),
                "goal_distance_m": float(metrics["goal_distance"]),
                "remaining_path_distance_m": float(
                    metrics["remaining_path_distance"]
                ),
                "ppo_observation_dim": self.PPO_OBSERVATION_DIM,
                "bc_observation_dim": self.RAW_OBSERVATION_DIM,
            }
        )

        return self.current_ppo_observation.copy(), info

    def step(self, action):
        if self.current_observation is None:
            raise RuntimeError("Call reset() before step().")

        # 1. Validate PPO residual action.
        residual = np.asarray(action, dtype=np.float32)

        if residual.shape != (2,):
            raise ValueError(
                f"Expected residual action shape (2,), got {residual.shape}."
            )

        if not np.isfinite(residual).all():
            raise ValueError("Residual action contains NaN or Inf.")

        residual = np.clip(residual, -1.0, 1.0)

        # 2. BC receives only the original 40-D observation.
        bc_action = np.asarray(
            self.bc_policy.predict(self.current_observation),
            dtype=np.float32,
        )

        if bc_action.shape != (2,) or not np.isfinite(bc_action).all():
            raise RuntimeError("BC policy returned an invalid action.")

        bc_v = float(bc_action[0])
        bc_w = float(bc_action[1])

        # 3. Convert normalized PPO residuals to physical corrections.
        delta_v = float(residual[0]) * self.delta_v_max
        delta_w = float(residual[1]) * self.delta_w_max

        # 4. Fuse BC baseline action and PPO residual action.
        final_v = float(
            np.clip(
                bc_v + delta_v,
                0.0,
                MAX_LINEAR_V,
            )
        )
        final_w = float(
            np.clip(
                bc_w + delta_w,
                -MAX_ANGULAR_W,
                MAX_ANGULAR_W,
            )
        )

        # 5. Convert physical velocities to the base environment's action space.
        base_action = np.asarray(
            [
                2.0 * final_v / MAX_LINEAR_V - 1.0,
                final_w / MAX_ANGULAR_W,
            ],
            dtype=np.float32,
        )

        # 6. Execute one action in the original 40-D navigation environment.
        (
            raw_observation,
            base_reward,
            terminated,
            truncated,
            info,
        ) = self.base_env.step(base_action)

        raw_observation = np.asarray(
            raw_observation,
            dtype=np.float32,
        )

        if raw_observation.shape != (self.RAW_OBSERVATION_DIM,):
            raise RuntimeError(
                f"Base environment returned unexpected shape "
                f"{raw_observation.shape}."
            )

        # 7. Read the post-step state and calculate path metrics.
        pose = self.base_env.robot.get_pose()
        position = np.asarray(pose[:2], dtype=np.float32)
        yaw = float(pose[2])

        metrics = self._get_path_metrics(position, yaw)

        goal_distance = metrics["goal_distance"]
        remaining_path_distance = metrics["remaining_path_distance"]
        path_deviation = metrics["path_deviation"]
        heading_error = metrics["heading_error"]

        linear_speed = float(info["linear_speed_mps"])
        angular_speed = float(info["angular_speed_radps"])

        # Progress is measured along the remaining A* route, not by
        # change in lateral distance from the path.
        path_progress = (
            self.previous_remaining_path_distance
            - remaining_path_distance
        )
        goal_progress = self.previous_goal_distance - goal_distance

        # 8. Reward shaping. Keep additions moderate because the base
        # environment already rewards waypoint progress and stable arrival.
        path_progress_reward = 1.0 * float(
            np.clip(path_progress, -0.1, 0.1)
        )

        path_deviation_penalty = -0.5 * min(
            path_deviation ** 2,
            1.0,
        )

        goal_progress_reward = 0.5 * float(
            np.clip(goal_progress, -0.1, 0.1)
        )

        heading_penalty = -0.02 * abs(heading_error)

        residual_penalty = -0.01 * float(
            np.sum(np.square(residual))
        )

        # Encourage low translational and rotational speed near the goal.
        goal_proximity = float(
            np.clip(
                (0.5 - goal_distance) / 0.5,
                0.0,
                1.0,
            )
        )
        normalized_linear_speed = min(
            linear_speed / MAX_LINEAR_V,
            1.0,
        )
        normalized_angular_speed = min(
            angular_speed / MAX_ANGULAR_W,
            1.0,
        )

        near_goal_speed_penalty = (
            -0.05 * goal_proximity * normalized_linear_speed ** 2
            -0.02 * goal_proximity * normalized_angular_speed ** 2
        )

        timeout_penalty = -5.0 if truncated else 0.0

        reward_components = {
            "path_progress": float(path_progress_reward),
            "path_deviation": float(path_deviation_penalty),
            "goal_progress": float(goal_progress_reward),
            "heading_alignment": float(heading_penalty),
            "residual_regularization": float(residual_penalty),
            "near_goal_speed": float(near_goal_speed_penalty),
            "timeout": float(timeout_penalty),
        }

        shaping_reward = float(sum(reward_components.values()))
        total_reward = float(base_reward + shaping_reward)

        # 9. Update state for the next control step.
        self.previous_goal_distance = goal_distance
        self.previous_remaining_path_distance = remaining_path_distance

        self.current_observation = raw_observation.copy()

        self.current_ppo_observation = (
            self._augment_ppo_observation(raw_observation, metrics)
        )

        # 10. Expose diagnostics for evaluation and GUI visualization.
        info = dict(info)
        info.update(
            {
                "bc_action": [bc_v, bc_w],
                "residual_action": residual.tolist(),
                "residual_correction": [delta_v, delta_w],
                "executed_action": [final_v, final_w],
                "position_xy": position.tolist(),
                "lookahead_point_xy": metrics["lookahead_point"].tolist(),
                "lookahead_local_xy": metrics["lookahead_local"].tolist(),
                "path_deviation_m": float(path_deviation),
                "path_progress_m": float(path_progress),
                "remaining_path_distance_m": float(remaining_path_distance),
                "heading_error_rad": float(heading_error),
                "goal_distance_m": float(goal_distance),
                "goal_progress_m": float(goal_progress),
                "base_reward": float(base_reward),
                "shaping_reward": float(shaping_reward),
                "reward_components": reward_components,
                "reward": total_reward,
                "ppo_observation_dim": self.PPO_OBSERVATION_DIM,
                "bc_observation_dim": self.RAW_OBSERVATION_DIM,
            }
        )

        return (
            self.current_ppo_observation.copy(),
            total_reward,
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self):
        self.base_env.close()
