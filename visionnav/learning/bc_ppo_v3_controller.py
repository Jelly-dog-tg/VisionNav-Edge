"""Inference controller that fuses the trained BC policy with PPO V3 residuals.

This controller is designed for the existing GUI semantic-navigation runner
(experiments/26_follow_visual_path.py). It keeps the original PyBullet robot,
scene, VLM-generated A* waypoints, and final safe-stop goal.

The BC policy receives the original 40-D observation.
The PPO V3 policy receives the exact 44-D V2 observation layout:
    raw 40-D BC observation
    + lookahead point in robot frame (x, y)
    + distance to remaining A* path
    + distance to final safe-stop goal

The saved VecNormalize statistics are used for PPO inference.
PPO residual scaling is applied exactly once here (0.25), followed by the
same physical conversion and BC/PPO fusion used by BCResidualNavigationEnvV2.
"""

from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from visionnav.learning.bc_policy import BehaviorCloningPolicy
from visionnav.learning.collect_expert import make_observation
from visionnav.learning.rl_navigation_env import MAX_ANGULAR_W, MAX_LINEAR_V


ROOT = Path(__file__).resolve().parents[2]

BC_CHECKPOINT = ROOT / "models" / "behavior_cloning" / "bc_best.pth"

DELTA_V_MAX = 0.05
DELTA_W_MAX = 0.4
LOOKAHEAD_DISTANCE_M = 0.5
WAYPOINT_TOLERANCE_M = 0.08


class _NormalizationOnlyEnv(gym.Env):
    """Dummy 44-D environment used only to attach saved VecNormalize stats."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(44, dtype=np.float32), {}

    def step(self, action):
        return (
            np.zeros(44, dtype=np.float32),
            0.0,
            False,
            False,
            {},
        )


class BCPPOV3Controller:
    """Drop-in compute_control() controller for the existing semantic runner."""

    def __init__(
        self,
        robot,
        waypoints,
        final_goal,
        seed=42,
        residual_scale=0.25,
        goal_threshold=0.025,
    ):
        self.robot = robot
        self.waypoints = np.asarray(waypoints, dtype=np.float32)
        self.final_goal = np.asarray(final_goal, dtype=np.float32)

        if (
            self.waypoints.ndim != 2
            or self.waypoints.shape[1] != 2
            or len(self.waypoints) < 1
            or not np.isfinite(self.waypoints).all()
        ):
            raise ValueError("waypoints must be a finite (N, 2) array")

        if self.final_goal.shape != (2,) or not np.isfinite(self.final_goal).all():
            raise ValueError("final_goal must be a finite (2,) vector")

        self.residual_scale = float(residual_scale)
        if not np.isfinite(self.residual_scale) or not 0.0 <= self.residual_scale <= 1.0:
            raise ValueError("residual_scale must be in [0, 1]")

        self.goal_threshold = float(goal_threshold)
        self.waypoint_index = 0
        self.step_count = 0

        # Kept for compatibility with the original runner's approach-speed
        # assignments. V3's learned policy does not use the PID thresholds.
        self.safety_distance = 0.6
        self.critical_distance = 0.4

        model_dir = ROOT / "models" / f"bc_residual_ppo_v3_seed_{int(seed)}"
        model_path = model_dir / "bc_residual_ppo_v3_final.zip"
        vecnormalize_path = model_dir / "vecnormalize.pkl"

        for path in (BC_CHECKPOINT, model_path, vecnormalize_path):
            if not path.is_file():
                raise FileNotFoundError(f"Required model file not found: {path}")

        self.bc_policy = BehaviorCloningPolicy(BC_CHECKPOINT)
        self.ppo_model = PPO.load(str(model_path), device="cpu")

        if self.ppo_model.observation_space.shape != (44,):
            raise RuntimeError(
                "PPO V3 must use a 44-D observation; got "
                f"{self.ppo_model.observation_space.shape}"
            )
        if self.ppo_model.action_space.shape != (2,):
            raise RuntimeError(
                "PPO V3 must use a 2-D action; got "
                f"{self.ppo_model.action_space.shape}"
            )

        dummy_vec = DummyVecEnv([_NormalizationOnlyEnv])
        try:
            self.vecnormalize = VecNormalize.load(
                str(vecnormalize_path), dummy_vec
            )
        except Exception:
            dummy_vec.close()
            raise

        self.vecnormalize.training = False
        self.vecnormalize.norm_reward = False

        if self.vecnormalize.observation_space.shape != (44,):
            self.vecnormalize.close()
            raise RuntimeError("Saved VecNormalize statistics are not 44-D")

        self.last_diagnostics = {}

        print("\n========== BC-PPO V3 CONTROLLER LOADED ==========")
        print("Seed:", seed)
        print("BC checkpoint:", BC_CHECKPOINT)
        print("PPO V3 checkpoint:", model_path)
        print("VecNormalize:", vecnormalize_path)
        print("Residual scale:", self.residual_scale)
        print("Observation dimensions: BC=40, PPO=44")

    def _update_waypoint_index(self, position):
        while self.waypoint_index < len(self.waypoints) - 1:
            distance = float(
                np.linalg.norm(position - self.waypoints[self.waypoint_index])
            )
            if distance >= WAYPOINT_TOLERANCE_M:
                break
            self.waypoint_index += 1

    def _project_onto_remaining_path(self, position):
        path = self.waypoints[self.waypoint_index:]

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
                ratio = float(np.dot(position - start, segment) / denominator)
                ratio = float(np.clip(ratio, 0.0, 1.0))

            projection = start + ratio * segment
            distance = float(np.linalg.norm(position - projection))

            if distance < best_distance:
                best_distance = distance
                best_projection = projection.astype(np.float32)
                best_segment_index = index

        return best_projection, best_segment_index, best_distance

    def _select_lookahead_point(self, position):
        path = self.waypoints[self.waypoint_index:]

        if len(path) == 1:
            return path[0].copy()

        projection, segment_index, _ = self._project_onto_remaining_path(position)

        remaining_distance = LOOKAHEAD_DISTANCE_M
        cursor = projection.copy()

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

        return path[-1].copy()

    def _build_ppo_observation(self, pose, raw_observation):
        position = np.asarray(pose[:2], dtype=np.float32)
        yaw = float(pose[2])

        lookahead_point = self._select_lookahead_point(position)
        target_vector_world = lookahead_point - position

        cos_yaw = float(np.cos(yaw))
        sin_yaw = float(np.sin(yaw))

        lookahead_local_x = (
            cos_yaw * float(target_vector_world[0])
            + sin_yaw * float(target_vector_world[1])
        )
        lookahead_local_y = (
            -sin_yaw * float(target_vector_world[0])
            + cos_yaw * float(target_vector_world[1])
        )

        _, _, path_deviation = self._project_onto_remaining_path(position)
        goal_distance = float(np.linalg.norm(position - self.final_goal))

        extra_features = np.asarray(
            [
                lookahead_local_x,
                lookahead_local_y,
                path_deviation,
                goal_distance,
            ],
            dtype=np.float32,
        )

        ppo_observation = np.concatenate(
            [np.asarray(raw_observation, dtype=np.float32), extra_features]
        ).astype(np.float32)

        if ppo_observation.shape != (44,) or not np.isfinite(ppo_observation).all():
            raise RuntimeError("Invalid 44-D PPO observation")

        return ppo_observation, path_deviation, goal_distance

    def compute_control(self, robot_pose, waypoint, lidar_scan):
        """Return (linear velocity m/s, angular velocity rad/s, reached)."""
        pose = np.asarray(robot_pose, dtype=np.float32)
        lidar = np.asarray(lidar_scan, dtype=np.float32)

        if pose.shape != (3,) or not np.isfinite(pose).all():
            raise ValueError("robot_pose must be a finite (3,) vector")
        if lidar.shape != (36,) or not np.isfinite(lidar).all():
            raise ValueError("lidar_scan must be a finite (36,) vector")

        position = pose[:2]
        self._update_waypoint_index(position)
        current_waypoint = self.waypoints[self.waypoint_index]

        velocity = np.asarray(self.robot.get_velocity(), dtype=np.float32)
        raw_observation = make_observation(
            pose,
            current_waypoint,
            lidar,
            velocity,
        )

        if raw_observation.shape != (40,):
            raise RuntimeError(
                f"BC observation must be 40-D, got {raw_observation.shape}"
            )

        ppo_observation, path_deviation, goal_distance = (
            self._build_ppo_observation(pose, raw_observation)
        )

        normalized_observation = self.vecnormalize.normalize_obs(
            ppo_observation.reshape(1, 44)
        )

        ppo_action, _ = self.ppo_model.predict(
            normalized_observation,
            deterministic=True,
        )
        ppo_action = np.asarray(ppo_action, dtype=np.float32).reshape(-1)

        if ppo_action.shape != (2,) or not np.isfinite(ppo_action).all():
            raise RuntimeError("PPO V3 produced an invalid residual action")

        ppo_action = np.clip(ppo_action, -1.0, 1.0)

        # Match ResidualScaleWrapper(0.25) exactly once.
        scaled_residual = ppo_action * self.residual_scale

        # BC produces physical velocity commands; PPO adds bounded corrections.
        bc_action = np.asarray(
            self.bc_policy.predict(raw_observation),
            dtype=np.float32,
        )
        if bc_action.shape != (2,) or not np.isfinite(bc_action).all():
            raise RuntimeError("BC policy produced an invalid action")

        delta_v = float(scaled_residual[0]) * DELTA_V_MAX
        delta_w = float(scaled_residual[1]) * DELTA_W_MAX

        final_v = float(
            np.clip(
                float(bc_action[0]) + delta_v,
                0.0,
                MAX_LINEAR_V,
            )
        )
        final_w = float(
            np.clip(
                float(bc_action[1]) + delta_w,
                -MAX_ANGULAR_W,
                MAX_ANGULAR_W,
            )
        )

        self.step_count += 1
        self.last_diagnostics = {
            "step": self.step_count,
            "waypoint_index": self.waypoint_index,
            "bc_action": bc_action.tolist(),
            "ppo_action": ppo_action.tolist(),
            "scaled_residual": scaled_residual.tolist(),
            "delta_v_mps": delta_v,
            "delta_w_radps": delta_w,
            "final_action": [final_v, final_w],
            "path_deviation_m": float(path_deviation),
            "goal_distance_m": float(goal_distance),
            "residual_scale": self.residual_scale,
        }

        if self.step_count % 30 == 0:
            print(
                f"[BC-PPO V3] step={self.step_count} "
                f"goal={goal_distance:.3f}m "
                f"path_dev={path_deviation:.3f}m "
                f"dv={delta_v:+.4f}m/s "
                f"dw={delta_w:+.4f}rad/s"
            )

        reached = goal_distance <= self.goal_threshold
        return final_v, final_w, reached

    def close(self):
        if getattr(self, "vecnormalize", None) is not None:
            self.vecnormalize.close()
            self.vecnormalize = None

    def set_path(self, waypoints, final_goal=None):
        points = np.asarray(waypoints, dtype=np.float32)

        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 1
            or not np.isfinite(points).all()
        ):
            raise ValueError("Invalid navigation path")

        goal = points[-1] if final_goal is None else np.asarray(
            final_goal, dtype=np.float32
        )

        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("Invalid final goal")

        self.waypoints = points
        self.final_goal = goal.copy()
        self.waypoint_index = 0
        self.step_count = 0
        self.last_diagnostics = {}