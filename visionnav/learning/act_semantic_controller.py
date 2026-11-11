"""ACT controller adapter for VisionNav semantic A* path following.

The observation feature layout exactly matches P5/P8 training:
  36 raw LiDAR distances,
  2 body-frame waypoint coordinates,
  signed forward speed,
  yaw rate.

The underlying ACTChunkController handles checkpoint normalization,
8-frame history, 16-action prediction, and 4-step receding-horizon execution.
This adapter converts its output into physical (v, w) commands expected by
DiffDriveRobot.set_velocity and the existing semantic navigation runner.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visionnav.learning.act_controller import ACTChunkController

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ACT_CHECKPOINT = ROOT / "models" / "act_distilled" / "act_best.pth"

# Must match the waypoint advancement rule used when P5 expert observations
# were collected. Changing this changes the 40-D observation distribution.
WAYPOINT_TOLERANCE_M = 0.12


def _finite_vector(value, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return array


def make_act_observation(pose, waypoint, raw_lidar, velocity) -> np.ndarray:
    """Build the exact 40-D observation used by the ACT training dataset."""
    pose_array = _finite_vector(pose, 3, "pose")
    goal = _finite_vector(waypoint, 2, "waypoint")
    lidar = _finite_vector(raw_lidar, 36, "raw_lidar")
    measured_velocity = _finite_vector(
        velocity, 3, "velocity [vx_world, vy_world, wz]"
    )
    if np.any(lidar < 0):
        raise ValueError("raw_lidar contains negative distances")

    c, s = np.cos(pose_array[2]), np.sin(pose_array[2])
    dx, dy = goal - pose_array[:2]
    local_goal = (c * dx + s * dy, -s * dx + c * dy)
    forward_speed = c * measured_velocity[0] + s * measured_velocity[1]

    observation = np.asarray(
        [*lidar, *local_goal, forward_speed, measured_velocity[2]],
        dtype=np.float32,
    )
    if observation.shape != (40,) or not np.isfinite(observation).all():
        raise RuntimeError("ACT observation must be finite float32 with shape (40,)")
    return observation


class ACTSemanticController:
    """Drop-in physical-action controller for the existing semantic runner."""

    def __init__(
        self,
        robot,
        waypoints,
        final_goal,
        checkpoint_path=DEFAULT_ACT_CHECKPOINT,
        goal_threshold=0.025,
        device="cpu",
    ):
        self.robot = robot
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Distilled ACT checkpoint not found: {self.checkpoint_path}"
            )

        self.waypoints = self._validate_waypoints(waypoints)
        self.final_goal = self._validate_goal(final_goal)
        self.goal_threshold = float(goal_threshold)
        if not np.isfinite(self.goal_threshold) or self.goal_threshold <= 0:
            raise ValueError("goal_threshold must be finite and positive")

        self.act = ACTChunkController(self.checkpoint_path, device=device)
        config = self.act.config
        if (config.obs_dim, config.action_dim, config.history_len, config.chunk_size) != (40, 2, 8, 16):
            raise RuntimeError(
                "Semantic ACT expects obs/action/history/chunk = 40/2/8/16; "
                f"checkpoint has {config.obs_dim}/{config.action_dim}/"
                f"{config.history_len}/{config.chunk_size}"
            )
        if self.act.execution_steps != 4:
            raise RuntimeError(
                f"Semantic ACT expects 4-step chunk execution; got {self.act.execution_steps}"
            )

        self.waypoint_index = 0
        self.step_count = 0
        self.last_diagnostics = {}
        self.history_reset_count = 0
        self.history_reset_events = []
        self._last_replans = 0
        self.closed = False

        print("\n========== DISTILLED ACT SEMANTIC CONTROLLER LOADED ==========")
        print("Checkpoint:", self.checkpoint_path)
        print("Training variant:", "bc_ppo_teacher_distilled_v1")
        print(
            f"Observation={config.obs_dim} | history={config.history_len} | "
            f"chunk={config.chunk_size} | execute={self.act.execution_steps}"
        )

    @staticmethod
    def _validate_waypoints(waypoints) -> np.ndarray:
        points = np.asarray(waypoints, dtype=np.float32)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or len(points) < 1
            or not np.isfinite(points).all()
        ):
            raise ValueError("waypoints must be a finite (N, 2) array")
        return points.copy()

    @staticmethod
    def _validate_goal(goal) -> np.ndarray:
        point = np.asarray(goal, dtype=np.float32)
        if point.shape != (2,) or not np.isfinite(point).all():
            raise ValueError("final_goal must be a finite (2,) vector")
        return point.copy()

    def reset(self, reason: str = "manual_reset") -> None:
        if self.closed:
            raise RuntimeError("ACTSemanticController is already closed")
        self.act.reset()
        self.waypoint_index = 0
        self.step_count = 0
        self.last_diagnostics = {}
        self._last_replans = 0
        self.history_reset_count += 1
        self.history_reset_events.append({"count": self.history_reset_count, "reason": str(reason),
            "history_frames_after": len(self.act.history), "pending_cleared": self.act.pending is None,
            "chunk_index_after": int(self.act.t)})

    def set_path(self, waypoints, final_goal=None, *, reset_reason="new_astar_path") -> None:
        """Switch A* leg and clear temporal history to avoid cross-leg leakage."""
        if self.closed:
            raise RuntimeError("Cannot set a path on a closed ACT controller")
        points = self._validate_waypoints(waypoints)
        goal = self._validate_goal(points[-1] if final_goal is None else final_goal)
        self.waypoints = points
        self.final_goal = goal
        self.reset(reason=reset_reason)

    def _update_waypoint_index(self, position: np.ndarray) -> None:
        while self.waypoint_index < len(self.waypoints) - 1:
            distance = float(
                np.linalg.norm(position - self.waypoints[self.waypoint_index])
            )
            if distance >= WAYPOINT_TOLERANCE_M:
                break
            self.waypoint_index += 1

    def _path_deviation(self, position: np.ndarray) -> float:
        path = self.waypoints[self.waypoint_index :]
        if len(path) == 1:
            return float(np.linalg.norm(position - path[0]))

        best = float("inf")
        for start, end in zip(path[:-1], path[1:]):
            segment = end - start
            denom = float(np.dot(segment, segment))
            if denom <= 1e-12:
                projection = start
            else:
                ratio = float(np.clip(np.dot(position - start, segment) / denom, 0.0, 1.0))
                projection = start + ratio * segment
            best = min(best, float(np.linalg.norm(position - projection)))
        return best

    def compute_control(self, robot_pose, waypoint, lidar_scan):
        """Return physical linear velocity, angular velocity, and reached flag."""
        if self.closed:
            raise RuntimeError("ACTSemanticController is closed")

        pose = _finite_vector(robot_pose, 3, "robot_pose").astype(np.float32)
        lidar = _finite_vector(lidar_scan, 36, "lidar_scan").astype(np.float32)
        position = pose[:2]
        self._update_waypoint_index(position)
        current_waypoint = self.waypoints[self.waypoint_index]
        velocity = _finite_vector(self.robot.get_velocity(), 3, "robot velocity")

        observation = make_act_observation(
            pose,
            current_waypoint,
            lidar,
            velocity,
        )
        normalized_action, act_diag = self.act.next_action(observation)

        # ACTChunkController has already de-normalized, physically clipped,
        # and converted the command to the RL environment's normalized action.
        # Use its physical diagnostics to avoid a second scaling operation.
        v_mps = float(act_diag["v_mps"])
        w_radps = float(act_diag["w_radps"])
        if not np.isfinite([v_mps, w_radps]).all():
            raise RuntimeError("Distilled ACT produced a non-finite command")

        self.step_count += 1
        goal_distance = float(np.linalg.norm(position - self.final_goal))
        path_deviation = self._path_deviation(position)
        replans_now = int(act_diag["replans"])
        previous_replans = int(getattr(self, "_last_replans", 0))
        chunk_replanned = replans_now > previous_replans
        self._last_replans = replans_now

        self.last_diagnostics = {
            "step": self.step_count,
            "waypoint_index": self.waypoint_index,
            "v_mps": v_mps,
            "w_radps": w_radps,
            "path_deviation_m": path_deviation,
            "goal_distance_m": goal_distance,
            "normalized_action": np.asarray(normalized_action, dtype=np.float32).tolist(),
            "action_clipped": bool(act_diag["action_clipped"]),
            "chunk_index": int(act_diag["chunk_index"]),
            "chunk_replanned": bool(chunk_replanned),
            "replans": replans_now,
            "history_frames": int(act_diag["history_frames"]),
        }

        if self.step_count % 30 == 0:
            print(
                f"[Distilled ACT] step={self.step_count} "
                f"goal={goal_distance:.3f}m path_dev={path_deviation:.3f}m "
                f"v={v_mps:.3f}m/s w={w_radps:+.3f}rad/s "
                f"chunk={act_diag['chunk_index']} replans={replans_now}"
            )

        return v_mps, w_radps, bool(goal_distance <= self.goal_threshold)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.last_diagnostics = {}
        self.act = None

