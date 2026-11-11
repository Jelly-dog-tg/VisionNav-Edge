"""P5 adapter around the existing waypoint PID; no planner or sensor changes."""

from functools import lru_cache
import importlib.util
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


@lru_cache(maxsize=1)
def _load_pid_controller_class() -> type:
    """Load the original controller whose script filename starts with a digit."""
    path = Path(__file__).resolve().parents[2] / "scripts/06_navigate_environment.py"
    spec = importlib.util.spec_from_file_location("visionnav_p5_original_pid", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load original PID controller: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PIDNavigationController


def _finite_vector(value: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    """Reject invalid packets instead of silently changing expert observations."""
    try:
        if np.iscomplexobj(value):
            raise ValueError("complex values are not physical sensor coordinates")
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a real numeric vector of shape ({size},)") from exc
    if vector.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must not contain NaN or Inf")
    return vector


class ExpertController:
    """Use the original PID to follow an A*-supplied waypoint.

    Inputs are world-frame pose [x_m, y_m, yaw_rad], waypoint [x_m, y_m],
    and the ORIGINAL 36-ray scan in metres from DiffDriveRobot.get_lidar_scan().
    Output is (v_mps, w_radps, reached), directly compatible with the original
    PID interface and robot.set_velocity(v_mps, w_radps).

    Only LiDAR indexing is adapted. PID gains, speed limits, goal stopping,
    side-selection heuristic, and derivative state retain their original logic.
    Pass raw scans, not already-shifted scans; reset between independent runs.
    """

    def __init__(self, goal_threshold: float = 0.4) -> None:
        try:
            threshold = float(goal_threshold)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("goal_threshold must be a finite positive distance in metres") from exc
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("goal_threshold must be a finite positive distance in metres")
        self._pid = _load_pid_controller_class()(goal_threshold=threshold)

    @staticmethod
    def adapt_lidar(raw_lidar: ArrayLike) -> NDArray[np.float64]:
        """Return a new 36-ray array, with distances/units/order direction intact.

        Raw i maps to PID (i + 18) % 36:
        front 0 -> 18, left 9 -> 27, rear 18 -> 0, right 27 -> 9.
        No reflection, clipping, normalization, or in-place mutation is applied.
        """
        scan = _finite_vector(raw_lidar, 36, "raw_lidar")
        if np.any(scan < 0):
            raise ValueError("raw_lidar distances must be nonnegative metres")
        return np.roll(scan, 18)

    def compute_control(
        self, pose: ArrayLike, waypoint: ArrayLike, raw_lidar: ArrayLike
    ) -> tuple[float, float, bool]:
        """Validate all inputs before updating PID state; reject invalid data."""
        robot_pose = _finite_vector(pose, 3, "pose")
        goal = _finite_vector(waypoint, 2, "waypoint")
        scan = self.adapt_lidar(raw_lidar)
        v, w, reached = self._pid.compute_control(robot_pose, goal, scan)
        v_mps, w_radps = float(v), float(w)
        if not np.isfinite(v_mps) or not np.isfinite(w_radps):
            raise ValueError("Original PID returned a non-finite expert action")
        return v_mps, w_radps, bool(reached)

    def reset(self) -> None:
        """Clear the original PID's derivative history for an independent run."""
        self._pid.last_heading_error = 0.0
