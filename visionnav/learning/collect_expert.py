"""P5-2: collect successful Warehouse A*/ExpertController demonstrations.

Run from the project root: python -m visionnav.learning.collect_expert
There is one fresh DIRECT world and one fresh ExpertController per episode.
Only four dataset files are written; failed trajectories remain statistics only.
"""

import argparse
from dataclasses import dataclass, field
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
import pybullet as p
import pybullet_data

from scripts.diff_drive_robot import DiffDriveRobot
from visionnav.learning.expert_controller import ExpertController


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "logs/p5_dataset"
ARENA_SIZE = 10.0
PHYSICS_DT = 1 / 240
PHYSICS_STEPS = 10
INITIAL_SETTLE_STEPS = 120
GOAL_THRESHOLD = 0.08
FINAL_TOLERANCE = 0.12
SPEED_TOLERANCE = 0.05
REQUIRED_SETTLE_CYCLES = 10
MIN_TASK_DISTANCE = 2.0
MAX_TASK_DISTANCE = 6.0
MAX_SAMPLING_ATTEMPTS = 256


def _load_experiment(name: str) -> ModuleType:
    path = ROOT / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"visionnav_p5_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load existing module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


occupancy = _load_experiment("occupancy_grid")
planner = _load_experiment("astar_planner")
EnvironmentBuilder = occupancy.load_environment_builder()


def _vector(value: ArrayLike, size: int, name: str) -> NDArray[np.float64]:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return array


def make_observation(
    pose: ArrayLike, waypoint: ArrayLike, raw_lidar: ArrayLike, velocity: ArrayLike
) -> NDArray[np.float32]:
    """Raw scan + body-frame waypoint + actual signed forward speed/yaw rate."""
    pose_array = _vector(pose, 3, "pose")
    goal = _vector(waypoint, 2, "waypoint")
    lidar = _vector(raw_lidar, 36, "raw_lidar")
    measured_velocity = _vector(velocity, 3, "velocity [vx_world, vy_world, wz]")
    if np.any(lidar < 0):
        raise ValueError("raw_lidar contains negative distances")
    c, s = np.cos(pose_array[2]), np.sin(pose_array[2])
    dx, dy = goal - pose_array[:2]
    local_goal = [c * dx + s * dy, -s * dx + c * dy]
    speed = c * measured_velocity[0] + s * measured_velocity[1]
    with np.errstate(over="raise", invalid="raise"):
        observation = np.asarray([*lidar, *local_goal, speed, measured_velocity[2]], dtype=np.float32)
    if observation.shape != (40,) or not np.all(np.isfinite(observation)):
        raise ValueError("Invalid float32 observation")
    return observation


@dataclass
class EpisodeResult:
    episode_id: int
    seed: int
    success: bool = False
    failure_reason: str | None = None
    start_xy: list[float] | None = None
    goal_xy: list[float] | None = None
    start_yaw_rad: float | None = None
    initial_pose: list[float] | None = None
    initial_speed_mps: float | None = None
    initial_yaw_rate_radps: float | None = None
    navigation_steps: int = 0
    path_length_m: float | None = None
    expanded_nodes: int | None = None
    sampling_attempts: int = 0
    collision: bool = False
    final_pose: list[float] | None = None
    final_goal_error_m: float | None = None
    final_speed_mps: float | None = None
    observations: list[NDArray[np.float32]] = field(default_factory=list, repr=False)
    actions: list[NDArray[np.float32]] = field(default_factory=list, repr=False)

    def summary(self, start: int, end: int) -> dict[str, Any]:
        return {name: getattr(self, name) for name in (
            "episode_id", "seed", "success", "failure_reason", "start_xy", "goal_xy",
            "start_yaw_rad", "initial_pose", "initial_speed_mps", "initial_yaw_rate_radps",
            "navigation_steps", "path_length_m", "expanded_nodes", "sampling_attempts",
            "collision", "final_pose", "final_goal_error_m", "final_speed_mps",
        )} | {"attempted_sample_count": len(self.observations),
              "collected_sample_count": end - start,
              "discarded_sample_count": len(self.observations) - (end - start),
              "sample_range": [start, end]}


def sample_task(
    grid: NDArray, rng: np.random.Generator, used_cells: set[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], float, int, int]:
    """Sample distinct free endpoints; accept only a reachable original-A* path."""
    cells = np.argwhere(grid == 0)
    # Avoid spawning near the outer walls, while retaining internal obstacles.
    points = -ARENA_SIZE + (cells[:, ::-1] + 0.5) * occupancy.RESOLUTION
    cells = cells[np.all(np.abs(points) <= ARENA_SIZE - 1.0, axis=1)]
    if len(cells) < 2:
        raise RuntimeError("NO_PATH: fewer than two free sampling cells")
    for attempt in range(1, MAX_SAMPLING_ATTEMPTS + 1):
        selected = rng.choice(len(cells), size=2, replace=False)
        start, goal = (tuple(int(v) for v in cells[i]) for i in selected)
        if start in used_cells or goal in used_cells:
            continue
        start_xy = np.asarray(planner.grid_to_world(start, occupancy.RESOLUTION, ARENA_SIZE))
        goal_xy = np.asarray(planner.grid_to_world(goal, occupancy.RESOLUTION, ARENA_SIZE))
        if not MIN_TASK_DISTANCE <= np.linalg.norm(goal_xy - start_xy) <= MAX_TASK_DISTANCE:
            continue
        path, expanded = planner.astar(grid, start, goal, occupancy.RESOLUTION)
        if path:
            used_cells.update((start, goal))
            return start, goal, path, float(rng.uniform(-np.pi, np.pi)), attempt, expanded
    raise RuntimeError("NO_PATH: sampling budget exhausted without a distinct reachable task")


def has_collision(robot_id: int, ground_id: int) -> bool:
    """Check all robot links; only ground contact is excluded."""
    return any(contact[2] not in (ground_id, robot_id)
               and (contact[8] <= 0.0 or contact[9] > 0.1)
               for contact in p.getContactPoints(bodyA=robot_id))


def run_episode(
    episode_id: int, seed: int, max_steps: int, used_cells: set[tuple[int, int]]
) -> EpisodeResult:
    result = EpisodeResult(episode_id=episode_id, seed=seed)
    client, robot = -1, None
    # Per-episode RNG streams make failures independent of later random draws.
    rng = np.random.default_rng(np.random.SeedSequence([seed, episode_id]))
    try:
        if p.isConnected():
            raise RuntimeError("DIRECT_CONNECTION_CONFLICT: existing default physics connection")
        client = p.connect(p.DIRECT)
        if client != 0:
            raise RuntimeError("DIRECT_CONNECTION_FAILED: existing modules require default client 0")
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(PHYSICS_DT)
        ground_id = p.loadURDF("plane.urdf")
        EnvironmentBuilder(client).create_warehouse_environment(arena_size=ARENA_SIZE)
        # Map construction precedes robot creation so its footprint is excluded.
        obstacles = occupancy.collect_obstacle_aabbs(ground_id)
        grid = occupancy.create_grid(obstacles, ARENA_SIZE)
        start, goal, path, yaw, attempts, expanded = sample_task(grid, rng, used_cells)
        result.start_xy = planner.grid_to_world(start, occupancy.RESOLUTION, ARENA_SIZE)
        result.goal_xy = planner.grid_to_world(goal, occupancy.RESOLUTION, ARENA_SIZE)
        result.start_yaw_rad, result.sampling_attempts = yaw, attempts
        result.expanded_nodes = int(expanded)
        result.path_length_m = float(planner.calculate_path_length(path, occupancy.RESOLUTION))
        waypoints = np.asarray([planner.grid_to_world(node, occupancy.RESOLUTION, ARENA_SIZE)
                                for node in path], dtype=float)
        robot = DiffDriveRobot(str(ROOT / "urdf/diff_drive_robot.urdf"),
                              start_pos=[*result.start_xy, 0.1], start_yaw=yaw)
        robot.set_velocity(0.0, 0.0)
        for _ in range(INITIAL_SETTLE_STEPS):
            p.stepSimulation()
            if has_collision(robot.robot_id, ground_id):
                result.collision = True
                raise RuntimeError("SPAWN_COLLISION")
        result.initial_pose = _vector(robot.get_pose(), 3, "initial pose").tolist()
        initial_velocity = _vector(robot.get_velocity(), 3, "initial velocity")
        result.initial_speed_mps = float(np.linalg.norm(initial_velocity[:2]))
        result.initial_yaw_rate_radps = float(initial_velocity[2])
        # Each episode gets its own derivative history and original PID defaults.
        expert = ExpertController(goal_threshold=GOAL_THRESHOLD)
        waypoint_index, stable_cycles = 0, 0
        for step in range(1, max_steps + 1):
            result.navigation_steps = step
            pose = robot.get_pose()
            velocity = robot.get_velocity()
            while (waypoint_index < len(waypoints) - 1
                   and np.linalg.norm(pose[:2] - waypoints[waypoint_index]) < 0.12):
                waypoint_index += 1
            waypoint = waypoints[waypoint_index]
            # Freeze measurements once; no physics step occurs between sampling
            # the observation, obtaining its action, and applying that action.
            lidar = np.asarray(robot.get_lidar_scan(), dtype=np.float32)
            observation = make_observation(pose, waypoint, lidar, velocity)
            v, w, reached = expert.compute_control(pose, waypoint, lidar)
            action = np.asarray([v, w], dtype=np.float32)
            if action.shape != (2,) or not np.all(np.isfinite(action)):
                raise ValueError("INVALID_ACTION")
            result.observations.append(observation)
            result.actions.append(action)
            # Apply exactly the float32 command saved as the label.
            robot.set_velocity(float(action[0]), float(action[1]))
            for _ in range(PHYSICS_STEPS):
                p.stepSimulation()
                if has_collision(robot.robot_id, ground_id):
                    result.collision = True
                    raise RuntimeError("COLLISION")
            current_pose, current_velocity = robot.get_pose(), robot.get_velocity()
            distance = float(np.linalg.norm(current_pose[:2] - np.asarray(result.goal_xy)))
            stable = (waypoint_index == len(waypoints) - 1 and reached
                      and distance <= FINAL_TOLERANCE
                      and np.linalg.norm(current_velocity[:2]) <= SPEED_TOLERANCE
                      and abs(current_velocity[2]) <= SPEED_TOLERANCE)
            stable_cycles = stable_cycles + 1 if stable else 0
            if stable_cycles >= REQUIRED_SETTLE_CYCLES:
                result.success = True
                break
        else:
            result.failure_reason = "TIMEOUT"
    except Exception as exc:
        result.success = False
        result.failure_reason = f"{type(exc).__name__}: {exc}"
    finally:
        if robot is not None and client >= 0 and p.isConnected(client):
            try:
                robot.set_velocity(0.0, 0.0)
                pose, velocity = robot.get_pose(), robot.get_velocity()
                if np.all(np.isfinite(pose)):
                    result.final_pose = pose.tolist()
                    if result.goal_xy is not None:
                        error = float(np.linalg.norm(pose[:2] - np.asarray(result.goal_xy)))
                        result.final_goal_error_m = error if np.isfinite(error) else None
                if np.all(np.isfinite(velocity)):
                    speed = float(np.linalg.norm(velocity[:2]))
                    result.final_speed_mps = speed if np.isfinite(speed) else None
            except p.error:
                result.success = False
                result.failure_reason = "PHYSICS_STATE_UNAVAILABLE"
        if client >= 0 and p.isConnected(client):
            p.disconnect(client)
    return result


def save_dataset(
    results: Sequence[EpisodeResult], output_dir: Path, seed: int, max_steps: int
) -> dict[str, Any]:
    """Commit complete successful episodes together, with auditable row ranges."""
    observations, actions, ids, summaries = [], [], [], []
    offset = 0
    for result in results:
        if len(result.observations) != len(result.actions):
            raise ValueError("Observation/action count mismatch")
        start = offset
        if result.success:
            if (result.collision or not result.observations
                    or result.final_goal_error_m is None or result.final_goal_error_m > FINAL_TOLERANCE
                    or result.final_speed_mps is None or result.final_speed_mps > SPEED_TOLERANCE):
                raise ValueError("Invalid success metadata; refusing contaminated training data")
            obs = np.asarray(result.observations, dtype=np.float32)
            act = np.asarray(result.actions, dtype=np.float32)
            if (obs.shape != (len(act), 40) or act.shape != (len(obs), 2)
                    or not np.all(np.isfinite(obs)) or not np.all(np.isfinite(act))):
                raise ValueError("Invalid successful episode arrays")
            observations.append(obs)
            actions.append(act)
            # Integer-valued identifiers, saved as float32 like all dataset arrays.
            ids.append(np.full(len(obs), result.episode_id, dtype=np.float32))
            offset += len(obs)
        summaries.append(result.summary(start, offset))
    obs_array = np.concatenate(observations) if observations else np.empty((0, 40), dtype=np.float32)
    action_array = np.concatenate(actions) if actions else np.empty((0, 2), dtype=np.float32)
    id_array = np.concatenate(ids) if ids else np.empty((0,), dtype=np.float32)
    metadata = {
        "format_version": 1, "seed": seed, "environment": "warehouse", "connection": "DIRECT",
        "reset_policy": "fresh physics connection, robot and ExpertController per episode",
        "episode_count": len(results), "successful_episodes": sum(r.success for r in results),
        "failed_episodes": sum(not r.success for r in results), "sample_count": offset,
        "array_dtype": "float32", "episode_ids": "integer-valued float32",
        "observation_schema": {"0:36": "raw LiDAR metres; front=0, left=9, rear=18, right=27",
            "36:38": "current waypoint in robot frame, x forward/y left, metres",
            "38": "actual signed forward linear velocity m/s", "39": "actual yaw rate rad/s"},
        "action_schema": ["commanded linear velocity m/s", "commanded angular velocity rad/s"],
        "physics_dt_s": PHYSICS_DT, "physics_steps_per_action": PHYSICS_STEPS,
        "control_dt_s": PHYSICS_DT * PHYSICS_STEPS, "initial_settle_steps": INITIAL_SETTLE_STEPS,
        "max_control_steps": max_steps, "goal_threshold_m": GOAL_THRESHOLD,
        "final_tolerance_m": FINAL_TOLERANCE, "speed_tolerance_mps": SPEED_TOLERANCE,
        "required_stable_cycles": REQUIRED_SETTLE_CYCLES,
        "sampling": {"min_distance_m": MIN_TASK_DISTANCE, "max_distance_m": MAX_TASK_DISTANCE,
            "max_attempts": MAX_SAMPLING_ATTEMPTS, "rng": "PCG64 SeedSequence([seed, episode_id])",
            "unique_endpoints": True, "initial_yaw": "uniform [-pi, pi]"},
        "map_resolution_m": occupancy.RESOLUTION, "map_safety_inflation_m": occupancy.SAFETY_MARGIN,
        "pybullet_api_version": p.getAPIVersion(), "numpy_version": np.__version__,
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in (
            "experiments/occupancy_grid.py", "experiments/astar_planner.py",
            "scripts/05_environment_builder.py", "scripts/diff_drive_robot.py",
            "scripts/06_navigate_environment.py", "visionnav/learning/expert_controller.py")},
        "episodes": summaries,
    }
    # Check JSON before touching the output files; NaN/Inf metadata is forbidden.
    text = json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "observations.npy", obs_array, allow_pickle=False)
    np.save(output_dir / "actions.npy", action_array, allow_pickle=False)
    np.save(output_dir / "episode_ids.npy", id_array, allow_pickle=False)
    (output_dir / "metadata.json").write_text(text, encoding="utf-8")
    return metadata


def collect(
    episodes: int = 20, seed: int = 42, max_steps: int = 2500, output_dir: Path = OUTPUT_DIR
) -> dict[str, Any]:
    if episodes <= 0 or episodes > 2**24 or seed < 0 or max_steps <= 0:
        raise ValueError("episodes/max_steps must be positive, seed nonnegative, IDs float32-exact")
    if p.isConnected():
        raise RuntimeError("Collector requires an unused default physics connection")
    results, used_cells = [], set()
    for episode_id in range(episodes):
        result = run_episode(episode_id, seed, max_steps, used_cells)
        results.append(result)
        print(f"Episode {episode_id + 1}/{episodes}: success={result.success}, "
              f"steps={result.navigation_steps}, collision={result.collision}, "
              f"error={result.final_goal_error_m}, reason={result.failure_reason}", flush=True)
    metadata = save_dataset(results, Path(output_dir), seed, max_steps)
    print(f"Saved {metadata['sample_count']} samples; success={metadata['successful_episodes']}, "
          f"failed={metadata['failed_episodes']}; output={output_dir}", flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=2500)
    args = parser.parse_args()
    collect(args.episodes, args.seed, args.max_steps)


if __name__ == "__main__":
    main()
