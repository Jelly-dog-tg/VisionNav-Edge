"""Generate 40 V3 selection-validation tasks, excluding all 80 prior endpoints.

The requested historical 27_generate_benchmark_tasks.py is absent. Reuse the
existing collect_expert.sample_task/Warehouse/A* instead; do not relax geometry
or reachability constraints. This is NOT a final unbiased test set.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pybullet as p
import pybullet_data

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionnav.learning import collect_expert as expert  # noqa: E402

OUTPUT = ROOT / "logs/p6_evaluation/benchmark_tasks_v3_validation.json"
TRAINING = ROOT / "logs/p5_dataset/metadata.json"
BENCHMARK = ROOT / "logs/p5_evaluation/benchmark_tasks.json"
SEED = 20261110


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def excluded_endpoints(grid):
    training = json.loads(TRAINING.read_text(encoding="utf-8"))["episodes"]
    benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))["tasks"]
    groups = []
    for records in (training, benchmark):
        cells = set()
        for row in records:
            for key in ("start_xy", "goal_xy"):
                xy = np.asarray(row[key], dtype=float)
                if xy.shape != (2,) or not np.isfinite(xy).all():
                    raise ValueError("Invalid prior endpoint; cannot guarantee independence")
                cell = expert.planner.world_to_grid(xy, expert.occupancy.RESOLUTION,
                                                    expert.ARENA_SIZE, grid)
                if cell is None:
                    raise ValueError("Prior endpoint outside the warehouse map")
                cells.add(cell)
        groups.append(cells)
    union = groups[0] | groups[1]
    # Clarified with the user: 40 dataset endpoints + 40 benchmark endpoints.
    if tuple(map(len, groups)) != (40, 40) or len(union) != 80:
        raise ValueError("Expected exactly 40 dataset + 40 benchmark = 80 unique prior endpoints")
    return union, groups


def generate(seed=SEED):
    if p.isConnected():
        raise RuntimeError("Task generation requires a separate default physics connection")
    client = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        ground = p.loadURDF("plane.urdf")
        expert.EnvironmentBuilder(client).create_warehouse_environment(arena_size=expert.ARENA_SIZE)
        grid = expert.occupancy.create_grid(expert.occupancy.collect_obstacle_aabbs(ground), expert.ARENA_SIZE)
        excluded, groups = excluded_endpoints(grid)
        used = set(excluded)
        rng = np.random.default_rng(seed)
        tasks = []
        for task_id in range(40):
            start, goal, path, yaw, attempts, expanded = expert.sample_task(grid, rng, used)
            if start in excluded or goal in excluded or grid[start] or grid[goal]:
                raise RuntimeError("Endpoint exclusion/traversability failed")
            if path[0] != start or path[-1] != goal:
                raise RuntimeError("A* path endpoints do not match the task")
            for a,b in zip(path[:-1], path[1:]):
                if b not in expert.planner.get_neighbors(a, grid):
                    raise RuntimeError("Invalid A* path edge")
            tasks.append({
                "episode_id": task_id, "seed": seed,
                "start_cell": list(start), "goal_cell": list(goal),
                "start_xy": expert.planner.grid_to_world(start, expert.occupancy.RESOLUTION, expert.ARENA_SIZE),
                "goal_xy": expert.planner.grid_to_world(goal, expert.occupancy.RESOLUTION, expert.ARENA_SIZE),
                "start_yaw_rad": yaw, "path_cells": [list(cell) for cell in path],
                "waypoints_world": [expert.planner.grid_to_world(cell, expert.occupancy.RESOLUTION,
                                                                  expert.ARENA_SIZE) for cell in path],
                "planned_path_length_m": expert.planner.calculate_path_length(path, expert.occupancy.RESOLUTION),
                "expanded_nodes": int(expanded), "sampling_attempts": attempts,
            })
        if len(used - excluded) != 80:
            raise RuntimeError("New tasks must have 80 distinct endpoints")
        original_keys = set(json.loads(BENCHMARK.read_text(encoding="utf-8"))["tasks"][0])
        if any(set(task) != original_keys for task in tasks):
            raise RuntimeError("Task fields differ from the original benchmark schema")
        return {
            "format_version": 1, "seed": seed, "environment": "warehouse", "task_count": 40,
            "arena_size": expert.ARENA_SIZE, "resolution": expert.occupancy.RESOLUTION,
            "dataset_role": "independent validation for parameter selection, NOT a final unbiased test set",
            "generation": {"sampler": "visionnav.learning.collect_expert.sample_task",
                "historical_generator": "experiments/27_generate_benchmark_tasks.py is absent",
                "min_distance_m": expert.MIN_TASK_DISTANCE, "max_distance_m": expert.MAX_TASK_DISTANCE,
                "max_sampling_attempts": expert.MAX_SAMPLING_ATTEMPTS,
                "obstacle_inflation_m": expert.occupancy.SAFETY_MARGIN,
                "excluded_dataset_endpoints": len(groups[0]), "excluded_benchmark_endpoints": len(groups[1]),
                "excluded_unique_endpoints": len(excluded), "new_unique_endpoints": 80,
                "clarification": "User confirmed that the stated 80 endpoints are the combined prior dataset and benchmark endpoints",
                "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                    TRAINING, BENCHMARK, ROOT / "visionnav/learning/collect_expert.py",
                    ROOT / "experiments/occupancy_grid.py", ROOT / "experiments/astar_planner.py",
                    ROOT / "scripts/05_environment_builder.py")},
                "grid_sha256": hashlib.sha256(grid.tobytes()).hexdigest()},
            "tasks": tasks,
        }
    finally:
        if p.isConnected(client):
            p.disconnect(client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite an existing validation set: {OUTPUT}")
    payload = generate(args.seed)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Generated {len(payload['tasks'])} reachable tasks; excluded 80 prior endpoints; {OUTPUT}")


if __name__ == "__main__":
    main()
