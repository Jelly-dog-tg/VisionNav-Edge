"""
VisionNav-Edge
Experiment 4.2: A* Global Path Planning

Input:  experiments/results/maps/warehouse_grid.npz
Output: experiments/results/paths/warehouse_astar.png
        experiments/results/paths/warehouse_path.json
"""

import heapq
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MAP_FILE = ROOT / "experiments/results/maps/warehouse_grid.npz"
OUTPUT_DIR = ROOT / "experiments/results/paths"
ENV_NAME = "warehouse"

SQRT2 = math.sqrt(2)
# 8-connected moves; diagonals come last so straight ties win in the heap
MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1),
         (-1, -1), (-1, 1), (1, -1), (1, 1)]


def world_to_grid(point, resolution, arena_size, grid):
    """Convert world coordinates (x, y) to (row, col), or None if outside the map."""
    x, y = point
    col = math.floor((x + arena_size) / resolution)
    row = math.floor((y + arena_size) / resolution)
    if 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]:
        return row, col
    return None


def grid_to_world(node, resolution, arena_size):
    """Convert a grid cell to its world-space center."""
    row, col = node
    return [-arena_size + (col + 0.5) * resolution,
            -arena_size + (row + 0.5) * resolution]


def heuristic(node, goal, resolution):
    """Octile distance for 8-connected movement."""
    dx = abs(node[1] - goal[1])
    dy = abs(node[0] - goal[0])
    diagonal = min(dx, dy)
    return resolution * (diagonal * SQRT2 + (max(dx, dy) - diagonal))


def get_neighbors(node, grid):
    """Traversable 8-connected neighbors; diagonal corner cutting is prohibited."""
    row, col = node
    rows, cols = grid.shape
    neighbors = []
    for dr, dc in MOVES:
        nr, nc = row + dr, col + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if grid[nr, nc] == 1:
            continue
        if dr and dc and (grid[row + dr, col] == 1 or grid[row, col + dc] == 1):
            continue
        neighbors.append((nr, nc))
    return neighbors


def astar(grid, start, goal, resolution):
    """Find a minimum-cost path. Returns (path, expanded_nodes) or (None, expanded_nodes)."""
    if grid[start] == 1:
        raise ValueError(f"Start cell is blocked: {start}")
    if grid[goal] == 1:
        raise ValueError(f"Goal cell is blocked: {goal}. Choose another free goal position.")

    open_heap = [(heuristic(start, goal, resolution), 0.0, start)]
    g_score = {start: 0.0}
    came_from = {}
    closed_set = set()
    expanded = 0

    while open_heap:
        _, g, current = heapq.heappop(open_heap)

        # Skip outdated queue entries
        if current in closed_set or g > g_score.get(current, math.inf) + 1e-10:
            continue

        if current == goal:
            path = [goal]
            while path[-1] != start:
                path.append(came_from[path[-1]])
            path.reverse()
            return path, expanded

        closed_set.add(current)
        expanded += 1

        for neighbor in get_neighbors(current, grid):
            if neighbor in closed_set:
                continue
            dr = neighbor[0] - current[0]
            dc = neighbor[1] - current[1]
            step = resolution * SQRT2 if dr and dc else resolution
            tentative = g + step
            if tentative < g_score.get(neighbor, math.inf):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                f = tentative + heuristic(neighbor, goal, resolution)
                heapq.heappush(open_heap, (f, tentative, neighbor))

    return None, expanded


def calculate_path_length(path, resolution):
    """Geometric length of a grid path."""
    total = 0.0
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        total += resolution * math.hypot(r2 - r1, c2 - c1)
    return total


def visualize_path(grid, path, start_world, goal_world, resolution, arena_size, output_file):
    """Draw occupancy grid and planned path. Returns the path in world coordinates."""
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap="gray_r", origin="lower", interpolation="nearest",
               extent=[-arena_size, arena_size, -arena_size, arena_size], vmin=0, vmax=1)

    path_world = np.array([grid_to_world(n, resolution, arena_size) for n in path])
    plt.plot(path_world[:, 0], path_world[:, 1], color="red", linewidth=2, label="A* Path")
    plt.scatter(*start_world, color="green", s=120, label="Start", zorder=3)
    plt.scatter(*goal_world, color="blue", s=120, marker="*", label="Goal", zorder=3)

    plt.title("VisionNav-Edge A* Path Planning")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.legend()
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(output_file, dpi=180)
    plt.close()

    return path_world


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MAP_FILE.exists():
        raise FileNotFoundError(f"Map not found: {MAP_FILE}")

    data = np.load(MAP_FILE)
    grid = data["grid"]
    resolution = float(data["resolution"])
    arena_size = float(data["arena_size"])
    start_world = data["start"].tolist()
    goal_world = data["goal"].tolist()

    print("=" * 50)
    print("VisionNav-Edge A* Planner")
    print("=" * 50)
    print("Environment:", ENV_NAME)
    print("Map shape:", grid.shape)
    print("Resolution:", resolution, "m")
    print("Start:", start_world)
    print("Goal:", goal_world)

    start = world_to_grid(start_world, resolution, arena_size, grid)
    goal = world_to_grid(goal_world, resolution, arena_size, grid)
    if start is None or goal is None:
        raise ValueError("Start or goal is outside map boundaries")

    print("Start grid:", start)
    print("Goal grid:", goal)

    print("\nSearching path...")
    path, expanded_nodes = astar(grid, start, goal, resolution)

    if path is None:
        print("No valid path found.")
        return

    path_length = calculate_path_length(path, resolution)
    image_file = OUTPUT_DIR / f"{ENV_NAME}_astar.png"
    path_world = visualize_path(grid, path, start_world, goal_world,
                                resolution, arena_size, image_file)

    result = {
        "environment": ENV_NAME,
        "algorithm": "A*",
        "resolution": resolution,
        "start_world": start_world,
        "goal_world": goal_world,
        "start_grid": list(start),
        "goal_grid": list(goal),
        "path_grid": [list(node) for node in path],
        "path_world": path_world.tolist(),
        "path_length_m": path_length,
        "path_nodes": len(path),
        "expanded_nodes": expanded_nodes,
    }
    json_file = OUTPUT_DIR / f"{ENV_NAME}_path.json"
    json_file.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 50)
    print("A* Planning Results")
    print("=" * 50)
    print("Path found: YES")
    print("Path nodes:", len(path))
    print("Expanded nodes:", expanded_nodes)
    print("Path length:", round(path_length, 3), "m")
    print("Visualization:", image_file)
    print("Path data:", json_file)
    print("\nA* PLANNING PASSED")


if __name__ == "__main__":
    main()
