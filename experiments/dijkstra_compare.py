"""
VisionNav-Edge
Experiment 4.3: A* vs Dijkstra

Compare two shortest-path algorithms using exactly the same occupancy grid.

Input:  experiments/results/maps/warehouse_grid.npz
Output: experiments/results/paths/warehouse_astar_dijkstra_comparison.png
        experiments/results/paths/astar_dijkstra_comparison.json
"""

import heapq
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from astar_planner import (
    SQRT2,
    astar,
    calculate_path_length,
    get_neighbors,
    grid_to_world,
    world_to_grid,
)

ROOT = Path(__file__).resolve().parent.parent
MAP_FILE = ROOT / "experiments/results/maps/warehouse_grid.npz"
OUTPUT_DIR = ROOT / "experiments/results/paths"


def dijkstra(grid, start, goal, resolution):
    """Find the shortest path using Dijkstra (no heuristic function)."""
    if grid[start] == 1:
        raise ValueError(f"Start cell is blocked: {start}")
    if grid[goal] == 1:
        raise ValueError(f"Goal cell is blocked: {goal}")

    open_heap = [(0.0, start)]
    g_score = {start: 0.0}
    came_from = {}
    closed_set = set()
    expanded = 0

    while open_heap:
        g, current = heapq.heappop(open_heap)

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
                heapq.heappush(open_heap, (tentative, neighbor))

    return None, expanded


def path_to_world(path, resolution, arena_size):
    return np.array([grid_to_world(n, resolution, arena_size) for n in path])


def draw_comparison(grid, astar_path, dijkstra_path, start_world, goal_world,
                    resolution, arena_size):
    """Draw both paths side by side. Returns the output file path."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    extent = [-arena_size, arena_size, -arena_size, arena_size]

    for ax, (title, path) in zip(axes, [("A* Path", astar_path),
                                        ("Dijkstra Path", dijkstra_path)]):
        ax.imshow(grid, cmap="gray_r", origin="lower", interpolation="nearest",
                  extent=extent, vmin=0, vmax=1)
        world_path = path_to_world(path, resolution, arena_size)
        ax.plot(world_path[:, 0], world_path[:, 1], color="red", linewidth=2, label=title)
        ax.scatter(*start_world, color="green", s=90, label="Start")
        ax.scatter(*goal_world, color="blue", marker="*", s=120, label="Goal")
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend()
        ax.grid(alpha=0.15)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "warehouse_astar_dijkstra_comparison.png"
    plt.savefig(output_file, dpi=180)
    plt.close()

    return output_file


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

    start = world_to_grid(start_world, resolution, arena_size, grid)
    goal = world_to_grid(goal_world, resolution, arena_size, grid)
    if start is None or goal is None:
        raise ValueError("Start or goal is outside map boundaries")

    print("=" * 55)
    print("VisionNav-Edge: A* vs Dijkstra")
    print("=" * 55)
    print("Map shape:", grid.shape)
    print("Resolution:", resolution)
    print("Start:", start)
    print("Goal:", goal)

    print("\nRunning A*...")
    astar_path, astar_expanded = astar(grid, start, goal, resolution)

    print("Running Dijkstra...")
    dijkstra_path, dijkstra_expanded = dijkstra(grid, start, goal, resolution)

    if astar_path is None:
        raise RuntimeError("A* failed")
    if dijkstra_path is None:
        raise RuntimeError("Dijkstra failed")

    astar_length = calculate_path_length(astar_path, resolution)
    dijkstra_length = calculate_path_length(dijkstra_path, resolution)

    print("\n" + "=" * 55)
    print("Comparison Results")
    print("=" * 55)
    print("\nA*:")
    print("Path nodes:", len(astar_path))
    print("Expanded nodes:", astar_expanded)
    print("Path length:", round(astar_length, 3), "m")
    print("\nDijkstra:")
    print("Path nodes:", len(dijkstra_path))
    print("Expanded nodes:", dijkstra_expanded)
    print("Path length:", round(dijkstra_length, 3), "m")

    reduction = None
    if dijkstra_expanded > 0:
        reduction = (dijkstra_expanded - astar_expanded) / dijkstra_expanded * 100
        print("\nA* expansion reduction:", round(reduction, 2), "%")

    same_length = math.isclose(astar_length, dijkstra_length,
                               rel_tol=1e-8, abs_tol=1e-8)
    print("\nSame optimal path cost:", same_length)
    if not same_length:
        raise AssertionError("A* and Dijkstra path lengths differ!")

    image_file = draw_comparison(grid, astar_path, dijkstra_path,
                                 start_world, goal_world, resolution, arena_size)

    result = {
        "environment": "warehouse",
        "map_resolution": resolution,
        "astar": {
            "path_nodes": len(astar_path),
            "expanded_nodes": astar_expanded,
            "path_length_m": astar_length,
        },
        "dijkstra": {
            "path_nodes": len(dijkstra_path),
            "expanded_nodes": dijkstra_expanded,
            "path_length_m": dijkstra_length,
        },
        "same_optimal_path_cost": same_length,
        "astar_expansion_reduction_percent": reduction,
    }
    json_file = OUTPUT_DIR / "astar_dijkstra_comparison.json"
    json_file.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    print("\nComparison image:", image_file)
    print("Comparison data:", json_file)
    print("\nASTAR DIJKSTRA COMPARISON PASSED")


if __name__ == "__main__":
    main()
