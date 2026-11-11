"""
VisionNav-Edge - Experiment 4.4

A* path simplification using line-of-sight checks.

Inputs:  experiments/results/maps/warehouse_grid.npz
         experiments/results/paths/warehouse_path.json
Outputs: experiments/results/paths/warehouse_smoothed.png
         experiments/results/paths/warehouse_waypoints.json
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from astar_planner import grid_to_world

ROOT = Path(__file__).resolve().parent.parent
MAP_FILE = ROOT / "experiments/results/maps/warehouse_grid.npz"
PATH_FILE = ROOT / "experiments/results/paths/warehouse_path.json"
OUTPUT_DIR = ROOT / "experiments/results/paths"


def segment_intersects_box(a, b, xmin, ymin, xmax, ymax):
    """
    Test whether a line segment touches a grid-cell box.

    Touching an obstacle boundary is treated as a collision.
    """
    x0, y0 = a
    x1, y1 = b
    dx = x1 - x0
    dy = y1 - y0

    tmin, tmax = 0.0, 1.0
    for origin, direction, low, high in (
        (x0, dx, xmin, xmax),
        (y0, dy, ymin, ymax),
    ):
        if abs(direction) < 1e-12:
            if origin < low or origin > high:
                return False
            continue

        t1 = (low - origin) / direction
        t2 = (high - origin) / direction
        tmin = max(tmin, min(t1, t2))
        tmax = min(tmax, max(t1, t2))
        if tmin > tmax:
            return False

    return True


def line_of_sight(grid, a, b, resolution, arena_size):
    """Check whether a straight segment between two cell centers intersects an occupied cell."""
    p0 = grid_to_world(a, resolution, arena_size)
    p1 = grid_to_world(b, resolution, arena_size)

    min_x, max_x = min(p0[0], p1[0]), max(p0[0], p1[0])
    min_y, max_y = min(p0[1], p1[1]), max(p0[1], p1[1])

    # Candidate cell index range, padded by one cell on each side
    c0 = max(0, math.floor((min_x + arena_size) / resolution) - 1)
    c1 = min(grid.shape[1] - 1, math.floor((max_x + arena_size) / resolution) + 1)
    r0 = max(0, math.floor((min_y + arena_size) / resolution) - 1)
    r1 = min(grid.shape[0] - 1, math.floor((max_y + arena_size) / resolution) + 1)

    for row in range(r0, r1 + 1):
        for col in range(c0, c1 + 1):
            if grid[row, col] == 0:
                continue
            xmin = -arena_size + col * resolution
            ymin = -arena_size + row * resolution
            if segment_intersects_box(p0, p1, xmin, ymin,
                                      xmin + resolution, ymin + resolution):
                return False

    return True


def simplify_path(grid, path, resolution, arena_size):
    """Keep the farthest visible future waypoint."""
    if len(path) <= 2:
        return path.copy()

    simplified = [path[0]]
    current = 0
    last = len(path) - 1

    while current < last:
        # Search backward from the goal for the farthest visible waypoint
        next_index = None
        for candidate in range(last, current, -1):
            if line_of_sight(grid, path[current], path[candidate],
                             resolution, arena_size):
                next_index = candidate
                break

        if next_index is None:
            raise RuntimeError("No collision-free next waypoint found")

        simplified.append(path[next_index])
        current = next_index

    return simplified


def path_length(path, resolution, arena_size):
    if len(path) < 2:
        return 0.0
    points = np.asarray([grid_to_world(p, resolution, arena_size) for p in path])
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def validate_path(grid, path, resolution, arena_size):
    for i in range(len(path) - 1):
        if not line_of_sight(grid, path[i], path[i + 1], resolution, arena_size):
            raise AssertionError(f"Segment {i} intersects an obstacle")
    return True


def plot_comparison(grid, original, simplified, resolution, arena_size):
    """Draw original path and simplified waypoints side by side. Returns the image file."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    extent = [-arena_size, arena_size, -arena_size, arena_size]

    for ax, (title, path, color) in zip(axes, [
            ("Original A* Path", original, "red"),
            ("Simplified Waypoints", simplified, "blue")]):
        ax.imshow(grid, cmap="gray_r", origin="lower", extent=extent,
                  interpolation="nearest", vmin=0, vmax=1)
        points = np.asarray([grid_to_world(p, resolution, arena_size) for p in path])
        ax.plot(points[:, 0], points[:, 1], "-o", color=color,
                linewidth=2, markersize=3)
        ax.scatter(points[0, 0], points[0, 1], color="green", s=90,
                   label="Start", zorder=5)
        ax.scatter(points[-1, 0], points[-1, 1], color="orange",
                   marker="*", s=130, label="Goal", zorder=5)
        ax.set_title(f"{title} ({len(path)} nodes)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect("equal")
        ax.legend()
        ax.grid(alpha=0.15)

    plt.tight_layout()
    image_file = OUTPUT_DIR / "warehouse_smoothed.png"
    plt.savefig(image_file, dpi=180)
    plt.close()

    return image_file


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with np.load(MAP_FILE) as data:
        grid = data["grid"]
        resolution = float(data["resolution"])
        arena_size = float(data["arena_size"])

    with PATH_FILE.open("r", encoding="utf-8") as f:
        path_data = json.load(f)

    original = [tuple(node) for node in path_data["path_grid"]]
    if not original:
        raise ValueError("Original A* path is empty")

    print("=" * 50)
    print("VisionNav-Edge - Path Simplification")
    print("=" * 50)
    print("Original nodes:", len(original))

    simplified = simplify_path(grid, original, resolution, arena_size)

    # Verify that endpoints are preserved
    assert simplified[0] == original[0]
    assert simplified[-1] == original[-1]

    valid = validate_path(grid, simplified, resolution, arena_size)
    original_length = path_length(original, resolution, arena_size)
    simplified_length = path_length(simplified, resolution, arena_size)
    reduction = 100 * (len(original) - len(simplified)) / len(original)

    result = {
        "environment": "warehouse",
        "algorithm": "A* + line-of-sight simplification",
        "resolution": resolution,
        "arena_size": arena_size,
        "original_nodes": len(original),
        "waypoint_nodes": len(simplified),
        "original_length_m": original_length,
        "simplified_length_m": simplified_length,
        "node_reduction_percent": reduction,
        "collision_free_on_grid": valid,
        "waypoints_grid": [list(node) for node in simplified],
        "waypoints_world": [grid_to_world(node, resolution, arena_size)
                            for node in simplified],
    }
    output_json = OUTPUT_DIR / "warehouse_waypoints.json"
    output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    image_file = plot_comparison(grid, original, simplified, resolution, arena_size)

    print("\nOriginal nodes:", len(original))
    print("Simplified nodes:", len(simplified))
    print("Node reduction:", round(reduction, 2), "%")
    print("Original length:", round(original_length, 3), "m")
    print("Simplified length:", round(simplified_length, 3), "m")
    print("Collision-free on grid:", valid)
    print("Image:", image_file)
    print("JSON:", output_json)
    print("\nPATH SIMPLIFICATION PASSED")


if __name__ == "__main__":
    main()
