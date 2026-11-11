
"""Build 2D occupancy grids from PyBullet environments."""

import sys
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import pybullet_data


# Locate project and original environment builder
ROOT = Path(__file__).resolve().parent.parent
BUILDER_FILE = ROOT / "scripts" / "05_environment_builder.py"
OUTPUT_DIR = ROOT / "experiments" / "results" / "maps"

# Map configuration
RESOLUTION = 0.1       # Meters per cell
SAFETY_MARGIN = 0.25   # Obstacle inflation in meters

START = (0.0, 0.0)
GOAL = (6.0, 6.0)


def load_environment_builder():
    """Import the original environment builder."""
    spec = importlib.util.spec_from_file_location(
        "environment_builder", BUILDER_FILE
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.EnvironmentBuilder


def build_scene(env_type):
    """Create an original PyBullet environment."""
    client = p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")

    EnvironmentBuilder = load_environment_builder()
    builder = EnvironmentBuilder(client)

    if env_type == "warehouse":
        builder.create_warehouse_environment()
        arena_size = 10.0

    elif env_type == "street":
        builder.create_street_environment()
        arena_size = 10.0

    elif env_type == "maze":
        builder.create_simple_maze()
        arena_size = 8.0

    else:
        raise ValueError(f"Unknown environment: {env_type}")

    return plane_id, arena_size


def collect_obstacle_aabbs(plane_id):
    """Collect bounding boxes of all scene obstacles."""
    obstacles = []

    for i in range(p.getNumBodies()):
        body_id = p.getBodyUniqueId(i)

        # Ground is not treated as a 2D obstacle
        if body_id == plane_id:
            continue

        aabb_min, aabb_max = p.getAABB(body_id)

        obstacles.append((
            aabb_min[0],
            aabb_min[1],
            aabb_max[0],
            aabb_max[1],
        ))

    return obstacles


def create_grid(obstacles, arena_size):
    """Rasterize inflated obstacles into a 2D grid."""
    grid_size = int(round(2 * arena_size / RESOLUTION))

    grid = np.zeros(
        (grid_size, grid_size),
        dtype=np.uint8
    )

    # Cell-center coordinates
    coords = (
        -arena_size
        + (np.arange(grid_size) + 0.5) * RESOLUTION
    )

    x_coords = coords[None, :]
    y_coords = coords[:, None]

    for xmin, ymin, xmax, ymax in obstacles:
        # Inflate obstacles to create clearance
        xmin -= SAFETY_MARGIN
        ymin -= SAFETY_MARGIN
        xmax += SAFETY_MARGIN
        ymax += SAFETY_MARGIN

        occupied_x = (x_coords >= xmin) & (x_coords <= xmax)
        occupied_y = (y_coords >= ymin) & (y_coords <= ymax)

        grid[occupied_y & occupied_x] = 1

    return grid


def world_to_grid(point, arena_size, grid):
    """Convert world coordinates into grid row and column."""
    x, y = point

    col = int(np.floor((x + arena_size) / RESOLUTION))
    row = int(np.floor((y + arena_size) / RESOLUTION))

    if (
        row < 0 or row >= grid.shape[0]
        or col < 0 or col >= grid.shape[1]
    ):
        return None

    return row, col


def check_position(grid, point, arena_size):
    """Report whether a world position is traversable."""
    index = world_to_grid(point, arena_size, grid)

    if index is None:
        return "OUT_OF_BOUNDS"

    row, col = index

    if grid[row, col] == 1:
        return "BLOCKED"

    return "FREE"


def visualize_map(grid, env_type, arena_size):
    """Save a plot of the generated occupancy grid."""
    plt.figure(figsize=(8, 8))

    plt.imshow(
        grid,
        cmap="gray_r",
        origin="lower",
        interpolation="nearest",
        extent=[
            -arena_size, arena_size,
            -arena_size, arena_size
        ],
        vmin=0,
        vmax=1
    )

    plt.scatter(
        *START,
        color="green",
        s=100,
        marker="o",
        label="Start",
        zorder=3
    )

    plt.scatter(
        *GOAL,
        color="red",
        s=130,
        marker="*",
        label="Original Goal",
        zorder=3
    )

    plt.title(f"{env_type.capitalize()} Occupancy Grid")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{env_type}_grid.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    return output_file


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    EnvironmentBuilder = load_environment_builder()
    print("Environment builder loaded:", EnvironmentBuilder.__name__)

    for env_type in ("warehouse", "street", "maze"):
        print("\n" + "=" * 45)
        print("Processing:", env_type)

        plane_id = None

        try:
            plane_id, arena_size = build_scene(env_type)

            obstacles = collect_obstacle_aabbs(plane_id)
            grid = create_grid(obstacles, arena_size)

            start_status = check_position(
                grid, START, arena_size
            )
            goal_status = check_position(
                grid, GOAL, arena_size
            )

            map_file = OUTPUT_DIR / f"{env_type}_grid.npz"

            np.savez_compressed(
                map_file,
                grid=grid,
                resolution=RESOLUTION,
                arena_size=arena_size,
                safety_margin=SAFETY_MARGIN,
                start=np.array(START),
                goal=np.array(GOAL),
            )

            image_file = visualize_map(
                grid, env_type, arena_size
            )

            print("Map shape:", grid.shape)
            print("Obstacle bodies:", len(obstacles))
            print("Occupied cells:", int(grid.sum()))
            print("Start:", start_status)
            print("Goal:", goal_status)
            print("Map file:", map_file)
            print("Image file:", image_file)

            if start_status != "FREE":
                print("WARNING: Start position is invalid")

            if goal_status != "FREE":
                print("WARNING: Goal position is invalid")

        finally:
            if p.isConnected():
                p.disconnect()

    print("\nAll three maps generated.")


if __name__ == "__main__":
    main()
