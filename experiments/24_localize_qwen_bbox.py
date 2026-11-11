
"""P3: Localize Qwen-predicted targets using RGB-D geometry."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "logs" / "p2_dataset"

TARGET_NAMES = {
    "red": "red box",
    "blue": "blue cylinder",
    "green": "green sphere",
}


def estimate_position(bbox, depth, camera):
    """Estimate a visible target surface point in world frame."""

    height, width = depth.shape
    x1, y1, x2, y2 = bbox

    # Use the central 60% of the predicted box to reduce
    # background contamination.
    margin_x = (x2 - x1) * 0.2
    margin_y = (y2 - y1) * 0.2

    left = max(0, int(x1 + margin_x))
    right = min(width, int(np.ceil(x2 - margin_x)))
    top = max(0, int(y1 + margin_y))
    bottom = min(height, int(np.ceil(y2 - margin_y)))

    if right <= left or bottom <= top:
        raise ValueError("Invalid target region")

    region = depth[top:bottom, left:right]

    vv, uu = np.mgrid[top:bottom, left:right]

    valid = np.isfinite(region) & (region > 0)

    if not np.any(valid):
        raise ValueError("No valid depth in target region")

    # Reject depth values far from the central depth cluster.
    median_depth = np.median(region[valid])

    valid &= np.abs(region - median_depth) < 0.15

    if np.count_nonzero(valid) < 10:
        raise ValueError("Insufficient target depth pixels")

    z = region[valid].astype(np.float64)
    u = uu[valid]
    v = vv[valid]

    # Pixel coordinates -> camera coordinates.
    x = (u - camera["cx"]) * z / camera["fx"]
    y = (v - camera["cy"]) * z / camera["fy"]

    points_camera = np.stack([x, y, z], axis=1)

    # Camera coordinates -> world coordinates.
    T = np.asarray(
        camera["T_world_camera"], dtype=np.float64
    ).reshape(4, 4)

    points_world = (
        points_camera @ T[:3, :3].T + T[:3, 3]
    )

    return np.median(points_world, axis=0)


def main():
    dataset = json.loads(
        (DATA_DIR / "dataset_index.json").read_text(
            encoding="utf-8"
        )
    )

    predictions = json.loads(
        (DATA_DIR / "p3_qwen_bboxes.json").read_text(
            encoding="utf-8"
        )
    )

    bbox_by_name = {
        item["target"]: item["bbox_pixels"]
        for item in predictions
    }

    samples = [
        s for s in dataset["samples"]
        if s["split"] == "test"
    ]

    if len(samples) != 3:
        raise ValueError("Expected three test targets")

    results = []

    for sample in samples:
        gt = sample["ground_truth"]
        inputs = sample["model_input"]

        target_name = TARGET_NAMES[gt["color"]]
        bbox = bbox_by_name.get(target_name)

        if bbox is None:
            raise ValueError(
                f"Missing predicted bbox: {target_name}"
            )

        depth = np.load(
            ROOT / inputs["depth_path"],
            allow_pickle=False,
        )

        camera = inputs["camera"]

        if depth.shape != (
            camera["height"], camera["width"]
        ):
            raise ValueError("Camera/depth size mismatch")

        estimated = estimate_position(
            bbox, depth, camera
        )

        # Ground truth is used only for evaluation.
        true_center = np.asarray(
            gt["position_m"],
            dtype=np.float64,
        )

        error_xy = float(
            np.linalg.norm(
                estimated[:2] - true_center[:2]
            )
        )

        results.append({
            "target": target_name,
            "estimated_surface_xyz_m": estimated.tolist(),
            "true_center_xyz_m": true_center.tolist(),
            "error_xy_m": error_xy,
        })

        print(
            f"{target_name}: "
            f"estimated=({estimated[0]:.3f}, "
            f"{estimated[1]:.3f}), "
            f"XY error={error_xy:.3f} m"
        )

    errors = [r["error_xy_m"] for r in results]

    output = DATA_DIR / "p3_qwen_localization.json"

    output.write_text(
        json.dumps(
            {
                "method": "qwen_bbox_rgbd",
                "mean_xy_error_m": float(np.mean(errors)),
                "max_xy_error_m": float(np.max(errors)),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nP3 LOCALIZATION COMPLETED")
    print(f"Mean XY error: {np.mean(errors):.3f} m")
    print(f"Max XY error: {np.max(errors):.3f} m")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
