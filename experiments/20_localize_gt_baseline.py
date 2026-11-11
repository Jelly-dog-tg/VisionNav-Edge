
"""P3-1: RGB-D localization using ground-truth instance masks."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "logs" / "p2_dataset"


def load_array(path):
    return np.load(
        ROOT / path,
        allow_pickle=False,
    )


def localize_target(sample):
    inputs = sample["model_input"]
    gt = sample["ground_truth"]

    depth = load_array(inputs["depth_path"])
    mask = load_array(gt["mask_path"])

    camera = inputs["camera"]

    assert depth.shape == mask.shape

    # Select pixels belonging to the target instance.
    pixels = mask == gt["mask_id"]

    if not np.any(pixels):
        raise ValueError(
            f"Target not visible: {gt['instance_id']}"
        )

    # Pixel positions: u = column, v = row.
    v, u = np.where(pixels)

    z = depth[pixels].astype(np.float64)

    valid = np.isfinite(z) & (z > 0)

    u = u[valid]
    v = v[valid]
    z = z[valid]

    if len(z) == 0:
        raise ValueError("No valid target depth")

    fx = camera["fx"]
    fy = camera["fy"]
    cx = camera["cx"]
    cy = camera["cy"]

    # Pixel + depth -> camera coordinates.
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points_camera = np.stack(
        [x, y, z],
        axis=1,
    )

    T = np.array(
        camera["T_world_camera"],
        dtype=np.float64,
    ).reshape(4, 4)

    R = T[:3, :3]
    t = T[:3, 3]

    # Camera coordinates -> world coordinates.
    points_world = points_camera @ R.T + t

    # Robust estimate of visible target surface.
    estimated = np.median(
        points_world,
        axis=0,
    )

    true_center = np.asarray(
        gt["position_m"],
        dtype=np.float64,
    )

    # Navigation uses ground-plane XY coordinates.
    error_xy = float(
        np.linalg.norm(
            estimated[:2] - true_center[:2]
        )
    )

    return {
        "sample_id": sample["sample_id"],
        "instruction": inputs["instruction"],
        "estimated_surface_xyz_m": estimated.tolist(),
        "true_center_xyz_m": true_center.tolist(),
        "error_xy_m": error_xy,
    }


def main():
    dataset = json.loads(
        (DATA_DIR / "dataset_index.json").read_text(
            encoding="utf-8"
        )
    )

    results = []

    for sample in dataset["samples"]:
        result = localize_target(sample)
        results.append(result)

        print(
            f"{result['sample_id']} | "
            f"XY error={result['error_xy_m']:.3f} m"
        )

    errors = np.array(
        [item["error_xy_m"] for item in results]
    )

    summary = {
        "method": "ground_truth_mask_rgbd",
        "sample_count": len(results),
        "mean_xy_error_m": float(errors.mean()),
        "max_xy_error_m": float(errors.max()),
        "results": results,
    }

    output = DATA_DIR / "p3_gt_baseline_results.json"

    output.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nP3-1 LOCALIZATION COMPLETED")
    print(f"Samples: {len(results)}")
    print(f"Mean XY error: {errors.mean():.3f} m")
    print(f"Max XY error: {errors.max():.3f} m")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
