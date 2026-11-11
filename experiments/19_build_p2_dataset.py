
"""P2-3: Build and validate layout-separated semantic data."""

import sys
import json
from pathlib import Path
from dataclasses import asdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pybullet_nav_env import PyBulletNavEnv
from visionnav.simulation.semantic_scene import (
    add_semantic_targets,
)
from visionnav.simulation.observation_adapter import (
    PyBulletObservationAdapter,
)


OUTPUT = ROOT / "logs" / "p2_dataset"

COLOR_ZH = {
    "red": "红色",
    "blue": "蓝色",
    "green": "绿色",
}

CLASS_ZH = {
    "box": "箱子",
    "cylinder": "圆柱体",
    "sphere": "球体",
}


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def collect_scene(split):
    env = PyBulletNavEnv(
        render_mode=None,
        env_type="warehouse",
    )

    try:
        env.reset(seed=42)
        env.robot.reset([0.0, 0.0], yaw=0.0)

        targets = add_semantic_targets(layout=split)

        scene_id = f"warehouse_{split}"
        folder = OUTPUT / split

        adapter = PyBulletObservationAdapter(
            robot=env.robot,
            output_dir=folder,
            width=640,
            height=480,
        )

        obs = adapter.capture(
            timestamp_s=0.0,
            scene_id=scene_id,
            task_id=f"semantic_{split}",
            frame_index=0,
        )

        raw_mask = np.load(
            obs.segmentation_path,
            allow_pickle=False,
        )

        depth = np.load(
            obs.depth_path,
            allow_pickle=False,
        )

        if raw_mask.shape != depth.shape:
            raise ValueError("Depth/mask size mismatch")

        if not np.all(np.isfinite(depth)):
            raise ValueError("Invalid depth values")

        body_mask = np.where(
            raw_mask >= 0,
            raw_mask.astype(np.int64) & 0xFFFFFF,
            -1,
        )

        instance_mask = np.zeros(
            raw_mask.shape,
            dtype=np.uint8,
        )

        samples = []

        for target in targets:
            pixels = body_mask == target["body_id"]

            count = int(np.count_nonzero(pixels))

            if count == 0:
                raise RuntimeError(
                    f"Invisible target in {split}: "
                    f"{target['instance_id']}"
                )

            instance_mask[pixels] = target["mask_id"]

            instruction = (
                "找到"
                + COLOR_ZH[target["color"]]
                + CLASS_ZH[target["class"]]
            )

            samples.append({
                "sample_id": (
                    f"{split}_{target['instance_id']}"
                ),
                "split": split,
                "scene_id": scene_id,
                "model_input": {
                    "instruction": instruction,
                    "rgb_path": relative(obs.rgb_path),
                    "depth_path": relative(obs.depth_path),
                    "camera": asdict(obs.camera),
                },
                "ground_truth": {
                    "instance_id": target["instance_id"],
                    "class": target["class"],
                    "color": target["color"],
                    "mask_id": target["mask_id"],
                    "position_m": target["position"],
                    "visible_pixels": count,
                    "mask_path": relative(
                        folder / "instance_mask.npy"
                    ),
                },
            })

            print(
                f"  {instruction}: "
                f"{count} visible pixels"
            )

        np.save(
            folder / "instance_mask.npy",
            instance_mask,
        )

        return samples

    finally:
        env.close()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    all_samples = []

    for split in ("train", "val", "test"):
        print(f"\nGenerating {split} layout...")
        all_samples.extend(collect_scene(split))

    if len(all_samples) != 9:
        raise RuntimeError("Expected 9 task samples")

    dataset = {
        "schema_version": 1,
        "split_method": "distinct_target_layouts",
        "samples": all_samples,
    }

    index_path = OUTPUT / "dataset_index.json"

    index_path.write_text(
        json.dumps(
            dataset,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    print("\nP2 DATASET VALIDATION PASSED")
    print("Train: 3 samples")
    print("Val:   3 samples")
    print("Test:  3 samples")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
