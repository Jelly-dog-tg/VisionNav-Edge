
"""P2-2: Build language-conditioned target index."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "logs" / "p2_semantic"


def locate(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def relative_path(path):
    return locate(path).resolve().relative_to(ROOT).as_posix()


def main():
    metadata = json.loads(
        (DATA_DIR / "targets.json").read_text(
            encoding="utf-8"
        )
    )

    mask = np.load(
        locate(metadata["instance_mask_path"]),
        allow_pickle=False,
    )

    color_names = {
        "red": "红色",
        "blue": "蓝色",
        "green": "绿色",
    }

    class_names = {
        "box": "箱子",
        "cylinder": "圆柱体",
        "sphere": "球体",
    }

    samples = []

    for target in metadata["targets"]:
        mask_id = target["mask_id"]

        pixel_count = int(
            np.count_nonzero(mask == mask_id)
        )

        if pixel_count == 0:
            raise ValueError(
                f"Target has no visible pixels: "
                f"{target['instance_id']}"
            )

        instruction = (
            "找到"
            + color_names[target["color"]]
            + class_names[target["class"]]
        )

        sample = {
            "sample_id": target["instance_id"],
            "scene_id": metadata["scene_id"],

            "model_input": {
                "instruction": instruction,
                "rgb_path": relative_path(
                    metadata["rgb_path"]
                ),
                "depth_path": relative_path(
                    metadata["depth_path"]
                ),
            },

            "ground_truth": {
                "instance_id": target["instance_id"],
                "mask_id": mask_id,
                "class": target["class"],
                "color": target["color"],
                "position_m": target["position"],
                "visible_pixels": pixel_count,
                "mask_path": relative_path(
                    metadata["instance_mask_path"]
                ),
            },
        }

        samples.append(sample)

        print(
            f"{instruction} -> "
            f"{target['instance_id']} "
            f"({pixel_count} pixels)"
        )

    output = DATA_DIR / "dataset_index.json"

    output.write_text(
        json.dumps(
            {"schema_version": 1, "samples": samples},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nP2-2 PASSED: {len(samples)} samples")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
