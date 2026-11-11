
"""Save and load VisionNav episodes using JSON metadata."""

import json
from pathlib import Path

from visionnav.core.types import (
    RobotState,
    Action,
    TaskCommand,
    CameraCalibration,
    Observation,
    EpisodeStep,
    Episode,
)

SCHEMA_VERSION = 1


def _observation_to_dict(obs):
    camera = None

    if obs.camera is not None:
        camera = {
            "width": obs.camera.width,
            "height": obs.camera.height,
            "fx": obs.camera.fx,
            "fy": obs.camera.fy,
            "cx": obs.camera.cx,
            "cy": obs.camera.cy,
            "T_world_camera": list(
                obs.camera.T_world_camera
            ),
        }

    return {
        "timestamp_s": obs.timestamp_s,
        "scene_id": obs.scene_id,
        "task_id": obs.task_id,
        "state": {
            "time_s": obs.state.time_s,
            "x_m": obs.state.x_m,
            "y_m": obs.state.y_m,
            "yaw_rad": obs.state.yaw_rad,
            "v_mps": obs.state.v_mps,
            "w_radps": obs.state.w_radps,
            "frame_id": obs.state.frame_id,
        },
        "lidar_m": list(obs.lidar_m),
        "rgb_path": obs.rgb_path,
        "depth_path": obs.depth_path,
        "segmentation_path": obs.segmentation_path,
        "camera": camera,
    }


def _observation_from_dict(data):
    state = RobotState(**data["state"])

    camera_data = data["camera"]
    camera = None

    if camera_data is not None:
        camera_data = dict(camera_data)
        camera_data["T_world_camera"] = tuple(
            camera_data["T_world_camera"]
        )
        camera = CameraCalibration(**camera_data)

    return Observation(
        timestamp_s=data["timestamp_s"],
        scene_id=data["scene_id"],
        task_id=data["task_id"],
        state=state,
        lidar_m=tuple(data["lidar_m"]),
        rgb_path=data["rgb_path"],
        depth_path=data["depth_path"],
        segmentation_path=data["segmentation_path"],
        camera=camera,
    )


def save_episode(episode, output_path):
    """Save an Episode as JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    payload = {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode.episode_id,
        "scene_id": episode.scene_id,

        "task": {
            "task_id": episode.task.task_id,
            "instruction": episode.task.instruction,
            "target_class": episode.task.target_class,
            "target_attributes": episode.task.target_attributes,
            "target_instance_id": episode.task.target_instance_id,
        },

        "success": episode.success,
        "termination_reason": episode.termination_reason,

        "num_steps": len(episode.steps),

        "steps": [
            {
                "observation": _observation_to_dict(
                    step.observation
                ),
                "action": {
                    "v_mps": step.action.v_mps,
                    "w_radps": step.action.w_radps,
                    "dt_s": step.action.dt_s,
                    "source": step.action.source,
                },
                "reward": step.reward,
                "collision": step.collision,
                "done": step.done,
            }
            for step in episode.steps
        ],
    }

    temp_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    temp_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # Atomic replacement after successful serialization.
    temp_path.replace(output_path)

    print(f"Episode saved: {output_path}")


def load_episode(input_path):
    """Restore an Episode from JSON."""

    input_path = Path(input_path)

    data = json.loads(
        input_path.read_text(encoding="utf-8")
    )

    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported Episode schema version")

    if data["num_steps"] != len(data["steps"]):
        raise ValueError("Episode step count mismatch")

    task = TaskCommand(**data["task"])

    episode = Episode(
        episode_id=data["episode_id"],
        scene_id=data["scene_id"],
        task=task,
        success=data["success"],
        termination_reason=data["termination_reason"],
    )

    for item in data["steps"]:

        observation = _observation_from_dict(
            item["observation"]
        )

        action = Action(**item["action"])

        step = EpisodeStep(
            observation=observation,
            action=action,
            reward=item["reward"],
            collision=item["collision"],
            done=item["done"],
        )

        episode.add_step(step)

    return episode


def check_episode_assets(episode, project_root):
    """Check whether referenced RGB/Depth/Mask files exist."""

    project_root = Path(project_root)
    missing = []

    for index, step in enumerate(episode.steps):
        obs = step.observation

        for field in (
            "rgb_path",
            "depth_path",
            "segmentation_path",
        ):
            value = getattr(obs, field)

            if not value:
                continue

            path = Path(value)

            if not path.is_absolute():
                path = project_root / path

            if not path.is_file():
                missing.append({
                    "step": index,
                    "field": field,
                    "path": str(path),
                })

    return missing
