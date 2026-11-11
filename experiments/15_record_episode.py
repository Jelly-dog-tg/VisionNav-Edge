
"""P1-3: Record and reload a complete robot Episode."""

import sys
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pybullet_nav_env import PyBulletNavEnv

from visionnav.core.types import (
    Action,
    TaskCommand,
    Episode,
    EpisodeStep,
)

from visionnav.core.episode_io import (
    save_episode,
    load_episode,
    check_episode_assets,
)

from visionnav.simulation.observation_adapter import (
    PyBulletObservationAdapter,
)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    episode_id = (
        f"warehouse_seed{args.seed}_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    task_id = episode_id + "_task"

    env = PyBulletNavEnv(
        render_mode=None,
        env_type="warehouse",
    )

    try:
        _, info = env.reset(seed=args.seed)

        task = TaskCommand(
            task_id=task_id,
            instruction="记录机器人在 Warehouse 中的短距离运动",
        )

        episode = Episode(
            episode_id=episode_id,
            scene_id="warehouse",
            task=task,
        )

        adapter = PyBulletObservationAdapter(
            robot=env.robot,
            output_dir=(
                Path("logs")
                / "episodes"
                / "assets"
                / episode_id
            ),
            width=160,
            height=120,
        )

        print("Recording Episode...")

        max_steps = 12

        for step_index in range(max_steps):

            # 1. Capture observation BEFORE action.
            observation = adapter.capture(
                timestamp_s=step_index * env.time_step,
                scene_id="warehouse",
                task_id=task_id,
                frame_index=step_index,
            )

            # 2. Execute a small test action.
            command = np.array(
                [0.15, 0.0],
                dtype=np.float32,
            )

            _, reward, terminated, truncated, info = env.step(
                command
            )

            # 3. Record action and transition result.
            action = Action(
                v_mps=float(command[0]),
                w_radps=float(command[1]),
                dt_s=float(env.time_step),
                source="manual_smoke_test",
            )

            done = bool(terminated or truncated)

            episode.add_step(
                EpisodeStep(
                    observation=observation,
                    action=action,
                    reward=float(reward),
                    collision=bool(info.get("collision", False)),
                    done=done,
                )
            )

            if step_index % 3 == 0:
                print(
                    f"Step {step_index}: "
                    f"position=("
                    f"{observation.state.x_m:.3f}, "
                    f"{observation.state.y_m:.3f}), "
                    f"reward={reward:.3f}"
                )

            if done:
                break

        # 4. Set Episode result.
        if terminated:
            episode.success = True
            episode.termination_reason = "goal_reached"

        elif truncated:
            episode.success = False
            episode.termination_reason = (
                "collision"
                if info.get("collision", False)
                else "time_limit"
            )

        else:
            episode.success = None
            episode.termination_reason = "capture_limit_reached"

        # 5. Save Episode.
        output_path = (
            ROOT
            / "logs"
            / "episodes"
            / f"{episode_id}.json"
        )

        save_episode(episode, output_path)

        # 6. Reload Episode.
        restored = load_episode(output_path)

        # 7. Verify data.
        assert restored.episode_id == episode.episode_id
        assert len(restored.steps) == len(episode.steps)
        assert restored.scene_id == "warehouse"

        missing = check_episode_assets(
            restored,
            ROOT,
        )

        assert len(missing) == 0, (
            f"Missing assets: {missing}"
        )

        print("\nEPISODE VERIFICATION PASSED")
        print(f"Episode ID: {restored.episode_id}")
        print(f"Scene: {restored.scene_id}")
        print(f"Steps: {len(restored.steps)}")
        print(f"Success: {restored.success}")
        print(f"Termination: {restored.termination_reason}")
        print(f"Missing assets: {len(missing)}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
