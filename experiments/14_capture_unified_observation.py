
"""P1-2: Capture unified observations in the real warehouse env."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.pybullet_nav_env import PyBulletNavEnv
from visionnav.simulation.observation_adapter import (
    PyBulletObservationAdapter,
)


def main():
    env = PyBulletNavEnv(
        render_mode=None,
        env_type="warehouse",
    )

    captured = []

    try:
        env.reset(seed=42)

        adapter = PyBulletObservationAdapter(
            robot=env.robot,
            output_dir="logs/p1_observations",
            width=160,
            height=120,
        )

        for step in range(12):
            # Small forward movement in the existing environment.
            action = np.array([0.15, 0.0], dtype=np.float32)

            _, reward, terminated, truncated, info = env.step(
                action
            )

            # Capture every 3 environment steps.
            if step % 3 == 0:
                obs = adapter.capture(
                    timestamp_s=(step + 1) * env.time_step,
                    scene_id="warehouse",
                    task_id="p1_sensor_smoke",
                    frame_index=step,
                )

                assert len(obs.lidar_m) == 36
                assert obs.camera is not None
                assert Path(obs.rgb_path).exists()
                assert Path(obs.depth_path).exists()

                depth = np.load(obs.depth_path)

                assert depth.shape == (120, 160)
                assert np.all(np.isfinite(depth))

                captured.append(obs)

                print(
                    f"Frame {step}: "
                    f"t={obs.timestamp_s:.3f}s, "
                    f"pose=({obs.state.x_m:.2f}, "
                    f"{obs.state.y_m:.2f}), "
                    f"LiDAR={len(obs.lidar_m)}, "
                    f"depth={depth.shape}"
                )

            if terminated or truncated:
                break

        print("\nP1-2 CAPTURE COMPLETE")
        print(f"Observations collected: {len(captured)}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
