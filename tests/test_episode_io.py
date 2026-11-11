
import tempfile
import unittest
from pathlib import Path

from visionnav.core.types import (
    RobotState,
    Action,
    TaskCommand,
    Observation,
    EpisodeStep,
    Episode,
)

from visionnav.core.episode_io import (
    save_episode,
    load_episode,
)


class TestEpisodeIO(unittest.TestCase):

    def test_episode_round_trip(self):

        state = RobotState(
            time_s=0.0,
            x_m=1.0,
            y_m=2.0,
            yaw_rad=0.5,
            v_mps=0.2,
            w_radps=0.1,
        )

        task = TaskCommand(
            task_id="task_test",
            instruction="前往红色箱子旁边",
            target_class="box",
            target_attributes={"color": "red"},
        )

        observation = Observation(
            timestamp_s=0.0,
            scene_id="warehouse",
            task_id="task_test",
            state=state,
            lidar_m=(5.0,) * 36,
        )

        action = Action(
            v_mps=0.3,
            w_radps=0.1,
            dt_s=1 / 30,
            source="expert",
        )

        episode = Episode(
            episode_id="episode_test",
            scene_id="warehouse",
            task=task,
        )

        episode.add_step(
            EpisodeStep(
                observation=observation,
                action=action,
                reward=1.5,
                collision=False,
                done=False,
            )
        )

        with tempfile.TemporaryDirectory() as tmp:

            path = Path(tmp) / "episode.json"

            save_episode(episode, path)

            restored = load_episode(path)

            self.assertEqual(
                restored.episode_id,
                episode.episode_id,
            )

            self.assertEqual(
                len(restored.steps),
                1,
            )

            self.assertEqual(
                restored.steps[0].action.v_mps,
                0.3,
            )

            self.assertEqual(
                restored.steps[0].reward,
                1.5,
            )

            self.assertEqual(
                restored.task.instruction,
                "前往红色箱子旁边",
            )

            self.assertEqual(
                restored.steps[0].observation.lidar_m,
                (5.0,) * 36,
            )


if __name__ == "__main__":
    unittest.main()
