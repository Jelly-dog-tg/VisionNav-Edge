
import unittest

from visionnav.core.types import (
    RobotState,
    Action,
    TaskCommand,
    Observation,
    EpisodeStep,
    Episode,
)


class TestP1Types(unittest.TestCase):

    def test_robot_state(self):
        state = RobotState(
            time_s=0.0,
            x_m=1.0,
            y_m=2.0,
            yaw_rad=0.5,
            v_mps=0.2,
            w_radps=0.1,
        )
        self.assertEqual(state.x_m, 1.0)

    def test_action(self):
        action = Action(
            v_mps=0.3,
            w_radps=0.2,
            dt_s=1 / 30,
            source="expert",
        )
        self.assertEqual(action.source, "expert")

    def test_invalid_action(self):
        with self.assertRaises(ValueError):
            Action(
                v_mps=0.3,
                w_radps=0.2,
                dt_s=-1.0,
            )

    def test_episode(self):
        state = RobotState(
            0.0, 1.0, 2.0, 0.0, 0.0, 0.0
        )

        task = TaskCommand(
            task_id="task_001",
            instruction="前往红色箱子旁边",
            target_class="box",
            target_attributes={"color": "red"},
        )

        observation = Observation(
            timestamp_s=0.0,
            scene_id="warehouse",
            task_id="task_001",
            state=state,
            lidar_m=(5.0,) * 36,
        )

        action = Action(
            v_mps=0.3,
            w_radps=0.0,
            dt_s=1 / 30,
            source="expert",
        )

        episode = Episode(
            episode_id="ep_001",
            scene_id="warehouse",
            task=task,
        )

        episode.add_step(
            EpisodeStep(
                observation=observation,
                action=action,
            )
        )

        self.assertEqual(len(episode.steps), 1)
        self.assertEqual(
            episode.task.instruction,
            "前往红色箱子旁边",
        )


if __name__ == "__main__":
    unittest.main()
