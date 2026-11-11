"""P5 data correspondence, filtering and DIRECT episode isolation tests."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

from visionnav.learning import collect_expert as collector
from visionnav.learning.expert_controller import ExpertController


class TestCollectExpert(unittest.TestCase):
    def test_observation_frame_units_dimension_and_raw_order(self):
        raw = np.arange(36, dtype=np.float32) / 10
        obs = collector.make_observation([1, 2, np.pi / 2], [1, 4], raw, [-0.1, 0.3, -0.2])
        self.assertEqual(obs.shape, (40,))
        self.assertEqual(obs.dtype, np.float32)
        np.testing.assert_array_equal(obs[:36], raw)
        np.testing.assert_allclose(obs[36:], [2, 0, .3, -.2], atol=1e-7)
        backwards = collector.make_observation([0, 0, 0], [1, 0], raw, [-.2, .1, .3])
        self.assertAlmostEqual(float(backwards[38]), -.2, places=6)

    def test_invalid_measurements_not_recordable(self):
        good = [np.zeros(3), np.ones(2), np.full(36, 5.), np.zeros(3)]
        for index in range(4):
            for invalid in (np.nan, np.inf, -np.inf):
                values = [a.copy() for a in good]
                values[index][0] = invalid
                with self.subTest(index=index, invalid=invalid), self.assertRaises(ValueError):
                    collector.make_observation(*values)
        with self.assertRaises(ValueError):
            collector.make_observation([0, 0, 0], [1, 1], np.ones(35), [0, 0, 0])
        with self.assertRaises(ValueError):
            collector.make_observation([0, 0, 0], [1, 1], -np.ones(36), [0, 0, 0])

    def test_sampler_reproducible_unique_free_and_reachable(self):
        grid = np.zeros((200, 200), dtype=np.uint8)
        streams = [np.random.default_rng(42), np.random.default_rng(42)]
        used = [set(), set()]
        for _ in range(5):
            first = collector.sample_task(grid, streams[0], used[0])
            second = collector.sample_task(grid, streams[1], used[1])
            self.assertEqual(first, second)
            start, goal, path, _, _, _ = first
            self.assertEqual(path[0], start)
            self.assertEqual(path[-1], goal)
            self.assertTrue(all(grid[cell] == 0 for cell in path))
        self.assertEqual(len(used[0]), 10)
        with self.assertRaisesRegex(RuntimeError, "NO_PATH"):
            collector.sample_task(np.ones((200, 200), dtype=np.uint8), streams[0], set())

    @staticmethod
    def fixed_task(grid, rng, used_cells):
        # Short, known-free task for the integration test, not dataset sampling.
        start, goal = (100, 100), (100, 110)
        path, expanded = collector.planner.astar(grid, start, goal, .1)
        return start, goal, path, 0., 1, expanded

    def test_real_direct_correspondence_episode_reset_and_release(self):
        robots, expected_observations, expected_actions, experts = [], [], [], []
        original_robot = collector.DiffDriveRobot

        class RecordingRobot(original_robot):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.commands = []
                robots.append(self)

            def set_velocity(self, v, w):
                self.commands.append([v, w])
                super().set_velocity(v, w)

        class RecordingExpert(ExpertController):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                experts.append(self)

            def compute_control(self, pose, waypoint, raw_lidar):
                expected_observations.append(collector.make_observation(
                    pose, waypoint, raw_lidar, robots[-1].get_velocity()))
                action = super().compute_control(pose, waypoint, raw_lidar)
                expected_actions.append(np.array(action[:2], dtype=np.float32))
                return action

        results = []
        with patch.object(collector, "sample_task", side_effect=self.fixed_task), \
                patch.object(collector, "DiffDriveRobot", RecordingRobot), \
                patch.object(collector, "ExpertController", RecordingExpert):
            for episode in range(2):
                begin = len(expected_observations)
                result = collector.run_episode(episode, 42, 1000, set())
                results.append(result)
                self.assertTrue(result.success, result.failure_reason)
                self.assertFalse(result.collision)
                self.assertFalse(p.isConnected())
                self.assertLess(result.initial_speed_mps, .001)
                self.assertLess(abs(result.initial_yaw_rate_radps), .001)
                np.testing.assert_array_equal(result.observations, expected_observations[begin:])
                np.testing.assert_array_equal(result.actions, expected_actions[begin:])
                # Initial zero command, exactly one command per label, final stop.
                np.testing.assert_array_equal(robots[-1].commands[1:-1], result.actions)
                self.assertEqual(result.navigation_steps, len(result.actions))
        self.assertIsNot(experts[0], experts[1])
        np.testing.assert_array_equal(results[0].observations, results[1].observations)
        np.testing.assert_array_equal(results[0].actions, results[1].actions)

    def test_timeout_collision_no_path_release_connections(self):
        with patch.object(collector, "sample_task", side_effect=self.fixed_task):
            timeout = collector.run_episode(0, 42, 1, set())
        self.assertFalse(timeout.success)
        self.assertEqual(timeout.failure_reason, "TIMEOUT")
        self.assertFalse(p.isConnected())
        # Simulate contact only after the initial settling phase.
        calls = iter([False] * collector.INITIAL_SETTLE_STEPS + [True])
        with patch.object(collector, "sample_task", side_effect=self.fixed_task), \
                patch.object(collector, "has_collision", side_effect=lambda *args: next(calls)):
            collision = collector.run_episode(1, 42, 10, set())
        self.assertFalse(collision.success)
        self.assertTrue(collision.collision)
        self.assertIn("COLLISION", collision.failure_reason)
        self.assertFalse(p.isConnected())
        with patch.object(collector, "sample_task", side_effect=RuntimeError("NO_PATH")):
            no_path = collector.run_episode(2, 42, 1, set())
        self.assertFalse(no_path.success)
        self.assertIn("NO_PATH", no_path.failure_reason)
        self.assertFalse(p.isConnected())

    @staticmethod
    def result(episode_id, success, marker):
        return collector.EpisodeResult(episode_id, 42, success=success,
            failure_reason=None if success else "TIMEOUT",
            final_goal_error_m=.05 if success else 2., final_speed_mps=0.,
            observations=[np.full(40, marker, np.float32) for _ in range(2)],
            actions=[np.full(2, .1, np.float32) for _ in range(2)])

    def test_success_filter_shapes_ids_and_only_four_files(self):
        results = [self.result(0, True, 1), self.result(1, False, np.nan), self.result(2, True, 2)]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            metadata = collector.save_dataset(results, output, 42, 10)
            self.assertEqual({p.name for p in output.iterdir()},
                {"observations.npy", "actions.npy", "episode_ids.npy", "metadata.json"})
            obs = np.load(output / "observations.npy", allow_pickle=False)
            actions = np.load(output / "actions.npy", allow_pickle=False)
            ids = np.load(output / "episode_ids.npy", allow_pickle=False)
            self.assertEqual(obs.shape, (4, 40))
            self.assertEqual(actions.shape, (4, 2))
            self.assertEqual(ids.shape, (4,))
            for array in (obs, actions, ids):
                self.assertEqual(array.dtype, np.float32)
                self.assertTrue(np.all(np.isfinite(array)))
            np.testing.assert_array_equal(ids, [0, 0, 2, 2])
            np.testing.assert_array_equal(obs[:, 0], [1, 1, 2, 2])
            self.assertEqual(metadata["episodes"][1]["collected_sample_count"], 0)
            self.assertEqual(metadata["episodes"][1]["discarded_sample_count"], 2)
            self.assertEqual(metadata["episodes"][2]["sample_range"], [2, 4])

    def test_batch_continues_after_failure_and_empty_dataset_is_valid(self):
        results = [self.result(0, False, 5), self.result(1, True, 2)]
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(collector, "run_episode", side_effect=results) as run:
            metadata = collector.collect(2, 42, 1, Path(folder))
            self.assertEqual(run.call_count, 2)
            self.assertEqual(metadata["failed_episodes"], 1)
            self.assertEqual(metadata["successful_episodes"], 1)
        with tempfile.TemporaryDirectory() as folder:
            collector.save_dataset(results[:1], Path(folder), 42, 1)
            self.assertEqual(np.load(Path(folder) / "observations.npy").shape, (0, 40))
            self.assertEqual(np.load(Path(folder) / "actions.npy").shape, (0, 2))

    def test_inconsistent_or_invalid_success_data_refused(self):
        for kind in ("unpaired", "nonfinite", "collision"):
            result = self.result(0, True, 1)
            if kind == "unpaired":
                result.actions.pop()
            elif kind == "nonfinite":
                result.actions[0][0] = np.inf
            else:
                result.collision = True
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
                collector.save_dataset([result], Path(folder), 42, 1)


if __name__ == "__main__":
    unittest.main()
