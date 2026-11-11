"""P5 index, validation, sensor convention and original-PID compatibility tests."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

from scripts.diff_drive_robot import DiffDriveRobot
from visionnav.learning import ExpertController


class TestExpertController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Independent reference import, not the adapter's loader or a PID replica.
        path = Path(__file__).resolve().parents[1] / "scripts/06_navigate_environment.py"
        spec = importlib.util.spec_from_file_location("p5_reference_pid", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.reference_pid = module.PIDNavigationController

    def test_lidar_permutation_preserves_values_and_input(self):
        raw = np.arange(36, dtype=float) + 0.125
        original = raw.copy()
        adapted = ExpertController.adapt_lidar(raw)
        np.testing.assert_array_equal(adapted, np.concatenate((raw[18:], raw[:18])))
        np.testing.assert_array_equal(np.sort(adapted), np.sort(raw))
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(adapted.shape, (36,))
        self.assertFalse(np.shares_memory(raw, adapted))

    def test_four_directions_match_pid_angles(self):
        for direction, raw_index, pid_index, angle in (
            ("front", 0, 18, 0.0), ("rear", 18, 0, -np.pi),
            ("left", 9, 27, np.pi / 2), ("right", 27, 9, -np.pi / 2),
        ):
            with self.subTest(direction=direction):
                raw = np.full(36, 5.0)
                raw[raw_index] = 0.2
                adapted = ExpertController.adapt_lidar(raw)
                self.assertEqual(int(np.argmin(adapted)), pid_index)
                self.assertAlmostEqual(2 * np.pi * pid_index / 36 - np.pi, angle)

    def test_original_sensor_ray_geometry_and_output_order(self):
        # Exercise the real sensor method; mock only physics ray returns.
        # No GUI, URDF or sensor reimplementation is required.
        robot = DiffDriveRobot.__new__(DiffDriveRobot)
        robot.lidar_rays, robot.lidar_range, robot.lidar_height = 36, 5.0, 0.11
        fractions = np.arange(36) / 72.0
        for yaw in (0.0, 0.7):
            with self.subTest(yaw=yaw):
                robot.get_pose = lambda: np.array([1.0, -2.0, yaw])
                with patch.object(p, "rayTestBatch", return_value=[
                    (-1, -1, float(f), (), ()) for f in fractions
                ]) as rays:
                    raw = robot.get_lidar_scan()
                starts, ends = map(np.asarray, rays.call_args.args)
                for index, relative_angle in ((0, 0.), (9, np.pi / 2), (18, np.pi), (27, -np.pi / 2)):
                    vector = ends[index, :2] - starts[index, :2]
                    np.testing.assert_allclose(vector, 5.0 * np.array([
                        np.cos(yaw + relative_angle), np.sin(yaw + relative_angle)
                    ]), atol=1e-12)
                np.testing.assert_array_equal(raw, fractions * 5.0)
                np.testing.assert_array_equal(ExpertController.adapt_lidar(raw),
                                              np.concatenate((raw[18:], raw[:18])))

    def test_front_obstacle_triggers_avoidance_rear_does_not(self):
        actions = {}
        for name, index in (("front", 0), ("rear", 18)):
            raw = np.full(36, 5.0)
            raw[index] = 0.3
            actions[name] = ExpertController().compute_control([0, 0, 0], [2, 0], raw)
        self.assertGreater(abs(actions["front"][1]), 0.0)
        self.assertEqual(actions["rear"][1], 0.0)
        # The legacy global minimum still slows for rear obstacles; preserve it.
        self.assertLess(actions["rear"][0], 0.5)

    def test_matches_original_pid_over_stateful_sequence(self):
        expert, reference = ExpertController(goal_threshold=0.08), self.reference_pid(goal_threshold=0.08)
        for pose, goal, near_index in (
            ([0, 0, 0], [3, 0], None), ([0.2, 0, 0.2], [1, 2], 0),
            ([0.3, 0.1, -0.5], [-2, 1], 1), ([0.3, 0.1, 0.5], [2, -1], 35),
            ([1, 1, 0], [1.02, 1.01], None),
        ):
            raw = np.full(36, 5.0)
            if near_index is not None:
                raw[near_index] = 0.2
            expected = reference.compute_control(np.array(pose), np.array(goal),
                                                 np.concatenate((raw[18:], raw[:18])))
            actual = expert.compute_control(pose, goal, raw)
            self.assertEqual(actual, tuple(expected))
            self.assertIsInstance(actual[0], float)
            self.assertIsInstance(actual[1], float)
            self.assertIsInstance(actual[2], bool)
            self.assertLessEqual(abs(actual[0]), reference.max_linear_vel)
            self.assertLessEqual(abs(actual[1]), reference.max_angular_vel)

    def test_original_velocity_caps_and_goal_stop(self):
        scan = np.full(36, 5.0)
        self.assertEqual(ExpertController().compute_control([0, 0, 0], [100, 0], scan), (0.5, 0., False))
        v, w, reached = ExpertController().compute_control([0, 0, 0], [-100, 0], scan)
        self.assertEqual(w, 2.0)
        self.assertLessEqual(v, 0.5)
        self.assertFalse(reached)
        self.assertEqual(ExpertController().compute_control([0, 0, 0], [0.1, 0.1], scan), (0., 0., True))

    def test_invalid_inputs_rejected_before_pid_state_changes(self):
        expert = ExpertController()
        expert.compute_control([0, 0, 0], [0, 2], np.full(36, 5.0))
        previous = expert._pid.last_heading_error
        valid = [np.zeros(3), np.ones(2), np.full(36, 5.0)]
        for field, size in enumerate((3, 2, 36)):
            bad_values = [np.zeros(size - 1), np.zeros((1, size)), ["not-a-number"] * size,
                          np.full(size, complex(1, 2))]
            for illegal in (np.nan, np.inf, -np.inf):
                vector = np.ones(size)
                vector[0] = illegal
                bad_values.append(vector)
            for bad in bad_values:
                with self.subTest(field=field, bad=repr(bad)):
                    values = valid.copy()
                    values[field] = bad
                    with self.assertRaises(ValueError):
                        expert.compute_control(*values)
                    self.assertEqual(expert._pid.last_heading_error, previous)
        negative = np.full(36, 5.0)
        negative[0] = -0.1
        with self.assertRaises(ValueError):
            expert.compute_control(*valid[:2], negative)

    def test_invalid_goal_threshold(self):
        for threshold in (0, -0.1, np.nan, np.inf, -np.inf, "invalid", None):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                ExpertController(goal_threshold=threshold)

    def test_reset_and_independent_instances(self):
        scan = np.full(36, 5.0)
        expert, fresh = ExpertController(), ExpertController()
        expert.compute_control([0, 0, 0], [0, 2], scan)
        self.assertNotEqual(expert._pid.last_heading_error, fresh._pid.last_heading_error)
        expert.reset()
        self.assertEqual(expert.compute_control([0, 0, 0], [2, 0], scan),
                         fresh.compute_control([0, 0, 0], [2, 0], scan))

    def test_legacy_side_selection_is_preserved(self):
        # This documents an inherited limitation, NOT a new steering rule.
        # Its left_space mean covers PID indices 0..17 (physical right half).
        for index, expected_sign in ((1, 1), (35, -1)):
            raw = np.full(36, 5.0)
            raw[0], raw[index] = 0.3, 0.2
            action = ExpertController().compute_control([0, 0, 0], [2, 0], raw)
            self.assertEqual(int(np.sign(action[1])), expected_sign)


if __name__ == "__main__":
    unittest.main()
