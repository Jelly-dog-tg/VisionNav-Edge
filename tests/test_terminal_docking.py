"""Mocked terminal docking tests: no GUI, VLM, weights, or physics run."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'experiments/27_follow_visual_path_bc_ppo.py'
tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
names = {'approach_metrics', 'safe_terminal_metrics', 'terminal_path', 'ActiveSearchMotion'}
nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
ns = {'np': np, 'visual_plan': Mock(), 'TERMINAL_PARAMETERS': {
    'max_linear_mps': .04, 'max_angular_radps': .25, 'max_distance_m': .20,
    'max_steps': 600, 'position_tolerance_m': .010, 'stable_steps': 10,
    'no_progress_steps': 120}, 'STOP_HEADING_TOLERANCE_RAD': .04,
    'STOP_LINEAR_SPEED_TOLERANCE_MPS': .05, 'STOP_ANGULAR_SPEED_TOLERANCE_RADPS': .05}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)


class TerminalDockingTests(unittest.TestCase):
    def setUp(self):
        previous=ns['visual_plan']
        ns['visual_plan']=Mock()
        ns['visual_plan'].select_local_docking.return_value=(np.zeros(2),0.)
        ns['visual_plan'].visual_footprint_clear.return_value=True
        self.addCleanup(lambda:ns.update(visual_plan=previous))

    def metrics(self, gap, lateral=0., yaw=0.):
        return ns['approach_metrics'](np.array([0., 0., yaw]),
            np.array([.2 + gap, lateral, .1]), {'front_m': .2, 'half_width_m': .1})

    def test_geometry(self):
        for gap, passed in ((.133, True), (.205, False), (.079, False)):
            with self.subTest(gap=gap):
                self.assertEqual(ns['safe_terminal_metrics'](self.metrics(gap)), passed)
        self.assertFalse(ns['safe_terminal_metrics'](self.metrics(.133, lateral=.14)))
        self.assertFalse(ns['safe_terminal_metrics'](self.metrics(.133, yaw=.3)))

    def test_no_path(self):
        ns['visual_plan'].planner.world_to_grid.return_value = (0, 0)
        ns['visual_plan'].planner.astar.return_value = ([], 0)
        with self.assertRaisesRegex(RuntimeError, 'NO_SAFE_TERMINAL_PATH'):
            ns['terminal_path'](np.zeros((2, 2)), [0., 0.], [.1, 0.], .1, 10.)

    def motion(self):
        motion = ns['ActiveSearchMotion'].__new__(ns['ActiveSearchMotion'])
        motion.robot = Mock()
        motion.robot.get_pose.return_value = np.array([0., 0., 0.])
        motion.robot.get_velocity.return_value = np.zeros(3)
        motion.log = {'actual_distance_m': 0., 'collision': False}
        motion.step = Mock()
        motion.command_map_safe = lambda *args: True
        return motion

    def test_collision_stops_immediately(self):
        motion = self.motion()
        def collision(*args):
            motion.log['collision'] = True
            raise RuntimeError('NAVIGATION_COLLISION')
        motion.step.side_effect = collision
        with self.assertRaisesRegex(RuntimeError, 'TERMINAL_REFINEMENT_COLLISION'):
            motion.terminal_refine_to_goal([[0., 0.], [.1, 0.]], np.array([.433, 0., .1]),
                                         {'front_m': .2, 'half_width_m': .1})
        self.assertEqual(motion.step.call_count, 1)
        motion.robot.set_velocity.assert_called_with(0., 0.)
        self.assertTrue(motion.log['terminal_refinement_collision'])

    def test_stable_ten_checks_and_no_policy_calls(self):
        motion = self.motion()
        motion.terminal_refine_to_goal([[0., 0.], [0., 0.]], np.array([.32, 0., .1]),
                                     {'front_m': .2, 'half_width_m': .1},local_context={})
        self.assertEqual(motion.log['terminal_stable_steps'], 10)
        self.assertEqual(motion.step.call_count, 9)
        self.assertTrue(motion.log['terminal_stable_stop'])

    def test_minimum_gap_prevents_motion(self):
        motion = self.motion()
        with self.assertRaisesRegex(RuntimeError, 'FINAL_GAP_TOO_SMALL'):
            motion.terminal_refine_to_goal([[0., 0.], [0., 0.]], np.array([.279, 0., .1]),
                                         {'front_m': .2, 'half_width_m': .1},local_context={})
        motion.step.assert_not_called()

    def test_no_online_ground_truth(self):
        online = '\n'.join(ast.unparse(n) for n in nodes)
        for forbidden in ('LAYOUTS', 'getVisualShapeData', 'resetBasePositionAndOrientation',
                          'getBasePositionAndOrientation'):
            self.assertNotIn(forbidden, online)


if __name__ == '__main__':
    unittest.main()
