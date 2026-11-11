import unittest
from unittest.mock import Mock
import numpy as np
import test_terminal_docking as terminal
import test_p452_vision as vision

Motion=terminal.ns['ActiveSearchMotion']
terminal.ns.update(visual_plan=vision.plan,FINAL_POSITION_TOLERANCE_M=.12,REQUIRED_SETTLE_STEPS=10,p=Mock())

class HybridNavigationTests(unittest.TestCase):
    def motion(self):
        motion=Motion.__new__(Motion)
        motion.robot=Mock(wheel_base=.22,wheel_radius=.05,max_wheel_vel=10.)
        motion.navigation_map=(np.zeros((200,200),dtype=np.uint8),.1,10.)
        return motion

    def test_progress(self):
        progress,deviation=Motion.path_progress(np.array([[0.,0.],[1.,0.],[1.,1.]]),np.array([1.,.3]))
        self.assertAlmostEqual(progress,1.3); self.assertAlmostEqual(deviation,0.)

    def test_turn_before_forward(self):
        m=self.motion();v,w,_=m.pure_pursuit_control(np.array([0.,0.,np.pi]),np.array([[0.,0.],[1.,0.]]),1)
        self.assertEqual(v,0.);self.assertLessEqual(abs(w),.5)

    def test_forward_wheel_limits(self):
        m=self.motion();v,w,_=m.pure_pursuit_control(np.zeros(3),np.array([[0.,0.],[1.,0.]]),1)
        self.assertGreater(v,0);self.assertLessEqual(v,.1)
        self.assertLessEqual(max(abs((2*v+w*.22)/.1),abs((2*v-w*.22)/.1)),10.)

    def test_no_obstacle_corner_cut(self):
        m=self.motion();grid=m.navigation_map[0];grid[100,101]=1
        with self.assertRaisesRegex(RuntimeError,'RECOVERY_NO_SAFE_TRACKING_SEGMENT'):
            m.pure_pursuit_control(np.array([.05,.05,0.]),np.array([[.05,.05],[.25,.05]]),1)

    def test_map_guard_before_occupied_cell(self):
        m=self.motion();m.navigation_map[0][100,101]=1
        self.assertFalse(m.command_map_safe(np.array([.09,.05,0.]),.1,0.))
        self.assertTrue(m.command_map_safe(np.array([.09,.05,0.]),0.,.5))

    def test_passed_corner_advances_projection(self):
        m=self.motion()
        points=np.array([[1.7967,.4743],[1.75,.35],[1.75,.25],[1.85,.15],[1.95,.15],[2.05,.15]])
        v,w,index=m.pure_pursuit_control(np.array([1.8214,.18,-.44]),points,2)
        self.assertGreaterEqual(index,3)
        self.assertGreater(v,0.)

    def test_stall_latches_recovery_and_stable_stop(self):
        m=self.motion();m.goal_threshold=.12
        pose=np.zeros(3);velocity=np.zeros(3)
        m.robot.get_pose.side_effect=lambda:pose.copy()
        m.robot.get_velocity.side_effect=lambda:velocity.copy()
        m.robot.get_lidar_scan.return_value=np.ones(36)*5
        controller=Mock(waypoint_index=0,waypoints=np.array([[0.,0.],[.4,0.]]),
                        last_diagnostics={'delta_v_mps':.001,'delta_w_radps':.01})
        controller.compute_control.return_value=(0.,0.,False)
        m._prepare_controller=lambda points:controller
        m.log={'collision':False,'actual_distance_m':0.}
        m._record_phase_metrics=lambda stats:m.log.update(stats=stats)
        def step(v,w):
            pose[:2]+=v*(10/240)*np.array([np.cos(pose[2]),np.sin(pose[2])]);pose[2]+=w*(10/240)
            velocity[:]=[v*np.cos(pose[2]),v*np.sin(pose[2]),w]
        m.step=step
        m.follow([[0.,0.],[.4,0.]],0.,[1.,1.,0.],phase='search_navigation')
        stats=m.log['stats']
        self.assertTrue(stats['fallback_triggered']);self.assertTrue(stats['completed'])
        self.assertGreater(stats['fallback_steps'],0)
        self.assertEqual(stats['settled_consecutive_steps'],10)
        self.assertLessEqual(stats['final_goal_error_m'],.12)
        self.assertLessEqual(controller.compute_control.call_count,80)

if __name__=='__main__':unittest.main()
