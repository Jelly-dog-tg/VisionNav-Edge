"""P8.8 interface/geometry tests; no GUI or live VLM calls."""
import importlib.util,json,tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('act_runner_test',ROOT/'experiments/28_follow_visual_path_act.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
plan=runner.visual_plan
from visionnav.learning.act_semantic_controller import ACTSemanticController

class ACTSemanticFixTests(unittest.TestCase):
    def test_real_model_path_reset_clears_temporal_state_without_reload(self):
        robot=Mock();robot.get_velocity.return_value=np.zeros(3)
        c=ACTSemanticController(robot,[[0,0],[1,0]],[1,0])
        try:
            model=c.act.model
            for _ in range(9):c.compute_control(np.zeros(3),[1,0],np.ones(36)*3)
            self.assertEqual(len(c.act.history),8);self.assertIsNotNone(c.act.pending)
            with patch('torch.load',side_effect=AssertionError('must not reload')):
                c.set_path([[0,0],[0,1]],[0,1],reset_reason='second_search_path')
                self.assertIs(c.act.model,model)
                self.assertEqual(len(c.act.history),0);self.assertIsNone(c.act.pending);self.assertEqual(c.act.t,0)
                self.assertEqual(c.history_reset_count,1)
                self.assertEqual(c.history_reset_events[-1]['reason'],'second_search_path')
                c.reset(reason='third_path');self.assertEqual(c.history_reset_count,2)
        finally:c.close()

    def historical(self):
        s=json.loads((ROOT/'logs/p8_8_act_semantic_evidence/run_1794378560971181700/summary.json').read_text(encoding='utf8'))
        return next(d for d in s['detections'] if d['frame']==1006)

    def test_historical_full_frame_uses_verified_local_box(self):
        d=self.historical();record={}
        box=plan.match_candidate_bboxes(d['detector_raw'],d['verification_raw'],'blue cylinder','寻找蓝色圆柱体并靠近它',(640,480),record)
        self.assertEqual(box,[789.,388.,972.,488.]);self.assertTrue(record['fallback_used'])
        self.assertEqual(record['bbox_source'],'verifier_fallback');self.assertLess(record['bbox_iou'],.5)
        self.assertFalse(record['iou_applicable']);self.assertEqual(record['detector_coordinate_scale'],'0_1')

    def test_explicit_scale_and_full_frame_rejection(self):
        q=plan.qwen_bbox
        self.assertEqual(q.normalize_bbox([0,0,1,1],scale='0_1')[0],[0.,0.,1000.,1000.])
        self.assertEqual(q.normalize_bbox([0,0,1,1],scale='0_1000')[0],[0.,0.,1.,1.])
        with self.assertRaises(ValueError):q.check_bbox_reasonableness([0,0,1000,1000],640,480)
        with self.assertRaises(ValueError):q.check_bbox_reasonableness([0,0,1,1],640,480)

    def test_two_valid_low_iou_boxes_are_not_fallback(self):
        d=self.historical();raw=json.loads(d['detector_raw']);raw['bbox_2d']=[100,100,200,200];raw['matching_boxes']=[raw['bbox_2d']]
        with self.assertRaisesRegex(ValueError,'BBOX_IOU_BELOW_THRESHOLD'):
            plan.match_candidate_bboxes(json.dumps(raw),d['verification_raw'],'blue cylinder','蓝色圆柱体',(640,480),{})

    def test_fallback_verifier_color_conflict_rejected(self):
        d=self.historical();raw=json.loads(d['verification_raw']);raw['observed_color']='red'
        with self.assertRaises(ValueError):
            plan.match_candidate_bboxes(d['detector_raw'],json.dumps(raw),'blue cylinder','蓝色圆柱体',(640,480),{})

    def test_fallback_still_runs_rgbd(self):
        d=self.historical();record={};obs=SimpleNamespace(camera=SimpleNamespace(width=640,height=480))
        with tempfile.TemporaryDirectory() as tmp:
            rgb=Path(tmp)/'rgb.png';rgb.write_bytes(b'frame')
            with patch.object(plan,'predict_bbox',return_value=(None,d['detector_raw'])),patch.object(plan,'verify_target',return_value=d['verification_raw']),patch.object(plan,'validate_candidate_position',side_effect=ValueError('RGBD_INVALID')) as localize:
                result=plan.evaluate_observation(obs,rgb,'depth.npy','blue cylinder','蓝色圆柱体',{},record,allow_verifier_bbox_fallback=True)
        self.assertIsNone(result);localize.assert_called_once()
        self.assertEqual(record['rgbd_verification_status'],'CANDIDATE_REJECTED')

    def test_fallback_still_requires_hsv_after_valid_rgbd(self):
        d=self.historical();record={};obs=SimpleNamespace(camera=SimpleNamespace(width=640,height=480))
        with tempfile.TemporaryDirectory() as tmp:
            rgb=Path(tmp)/'rgb.png';rgb.write_bytes(b'frame')
            with patch.object(plan,'predict_bbox',return_value=(None,d['detector_raw'])),patch.object(plan,'verify_target',return_value=d['verification_raw']),patch.object(plan,'validate_candidate_position',return_value=(np.array([2.,.7,.18]),False)):
                result=plan.evaluate_observation(obs,rgb,'depth.npy','blue cylinder','蓝色圆柱体',{},record,allow_verifier_bbox_fallback=True)
        self.assertIsNotNone(result);self.assertEqual(record['rgbd_verification_status'],'TARGET_VALIDATED')
        with patch.object(runner._recovery_module,'target_color_fraction',return_value=.079),self.assertRaisesRegex(ValueError,'HSV_COLOR_CHECK_FAILED'):
            runner._recovery_module.verify_candidate_color(None,record['bbox_pixels'],'blue',record)
        self.assertEqual(record['hsv_verification_status'],'FAILED');self.assertFalse(record['candidate_accepted'])

    def test_recovery_latches_and_stabilizes(self):
        m=runner.ActiveSearchMotion.__new__(runner.ActiveSearchMotion)
        pose=np.zeros(3);velocity=np.zeros(3)
        robot=Mock(wheel_base=.22,wheel_radius=.05,max_wheel_vel=10.)
        robot.get_pose.side_effect=lambda:pose.copy();robot.get_velocity.side_effect=lambda:velocity.copy()
        robot.get_lidar_scan.return_value=np.ones(36)*3
        m.robot=robot;m.goal_threshold=.05;m.control_steps=0
        m.navigation_map=(np.zeros((200,200),dtype=np.uint8),.1,10.)
        c=Mock(waypoint_index=0,waypoints=np.array([[0.,0.],[.4,0.]]),history_reset_count=2,
               act=SimpleNamespace(history=[],pending=None),last_diagnostics={})
        c.compute_control.return_value=(0.,0.,False)
        m._prepare_controller=lambda points,phase:c
        m.log={'controller':{'phase_runs':[]},'actual_distance_m':0.,'collision':False}
        def step(v,w):
            pose[:2]+=v*(10/240)*np.array([np.cos(pose[2]),np.sin(pose[2])]);pose[2]+=w*(10/240)
            velocity[:]=[v*np.cos(pose[2]),v*np.sin(pose[2]),w];m.control_steps+=1
        m.step=step
        with patch.object(runner.p,'addUserDebugLine'):
            m.follow([[0,0],[.4,0]],0,[1,1,0],phase='search_navigation')
        r=m.log['controller']['phase_runs'][0]
        self.assertTrue(r['fallback_completion']);self.assertEqual(r['settled_consecutive_steps'],10)
        self.assertLessEqual(r['final_goal_error_m'],.05)
        self.assertIn('recovery_started_at_utc',r);self.assertIn('recovery_ended_at_utc',r)
        self.assertLessEqual(c.compute_control.call_count,80)

if __name__=='__main__':unittest.main()
