"""Relocated docking geometry/state tests: no simulator or model calls."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch
import numpy as np
import test_terminal_docking as terminal
import test_p452_vision as vision

plan=vision.plan
G={'front_m':.15,'rear_m':.15,'half_width_m':.11,'length_m':.3,'width_m':.22,
   'radius_m':float(np.hypot(.15,.11)),'top_z_m':.15}


class RelocatedDockingTests(unittest.TestCase):
    def context(self):
        grid=np.zeros((200,200),dtype=np.uint8);grid[:,101:]=1
        # Synthetic visual surface and depth, not a scene object or instance lookup.
        points=np.array([[.45,y,z] for y in np.linspace(-.12,.12,25) for z in (.05,.1,.15,.2)])
        T=np.array([[0,-np.sin(.06),np.cos(.06),.1],[-1,0,0,0],[0,-np.cos(.06),-np.sin(.06),.16],[0,0,0,1.]])
        c=SimpleNamespace(width=640,height=480,fx=415.,fy=415.,cx=319.5,cy=239.5)
        return dict(points=points,ordinary_points=np.empty((0,3)),geometry=G,grid=grid,
                    resolution=.1,arena_size=10.,T=T,camera=c,depth=np.ones((480,640))*.35)

    def test_continuous_green_style_stop_and_footprint(self):
        context=self.context()
        goal,yaw=plan.select_local_docking(np.zeros(2),np.array([.45,0,.18]),context)
        np.testing.assert_allclose(goal,[.18,0.])
        self.assertTrue(plan.visual_footprint_clear(np.array([*goal,yaw]),context))
        cell=plan.planner.world_to_grid(goal,.1,10.,context['grid'])
        self.assertEqual(context['grid'][cell],1) # global inflation unchanged
        self.assertAlmostEqual(np.linalg.norm(goal-np.array([.45,0]))-G['front_m'],.12)

    def test_ordinary_obstacle_and_target_geometry_retained(self):
        context=self.context();context['ordinary_points']=np.array([[.2,.13,.1]])
        with self.assertRaisesRegex(RuntimeError,'NO_SAFE_LOCAL_DOCKING_CORRIDOR'):
            plan.select_local_docking(np.zeros(2),np.array([.45,0,.18]),context)
        context=self.context()
        self.assertFalse(plan.visual_footprint_clear(np.array([.40,0,0.]),context))

    def motion(self,yaw=0.):
        motion=terminal.ns['ActiveSearchMotion'].__new__(terminal.ns['ActiveSearchMotion'])
        pose=np.array([0.,0.,yaw]);velocity=np.zeros(3)
        robot=Mock();robot.get_pose.side_effect=lambda:pose.copy()
        robot.get_velocity.side_effect=lambda:velocity.copy();motion.robot=robot
        motion.log={'actual_distance_m':0.,'collision':False}
        calls=[]
        def step(v,w):
            calls.append((v,w,pose.copy(),motion.log.get('terminal_phase')))
            previous=pose[:2].copy()
            pose[:2]+=v*(10/240)*np.array([np.cos(pose[2]),np.sin(pose[2])]);pose[2]+=w*(10/240)
            velocity[:]=[v*np.cos(pose[2]),v*np.sin(pose[2]),w]
            motion.log['actual_distance_m']+=float(np.linalg.norm(pose[:2]-previous))
        motion.step=step;motion.command_map_safe=lambda *args:True
        return motion,calls

    def test_alignment_then_safe_forward_and_stable_stop(self):
        m,calls=self.motion(np.pi/2)
        helper=Mock();helper.visual_footprint_clear.return_value=True
        helper.select_local_docking.return_value=(np.array([.15,0.]),0.)
        with patch.dict(terminal.ns,visual_plan=helper):
            m.terminal_refine_to_goal([[0.,0.],[0.,0.]],np.array([.42,0,.18]),G,local_context={})
        forward=[c for c in calls if c[0]>0]
        self.assertTrue(forward)
        for v,w,pose,phase in forward:
            delta=np.array([.42,0])-pose[:2]
            error=np.arctan2(np.sin(np.arctan2(delta[1],delta[0])-pose[2]),
                             np.cos(np.arctan2(delta[1],delta[0])-pose[2]))
            self.assertLessEqual(abs(error),.04)
            predicted=pose.copy();predicted[:2]+=v*(10/240)*np.array([np.cos(pose[2]),np.sin(pose[2])])
            self.assertGreaterEqual(terminal.ns['approach_metrics'](predicted,[.42,0,.18],G)['front_gap_projected_m'],.08)
        self.assertTrue(m.log['terminal_alignment_completed']);self.assertTrue(m.log['terminal_approach_completed'])
        self.assertEqual(m.log['terminal_stable_steps'],10)
        self.assertTrue(.08<=m.log['terminal_current_gap_m']<=.15)
        self.assertFalse(m.log['post_docking_target_verified'])

    def test_unaligned_pre_stop_motion_does_not_use_signed_gap(self):
        m,calls=self.motion(np.pi/2)
        helper=Mock();helper.visual_footprint_clear.return_value=True
        helper.select_local_docking.return_value=(np.array([.15,.02]),0.)
        with patch.dict(terminal.ns,visual_plan=helper):
            m.terminal_refine_to_goal([[0.,0.],[0.,.04]],np.array([.42,0,.18]),G,local_context={})
        self.assertTrue(any(v>0 and phase=='PRE_DOCKING_PATH' for v,w,p,phase in calls))
        self.assertTrue(m.log['terminal_stable_stop'])

    def test_terminal_failure_never_retains_final_verification(self):
        m,_=self.motion();m.log.update(post_docking_target_verified=True,final_target_verified=True)
        helper=Mock();helper.visual_footprint_clear.return_value=False
        with patch.dict(terminal.ns,visual_plan=helper),self.assertRaises(RuntimeError):
            m.terminal_refine_to_goal([[0.,0.],[.1,0.]],np.array([.42,0,.18]),G,local_context={})
        self.assertFalse(m.log['post_docking_target_verified']);self.assertFalse(m.log['final_target_verified'])

    def test_new_blind_area_reuses_only_prior_free_evidence(self):
        old=self.context()
        new=self.context()
        new['T']=new['T'].copy();new['T'][0,3]=.30
        pose=np.array([.18,0.,0.])
        self.assertTrue(plan.visual_footprint_clear(pose,old))
        self.assertFalse(plan.visual_footprint_clear(pose,new))
        new['previous_context']=old
        self.assertTrue(plan.visual_footprint_clear(pose,new))
        new['points']=np.vstack([new['points'],[.20,0.,.10]])
        self.assertFalse(plan.visual_footprint_clear(pose,new))

    def test_real_target_contact_is_collision(self):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'experiments/27_follow_visual_path_bc_ppo.py').read_text(encoding='utf8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='has_collision')
        fake=Mock();contact=[0]*10;contact[2]=42;contact[9]=1.
        fake.getContactPoints.side_effect=lambda **kwargs: [] if 'bodyB' in kwargs else [contact]
        ns={'p':fake};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<contact>','exec'),ns)
        self.assertTrue(ns['has_collision'](3,1,99))

    def test_ordinary_inflation_uses_full_global_margin(self):
        context=self.context();context['inflation_m']=.265
        context['ordinary_points']=np.array([[.44,0.,.1]])
        self.assertFalse(plan.visual_footprint_clear(np.array([.18,0.,0.]),context))
        self.assertIn('ORDINARY_INFLATION',context['last_rejection'])

    def test_evaluator_complete_on_early_failure(self):
        source=Path(__file__).resolve().parents[1]/'experiments/27_follow_visual_path_bc_ppo.py'
        tree=ast.parse(source.read_text(encoding='utf8'))
        nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in
               ('evaluate_completed_active_run','_evaluate_completed_active_run')]
        fake_p=Mock();fake_p.isConnected.return_value=True
        fake_p.getNumBodies.return_value=0
        ns={'np':np,'p':fake_p,'FINAL_POSITION_TOLERANCE_M':.12,'STOP_LINEAR_SPEED_TOLERANCE_MPS':.05,
            'STOP_ANGULAR_SPEED_TOLERANCE_RADPS':.05}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'<eval>','exec'),ns)
        robot=Mock();robot.get_pose.return_value=np.zeros(3);robot.get_velocity.return_value=np.zeros(3)
        log={'failure_reason':'NO_SAFE_TERMINAL_PATH','collision':False,'task_success':False,
             'stages':['TARGET_VERIFIED'],'stop_goal_xy':[.1,0.], 'target_attributes':{'class':'sphere','color':'green'}}
        ns['evaluate_completed_active_run'](log,robot)
        self.assertIn('reached_planned_stop',log['post_run_evaluation'])
        self.assertEqual(log['failure_reason'],'NO_SAFE_TERMINAL_PATH')
        self.assertEqual(log['reached_planned_stop'],log['post_run_reached_legacy_stop'])
        self.assertIn('actual_target_position_xyz_m',log['post_run_evaluation'])
        self.assertTrue(log['post_run_evaluation']['target_found'])

    def test_online_planner_has_no_truth_access(self):
        for fn in (plan.build_visual_docking_context,plan.visual_footprint_clear,plan.select_local_docking):
            import inspect
            text=inspect.getsource(fn)
            for banned in ('getAABB','getBody','getVisualShapeData','LAYOUTS','segmentation_path','body_id'):
                self.assertNotIn(banned,text)

if __name__=='__main__':unittest.main()
