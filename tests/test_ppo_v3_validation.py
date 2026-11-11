"""V3 validation independence, pairing, readiness gates and real-policy smoke."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load("v3_test_generator", "36_generate_ppo_v3_validation_tasks.py")
evaluator = load("v3_test_evaluator", "37_evaluate_ppo_v3_validation.py")


class TestPPOV3Validation(unittest.TestCase):
    def test_40_tasks_reproducible_unique_and_original_schema(self):
        first = generator.generate()
        second = generator.generate()
        self.assertEqual(first, second)
        self.assertFalse(p.isConnected())
        self.assertEqual(first['task_count'], 40)
        endpoints = [tuple(task[key]) for task in first['tasks'] for key in ('start_cell','goal_cell')]
        self.assertEqual(len(set(endpoints)), 80)
        metadata = json.loads(generator.TRAINING.read_text(encoding='utf-8'))['episodes']
        original = json.loads(generator.BENCHMARK.read_text(encoding='utf-8'))['tasks']
        previous = [tuple(row[key]) for row in metadata+original for key in ('start_xy','goal_xy')]
        for task in first['tasks']:
            self.assertEqual(set(task), set(original[0]))
            for key in ('start_xy','goal_xy'):
                self.assertTrue(all(np.linalg.norm(np.array(task[key])-xy) > .09 for xy in previous))
            self.assertEqual(task['path_cells'][0], task['start_cell'])
            self.assertEqual(task['path_cells'][-1], task['goal_cell'])
            self.assertGreater(task['planned_path_length_m'], 0)

    def test_missing_endpoint_constraints_are_not_silently_relaxed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'metadata.json'
            path.write_text(json.dumps({'episodes':[{'start_xy':[.05,.05],'goal_xy':[1.05,.05]}]}))
            with patch.object(generator,'TRAINING',path), self.assertRaisesRegex(ValueError,'80 unique'):
                generator.generate()
        self.assertFalse(p.isConnected())

    @staticmethod
    def decision_inputs():
        per_seed, comparisons = [], []
        for dataset in ('regression','validation'):
            for scale,rate,std,error in ((0.,.7,0.,.1),(.25,.8,5.,.08),(.5,.85,7.,.09)):
                comparisons.append({'dataset':dataset,'scale':scale,'mean_success_rate':rate,
                    'std_success_rate_pp':std,'mean_collision_rate':0.,'mean_final_error_m':error,
                    'invalid_action_count':0})
                per_seed.extend({'dataset':dataset,'scale':scale,'seed':seed,'success_rate':rate}
                                for seed in evaluator.SEEDS)
        return per_seed,comparisons

    def test_readiness_prefers_quarter_and_rejects_failed_gates(self):
        per_seed,comparisons=self.decision_inputs()
        result=evaluator.readiness(per_seed,comparisons,{'passed':True})
        self.assertTrue(result['recommend_training'])
        self.assertEqual(result['recommended_residual_scale'],.25)
        for row in comparisons:
            if row['dataset']=='validation' and row['scale'] in (.25,.5):
                row['mean_final_error_m']=.5
        result=evaluator.readiness(per_seed,comparisons,{'passed':True})
        self.assertFalse(result['recommend_training'])
        self.assertTrue(all(not r['gates']['final_error_not_worse_by_over_2cm'] for r in result['decisions']))
        per_seed,comparisons=self.decision_inputs()
        self.assertFalse(evaluator.readiness(per_seed,comparisons,{'passed':False})['recommend_training'])

    def test_half_requires_both_mean_and_stability_advantage(self):
        per_seed,comparisons=self.decision_inputs()
        for row in comparisons:
            if row['dataset']=='validation' and row['scale']==.5:
                row['std_success_rate_pp']=2.
        self.assertEqual(evaluator.readiness(per_seed,comparisons,{'passed':True})['recommended_residual_scale'],.5)

    def test_full_paired_matrix_has_no_dataset_id_collisions(self):
        rows=[]
        for seed in evaluator.SEEDS:
            for dataset,count in (('regression',20),('validation',40)):
                for scale in evaluator.SCALES:
                    for task_id in range(count):
                        rows.append({'seed':seed,'dataset':dataset,'scale':scale,'task_id':task_id,
                            'success':bool(scale>0 or task_id%2),'collision':False,'steps':100,
                            'final_goal_error_m':.1,'actual_distance_m':3.,'mean_path_deviation_m':.1,
                            'episode_return':50.})
        paired=evaluator.paired_comparisons(rows)
        self.assertEqual(len(paired),540)
        self.assertEqual(len({(r['seed'],r['dataset'],r['task_id'],r['baseline_scale'],r['candidate_scale'])
                              for r in paired}),540)

    def test_real_v2_ppo_vecnormalize_worker_smoke_and_progress(self):
        task=evaluator.ablation.extract_tasks(json.loads(evaluator.ORIGINAL_TASKS.read_text(encoding='utf-8')))[0]
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(evaluator.PPO,'load',wraps=evaluator.PPO.load) as loads:
                rows=evaluator.run_seed(42,[{'dataset':'regression','task':task}],folder,max_steps=8)
                self.assertEqual(loads.call_count,2)  # BC-only does not load PPO.
            self.assertEqual(len(rows),3)
            self.assertEqual(len(evaluator.read_rows(Path(folder)/'v3_seed_42_progress.csv')),3)
            for row in rows:
                self.assertEqual(row['steps'],8)
                self.assertLessEqual(row['max_abs_delta_v_mps'],row['scale']*.05+1e-7)
                self.assertLessEqual(row['max_abs_delta_w_radps'],row['scale']*.4+1e-7)
                self.assertEqual(row['invalid_action_count'],0)
            # Resume never repeats a completed case or loads its model again.
            with patch.object(evaluator.PPO,'load',side_effect=AssertionError('unexpected reload')):
                self.assertEqual(len(evaluator.run_seed(42,[{'dataset':'regression','task':task}],folder,max_steps=8)),3)
        self.assertFalse(p.isConnected())


if __name__=='__main__':
    unittest.main()
