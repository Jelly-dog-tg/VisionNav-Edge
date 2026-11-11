"""Offline visual protocol tests. No simulator, network or model inference."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('vision_fix_plan', ROOT/'experiments/25_visual_goal_plan.py')
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)
qwen = plan.qwen_bbox


def answer(box=None, status='TARGET_FOUND', color='green', category='sphere', relation='unspecified'):
    box = [100, 200, 300, 400] if box is None and status == 'TARGET_FOUND' else box
    found = status == 'TARGET_FOUND'
    return dict(status=status, category_match=found, color_match=found, spatial_match=True,
                observed_category=category if found else None, observed_color=color if found else None,
                spatial_relation=relation, bbox_valid=box is not None, bbox_2d=box,
                matching_boxes=[box] if box else [], failure_reason='' if found else 'NOT_VISIBLE')


class VisionFixTests(unittest.TestCase):
    def test_conditional_prompts(self):
        plain = qwen.visual_prompt('green sphere', '移动到绿色球体', True)
        self.assertNotIn('shelf', plain)
        self.assertIn('No spatial relation', plain)
        self.assertIn('ON TOP of the shelf', qwen.visual_prompt('green sphere', '找到货架顶部的绿色球体', True))
        floor = qwen.visual_prompt('red box', '找到地面上的红色方块', True)
        self.assertIn('on the floor', floor)
        self.assertNotIn('shelf', floor)

    def test_success_all_relations(self):
        for target, instruction, color, category, relation in (
            ('green sphere', '绿色球体', 'green', 'sphere', 'unspecified'),
            ('green sphere on top of the shelf', '货架顶部绿色球体', 'green', 'sphere', 'shelf_top'),
            ('red box on the floor', '地面红色方块', 'red', 'box', 'floor')):
            with self.subTest(relation=relation):
                data = answer(color=color, category=category, relation=relation)
                self.assertEqual(plan.classify_detection(data['bbox_2d'], json.dumps(data), target, instruction), 'TARGET_FOUND')

    def test_structured_conflicts_rejected(self):
        for update in ({'observed_category': 'cylinder'}, {'observed_color': 'blue'},
                       {'category_match': False}, {'color_match': False},
                       {'spatial_match': False}, {'spatial_relation': 'unknown'},
                       {'bbox_valid': False}):
            with self.subTest(update=update):
                data = answer(); data.update(update)
                with self.assertRaises(ValueError):
                    plan.classify_detection(data['bbox_2d'], json.dumps(data), 'green sphere')

    def test_low_iou_and_exact_boundary(self):
        self.assertEqual(plan.classify_detection([500, 500, 600, 600], json.dumps(answer()), 'green sphere'),
                         'TARGET_RECOGNITION_MISMATCH')
        data=answer([0, 0, 100, 100])
        self.assertEqual(plan.classify_detection([0, 0, 50, 100], json.dumps(data), 'green sphere'), 'TARGET_FOUND')

    def test_null_absence(self):
        data=answer(status='TARGET_NOT_VISIBLE')
        self.assertIsNone(qwen.validate_visual_answer(data, 'green sphere', 'unspecified'))
        self.assertEqual(plan.classify_detection(None, json.dumps(data), 'green sphere'), 'TARGET_NOT_VISIBLE')

    def test_invalid_boxes(self):
        for box in ([None]*4, [True, 2, 3, 4], ['1', 2, 3, 4], [-1, 0, 2, 3],
                    [0, 0, 1001, 3], [0, 0, float('nan'), 3], [0, 0, float('inf'), 3],
                    [2, 0, 1, 3], [0, 0, 0, 3], [0, 1, 2]):
            with self.subTest(box=box), self.assertRaises(ValueError):
                plan.checked_bbox(box)

    def test_json_separate_error(self):
        with self.assertRaises(json.JSONDecodeError):
            plan.classify_detection([1, 2, 3, 4], '{"status":', 'green sphere')

    def test_legacy_missing_flags_fail_closed(self):
        raw=json.dumps({'status':'TARGET_FOUND','matching_boxes':[[1,2,3,4]],'reason':'green sphere'})
        with self.assertRaises(ValueError):
            plan.classify_detection([1,2,3,4], raw, 'green sphere')

    def test_request_saves_prompt_raw_metadata(self):
        response=Mock(); response.read.return_value=json.dumps({'message':{'content':'{'},
            'done':True,'done_reason':'length','eval_count':384}).encode()
        response.__enter__=Mock(return_value=response); response.__exit__=Mock(return_value=False)
        diagnostics=[]
        with patch.object(qwen.urllib.request, 'urlopen', return_value=response) as request:
            box, raw=qwen.predict_bbox('SAME_FRAME', 'green sphere', diagnostics=diagnostics)
        self.assertIsNone(box); self.assertEqual(request.call_count, 2)
        self.assertEqual(len(diagnostics),2)
        for d in diagnostics:
            self.assertEqual(d['raw'],'{')
            self.assertEqual(d['error_type'],'JSONDecodeError')
            self.assertEqual(d['response_metadata']['done_reason'],'length')
            self.assertIn('green sphere',d['prompt'])

    def evaluate_with(self, outputs):
        seen=[]; record={}
        def request(image, prompt, diagnostics):
            seen.append(image)
            raw=next(outputs)
            diagnostics.append({'prompt':prompt,'raw':raw,'response_metadata':{'mock':True}})
            return raw
        with tempfile.TemporaryDirectory() as tmp:
            rgb=Path(tmp)/'rgb.png'; rgb.write_bytes(b'same-rgb')
            obs=SimpleNamespace(camera=SimpleNamespace(width=640,height=480))
            with patch.object(qwen,'visual_request',side_effect=request), patch.object(
                    plan,'validate_candidate_position',return_value=(plan.np.array([1., 2., .2]), False)) as localize:
                result=plan.evaluate_observation(obs,rgb,'depth.npy','green sphere','绿色球体',{},record)
        return result,record,seen,localize

    def test_one_retry_succeeds_same_frame(self):
        result, record, seen, localize=self.evaluate_with(iter(['{',json.dumps(answer()),json.dumps(answer())]))
        self.assertIsNotNone(result); self.assertFalse(result[1]); self.assertEqual(len(record['visual_attempts']),2)
        self.assertEqual(len(set(seen)),1); localize.assert_called_once()

    def test_retry_still_fails_no_rgbd(self):
        result,record,seen,localize=self.evaluate_with(iter(['{','{']))
        self.assertIsNone(result); self.assertEqual(len(seen),2)
        self.assertEqual(record['status'],'CANDIDATE_REJECTED'); localize.assert_not_called()

    def test_verifier_retry_budget_is_total_per_frame(self):
        result,record,seen,localize=self.evaluate_with(iter([json.dumps(answer()),'{',json.dumps(answer()),'{']))
        self.assertIsNone(result); self.assertEqual(len(record['visual_attempts']),2)
        self.assertEqual(len(seen),4); localize.assert_not_called()

    def test_invalid_box_never_localizes(self):
        data=answer(); data['bbox_2d']=[None]*4
        result,record,_,localize=self.evaluate_with(iter([json.dumps(data)]*2))
        self.assertIsNone(result); localize.assert_not_called()

    def test_absence_and_iou_failure_never_localize(self):
        result, record, seen, localize = self.evaluate_with(iter([
            json.dumps(answer(status='TARGET_NOT_VISIBLE'))]))
        self.assertIsNone(result); self.assertEqual(len(seen), 1)
        localize.assert_not_called()
        result, record, seen, localize = self.evaluate_with(iter([
            json.dumps(answer()), json.dumps(answer([500, 500, 600, 600]))]))
        self.assertIsNone(result); self.assertEqual(len(seen), 2)
        self.assertEqual(record['detection_status'], 'TARGET_RECOGNITION_MISMATCH')
        localize.assert_not_called()

    def test_no_free_text_success_override(self):
        data = answer(); data['category_match'] = False
        data['reason'] = 'This is definitely the correct green sphere'
        with self.assertRaises(ValueError):
            plan.classify_detection(data['bbox_2d'], json.dumps(data), 'green sphere')

    def test_historical_seven_frames_fail_closed(self):
        data=json.loads((ROOT/'logs/p4_5_1_evidence/run_1794314441677038900/summary.json').read_text(encoding='utf8'))
        self.assertEqual([d['frame'] for d in data['detections']],list(range(1000,1007)))
        for d in data['detections']:
            with self.subTest(frame=d['frame']):
                raw=d.get('verification_raw')
                with self.assertRaises((ValueError,TypeError)):
                    if d['frame']==1002:
                        plan.checked_bbox(json.loads(d['detector_raw'])['bbox_2d'])
                    else:
                        plan.classify_detection(d['bbox_normalized'],raw,'green sphere','移动到绿色球体')


if __name__ == '__main__':
    unittest.main()
