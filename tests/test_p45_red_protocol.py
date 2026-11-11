"""Red-target protocol regressions: saved model outputs and mocks only."""
import ast
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import test_p452_vision as old

plan, qwen, ROOT = old.plan, old.qwen, old.ROOT


def red(box=None, relation='unspecified'):
    return old.answer(box, color='red', category='box', relation=relation)


class RedProtocolTests(unittest.TestCase):
    def parse(self, data, target='red box', relation='unspecified'):
        return qwen.parse_visual_answer(json.dumps(data), target, relation)

    def test_arrays_and_objects(self):
        for objects in (False, True):
            data=red([800,210,980,370])
            if objects:
                data['matching_boxes']=[{'bbox_2d':data['bbox_2d'],'label':'red box'}]
            result=self.parse(data)
            self.assertEqual(result['bbox_2d'],[800.,210.,980.,370.])
            self.assertEqual(result['matching_boxes'],[result['bbox_2d']])

    def test_category_aliases_and_color(self):
        for category in ('box','red box','cube','block'):
            data=red(); data['observed_category']=category
            self.assertEqual(self.parse(data)['status'],'TARGET_FOUND')
            self.assertEqual(self.parse(data)['observed_category'],'box')
        for category in ('sphere','ball'):
            data=old.answer();data['observed_category']=category
            self.assertEqual(self.parse(data,'green sphere')['status'],'TARGET_FOUND')
        for update in ({'observed_category':'not a box'}, {'observed_color':'red and blue'},
                       {'observed_color':'blue'}, {'observed_category':'cylinder'}):
            data=red();data.update(update)
            with self.assertRaises(ValueError): self.parse(data)

    def test_spatial_aliases_and_conflicts(self):
        for value in ('floor','on the floor','on the ground'):
            self.assertEqual(self.parse(red(relation=value),'red box on the floor','floor')['spatial_relation'],'floor')
        for value in ('shelf_top','on top of the shelf','atop the shelf'):
            self.assertEqual(self.parse(red(relation=value),'red box on top of the shelf','shelf_top')['spatial_relation'],'shelf_top')
        self.assertEqual(self.parse(red(relation='none'))['spatial_relation'],'unspecified')
        for value in ('unknown','ceiling','unspecified','shelf_top'):
            with self.assertRaises(ValueError): self.parse(red(relation=value),'red box on the floor','floor')
        # An observed floor placement is not an invented user requirement.
        self.assertEqual(self.parse(red(relation='on the floor'))['required_spatial_relation'],'unspecified')

    def test_fractional_scale_and_pixel_size(self):
        data=red([.33,.33,.55,.55]); result=self.parse(data)
        self.assertEqual(result['bbox_2d'],[330.,330.,550.,550.])
        self.assertEqual(result['coordinate_scale'],'0_1')
        pixels=qwen.bbox_to_pixels(result['bbox_2d'],640,480)
        self.assertGreater(pixels[2]-pixels[0],1)
        result=self.parse(red([.00001,.00001,.00002,.00002]))
        with self.assertRaisesRegex(ValueError,'TOO_SMALL'):
            qwen.bbox_to_pixels(result['bbox_2d'],640,480)
        # Mixed scales for top-level/matching box normalize to the same box.
        data['matching_boxes']=[[330,330,550,550]]
        self.assertEqual(self.parse(data)['matching_boxes'],[[330.,330.,550.,550.]])

    def test_absent_forms(self):
        for form in ('missing','null','four_nulls'):
            data=old.answer(status='TARGET_NOT_VISIBLE')
            if form=='missing': del data['bbox_2d']
            elif form=='four_nulls': data['bbox_2d']=[None]*4
            self.assertIsNone(self.parse(data)['bbox_2d'])
        data=old.answer(status='TARGET_NOT_VISIBLE');data['bbox_2d']=[None]*4;data['color_match']=True
        with self.assertRaises(ValueError):self.parse(data)
        data=old.answer(status='TARGET_NOT_VISIBLE');data['bbox_valid']=True
        with self.assertRaises(ValueError):self.parse(data)

    def test_found_missing_box_and_box_disagreement(self):
        data=red(); del data['bbox_2d']
        with self.assertRaises(ValueError):self.parse(data)
        data=red(); data['matching_boxes']=[[500,500,600,600]]
        with self.assertRaises(ValueError):self.parse(data)
        data=red(); data['matching_boxes']=[{'bbox_2d':data['bbox_2d'],'label':'blue cylinder'}]
        with self.assertRaises(ValueError):self.parse(data)

    def test_ambiguous_never_found(self):
        data=red();data['status']='TARGET_AMBIGUOUS'
        self.assertIsNone(qwen.validate_visual_answer(data,'red box','unspecified'))
        self.assertEqual(plan.classify_detection(data['bbox_2d'],json.dumps(data),'red box'),'TARGET_AMBIGUOUS')

    def test_bad_json_and_coordinates(self):
        with self.assertRaises(json.JSONDecodeError):qwen.parse_visual_answer('{','red box','unspecified')
        for box in ([None]*4,['.33',.33,.55,.55],[False,0,3,4],[0,0,1001,4],
                    [0,0,float('inf'),4],[2,2,1,1]):
            with self.subTest(box=box), self.assertRaises(ValueError):self.parse(red(box))

    def test_understand_instruction_strips_invented_placement(self):
        for instruction, expected in (
            ('移动到红色方块','red box'),('地上的红色方块','red box on the floor'),
            ('货架上的红色方块','red box on top of the shelf')):
            response=io.BytesIO(json.dumps({'message':{'content':json.dumps({'target':'red box on the floor'})}}).encode())
            with patch.object(plan.urllib.request,'urlopen',return_value=response):
                self.assertEqual(plan.understand_instruction(instruction),expected)

    def test_resolve_defends_against_invented_placement(self):
        tree=ast.parse((ROOT/'experiments/27_follow_visual_path_bc_ppo.py').read_text(encoding='utf8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_active_target')
        import re
        namespace={'visual_plan':plan,'re':re,'ACTIVE_TARGETS':{
            'red box':{'color':'red','class':'box'},'blue cylinder':{'color':'blue','class':'cylinder'},
            'green sphere':{'color':'green','class':'sphere'}}}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<resolve>','exec'),namespace)
        with patch.object(plan,'understand_instruction',return_value='red box on the floor'):
            self.assertEqual(namespace['resolve_active_target']('移动到红色方块'),('red box','red box'))
        with patch.object(plan,'understand_instruction',return_value='blue cylinder'):
            with self.assertRaises(ValueError):namespace['resolve_active_target']('移动到红色方块')

    def test_diagnostics_independent_verifier_and_iou(self):
        diagnostics={}
        self.assertEqual(plan.classify_detection([800,210,980,370],json.dumps(red([800,210,980,370])),
            'red box',diagnostics=diagnostics,image_size=(640,480)),'TARGET_FOUND')
        self.assertEqual(diagnostics['bbox_iou'],1.)
        self.assertEqual(diagnostics['verifier_bbox'],[800.,210.,980.,370.])
        diagnostics={}
        self.assertEqual(plan.classify_detection([0,0,100,100],json.dumps(red([800,210,980,370])),
            'red box',diagnostics=diagnostics,image_size=(640,480)),'TARGET_RECOGNITION_MISMATCH')
        self.assertIn('BBOX_IOU_BELOW_THRESHOLD',diagnostics['rejection_detail'])

    def test_response_schema_does_not_force_recognition(self):
        schema=qwen.visual_response_schema()
        self.assertIn('bbox_2d',schema['required'])
        self.assertIn('failure_reason',schema['required'])
        self.assertIn('TARGET_NOT_VISIBLE',schema['properties']['status']['enum'])
        self.assertIn('TARGET_AMBIGUOUS',schema['properties']['status']['enum'])
        self.assertEqual(schema['properties']['category_match'],{'type':'boolean'})
        self.assertNotIn('rectangle',schema['properties']['observed_category']['enum'])

    def test_hsv_failure_keeps_qwen_rgbd_status(self):
        tree=ast.parse((ROOT/'experiments/27_follow_visual_path_bc_ppo.py').read_text(encoding='utf8'))
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_candidate_color')
        namespace={'target_color_fraction':lambda *args:.079}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<color>','exec'),namespace)
        record={'detector_status':'TARGET_FOUND','semantic_verification_status':'TARGET_FOUND',
                'rgbd_verification_status':'TARGET_VALIDATED','candidate_accepted':False}
        with self.assertRaisesRegex(ValueError,'HSV_COLOR_CHECK_FAILED'):
            namespace['verify_candidate_color'](None,[1,2,3,4],'red',record)
        self.assertEqual(record['hsv_verification_status'],'FAILED')
        self.assertEqual(record['semantic_verification_status'],'TARGET_FOUND')
        self.assertEqual(record['rgbd_verification_status'],'TARGET_VALIDATED')
        self.assertFalse(record['candidate_accepted'])
        namespace['target_color_fraction']=lambda *args:.08
        namespace['verify_candidate_color'](None,[1,2,3,4],'red',record)
        self.assertEqual(record['hsv_verification_status'],'PASSED')

    def test_historical_real_detector_outputs(self):
        path=ROOT/'logs/p4_5_1_evidence/run_1794316696517255500/summary.json'
        data=json.loads(path.read_text(encoding='utf8'))
        found,absent=0,0
        for d in data['detections']:
            with self.subTest(frame=d['frame']):
                result=qwen.parse_visual_answer(d['detector_raw'],'red box','unspecified')
                if result['status']=='TARGET_FOUND':
                    found+=1
                    qwen.bbox_to_pixels(result['bbox_2d'],640,480)
                else:
                    absent+=1; self.assertIsNone(result['bbox_2d'])
        self.assertEqual((found,absent),(4,9))
        totals = {'TARGET_FOUND':0, 'TARGET_NOT_VISIBLE':0}
        for d in data['detections']:
            for attempt in d['visual_attempts']:
                for response in attempt['detector']:
                    parsed=qwen.parse_visual_answer(response['raw'],'red box','unspecified')
                    totals[parsed['status']]+=1
        self.assertEqual(totals,{'TARGET_FOUND':8,'TARGET_NOT_VISIBLE':18})
        # Detector-only historical candidates are not verified navigation targets.
        self.assertTrue(all(d.get('verification_raw') is None for d in data['detections']))


if __name__=='__main__':unittest.main()
