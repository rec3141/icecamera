import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from dashboard.camera_seawater import Filter,image_guards,audit_sample,audit_contradiction

class SeawaterTests(unittest.TestCase):
    def test_flat_dark_and_bright_objects_review(self):
        self.assertTrue(image_guards(Image.new('RGB',(1200,600),'black')))
        self.assertTrue(image_guards(Image.new('RGB',(1200,600),(80,80,80))))
        a=np.random.default_rng(42).integers(40,90,(600,1200),dtype=np.uint8);a[10:20,10:20]=255
        self.assertTrue(image_guards(Image.fromarray(a)))
    def test_sample_stable(self):
        self.assertEqual(audit_sample('photo','model'),audit_sample('photo','model'))
        self.assertTrue(70<sum(audit_sample(str(i),'model') for i in range(1000))<130)
    def test_missing_config_fails_open(self):
        self.assertEqual(Filter('/nonexistent/seawater-config').decide('x',None)['route'],'gemma')
    def test_audit_any_ice_disables(self):
        row=dict(surface_percentages={'icy bits':1,'unknown':0},artifact_percentages={'blurry':0})
        self.assertTrue(audit_contradiction(json.dumps(row)))
        row['surface_percentages']['icy bits']=0
        self.assertFalse(audit_contradiction(json.dumps(row)))
    def test_threshold_and_disabled_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'model').write_text('{}');cfg=p/'config';flag=p/'disabled'
            cfg.write_text(json.dumps(dict(enabled=True,model=str(p/'model'),disabled_flag=str(flag),threshold=.995)))
            f=Filter(cfg)
            with patch('dashboard.camera_seawater.image_guards',return_value=[]),patch('dashboard.camera_seawater.features',return_value=[]),patch('dashboard.camera_seawater.infer',return_value={'scores':{'clear_seawater_score':.8},'outside_training_features':False}):
                self.assertEqual(f.decide('x',None)['route'],'gemma')
            flag.write_text('{}');self.assertEqual(f.decide('x',None)['reason'],'filter disabled')
    def test_clear_water_budget_ambiguity_not_ice(self):
        row=dict(surface_percentages={'icy bits':0,'unknown':15},artifact_percentages={'blurry':0},visibility='clear',confidence='high')
        self.assertFalse(audit_contradiction(json.dumps(row)))
        row['visibility']='uncertain';self.assertTrue(audit_contradiction(json.dumps(row)))
