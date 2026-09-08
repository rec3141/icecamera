import importlib.util
from pathlib import Path
import unittest
import json
import numpy as np
from PIL import Image
spec=importlib.util.spec_from_file_location('native',Path(__file__).resolve().parents[1]/'tools/ice-native-triage.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class NativeTests(unittest.TestCase):
    def test_small_bright_fragment_detected(self):
        a=np.full((600,1200),60,dtype=np.uint8);a[40:45,60:65]=240
        self.assertGreater(len(m.objects(Image.fromarray(a))['bright_objects']),0)
    def test_dark_is_obscured(self):self.assertTrue(m.objects(Image.new('L',(1200,600),5))['obscured'])
    def test_report_serializable(self):json.dumps(m.objects(Image.new('L',(1200,600),80)))
    def test_large_time_gap_breaks_support(self):
        r=[{'file':f'2026_LEG_03/20260908/{t}/Camera360_20260908{t}_cam_3.jpg'} for t in ['000000','000200','010000']]
        self.assertEqual(m.neighbors(r,1),[])
