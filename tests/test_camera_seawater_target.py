import importlib.util
import json
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('seawater',Path(__file__).resolve().parents[1]/'tools/train-camera-seawater.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class TargetTests(unittest.TestCase):
    def row(self,water=100,ice=0,blur=0):
        return dict(finish_reason='stop',response=json.dumps(dict(surface_percentages={'calm water':water,'icy bits':ice,'unknown':0},artifact_percentages=dict(blurry=blur,fog=0,reflection=0,night=0),visibility='clear',confidence='high')))
    def test_clear_water(self):self.assertEqual(module.target(self.row())[0],1)
    def test_tiny_ice_is_not_water(self):self.assertEqual(module.target(self.row(99,1))[0],0)
    def test_blur_routes_to_review(self):self.assertEqual(module.target(self.row(blur=30))[0],0)
    def test_bad_sum_rejected(self):
        with self.assertRaises(ValueError):module.target(self.row(99))
