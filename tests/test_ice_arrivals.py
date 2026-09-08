import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('arrivals',Path(__file__).resolve().parents[1]/'tools/ice-camera-arrivals.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class ArrivalsTests(unittest.TestCase):
    def test_newest_first_unique_and_old_preserved(self):
        old=[dict(file='old',capture_time='20260901'),dict(file='mid',capture_time='20260902')]
        new=[dict(file='new',capture_time='20260903'),dict(file='mid',capture_time='20260902')]
        self.assertEqual([r['file'] for r in module.merged_queue(old,new)],['new','mid','old'])
