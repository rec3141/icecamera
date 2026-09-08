import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('consensus',Path(__file__).resolve().parents[1]/'tools/ice-batch-consensus.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class BatchTests(unittest.TestCase):
    def test_disagreement_not_water(self):
        r=m.compare([{'file':'a'}],{'a':'water'},{'a':'ice'})
        self.assertEqual(r[0]['route'],'full_review')
    def test_missing_ids_rejected(self):
        with self.assertRaises(ValueError):m.decisions([{'file':'a'}],{'items':[]})
