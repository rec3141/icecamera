import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('similarity',Path(__file__).resolve().parents[1]/'tools/ice-image-similarity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class SimilarityTests(unittest.TestCase):
    def test_identical_matches_and_colour_change_penalized(self):
        rgb=np.ones((24,48,3),dtype=np.float32)*.2;gray=np.zeros((24,48),dtype=np.float32)
        d=m.distances(rgb,gray,np.stack([rgb,rgb+.2]),np.stack([gray,gray]))
        self.assertAlmostEqual(d[0],0);self.assertGreater(d[1],.19)
