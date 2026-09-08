import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('patch',Path(__file__).resolve().parents[1]/'tools/ice-patch-similarity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class PatchTests(unittest.TestCase):
    def test_local_outlier_cannot_hide_in_mean(self):
        water=np.zeros((1,256),dtype=np.float32);x=np.zeros((18,256),dtype=np.float32);x[0]=1
        d=m.nearest(x,water)
        self.assertGreater(d.max(),10);self.assertEqual(int((d>1).sum()),1)
