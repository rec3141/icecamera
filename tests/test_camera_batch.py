import unittest
from dashboard.camera_batch import labels,consensus,PROMPT

def response(text,finish='stop'):
    return {'choices':[{'finish_reason':finish,'message':{'content':text}}]}

class CameraBatchTests(unittest.TestCase):
    def test_reversed_ids_map_to_files(self):
        r=labels(response('{"items":[{"id":2,"label":"water"},{"id":1,"label":"ice"}]}'),[{'file':'b'},{'file':'a'}])
        self.assertEqual(r,{'a':'water','b':'ice'})
    def test_incomplete_and_duplicate_ids_rejected(self):
        for raw in [response('{}','length'),response('{"items":[{"id":1,"label":"water"},{"id":1,"label":"water"}]}')]:
            with self.assertRaises(ValueError):labels(raw,[{'file':'a'},{'file':'b'}])
    def test_consensus_needs_two_water_votes(self):
        r=consensus({'a':'water','b':'water','c':'unclear'},{'a':'water','b':'ice','c':'water'})
        self.assertTrue(r['a']['water']);self.assertFalse(r['b']['water']);self.assertFalse(r['c']['water'])
    def test_partial_batch_prompt(self):self.assertIn('all 3 ids',PROMPT.format(n=3))
