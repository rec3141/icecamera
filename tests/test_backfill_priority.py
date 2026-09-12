import importlib.util
import json
from pathlib import Path


def test_human_cluster_ranking_and_capture_boundary(tmp_path,monkeypatch):
    monkeypatch.setenv('ICE_GEMMA_OUT',str(tmp_path))
    monkeypatch.setenv('ICE_GEMMA_CAPTURE_BEFORE','20260911062817')
    spec=importlib.util.spec_from_file_location('gemma_worker',Path(__file__).parents[1]/'tools'/'ice-gemma-full.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    policy=dict(capture_from='20260910000000',allowed_files=['a','b','c','d'],
                ice_fraction_by_file={'a':1,'b':.7,'c':1,'d':0},cluster_by_file={'a':1,'b':2,'c':3,'d':4},
                cluster_rank={'1':0,'3':1,'2':2,'4':3})
    (tmp_path/'backfill-paused.json').write_text(json.dumps(policy))
    queue=[dict(file=file,capture_time='20260910000001') for file in ['b','c','d','a','unlabelled']]
    selected=sorted((r for r in queue if worker.eligible_for_processing(r)),key=worker.backfill_priority,reverse=True)
    assert [r['file'] for r in selected]==['a','c','b','d']
    assert not worker.eligible_for_processing(dict(file='a',capture_time='20260912000000'))
