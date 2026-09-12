import importlib.util
import json
from pathlib import Path


def load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).parents[1]/'tools'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_combined_policy_excludes_qwen_without_changing_source_or_other_portals(tmp_path):
    explorer=load('ice-region-explorer')
    points=[dict(id=str(i),file=str(i),x=i,y=i) for i in range(3)]
    (tmp_path/'embedding.json').write_text(json.dumps(points))
    rows=[dict(file=str(i),response='{}',finish_reason='stop',**extra)
          for i,extra in enumerate([{},dict(model='Qwen3'),dict(model='gemma-current')])]
    explorer.render(tmp_path,rows,False)
    assert len(json.loads((tmp_path/'annotations-live.js').read_text().split('(',1)[1][:-2]))==3
    (tmp_path/'annotation-policy.json').write_text('{"allow_legacy_qwen":false}')
    explorer.render(tmp_path,rows,False)
    annotations=json.loads((tmp_path/'annotations-live.js').read_text().split('(',1)[1][:-2])
    assert list(annotations)==['2']
    assert annotations['2']['model']=='gemma-current'
    assert len(rows)==3


def test_refit_archives_stale_derived_projections(tmp_path):
    worker=load('ice-full-leg-tsne')
    for name in ('embedding.json','layout-state.json','projections.json','ward-k64.json','feature-pcs.json'):
        (tmp_path/name).write_text('{}')
    worker.archive_layout(tmp_path)
    backup=next((tmp_path/'layout-history').iterdir())
    assert len(list(backup.iterdir()))==5
    assert (tmp_path/'embedding.json').exists()
    assert not (tmp_path/'projections.json').exists()
    assert (backup/'projections.json').read_text()=='{}'
