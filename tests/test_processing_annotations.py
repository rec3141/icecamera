import importlib.util
import json
import sqlite3
from pathlib import Path


def explorer():
    spec=importlib.util.spec_from_file_location('explorer',Path(__file__).parents[1]/'tools/ice-region-explorer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def response(file, **extra):
    return dict(file=file, model='gemma-current', response='```json\n{"surface_percentages":{"calm water":100}}\n```', finish_reason='stop', **extra)


def test_processing_sources_precedence_identity_and_disabled_filters(tmp_path):
    source=tmp_path/'development';source.mkdir()
    rows=[response('classified',images=['images/original.jpg']),response('incomplete'),response('both')]
    rows[1]['finish_reason']='length'
    (source/'results.json').write_text(json.dumps(rows))
    skips=[dict(file=file,route='filter',stage=stage,score=.999) for file,stage in [('classified','local'),('filtered','local'),('disabled','batch'),('both','local')]]
    (source/'seawater-skips.json').write_text(json.dumps(skips))
    (source/'batch-filter-disabled.json').touch()
    (tmp_path/'annotation-sources.json').write_text(json.dumps([str(source/'results.json')]))
    (tmp_path/'annotation-policy.json').write_text('{"allow_legacy_qwen":false}')
    names=['classified','incomplete','filtered','disabled','both','live','live-filter','imported','unseen']
    points=[dict(id=f'chart-{i}',file=file) for i,file in enumerate(names)]
    db=tmp_path/'production.sqlite'
    with sqlite3.connect(db) as connection:
        connection.execute('CREATE TABLE photos(file TEXT,status TEXT,detail TEXT)')
        live=dict(answer=dict(surface_percentages={'nilas':75,'unknown':25}),version='production-model')
        records=[('live','gemma',live),('live-filter','filtered',dict(triage=dict(route='filter',stage='batch'))),('imported','gemma',dict(imported=True,record=response('imported',images=['original.jpg']))),('both','filtered',dict(triage=dict(route='filter'))),('outside-map','gemma',live)]
        connection.executemany('INSERT INTO photos VALUES (?,?,?)',[(f,s,json.dumps(d)) for f,s,d in records])
    (tmp_path/'annotation-processing-sources.json').write_text(json.dumps([dict(kind='production_sqlite',path=str(db))]))
    before=db.read_bytes()
    annotations,legacy=explorer().collect_annotations(tmp_path,points,[])
    actual={p['file']:annotations.get(p['id'],{}).get('status','pending') for p in points}
    assert actual==dict(classified='classified',incomplete='pending',filtered='filtered',disabled='pending',both='classified',live='classified',**{'live-filter':'filtered'},imported='classified',unseen='pending')
    assert legacy is False and db.read_bytes()==before
    classified=annotations['chart-0']
    assert classified['images']==['images/original.jpg']
    assert {s['kind'] for s in classified['sources']}=={'response_json','seawater_json'}
    for id in ('chart-2','chart-6'):
        assert 'response' not in annotations[id]
        assert 'surface_percentages' not in annotations[id]
        assert 'human_label' not in annotations[id]
    assert annotations['chart-7']['source']['imported'] is True
    assert annotations['chart-7']['images']==['original.jpg']
    assert len(annotations)==7


def test_incomplete_later_response_cannot_hide_accepted_response(tmp_path):
    rows=[response('photo'),dict(file='photo',model='gemma-current',response='{',finish_reason='length')]
    annotations,_=explorer().collect_annotations(tmp_path,[dict(id='point',file='photo')],rows)
    assert annotations['point']['status']=='classified'
    assert annotations['point']['finish_reason']=='stop'


def test_missing_sqlite_is_never_created(tmp_path):
    import pytest
    path=tmp_path/'absent.sqlite'
    with pytest.raises(sqlite3.OperationalError):
        list(explorer().sqlite_annotations(path,{'photo'},False))
    assert not path.exists()


def test_ui_keeps_filter_estimates_out_of_percentage_colours_and_pending():
    import subprocess
    script=Path(__file__).parents[1]/'tools/ice-region-ui.js'
    helpers='\n'.join(line for line in script.read_text().splitlines() if any(line.startswith('function '+name+'(') for name in ('processingStatus','processingSummary','responseText','annotation','fill')))
    setup="""
const assert=require('node:assert/strict');
const data={points:[{id:'classified'},{id:'filtered'},{id:'pending'}],annotations:{
 classified:{status:'classified',response:'{"surface_percentages":{"nilas":80}}'},
 filtered:{status:'filtered',filter:{route:'filter',score:.999},source:{kind:'seawater_json',path:'/source/skips.json'}}
}};
let colour='surface:nilas',highlightLabel=null;
const $=()=>({value:colour});
"""
    checks=r"""
assert.equal(annotation('filtered'),null);
assert.equal(fill({id:'filtered'},1),'#777');
assert.notEqual(fill({id:'classified'},0),'#777');
assert.match(responseText(data.annotations.filtered),/no full model classification/);
assert.match(responseText(data.annotations.filtered),/source\/skips.json/);
assert.equal(responseText(undefined),'Awaiting model annotation');
assert.equal(processingSummary(),'1 model classifications · 1 seawater filter estimates · 1 pending / 3 photos');
assert.deepEqual(data.points.filter(p=>processingStatus(p.id)==='pending').map(p=>p.id),['pending']);
colour='processing';
assert.equal(fill({id:'filtered'},1),'#69c');
assert.equal(fill({id:'classified'},0),'#5c9');
assert.equal(fill({id:'pending'},2),'#777');
"""
    subprocess.run(['node','-e',setup+helpers+checks],check=True,capture_output=True,text=True)
