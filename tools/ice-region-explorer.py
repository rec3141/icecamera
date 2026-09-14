"""Whole-region projection explorer with live annotations and human labels."""
import argparse
import importlib.util
import json
import fcntl
import sqlite3
import re
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.camera_completion import accepted_response


def response_annotation(row, source, allow_legacy_qwen):
    model = row.get('model', 'legacy Qwen')
    if not allow_legacy_qwen and (not row.get('model') or 'qwen' in model.lower()):
        return None
    try:
        answer = json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', row.get('response', '').strip()))
    except (ValueError, TypeError):
        answer = None
    accepted = row.get('finish_reason') == 'stop' and isinstance(answer, dict)
    if not allow_legacy_qwen:
        accepted = accepted_response(row)
    result = {key: row[key] for key in ('response', 'finish_reason', 'quality_flags', 'images', 'image', 'utc', 'seawater_triage') if key in row}
    result.update(status='classified' if accepted else 'pending', model=model, source=source)
    return result


def sqlite_annotations(path, files, allow_legacy_qwen):
    # A read-only connection observes live commits without owning the production queue.
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True, timeout=5)) as connection:
        for file, status, raw in connection.execute('SELECT file,status,detail FROM photos'):
            if file not in files:
                continue
            detail = json.loads(raw or '{}')
            record = detail.get('record', {})
            source = dict(kind='production_sqlite', path=str(path), imported=bool(detail.get('imported')))
            if status == 'gemma':
                if not record and isinstance(detail.get('answer'), dict):
                    record = dict(response=json.dumps(detail['answer']), finish_reason='stop', model=detail.get('version', 'production Gemma'), utc=detail.get('utc'))
                row = response_annotation(record, source, allow_legacy_qwen)
                if row:
                    yield file, row
            elif status == 'filtered':
                decision = record or detail.get('triage', {})
                yield file, dict(status='filtered', source=source, filter=decision)


def collect_annotations(output, points, responses):
    policy = output/'annotation-policy.json'
    allow_legacy_qwen = json.loads(policy.read_text()).get('allow_legacy_qwen', True) if policy.exists() else True
    files = {point['file'] for point in points}
    by_file = {}
    rank = {'pending': 0, 'filtered': 1, 'classified': 2}

    def add(file, row):
        if file not in files or row is None:
            return
        previous = by_file.get(file)
        origins = previous.get('sources', []) if previous else []
        if row['source'] not in origins:
            origins = [*origins, row['source']]
        if previous is None or rank[row['status']] >= rank[previous['status']]:
            by_file[file] = dict(row, sources=origins)
        else:
            previous['sources'] = origins

    sources = output/'annotation-sources.json'
    for name in json.loads(sources.read_text()) if sources.exists() else []:
        path = Path(name)
        if path.exists():
            source = dict(kind='response_json', path=str(path), image_root=str(path.parent))
            for row in json.loads(path.read_text()):
                add(row['file'], response_annotation(row, source, allow_legacy_qwen))
        skips = path.with_name('seawater-skips.json')
        if skips.exists():
            for row in json.loads(skips.read_text()):
                disabled = 'batch-filter-disabled.json' if row.get('stage') == 'batch' else 'seawater-filter-disabled.json'
                if not skips.with_name(disabled).exists():
                    add(row['file'], dict(status='filtered', source=dict(kind='seawater_json', path=str(skips)), filter=row))
    processing = output/'annotation-processing-sources.json'
    for source in json.loads(processing.read_text()) if processing.exists() else []:
        if source['kind'] != 'production_sqlite':
            raise ValueError('Unknown processing source: ' + source['kind'])
        for file, row in sqlite_annotations(Path(source['path']), files, allow_legacy_qwen):
            add(file, row)
    for row in responses:
        add(row['file'], response_annotation(row, dict(kind='provided_response'), allow_legacy_qwen))
    return {point['id']: by_file[point['file']] for point in points if point['file'] in by_file}, allow_legacy_qwen


def render(output, responses, follow_links=True, live_only=False):
    with (output/'render.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        _render(output,responses,follow_links,live_only)


def _render(output, responses, follow_links=True, live_only=False):
    layout=output/'embedding.json'
    if not layout.exists(): return
    points=json.loads(layout.read_text())
    annotations,allow_legacy_qwen=collect_annotations(output,points,responses)
    live=output/'annotations-live.js';temp=live.with_suffix('.tmp')
    temp.write_text('window.updateIceAnnotations('+json.dumps(annotations).replace('<','\\u003c')+');');temp.replace(live)
    if live_only:return
    projections={'tsne':dict(coords=[[p['x'],p['y']] for p in points],trees={})}
    if (output/'projections.json').exists():
        extra=json.loads((output/'projections.json').read_text())
        if extra['ids']!=[p['id'] for p in points]: raise ValueError('Projection identities differ')
        projections.update(extra['projections'])
    fixed=output/'ward-k64.json'
    fixed=json.loads(fixed.read_text()) if fixed.exists() else None
    if fixed and fixed['ids']!=[p['id'] for p in points]:raise ValueError('Fixed cut identities differ')
    pcs_path=output/'feature-pcs.json'
    pcs=json.loads(pcs_path.read_text()) if pcs_path.exists() else None
    if pcs and pcs['ids']!=[p['id'] for p in points]:raise ValueError('Feature PC identities differ')
    payload=json.dumps(dict(points=points,annotations=annotations,projections=projections,fixed=fixed,pcs=pcs,allow_legacy_qwen=allow_legacy_qwen)).replace('<','\\u003c')
    script=Path(__file__).with_name('ice-region-ui.js').read_text()
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Ice region projections</title>
<style>body{font:16px system-ui;background:#17212b;color:#eee;margin:18px}main{display:grid;grid-template-columns:1fr 1fr;gap:16px}canvas{width:100%;background:#0d1720;touch-action:none}img{max-width:100%}pre{white-space:pre-wrap}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:8px 0}button,select,input{padding:5px}#tip{position:fixed;pointer-events:none;background:#17212b;padding:6px;display:none}#tip img{width:250px}@media(max-width:800px){main{grid-template-columns:1fr}}</style>
<h1>Whole-region projections</h1><p>Visual similarity is not verified ice type. No labels automatically propagate. Scroll to zoom; drag to pan or lasso. Shift adds to selection; Alt subtracts. Overlapping selections can have multiple human labels.</p>
<div class="controls"><label>Projection <select id="projection"></select></label><label>Clustering <select id="linkage"><option value="none">None</option><option value="ward">Ward</option><option value="single">Single linkage</option></select></label><label>Tree cut <input id="height" type="range" min="0" max="1000" value="500"><output id="cut"></output></label><label>Colour <select id="colour"><option value="average">Mean RGB</option><option value="brightness">Brightness</option><option value="cluster">Cluster</option><option value="human">Human labels</option><option value="processing">Processing status</option><option value="total">Total ice %</option><option value="thin_new_ice">Thin/new %</option><option value="broken_ice">Broken/rubble %</option><option value="consolidated_ice">Consolidated %</option><option value="unknown">Unknown %</option><option value="visibility">Visibility</option></select></label><span id="status"></span></div>
<div class="controls"><label>Drag <select id="mode"><option value="pan">Pan</option><option value="lasso">Lasso</option></select></label><button id="clear">Clear selection</button><span id="selection"></span><input id="label" placeholder="Label name" maxlength="100"><button id="apply">Add label</button><button id="remove">Remove this label</button><button id="export">Export labels</button><label>Import <input id="import" type="file" accept=".json"></label><button id="queue">Export Qwen queue</button><button id="refresh">Refresh annotations</button><span id="save"></span></div>
<main><section><canvas id="map" width="800" height="650"></canvas><p>Selected points outlined cyan. Processing status: green = model classification, blue = seawater filter estimate, grey = pending. Filter estimates have no detailed percentages. Grey in other modes means no colour/annotation available. Tree heights use each projection's own units and are not comparable across projections.</p></section><section><h2 id="title"></h2><p id="clusterinfo"></p><p id="human"></p><img id="crop"><img id="source"><pre id="response"></pre></section></main><div id="tip"></div><script>window.ICE_DATA=PAYLOAD;
SCRIPT</script>'''.replace('PAYLOAD',payload).replace('SCRIPT',script)
    tmp=output/'regions.html.tmp';tmp.write_text(page);tmp.replace(output/'regions.html')
    linked=output/'linked-explorers.json'
    if follow_links and linked.exists():
        for folder in json.loads(linked.read_text()):render(Path(folder),responses,False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--annotations',type=Path);p.add_argument('--render-only',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not a.render_only:
        import sys,base64,io
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from dashboard.ice_classifier import features
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        from PIL import Image
        spec=importlib.util.spec_from_file_location('overnight',Path(__file__).with_name('ice-overnight.py'))
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        data=json.loads((Path.home()/'Downloads/amundsen-ice-texture-brightness/tiles.json').read_text());points=[];vectors=[]
        for job in mod.jobs(data,{'tiles':[]},Path('/media/cryomics/T7 Shield/Amundsen/Camera_360/2025_LEG_04')):
            with Image.open(io.BytesIO(base64.b64decode(job['images'][1].split(',',1)[1]))) as crop: vector=features(crop)
            vectors.append(vector);points.append(dict(**{k:job[k] for k in ('id','file','images')},brightness=float(vector[21]),mean_rgb=[float(vector[i]) for i in (0,7,14)]))
        coords=TSNE(n_components=2,perplexity=min(15,(len(points)-1)/3),random_state=42,init='pca',learning_rate='auto').fit_transform(StandardScaler().fit_transform(np.array(vectors)))
        for point,xy in zip(points,coords):point.update(x=float(xy[0]),y=float(xy[1]))
        (a.output/'embedding.json').write_text(json.dumps(points))
    results=a.annotations or a.output/'results.json';render(a.output,json.loads(results.read_text()) if results.exists() else [])
    print(a.output/'regions.html')


if __name__=='__main__':main()
