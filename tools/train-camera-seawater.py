"""Shadow-only seawater triage pilot; teacher targets are weak evidence, not truth."""
import hashlib
from html import escape
import json
from pathlib import Path
import sqlite3
import time
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dashboard.camera_model import infer
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

ROOT=Path('/home/cryomics/Downloads')
OUT=ROOT/'amundsen-ice-seawater';OUT.mkdir(exist_ok=True)
WATER={'calm water','smooth water','small waves','whitecap'}
ICE={'grease ice','nilas','thin fyi','thin ice floe','icy bits','brash ice','ice floe','thick ice floe'}

def target(row):
    if row.get('finish_reason')!='stop':raise ValueError('incomplete')
    d=json.loads(row['response'].strip().removeprefix('```json').removesuffix('```').strip())
    s=d['surface_percentages'];a=d['artifact_percentages']
    values=np.asarray(list(s.values()),dtype=float)
    if not np.isfinite(values).all() or (values<0).any() or (values>100).any() or abs(values.sum()-100)>.01:raise ValueError('invalid budget')
    if not {'blurry','fog','reflection','night'}<=a.keys():raise ValueError('missing artifact fields')
    artifacts=np.asarray(list(a.values()),dtype=float)
    if not np.isfinite(artifacts).all() or (artifacts<0).any() or (artifacts>100).any():raise ValueError('invalid artifacts')
    water=sum(s.get(k,0) for k in WATER);ice=sum(s.get(k,0) for k in ICE)
    pure=water==100 and s.get('unknown',100)==0 and d.get('visibility')=='clear' and d.get('confidence')=='high' and artifacts.max()<=5
    return int(pure),dict(teacher_water=water,teacher_ice=ice,teacher_unknown=s.get('unknown'),artifacts=a)

def forest():return ExtraTreesRegressor(n_estimators=64,max_depth=12,min_samples_leaf=3,random_state=42,n_jobs=1)

def metrics(y,s,threshold):
    chosen=s>=threshold;n=int(chosen.sum());errors=int(((y==0)&chosen).sum())
    return dict(threshold=threshold,screened=n,total=len(y),screened_fraction=n/len(y),
                teacher_disagreements=errors,teacher_agreement=None if not n else 1-errors/n,
                note='Agreement with weak teacher eligibility, NOT verified seawater accuracy')

def main():
    vectors={}
    with sqlite3.connect('file:'+str(ROOT/'amundsen-ice-2025-2026-tsne/features.sqlite')+'?mode=ro',uri=True) as db:
        vectors={f:json.loads(v) for f,v in db.execute('SELECT file,vector FROM features WHERE error IS NULL AND vector IS NOT NULL')}
    human={r['file']:r['labels'] for r in json.loads((ROOT/'amundsen-ice-qwen-k32-classes/labels-clean.json').read_text())['labels']}
    samples=[];skipped=[]
    for name in ['amundsen-ice-gemma-leg4-size-evidence','amundsen-ice-gemma-2026']:
        for row in json.loads((ROOT/name/'results.json').read_text()):
            try:
                y,details=target(row);x=vectors[row['file']]
                samples.append(dict(file=row['file'],id=row['id'],x=x,y=y,day=row['file'].split('/')[-3],
                    year=row['file'].split('/')[-3][:4],human=human.get(row['file'],[]),
                    images=[str(ROOT/name/p) for p in row['images']],**details))
            except (ValueError,KeyError,TypeError) as e:skipped.append(dict(file=row['file'],reason=str(e)))
    x=np.asarray([r['x'] for r in samples],dtype=np.float32);y=np.array([r['y'] for r in samples]);groups=np.array([r['day'] for r in samples])
    held=np.zeros(len(y));folds=[]
    for train,test in GroupKFold(n_splits=5).split(x,y,groups):
        held[test]=forest().fit(x[train],y[train]).predict(x[test]);folds.append(dict(test_days=sorted(set(groups[test]))))
        print('Held-out fold',len(folds),flush=True)
    thresholds=[.5,.7,.8,.9,.95,.98,.99]
    transfer={}
    for year in ['2025','2026']:
        mask=np.array([r['year']==year for r in samples]);score=forest().fit(x[mask],y[mask]).predict(x[~mask])
        transfer[year+' train, other year test']=[metrics(y[~mask],score,t) for t in thresholds]
    fitted=forest().fit(x,y);trees=[]
    for estimator in fitted.estimators_:
        t=estimator.tree_;trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value[:,:,0].tolist()))
    model=dict(id=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()[:16],kind='seawater-triage-shadow',
        feature_size=[600,300],geometry='original-roi-v1',outputs=['clear_seawater_score'],trees=trees,
        mean=x.mean(0).tolist(),scale=np.maximum(x.std(0),.01).tolist(),auto_filter_enabled=False,
        note='Weak teacher target; score is not calibrated probability. Do not skip Gemma based on this candidate.')
    (OUT/'candidate-model.json').write_text(json.dumps(model,separators=(',',':')))
    begin=time.perf_counter()
    portable=np.array([infer(v,model)['scores']['clear_seawater_score'] for v in x[:100]])
    inference_ms=(time.perf_counter()-begin)*10
    np.testing.assert_allclose(portable,fitted.predict(x[:100]),atol=1e-10)
    report=dict(rows=len(y),eligible_water=int(y.sum()),days=len(set(groups)),skipped=skipped,folds=folds,
        heldout=[metrics(y,held,t) for t in thresholds],cross_year=transfer,auto_filter_enabled=False,inference_ms=inference_ms)
    for r,s in zip(samples,held):r.pop('x');r['heldout_water_score']=float(s)
    (OUT/'predictions.json').write_text(json.dumps(samples,indent=2));(OUT/'evaluation.json').write_text(json.dumps(report,indent=2))
    # Prioritize false-safe candidates, human/teacher contradictions and confident
    # teacher-water examples; retain distinct days before nearby repetitions.
    review=[]
    buckets=[sorted([r for r in samples if not r['y']],key=lambda r:-r['heldout_water_score']),
             sorted([r for r in samples if r['y'] and set(r['human'])&ICE],key=lambda r:-r['heldout_water_score']),
             sorted([r for r in samples if r['y']],key=lambda r:-r['heldout_water_score'])]
    for bucket in buckets:
        days=set();selected=0
        for r in bucket:
            if r['file'] in {s['file'] for s in review} or r['day'] in days:continue
            review.append(r);days.add(r['day']);selected+=1
            if selected==8:break
    (OUT/'review.json').write_text(json.dumps(review,indent=2))
    page='<!doctype html><meta charset="utf-8"><title>Seawater filter pilot</title><style>body{font:16px system-ui;max-width:1100px;margin:2rem auto}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}img{max-width:48%}pre{white-space:pre-wrap}</style><h1>Seawater filter · shadow pilot</h1><p>No photos are being filtered. Target: confidently visible 100% seawater in the ROI, with low artifacts. Human labels are comparison evidence, not training truth. Teacher labels can be wrong; held-out agreement does not establish safety.</p>'
    page+=f'<p>{len(y)} teacher examples, {int(y.sum())} eligible water, {len(set(groups))} days. Five-fold whole-day holdout. Threshold sweep is exploratory, not an independent deployment test. Portable inference: {inference_ms:.2f} ms per cached feature vector (feature extraction excluded).</p><p><a href="seawater-recheck.html">Live ROI-only Gemma recheck</a></p><table><tr><th>Score cutoff</th><th>Would screen</th><th>Teacher disagreements</th></tr>'
    for m in report['heldout']:page+=f'<tr><td>{m["threshold"]}</td><td>{m["screened"]} ({m["screened_fraction"]:.1%})</td><td>{m["teacher_disagreements"]}</td></tr>'
    page+='</table><h2>Cross-year tests</h2><pre>'+escape(json.dumps(transfer,indent=2))+'</pre><h2>Review candidates</h2>'
    for i,r in enumerate(review):
        page+=f'<h3>{i+1}: {escape(r["file"])}</h3><p>Held-out score {r["heldout_water_score"]:.3f}; teacher water {r["teacher_water"]}%; teacher ice {r["teacher_ice"]}%; human: {escape(", ".join(r["human"]))}</p>'
        page+=''.join('<img loading="lazy" src="'+escape(Path(p).as_uri(),quote=True)+'">' for p in r['images'])
    (ROOT/'seawater.html').write_text(page)
    print(json.dumps(report['heldout'],indent=2),flush=True)

if __name__=='__main__':main()
