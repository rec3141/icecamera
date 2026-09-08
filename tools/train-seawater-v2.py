"""Human-label candidate with day holdouts and independent ROI-only checks."""
import json
import hashlib
import sqlite3
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold

ROOT=Path('/home/cryomics/Downloads');OUT=ROOT/'amundsen-ice-seawater'
ICE={'grease ice','nilas','thin fyi','icy bits','brash ice','ice floe'}
WATER={'smooth water','small waves','whitecap'}
def forest():return ExtraTreesRegressor(n_estimators=64,max_depth=16,min_samples_leaf=2,random_state=42,n_jobs=1)

def main():
    labels=json.loads((ROOT/'amundsen-ice-qwen-k32-classes/labels-clean.json').read_text())['labels']
    eligible={r['file']:int(not (set(r['labels'])&ICE)) for r in labels if set(r['labels'])&ICE or (r['labels'] and set(r['labels'])<=WATER)}
    with sqlite3.connect('file:'+str(ROOT/'amundsen-ice-2025-2026-tsne/features.sqlite')+'?mode=ro',uri=True) as db:
        rows=[(f,json.loads(v)) for f,v in db.execute('SELECT file,vector FROM features WHERE error IS NULL ORDER BY file')]
    vectors=dict(rows);files=sorted(set(eligible)&vectors.keys());x=np.asarray([vectors[f] for f in files],dtype=np.float32);y=np.asarray([eligible[f] for f in files]);groups=np.array([f[:8] for f in files]);held=np.zeros(len(y))
    for train,test in GroupKFold(5).split(x,y,groups):held[test]=forest().fit(x[train],y[train]).predict(x[test])
    report=[]
    for t in [.9,.95,.98,.99,.995,.999]:
        selected=held>=t;report.append(dict(threshold=t,selected=int(selected.sum()),human_ice_disagreements=int(((y==0)&selected).sum())))
    fitted=forest().fit(x,y);trees=[]
    for e in fitted.estimators_:
        t=e.tree_;trees.append(dict(left=t.children_left.tolist(),right=t.children_right.tolist(),feature=t.feature.tolist(),threshold=t.threshold.tolist(),value=t.value[:,:,0].tolist()))
    model=dict(id=hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest()[:16],kind='seawater-triage-shadow',feature_size=[600,300],geometry='original-roi-v1',outputs=['clear_seawater_score'],trees=trees,mean=x.mean(0).tolist(),scale=np.maximum(x.std(0),.01).tolist(),auto_filter_enabled=False)
    (OUT/'v2-model.json').write_text(json.dumps(model,separators=(',',':')))
    allx=np.asarray([v for _,v in rows],dtype=np.float32);scores=fitted.predict(allx)
    predictions=[dict(file=f,score=float(s)) for (f,_),s in zip(rows,scores)]
    (OUT/'v2-scores.json').write_text(json.dumps(predictions))
    byfile={r['file']:r['score'] for r in predictions}
    checks=[dict(file=r['file'],decision=r['decision'],score=byfile.get(r['file'])) for r in json.loads((OUT/'binary-recheck.json').read_text())]
    result=dict(training=len(y),water=int(y.sum()),day_holdout=report,roi_only_checks=checks,note='Human-label agreement is not verified safety; ROI checks were not training targets.')
    (OUT/'v2-evaluation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
