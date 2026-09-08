"""Cache PC1–4 from all standardized 89-feature rows for an explorer."""
import argparse
import json
import sqlite3
import numpy as np
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--portal',type=Path,required=True);a=p.parse_args()
points=json.loads((a.portal/'embedding.json').read_text())
with sqlite3.connect((a.portal/'features.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
    vectors={f:json.loads(v) for f,v in db.execute('SELECT file,vector FROM features WHERE error IS NULL AND vector IS NOT NULL')}
x=np.array([vectors[p['file']] for p in points]);assert x.shape==(len(points),89) and np.isfinite(x).all()
mean=x.mean(0);scale=x.std(0);scale[scale==0]=1;z=(x-mean)/scale
_,s,vt=np.linalg.svd(z,full_matrices=False);axes=vt[:4].copy()
for axis in axes:
    if axis[np.argmax(np.abs(axis))]<0:axis*=-1
scores=z@axes.T
out=dict(ids=[p['id'] for p in points],scores=scores.tolist(),variance=(s[:4]**2/(s**2).sum()).tolist(),mean=mean.tolist(),scale=scale.tolist(),axes=axes.tolist())
target=a.portal/'feature-pcs.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(out));temp.replace(target)
print(len(points),'photos; explained variance:',out['variance'])
