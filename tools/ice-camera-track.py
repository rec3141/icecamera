"""Export every cached camera feature row as a mean-RGB coloured PC1 timeline."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import numpy as np


def project(matrix):
    matrix=np.asarray(matrix,dtype=float)
    if matrix.ndim!=2 or matrix.shape[1]!=89 or not np.isfinite(matrix).all():
        raise ValueError('Expected finite 89-feature matrix')
    mean=matrix.mean(axis=0);scale=matrix.std(axis=0);scale[scale==0]=1
    z=(matrix-mean)/scale
    _,singular,vt=np.linalg.svd(z,full_matrices=False)
    axis=vt[0]
    # Resolve SVD sign ambiguity deterministically, not separately per window.
    if axis[np.argmax(np.abs(axis))]<0:axis=-axis
    variance=float(singular[0]**2/np.sum(singular**2)) if np.any(singular) else 0
    return z@axis,dict(mean=mean.tolist(),scale=scale.tolist(),axis=axis.tolist(),explained_variance_ratio=variance)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--portal',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--leg',required=True)
    a=p.parse_args()
    points=json.loads((a.portal/'embedding.json').read_text())
    allowed={p['file'] for p in points}
    with sqlite3.connect((a.portal/'features.sqlite').resolve().as_uri()+'?mode=ro',uri=True) as db:
        rows=[(file,json.loads(vector)) for file,vector in db.execute('SELECT file,vector FROM features WHERE error IS NULL AND vector IS NOT NULL ORDER BY file') if file in allowed]
    if not rows:raise ValueError('No features available')
    matrix=np.array([v for _,v in rows]);scores,model=project(matrix)
    pcs=json.loads((a.portal/'feature-pcs.json').read_text())
    if pcs['ids']!=[p['id'] for p in points]:raise ValueError('PC identities differ')
    pcs_by_file={p['file']:values for p,values in zip(points,pcs['scores'])}
    result=[]
    for (file,vector),score in zip(rows,scores):
        match=re.search(r'Camera360_(\d{14})_cam_3\.jpg$',file)
        if not match:raise ValueError(f'Invalid camera timestamp: {file}')
        utc=datetime.strptime(match[1],'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc).isoformat()
        result.append(dict(time=utc,file=file,leg=a.leg,pc1=round(float(score),6),pcs=pcs_by_file[file],rgb=[round(float(vector[i])*255) for i in (0,7,14)]))
    payload=dict(schema='camera-feature-pc1-v1',feature_count=89,
                 feature_metadata=json.loads((a.portal/'layout-state.json').read_text()),
                 note='PC1 of standardized full feature matrix; colour is mean RGB of the reviewed sea-surface crop, not the whole image. Not ice concentration.',
                 pca=model,photos=result)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    temp=a.output.with_suffix('.tmp');temp.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False));temp.replace(a.output)
    print(f'{len(result)} photos; PC1 explains {model["explained_variance_ratio"]:.1%}; {a.output}')


if __name__=='__main__':main()
