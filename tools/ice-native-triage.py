"""Native-ROI object checks plus bounded temporal corroboration; shadow only."""
import argparse
from datetime import datetime
import hashlib
from html import escape
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy import ndimage
from PIL import Image,ImageDraw,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dashboard.camera_features import crop_region

def objects(crop):
    gray=np.asarray(crop.convert('L'),dtype=np.float32)/255
    # Native pixels retain small fragments erased by 48x24 matching.
    background=ndimage.uniform_filter(gray,size=31,mode='reflect')
    mask=(gray-background>.12)&(gray>.45)
    labels,n=ndimage.label(mask)
    boxes=[]
    for i,sl in enumerate(ndimage.find_objects(labels),1):
        if sl is None:continue
        local=labels[sl]==i;area=int(local.sum())
        if area<6:continue
        yy,xx=np.nonzero(local)
        cov=np.cov(np.stack([xx,yy]));eig=np.linalg.eigvalsh(cov)
        compactness=float(eig[0]/max(eig[1],.01))
        # Elongated wave crests pass; compact bright fragments warrant inspection.
        if compactness<.12:continue
        boxes.append(dict(box=[sl[1].start,sl[0].start,sl[1].stop,sl[0].stop],area=area,compactness=compactness))
    edges=[]
    for band in np.array_split(gray,4):
        for tile in np.array_split(band,6,axis=1):
            edges.append(float((np.abs(np.diff(tile,axis=0)).mean()+np.abs(np.diff(tile,axis=1)).mean())/2))
    ratio=min(edges)/max(float(np.median(edges)),1e-6)
    return dict(bright_objects=boxes,local_texture_ratio=ratio,mean=float(gray.mean()),
                texture=float(np.median(edges)),obscured=bool(ratio<.18 or np.median(edges)<.002 or gray.mean()<.075))

def stamp(file):return datetime.strptime(Path(file).name.split('_')[1],'%Y%m%d%H%M%S')

def neighbors(rows,index,max_seconds=300):
    r=rows[index];result=[]
    for j in [index-1,index+1]:
        if j<0 or j>=len(rows):return []
        other=rows[j]
        if r['file'].split('/')[0]!=other['file'].split('/')[0]:return []
        if abs((stamp(r['file'])-stamp(other['file'])).total_seconds())>max_seconds:return []
        result.append(other)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=240);a=p.parse_args()
    root=Path('/home/cryomics/Downloads');out=root/'ice-native-triage';out.mkdir(exist_ok=True)
    rows=sorted(json.loads((root/'ice-patch-similarity/scores.json').read_text()),key=lambda r:r['file'])
    candidates=[]
    for i,r in enumerate(rows):
        if r['seed'] or r['worst_water']>=.08 or r['negative_tiles'] or r['water_tiles']<17:continue
        ns=neighbors(rows,i)
        temporal=bool(ns) and all(n['worst_water']<.08 and not n['negative_tiles'] and n['water_tiles']>=17 for n in ns)
        candidates.append(dict(r,temporal=temporal,neighbors=[n['file'] for n in ns]))
    # Deterministic broad sample plus known suspicious strict-patch cases.
    candidates.sort(key=lambda r:hashlib.sha256(r['file'].encode()).hexdigest())
    selected=candidates[:a.limit] if a.limit else candidates
    concerns=json.loads((root/'ice-patch-similarity/strict-review.json').read_text())
    for n in [5,15,20,25,30,32]:
        row=next((r for r in candidates if r['file']==concerns[n-1]['file']),None)
        if row:
            existing=next((x for x in selected if x['file']==row['file']),None)
            if existing is not None:existing['development_case']=n
            else:selected.append(dict(row,development_case=n))
    results=[];started=time.monotonic()
    for i,r in enumerate(selected):
        source=(Path('/data/scratch/camera-originals')/r['file']) if r['file'].startswith('2026_') else Path('/media/cryomics/T7 Shield/Amundsen/Camera_360/2025_LEG_04')/r['file']
        try:
            with Image.open(source) as im:crop=crop_region(im)
            check=objects(crop);passed=r['temporal'] and not check['bright_objects'] and not check['obscured']
            result=dict(file=r['file'],image=r['image'],temporal=r['temporal'],neighbors=r['neighbors'],development_case=r.get('development_case'),candidate=passed,**check)
            results.append(result)
        except (OSError,ValueError) as e:results.append(dict(file=r['file'],error=str(e),candidate=False))
        if i%25==0:print(i+1,'/',len(selected),flush=True)
    (out/'results.json').write_text(json.dumps(results,indent=2))
    fresh=[r for r in results if not r.get('development_case')];passed=[r for r in fresh if r['candidate']]
    review=passed[:40];sheet=Image.new('RGB',(1600,max(1,230*((len(review)+3)//4))))
    draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(review):
        x=i%4*400;y=i//4*230
        with Image.open(r['image']) as im:sheet.paste(ImageOps.contain(im,(396,198)),(x,y+28))
        draw.text((x+3,y+4),f'{i+1} {r["file"].split("/")[-3]}',fill='#00ff66')
    sheet.save(out/'review.jpg',quality=90);(out/'review.json').write_text(json.dumps(review,indent=2))
    report=dict(patch_candidates=len(candidates),temporally_supported=sum(r['temporal'] for r in candidates),sampled=len(fresh),sample_passed=len(passed),seconds=time.monotonic()-started,
        development_checks=[dict(number=r['development_case'],passed=r['candidate'],objects=len(r.get('bright_objects',[])),obscured=r.get('obscured')) for r in results if r.get('development_case')],live_enabled=False,
        note='Native bright-object and texture checks are heuristics. Adjacent frames are correlated evidence, not independent labels. Sample counts are not full-dataset bypass counts.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    page='<!doctype html><meta charset="utf-8"><title>Native and temporal triage</title><style>body{font:17px system-ui;max-width:1100px;margin:auto}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>Native-resolution + temporal checks</h1><pre>'+escape(json.dumps(report,indent=2))+'</pre><img src="ice-native-triage/review.jpg">'
    for i,r in enumerate(review):page+=f'<p>{i+1}: {escape(r["file"])}</p>'
    (root/'native-triage.html').write_text(page);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
