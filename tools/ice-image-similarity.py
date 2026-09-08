"""Image-reference experiment: direct pixel matches, never automatic live labels."""
import argparse
from html import escape
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageOps,ImageDraw

def pixels(path):
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB').resize((48,24),Image.Resampling.BOX),dtype=np.float32)/255
    gray=rgb@np.array([.299,.587,.114],dtype=np.float32)
    # Keep both absolute colour and contrast-normalized spatial image structure.
    normalized=(gray-gray.mean())/max(float(gray.std()),.02)
    return rgb,normalized

def distances(rgb,gray,ref_rgb,ref_gray):
    colour=np.sqrt(np.mean((ref_rgb-rgb)**2,axis=(1,2,3)))
    structure=np.sqrt(np.mean((ref_gray-gray)**2,axis=(1,2)))
    return colour+.04*structure

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    root=Path('/home/cryomics/Downloads');portal=root/'amundsen-ice-2025-2026-tsne';out=root/'ice-image-similarity';out.mkdir(exist_ok=True)
    points=json.loads((portal/'embedding.json').read_text());byfile={r['file']:r for r in points}
    previous=root/'amundsen-ice-seawater'
    reviews=json.loads((previous/'v2-review.json').read_text());decisions=json.loads((previous/'v2-review-decisions.json').read_text())
    chosen={r['file'] for r in decisions if r['route']=='filter'}
    seeds=[dict(file=r['file'],path=r['roi'],label='water',provenance='Astra visually reviewed wave-water candidate, development seed') for r in reviews if r['file'] in chosen]
    old=json.loads((previous/'review.json').read_text())
    for index in [0,13]:
        r=old[index];seeds.append(dict(file=r['file'],path=r['images'][1],label='ice',provenance='Visible isolated fragments or coherent ice at ROI edges; development counterexample'))
    # Use the SAME preview representation for references and query images.
    for seed in seeds:seed['path']=str((portal/byfile[seed['file']]['images'][1]).resolve()) if not byfile[seed['file']]['images'][1].startswith('file:') else byfile[seed['file']]['images'][1][7:]
    refs=[pixels(r['path']) for r in seeds];ref_rgb=np.stack([r[0] for r in refs]);ref_gray=np.stack([r[1] for r in refs]);water=np.array([r['label']=='water' for r in seeds]);seedfiles={r['file'] for r in seeds}
    (out/'seeds.json').write_text(json.dumps(seeds,indent=2))
    paths=out/'matches.jsonl';done={}
    if paths.exists():
        for line in paths.read_text().splitlines():
            r=json.loads(line);done[r['file']]=r
    targets=points[:a.limit] if a.limit else points
    with paths.open('a') as log:
        for i,point in enumerate(targets):
            if point['file'] in done:continue
            path=point['images'][1];path=Path(path[7:]) if path.startswith('file:') else portal/path
            try:
                rgb,gray=pixels(path);d=distances(rgb,gray,ref_rgb,ref_gray)
                # Exclude self matches; a reference cannot validate itself.
                for j,seed in enumerate(seeds):
                    if seed['file']==point['file']:d[j]=np.inf
                wi=int(np.argmin(np.where(water,d,np.inf)));ni=int(np.argmin(np.where(~water,d,np.inf)))
                row=dict(file=point['file'],id=point['id'],image=str(path),water_distance=float(d[wi]),ice_distance=float(d[ni]),water_reference=seeds[wi]['file'],seed=point['file'] in seedfiles)
            except (OSError,ValueError) as error:row=dict(file=point['file'],error=str(error))
            done[point['file']]=row;log.write(json.dumps(row)+'\n');log.flush()
            if i%500==0:
                (out/'progress.json').write_text(json.dumps(dict(stage='image matching',processed=i+1,total=len(targets))))
                print('Image matches',i+1,'/',len(targets),flush=True)
    good=[r for r in done.values() if 'error' not in r and not r['seed']]
    curve=[]
    for cutoff in [.04,.06,.08,.10,.12,.15,.18]:
        selected=[r for r in good if r['water_distance']<cutoff and r['water_distance']<.8*r['ice_distance']]
        curve.append(dict(cutoff=cutoff,candidates=len(selected),fraction=len(selected)/len(good)))
    # Review possible propagation errors, distributed across distance bands/days.
    review=[];seen=set()
    for lo,hi in [(0,.06),(.06,.08),(.08,.10),(.10,.12),(.12,.15),(.15,.18)]:
        candidates=sorted([r for r in good if lo<=r['water_distance']<hi and r['water_distance']<.8*r['ice_distance']],key=lambda r:r['water_distance'],reverse=True)
        days=set()
        for r in candidates:
            day=r['file'].split('/')[-3]
            if day in days or r['file'] in seen:continue
            days.add(day);seen.add(r['file']);review.append(r)
            if len(days)==6:break
    (out/'review.json').write_text(json.dumps(review,indent=2))
    sheet=Image.new('RGB',(1600,230*((len(review)+3)//4)),'#17212b');draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(review):
        x=i%4*400;y=i//4*230
        with Image.open(r['image']) as im:sheet.paste(ImageOps.contain(im,(396,198)),(x,y+28))
        draw.text((x+4,y+3),f'{i+1} water {r["water_distance"]:.3f} ice {r["ice_distance"]:.3f}',fill='#00ff66')
    if review:sheet.save(out/'review.jpg',quality=90)
    report=dict(stage='complete',matched=len(done),errors=sum('error' in r for r in done.values()),threshold_sweep=curve,live_enabled=False,note='Direct ROI pixel/colour and normalized structure comparisons; ten water seeds and two ice counterexamples. Development coverage only, not accuracy. No labels propagated into live queue.')
    (out/'progress.json').write_text(json.dumps(report,indent=2))
    page='<!doctype html><meta charset="utf-8"><title>Image-similarity pilot</title><style>body{font:17px system-ui;max-width:1100px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Actual-image similarity pilot</h1><p>Shadow experiment only. All matches retain their reference image and distance. Self-matches excluded. These are candidate water matches, not validated propagated labels.</p><pre>'+escape(json.dumps(report,indent=2))+'</pre><img src="ice-image-similarity/review.jpg">'
    for i,r in enumerate(review):page+=f'<p>{i+1}: {escape(r["file"])}<br>Reference: {escape(r["water_reference"])}</p>'
    (root/'similarity.html').write_text(page);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
