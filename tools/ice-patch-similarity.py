"""Local image-patch reference matching with explicit hard negatives; shadow only."""
from html import escape
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image,ImageOps,ImageDraw

def patches(path):
    with Image.open(path) as im:
        rgb=np.asarray(im.convert('RGB').resize((48,24),Image.Resampling.BOX),dtype=np.float32)/255
    tiles=rgb.reshape(3,8,6,8,3).transpose(0,2,1,3,4).reshape(18,8,8,3)
    gray=tiles@np.array([.299,.587,.114],dtype=np.float32)
    z=(gray-gray.mean((1,2),keepdims=True))/np.maximum(gray.std((1,2),keepdims=True),.02)
    # Distances combine actual small RGB images and normalized texture images.
    return np.concatenate([tiles.reshape(18,-1)/np.sqrt(192),z.reshape(18,-1)*.04/8],axis=1).astype(np.float32)

def nearest(x,bank):
    return np.sqrt(np.maximum((x*x).sum(1)[:,None]+(bank*bank).sum(1)[None,:]-2*x@bank.T,0)).min(1)

def main():
    root=Path('/home/cryomics/Downloads');base=root/'ice-image-similarity';out=root/'ice-patch-similarity';out.mkdir(exist_ok=True)
    review=json.loads((base/'review.json').read_text());seeds=json.loads((base/'seeds.json').read_text())
    water=[(r['file'],r['path']) for r in seeds if r['label']=='water']
    for n in [1,3,5,7,8,10,11,12,13,14,15,19,22,23,24,28,30,32]:
        r=review[n-1];water.append((r['file'],r['image']))
    # Human-readable normalized boxes identify the specific non-water regions
    # in previously visually reviewed failures, not all background in that photo.
    bad=[(2,(0,0,1,.34)),(17,(0,0,.75,.5)),(20,(0,0,1,.34)),
         (21,(0,0,1,1)),(25,(0,0,1,1)),(33,(0,0,1,1)),(35,(.3,.15,.7,.65))]
    wb=np.concatenate([patches(p) for _,p in water]);nb=[];manifest=[]
    for n,box in bad:
        r=review[n-1];tile=patches(r['image']);indices=[]
        for i in range(18):
            x=(i%6+.5)/6;y=(i//6+.5)/3
            if box[0]<=x<=box[2] and box[1]<=y<=box[3]:indices.append(i)
        nb.extend(tile[indices]);manifest.append(dict(file=r['file'],box=box,tiles=indices))
    nb=np.asarray(nb);seedfiles={f for f,_ in water}|{r['file'] for r in manifest}
    (out/'references.json').write_text(json.dumps(dict(water=[f for f,_ in water],negative_regions=manifest,note='Development references selected by visual inspection; not independent evaluation'),indent=2))
    np.savez_compressed(out/'patch-bank.npz',water=wb,negative=nb)
    points=[json.loads(line) for line in (base/'matches.jsonl').read_text().splitlines() if 'error' not in json.loads(line)]
    scores=[];begin=time.monotonic()
    for i,r in enumerate(points):
        x=patches(r['image']);dw=nearest(x,wb);dn=nearest(x,nb)
        veto=int(((dn<.065)&(dn<dw*.85)).sum())
        scores.append(dict(file=r['file'],image=r['image'],seed=r['file'] in seedfiles,
            worst_water=float(dw.max()),mean_water=float(dw.mean()),negative_tiles=veto,
            water_tiles=int((dw<dn*.95).sum()),tile_water=dw.tolist(),tile_negative=dn.tolist()))
        if i%1000==0:print(i+1,'/',len(points),flush=True)
    (out/'scores.json').write_text(json.dumps(scores))
    usable=[r for r in scores if not r['seed']];curve=[]
    for threshold in [.06,.08,.10,.12,.15]:
        chosen=[r for r in usable if r['worst_water']<threshold and r['negative_tiles']==0 and r['water_tiles']>=17]
        curve.append(dict(cutoff=threshold,candidates=len(chosen),fraction=len(chosen)/len(usable)))
    previous_failures=[dict(number=n,**next(r for r in scores if r['file']==review[n-1]['file'])) for n,_ in bad]
    newreview=[];seen=set()
    for lo,hi in [(0,.06),(.06,.08),(.08,.10),(.10,.12),(.12,.15)]:
        candidates=sorted([r for r in usable if lo<=r['worst_water']<hi and r['negative_tiles']==0 and r['water_tiles']>=17],key=lambda r:r['worst_water'],reverse=True);days=set()
        for r in candidates:
            day=r['file'].split('/')[-3]
            if day in days or r['file'] in seen:continue
            newreview.append(r);seen.add(r['file']);days.add(day)
            if len(days)==6:break
    (out/'review.json').write_text(json.dumps(newreview,indent=2))
    if newreview:
        sheet=Image.new('RGB',(1600,230*((len(newreview)+3)//4)),'#17212b');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(newreview):
            x=i%4*400;y=i//4*230
            with Image.open(r['image']) as im:sheet.paste(ImageOps.contain(im,(396,198)),(x,y+28))
            draw.text((x+4,y+4),f'{i+1} worst {r["worst_water"]:.3f} water tiles {r["water_tiles"]}/18',fill='#00ff66')
        sheet.save(out/'review.jpg',quality=90)
    report=dict(matched=len(scores),seconds=time.monotonic()-begin,water_patches=len(wb),negative_patches=len(nb),threshold_sweep=curve,live_enabled=False,
        known_failures=[dict(number=r['number'],negative_tiles=r['negative_tiles'],worst_water=r['worst_water'],water_tiles=r['water_tiles']) for r in previous_failures],
        note='Known failures were used to build the negative bank, so their rejection is a regression check, not independent validation. Coverage counts exclude all reference images.')
    (out/'report.json').write_text(json.dumps(report,indent=2))
    page='<!doctype html><meta charset="utf-8"><title>Patch similarity</title><style>body{font:17px system-ui;max-width:1100px;margin:auto}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>Local patch similarity</h1><p>Shadow experiment: 18 spatial patches per actual ROI. Local non-water matches veto global water appearance. No live labels changed.</p><pre>'+escape(json.dumps(report,indent=2))+'</pre><h2>Fresh non-reference review sample</h2><img src="ice-patch-similarity/review.jpg">'
    for i,r in enumerate(newreview):page+=f'<p>{i+1}: {escape(r["file"])}</p>'
    (root/'patches.html').write_text(page);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
