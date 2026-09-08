"""Six paired local-only trials per model, serial, with thermal cutoffs."""
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import urllib.request

spec=importlib.util.spec_from_file_location('overnight',Path(__file__).with_name('ice-overnight.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prompts=m.module('context_prompt','ice-context-prompt.py')
quality=m.module('quality','ice-response-quality.py')
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,default=Path('/home/cryomics/Downloads/amundsen-ice-model-comparison'))
p.add_argument('--model',choices=['gemma4','muse-glimmer'])
p.add_argument('--queue',type=Path,help='Run all entries in this queue, instead of six test cases')
p.add_argument('--hours',type=float,default=2)
a=p.parse_args();assert 0<a.hours<=8
root=a.output
root.mkdir(exist_ok=True)
baseline=json.loads(Path('/home/cryomics/Downloads/amundsen-ice-qwen-k32-parallel/results.json').read_text())
by_file={r['file']:r for r in baseline}
queue=json.loads(Path('/home/cryomics/Downloads/amundsen-ice-qwen-k32-classes/queue.json').read_text())['queue']
chosen=json.loads(a.queue.read_text())['queue'] if a.queue else [next(q for q in queue if q['cluster']==c) for c in (0,6,12,18,24,31)]
assert len(chosen)==len({q['file'] for q in chosen}), 'Duplicate queue entries'
key='data:image/png;base64,'+base64.b64encode(Path('/home/cryomics/Downloads/ice-reference-key/ice-reference-key.png').read_bytes()).decode()
models=[('gemma4','google/gemma-4-26b-a4b-qat','off'),('muse-glimmer','lmstudio-community/muse-glimmer-30b',None)]
if a.model:models=[row for row in models if row[0]==a.model]
m.atomic(root/'manifest.json',json.dumps(dict(queue=chosen,models=models,temperature=0,max_output_tokens=6000),indent=2))
window=m.RunningTemperature();deadline=time.monotonic()+a.hours*3600

def check_temperature(out):
    cpu,gpu,power,mem=m.monitor.temperatures();avg=window.add(time.monotonic(),cpu)
    with (out/'telemetry.csv').open('a') as f:
        f.write(f'{datetime.now(timezone.utc).isoformat()},{cpu},{gpu},{power},{mem},{avg}\n')
    if avg>=95 or gpu>=83:raise RuntimeError('Thermal cutoff')
    if time.monotonic()>deadline:raise RuntimeError('Bounded run deadline')
    return gpu

def request(model,reasoning,blocks):
    payload=dict(model=model,input=blocks,temperature=0,max_output_tokens=6000,store=False)
    if reasoning is not None:payload['reasoning']=reasoning
    client=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req=urllib.request.Request('http://127.0.0.1:1234/api/v1/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    try:
        with client.open(req,timeout=600) as response:return json.load(response)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f'HTTP {e.code}: {e.read().decode()[:2000]}') from e

for slug,model,reasoning in models:
    out=root/slug;out.mkdir(exist_ok=True)
    rows=json.loads((out/'results.json').read_text()) if (out/'results.json').exists() else []
    done={r['file'] for r in rows if r.get('finish_reason')=='stop'}
    pool=ThreadPoolExecutor(max_workers=1)
    try:
        while check_temperature(out)>=78:time.sleep(3)
        print('Loading',model,flush=True)
        subprocess.run([m.LMS,'load',model,'--gpu','max','--context-length','16384','--parallel','1','--identifier',model,'--ttl','1800','-y'],check=True,timeout=180)
        for q in chosen:
            if q['file'] in done:continue
            while check_temperature(out)>=78:time.sleep(3)
            old=by_file.get(q['file'])
            if old is None:
                old=next(m.jobs({'scenes':[dict(scene=q['file'],file=q['file'])]},{'tiles':[]},Path('/media/cryomics/T7 Shield/Amundsen/Camera_360/2025_LEG_04')))
                old['prompt']=baseline[0]['prompt']
            images=old['images'][:2]+[key]
            texts,blocks=prompts.build(old['prompt'],images)
            started=time.monotonic();future=pool.submit(request,model,reasoning,blocks)
            print('Started',slug,q['file'],flush=True)
            while not future.done():
                check_temperature(out);time.sleep(2)
            raw=future.result();stats=raw.get('stats',{})
            response='\n'.join(i['content'] for i in raw.get('output',[]) if i['type']=='message')
            audit=quality.assess(response)
            try:
                parsed=json.loads(response.strip().removeprefix('```json').removesuffix('```').strip())
                for field in ('context_photo_description','roi_description'):
                    if not isinstance(parsed.get(field),str) or not parsed[field].strip():audit['quality_flags'].append('Missing '+field)
            except (ValueError,AttributeError):audit['quality_flags'].append('Cannot verify descriptions')
            row=dict(id=old['id'],file=q['file'],images=images,prompt='\n\n'.join(texts),prompt_segments=texts,
                     input_image_order=['context','reference_key','roi'],model=model,reasoning=reasoning or 'not exposed',
                     human_label=old.get('human_label',[]),queue_metadata=q,response=response,usage=stats,
                     finish_reason='length' if stats.get('total_output_tokens',0)>=6000 else 'stop',
                     elapsed_s=time.monotonic()-started,utc=datetime.now(timezone.utc).isoformat(),**audit)
            if a.queue:
                # Keep one copy of each image on disk instead of rewriting many
                # hundreds of megabytes of base64 on every completed request.
                import hashlib
                image_dir=out/'images';image_dir.mkdir(exist_ok=True)
                identity=hashlib.sha256(q['file'].encode()).hexdigest()[:20]
                paths=[]
                for index,image in enumerate(images):
                    name='reference-key.png' if index==2 else f'{identity}-{index}.jpg'
                    path=image_dir/name
                    if not path.exists():path.write_bytes(base64.b64decode(image.split(',',1)[1]))
                    paths.append('images/'+name)
                row['images']=paths
            rows.append(row);m.atomic(out/'results.json',json.dumps(rows));m.render(out,rows)
            print('Saved',slug,len(rows),row['elapsed_s'],'seconds',audit['quality_flags'],flush=True)
        m.atomic(out/'status.json',json.dumps(dict(saved=len(rows),complete=True)))
    except Exception as e:
        print('Stopped',slug,repr(e),flush=True)
        m.atomic(out/'status.json',json.dumps(dict(saved=len(rows),complete=False,error=str(e))))
    finally:
        subprocess.run([m.LMS,'unload',model],timeout=45,check=False)
        pool.shutdown(wait=False,cancel_futures=True)
    if time.monotonic()>deadline:break
m.atomic(root/'index.html','<meta charset="utf-8"><h1>Paired ice model tests</h1>'+''.join(f'<p><a href="{slug}/index.html">{slug}</a> · <a href="{slug}/status.json">status</a></p>' for slug,_,_ in models))
