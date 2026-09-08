"""Bounded four-slot, no-thinking review; one writer and shared thermal monitor."""
import argparse
import base64
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import subprocess
import time

spec=importlib.util.spec_from_file_location('overnight',Path(__file__).with_name('ice-overnight.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
quality=m.module('response_quality','ice-response-quality.py')
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--queue',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--hours',type=float,default=4)
p.add_argument('--reference-key',type=Path,help='Append this labeled PNG as image 3; replay exact baseline inputs')
p.add_argument('--parallel',type=int,choices=(1,2,4),default=4)
p.add_argument('--context-first',action='store_true',help='Interleave context, key definitions, ROI; include context_photo_description and roi_description in JSON')
a=p.parse_args()
assert not a.context_first or a.reference_key, '--context-first requires --reference-key'
assert 0<a.hours<=4
a.output.mkdir(parents=True,exist_ok=True)
rows=json.loads((a.output/'results.json').read_text()) if (a.output/'results.json').exists() else []
baseline=json.loads(a.baseline.read_text());prompt=baseline[0]['prompt'];labels=json.loads(a.labels.read_text());human={r['file']:r['labels'] for r in labels['labels']}
queue=json.loads(a.queue.read_text())['queue'];done={r['file'] for r in rows if r.get('finish_reason')=='stop'}
data={'scenes':[dict(scene=f'cluster-{q["cluster"]}-{i}',file=q['file']) for i,q in enumerate(queue) if q['file'] not in done]}
metadata={q['file']:q for q in queue}
jobs=iter(m.jobs(data,{'tiles':[]},Path('/media/cryomics/T7 Shield/Amundsen/Camera_360/2025_LEG_04')))
if a.reference_key:
    reference=a.reference_key.read_bytes()
    reference_uri='data:image/png;base64,'+base64.b64encode(reference).decode()
    reference_hash=hashlib.sha256(reference).hexdigest()
    by_file={r['file']:r for r in baseline if r.get('finish_reason')=='stop'}
    assert len(queue)==len({q['file'] for q in queue})
    assert all(q['file'] in by_file for q in queue), 'Complete paired baseline required'
    addition=('\nImage 3 is a labeled reference key, NOT the target scene. '
              'Its six rows show three human-approved examples each of grease ice, nilas, '
              'thin fyi, icy bits, brash ice, and ice floe. Use their visual appearances '
              'to interpret those category names. Each example can contain water/background; '
              'a label does not mean 100% coverage. Do not measure the key or its green text. '
              'Estimate percentages ONLY for the target region in images 1 and 2, '
              'using the original JSON schema and all original categories. '
              'When ambiguous, retain uncertainty rather than forcing a match.')
    jobs=iter([dict(**{k:by_file[q['file']][k] for k in ('id','file','polygon')},
                    images=by_file[q['file']]['images'][:2]+[reference_uri],
                    prompt=by_file[q['file']]['prompt']+addition,
                    reference_sha256=reference_hash)
               for q in queue if q['file'] not in done])
    (a.output/'reference-key.png').write_bytes(reference)
    m.atomic(a.output/'experiment.json',json.dumps(dict(baseline=str(a.baseline),
             queue=str(a.queue),reference_sha256=reference_hash,reasoning='off',parallel=a.parallel,
             context_first=a.context_first,
             change='Context/key/ROI single-call separate context_photo_description and roi_description' if a.context_first else 'Exact baseline target images and prompts plus reference key and usage instructions')))
context_prompt=m.module('context_prompt','ice-context-prompt.py') if a.context_first else None
deadline=time.monotonic()+a.hours*3600;window=m.RunningTemperature();active={};loaded=False;exhausted=False;errors=0
ice=('grease ice','nilas','thin fyi','icy bits','brash ice','ice floe')
pool=ThreadPoolExecutor(max_workers=a.parallel)
def unload():
    global loaded
    subprocess.run([m.LMS,'unload','qwen3.5-9b'],timeout=45,check=False);loaded=False
try:
    while (not exhausted or active) and time.monotonic()<deadline:
        cpu,gpu,power,mem=m.monitor.temperatures();avg=window.add(time.monotonic(),cpu)
        with (a.output/'telemetry.csv').open('a') as f:f.write(f'{datetime.now(timezone.utc).isoformat()},{cpu},{gpu},{power},{mem},{avg}\n')
        if avg>=95 or gpu>=83:
            unload()
            raise RuntimeError('Thermal cutoff: saved results retained; resume after cooling')
        if not loaded:
            if gpu>=78:
                time.sleep(5);continue
            subprocess.run([m.LMS,'load','qwen3.5-9b','--gpu','max','--context-length','16384','--parallel',str(a.parallel),'--identifier','qwen3.5-9b','--ttl','1800','-y'],check=True,timeout=120)
            loaded=True;m.limit_qwen_cpu()
        while len(active)<a.parallel and not exhausted and gpu<78:
            try:job=next(jobs)
            except StopIteration:exhausted=True;break
            if not a.reference_key:job['prompt']=prompt
            job.update(human_label=human.get(job['file'],[]),queue_metadata=metadata[job['file']])
            ordered_input=None
            if a.context_first:
                texts,ordered_input=context_prompt.build(by_file[job['file']]['prompt'],job['images'])
                job.update(prompt='\n\n'.join(texts),prompt_segments=texts,
                           input_image_order=['context','reference_key','roi'])
            future=pool.submit(m.request,job['images'],job['prompt'],True,ordered_input);active[future]=(job,time.monotonic())
            print('Started',job['id'],'in flight',len(active),flush=True)
        for future in list(active):
            if not future.done():continue
            job,started=active.pop(future)
            try:
                response=future.result();choice=response['choices'][0];text=choice['message']['content']
                audit=quality.assess(text)
                if a.context_first:
                    try:
                        parsed=json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
                        for field in ('context_photo_description','roi_description'):
                            if not isinstance(parsed.get(field),str) or not parsed[field].strip():
                                audit['quality_flags'].append('Missing '+field)
                    except (ValueError,AttributeError):
                        audit['quality_flags'].append('Cannot verify context/ROI descriptions')
                row=dict(**job,response=text,finish_reason=choice['finish_reason'],usage=response['usage'],model='qwen3.5-9b',reasoning='off',parallel=a.parallel,elapsed_s=time.monotonic()-started,utc=datetime.now(timezone.utc).isoformat(),**audit)
                rows.append(row);m.atomic(a.output/'results.json',json.dumps(rows));m.render(a.output,rows)
                print('Saved',job['id'],len(rows),'results',flush=True)
                if audit['quality_flags']:print('Flagged for review',job['id'],audit['quality_flags'],flush=True)
            except Exception as e:
                errors+=1;print('Failed',job['file'],repr(e),flush=True)
                with (a.output/'errors.jsonl').open('a') as f:f.write(json.dumps(dict(file=job['file'],error=str(e)))+'\n')
                if errors>=3:raise
        time.sleep(2)
finally:
    if loaded:unload()
    pool.shutdown(wait=False,cancel_futures=True)
    m.atomic(a.output/'status.json',json.dumps(dict(saved=len(rows),errors=errors,pending=len(queue)-len({r['file'] for r in rows}),ended_utc=datetime.now(timezone.utc).isoformat())))
