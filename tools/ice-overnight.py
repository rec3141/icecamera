"""Bounded, local-only exploratory review; labels never enter model prompts."""
import argparse
import concurrent.futures
from collections import deque
import html
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from PIL import Image, ImageDraw, ImageOps


def module(name, file):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(file))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


rot = module('rot', 'ice-rotated-preview.py')
monitor = module('monitor', 'ice-monitored-review.py')
MODEL = 'ice-qwen35'
LMS = str(Path.home()/'.lmstudio/bin/lms')
PROMPT = ('Exploratory sea ice interpretation, not navigation or scientific measurement. '
          'Do not assume dark means water or bright means ice. Distinguish glare, foam, '
          'open water, dark new/nilas/grease ice, broken ice/rubble, and thicker snow-covered ice. '
          'Estimate mutually exclusive percentages of image-plane sea-surface area for these '
          'categories, plus unknown; total 100%. Give total ice percentage, confidence and '
          'alternative explanations. Unknown is acceptable. Answer under 180 words. ')

STRUCTURED_PROMPT = '''Return one JSON object only, with this schema:
{"visibility":"clear|degraded|unusable|uncertain", "visibility_causes":["fog","wet_lens","blur","darkness","glare"], "confidence":"high|moderate|low", "percentages":{"open_water":0,"thin_new_ice":0,"broken_ice":0,"consolidated_ice":0,"unknown":0}, "explanation":"short explanation"}.
First assess visibility: look for fog, droplets/water smears on the lens, defocus/motion blur, darkness and glare. Do not interpret missing texture caused by optical obstruction as open water. Cause can be uncertain; visibility_causes may be empty. Degraded or uncertain visibility is valid even if some ice can be seen.
Percentages describe image-plane area of the sea surface INSIDE the orange quadrilateral in image 1, shown extracted/rotated in image 2. Ignore everything outside it. The five mutually exclusive percentages must sum to 100. Thin_new_ice includes relatively continuous dark nilas/grease/frazil/new ice, including frost flowers. Broken_ice includes distinct floes/rubble/fragments even if snow-covered; prioritize this category over snow cover. Consolidated_ice means continuous established/snow-covered ice not assigned to the other two. Open_water includes identifiable water with glare/foam; unresolvable area is unknown. Do not assume dark means water or white means ice. If unusable, unknown=100. These are rough visual classes, not physical thickness/age determinations. Do not provide navigation/safety advice. No human labels are supplied.'''


def request(images, prompt, reasoning_off=False, ordered_input=None):
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    if reasoning_off:
        payload=dict(model='qwen3.5-9b',temperature=0,max_output_tokens=6000,reasoning='off',store=False,
                     input=ordered_input if ordered_input is not None else [dict(type='text',content=prompt)]+[dict(type='image',data_url=i) for i in images])
        req=urllib.request.Request('http://127.0.0.1:1234/api/v1/chat',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
        with client.open(req,timeout=600) as response:raw=json.load(response)
        stats=raw.get('stats',{})
        if stats.get('reasoning_output_tokens',0):raise ValueError('Reasoning-off request generated reasoning tokens')
        text='\n'.join(item['content'] for item in raw.get('output',[]) if item['type']=='message')
        return dict(choices=[dict(message=dict(content=text),finish_reason='length' if stats.get('total_output_tokens',0)>=6000 else 'stop')],usage=stats,native_response=raw)
    payload = dict(model=MODEL, temperature=0, max_tokens=6000, messages=[dict(
        role='user', content=[dict(type='text', text=prompt)]+[
            dict(type='image_url', image_url=dict(url=i)) for i in images])])
    req = urllib.request.Request('http://127.0.0.1:1234/v1/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
    with client.open(req, timeout=600) as response:
        return json.load(response)


class RunningTemperature:
    """Time-weighted trailing mean; startup uses observed time, never zero padding."""
    def __init__(self, seconds=120):
        self.seconds=seconds;self.samples=deque()

    def add(self, now, value):
        self.samples.append((now,value))
        cutoff=now-self.seconds
        while len(self.samples)>1 and self.samples[1][0]<=cutoff:self.samples.popleft()
        start=max(cutoff,self.samples[0][0]);total=0
        for (t,v),(end,_) in zip(self.samples,list(self.samples)[1:]):
            total+=v*max(0,end-max(t,start))
        return total/(now-start) if now>start else value


def limit_qwen_cpu(proc_root=None):
    """Linux-only: constrain the exact LM Studio Qwen backend, not Gemma."""
    if not hasattr(os,'sched_setaffinity'): return
    cpus=set(sorted(os.sched_getaffinity(0))[:4])
    matched=False
    for proc in (proc_root or Path('/proc')).glob('[0-9]*'):
        try:
            cmd=(proc/'cmdline').read_bytes().split(b'\0')
            if not cmd or b'/.lmstudio/extensions/backends/' not in cmd[0]: continue
            if b'/data/scratch/models/lmstudio-community/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q4_K_M.gguf' not in cmd: continue
            for task in (proc/'task').iterdir():
                try:
                    os.sched_setaffinity(int(task.name),cpus);matched=True
                except (ProcessLookupError,FileNotFoundError):continue
        except (ProcessLookupError,FileNotFoundError,PermissionError): continue
    if not matched: raise RuntimeError('Could not identify Qwen backend for CPU limit')


def atomic(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp'); tmp.write_text(value); tmp.replace(path)


def render(out, rows):
    if (out/'embedding.json').exists():
        module('region_explorer','ice-region-explorer.py').render(out, rows)
    elif (out/'linked-explorers.json').exists():
        explorer=module('region_explorer','ice-region-explorer.py')
        for folder in json.loads((out/'linked-explorers.json').read_text()):
            explorer.render(Path(folder),rows,follow_links=False)
    page = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Overnight ice experiments</title><style>body{font:16px system-ui;max-width:1200px;margin:auto;padding:20px}img{max-width:48%;max-height:500px}pre{white-space:pre-wrap}article{border-top:1px solid;padding:20px 0}</style><h1>Blind local Qwen experiments</h1><p>Unvalidated exploratory predictions. Human labels were withheld from prompts. Cluster descriptions are weak labels, not individual coupon truth. Region predictions use image-plane area, not perspective-corrected concentration. Refresh for new results.</p>'
    page += f'<p>{len(rows)} completed requests.</p>'
    baseline_path=out/'comparison-baseline.json'
    if baseline_path.exists():
        baseline={r['file']:r for r in json.loads(baseline_path.read_text())}
        pairs=[]
        for row in rows:
            old=baseline.get(row['file'])
            if not old:continue
            record=dict(file=row['file'],thinking_seconds=old['elapsed_s'],off_seconds=row['elapsed_s'],reasoning_tokens=row.get('usage',{}).get('reasoning_output_tokens'))
            try:
                parse=lambda r:json.loads(r['response'].strip().removeprefix('```json').removesuffix('```').strip())
                left,right=parse(old),parse(row)
                record['percentage_differences']={group:{name:right[group][name]-value for name,value in left[group].items()} for group in ('surface_percentages','artifact_percentages')}
                ice=('grease ice','nilas','thin fyi','icy bits','brash ice','ice floe')
                record['thinking_ice_sum']=sum(left['surface_percentages'][name] for name in ice)
                record['off_ice_sum']=sum(right['surface_percentages'][name] for name in ice)
                record['reported_totals']=[left.get('total_ice_percent'),right.get('total_ice_percent')]
                record['total_ice_difference']=record['off_ice_sum']-record['thinking_ice_sum']
            except (ValueError,KeyError,TypeError):record['comparison_error']='Invalid or incompatible JSON'
            pairs.append(record)
        atomic(out/'comparison.json',json.dumps(pairs,indent=2))
        page+='<h2>Thinking on/off paired comparison</h2><p>Signed differences are thinking-off minus thinking-on, in percentage points; agreement is not accuracy.</p><pre>'+html.escape(json.dumps(pairs,indent=2))+'</pre>'
    for r in rows:
        page += '<article><h2>'+html.escape(r['id'])+'</h2>'
        page += ''.join(f'<img src="{x}">' for x in r['images'])
        page += '<p>Human comparison (not supplied to model): '+html.escape(str(r.get('human_label','No matching label')))+' </p>'
        if r.get('quality_flags'):page+='<p><strong>Review flags: '+html.escape(', '.join(r['quality_flags']))+'</strong> · surface sum: '+html.escape(str(r.get('surface_sum')))+'</p>'
        page += '<pre>'+html.escape(r['response'])+'</pre><p>Finish: '+html.escape(str(r['finish_reason']))+'</p></article>'
    atomic(out/'index.html', page)


def jobs(data, labels, source):
    # Interleave context-region requests and exact original labelled footprints.
    groups = {}
    for tile in labels['tiles']:
        groups.setdefault(tile['cluster'], []).append(tile)
    rng = random.Random(42)
    for tiles in groups.values(): rng.shuffle(tiles)
    coupons = [groups[g][i] for i in range(24) for g in sorted(groups) if i < len(groups[g])]
    for i in range(max(len(data['scenes']), len(coupons))):
        if i < len(data['scenes']):
            s = data['scenes'][i]
            with Image.open(source/s['file']) as im:
                im = im.convert('RGB')
                affine, polygon, _ = rot.geometry(im.size, angle=-30)
                crop = im.transform((1200,600), Image.Transform.AFFINE, affine, Image.Resampling.BICUBIC)
                full = im.copy(); ImageDraw.Draw(full).line(polygon+[polygon[0]], fill='orange', width=12)
                images = [rot.uri(ImageOps.contain(full,(1400,1400))), rot.uri(crop)]
            yield dict(id=f"region-scene-{s['scene']}", file=s['file'], polygon=polygon,
                       images=images, prompt=PROMPT+'Image 1 is context: assess ONLY inside the orange quadrilateral. Image 2 is that exact region extracted and rotated. Exclude the rest of the photo. Avoid double-counting frost flowers on new ice.')
        if i < len(coupons):
            t = coupons[i]
            with Image.open(source/t['file']) as im:
                images = [rot.uri(im.convert('RGB').crop(t['box']))]
            names = labels['group_names']
            label = names[str(t['cluster'])] if isinstance(names,dict) else names[t['cluster']]
            yield dict(id=f"coupon-{t['id']}", file=t['file'], box=t['box'], cluster=t['cluster'],
                       human_label=dict(cluster_description=label, individual=t.get('reviewed_label')),
                       images=images, prompt=PROMPT+'This is an exact 100x100 pixel coupon. If scale or lack of context prevents classification, say so.')


def main():
    global MODEL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hours', type=float, default=4)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--regions-only', action='store_true')
    p.add_argument('--structured', action='store_true')
    p.add_argument('--queue',type=Path,help='Blind region review queue, matched to source filenames')
    p.add_argument('--class-labels',type=Path,help='Region labels: supply category vocabulary, withhold per-photo labels')
    p.add_argument('--reasoning-off',action='store_true')
    p.add_argument('--replay',type=Path,help='Replay exact saved images and prompts, for paired comparisons')
    a = p.parse_args()
    if a.reasoning_off:MODEL='qwen3.5-9b'
    if not 0 < a.hours <= 4: p.error('hours must be >0 and <=4')
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    data = json.loads((Path.home()/'Downloads/amundsen-ice-texture-brightness/tiles.json').read_text())
    label_path = Path.home()/'Downloads/ice-texture-labels(1).json'
    labels = json.loads(label_path.read_text())
    queue_by_file={}
    if a.queue:
        if not a.regions_only or not a.structured:raise ValueError('Queues require structured region mode')
        queued=json.loads(a.queue.read_text())['queue']
        queue_by_file={q['file']:q for q in queued}
        if len(queue_by_file)!=len(queued):raise ValueError('Duplicate photos in queue')
        data={'scenes':[{'scene':f'cluster-{q["cluster"]}-{i}','file':q['file']} for i,q in enumerate(queued)]}
        labels={'tiles':[]}
    source = Path('/media/cryomics/T7 Shield/Amundsen/Camera_360/2025_LEG_04')
    atomic(out/'human-labels-snapshot.json', json.dumps(labels))
    existing=out/'results.json'
    rows=json.loads(existing.read_text()) if existing.exists() else []
    expected=STRUCTURED_PROMPT if a.structured else None
    comparisons={}
    if a.class_labels:
        if not a.queue:raise ValueError('Class comparison requires a structured region queue')
        classes=json.loads(a.class_labels.read_text())
        comparisons={r['file']:r['labels'] for r in classes['labels']}
        names=classes['labelOrder']
        expected='''Return one JSON object only. Exploratory visual assessment, not navigation or scientific measurement. Image 1 shows context and an orange quadrilateral; image 2 is that exact extracted region. Assess ONLY that region. Per-photo human labels are withheld.
Estimate area percentages (0–100) for these surface classes: whitecap (foam on open water), small waves (rippled/wavy open water), smooth water, grease ice, nilas, thin fyi (visually thin first-year-looking sheet ice, not a measured age/thickness), icy bits (sparse small ice fragments), brash ice (dense broken rubble), ice floe (distinct larger pieces). Allocate each patch only once; surface classes plus unknown must sum to 100. Prioritize brash/icy bits/floe over underlying ice age categories. Do not assume dark means water or white means ice.
Separately estimate the fraction of the region affected by blurry (including water/droplets/smears on lens), fog, reflection/glare, and night/darkness. These are overlapping optical conditions, NOT surface types; do not include them in the 100% surface budget. Each is independently 0–100. If surface unresolvable use unknown, not smooth water. Assess visibility first.
Schema: {"visibility":"clear|degraded|unusable|uncertain","confidence":"high|moderate|low","surface_percentages":{"whitecap":0,"small waves":0,"smooth water":0,"grease ice":0,"nilas":0,"thin fyi":0,"icy bits":0,"brash ice":0,"ice floe":0,"unknown":0},"artifact_percentages":{"blurry":0,"fog":0,"reflection":0,"night":0},"total_ice_percent":0,"explanation":"brief, include ambiguity"}. Total ice is the sum of the six ice classes. Available human vocabulary: '''+json.dumps(names)
        atomic(out/'human-labels-snapshot.json',json.dumps(classes))
    if rows and (not a.structured or any(r['prompt']!=expected for r in rows)):
        raise ValueError('Output already has incompatible results; use a new directory')
    done={r['id'] for r in rows if r.get('finish_reason')=='stop' and r.get('response','').strip()}
    render(out, rows)
    deadline = time.monotonic()+a.hours*3600
    loaded = False; trips = 0; errors = 0; force_cool=False
    cpu_window=RunningTemperature();cpu_average=0
    def unload():
        nonlocal loaded
        subprocess.run([LMS,'unload',MODEL], timeout=45, check=False); loaded=False
    def sample():
        nonlocal cpu_average
        values = monitor.temperatures()
        cpu_average=cpu_window.add(time.monotonic(),values[0])
        with (out/'telemetry.csv').open('a') as f:
            f.write(datetime.now(timezone.utc).isoformat()+','+','.join(map(str,values))+f',{cpu_average}\n')
        return values
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        replay=json.loads(a.replay.read_text()) if a.replay else None
        if replay is not None:
            atomic(out/'comparison-baseline.json',json.dumps(replay))
            replay=[{k:r[k] for k in ('id','file','polygon','images','prompt','human_label','queue_metadata') if k in r} for r in replay if r.get('finish_reason')=='stop' and r.get('response','').strip()]
        for job in (replay if replay is not None else jobs(data, labels, source)):
            if a.queue:job['queue_metadata']=queue_by_file[job['file']]
            if a.regions_only and not job['id'].startswith('region-'): continue
            if a.structured:
                if not a.regions_only: raise ValueError('Structured mode requires --regions-only')
                if not a.replay:job['prompt']=expected
                if a.class_labels:job['human_label']=comparisons.get(job['file'],[])
            if job['id'] in done: continue
            if a.prepare_only:
                atomic(out/'prepared-example.json', json.dumps(job)); return
            if time.monotonic() >= deadline: break
            cpu,gpu,*_ = sample()
            if force_cool or cpu_average >= 95 or gpu >= 78:
                if loaded: unload()
                cooling_start = time.monotonic()
                while cpu > 75 or gpu > 65:
                    if time.monotonic() >= deadline or time.monotonic()-cooling_start > 900:
                        raise RuntimeError('Cooling timeout or run deadline')
                    time.sleep(15); cpu,gpu,*_ = sample()
                force_cool=False
            if not loaded:
                subprocess.run([LMS,'load','qwen3.5-9b','--gpu','max','--context-length','16384',
                    '--parallel','1','--identifier',MODEL,'--ttl','1800','-y'],check=True,timeout=120)
                loaded=True
                limit_qwen_cpu()
            started=time.monotonic()
            print('Starting',job['id'],'reasoning', 'off' if a.reasoning_off else 'default',flush=True)
            future=pool.submit(request,job['images'],job['prompt'],a.reasoning_off)
            interrupted=False
            while not future.done():
                cpu,gpu,*_=sample()
                if cpu_average >= 95 or gpu >= 83 or time.monotonic() >= deadline:
                    unload(); interrupted=True; trips+=1; force_cool=True
                    print('Interrupted for temperature/deadline; CPU instantaneous/120s mean/GPU',cpu,cpu_average,gpu,flush=True)
                    break
                time.sleep(3)
            try:
                response=future.result(timeout=45 if interrupted else 5)
                if interrupted:
                    if trips >= 3: raise RuntimeError('Three thermal interruptions; stopping')
                    continue
                choice=response['choices'][0]
                rows.append(dict(**job,response=choice['message']['content'] or '',
                    finish_reason=choice.get('finish_reason'),usage=response.get('usage'),
                    model=MODEL,max_tokens=6000,reasoning='off' if a.reasoning_off else 'default',elapsed_s=time.monotonic()-started,
                    utc=datetime.now(timezone.utc).isoformat()))
                atomic(out/'results.json',json.dumps(rows));render(out,rows)
                print(job['id'],len(rows),'completed',flush=True)
            except Exception as e:
                errors+=1;print(type(e).__name__,str(e),flush=True)
                if not future.done() or errors>=3 or trips>=3: raise
                if loaded: unload()
            time.sleep(15)
    finally:
        unload()
        pool.shutdown(wait=False,cancel_futures=True)
        atomic(out/'status.json',json.dumps(dict(completed=len(rows),ended_utc=datetime.now(timezone.utc).isoformat())))


if __name__=='__main__': main()
