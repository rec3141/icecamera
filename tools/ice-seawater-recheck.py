"""Bounded ROI-only Gemma recheck of seawater-filter review cases; no live skips."""
import base64
from html import escape
import importlib.util
import json
from pathlib import Path
import time
import urllib.request

ROOT=Path('/home/cryomics/Downloads');OUT=ROOT/'amundsen-ice-seawater'
PROMPT='''Assess ONLY this image, a cropped sea-surface region. Is the ENTIRE visible
region confidently seawater without any ice? Inspect all four edges and corners
as well as the centre. Any visible ice fragment, however small, means no.
Foam and ordinary wave crests are water, but do not dismiss a coherent ice edge
as foam. Smooth surfaces may be grease ice or nilas: if ambiguous, do not claim
clear seawater. Wet-lens blur, fog, darkness or severe glare can make the absence
of ice unassessable. Do not estimate ice types or percentages.
Return only JSON with fields:
"decision": "clear_seawater" | "possible_or_visible_ice" | "unassessable",
"ice_locations": a short list of image positions, empty if none,
"visibility_issue": short text or "none",
"evidence": one short sentence of observable visual evidence.
This is conservative triage, not scientific verification.'''

def main():
    spec=importlib.util.spec_from_file_location('monitor',Path(__file__).resolve().with_name('ice-monitored-review.py'))
    monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    candidates=json.loads((OUT/'review.json').read_text())
    # Verified ice-at-edge contradiction first, then the wet-lens example.
    candidates.sort(key=lambda r:(r['id']!='dd1bc6595c7a0e66b25a',r['id']!='23d28f77a7b1e986de30'))
    path=OUT/'binary-recheck.json';rows=json.loads(path.read_text()) if path.exists() else []
    done={r['file'] for r in rows};deadline=time.monotonic()+3600
    def render():
        page='<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>Seawater binary recheck</title><style>body{font:16px system-ui;max-width:1000px;margin:2rem auto}img{max-width:100%;max-height:500px}pre{white-space:pre-wrap}</style><h1>ROI-only seawater recheck</h1><p>No live images are skipped. Gemma sees only the ROI, with no key or prior labels. '+str(len(rows))+'/'+str(len(candidates))+' complete.</p><pre>'+escape(PROMPT)+'</pre>'
        for r in rows:page+='<h2>'+escape(r['file'])+'</h2><img loading="lazy" src="'+Path(r['image']).as_uri()+'"><pre>'+escape(r['response'])+'</pre>'
        tmp=ROOT/'seawater-recheck.html.tmp';tmp.write_text(page);tmp.replace(ROOT/'seawater-recheck.html')
    render()
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for r in candidates:
        if r['file'] in done:continue
        for attempt in range(3):
            if time.monotonic()>deadline:raise TimeoutError('One-hour pilot bound reached')
            cpu,gpu,*_=monitor.temperatures()
            while cpu>=85 or gpu>=78:
                if time.monotonic()>deadline:raise TimeoutError('Pilot cooling deadline')
                time.sleep(5);cpu,gpu,*_=monitor.temperatures()
            image=Path(r['images'][1]);uri='data:image/jpeg;base64,'+base64.b64encode(image.read_bytes()).decode()
            payload=dict(model='gemma-camera',temperature=0,seed=42,max_tokens=450,stream=False,
                chat_template_kwargs=dict(enable_thinking=False),messages=[dict(role='user',content=[dict(type='text',text=PROMPT),dict(type='image_url',image_url=dict(url=uri))])])
            req=urllib.request.Request('http://127.0.0.1:18043/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            begin=time.monotonic()
            try:
                with opener.open(req,timeout=300) as response:raw=json.load(response)
                choice=raw['choices'][0];text=choice['message']['content']
                if choice['finish_reason']!='stop':raise ValueError('Incomplete response')
                parsed=json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
                if parsed.get('decision') not in {'clear_seawater','possible_or_visible_ice','unassessable'}:raise ValueError('Invalid decision')
                rows.append(dict(file=r['file'],image=str(image),response=text,decision=parsed['decision'],elapsed_s=time.monotonic()-begin,prompt=PROMPT,model='gemma-camera-roi-only'))
                tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(rows,indent=2));tmp.replace(path);done.add(r['file']);render()
                print(len(rows),r['file'],parsed['decision'],flush=True);break
            except (OSError,ValueError,KeyError) as e:
                print('Retry',repr(e),flush=True)
                if attempt==2:raise
                time.sleep(5)

if __name__=='__main__':main()
