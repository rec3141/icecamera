"""Sixteen-ROI semantic triage speed pilot; no live decisions changed."""
import base64
import argparse
from html import escape
import json
from pathlib import Path
import time
import urllib.request
from PIL import Image,ImageOps,ImageDraw,ImageFont

parser=argparse.ArgumentParser();parser.add_argument('--reverse',action='store_true');args=parser.parse_args()
root=Path('/home/cryomics/Downloads');out=root/('ice-batch-triage-reversed' if args.reverse else 'ice-batch-triage');out.mkdir(exist_ok=True)
patch=json.loads((root/'ice-patch-similarity/review.json').read_text())
native=json.loads((root/'ice-native-triage/review.json').read_text())
cases=[]
for i in [1,2,7,8,13,16,17,19]:cases.append(dict(file=patch[i-1]['file'],image=patch[i-1]['image']))
for i in [1,4,10,15,18,22,28,31]:cases.append(dict(file=native[i-1]['file'],image=native[i-1]['image']))
if args.reverse:cases.reverse()
sheet=Image.new('RGB',(1600,920),'#17212b');draw=ImageDraw.Draw(sheet)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
for i,r in enumerate(cases):
    x=i%4*400;y=i//4*230
    with Image.open(r['image']) as im:sheet.paste(ImageOps.contain(im,(396,198)),(x,y+30))
    draw.text((x+4,y+2),str(i+1),font=font,fill='#00ff66')
path=out/'sheet.jpg';sheet.save(path,quality=92);(out/'cases.json').write_text(json.dumps(cases,indent=2))
prompt='''This is a numbered contact sheet of 16 separate sea-surface ROI photos.
For EACH numbered image independently, classify it as "water", "ice", or "unclear".
Water includes waves and ordinary foam. Ice includes any visible ice fragments or
coherent ice surface. Use unclear for ship/deck, severe wet-lens blur, fog, darkness,
or an ambiguous smooth surface. Mild blur need not obscure otherwise clear water.
Inspect the entire tile including its edges. Do not let neighbouring tiles affect
the label. Return only JSON: {"items":[{"id":1,"label":"water"}, ...]}.
Include all 16 ids exactly once. No percentages and no explanation.'''
payload=dict(model='gemma-camera',messages=[dict(role='user',content=[dict(type='text',text=prompt),dict(type='image_url',image_url=dict(url='data:image/jpeg;base64,'+base64.b64encode(path.read_bytes()).decode()))])],temperature=0,seed=42,max_tokens=1000,stream=False,chat_template_kwargs=dict(enable_thinking=False))
request=urllib.request.Request('http://127.0.0.1:18043/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
start=time.monotonic()
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=300) as response:raw=json.load(response)
elapsed=time.monotonic()-start;choice=raw['choices'][0];text=choice['message']['content']
data=json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
assert choice['finish_reason']=='stop' and sorted(x['id'] for x in data['items'])==list(range(1,17))
assert all(x['label'] in {'water','ice','unclear'} for x in data['items'])
result=dict(elapsed_s=elapsed,seconds_per_photo=elapsed/16,response=data,prompt=prompt,usage=raw.get('usage'),live_enabled=False)
(out/'result.json').write_text(json.dumps(result,indent=2))
(root/('batch-triage-reversed.html' if args.reverse else 'batch-triage.html')).write_text('<!doctype html><meta charset="utf-8"><title>Batch triage pilot</title><style>body{font:17px system-ui;max-width:1200px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>16-image Gemma triage pilot</h1><p>Shadow experiment. End-to-end timing includes shared-server queue wait. No live photos skipped.</p><img src="'+out.name+'/sheet.jpg"><pre>'+escape(json.dumps(result,indent=2))+'</pre>')
print(json.dumps(result),flush=True)
