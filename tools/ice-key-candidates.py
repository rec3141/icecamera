"""Build an unaltered-photo contact sheet to review ice-key candidates."""
import json
from pathlib import Path
import html
import numpy as np
import sys

wide = '--wide' in sys.argv

root=Path('/home/cryomics/Downloads/amundsen-ice-full-leg-600x300')
out=Path('/home/cryomics/Downloads/ice-reference-key');out.mkdir(exist_ok=True)
points=json.loads((root/'embedding.json').read_text())
labels=json.loads(Path('/home/cryomics/Downloads/amundsen-ice-qwen-k32-classes/labels-clean.json').read_text())
by_file={r['file']:set(r['labels']) for r in labels['labels']}
types=['grease ice','nilas','thin fyi','icy bits','brash ice','ice floe']
pcs=json.loads((root/'feature-pcs.json').read_text())
scores=np.array(pcs['scores'])
manifest=[]
for name in types:
    eligible=[i for i,p in enumerate(points) if by_file.get(p['file'])=={name}]
    center=np.median(scores[eligible],axis=0)
    ranked=sorted(eligible,key=lambda i:np.linalg.norm(scores[i]-center))
    if wide:
        ranked=list(np.random.default_rng(42).permutation(eligible))
    selected=[];used=set()
    for i in ranked:
        # Avoid adjacent frames: at most one candidate per UTC hour.
        hour=points[i]['file'][:11]
        if hour in used:continue
        selected.append(i);used.add(hour)
        if len(selected)==(24 if wide else 12):break
    print(name,len(eligible),'clean single-label candidates')
    cards=[]
    for i in selected:
        p=points[i];number=len(manifest)+1
        manifest.append(dict(number=number,type=name,**p))
        cards.append(f'<figure><img src="{(root/p["images"][1]).as_uri()}"><figcaption>{number} · {html.escape(p["file"])}</figcaption></figure>')
    page='<meta charset="utf-8"><style>body{background:#17212b;color:white;font:16px sans-serif}main{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}figure{margin:0}img{width:100%}figcaption{font:12px sans-serif}</style><h1>'+name+'</h1><main>'+''.join(cards)+'</main>'
    (out/(name.replace(' ','-')+('-wide' if wide else '')+'.html')).write_text(page)
(out/('candidates-wide.json' if wide else 'candidates.json')).write_text(json.dumps(manifest,indent=2))
