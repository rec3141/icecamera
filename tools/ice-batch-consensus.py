"""Compare independent tile arrangements without treating disagreement as water."""
from html import escape
import json
from pathlib import Path

def decisions(cases,response):
    items=response['items']
    if sorted(r['id'] for r in items)!=list(range(1,len(cases)+1)):raise ValueError('Missing or duplicate tile ids')
    if any(r['label'] not in {'water','ice','unclear'} for r in items):raise ValueError('Invalid label')
    return {cases[r['id']-1]['file']:r['label'] for r in items}

def compare(cases,a,b):
    if set(a)!=set(b):raise ValueError('Different photo sets')
    return [dict(file=r['file'],first=a[r['file']],reversed=b[r['file']],
                 route='water_candidate' if a[r['file']]==b[r['file']]=='water' else 'full_review') for r in cases]

if __name__=='__main__':
    root=Path('/home/cryomics/Downloads');pa=root/'ice-batch-triage';pb=root/'ice-batch-triage-reversed'
    ca=json.loads((pa/'cases.json').read_text());cb=json.loads((pb/'cases.json').read_text())
    a=json.loads((pa/'result.json').read_text());b=json.loads((pb/'result.json').read_text())
    rows=compare(ca,decisions(ca,a['response']),decisions(cb,b['response']))
    report=dict(rows=rows,total_seconds=a['elapsed_s']+b['elapsed_s'],seconds_per_photo=(a['elapsed_s']+b['elapsed_s'])/len(ca),
        water_candidates=sum(r['route']=='water_candidate' for r in rows),disagreements=sum(r['first']!=r['reversed'] for r in rows),
        live_enabled=False,note='Small, deliberately difficult development set, not a representative throughput or accuracy evaluation. Both arrangements must agree on water; all other cases go to full review.')
    (pa/'consensus.json').write_text(json.dumps(report,indent=2))
    (root/'batch-consensus.html').write_text('<!doctype html><meta charset="utf-8"><title>Two-order batch triage</title><style>body{font:17px system-ui;max-width:1100px;margin:auto}pre{white-space:pre-wrap}img{max-width:100%}</style><h1>Two-order Gemma batch triage</h1><p>The isolated-ice image missed in the first order was recognized as ice in the reversed order. Consensus keeps that image for full review. No live filtering changed.</p><img src="ice-batch-triage/sheet.jpg"><pre>'+escape(json.dumps(report,indent=2))+'</pre>')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
