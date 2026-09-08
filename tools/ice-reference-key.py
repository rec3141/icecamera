"""Assemble a portable HTML reference figure from visually reviewed real crops.

No model-generated images, relabeling, crop shifts, or contrast adjustments.
Render the HTML with Chromium at 1464 x 1532 to export the PNG.
"""
import base64
import html
import json
from pathlib import Path

ROOT = Path('/home/cryomics/Downloads/amundsen-ice-full-leg-600x300')
OUT = Path('/home/cryomics/Downloads/ice-reference-key')
SETS = {
    'median': json.loads((OUT / 'candidates.json').read_text()),
    'wide': json.loads((OUT / 'candidates-wide.json').read_text()),
}
# Selected after viewing the candidate contact sheets. All are single-label
# human annotations, but appearance alone does not establish ice age/thickness.
CHOICES = [
    ('grease ice', [('median', 2), ('wide', 1), ('wide', 13)]),
    ('nilas', [('median', 19), ('median', 22), ('median', 14)]),
    ('thin fyi', [('median', 28), ('median', 33), ('median', 36)]),
    ('icy bits', [('wide', 67), ('wide', 69), ('wide', 79)]),
    ('brash ice', [('wide', 83), ('wide', 92), ('wide', 99)]),
    ('ice floe', [('median', 61), ('median', 63), ('median', 69)]),
]
cards, manifest = [], []
for row, (label, choices) in enumerate(CHOICES, 1):
    for col, (pool, number) in enumerate(choices, 1):
        point = next(p for p in SETS[pool] if p['number'] == number)
        assert point['type'] == label
        crop = ROOT / point['images'][1]
        encoded = base64.b64encode(crop.read_bytes()).decode()
        manifest.append(dict(row=row, column=col, label=label,
                             id=point['id'], file=point['file'],
                             crop=str(crop), source_preview=str(ROOT / point['images'][0])))
        cards.append(f'<figure title="{html.escape(point["file"])}">'
                     f'<img alt="{html.escape(label)} example {col}" '
                     f'src="data:image/jpeg;base64,{encoded}">'
                     f'<figcaption>{html.escape(label)}</figcaption></figure>')
assert len({p['id'] for p in manifest}) == 18
page = '''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Amundsen ice reference key</title><style>
*{box-sizing:border-box}body{margin:0;padding:8px;background:#111820;color:#edf4f8;
width:1464px;font:16px Arial,sans-serif}header{height:56px;padding:2px 4px}
h1{font-size:23px;margin:0 0 5px}p{margin:0;color:#b8c6d2;font-size:14px}
main{display:grid;grid-template-columns:repeat(3,480px);gap:4px}
figure{margin:0;position:relative;width:480px;height:240px}
img{display:block;width:480px;height:240px}
figcaption{position:absolute;top:6px;left:8px;color:#00ff6a;font-size:23px;
font-weight:800;text-shadow:-1px -1px 2px #000,1px -1px 2px #000,
-1px 1px 2px #000,1px 1px 2px #000,0 0 5px #000}
</style><header><h1>Amundsen · ice reference key</h1>
<p>2025 leg 4 · three examples per human-label category · review draft, not validated ice age/thickness</p>
</header><main>''' + ''.join(cards) + '</main></html>'
(OUT / 'ice-reference-key.html').write_text(page)
(OUT / 'selected-examples.json').write_text(json.dumps({
    'status': 'visually reviewed draft; supervisor approval recommended before model use',
    'notes': [
        'Labels come from the existing human single-label annotations, not Qwen.',
        'Grease ice is the least visually conclusive category; review against smooth water.',
        'Thin fyi is the project appearance label, not independently established age/thickness.',
        'Icy bits and brash ice may overlap; selected examples emphasize sparse pieces versus rubble.',
        'Whole crops retain their water/background; labels are category examples, not 100% coverage claims.',
        'Existing approved cutout geometry and colors are unchanged.',
    ], 'examples': manifest,
}, indent=2))
print(OUT / 'ice-reference-key.html')
