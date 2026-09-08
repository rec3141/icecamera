"""Review threshold candidates across every represented 2026 day."""
import json
import sys
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dashboard.camera_features import crop_region

root=Path('/home/cryomics/Downloads/amundsen-ice-seawater')
rows=json.loads((root/'v2-scores.json').read_text());groups={}
for r in rows:
    if r['file'].startswith('2026_') and r['score']>=.995:groups.setdefault(r['file'].split('/')[1],[]).append(r)
selected=[min(group,key=lambda r:r['score']) for day,group in sorted(groups.items())]
sheet=Image.new('RGB',(1600,230*((len(selected)+3)//4)),'#17212b');draw=ImageDraw.Draw(sheet)
(root/'v2-review').mkdir(exist_ok=True)
for i,r in enumerate(selected):
    with Image.open(Path('/data/scratch/camera-originals')/r['file']) as im:crop=crop_region(im)
    path=root/'v2-review'/f'{i+1}.jpg';crop.save(path,quality=92);r['roi']=str(path)
    x=i%4*400;y=i//4*230;sheet.paste(ImageOps.contain(crop,(396,198)),(x,y+28))
    draw.text((x+4,y+4),f'{i+1} {r["file"].split("/")[1]} score {r["score"]:.4f}',fill='#00ff66')
(root/'v2-review.json').write_text(json.dumps(selected,indent=2));sheet.save(root/'v2-review.jpg',quality=90)
print(len(selected),'day-diverse threshold candidates')
