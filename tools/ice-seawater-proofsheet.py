"""Make a numbered review sheet from saved real model ROI inputs."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps

root=Path('/home/cryomics/Downloads/amundsen-ice-seawater')
rows=json.loads((root/'review.json').read_text())
sheet=Image.new('RGB',(1600,240*((len(rows)+3)//4)),'#17212b');draw=ImageDraw.Draw(sheet)
for i,row in enumerate(rows):
    x=(i%4)*400;y=(i//4)*240
    with Image.open(row['images'][1]) as im:sheet.paste(ImageOps.contain(im,(396,198)),(x,y+35))
    draw.text((x+4,y+4),f'{i+1}  teacher water {row["teacher_water"]}% ice {row["teacher_ice"]}%',fill='#00ff66')
    draw.text((x+4,y+18),f'score {row["heldout_water_score"]:.3f} | {row["day"]}',fill='white')
sheet.save(root/'review.jpg',quality=90)
print(root/'review.jpg')
