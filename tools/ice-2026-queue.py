"""Snapshot available 2026 camera-3 originals in reverse capture-time order."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import argparse
import subprocess
from PIL import Image

root=Path('/mnt/ship/Data/Camera_360')
out=Path('/home/cryomics/Downloads/amundsen-ice-gemma-2026')
parser=argparse.ArgumentParser();parser.add_argument('--refresh',action='store_true');args=parser.parse_args()
path=out/'selection.json'
previous=json.loads(path.read_text()) if path.exists() else None
if previous and not args.refresh:raise RuntimeError('Existing selection preserved; use --refresh')
known={r['file']:r for r in previous['queue']} if previous else {}
queue=[]
for leg in sorted(root.glob('2026_LEG_*')):
    for day in leg.glob('2026????'):
        for path in day.glob('*/Camera360_*_cam_3.jpg'):
            match=re.fullmatch(r'Camera360_(2026\d{10})_cam_3\.jpg',path.name)
            if not match:continue
            file=path.relative_to(root).as_posix()
            if file not in known and time.time()-path.stat().st_mtime<60:continue
            queue.append(dict(id=hashlib.sha256(file.encode()).hexdigest()[:20],file=file,
                              capture_time=match[1],leg=leg.name,queue_source='2026-reverse-chronological'))
discovered={r['file']:r for r in queue}
added=len(discovered.keys()-known.keys())
queue=list((known|discovered).values())
queue.sort(key=lambda r:(r['capture_time'],r['file']),reverse=True)
assert queue and len(queue)==len({r['file'] for r in queue})
with Image.open(root/queue[0]['file']) as image:
    assert image.size==(3648,2052),image.size
out.mkdir(exist_ok=True)
path=out/'selection.json'
if args.refresh and added:
    subprocess.run(['systemctl','--user','stop','ice-gemma-2026-watch.timer','ice-gemma-2026-watch.service','ice-gemma-2026.service','ice-2026-projection.service'],check=True)
    backup=out/('selection-before-refresh-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
    backup.write_text(json.dumps(previous,indent=2))
if not previous or added:
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(dict(queue=queue,created_utc=datetime.now(timezone.utc).isoformat(),
        order='Reverse filename capture timestamp; retain previous queued photos; new files must be at least 60 seconds old'),indent=2));temp.replace(path)
if args.refresh and added:subprocess.run(['systemctl','--user','start','ice-cache-2026.service'],check=True)
print(json.dumps(dict(total=len(queue),added=added,newest=queue[0]['file'],oldest=queue[-1]['file'])),flush=True)
