"""Sync unseen 2026 originals and atomically prioritize them without stopping jobs."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from PIL import Image

ROOT=Path('/home/cryomics/Downloads/amundsen-ice-gemma-2026')
SOURCE=Path('/mnt/ship/Data/Camera_360')
DEST=Path('/data/scratch/camera-originals')

def merged_queue(previous, additions):
    rows={r['file']:r for r in previous}
    rows.update({r['file']:r for r in additions})
    return sorted(rows.values(),key=lambda r:(r['capture_time'],r['file']),reverse=True)

def main():
    path=ROOT/'selection.json';previous=json.loads(path.read_text())
    known={r['file'] for r in previous['queue']};candidates=[]
    if not SOURCE.is_dir():raise RuntimeError('Ship share unavailable; local queue unchanged')
    for leg in sorted(SOURCE.glob('2026_LEG_*'),reverse=True):
        for day in sorted(leg.glob('2026????'),reverse=True):
            for source in sorted(day.glob('*/Camera360_*_cam_3.jpg'),reverse=True):
                file=source.relative_to(SOURCE).as_posix()
                if file in known:continue
                match=re.fullmatch(r'Camera360_(2026\d{10})_cam_3\.jpg',source.name)
                if not match:continue
                stat=source.stat()
                if time.time()-stat.st_mtime<60:continue
                candidates.append((match[1],file,stat.st_size,stat.st_mtime_ns))
    candidates.sort(reverse=True)
    added=0
    for start in range(0,len(candidates),20):
        if shutil.disk_usage(DEST).free<10*1024**3:raise RuntimeError('Less than 10 GiB free')
        batch=candidates[start:start+20]
        subprocess.run(['rsync','-rt','--from0','--files-from=-','--timeout=60','--partial-dir=.rsync-partial',str(SOURCE)+'/',str(DEST)+'/'],
            input=b''.join(file.encode()+b'\0' for _,file,_,_ in batch),check=True,timeout=180)
        additions=[]
        for capture,file,size,mtime in batch:
            local=DEST/file;before=local.stat();remote=(SOURCE/file).stat()
            if (remote.st_size,remote.st_mtime_ns)!=(size,mtime) or before.st_size!=size:continue
            try:
                with Image.open(local) as im:
                    if im.size!=(3648,2052):continue
                    im.load()
            except (OSError,ValueError):continue
            additions.append(dict(id=hashlib.sha256(file.encode()).hexdigest()[:20],file=file,
                capture_time=capture,leg=Path(file).parts[0],queue_source='live-new-arrivals'))
        # Re-read so an earlier successful batch is never lost.
        data=json.loads(path.read_text());data['queue']=merged_queue(data['queue'],additions)
        data['updated_utc']=datetime.now(timezone.utc).isoformat()
        temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)
        added+=len(additions);print(f'Published {added} new locally cached photos',flush=True)
    status=dict(checked_utc=datetime.now(timezone.utc).isoformat(),discovered=len(candidates),added=added,
                total=len(json.loads(path.read_text())['queue']))
    target=ROOT/'arrivals-status.json';temp=target.with_suffix('.tmp');temp.write_text(json.dumps(status,indent=2));temp.replace(target)
    print(json.dumps(status),flush=True)

if __name__=='__main__':main()
