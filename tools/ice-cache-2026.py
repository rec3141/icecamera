"""Copy the saved newest-first camera queue locally, then resume its consumers."""
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

ROOT=Path('/home/cryomics/Downloads/amundsen-ice-gemma-2026')
SOURCE=Path('/mnt/ship/Data/Camera_360')
DEST=Path('/data/scratch/camera-originals')

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    queue=json.loads((ROOT/'selection.json').read_text())['queue']
    for row in queue:
        p=Path(row['file'])
        if p.is_absolute() or '..' in p.parts or not p.parts[0].startswith('2026_LEG_'):
            raise ValueError('Unsafe queued path')
    def status(**value):
        path=ROOT/'local-cache-status.json';tmp=path.with_suffix('.tmp')
        tmp.write_text(json.dumps(dict(total=len(queue),local_root=str(DEST),**value),indent=2));tmp.replace(path)
    for start in range(0,len(queue),200):
        if shutil.disk_usage(DEST).free<10*1024**3:raise RuntimeError('Less than 10 GiB free')
        batch=queue[start:start+200]
        payload=b''.join(r['file'].encode()+b'\0' for r in batch)
        status(state='copying',checked=start)
        for attempt in range(5):
            try:
                subprocess.run(['rsync','-rt','--from0','--files-from=-','--timeout=60',
                    '--partial-dir=.rsync-partial',str(SOURCE)+'/',str(DEST)+'/'],
                    input=payload,check=True,timeout=600)
                break
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
                status(state='retrying share copy',checked=start,attempt=attempt+1)
                if attempt==4:raise
                time.sleep(30)
        print(f'Cached {min(start+200,len(queue))}/{len(queue)}',flush=True)
    # Preserve extracted feature rows when only the source root changed. rsync
    # preserves mtimes; require the same size and timestamp before relinking.
    dbpath=Path('/home/cryomics/Downloads/amundsen-ice-2025-2026-tsne/features.sqlite')
    migrated=0
    with sqlite3.connect(dbpath) as db:
        for file,fingerprint in db.execute("SELECT file,fingerprint FROM features WHERE file LIKE '2026_LEG_%' AND error IS NULL").fetchall():
            old=str(SOURCE/file);new=str(DEST/file)
            if old not in fingerprint:continue
            stat=(DEST/file).stat()
            if fingerprint.endswith(f':{stat.st_size}:{stat.st_mtime_ns}'):
                db.execute('UPDATE features SET fingerprint=? WHERE file=?',(fingerprint.replace(old,new),file));migrated+=1
    status(state='complete',checked=len(queue),feature_rows_relinked=migrated)
    subprocess.run(['systemctl','--user','start','ice-gemma-2026.service','ice-2026-projection.service'],check=True)
    subprocess.run(['systemctl','--user','start','ice-gemma-2026-watch.timer'],check=True)
    print('Local cache complete; processors resumed',flush=True)

if __name__=='__main__':main()
