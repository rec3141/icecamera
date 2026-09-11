"""Build an offline ice-profile design comparison from saved Gemma results."""
import json
import re
import sqlite3
from bisect import bisect_left
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from PIL import Image

def main():
    root=Path.home()/'Downloads'
    source=root/'amundsen-ice-gemma-2026'
    records=json.loads((source/'results.json').read_text())
    types=['grease ice','nilas','thin ice floe','icy bits','brash ice','thick ice floe']
    full_files={r['file'] for r in records}
    skips={r['file']:r for r in json.loads((source/'seawater-skips.json').read_text()) if r['file'] not in full_files}
    for r in skips.values():
        original=Path('/data/scratch/camera-originals')/r['file']
        records.append(dict(id=r['id'],file=r['file'],images=[str(original),r['image']],provenance=r.get('stage','local')+' seawater filter',response=json.dumps(dict(surface_percentages={t:0 for t in types},confidence='filter estimate'))))
    pending=[]
    for r in json.loads((source/'selection.json').read_text())['queue']:
        if r['file'] in full_files or r['file'] in skips:continue
        stamp=datetime.strptime(r['capture_time'],'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
        pending.append(int(stamp.timestamp()*1000))
    rows=[]
    navigation={}
    for path in Path('/data/underway_server/db').glob('2026_LEG_*.db'):
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
            columns=dict(db.execute('SELECT key,col FROM columns'))
            lat=columns.get('posmv — latitude (deg n)');lon=columns.get('posmv — longitude (deg e)')
            if not lat or not lon:continue
            if not all(re.fullmatch(r'c\d+',c) for c in (lat,lon)):continue
            fixes=db.execute(f'SELECT MIN(t),AVG({lat}),AVG({lon}) FROM obs WHERE {lat} BETWEEN -90 AND 90 AND {lon} BETWEEN -180 AND 180 GROUP BY CAST(t/60 AS INTEGER) ORDER BY MIN(t)').fetchall()
            navigation[path.stem]=([f[0] for f in fixes],fixes)
    slices=root/'ice-profile-slices';slices.mkdir(exist_ok=True)
    for r in records:
        match=re.search(r'Camera360_(\d{14})_',r['file'])
        if not match:continue
        try:
            response=r['response'];a=response.index('{');b=response.rindex('}')
            answer=json.loads(response[a:b+1]);surface=answer['surface_percentages']
            values=[float(surface.get(t,0)) for t in types]
            if any(v<0 or v>100 for v in values):continue
            stamp=datetime.strptime(match[1],'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc)
            times,fixes=navigation.get(r['file'].split('/')[0],([],[]));pos=None
            idx=bisect_left(times,stamp.timestamp())
            candidates=fixes[max(0,idx-1):idx+1]
            if candidates:
                nearest=min(candidates,key=lambda f:abs(f[0]-stamp.timestamp()))
                if abs(nearest[0]-stamp.timestamp())<=120:pos=list(nearest[1:])
            strip=slices/(r['id']+'.jpg')
            if not strip.exists():
                with Image.open(source/r['images'][1]) as im:
                    middle=im.width//2
                    im.crop((middle-2,0,middle+2,im.height)).resize((4,180)).convert('RGB').save(strip,quality=85)
            rows.append(dict(t=int(stamp.timestamp()*1000),ice=sum(values),v=values,
                slice=strip.as_uri(),
                pos=pos,
                provenance=r.get('provenance','Gemma full classification'),
                unknown=surface.get('unknown',0),confidence=answer.get('confidence','unknown'),
                image=(source/r['images'][0]).as_uri(),roi=(source/r['images'][1]).as_uri(),file=r['file']))
        except (ValueError,KeyError,TypeError,IndexError,OSError):continue
    rows.sort(key=lambda r:r['t'])
    tz=ZoneInfo('America/Toronto')
    dates=sorted({datetime.fromtimestamp(t/1000,tz).date() for t in [r['t'] for r in rows]+pending})
    windows={str(d):[int(datetime.combine(d,datetime.min.time(),tz).timestamp()*1000),int(datetime.combine(d+timedelta(days=1),datetime.min.time(),tz).timestamp()*1000)] for d in dates}
    payload=json.dumps(dict(rows=rows,types=types,windows=windows,pending=pending)).replace('<','\\u003c')
    template=Path(__file__).with_name('ice-profile-demo.html').read_text()
    target=root/'ice-profiles.html'
    target.write_text(template.replace('__DATA__',payload))
    print(f'{target}: {len(rows)} full classifications, {target.stat().st_size:,} bytes')

if __name__=='__main__':main()
