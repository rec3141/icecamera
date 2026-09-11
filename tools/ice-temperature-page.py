"""Sample CPU/GPU temperature and render a self-contained rolling history page."""
import argparse
import csv
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import sqlite3
import time


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(a.output/'temperatures.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS samples (utc REAL PRIMARY KEY,cpu REAL,gpu REAL)')
    db.execute('CREATE TABLE IF NOT EXISTS imported (file TEXT PRIMARY KEY)')
    for path in (Path.home()/'Downloads').glob('amundsen-ice-*/telemetry.csv'):
        if db.execute('SELECT 1 FROM imported WHERE file=?',(str(path),)).fetchone():continue
        with path.open() as f:
            for row in csv.reader(f):
                try:
                    stamp=datetime.fromisoformat(row[0]).timestamp();cpu=float(row[1]);gpu=float(row[2])
                    db.execute('INSERT OR IGNORE INTO samples VALUES (?,?,?)',(stamp,cpu,gpu))
                except (ValueError,IndexError):continue
        db.execute('INSERT INTO imported VALUES (?)',(str(path),))
    spec=importlib.util.spec_from_file_location('monitor',Path(__file__).with_name('ice-monitored-review.py'))
    monitor=importlib.util.module_from_spec(spec);spec.loader.exec_module(monitor)
    now=time.time();error=None
    try:
        cpu,gpu,*_=monitor.temperatures();db.execute('INSERT OR REPLACE INTO samples VALUES (?,?,?)',(now,cpu,gpu))
    except Exception as e:error=str(e)
    db.execute('DELETE FROM samples WHERE utc<?',(now-7*86400,));db.commit()
    rows=db.execute('SELECT utc,cpu,gpu FROM samples ORDER BY utc').fetchall();db.close()
    data=json.dumps(dict(rows=rows,updated=datetime.now(timezone.utc).isoformat(),error=error)).replace('<','\\u003c')
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="refresh" content="30"><title>CPU and GPU temperatures</title><style>body{font:17px system-ui;background:#17212b;color:#eee;margin:25px}canvas{width:100%;max-height:650px;background:#0d1720}label{margin:15px}</style><h1>CPU and GPU temperatures</h1><p id="status"></p><p>CPU: hottest core/package sensor (orange). GPU: hottest GPU sensor (cyan). Last 24 hours; UTC time. Gaps over two minutes are not connected. Dashed lines are experiment cutoffs, not damage thresholds. Seven days retained locally.</p><canvas id="plot" width="1400" height="600"></canvas><p id="hover"></p><script>
const data=DATA,c=document.getElementById('plot'),ctx=c.getContext('2d'),rows=data.rows;
const controls=document.createElement('div');
controls.innerHTML='<button id="older">← Older</button> <button id="newer">Newer →</button> <button id="live">Live</button><label>History <input id="history" type="range" step="60" style="width:55%;vertical-align:middle"></label><p id="window"></p>';
c.before(controls);
const latest=Date.parse(data.updated)/1000,span=6*3600,earliest=Math.min(latest,Math.max(latest-7*86400,rows.length?rows[0][0]:latest)+span);
const slider=document.getElementById('history');slider.min=earliest;slider.max=latest;
let end=latest,start=end-span;
const saved=Number(new URLSearchParams(location.hash.slice(1)).get('end'));
if(saved)end=Math.max(earliest,Math.min(latest,saved));
function navigate(value){end=Math.max(earliest,Math.min(latest,value));history.replaceState(null,'',location.pathname+location.search+(end>=latest?'':'#end='+Math.round(end)));draw();}
slider.oninput=()=>navigate(Number(slider.value));
document.getElementById('older').onclick=()=>navigate(end-span);
document.getElementById('newer').onclick=()=>navigate(end+span);
document.getElementById('live').onclick=()=>navigate(latest);
c.addEventListener('wheel',e=>{e.preventDefault();navigate(end+(e.deltaX||e.deltaY)*30);},{passive:false});
document.getElementById('status').textContent=(data.error?'Sensor error: '+data.error:'Updated '+data.updated)+(rows.length?` · Latest CPU ${rows.at(-1)[1]}°C / GPU ${rows.at(-1)[2]}°C`:' · No samples');
const x=t=>70+(t-start)/span*1300,y=v=>540-v/110*500;
function draw(){
start=end-span;slider.value=end;
document.getElementById('older').disabled=end<=earliest;
document.getElementById('newer').disabled=end>=latest;
document.getElementById('window').textContent=new Date(start*1000).toISOString().slice(0,16)+' — '+new Date(end*1000).toISOString().slice(0,16)+' UTC'+(end>=latest?' · LIVE':' · historical');
ctx.clearRect(0,0,c.width,c.height);
ctx.font='16px system-ui';ctx.fillStyle='#ddd';ctx.strokeStyle='#354453';
for(let v=0;v<=110;v+=10){ctx.beginPath();ctx.moveTo(70,y(v));ctx.lineTo(1370,y(v));ctx.stroke();ctx.fillText(v+'°C',10,y(v)+5);}
for(let i=0;i<=6;i++){let t=start+i*3600;ctx.fillText(new Date(t*1000).toISOString().slice(11,16),x(t)-20,575);}
for(const [v,color] of [[95,'#fa4'],[83,'#5df']]){ctx.setLineDash([6,6]);ctx.strokeStyle=color;ctx.beginPath();ctx.moveTo(70,y(v));ctx.lineTo(1370,y(v));ctx.stroke();}ctx.setLineDash([]);
for(const [column,color] of [[1,'#fa4'],[2,'#5df']]){ctx.strokeStyle=color;ctx.lineWidth=1.5;ctx.beginPath();let previous=null;for(const row of rows){if(row[0]<start||row[0]>end)continue;if(previous===null||row[0]-previous>120)ctx.moveTo(x(row[0]),y(row[column]));else ctx.lineTo(x(row[0]),y(row[column]));previous=row[0];}ctx.stroke();}
}
draw();
c.onmousemove=e=>{const visible=rows.filter(r=>r[0]>=start&&r[0]<=end);if(!visible.length)return;const b=c.getBoundingClientRect(),t=start+(((e.clientX-b.left)*1400/b.width)-70)/1300*span;const r=visible.reduce((a,b)=>Math.abs(a[0]-t)<Math.abs(b[0]-t)?a:b);document.getElementById('hover').textContent=new Date(r[0]*1000).toISOString()+` · CPU ${r[1]}°C · GPU ${r[2]}°C`;};
</script>'''.replace('DATA',data)
    page=page.replace('Last 24 hours; UTC time.', 'Rolling six-hour window; UTC time. Scroll over the chart or use the slider to browse seven days. Live returns to the newest readings.')
    tmp=a.output/'index.html.tmp';tmp.write_text(page);tmp.replace(a.output/'index.html')


if __name__=='__main__':main()
