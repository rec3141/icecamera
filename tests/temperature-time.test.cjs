const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const vm=require('node:vm');
const {test}=require('node:test');

test('temperature page uses local time for status, range, ticks and hover, including DST',()=>{
  const source=fs.readFileSync(path.join(__dirname,'../tools/ice-temperature-page.py'),'utf8');
  const script=source.match(/<script>\n([\s\S]*?)<\/script>/)[1];
  const previous=process.env.TZ;
  try {
    process.env.TZ='America/Toronto';
    for(const [stamp,hour,zone] of [['2026-09-12T12:00:00Z','08:00','EDT'],['2026-01-12T12:00:00Z','07:00','EST']]) {
      const ticks=[],elements={};
      const ctx=new Proxy({fillText:text=>ticks.push(text)},{get:(target,key)=>target[key]||(()=>{})});
      const element=()=>({textContent:'',before(){},addEventListener(){},getContext:()=>ctx,getBoundingClientRect:()=>({left:0,width:1400})});
      const document={getElementById:id=>elements[id]||(elements[id]=element()),createElement:element};
      const data={updated:stamp,rows:[[Date.parse(stamp)/1000,71,72]]};
      const context={document,Date,URLSearchParams,location:{hash:''},history:{replaceState(){}}};
      vm.runInNewContext(script.replace('DATA',JSON.stringify(data)),context);
      elements.plot.onmousemove({clientX:1370});
      for(const id of ['status','window','hover']) {
        assert.match(elements[id].textContent,new RegExp(hour));
        assert.match(elements[id].textContent,new RegExp(zone+'|GMT-0?'+(zone==='EDT'?'4':'5')));
      }
      assert(ticks.includes(hour));
    }
  } finally {if(previous===undefined)delete process.env.TZ;else process.env.TZ=previous;}
});
