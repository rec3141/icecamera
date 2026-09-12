"""Refresh completed combined projection annotations without refitting its layout."""
import importlib.util
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

portal=Path('/home/cryomics/Downloads/amundsen-ice-2025-2026-tsne')
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--finish-layout',action='store_true',help='Wait for the feature/t-SNE service, then rebuild its linkage trees with thermal guards')
args=parser.parse_args()
if args.finish_layout:
    unit='ice-2026-projection.service'
    while subprocess.check_output(['systemctl','--user','show',unit,'-p','ActiveState','--value'],text=True).strip() in ('active','activating','deactivating'):
        time.sleep(10)
    result=subprocess.check_output(['systemctl','--user','show',unit,'-p','Result','--value'],text=True).strip()
    if result!='success' or json.loads((portal/'progress.json').read_text()).get('stage')!='complete':
        raise RuntimeError('Layout did not complete successfully; previous published explorer preserved')
    subprocess.run([sys.executable,str(Path(__file__).with_name('ice-projections.py')),'--output',str(portal),
                    '--annotations','/home/cryomics/Downloads/amundsen-ice-gemma-2026/results.json','--methods','tsne'],check=True)
    layout=json.loads((portal/'embedding.json').read_text())
    projections=json.loads((portal/'projections.json').read_text())
    if projections['ids']!=[point['id'] for point in layout] or any(
        len(projections.get('projections',{}).get('tsne',{}).get('trees',{}).get(method,[]))!=len(layout)-1
        for method in ('ward','single')
    ):raise RuntimeError('Updated t-SNE linkage trees are incomplete; inspect the projection log')
if (portal/'embedding.json').exists():
    spec=importlib.util.spec_from_file_location('explorer',Path(__file__).resolve().with_name('ice-region-explorer.py'))
    explorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(explorer)
    explorer.render(portal,[],False)
    print('Updated combined 2025–2026 model annotations')
