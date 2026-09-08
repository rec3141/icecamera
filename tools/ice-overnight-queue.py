"""Keep the original paired 96 first, then cover 128 Ward clusters. No UI mutations."""
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import cut_tree
root=Path('/home/cryomics/Downloads/amundsen-ice-full-leg-600x300')
out=Path('/home/cryomics/Downloads/amundsen-ice-gemma-overnight');out.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('reps',Path(__file__).with_name('ice-cluster-representatives.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
points=json.loads((root/'embedding.json').read_text());state=json.loads((root/'projections.json').read_text())
assert state['ids']==[p['id'] for p in points]
tree=np.array(state['projections']['tsne']['trees']['ward']);coords=np.array(state['projections']['tsne']['coords'])
labels=cut_tree(tree,n_clusters=[128]).ravel()
queue=json.loads(Path('/home/cryomics/Downloads/amundsen-ice-qwen-k32-classes/queue.json').read_text())['queue']
for q in queue:q['queue_source']='original-k32-paired'
done={q['file'] for q in queue}
for cluster,i,n,reason in m.representatives(coords,labels):
    p=points[i]
    if p['file'] in done:continue
    queue.append(dict(cluster=cluster,id=p['id'],file=p['file'],cluster_size=n,reason=reason,queue_source='600x300-k128'))
    done.add(p['file'])
(out/'queue.json').write_text(json.dumps(dict(queue=queue,feature_size=[600,300],note='Original 96 paired photos followed by 3 representatives per 128 clusters, deduplicated'),indent=2))
print(len(queue),'unique photos')
