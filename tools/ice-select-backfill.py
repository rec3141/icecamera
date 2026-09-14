"""Rank every human-labelled joint t-SNE Ward-64 cluster by its ice-label fraction."""
import json
import re
from collections import defaultdict
from pathlib import Path
from scipy.cluster.hierarchy import cut_tree

def main():
    root=Path.home()/'Downloads'
    portal=root/'amundsen-ice-2025-2026-tsne'
    source=root/'ice-region-human-labels(2).json'
    state=json.loads((portal/'projections.json').read_text())
    points={r['id']:r for r in json.loads((portal/'embedding.json').read_text())}
    human={r['id']:r['labels'] for r in json.loads(source.read_text())['labels']}
    labels=cut_tree(state['projections']['tsne']['trees']['ward'],n_clusters=64).ravel()
    groups=defaultdict(list)
    for identifier,cluster in zip(state['ids'],labels):groups[int(cluster)].append(identifier)
    ice={'grease ice','nilas','thin fyi','thin ice floe','icy bits','brash ice','ice floe','thick ice floe'}
    if state['ids']!=list(points):raise ValueError('Stale Ward tree: projection identities differ')
    report=[];allowed=[];fractions={};cluster_by_file={}
    for cluster,ids in sorted(groups.items()):
        labeled=[i for i in ids if human.get(i)]
        positive=sum(any(re.sub(r'^[0-9A-F] ', '',v).lower() in ice for v in human[i]) for i in labeled)
        selected=bool(labeled)
        for i in ids:
            fractions[points[i]['file']]=positive/len(labeled) if labeled else 0
            cluster_by_file[points[i]['file']]=cluster
        report.append(dict(cluster=cluster,photos=len(ids),human_labeled=len(labeled),ice_labeled=positive,ice_fraction=positive/len(labeled) if labeled else None,selected=selected))
        if selected:allowed.extend(points[i]['file'] for i in ids)
    out=root/'amundsen-ice-gemma-2026'
    control=out/'backfill-paused.json'
    policy=json.loads(control.read_text())
    allowed=list(dict.fromkeys(allowed+policy.get('additional_allowed_files',[])))
    report.sort(key=lambda r:(-(r['ice_fraction'] if r['ice_fraction'] is not None else -1),r['cluster']))
    policy.update(reason='All human-labelled joint t-SNE Ward k=64 clusters, highest human ice-label fraction first; unlabelled clusters excluded.',label_source=str(source),allowed_files=allowed,clusters=report,ice_fraction_by_file=fractions,cluster_by_file=cluster_by_file,cluster_rank={str(r['cluster']):i for i,r in enumerate(report)})
    if policy.get('additional_allowed_files'):
        policy['reason']='Human-labelled clusters ranked by ice-label fraction, plus explicitly authorized archive photos.'
    tmp=control.with_suffix('.tmp');tmp.write_text(json.dumps(policy,indent=2));tmp.replace(control)
    archive_selection=root/'amundsen-ice-gemma-leg4-size-evidence/selection.json'
    if archive_selection.exists():
        archive=json.loads(archive_selection.read_text())
        archive['queue'].sort(key=lambda r:(
            -fractions.get(r['file'],0),
            policy['cluster_rank'].get(str(cluster_by_file.get(r['file'])),9999),
            r['file']))
        archive['description']='All Leg 4 photos, ordered by combined-map human ice-label fraction, highest first.'
        temp=archive_selection.with_suffix('.priority.tmp')
        temp.write_text(json.dumps(archive,indent=2));temp.replace(archive_selection)
    queue=json.loads((out/'selection.json').read_text())['queue']
    done={r['file'] for name in ('results.json','seawater-skips.json') for r in json.loads((out/name).read_text())}
    allowed=set(allowed)
    for threshold in (.8,.6,.4):
        matching=[r for r in queue if fractions.get(r['file'],0)>threshold]
        print(f'>{threshold:.0%}: {len(matching)} queued photos, {sum(r["file"] not in done for r in matching)} pending')
    print(json.dumps(dict(selected_clusters=sum(r['selected'] for r in report),eligible_queued=sum(r['file'] in allowed for r in queue),pending_selected=sum(r['file'] in allowed and r['file'] not in done for r in queue)),indent=2))

if __name__=='__main__':main()
