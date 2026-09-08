"""Audit raw Qwen percentages without repairing or discarding model responses."""
import json
import math

ICE=('grease ice','nilas','thin fyi','icy bits','brash ice','ice floe')
SURFACE=('whitecap','small waves','smooth water',*ICE,'unknown')
ARTIFACT=('blurry','fog','reflection','night')


def assess(text):
    flags=[]
    try:
        parsed=json.loads(text.strip().removeprefix('```json').removesuffix('```').strip())
        if not isinstance(parsed,dict):raise ValueError('Expected object')
    except (ValueError,TypeError,AttributeError):
        return dict(quality_flags=['invalid_json'],surface_sum=None,derived_total_ice_percent=None,reported_total_ice_percent=None)
    surface=parsed.get('surface_percentages');artifact=parsed.get('artifact_percentages')
    def valid(group,keys):
        return isinstance(group,dict) and set(group)==set(keys) and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=100 for v in group.values())
    surface_ok=valid(surface,SURFACE)
    if not surface_ok:flags.append('invalid_surface_percentages')
    if not valid(artifact,ARTIFACT):flags.append('invalid_artifact_percentages')
    total=sum(surface.values()) if surface_ok else None
    derived=sum(surface[name] for name in ICE) if surface_ok else None
    if total is not None and abs(total-100)>1:flags.append('surface_sum_not_100')
    if derived is not None and parsed.get('total_ice_percent')!=derived:flags.append('reported_total_disagrees')
    return dict(quality_flags=flags,surface_sum=total,derived_total_ice_percent=derived,reported_total_ice_percent=parsed.get('total_ice_percent'))
