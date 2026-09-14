"""Accepted response identities and provenance shared by archive workers."""
import json
from pathlib import Path


def accepted_response(row, allow_legacy_qwen=False):
    model = row.get('model', '')
    if not allow_legacy_qwen and (not model or 'qwen' in model.lower()):
        return False
    if row.get('finish_reason') != 'stop':
        return False
    response = row.get('response')
    if not isinstance(response, str):
        return False
    try:
        parsed = json.loads(response.strip().removeprefix('```json').removeprefix('```').removesuffix('```').strip())
    except ValueError:
        return False
    return isinstance(parsed, dict) and isinstance(parsed.get('surface_percentages'), dict) and bool(parsed['surface_percentages'])


def external_completions(config, own_results, queue_files):
    """Read source records without relocating responses or their relative images.

    Explicitly configured missing or malformed sources raise an error so a
    temporarily unavailable source cannot silently cause duplicate inference.
    """
    if not config:
        return {}
    config = Path(config).resolve()
    own_results = Path(own_results).resolve()
    completed = {}
    sources = json.loads(config.read_text())
    if not isinstance(sources, list):
        raise ValueError('Completion sources must be a JSON list of result paths')
    for source in dict.fromkeys(sources):
        source = (config.parent / source).resolve()
        if source == own_results:
            continue
        rows = json.loads(source.read_text())
        if not isinstance(rows, list):
            raise ValueError(f'Completion source must contain a list: {source}')
        for row in rows:
            file = row.get('file')
            if file in queue_files and accepted_response(row):
                completed.setdefault(file, []).append(dict(
                    source=str(source), image_root=str(source.parent),
                    images=row.get('images', []), model=row.get('model'), utc=row.get('utc')))
    return completed


def completion_counts(classified, filtered, external, queue_files):
    """Count queue identities once, preferring classifications over estimates."""
    classified = set(classified) & queue_files
    external = (set(external) & queue_files) - classified
    filtered = (set(filtered) & queue_files) - classified - external
    return dict(completed=len(classified | external | filtered),
                classified=len(classified), externally_completed=len(external),
                filtered_seawater=len(filtered))
