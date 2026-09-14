import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
from types import SimpleNamespace

import pytest

from dashboard.camera_completion import accepted_response, completion_counts, external_completions


def response(file, **extra):
    return dict(dict(file=file, model='gemma', finish_reason='stop',
                     response='```json\n{"surface_percentages":{"calm water":100}}\n```',
                     images=['images/roi.jpg']), **extra)


@pytest.mark.parametrize('extra', [dict(model='Qwen'), dict(model=''), dict(finish_reason='length'),
                                  dict(response='invalid'), dict(response='{}'), dict(response='[]')])
def test_unaccepted_responses_remain_pending(extra):
    assert not accepted_response(response('pending', **extra))


def test_external_sources_preserve_provenance_and_do_not_copy_responses(tmp_path):
    source = tmp_path / 'source.json'
    records = [response('existing'), response('outside'), response('pending', finish_reason='length')]
    source.write_text(json.dumps(records))
    own = tmp_path / 'own.json'
    own.write_text(json.dumps([response('own')]))
    config = tmp_path / 'annotation-sources.json'
    config.write_text(json.dumps(['source.json', 'own.json', 'source.json']))
    completed = external_completions(config, own, {'existing', 'pending', 'own'})
    assert set(completed) == {'existing'}
    assert completed['existing'] == [dict(source=str(source), image_root=str(tmp_path),
                                         images=['images/roi.jpg'], model='gemma', utc=None)]
    assert json.loads(source.read_text()) == records
    assert json.loads(own.read_text()) == [response('own')]


def test_missing_explicit_source_stops_deduplication(tmp_path):
    config = tmp_path / 'annotation-sources.json'
    config.write_text('["missing.json"]')
    with pytest.raises(FileNotFoundError):
        external_completions(config, tmp_path / 'own.json', {'a'})
    assert external_completions(None, tmp_path / 'own.json', {'a'}) == {}


def test_completion_counts_are_disjoint_and_limited_to_queue():
    assert completion_counts({'own', 'outside'}, {'own', 'external', 'filtered'},
                             {'own', 'external', 'outside'}, {'own', 'external', 'filtered', 'pending'}) == dict(
        completed=3, classified=1, externally_completed=1, filtered_seawater=1)


def test_watchdog_stops_completed_queue_with_external_responses(tmp_path, monkeypatch):
    for name, value in dict(ICE_TEACHER_ROOT=str(tmp_path), ICE_TEACHER_STATE=str(tmp_path / 'state.json'),
                            ICE_TEACHER_PAUSE=str(tmp_path / 'paused'), ICE_TEACHER_RESULTS='own.json',
                            ICE_TEACHER_UNIT='archive.service', ICE_TEACHER_TIMER='archive.timer',
                            ICE_GEMMA_COMPLETED_SOURCES=str(tmp_path / 'sources.json')).items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv('ICE_TEACHER_LIVE', raising=False)
    (tmp_path / 'queue.json').write_text(json.dumps(dict(queue=[dict(file='own'), dict(file='external')])))
    (tmp_path / 'own.json').write_text(json.dumps([response('own'), response('outside')]))
    (tmp_path / 'source.json').write_text(json.dumps([response('external')]))
    (tmp_path / 'sources.json').write_text('["source.json","own.json"]')
    class MonitorLoader:
        def create_module(self, spec):
            return None
        def exec_module(self, module):
            module.temperatures = lambda: (70, 60, 0, 0)
    original_spec = importlib.util.spec_from_file_location
    monkeypatch.setattr(importlib.util, 'spec_from_file_location',
                        lambda name, path: importlib.util.spec_from_loader(name, MonitorLoader())
                        if name == 'monitor' else original_spec(name, path))
    calls = []
    def run(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(stdout='inactive')
    monkeypatch.setattr(subprocess, 'run', run)
    runpy.run_path(str(Path(__file__).parents[1] / 'tools/ice-teacher-watch.py'))
    state = json.loads((tmp_path / 'state.json').read_text())
    assert (state['saved'], state['expected'], state['externally_completed'], state['filtered_seawater']) == (2, 2, 1, 0)
    assert state['status'] == 'complete'
    assert calls[-1] == ['systemctl', '--user', 'stop', 'archive.timer']


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location('completion_worker', Path(__file__).parents[1] / 'tools/ice-gemma-full.py')
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_priority_wait_preserves_thermal_checks_and_resumes(worker, monkeypatch):
    states = iter(['active', 'activating', 'inactive'])
    def run(*args, **kwargs):
        state = next(states)
        return SimpleNamespace(stdout=state, stderr='', returncode=0 if state == 'active' else 3)
    monkeypatch.setattr(worker.subprocess, 'run', run)
    sleeps = []
    monkeypatch.setattr(worker.time, 'sleep', sleeps.append)
    temperatures = iter([(80, 70, 80), (90, 83, 90)])
    statuses, cools = [], []
    worker.wait_for_priority_unit('archive.service', lambda state, **extra: statuses.append((state, extra)),
                                  lambda: next(temperatures), lambda: cools.append(True))
    assert sleeps == [10, 10]
    assert cools == [True]
    assert statuses == [('waiting for priority archive', {'priority_unit': 'archive.service'})] * 2


def test_unavailable_service_manager_does_not_bypass_priority(worker, monkeypatch):
    monkeypatch.setattr(worker.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stdout='', stderr='No bus'))
    with pytest.raises(RuntimeError, match='Cannot inspect priority worker'):
        worker.wait_for_priority_unit('archive.service', None, None, None)
    worker.wait_for_priority_unit(None, None, None, None)
