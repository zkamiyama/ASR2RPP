"""Desired regression contracts for ASR2RPP 7fdffae.

Run from the repository root:
  PYTHONPATH=. python -m pytest /path/to/test_audit_regressions.py -v -rx

These formerly failing contracts must pass on every supported platform.
Network and native inference are mocked. The memory-safety test intercepts the
unsafe NumPy operation: it NEVER performs an out-of-bounds memory read.
"""
import hashlib
import io
import json
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from asr2rpp import catalog, pipeline, queue_runner, model_conversion
from asr2rpp.adapters import Unit, Result
from asr2rpp.catalog import Model
from asr2rpp.pipeline import Settings, Stage


class Response(io.BytesIO):
    def __init__(self, value):
        super().__init__(value)
        self.headers = {'Content-Length': str(len(value))}


def test_installed_cache_does_not_report_stale_hash(tmp_path, monkeypatch):
    monkeypatch.setenv('ASR2RPP_WEIGHTS_DIR', str(tmp_path))
    source = {'repo': 'example/model', 'revision': 'a' * 40, 'files': ['model.bin']}
    model = Model('audit', 'whisper_cpp', 'asr', source)
    identity = hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()[:16]
    directory = tmp_path / model.id / identity
    directory.mkdir(parents=True)
    original = b'A' * 2048
    (directory / 'model.bin').write_bytes(b'B' * len(original))
    state = {'files': {'model.bin': {'size': len(original), 'sha256': hashlib.sha256(original).hexdigest()}}}
    (directory / 'installed.json').write_text(json.dumps(state))
    try:
        path, provenance = catalog.resolve_model(model, threading.Event(), lambda _: None)
    except (ValueError, FileNotFoundError):
        return  # rejecting/requiring re-install is also a valid repair
    assert hashlib.sha256(path.read_bytes()).hexdigest() == provenance['files']['model.bin']['sha256']


def test_small_verified_auxiliary_asset_is_allowed(tmp_path, monkeypatch):
    monkeypatch.setenv('ASR2RPP_WEIGHTS_DIR', str(tmp_path))
    blobs = {'model.bin': b'W' * 2048, 'config.json': b'{"sample_rate":16000}'}
    source = {'repo': 'example/model', 'files': list(blobs),
              'sha256': {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}}
    model = Model('audit', 'whisper_cpp', 'asr', source)
    def fetch(request, timeout=30):
        url = request.full_url
        return Response(json.dumps({'sha': 'a' * 40}).encode() if '/api/models/' in url else blobs[url.rsplit('/', 1)[1]])
    monkeypatch.setattr(catalog, 'urlopen', fetch)
    path, _ = catalog.resolve_model(model, threading.Event(), lambda _: None, download=True)
    assert (path.parent / 'config.json').read_bytes() == blobs['config.json']


def test_invalid_tensor_never_reaches_unsafe_view(monkeypatch):
    import numpy as np
    class UnsafeViewReached(RuntimeError):
        pass
    def intercept(*args, **kwargs):
        raise UnsafeViewReached('unsafe view requested; deliberately blocked before memory access')
    monkeypatch.setattr(np.lib.stride_tricks, 'as_strided', intercept)
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as z:
        z.writestr('data/0', np.array([1., 2., 3., 4.], dtype=np.float32).tobytes())
    data.seek(0)
    ref = model_conversion.TensorRef('0', 'FloatStorage', 0, (2,), (-1,))
    with zipfile.ZipFile(data) as z, pytest.raises(model_conversion.ConversionError):
        model_conversion._materialize(z, '', ref)


def jobs_at(tmp_path):
    jobs, paths = [], {}
    for i in range(2):
        report = tmp_path / f'report{i}'
        report.mkdir()
        audio = tmp_path / f'audio{i}.wav'
        audio.write_bytes(b'PCM')
        job = queue_runner.QueueJob(i, f'q{i:06d}', audio, tmp_path / f'out{i}.rpp', report, '', {'status': 'running'})
        jobs.append(job)
        paths[job.key] = audio
    return jobs, paths


def mock_audio_batch(tmp_path, monkeypatch, mode):
    jobs, paths = jobs_at(tmp_path)
    model = Model('audit', 'audio_cpp', 'asr', {'path': 'unused'}, family='vibevoice_asr', sample_rate=24000)
    monkeypatch.setattr('asr2rpp.native_batches.executable', lambda *args: tmp_path / 'audiocpp_cli')
    def execute(argv, cancel, progress, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text('diagnostic: this output must survive failure')
        if mode == 'crash':
            raise RuntimeError('native process failed')
        base = Path(argv[argv.index('--segments-out') + 1])
        (base.parent / f'{base.stem}_{jobs[0].key}.json').write_text(json.dumps({'segments': [{'start': 0, 'end': 1, 'text': 'ok'}]}))
        (base.parent / f'{base.stem}_{jobs[1].key}.json').write_text('{broken')
    monkeypatch.setattr('asr2rpp.native_batches.run_process', execute)
    results = queue_runner._audio_batch(model, tmp_path / 'weights', Stage('audit'), jobs, paths,
                tmp_path / 'work', threading.Event(), lambda _: None, lambda *args: None, 'asr', 1024 * 1024)
    return jobs, results


def test_bad_result_does_not_discard_successful_sibling(tmp_path, monkeypatch):
    jobs, results = mock_audio_batch(tmp_path, monkeypatch, 'bad_result')
    assert not jobs[0].failed
    assert jobs[1].failed
    assert jobs[0].key in results


def test_native_failure_log_is_persisted_in_report(tmp_path, monkeypatch):
    jobs, _ = mock_audio_batch(tmp_path, monkeypatch, 'crash')
    assert any('diagnostic:' in p.read_text() for p in jobs[0].report.rglob('*.log'))


def test_diarization_uses_fine_native_asr_units(tmp_path, monkeypatch):
    monkeypatch.setenv('ASR2RPP_CACHE_DIR', str(tmp_path / 'cache'))
    source = tmp_path / 'source.wav'
    source.write_bytes(b'placeholder: native decode is mocked')
    asr = Model('asr', 'audio_cpp', 'asr', {'path': str(source)}, family='nemotron_asr')
    diar = Model('diar', 'audio_cpp', 'diar', {'path': str(source)}, family='nemotron_3_diar')
    monkeypatch.setattr(pipeline, 'executable', lambda *args: source)
    monkeypatch.setattr(pipeline, 'ffmpeg_path', lambda *args: 'ffmpeg')
    monkeypatch.setattr(pipeline, 'resolve_model', lambda *args: (source, {}))
    def decode(_src, dest, *args, **kwargs):
        dest.write_bytes(b'PCM')
        return 3.0
    monkeypatch.setattr(pipeline, 'decode', decode)
    monkeypatch.setattr(pipeline, 'full_reference_duration', lambda *args: 3.0)
    def infer(model, *args, **kwargs):
        if model.task == 'asr':
            return Result([Unit(0, 1, 'first', granularity='word'), Unit(1, 3, 'second', granularity='word')], {})
        return Result([Unit(0, 1, speaker='A'), Unit(1, 3, speaker='B')], {})
    monkeypatch.setattr(pipeline, 'infer', infer)
    output = pipeline.run_job(source, Settings(Stage('asr'), diar=Stage('diar')),
                     {'asr': asr, 'diar': diar}, threading.Event(), lambda _: None)
    exported = json.loads((output.with_suffix('.asr2rpp')/'transcript.json').read_text())['units']
    assert [(u['text'], u['speaker']) for u in exported] == [('first', 'A'), ('second', 'B')]
