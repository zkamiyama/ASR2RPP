"""Independent synthetic regression audit; no hardware/ASR-quality claims."""
import json
import threading
import wave
from pathlib import Path
import pytest
from asr2rpp.catalog import load_catalog
from asr2rpp.timing import TimingSettings, plan_for, asr_parameters
from asr2rpp.nemotron_transcribe import parse_text_result, parse_result, SCHEMA
from asr2rpp.providers import Request


def model():
    catalog, errors = load_catalog(Path(__file__).parents[1] / 'models')
    assert not errors
    return catalog['nemotron-transcribe-iq4-nl']


def raw(termination='end_of_turn'):
    return {'text': 'テスト。', 'custom_schema_output': {'schema': SCHEMA,
        'data': {'template_name': 'default', 'termination': termination}}}


def test_independent_timing_defaults_and_explicit_native():
    m = model()
    for align in (False, True):
        t = TimingSettings()
        p = plan_for(m, t, align)
        assert p.segmented and p.align == align
        assert asr_parameters(m, t, align, {}) == {'diarization': False, 'return_timestamps': False}
    t = TimingSettings(mode='native', speaker_source='native')
    assert not plan_for(m, t).segmented
    assert asr_parameters(m, t, False, {}) == {'diarization': True, 'return_timestamps': True}
    with pytest.raises(ValueError): plan_for(m, TimingSettings(speaker_source='native'), True)
    with pytest.raises(ValueError): asr_parameters(m, TimingSettings(), False, {'diarization': True})


@pytest.mark.parametrize('termination', ['token_limit', 'repetition_detected', 'cancelled', 'unknown', None])
def test_text_only_rejects_noncomplete_generation(termination):
    with pytest.raises(ValueError): parse_text_result(raw(termination))


def test_text_only_does_not_require_or_use_ctc():
    r = raw()
    assert parse_text_result(r).text == 'テスト。'
    r['custom_schema_output']['data'].update(alignment_status='failed', words=[{'start': -999}], failed_streams=['bad'])
    result = parse_text_result(r)
    assert result.units == [] and result.raw is r
    with pytest.raises(ValueError): parse_result(r)
    with pytest.raises(ValueError): parse_text_result({'text': 'テスト。'})


@pytest.mark.parametrize('failure', [None, 'missing', 'truncated'])
def test_text_manifest_reuse_and_failure_retention(tmp_path, monkeypatch, failure):
    from asr2rpp import native_batches as n
    audio = tmp_path / 'a.wav'
    with wave.open(str(audio), 'wb') as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes(b'\0' * 32000)
    calls = []
    monkeypatch.setattr(n, 'executable', lambda *a: Path('/mock/audio'))
    def run(argv, cancel, progress, log):
        calls.append(argv)
        assert '--words-out' not in argv and '--turns-out' not in argv
        seq = json.loads(Path(argv[argv.index('--request-sequence') + 1]).read_text())['requests']
        rows = []
        for req in seq:
            assert req['options']['return_timestamps'] is False
            assert req['options']['diarization'] is False
            Path(log).with_name('text_' + req['id'] + '.txt').write_text('テスト。')
            rows.append(dict(id=req['id'], custom_schema_output=raw('token_limit' if failure == 'truncated' else 'end_of_turn')['custom_schema_output']))
        Path(log).write_text('mock native run')
        if failure != 'missing':
            Path(argv[argv.index('--batch-manifest-out') + 1]).write_text(json.dumps({'requests': rows}))
    monkeypatch.setattr(n, 'run_process', run)
    requests = [Request('r' + str(i), audio) for i in range(17)]
    result = n.audio_many(model(), tmp_path/'weights', requests, tmp_path/'out',
        {'_text_only': True, 'parameters': {'return_timestamps': False, 'diarization': False}},
        threading.Event(), lambda _: None, 64*1024**2)
    assert len(calls) == 2  # 16 requests reuse one loaded native process.
    if failure:
        assert len(result.errors) == 17 and not result.results
    else:
        assert len(result.results) == 17 and not result.errors
        assert all(r.units == [] for r in result.results.values())
    assert (tmp_path/'out/r0/native/requests.json').is_file()
    assert (tmp_path/'out/r0/native/text_r0.txt').is_file()
