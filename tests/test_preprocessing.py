"""Asset lifetime and time mapping tests. Separator/recognition mocks are explicit."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import threading
import wave
import pytest
from asr2rpp import preprocessing as prep
from asr2rpp.pipeline import Stage
from asr2rpp.catalog import Model, digest, Cancelled
from asr2rpp.adapters import Unit
from asr2rpp.media import WaveInfo


def make_wave(path, seconds=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as out:
        out.setparams((2, 2, 44100, 0, 'NONE', 'not compressed'))
        out.writeframes(b'\x01\x00' * (44100 * seconds * 2))


def test_processed_reference_requires_preprocessing():
    asr = Model('asr', 'whisper_cpp', 'asr', {'path': 'weights.bin'})
    with pytest.raises(ValueError, match='requires enabled'):
        prep.Settings(Stage('asr'), reference_audio='processed').validate({'asr': asr})


def test_distinct_file_and_timeline_origins(tmp_path):
    reference = tmp_path / 'project_vocals.wav'
    make_wave(reference)
    output = tmp_path / 'project.rpp'
    prep.write_reference(output, reference, [Unit(.25, 1.5, '発話')], 10, 10, False, 44100, reference_duration=2)
    text = output.read_text(encoding='utf-8')
    assert 'POSITION 10.25' in text and 'SOFFS 0.25' in text
    assert 'FILE "project_vocals.wav"' in text
    assert 'LENGTH 1.25' in text and 'SAMPLERATE 44100' in text


def test_original_source_offset_keeps_clip_start(tmp_path):
    source, output = tmp_path / 'original.mp4', tmp_path / 'project.rpp'
    source.write_bytes(b'original')
    prep.write_reference(output, source, [Unit(.25, 1.5, '発話')], 10, 0, False, reference_duration=15)
    text = output.read_text(encoding='utf-8')
    assert 'POSITION 10.25' in text and 'SOFFS 10.25' in text


@pytest.mark.parametrize('mode', ['original', 'processed'])
@pytest.mark.parametrize('same', [True, False])
def test_pipeline_reference_and_cleanup(tmp_path, monkeypatch, mode, same):
    source = tmp_path / 'original.wav'
    make_wave(source, 15)
    original_hash = digest(source)
    weights = tmp_path / 'converted.gguf'
    weights.write_bytes(b'mocked weights')
    catalog = {'asr': Model('asr', 'whisper_cpp', 'asr', {'path': str(weights)}),
               'sep': Model('sep', 'audio_cpp', 'sep', {'path': str(weights)}, family='mel_band_roformer', sample_rate=44100)}
    captured = []
    def fake_separator(input_path, work, settings, model, weight_path, cancel, progress, **kwargs):
        assert input_path == source and settings.clip_start == 10
        path = work / 'stems' / 'vocals.wav'
        make_wave(path)
        return path, {'test_fixture': True}
    def fake_recognition(path, settings, models, cancel, progress, *, _analysis_report):
        assert path != source and settings.clip_start == 0 and settings.preprocess is None
        captured.append(path)
        report = Path(_analysis_report)
        report.mkdir(parents=True)
        (report / 'transcript.json').write_text(json.dumps({'units': [asdict(Unit(.25, 1.5, '発話'))], 'warnings': []}), encoding='utf-8')
        (report/'manifest.json').write_text(json.dumps({'timestamp_source':'native_asr'}))
        return report
    monkeypatch.setattr(prep, 'separate', fake_separator)
    monkeypatch.setattr(prep.core, 'run_job', fake_recognition)
    settings = prep.Settings(Stage('asr'), preprocess=Stage('sep'), reference_audio=mode,
                             clip_start=10, clip_duration=2, same_directory=same,
                             output_directory=str(tmp_path / 'output'))
    out = prep.run_job(source, settings, catalog, threading.Event(), lambda _: None)
    report = out.with_suffix('.asr2rpp')
    assert not captured[0].exists()
    assert not list(report.glob('.preprocess-*'))
    assert digest(source) == original_hash
    text = out.read_text(encoding='utf-8')
    assert '.preprocess-' not in text
    assert 'POSITION 10.25' in text
    metadata = json.loads((report / 'manifest.json').read_text())
    assert metadata['source_unchanged'] is True
    assert metadata['temporary_preprocessed_audio_removed'] is True
    if mode == 'processed':
        saved = out.with_name(out.stem + '_vocals.wav')
        assert saved.exists() and prep.wave_info(saved).frames == 88200
        assert 'SOFFS 0.25' in text and f'{out.stem}_vocals.wav' in text
    else:
        assert 'SOFFS 10.25' in text and not out.with_name(out.stem + '_vocals.wav').exists()
    assert out.parent == (source.parent if same else tmp_path / 'output')



def test_processed_audio_collision_advances_project_suffix(tmp_path):
    source = tmp_path / 'meeting.wav'
    make_wave(source, 2)
    (tmp_path / 'meeting_vocals.wav').write_bytes(b'existing')
    settings = prep.Settings(Stage('asr'), preprocess=Stage('sep'), reference_audio='processed')
    output, report = prep.core.reserve_output(source, settings, sibling_suffixes=('_vocals.wav',))
    assert output.name == 'meeting_2.rpp'
    assert output.with_name('meeting_2_vocals.wav').name.endswith('_vocals.wav')
    report.rmdir()


def test_long_output_names_are_compacted_and_keep_vocals_suffix(tmp_path):
    long_stem = 'a' * 180
    source = tmp_path / (long_stem + '.wav')
    source.write_bytes(b'x')
    settings = prep.Settings(Stage('asr'))
    output, report = prep.core.reserve_output(source, settings, sibling_suffixes=('_vocals.wav',))
    vocals = output.with_name(output.stem + '_vocals.wav')
    assert len(output.stem) <= prep.core.MAX_OUTPUT_STEM_CHARS
    assert vocals.name.endswith('_vocals.wav')
    assert '~' in output.stem
    assert len(vocals.name) < 255
    report.rmdir()

def test_off_does_not_call_separator(tmp_path, monkeypatch):
    source = tmp_path / 'input.wav'
    model = Model('a', 'whisper_cpp', 'asr', {'path': 'weights'})
    monkeypatch.setattr(prep.core, 'run_job', lambda *a: source.with_suffix('.rpp'))
    monkeypatch.setattr(prep, 'separate', lambda *a: pytest.fail('OFF must not run separator'))
    assert prep.run_job(source, prep.Settings(Stage('a')), {'a': model}, threading.Event(), print) == source.with_suffix('.rpp')


def test_bad_length_is_rejected():
    a = WaveInfo(44100, 2, 44100, 32, 3)
    b = WaveInfo(44100, 2, 43000, 32, 3)
    with pytest.raises(ValueError, match='duration'):
        prep.validate_duration(a, b)


def test_raw_checkpoint_is_not_a_different_model(tmp_path):
    settings = prep.Settings(Stage('a'), preprocess=Stage('b'))
    model = Model('b', 'audio_cpp', 'sep', {'path': 'big_beta7.ckpt'}, family='mel_band_roformer', sample_rate=44100)
    with pytest.raises(ValueError, match='converted'):
        prep.separate(tmp_path/'input.wav', tmp_path/'work', settings, model,
                      tmp_path/'big_beta7.ckpt', threading.Event(), print)
