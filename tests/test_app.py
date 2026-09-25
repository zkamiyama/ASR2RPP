import json
import os
from pathlib import Path
from fractions import Fraction
import subprocess
import sys
import threading
import time
import wave
import pytest
from rpp_writer import Source, Item, Track, Project, dumps
from asr2rpp.catalog import load_catalog, Model, Cancelled, safe_relative, weights_root, cache_root, resolve_model
from asr2rpp.adapters import Unit, Result, parse_whisper, parse_audio, run_process
import asr2rpp.adapters as adapters
from asr2rpp.pipeline import Settings, Stage, reserve_output, clean_bounds, group_units, has_alignable_text, export_rpp, run_job


def test_independent_writer():
    source = Source('media/音声.wav')
    item = Item('こんにちは "world"', source, Fraction(5, 2), Fraction(5, 2), Fraction(1, 48000))
    output = dumps(Project((Track('Speaker 01', (item,)),)))
    assert 'SOFFS 2.5' in output and 'POSITION 2.5' in output
    assert 'media/音声.wav' in output
    assert 'torch' not in sys.modules


def test_catalog_isolation(tmp_path):
    (tmp_path / 'good.toml').write_text('runtime="whisper_cpp"\ntask="asr"\n[source]\npath="model.bin"', encoding='utf-8')
    (tmp_path / 'bad.toml').write_text('invalid = [', encoding='utf-8')
    models, errors = load_catalog(tmp_path)
    assert list(models) == ['good'] and len(errors) == 1


def test_configurable_model_and_temp_roots(tmp_path, monkeypatch):
    monkeypatch.setenv('ASR2RPP_HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('ASR2RPP_WEIGHTS_DIR', raising=False)
    monkeypatch.delenv('ASR2RPP_CACHE_DIR', raising=False)
    assert weights_root() == tmp_path / 'home' / 'weights'
    assert cache_root() == tmp_path / 'home' / 'cache'
    monkeypatch.setenv('ASR2RPP_WEIGHTS_DIR', str(tmp_path / 'models-custom'))
    monkeypatch.setenv('ASR2RPP_CACHE_DIR', str(tmp_path / 'temp-custom'))
    assert weights_root() == tmp_path / 'models-custom'
    assert cache_root() == tmp_path / 'temp-custom'


@pytest.mark.parametrize('value', ['../secret', '/etc/passwd', 'C:\\file', 'a/../../b'])
def test_safe_paths(value):
    with pytest.raises(ValueError):
        safe_relative(value)


def test_all_templates():
    models, errors = load_catalog(Path('models'))
    assert not errors
    assert len(models) >= 6
    assert {m.task for m in models.values()} == {'asr', 'diar', 'align', 'sep'}


def test_runtime_messages_use_stable_model_id_not_editable_name(tmp_path, monkeypatch):
    monkeypatch.setenv('ASR2RPP_WEIGHTS_DIR', str(tmp_path / 'weights'))
    model = Model(
        'anime-whisper', 'whisper_cpp', 'asr',
        {'repo': 'owner/repo', 'files': ['model.bin']},
        name='Anime Whisper · 実験的変換版',
    )
    with pytest.raises(FileNotFoundError) as caught:
        resolve_model(model, threading.Event(), lambda _text: None, download=False)
    message = str(caught.value)
    assert 'anime-whisper' in message
    assert '実験的変換版' not in message


def test_output_collision_and_same_directory(tmp_path):
    source = tmp_path / 'source.wav'
    source.touch()
    settings = Settings(Stage('test'))
    first, report1 = reserve_output(source, settings)
    second, report2 = reserve_output(source, settings)
    assert first.name == 'source.rpp' and second.name == 'source_2.rpp'
    assert first.parent == tmp_path and report1 != report2


def test_custom_output(tmp_path):
    source = tmp_path / 'source.wav'
    settings = Settings(Stage('test'), same_directory=False, output_directory=str(tmp_path / 'out'))
    output, _ = reserve_output(source, settings)
    assert output.parent == tmp_path / 'out'


def test_required_output():
    with pytest.raises(ValueError, match='Output directory'):
        Settings(Stage('x'), same_directory=False).validate({})


def test_whisper_offsets_are_milliseconds():
    result = parse_whisper({'transcription': [{'offsets': {'from': 1200, 'to': 3400}, 'text': ' hello'}]})
    assert result.units[0].start == 1.2 and result.units[0].end == 3.4


def test_nemotron_does_not_claim_word_intervals():
    result = parse_audio({'words': [{'start': 1, 'end': 1.08, 'word': 'test'}]}, 'asr', 'nemotron_asr')
    assert result.units[0].granularity == 'token'
    assert result.units[0].method == 'emission_frame'


def test_nemotron_infer_uses_streaming_session_and_keeps_timestamps(tmp_path, monkeypatch):
    model = Model(
        'nemotron-asr', 'audio_cpp', 'asr', {'path': 'unused.gguf'},
        defaults={'language': 'ja-JP'}, family='nemotron_asr', sample_rate=16000,
    )
    weights = tmp_path / 'model.gguf'
    audio = tmp_path / 'audio.wav'
    weights.write_bytes(b'GGUF')
    audio.write_bytes(b'WAV')
    commands = []

    monkeypatch.setattr(adapters, 'executable', lambda *_args, **_kwargs: tmp_path / 'audiocpp_cli.exe')

    def fake_run(argv, _cancel, _progress, _log, timeout=7200):
        commands.append(list(argv))
        out = Path(argv[argv.index('--words-out') + 1])
        out.write_text(json.dumps({
            'words': [{'start': 0.0, 'end': 0.32, 'word': 'テスト'}]
        }, ensure_ascii=False), encoding='utf-8')

    monkeypatch.setattr(adapters, 'run_process', fake_run)
    result = adapters.infer(
        model, weights, audio, tmp_path / 'work',
        {'device': 'vulkan', 'language': 'ja-JP', 'threads': 4},
        threading.Event(), lambda _text: None,
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index('--mode') + 1] == 'streaming'
    assert '--words-out' in command
    assert '--batch-audio-dir' not in command
    assert result.units[0].granularity == 'token'
    assert result.units[0].start == 0.0
    assert result.units[0].end == 0.32


def test_bounds_preserve_raw():
    raw = [Unit(-1, 2, 'a'), Unit(3, 6, 'b'), Unit(4, 4, 'c')]
    warnings = []
    result = clean_bounds(raw, 5, warnings)
    assert raw[0].start == -1 and raw[1].end == 6
    assert result[0].start == 0 and result[-1].end == 5 and len(warnings) == 3


def test_no_false_word_timing():
    units = group_units([Unit(0, 4, '今日はいい天気です')])
    assert len(units) == 1 and units[0].granularity == 'segment'


def test_forced_alignment_keeps_punctuation_only_asr_span(tmp_path, monkeypatch):
    source = tmp_path / 'source.wav'
    with wave.open(str(source), 'wb') as handle:
        handle.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        handle.writeframes(b'\0\0' * 16000)

    asr_model = Model('asr-test', 'whisper_cpp', 'asr', {'path': str(source)})
    align_model = Model(
        'align-test', 'audio_cpp', 'align', {'path': str(source)},
        family='qwen3_forced_aligner',
    )
    catalog = {'asr-test': asr_model, 'align-test': align_model}
    settings = Settings(Stage('asr-test'), align=Stage('align-test'))
    monkeypatch.setenv('ASR2RPP_CACHE_DIR', str(tmp_path / 'cache'))
    monkeypatch.setattr('asr2rpp.pipeline.executable', lambda *args: source)
    monkeypatch.setattr('asr2rpp.pipeline.ffmpeg_path', lambda *args: 'ffmpeg')
    monkeypatch.setattr('asr2rpp.pipeline.resolve_model', lambda *args: (source, {}))
    monkeypatch.setattr('asr2rpp.pipeline.decode', lambda *args, **kwargs: 1.0)

    import shutil
    from asr2rpp import alignment
    def fake_decode(_source, destination, *args, **kwargs):
        shutil.copyfile(source, destination)
        return 1.0
    monkeypatch.setattr('asr2rpp.pipeline.decode', fake_decode)
    aligned_texts = []
    def fake_infer(*args, **kwargs):
        return Result([
            Unit(0.0, 0.2, 'こんにちは。', granularity='segment'),
            Unit(0.3, 0.4, '、', granularity='segment'),
        ], {})
    def fake_requests(_model, _weights, _stage, requests, *args):
        aligned_texts.extend(req.text for req in requests)
        return {req.key: Result([Unit(0.0, 0.1, req.text, granularity='word')], {})
                for req in requests}
    monkeypatch.setattr('asr2rpp.pipeline.infer', fake_infer)
    monkeypatch.setattr(alignment, 'infer_requests', fake_requests)
    output = run_job(source, settings, catalog, threading.Event(), lambda _text: None)

    assert aligned_texts == ['こんにちは。']
    assert has_alignable_text('こんにちは') is True
    assert has_alignable_text('、 。！？  ') is False
    text = output.read_text(encoding='utf-8')
    assert 'こんにちは。' in text
    assert '、' in text
    manifest = json.loads((tmp_path / 'source.asr2rpp' / 'manifest.json').read_text(encoding='utf-8'))
    assert any('punctuation-only' in warning for warning in manifest['warnings'])


def test_non_destructive_export(tmp_path):
    source = tmp_path / '音声.wav'
    source.write_bytes(b'unchanged')
    output = tmp_path / 'out.rpp'
    export_rpp(source, output, [Unit(1.5, 3, 'テスト')], 10, False, 15)
    assert source.read_bytes() == b'unchanged'
    assert 'SOFFS 11.5' in output.read_text(encoding='utf-8')
    assert 'NAME "Transcript"' in output.read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        export_rpp(source, output, [Unit(0, 1, 'x')], 0, False, 15)


def test_pipeline_options_disabled(tmp_path, monkeypatch):
    source = tmp_path / 'sample.wav'
    with wave.open(str(source), 'wb') as handle:
        handle.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        handle.writeframes(b'\0\0' * 16000)
    model = Model('test', 'whisper_cpp', 'asr', {'path': str(source)})
    calls = []
    monkeypatch.setattr('asr2rpp.pipeline.executable', lambda *a: source)
    monkeypatch.setattr('asr2rpp.pipeline.ffmpeg_path', lambda *a: 'ffmpeg')
    monkeypatch.setattr('asr2rpp.pipeline.resolve_model', lambda *a: (source, {}))
    monkeypatch.setattr('asr2rpp.pipeline.decode', lambda *a, **kw: 1.0)
    def fake_infer(model, *a, **kw):
        calls.append(model.task)
        return Result([Unit(0, 0.8, 'テスト', 'built-in-speaker')], {})
    monkeypatch.setattr('asr2rpp.pipeline.infer', fake_infer)
    output = run_job(source, Settings(Stage('test')), {'test': model}, threading.Event(), lambda x: None)
    assert calls == ['asr']
    assert 'built-in-speaker' not in output.read_text(encoding='utf-8')
    manifest = json.loads((tmp_path / 'sample.asr2rpp/manifest.json').read_text())
    assert manifest['source_unchanged'] is True



def test_native_engine_work_isolated_from_unicode_report_path(tmp_path, monkeypatch):
    source_dir = tmp_path / '日本語入力'
    source_dir.mkdir()
    source = source_dir / '試験音声.wav'
    with wave.open(str(source), 'wb') as handle:
        handle.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        handle.writeframes(b'\0\0' * 16000)

    cache = tmp_path / 'engine-cache'
    monkeypatch.setenv('ASR2RPP_CACHE_DIR', str(cache))
    model = Model('test', 'whisper_cpp', 'asr', {'path': str(source)})
    monkeypatch.setattr('asr2rpp.pipeline.executable', lambda *a: source)
    monkeypatch.setattr('asr2rpp.pipeline.ffmpeg_path', lambda *a: 'ffmpeg')
    monkeypatch.setattr('asr2rpp.pipeline.resolve_model', lambda *a: (source, {}))
    monkeypatch.setattr('asr2rpp.pipeline.decode', lambda *a, **kw: 1.0)
    engine_dirs = []

    def fake_infer(_model, _weights, _audio, work, *_args, **_kwargs):
        engine_dirs.append(Path(work))
        Path(work).mkdir(parents=True, exist_ok=True)
        (Path(work) / 'engine.log').write_text('native log', encoding='utf-8')
        return Result([Unit(0, 0.8, 'test')], {})

    monkeypatch.setattr('asr2rpp.pipeline.infer', fake_infer)
    output = run_job(
        source, Settings(Stage('test')), {'test': model},
        threading.Event(), lambda _text: None)

    assert output.is_file()
    assert len(engine_dirs) == 1
    assert cache in engine_dirs[0].parents
    assert '試験音声.asr2rpp' not in str(engine_dirs[0])
    persisted = source_dir / '試験音声.asr2rpp' / 'asr' / 'engine.log'
    assert persisted.read_text(encoding='utf-8') == 'native log'


def test_auto_runtime_finds_packaged_vulkan_then_explicit_cpu(tmp_path, monkeypatch):
    engines = tmp_path / 'engines'
    suffix = '.exe' if sys.platform == 'win32' else ''
    vulkan = engines / 'whisper_cpp-vulkan' / ('whisper-cli' + suffix)
    cpu = engines / 'whisper_cpp-cpu' / ('whisper-cli' + suffix)
    vulkan.parent.mkdir(parents=True)
    cpu.parent.mkdir(parents=True)
    vulkan.write_bytes(b'vulkan')
    cpu.write_bytes(b'cpu')
    monkeypatch.setattr(adapters, 'assets_root', lambda: tmp_path)
    assert adapters.executable('whisper_cpp', 'auto') == vulkan
    assert adapters.executable('whisper_cpp', 'cpu') == cpu

def test_run_process_serializes_path_arguments_as_strings(tmp_path):
    log = tmp_path / 'command.log'
    run_process(
        [Path(sys.executable), '-c', 'print("path-argv-ok")'],
        threading.Event(), lambda _text: None, log,
    )
    first = json.loads(log.read_text(encoding='utf-8').splitlines()[0])
    assert first['command'][0] == sys.executable
    assert all(isinstance(value, str) for value in first['command'])


def test_process_cancellation(tmp_path):
    cancel = threading.Event()
    timer = threading.Timer(0.3, cancel.set)
    timer.start()
    started = time.monotonic()
    with pytest.raises(Cancelled):
        run_process([sys.executable, '-c', 'import time; time.sleep(30)'], cancel, lambda x: None, tmp_path / 'log.txt')
    timer.join()
    assert time.monotonic() - started < 8


