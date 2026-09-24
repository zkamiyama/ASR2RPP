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
from asr2rpp.catalog import load_catalog, Model, Cancelled, safe_relative, weights_root, cache_root
from asr2rpp.adapters import Unit, Result, parse_whisper, parse_audio, run_process
import asr2rpp.adapters as adapters
from asr2rpp.pipeline import Settings, Stage, reserve_output, clean_bounds, group_units, export_rpp, run_job


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


def test_bounds_preserve_raw():
    raw = [Unit(-1, 2, 'a'), Unit(3, 6, 'b'), Unit(4, 4, 'c')]
    warnings = []
    result = clean_bounds(raw, 5, warnings)
    assert raw[0].start == -1 and raw[1].end == 6
    assert result[0].start == 0 and result[-1].end == 5 and len(warnings) == 3


def test_no_false_word_timing():
    units = group_units([Unit(0, 4, '今日はいい天気です')])
    assert len(units) == 1 and units[0].granularity == 'segment'


def test_non_destructive_export(tmp_path):
    source = tmp_path / '音声.wav'
    source.write_bytes(b'unchanged')
    output = tmp_path / 'out.rpp'
    export_rpp(source, output, [Unit(1.5, 3, 'テスト')], 10, False)
    assert source.read_bytes() == b'unchanged'
    assert 'SOFFS 11.5' in output.read_text(encoding='utf-8')
    assert 'NAME "Transcript"' in output.read_text(encoding='utf-8')
    with pytest.raises(FileExistsError):
        export_rpp(source, output, [Unit(0, 1, 'x')], 0, False)


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

def test_process_cancellation(tmp_path):
    cancel = threading.Event()
    timer = threading.Timer(0.3, cancel.set)
    timer.start()
    started = time.monotonic()
    with pytest.raises(Cancelled):
        run_process([sys.executable, '-c', 'import time; time.sleep(30)'], cancel, lambda x: None, tmp_path / 'log.txt')
    timer.join()
    assert time.monotonic() - started < 8


def test_gui_states_and_screenshots(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    monkeypatch.setenv('ASR2RPP_HOME', str(tmp_path / 'home'))
    from PySide6.QtWidgets import QApplication, QDialog, QLineEdit
    from PySide6.QtCore import QSettings, QTimer
    from asr2rpp.gui import MainWindow, STYLE, default_storage_hint
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope, str(tmp_path / 'settings'))
    legacy = QSettings('ASR2RPP', 'ASR2RPP')
    legacy.setValue('asr/device', 'cpu')
    legacy.sync()
    window = MainWindow()
    window.diar.toggle.setChecked(False)
    window.align.toggle.setChecked(False)
    window.same.setChecked(True)
    window.show()
    for name in ['interview.wav', 'conversation.mp4', 'narration.flac']:
        path = tmp_path / name
        path.write_bytes(b'test fixture')
        window.add_paths([str(path)])
    app.processEvents()
    assert not window.diar.body.isEnabled()
    assert not window.align.body.isEnabled()
    assert not window.output_dir.isEnabled()
    assert not window.output_browse.isEnabled()
    assert window.asr.body.isEnabled()
    assert window.runtime_button.text() == '⚙'
    assert window.runtime_button.accessibleName() == '設定'
    assert window.asr.device.currentData() == 'default'
    assert window.queue_strategy == 'stage'
    assert window.model_storage_dir == '' and window.temp_storage_dir == ''
    assert window.clip_length.maximum() >= 3600
    assert window.asr.param_summary.text()
    if sys.platform == 'win32':
        assert '%LOCALAPPDATA%' in default_storage_hint('weights')
        assert '%LOCALAPPDATA%' in default_storage_hint('cache')
    window.runtime_defaults['whisper_cpp'] = 'vulkan'
    assert window.asr.stage({}, 4, window.runtime_defaults).device == 'vulkan'
    window.update_summary()
    assert 'whisper.cpp Vulkan' in window.summary.text()
    window.runtime_defaults['whisper_cpp'] = 'cpu'
    window.runtime_defaults['audio_cpp'] = 'vulkan'
    window.update_summary()
    assert 'whisper.cpp CPU' in window.summary.text()
    assert 'audio.cpp Vulkan' in window.summary.text()
    audio_models = [m for m in window.catalog.values() if m.runtime == 'audio_cpp']
    if audio_models:
        window.diar.toggle.setChecked(True)
        audio_index = window.diar.model.findData(next((m.id for m in audio_models if m.task == 'diar'), ''))
        if audio_index >= 0:
            window.diar.model.setCurrentIndex(audio_index)
            assert window.diar.stage({}, 4, window.runtime_defaults).device == 'vulkan'
    assert len(window.entries) == 3
    window.add_paths([str(tmp_path / 'interview.wav')])
    assert len(window.entries) == 3
    reports = Path('reports')
    reports.mkdir(exist_ok=True)
    assert window.align.y() < window.diar.y()
    window.grab().save(str(reports / 'gui-options-off.png'))
    window.diar.toggle.setChecked(True)
    assert window.diar.body.isEnabled() and not window.align.body.isEnabled()
    window.align.toggle.setChecked(True)
    window.same.setChecked(False)
    window.output_dir.clear()
    assert not window.start_button.isEnabled()
    window.output_dir.setText(str(tmp_path / 'output'))
    assert window.start_button.isEnabled()
    assert window.align.body.isEnabled() and window.output_dir.isEnabled()
    app.processEvents()
    window.grab().save(str(reports / 'gui-options-on.png'))

    settings_state = {}
    def capture_settings():
        dialogs = [w for w in app.topLevelWidgets()
                   if isinstance(w, QDialog) and w.windowTitle() == '設定']
        assert dialogs
        dialog = dialogs[-1]
        placeholders = [edit.placeholderText() for edit in dialog.findChildren(QLineEdit)]
        settings_state['placeholders'] = placeholders
        dialog.grab().save(str(reports / 'settings-dialog.png'))
        dialog.reject()
    QTimer.singleShot(0, capture_settings)
    window.runtime_dialog()
    assert any('weights' in text.lower() for text in settings_state['placeholders'])
    assert any('cache' in text.lower() for text in settings_state['placeholders'])
    if sys.platform == 'win32':
        assert sum('%LOCALAPPDATA%' in text for text in settings_state['placeholders']) >= 2

    window.set_busy(True)
    assert not window.diar.isEnabled() and not window.start_button.isEnabled()
    window.set_busy(False)
    window.close()
    app.processEvents()
