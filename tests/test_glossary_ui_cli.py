"""Glossary controls pass invocation data without executing inference."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('newline', ['\n', '\r\n', '\r'])
def test_cli_terms_file_flows_into_invocation(tmp_path, monkeypatch, newline):
    from asr2rpp import cli, providers, timing
    text = newline.join(('ASR2RPP\tえーえすあーる', 'Nemotron', ''))
    terms = tmp_path / 'terms.txt'
    terms.write_bytes(('\ufeff' + text).encode('utf-8'))
    seen = []
    monkeypatch.setattr(cli.Settings, 'validate', lambda *a: None)
    monkeypatch.setattr(timing, 'plan_for', lambda *a: SimpleNamespace(segmented=False))
    monkeypatch.setattr(providers, 'preflight', lambda *a, **k: {})
    def run(jobs, settings, *args):
        seen.append(settings)
        return {0: tmp_path / 'result.rpp'}
    monkeypatch.setattr(cli, 'run_queue', run)
    assert cli.main(['run', str(tmp_path / 'input.wav'), '--terms-file', str(terms)]) == 0
    assert seen[0].terms_text == text


def test_cli_invalid_utf8_fails_before_preflight(tmp_path, monkeypatch, capsys):
    from asr2rpp import cli, providers
    terms = tmp_path / 'invalid.txt'
    terms.write_bytes(b'\xff')
    calls = []
    monkeypatch.setattr(providers, 'preflight', lambda *a, **k: calls.append(True))
    monkeypatch.setattr(cli, 'run_queue', lambda *a, **k: pytest.fail('Must not run inference'))
    assert cli.main(['run', 'input.wav', '--terms-file', str(terms)]) == 2
    assert not calls
    assert capsys.readouterr().err


@pytest.fixture
def ui(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    monkeypatch.setenv('ASR2RPP_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP', '1')
    monkeypatch.delenv('ASR2RPP_WEIGHTS_DIR', raising=False)
    monkeypatch.delenv('ASR2RPP_CACHE_DIR', raising=False)
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from asr2rpp.gui_dcc import MainWindow
    app = QApplication.instance() or QApplication([])
    prefs = QSettings(str(tmp_path / 'settings.ini'), QSettings.Format.IniFormat)
    prefs.setValue('ui/language', 'en')
    window = MainWindow(preferences=prefs)
    yield window, app, prefs
    window.close()
    app.processEvents()


def test_glossary_paste_import_and_failed_import_preserves_text(ui, tmp_path, monkeypatch):
    from asr2rpp.gui_dcc import PreferencesDialog, QFileDialog, QMessageBox
    window, app, _ = ui
    window.terms_text = 'Original'
    dialog = PreferencesDialog(window)
    assert dialog.nav.count() == 5
    assert dialog.terms_edit.toPlainText() == 'Original'
    assert dialog.speaker_source.currentData() == 'auto'
    assert 'does not support' in dialog.terms_notice.text()  # Anime Whisper prohibits prompts.
    terms = tmp_path / 'terms.txt'
    terms.write_text('\ufeff表記\tひょうき\nASR2RPP', encoding='utf-8')
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **k: (str(terms), ''))
    dialog._import_terms()
    assert dialog.terms_edit.toPlainText() == '表記\tひょうき\nASR2RPP'
    terms.write_bytes(b'\xff')
    errors = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: errors.append(a[-1]))
    dialog._import_terms()
    assert errors
    assert dialog.terms_edit.toPlainText() == '表記\tひょうき\nASR2RPP'
    dialog.reject()
    assert window.terms_text == 'Original'


def test_saved_glossary_is_run_data_not_global_preference(ui, monkeypatch):
    from asr2rpp.gui_dcc import PreferencesDialog
    window, app, prefs = ui
    text = 'Nemotron\tねもとろん'
    def accept_with_changes(dialog):
        dialog.terms_edit.setPlainText(text)
        dialog.speaker_source.setCurrentIndex(dialog.speaker_source.findData('native'))
        return 1
    monkeypatch.setattr(PreferencesDialog, 'exec', accept_with_changes)
    window.open_settings()
    assert window.terms_text == text
    assert window.current_settings().terms_text == text
    assert window.timing_settings.speaker_source == 'native'
    window.save_preferences()
    assert not any('terms' in key or 'glossary' in key for key in prefs.allKeys())


def test_gui_unsupported_glossary_blocks_model_preparation(ui, monkeypatch):
    from asr2rpp import gui_dcc
    window, app, _ = ui
    window.terms_text = 'Nemotron'
    errors = []
    monkeypatch.setattr(gui_dcc.QMessageBox, 'warning', lambda *a: errors.append(a[-1]))
    monkeypatch.setattr(gui_dcc, 'Worker', lambda *a, **k: pytest.fail('Must validate before download worker'))
    window.start_work(prepare_only=True)
    assert errors
    assert window.worker is None


@pytest.mark.parametrize('prepare_only', [False, True])
def test_direct_worker_rejects_terms_before_any_probe_or_download(monkeypatch, prepare_only):
    pytest.importorskip('PySide6')
    from asr2rpp import gui_dcc, providers
    from asr2rpp.catalog import load_catalog
    from asr2rpp.pipeline import Stage
    from asr2rpp.preprocessing import Settings
    catalog, errors = load_catalog(Path(__file__).resolve().parents[1] / 'models')
    assert not errors
    settings = Settings(Stage('anime-whisper'), terms_text='Unsupported',
                        same_directory=not prepare_only, output_formats=() if prepare_only else ('rpp',))
    monkeypatch.setattr(providers, 'preflight', lambda *a, **k: pytest.fail('Must validate before probing'))
    monkeypatch.setattr(gui_dcc, 'resolve_model', lambda *a, **k: pytest.fail('Must validate before acquisition'))
    worker = gui_dcc.Worker([], settings, catalog, prepare_only=prepare_only)
    with pytest.raises(ValueError, match='unsupported'):
        worker._prepare_models()
    assert worker.settings.same_directory == settings.same_directory
    assert worker.settings.output_formats == settings.output_formats


def test_prepare_only_worker_ignores_unselected_output_without_mutating_settings(monkeypatch):
    pytest.importorskip('PySide6')
    from asr2rpp import gui_dcc, providers
    from asr2rpp.catalog import load_catalog
    from asr2rpp.pipeline import Stage
    from asr2rpp.preprocessing import Settings
    catalog, errors = load_catalog(Path(__file__).resolve().parents[1] / 'models')
    assert not errors
    settings = Settings(Stage('whisper-base'), same_directory=False, output_formats=())
    calls = []
    monkeypatch.setattr(providers, 'preflight', lambda *a, **k: {})
    monkeypatch.setattr(gui_dcc, 'resolve_model', lambda *a, **k: calls.append(a[0].id))
    worker = gui_dcc.Worker([], settings, catalog, prepare_only=True)
    worker._prepare_models()
    assert calls == ['whisper-base']
    assert worker.settings.same_directory is False
    assert worker.settings.output_formats == ()
