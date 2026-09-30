"""The shipped first-run experience is Anime Whisper / ja, never a saved override."""
import json
from pathlib import Path
import pytest


@pytest.mark.parametrize('extra,expected,language', [([], 'anime-whisper', 'ja'),
    (['--asr','whisper-base','--asr-language','en'], 'whisper-base', 'en')])
def test_cli_initial_model_and_explicit_override(tmp_path, monkeypatch, extra, expected, language):
    from asr2rpp import cli, providers
    monkeypatch.setenv('ASR2RPP_HOME', str(tmp_path/'home'))
    monkeypatch.setattr(providers, 'preflight', lambda *a, **kw: {})
    observed = []
    def run(jobs, settings, *args):
        observed.append((settings.asr.model_id, settings.asr.language, settings.output_formats))
        return {0:tmp_path/'result.rpp'}
    monkeypatch.setattr(cli,'run_queue',run)
    assert cli.main(['run',str(tmp_path/'voice.wav'),'--asr-device','cpu',*extra]) == 0
    assert observed == [(expected, language, ('rpp',))]


def test_first_run_gui_and_saved_choice(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    monkeypatch.setenv('ASR2RPP_HOME',str(tmp_path/'home'))
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP','1')
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from asr2rpp.gui_dcc import MainWindow
    app=QApplication.instance() or QApplication([])
    prefs=QSettings(str(tmp_path/'preferences.ini'),QSettings.Format.IniFormat)
    window=MainWindow(prefs)
    try:
        config=window.current_settings()
        assert config.asr.model_id=='anime-whisper' and config.asr.language=='ja'
        assert config.output_formats==('rpp',)
        window.asr.model.setCurrentIndex(window.asr.model.findData('whisper-base'))
        window.asr.language.setCurrentText('en')
        window.save_preferences()
    finally:
        window.close();app.processEvents()
    restored=MainWindow(prefs)
    try:
        restored.reload_catalog()
        config=restored.current_settings()
        assert config.asr.model_id=='whisper-base' and config.asr.language=='en'
    finally:
        restored.close();app.processEvents()


def test_representative_catalog_validation_requests_every_export(tmp_path):
    from tools.validate_catalog import case_arguments
    _source,_device,args=case_arguments('anime-whisper',
        {'task':'asr','provider':'whisper_cpp','defaults':{'language':'ja'}},
        'vulkan',{'ja':tmp_path/'ja.wav','en':tmp_path/'en.wav'},tmp_path/'output')
    assert args[args.index('--format')+1]=='rpp,otio,json'
