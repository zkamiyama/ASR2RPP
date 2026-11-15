"""Shipped source catalog selection and document-copy checks, not frozen QA."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from asr2rpp.catalog import load_catalog
from asr2rpp.native_profile import profile

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = 'nemotron-transcribe-iq4-nl'


def catalog():
    models, errors = load_catalog(ROOT / 'models')
    assert not errors
    assert len(models) == 21
    return models


def test_exact_30b_definition_is_shipped_and_default_off():
    models = catalog()
    model = models[MODEL_ID]
    assert model.definition.resolve() == ROOT / 'models' / f'{MODEL_ID}.toml'
    assert model.family == 'nemotron_transcribe'
    assert models['nemotron-asr'].family != model.family
    assert models['nemotron-diarization'].task == 'diar'
    assert model.source['repo'] == 'Jmica/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF'
    assert model.source['revision'] == '304032f14e1e9e4042f5491d2d0194c044710bb7'
    assert model.source['entry'] == 'nemotron-transcribe-30b-a4b-experts-IQ4_NL.gguf'
    assert model.defaults['request']['diarization'] is False
    assert model.defaults['request']['return_timestamps'] is False
    assert model.capabilities['glossary']['adapter'] == 'nemotron_instruct_v1'
    assert model.capabilities['glossary']['readings'] == 'experimental'
    assert profile(model).max_batch_items == 1
    assert model.execution['text_batch_items'] == 16
    assert not models['anime-whisper'].capabilities.get('glossary')
    assert len([m for m in models.values() if m.runtime == 'whisper_cpp' and m.capabilities.get('glossary')]) == 6


def test_cli_models_list_includes_exact_30b_source_without_acquisition(monkeypatch, capsys):
    from asr2rpp import cli
    monkeypatch.setattr(cli, 'load_catalog', lambda: (catalog(), []))
    monkeypatch.setattr(cli, 'resolve_model', lambda *a, **k: pytest.fail('Listing must not acquire weights'))
    assert cli.main(['models', 'list', '--json']) == 0
    records = json.loads(capsys.readouterr().out)
    assert len(records) == 21
    selected = next(row for row in records if row['id'] == MODEL_ID)
    assert selected['task'] == 'asr'
    assert Path(selected['definition']).resolve() == ROOT / 'models' / f'{MODEL_ID}.toml'
    assert selected['sha256'] == hashlib.sha256(Path(selected['definition']).read_bytes()).hexdigest()


def test_gui_can_select_30b_without_enabling_native_speakers(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    monkeypatch.setenv('ASR2RPP_HOME', str(tmp_path / 'home'))
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP', '1')
    monkeypatch.delenv('ASR2RPP_WEIGHTS_DIR', raising=False)
    monkeypatch.delenv('ASR2RPP_CACHE_DIR', raising=False)
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from asr2rpp.gui_dcc import MainWindow, PreferencesDialog
    from asr2rpp.timing import asr_parameters
    app = QApplication.instance() or QApplication([])
    prefs = QSettings(str(tmp_path / 'prefs.ini'), QSettings.Format.IniFormat)
    window = MainWindow(preferences=prefs)
    try:
        index = window.asr.model.findData(MODEL_ID)
        assert index >= 0
        window.asr.model.setCurrentIndex(index)
        settings = window.current_settings()
        assert settings.asr.model_id == MODEL_ID
        assert settings.timing.speaker_source == 'auto'
        parameters = asr_parameters(window.catalog[MODEL_ID], settings.timing,
                                    settings.align is not None, settings.asr.parameters)
        assert parameters['diarization'] is False
        dialog = PreferencesDialog(window)
        assert dialog.speaker_source.currentData() == 'auto'
        assert '4096' in dialog.terms_notice.text()
        dialog.close()
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize('platform', ['windows', 'macos'])
def test_packager_document_copy_preserves_21_tomls_and_new_docs(tmp_path, platform):
    """Execute only packager data-copy code, never PyInstaller/native/download steps."""
    package = tmp_path / 'resources'
    package.mkdir()
    source = ROOT / 'tools' / f'package_{platform}.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    namespace = {'ROOT': ROOT, 'shutil': shutil, 'package': package,
                 'distribution': lambda _: SimpleNamespace(files=[]), 'Path': Path}
    if platform == 'macos':
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'copy_documents')
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        namespace['copy_documents'](package)
    else:
        nodes = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                 and ast.unparse(n.value.func) == 'shutil.copytree'
                 and ast.unparse(n.value.args[0]) in ("ROOT / 'models'", "ROOT / 'docs'")]
        assert len(nodes) == 2
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    copied, errors = load_catalog(package / 'models')
    assert not errors and len(copied) == 21 and MODEL_ID in copied
    for path in (ROOT / 'models').glob('*.toml'):
        assert (package / 'models' / path.name).read_bytes() == path.read_bytes()
    for name in ('glossary.md', 'nemotron-transcribe.md', 'models.md'):
        assert (package / 'docs' / name).read_bytes() == (ROOT / 'docs' / name).read_bytes()
