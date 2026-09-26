"""Model policy tests deliberately use arbitrary model IDs (no name special cases)."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import pytest

from asr2rpp.catalog import Model, load_catalog
from asr2rpp.inference_policy import policy_for, constrain_parameters, ui_parameters
from asr2rpp.pipeline import Settings, Stage

POLICY = {'segmentation': 'vad', 'timestamps': False, 'history': False, 'max_segment_seconds': 25.0}


def constrained(**kwargs):
    data = dict(id='unrelated-custom-model', runtime='whisper_cpp', task='asr', source={'path': 'model.bin'},
                constraints={'disabled_parameters': ['initial_prompt', 'carry_initial_prompt'], 'inference': dict(POLICY)})
    data.update(kwargs)
    return Model(**data)


def test_model_id_independent_policy_and_normal_models_unchanged():
    model = constrained()
    model.validate()
    assert policy_for(model).uses_vad_timing
    assert constrain_parameters(model, {}) == {'vad': True, 'no_timestamps': True, 'max_context': 0}
    assert constrain_parameters(replace(model, id='another-id'), {}) == constrain_parameters(model, {})
    legacy = replace(model, constraints={})
    assert constrain_parameters(legacy, {'beam_size': 5}) == {'beam_size': 5}
    assert not policy_for(legacy).uses_vad_timing


@pytest.mark.parametrize('change', [
    {'segmentation': 'silence-delete'}, {'timestamps': 'false'}, {'history': 0},
    {'history': True}, {'timestamps': True}, {'max_segment_seconds': True},
    {'max_segment_seconds': float('nan')}, {'max_segment_seconds': float('inf')},
    {'max_segment_seconds': 30}, {'max_segment_seconds': 0}, {'typo': False},
])
def test_invalid_policy_fails_closed(change):
    with pytest.raises(ValueError):
        constrained(constraints={'inference': dict(POLICY, **change)}).validate()


@pytest.mark.parametrize('change', [
    {'runtime': 'audio_cpp', 'family': 'nemotron_asr'}, {'sample_rate': 48000},
    {'constraints': {'inference': []}}, {'constraints': {'inference': {'timestamps': False}}},
    {'constraints': {'inference': {'max_segment_seconds': 25}}},
    {'constraints': {'disabled_parameters': ['vad'], 'inference': POLICY}},
])
def test_unsupported_policy_is_not_silently_ignored(change):
    with pytest.raises(ValueError):
        constrained(**change).validate()


@pytest.mark.parametrize('override', [
    {'vad': False}, {'vad': 1}, {'max_context': -1}, {'max_context': 0.0},
    {'no_timestamps': False}, {'vad_max_speech_duration_s': 40},
    {'processors': 2}, {'translate': True},
])
def test_conflicting_overrides_fail_before_native_launch(override):
    model = constrained()
    alignment = Model('align', 'audio_cpp', 'align', {'path': 'align.gguf'}, family='qwen3_forced_aligner')
    settings = Settings(Stage(model.id, parameters=override), align=Stage('align'))
    with pytest.raises(ValueError):
        settings.validate({model.id: model, 'align': alignment})


def test_vad_timing_without_alignment_and_gui_stale_values_removed():
    model = constrained()
    Settings(Stage(model.id)).validate({model.id: model})
    saved = dict(vad=False, max_context=-1, no_timestamps=False, initial_prompt='old', beam_size=3)
    assert ui_parameters(model, saved) == {'beam_size': 3}
    assert constrain_parameters(model, ui_parameters(model, saved))['max_context'] == 0


def test_literal_toml_custom_model_controls_runtime(tmp_path):
    (tmp_path/'custom.toml').write_text('''runtime="whisper_cpp"
task="asr"
[source]
path="weights.bin"
[constraints.inference]
segmentation="vad"
timestamps=false
history=false
max_segment_seconds=12.0
''', encoding='utf-8')
    models, errors = load_catalog(tmp_path)
    assert not errors
    assert policy_for(models['custom']).max_segment_seconds == 12
    assert policy_for(models['custom']).uses_vad_timing


def test_model_directory_never_copies_or_rewrites_templates(tmp_path, monkeypatch):
    import asr2rpp.catalog as catalog
    assets, home = tmp_path/'assets', tmp_path/'home'
    (assets/'models').mkdir(parents=True); (home/'models').mkdir(parents=True)
    (assets/'models'/'a.toml').write_text('new shipped definition')
    (home/'models'/'a.toml').write_text('older user file')
    monkeypatch.setattr(catalog, 'assets_root', lambda: assets)
    monkeypatch.setattr(catalog, 'data_root', lambda: home)
    assert catalog.model_directory() == assets/'models'
    assert (home/'models'/'a.toml').read_text() == 'older user file'
    assert not list(home.rglob('*.bak')) and not list(home.rglob('*.tmp'))


def test_gui_policy_locks_values_but_alignment_remains_optional(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from asr2rpp.gui_dcc import MainWindow, ParameterDialog
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    monkeypatch.setenv('ASR2RPP_HOME', str(tmp_path/'home'))
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP','1')
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path/'prefs'))
    app=QApplication.instance() or QApplication([])
    model=constrained()
    dialog=ParameterDialog(model, {'vad':False,'max_context':-1}, 'ja')
    for key in ('vad','no_timestamps','max_context','processors','translate'):
        control,spec=dialog.controls[key]
        assert not control.isEnabled() and spec['locked']
    assert dialog.controls['vad'][0].isChecked()
    assert dialog.controls['no_timestamps'][0].isChecked()
    assert dialog.controls['max_context'][0].value() == 0
    assert 'initial_prompt' not in dialog.controls
    assert not (set(dialog.result_parameters()) & {'vad','max_context','no_timestamps'})
    window=MainWindow()
    window.asr.model.setCurrentIndex(window.asr.model.findData('whisper-base'))
    window.align.toggle.setChecked(False)
    window.catalog[model.id]=model
    window.asr.set_catalog(window.catalog)
    window.asr.model.setCurrentIndex(window.asr.model.findData(model.id))
    app.processEvents()
    assert not window.align.toggle.isChecked() and window.align.toggle.isEnabled()
    assert window.asr.policy_label.text().startswith('TOML')
    window.align.toggle.setChecked(True)
    assert window.align.toggle.isChecked()
    window.align.toggle.setChecked(False)
    assert not window.align.toggle.isChecked()
    window.asr.model.setCurrentIndex(window.asr.model.findData('whisper-base'))
    assert window.align.toggle.isEnabled() and not window.align.toggle.isChecked()
    dialog.close(); window.close()
