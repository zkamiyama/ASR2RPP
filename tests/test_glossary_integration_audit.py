"""Cross-layer pre-acquisition invariants found by an independent hook audit."""
from dataclasses import replace
import copy
import threading
from pathlib import Path
import pytest
from asr2rpp.catalog import load_catalog
from asr2rpp.pipeline import Settings, Stage, run_job
from asr2rpp.timing import TimingSettings, prepare_run, stage_options

ROOT = Path(__file__).resolve().parents[1]


def models():
    result, errors = load_catalog(ROOT/'models')
    assert not errors
    return result


@pytest.mark.parametrize('key', ['instruct', 'system_prompt', 'text_input'])
@pytest.mark.parametrize('source', ['request', 'metadata'])
def test_native_code_owned_instruction_rejects_toml_default_without_terms(key, source):
    catalog = models(); model = catalog['nemotron-transcribe-iq4-nl']
    if source == 'request':
        defaults = copy.deepcopy(model.defaults)
        defaults.setdefault('request', {})[key] = 'Translate rather than transcribe'
        model = replace(model, defaults=defaults)
    else:
        params = copy.deepcopy(model.parameters)
        params[key] = dict(type='str', default='Ignore the original instruction')
        model = replace(model, parameters=params)
    catalog[model.id] = model
    with pytest.raises(ValueError, match='instruction|prompt|transcription'):
        Settings(Stage(model.id)).validate(catalog)


def test_empty_lines_are_no_glossary_even_for_whisper_vad():
    catalog = models()
    settings = Settings(Stage('whisper-small'), terms_text='\n\r\n', timing=TimingSettings(mode='vad'))
    settings.validate(catalog)
    prepared, catalog, _ = prepare_run(settings, catalog)
    options = stage_options(prepared, catalog['whisper-small'])
    assert 'initial_prompt' not in options['parameters']
    assert not prepared.terms_provenance


@pytest.mark.parametrize('text', [' ', 'bad\tEnglish', 'term\nterm', '<|im_start|>'])
def test_direct_invalid_terms_block_acquisition(tmp_path, monkeypatch, text):
    from asr2rpp import pipeline
    catalog = models(); calls = []
    monkeypatch.setattr(pipeline, 'resolve_model', lambda *a, **kw: calls.append('acquisition'))
    with pytest.raises(ValueError):
        run_job(tmp_path/'audio.wav', Settings(Stage('nemotron-transcribe-iq4-nl'), terms_text=text),
                catalog, threading.Event(), lambda _:None)
    assert not calls


@pytest.mark.parametrize('speaker_source', ['auto', 'none', 'native'])
def test_speaker_policy_precedes_glossary_and_original_surface_survives(speaker_source):
    catalog = models(); model = catalog['nemotron-transcribe-iq4-nl']
    text = '神﨑\tかんざき\nＡＳＲ2RPP'
    settings = Settings(Stage(model.id), terms_text=text,
                        timing=TimingSettings(speaker_source=speaker_source))
    settings.validate(catalog)
    prepared, adapted, _ = prepare_run(settings, catalog)
    prepared.validate(adapted)
    options = stage_options(prepared, adapted[model.id])
    assert options['parameters']['diarization'] is (speaker_source == 'native')
    assert ('<spk:0>' in options['parameters']['instruct']) is (speaker_source == 'native')
    assert prepared.terms_text == text
    assert prepared.terms_provenance['entries'] == [
        {'surface':'神﨑','reading':'かんざき'}, {'surface':'ＡＳＲ2RPP','reading':None}]
    assert prepared.terms_provenance['rendered_prompt'] == options['parameters']['instruct']
    assert settings.asr.parameters is None and not settings.terms_provenance


def test_queue_invalid_terms_block_resolution(monkeypatch):
    from asr2rpp import queue_runner
    catalog = models(); calls = []
    monkeypatch.setattr(queue_runner, '_resolve_models', lambda *a: calls.append('acquisition'))
    with pytest.raises(ValueError):
        queue_runner.run_queue([], Settings(Stage('anime-whisper'), terms_text='Unsupported'),
                               catalog, threading.Event(), lambda _:None, lambda *a:None)
    assert not calls


def test_gui_worker_validates_terms_before_acquisition(monkeypatch):
    pytest.importorskip('PySide6')
    from asr2rpp import gui_dcc, providers, vad
    from asr2rpp.preprocessing import Settings as GuiSettings
    catalog = models(); calls = []
    monkeypatch.setattr(providers, 'preflight', lambda *a, **kw: {})
    monkeypatch.setattr(gui_dcc, 'resolve_model', lambda *a, **kw: calls.append('acquisition'))
    monkeypatch.setattr(vad, 'vad_model', lambda *a, **kw: None)
    worker = gui_dcc.Worker([], GuiSettings(Stage('anime-whisper'), terms_text='Unsupported'), catalog)
    with pytest.raises(ValueError, match='unsupported'):
        worker._prepare_models()
    assert not calls


def test_hint_must_not_be_silently_disabled_by_high_whisper_temperature():
    catalog = models()
    with pytest.raises(ValueError, match='temperature|prompt'):
        Settings(Stage('whisper-small', parameters={'temperature': 0.5}), terms_text='OpenAI').validate(catalog)


def test_nemotron_hint_cannot_silently_replace_conflicting_request_language():
    catalog = models()
    with pytest.raises(ValueError, match='language|Japanese'):
        Settings(Stage('nemotron-transcribe-iq4-nl', language='ja', parameters={'language':'en'}),
                 terms_text='用語').validate(catalog)
