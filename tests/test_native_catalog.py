"""Shipped native definitions must be reproducible and compiled into the pack."""
from pathlib import Path
import json
import re
import tomllib
from asr2rpp.catalog import load_catalog
from asr2rpp.native_profile import profile

ROOT = Path(__file__).resolve().parents[1]


def test_every_shipped_remote_file_is_revision_and_sha_pinned():
    catalog, errors = load_catalog(ROOT / 'models')
    assert not errors
    for model in catalog.values():
        assert re.fullmatch(r'[0-9a-f]{40}', model.source['revision']), model.id
        for name in model.source['files']:
            assert re.fullmatch(r'[0-9a-f]{64}', model.source['sha256'][name]), (model.id, name)


def test_representative_native_families_are_compiled():
    catalog, errors = load_catalog(ROOT / 'models')
    assert not errors
    compiled = set(json.loads((ROOT / 'native/versions.json').read_text())['audio_cpp']['models'])
    aliases = {'mel_band_roformer': 'roformer'}
    families = {aliases.get(m.family, m.family) for m in catalog.values() if m.runtime == 'audio_cpp'}
    assert families <= compiled
    assert {'canary_asr', 'moonshine_asr', 'fun_asr_nano', 'voxtral_realtime', 'sense_asr'} <= families


def test_text_only_native_models_request_text_not_fake_word_intervals():
    catalog, _ = load_catalog(ROOT / 'models')
    for model in catalog.values():
        if model.runtime == 'audio_cpp' and model.task == 'asr' and model.capabilities.get('timestamps') == 'none':
            assert profile(model).output == 'text', model.id
            assert 0 < profile(model).max_audio_seconds <= 28, model.id
            assert 'inference' not in model.constraints, model.id


def test_english_only_models_do_not_default_to_japanese():
    catalog, _ = load_catalog(ROOT / 'models')
    for name in ('canary-180m-flash', 'moonshine-streaming-tiny'):
        assert catalog[name].defaults['language'] == 'en'


def test_catalog_has_no_faster_whisper_standard_dependency():
    catalog, _ = load_catalog(ROOT / 'models')
    assert all(m.runtime != 'faster_whisper' for m in catalog.values())
