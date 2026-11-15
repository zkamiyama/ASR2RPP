"""Pinned Nemotron Transcribe definition; no downloads or GPU claims."""
from pathlib import Path
import json
import pytest
from asr2rpp.catalog import load_catalog
from asr2rpp.adapters import split_engine_parameters, audio_session_args
from asr2rpp.native_profile import profile
from asr2rpp.parameter_specs import specs_for

ROOT = Path(__file__).resolve().parents[1]


def model():
    catalog, errors = load_catalog(ROOT / 'models')
    assert not errors
    return catalog['nemotron-transcribe-iq4-nl']


def test_transcribe_is_distinct_offline_family_with_two_pinned_files():
    m = model()
    assert m.family == 'nemotron_transcribe'
    assert m.runtime == 'audio_cpp' and m.sample_rate == 16000
    assert m.source['revision'] == '304032f14e1e9e4042f5491d2d0194c044710bb7'
    assert m.source['entry'] == 'nemotron-transcribe-30b-a4b-experts-IQ4_NL.gguf'
    assert len(m.source['files']) == 2
    assert m.source['files'][1].startswith('mmproj-')
    assert set(m.source['sha256']) == set(m.source['files'])
    assert profile(m).mode == 'offline'
    assert profile(m).output == 'custom'
    assert profile(m).max_batch_items == 1
    assert profile(m).max_audio_seconds == 3600


def test_runtime_pin_and_family_are_version_correct():
    data = json.loads((ROOT / 'native/versions.json').read_text())['audio_cpp']
    assert data['commit'] == '61c7838baf1f7f34769b5b7bd369c0d78edf9535'
    assert 'nemotron_transcribe' in data['models']
    assert 'nemotron_asr' in data['models']


def test_transcribe_defaults_preserve_timing_and_bound_expert_storage():
    m = model()
    request, session = split_engine_parameters(m, {})
    assert request['return_timestamps'] is False and request['diarization'] is False
    assert 'max_tokens' not in request  # native duration budget, not a short cap
    assert session['moe_disk_experts'] is True
    assert session['weight_type'] == 'native'
    assert session['moe_expert_cache_size'] == 8
    assert session['prefill_batch_size'] == 64
    assert session['moe_pin_experts'] is False
    argv = audio_session_args(m, session)
    assert 'moe_disk_experts=true' in argv
    assert 'nemotron_transcribe.moe_disk_experts=true' not in argv
    assert 'nemotron_transcribe.weight_type=native' in argv
    assert 'nemotron_transcribe.prefill_batch_size=64' in argv


def test_transcribe_typed_offload_controls_are_available_and_validated():
    m = model()
    specs = {s['key']: s for s in specs_for(m)}
    assert specs['session.moe_expert_cache_size']['min'] == 6
    assert specs['session.moe_cpu_expert_layers']['max'] == 23
    for key, value in [('session.moe_expert_cache_size', 5),
                       ('session.moe_cpu_expert_layers', 24),
                       ('session.prefill_batch_size', 2049),
                       ('session.weight_type', 'f32')]:
        with pytest.raises(ValueError, match='Invalid parameter'):
            split_engine_parameters(m, {key: value})
