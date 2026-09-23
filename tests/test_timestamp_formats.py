import pytest
from asr2rpp.adapters import parse_audio


def test_sample_offsets_are_not_seconds():
    raw = [{'start_sample': 16000, 'end_sample': 32000, 'speaker_id': 'speaker_0'}]
    result = parse_audio(raw, 'diar', 'nemotron_3_diar', 16000)
    assert result.units[0].start == 1 and result.units[0].end == 2
    assert result.units[0].speaker == 'speaker_0'
    assert raw[0]['start_sample'] == 16000


def test_sample_offsets_need_rate():
    with pytest.raises(ValueError, match='sample rate'):
        parse_audio([{'start_sample': 1, 'end_sample': 2}], 'align', 'qwen3_forced_aligner')


def test_24k_segment():
    raw = [{'start_sample': 12000, 'end_sample': 36000, 'text': 'test'}]
    unit = parse_audio(raw, 'asr', 'vibevoice_asr', 24000).units[0]
    assert unit.start == 0.5 and unit.end == 1.5


def test_aligner_samples():
    raw = [{'start_sample': 8000, 'end_sample': 16000, 'word': 'hello'}]
    unit = parse_audio(raw, 'align', 'qwen3_forced_aligner', 16000).units[0]
    assert unit.start == 0.5 and unit.method == 'forced_alignment'


def test_invalid_structure_is_not_silently_accepted():
    with pytest.raises(ValueError):
        parse_audio({'unknown': []}, 'asr', 'other', 16000)
