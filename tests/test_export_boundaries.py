"""Native resampling must not put edit clips beyond the original WAV tail."""
from fractions import Fraction
import pytest
from asr2rpp.domain import Unit
from asr2rpp.export_timeline import build_timeline


@pytest.mark.parametrize('origin', [0, 90])
def test_submillisecond_resampling_tail_is_capped_not_guessed(tmp_path, origin):
    duration = Fraction(691856, 44100)
    rounded = 15.688375
    units = [Unit(10.24, rounded, 'unchanged recognition')]
    timeline = build_timeline(tmp_path/'source.wav', units, origin, origin, False,
                              reference_duration=duration)
    clip = timeline['tracks'][1]['clips'][0]
    assert clip['reference_boundary_clamped']
    assert clip['source_start_seconds'] + clip['duration_seconds'] == pytest.approx(float(duration))
    assert clip['start_seconds'] == pytest.approx(origin+10.24)
    assert units[0].end == rounded
    assert timeline['tracks'][0]['clips'][0]['duration_seconds'] == float(duration)


def test_materially_wrong_reference_duration_remains_an_error(tmp_path):
    with pytest.raises(ValueError, match='outside'):
        build_timeline(tmp_path/'source.wav', [Unit(0, 2.1, 'bad mapping')], 0, 0, False,
                       reference_duration=2)
