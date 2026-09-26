"""VAD times are region intervals, never invented word alignment."""
import json
import threading
from pathlib import Path
import pytest
from asr2rpp.vad import SpeechWindow, bounded_windows
from asr2rpp.vad_asr import infer_vad_whisper
from asr2rpp.rpp_export import write_reference
from asr2rpp.transcript import group_units
from test_inference_policy import constrained
from test_vad_asr import pcm


def fake_asr(monkeypatch, tmp_path, windows):
    import asr2rpp.vad_asr as asr
    calls = []
    monkeypatch.setattr(asr, 'executable', lambda *a: tmp_path/'whisper')
    def detect(*args, **kwargs):
        calls.append(kwargs['context_overlap'])
        return windows, {'windows': []}
    monkeypatch.setattr(asr, 'detect_windows', detect)
    def native(argv, *args):
        for i, arg in enumerate(argv):
            if arg == '-of':
                Path(argv[i+1]).with_suffix('.json').write_text(json.dumps({
                    'transcription': [{'text': 'はい。', 'offsets': {'from': 0, 'to': 30000}}]}))
    monkeypatch.setattr(asr, 'run_process', native)
    return calls

@pytest.mark.parametrize('reference_origin', [0, 90])
def test_vad_times_export_without_aligner_and_preserve_gaps(tmp_path, monkeypatch, reference_origin):
    windows = [SpeechWindow(1, 2, 1, 2), SpeechWindow(8, 10, 8, 10)]
    calls = fake_asr(monkeypatch, tmp_path, windows)
    audio = pcm(tmp_path/'input.wav', 12)
    result = infer_vad_whisper(constrained(), tmp_path/'weights', audio, tmp_path/'work',
                              {}, threading.Event(), lambda _: None)
    assert calls == [False]
    assert result.raw['timing_kind'] == 'vad_segment'
    assert [(u.start, u.end, u.method, u.granularity) for u in result.units] == [
        (1, 2, 'vad_segment', 'segment'), (8, 10, 'vad_segment', 'segment')]
    assert all(u.owner_start is None for u in result.units)
    assert len(group_units(result.units)) == 2  # repeated speech not removed
    output = tmp_path/'out.rpp'
    write_reference(output, audio, result.units, 90, reference_origin, False,
                    reference_duration=12 if reference_origin else 200)
    text = output.read_text(encoding='utf-8')
    assert text.index('ORIGINAL') < text.index('Transcript')
    assert text.count('<ITEM') == 3
    assert 'POSITION 91' in text and 'POSITION 98' in text
    assert f'SOFFS {91-reference_origin}' in text and f'SOFFS {98-reference_origin}' in text
    assert 'LENGTH 1' in text and 'LENGTH 2' in text
    assert not list((tmp_path/'work').rglob('*.wav'))

def test_overlap_cannot_be_mislabeled_as_vad_segment_times(tmp_path, monkeypatch):
    fake_asr(monkeypatch, tmp_path, [SpeechWindow(1, 3, 1, 2.8)])
    with pytest.raises(ValueError, match='Overlapping ASR context'):
        infer_vad_whisper(constrained(), tmp_path/'weights', pcm(tmp_path/'a.wav', 4),
                          tmp_path/'work', {}, threading.Event(), lambda _: None)


def test_no_aligner_long_speech_uses_disjoint_bounded_inputs(tmp_path):
    windows, splits = bounded_windows([(0, 65), (70, 72)], pcm(tmp_path/'a.wav', 80),
                                      25, 0, threading.Event())
    assert splits > 0
    assert windows[0].start == 0 and windows[-2].end == 65
    assert windows[-1].start == 70
    assert all(w.start == w.owner_start and w.end == w.owner_end for w in windows)
    assert all(0 < w.end-w.start <= 25 for w in windows)
    assert all(a.end <= b.start for a,b in zip(windows, windows[1:]))


def test_selected_aligner_still_receives_full_context_windows(tmp_path, monkeypatch):
    calls = fake_asr(monkeypatch, tmp_path, [SpeechWindow(1, 3, 1, 2.8)])
    result = infer_vad_whisper(constrained(), tmp_path/'weights', pcm(tmp_path/'a.wav', 4),
                              tmp_path/'work', {'alignment_requested': True}, threading.Event(), lambda _: None)
    assert calls == [True]
    assert result.units[0].method == 'vad_window'
    assert result.units[0].owner_end == 2.8
    assert result.raw['timing_kind'] == 'pending_alignment'


@pytest.mark.parametrize('source', ['typo', '', False, 1, 'native'])
def test_invalid_vad_timestamp_source_is_rejected(source):
    from test_inference_policy import POLICY
    with pytest.raises(ValueError):
        constrained(constraints={'inference': dict(POLICY, timestamp_source=source)}).validate()


def test_explicit_alignment_policy_remains_enforced(tmp_path, monkeypatch):
    from test_inference_policy import POLICY
    from asr2rpp.pipeline import Settings, Stage
    from asr2rpp.catalog import Model
    from asr2rpp.inference_policy import policy_for
    model = constrained(constraints={'inference': dict(POLICY, timestamp_source='alignment')})
    assert policy_for(model).requires_alignment
    assert not policy_for(model).uses_vad_timing
    with pytest.raises(ValueError, match='requires forced alignment'):
        Settings(Stage(model.id)).validate({model.id: model})
    align = Model('align', 'audio_cpp', 'align', {'path': 'x'}, family='qwen3_forced_aligner')
    Settings(Stage(model.id), align=Stage('align')).validate({model.id: model, 'align': align})
    calls = fake_asr(monkeypatch, tmp_path, [])
    with pytest.raises(ValueError, match='requires forced alignment'):
        infer_vad_whisper(model, tmp_path/'w', tmp_path/'a', tmp_path/'work',
                          {}, threading.Event(), lambda _: None)
    assert not calls
