"""Reference-track, shared alignment and cleanup regressions without private media."""
from fractions import Fraction
from pathlib import Path
import hashlib
import json
import random
import shutil
import threading
import wave

import pytest
from asr2rpp.adapters import Unit, Result
from asr2rpp.catalog import Model, Cancelled
from asr2rpp.pipeline import Stage, Settings
from asr2rpp.rpp_export import write_reference
from asr2rpp.media import slice_pcm, wave_info, full_reference_duration
from asr2rpp.transcript import assign_speakers
from asr2rpp import alignment, pipeline, queue_runner


def wav(path, seconds=2, rate=16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as f:
        f.setparams((1, 2, rate, 0, 'NONE', 'not compressed'))
        f.writeframes(b'\x01\x00' * round(seconds * rate))
    return path


def first_track(text):
    return text.split('  <TRACK\n')[1]


@pytest.mark.parametrize('diar', [False, True])
@pytest.mark.parametrize('processed', [False, True])
def test_original_is_full_reference_first_and_muted(tmp_path, diar, processed):
    ref = wav(tmp_path / ('音声_vocals.wav' if processed else '音声.wav'), seconds=7)
    digest = hashlib.sha256(ref.read_bytes()).hexdigest()
    out = tmp_path / 'project.rpp'
    origin = 10 if processed else 0
    units = [Unit(1, 2, '発話', 'S1'), Unit(4, 5, '発話2', 'S2')]
    write_reference(out, ref, units, origin, origin, diar, 16000, reference_duration=Fraction(7))
    text = out.read_text(encoding='utf-8')
    first = first_track(text)
    assert first.startswith('    NAME "ORIGINAL"\n')
    assert first.count('<ITEM') == 1
    assert 'LENGTH 7\n' in first  # includes leading/trailing silence, not only speech
    assert f'POSITION {origin}\n' in first
    assert 'SOFFS 0\n' in first and 'LOOP 0\n' in first
    assert 'MUTESOLO 1 0 0' in first
    assert text.count('MUTESOLO 1 0 0') == 1
    assert text.count('  <TRACK\n') == (3 if diar else 2)
    assert ref.name in first and ref.name in text.split('  <TRACK\n')[2]
    assert hashlib.sha256(ref.read_bytes()).hexdigest() == digest
    with pytest.raises(FileExistsError):
        write_reference(out, ref, units, origin, origin, diar, reference_duration=7)


def test_full_original_not_trimmed_to_cli_inference_range(tmp_path):
    reference = wav(tmp_path / 'full.wav', seconds=20)
    settings = Settings(Stage('asr'), clip_start=10, clip_duration=2)
    duration = full_reference_duration(reference, settings, 2, tmp_path, threading.Event(), lambda _: None)
    assert duration == 20
    out = tmp_path / 'clip.rpp'
    write_reference(out, reference, [Unit(.25, 1.5, 'clip')], 10, 0, False, reference_duration=duration)
    assert 'LENGTH 20\n' in first_track(out.read_text())
    assert 'POSITION 10.25\n' in out.read_text()
    assert 'SOFFS 10.25\n' in out.read_text()


def test_reference_duration_never_uses_last_speech_or_reprobes_full_media(tmp_path, monkeypatch):
    from asr2rpp import media
    monkeypatch.setattr(media, 'run_process', lambda *args: pytest.fail('unnecessary full scan'))
    length = full_reference_duration(tmp_path / 'video.mp4', Settings(Stage('a')),
                                     120.5, tmp_path, threading.Event(), lambda _: None)
    assert length == Fraction(241, 2)


def test_clipped_video_duration_scan_ignores_inference_clip(tmp_path, monkeypatch):
    from asr2rpp import media
    monkeypatch.setattr(media, 'ffmpeg_path', lambda *args: 'ffmpeg')
    def scan(argv, cancel, progress, log):
        assert 'atrim' not in ' '.join(argv)
        assert argv[-3:] == ['-f', 'null', '-']
        progress('out_time_us=N/A')
        progress('out_time_us=20500000')
    monkeypatch.setattr(media, 'run_process', scan)
    assert full_reference_duration(tmp_path / 'video.mp4', Settings(Stage('a'), clip_start=10),
                                   2, tmp_path, threading.Event(), lambda _: None) == Fraction(41, 2)


def test_pcm_slice_is_sample_exact_and_cancellable(tmp_path):
    source = wav(tmp_path / 'pcm.wav')
    destination = tmp_path / 'slice.wav'
    assert slice_pcm(source, destination, .25, .75, threading.Event()) == .75
    info = wave_info(destination)
    assert info.frames == 12000 and info.sample_rate == 16000
    cancel = threading.Event(); cancel.set()
    with pytest.raises(Cancelled):
        slice_pcm(source, destination, 0, 1, cancel)


def fake_alignment_native(monkeypatch, commands):
    monkeypatch.setattr(alignment, 'executable', lambda *args: Path('audiocpp_cli'))
    def run(argv, cancel, progress, log):
        commands.append(argv)
        request_file = Path(argv[argv.index('--request-sequence') + 1])
        data = json.loads(request_file.read_text(encoding='utf-8'))
        output = Path(argv[argv.index('--words-out') + 1]).parent
        for request in data['requests']:
            assert (request_file.parent / request['audio']).is_file()
            assert any(ch.isalnum() for ch in request['text'])
            (output / f"words_{request['id']}.json").write_text(json.dumps({
                'words': [{'start': .01, 'end': .1, 'word': request['text']}]
            }), encoding='utf-8')
        Path(log).write_text('one native session', encoding='utf-8')
    monkeypatch.setattr(alignment, 'run_process', run)


def test_294_alignment_segments_reuse_one_native_session(tmp_path, monkeypatch):
    commands = []; fake_alignment_native(monkeypatch, commands)
    model = Model('align', 'audio_cpp', 'align', {}, family='qwen3_forced_aligner')
    stage = Stage('align', parameters={'clamp_timestamps_to_audio': True})
    pcm = wav(tmp_path / 'pcm.wav', seconds=300)
    units = [Unit(n, n + .25, 'test') for n in range(294)]
    result = alignment.align_segments(model, Path('weights'), stage, units, pcm, 300,
                                      tmp_path / 'work', threading.Event(), lambda _: None, [])
    assert len(commands) == 1 and len(result) == 294
    assert result[-1].start == pytest.approx(292.86)
    assert '--request-sequence' in commands[0]
    request_file = Path(commands[0][commands[0].index('--request-sequence') + 1])
    assert all(r['options'] == {'clamp_timestamps_to_audio': True}
               for r in json.loads(request_file.read_text())['requests'])


def test_queue_alignment_shares_protocol_and_passes_session_parameters(tmp_path, monkeypatch):
    commands = []; fake_alignment_native(monkeypatch, commands)
    model = Model('align', 'audio_cpp', 'align', {}, family='qwen3_forced_aligner')
    stage = Stage('align', parameters={'session.example': 1})
    requests, jobs = [], []
    for n in range(2):
        report = tmp_path / f'report{n}'; report.mkdir()
        job = queue_runner.QueueJob(n, f'q{n}', tmp_path/'source.wav', tmp_path/f'{n}.rpp', report, '', {})
        jobs.append(job)
        requests.append(queue_runner.AlignRequest(f'q{n}s0', job, wav(tmp_path/f'{n}.wav'), 0, 2, 'test'))
    results = queue_runner._align_batch(model, Path('model.gguf'), stage, requests, tmp_path/'batch',
                                        threading.Event(), lambda _: None, lambda *args: None, 1024**2)
    assert len(commands) == 1 and len(results) == 2
    assert 'qwen3_forced_aligner.example=1' in commands[0]
    assert not any(j.failed for j in jobs)


def test_empty_alignment_does_not_load_model(tmp_path, monkeypatch):
    monkeypatch.setattr(alignment, 'executable', lambda *args: pytest.fail('empty batch loaded a model'))
    assert alignment.infer_requests(None, None, None, [], tmp_path, threading.Event(), lambda _: None) == {}


def test_queue_original_track_and_alignment_end_to_end(tmp_path, monkeypatch):
    from asr2rpp.preprocessing import Settings as QueueSettings
    sources = [wav(tmp_path / f'{n}.wav', seconds=3) for n in range(2)]
    models = {'asr': Model('asr', 'whisper_cpp', 'asr', {}),
              'align': Model('align', 'audio_cpp', 'align', {}, family='qwen3_forced_aligner')}
    monkeypatch.setenv('ASR2RPP_CACHE_DIR', str(tmp_path/'cache'))
    monkeypatch.setattr(queue_runner, 'executable', lambda *args: Path('tool'))
    monkeypatch.setattr(queue_runner, 'ffmpeg_path', lambda *args: 'ffmpeg')
    monkeypatch.setattr(queue_runner, 'resolve_model', lambda *args: (Path('model'), {}))
    decodes = []
    def decode(source, destination, *args, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination); decodes.append(source)
        return 3.0
    monkeypatch.setattr(pipeline, 'decode', decode)
    def asr(model, weights, stage, jobs, *args, alignment_requested=False):
        assert alignment_requested
        return {j.key: Result([Unit(.1, .3, 'speech'), Unit(1, 1.1, '、')], {}) for j in jobs}
    monkeypatch.setattr(queue_runner, '_whisper_batch', asr)
    commands = []; fake_alignment_native(monkeypatch, commands)
    settings = QueueSettings(Stage('asr'), align=Stage('align'))
    events = []
    outputs = queue_runner.run_queue(list(enumerate(map(str, sources))), settings, models,
                                     threading.Event(), lambda _: None, lambda *args: events.append(args))
    assert set(outputs) == {0, 1}, events
    assert len(commands) == 1  # shares model across both files
    assert len(decodes) == 4  # one ASR and one alignment decode per file, not per segment
    for out in outputs.values():
        assert 'LENGTH 3\n' in first_track(out.read_text())
        assert '、' in out.read_text()
        manifest = json.loads((out.with_suffix('.asr2rpp')/'manifest.json').read_text())
        assert manifest['original_track']['duration_seconds'] == 3
        assert manifest['source_unchanged']


def test_speaker_sweep_preserves_bruteforce_results():
    rng = random.Random(51)
    turns = [Unit(x := rng.uniform(0, 100), x + rng.uniform(.1, 5), speaker=str(n % 5)) for n in range(400)]
    units = [Unit(x := rng.uniform(0, 100), x + rng.uniform(.1, 3), 'speech') for _ in range(200)]
    expected = []
    for u in units:
        overlap = {}
        for t in turns:
            amount = max(0, min(u.end, t.end) - max(u.start, t.start))
            if amount:
                overlap[t.speaker] = overlap.get(t.speaker, 0) + amount
        speaker = max(overlap, key=overlap.get) if overlap else 'UNKNOWN'
        if not overlap or overlap[speaker] / (u.end-u.start) < .55:
            speaker = 'UNKNOWN'
        expected.append(speaker)
    assert [u.speaker for u in assign_speakers(units, turns, [])] == expected


def test_only_one_gui_implementation_and_modes(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    monkeypatch.setenv('ASR2RPP_HOME', str(tmp_path/'home'))
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP', '1')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from PySide6.QtWidgets import QApplication
    from asr2rpp import gui, gui_preprocessing, gui_dcc
    assert gui.MainWindow is gui_preprocessing.MainWindow is gui_dcc.MainWindow
    app = QApplication.instance() or QApplication([])
    window = gui_dcc.MainWindow()
    window.asr.model.setCurrentIndex(window.asr.model.findData('whisper-base'))
    for sep in (False, True):
        window.preprocess.toggle.setChecked(sep)
        window.preprocess.reference.setCurrentIndex(window.preprocess.reference.findData('processed'))
        for diar, align in ((False, True), (True, False), (True, True), (False, False)):
            window.diar.toggle.setChecked(diar); window.align.toggle.setChecked(align)
            settings = window.current_settings()
            assert (settings.preprocess is not None) == sep
            assert (settings.diar is not None) == diar
            assert (settings.align is not None) == align
            assert settings.reference_audio == ('processed' if sep else 'original')
    window.close(); app.processEvents()


@pytest.mark.parametrize('key', ['num_overlap', 'session.num_overlap'])
def test_separator_uses_shared_session_options(tmp_path, monkeypatch, key):
    from asr2rpp import preprocessing as pre
    model = Model('sep', 'audio_cpp', 'sep', {}, family='mel_band_roformer',
                  defaults={'session': {'num_overlap': 2}})
    settings = pre.Settings(Stage('asr'), preprocess=Stage('sep', parameters={key: 3}))
    monkeypatch.setattr(pre, 'executable', lambda *args: Path('native-tool'))
    monkeypatch.setattr(pre, 'ffmpeg_path', lambda *args: 'ffmpeg')
    commands = []
    def run(argv, *args):
        commands.append(argv)
        if '--task' not in argv:
            wav(Path(argv[-1]))
        else:
            out = Path(argv[argv.index('--out-dir') + 1]); out.mkdir(exist_ok=True)
            wav(out/'vocals.wav')
    monkeypatch.setattr(pre, 'run_process', run)
    pre.separate(tmp_path/'source.wav', tmp_path/'work', settings, model, Path('model.gguf'),
                 threading.Event(), lambda _: None)
    assert 'mel_band_roformer.num_overlap=3' in commands[-1]
    assert not any('session.num_overlap' in str(arg) for arg in commands[-1])
