"""Synthetic v0.9.5 contract tests. No model/GPU/accuracy claims."""
from dataclasses import replace
from pathlib import Path
import copy
import json
import threading
import wave
import pytest
from asr2rpp.adapters import parse_audio
from asr2rpp.catalog import Model
from asr2rpp.domain import Unit, Result
from asr2rpp.nemotron_transcribe import parse_result, read_manifest
from asr2rpp.providers import Request
from asr2rpp.timing import TimingSettings, plan_for
from asr2rpp.transcript import group_units


@pytest.fixture
def raw():
    return json.loads((Path(__file__).parent / 'fixtures/nemotron_transcribe_v095.json').read_text())


def model():
    return Model('nemotron-test', 'audio_cpp', 'asr', {'path': 'unused.gguf'},
        family='nemotron_transcribe', sample_rate=16000, schema_version=2,
        capabilities={'timestamps': 'word', 'speakers': True},
        execution={'output': 'custom', 'max_batch_items': 1, 'max_audio_seconds': 3600},
        defaults={'request': {'return_timestamps': True, 'diarization': True}})


def pcm(path, seconds=50):
    with wave.open(str(path), 'wb') as f:
        f.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        f.writeframes(b'\0\0' * 16000 * seconds)
    return path


def test_rich_schema_preserves_native_ids_characters_and_original_turns(raw):
    before = copy.deepcopy(raw)
    result = parse_audio(raw, 'asr', 'nemotron_transcribe', 48000)
    assert result.text == raw['text']
    assert result.raw == before == raw
    assert result.units[0].speaker == 'speaker_2'  # NOT spk:0 / speaker_0
    assert result.units[0].start == 1.25
    assert result.units[0].method == 'ctc_alignment'
    assert [u.text for u in group_units(result.units)] == ['今日は。', 'はい。']
    assert result.units[2].end > result.units[3].start
    assert parse_audio(raw['speaker_turns'], 'diar', 'nemotron_3_diar', 48000).units[0].start == 1
    assert result.warnings and 'characters' in result.warnings[0]


@pytest.mark.parametrize('termination', ['token_limit', 'repetition_detected', 'cancelled', 'unknown'])
def test_truncated_text_cannot_be_exported_as_complete(raw, termination):
    raw['custom_schema_output']['data']['termination'] = termination
    with pytest.raises(ValueError, match='timed export refused'): parse_result(raw)


@pytest.mark.parametrize('status', ['partial', 'unavailable'])
def test_failed_alignment_never_uses_turns_as_transcript_intervals(raw, status):
    raw['custom_schema_output']['data']['alignment_status'] = status
    raw['custom_schema_output']['data']['failed_streams'] = [{'tag': 1, 'reason': 'no legal path'}]
    with pytest.raises(ValueError, match='not complete'): parse_result(raw)
    assert raw['text'] and raw['speaker_turns']


def test_sparse_alignment_detects_missing_repeated_character(raw):
    raw['custom_schema_output']['data']['words'].pop(3)
    with pytest.raises(ValueError, match='coverage'): parse_result(raw)


def test_coverage_keeps_event_style_and_malformed_tags(raw):
    raw['text'] += ' <ev:laughter> <sty:happy> <spk:bad>'
    with pytest.raises(ValueError, match='coverage'): parse_result(raw)


def test_unknown_acoustic_id_does_not_fall_back_to_transcript_tag(raw):
    row = raw['custom_schema_output']['data']['words'][0]
    row.update(speaker_id='unknown', speaker_column=-1)
    assert parse_result(raw).units[0].speaker is None
    row['speaker_id'] = 'speaker_0'
    with pytest.raises(ValueError, match='ID/column'): parse_result(raw)


@pytest.mark.parametrize('bad', [float('nan'), True, -1])
def test_invalid_timestamps_fail_closed(raw, bad):
    raw['custom_schema_output']['data']['words'][0]['start'] = bad
    with pytest.raises((ValueError, TypeError)): parse_result(raw)


def test_silence_is_not_a_missing_schema_error():
    result = parse_result({'text': '', 'custom_schema_output': {
        'schema': 'nemotron_transcribe.result.v1', 'data': {'termination': 'near_silent',
        'generated_tokens': 0, 'alignment_status': 'unavailable', 'alignment_failure_reason': 'empty_transcript'}}})
    assert not result.units and not result.text and 'silence gate' in result.warnings[0]


def test_generic_words_are_not_accepted_for_native_speakers(raw):
    with pytest.raises(ValueError, match='custom schema'): parse_audio(raw['word_timestamps'], 'asr', 'nemotron_transcribe', 48000)
    raw['custom_schema_output']['schema'] = 'nemotron_transcribe.result.v2'
    with pytest.raises(ValueError, match='custom schema'): parse_result(raw)


@pytest.mark.parametrize('records', [[{'id': 'wrong'}], [{'id': 'one'}, {'id': 'one'}], [None]])
def test_manifest_request_identity(tmp_path, records):
    p = tmp_path / 'manifest.json'; p.write_text(json.dumps({'requests': records}))
    with pytest.raises(ValueError): read_manifest(p, {'one'})


@pytest.mark.parametrize('timing,align', [(TimingSettings(mode='vad', speaker_source='native'), False),
    (TimingSettings(mode='alignment', segment_before_alignment=True, speaker_source='native'), True)])
def test_no_silent_vad_chunking_or_speaker_id_reset(timing, align):
    with pytest.raises(ValueError, match='one complete recording'): plan_for(model(), timing, align)


def mock_native(tmp_path, monkeypatch, raw, *, omit_empty=False):
    from asr2rpp import native_batches
    calls = []
    monkeypatch.setattr(native_batches, 'executable', lambda *a: tmp_path / 'audiocpp_cli')
    def run(argv, cancel, progress, log):
        calls.append(argv)
        seq = json.loads(Path(argv[argv.index('--request-sequence') + 1]).read_text())
        assert len(seq['requests']) == 1
        key = seq['requests'][0]['id']
        folder = Path(log).parent
        Path(log).write_text('text_output=misleading\ncustom_schema_output=not-json\n')
        manifest = {'requests': [{'id': key, 'custom_schema_output': raw['custom_schema_output']}], 'chapters': []}
        (folder / 'manifest.json').write_text(json.dumps(manifest))
        (folder / f'text_{key}.txt').write_text(raw['text'])
        if not omit_empty:
            for name, field in [('words', 'word_timestamps'), ('turns', 'speaker_turns')]:
                (folder / f'{name}_{key}.json').write_text(json.dumps(raw[field]))
    monkeypatch.setattr(native_batches, 'run_process', run)
    return calls


def test_whole_recording_single_and_batch_routes_keep_full_evidence(tmp_path, monkeypatch, raw):
    from asr2rpp import adapters, native_batches
    calls = mock_native(tmp_path, monkeypatch, raw)
    audio = pcm(tmp_path / '50-seconds.wav')
    result = adapters.infer(model(), tmp_path / 'w', audio, tmp_path / 'single', {}, threading.Event(), lambda _: None)
    assert len(calls) == 1 and result.raw['speaker_turns'] == raw['speaker_turns']
    assert result.text == raw['text'] and result.units[0].speaker == 'speaker_2'
    batch = native_batches.audio_many(model(), tmp_path / 'w', [Request('one', audio), Request('two', audio)],
        tmp_path / 'queue', {}, threading.Event(), lambda _: None, 1024**2)
    assert len(calls) == 3 and not batch.errors
    assert batch.results['one'].raw['id'] == 'one' and batch.results['two'].raw['id'] == 'two'
    # Identical local labels do not establish cross-recording speaker identity.
    assert all(r.units[0].speaker == 'speaker_2' for r in batch.results.values())


def test_partial_failure_retains_raw_and_full_transcript(tmp_path, monkeypatch, raw):
    from asr2rpp import native_batches
    raw['custom_schema_output']['data']['alignment_status'] = 'partial'
    mock_native(tmp_path, monkeypatch, raw)
    batch = native_batches.audio_many(model(), tmp_path / 'w', [Request('one', pcm(tmp_path / 'a.wav'))],
        tmp_path / 'work', {}, threading.Event(), lambda _: None, 1024**2)
    assert not batch.results and 'one' in batch.errors
    saved = json.loads((tmp_path / 'work/one/raw.json').read_text())
    assert saved['text'] == raw['text'] and saved['speaker_turns'] == raw['speaker_turns']


def test_silence_missing_optional_files_is_valid_empty_result(tmp_path, monkeypatch, raw):
    from asr2rpp import native_batches
    raw['text'] = ''
    raw['custom_schema_output']['data'] = {'termination': 'near_silent', 'alignment_status': 'unavailable'}
    mock_native(tmp_path, monkeypatch, raw, omit_empty=True)
    batch = native_batches.audio_many(model(), tmp_path / 'w', [Request('one', pcm(tmp_path / 'a.wav'))],
        tmp_path / 'work', {}, threading.Event(), lambda _: None, 1024**2)
    assert not batch.errors and batch.results['one'].units == []


@pytest.mark.parametrize('parameters', [{'return_timestamps': False}, {'template_name': 'translation'}])
def test_contradictory_timed_options_rejected_before_launch(tmp_path, parameters):
    from asr2rpp import native_batches
    with pytest.raises(ValueError):
        native_batches.custom_many(model(), tmp_path / 'w', [], tmp_path, {'parameters': parameters},
            threading.Event(), lambda _: None, 1024**2)


@pytest.mark.parametrize('speaker_source,external,expected', [
    ('auto', False, {None}),
    ('native', False, {'speaker_0', 'speaker_2'}),
    ('none', False, {None}),
    ('auto', True, {'external'}),
    ('diarizer', True, {'external'}),
])
def test_single_and_queue_export_keep_fine_timing_raw_turns_and_speaker_policy(
        tmp_path, monkeypatch, raw, speaker_source, external, expected):
    from asr2rpp import pipeline, queue_runner
    from asr2rpp.pipeline import Settings, Stage
    import shutil
    source = pcm(tmp_path / 'source.wav', 4)
    asr = model()
    diar = Model('diar', 'audio_cpp', 'diar', {'path': str(source)}, family='nemotron_3_diar')
    catalog = {asr.id: asr, diar.id: diar}
    settings = Settings(Stage(asr.id), diar=Stage('diar') if external else None,
        timing=TimingSettings(mode='native', speaker_source=speaker_source), output_formats=('rpp', 'json', 'otio'))
    monkeypatch.setenv('ASR2RPP_CACHE_DIR', str(tmp_path / 'cache'))
    for module in (pipeline, queue_runner):
        monkeypatch.setattr(module, 'executable', lambda *a: source)
        monkeypatch.setattr(module, 'ffmpeg_path', lambda *a: 'ffmpeg')
        monkeypatch.setattr(module, 'resolve_model', lambda *a: (source, {}))
    def decode(src, dest, *args, **kwargs):
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(src, dest); return 4.0
    monkeypatch.setattr(pipeline, 'decode', decode)
    def recognized(): return parse_result(copy.deepcopy(raw))
    def turns(): return Result([Unit(0, 4, speaker='external')], {'turns': 'external'})
    monkeypatch.setattr(pipeline, 'infer', lambda m, *a, **k: recognized() if m.task == 'asr' else turns())
    monkeypatch.setattr(queue_runner, '_asr_batch', lambda m, w, s, jobs, *a, **k: {j.key: recognized() for j in jobs})
    monkeypatch.setattr(queue_runner, '_audio_batch', lambda m, w, s, jobs, *a, **k: {j.key: turns() for j in jobs})
    single = pipeline.run_job(source, settings, catalog, threading.Event(), lambda _: None)
    queued = queue_runner.run_queue([(0, source)], settings, catalog, threading.Event(), lambda _: None, lambda *a: None)[0]
    for output in (single, queued):
        doc = json.loads(output.with_suffix('.json').read_text())
        units = doc['transcript']['units']
        assert len(units) == 5 and {(u['start'], u['end']) for u in units} == {(u.start, u.end) for u in recognized().units}
        assert {u['speaker'] for u in units} == expected
        assert doc['results']['asr/raw.json']['data']['speaker_turns'] == raw['speaker_turns']
        assert doc['results']['asr/raw.json']['data']['text'] == raw['text']
        assert 'experimental' in doc['transcript']['warnings'][0]
        assert output.with_suffix('.otio').is_file()
        rpp = output.read_text()
        assert 'ORIGINAL' in rpp and '今日は。' in rpp and 'はい。' in rpp
        if not external and speaker_source == 'native':
            assert 'speaker_2' in rpp and 'speaker_0' in rpp


def test_joins_previous_preserves_latin_fragments_and_thai_clusters():
    units = [Unit(0, 1, 'a', granularity='word', joins_previous=False),
             Unit(1, 2, 'b', granularity='word', joins_previous=True),
             Unit(2, 3, 'คำ', granularity='word', joins_previous=False),
             Unit(3, 4, 'ไทย', granularity='word', joins_previous=True)]
    assert group_units(units)[0].text == 'ab คำไทย'


def test_family_preflight_rejects_old_runtime_before_weights(monkeypatch):
    from asr2rpp.providers import preflight
    from asr2rpp.pipeline import Stage
    from asr2rpp import runtime_registry
    monkeypatch.setattr(runtime_registry, 'probe', lambda *a: {'capabilities': {
        'loaders': {'nemotron_asr': {'tasks': {'asr': ['streaming']}}}}})
    with pytest.raises(ValueError, match='not compiled'): preflight(model(), Stage(model().id))


def test_out_of_audio_ctc_word_cannot_be_dropped_into_partial_export(tmp_path, monkeypatch, raw):
    from asr2rpp import native_batches
    raw['custom_schema_output']['data']['words'][-1].update(start=90, end=91)
    mock_native(tmp_path, monkeypatch, raw)
    batch = native_batches.audio_many(model(), tmp_path / 'w', [Request('one', pcm(tmp_path / 'a.wav'))],
        tmp_path / 'work', {}, threading.Event(), lambda _: None, 1024**2)
    assert not batch.results and 'request duration' in str(batch.errors['one'])
    saved = json.loads((tmp_path / 'work/one/raw.json').read_text())
    assert saved['custom_schema_output']['data']['words'][-1]['end'] == 91


def test_native_millisecond_rounding_does_not_fail_complete_result(raw):
    from asr2rpp.nemotron_transcribe import validate_duration
    result = parse_result(raw)
    assert validate_duration(result, 2.0998) is result
    with pytest.raises(ValueError, match='duration'): validate_duration(result, 2.0)


@pytest.mark.parametrize('complete,sparse', [('ㇷ゚', 'ㇷ'), ('ก่', 'ก')])
def test_missing_combining_marks_are_not_complete_coverage(raw, complete, sparse):
    raw['text'] = '<spk:0>' + complete
    row = raw['custom_schema_output']['data']['words'][0]
    row.update(word=sparse, display=sparse)
    raw['custom_schema_output']['data']['words'] = [row]
    with pytest.raises(ValueError, match='coverage'): parse_result(raw)


@pytest.mark.parametrize('filename,payload', [('manifest.json', b'{broken'),
    ('text_one.txt', b'bad \xff text'), ('turns_one.json', b'{partial')])
def test_malformed_native_evidence_survives_queue_folder_cleanup(tmp_path, monkeypatch, raw, filename, payload):
    from asr2rpp import native_batches
    from asr2rpp.diagnostics import persist_tree
    mock_native(tmp_path, monkeypatch, raw)
    original = native_batches.run_process
    def corrupt(argv, cancel, progress, log):
        original(argv, cancel, progress, log)
        (Path(log).parent / filename).write_bytes(payload)
    monkeypatch.setattr(native_batches, 'run_process', corrupt)
    work = tmp_path / 'work'
    batch = native_batches.audio_many(model(), tmp_path / 'w', [Request('one', pcm(tmp_path / 'a.wav'))],
        work, {}, threading.Event(), lambda _: None, 1024**2)
    assert not batch.results and 'one' in batch.errors
    persist_tree(work / 'one', tmp_path / 'report/asr')
    import shutil
    shutil.rmtree(work)
    saved = tmp_path / 'report/asr/native'
    assert (saved / filename).read_bytes() == payload
    assert (saved / 'requests.json').is_file() and (saved / 'text_one.txt').is_file()


@pytest.mark.parametrize('parameters', [{'return_timestamps': False}, {'template_name': 'translation'}])
def test_invalid_timed_contract_fails_settings_before_any_model_download(parameters):
    from asr2rpp.pipeline import Stage, Settings
    m = model()
    with pytest.raises(ValueError, match='requires|requires the|controlled by'):
        Settings(Stage(m.id, parameters=parameters), timing=TimingSettings(mode='native')).validate({m.id: m})



def test_cancellation_keeps_native_evidence_before_reraising(tmp_path, monkeypatch, raw):
    from asr2rpp import native_batches
    from asr2rpp.catalog import Cancelled
    mock_native(tmp_path, monkeypatch, raw)
    original = native_batches.run_process
    def cancel_run(argv, cancel, progress, log):
        original(argv, cancel, progress, log)
        cancel.set()
        raise Cancelled('cancelled')
    monkeypatch.setattr(native_batches, 'run_process', cancel_run)
    work = tmp_path / 'work'
    with pytest.raises(Cancelled):
        native_batches.audio_many(model(), tmp_path / 'w', [Request('one', pcm(tmp_path / 'a.wav'))],
            work, {}, threading.Event(), lambda _: None, 1024**2)
    assert (work / 'one/native/manifest.json').is_file()
    assert (work / 'one/native/text_one.txt').read_text() == raw['text']
    assert (work / 'one/engine.log').is_file()
