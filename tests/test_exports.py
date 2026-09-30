"""Public export contract: real OTIO round-trips, no overwrites, no fake times."""
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
import json
import threading

import pytest
from asr2rpp.domain import Unit, Result
from asr2rpp.pipeline import Stage, Settings, reserve_output
from asr2rpp.export_timeline import build_timeline, validate_timeline
from asr2rpp import outputs


def document(tmp_path, *, processed=False):
    source = tmp_path/'音声 #1.wav'
    source.write_bytes(b'referenced, never modified')
    report = tmp_path/'report'
    report.mkdir(exist_ok=True)
    (report/'asr').mkdir(exist_ok=True)
    (report/'asr/raw.json').write_text('{"segments":[{"text":"raw recognition","start":0.25,"end":1.5}]}')
    (report/'asr/native.txt').write_text('uncorrected text')
    (report/'asr/cache.wav').write_bytes(b'never embed')
    (report/'asr/engine.log').write_text('log is not recognition data')
    units = [Unit(.25,1.5,'一人目','Alice'), Unit(2,3,'二人目','Bob'), Unit(3.5,4,'再び','Alice')]
    settings = Settings(Stage('a'), output_formats=('rpp','otio','json'), clip_start=10)
    outputs.snapshot(report, 'aligned', units)
    return outputs.make_document(source, source, units, units, settings, report,
        {'source_sha256':'verified-source-digest', 'models':{'asr':{'sha256':'model-digest'}}},
        5, 10 if processed else 0, 5 if processed else 20)


@pytest.mark.parametrize('formats', [(), [], ('unknown',), ('rpp','rpp'), 'rpp', None, (True,)])
def test_invalid_or_empty_formats_fail_before_runtime(tmp_path, monkeypatch, formats):
    from asr2rpp import queue_runner
    monkeypatch.setattr(queue_runner, '_resolve_models', lambda *a: pytest.fail('must not load/download a model'))
    with pytest.raises(ValueError):
        queue_runner.run_queue([], Settings(Stage('a'), output_formats=formats), {}, threading.Event(), print, print)
    assert not list(tmp_path.iterdir())


def test_format_parsing_is_explicit_and_default_is_rpp():
    assert outputs.parse_formats() == ('rpp',)
    assert outputs.parse_formats(['OTIO,json','rpp']) == ('otio','json','rpp')
    with pytest.raises(ValueError): outputs.parse_formats([''])


@pytest.mark.parametrize('occupied', ['rpp','otio','json'])
def test_shared_stem_reservation_checks_all_export_siblings(tmp_path, occupied):
    source = tmp_path/'a.wav'
    source.write_bytes(b'input')
    (tmp_path/('a.'+occupied)).write_text('existing')
    path, report = reserve_output(source, Settings(Stage('a'),output_formats=('json',)))
    assert path.name == 'a_2.json' and report.name == 'a_2.asr2rpp'
    assert (tmp_path/('a.'+occupied)).read_text() == 'existing'


@pytest.mark.parametrize('formats', [x for n in range(1,4) for x in combinations(outputs.FORMATS,n)])
def test_each_format_combination_writes_only_requested_files(tmp_path, formats):
    if 'otio' in formats: pytest.importorskip('opentimelineio')
    doc = document(tmp_path)
    primary = tmp_path/('edits.'+formats[0])
    result = outputs.write_document(doc, primary, formats)
    assert tuple(result) == formats
    for key in outputs.FORMATS:
        assert primary.with_suffix('.'+key).exists() == (key in formats)
    if 'json' in formats:
        loaded = json.loads(result['json'].read_text())
        outputs.validate_document(loaded)
        assert loaded == doc


@pytest.mark.parametrize('processed', [True, False])
def test_otio_original_mute_speakers_gaps_and_source_offsets(tmp_path, processed):
    otio = pytest.importorskip('opentimelineio')
    doc = document(tmp_path, processed=processed)
    result = outputs.write_document(doc, tmp_path/'timeline.otio', ('otio',))
    value = otio.core.deserialize_json_from_file(str(result['otio']))
    assert isinstance(value, otio.schema.Timeline)
    tracks = list(value.tracks)
    assert [t.name for t in tracks] == ['ORIGINAL','Alice','Bob']
    assert tracks[0].enabled is False
    original = next(iter(tracks[0].find_clips()))
    assert original.enabled is False
    assert original.source_range.start_time.to_seconds() == 0
    assert original.source_range.duration.to_seconds() == (5 if processed else 20)
    assert tracks[0].range_of_child(original).start_time.to_seconds() == (10 if processed else 0)
    for track in tracks[1:]:
        assert track.enabled
        for clip in track.find_clips():
            metadata = dict(clip.metadata['asr2rpp'])
            assert track.range_of_child(clip).start_time.to_seconds() == pytest.approx(metadata['start_seconds'])
            assert clip.source_range.start_time.to_seconds() == pytest.approx(metadata['source_start_seconds'])
            assert clip.media_reference.target_url == doc['media']['reference']['url']
            assert '%23' in clip.media_reference.target_url and '%20' in clip.media_reference.target_url


def test_same_speaker_overlap_preserves_time_in_distinct_lanes(tmp_path):
    units=[Unit(0,2,'one','A'),Unit(1,3,'overlap','A'),Unit(3,4,'later','A'),Unit(1,2,'other','ORIGINAL')]
    value=build_timeline(tmp_path/'audio.wav',units,0,0,True,reference_duration=6)
    assert [t['name'] for t in value['tracks']] == ['ORIGINAL','A','Speaker ORIGINAL','A (2)']
    assert sum(len(t['clips']) for t in value['tracks'][1:]) == 4
    assert value['tracks'][1]['clips'][1]['start_seconds'] == 3
    assert value['tracks'][3]['clips'][0]['start_seconds'] == 1


def test_json_preserves_results_fine_units_and_explicit_time_origins(tmp_path):
    doc = document(tmp_path)
    assert doc['schema'] == outputs.SCHEMA and doc['schema_version'] == 1
    assert doc['timebase']['unit_to_timeline_offset_seconds'] == 10
    assert doc['transcript']['units'][0]['start'] == .25
    assert doc['timeline']['tracks'][1]['clips'][0]['start_seconds'] == 10.25
    assert doc['results']['asr/raw.json']['data']['segments'][0]['text'] == 'raw recognition'
    assert doc['results']['asr/native.txt']['data'] == 'uncorrected text'
    assert doc['results']['history/aligned.json']['data'][0]['text'] == '一人目'
    assert not any(name.endswith(('.wav','.log')) for name in doc['results'])
    assert doc['provenance']['models']['asr']['sha256'] == 'model-digest'


def test_export_failure_rolls_back_only_owned_files(tmp_path):
    doc = document(tmp_path)
    (tmp_path/'edits.json').write_text('user-owned')
    with pytest.raises(FileExistsError):
        outputs.write_document(doc, tmp_path/'edits.rpp', ('rpp','json'))
    assert not (tmp_path/'edits.rpp').exists()
    assert (tmp_path/'edits.json').read_text() == 'user-owned'


def test_export_cancellation_rolls_back_outputs(tmp_path, monkeypatch):
    from asr2rpp.catalog import Cancelled
    doc=document(tmp_path)
    stop=threading.Event()
    real=outputs._render
    def render(*args):
        value=real(*args);stop.set();return value
    monkeypatch.setattr(outputs,'_render',render)
    with pytest.raises(Cancelled): outputs.write_document(doc,tmp_path/'edits.json',('json',),stop)
    assert not (tmp_path/'edits.json').exists()


def test_standalone_json_reexports_without_models_or_report(tmp_path, capsys):
    from asr2rpp.cli import main
    doc=document(tmp_path)
    path=tmp_path/'all-results.json';path.write_text(json.dumps(doc))
    import shutil
    shutil.rmtree(tmp_path/'report')
    assert main(['convert',str(path),'--format','rpp']) == 0
    output=Path(capsys.readouterr().out.strip())
    assert output.exists() and 'MUTESOLO 1' in output.read_text()
    assert json.loads(path.read_text()) == doc
    assert main(['convert',str(path),'--format','']) == 2


def test_unknown_json_schema_is_not_silently_guessed(tmp_path):
    doc=document(tmp_path);doc['schema_version']=2
    with pytest.raises(ValueError): outputs.validate_document(doc)


def test_raw_json_single_and_queue_paths_keep_fine_and_grouped_units(tmp_path, monkeypatch):
    from asr2rpp import pipeline, queue_runner
    from asr2rpp.catalog import Model
    from test_release_regressions import wav
    source=wav(tmp_path/'voice.wav', 4)
    model=Model('a','whisper_cpp','asr',{'path':str(source)})
    settings=Settings(Stage('a'),output_formats=('json',))
    monkeypatch.setenv('ASR2RPP_CACHE_DIR',str(tmp_path/'cache'))
    def decode(src,dest,*args,**kwargs):
        import shutil
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);return 4.0
    for module in [pipeline,queue_runner]:
        monkeypatch.setattr(module,'executable',lambda *a:source)
        monkeypatch.setattr(module,'ffmpeg_path',lambda *a:'ffmpeg')
        monkeypatch.setattr(module,'resolve_model',lambda *a:(source,{}))
    monkeypatch.setattr(pipeline,'decode',decode)
    result=lambda:Result([Unit(.25,.75,'hello',granularity='word'),Unit(.8,1.2,'world',granularity='word')],{'original':'hello world'})
    monkeypatch.setattr(pipeline,'infer',lambda *a,**kw:result())
    monkeypatch.setattr(queue_runner,'_asr_batch',lambda _m,_w,_s,jobs,*a,**kw:{j.key:result() for j in jobs})
    a=pipeline.run_job(source,settings,{'a':model},threading.Event(),lambda _:None)
    b=queue_runner.run_queue([(0,source)],settings,{'a':model},threading.Event(),lambda _:None,lambda *a:None)[0]
    for output in [a,b]:
        data=json.loads(output.read_text())
        assert len(data['transcript']['units'])==2 and len(data['transcript']['edit_units'])==1
        assert data['results']['asr/raw.json']['data']=={'original':'hello world'}
        assert not output.with_suffix('.rpp').exists()
        assert data['provenance']['outputs']=={'json':str(output)}
