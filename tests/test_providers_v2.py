"""Data-only model extension and provider protocol regression contracts."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import io
import json
import hashlib
import threading
import wave
import pytest
from asr2rpp.catalog import Model,load_catalog,resolve_model
from asr2rpp.pipeline import Stage
from asr2rpp.domain import Unit,Result
from asr2rpp.providers import Request,Batch,preflight,provider_for
from asr2rpp import runtime_registry,worker_client


def make_model(**kwargs):
    values=dict(id='custom',runtime='sherpa_onnx',task='asr',family='transducer',
        source={'path':'weights'},schema_version=2,capabilities={'timestamps':'none'})
    values.update(kwargs)
    return Model(**values)


def test_provider_alias_and_versioned_parameters(tmp_path):
    (tmp_path/'custom.toml').write_text('''schema_version = 2
provider = "sherpa_onnx"
task = "asr"
family = "transducer"
[source]
path = "weights"
[capabilities]
timestamps = "none"
[artifacts]
encoder = "encoder.onnx"
tokens = "tokens.txt"
[parameters.max_active_paths]
type = "int"
default = 4
min = 1
max = 16
ja = "探索幅"
''')
    models,errors=load_catalog(tmp_path)
    assert not errors
    from asr2rpp.parameter_specs import specs_for
    from asr2rpp.adapters import split_engine_parameters
    model=models['custom']
    assert model.runtime=='sherpa_onnx'
    assert specs_for(model)[0]['ja']=='探索幅'
    assert split_engine_parameters(model,None)[0]['max_active_paths']==4
    with pytest.raises(ValueError): split_engine_parameters(model,{'max_active_paths':99})


@pytest.mark.parametrize('extra',[
    {'schema_version':True},{'schema_version':99},{'runtime':'python:arbitrary'},
    {'execution':{'command':'anything'}},{'execution':{'mode':'unknown'}},
    {'execution':{'max_batch_items':0}},{'execution':{'max_audio_seconds':float('nan')}},
    {'execution':{'protocol':True}},{'artifacts':{'encoder':'../escape'}},
    {'artifacts':{'encoder':'.'}},{'parameters':{'x':{'type':'int','default':True}}},
    {'parameters':{'x':{'type':'float','default':float('inf')}}},
    {'parameters':{'x':{'type':'enum','default':'bad','values':['good']}}},
    {'defaults':{'session':'wrong'}},{'defaults':{'request':{'x':float('nan')}}},
])
def test_invalid_extensions(extra):
    with pytest.raises(ValueError): make_model(**extra).validate()


def test_model_toml_cannot_register_an_executable(tmp_path):
    (tmp_path/'unsafe.toml').write_text('''schema_version=2
provider="external_json"
task="asr"
family="custom"
command="arbitrary.exe"
[source]
path="weights"
''')
    models,errors=load_catalog(tmp_path)
    assert not models and any('Unknown fields' in e for e in errors)


def test_local_directory_provenance_changes_with_any_artifact(tmp_path):
    (tmp_path/'encoder.onnx').write_bytes(b'abcd')
    (tmp_path/'tokens.txt').write_text('a 1')
    model=make_model(source={'path':str(tmp_path)},artifacts={'encoder':'encoder.onnx','tokens':'tokens.txt'})
    path,first=resolve_model(model,threading.Event(),lambda _:None)
    assert path==tmp_path
    (tmp_path/'tokens.txt').write_text('b 1')
    _,second=resolve_model(model,threading.Event(),lambda _:None)
    assert first['sha256']!=second['sha256']
    assert first['files']['encoder.onnx']==second['files']['encoder.onnx']


def test_artifact_symlink_escape_is_rejected(tmp_path):
    from asr2rpp.model_schema import artifact_paths
    root=tmp_path/'weights';root.mkdir()
    outside=tmp_path/'outside';outside.write_text('secret')
    try:(root/'tokens.txt').symlink_to(outside)
    except OSError:pytest.skip('Symlink privilege unavailable')
    with pytest.raises(ValueError):artifact_paths(make_model(artifacts={'tokens':'tokens.txt'}),root)


def test_remote_directory_entry_and_small_auxiliary_files(tmp_path,monkeypatch):
    from asr2rpp import catalog
    class Response(io.BytesIO):
        @property
        def headers(self):return {'Content-Length':str(len(self.getvalue()))}
    blobs={'encoder.onnx':b'weights','tokens.txt':b'a 1'}
    model=make_model(source={'repo':'owner/model','revision':'a'*40,'files':list(blobs),'entry':'.',
        'sha256':{k:hashlib.sha256(v).hexdigest() for k,v in blobs.items()}},artifacts={'encoder':'encoder.onnx','tokens':'tokens.txt'})
    model.validate()
    monkeypatch.setenv('ASR2RPP_WEIGHTS_DIR',str(tmp_path))
    monkeypatch.setattr(catalog,'urlopen',lambda req,timeout=30:Response(
        json.dumps({'sha':'a'*40}).encode() if '/api/' in req.full_url else blobs[req.full_url.rsplit('/',1)[1]]))
    path,_=resolve_model(model,threading.Event(),lambda _:None,download=True)
    assert path.is_dir() and (path/'tokens.txt').read_bytes()==b'a 1'
    monkeypatch.setattr(catalog,'urlopen',lambda *a,**k:pytest.fail('verified installation downloaded again'))
    assert resolve_model(model,threading.Event(),lambda _:None)[0]==path


def test_runtime_registration_requires_trust_and_checks_changes(tmp_path,monkeypatch):
    monkeypatch.setenv('ASR2RPP_HOME',str(tmp_path/'home'))
    one=tmp_path/'first.exe';one.write_bytes(b'first')
    two=tmp_path/'second.exe';two.write_bytes(b'second')
    with pytest.raises(ValueError):runtime_registry.register('audio_cpp','cpu',one)
    runtime_registry.register('audio_cpp','cpu',one,trust=True)
    runtime_registry.register('audio_cpp','cpu',two,trust=True)
    assert runtime_registry.resolve('audio_cpp','cpu')==two
    runtime_registry.rollback('audio_cpp','cpu')
    assert runtime_registry.resolve('audio_cpp','cpu')==one
    one.write_bytes(b'wrong')
    with pytest.raises(ValueError,match='changed'):runtime_registry.resolve('audio_cpp','cpu')


def fake_probe(monkeypatch,loaders):
    monkeypatch.setattr(runtime_registry,'probe',lambda *a,**k:{'capabilities':{'schema_version':1,'loaders':loaders},'sha256':'hash'})


def test_preflight_rejects_unsupported_compiled_family_before_weights(monkeypatch):
    fake_probe(monkeypatch,{'other':{'tasks':{'asr':['offline']}}})
    model=make_model(runtime='audio_cpp',family='new_family')
    with pytest.raises(ValueError,match='not compiled'):preflight(model,Stage(model.id))


def test_preflight_rejects_wrong_device(monkeypatch):
    fake_probe(monkeypatch,{'sherpa_onnx/transducer':{'tasks':{'asr':['offline']},'devices':['cpu'],'timestamps':['none']}})
    model=make_model()
    with pytest.raises(ValueError,match='does not support device'):preflight(model,Stage(model.id,device='cuda'),segmented=True)
    assert preflight(model,Stage(model.id,device='cpu'),segmented=True)['sha256']=='hash'


def test_preflight_does_not_trust_invented_timestamps(monkeypatch):
    fake_probe(monkeypatch,{'sherpa_onnx/transducer':{'tasks':{'asr':['offline']},'devices':['cpu'],'timestamps':['none']}})
    with pytest.raises(ValueError,match='no complete intervals'):
        preflight(make_model(capabilities={'timestamps':'word'}),Stage('custom'))


def ok_record(key='a'):
    return dict(schema_version=1,id=key,status='ok',text='hello',time_origin='request',
                units=[dict(start=0.,end=1.,text='hello')])


def test_worker_partial_trailing_result_preserves_success(tmp_path):
    path=tmp_path/'out.jsonl';path.write_text(json.dumps(ok_record())+'\n{"schema_')
    results,errors=worker_client.read_results(path,{'a','b'})
    assert set(results)=={'a'} and set(errors)=={'b'}


@pytest.mark.parametrize('record',[
    dict(ok_record(),schema_version=True),dict(ok_record(),schema_version=2),
    dict(ok_record(),id='unrequested')])
def test_worker_schema_and_ownership(tmp_path,record):
    path=tmp_path/'out.jsonl';path.write_text(json.dumps(record)+'\n')
    with pytest.raises(ValueError):worker_client.read_results(path,{'a'})


def test_worker_duplicate_result_id_is_rejected(tmp_path):
    path=tmp_path/'out.jsonl';path.write_text((json.dumps(ok_record())+'\n')*2)
    with pytest.raises(ValueError,match='duplicate'):worker_client.read_results(path,{'a'})


@pytest.mark.parametrize('change',[{'start':True},{'end':float('nan')},{'start':-1},
    {'start':2,'end':1},{'speaker':7},{'granularity':'made_up'},{'command':'untrusted'}])
def test_worker_invalid_timestamps_fail_one_request(tmp_path,change):
    record=ok_record();record['units'][0].update(change)
    path=tmp_path/'out.jsonl';path.write_text(json.dumps(record)+'\n'+json.dumps(ok_record('b'))+'\n')
    results,errors=worker_client.read_results(path,{'a','b'})
    assert set(results)=={'b'} and set(errors)=={'a'}


def test_worker_requires_time_origin(tmp_path):
    record=ok_record();record.pop('time_origin')
    with pytest.raises(ValueError):worker_client.normalize(record)


def pcm(path,seconds=4):
    with wave.open(str(path),'wb') as w:
        w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(b'\0\0'*16000*seconds)
    return path


def test_shared_generic_vad_sessions_preserve_source_gaps(tmp_path,monkeypatch):
    from asr2rpp import region_batch
    from asr2rpp.vad import SpeechWindow
    calls=[]
    class Provider:
        def infer_many(self,model,weights,requests,*args):
            calls.append([r.id for r in requests])
            return Batch({r.id:Result([],{},'発話') for r in requests})
    monkeypatch.setattr(region_batch,'provider_for',lambda _:Provider())
    monkeypatch.setattr(region_batch,'detect_windows',lambda *a,**k:([SpeechWindow(0,1,0,1),SpeechWindow(2,3,2,3)],{}))
    requests=[Request('file0',pcm(tmp_path/'a.wav')),Request('file1',pcm(tmp_path/'b.wav'))]
    out=region_batch.infer_many(make_model(),tmp_path/'w',requests,tmp_path/'work',
        {'timing':{'mode':'vad'}},threading.Event(),lambda _:None,1024**2)
    assert len(calls)==1 and len(set(calls[0]))==4
    assert set(out.results)=={'file0','file1'} and not out.errors
    for result in out.results.values():
        assert [(u.start,u.end,u.method) for u in result.units]==[(0,1,'vad_segment'),(2,3,'vad_segment')]
    assert not list((tmp_path/'work').rglob('*.wav'))


def test_shared_generic_vad_failure_isolated_to_owner(tmp_path,monkeypatch):
    from asr2rpp import region_batch
    from asr2rpp.vad import SpeechWindow
    class Provider:
        def infer_many(self,model,weights,requests,*args):
            return Batch({requests[0].id:Result([],{},'ok')},{requests[1].id:ValueError('bad')})
    monkeypatch.setattr(region_batch,'provider_for',lambda _:Provider())
    monkeypatch.setattr(region_batch,'detect_windows',lambda *a,**k:([SpeechWindow(0,1,0,1)],{}))
    requests=[Request('a',pcm(tmp_path/'a.wav')),Request('b',pcm(tmp_path/'b.wav'))]
    out=region_batch.infer_many(make_model(),tmp_path/'w',requests,tmp_path/'work',
        {'timing':{'mode':'vad'}},threading.Event(),lambda _:None,1024**2)
    assert set(out.results)=={'a'} and set(out.errors)=={'b'}


def test_provider_registry_is_not_a_python_import_registry():
    with pytest.raises(ValueError):provider_for(make_model(runtime='some.module.Class'))


def test_queue_prefetch_windows_preserve_indices_and_partial_success(tmp_path, monkeypatch):
    from asr2rpp import queue_runner as q
    from asr2rpp.pipeline import Settings, Stage
    from asr2rpp.catalog import Model
    settings=Settings(Stage('m'),queue_window_items=2)
    m=Model('m','whisper_cpp','asr',{'path':'weights'})
    batches=[]
    def window(jobs,*args):
        batches.append([i for i,_ in jobs])
        return {i:Path(p).with_suffix('.rpp') for i,p in jobs if i!=3}
    monkeypatch.setattr(q,'_run_queue_window',window)
    output=q.run_queue([(i,str(tmp_path/f'{i}.wav')) for i in range(5)],settings,{'m':m},
        threading.Event(),lambda _:None,lambda *a:None)
    assert batches==[[0,1],[2,3],[4]]
    assert set(output)=={0,1,2,4}


def test_alignment_preserves_native_speaker_without_coarsening():
    from asr2rpp.alignment import AlignmentInput, aligned_units
    from asr2rpp.domain import Unit, Result
    request=AlignmentInput('s1',Path('speech.wav'),10,2,'two words',speaker='native-A')
    result=Result([Unit(0,.4,'two',granularity='word'),Unit(.5,1,'words',granularity='word')],{})
    units=aligned_units(result,request,[])
    assert [(u.start,u.end,u.speaker) for u in units]==[(10,10.4,'native-A'),(10.5,11,'native-A')]


def test_schema_does_not_accept_boolean_as_version(tmp_path):
    from asr2rpp.catalog import load_catalog
    path=tmp_path/'m.toml'
    path.write_text('schema_version=true\nprovider="audio_cpp"\ntask="asr"\nfamily="new"\n[source]\npath="model.gguf"\n')
    models,errors=load_catalog(tmp_path)
    assert not models and errors
