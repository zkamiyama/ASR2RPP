"""Performance changes may not change text, sample boundaries, or error semantics."""
from pathlib import Path
import json
import threading
import pytest
from asr2rpp.catalog import Cancelled
from asr2rpp.vad import SpeechWindow
from asr2rpp.whisper_io import write_plan, response_command
from asr2rpp import vad_asr
from test_vad_asr import pcm
from test_inference_policy import constrained


def setup_native(monkeypatch,tmp_path,windows,feature='ASR2RPP_PCM_PLAN_V1'):
    monkeypatch.setattr(vad_asr,'executable',lambda *a:tmp_path/'whisper')
    monkeypatch.setattr(vad_asr,'capabilities',lambda *a:frozenset({feature}))
    monkeypatch.setattr(vad_asr,'detect_windows',lambda *a,**k:(windows,{'windows':[]}))
    commands=[]
    def run(argv,*args):
        commands.append(argv)
        effective=argv
        if len(argv)==2 and argv[1].startswith('@'):
            effective=[argv[0]]+Path(argv[1][1:]).read_text().splitlines()
        prefixes=[]
        for i,arg in enumerate(effective):
            if arg=='--asr2rpp-pcm-plan':
                lines=Path(effective[i+1]).read_text().splitlines()
                assert lines[0]=='ASR2RPP_PCM_PLAN_V1'
                prefixes += [Path(lines[j+1]) for j in range(2,len(lines),4)]
            elif arg=='-of': prefixes.append(Path(effective[i+1]))
        assert '-nt' in effective and effective[effective.index('-mc')+1]=='0'
        for prefix in prefixes:
            prefix.with_suffix('.json').write_text(json.dumps({'transcription':[
                {'text':'はい。','offsets':{'from':0,'to':30000}}]}),encoding='utf-8')
    monkeypatch.setattr(vad_asr,'run_process',run)
    return commands


def test_186_windows_one_load_and_zero_segment_files(tmp_path,monkeypatch):
    windows=[SpeechWindow(i/2,i/2+.25,i/2,i/2+.25) for i in range(186)]
    commands=setup_native(monkeypatch,tmp_path,windows)
    monkeypatch.setattr(vad_asr,'slice_pcm',lambda *a:pytest.fail('Segment WAV should not be created'))
    result=vad_asr.infer_vad_whisper(constrained(),tmp_path/'w',pcm(tmp_path/'a.wav',94),
                                    tmp_path/'work',{},threading.Event(),lambda _:None)
    assert len(commands)==1 and len(result.units)==186
    assert all(u.method=='vad_segment' for u in result.units)
    assert [(u.start,u.end) for u in result.units]==[(w.start,w.end) for w in windows]
    assert result.raw['performance']['model_processes']==1
    assert result.raw['performance']['segment_wav_files']==0
    assert not list((tmp_path/'work').rglob('*.wav'))


def test_response_fallback_also_loads_once(tmp_path,monkeypatch):
    windows=[SpeechWindow(i/2,i/2+.25,i/2,i/2+.25) for i in range(186)]
    commands=setup_native(monkeypatch,tmp_path,windows,'ASR2RPP_RESPONSE_V1')
    result=vad_asr.infer_vad_whisper(constrained(),tmp_path/'w',pcm(tmp_path/'a.wav',94),
                                  tmp_path/'work',{},threading.Event(),lambda _:None)
    assert len(commands)==1 and commands[0][1].startswith('@')
    assert len(result.units)==186 and not list((tmp_path/'work').rglob('*.wav'))
    assert result.raw['performance']['input_mode']=='segment_wav'


def test_queue_files_share_one_native_model(tmp_path,monkeypatch):
    windows=[SpeechWindow(0,1,0,1)]
    commands=setup_native(monkeypatch,tmp_path,windows)
    audio=pcm(tmp_path/'a.wav',2)
    items=[('one',audio,tmp_path/'one'),('two',audio,tmp_path/'two')]
    errors=[]
    results=vad_asr.infer_vad_many(constrained(),tmp_path/'w',items,{},threading.Event(),
        lambda _:None,lambda *e:errors.append(e),tmp_path)
    assert not errors and set(results)=={'one','two'}
    assert len(commands)==1 and commands[0].count('--asr2rpp-pcm-plan')==2
    assert all(x.raw['performance']['shared_session_files']==2 for x in results.values())


def test_pcm_failure_is_not_retried_with_legacy_or_fake_output(tmp_path,monkeypatch):
    setup_native(monkeypatch,tmp_path,[SpeechWindow(0,1,0,1)])
    def fail(*args): raise RuntimeError('native failed')
    monkeypatch.setattr(vad_asr,'run_process',fail)
    with pytest.raises(RuntimeError,match='native failed'):
        vad_asr.infer_vad_whisper(constrained(),tmp_path/'w',pcm(tmp_path/'a.wav',2),
                                 tmp_path/'work',{},threading.Event(),lambda _:None)
    assert not (tmp_path/'work/raw.json').exists()
    assert not (tmp_path/'work/inputs').exists()

@pytest.mark.parametrize('bad',['line\nbreak','carriage\rreturn','nul\0byte'])
def test_response_arguments_are_not_shell_parsed(tmp_path,bad):
    with pytest.raises(ValueError): response_command(['whisper','--prompt',bad],tmp_path/'args')
    argv=['whisper','--prompt','two words "日本語"','-m','space path/model.bin']
    command=response_command(argv,tmp_path/'args')
    assert Path(command[1][1:]).read_text().splitlines()==argv[1:]


def test_plan_is_sample_exact_and_bounded(tmp_path):
    audio=pcm(tmp_path/'元 音声.wav',2);prefix=tmp_path/'out'
    w=SpeechWindow(17/16000,777/16000,17/16000,777/16000)
    out=tmp_path/'plan'
    assert write_plan(audio,[w],[prefix],out,threading.Event())==[(17/16000,760/16000)]
    assert out.read_text().splitlines()[-2:]==['17','760']
    with pytest.raises(ValueError):
        write_plan(audio,[SpeechWindow(3,4,3,4)],[prefix],out,threading.Event())
    cancel=threading.Event();cancel.set()
    with pytest.raises(Cancelled): write_plan(audio,[w],[prefix],out,cancel)


def test_diagnostics_move_without_retaining_audio(tmp_path):
    from asr2rpp.diagnostics import persist_tree
    source=tmp_path/'cache';source.mkdir();(source/'result.json').write_text('{}')
    target=tmp_path/'report'
    persist_tree(source,target)
    assert not source.exists() and (target/'result.json').read_text()=='{}'
    source.mkdir();(source/'audio.wav').write_bytes(b'private');(source/'log.txt').write_text('retained')
    persist_tree(source,target)
    assert not (target/'audio.wav').exists() and (source/'audio.wav').exists()
    assert (target/'log.txt').read_text()=='retained'


def test_performance_scope_resets_on_error(tmp_path):
    from asr2rpp.performance import profiled,timed,report_directory,_CURRENT
    @timed('a')
    def step(): return 42
    @profiled
    def pipeline():
        report_directory(tmp_path)
        assert step()==42
        raise RuntimeError('test')
    with pytest.raises(RuntimeError): pipeline()
    assert _CURRENT.get() is None
    data=json.loads((tmp_path/'performance.json').read_text())
    assert data['status']=='failed' and data['timings']['a']['calls']==1
    assert data['inclusive_times']


def test_plan_feature_is_cached_but_invalidated_by_binary_change(tmp_path,monkeypatch):
    from asr2rpp import whisper_io,adapters
    binary=tmp_path/'binary';binary.write_bytes(b'a');calls=[]
    def help_cmd(argv,cancel,progress,log,timeout=30):
        calls.append(1);Path(log).write_text('ASR2RPP_PCM_PLAN_V1 ASR2RPP_RESPONSE_V1')
    monkeypatch.setattr(adapters,'run_process',help_cmd)
    assert 'ASR2RPP_PCM_PLAN_V1' in whisper_io.capabilities(binary,tmp_path,threading.Event(),print)
    whisper_io.capabilities(binary,tmp_path,threading.Event(),print);assert len(calls)==1
    binary.write_bytes(b'changed');whisper_io.capabilities(binary,tmp_path,threading.Event(),print)
    assert len(calls)==2


def test_gui_prepare_does_not_hash_local_weights_twice(tmp_path,monkeypatch):
    pytest.importorskip('PySide6')
    from asr2rpp import gui_dcc,catalog
    from asr2rpp.pipeline import Stage
    from asr2rpp.preprocessing import Settings
    weights=tmp_path/'model.bin';weights.write_bytes(b'weights')
    model=catalog.Model('local','whisper_cpp','asr',{'path':str(weights)})
    calls=[]
    monkeypatch.setattr(gui_dcc,'resolve_model',lambda *a,**kw:calls.append(1))
    worker=gui_dcc.Worker([],Settings(Stage('local')),{'local':model})
    worker._prepare_models();assert calls==[]
    path,info=catalog.resolve_model(model,threading.Event(),lambda _:None)
    import hashlib
    assert info['sha256']==hashlib.sha256(b'weights').hexdigest()
    worker.prepare_only=True;worker._prepare_models();assert calls==[1]


def test_chatty_native_process_cancels_with_bounded_reader(tmp_path):
    from asr2rpp.adapters import run_process
    import sys,time
    stop=threading.Event();timer=threading.Timer(.2,stop.set);timer.start()
    started=time.monotonic()
    try:
        with pytest.raises(Cancelled):
            run_process([sys.executable,'-u','-c',
                         'import time\nfor i in range(100000): print("x"*512)\ntime.sleep(60)'],
                        stop,lambda _:None,tmp_path/'chatty.log')
    finally: timer.cancel()
    assert time.monotonic()-started<8


@pytest.mark.parametrize('payload',[{}, {'schema':True,'results':[]},
    {'schema':1,'results':[{}]}, {'schema':2,'results':[]}, []])
def test_bundle_rejects_incomplete_or_wrong_schema(tmp_path,payload):
    from asr2rpp.whisper_io import read_bundle
    p=tmp_path/'bundle.json';p.write_text(json.dumps(payload))
    with pytest.raises(ValueError): read_bundle(p,0)


def test_consolidated_results_preserve_order_and_unicode_validation(tmp_path,monkeypatch):
    setup_native(monkeypatch,tmp_path,[SpeechWindow(0,1,0,1),SpeechWindow(1,2,1,2)])
    monkeypatch.setattr(vad_asr,'capabilities',lambda *a:frozenset({'ASR2RPP_PCM_PLAN_V1','ASR2RPP_PCM_RESULT_V1'}))
    def native(argv,*args):
        result=Path(argv[argv.index('--asr2rpp-pcm-results')+1])
        result.write_text(json.dumps({'schema':1,'results':[
            {'transcription':[{'text':'先'}]},{'transcription':[{'text':'後'}]}]}),encoding='utf-8')
    monkeypatch.setattr(vad_asr,'run_process',native)
    result=vad_asr.infer_vad_whisper(constrained(),tmp_path/'w',pcm(tmp_path/'a.wav',3),
                                  tmp_path/'work',{},threading.Event(),lambda _:None)
    assert [u.text for u in result.units]==['先','後']
    assert result.raw['performance']['result_mode']=='bundle'
    assert not list((tmp_path/'work/out').glob('*.json'))
    with pytest.raises(ValueError): vad_asr.read_text_payload({'transcription':[{'text':'\ufffd'}]},'bad')
