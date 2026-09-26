from dataclasses import replace
from pathlib import Path
import io
import json
import threading
import wave
import pytest

from asr2rpp.catalog import Cancelled
from asr2rpp.adapters import Unit, Result
from asr2rpp.vad import parse_segments, bounded_windows, SpeechWindow, VadOptions
from asr2rpp.vad_asr import infer_vad_whisper, command_batches, read_text_result
from asr2rpp.alignment import AlignmentInput, aligned_units
from asr2rpp.rpp_export import write_reference
from test_inference_policy import constrained


def pcm(path, seconds):
    with wave.open(str(path), 'wb') as w:
        w.setparams((1,2,16000,0,'NONE','not compressed'))
        w.writeframes(b'\0\0'*int(seconds*16000))
    return path


def test_native_centiseconds_and_silence_gaps():
    text='Detected 2 speech segments:\nSpeech segment 0: start = 100.00, end = 250.00\nSpeech segment 1: start = 900.00, end = 999.00\n'
    assert parse_segments(text,10)==[(1,2.5),(9,9.99)]
    assert parse_segments('Detected 0 speech segments:\n',10)==[]


@pytest.mark.parametrize('text', [
    '', 'Detected 1 speech segments:\n',
    'Detected 0 speech segments:\nSpeech segment 0: start = 100, end = 200\n',
    'Detected 1 speech segments:\nSpeech segment 1: start = 100, end = 200\n',
    'Detected 1 speech segments:\nSpeech segment 0: start = 200, end = 100\n',
    'Detected 1 speech segments:\nSpeech segment 0: start = -1, end = 100\n',
    'Detected 1 speech segments:\nSpeech segment 0: start = 100, end = 1200\n',
    'Detected 2 speech segments:\nSpeech segment 0: start = 100, end = 300\nSpeech segment 1: start = 200, end = 400\n',
])
def test_malformed_vad_output_fails(text):
    with pytest.raises(ValueError):
        parse_segments(text,10)


def test_postmerged_long_vad_region_still_obeys_bound(tmp_path):
    audio=pcm(tmp_path/'audio.wav',85)
    windows, splits=bounded_windows([(0,60),(70,72)],audio,25,0.2,threading.Event())
    assert splits>0
    assert all(0<w.end-w.start<=25+1/16000 for w in windows)
    assert windows[0].owner_start==0 and windows[-2].owner_end==60
    assert windows[-1].start==70  # no silence deletion/time compression
    for left,right in zip(windows[:-2],windows[1:-1]):
        assert left.owner_end==right.owner_start
        assert left.end>right.start  # context overlap at a forced cut
    stop=threading.Event(); stop.set()
    with pytest.raises(Cancelled):
        bounded_windows([(0,60)],audio,25,0.2,stop)


def test_word_ownership_not_string_deduplication(tmp_path):
    a=AlignmentInput('a',tmp_path/'a.wav',0,2.2,'はい',0,2)
    b=AlignmentInput('b',tmp_path/'b.wav',1.8,2.2,'はい',2,4)
    left=Result([Unit(0.4,0.6,'はい'),Unit(1.9,2.1,'はい')],{})
    right=Result([Unit(0.1,0.3,'はい'),Unit(1.2,1.4,'はい')],{})
    units=aligned_units(left,a,[])+aligned_units(right,b,[])
    assert len(units)==3  # the boundary word once; genuine repeated words retained
    assert [round(u.start,1) for u in units]==[0.4,1.9,3.0]
    assert all(u.owner_start is None and u.owner_end is None for u in units)


@pytest.mark.parametrize('key,value', [
    ('vad_threshold',float('nan')),('vad_threshold',1.1),
    ('vad_min_speech_duration_ms',True),('vad_min_speech_duration_ms',250.5),
    ('vad_min_silence_duration_ms',0),('vad_speech_pad_ms',2000),
    ('vad_max_speech_duration_s',26),('vad_max_speech_duration_s',0.5),
])
def test_invalid_vad_parameters_rejected(key,value):
    with pytest.raises(ValueError):
        VadOptions.from_parameters({key:value},25)


def test_vad_asr_uses_independent_windows_ignores_native_fake_times(tmp_path,monkeypatch):
    import asr2rpp.vad_asr as asr
    audio=pcm(tmp_path/'audio.wav',12)
    binary=tmp_path/'whisper-cli';binary.touch()
    windows=[SpeechWindow(1,2,1,2),SpeechWindow(8,10,8,10)]
    monkeypatch.setattr(asr,'executable',lambda *a:binary)
    monkeypatch.setattr(asr,'detect_windows',lambda *a,**kw:(windows,{'windows':[]}))
    commands=[]
    def native(argv,cancel,progress,log):
        commands.append(argv)
        assert '-nt' in argv and argv[argv.index('-mc')+1]=='0'
        assert '--vad' not in argv and '-ojf' not in argv
        prefixes=[Path(argv[i+1]) for i,x in enumerate(argv) if x=='-of']
        for prefix in prefixes:
            prefix.with_suffix('.json').write_text(json.dumps({'transcription':[{'text':'こんにちは。','offsets':{'from':0,'to':30000}}]}))
    monkeypatch.setattr(asr,'run_process',native)
    result=infer_vad_whisper(constrained(),tmp_path/'model.bin',audio,tmp_path/'work',{'device':'cpu','alignment_requested':True},threading.Event(),lambda x:None)
    assert len(commands)==1 and commands[0].count('-m')==1 and commands[0].count('-f')==2
    assert [(u.start,u.end) for u in result.units]==[(1,2),(8,10)]
    assert all(u.method=='vad_window' for u in result.units)
    assert not list((tmp_path/'work').rglob('*.wav'))
    with pytest.raises(ValueError,match='Cannot export VAD'):
        write_reference(tmp_path/'out.rpp',audio,result.units,0,0,False,reference_duration=12)
    assert not (tmp_path/'out.rpp').exists()


def test_no_vad_speech_does_not_invent_an_asr_request(tmp_path,monkeypatch):
    import asr2rpp.vad_asr as asr
    monkeypatch.setattr(asr,'executable',lambda *a:tmp_path/'whisper')
    monkeypatch.setattr(asr,'detect_windows',lambda *a,**kw:([],{'windows':[]}))
    monkeypatch.setattr(asr,'run_process',lambda *a:pytest.fail('No speech must not start ASR'))
    result=infer_vad_whisper(constrained(),tmp_path/'model',tmp_path/'silence.wav',tmp_path/'work',{},threading.Event(),lambda x:None)
    assert result.units==[]


def test_native_invalid_utf8_not_silently_repaired(tmp_path):
    p=tmp_path/'raw.json';p.write_bytes(b'{"transcription":[{"text":"bad\xff"}]}')
    with pytest.raises(ValueError,match='Invalid UTF-8'):
        read_text_result(p)
    assert b'\xff' in p.read_bytes()
    p.write_text(json.dumps({'transcription':[{'text':'bad\ufffd'}]}))
    with pytest.raises(ValueError,match='replacement'):
        read_text_result(p)


def test_command_length_counts_input_output_and_fixed_args():
    pairs=[(Path('a'*200),Path('b'*200)) for _ in range(100)]
    chunks=list(command_batches(['whisper','-m','model'],pairs,max_chars=1400,max_items=64))
    assert sum(map(len,chunks))==100 and max(map(len,chunks))==3
    with pytest.raises(ValueError):
        list(command_batches(['whisper'],[(Path('a'*30000),Path('b'))]))


def test_vad_download_verifies_reuses_and_cancels(tmp_path,monkeypatch):
    import asr2rpp.vad as vad
    data=b'synthetic vad weights'
    import hashlib
    sha=hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(vad,'weights_root',lambda:tmp_path)
    monkeypatch.setattr(vad,'VAD_SHA256',sha)
    monkeypatch.setattr(vad,'VAD_SIZE',len(data))
    calls=[]
    def fetch(*a,**kw):
        calls.append(1);return io.BytesIO(data)
    monkeypatch.setattr(vad,'urlopen',fetch)
    path,info=vad.vad_model(constrained(),{},threading.Event(),lambda x:None)
    assert path.read_bytes()==data and info['sha256']==sha
    assert vad.vad_model(constrained(),{},threading.Event(),lambda x:None)[0]==path
    assert len(calls)==1
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        vad.vad_model(constrained(),{},threading.Event(),lambda x:None)
    path.unlink()
    stop=threading.Event();stop.set()
    with pytest.raises(Cancelled):
        vad.vad_model(constrained(),{},stop,lambda x:None)
    assert not list(tmp_path.rglob('*.part'))


def test_queue_routes_custom_policy_to_shared_vad_adapter(tmp_path,monkeypatch):
    from asr2rpp import queue_runner as queue, vad_asr
    from asr2rpp.pipeline import Stage
    jobs=[]
    for i in range(2):
        report=tmp_path/f'report{i}';report.mkdir()
        jobs.append(queue.QueueJob(i,f'j{i}',tmp_path/f'{i}.wav',tmp_path/f'{i}.rpp',report,'hash',{}))
    calls=[]
    def infer(model,weights,audio,work,options,cancel,progress):
        calls.append(audio)
        if audio.name=='1.wav':
            raise ValueError('bad second file')
        return Result([Unit(1,2,'text',method='vad_window')],{})
    monkeypatch.setattr(vad_asr,'infer_vad_whisper',infer)
    result=queue._whisper_batch(constrained(),tmp_path/'weights',Stage('unrelated-custom-model'),jobs,{j.key:j.source for j in jobs},tmp_path/'cache',threading.Event(),lambda x:None,lambda *a:None)
    assert list(result)==['j0'] and len(calls)==2
    assert not jobs[0].failed and jobs[1].failed
