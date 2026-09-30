"""Timing is selected by the user, never by a model-specific GUI branch."""
from dataclasses import replace
from pathlib import Path
import json
import threading
import wave
import pytest
from asr2rpp.catalog import Model
from asr2rpp.pipeline import Stage, Settings
from asr2rpp.timing import TimingSettings, plan_for, prepare_run


def model(runtime='audio_cpp', timed=False):
    return Model('m', runtime, 'asr', {'path':'unused'}, family='example',
                 capabilities={'timestamps':'word' if timed else 'none'})


@pytest.mark.parametrize('timed,align,expected', [(True,False,'native'), (False,False,'vad'),
                                                  (True,True,'alignment'), (False,True,'alignment')])
def test_automatic_policy(timed, align, expected):
    plan=plan_for(model(timed=timed),TimingSettings(),align)
    assert plan.mode==expected
    assert plan.segmented == (not timed)


@pytest.mark.parametrize('mode,align', [('native',False),('alignment',False),('vad',True),('native',True)])
def test_incompatible_policy_is_rejected(mode,align):
    with pytest.raises(ValueError): plan_for(model(),TimingSettings(mode),align)


def test_user_can_override_legacy_mandatory_alignment():
    from test_inference_policy import constrained, POLICY
    legacy=constrained(constraints={'inference':dict(POLICY,timestamp_source='alignment')})
    assert plan_for(legacy,TimingSettings(mode='vad'),False).mode=='vad'


def test_user_settings_do_not_mutate_model_definition():
    m=model('whisper_cpp')
    settings=Settings(Stage('m'),timing=TimingSettings('vad',max_seconds=12,threshold=.7))
    copy,catalog,plan=prepare_run(settings,{'m':m})
    assert not m.constraints and not settings.asr.parameters
    assert plan.segmented
    assert catalog['m'].constraints['inference']['max_segment_seconds']==12
    assert copy.asr.parameters['vad_threshold']==.7


@pytest.mark.parametrize('kwargs',[{'max_seconds':float('nan')},{'max_seconds':29},
    {'threshold':True},{'min_silence_ms':1},{'segment_before_alignment':1},{'mode':'fabricated'}])
def test_bad_settings(kwargs):
    with pytest.raises(ValueError): TimingSettings(**kwargs).validate()


def pcm(path,rate=16000):
    with wave.open(str(path),'wb') as w:
        w.setparams((1,2,rate,0,'NONE','not compressed'))
        w.writeframes(b'\0\0'*rate*4)
    return path


def test_generic_text_model_uses_vad_ownership_and_one_batch(tmp_path,monkeypatch):
    from asr2rpp import region_asr,text_requests
    from asr2rpp.adapters import Result
    from asr2rpp.vad import SpeechWindow
    calls=[]
    monkeypatch.setattr(region_asr,'detect_windows',lambda *a,**k:(
        [SpeechWindow(0,1,0,1),SpeechWindow(2,3,2,3)],{}))
    def text_batch(m,w,reqs,*a):
        calls.append(len(reqs));return {r['id']:Result([],{},'言葉') for r in reqs}
    monkeypatch.setattr(text_requests,'infer_text_requests',text_batch)
    output=region_asr.infer_regions(model(),tmp_path/'w',pcm(tmp_path/'a.wav'),tmp_path/'work',
        {'timing':{'mode':'vad'}},threading.Event(),lambda _:None)
    assert calls==[2]
    assert [(u.start,u.end,u.method) for u in output.units]==[(0,1,'vad_segment'),(2,3,'vad_segment')]
    assert not list((tmp_path/'work').rglob('*.wav'))


def test_generic_alignment_preserves_context_ownership(tmp_path,monkeypatch):
    from asr2rpp import region_asr,text_requests
    from asr2rpp.adapters import Result
    from asr2rpp.vad import SpeechWindow
    monkeypatch.setattr(region_asr,'detect_windows',lambda *a,**k:([SpeechWindow(0,2,0,1.5)],{}))
    monkeypatch.setattr(text_requests,'infer_text_requests',lambda m,w,rs,*a:
        {r['id']:Result([],{},'発話') for r in rs})
    output=region_asr.infer_regions(model(),tmp_path/'w',pcm(tmp_path/'a.wav'),tmp_path/'work',
        {'timing':{'mode':'alignment'},'alignment_requested':True},threading.Event(),lambda _:None)
    unit=output.units[0]
    assert (unit.start,unit.end,unit.owner_start,unit.owner_end,unit.method)==(0,2,0,1.5,'vad_window')


def test_timing_ui_roundtrip_and_forced_alignment(tmp_path,monkeypatch):
    pytest.importorskip('PySide6')
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from asr2rpp.gui_dcc import MainWindow, PreferencesDialog
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP','1')
    app=QApplication.instance() or QApplication([])
    prefs=QSettings(str(tmp_path/'prefs.ini'),QSettings.Format.IniFormat)
    window=MainWindow(preferences=prefs)
    window.timing_settings=TimingSettings('alignment',12,.65,min_silence_ms=400)
    window.update_state()
    assert window.align.enabled_stage() and not window.align.toggle.isEnabled()
    dialog=PreferencesDialog(window)
    assert dialog.nav.count()==4
    assert dialog.timing_mode.currentData()=='alignment'
    assert dialog.vad_maximum.value()==12
    window.save_preferences()
    dialog.close();window.close()
    restored=MainWindow(preferences=prefs)
    assert restored.timing_settings==TimingSettings('alignment',12,.65,min_silence_ms=400)
    restored.close();app.processEvents()
