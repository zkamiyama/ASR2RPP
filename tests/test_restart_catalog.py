"""Restart state and authoritative catalog regression coverage."""
from pathlib import Path
from dataclasses import replace
import hashlib
import itertools
import json
import threading
import pytest
from asr2rpp import catalog
from asr2rpp.run_state import RunLedger, TERMINAL


@pytest.mark.parametrize('mode', ['prepare-stop','file-stop','queue-stop','prepare-error',
    'queue-error','missing-result','file-error','immediate-stop','close-stop'])
def test_gui_go_stop_go_and_error_recovery(tmp_path, monkeypatch, mode):
    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM', 'offscreen')
    from asr2rpp.lifecycle_smoke import exercise
    assert exercise(tmp_path, [mode])['passed']


@pytest.mark.parametrize('terminal', ['done', 'failed', 'stopped'])
def test_terminal_events_cannot_regress_or_overcount(terminal):
    for later in itertools.product(('running', 'waiting', 'done', 'failed', 'stopped'), repeat=3):
        state = RunLedger((0, 1))
        assert state.accept(0, terminal, 'first') is not None
        for event in later:
            assert state.accept(0, event, 'late') is None
        assert state.records[0] == (terminal, 'first') and state.completed == 1
        assert state.accept(42, 'done') is None


def setup_roots(tmp_path, monkeypatch):
    bundled, home = tmp_path/'app/models', tmp_path/'home'
    bundled.mkdir(parents=True)
    monkeypatch.setattr(catalog, 'assets_root', lambda: bundled.parent)
    monkeypatch.setattr(catalog, 'data_root', lambda: home)
    return bundled, home


def definition(language='ja'):
    return f'runtime="whisper_cpp"\ntask="asr"\n[source]\npath="relative.bin"\n[defaults]\nlanguage="{language}"\n'


def test_direct_catalog_edit_is_seen_without_any_copy(tmp_path, monkeypatch):
    bundled, home = setup_roots(tmp_path, monkeypatch)
    file = bundled/'custom.toml'; file.write_text(definition())
    first, errors = catalog.load_catalog(); assert not errors
    assert first['custom'].definition == file.resolve()
    assert not home.exists()
    file.write_text(definition('en'))
    second, errors = catalog.load_catalog(); assert not errors
    assert second['custom'].defaults['language'] == 'en'
    assert first['custom'].defaults['language'] == 'ja'
    assert first['custom'].definition_sha256 != second['custom'].definition_sha256
    assert second['custom'].definition_sha256 == hashlib.sha256(file.read_bytes()).hexdigest()
    (bundled/'relative.bin').write_bytes(b'weights')
    assert catalog.local_model_path(second['custom']) == bundled/'relative.bin'


def test_frozen_catalog_ignores_meipass_and_working_directory(tmp_path, monkeypatch):
    app, extraction = tmp_path/'app', tmp_path/'temp/_MEI'
    (app/'models').mkdir(parents=True); (extraction/'models').mkdir(parents=True)
    (app/'models/a.toml').write_text(definition('ja'))
    (extraction/'models/a.toml').write_text(definition('en'))
    monkeypatch.setattr(catalog.sys, 'frozen', True, raising=False)
    monkeypatch.setattr(catalog.sys, 'executable', str(app/'asr2rpp-cli.exe'))
    monkeypatch.setattr(catalog.sys, '_MEIPASS', str(extraction), raising=False)
    monkeypatch.setattr(catalog, 'data_root', lambda: tmp_path/'home')
    models, errors = catalog.load_catalog(); assert not errors
    assert models['a'].defaults['language'] == 'ja'
    (app/'models/a.toml').unlink(); (app/'models').rmdir()
    models, errors = catalog.load_catalog()
    assert not models and errors and 'Missing model definitions' in errors[0]


def test_known_legacy_copy_ignored_and_custom_collision_visible(tmp_path, monkeypatch):
    bundled, home = setup_roots(tmp_path, monkeypatch)
    (home/'models').mkdir(parents=True)
    source=bundled/'a.toml'; source.write_text(definition('ja'))
    old=home/'models/a.toml'; old.write_text(definition('en'))
    fingerprint=hashlib.sha256(old.read_text().encode()).hexdigest()
    (bundled/'template-migrations.json').write_text(json.dumps({'a.toml':[fingerprint]}))
    models, errors=catalog.load_catalog(); assert not errors
    assert models['a'].defaults['language']=='ja' and old.read_text()==definition('en')
    old.write_text(definition('de'))
    models, errors=catalog.load_catalog()
    assert models['a'].definition==source and len(errors)==1 and 'Duplicate model ID' in errors[0]
    old.rename(home/'models/user-a.toml')
    models, errors=catalog.load_catalog(); assert not errors
    assert models['user-a'].defaults['language']=='de'
    assert not list(home.rglob('*.bak'))


def test_bad_or_duplicate_custom_id_does_not_silently_select_another(tmp_path, monkeypatch):
    bundled, home = setup_roots(tmp_path, monkeypatch)
    (bundled/'a.toml').write_text(definition())
    custom=home/'custom-models';custom.mkdir(parents=True)
    (custom/'A.toml').write_text(definition('en'))
    (custom/'bad.toml').write_text('[broken')
    models, errors=catalog.load_catalog()
    assert set(models)=={'a'} and len(errors)==2
    assert any('Duplicate model ID' in e for e in errors)


@pytest.mark.parametrize('where', ['resolve', 'asr', 'align', 'diar', 'after_first_output'])
def test_actual_queue_cancel_preserves_terminal_manifests(tmp_path, monkeypatch, where):
    from asr2rpp import queue_runner as queue, pipeline as core
    from asr2rpp.preprocessing import Settings
    from asr2rpp.pipeline import Stage
    from asr2rpp.adapters import Result, Unit
    from test_release_regressions import wav
    import shutil
    stop=threading.Event(); events=[]
    paths=[wav(tmp_path/f'{n}.wav', 3) for n in range(2)]
    models={'a':catalog.Model('a','whisper_cpp','asr',{}),
        'al':catalog.Model('al','audio_cpp','align',{},family='qwen3_forced_aligner'),
        'd':catalog.Model('d','audio_cpp','diar',{},family='nemotron_3_diar')}
    settings=Settings(Stage('a'), align=Stage('al') if where=='align' else None,
                      diar=Stage('d') if where=='diar' else None)
    monkeypatch.setenv('ASR2RPP_CACHE_DIR',str(tmp_path/'cache'))
    monkeypatch.setattr(queue,'executable',lambda *a:Path('native'))
    monkeypatch.setattr(queue,'ffmpeg_path',lambda *a:'ffmpeg')
    def resolve(*args):
        if where=='resolve': stop.set(); catalog.checkpoint(stop)
        return Path('weights'),{}
    monkeypatch.setattr(queue,'resolve_model',resolve)
    def decode(source,destination,*a,**kw):
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,destination);return 3.0
    monkeypatch.setattr(core,'decode',decode)
    def asr(model,weights,stage,jobs,*a,**kw):
        if where=='asr': stop.set(); catalog.checkpoint(stop)
        return {j.key:Result([Unit(.5,1,'test')],{}) for j in jobs}
    monkeypatch.setattr(queue,'_whisper_batch',asr)
    def interrupt(*a,**kw): stop.set();catalog.checkpoint(stop)
    monkeypatch.setattr(queue,'_align_batch',interrupt)
    monkeypatch.setattr(queue,'_audio_batch',interrupt)
    def item(i,state,detail):
        events.append((i,state,detail))
        if where=='after_first_output' and state=='完了': stop.set()
    with pytest.raises(catalog.Cancelled):
        queue.run_queue(list(enumerate(map(str,paths))),settings,models,stop,lambda _:None,item)
    manifests=[json.loads((tmp_path/f'{n}.asr2rpp/manifest.json').read_text()) for n in range(2)]
    expected=['completed','cancelled'] if where=='after_first_output' else ['cancelled','cancelled']
    assert [m['status'] for m in manifests]==expected
    for i in range(2):
        terminal=[s for n,s,_ in events if n==i and s in ('完了','中断','失敗')]
        assert terminal==(['完了'] if expected[i]=='completed' else ['中断'])
    assert not list((tmp_path/'cache').glob('queue-*'))


def test_worker_snapshot_and_late_signals_cannot_mutate_next_attempt(tmp_path, monkeypatch):
    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    monkeypatch.setenv('ASR2RPP_HOME',str(tmp_path/'home'))
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP','1')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from asr2rpp import gui_dcc as gui
    from asr2rpp.preprocessing import Settings
    from asr2rpp.pipeline import Stage
    app=QApplication.instance() or QApplication([])
    model=catalog.Model('a','whisper_cpp','asr',{'path':'old'},defaults={'language':'ja'})
    old=gui.Worker([(0,'x.wav')],Settings(Stage('a')),{'a':model})
    current=gui.Worker([(0,'x.wav')],Settings(Stage('a')),{'a':model})
    model.defaults['language']='en'
    assert current.catalog['a'].defaults['language']=='ja'
    window=gui.MainWindow(QSettings(str(tmp_path/'settings.ini'),QSettings.Format.IniFormat))
    window.entries=[{'path':str(tmp_path/'x.wav'),'status':'waiting','output':''}]
    window.worker=current;window._run_ledger=RunLedger((0,))
    old.item.connect(window.item_changed);old.finished.connect(window.work_finished)
    current.item.connect(window.item_changed)
    old.item.emit(0,'done','wrong.rpp');old.finished.emit();app.processEvents()
    assert window.worker is current and window.entries[0]['status']=='waiting'
    current.item.emit(0,'done','correct.rpp');app.processEvents()
    old.item.emit(0,'failed','late');app.processEvents()
    assert window.entries[0]['output']=='correct.rpp' and window.completed==1
    window.worker=None;window.close()


def test_missing_selected_model_on_go_is_not_replaced(tmp_path,monkeypatch):
    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    monkeypatch.setenv('ASR2RPP_HOME',str(tmp_path/'home'))
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP','1')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from asr2rpp import gui_dcc as gui
    app=QApplication.instance() or QApplication([])
    window=gui.MainWindow(QSettings(str(tmp_path/'prefs.ini'),QSettings.Format.IniFormat))
    window.asr.model.setCurrentIndex(window.asr.model.findData('whisper-base'))
    source=tmp_path/'a.wav';source.write_bytes(b'test');window.add_paths([str(source)])
    other=catalog.Model('different','whisper_cpp','asr',{'path':'x'})
    monkeypatch.setattr(gui,'load_catalog',lambda:({'different':other},[]))
    errors=[];monkeypatch.setattr(gui.QMessageBox,'warning',lambda *args:errors.append(args[-1]))
    window.run_button.click();app.processEvents()
    assert window.worker is None and errors and window.asr.model.currentIndex()==-1
    window.close()


def test_invalid_saved_preferences_do_not_crash_or_replace_missing_model(tmp_path,monkeypatch):
    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    monkeypatch.setenv('ASR2RPP_HOME',str(tmp_path/'home'))
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP','1')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from asr2rpp.gui_dcc import MainWindow
    app=QApplication.instance() or QApplication([])
    prefs=QSettings(str(tmp_path/'prefs.ini'),QSettings.Format.IniFormat)
    for k,v in {'runtime_paths':'[]','threads':'invalid','queue/batch_audio_ram_mb':'invalid',
                'asr/model':'removed-custom-model','asr/parameters':'123'}.items():prefs.setValue(k,v)
    w=MainWindow(prefs)
    assert w.threads==4 and w.batch_audio_ram_mb==512 and w.runtime_paths=={}
    assert w.asr.model.currentIndex()==-1 and w.asr.parameters=={}
    w.close();app.processEvents()
