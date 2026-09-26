"""Exercise the real Qt/Worker lifecycle with deterministic inference stubs.

Explicit developer self-test only. Uses isolated INI preferences and data roots;
never reads or writes a user's settings, models, or media. Native inference is
covered separately by the frozen native smoke gate.
"""
from pathlib import Path
import json
import os
import threading
import time


def exercise(destination, cases=None):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from . import gui_dcc as gui
    from .catalog import checkpoint, Cancelled, model_directory
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    keys = ('ASR2RPP_HOME', 'ASR2RPP_CACHE_DIR', 'ASR2RPP_WEIGHTS_DIR', 'ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP')
    old_env = {k: os.environ.get(k) for k in keys}
    os.environ['ASR2RPP_HOME'] = str(root/'home')
    os.environ['ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP'] = '1'
    os.environ.pop('ASR2RPP_CACHE_DIR', None)
    os.environ.pop('ASR2RPP_WEIGHTS_DIR', None)
    originals = gui.Worker._prepare_models, gui.run_job, gui.run_queue
    app = QApplication.instance() or QApplication([])
    results = []
    window = None

    def until(predicate):
        deadline = time.monotonic() + 8
        while not predicate():
            app.processEvents()
            if time.monotonic() > deadline:
                raise AssertionError('GUI lifecycle timeout')
            time.sleep(.005)
        app.processEvents()

    try:
        for mode in cases or ('prepare-stop', 'file-stop', 'queue-stop', 'prepare-error',
                               'queue-error', 'missing-result', 'file-error', 'immediate-stop', 'close-stop'):
            entered = threading.Event()
            stage = {'attempt': 1, 'calls': []}
            def hold(cancel):
                entered.set()
                while True:
                    checkpoint(cancel)
                    time.sleep(.005)
            def prepare(worker):
                if stage['attempt'] == 1 and mode in ('prepare-stop', 'immediate-stop'):
                    hold(worker.cancel)
                if stage['attempt'] == 1 and mode == 'prepare-error':
                    raise RuntimeError('test prepare failure')
            def single(source, settings, catalog, cancel, progress):
                stage['calls'].append(source.name)
                if stage['attempt'] == 1 and source.stem == 'input1':
                    if mode in ('file-stop', 'close-stop'):
                        hold(cancel)
                    if mode == 'file-error':
                        raise RuntimeError('test item failure')
                output = root/f'{mode}-{stage["attempt"]}-{source.stem}.rpp'
                output.write_text('test result', encoding='utf-8')
                return output
            def queue(jobs, settings, catalog, cancel, progress, item, limit):
                if stage['attempt'] != 1:
                    for index, source in jobs:
                        item(index, 'done', str(single(Path(source), settings, catalog, cancel, progress)))
                    return
                index, source = jobs[0]
                item(index, 'done', str(single(Path(source), settings, catalog, cancel, progress)))
                # Duplicate/out-of-order callbacks cannot regress a completed item.
                item(index, 'running', '')
                item(index, 'stopped', '')
                if mode == 'queue-stop':
                    hold(cancel)
                if mode == 'queue-error':
                    raise RuntimeError('test queue failure')
                # A returned but unfinished queue is reconciled as failure.
            gui.Worker._prepare_models, gui.run_job, gui.run_queue = prepare, single, queue
            prefs = QSettings(str(root/f'{mode}.ini'), QSettings.Format.IniFormat)
            window = gui.MainWindow(preferences=prefs)
            window.queue_strategy = 'stage' if mode.startswith('queue') or mode == 'missing-result' else 'file'
            window.asr.model.setCurrentIndex(window.asr.model.findData('whisper-base'))
            for panel in (window.preprocess, window.align, window.diar):
                panel.toggle.setChecked(False)
            window.asr.language.setCurrentText('en')
            for i in range(3):
                path = root/f'input{i}.wav'
                path.write_bytes(b'test input')
                window.add_paths([str(path)])
            window.show()
            window.run_button.click()
            assert window.worker is not None
            first_worker = window.worker
            window.start_work(False)
            assert window.worker is first_worker
            assert window.worker.settings.asr.language == 'en'  # GO reload retained user settings.
            before = len(window.entries)
            window.clear_queue(); window.retry_failed(); window.remove_selected()
            assert len(window.entries) == before
            if mode.endswith('stop'):
                if mode != 'immediate-stop':
                    until(entered.is_set)
                if mode == 'close-stop':
                    window.close()
                else:
                    window.run_button.click()
                assert not window.run_button.isEnabled() and window._stopping
            until(lambda: window.worker is None)
            statuses = [e['status'] for e in window.entries]
            assert 'running' not in statuses and 'waiting' not in statuses
            assert window.completed == 3
            expected = 'stopped' if mode.endswith('stop') else 'failed'
            assert expected in statuses, (mode, statuses)
            successful = {i:e['output'] for i,e in enumerate(window.entries) if e['status']=='done'}
            if mode not in ('prepare-stop', 'prepare-error', 'immediate-stop'):
                assert statuses[0] == 'done'
            if mode != 'close-stop':
                assert window.run_button.isEnabled() and window.run_button.text() == 'GO'
                stage['attempt'] = 2
                stage['calls'].clear()
                window.run_button.click()
                until(lambda: window.worker is None)
                assert all(e['status']=='done' for e in window.entries)
                assert all(window.entries[i]['output']==out for i,out in successful.items())
                assert len(stage['calls']) == 3-len(successful)
                assert window.completed == 3-len(successful) and not window.run_button.isEnabled()
            window.close(); app.processEvents(); window = None
            results.append({'case': mode, 'passed': True})
        report = {'passed': True, 'cases': results, 'model_directory': str(model_directory()),
                  'inference': 'deterministic stubs; native execution tested separately'}
        (root/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report
    finally:
        if window is not None:
            if window.worker:
                window.worker.cancel.set()
                window.worker.wait(3000)
                app.processEvents()
            window.close()
        gui.Worker._prepare_models, gui.run_job, gui.run_queue = originals
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
