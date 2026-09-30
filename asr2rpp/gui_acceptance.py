"""Opt-in frozen-GUI acceptance with real inference and isolated preferences.

Run only with an explicitly supplied fixture and a NEW report directory. This
never uses or changes the user's GUI settings. Model/cache paths may be provided
via the normal environment variables by the acceptance runner.
"""
from pathlib import Path
import hashlib
import json
import os
import time


def exercise(fixture, destination):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from .gui_dcc import MainWindow, STYLE
    from .catalog import digest

    fixture = Path(fixture).resolve()
    if not fixture.is_file():
        raise FileNotFoundError(fixture)
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    env_keys = ('ASR2RPP_HOME', 'ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP',
                'ASR2RPP_WEIGHTS_DIR', 'ASR2RPP_CACHE_DIR')
    previous = {key: os.environ.get(key) for key in env_keys}
    os.environ['ASR2RPP_HOME'] = str(root/'home')
    os.environ['ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP'] = '1'
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    prefs = QSettings(str(root/'isolated.ini'), QSettings.Format.IniFormat)
    prefs.setValue('storage/model_dir', previous['ASR2RPP_WEIGHTS_DIR'] or '')
    prefs.setValue('storage/temp_dir', str(root/'cache'))
    device = os.environ.get('ASR2RPP_ACCEPTANCE_DEVICE', 'cpu')
    prefs.setValue('runtime_default/whisper_cpp', device)
    prefs.setValue('runtime_default/audio_cpp', device)
    ffmpeg = os.environ.get('ASR2RPP_ACCEPTANCE_FFMPEG', '')
    if ffmpeg:
        prefs.setValue('runtime_paths', json.dumps({'ffmpeg': ffmpeg}))
    window = None
    result = {'passed': False, 'inference': 'real frozen GUI; no stubs',
              'requested_device': device, 'source_sha256': digest(fixture)}
    try:
        window = MainWindow(preferences=prefs)
        settings = window.current_settings()
        result.update(default_asr=settings.asr.model_id, default_language=settings.asr.language,
                      default_formats=list(settings.output_formats))
        assert settings.asr.model_id == 'anime-whisper' and settings.asr.language == 'ja'
        assert settings.output_formats == ('rpp',)
        assert settings.preprocess is None and settings.align is None and settings.diar is None
        for key in ('otio', 'json'):
            action = next(a for a in window.output_formats.menu.actions() if a.data() == key)
            action.trigger()
        window.output_mode.setCurrentIndex(window.output_mode.findData('custom'))
        window.output_dir.setText(str(root/'outputs'))
        window.add_paths([str(fixture)])
        window.show()
        app.processEvents()
        window.run_button.click()
        assert window.worker is not None
        deadline = time.monotonic() + 1200
        while window.worker is not None:
            app.processEvents()
            if time.monotonic() > deadline:
                window.worker.cancel.set()
                raise TimeoutError('Real-GUI acceptance exceeded 20 minutes')
            time.sleep(.02)
        assert len(window.entries) == 1 and window.entries[0]['status'] == 'done', window.entries
        primary = Path(window.entries[0]['output'])
        doc = json.loads(primary.with_suffix('.json').read_text(encoding='utf-8'))
        assert doc['transcript']['units'] and doc['provenance']['source_unchanged']
        assert all(primary.with_suffix('.'+key).is_file() for key in ('rpp', 'otio', 'json'))
        result.update(passed=digest(fixture) == result['source_sha256'],
                      source_unchanged=digest(fixture) == result['source_sha256'],
                      timestamp_source=doc['provenance'].get('timestamp_source'),
                      formats=['rpp', 'otio', 'json'],
                      unit_count=len(doc['transcript']['units']),
                      output=str(primary))
        app.processEvents()
        window.grab().save(str(root/'gui.png'))
        (root/'run.log').write_text(window.run_log.toPlainText(), encoding='utf-8')
        return result
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        if window is not None:
            if window.worker is not None:
                window.worker.cancel.set()
                deadline = time.monotonic()+30
                while window.worker is not None and time.monotonic() < deadline:
                    app.processEvents()
                    time.sleep(.02)
            window.close()
            app.processEvents()
        (root/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
