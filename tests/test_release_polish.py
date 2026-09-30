"""Release UX regressions: real Qt ownership, assets, docs and ICO payloads."""
from pathlib import Path
import hashlib
import re
import threading
import time
import pytest
from tools.build_icons import build_icons, ico_frames, SIZES, ROLES

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def app():
    pytest.importorskip('PySide6')
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def window_for_test(tmp_path, monkeypatch, app):
    from PySide6.QtCore import QSettings
    from asr2rpp.gui_dcc import MainWindow
    monkeypatch.setenv('ASR2RPP_HOME', str(tmp_path/'home'))
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP', '1')
    window = MainWindow(preferences=QSettings(str(tmp_path/'test.ini'), QSettings.Format.IniFormat))
    window.asr.model.setCurrentIndex(window.asr.model.findData('whisper-base'))
    for panel in (window.preprocess, window.align, window.diar):
        panel.toggle.setChecked(False)
    return window


def test_thread_start_failure_unlocks_go_and_can_retry(tmp_path, monkeypatch, app):
    from asr2rpp import gui_dcc as gui
    window = window_for_test(tmp_path, monkeypatch, app)
    errors = []
    original_start = gui.Worker.start
    monkeypatch.setattr(gui.QMessageBox, 'warning', lambda *a: errors.append(a[-1]))
    monkeypatch.setattr(gui.Worker, '_prepare_models', lambda _: None)
    def cannot_start(worker):
        raise RuntimeError('test: thread could not start')
    monkeypatch.setattr(gui.Worker, 'start', cannot_start)
    source = tmp_path/'input.wav'
    source.write_bytes(b'test input; native inference stubbed')
    window.add_paths([str(source)])
    try:
        window.run_button.click()
        assert errors and 'could not start' in errors[-1]
        assert window.worker is None and not window._stopping
        assert window.entries[0]['status'] == 'failed'
        assert window.completed == window._run_ledger.completed == 1
        assert window.run_button.isEnabled() and window.run_button.text() == 'GO'
        assert window.settings_button.isEnabled()
        monkeypatch.setattr(gui.Worker, 'start', original_start)
        def success(source, *args):
            output = source.with_suffix('.rpp')
            output.write_text('test result')
            return output
        monkeypatch.setattr(gui, 'run_job', success)
        window.run_button.click()
        deadline = time.monotonic()+5
        while window.worker is not None:
            app.processEvents()
            assert time.monotonic() < deadline
            time.sleep(.005)
        assert window.entries[0]['status'] == 'done'
        assert not window.run_button.isEnabled()
    finally:
        if window.worker:
            window.worker.cancel.set()
            window.worker.wait(3000)
            app.processEvents()
        window.close()


def test_bad_configuration_keeps_idle_state_and_items(tmp_path, monkeypatch, app):
    from asr2rpp import gui_dcc as gui
    window = window_for_test(tmp_path, monkeypatch, app)
    errors = []
    monkeypatch.setattr(gui.QMessageBox, 'warning', lambda *a: errors.append(a[-1]))
    source = tmp_path/'input.wav'; source.write_bytes(b'test')
    window.add_paths([str(source)])
    def invalid():
        raise ValueError('invalid selected TOML')
    monkeypatch.setattr(window, 'current_settings', invalid)
    window.start_work()
    assert window.worker is None and window.entries[0]['status'] == 'waiting'
    assert window.run_button.isEnabled() and errors
    window.close()


def test_gui_icon_works_in_all_common_sizes(tmp_path, monkeypatch, app):
    window = window_for_test(tmp_path, monkeypatch, app)
    try:
        for n in SIZES:
            assert not window.windowIcon().pixmap(n, n).isNull()
        assert not app.windowIcon().isNull()
    finally:
        window.close()


def test_icons_are_distinct_multisize_and_reproducible(tmp_path, app):
    from PySide6.QtGui import QImage, QIcon
    first = build_icons(tmp_path/'one', tmp_path/'preview.png')
    second = build_icons(tmp_path/'two')
    fingerprints = set()
    for role in ROLES:
        assert first[role].read_bytes() == second[role].read_bytes()
        fingerprints.add(hashlib.sha256(first[role].read_bytes()).hexdigest())
        frames = ico_frames(first[role])
        assert [(w,h) for w,h,_ in frames] == [(n,n) for n in SIZES]
        for w, h, data in frames:
            image = QImage.fromData(data, 'PNG')
            assert (image.width(), image.height()) == (w,h)
            assert image.pixelColor(0,0).alpha() == 0
            assert not QIcon(str(first[role])).pixmap(w,h).isNull()
    assert len(fingerprints) == len(ROLES)
    assert not QImage(str(tmp_path/'preview.png')).isNull()


@pytest.mark.parametrize('payload', [b'', b'not an icon', b'\x00\x00\x01\x00\x20\x00'])
def test_ico_bounds_checked(tmp_path, payload):
    file = tmp_path/'bad.ico'; file.write_bytes(payload)
    with pytest.raises(ValueError):
        ico_frames(file)


def test_readmes_link_each_language_and_have_no_broken_relative_links():
    for name in ('README.md', 'README.ja.md'):
        text = (ROOT/name).read_text(encoding='utf-8')
        assert 'assets/branding/app.svg' in text.splitlines()[0]
        assert '[English](README.md)' in text or '[日本語](README.ja.md)' in text
        assert 'PyInstaller' not in text and '37.01' not in text
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' not in target and not target.startswith('#'):
                assert (ROOT/target.split('#')[0]).is_file(), target
    assert (ROOT/'docs/history/pre-release-readme.md').is_file()


def test_all_known_packaged_tools_have_icons():
    from tools.brand_windows import ROLE_BY_NAME
    assert set(ROLE_BY_NAME.values()) == set(ROLES)
    for role in ROLE_BY_NAME.values():
        assert (ROOT/'assets/branding'/f'{role}.svg').is_file()
