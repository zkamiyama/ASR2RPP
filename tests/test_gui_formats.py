"""Output chips work with mouse/keyboard and preserve an intentionally empty set."""
import json
import pytest


def setup(tmp_path,monkeypatch):
    pytest.importorskip('PySide6')
    monkeypatch.setenv('QT_QPA_PLATFORM','offscreen')
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP','1')
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import QSettings
    from asr2rpp.gui_dcc import MainWindow
    app=QApplication.instance() or QApplication([])
    prefs=QSettings(str(tmp_path/'settings.ini'),QSettings.Format.IniFormat)
    window=MainWindow(prefs)
    return app,prefs,window


def test_chips_menu_only_offers_unselected_formats(tmp_path,monkeypatch):
    app,prefs,window=setup(tmp_path,monkeypatch)
    try:
        widget=window.output_formats
        assert widget.formats()==('rpp',)
        assert [a.data() for a in widget.menu.actions()]==['otio','json']
        widget.menu.actions()[0].trigger()
        assert widget.formats()==('rpp','otio')
        widget.menu.actions()[0].trigger()
        assert widget.formats()==('rpp','otio','json')
        assert not widget.add_button.isEnabled() and not widget.menu.actions()
        for button in list(widget.buttons.values()):
            assert not button.icon().isNull()
            button.click()
        assert widget.formats()==()
        widget.add_format('json')
        assert widget.formats()==('json',)
        assert [a.data() for a in widget.menu.actions()]==['rpp','otio']
        assert window.current_settings().output_formats==('json',)
    finally:
        window.close();app.processEvents()


def test_empty_formats_show_dialog_and_never_start_worker(tmp_path,monkeypatch):
    app,prefs,window=setup(tmp_path,monkeypatch)
    try:
        source=tmp_path/'voice.wav';source.write_bytes(b'test')
        window.add_paths([str(source)])
        window.output_formats.set_formats(())
        from asr2rpp import gui_dcc
        warnings=[]
        monkeypatch.setattr(gui_dcc.QMessageBox,'warning',lambda *args:warnings.append(args[-1]))
        monkeypatch.setattr(gui_dcc.Worker,'start',lambda *args:pytest.fail('no output selected'))
        window.run_button.click()
        assert warnings and '出力形式' in warnings[0]
        assert window.worker is None and window.entries[0]['status']=='waiting'
        assert not source.with_suffix('.asr2rpp').exists()
        window.save_preferences()
        assert json.loads(prefs.value('output/formats'))==[]
    finally:
        window.close();app.processEvents()
    from asr2rpp.gui_dcc import MainWindow
    again=MainWindow(prefs)
    try: assert again.output_formats.formats()==()
    finally: again.close();app.processEvents()
