"""Thin PySide6 GUI; all jobs execute via the same CLI in a separate process."""
import json
import sys
from datetime import datetime
from pathlib import Path
from PySide6.QtCore import QProcess, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QLineEdit, QComboBox, QPlainTextEdit, QPushButton, QFileDialog,
    QLabel, QDoubleSpinBox, QMessageBox)
from common import VERSION, app_dir
from registry import Registry


class Panel(QGroupBox):
    def __init__(self, title, registry, kinds, allow_none=False):
        super().__init__(title)
        self.registry = registry
        form = QFormLayout(self)
        self.models, self.devices = QComboBox(), QComboBox()
        if allow_none:
            self.models.addItem('なし / 統合モデルの話者情報を使う', 'none')
        for model in registry.models.values():
            if model['type'] in kinds:
                self.models.addItem(model.get('label', model['id']), model['id'])
        self.params = QPlainTextEdit()
        self.params.setMinimumHeight(140)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.download = QPushButton('このモデルを取得')
        form.addRow('モデル', self.models)
        form.addRow('実行先（実測未確認）', self.devices)
        form.addRow('独立パラメータ JSON', self.params)
        form.addRow(self.note)
        form.addRow(self.download)
        self.models.currentIndexChanged.connect(self.changed)
        self.changed()

    def refresh_status(self):
        mid = self.models.currentData()
        if mid == 'none' or mid is None:
            self.download.setEnabled(False)
            self.note.setText('ASRのみ、または統合モデルの話者情報を使用します。')
            return
        model = self.registry.model(mid)
        present = self.registry.model_path(model).is_file()
        self.note.setText(('モデル取得済み。' if present else 'モデル未取得。') + ' ' + model.get('warning', ''))
        self.download.setEnabled(bool(model.get('artifact')) and not present)

    def changed(self):
        mid = self.models.currentData()
        self.devices.clear()
        if mid == 'none' or mid is None:
            self.params.setPlainText('{}')
        else:
            model = self.registry.model(mid)
            for device in self.registry.data.get('engines', {}).get(model['engine'], {}):
                self.devices.addItem(device)
            self.params.setPlainText(json.dumps(model.get('defaults', {}), ensure_ascii=False, indent=2))
        self.refresh_status()


class Window(QMainWindow):
    def __init__(self, registry_path=None):
        super().__init__()
        self.registry = Registry(registry_path)
        self.setWindowTitle(f'ASR2RPP {VERSION}')
        self.resize(1080, 830)
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        heading = QLabel('ASR2RPP — 非破壊参照のREAPERプロジェクトを作成')
        heading.setStyleSheet('font-size: 19px; font-weight: 600; margin: 6px;')
        layout.addWidget(heading)
        layout.addWidget(QLabel('プレビュー版。ASR・話者推定・時刻精度は別々に検証してください。音源分離は行いません。'))
        inputs = QFormLayout()
        self.source, self.output = QLineEdit(), QLineEdit()
        source_row = QHBoxLayout()
        source_row.addWidget(self.source)
        browse = QPushButton('入力ファイル…')
        browse.clicked.connect(self.choose_source)
        source_row.addWidget(browse)
        inputs.addRow('入力音声 / 動画', source_row)
        inputs.addRow('新規出力フォルダ', self.output)
        times = QHBoxLayout()
        self.start, self.duration = QDoubleSpinBox(), QDoubleSpinBox()
        for box in (self.start, self.duration):
            box.setRange(0, 864000)
            box.setDecimals(3)
            box.setSuffix(' 秒')
        self.duration.setMinimum(0.01)
        self.duration.setValue(55)
        times.addWidget(QLabel('開始（音声ストリーム基準）'))
        times.addWidget(self.start)
        times.addWidget(QLabel('長さ'))
        times.addWidget(self.duration)
        inputs.addRow('処理する範囲', times)
        layout.addLayout(inputs)
        panels = QHBoxLayout()
        self.asr = Panel('ASR', self.registry, {'asr','joint'})
        self.diar = Panel('Diarization / 話者推定', self.registry, {'diarization'}, True)
        panels.addWidget(self.asr)
        panels.addWidget(self.diar)
        layout.addLayout(panels)
        actions = QHBoxLayout()
        self.run_button = QPushButton('RPPを作成')
        self.cancel_button = QPushButton('中止')
        self.cancel_button.setEnabled(False)
        config = QPushButton('モデル定義 JSONを開く')
        config.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.registry.path))))
        reload_button = QPushButton('定義を再読込')
        reload_button.clicked.connect(self.reload_registry)
        self.run_button.clicked.connect(self.run_job)
        self.cancel_button.clicked.connect(self.cancel)
        for button in (self.run_button, self.cancel_button, config, reload_button):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(130)
        layout.addWidget(self.log)
        self.log.appendPlainText('モデル重みは別取得です。MP4などの入力には設定済みのFFmpeg / ffprobeが必要です。')
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
        self.asr.download.clicked.connect(lambda: self.download(self.asr))
        self.diar.download.clicked.connect(lambda: self.download(self.diar))
        self.cancel_path = None
        self.installing = False

    def process_error(self, error):
        self.log.appendPlainText(self.process.errorString())
        if error == QProcess.FailedToStart:
            self.run_button.setEnabled(True)
            self.cancel_button.setEnabled(False)

    def choose_source(self):
        path, _ = QFileDialog.getOpenFileName(self, '音声・動画を選択', '', 'Media (*.wav *.mp4 *.mkv *.mov *.mp3 *.flac *.ogg);;All (*)')
        if path:
            self.source.setText(path)
            self.output.setText(str(Path(path).parent / ('ASR2RPP-' + datetime.now().strftime('%Y%m%d-%H%M%S'))))

    def launch_cli(self, argv, installing=False):
        if self.process.state() != QProcess.NotRunning:
            return
        self.installing = installing
        if getattr(sys, 'frozen', False):
            executable = app_dir() / 'asr2rpp-cli.exe'
            base = []
        else:
            executable = Path(sys.executable)
            base = [str(app_dir() / 'main.py')]
        args = base + ['--registry', str(self.registry.path)] + argv
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.process.start(str(executable), args)

    def download(self, panel):
        mid = panel.models.currentData()
        model = self.registry.model(mid)
        reply = QMessageBox.question(self, 'モデルの取得',
            f'{model.get("repository", mid)}\n\nモデルの利用条件を確認して取得します。数百MB〜十数GBになる場合があります。\n' + model.get('warning',''))
        if reply == QMessageBox.Yes:
            self.launch_cli(['install', mid], installing=True)

    def run_job(self):
        try:
            for panel in (self.asr, self.diar):
                if not isinstance(json.loads(panel.params.toPlainText()), dict):
                    raise ValueError('パラメータはJSONオブジェクトで指定してください。')
            if not self.source.text() or not self.output.text():
                raise ValueError('入力と出力フォルダを指定してください。')
            self.cancel_path = Path(self.output.text()) / '.cancel'
            self.launch_cli(['run', self.source.text(), '--out', self.output.text(),
                '--asr', self.asr.models.currentData(), '--diarization', self.diar.models.currentData(),
                '--asr-device', self.asr.devices.currentText() or 'cpu',
                '--diarization-device', self.diar.devices.currentText() or 'cpu',
                '--asr-params', self.asr.params.toPlainText(), '--diarization-params', self.diar.params.toPlainText(),
                '--start', str(self.start.value()), '--duration', str(self.duration.value()),
                '--cancel-file', str(self.cancel_path)])
        except Exception as exc:
            QMessageBox.warning(self, '設定エラー', str(exc))

    def read_output(self):
        self.log.insertPlainText(bytes(self.process.readAllStandardOutput()).decode('utf-8', errors='replace'))
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def finished(self, code, status):
        self.read_output()
        self.log.appendPlainText(f'\n終了コード: {code}')
        self.run_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.asr.refresh_status()
        self.diar.refresh_status()

    def cancel(self):
        if self.installing:
            self.process.kill()
            self.log.appendPlainText('ダウンロード中止。再取得時はモデルフォルダの .part を削除してください。')
        elif self.cancel_path:
            self.cancel_path.parent.mkdir(parents=True, exist_ok=True)
            self.cancel_path.touch()

    def reload_registry(self):
        if self.process.state() != QProcess.NotRunning:
            return
        try:
            registry = Registry(self.registry.path)
            self.registry = registry
            for panel, kinds, none in [(self.asr, {'asr','joint'}, False), (self.diar, {'diarization'}, True)]:
                panel.registry = registry
                panel.models.blockSignals(True)
                panel.models.clear()
                if none:
                    panel.models.addItem('なし / 統合モデルの話者情報を使う', 'none')
                for m in registry.models.values():
                    if m['type'] in kinds:
                        panel.models.addItem(m.get('label', m['id']), m['id'])
                panel.models.blockSignals(False)
                panel.changed()
        except Exception as exc:
            QMessageBox.warning(self, 'モデル定義エラー', str(exc))

    def closeEvent(self, event):
        if self.process.state() != QProcess.NotRunning:
            self.cancel()
            self.log.appendPlainText('処理を停止中です。終了してからウィンドウを閉じてください。')
            event.ignore()
        else:
            event.accept()


def launch(registry_path=None, screenshot=None):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    window = Window(registry_path)
    window.show()
    if screenshot:
        def save_and_quit():
            ok = window.grab().save(str(screenshot))
            app.exit(0 if ok else 1)
        QTimer.singleShot(700, save_and_quit)
    return app.exec()
