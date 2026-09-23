"""Desktop queue UI. All inference runs outside the GUI thread in native processes."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import threading
import tomllib
from PySide6.QtCore import Qt, QThread, Signal, QSettings, QUrl
from PySide6.QtGui import QDesktopServices, QCloseEvent
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView, QFileDialog, QMessageBox,
    QProgressBar, QPlainTextEdit, QScrollArea, QFrame, QSplitter, QDialog,
    QDialogButtonBox, QFormLayout, QAbstractItemView)
from .catalog import load_catalog, model_directory, resolve_model, Cancelled
from .pipeline import Stage, Settings, MEDIA_EXTENSIONS, run_job

STYLE = '''
QWidget { color: #243247; font-family: "Segoe UI", "Noto Sans CJK JP", sans-serif; font-size: 13px; }
QMainWindow, QDialog { background: #f3f5f8; }
QFrame#card { background: white; border: 1px solid #dce2eb; border-radius: 8px; }
QLabel#title { font-size: 25px; font-weight: 700; }
QLabel#subtitle { color: #657287; }
QLabel#section { font-size: 15px; font-weight: 600; }
QLabel#badge { font-size: 11px; font-weight: 700; padding: 3px 8px; background: #e8edf4; border-radius: 4px; }
QPushButton { background: #fff; border: 1px solid #cbd4df; padding: 7px 12px; border-radius: 5px; }
QPushButton:hover { border-color: #4876ab; background: #edf4fc; }
QPushButton#primary { background: #245b91; color: white; font-weight: 600; border-color: #245b91; }
QPushButton#primary:hover { background: #1a4b7d; }
QPushButton#stop { color: #a43535; border-color: #cf9d9d; }
QPushButton#primary:disabled, QPushButton#stop:disabled, QPushButton:disabled { color: #8993a1; background: #e9edf2; border-color: #dce2eb; }
QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox { background: white; border: 1px solid #cbd4df; border-radius: 4px; padding: 5px; min-height: 20px; }
QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: #9099a5; background: #edf0f4; border-color: #e0e4eb; }
QLabel:disabled { color: #929ba8; }
QCheckBox { spacing: 8px; font-weight: 600; }
QCheckBox::indicator { width: 18px; height: 18px; }
QTableWidget { background: white; border: 1px solid #dce2eb; gridline-color: #edf0f5; selection-background-color: #e4effb; selection-color: #173f66; }
QHeaderView::section { background: #eaf0f6; border: none; padding: 10px 8px; font-weight: 600; }
QTableWidget::item { padding: 8px; }
QPlainTextEdit { background: #f9fbfd; border: 1px solid #dce2eb; border-radius: 4px; padding: 6px; }
QProgressBar { border: none; background: #e2e8ef; border-radius: 4px; min-height: 8px; max-height: 8px; }
QProgressBar::chunk { background: #3674ad; border-radius: 4px; }
QScrollArea { border: none; background: transparent; }
'''


def card():
    frame = QFrame()
    frame.setObjectName('card')
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(14, 12, 14, 12)
    layout.setSpacing(9)
    return frame, layout


class StagePanel(QFrame):
    changed = Signal()

    def __init__(self, title: str, task: str, optional: bool):
        super().__init__()
        self.setObjectName('card')
        self.task, self.optional, self.catalog = task, optional, {}
        self.parameters = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        header = QHBoxLayout()
        self.toggle = QCheckBox(title)
        self.toggle.setChecked(not optional)
        self.toggle.setEnabled(optional)
        if not optional:
            label = QLabel(title)
            label.setObjectName('section')
            header.addWidget(label)
        else:
            header.addWidget(self.toggle)
        header.addStretch()
        self.badge = QLabel()
        self.badge.setObjectName('badge')
        header.addWidget(self.badge)
        layout.addLayout(header)
        self.body = QWidget()
        form = QGridLayout(self.body)
        form.setContentsMargins(0, 0, 0, 0)
        self.model = QComboBox()
        self.model.setMinimumContentsLength(18)
        self.model.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        form.addWidget(QLabel('モデル'), 0, 0)
        form.addWidget(self.model, 0, 1, 1, 3)
        self.device = QComboBox()
        self.device.addItems(['cpu', 'vulkan', 'metal', 'cuda', 'auto'])
        self.device.setToolTip('選んだデバイスに対応したネイティブ実行ファイルが必要です。GPUの実機確認とは別です。')
        form.addWidget(QLabel('実行先'), 1, 0)
        form.addWidget(self.device, 1, 1)
        self.language = QLineEdit('Japanese' if task == 'align' else 'ja')
        self.language.setMaximumWidth(100)
        form.addWidget(QLabel('言語'), 1, 2)
        form.addWidget(self.language, 1, 3)
        if task == 'diar':
            self.language.setEnabled(False)
        self.params = QPushButton('詳細パラメーター…')
        self.params.clicked.connect(self.edit_parameters)
        form.addWidget(self.params, 2, 0, 1, 4)
        layout.addWidget(self.body)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setObjectName('subtitle')
        layout.addWidget(self.note)
        self.toggle.toggled.connect(self.sync)
        self.model.currentIndexChanged.connect(self.model_changed)
        self.device.currentTextChanged.connect(self.changed)
        self.sync()

    def enabled_stage(self):
        return not self.optional or self.toggle.isChecked()

    def sync(self):
        enabled = self.enabled_stage()
        self.body.setEnabled(enabled)
        self.badge.setText('必須 · ON' if not self.optional else ('ON · 実行する' if enabled else 'OFF · 実行しない'))
        notes = {'asr': '文字起こしとネイティブ時刻を取得します。',
                 'diar': '話者別のトラックを作成します。' if enabled else 'オフ：話者推定せず、Transcriptトラックに配置します。',
                 'align': '文字起こしと音声を再対応付けします。' if enabled else 'オフ：ASRの時刻を使います。追加推論は行いません。'}
        self.note.setText(notes[self.task])
        self.changed.emit()

    def set_catalog(self, models):
        selected = self.model.currentData()
        self.catalog = models
        self.model.blockSignals(True)
        self.model.clear()
        for model in models.values():
            if model.task == self.task:
                self.model.addItem(model.label, model.id)
                self.model.setItemData(self.model.count() - 1, model.description or model.id, Qt.ItemDataRole.ToolTipRole)
        index = self.model.findData(selected)
        if index >= 0:
            self.model.setCurrentIndex(index)
        elif self.task == 'asr':
            index = self.model.findData('whisper-base')
            if index >= 0:
                self.model.setCurrentIndex(index)
        self.model.blockSignals(False)
        self.model_changed()

    def model_changed(self):
        self.parameters = {}
        model = self.catalog.get(self.model.currentData())
        if model:
            self.language.setText(str(model.defaults.get('language', 'Japanese' if self.task == 'align' else 'ja')))
        self.changed.emit()

    def edit_parameters(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('モデル固有パラメーター（TOML）')
        dialog.resize(500, 300)
        layout = QVBoxLayout(dialog)
        model = self.catalog.get(self.model.currentData())
        hint = ('Whisper: beam_size, temperature, no_speech_thold' if model and model.runtime == 'whisper_cpp'
                else 'audio.cppのrequest-optionを key = value で指定します。')
        layout.addWidget(QLabel(hint))
        editor = QPlainTextEdit()
        editor.setPlaceholderText('# 例\n# speaker_threshold = 0.5\n# 空ならモデルTOMLの既定値を使用')
        editor.setPlainText('\n'.join(f'{key} = {json.dumps(value, ensure_ascii=False)}' for key, value in self.parameters.items()))
        layout.addWidget(editor)
        error = QLabel('')
        error.setWordWrap(True)
        layout.addWidget(error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        def accept():
            try:
                values = tomllib.loads(editor.toPlainText())
                if any(not isinstance(v, (str, bool, int, float)) for v in values.values()):
                    raise ValueError('値は文字列・数値・真偽値のみです。テーブルや配列は使えません。')
                self.parameters = values
                dialog.accept()
            except ValueError as exc:
                error.setText(str(exc))
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def stage(self, runtime_paths, threads):
        if not self.enabled_stage():
            return None
        model_id = self.model.currentData() or ''
        model = self.catalog.get(model_id)
        device = self.device.currentText()
        key = f'{model.runtime}:{device}' if model else ''
        return Stage(model_id, device, runtime_paths.get(key, ''), self.language.text().strip(), threads, copy.deepcopy(self.parameters))


class Worker(QThread):
    progress = Signal(str)
    item = Signal(int, str, str)
    error = Signal(str)

    def __init__(self, jobs, settings, catalog, download=False):
        super().__init__()
        self.jobs, self.settings, self.catalog = jobs, copy.deepcopy(settings), catalog.copy()
        self.download = download
        self.cancel = threading.Event()

    def run(self):
        if self.download:
            try:
                ids = {s.model_id for s in [self.settings.asr, self.settings.diar, self.settings.align] if s}
                for model_id in ids:
                    resolve_model(self.catalog[model_id], self.cancel, self.progress.emit, download=True)
                self.progress.emit('選択モデルのダウンロードが完了しました。')
            except Exception as exc:
                self.error.emit(str(exc))
            return
        for index, path in self.jobs:
            if self.cancel.is_set():
                break
            self.item.emit(index, '実行中', '')
            try:
                output = run_job(Path(path), self.settings, self.catalog, self.cancel, self.progress.emit)
                self.item.emit(index, '完了', str(output))
            except Cancelled:
                self.item.emit(index, '中断', 'ユーザーが停止しました')
                break
            except Exception as exc:
                self.item.emit(index, '失敗', str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('ASR2RPP — 音声からREAPERプロジェクトへ')
        self.resize(1280, 920)
        self.setMinimumSize(1000, 720)
        self.setAcceptDrops(True)
        self.preferences = QSettings('ASR2RPP', 'ASR2RPP')
        try:
            self.runtime_paths = json.loads(self.preferences.value('runtime_paths', '{}'))
        except (ValueError, TypeError):
            self.runtime_paths = {}
        self.worker = None
        self.entries = []
        self.catalog = {}
        self.completed = 0
        outer = QWidget()
        self.setCentralWidget(outer)
        root = QVBoxLayout(outer)
        root.setContentsMargins(24, 20, 24, 18)
        root.setSpacing(14)
        top = QHBoxLayout()
        titlebox = QVBoxLayout()
        title = QLabel('ASR2RPP')
        title.setObjectName('title')
        titlebox.addWidget(title)
        subtitle = QLabel('音声・動画をキューに追加して、編集できるREAPERプロジェクトへ。')
        subtitle.setObjectName('subtitle')
        titlebox.addWidget(subtitle)
        top.addLayout(titlebox)
        top.addStretch()
        self.runtime_button = QPushButton('実行環境…')
        self.runtime_button.clicked.connect(self.runtime_dialog)
        top.addWidget(self.runtime_button)
        self.models_button = QPushButton('モデル定義を開く')
        self.models_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(model_directory()))))
        top.addWidget(self.models_button)
        self.reload_button = QPushButton('再読込')
        self.reload_button.clicked.connect(self.reload_catalog)
        top.addWidget(self.reload_button)
        root.addLayout(top)
        splitter = QSplitter()
        root.addWidget(splitter, 1)
        left = QWidget()
        leftlayout = QVBoxLayout(left)
        leftlayout.setContentsMargins(0, 0, 14, 0)
        toolbar = QHBoxLayout()
        self.add_button = QPushButton('＋ ファイルを追加')
        self.add_button.clicked.connect(self.choose_files)
        self.folder_button = QPushButton('フォルダーから')
        self.folder_button.clicked.connect(self.choose_folder)
        self.remove_button = QPushButton('選択を削除')
        self.remove_button.clicked.connect(self.remove_selected)
        self.retry_button = QPushButton('失敗・中断を再キュー')
        self.retry_button.clicked.connect(self.retry_failed)
        for button in (self.add_button, self.folder_button, self.remove_button, self.retry_button):
            toolbar.addWidget(button)
        leftlayout.addLayout(toolbar)
        self.queue_note = QLabel('キューは空です。ここに音声・動画ファイルをドロップできます。')
        self.queue_note.setObjectName('subtitle')
        self.queue_note.setWordWrap(True)
        leftlayout.addWidget(self.queue_note)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['入力ファイル', '状態', 'サイズ', '出力 / 詳細'])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().hide()
        self.table.setWordWrap(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.cellDoubleClicked.connect(self.open_output)
        leftlayout.addWidget(self.table, 1)
        explanation = QLabel('１入力 → １RPP  /  元メディアは非破壊参照  /  同名出力は連番で保護')
        explanation.setObjectName('subtitle')
        explanation.setWordWrap(True)
        leftlayout.addWidget(explanation)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(500)
        self.log.setMaximumHeight(160)
        self.log.setPlaceholderText('実行ログ：準備・ダウンロード・推論の進行とエラーを表示します。')
        leftlayout.addWidget(self.log)
        splitter.addWidget(left)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(350)
        right = QWidget()
        rightlayout = QVBoxLayout(right)
        rightlayout.setContentsMargins(0, 0, 4, 0)
        rightlayout.setSpacing(10)
        self.asr = StagePanel('ASR · 文字起こし', 'asr', False)
        self.diar = StagePanel('Diarization · 話者推定', 'diar', True)
        self.align = StagePanel('Forced Alignment · 時刻調整', 'align', True)
        for panel in (self.asr, self.diar, self.align):
            rightlayout.addWidget(panel)
            panel.changed.connect(self.update_summary)
        output, out = card()
        heading = QLabel('出力先')
        heading.setObjectName('section')
        out.addWidget(heading)
        self.same = QCheckBox('Same directory — 入力と同じフォルダー')
        self.same.setChecked(True)
        out.addWidget(self.same)
        directoryrow = QHBoxLayout()
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText('Output directory を指定')
        self.output_browse = QPushButton('選択…')
        self.output_browse.clicked.connect(self.choose_output)
        directoryrow.addWidget(self.output_dir, 1)
        directoryrow.addWidget(self.output_browse)
        out.addLayout(directoryrow)
        self.output_hint = QLabel()
        self.output_hint.setWordWrap(True)
        self.output_hint.setObjectName('subtitle')
        out.addWidget(self.output_hint)
        self.same.toggled.connect(self.output_changed)
        self.output_dir.textChanged.connect(self.update_summary)
        rightlayout.addWidget(output)
        clip, cl = card()
        self.clip_on = QCheckBox('検証用に時間範囲を限定する')
        cl.addWidget(self.clip_on)
        self.clip_body = QWidget()
        cliprow = QHBoxLayout(self.clip_body)
        cliprow.setContentsMargins(0, 0, 0, 0)
        self.clip_start = QDoubleSpinBox()
        self.clip_start.setRange(0, 864000)
        self.clip_start.setSuffix(' 秒から')
        self.clip_length = QDoubleSpinBox()
        self.clip_length.setRange(0.1, 60)
        self.clip_length.setValue(55)
        self.clip_length.setSuffix(' 秒間')
        cliprow.addWidget(self.clip_start)
        cliprow.addWidget(self.clip_length)
        cl.addWidget(self.clip_body)
        self.clip_body.setEnabled(False)
        self.clip_on.toggled.connect(self.clip_body.setEnabled)
        rightlayout.addWidget(clip)
        rightlayout.addStretch()
        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setSizes([780, 400])
        footer = QHBoxLayout()
        statusbox = QVBoxLayout()
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        statusbox.addWidget(self.summary)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        statusbox.addWidget(self.progress)
        footer.addLayout(statusbox, 1)
        self.download_button = QPushButton('選択モデルを取得')
        self.download_button.clicked.connect(lambda: self.start_work(True))
        footer.addWidget(self.download_button)
        self.stop_button = QPushButton('停止')
        self.stop_button.setObjectName('stop')
        self.stop_button.clicked.connect(self.stop_work)
        self.stop_button.setEnabled(False)
        footer.addWidget(self.stop_button)
        self.start_button = QPushButton('キューを実行')
        self.start_button.setObjectName('primary')
        self.start_button.setMinimumWidth(150)
        self.start_button.clicked.connect(lambda: self.start_work(False))
        footer.addWidget(self.start_button)
        root.addLayout(footer)
        self.reload_catalog()
        self.restore_preferences()
        self.output_changed()

    def reload_catalog(self):
        self.catalog, errors = load_catalog()
        for panel in (self.asr, self.diar, self.align):
            panel.set_catalog(self.catalog)
        for error in errors:
            self.log.appendPlainText('モデル定義エラー: ' + error)
        self.update_summary()

    def output_changed(self):
        self.output_dir.setEnabled(not self.same.isChecked())
        self.output_browse.setEnabled(not self.same.isChecked())
        self.output_hint.setText('各入力ファイルの隣にRPPを保存します。' if self.same.isChecked()
                                 else '指定先に各ファイルのRPPを保存します。空欄のまま実行できません。')
        self.update_summary()

    def update_summary(self):
        if not hasattr(self, 'summary'):
            return
        if self.worker is None:
            self.summary.setText('ASR → ' + ('時刻調整 ON' if self.align.enabled_stage() else '時刻調整 OFF') +
                                 ' → ' + ('話者推定 ON' if self.diar.enabled_stage() else '話者推定 OFF') + ' → RPP')
        valid_output = self.same.isChecked() or bool(self.output_dir.text().strip())
        self.start_button.setEnabled(self.worker is None and valid_output and any(e['status'] == '待機' for e in self.entries))

    def choose_files(self):
        extensions = ' '.join('*' + x for x in sorted(MEDIA_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(self, '音声・動画を追加', '', f'Audio / Video ({extensions})')
        self.add_paths(files)

    def choose_folder(self):
        directory = QFileDialog.getExistingDirectory(self, 'フォルダー内の音声・動画を追加（直下のみ）')
        if directory:
            self.add_paths(sorted(str(x) for x in Path(directory).iterdir() if x.is_file()))

    def add_paths(self, paths):
        if self.worker:
            return
        known = {e['path'] for e in self.entries}
        for value in paths:
            path = Path(value).expanduser().resolve()
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS and str(path) not in known:
                self.entries.append({'path': str(path), 'status': '待機', 'output': ''})
                known.add(str(path))
        self.render_queue()

    def render_queue(self):
        self.table.setRowCount(len(self.entries))
        for row, entry in enumerate(self.entries):
            path = Path(entry['path'])
            size = f'{path.stat().st_size / 1024**2:.1f} MB' if path.exists() else '不明'
            for col, text in enumerate([path.name, entry['status'], size, entry['output']]):
                item = QTableWidgetItem(text)
                item.setToolTip(entry['path'] if col == 0 else text)
                self.table.setItem(row, col, item)
            self.table.setRowHeight(row, 40)
        self.queue_note.setText(f'{len(self.entries)} ファイル  ·  上から順番に実行  ·  完了行をダブルクリックで出力先を開く' if self.entries
                               else 'キューは空です。ここに音声・動画ファイルをドロップできます。')
        self.update_summary()

    def remove_selected(self):
        rows = {x.row() for x in self.table.selectionModel().selectedRows()}
        self.entries = [e for i, e in enumerate(self.entries) if i not in rows]
        self.render_queue()

    def retry_failed(self):
        for entry in self.entries:
            if entry['status'] in {'失敗', '中断'}:
                entry.update(status='待機', output='')
        self.render_queue()

    def choose_output(self):
        directory = QFileDialog.getExistingDirectory(self, 'Output directory', self.output_dir.text())
        if directory:
            self.output_dir.setText(directory)

    def settings(self):
        threads = int(self.preferences.value('threads', 4))
        return Settings(self.asr.stage(self.runtime_paths, threads), self.diar.stage(self.runtime_paths, threads),
                        self.align.stage(self.runtime_paths, threads), self.same.isChecked(),
                        self.output_dir.text().strip(), self.runtime_paths.get('ffmpeg', ''),
                        self.clip_start.value() if self.clip_on.isChecked() else 0,
                        self.clip_length.value() if self.clip_on.isChecked() else 0)

    def set_busy(self, busy):
        for widget in [self.add_button, self.folder_button, self.remove_button, self.retry_button,
                       self.runtime_button, self.models_button, self.reload_button, self.download_button,
                       self.asr, self.diar, self.align, self.same, self.clip_on, self.clip_body]:
            widget.setEnabled(not busy)
        self.output_dir.setEnabled(not busy and not self.same.isChecked())
        self.output_browse.setEnabled(not busy and not self.same.isChecked())
        self.clip_body.setEnabled(not busy and self.clip_on.isChecked())
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(busy)

    def start_work(self, download=False):
        if self.worker:
            return
        try:
            settings = self.settings()
            validation = copy.deepcopy(settings)
            if download:
                validation.same_directory = True
            validation.validate(self.catalog)
            jobs = [(i, e['path']) for i, e in enumerate(self.entries) if e['status'] == '待機']
            if not download and not jobs:
                return
            self.save_preferences()
            self.completed = 0
            self.progress.setRange(0, 0 if download else len(jobs))
            self.worker = Worker(jobs, settings, self.catalog, download)
            self.worker.progress.connect(self.show_progress)
            self.worker.item.connect(self.item_changed)
            self.worker.error.connect(self.show_error)
            self.worker.finished.connect(self.work_finished)
            self.set_busy(True)
            self.worker.start()
        except Exception as exc:
            QMessageBox.warning(self, '設定を確認してください', str(exc))

    def show_progress(self, text):
        self.summary.setText(text[:150])
        self.log.appendPlainText(text)

    def show_error(self, text):
        self.log.appendPlainText('ERROR: ' + text)
        self.summary.setText(text[:150])

    def item_changed(self, index, status, detail):
        self.entries[index].update(status=status, output=detail)
        if status in {'完了', '失敗', '中断'}:
            self.completed += 1
            self.progress.setValue(self.completed)
        if status == '失敗':
            self.log.appendPlainText('ERROR: ' + detail)
        self.render_queue()

    def stop_work(self):
        if self.worker:
            self.worker.cancel.set()
            self.stop_button.setEnabled(False)
            self.summary.setText('停止処理中… 実行プロセスを終了し、残りは待機のまま保持します。')

    def work_finished(self):
        old = self.worker
        self.worker = None
        self.set_busy(False)
        self.progress.setRange(0, max(1, self.completed))
        self.progress.setValue(self.completed)
        self.update_summary()
        if old:
            old.deleteLater()
        self.log.appendPlainText('処理が終了しました。各行の状態と詳細をご確認ください。')

    def runtime_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('実行環境 — 空欄なら同梱エンジンを自動検出')
        dialog.resize(700, 480)
        layout = QVBoxLayout(dialog)
        note = QLabel('モデル重みは「選択モデルを取得」で別途ダウンロードします。\nGPUを使う場合は対応ビルドを指定してください。CPUビルドではGPUを使用できません。')
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        fields = {}
        keys = ['ffmpeg', 'whisper_cpp:cpu', 'whisper_cpp:vulkan', 'whisper_cpp:metal',
                'whisper_cpp:cuda', 'audio_cpp:cpu', 'audio_cpp:vulkan', 'audio_cpp:metal', 'audio_cpp:cuda']
        for key in keys:
            row = QHBoxLayout()
            edit = QLineEdit(self.runtime_paths.get(key, ''))
            edit.setPlaceholderText('同梱版 / PATHから自動検出')
            button = QPushButton('…')
            button.setMaximumWidth(40)
            def browse(checked=False, field=edit):
                filename, _ = QFileDialog.getOpenFileName(dialog, '実行ファイルを選択')
                if filename:
                    field.setText(filename)
            button.clicked.connect(browse)
            row.addWidget(edit)
            row.addWidget(button)
            form.addRow(key, row)
            fields[key] = edit
        threads = QSpinBox()
        threads.setRange(1, 128)
        threads.setValue(int(self.preferences.value('threads', 4)))
        form.addRow('CPU threads', threads)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec():
            self.runtime_paths = {k: v.text().strip() for k, v in fields.items() if v.text().strip()}
            self.preferences.setValue('threads', threads.value())
            self.save_preferences()

    def save_preferences(self):
        self.preferences.setValue('runtime_paths', json.dumps(self.runtime_paths))
        self.preferences.setValue('output', self.output_dir.text())
        self.preferences.setValue('same', self.same.isChecked())
        for key, panel in [('asr', self.asr), ('diar', self.diar), ('align', self.align)]:
            self.preferences.setValue(key + '/model', panel.model.currentData() or '')
            self.preferences.setValue(key + '/device', panel.device.currentText())
            self.preferences.setValue(key + '/enabled', panel.enabled_stage())
            self.preferences.setValue(key + '/language', panel.language.text())
            self.preferences.setValue(key + '/parameters', json.dumps(panel.parameters))
        self.preferences.sync()

    def restore_preferences(self):
        self.output_dir.setText(self.preferences.value('output', ''))
        self.same.setChecked(self.preferences.value('same', True, type=bool))
        for key, panel in [('asr', self.asr), ('diar', self.diar), ('align', self.align)]:
            index = panel.model.findData(self.preferences.value(key + '/model', ''))
            if index >= 0:
                panel.model.setCurrentIndex(index)
            panel.device.setCurrentText(self.preferences.value(key + '/device', 'cpu'))
            if panel.optional:
                panel.toggle.setChecked(self.preferences.value(key + '/enabled', False, type=bool))
            panel.language.setText(self.preferences.value(key + '/language', panel.language.text()))
            try:
                panel.parameters = json.loads(self.preferences.value(key + '/parameters', '{}'))
            except ValueError:
                panel.parameters = {}

    def open_output(self, row, column):
        entry = self.entries[row]
        if entry['status'] == '完了':
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(entry['output']).parent)))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and self.worker is None:
            event.acceptProposedAction()

    def dropEvent(self, event):
        self.add_paths([u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()])
        event.acceptProposedAction()

    def closeEvent(self, event: QCloseEvent):
        if self.worker is not None:
            self.stop_work()
            event.ignore()
            return
        self.save_preferences()
        event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()
