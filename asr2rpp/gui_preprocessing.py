"""Preprocessing UI extensions; shares the existing queue and stage widgets."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import tomllib
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QLabel,
    QRadioButton, QButtonGroup, QDialog, QDialogButtonBox, QPlainTextEdit, QMessageBox, QSpinBox)
from . import gui as base
from .catalog import Cancelled, resolve_model
from .preprocessing import Settings, run_job


class PreprocessPanel(base.StagePanel):
    def __init__(self):
        super().__init__('前処理 · 背景音除去', 'sep', True)
        form = self.body.layout()
        self.language.hide()
        form.itemAtPosition(1, 2).widget().hide()
        form.removeWidget(self.device)
        form.addWidget(self.device, 1, 1, 1, 3)
        form.removeWidget(self.params)
        form.removeWidget(self.param_summary)
        self.overlap_label = QLabel('Overlap')
        self.overlap = QSpinBox()
        self.overlap.setRange(1, 8)
        self.overlap.setValue(2)
        self.overlap.setToolTip('大きいほど分離境界の安定性を上げやすい一方、処理時間とメモリ使用量が増えます。')
        form.addWidget(self.overlap_label, 2, 0)
        form.addWidget(self.overlap, 2, 1)
        form.addWidget(self.params, 3, 0, 1, 4)
        form.addWidget(self.param_summary, 4, 0, 1, 4)
        self.reference_box = QWidget()
        layout = QVBoxLayout(self.reference_box)
        layout.setContentsMargins(0, 4, 0, 0)
        title = QLabel('RPPで再生する音声')
        title.setObjectName('section')
        layout.addWidget(title)
        self.original = QRadioButton('元メディア — 前処理は推論にだけ使用')
        self.processed = QRadioButton('前処理済み音声 — WAVを保存して参照')
        self.references = QButtonGroup(self)
        self.references.addButton(self.original)
        self.references.addButton(self.processed)
        self.original.setChecked(True)
        layout.addWidget(self.original)
        layout.addWidget(self.processed)
        form.addWidget(self.reference_box, 5, 0, 1, 4)
        self.original.toggled.connect(self.sync)
        self.processed.toggled.connect(self.sync)
        self.overlap.valueChanged.connect(self.update_parameter_summary)
        self.sync()

    def model_changed(self):
        super().model_changed()
        if hasattr(self, 'overlap'):
            model = self.catalog.get(self.model.currentData())
            if model and model.family == 'mel_band_roformer':
                self.overlap_label.show()
                self.overlap.show()
                self.overlap.setValue(int(model.defaults.get('session', {}).get('num_overlap', 2)))
            else:
                self.overlap_label.hide()
                self.overlap.hide()
            self.update_parameter_summary()

    def update_parameter_summary(self):
        super().update_parameter_summary()
        model = self.catalog.get(self.model.currentData())
        if hasattr(self, 'overlap') and model and model.family == 'mel_band_roformer':
            self.param_summary.setText(self.param_summary.text() + f'  /  Overlap={self.overlap.value()}')

    def stage(self, runtime_paths, threads, runtime_defaults=None):
        stage = super().stage(runtime_paths, threads, runtime_defaults)
        model = self.catalog.get(self.model.currentData())
        if stage and model and model.family == 'mel_band_roformer':
            stage.parameters['num_overlap'] = self.overlap.value()
        return stage

    def sync(self):
        enabled = self.enabled_stage()
        self.body.setEnabled(enabled)
        self.badge.setText('ON · 実行する' if enabled else 'OFF · 実行しない')
        saved = hasattr(self, 'processed') and self.processed.isChecked()
        self.note.setText('OFF：元メディアをそのまま使用。' if not enabled else
                         ('処理済み音声を <RPP名>_vocals.wav として保存・参照。' if saved else
                          '処理済み音声は推論だけに使用。RPPは元メディアを参照。'))
        self.changed.emit()

    def reference_mode(self):
        return 'processed' if self.enabled_stage() and self.processed.isChecked() else 'original'

    def edit_parameters(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('背景音除去のパラメーター（TOML）')
        dialog.resize(500, 280)
        layout = QVBoxLayout(dialog)
        hint = QLabel('追加の audio.cpp session-option を指定します。Overlap は前画面で変更できます。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        editor = QPlainTextEdit()
        editor.setPlainText('\n'.join(f'{key} = {json.dumps(value, ensure_ascii=False)}' for key, value in self.parameters.items()))
        layout.addWidget(editor)
        error = QLabel()
        error.setWordWrap(True)
        layout.addWidget(error)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        def accept():
            try:
                values = tomllib.loads(editor.toPlainText())
                if any(not isinstance(v, (str, int, float, bool)) for v in values.values()):
                    raise ValueError('文字列・数値・真偽値のみ指定できます。')
                if 'num_overlap' in values:
                    self.overlap.setValue(int(values.pop('num_overlap')))
                self.parameters = values
                self.update_parameter_summary()
                dialog.accept()
            except ValueError as exc:
                error.setText(str(exc))
        buttons.accepted.connect(accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()


class Worker(base.Worker):
    def run(self):
        if self.download:
            try:
                stages = [self.settings.asr, self.settings.diar, self.settings.align, self.settings.preprocess]
                for model_id in dict.fromkeys(stage.model_id for stage in stages if stage):
                    resolve_model(self.catalog[model_id], self.cancel, self.progress.emit, download=True, keep_source=self.keep_model_sources)
                self.progress.emit('選択モデルの準備が完了しました。')
            except Exception as exc:
                self.error.emit(str(exc))
            return
        if self.queue_strategy == 'stage' and len(self.jobs) > 1:
            from .queue_runner import run_queue
            try:
                run_queue(self.jobs, self.settings, self.catalog, self.cancel, self.progress.emit,
                          self.item.emit, self.batch_audio_ram_mb)
            except Cancelled:
                pass
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


class MainWindow(base.MainWindow):
    def __init__(self):
        super().__init__()
        self.preprocess = PreprocessPanel()
        self.asr.parentWidget().layout().insertWidget(0, self.preprocess)
        self.preprocess.set_catalog(self.catalog)
        self.preprocess.changed.connect(self.update_summary)
        index = self.preprocess.model.findData(self.preferences.value('preprocess/model', ''))
        if index >= 0:
            self.preprocess.model.setCurrentIndex(index)
        saved_device = str(self.preferences.value('preprocess/device', 'default'))
        device_index = self.preprocess.device.findData(saved_device)
        if device_index < 0:
            device_index = self.preprocess.device.findText(saved_device)
        self.preprocess.device.setCurrentIndex(max(0, device_index))
        try:
            self.preprocess.parameters = json.loads(self.preferences.value('preprocess/parameters', '{}'))
        except (TypeError, ValueError):
            self.preprocess.parameters = {}
        self.preprocess.overlap.setValue(int(self.preferences.value('preprocess/overlap', self.preprocess.overlap.value())))
        self.preprocess.processed.setChecked(self.preferences.value('preprocess/reference', 'original') == 'processed')
        if not self.preprocess.processed.isChecked():
            self.preprocess.original.setChecked(True)
        self.preprocess.toggle.setChecked(self.preferences.value('preprocess/enabled', False, type=bool))
        self.preprocess.sync()
        for label in self.findChildren(QLabel):
            if label.text().startswith('１入力 → １RPP'):
                label.setText('１入力 → １RPP  /  選択した音声を非破壊参照  /  同名出力は連番で保護')
        self.update_summary()

    def reload_catalog(self):
        super().reload_catalog()
        if hasattr(self, 'preprocess'):
            self.preprocess.set_catalog(self.catalog)

    def settings(self):
        settings = super().settings()
        if not hasattr(self, 'preprocess'):
            return Settings(**vars(settings))
        stage = self.preprocess.stage(self.runtime_paths, int(self.preferences.value('threads', 4)),
                                      self.runtime_defaults)
        return Settings(**vars(settings), preprocess=stage, reference_audio=self.preprocess.reference_mode())

    def save_preferences(self):
        super().save_preferences()
        if hasattr(self, 'preprocess'):
            panel = self.preprocess
            self.preferences.setValue('preprocess/model', panel.model.currentData() or '')
            self.preferences.setValue('preprocess/device', panel.device.currentData() or 'default')
            self.preferences.setValue('preprocess/enabled', panel.enabled_stage())
            self.preferences.setValue('preprocess/reference', 'processed' if panel.processed.isChecked() else 'original')
            self.preferences.setValue('preprocess/overlap', panel.overlap.value())
            self.preferences.setValue('preprocess/parameters', json.dumps(panel.parameters))
            self.preferences.sync()

    def set_busy(self, busy):
        super().set_busy(busy)
        if hasattr(self, 'preprocess'):
            self.preprocess.setEnabled(not busy)

    def update_summary(self):
        super().update_summary()
        if hasattr(self, 'preprocess') and self.worker is None:
            enabled = self.preprocess.enabled_stage()
            mode = '前処理済みWAV（保存）' if self.preprocess.reference_mode() == 'processed' else '元メディア'
            self.summary.setText(('背景音除去 ON → ' if enabled else '背景音除去 OFF → ') +
                                 'ASR → ' + ('時刻調整 ON' if self.align.enabled_stage() else '時刻調整 OFF') +
                                 ' → ' + ('話者推定 ON' if self.diar.enabled_stage() else '話者推定 OFF') +
                                 '  |  RPP参照: ' + mode + self.runtime_summary())

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
            self.worker = Worker(jobs, settings, self.catalog, download, self.keep_model_sources,
                                 self.queue_strategy, self.batch_audio_ram_mb)
            self.worker.progress.connect(self.show_progress)
            self.worker.item.connect(self.item_changed)
            self.worker.error.connect(self.show_error)
            self.worker.finished.connect(self.work_finished)
            self.set_busy(True)
            self.worker.start()
        except Exception as exc:
            QMessageBox.warning(self, '設定を確認してください', str(exc))


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(base.STYLE + '\nQRadioButton:disabled { color: #929ba8; }')
    window = MainWindow()
    window.show()
    return app.exec()
