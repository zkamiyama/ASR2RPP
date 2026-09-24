"""Compact DCC-style desktop UI for ASR2RPP.

The shell is intentionally workflow-first: queue on the left, processing inspector on
the right, one status/action bar at the bottom. Explanatory prose lives in tooltips.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys
import threading
import tomllib

from PySide6.QtCore import Qt, QThread, Signal, QSettings, QUrl, QSize
from PySide6.QtGui import QAction, QDesktopServices, QIcon, QPainter, QColor, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog,
    QDialogButtonBox, QFileDialog, QFormLayout, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QMainWindow,
    QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QDoubleSpinBox, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)

from .catalog import (
    Cancelled, assets_root, cache_root, data_root, load_catalog, model_directory,
    resolve_model, weights_root,
)
from .pipeline import MEDIA_EXTENSIONS, Stage
from .preprocessing import Settings, run_job
from .queue_runner import run_queue
from .parameter_specs import specs_for
from .adapters import executable as runtime_executable


STYLE = r"""
* {
    font-size: 11px;
    color: #d7dde6;
}
QMainWindow, QDialog, QWidget { background: #1b1f24; }
QLabel, QCheckBox { background: transparent; }
QToolTip {
    background: #111419;
    color: #e6ebf2;
    border: 1px solid #3a424d;
    padding: 5px 7px;
}
QSplitter::handle { background: #111419; width: 1px; }
QFrame#inspector, QFrame#statusBar { background: #20252b; }
QFrame#stage {
    background: #242a31;
    border: 1px solid #313944;
    border-radius: 4px;
}
QFrame#stage[disabledStage="true"] { background: #20252a; }
QWidget#stageBody { background: transparent; }
QLabel#section {
    color: #eef2f6;
    background: transparent;
    font-size: 10px;
    font-weight: 700;
}
QLabel#muted { color: #8993a0; }
QLabel#status { color: #aeb7c4; }
QTableWidget {
    background: #171b20;
    alternate-background-color: #1a1f25;
    border: none;
    gridline-color: #272e37;
    selection-background-color: #334b68;
    selection-color: #ffffff;
}
QTableWidget::item { padding: 4px 6px; border-bottom: 1px solid #252c34; }
QHeaderView::section {
    background: #20252b;
    color: #9fa9b6;
    border: none;
    border-right: 1px solid #2c333d;
    border-bottom: 1px solid #343c47;
    padding: 5px 6px;
    font-weight: 600;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background: #181d22;
    color: #dce2ea;
    border: 1px solid #3a424d;
    border-radius: 2px;
    padding: 0 4px;
    min-height: 18px;
    max-height: 18px;
    selection-background-color: #3f6f9f;
}

QFrame#editablePreset {
    background: #181d22;
    border: 1px solid #3a424d;
    border-radius: 2px;
    min-height: 18px; max-height: 18px;
}
QFrame#editablePreset QLineEdit {
    background: transparent; border: none; padding: 0 4px;
    min-height: 18px; max-height: 18px;
}
QFrame#editablePreset QToolButton {
    background: #20262c; border: none; border-left: 1px solid #303842;
    border-radius: 0; padding: 0; min-width: 18px; max-width: 18px;
    min-height: 18px; max-height: 18px;
}
QFrame#editablePreset QToolButton:hover { background: #252b32; }
QPlainTextEdit {
    background: #15191e;
    color: #dce2ea;
    border: 1px solid #3a424d;
    border-radius: 2px;
    padding: 5px;
    selection-background-color: #3f6f9f;
}
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
    color: #68717d;
    background: #1b2026;
    border-color: #2a3139;
}
QComboBox::drop-down {
    border: none;
    border-left: 1px solid #303842;
    width: 18px;
    background: #20262c;
}
QComboBox QLineEdit, QSpinBox QLineEdit, QDoubleSpinBox QLineEdit {
    background: transparent;
    border: none;
    padding: 0 3px;
    min-height: 16px;
    max-height: 16px;
}
QComboBox QAbstractItemView {
    background: #20252b;
    color: #dce2ea;
    border: 1px solid #444d59;
    selection-background-color: #365c83;
}
QPushButton, QToolButton {
    background: #2a3038;
    color: #dce2ea;
    border: 1px solid #3a424d;
    border-radius: 2px;
    padding: 0 6px;
    min-height: 18px;
    max-height: 18px;
}
QPushButton:hover, QToolButton:hover { background: #333b45; border-color: #53606e; }
QPushButton:pressed, QToolButton:pressed { background: #20262d; }
QPushButton:disabled, QToolButton:disabled { color: #626c78; background: #23282e; border-color: #2d343d; }
QToolButton#stageToggle {
    min-width: 30px; max-width: 30px; min-height: 16px; max-height: 16px;
    padding: 0; font-weight: 700; color: #7f8995; background: #191d22;
}
QToolButton#stageToggle:checked { color: #ffffff; background: #2f76b7; border-color: #4b8cca; }
QToolButton#iconButton { min-width: 18px; max-width: 18px; min-height: 18px; max-height: 18px; padding: 0; }
QToolButton#footerIcon, QToolButton#languageButton {
    min-width: 24px; max-width: 24px; min-height: 24px; max-height: 24px; padding: 0;
}
QToolButton#runButton {
    min-width: 76px; max-width: 76px; min-height: 24px; max-height: 24px;
    padding: 0 7px; font-weight: 800; letter-spacing: 1px;
    color: #ffffff; background: #247f5d; border-color: #319b75;
}
QToolButton#runButton:hover { background: #2b906a; }
QToolButton#runButton[running="true"] { background: #a84343; border-color: #ca5c5c; }
QToolButton#runButton[running="true"]:hover { background: #ba4b4b; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator {
    width: 13px; height: 13px; border-radius: 2px;
    background: #15191e; border: 1px solid #596472;
}
QCheckBox::indicator:checked { background: #347fbd; border-color: #5597cd; }
QCheckBox::indicator:disabled { background: #20252b; border-color: #353d47; }
QProgressBar {
    border: none; background: #13171b; min-height: 3px; max-height: 3px;
}
QProgressBar::chunk { background: #3d8ed0; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #181c21; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #404955; min-height: 24px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QListWidget {
    background: #171b20; border: none; border-right: 1px solid #303741; outline: none;
}
QListWidget::item { padding: 10px 12px; color: #9ca6b2; }
QListWidget::item:selected { background: #29323c; color: #ffffff; border-left: 2px solid #4b91ca; }
QMenu {
    background: #20252b; color: #dce2ea; border: 1px solid #3a424d; padding: 4px;
}
QMenu::item { padding: 5px 26px 5px 24px; }
QMenu::item:selected { background: #355c82; }
QMenu::separator { height: 1px; background: #343c46; margin: 4px 7px; }
"""
STYLE += '\nQComboBox::down-arrow { image: url("' + (assets_root() / "assets" / "icons" / "arrow_drop_down.svg").as_posix() + '"); width: 14px; height: 14px; }'
STYLE += '\nQCheckBox::indicator:checked { image: url("' + (assets_root() / "assets" / "icons" / "check.svg").as_posix() + '"); }'


TEXT = {
    "ja": {
        "queue_name": "入力",
        "queue_status": "状態",
        "queue_size": "サイズ",
        "queue_output": "出力",
        "drop": "音声・動画をここへドロップ",
        "add_files": "ファイルを追加",
        "add_folder": "フォルダーから追加",
        "remove": "選択を削除",
        "retry": "失敗・中断を再試行",
        "open_output": "出力先を開く",
        "clear": "キューをクリア",
        "sep": "BACKGROUND REMOVAL",
        "asr": "ASR",
        "align": "FORCED ALIGN",
        "diar": "DIARIZATION",
        "model": "MODEL",
        "backend": "BACKEND",
        "language": "LANG",
        "parameters": "パラメータ",
        "rpp_audio": "RPP AUDIO",
        "original": "元メディア",
        "processed": "処理済み",
        "output": "OUTPUT",
        "same_dir": "入力ファイルと同じ場所",
        "choose": "選択",
        "ready": "準備完了",
        "settings": "設定",
        "ui_language": "UI言語を切替",
        "general": "一般",
        "runtime": "実行環境",
        "advanced": "詳細",
        "model_dir": "モデル保存先",
        "temp_dir": "一時ファイル先",
        "queue_order": "キュー処理順",
        "stage_major": "ステージ優先",
        "file_major": "ファイル優先",
        "batch_limit": "audio.cppバッチ上限",
        "threads": "CPU threads",
        "whisper_backend": "whisper.cpp",
        "audio_backend": "audio.cpp",
        "keep_source": "変換成功後も元チェックポイントを保持",
        "prepare_models": "選択モデルを準備",
        "open_models": "モデル保存先を開く",
        "open_toml": "モデルTOMLを開く",
        "reload_toml": "モデル定義を再読込",
        "exe_override": "実行ファイル上書き",
        "save": "保存",
        "cancel": "キャンセル",
        "reset": "既定値へ戻す",
        "apply": "適用",
        "default": "Default",
        "cpu": "CPU",
        "vulkan": "Vulkan",
        "waiting": "待機",
        "running": "実行中",
        "done": "完了",
        "failed": "失敗",
        "stopped": "中断",
        "preparing": "モデル準備中",
        "select_output": "出力先を選択",
        "no_params": "このモデルには追加の公開推論パラメータがありません。",
    },
    "en": {
        "queue_name": "Input",
        "queue_status": "Status",
        "queue_size": "Size",
        "queue_output": "Output",
        "drop": "Drop audio or video files here",
        "add_files": "Add files",
        "add_folder": "Add folder",
        "remove": "Remove selected",
        "retry": "Retry failed/stopped",
        "open_output": "Open output folder",
        "clear": "Clear queue",
        "sep": "BACKGROUND REMOVAL",
        "asr": "ASR",
        "align": "FORCED ALIGN",
        "diar": "DIARIZATION",
        "model": "MODEL",
        "backend": "BACKEND",
        "language": "LANG",
        "parameters": "Parameters",
        "rpp_audio": "RPP AUDIO",
        "original": "Original",
        "processed": "Processed",
        "output": "OUTPUT",
        "same_dir": "Save next to each input",
        "choose": "Browse",
        "ready": "Ready",
        "settings": "Settings",
        "ui_language": "Switch UI language",
        "general": "General",
        "runtime": "Runtime",
        "advanced": "Advanced",
        "model_dir": "Model directory",
        "temp_dir": "Temporary directory",
        "queue_order": "Queue order",
        "stage_major": "Stage-major",
        "file_major": "File-major",
        "batch_limit": "audio.cpp batch limit",
        "threads": "CPU threads",
        "whisper_backend": "whisper.cpp",
        "audio_backend": "audio.cpp",
        "keep_source": "Keep source checkpoint after conversion",
        "prepare_models": "Prepare selected models",
        "open_models": "Open model directory",
        "open_toml": "Open model TOML",
        "reload_toml": "Reload model definitions",
        "exe_override": "Executable overrides",
        "save": "Save",
        "cancel": "Cancel",
        "reset": "Reset defaults",
        "apply": "Apply",
        "default": "Default",
        "cpu": "CPU",
        "vulkan": "Vulkan",
        "waiting": "Queued",
        "running": "Running",
        "done": "Done",
        "failed": "Failed",
        "stopped": "Stopped",
        "preparing": "Preparing models",
        "select_output": "Choose output directory",
        "no_params": "This model exposes no additional inference parameters.",
    },
}

MODEL_TEXT = {
    "anime-whisper": {
        "ja": ("Anime Whisper · 実験的変換版", "GGML変換版。配布者から認識異常の注意があります。"),
        "en": ("Anime Whisper · Experimental", "Experimental GGML conversion; the distributor warns of possible recognition anomalies."),
    },
    "mel-big-beta7": {
        "ja": ("Mel-Band RoFormer big beta7", "背景音除去。元CKPTをローカルで検証し、PyTorchなしでaudio.cpp F16 GGUFへ変換します。"),
        "en": ("Mel-Band RoFormer big beta7", "Background removal. The exact source checkpoint is verified and converted locally to audio.cpp F16 GGUF without PyTorch."),
    },
    "nemotron-asr": {
        "ja": ("Nemotron 3.5 ASR · Q8", "token emission frame由来の時刻。精密な境界にはForced Alignmentを併用できます。"),
        "en": ("Nemotron 3.5 ASR · Q8", "Timing is derived from token emission frames. Forced Alignment can refine edit boundaries."),
    },
    "nemotron-diarization": {
        "ja": ("Nemotron 3 Diarization · 8話者", "最大8話者の話者推定。音源分離は行いません。"),
        "en": ("Nemotron 3 Diarization · 8 speakers", "Speaker diarization for up to eight speakers. It does not perform source separation."),
    },
    "qwen-forced-aligner": {
        "ja": ("Qwen3 Forced Aligner · Q8", "認識テキストと対応音声を再整列します。"),
        "en": ("Qwen3 Forced Aligner · Q8", "Re-aligns recognized transcript text to the corresponding audio."),
    },
    "vibevoice-asr": {
        "ja": ("VibeVoice ASR · Q8 / 大容量", "長時間向けオフライン統合ASR。メモリ使用量が大きいモデルです。"),
        "en": ("VibeVoice ASR · Q8 / Large", "Long-form offline integrated ASR with comparatively high memory usage."),
    },
    "whisper-base": {
        "ja": ("Whisper Base · 軽量", "標準Whisper Base。軽量な既定モデル・動作確認向け。"),
        "en": ("Whisper Base · Lightweight", "Standard Whisper Base, used as the lightweight default and smoke-test model."),
    },
}


def model_text(model, lang: str) -> tuple[str, str]:
    localized = MODEL_TEXT.get(model.id, {}).get(lang)
    if localized:
        return localized
    return model.label, model.description or model.id


def icon(name: str) -> QIcon:
    return QIcon(str(assets_root() / "assets" / "icons" / f"{name}.svg"))


def default_storage_hint(kind: str) -> str:
    if sys.platform == "win32":
        return rf"%LOCALAPPDATA%\ASR2RPP\{kind}"
    return str(data_root() / kind)


class QueueTable(QTableWidget):
    filesDropped = Signal(list)

    def __init__(self):
        super().__init__(0, 4)
        self.placeholder = ""
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DropOnly)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.verticalHeader().hide()
        self.setShowGrid(False)
        self.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.filesDropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.rowCount() == 0 and self.placeholder:
            painter = QPainter(self.viewport())
            painter.setPen(QColor("#707b88"))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self.placeholder)



class EditablePresetField(QFrame):
    """Free text plus an explicit preset menu, avoiding ambiguous editable combos."""
    currentTextChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("editablePreset")
        self._presets = []
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.edit = QLineEdit()
        self.edit.textChanged.connect(self.currentTextChanged)
        row.addWidget(self.edit, 1)
        self.menu_button = QToolButton()
        self.menu_button.setIcon(icon("arrow_drop_down"))
        self.menu_button.setIconSize(QSize(14, 14))
        self.menu_button.clicked.connect(self._show_menu)
        row.addWidget(self.menu_button)
        self.setFixedHeight(20)

    def setPresets(self, values):
        seen = set()
        self._presets = []
        for value in values:
            value = str(value)
            if value and value not in seen:
                seen.add(value)
                self._presets.append(value)

    def _show_menu(self):
        menu = QMenu(self)
        for value in self._presets:
            action = menu.addAction(value)
            action.triggered.connect(lambda checked=False, v=value: self.setCurrentText(v))
        if menu.actions():
            menu.exec(self.menu_button.mapToGlobal(self.menu_button.rect().bottomLeft()))

    def currentText(self):
        return self.edit.text()

    def setCurrentText(self, value):
        self.edit.setText(str(value or ""))

    def setToolTip(self, text):
        super().setToolTip(text)
        self.edit.setToolTip(text)
        self.menu_button.setToolTip(text)

def _param_value(widget, spec):
    kind = spec["type"]
    if kind == "bool":
        return widget.isChecked()
    if kind == "int":
        return widget.value()
    if kind == "float":
        return widget.value()
    if kind == "enum":
        return widget.currentData()
    return widget.text()


def _set_param_value(widget, spec, value):
    kind = spec["type"]
    if kind == "bool":
        widget.setChecked(bool(value))
    elif kind in {"int", "float"}:
        widget.setValue(value)
    elif kind == "enum":
        index = widget.findData(value)
        widget.setCurrentIndex(max(0, index))
    else:
        widget.setText(str(value or ""))


class ParameterDialog(QDialog):
    def __init__(self, model, parameters: dict, ui_lang: str, parent=None):
        super().__init__(parent)
        self.model = model
        self.parameters = dict(parameters or {})
        self.ui_lang = ui_lang
        self.specs = specs_for(model)
        self.controls = {}
        self.setWindowTitle(f"{model.label} — {TEXT[ui_lang]['parameters']}")
        self.resize(560, 560)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        form = QFormLayout(body)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(8)
        for spec in self.specs:
            current = self.parameters.get(spec["key"], spec.get("default"))
            kind = spec["type"]
            if kind == "bool":
                control = QCheckBox()
            elif kind == "int":
                control = QSpinBox()
                control.setRange(int(spec.get("min", -2147483647)), int(spec.get("max", 2147483647)))
                if spec.get("suffix"):
                    control.setSuffix(spec["suffix"])
            elif kind == "float":
                control = QDoubleSpinBox()
                control.setDecimals(4)
                control.setRange(float(spec.get("min", -1e9)), float(spec.get("max", 1e9)))
                control.setSingleStep(float(spec.get("step", 0.1)))
                if spec.get("suffix"):
                    control.setSuffix(spec["suffix"])
            elif kind == "enum":
                control = QComboBox()
                for value in spec["values"]:
                    control.addItem(str(value), value)
            else:
                control = QLineEdit()
            _set_param_value(control, spec, current)
            control.setToolTip(spec.get("tip_" + ui_lang, spec.get("tip_en", "")))
            label = QLabel(spec.get(ui_lang, spec.get("en", spec["key"])))
            label.setToolTip(control.toolTip())
            form.addRow(label, control)
            self.controls[spec["key"]] = (control, spec)

        if not self.specs:
            empty = QLabel(TEXT[ui_lang]["no_params"])
            empty.setObjectName("muted")
            empty.setWordWrap(True)
            form.addRow(empty)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        buttons = QHBoxLayout()
        reset = QPushButton(TEXT[ui_lang]["reset"])
        reset.clicked.connect(self.reset_defaults)
        buttons.addWidget(reset)
        buttons.addStretch()
        cancel = QPushButton(TEXT[ui_lang]["cancel"])
        apply = QPushButton(TEXT[ui_lang]["apply"])
        cancel.clicked.connect(self.reject)
        apply.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(apply)
        outer.addLayout(buttons)

    def reset_defaults(self):
        for _key, (widget, spec) in self.controls.items():
            _set_param_value(widget, spec, spec.get("default"))

    def result_parameters(self) -> dict:
        result = {}
        known = {spec["key"] for spec in self.specs}
        # Preserve unknown options from hand-edited preferences/custom model definitions.
        disabled = self.model.disabled_parameters
        for key, value in self.parameters.items():
            if key not in known and key not in disabled:
                result[key] = value
        for key, (widget, spec) in self.controls.items():
            value = _param_value(widget, spec)
            if value != spec.get("default"):
                result[key] = value
        return result


class StagePanel(QFrame):
    changed = Signal()

    def __init__(self, task: str, optional: bool, ui_lang: str):
        super().__init__()
        self.task = task
        self.optional = optional
        self.ui_lang = ui_lang
        self.catalog = {}
        self.parameters = {}
        self.setObjectName("stage")

        root = QVBoxLayout(self)
        root.setContentsMargins(5, 3, 5, 4)
        root.setSpacing(2)

        head = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("section")
        head.addWidget(self.title)
        head.addStretch()
        if optional:
            self.toggle = QToolButton()
            self.toggle.setObjectName("stageToggle")
            self.toggle.setCheckable(True)
            self.toggle.setChecked(False)
            self.toggle.toggled.connect(self._sync_enabled)
            head.addWidget(self.toggle)
        else:
            self.toggle = None
        root.addLayout(head)

        self.body = QWidget()
        self.body.setObjectName("stageBody")
        grid = QGridLayout(self.body)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)
        grid.setColumnMinimumWidth(0, 54)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(2, 20)

        self.model_label = QLabel()
        self.model_label.setObjectName("muted")
        self.model_label.setFixedWidth(54)
        self.model = QComboBox()
        self.model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model.setFixedHeight(20)
        self.model.currentIndexChanged.connect(self._model_changed)
        self.params = QToolButton()
        self.params.setObjectName("iconButton")
        self.params.setIcon(icon("tune"))
        self.params.setIconSize(QSize(13, 13))
        self.params.setFixedSize(20, 20)
        self.params.clicked.connect(self.edit_parameters)
        grid.addWidget(self.model_label, 0, 0)
        grid.addWidget(self.model, 0, 1)
        grid.addWidget(self.params, 0, 2)

        self.backend_label = QLabel()
        self.backend_label.setObjectName("muted")
        self.backend_label.setFixedWidth(54)
        self.device = QComboBox()
        self.device.setFixedHeight(20)
        grid.addWidget(self.backend_label, 1, 0)
        grid.addWidget(self.device, 1, 1, 1, 2)

        self.language_label = QLabel()
        self.language_label.setObjectName("muted")
        self.language_label.setFixedWidth(54)
        self.language = EditablePresetField()
        self.language.setToolTip("Free input. Use the arrow for common runtime language values.")
        self.language.setPresets(("auto", "ja", "ja-JP", "en", "en-US", "Japanese", "English"))
        grid.addWidget(self.language_label, 2, 0)
        grid.addWidget(self.language, 2, 1, 1, 2)

        self.reference_label = QLabel()
        self.reference_label.setObjectName("muted")
        self.reference_label.setFixedWidth(54)
        self.reference = QComboBox()
        self.reference.setFixedHeight(20)
        self.reference.addItem("Original", "original")
        self.reference.addItem("Processed", "processed")
        self.reference.currentIndexChanged.connect(self.changed)
        grid.addWidget(self.reference_label, 3, 0)
        grid.addWidget(self.reference, 3, 1, 1, 2)

        if task == "diar":
            self.language_label.hide()
            self.language.hide()
        if task != "sep":
            self.reference_label.hide()
            self.reference.hide()

        root.addWidget(self.body)
        self.device.currentIndexChanged.connect(self.changed)
        self.language.currentTextChanged.connect(self.changed)
        self.apply_language(ui_lang)
        self._sync_enabled()

    def enabled_stage(self):
        return True if not self.optional else self.toggle.isChecked()

    def _sync_enabled(self):
        enabled = self.enabled_stage()
        self.body.setEnabled(enabled)
        # Optional stages collapse to a single header row while disabled.
        # This keeps the inspector dense like a DCC properties panel instead
        # of reserving disabled-form space.
        if self.optional:
            self.body.setVisible(enabled)
        self.setProperty("disabledStage", not enabled)
        self.style().unpolish(self)
        self.style().polish(self)
        if self.toggle:
            self.toggle.setText("ON" if enabled else "OFF")
        self.changed.emit()

    def set_catalog(self, catalog):
        selected = self.model.currentData()
        self.catalog = catalog
        self.model.blockSignals(True)
        self.model.clear()
        for model in catalog.values():
            if model.task == self.task:
                label, description = model_text(model, self.ui_lang)
                self.model.addItem(label, model.id)
                index = self.model.count() - 1
                self.model.setItemData(index, description, Qt.ItemDataRole.ToolTipRole)
        if selected:
            index = self.model.findData(selected)
            if index >= 0:
                self.model.setCurrentIndex(index)
        if self.model.count() and self.model.currentIndex() < 0:
            self.model.setCurrentIndex(0)
        self.model.blockSignals(False)
        self._model_changed()

    def _model_changed(self):
        model = self.catalog.get(self.model.currentData())
        if model:
            allowed = {spec["key"] for spec in specs_for(model)}
            self.parameters = {key: value for key, value in self.parameters.items() if key in allowed}
            default_language = str(model.defaults.get("language", "ja"))
            presets = [default_language, "auto", "ja", "ja-JP", "en", "en-US"]
            if model.family == "qwen3_forced_aligner":
                presets += ["Japanese", "English"]
            self.language.setPresets(presets)
            self.language.setCurrentText(default_language)
        self.changed.emit()

    def apply_language(self, lang):
        self.ui_lang = lang
        tr = TEXT[lang]
        self.title.setText(tr[self.task])
        self.model_label.setText(tr["model"])
        self.backend_label.setText(tr["backend"])
        self.language_label.setText(tr["language"])
        self.reference_label.setText(tr["rpp_audio"])
        selected_ref = self.reference.currentData()
        self.reference.blockSignals(True)
        self.reference.clear()
        self.reference.addItem(tr["original"], "original")
        self.reference.addItem(tr["processed"], "processed")
        idx = self.reference.findData(selected_ref)
        self.reference.setCurrentIndex(max(0, idx))
        self.reference.blockSignals(False)
        for i in range(self.model.count()):
            model = self.catalog.get(self.model.itemData(i))
            if model:
                label, description = model_text(model, lang)
                self.model.setItemText(i, label)
                self.model.setItemData(i, description, Qt.ItemDataRole.ToolTipRole)
        self.params.setToolTip(tr["parameters"])
        selected = self.device.currentData()
        self.device.blockSignals(True)
        self.device.clear()
        self.device.addItem(tr["default"], "default")
        self.device.addItem("CPU", "cpu")
        self.device.addItem("Vulkan", "vulkan")
        if selected:
            idx = self.device.findData(selected)
            if idx >= 0:
                self.device.setCurrentIndex(idx)
        self.device.blockSignals(False)
        if self.toggle:
            self.toggle.setToolTip(self.title.text())
        self.changed.emit()

    def edit_parameters(self):
        model = self.catalog.get(self.model.currentData())
        if not model:
            return
        dialog = ParameterDialog(model, self.parameters, self.ui_lang, self)
        if dialog.exec():
            self.parameters = dialog.result_parameters()
            self.changed.emit()

    def stage(self, runtime_paths, threads, runtime_defaults):
        if not self.enabled_stage():
            return None
        model_id = self.model.currentData() or ""
        model = self.catalog.get(model_id)
        if not model:
            raise ValueError(f"No model selected for {self.task}")
        requested = self.device.currentData() or "default"
        device = runtime_defaults.get(model.runtime, "vulkan") if requested == "default" else requested
        if device not in {"cpu", "vulkan"}:
            device = "vulkan"
        executable = runtime_paths.get(f"{model.runtime}:{device}", "")
        return Stage(model_id, device, executable, self.language.currentText().strip(),
                     threads, copy.deepcopy(self.parameters))


class Worker(QThread):
    progress = Signal(str)
    item = Signal(int, str, str)
    error = Signal(str)

    def __init__(self, jobs, settings, catalog, prepare_only=False,
                 keep_sources=False, queue_strategy="stage", batch_audio_ram_mb=512):
        super().__init__()
        self.jobs = jobs
        self.settings = copy.deepcopy(settings)
        self.catalog = catalog.copy()
        self.prepare_only = prepare_only
        self.keep_sources = keep_sources
        self.queue_strategy = queue_strategy
        self.batch_audio_ram_mb = batch_audio_ram_mb
        self.cancel = threading.Event()

    def _stages(self):
        return [self.settings.preprocess, self.settings.asr, self.settings.align, self.settings.diar]

    def _prepare_models(self):
        seen = set()
        for stage in self._stages():
            if not stage or stage.model_id in seen:
                continue
            seen.add(stage.model_id)
            resolve_model(self.catalog[stage.model_id], self.cancel, self.progress.emit,
                          download=True, keep_source=self.keep_sources)

    def run(self):
        try:
            self._prepare_models()
            if self.prepare_only:
                self.progress.emit("Models ready")
                return
            if self.queue_strategy == "stage" and len(self.jobs) > 1:
                run_queue(self.jobs, self.settings, self.catalog, self.cancel,
                          self.progress.emit, self.item.emit, self.batch_audio_ram_mb)
                return
            for index, path in self.jobs:
                if self.cancel.is_set():
                    break
                self.item.emit(index, "running", "")
                try:
                    output = run_job(Path(path), self.settings, self.catalog,
                                     self.cancel, self.progress.emit)
                    self.item.emit(index, "done", str(output))
                except Cancelled:
                    self.item.emit(index, "stopped", "")
                    break
                except Exception as exc:
                    self.item.emit(index, "failed", str(exc))
        except Cancelled:
            pass
        except Exception as exc:
            self.error.emit(str(exc))


class PreferencesDialog(QDialog):
    prepareRequested = Signal()

    def __init__(self, owner):
        super().__init__(owner)
        self.owner = owner
        self.lang = owner.ui_lang
        tr = TEXT[self.lang]
        self.setWindowTitle(tr["settings"])
        self.resize(760, 560)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        self.nav = QListWidget()
        self.nav.setFixedWidth(138)
        for key in ("general", "runtime", "advanced"):
            self.nav.addItem(tr[key])
        content.addWidget(self.nav)

        self.pages = QStackedWidget()
        content.addWidget(self.pages, 1)
        root.addLayout(content, 1)

        self.model_dir = QLineEdit(owner.model_storage_dir)
        self.model_dir.setPlaceholderText(default_storage_hint("weights"))
        self.temp_dir = QLineEdit(owner.temp_storage_dir)
        self.temp_dir.setPlaceholderText(default_storage_hint("cache"))
        self.queue_strategy = QComboBox()
        self.queue_strategy.addItem(tr["stage_major"], "stage")
        self.queue_strategy.addItem(tr["file_major"], "file")
        self.queue_strategy.setCurrentIndex(max(0, self.queue_strategy.findData(owner.queue_strategy)))
        self.batch_ram = QSpinBox()
        self.batch_ram.setRange(128, 8192)
        self.batch_ram.setSingleStep(128)
        self.batch_ram.setSuffix(" MB")
        self.batch_ram.setValue(owner.batch_audio_ram_mb)
        self.threads = QSpinBox()
        self.threads.setRange(1, 128)
        self.threads.setValue(owner.threads)

        general = QWidget()
        form = QFormLayout(general)
        form.setContentsMargins(16, 16, 16, 16)
        form.setSpacing(10)
        form.addRow(tr["model_dir"], self._path_row(self.model_dir))
        form.addRow(tr["temp_dir"], self._path_row(self.temp_dir))
        form.addRow(tr["queue_order"], self.queue_strategy)
        form.addRow(tr["batch_limit"], self.batch_ram)
        form.addRow(tr["threads"], self.threads)
        self.pages.addWidget(general)

        self.whisper_backend = QComboBox()
        self.audio_backend = QComboBox()
        for combo, value in ((self.whisper_backend, owner.runtime_defaults["whisper_cpp"]),
                             (self.audio_backend, owner.runtime_defaults["audio_cpp"])):
            combo.addItem("CPU", "cpu")
            combo.addItem("Vulkan", "vulkan")
            combo.setCurrentIndex(max(0, combo.findData(value)))
        runtime = QWidget()
        runtime_form = QFormLayout(runtime)
        runtime_form.setContentsMargins(16, 16, 16, 16)
        runtime_form.setSpacing(10)
        runtime_form.addRow(tr["whisper_backend"], self.whisper_backend)
        runtime_form.addRow(tr["audio_backend"], self.audio_backend)
        self.whisper_status = QLabel(self._runtime_status("whisper_cpp"))
        self.audio_status = QLabel(self._runtime_status("audio_cpp"))
        self.whisper_status.setObjectName("muted")
        self.audio_status.setObjectName("muted")
        runtime_form.addRow("", self.whisper_status)
        runtime_form.addRow("", self.audio_status)
        self.pages.addWidget(runtime)

        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(16, 16, 16, 16)
        advanced_layout.setSpacing(9)
        prepare = QPushButton(icon("download"), tr["prepare_models"])
        prepare.clicked.connect(self._prepare)
        advanced_layout.addWidget(prepare)
        row = QHBoxLayout()
        open_models = QPushButton(icon("folder_open"), tr["open_models"])
        open_models.clicked.connect(lambda: owner.open_path(owner.storage_path("weights")))
        open_toml = QPushButton(icon("folder_open"), tr["open_toml"])
        open_toml.clicked.connect(lambda: owner.open_path(model_directory()))
        reload_toml = QPushButton(icon("refresh"), tr["reload_toml"])
        reload_toml.clicked.connect(owner.reload_catalog)
        row.addWidget(open_models)
        row.addWidget(open_toml)
        row.addWidget(reload_toml)
        advanced_layout.addLayout(row)

        self.keep_source = QCheckBox(tr["keep_source"])
        self.keep_source.setChecked(owner.keep_model_sources)
        advanced_layout.addWidget(self.keep_source)

        title = QLabel(tr["exe_override"])
        title.setObjectName("section")
        advanced_layout.addWidget(title)
        self.runtime_fields = {}
        for key in ("ffmpeg", "whisper_cpp:cpu", "whisper_cpp:vulkan",
                    "audio_cpp:cpu", "audio_cpp:vulkan"):
            line = QLineEdit(owner.runtime_paths.get(key, ""))
            line.setPlaceholderText("Bundled / PATH")
            advanced_layout.addLayout(self._file_row(key, line))
            self.runtime_fields[key] = line
        advanced_layout.addStretch()
        self.pages.addWidget(advanced)

        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.nav.setCurrentRow(0)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save |
                                   QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Save).setText(tr["save"])
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText(tr["cancel"])
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _path_row(self, edit):
        box = QWidget()
        row = QHBoxLayout(box)
        row.setContentsMargins(0, 0, 0, 0)
        button = QToolButton()
        button.setObjectName("iconButton")
        button.setIcon(icon("folder_open"))
        button.clicked.connect(lambda: self._choose_dir(edit))
        row.addWidget(edit, 1)
        row.addWidget(button)
        return box

    def _file_row(self, key, edit):
        row = QHBoxLayout()
        label = QLabel(key)
        label.setMinimumWidth(150)
        button = QToolButton()
        button.setObjectName("iconButton")
        button.setIcon(icon("folder_open"))
        button.clicked.connect(lambda: self._choose_file(edit))
        row.addWidget(label)
        row.addWidget(edit, 1)
        row.addWidget(button)
        return row

    def _choose_dir(self, edit):
        value = QFileDialog.getExistingDirectory(self, "", edit.text().strip())
        if value:
            edit.setText(value)

    def _choose_file(self, edit):
        value, _ = QFileDialog.getOpenFileName(self, "")
        if value:
            edit.setText(value)

    def _runtime_status(self, runtime):
        values = []
        for device in ("cpu", "vulkan"):
            try:
                path = runtime_executable(runtime, device,
                    self.owner.runtime_paths.get(f"{runtime}:{device}", ""))
                values.append(f"{device.upper()}: {path.name}")
            except Exception:
                values.append(f"{device.upper()}: --")
        return "   ".join(values)

    def _prepare(self):
        self.accept()
        self.prepareRequested.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ASR2RPP")
        self.resize(1180, 760)
        self.setMinimumSize(920, 600)

        self.preferences = QSettings("ASR2RPP", "ASR2RPP")
        # The legacy preview defaulted native runtimes to CPU. The DCC UI changes the
        # product default to Vulkan, so migrate once; later explicit CPU choices persist.
        if not self.preferences.value("runtime_defaults_v2", False, type=bool):
            self.preferences.setValue("runtime_default/whisper_cpp", "vulkan")
            self.preferences.setValue("runtime_default/audio_cpp", "vulkan")
            self.preferences.setValue("runtime_defaults_v2", True)
            self.preferences.sync()
        self.ui_lang = str(self.preferences.value("ui/language", "ja"))
        if self.ui_lang not in TEXT:
            self.ui_lang = "ja"
        self.runtime_defaults = {
            "whisper_cpp": str(self.preferences.value("runtime_default/whisper_cpp", "vulkan")),
            "audio_cpp": str(self.preferences.value("runtime_default/audio_cpp", "vulkan")),
        }
        for key in self.runtime_defaults:
            if self.runtime_defaults[key] not in {"cpu", "vulkan"}:
                self.runtime_defaults[key] = "vulkan"
        try:
            self.runtime_paths = json.loads(self.preferences.value("runtime_paths", "{}"))
        except Exception:
            self.runtime_paths = {}
        self.model_storage_dir = str(self.preferences.value("storage/model_dir", "")).strip()
        self.temp_storage_dir = str(self.preferences.value("storage/temp_dir", "")).strip()
        self.keep_model_sources = self.preferences.value(
            "models/keep_source_checkpoints", False, type=bool)
        self.queue_strategy = str(self.preferences.value("queue/strategy", "stage"))
        if self.queue_strategy not in {"stage", "file"}:
            self.queue_strategy = "stage"
        self.batch_audio_ram_mb = int(self.preferences.value("queue/batch_audio_ram_mb", 512))
        self.threads = int(self.preferences.value("threads", 4))
        self.entries = []
        self.worker = None
        self.completed = 0
        self.catalog = {}
        self.apply_storage_roots()

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.table = QueueTable()
        self.table.filesDropped.connect(self.add_paths)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.queue_menu)
        self.table.cellDoubleClicked.connect(self.open_output)
        splitter.addWidget(self.table)

        inspector = QFrame()
        inspector.setObjectName("inspector")
        inspector.setMinimumWidth(300)
        inspector.setMaximumWidth(360)
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(8, 8, 8, 8)
        inspector_layout.setSpacing(7)

        self.preprocess = StagePanel("sep", True, self.ui_lang)
        self.asr = StagePanel("asr", False, self.ui_lang)
        self.align = StagePanel("align", True, self.ui_lang)
        self.diar = StagePanel("diar", True, self.ui_lang)
        for panel in (self.preprocess, self.asr, self.align, self.diar):
            inspector_layout.addWidget(panel)
            panel.changed.connect(self.update_state)

        output = QFrame()
        output.setObjectName("stage")
        out_layout = QVBoxLayout(output)
        out_layout.setContentsMargins(9, 8, 9, 8)
        out_layout.setSpacing(6)
        self.output_title = QLabel()
        self.output_title.setObjectName("section")
        out_layout.addWidget(self.output_title)
        self.same_directory = QCheckBox()
        self.same_directory.setChecked(self.preferences.value("output/same", True, type=bool))
        self.same_directory.toggled.connect(self.output_changed)
        out_layout.addWidget(self.same_directory)
        row = QHBoxLayout()
        self.output_dir = QLineEdit(str(self.preferences.value("output/directory", "")))
        self.output_dir.textChanged.connect(self.update_state)
        self.output_browse = QToolButton()
        self.output_browse.setObjectName("iconButton")
        self.output_browse.setIcon(icon("folder_open"))
        self.output_browse.clicked.connect(self.choose_output)
        row.addWidget(self.output_dir, 1)
        row.addWidget(self.output_browse)
        out_layout.addLayout(row)
        inspector_layout.addWidget(output)
        inspector_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inspector)
        splitter.addWidget(scroll)
        splitter.setSizes([840, 340])
        root.addWidget(splitter, 1)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        status = QFrame()
        status.setObjectName("statusBar")
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(7, 4, 7, 4)
        status_layout.setSpacing(4)
        self.status_label = QLabel()
        self.status_label.setObjectName("status")
        status_layout.addWidget(self.status_label, 1)

        self.language_button = QToolButton()
        self.language_button.setObjectName("languageButton")
        self.language_button.setIcon(icon("language"))
        self.language_button.setIconSize(QSize(15, 15))
        self.language_button.setFixedSize(26, 26)
        self.language_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        self.language_button.clicked.connect(self.toggle_language)
        status_layout.addWidget(self.language_button)

        self.settings_button = QToolButton()
        self.settings_button.setObjectName("footerIcon")
        self.settings_button.setIcon(icon("settings"))
        self.settings_button.setIconSize(QSize(16, 16))
        self.settings_button.setFixedSize(26, 26)
        self.settings_button.clicked.connect(self.open_settings)
        status_layout.addWidget(self.settings_button)

        self.run_button = QPushButton()
        self.run_button.setObjectName("runButton")
        self.run_button.setIcon(icon("play"))
        self.run_button.setIconSize(QSize(14, 14))
        self.run_button.setFixedSize(78, 26)
        self.run_button.clicked.connect(self.toggle_run)
        status_layout.addWidget(self.run_button)
        root.addWidget(status)

        self.language_shortcut = QShortcut(QKeySequence("Ctrl+Shift+L"), self)
        self.language_shortcut.activated.connect(self.toggle_language)

        self.reload_catalog()
        self.restore_stage_preferences()
        self.apply_language()
        self.output_changed()

    def tr(self, key):
        return TEXT[self.ui_lang][key]

    def apply_storage_roots(self):
        if self.model_storage_dir:
            os.environ["ASR2RPP_WEIGHTS_DIR"] = self.model_storage_dir
        else:
            os.environ.pop("ASR2RPP_WEIGHTS_DIR", None)
        if self.temp_storage_dir:
            os.environ["ASR2RPP_CACHE_DIR"] = self.temp_storage_dir
        else:
            os.environ.pop("ASR2RPP_CACHE_DIR", None)

    def storage_path(self, kind):
        if kind == "weights":
            return Path(self.model_storage_dir).expanduser() if self.model_storage_dir else weights_root()
        return Path(self.temp_storage_dir).expanduser() if self.temp_storage_dir else cache_root()

    def open_path(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def reload_catalog(self):
        self.catalog, errors = load_catalog()
        for panel in (self.preprocess, self.asr, self.align, self.diar):
            panel.set_catalog(self.catalog)
        if errors:
            self.status_label.setText(errors[0])

    def restore_stage_preferences(self):
        for key, panel in (("preprocess", self.preprocess), ("asr", self.asr),
                           ("align", self.align), ("diar", self.diar)):
            model_id = str(self.preferences.value(f"{key}/model", ""))
            index = panel.model.findData(model_id)
            if index >= 0:
                panel.model.setCurrentIndex(index)
            device = str(self.preferences.value(f"{key}/device", "default"))
            index = panel.device.findData(device)
            if index >= 0:
                panel.device.setCurrentIndex(index)
            if panel.optional:
                panel.toggle.setChecked(self.preferences.value(f"{key}/enabled", False, type=bool))
            language = str(self.preferences.value(f"{key}/language", panel.language.currentText()))
            panel.language.setCurrentText(language)
            try:
                panel.parameters = json.loads(self.preferences.value(f"{key}/parameters", "{}"))
            except Exception:
                panel.parameters = {}
            model = self.catalog.get(panel.model.currentData())
            if model and model.disabled_parameters:
                panel.parameters = {
                    name: value for name, value in panel.parameters.items()
                    if name not in model.disabled_parameters
                }
        ref = str(self.preferences.value("preprocess/reference", "original"))
        idx = self.preprocess.reference.findData(ref)
        if idx >= 0:
            self.preprocess.reference.setCurrentIndex(idx)

    def save_preferences(self):
        self.preferences.setValue("ui/language", self.ui_lang)
        self.preferences.setValue("runtime_paths", json.dumps(self.runtime_paths))
        self.preferences.setValue("storage/model_dir", self.model_storage_dir)
        self.preferences.setValue("storage/temp_dir", self.temp_storage_dir)
        self.preferences.setValue("models/keep_source_checkpoints", self.keep_model_sources)
        self.preferences.setValue("queue/strategy", self.queue_strategy)
        self.preferences.setValue("queue/batch_audio_ram_mb", self.batch_audio_ram_mb)
        self.preferences.setValue("threads", self.threads)
        for runtime, value in self.runtime_defaults.items():
            self.preferences.setValue(f"runtime_default/{runtime}", value)
        self.preferences.setValue("output/same", self.same_directory.isChecked())
        self.preferences.setValue("output/directory", self.output_dir.text().strip())
        for key, panel in (("preprocess", self.preprocess), ("asr", self.asr),
                           ("align", self.align), ("diar", self.diar)):
            self.preferences.setValue(f"{key}/model", panel.model.currentData() or "")
            self.preferences.setValue(f"{key}/device", panel.device.currentData() or "default")
            self.preferences.setValue(f"{key}/enabled", panel.enabled_stage())
            self.preferences.setValue(f"{key}/language", panel.language.currentText())
            self.preferences.setValue(f"{key}/parameters", json.dumps(panel.parameters))
        self.preferences.setValue("preprocess/reference",
                                  self.preprocess.reference.currentData() or "original")
        self.preferences.sync()

    def apply_language(self):
        tr = TEXT[self.ui_lang]
        self.table.setHorizontalHeaderLabels([
            tr["queue_name"], tr["queue_status"], tr["queue_size"], tr["queue_output"]])
        self.table.placeholder = tr["drop"]
        self.table.viewport().update()
        for panel in (self.preprocess, self.asr, self.align, self.diar):
            panel.apply_language(self.ui_lang)
        self.output_title.setText(tr["output"])
        self.same_directory.setText(tr["same_dir"])
        self.output_dir.setPlaceholderText(tr["select_output"])
        self.settings_button.setToolTip(tr["settings"])
        target = "English" if self.ui_lang == "ja" else "日本語"
        self.language_button.setToolTip(tr["ui_language"] + f" → {target} (Ctrl+Shift+L)")
        self.language_button.setText("")
        self.render_queue()

    def toggle_language(self):
        self.ui_lang = "en" if self.ui_lang == "ja" else "ja"
        self.apply_language()
        self.save_preferences()

    def queue_menu(self, position):
        menu = QMenu(self)
        add_files = QAction(icon("add"), self.tr("add_files"), self)
        add_files.triggered.connect(self.choose_files)
        add_folder = QAction(icon("new_folder"), self.tr("add_folder"), self)
        add_folder.triggered.connect(self.choose_folder)
        menu.addAction(add_files)
        menu.addAction(add_folder)
        menu.addSeparator()
        remove = QAction(icon("delete"), self.tr("remove"), self)
        remove.setEnabled(bool(self.table.selectionModel().selectedRows()) and self.worker is None)
        remove.triggered.connect(self.remove_selected)
        menu.addAction(remove)
        retry = QAction(icon("refresh"), self.tr("retry"), self)
        retry.setEnabled(any(e["status"] in {"failed", "stopped"} for e in self.entries)
                         and self.worker is None)
        retry.triggered.connect(self.retry_failed)
        menu.addAction(retry)
        open_action = QAction(icon("open"), self.tr("open_output"), self)
        row = self.table.rowAt(position.y())
        open_action.setEnabled(row >= 0 and self.entries[row]["status"] == "done")
        open_action.triggered.connect(lambda: self.open_output(row, 0))
        menu.addAction(open_action)
        menu.addSeparator()
        clear = QAction(self.tr("clear"), self)
        clear.setEnabled(bool(self.entries) and self.worker is None)
        clear.triggered.connect(self.clear_queue)
        menu.addAction(clear)
        menu.exec(self.table.viewport().mapToGlobal(position))

    def choose_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "", "",
            "Audio / Video (" + " ".join("*" + x for x in sorted(MEDIA_EXTENSIONS)) + ")")
        self.add_paths(files)

    def choose_folder(self):
        directory = QFileDialog.getExistingDirectory(self, "")
        if directory:
            self.add_paths([str(x) for x in sorted(Path(directory).iterdir()) if x.is_file()])

    def add_paths(self, values):
        if self.worker:
            return
        known = {e["path"] for e in self.entries}
        expanded = []
        for value in values:
            path = Path(value).expanduser()
            if path.is_dir():
                expanded.extend(x for x in sorted(path.iterdir()) if x.is_file())
            else:
                expanded.append(path)
        for path in expanded:
            path = path.resolve()
            if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS and str(path) not in known:
                self.entries.append({"path": str(path), "status": "waiting", "output": ""})
                known.add(str(path))
        self.render_queue()

    def render_queue(self):
        self.table.setRowCount(len(self.entries))
        for row, entry in enumerate(self.entries):
            path = Path(entry["path"])
            size = f"{path.stat().st_size / 1024**2:.1f} MB" if path.exists() else "--"
            status = self.tr(entry["status"]) if entry["status"] in TEXT[self.ui_lang] else entry["status"]
            values = [path.name, status, size, entry["output"]]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(entry["path"] if col == 0 else str(value))
                self.table.setItem(row, col, item)
            self.table.setRowHeight(row, 32)
        self.table.viewport().update()
        self.update_state()

    def remove_selected(self):
        rows = {index.row() for index in self.table.selectionModel().selectedRows()}
        self.entries = [entry for i, entry in enumerate(self.entries) if i not in rows]
        self.render_queue()

    def retry_failed(self):
        for entry in self.entries:
            if entry["status"] in {"failed", "stopped"}:
                entry.update(status="waiting", output="")
        self.render_queue()

    def clear_queue(self):
        self.entries.clear()
        self.render_queue()

    def open_output(self, row, _column):
        if 0 <= row < len(self.entries):
            entry = self.entries[row]
            if entry["status"] == "done" and entry["output"]:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(entry["output"]).parent)))

    def choose_output(self):
        directory = QFileDialog.getExistingDirectory(self, "", self.output_dir.text())
        if directory:
            self.output_dir.setText(directory)

    def output_changed(self):
        enabled = not self.same_directory.isChecked()
        self.output_dir.setEnabled(enabled)
        self.output_browse.setEnabled(enabled)
        self.update_state()

    def current_settings(self):
        preprocess = self.preprocess.stage(self.runtime_paths, self.threads, self.runtime_defaults)
        return Settings(
            asr=self.asr.stage(self.runtime_paths, self.threads, self.runtime_defaults),
            diar=self.diar.stage(self.runtime_paths, self.threads, self.runtime_defaults),
            align=self.align.stage(self.runtime_paths, self.threads, self.runtime_defaults),
            same_directory=self.same_directory.isChecked(),
            output_directory=self.output_dir.text().strip(),
            ffmpeg=self.runtime_paths.get("ffmpeg", ""),
            clip_start=0.0,
            clip_duration=0.0,
            preprocess=preprocess,
            reference_audio=(self.preprocess.reference.currentData() or "original")
                if preprocess else "original",
        )

    def pipeline_text(self):
        stages = []
        if self.preprocess.enabled_stage():
            stages.append("SEP")
        stages.append("ASR")
        if self.align.enabled_stage():
            stages.append("ALIGN")
        if self.diar.enabled_stage():
            stages.append("DIAR")
        return " > ".join(stages)

    def update_state(self):
        if self.worker is None:
            self.status_label.setText(
                f"{self.tr('ready')}   {self.pipeline_text()}   "
                f"W:{self.runtime_defaults['whisper_cpp'].upper()}  "
                f"A:{self.runtime_defaults['audio_cpp'].upper()}")
        valid_output = self.same_directory.isChecked() or bool(self.output_dir.text().strip())
        can_go = valid_output and any(e["status"] == "waiting" for e in self.entries)
        self.run_button.setEnabled(True if self.worker else can_go)
        self.run_button.setProperty("running", bool(self.worker))
        self.run_button.setText("STOP" if self.worker else "GO!")
        self.run_button.setIcon(icon("stop" if self.worker else "play"))
        self.run_button.style().unpolish(self.run_button)
        self.run_button.style().polish(self.run_button)

    def set_busy(self, busy):
        self.table.setEnabled(not busy)
        for panel in (self.preprocess, self.asr, self.align, self.diar):
            panel.setEnabled(not busy)
        self.same_directory.setEnabled(not busy)
        self.output_dir.setEnabled(not busy and not self.same_directory.isChecked())
        self.output_browse.setEnabled(not busy and not self.same_directory.isChecked())
        self.settings_button.setEnabled(not busy)
        self.update_state()

    def toggle_run(self):
        if self.worker:
            self.worker.cancel.set()
            self.status_label.setText("Stopping..." if self.ui_lang == "en" else "停止処理中...")
            return
        self.start_work(False)

    def start_work(self, prepare_only=False):
        if self.worker:
            return
        try:
            settings = self.current_settings()
            validation = copy.deepcopy(settings)
            if prepare_only:
                validation.same_directory = True
            validation.validate(self.catalog)
            jobs = [(i, e["path"]) for i, e in enumerate(self.entries)
                    if e["status"] == "waiting"]
            if not prepare_only and not jobs:
                return
            self.save_preferences()
            self.completed = 0
            self.progress.setRange(0, 0 if prepare_only else max(1, len(jobs)))
            self.progress.setValue(0)
            self.worker = Worker(
                jobs, settings, self.catalog, prepare_only, self.keep_model_sources,
                self.queue_strategy, self.batch_audio_ram_mb)
            self.worker.progress.connect(self.show_progress)
            self.worker.item.connect(self.item_changed)
            self.worker.error.connect(self.show_error)
            self.worker.finished.connect(self.work_finished)
            self.set_busy(True)
            if prepare_only:
                self.status_label.setText(self.tr("preparing"))
            self.worker.start()
        except Exception as exc:
            QMessageBox.warning(self, "ASR2RPP", str(exc))

    def show_progress(self, value):
        self.status_label.setText(value[:180])

    def show_error(self, value):
        self.status_label.setText(value[:180])

    def item_changed(self, index, status, detail):
        mapping = {
            "実行中": "running", "完了": "done", "失敗": "failed", "中断": "stopped",
            "ASR": "running", "SEP": "running", "Forced Align": "running",
            "Diarization": "running", "RPP": "running", "ASR準備": "running",
            "前処理準備": "running",
        }
        canonical = mapping.get(status, status if status in {"running", "done", "failed", "stopped"} else "running")
        self.entries[index].update(status=canonical, output=detail if canonical in {"done", "failed"} else "")
        if canonical in {"done", "failed", "stopped"}:
            self.completed += 1
            self.progress.setValue(self.completed)
        self.render_queue()

    def work_finished(self):
        worker = self.worker
        self.worker = None
        self.set_busy(False)
        if self.progress.maximum() == 0:
            self.progress.setRange(0, 1)
            self.progress.setValue(1)
        self.update_state()
        if worker:
            worker.deleteLater()

    def open_settings(self):
        dialog = PreferencesDialog(self)
        prepare = {"value": False}
        dialog.prepareRequested.connect(lambda: prepare.__setitem__("value", True))
        if dialog.exec():
            self.model_storage_dir = dialog.model_dir.text().strip()
            self.temp_storage_dir = dialog.temp_dir.text().strip()
            for value in (self.model_storage_dir, self.temp_storage_dir):
                if value:
                    Path(value).expanduser().mkdir(parents=True, exist_ok=True)
            self.apply_storage_roots()
            self.queue_strategy = dialog.queue_strategy.currentData() or "stage"
            self.batch_audio_ram_mb = dialog.batch_ram.value()
            self.threads = dialog.threads.value()
            self.runtime_defaults = {
                "whisper_cpp": dialog.whisper_backend.currentData(),
                "audio_cpp": dialog.audio_backend.currentData(),
            }
            self.keep_model_sources = dialog.keep_source.isChecked()
            self.runtime_paths = {
                key: edit.text().strip() for key, edit in dialog.runtime_fields.items()
                if edit.text().strip()
            }
            self.save_preferences()
            self.update_state()
            if prepare["value"]:
                self.start_work(True)

    def closeEvent(self, event):
        if self.worker:
            self.worker.cancel.set()
            event.ignore()
            return
        self.save_preferences()
        event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()
