"""Shared DCC palette and localized labels, separate from widget behavior."""
from .catalog import assets_root

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
QFrame#logPanel { background: #1b1f24; border-top: 1px solid #303741; }
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
QPushButton#runButton {
    min-width: 76px; max-width: 76px; min-height: 24px; max-height: 24px;
    padding: 0; text-align: center; font-weight: 800; letter-spacing: 1px;
    color: #ffffff; background: #247f5d; border-color: #319b75;
}
QPushButton#runButton:hover { background: #2b906a; }
QPushButton#runButton[running="true"] { background: #a84343; border-color: #ca5c5c; }
QPushButton#runButton[running="true"]:hover { background: #ba4b4b; }
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
QListWidget::item { padding: 7px 9px; color: #9ca6b2; }
QListWidget::item:selected { background: #29323c; color: #ffffff; border-left: 2px solid #4b91ca; }
QMenu {
    background: #20252b; color: #dce2ea; border: 1px solid #3a424d; padding: 4px;
}
QMenu::item { padding: 5px 26px 5px 24px; }
QMenu::item:selected { background: #355c82; }
QMenu::separator { height: 1px; background: #343c46; margin: 4px 7px; }
"""
STYLE += """
QFrame#formatChip { background: #242e36; border: 1px solid #45515d; border-radius: 3px; min-height: 20px; max-height: 20px; }
QFrame#formatChip QLabel { font-size: 10px; font-weight: 600; color: #e0e9ed; }
QToolButton#formatRemove { background: transparent; border: none; padding: 0; min-width: 14px; max-width: 14px; min-height: 16px; max-height: 16px; }
QToolButton#formatRemove:hover { background: #4c3539; border-radius: 2px; }
QToolButton#formatAdd { padding: 0; min-width: 18px; max-width: 18px; }
QToolButton#formatAdd::menu-indicator { image: none; }
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
        "sep": "背景音除去",
        "asr": "音声認識",
        "align": "強制アライメント",
        "diar": "話者ダイアライゼーション",
        "model": "モデル",
        "backend": "実行環境",
        "language": "言語",
        "parameters": "パラメータ",
        "rpp_audio": "参照音声",
        "original": "元メディア",
        "processed": "処理済み",
        "output": "出力",
        "same_dir": "入力ファイルと同じ場所",
        "location": "出力先",
        "custom_dir": "指定フォルダー",
        "choose": "選択",
        "ready": "準備完了",
        "settings": "設定",
        "ui_language": "UI言語を切替",
        "general": "一般",
        "timing": "時刻付与",
        "runtime": "実行環境",
        "advanced": "詳細",
        "model_dir": "モデル保存先",
        "temp_dir": "一時ファイル先",
        "queue_order": "キュー処理順",
        "stage_major": "ステージ優先",
        "file_major": "ファイル優先",
        "batch_limit": "audio.cppバッチ上限",
        "threads": "CPUスレッド",
        "whisper_backend": "whisper.cpp",
        "audio_backend": "audio.cpp",
        "keep_source": "変換成功後も元チェックポイントを保持",
        "prepare_models": "選択モデルを準備",
        "open_models": "モデル保存先を開く",
        "open_toml": "同梱TOMLを開く",
        "open_custom_toml": "独自TOMLを開く",
        "reload_toml": "モデル定義を再読込",
        "exe_override": "実行ファイル上書き",
        "save": "保存",
        "cancel": "キャンセル",
        "reset": "既定値へ戻す",
        "apply": "適用",
        "default": "既定",
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
        "run_log": "実行ログ",
        "ffmpeg_auto": "自動取得 / PATH",
        "bundled_path": "同梱 / PATH",
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
        "rpp_audio": "REFERENCE",
        "original": "Original",
        "processed": "Processed",
        "output": "OUTPUT",
        "same_dir": "Same as input",
        "location": "LOCATION",
        "custom_dir": "Custom directory",
        "choose": "Browse",
        "ready": "Ready",
        "settings": "Settings",
        "ui_language": "Switch UI language",
        "general": "General",
        "timing": "Timing",
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
        "open_toml": "Open bundled TOML",
        "open_custom_toml": "Open custom TOML",
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
        "run_log": "Run Log",
        "ffmpeg_auto": "Auto-download / PATH",
        "bundled_path": "Bundled / PATH",
    },
}

MODEL_TEXT = {
    "anime-whisper": {
        "ja": ("Anime Whisper", "GGML変換版。配布者から認識異常の注意があります。"),
        "en": ("Anime Whisper", "Experimental GGML conversion; the distributor warns of possible recognition anomalies."),
    },
    "mel-big-beta7": {
        "ja": ("Mel-Band RoFormer big beta7 · F16", "背景音除去。元CKPTをローカルで検証し、PyTorchなしでaudio.cpp F16 GGUFへ変換します。"),
        "en": ("Mel-Band RoFormer big beta7 · F16", "Background removal. The exact source checkpoint is verified and converted locally to audio.cpp F16 GGUF without PyTorch."),
    },
    "nemotron-asr": {
        "ja": ("Nemotron 3.5 ASR · Q8", "token emission frame由来の時刻。精密な境界にはForced Alignmentを併用できます。"),
        "en": ("Nemotron 3.5 ASR · Q8", "Timing is derived from token emission frames. Forced Alignment can refine edit boundaries."),
    },
    "nemotron-diarization": {
        "ja": ("Nemotron 3 Diarization · BF16", "最大8話者の話者推定。音源分離は行いません。"),
        "en": ("Nemotron 3 Diarization · BF16", "Speaker diarization for up to eight speakers. It does not perform source separation."),
    },
    "qwen-forced-aligner": {
        "ja": ("Qwen3 Forced Aligner · Q8", "認識テキストと対応音声を再整列します。"),
        "en": ("Qwen3 Forced Aligner · Q8", "Re-aligns recognized transcript text to the corresponding audio."),
    },
    "vibevoice-asr": {
        "ja": ("VibeVoice ASR · Q8", "長時間向けオフライン統合ASR。メモリ使用量が大きいモデルです。"),
        "en": ("VibeVoice ASR · Q8", "Long-form offline integrated ASR with comparatively high memory usage."),
    },
    "whisper-base": {
        "ja": ("Whisper Base", "標準Whisper Base。既定モデル・動作確認向け。"),
        "en": ("Whisper Base", "Standard Whisper Base, used as the default and smoke-test model."),
    },
}


