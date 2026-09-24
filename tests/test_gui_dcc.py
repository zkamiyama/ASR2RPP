import os
from pathlib import Path

import pytest

from asr2rpp.adapters import whisper_parameter_args, audio_session_args
from asr2rpp.catalog import Model
from asr2rpp.parameter_specs import specs_for


def test_native_parameter_mapping():
    args = whisper_parameter_args({
        "processors": 2,
        "gpu_device": 1,
        "beam_size": 7,
        "temperature": 0.2,
        "split_on_word": True,
        "flash_attn": False,
        "suppress_regex": r"\[.*?\]",
        "initial_prompt": "REAPER",
        "vad": True,
        "vad_threshold": 0.55,
    })
    assert args == [
        "-p", "2", "-dev", "1", "-bs", "7", "-tp", "0.2", "-sow",
        "-nfa", "--suppress-regex", r"\[.*?\]", "--prompt", "REAPER",
        "--vad", "-vt", "0.55",
    ]
    model = Model("d", "audio_cpp", "diar", {"path": "x"},
                  family="nemotron_3_diar")
    session = audio_session_args(model, {"graph_arena_mb": 512, "weight_type": "native"})
    assert session == [
        "--session-option", "nemotron_3_diar.graph_arena_mb=512",
        "--session-option", "nemotron_3_diar.weight_type=native",
    ]


def test_parameter_specs_surface_cpp_controls():
    whisper = Model("w", "whisper_cpp", "asr", {"path": "x"})
    keys = {x["key"] for x in specs_for(whisper)}
    assert {"beam_size", "best_of", "temperature", "temperature_inc",
            "no_speech_thold", "entropy_thold", "logprob_thold", "word_thold",
            "audio_ctx", "max_context", "max_len", "split_on_word",
            "no_fallback", "suppress_nst", "suppress_regex", "processors",
            "gpu_device", "flash_attn", "grammar", "grammar_rule",
            "grammar_penalty", "vad", "vad_model", "vad_threshold",
            "initial_prompt"} <= keys
    anime = Model("anime-whisper", "whisper_cpp", "asr", {"path": "x"})
    anime_keys = {x["key"] for x in specs_for(anime)}
    assert "initial_prompt" not in anime_keys
    assert "carry_initial_prompt" not in anime_keys
    diar = Model("d", "audio_cpp", "diar", {"path": "x"}, family="nemotron_3_diar")
    qwen = Model("q", "audio_cpp", "align", {"path": "x"}, family="qwen3_forced_aligner")
    qwen_keys = {x["key"] for x in specs_for(qwen)}
    assert "clamp_timestamps_to_audio" in qwen_keys
    assert "session.weight_type" not in qwen_keys
    diar_keys = {x["key"] for x in specs_for(diar)}
    assert {"speaker_threshold", "speaker_min_frames", "speaker_pad_frames",
            "session.latency_profile", "session.graph_arena_mb",
            "session.weight_context_mb", "session.weight_type"} <= diar_keys


def test_dcc_gui_structure_and_screens(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("ASR2RPP_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ASR2RPP_WEIGHTS_DIR", raising=False)
    monkeypatch.delenv("ASR2RPP_CACHE_DIR", raising=False)

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QPushButton
    from asr2rpp.gui_dcc import MainWindow, PreferencesDialog, STYLE

    QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope,
                      str(tmp_path / "settings"))
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    window = MainWindow()
    window.show()
    app.processEvents()

    assert window.windowTitle() == "ASR2RPP"
    assert window.runtime_defaults == {"whisper_cpp": "vulkan", "audio_cpp": "vulkan"}
    assert window.run_button.text() == "GO!"
    assert not hasattr(window, "clip_start")
    assert "#1b1f24" in STYLE
    assert "font-family" not in STYLE
    # Disabled optional stages collapse to their header only. Explicitly
    # switch them off because earlier GUI migration tests may persist settings.
    window.preprocess.toggle.setChecked(False)
    window.align.toggle.setChecked(False)
    window.diar.toggle.setChecked(False)
    app.processEvents()
    assert not window.preprocess.body.isVisible()
    assert not window.align.body.isVisible()
    assert not window.diar.body.isVisible()
    window.align.toggle.setChecked(True)
    app.processEvents()
    assert window.align.body.isVisible()
    window.align.toggle.setChecked(False)
    app.processEvents()
    visible_buttons = [button.text() for button in window.findChildren(QPushButton)]
    assert "キューを実行" not in visible_buttons
    assert "選択モデルを準備" not in visible_buttons

    # Queue has no permanent add toolbar; files are accepted directly.
    sample = tmp_path / "meeting.wav"
    sample.write_bytes(b"x")
    window.add_paths([str(sample)])
    assert len(window.entries) == 1
    assert window.run_button.isEnabled()

    # Runtime selectors are deliberately limited to Default / CPU / Vulkan.
    for panel in (window.preprocess, window.asr, window.align, window.diar):
        devices = [panel.device.itemData(i) for i in range(panel.device.count())]
        assert devices == ["default", "cpu", "vulkan"]

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    window.grab().save(str(reports / "gui-dcc-ja.png"))

    window.toggle_language()
    app.processEvents()
    assert window.ui_lang == "en"
    assert window.table.horizontalHeaderItem(0).text() == "Input"
    assert window.table.item(0, 1).text() == "Queued"
    assert "実験的変換版" not in window.asr.model.currentText()
    assert window.preprocess.reference.itemText(0) == "Original"
    window.grab().save(str(reports / "gui-dcc-en.png"))

    dialog = PreferencesDialog(window)
    dialog.show()
    app.processEvents()
    assert dialog.nav.count() == 3
    assert dialog.model_dir.placeholderText()
    assert dialog.temp_dir.placeholderText()
    assert [dialog.whisper_backend.itemData(i) for i in range(dialog.whisper_backend.count())] == ["cpu", "vulkan"]
    assert [dialog.audio_backend.itemData(i) for i in range(dialog.audio_backend.count())] == ["cpu", "vulkan"]
    dialog.grab().save(str(reports / "settings-dcc.png"))
    dialog.close()
    window.close()
    app.processEvents()
