# ASR2RPP Windows preview

[English guide](https://github.com/zkamiyama/ASR2RPP/blob/main/README.md) · [日本語ガイド](https://github.com/zkamiyama/ASR2RPP/blob/main/README.ja.md)

Download **ASR2RPP-Windows-x64.zip**, extract everything into a new folder, then open **ASR2RPP.exe**.
Python/PyTorch are not required. This preview is unsigned. Models and FFmpeg are not bundled.

## Changes

- GO works again after STOP. Stopped, failed and waiting items can be retried; completed items stay completed.
- Preparation errors, delayed callbacks, repeated stops, close-during-work and a failed worker start are reconciled consistently.
- The GUI and CLI read the EXE-adjacent `models` directory directly. No temporary or user-folder copies of bundled TOMLs.
- Custom definitions use `custom-models` with distinct IDs. Conflicts are reported without changing the user's files.
- Whole-reference duration remains measurable when FFmpeg reports unavailable progress timestamps.
- Minimal high-contrast icons for the app, CLI and every bundled native executable; the application window and taskbar use the same identity.
- Plain English instructions in README, with a Japanese translation linked at the top. Technical background is in `docs`.
- Includes the VAD/TOML controls and PCM/model-reuse optimizations from earlier previews.

Windows/Linux tests, the frozen GO-STOP-GO lifecycle, direct-TOML provenance, native stages,
PCM/WAV equivalence and packaged icon resources are checked before this ZIP is published.

## 日本語

ZIPを新しいフォルダーへ**すべて展開**して起動してください。EXEだけを移動しないでください。
停止後はGOで中断・失敗・待機中の項目を再実行でき、完了済み項目は再処理しません。
同梱TOMLの一時コピーを廃止し、EXEの隣の `models` を直接読みます。
各実行ファイルのアイコンと利用者向けガイドも更新しました。
取得済みモデル・元音源・独自TOMLは上書きしません。

Recognition can be wrong. VAD timestamps describe regions, not exact word boundaries.
GPU speed measurements from WSL/CUDA do not imply identical Windows/Vulkan performance.
