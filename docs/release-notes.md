# ASR2RPP 0.2.1 — Windows Vulkan / macOS Metal

Windows: extract ASR2RPP-Windows-x64.zip into a fresh folder and start ASR2RPP.exe.
CPU and Vulkan only; no CUDA/CT2/faster-whisper libraries in the standard archive.
The CPU sherpa-onnx worker and Qwen3-ASR/ReazonSpeech definitions remain included.

macOS: extract ASR2RPP-macOS-arm64.zip and open ASR2RPP.app. Apple Silicon,
macOS 14+, CPU and Metal. The app is ad-hoc signed, not notarized. FFmpeg must be
installed separately. Add model definitions outside the signed bundle using Settings.

Timing is a user setting: Automatic / Native / VAD / Forced alignment.
Recognition can be wrong. VAD timing is a speech interval, not a word boundary.
Both build paths test the final archive; GPU execution availability and skip reasons
are recorded separately. Compilation is not proof that a GPU inference test ran.

WindowsはCPU/Vulkanに限定し、CUDA・CTranslate2・faster-whisperを標準配布から除外。
Apple Silicon向けCPU/Metalの.app作成と検証をCIへ追加しました。モデルは初回取得、
macOSのFFmpegは別途用意してください。元音源や取得済みモデルは上書きしません。
