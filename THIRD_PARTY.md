# Third-party components / 第三者コンポーネント

ASR2RPP source is MIT licensed; see LICENSE. Third-party libraries and models retain their own terms.
The native engines are separate processes and can be replaced in the Runtime settings.

## Bundled application libraries

- Python: Python Software Foundation License. https://www.python.org/downloads/source/
- PySide6 / Qt 6.8.3 / Shiboken: Qt for Python and Qt library licenses, including LGPL v3. Only Qt Core, Gui and Widgets are required by the app. License files from installed distributions are retained in `licenses/`. Qt libraries remain dynamically loaded and replaceable; modification/debugging of LGPL components is not prohibited by this application. https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v6.8.3 and https://download.qt.io/archive/qt/6.8/6.8.3/
- PyInstaller 6.22.3: GPL with bootloader exception; application code retains its own license. https://github.com/pyinstaller/pyinstaller/tree/v6.22.3

## UI assets

- Google Material Icons SVGs are used in the desktop UI. The included SVG files are color-adjusted derivatives of Google's official Material Design Icons and remain under Apache License 2.0. A full copy is in `assets/icons/LICENSE.txt`. Upstream: https://github.com/google/material-design-icons

## Native engines

- whisper.cpp, ggml and bundled dependencies: upstream licenses in `engines/whisper_cpp-cpu/licenses`. Pinned source https://github.com/ggml-org/whisper.cpp/tree/a664346ea5c6dddff3e61a2b7b32dd4514613f50
- audio.cpp and bundled dependencies: upstream LICENSE and collected notices in `engines/audio_cpp-cpu/licenses`. Pinned source https://github.com/0xShug0/audio.cpp/tree/9bdd1d908bbd128e9eb405f5a8e38d0defb84c72
- Native build instructions and flags are recorded in `tools/build_native.py`; per-file hashes and revision appear in each `build-manifest.json`.
- FFmpeg is NOT bundled in the ASR2RPP Windows ZIP. The app first uses a user-selected executable or `PATH`; if none is found on Windows, it downloads BtbN/FFmpeg-Builds' current `win64-lgpl-shared` archive from the provider's `latest` release, verifies it against that release's `checksums.sha256`, and installs only the runtime `bin` files under the user's ASR2RPP data directory. FFmpeg remains separately licensed under its applicable LGPL terms and the downloaded build retains the provider/upstream terms. Upstream: https://ffmpeg.org/ ; build provider: https://github.com/BtbN/FFmpeg-Builds .

## Model weights

Model weights are NOT included in the Windows ZIP. Model TOML files identify their download repositories.
Original model and conversion publisher terms must be checked before downloading or redistributing weights.
The app records repository revision, downloaded file SHA-256 and sizes locally.
Anime Whisper's community conversion is experimental; availability does not imply verified transcription parity.

No REAPER binary, user media, credential, or system font is distributed with ASR2RPP.

- Silero VAD v5.1.2 is downloaded on demand, not bundled. The GGML is from ggml-org/whisper-vad at e5614ed76a5dd4b03fad5068c89efcd2617a9d1e, SHA-256 29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf (885098 bytes). Upstream Silero VAD and the model repository declare MIT licensing. The native helper is built from the same pinned whisper.cpp revision and its license is retained with that runtime. https://huggingface.co/ggml-org/whisper-vad and https://github.com/snakers4/silero-vad
