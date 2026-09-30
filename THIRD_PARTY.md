# Third-party components / 第三者コンポーネント

ASR2RPP source uses the [MIT license](LICENSE). Libraries, runtimes and weights keep their own licenses. 本体・ライブラリ・モデルの利用条件は別です。
This inventory describes the standard **0.2.2 CPU/Vulkan (Windows) and CPU/Metal (Apple Silicon)** distributions, not earlier CUDA previews.

## Bundled application libraries

| Component | Source / terms |
|---|---|
| Python runtime | [Python source and PSF license](https://www.python.org/downloads/source/) |
| PySide6, Shiboken and Qt | [Qt for Python 6.8.3](https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v6.8.3), [Qt 6.8.3 sources](https://download.qt.io/archive/qt/6.8/6.8.3/); applicable Qt licenses include LGPL v3 |
| NumPy | [NumPy](https://github.com/numpy/numpy); BSD terms and dependency notices supplied by its distribution |
| safetensors | [safetensors](https://github.com/huggingface/safetensors); Apache-2.0 |
| PyInstaller | [PyInstaller](https://github.com/pyinstaller/pyinstaller); GPL with its bootloader exception |
| sherpa-onnx / sherpa-onnx-core, CPU worker | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx); Apache-2.0 and bundled dependency notices |
| ONNX Runtime used by that worker | [ONNX Runtime](https://github.com/microsoft/onnxruntime); MIT and bundled dependency notices |

Installed distribution license files are copied into `licenses/` and `engines/python_worker/licenses/`. Worker package versions and file hashes are in its `build-manifest.json`.
Qt libraries remain dynamically loaded and replaceable. Modification/debugging of LGPL components is not prohibited by ASR2RPP. GUI dependencies include Qt Core, Gui, Widgets and SVG support, plus the platform/plugin dependencies collected by PyInstaller.

## Native engines

[whisper.cpp](https://github.com/ggml-org/whisper.cpp) and [audio.cpp](https://github.com/0xShug0/audio.cpp) run as separate processes. Their ggml and other dependency licenses/notices are collected in each native pack's `licenses/` directory. Exact source revisions, build flags and file hashes are recorded in the pack's `build-manifest.json`; `native/versions.json` defines the source pins.
Windows GPU acceleration uses Vulkan and the installed GPU driver. Mac uses the system Metal framework. No GPU driver is redistributed.

**Not bundled:** CUDA, cuBLAS, cuDNN, CTranslate2, faster-whisper, PyAV, their VAD assets, PyTorch, Transformers, REAPER, user media, credentials or system fonts. An optional external worker has its own dependencies and terms; it does not change this standard inventory.

## FFmpeg, obtained separately

FFmpeg binaries are not included. The app uses a selected executable or a discovered installation.
When Windows needs a download, the bootstrap obtains the `win64-lgpl-shared` build from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds), checks the published release checksum, and installs the runtime files in user data. That separate build retains its provider/upstream terms; it is not covered by ASR2RPP's MIT license. See [FFmpeg](https://ffmpeg.org/).
On Mac, the user provides FFmpeg separately.

## Model weights

Only TOML definitions are bundled. Selected ASR, alignment, diarization, separation and VAD weights are obtained on demand. Every shipped catalog file has a pinned revision and SHA-256; acquisition records downloaded sizes and actual hashes.
A conversion's license does not replace its original model's license. Review both publishers' model cards before use or redistribution. The [model table](docs/models.md) links the repositories; in particular, not every model uses MIT/Apache terms.

Silero VAD v5.1.2 is separately downloaded from [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad) at `e5614ed76a5dd4b03fad5068c89efcd2617a9d1e`, SHA-256 `29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf` (885,098 bytes). [Silero VAD](https://github.com/snakers4/silero-vad) and that model repository declare MIT licensing.
Anime Whisper's community GGML conversion is experimental; a functional smoke test is not proof of parity with its original implementation.

## Icons

UI and executable artwork use adapted [Google Material Icons](https://github.com/google/material-design-icons), Apache-2.0. ASR2RPP adds backgrounds/colors. The full license is in `assets/icons/LICENSE.txt`; provenance is in `assets/branding/NOTICE.md` and packaged `licenses/material-icons/`. No icon font is bundled.
