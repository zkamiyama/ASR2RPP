# Third-party components / 第三者コンポーネント

ASR2RPP source and original brand marks use the [MIT license](LICENSE). Libraries, runtimes and model weights retain their own licenses. 本体・依存ライブラリ・モデルの条件は別です。
This inventory describes the standard **0.3 CPU/Vulkan (Windows) and CPU/Metal (Apple Silicon)** distributions, not earlier CUDA previews.

## Included in the application

| Component | Source / terms |
|---|---|
| Python runtime | [Python source and PSF license](https://www.python.org/downloads/source/) |
| PySide6, Shiboken, Qt | [Qt for Python 6.8.3](https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v6.8.3), [Qt 6.8.3 sources](https://download.qt.io/archive/qt/6.8/6.8.3/); applicable licenses include LGPL v3 |
| NumPy | [NumPy](https://github.com/numpy/numpy); BSD and dependency notices |
| safetensors | [safetensors](https://github.com/huggingface/safetensors); Apache-2.0 |
| OpenTimelineIO 0.18.1 | [OpenTimelineIO](https://github.com/AcademySoftwareFoundation/OpenTimelineIO); Apache-2.0 |
| PyInstaller bootloader | [PyInstaller](https://github.com/pyinstaller/pyinstaller); GPL with bootloader exception |
| sherpa-onnx / sherpa-onnx-core | [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx); Apache-2.0 and dependency notices |
| ONNX Runtime in the CPU worker | [ONNX Runtime](https://github.com/microsoft/onnxruntime); MIT and dependency notices |

Distribution license files are copied into `licenses/` and `engines/python_worker/licenses/`. Worker versions and hashes are in `build-manifest.json`. Qt is dynamically loaded and replaceable; modification/debugging of LGPL components is not prohibited by ASR2RPP. OpenTimelineIO is used through its core API without discovering external adapter plugins.

## Native engines

[whisper.cpp](https://github.com/ggml-org/whisper.cpp) and [audio.cpp](https://github.com/0xShug0/audio.cpp) run as separate processes. Their ggml and dependency licenses are collected in each runtime pack's `licenses/`. `native/versions.json` pins sources; each pack's `build-manifest.json` records revisions, build flags and hashes. Windows uses Vulkan and the installed GPU driver; Mac uses system Metal. No GPU driver is redistributed.

**Not bundled:** CUDA, cuBLAS, cuDNN, CTranslate2, faster-whisper, PyAV, their VAD assets, PyTorch, Transformers, REAPER, user media, credentials or system fonts. Optional external workers have their own dependencies and terms.

## FFmpeg: separate first-use download

FFmpeg binaries are **not in the application ZIP**. A user-selected or discovered executable takes precedence.

Windows acquisition uses the `win64-lgpl-shared` build from [BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds), verifies its published checksum and installs it under user data.

Apple Silicon acquisition downloads a pinned [imageio-ffmpeg 0.6.0](https://pypi.org/project/imageio-ffmpeg/0.6.0/) arm64 wheel, verifies the SHA-256 in `asr2rpp/ffmpeg_macos.py`, and extracts only standalone FFmpeg 7.1 and notices. No Python package is installed. **That FFmpeg build is GPL-enabled**; the wrapper's BSD license does not replace the executable's license. Its provenance is retained and `ffmpeg -L` prints its license. Users may supply a compatible alternative.

See [imageio-ffmpeg sources](https://github.com/imageio/imageio-ffmpeg) and [FFmpeg legal information](https://ffmpeg.org/legal.html). These separate distributions are not relicensed under ASR2RPP's MIT license.

## Model weights

Only TOMLs are included. Selected recognition, alignment, diarization, separation and VAD weights are obtained on demand. Shipped catalog definitions pin source revisions and SHA-256; installations record actual hashes. Conversion does not replace the original model's license. Read both publishers' model cards before use or redistribution; [the model table](docs/models.md) links them.

Silero VAD v5.1.2 is downloaded separately from [ggml-org/whisper-vad](https://huggingface.co/ggml-org/whisper-vad) at `e5614ed76a5dd4b03fad5068c89efcd2617a9d1e`, SHA-256 `29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf`. [Silero VAD](https://github.com/snakers4/silero-vad) and that repository declare MIT licensing. Anime Whisper's community conversion is experimental; a functional test does not establish accuracy parity.

## Icons and sample

`assets/branding/*.svg` are original geometric ASR2RPP marks under MIT. Interface icons adapted from [Google Material Icons](https://github.com/google/material-design-icons) retain Apache-2.0; the full license is `assets/icons/LICENSE.txt`. No font files are bundled.

The small `examples/sample.wav` is the pinned public speech fixture from whisper.cpp's `samples/jfk.wav`. Its source URL and SHA-256 accompany it in `examples/README.txt`. It is test media, not a user's recording.
