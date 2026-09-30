# Development and upstream maintenance

[User guide](../README.md) · [日本語](../README.ja.md) · [Customization](provider-models.md)

## Source setup

Use Python 3.11+ (CI uses 3.12):

```sh
python -m pip install -e '.[dev,workers]'
python -m pytest tests
python launcher.py
```

Source execution needs FFmpeg and built native runtimes. `workers` installs CPU sherpa-onnx. `faster` is an optional developer extra, excluded from standard packages. Python, Qt, NumPy, safetensors and OTIO are included in portable packages; build toolchains are not end-user requirements.

## Native builds

`native/versions.json` pins whisper.cpp/audio.cpp revisions and compiled audio.cpp families. The builder rejects modified upstream files. Ordinary Whisper uses stock `whisper-cli`; bounded regions use our public `whisper.h` API helper. The old CLI text-patching script is removed.

Windows: use an x64 MSVC Developer Command Prompt with CMake and Vulkan SDK:

```text
python tools/build_native.py whisper_cpp:cpu whisper_cpp:vulkan audio_cpp:cpu audio_cpp:vulkan
python tools/package_windows.py
```

The native Windows build requires `/utf-8` for upstream CJK prompt literals and `/EHsc` for C++ exception unwinding. Apple Silicon requires Xcode/Metal, CMake and Ninja:

```sh
python tools/build_native.py whisper_cpp:cpu whisper_cpp:metal audio_cpp:cpu audio_cpp:metal
python tools/package_macos.py
```

Mac targets arm64/macOS 14+, embeds Metal resources, audits library dependencies and ad-hoc signs the complete app. It is not notarized. Do not change resources after signing.

Build manifests record source/recipe hashes, flags and binary hashes. Verified-cache reuse is opt-in. Packaging explicitly selects CPU/Vulkan or CPU/Metal packs plus the CPU worker, rejects CUDA/CT2 dependencies and excludes fonts, model weights and FFmpeg. Licenses are retained.

## Code boundaries

`providers.py` and worker contracts isolate native inference. Model TOML describes assets, capabilities and defaults; it cannot execute arbitrary code or implement an unknown architecture. User timing policy is independent of model definitions.

`export_timeline.py` owns positions, reference origins, speaker lanes and the muted ORIGINAL track. `rpp_export.py` and `otio_export.py` serialize the same timeline. `outputs.py` owns versioned JSON, stage history and exclusive multi-format writes. `gui_formats.py` owns chip interaction; `gui_theme.py` contains style/localization separate from behavior. CLI `convert` reuses exporters without model loading.

Keep fine timestamps through speaker assignment; group editable clips last. Preserve native result data and distinguish timing methods. Separate individual result failures from process failures. The obsolete queue parameter/chunk helpers and one-off CUDA delivery workflow are removed; batching tests target the actual implementation.

## Tests

```sh
python -m pytest tests
cmake -S native -B build/pcm-tests -DCMAKE_BUILD_TYPE=Release
cmake --build build/pcm-tests --config Release
ctest --test-dir build/pcm-tests -C Release --output-on-failure
```

Assertions stay enabled in native Release tests. The application CI builds both OS packages, runs frozen GUI cancellation/restart tests and reads OTIO outputs through the official core. `tools/smoke_outputs.py` checks three-format agreement, mute, gaps, overlaps, offsets, JSON replay, individual format selection and empty-selection rejection. Mac additionally verifies a relocated copy of the exact signed ZIP.

Metal availability is measured. Missing GPU features are explicit skipped cases, not successes; `ASR2RPP_REQUIRE_METAL_GPU=1` makes skips fail on hardware runners. A virtual Metal result does not certify every model on a physical Mac.

## Real-model validation

Use public or synthetic fixtures on an authorized computer:

```text
python tools/validate_catalog.py --package PATH_TO_APP --fixture-ja JA.wav --fixture-en EN.wav --report NEW_REPORT_DIRECTORY --weights-dir WEIGHTS_DIRECTORY --ffmpeg PATH_TO_FFMPEG --device vulkan --install
```

Use `--source-root REPO` instead of `--package` for developer validation. `--models ID ID` limits selection; otherwise all bundled definitions run. `--install` can download many gigabytes. Reports must distinguish source/frozen execution, CPU/physical GPU and model-quality evaluation. Nonempty results and valid output structure are functional evidence, not WER/CER or long-recording certification.

## Updates and releases

Change the appropriate source revision/family list or TOML, rebuild, inspect capabilities, run unit/PCM checks and relevant real-model cases. The public API eliminates CLI text-patching, not all compatibility work.

Build and release are separate decisions. Before publishing, verify both CI jobs, inspect exact ZIP contents and hashes, and exercise the Windows package on the real model matrix. Do not substitute older model or runtime results. Never overwrite a released asset with different bytes under the same version.

[Model catalog](models.md) · [Output specification](outputs.md) · [Historical performance](performance.md) · [Implementation records](implementation-stages.md)
