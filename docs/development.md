# Development and upstream maintenance

[User guide](../README.md) · [日本語](../README.ja.md) · [Model/runtime customization](provider-models.md)

## Source development

Use Python 3.11 or newer; CI uses 3.12. From the repository root:

```sh
python -m pip install -e '.[dev,workers]'
python -m pytest tests
python launcher.py
```

The source tree needs FFmpeg and built native runtimes. PyTorch, Transformers, CUDA and CTranslate2 are not standard dependencies. `workers` installs CPU sherpa-onnx; `faster` is a separate optional developer extra, not part of the portable builds.

## Native builds and packages

`native/versions.json` is the source of truth for whisper.cpp/audio.cpp revisions and compiled audio.cpp model families.
The build rejects modified upstream files. It does **not** patch the upstream CLI. Ordinary Whisper uses stock `whisper-cli`; bounded-region requests use our `asr2rpp-whisper-regions`, compiled against the public `whisper.h` API. The legacy patch script is retained for historical reproduction only.

Windows requires a native x64 MSVC environment, CMake and the Vulkan SDK for building. In a Developer Command Prompt:

```text
python tools/build_native.py whisper_cpp:cpu whisper_cpp:vulkan audio_cpp:cpu audio_cpp:vulkan
python tools/package_windows.py
```

Source and execution character sets use `/utf-8`, and C++ exception unwinding stays enabled with `/EHsc`. This matters for upstream CJK prompt literals on non-UTF-8 Windows installations.

On Apple Silicon, build with the Xcode/Metal toolchain, CMake and Ninja:

```sh
python tools/build_native.py whisper_cpp:cpu whisper_cpp:metal audio_cpp:cpu audio_cpp:metal
python tools/package_macos.py
```

Mac targets arm64/macOS 14+, embeds Metal resources and avoids external Homebrew library references in packaged engines. The app is ad-hoc signed, not notarized. Do not mutate its resources after signing.
These toolchains are **build dependencies**, not end-user requirements.

Build manifests record revisions, recipe hashes, compiler flags and per-file hashes. Verified-cache reuse is opt-in. The packager copies only the platform's CPU/GPU packs and CPU worker, rejects accidental CUDA/CT2 inclusion, retains notices and excludes fonts/model weights/FFmpeg binaries. Excluded Python package metadata is also removed from the copied worker, never from the build environment.

## Tests and release gate

Common unit tests include model-schema validation, duplicate/unsafe paths, request-level errors, native timing, speaker preservation and GUI cancellation/retry. Native PCM checks run with assertions enabled even in Release:

```sh
cmake -S native -B build/pcm-tests -DCMAKE_BUILD_TYPE=Release
cmake --build build/pcm-tests --config Release
ctest --test-dir build/pcm-tests -C Release --output-on-failure
```

The application workflow builds Windows CPU/Vulkan and macOS CPU/Metal packages. Windows tests the frozen CLI/GUI and optional stages on CPU. Mac verifies arm64 slices, library references, signatures and a relocated copy of the exact ZIP before exercising the app.
Metal hardware capabilities are probed: an unavailable GPU or required feature is recorded as a skipped GPU test, not success. `ASR2RPP_REQUIRE_METAL_GPU=1` makes such skips fail on a hardware runner.

Use the explicit real-model matrix on an authorized GPU machine, with synthetic/public inputs:

```text
python tools/validate_catalog.py --package PATH_TO_APP --fixture-ja JA.wav --fixture-en EN.wav --report NEW_REPORT_DIRECTORY --weights-dir WEIGHTS_DIRECTORY --ffmpeg PATH_TO_FFMPEG --device vulkan --install
```

Mac uses `--device metal`. `--source-root REPO` replaces `--package` for source validation; reports distinguish those modes.
`--models ID ID` limits the selection. Without it, all shipped definitions are exercised and **many gigabytes of weights may be downloaded** when `--install` is given. Without `--install`, weights must already exist.
Each test requires a nonempty bounded transcript/RPP, original-file integrity, correct timing provenance and expected optional-stage output. Speaker-capable ASR must preserve a speaker label. GPU failures are not replaced with CPU successes. This is a functional matrix, not WER/CER, multi-speaker quality, throughput or long-recording certification.

Never call a build successful before its job, artifact and actual report are inspected. Keep application, runtime and model revisions distinct. Do not replace published assets with different untested bytes under the same identity.

## Update an upstream runtime or model

Change the pinned revision/family list or model TOML, rebuild, run unit/PCM checks, verify capabilities and perform the relevant model smoke tests. The public API removes the CLI-text patch dependency but does not promise permanent ABI/API compatibility.
Unknown architectures still require an implementation. Adding a compiled family and a valid TOML is enough only when the runtime's input/output contract is already supported.
Keep raw native output and fine-grained timestamps until speaker assignment; construct editable coarse items last. Separate a process failure from an individual malformed result. Persist logs on failure and leave source media unchanged.

[Current model matrix](models.md) · [Historical performance records](performance.md) · [Earlier implementation records](implementation-stages.md)
Historical CUDA measurements and previous release reports do not describe the standard Windows Vulkan package.
