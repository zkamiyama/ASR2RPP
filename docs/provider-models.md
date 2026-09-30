# Model and runtime extension (0.2)

## Timing belongs to the user

Settings → Timing selects Automatic, Native model intervals, VAD speech regions,
or Forced alignment. The alignment model is selected in the main inspector.
VAD intervals are deliberately labelled speech-region timing, not word timing.
The optional “VAD before alignment” switch bounds requests for long/coarse output.
Native timing cannot be selected for a text-only model. An inconsistent stage
combination fails before inference rather than silently changing the workflow.

A model TOML describes capabilities, artifacts and safe scalar parameters. It
cannot execute a shell command, import Python, install dependencies, or register
an executable. Old schema-1 definitions remain accepted. Explicit GUI timing
choices take precedence over a legacy TOML's workflow default.

## TOML-only models

Place a uniquely named TOML in the directory opened by Settings → Advanced →
Custom TOMLs. Definitions are reloaded before GO. Copy one of the bundled v2
examples: qwen3-asr-06b.toml or reazonspeech-k2.toml.
Use `provider`, `family`, `source`, `artifacts`, `capabilities` and `execution`.
Declare `capabilities.timestamps = "none"` for text-only ASR. Choose timing in
the GUI, not in the model file. `[parameters.<name>]` supplies a type, default,
range/enum and Japanese/English UI labels. Unknown native scalar options can be
expressed here without changing the GUI, provided the chosen runtime supports them.

Multi-file downloads use a pinned Hugging Face revision, `source.files`,
`source.entry = "."` and an `[artifacts]` role-to-file table. All download files,
including small JSON/tokenizer files, are supported. Installation is locked and
atomic; retained artifacts are checked before use. Local models may point at a
file or directory. Artifact paths must stay inside that directory.

## Adding an implementation

A different checkpoint of an implemented architecture requires only a TOML.
An architecture not implemented by an installed runtime needs a new runtime or
worker. A format name such as ONNX or GGUF alone does not define preprocessing,
network operators or decoding. The app probes actual compiled family/mode support.

The bundled isolated worker implements sherpa-onnx (transducer, SenseVoice,
Paraformer, NeMo CTC) on CPU. Faster-whisper/CTranslate2 are not bundled. A trusted third-party executable can use
`provider = "external_json"` and worker protocol 1, documented by worker_client.py
and provider_worker.py. The executable is selected in settings or registered:

```
asr2rpp-cli runtimes register --provider external_json --device cpu --exe PATH --trust
asr2rpp-cli runtimes probe --provider external_json --device cpu
asr2rpp-cli runtimes rollback --provider external_json --device cpu
```

Registration is explicit code trust and pins the selected executable hash.
Changing it requires re-registration. A model definition cannot do this for you.

## Upstream updates

`native/versions.json` pins upstream revisions and compiled audio.cpp families.
The builder does not alter upstream source files. Ordinary transcription uses
stock whisper-cli; bounded VAD transcription uses asr2rpp-whisper-regions, linked
only through whisper.h's public API. Unsupported helper options use the stock
CLI path, not silently ignored flags. Legacy patched CLIs are still readable;
tools/patch_whisper.py is retained only for legacy reproduction, not called by
normal builds. Public API changes still require compiling and testing the helper.

Run Python tests, native PCM tests, optional-stage smoke and text-equivalence
checks for every candidate update. Runtime binaries can be replaced/registered
independently of the GUI; they must pass capability checks. Standard Windows GPU
execution uses Vulkan only; macOS Apple Silicon uses Metal only. CPU is available
on both. These choices are enforced by the GUI, CLI and registry, including
explicit custom executable paths. CUDA is not selected automatically on Windows.

## Distribution (0.2.1)

Windows ZIP: GUI, CLI, CPU/Vulkan native engines and a CPU-only sherpa worker.
Apple Silicon ZIP: a self-contained ASR2RPP.app with CPU/Metal native engines,
CLI in Contents/MacOS and assets in Contents/Resources. It targets macOS 14 or later;
Intel/universal2 is not part of this build. The .app is ad-hoc signed, not notarized.
Keep bundled definitions unmodified to preserve its signature. Add personal TOMLs
under Settings → Advanced → Custom TOMLs (Library/Application Support/ASR2RPP on Mac).

No Python, PyTorch, CUDA Toolkit, CTranslate2 or faster-whisper is required by
these standard packages. ASR, alignment and VAD weights are downloaded on demand.
FFmpeg is not redistributed: Windows can acquire it through the verified bootstrap;
on macOS install FFmpeg separately and select it in Settings if needed. The app
also recognizes /opt/homebrew/bin/ffmpeg and /usr/local/bin/ffmpeg for Finder launches.

The faster-whisper example was moved to docs/optional-models. It is not an available
model until the user explicitly installs/registers a compatible external worker.
On Windows and macOS that optional path is CPU-only; a Vulkan or Metal flag does
not make CTranslate2 support those backends. It is never downloaded or bundled
by the standard build.

CI builds both platforms, audits packaged files for accidental CUDA/CT2 inclusion,
and tests the extracted archives. macOS signatures, arm64 slices and Mach-O library
references are verified after relocation. The Metal hardware probe distinguishes
compiled support from real inference: when the VM exposes no MTLDevice the report
records GPU testing as skipped. ASR2RPP_REQUIRE_METAL_GPU=1 makes that a failure on
a hardware runner. The absence of a GPU never counts as a successful Metal test.
