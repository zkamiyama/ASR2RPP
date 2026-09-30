[English](README.md) | [日本語](README.ja.md)

# ASR2RPP

**Turn speech in audio or video into an editable REAPER project, locally.**

ASR2RPP writes recognized text and timing into `.rpp` items that reference your media.
Optional stages remove background audio, refine timing and organize items by speaker.
The original file is not modified.

## Start here

1. Get the complete ZIP for your OS from [Releases](https://github.com/zkamiyama/ASR2RPP/releases). Development builds are the successful CI artifacts linked from [PR #7](https://github.com/zkamiyama/ASR2RPP/pull/7). **This guide describes 0.2.2 and later; the older CUDA-containing 0.2.0 archive is different.**
2. Extract everything into a **new folder**. Start `ASR2RPP.exe` on Windows or `ASR2RPP.app` on Mac. Do not move only the EXE.
3. Drag recordings into the queue, choose an ASR model and press **GO**. For a first check, use **Whisper Base**, select the recording's language, leave optional stages OFF, and use **Settings → Timing → Automatic**.
4. Only the selected weights are downloaded. Open the resulting `.rpp` in REAPER. The default output location is beside the input; change it in the main window. Double-click a completed row to open its output folder.

The gear opens settings. The globe button or `Ctrl+Shift+L` switches English/Japanese.

## Requirements and dependencies

| Package | Target | GPU path |
|---|---|---|
| Windows | x64 CPU with AVX2 | Vulkan only, with a compatible GPU driver |
| macOS | Apple Silicon, macOS 14 or later; not Intel | Metal only |

Both packages support CPU execution. Select **CPU** for a stage when its GPU path is unavailable or fails on your system. The bundled ReazonSpeech K2 worker is CPU-only.

**Included:** application Python runtime, Qt/PySide6, NumPy, safetensors, whisper.cpp, audio.cpp, CPU sherpa-onnx worker and model TOML definitions.
**Separate:** REAPER to open projects, FFmpeg to decode media, and the selected model weights.
**Not required for users:** a Python installation, PyTorch, CUDA Toolkit, Vulkan SDK or Xcode. Standard packages exclude CUDA, CTranslate2 and faster-whisper.

Windows acquires a checksum-verified FFmpeg build when none is found. On Mac, install FFmpeg separately; choose its executable under **Settings → Advanced → ffmpeg** when necessary. Standard Homebrew paths are also detected.
Model/FFmpeg acquisition uses the network. Once those files are available, inference runs locally; recordings are not sent to a cloud transcription service.

Windows builds are unsigned; Mac builds are ad-hoc signed and not notarized. Check the distribution source and SHA-256. Libraries and weights retain their own terms, separate from the app's MIT license; see [third-party notices](THIRD_PARTY.md).

## Models and timing

The catalog contains **20 model definitions, not 20 bundled sets of weights**. Nothing downloads the entire catalog automatically.
It covers Whisper Tiny through Large/Turbo, Anime Whisper, Qwen3-ASR, SenseVoice, Fun-ASR-Nano, Nemotron, Canary, Moonshine, Voxtral, VibeVoice and optional analysis stages.
Use the [model table](docs/models.md) for language, download size and exact validation scope. The included Canary and Moonshine models do not support Japanese.

Choose timing in **Settings → Timing**, not by editing model definitions:

| Choice | Behavior |
|---|---|
| Automatic | Uses model intervals when available; otherwise uses VAD speech regions. Enabling alignment selects alignment. Legacy TOML workflow defaults are also considered for compatibility. |
| Native model intervals | Keeps the model's intervals; unavailable for text-only ASR. |
| VAD speech regions | Recognizes bounded speech regions independently. **These are utterance intervals, not word boundaries.** |
| Forced alignment | Aligns recognized text to audio. Select the aligner in the main window. Enable “VAD before alignment” for long/coarse segments. |

The same settings page controls maximum VAD duration, threshold and minimum silence.
Recognition, timing and speaker labels can be wrong. Alignment does not correct a wrong transcript.
Background removal and speaker diarization are optional. Native ASR speaker labels, such as VibeVoice's, can also be retained. Assigning labels does not separate overlapping voices into distinct recordings.

## Output, cancellation and errors

The first track, **ORIGINAL**, contains the complete reference audio as one muted item.
Transcript or speaker tracks follow. Audio is referenced, not embedded: keep the media with the project.
With background removal and **RPP Audio → Processed**, a continuous `*_vocals.wav` is saved and becomes the reference; **Original** keeps references to the original media.
Existing output names receive a numeric suffix instead of being overwritten.
The adjacent `.asr2rpp` directory contains settings, timing provenance, logs and recognized text. **Review it for private information before sharing.**

Press **STOP** and wait for cleanup. **GO** retries waiting, stopped or failed items without regenerating completed items. Remove and re-add a completed row to process it again.
The bottom log is selectable/copyable. An unsupported `family` means the runtime does not contain that implementation; a missing model means its weights need to be installed.

## Updates and customization

Change model and temporary storage under **Settings → General**. Default data locations are `%LOCALAPPDATA%\ASR2RPP` on Windows and `~/Library/Application Support/ASR2RPP` on Mac.
Extract updates into a new folder. UI preferences use the OS settings store, and verified weights for unchanged definitions are normally reused. Changing the source/revision may require another download.

Add a uniquely named `.toml` under **Settings → Advanced → Open custom TOML**. Do not reuse a bundled model ID or edit the signed `.app` contents. Definitions reload before GO or through the reload button.
**TOML-only additions work for architectures already implemented by an installed runtime. A GGUF or ONNX extension alone does not make an unknown architecture executable.**

[Custom models and runtimes](docs/provider-models.md) · [Model catalog and test scope](docs/models.md) · [Development/upstream updates](docs/development.md) · [License](LICENSE)
