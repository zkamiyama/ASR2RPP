[English](README.md) | [日本語](README.ja.md)

# ASR2RPP

**Turn audio or video into an editable REAPER project.**

ASR2RPP transcribes speech and creates an `.rpp` project with the text attached to
items that reference your audio. You can also remove background audio, refine
speech timing, and organize speech by speaker. Your original recording is not modified.

## Get started on Windows

1. Download **ASR2RPP-Windows-x64.zip** from the newest [release](https://github.com/zkamiyama/ASR2RPP/releases).
2. Extract **the entire ZIP** to a new folder. Do not run it inside the ZIP or move only the EXE.
3. Open **ASR2RPP.exe**. Drag your audio or video files into the queue.
4. Choose a speech-recognition model and press **GO**. For a first test, select **Whisper Base** and leave the optional steps OFF.
5. Open the resulting **`.rpp` file in REAPER**. By default it is saved next to the input file. Double-click a completed queue row to open its output folder.

The Windows build is for x64 PCs with AVX2 support. REAPER is needed to open the
project; it is not included. Python and PyTorch do not need to be installed.
The app is currently unsigned, so Windows may display a warning. Use a ZIP from
this repository and check the published SHA-256 when verifying a download.

The first run needs internet access to download the selected models. Large models
can take time and several gigabytes of disk space. FFmpeg is used to read media;
if it is not already available, the app downloads a verified build automatically.
Existing downloads are reused. Each model has its own license and use conditions.

## Choose what to run

| Step | What it does |
|---|---|
| Speech recognition | Turns speech into text. This step is always enabled. |
| Background removal | Helps isolate speech from music and other audio. Optional. |
| Forced alignment | Matches the recognized text to the audio for finer timing. Optional, unless required by the model. |
| Speaker diarization | Assigns speech to speaker tracks. It does not separate overlapping voices into different recordings. |

For **Anime Whisper**, the app detects speech regions and recognizes them
independently. With alignment OFF, each item uses the time range of its speech
region—not an exact word boundary. Turn alignment ON for finer timing.
Recognition and speaker labels can be wrong; review the result before editing or publishing.

Use **CPU** if Vulkan is unavailable or fails on your computer. The settings button
at the bottom right lets you change output folders, model storage and execution settings.
Japanese systems start in Japanese; other systems start in English. Use the language
button, or **Ctrl+Shift+L**, to switch. Your selection is remembered.

## Stop and continue

Press **STOP** and wait for cancellation to finish. When **GO** becomes available,
press it again to retry stopped or failed items and process waiting items.
Completed items are left alone and are not generated a second time. To process a
completed file again, remove its queue row and add the file again.

An error is shown in the selectable, copyable log at the bottom of the window.
Check the message, adjust the settings or file, then press GO to retry.

## Understand the output

The top track, **ORIGINAL**, contains the complete reference audio as one item,
including silence. It starts **muted** so it does not double the sound from the
edited tracks. To compare, unmute ORIGINAL and mute the other tracks, or solo ORIGINAL.

Below it are a **Transcript** track or speaker tracks. Items reference the original
media; the app does not save a separate audio file for every phrase. Keep the
referenced media with the project. Moving or deleting it can make REAPER report missing files.

When background removal is enabled, **RPP Audio → Processed** saves a continuous
`*_vocals.wav` next to the project and uses it as the reference instead.
Existing output files are not overwritten; a numeric suffix is added when needed.
The adjacent `.asr2rpp` folder contains diagnostics and recognized text. It may
contain private information: do not post it publicly without reviewing it.

## Updating and model definitions

Extract an update into a **new folder**. Your downloaded models and settings are
normally kept under `%LOCALAPPDATA%\ASR2RPP` and can be reused.

The app reads the **`models` folder next to the EXE directly**. It does not copy
these TOML definitions into a temporary folder. Most users do not need to edit them.
For a custom model, use **Settings → Advanced → Open custom TOML** and give the
TOML a unique filename. Do not reuse a built-in model's filename. Definitions are
reloaded before GO; invalid or conflicting definitions are reported instead of
silently selecting a different model.

For model setup details, see the [model guide](docs/inference-policy.md).
[Development notes](docs/development.md) and technical history are separate from this guide.

[License](LICENSE) · [Third-party notices](THIRD_PARTY.md)


## 0.2: timing and model runtimes

Settings → Timing selects automatic, native, VAD speech-region or forced-alignment
timing without editing model TOMLs. Qwen3-ASR, ReazonSpeech K2 and faster-whisper
model definitions are included. See the [provider/model guide](docs/provider-models.md)
for custom models, isolated workers and independently replaceable runtimes.

## 0.2.1: lightweight Windows Vulkan and Apple Silicon Metal

Windows supports CPU and Vulkan only. CUDA/cuDNN/cuBLAS, CTranslate2 and
faster-whisper are excluded from the standard ZIP and from automatic GPU selection.
Extract updates into a fresh folder. Previous CUDA preferences migrate to Vulkan.

The macOS ZIP contains **ASR2RPP.app** for **Apple Silicon, macOS 14+** (not Intel
or universal2). It includes CPU/Metal native engines and the CPU sherpa worker.
Install FFmpeg separately; select its path in Settings if it is not detected.
The app is ad-hoc signed, **not notarized**. Verify its origin/checksum before
allowing it in Privacy & Security. Add custom TOMLs through Settings, outside the
signed .app. CI checks relocation, architecture, signatures and native execution.
Metal execution is recorded separately from compilation when a CI VM has no GPU.
