<p align="center"><img src="assets/branding/app.svg" width="96" height="96" alt="ASR2RPP: speech into an editable timeline"></p>

# ASR2RPP

**Local speech recognition. Editable timelines. Your original media stays untouched.**

[日本語](README.ja.md) · [Download](https://github.com/zkamiyama/ASR2RPP/releases/latest) · [Models](docs/models.md)

Turn audio or video into a **REAPER project, OpenTimelineIO timeline, or reusable JSON**.
Optional background removal, alignment and speaker labels all work before export, so the selected formats share the same edit decisions.

## Start in four steps

1. Download your OS's **complete ZIP** from Releases and extract it into a **new folder**.
2. Open `ASR2RPP.exe` on Windows or `ASR2RPP.app` on Mac. Keep the other files together.
3. Drop a recording into the window. For a first test, use the included `examples/sample.wav`, **Whisper Base**, language **en**, optional stages OFF, and **Settings → Timing → Automatic**.
4. Press **GO**. Selected model weights and a missing FFmpeg are acquired on first use. Double-click a completed row to open the output folder.

The first-run ASR is **Anime Whisper / ja** in both GUI and CLI. Saved GUI model/language choices are preserved. The included sample is English, so select Whisper Base / en explicitly for that sample. Anime Whisper is a larger, experimental Japanese conversion; its initial download is not the lightweight sample setup.

For your own recordings, choose the appropriate language. The default destination is beside the input. The gear opens settings; the globe switches English/Japanese. Downloads require a network connection. Once dependencies are present, recognition stays on your computer.

## Choose your output

The main window starts with **Format: `RPP ×` `+`**. Click **+** to add OTIO or JSON; click a chip's **×** to remove it. The menu offers only unselected formats. An empty selection shows an error and does **not** start inference. Your selection is remembered.

| Format | Use it for |
|---|---|
| **RPP** | Open directly in REAPER. |
| **OTIO** | Exchange a timeline with an OpenTimelineIO-capable editor or adapter. |
| **JSON** | Keep raw stage results, corrected units, model provenance, media paths and an explicit edit timeline for your own converters. |

RPP and OTIO contain a full-length, **muted ORIGINAL** reference track, followed by transcript or speaker tracks. Gaps stay in place. Overlapping clips from one speaker use additional lanes. Media is **referenced, not embedded**. OTIO writes `enabled=false` for the muted track and clip; the receiving editor must honor this property. Check its import behavior before production use.

Background removal can save a continuous `*_vocals.wav`; choose **Reference audio → Processed** to edit against it. Otherwise the timeline references the original media. Existing outputs are never overwritten. The adjacent `.asr2rpp` folder holds diagnostics; JSON also includes the result data needed for conversion without that folder. **Both can contain private text and paths.**

[Output structure, JSON schema and conversion examples →](docs/outputs.md)

## Models and timing

The ZIP contains **20 model definitions, not 20 sets of weights**. Only selected weights are downloaded. The [model table](docs/models.md) lists languages, sizes and validation scope. Canary and Moonshine's included models are English-oriented, not Japanese ASR.

Choose timing under **Settings → Timing**:

| Setting | Meaning |
|---|---|
| **Automatic** | Use native model intervals when available, otherwise VAD speech regions. An enabled aligner takes precedence. Legacy TOML defaults are supported. |
| **Native model intervals** | Use intervals returned by the ASR. Not available for text-only models. |
| **VAD speech regions** | Locate utterances and recognize each region. **These are not word boundaries.** |
| **Forced alignment** | Align the recognized text to the audio. Select the aligner in the main window; use VAD before alignment for long/coarse segments. |

Recognition, timing and speaker labels can be wrong. Alignment does not fix a wrong transcript. Speaker labeling does not separate overlapping voices into separate recordings.

## Requirements

| OS | Supported package | GPU |
|---|---|---|
| **Windows** | x64 CPU with AVX2 | Vulkan only; a compatible GPU driver is required |
| **macOS** | Apple Silicon, macOS 14+; not Intel | Metal only |

CPU is available on both. The bundled ReazonSpeech K2 worker is CPU-only.

**Included:** Python runtime, Qt/PySide6, NumPy, safetensors, OpenTimelineIO, whisper.cpp, audio.cpp and a CPU sherpa-onnx worker.
**Acquired separately on first use:** selected model weights and, when missing, checksum-verified FFmpeg. A custom FFmpeg path can be set in **Settings → Advanced**.
**Not needed:** a separate Python install, PyTorch, CUDA Toolkit, Vulkan SDK or Xcode. CUDA, CTranslate2 and faster-whisper are not bundled.

REAPER is only needed to open RPP; it is not needed to generate JSON or OTIO. Windows executables are unsigned; the Mac app is ad-hoc signed, not Apple-notarized. Verify the release source and checksum. Dependency/model terms are separate from the app's [MIT license](LICENSE); see [third-party notices](THIRD_PARTY.md).

## Automate and customize

```powershell
.\asr2rpp-cli.exe run recording.wav --asr whisper-base --asr-device vulkan --format rpp,otio,json --output-dir exports
.\asr2rpp-cli.exe convert exports\recording.json --format otio --output-dir converted
```

`--format` may be repeated; omission means RPP only. Conversion from JSON does not run or download ASR models. On Mac the CLI is inside `ASR2RPP.app/Contents/MacOS/asr2rpp-cli`. [CLI reference →](docs/outputs.md#command-line)

Add a unique `.toml` under **Settings → Advanced → Open custom TOML**; do not modify the signed `.app`. Definitions reload before GO. **TOML adds checkpoints for architectures supported by an installed runtime; it cannot implement an unknown architecture.** [Model and runtime examples →](docs/provider-models.md)

Press **STOP** and wait for cleanup. GO retries waiting/stopped/failed items, not completed items. To redo a completed input, remove and re-add it. Save model/cache locations under **Settings → General**. Updates go into a new folder; unchanged model definitions reuse verified cached weights.

[Model catalog](docs/models.md) · [Development and upstream updates](docs/development.md) · [Output schema](docs/outputs.md) · [Validation](docs/implementation-stages.md)
