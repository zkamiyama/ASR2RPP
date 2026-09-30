# Outputs / 出力仕様

[日本語README](../README.ja.md) · [English README](../README.md)

## One inference, several outputs

The selected formats use **one shared edit timeline**. Adding OTIO or JSON does not run ASR again.
The default is RPP. GUI chips may be empty while editing preferences, but GO rejects an empty selection before starting a worker. CLI validation rejects unknown/empty/duplicate formats.

All selected outputs share a basename. Existing `.rpp`, `.otio`, `.json`, diagnostic directories and saved vocals reserve that basename, so no earlier output is overwritten. Caught export failures roll back only the new format files created by the current export; diagnostic reports remain for investigation. An external process kill or power loss can leave incomplete files.

## Timeline semantics

Times are **half-open `[start,end)` intervals in seconds** on FFmpeg's normalized media timeline. The first audio stream (`0:a:0`) is used. Media is linked, never embedded.

* **ORIGINAL:** one full-length reference clip, muted. This is the source media, or the entire saved vocals WAV when Processed reference is selected.
* **Transcript / speaker tracks:** recognized clips at their actual timeline positions, including silence gaps. Unknown speakers remain UNKNOWN. Same-speaker overlaps use extra lanes rather than shifting or cutting speech.
* **Text:** clip/item names contain the recognized text. Individual token/word detail remains in JSON, even when clips are grouped into readable utterances.

For clipped recognition, the original track still references the full source. For a processed selection that starts at 90 seconds, a unit at 2 seconds has timeline position **92**, but source offset **2** in the saved vocals file. JSON explicitly records both origins. It never invents word timing by dividing text length.

## RPP and OTIO

RPP uses relative media paths when possible. OTIO uses percent-encoded absolute `file:` URLs, official `RationalTime`/`TimeRange`, Audio Tracks, Clips and Gaps. Times are not rounded to video frames. ORIGINAL has both `Track.enabled=false` and `Clip.enabled=false`; the same intent is repeated in `asr2rpp` metadata.

**Import compatibility is editor-specific.** A valid `.otio` file does not guarantee every NLE supports audio-only timelines, file URLs, sample-rate timebases, speaker metadata or disabled tracks. Check the receiver, or use its supported OTIO adapter. There is no bundled NLE integration or audio rendering.

Official references: [OTIO file specification](https://opentimelineio.readthedocs.io/en/latest/tutorials/otio-file-format-specification.html), [schema API](https://opentimelineio.readthedocs.io/en/latest/api/python/opentimelineio.schema.html).

## Raw JSON: `asr2rpp.export`, version 1

This is a **versioned interchange document**, not a dump of Python objects. It contains all persisted stage result JSON/JSONL/text plus normalized and edited results. Audio, model weights and executable files are deliberately not embedded.

| Field | Meaning |
|---|---|
| `schema`, `schema_version` | `asr2rpp.export`, integer `1`. Check these before conversion. |
| `generator` | Application name and version. |
| `media.source` | Absolute input path, file URL, SHA-256 and selected audio stream. |
| `media.reference` | Actual linked file, timeline origin and full duration. |
| `media.inference` | Selection offset/duration and optional preprocessing information. |
| `timebase` | Unit origin, timeline origin and the explicit offsets between them. |
| `settings`, `provenance` | User options, models/revisions/hashes, runtime records and processing manifest. |
| `transcript.units` | Fine normalized/aligned/speaker-assigned units before clip grouping. |
| `transcript.edit_units` | Grouped utterance units used for exports; warnings are kept alongside them. |
| `timeline` | Format-independent tracks/clips with mute, speaker, `start_seconds`, `source_start_seconds`, `duration_seconds`. |
| `results` | A map of relative diagnostic filenames to their original-byte SHA-256, size, encoding and parsed data. Includes native results and correction history. |

`results` uses `json`, `jsonl`, `utf-8` or `base64`. Invalid JSON/UTF-8 is preserved as base64 rather than silently corrected. A SHA refers to **the original file bytes**, not a reserialized parsed JSON object. Filenames inside native requests may point to temporary audio that has already been removed; use **`media` and `timeline` for lasting media references**. Final cleanup and performance telemetry are recorded in the adjacent report and are not promised to be in the interchange JSON.

`results/history/*` preserves ASR-normalized, aligned and speaker-assigned units where those stages run. Native timestamp granularity and `method` remain distinct: VAD utterance bounds, emission frames, native intervals and forced alignment are not interchangeable.

For a custom converter, use `timeline` for edit placement and `transcript.units`/`results` for additional detail. Paths and text can disclose personal information: review before sharing. The file carries data only; conversion does not execute commands from it. Keep the referenced media available. A moved file must be relinked in the editor, or update the JSON's reference path/URL before conversion.

[Machine-readable schema](export-schema.json)

## Command line

Windows:

```powershell
# Default output: RPP only
.\asr2rpp-cli.exe run recording.wav --asr whisper-base --asr-language ja --asr-device vulkan

# One ASR run, three outputs
.\asr2rpp-cli.exe run recording.wav --format rpp --format otio,json --output-dir exports

# VAD for a text-only ASR
.\asr2rpp-cli.exe run recording.wav --asr qwen3-asr-06b --asr-device vulkan --timing vad --format json

# Optional alignment and diarization
.\asr2rpp-cli.exe run recording.wav --align qwen-forced-aligner --diar nemotron-diarization --timing alignment --vad-before-alignment --format rpp,otio,json

# No models, downloads or inference: re-export a saved JSON
.\asr2rpp-cli.exe convert exports\recording.json --format rpp,otio --output-dir converted
```

On Mac replace the executable with `./ASR2RPP.app/Contents/MacOS/asr2rpp-cli` and Vulkan with Metal. Use `--help` for parameters, model acquisition and runtime registration. `--reference-audio processed` requires `--preprocess`; the old `--rpp-audio` spelling remains an alias.

Paths to generated files are printed to stdout; progress/errors go to stderr. Success returns 0. Invalid arguments, configuration or conversion return nonzero. Partial queue failures are retained as failures, not reported as a successful whole queue. `convert` creates a new `_export` basename and never overwrites its input JSON.

## Minimal custom conversion

```python
import json
from pathlib import Path

result = json.loads(Path("recording.json").read_text(encoding="utf-8"))
if (result.get("schema"), result.get("schema_version")) != ("asr2rpp.export", 1):
    raise ValueError("Unsupported result schema")

reference = result["timeline"]["reference"]["path"]
for track in result["timeline"]["tracks"]:
    for clip in track["clips"]:
        print(track["name"], track["muted"], reference,
              clip["start_seconds"], clip["source_start_seconds"],
              clip["duration_seconds"], clip["text"])
```
