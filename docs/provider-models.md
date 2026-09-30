# Custom models and independent runtimes

[Quick start](../README.md) · [日本語の具体例](custom-models.ja.md) · [Model catalog](models.md)

## What a TOML can and cannot add

A TOML describes weights, model capabilities and scalar options. Adding another checkpoint of a supported architecture normally needs no application changes.
It does **not** implement new operators, preprocessing, tokenization or decoding. GGUF/ONNX are formats, not a universal model implementation.
A new architecture needs a runtime/provider implementing it. Standard Windows GPU execution is Vulkan; Apple Silicon GPU execution is Metal. CPU works on both.

Open **Settings → Advanced → Open custom TOML**, create a unique UTF-8 filename, then reload or press GO. The filename stem is the model ID.
Do not shadow a bundled ID, edit the signed Mac app, or copy every bundled template into the user directory. The app reads bundled definitions in place and reports invalid/duplicate definitions.
Schema 1 remains accepted; use schema 2 for new definitions.

## Local Whisper example

Replace the path with an actual whisper.cpp-compatible model. Relative paths are relative to the TOML, not the process working directory.

```toml
schema_version = 2
provider = "whisper_cpp"
task = "asr"
name = "My Whisper"
sample_rate = 16000
[source]
path = "D:/ASR/models/my-whisper.bin"
[capabilities]
timestamps = "segment"
speakers = false
[defaults]
language = "en"
```

Text-only derivatives declare `timestamps = "none"`. The **user** chooses VAD or forced alignment in Settings → Timing. Do not manufacture word timestamps or add a mandatory alignment workflow to new model definitions.

## Native text-only ASR example

This example requires a Qwen3-ASR implementation in the installed audio.cpp runtime and a compatible GGUF, not an arbitrary file renamed to GGUF.

```toml
schema_version = 2
provider = "audio_cpp"
family = "qwen3_asr"
task = "asr"
name = "My Qwen ASR"
sample_rate = 16000
[source]
path = "D:/ASR/models/my-qwen-asr.gguf"
[capabilities]
timestamps = "none"
speakers = false
[execution]
mode = "offline"
output = "text"
max_batch_items = 16
max_audio_seconds = 28
[defaults]
language = "English"
```

`execution.output` is `text`, `words`, `segments` or `turns`; it selects the actual native output contract.
Use `turns` with `capabilities.speakers = true` when the engine returns speaker-labelled turns, as the VibeVoice example does. A `segments` file can omit speakers even when another output carries them.
`max_batch_items` bounds session requests; it does not promise simultaneous GPU batching. `max_audio_seconds` bounds segmented text-only requests.
Fixed-language engines that reject a language option declare `execution.pass_language = false` (Moonshine). The visible language field is then not forwarded; the engine's fixed language applies.

## Remote and multi-file weights

Replace `source.path` with `source.repo` (`owner/repository` on Hugging Face), an immutable 40-hex `source.revision`, `source.files`, and a `[source.sha256]` entry for every file. Use the file's SHA-256, not a Git blob SHA.
Shipped definitions provide tested, pinned examples. Revision changes can create a new installation rather than reusing old files.
A directory model uses `source.entry = "."` and `[artifacts]` role-to-relative-file mappings. See `models/reazonspeech-k2.toml`: encoder, decoder, joiner and tokens are separate files. Small configuration files are supported. Assets must remain within the model directory.

Installation uses per-model locks and atomic state updates. Retained file contents are verified before use. Verification detects corruption; it does not make an untrusted publisher trustworthy.
Model weights and conversion publishers retain their own licenses.

## Declarative parameter controls

`[defaults.request]` supplies inference options. `[defaults.session]` supplies session options.
A `[parameters.option_name]` table adds a typed control with `default`, optional `min`/`max`/`step`, `values` for enums, and `ja`/`en` labels or `tip_ja`/`tip_en` help.
Supported types are `bool`, `int`, `float`, `str` and `enum`. Session control names use `[parameters."session.option_name"]`.
The option must exist in the runtime; a GUI declaration does not implement it. Language spelling is model-specific: for example, the bundled aligner uses `Japanese`/`English`, while other models use `ja`/`en` or locale codes.

TOMLs cannot contain executable commands, import Python, install dependencies or silently register an executable.

## Install a compatible runtime independently

Native runtime overrides are under Settings → Advanced. The CLI also supports explicit registration, inspection and rollback:

```text
asr2rpp-cli runtimes register --provider audio_cpp --device vulkan --exe PATH --trust
asr2rpp-cli runtimes probe --provider audio_cpp --device vulkan
asr2rpp-cli runtimes rollback --provider audio_cpp --device vulkan
```

Use `metal` instead of `vulkan` on Mac. Use `cpu` for the bundled sherpa worker. On Windows the executable is `asr2rpp-cli.exe`; on Mac it is inside `ASR2RPP.app/Contents/MacOS/`.
Registration is a separate **code-trust** decision and pins the executable hash. Register again after replacing that executable. Capabilities/family/mode checks still apply.

For a different engine, a trusted separate process can implement `provider = "external_json"`, protocol 1.
The contract is implemented in `asr2rpp/worker_client.py` and `asr2rpp/provider_worker.py`: advertise capabilities, load local assets, process request IDs, return validated intervals/text with an explicit time origin, and report request errors independently. No shell interpretation is needed.
The standard worker contains CPU sherpa-onnx only. The faster-whisper TOML under `docs/optional-models` is a reference for an explicitly installed external CPU worker, not a bundled or automatically downloaded engine. Vulkan/Metal flags do not turn CTranslate2 into a Vulkan/Metal runtime.

For upstream revision changes, compilation and distribution checks, see [development](development.md).
