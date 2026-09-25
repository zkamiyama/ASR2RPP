# Anime Whisper: torch-free conversion trial

This experiment does not change main, the application, or existing model definitions. It reads the original `litagin/anime-whisper` SafeTensors and writes Whisper GGML using only NumPy and safetensors. PyTorch and Transformers are not installed or imported.

## Run (Python 3.11+, CI uses 3.12)

```bat
py -3 -m venv .venv
.venv\Scripts\python.exe -m pip install --only-binary=:all: numpy==2.3.5 safetensors==0.7.0
.venv\Scripts\python.exe convert_anime_whisper.py --download --model-dir anime-whisper-source --precision f16 --output converted\ggml-anime-whisper-local-f16.bin
.venv\Scripts\python.exe convert_anime_whisper.py --download --model-dir anime-whisper-source --precision f32 --output converted\ggml-anime-whisper-local-f32.bin
```

The original weights download is approximately 3.03 GB. The source revision is pinned to `22e2008a8182b357da3922a6308d095008f72973`; the source weights SHA-256 is checked. Allow about 10 GB free for source plus both outputs. Existing output files are never overwritten. `--download` may be omitted for a fully local conversion when config, vocabulary, weights and mel filters are already present.

Each conversion also writes `.conversion.json` with hashes and tensor metadata, and a `.toml` pointing to the local `.bin`. Copy only that generated TOML to the ASR2RPP model-definitions folder, restart the app and select `Anime Whisper · F16` or `Anime Whisper · F32`. Keep the binary at the recorded path. Existing Anime Whisper remains available for comparison. Initial prompts are still disabled.

## What differs from the upstream conversion path

- No PyTorch model construction, no Transformers, no pickle execution.
- Explicit tensor allowlist, dimensions, vocabulary order and special-token checks.
- A separate output projection is never silently discarded unless equal to the tied embeddings.
- F16 keeps native F32 auxiliary tensors at source precision.
- F32 never passes through F16, unlike the unconditional intermediate F16 cast in the referenced upstream script.
- Atomic output and per-tensor hashes support auditing.

The inference algorithm remains whisper.cpp. Transformers generation settings are recorded as inputs but are not embedded as executable behavior in GGML. Converting successfully does not prove that anomalous leading characters are fixed.

## Validation

`test_converter.py` tests F32 round trips, F16 auxiliaries, mapping, invalid inputs, projection handling and no-overwrite behavior. `inspect_ggml.py` reads back and compares GGML headers, mel filters, vocabulary and tensors. `validate_trial.py` requires a torch/transformers-free environment, converts the full original, compares it to the existing non-Q8 GGML and runs the pinned whisper.cpp CPU executable in timed and text-only modes.

The real-inference smoke uses four seconds of the public English JFK fixture. It is a model-loading/inference check, **not** a Japanese recognition benchmark or reproduction of the user's leading-character issue. No private user audio, credentials, model weights or runtime binaries are uploaded as artifacts. GPU behavior is not validated by this CPU trial.

## Sources and licensing

Original model: https://huggingface.co/litagin/anime-whisper

Existing conversion: https://huggingface.co/Aratako/anime-whisper-ggml

GGML layout and mapping reference: https://github.com/ggml-org/whisper.cpp/blob/a664346ea5c6dddff3e61a2b7b32dd4514613f50/models/convert-h5-to-ggml.py

Mel filters: https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/assets/mel_filters.npz

See LICENSE for the MIT terms. The trial does not redistribute model weights or installed libraries.
