# Provider / timing implementation

Baseline: 7fdffae93e9e64643b2e6d71569dd541534f09df.

1. Integrity and correctness: safe tensor bounds, cache verification and locking,
   request-level partial success, persistent failure logs, fine-grained diarization.
2. User-selected timing: automatic / native / VAD speech regions / forced alignment;
   model definitions describe capabilities, not mandatory user workflow.
3. Provider boundary, schema-v2 multi-file models, runtime capability inspection,
   native and optional external workers without executing code from model TOMLs.
4. Additional ASR providers, CUDA runtime builds, upstream compatibility checks,
   Windows frozen validation and complete distribution archive.

Each stage is committed only after its regression suite is run. Model weights,
private media and local credentials are never committed or bundled.

## Stage 2 validation

235 Python tests pass, including GUI setting persistence and 19 user-timing
regression cases. RTX 4080 / WSL native smoke on a 25-second Japanese clip:
VAD mode created 8 speech-region items; forced alignment created 8 grouped items
with forced-alignment provenance. Both RPPs were generated and all intervals were
inside the audio. This is a functionality check, not an accuracy benchmark.

## Stage 3 contract

The stage-major GUI path and CLI use the same bounded queue scheduler. Provider implementations
encapsulate native CLI differences; a version-1 JSONL worker interface supports
isolated sherpa-onnx, faster-whisper and explicitly selected external executables.
Model schema 2 supports multiple asset roles, directory entries, declarative
parameter controls and execution capabilities. Model files never import modules,
register executables or install software. User runtime registration is separate,
requires explicit trust and checks the executable hash; it is not an OS sandbox
or a DLL supply-chain verifier. Keep registered runtime folders read-only.

Native speaker labels survive normalization/alignment unless the user selects
external diarization or disables speakers. Fine timestamps are kept until export.
The previously tested Stage 2 GUI is retained; the proposed GUI module split was
not published because a source-write operation was blocked.

The new Qwen3-ASR 0.6B Q8_0 native CUDA runtime was built on the authorized RTX 4080
host. A 25-second private Japanese fixture produced nonempty text successfully
(exit 0, 5.04 seconds including process/model setup). Only these aggregate
measurements are recorded; neither the audio nor transcript is published. This
is a native functionality smoke, not a recognition accuracy benchmark.


## Stage 4: portable runtimes and upstream separation

- Windows Python/Qt regression suite: 281 passed, 1 symlink-privilege skip.
- Added public-C-API Whisper region runner; default builds never modify upstream CLI source.
- Added pinned Qwen3-ASR 0.6B, ReazonSpeech K2 and faster-whisper model definitions.
- Added isolated frozen worker and shared CUDA dependency layout; worker capability probe succeeds after freezing.
- Native CPU builds completed locally; GPU/Vulkan build and complete ZIP validation remain release gates.
- GUI and CLI preflight capabilities before downloads and retain runtime identity in job settings.

The branch remains a draft until the complete portable ZIP has passed native inference and lifecycle checks.
