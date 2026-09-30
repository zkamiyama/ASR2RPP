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
