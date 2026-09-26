# Pipeline performance and invariants

This change is based on the VAD-region implementation at `13e84038`.
It does not lower precision, beam width, sample rate, VAD thresholds or separation
quality to improve benchmark numbers. Decoder history remains disabled for each
TOML-constrained VAD input, even when a model context is reused.

## What changed

- **Whisper VAD input:** one bounded PCM plan replaces a WAV per region and the
  64-input command batches. The native process loads the model once, opens the
  shared decoded PCM, seeks to integer sample offsets and converts only the current
  region to floats. One model session also serves compatible stage-major queue
  files. Model state is freed before alignment/diarization/separation changes.
- **Bounded memory:** no entire-video byte array or duplicated in-memory WAVs.
  Region audio buffers are at most 28 seconds, approximately 2.7 MB for PCM16 plus
  float samples, excluding model/graph memory. Large result sets retain per-region
  outputs; ordinary sets use a bounded consolidated result document.
- **Native compatibility:** the app-pinned whisper CLI has a small, fingerprinted
  input/output adapter. Upstream decoding, parameter handling and JSON serialization
  remain unchanged. Unknown custom binaries use the existing WAV path. Response
  files remove command-length-induced reloads for compatible older runtimes.
- **Queue decode:** same-file, same-sample-rate PCM is reused by ASR, alignment and
  diarization. It is released with the queue workspace after its final consumer.
  This does not reuse audio across different files, clip settings or preprocessing.
- **Separation wrapper:** recognition writes diagnostics directly to its final
  analysis directory. It no longer generates a nested RPP only to discard it or
  copies a complete diagnostics tree a second time. Final original/processed audio
  ownership and the full muted ORIGINAL track remain unchanged.
- **Diagnostics:** completed, media-free engine directories are renamed into place
  on the same filesystem. Cross-filesystem cases retain safe copying semantics.
  Native logs flush periodically rather than after every line; progress is still
  delivered immediately and a bounded reader queue prevents unbounded log memory.
- **Model preparation:** GUI preparation no longer hashes a local model and then
  immediately hashes it again in execution. Actual local model SHA-256 resolution
  and before/after source-integrity checks remain. Download SHA-256 is computed
  while receiving the stream rather than rereading the newly downloaded file.
- **Observation:** `performance.json` records per-invocation inclusive stage totals.
  Nested totals overlap; do not sum them. Queue-wide totals appear in each job's
  report and are explicitly labelled as shared. ASR raw metadata records actual
  input mode, result mode, process count and region-WAV count.

## What was deliberately not changed

Native separation overlap settings, Nemotron streaming mode, alignment precision,
word ownership, speaker assignment and output normalization are retained. A faster
but different transcript is not accepted as evidence of an optimization. Persistent
large-model residency across different stages was not introduced: avoiding VRAM
contention/OOM is more important than speculative overlap. SHA-256 checks were not
replaced by size/mtime-only trust. PCM/result parsing remains strict and errors or
cancelled processes are not retried under a silently different inference mode.

## Validation protocol

Compare old/new on the same pre-existing local weights, audio, precision, backend,
threads and settings. Repeat the full-video ASR test, report individual runs and
median, and compare normalized units including text, speaker and time fields.
Also run single-file alignment, diarization, separation+ASR+alignment+diarization,
Nemotron and multi-file queue cases. Short-stage timing changes can be noise; do not
present unchanged stages as universal speedups. Private media/transcripts stay local.
