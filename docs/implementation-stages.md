# Staged implementation and validation

## Stage 1 — correctness and integrity

Tensor bounds and stride validation, per-request failure isolation, persistent
failure diagnostics, artifact hashing, installation locking/atomic state, and
fine native timing before speaker assignment. Same-size rewrites are checked by
content, not by a cached file timestamp (including Windows coarse timestamps).

## Stage 2 — user-owned timing

GUI/CLI Automatic, Native, VAD speech-region and Forced-alignment modes. Settings
persist independently of model definitions; bounded text-only ASR supports the
same timing workflow outside Whisper. Native word intervals are never fabricated.

## Stage 3 — provider/model separation

Data-only TOML schema 2, artifact-role/multi-file models, declarative scalar UI
parameters, fixed provider interfaces, trusted external-worker protocol 1,
explicit runtime registration/probing/rollback, native speaker preservation,
and bounded queue windows. Unknown architectures still need an implementation;
TOML selects an existing implementation rather than executing arbitrary code.

## Stage 4 — upstream separation and portable release

Default native builds use unmodified pinned upstreams. A small public-whisper.h
region runner replaces CLI text patches for bounded transcription; unsupported
helper options fall back to stock CLI without discarding those options.
CPU/Vulkan/CUDA native runtimes and an independent frozen sherpa-onnx /
faster-whisper worker are packaged with dependency notices and manifests.
Qwen3-ASR 0.6B, ReazonSpeech K2 and faster-whisper-base definitions are pinned.

## Verified release, 2026-09-30

Application commit: `dd5846731465f50eab6786c90655c3a13ea7afba`.

- Linux/Qt: 288 tests passed.
- Windows/Qt: 287 passed; one symlink-privilege-dependent test skipped.
- Native PCM Release test: 1 passed (assertions enabled).
- Extracted portable ZIP: 11/11 inference cases passed, with Python/toolkit paths
  removed. Includes Whisper and Qwen CUDA/Vulkan, Reazon CPU, faster-whisper CPU,
  CUDA and batch=4, and Qwen CUDA forced alignment. All preserve original media.
- Frozen optional-stage checks: all expected outcomes passed, including native
  diarization, alignment, Unicode paths, required-alignment rejection, full muted
  ORIGINAL track and four independent regions with exact stock/helper text match.
- Frozen GUI lifecycle, CLI commands, native inference and 14 EXE icons verified.
- ZIP hash checked after extraction; font files and ASR model weights excluded.
  Faster-whisper's packaged Silero VAD asset is included and declared separately.

[Machine-readable results](validation/windows-20260930.json).
These are functional tests on synthetic Japanese/public English speech, not a
quality benchmark or a claim that every model/backend combination is certified.
The local Windows CUDA build used CUDA 12.6 and an explicitly recorded compiler
compatibility flag. Other GPUs may require a different runtime build.

## Boundaries kept explicit

The app is unsigned. GPU drivers and REAPER are external prerequisites; model
weights and ffmpeg.exe are acquired separately. PyAV's FFmpeg shared libraries
and all redistribution notices remain in the worker pack. Workers are reused
within bounded request groups, not promised to remain resident between GO runs.
Optional future work includes cross-run resident workers and corpus-level CER,
word-boundary and diarization quality benchmarks. Native updates still require
capability/contract tests even though they no longer patch CLI source text.
