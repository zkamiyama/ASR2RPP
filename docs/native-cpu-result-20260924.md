# Native CPU ASR validation — 2026-09-24 JST

## Outcome

GitHub Actions run [35914889095](https://github.com/zkamiyama/ASR2RPP/actions/runs/35914889095) completed successfully. This is a native CPU runtime/download/transcription smoke test, not a Japanese speech accuracy benchmark and not a Nemotron 3 Diarization validation.

- Tested code commit: `ce3c7fe9b11160a3e709dac26e0cd99e8acee937`.
- Runner: standard `ubuntu-24.04`, x86_64, with no GPU runner requested.
- Detected CPU model: AMD EPYC 9V74; this name does not imply that the VM received all of the physical processor's cores.
- Python: 3.12.3, used only for fixture/result validation.
- FFmpeg: 6.1.1-3ubuntu5.
- Native runtime: NVIDIA NeMo-Speech.cpp 0.1.0, CPU archive.
- Runtime archive SHA-256: `0f74131d631ad2c694cf0ec53490866bb6461147959589a69fb6fc231944065b`.
- ASR model: `nvidia/nemotron-3.5-asr-streaming-0.6b`.
- Model revision from runtime index: `1c8deaecc64b91f034d73e08dd8b64625eb3395d`.
- Loaded model: `nemotron-3.5-asr-streaming-0.6b.q8_0.gguf`; downloader reported 707.2 MiB and verified size/SHA-256.
- The workflow did not install PyTorch; inference ran through the native executable.

## Measurements from retrieved artifacts

The input was locally synthesized English, not user-provided media. It contained 89,085 samples at 16,000 Hz: **5.5678125 seconds**.

| Measurement | Observed value |
|---|---:|
| Model download and integrity verification | 41.58 seconds |
| Transcribe process wall time, including model loading | 4.12 seconds |
| Transcribe maximum resident set size | 1,724,156 KiB, approximately 1.644 GiB |
| Transcribe exit status | 0 |
| Word entries returned | 14 |

These are one-run, short synthetic-audio observations. They do not establish 60-second/long-form performance, Japanese accuracy, or PyTorch-vs-GGUF equivalence.

Input text: `This is a local speech recognition test. The sample is shorter than one minute.`

Returned text: `It is a local speech recognition test. The sample is shorter than one minute.`

Thus runtime success does not mean transcription perfection: the first word changed from `This` to `It`.

## Timestamp audit

The JSON contained `words[].start` and `words[].end`. Starts were nondecreasing and individual durations were positive, but:

- `test.` ended at 3.60 seconds, while `The` started at 3.52 seconds: adjacent intervals overlap by 80 ms.
- `minute.` ended at 5.60 seconds, exceeding the measured input duration by 32.1875 ms.

These are structural warnings, not a measurement of alignment accuracy against human-annotated boundaries. Preserve these raw timestamps. Any later clipping, alignment, or final edit-boundary correction must be a separate, logged transformation rather than silently overwriting raw outputs. Do not automatically map every raw interval directly to an RPP item.

## Artifact provenance

- GitHub artifact ID: `10775135835`.
- Artifact name: `native-cpu-smoke-35914889095`.
- ZIP SHA-256: `22ca2acdd18d4a147ac2b99a26f118472ac61bb439ec71240401ba83aeebe6b2`.
- Reports were downloaded through the GitHub connector and the ZIP hash was checked before inspection.
- Artifact retention was set to three days. The workflow regenerates reports; long-term retention requires an explicit storage policy.

The first run (`35914403916`) failed because the test script exported a bundled C++ library path globally, making system FFmpeg load incompatible libraries. The test script was corrected to scope the library path to the native subprocess, and the second run above passed. This failure was not evidence that CPU ASR was unsupported.

## Not tested

The user's private video, Japanese real speech, speaker assignment, the new Nemotron 3 Diarization checkpoint, RPP generation in this workflow, forced alignment, Metal, DirectML, Vulkan, CUDA, and model quantization parity remain untested by this smoke test. No private media or credentials were committed or uploaded.
