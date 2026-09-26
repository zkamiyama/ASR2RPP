# Release regression checklist

These checks are a build gate, not a promise of perfect transcription accuracy.
Use the exact commit's Windows ZIP, not a stale artifact from an earlier run.

## Lifecycle and definitions

- GO / immediate STOP / GO; stop during preparation, one file or a stage-major queue.
- Failed preparation, missing queue results, per-file failure, late/duplicate signals.
- Close while busy; wait for process cleanup; no QThread destroyed while running.
- Worker-start exception restores idle controls and incomplete items can be retried.
- Completed items and their outputs stay unchanged across retries.
- GUI and CLI report the hash and real path of the EXE-adjacent TOML.
- No `_internal/models`, `_MEIPASS` model definitions or user-folder template copies.
- Custom/legacy ID collisions are reported without overwriting files.

## Executable identity and documentation

- Every packaged EXE contains a primary RT_GROUP_ICON with all nine configured sizes.
- Each embedded image matches its generated ICO; code section hashes are unchanged.
- Native build-manifest hashes describe the branded package, not the unbranded cache.
- The frozen window icon renders at 16 and 64 pixels; CLI remains Qt-free.
- English and Japanese guides link to each other at the top and ship with the ZIP.
- Source SVGs and Apache license/attribution ship without any font binaries.

## Native processing and publication

- Public-fixture ASR, alignment, diarization, combined optional stages and Unicode paths.
- PCM-plan versus WAV transcription equivalence, independent history/timestamp settings.
- Full ORIGINAL duration survives clipped inference and unavailable FFmpeg time progress.
- No FFmpeg or model weights in the package; no private fixtures/transcripts in CI.
- Only after all tests succeed: main release job verifies SHA-256 and embedded commit,
  then publishes a new immutable preview tag. Never replace existing release assets.
