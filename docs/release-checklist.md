# Release checklist

Use the exact committed source and extracted ZIP. Build success alone is not inference validation.

## Application behavior

- Unit/Qt tests and native PCM Release assertions pass on each supported OS.
- GO/STOP/GO, errors and close-while-busy clean up without changing completed items.
- Unique model IDs, direct bundled definitions, custom definitions and pre-download runtime checks work.
- Recognition, VAD/alignment timing and native/external speaker labels keep their distinct provenance.
- ORIGINAL remains the full muted reference. Input hashes are unchanged; output collisions do not overwrite files.

## Packaging

- Windows: x64/AVX2, CPU/Vulkan only. Mac: arm64/macOS 14+, CPU/Metal only.
- No CUDA, CT2, faster-whisper, model weights, FFmpeg binaries, private media or font files.
- No orphaned excluded-provider metadata can advertise an absent engine.
- CPU sherpa worker and each native pack have their required libraries, notices and accurate hashes.
- Windows EXE icons and code-section integrity are checked. Mac architectures, library references and signatures pass after relocation.
- Both READMEs, the model table, customization examples and current third-party inventory are bundled.

## Evidence and delivery

- Check the actual frozen GUI/CLI, not just the source import path.
- Model changes get a real-weight smoke test where hardware is available. The [catalog matrix](models.md) distinguishes source/native tests from final-package tests.
- A missing GPU/operation is an explicit skip, never an inferred success. Hardware-required runs fail on skips.
- Record skipped tests and limits; short functional fixtures are not recognition-quality or long-audio benchmarks.
- Inspect CI reports, archive integrity, embedded commit and SHA-256 before delivery. Keep earlier releases identifiable and never silently replace their bytes.
