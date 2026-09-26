# Development and maintenance

This is developer documentation. End-user setup is in [English](../README.md) and
[Japanese](../README.ja.md); do not put build history or implementation caveats
back into the quick-start guide.

## Local development

Use Python 3.11 or newer (CI uses 3.12). Install the project and GUI/test dependencies:

```sh
python -m pip install -e '.[gui]' pytest
python -m pytest tests
python launcher.py
```

The standalone RPP serializer uses the standard library. The application uses
separate whisper.cpp/audio.cpp processes, NumPy and safetensors. It must not
require PyTorch or Transformers on the user's computer.

Build the pinned native engines with `tools/build_native.py`. The small native
PCM-plan patch is fingerprinted; never replace or force-reset a developer's
modified upstream checkout. Windows packaging runs `tools/package_windows.py`.
CPU and Vulkan builds are included; CUDA measurements in the performance report
are not Windows/Vulkan performance claims.

## Runtime structure and invariants

- `catalog.py` reads shipped TOMLs in place from the EXE-adjacent `models` directory.
  User definitions live in `custom-models`; IDs must be unique. Do not reintroduce
  `_MEIPASS` model copies or silently let an old definition shadow the package.
- `run_state.py`, the Qt worker, and `lifecycle_smoke.py` maintain one terminal event
  per input per attempt. STOP waits for cleanup; GO retries incomplete items only.
  Late signals must not mutate a later run. A thread-start failure must also unlock GO.
- `media.py` normalizes media time, preserves samples, and measures whole-reference
  duration from complete PCM output when FFmpeg's progress timestamp is unavailable.
- `vad_asr.py` uses independent bounded input regions without decoder history.
  `alignment.py` refines timestamps when selected. `transcript.py` groups and assigns
  speakers without inventing text boundaries. `rpp_export.py` keeps ORIGINAL first.
- Original media is never modified. Temporary inference audio is removed on normal
  completion, failure and cancellation. Diagnostics contain private text and paths.

## Release gate

The application workflow runs Linux/Windows unit tests and native PCM tests,
then tests the actual frozen Windows programs: catalog provenance, GUI lifecycle,
recognition, optional stages, original references, Unicode paths and PCM/WAV parity.
The package step verifies icon resources for every staged EXE and updates native
manifest hashes after resource-only branding. It never modifies the native cache,
an external FFmpeg, code sections, or signed third-party executables.

Main publishes a commit-tagged preview ZIP only after all jobs succeed. The
publication step checks the ZIP SHA-256 and embedded source commit. Never replace
an existing release's assets with a new untested build. Fonts, model weights and
FFmpeg binaries must not be included in the application ZIP.

## Technical background

[Inference policy](inference-policy.md), [performance](performance.md),
[measurements](performance-results.json), and [release notes](release-notes.md)
record the design and known limitations. The [earlier README](history/pre-release-readme.md)
is retained as a historical record, not current user instructions.
Executable artwork provenance is in [the icon notice](../assets/branding/NOTICE.md).
