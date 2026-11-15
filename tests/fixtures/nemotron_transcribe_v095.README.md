Hand-authored synthetic contract fixture, not a real inference/accuracy result.
Matches audio.cpp v0.9.5, commit 61c7838baf1f7f34769b5b7bd369c0d78edf9535:
- src/community_models/nemotron_transcribe/session.cpp:252-331 (custom words/diagnostics)
- app/workflow/file_sink.cpp:255-284,291-306,420-454 (manifest, generic words/turns)
- app/cli/main.cpp:1048-1053 (separate transcript text)

The envelope is ASR2RPP's lossless combination of those separate files. Generic
sample offsets deliberately use 48 kHz; custom intervals are always seconds.
Speaker_2 maps to transcript tag spk:0 to test the distinct acoustic/tag namespaces.
The final two words overlap the first speaker. Japanese words here are characters.
