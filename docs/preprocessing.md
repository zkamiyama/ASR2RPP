# Background removal and media references

[日本語ガイド](../README.ja.md) · [English guide](../README.md) · [Model table](models.md)

Background removal is optional. OFF uses original media for inference and output. ON runs the separator once and feeds its vocal stem to enabled ASR/alignment/diarization stages. The original input is never overwritten.

**RPP Audio → Original** keeps the project's references on the original media: background audio remains audible in REAPER, although recognition used the cleaned stem. The temporary stem is discarded.
**RPP Audio → Processed** saves one continuous `<project>_vocals.wav` beside the RPP and uses it as the reference. It is not one WAV per utterance. Keep it with the project. Selecting processed output without enabling a separator is rejected.
The full reference goes into the first, muted ORIGINAL track. A limited input-range run produces a processed reference for that range, not a reconstructed full-length source.

The bundled `mel-big-beta7` definition downloads a pinned CKPT and YAML, validates their hashes, and converts to F16 GGUF using the restricted torch-free reader, NumPy/safetensors and `audiocpp_gguf`. No manual conversion or PyTorch installation is required. Checkpoint plus conversion intermediates need more disk than the final GGUF. The source checkpoint and successful intermediate are removed by default; enable **Keep source checkpoints** to retain the original. Failed conversion retains its source for retry. An unrelated preconverted GGUF is never silently substituted.

Separator settings are audio.cpp **session** options, for example `num_overlap`. Each model has its own supported options; do not assume one separator's values work for another. Windows uses Vulkan/CPU; Mac uses Metal/CPU.

Output-name collisions advance both RPP and WAV suffixes without overwriting files. Timeline position and reference-file offset remain separate for cropped media. Sample rate, channels and length are checked; an unexpected duration change is an error, not a reason to stretch the transcript.
These checks and the [real-model matrix](models.md) do not certify separation quality, zero algorithmic latency or word-boundary accuracy.
