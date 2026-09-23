# Optional preprocessing and RPP audio references

Preprocessing and RPP source selection are independent settings. Preprocessing OFF uses original media everywhere. ON sends the resulting vocal stem to every enabled inference stage. RPP source selection then chooses original media or the saved processed stem. Original inputs are never overwritten; items reference one source, not separate utterance files.

## GUI

Launch `python launcher.py` or `python -m asr2rpp.cli gui`. The first card controls background removal, model, runtime device and session parameters. Inside it, `RPPで再生する音声` offers original media (inference only) or processed WAV (persistent). Its controls are greyed out with preprocessing OFF. The summary states the effective RPP reference. Each queue run snapshots the settings.

## CLI

```
python -m asr2rpp.cli run input.mp4 --preprocess mel-big-beta7 --rpp-audio original
python -m asr2rpp.cli run input.mp4 --preprocess mel-big-beta7 --rpp-audio processed --output-dir output
```

These require the exact model to have been converted and installed, as below. `--rpp-audio processed` without `--preprocess` is an error, not a silent fallback. Omitting both leaves the existing non-preprocessed path unchanged. The separator parameters are audio.cpp **session** options, for example `--preprocess-params '{"num_overlap":2}'`.

## Asset lifetime

Original reference: generated preprocessing audio is temporary. The RPP points to the original media, so playback retains the original background audio. Processed reference: the chosen stem is copied, without further conversion, to `<project-stem>.media/vocals.wav`. This is a persistent project asset outside the temporary directory. Keep the `.media` folder with the RPP. Same directory and Output directory both use this placement rule. Other drives use an absolute path when relative paths are not possible.

RPP generation remains non-destructive in both modes. A processed WAV is a new source master, not one file per spoken segment. A limited validation interval creates a master for that interval, not a full-length reconstructed original.

For a crop beginning at source time 10 s and a transcript interval starting 0.25 s into the crop, POSITION is 10.25 in both modes. Original media uses SOFFS=10.25; the saved crop uses SOFFS=0.25. Project position and file origin are intentionally distinct.

The separator input is stereo at its configured rate (44.1 kHz for this model). Model-specific mono/16 kHz/24 kHz files are generated only downstream for inference. They are never substituted for the persistent stem. The adapter checks channel count, sample rate and output length (maximum one-sample discrepancy); unexpected duration changes fail rather than stretch timestamps. This is not a proof of zero waveform latency or alignment accuracy. RF64 is not supported in this preview.

## Exact big_beta7 checkpoint status

Requested checkpoint: https://huggingface.co/pcunwa/Mel-Band-Roformer-big/blob/main/big_beta7.ckpt

Matching configuration: https://huggingface.co/pcunwa/Mel-Band-Roformer-big/blob/main/big_beta7.yaml

Checkpoint SHA-256: `9d68b9a8689a3500c45d3418a7811934e557760db07285f483088a0965b0eb88`.

The included TOML is a **conversion-required template**, not a ready-to-run model download. Set `source.path` to a verified audio.cpp SafeTensors/GGUF package converted from this exact checkpoint. No conversion has been completed or validated by this feature addition. A generic existing Mel-Band-RoFormer GGUF is not evidence of matching weights and is not silently substituted. Raw `.ckpt`/`.pt`/`.pth` paths are rejected with a conversion-required error.

The asset-routing tests use an explicitly mocked separator and recognizer. They validate cleanup, persistent references, original-file integrity, time origins and GUI control state, not the neural model's separation quality. Private sample audio is not uploaded to public CI.
