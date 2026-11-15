# Nemotron 3.5 Transcribe (experimental)

The `nemotron-transcribe-iq4-nl` definition integrates **NVIDIA-Nemotron-3.5-Transcribe-30B-A4B**, not the existing 0.6B `nemotron-asr` model or the standalone `nemotron-diarization` model. The implementation uses the experimental community `nemotron_transcribe` family in **audio.cpp v0.9.5**, pinned at `61c7838baf1f7f34769b5b7bd369c0d78edf9535`. Rebuild the native packs before use; the previously released pack cannot acquire a new family by downloading a TOML file.

## Exact weights

Source: [Jmica community GGUF packages](https://huggingface.co/Jmica/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF/tree/304032f14e1e9e4042f5491d2d0194c044710bb7), pinned to the package revision in [v0.9.5's spec](https://github.com/0xShug0/audio.cpp/blob/v0.9.5/model_specs/nemotron_transcribe.json). Both SHA-256 values are pinned in the TOML, checked against the Hub revision metadata; isolated local tests also verified the downloaded files against these hashes.

| Required file | Bytes | SHA-256 |
|---|---:|---|
| `nemotron-transcribe-30b-a4b-experts-IQ4_NL.gguf` | 20,370,389,728 | `81af0d93cd98e778fef906bb350220215396c3bfdf99d6240c0607eba51933ef` |
| `mmproj-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf` | 1,670,410,080 | `4221029c8a565c4a80c507ffa760c716e595acf663f9f4273ce69910050e7ba2` |

The decoder is the explicit entrypoint; the encoder must remain beside it with its original filename. IQ4_NL refers to routed experts, not every tensor. Dense decoder linears are Q8_0, other tensors retain BF16/F32, and the encoder is F16 with an F32 CTC head. These are community conversions, not NVIDIA's original BF16/FP8/NVFP4 release. Upstream conversion references NVIDIA revision `a99e33f698257b8b69ca7834f19c73582eb1fa7e`. Weights retain OpenMDW-1.1 and component notices embedded in the package.

## RTX 4080 16 GB / 64 GB RAM

The combined download is 22,040,799,808 bytes (22.04 decimal GB). **File size is not peak VRAM or RAM. Short RTX 4080/CUDA tests are not a 30-minute quality or memory guarantee.** The catalog enables `moe_disk_experts=true`, preserves `weight_type=native`, keeps the native eight expert cache slots, disables host pinning, and uses prefill size 64. These defaults retain the bounded expert-staging configuration; they are not a guarantee for every recording or GPU.

Expert staging retains banks in GGUF/host memory and transfers selected experts into a bounded device cache. Dense weights, encoder, state and activations still consume backend memory. The six-experts-per-token checkpoint permits cache sizes 6–128; eight is the upstream default. Increasing cache or prefill size can increase VRAM. `moe_cpu_expert_layers` counts the first N of 23 routed MoE layers, skipping dense-only layers; `moe_cpu_share` distributes eligible cache misses to CPU while staging is enabled. Both default to zero and can reduce throughput. Pinning requires backend support and consumes pinned host memory. Staging workers default to four; use one without OpenMP/Apple dispatch.

Shared `moe_*` options are global session keys, not `nemotron_transcribe.*` keys. Do not combine canonical options with legacy family aliases. Weight storage overrides can expand the model drastically; this definition exposes only `native`.

Standard Windows packaging uses Vulkan, not CUDA. The v0.9.5 port documents Vulkan offload support but explicitly **does not validate full-model Vulkan or Metal execution**. Its reported real-model CUDA timings use a 96 GB RTX PRO 6000 Blackwell, not an RTX 4080. Updating the shared runtime pin also requires regression testing the existing catalog before release. See [native build instructions](development.md). No package or release is produced by adding this definition.

## Default-off native speakers and independent timing

Requests default to `language=ja`, `diarization=false`, and `return_timestamps=false`. The app owns these last two switches through **Settings → Timing** / `--timing` and `--speaker-source`, rather than an independent decoder checkbox that could contradict the pipeline. Conflicting raw ASR overrides fail before model acquisition. Other models retain their existing timing choices.

| Selection for this 30B model | Execution and timestamp provenance |
|---|---|
| `--timing auto`, no aligner, speaker source `auto`/`none` | VAD constructs bounded audio windows; text-only ASR per window. Export uses **coarse VAD speech regions, not word times**. |
| `--timing auto` or `alignment`, Qwen aligner enabled, source `auto`/`none`/`diarizer` | VAD constructs independent audio windows (default 25 s, maximum 28 s), text-only ASR recognizes each window, then the aligner processes the matched window and text. No native 30B CTC intervals are used to construct crops. |
| `--timing native --speaker-source native` | Explicit experimental opt-in: one full-recording request, native CTC and speaker labels, strict validity checks. Known timing failures can stop export. |
| `--timing native`, speaker source `auto`/`none` | Explicit native CTC timing without native speaker inference. This remains experimental, not the safe automatic route. |

An enabled separate diarizer is selected by `auto`, or explicitly with `--speaker-source diarizer`. It receives the original decoded audio once for the file, not each ASR window. Its labels are assigned to the independently aligned intervals by the existing overlap policy. It does not alter those times, merge its IDs with native IDs, or cause a second full 30B ASR pass. Native labels require explicit `--speaker-source native`; merely omitting `--diar` does **not** enable them. Native speaker selection conflicts with an enabled separate diarizer and with segmented/alignment timing.

Text windows use `execution.text_batch_items=16` to share bounded runtime sessions; native speaker-bearing requests retain `max_batch_items=1`. The window boundaries come from VAD, with existing overlap/ownership handling, not from corrupted native CTC. A long recording is **not** sent as a 30-minute transcript to one forced-aligner request. Segmentation and recognition can still omit or repeat speech, and alignment cannot correct a wrong transcript. The existing diarizer may choose the largest-overlap candidate and emit a warning rather than always abstaining on overlap; review ambiguous assignments.

## Glossary and Japanese readings

Use **Settings → Glossary** to paste or import UTF-8 TXT, or `--terms-file terms.txt`. Each line is a surface term, optionally followed by a literal TAB and a kana reading. No CSV or JSON is needed. Original terms belong to the run/project data, never to the model TOML.

The versioned `[capabilities.glossary]` declaration selects the code-owned `nemotron_instruct_v1` adapter. It preserves the Japanese verbatim-transcription instruction, including speaker-tag instructions only for explicit native-speaker mode, and adds the bounded glossary through native `instruct`. Readings are **experimental natural-language hints**, not a phonetic dictionary; unrelated hints can worsen recognition or introduce words absent from the audio. Limits and conflicting prompt settings are checked before acquisition, without silently dropping terms/readings. See [TXT format, capability schema, supported models and limitations](glossary.md).

## Strict native diagnostics

The native route consumes `nemotron_transcribe.result.v1` from structured `--batch-manifest-out` JSON, not stdout parsing. Separate text/word/turn files remain diagnostics. Generic `--words-out` has no speaker IDs; generic activity turns have no transcript text. Custom words contain seconds, acoustic speaker columns and separate transcript tags. `spk:N` is **not** necessarily acoustic column N. Japanese units can be characters, not whitespace-delimited words. Raw activity and custom output remain available alongside normalized units.

The adapter fails closed on sparse/failed/partial alignment, zero-width or out-of-bounds intervals, and truncated token/repetition termination. It does not drop invalid characters, invent epsilon durations, or relabel independent chunks. `aligned` from the runtime does not itself prove valid edit boundaries. Text-only independent timing bypasses CTC parsing, but still validates the native completion contract and retains raw evidence; a partial transcript is not silently exported as complete.

The upstream one-hour input cap is an acceptance limit, not a 30-minute benchmark. Native speaker identities reset across independent requests. Native mode therefore retains one full-recording request; the default segmented mode disables native speakers and uses a separate whole-file diarizer when labels are needed.

## Running after native rebuild and deliberate model installation

The install command downloads about 22.04 GB. The aligner and optional standalone diarizer require their own model files.

```sh
python -m asr2rpp.cli models install nemotron-transcribe-iq4-nl qwen-forced-aligner
python -m asr2rpp.cli run recording.wav --asr nemotron-transcribe-iq4-nl --asr-device vulkan --align qwen-forced-aligner --align-device vulkan --align-language Japanese --timing alignment --speaker-source none --terms-file terms.txt --format rpp,json
```

For separately estimated speakers, add `--diar nemotron-diarization --diar-device vulkan --speaker-source diarizer` in place of `--speaker-source none`. For deliberate native diagnostics only, omit the aligner and use `--timing native --speaker-source native`. That path may stop on known CTC defects; it is not the default recommendation.

Use `--asr-exe PATH` / `--align-exe PATH` / `--diar-exe PATH` for trusted separately built v0.9.5 runtimes. Linux CUDA builds can use `cuda`; standard Windows packages use Vulkan. Default `auto` without an aligner is a coarse VAD export, not an implicit Qwen download or hidden forced-alignment step.

## Validation scope and release gate

Earlier real RTX 4080/CUDA tests on a natural Japanese 60.2-second two-speaker recording found a zero-width native CTC character at 15.590 seconds; a separate repeated-speaker fixture exposed cross-turn timing drift. This is why native timing/speakers are experimental and are no longer the automatic 30B route. A speech.cpp runtime-package comparison also retained the zero-width failure and is not the runtime used by this definition.

The independent route was then exercised on that same 60.2-second natural dialogue on an RTX 4080 under WSL/CUDA, with no hints, surface hints, and surface-plus-kana hints. Each run completed 16 bounded text requests with native timestamps/speakers disabled, Qwen alignment, and one standalone whole-file diarization request. All 92/93/90 final fine intervals were positive and in bounds; JSON, RPP and OTIO agreed. The earlier zero-width example instead aligned to 15.580–15.900 seconds, against a human utterance onset of 15.570 seconds. This is evidence for the independent path, not a repair of native CTC.

[Source-checkout measurements and provenance](validation/nemotron-glossary-source-20261115.json) distinguish these runs from release acceptance. Sampled whole-device GPU memory peaked near 6,490 MiB; maximum process RSS was about 16 GiB. Outer wall times were 322/229/237 seconds, including substantial file hashing. The hinted ASR requests were actually slower than the no-hint requests; cache/hash differences explain the misleading outer-time comparison.

**No glossary accuracy improvement is established.** All five target terms were already recognized without hints. Both hinted cases introduced an extra target wording where the reference uses different wording, while neither emitted the two distractors. Readings showed no advantage over surface hints here. The rough normalized text edit ratios were 24.3%/22.2%/22.5%, but overlapping speakers and orthographic variation make these sanity comparisons, not benchmark CER. Speaker agreement was correct on all 60/63/58 unambiguous eligible units after ID permutation; 32/30/32 other units were excluded, and 24/23/25 ambiguity warnings remained. Review overlap rather than interpreting this as perfect diarization.

This short source test does **not** establish 30-minute stability, rare-name recognition, general reading effectiveness, full overlap accuracy, or Windows/Vulkan package behavior. The audio and full transcripts stay local; only aggregate evidence and licensed fixture provenance are included. Earlier long tests of a different ASR pipeline do not validate this new 30B path.

Release additionally requires the repository's Windows and macOS packaging checks, exact package/source/manifest hashes and release approval evidence. Adding the TOML or passing source tests does not itself publish or validate a frozen package. No long-recording quality or package-completion claim is made here.

Sources: [versioned upstream documentation](https://github.com/0xShug0/audio.cpp/blob/v0.9.5/docs/models/nemotron_transcribe.md), [NVIDIA model](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B), [upstream validation record](https://github.com/0xShug0/audio.cpp/blob/v0.9.5/docs/reports/nemotron_transcribe_validation.md).
