# Anime Whisper conversion trial — 2026-09-25

Tested code: `155382ccdd789a087ad18adc980c7f677468f82d`.
Workflow: https://github.com/zkamiyama/ASR2RPP/actions/runs/36103056463
Artifact: `anime-conversion-trial-36103056463`, ID `10850780109`.
Artifact SHA-256: `74990263e217dd52bb09489fad569af0b5e3559a521845306bfaa0c47bb4b7d4`.
The artifact was downloaded and its checksum, summary, package list and script hashes inspected.
Later documentation/launcher commits do not change the four tested Python scripts.

## Environment and conversion

The ten synthetic converter tests passed on Windows 2022 and Ubuntu 24.04, using Python 3.12. The full-model conversion and real native inference ran on Ubuntu CPU, not Windows/Vulkan. PyTorch and Transformers were absent; `packages.txt` contains only `numpy==2.3.5` and `safetensors==0.7.0`.

Original: `litagin/anime-whisper@22e2008a8182b357da3922a6308d095008f72973`.
Original SafeTensors SHA-256: `15c672f0bf687b1c67aa14325f9c382c6919f1fce976f990513618b464a3c626`.

Both conversions contain 539 tensors and passed independent GGML read-back verification.

| Output | Bytes | SHA-256 |
|---|---:|---|
| F16 with source-precision F32 auxiliary tensors | 1,519,521,155 | `9cbe6ff83c8cc136431c17aeeb761bf7c7615e3f3a16fbc2be8b2f80fbc87066` |
| F32, without an intermediate F16 cast | 3,026,260,355 | `454bc4532aa8d868944802e1a8364dd9469a151556eb26af9de9dd1544a77411` |

## Comparison with existing GGML

Reference: `Aratako/anime-whisper-ggml@35b467f144c62a3ab2d84bfbb517d6af5135444a`, file `ggml-anime-whisper.bin`, SHA-256 `bce42c15312fcdbdcc8432b2ce63ba967f4cc3d2cc783fd6ab2ba264caf0db8d`.

Against the new F16 file: headers, mel filters and vocabulary are identical. All 539 tensor names are present in both files. Of these, 216 tensor payloads are identical and 323 differ. All 323 differences are exactly reproduced by applying the legacy F32 -> F16 -> F32 rounding to the original auxiliary tensors. There are no other tensor differences in this comparison. This numerical finding does not establish the cause of anomalous leading text.

## Real inference smoke and limitations

The same application-pinned whisper.cpp (`a664346ea5c6dddff3e61a2b7b32dd4514613f50`) loaded and ran all three models: existing GGML, new F16 and new F32. Both timed (`-ojf`) and text-only (`-oj -nt`) cases completed with exit code zero and output files: six cases in total.

**The fixture is four seconds of public English JFK speech with `-l ja`, intentionally used only to check loading/execution. It is not a Japanese accuracy benchmark or a reproduction of the user's failure.**

All three models produced the same text within each mode. Timed output was `SANANANANAAXAaaaa`; text-only output was `…そう、マイフロイヤルイカス…`. Neither is a correct transcription of the English fixture. Thus this trial does not demonstrate improved recognition from re-conversion. Changing timestamp mode changed the text; attributing the user's issue to timestamps still requires a relevant Japanese audio comparison.

No private media, credentials or model binaries are included in the reports. GPU/Vulkan quality and memory behavior were not measured. Production `main`, application defaults and existing user model files are unchanged. The converter is offered for an isolated comparison, not as a confirmed fix.
