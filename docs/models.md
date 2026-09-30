# 同梱モデル / Model catalog

[使い始め方](../README.ja.md) · [English](../README.md) · [カスタマイズ](custom-models.ja.md)

**20定義：whisper.cpp 7、audio.cppのASR 9と補助工程3、sherpa-onnxのReazonSpeech 1。**
All 20 definitions are bundled; weights are downloaded only when selected. The table is not a quality ranking.

## 一覧 / Catalog

取得サイズはモデルファイルの概数（十進MB/GB）です。**必要RAM/VRAMや一時ディスク量ではありません。** 量子化形式・入力長・実行環境によって実行時メモリは変わります。
Download size is not a memory requirement. VibeVoice alone downloads about 9.86 GB; plan additional memory and disk space.

| モデルID / TOML | エンジン・用途 | 重みの取得サイズ | 言語・特徴 | 今回の動作確認 |
|---|---|---:|---|---|
| [`whisper-tiny`](../models/whisper-tiny.toml) | `whisper_cpp` · `asr` | 77.7 MB | 多言語 / multilingual · 導入確認向け | RTX 4080 / Vulkan |
| [`whisper-base`](../models/whisper-base.toml) | `whisper_cpp` · `asr` | 148.0 MB | 多言語 / multilingual · 最初の確認に | RTX 4080 / Vulkan |
| [`whisper-small`](../models/whisper-small.toml) | `whisper_cpp` · `asr` | 487.6 MB | 多言語 / multilingual | RTX 4080 / Vulkan |
| [`whisper-medium-q5`](../models/whisper-medium-q5.toml) | `whisper_cpp` · `asr` | 539.2 MB | 多言語 / multilingual · Q5量子化 | RTX 4080 / Vulkan |
| [`whisper-large-v3-q5`](../models/whisper-large-v3-q5.toml) | `whisper_cpp` · `asr` | 1.08 GB | 多言語 / multilingual · Q5量子化 | RTX 4080 / Vulkan |
| [`whisper-large-v3-turbo-q5`](../models/whisper-large-v3-turbo-q5.toml) | `whisper_cpp` · `asr` | 574.0 MB | 多言語 / multilingual · Turbo、Q5量子化 | RTX 4080 / Vulkan |
| [`anime-whisper`](../models/anime-whisper.toml) | `whisper_cpp` · `asr` | 1.52 GB | 日本語 / Japanese · 実験的なGGML変換、初期プロンプト非対応 | RTX 4080 / Vulkan |
| [`qwen3-asr-06b`](../models/qwen3-asr-06b.toml) | `audio_cpp` · `asr` | 1.15 GB | 日本語を含む多言語 / multilingual | RTX 4080 / Vulkan |
| [`qwen3-asr-17b`](../models/qwen3-asr-17b.toml) | `audio_cpp` · `asr` | 2.47 GB | 日本語を含む多言語 / multilingual · 0.6Bより大きい | RTX 4080 / Vulkan |
| [`sensevoice-small`](../models/sensevoice-small.toml) | `audio_cpp` · `asr` | 254.2 MB | ja/zh/en/yue/ko等 · ITN、タグの保持を変更可能 | RTX 4080 / Vulkan |
| [`fun-asr-nano`](../models/fun-asr-nano.toml) | `audio_cpp` · `asr` | 1.05 GB | 日本語・中国語・英語 / Japanese, Chinese, English | RTX 4080 / Vulkan |
| [`nemotron-asr`](../models/nemotron-asr.toml) | `audio_cpp` · `asr` | 930.6 MB | 多言語 / multilingual · 時刻は出力フレーム由来 | RTX 4080 / Vulkan |
| [`canary-180m-flash`](../models/canary-180m-flash.toml) | `audio_cpp` · `asr` | 249.5 MB | en/de/es/fr · 日本語非対応 / not Japanese | RTX 4080 / Vulkan |
| [`moonshine-streaming-tiny`](../models/moonshine-streaming-tiny.toml) | `audio_cpp` · `asr` | 60.4 MB | 英語専用 / English only · 言語指定を渡さない | RTX 4080 / Vulkan |
| [`voxtral-mini-realtime-q4`](../models/voxtral-mini-realtime-q4.toml) | `audio_cpp` · `asr` | 3.10 GB | 多言語 / multilingual · 今回は英語で検証 | RTX 4080 / Vulkan |
| [`vibevoice-asr`](../models/vibevoice-asr.toml) | `audio_cpp` · `asr` | 9.86 GB | 多言語 / multilingual · ネイティブの時刻と話者 | RTX 4080 / Vulkan |
| [`reazonspeech-k2`](../models/reazonspeech-k2.toml) | `sherpa_onnx` · `asr` | 160.4 MB / 4 files | 日本語 / Japanese · 別エンジンのCPU比較用 | CPU |
| [`qwen-forced-aligner`](../models/qwen-forced-aligner.toml) | `audio_cpp` · `align` | 1.13 GB | テキストと音声を整列 / alignment | RTX 4080 / Vulkan |
| [`nemotron-diarization`](../models/nemotron-diarization.toml) | `audio_cpp` · `diar` | 198.7 MB | 話者区間 / speaker diarization | RTX 4080 / Vulkan |
| [`mel-big-beta7`](../models/mel-big-beta7.toml) | `audio_cpp` · `sep` | 約945 MB + 変換領域 | 音声・背景音の分離 / separation · 初回に自動変換 | RTX 4080 / Vulkan |

## 確認したこと・していないこと / Validation scope

2026-09-30 UTC、WindowsのRTX 4080でDesktop Commander経由の実行を確認しました。日本語の合成音声と公開英語音声を使い、全モデルの取得・ハッシュ検証からアプリを通したRPP作成までを検証しています。音声を含む個人の資料は公開していません。

19個のネイティブモデルはVulkanを明示し、ReazonSpeechはCPUで確認しました。元音源の不変性、空でない認識結果、元音源内に収まる時刻、VAD由来時刻の正しい表示を検査しました。補助工程は分離済みWAV・強制アライメント・話者ラベル、VibeVoiceはネイティブ話者ラベルも確認しました。

**これは短い音声の機能試験です。** 認識精度の順位、長尺での安定性、同時発話の話者精度、すべての言語、すべてのGPU、全モデルのMac Metal動作を保証するものではありません。ソース＋ネイティブ実行環境の実機検証と、CIでの凍結アプリ検証を区別しています。モデル別の記録は[検証JSON](validation/native-catalog-20260930.json)を参照してください。

Functional tests used source code with native binaries and an isolated CPU worker, not a claim that all 20 models were exercised on every frozen OS package. CI tests packaged application startup and selected native/optional-stage paths separately. Mac Metal validation depends on the capabilities exposed by the CI device; the virtual GPU cannot exercise every Qwen encoder operation.

今回の検証で、Moonshineが不要なlanguage引数を拒否する問題、VibeVoiceのsegments出力に話者ラベルが含まれない問題、Whisperのログ抑止で実GPUを追跡できない問題を修正しました。モデル固有の入出力指定はTOMLへ置き、同じ仕組みで追加モデルにも適用できます。

## 選び方・時刻 / Choosing a model and timing

最初は小さいモデルで入出力を確認し、手持ちの音声で本文・時刻を比較してください。大きいモデルが常に用途に適するとは限りません。VibeVoiceは特に重く、16GB GPUで今回の短い音声が動いたことから長尺の使用量は推測できません。

Whisperは通常ネイティブ区間、Nemotronは出力フレーム推定、VibeVoiceはネイティブ区間・話者を返します。Anime Whisper、Qwen3-ASR、SenseVoice、Fun-ASR、Canary、Moonshine、Voxtral、ReazonSpeechは、この構成ではVAD発話区間か強制アライメントを使います。設定 → 時刻付与で変更します。VAD時刻を単語境界として扱わないでください。

Moonshine/Realtime/Streamingというモデル名でも、ASR2RPPは録音済みファイルを処理します。このアプリにマイク入力のリアルタイムUIが付くわけではありません。

## 取得元とモデルの条件 / Publishers and terms

各TOMLに固定revisionとファイルSHA-256を記載しています。ファイルの一致と、モデルの利用許諾・精度は別です。元モデルと変換元の両方のモデルカードを確認してください。すべてのモデルにアプリのMITライセンスが適用されるわけではありません。

- Whisper: [original model](https://github.com/openai/whisper) / [GGML publisher](https://huggingface.co/ggerganov/whisper.cpp)。Anime Whisper: [GGML conversion and source references](https://huggingface.co/Aratako/anime-whisper-ggml)。
- audio.cpp GGUFs: [aggregate packages](https://huggingface.co/audio-cpp/audio.cpp-gguf) / [upstream model docs](https://github.com/0xShug0/audio.cpp/tree/9bdd1d908bbd128e9eb405f5a8e38d0defb84c72/docs/models)。SenseVoice: [compatible GGUF](https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF-audiocpp)。
- [Nemotron diarization](https://huggingface.co/audio-cpp/Nemotron-3-Diarization-GGUF) / [Mel-Band RoFormer checkpoint](https://huggingface.co/pcunwa/Mel-Band-Roformer-big) / [ReazonSpeech K2](https://huggingface.co/reazon-research/reazonspeech-k2-v2)。

## 他のモデル / Other models

この一覧は代表例であり、audio.cppの全TTS・音楽生成・変換機能を同梱したものではありません。対応済みfamilyの別チェックポイントは独自TOMLで追加できます。まだ実行環境にないfamilyは、新しいネイティブビルドや対応ワーカーが必要です。[追加方法](provider-models.md)を参照してください。
