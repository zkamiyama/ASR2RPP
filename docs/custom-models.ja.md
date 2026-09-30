# モデルを追加・カスタマイズする

[使い始め方](../README.ja.md) · [モデル一覧](models.md) · [English technical guide](provider-models.md)

## まず設定画面でできること

時刻の付け方は「設定 → 時刻付与」、言語・実行環境は各工程、認識パラメーターはモデル選択の横の調整ボタンで変更します。
モデル取得先やモデル構造を変えない限り、TOMLの編集は必要ありません。

## 同じモデル構造の重みを追加する

「設定 → 詳細 → 独自TOMLを開く」で開いた `custom-models` に、UTF-8で `my-whisper.toml` を保存します。
ファイル名から拡張子を除いたものがモデルIDです。組み込みや別の独自TOMLと重複させないでください。
以下のパスは、自分の手元にある **whisper.cpp対応のGGML重み**に置き換えます。

```toml
schema_version = 2
provider = "whisper_cpp"
task = "asr"
name = "My Whisper"
sample_rate = 16000

[source]
path = "D:/ASR/models/my-whisper.bin"

[capabilities]
timestamps = "segment"
speakers = false

[defaults]
language = "ja"
```

Macでは `/Users/yourname/ASR/models/my-whisper.bin` のようなパスにします。Windowsは `/` 区切りにするとバックスラッシュのエスケープを避けられます。
相対パスはTOMLのあるフォルダーを基準にします。取得済みファイルを指定したローカル定義では、ネット上の別モデルを自動取得しません。
時刻を返さない派生モデルなら `timestamps = "none"` とし、VADか強制アライメントを設定画面で選びます。

## audio.cppの時刻なしASRを追加する

これは **Qwen3-ASR対応GGUF**を使う例です。別構造のモデルを `family` だけ変更して動かすことはできません。

```toml
schema_version = 2
provider = "audio_cpp"
family = "qwen3_asr"
task = "asr"
name = "My Qwen ASR"
sample_rate = 16000

[source]
path = "D:/ASR/models/my-qwen-asr.gguf"

[capabilities]
timestamps = "none"
speakers = false

[execution]
mode = "offline"
output = "text"
max_batch_items = 16
max_audio_seconds = 28

[defaults]
language = "Japanese"
```

`family`・入力サンプルレート・出力形式は使用する実行環境の契約と一致させます。
`max_batch_items` は一つのセッションで受け付ける件数上限で、GPUが同時に計算するバッチサイズとは限りません。
Moonshineのように言語指定そのものを受け付けない実装には、`[execution]` の `pass_language = false` を使います。画面の言語欄はその実装には渡されず、モデルの固定言語で動きます。
VibeVoiceの話者情報を使うには `speakers = true` と `output = "turns"` の組み合わせが必要です。

## 取得元を指定する・複数ファイルを使う

ローカルの `source.path` の代わりに、組み込みTOMLと同じ `source.repo`、固定した40桁の `source.revision`、`source.files`、`source.sha256` を記載できます。
ハッシュは**ファイルのSHA-256**です。Gitのblob SHAとは異なります。
複数ファイル型は `source.entry = "."` とし、`[artifacts]` で `encoder`、`decoder`、`joiner`、`tokens` などの役割をファイル名に対応させます。
実例は組み込みの `reazonspeech-k2.toml` です。ファイルはモデルフォルダー内に置き、`../` で外を参照しないでください。

## パラメーターを画面へ追加する

同じTOMLへ次のような宣言を追加すると、パラメーター画面に型・範囲付きの項目を出せます。
以下は `enable_itn` を受け付けるSenseVoice/Fun-ASR用の例で、すべてのモデル共通ではありません。

```toml
[defaults.request]
enable_itn = true

[parameters.enable_itn]
type = "bool"
default = true
ja = "数値などの表記を正規化"
en = "Inverse text normalization"
```

`type` は `bool`、`int`、`float`、`str`、`enum`。数値には `min`/`max`、選択肢には `values` を指定できます。
通常の推論オプションは `[defaults.request]`、モデル読み込み時のセッション設定は `[defaults.session]` です。
セッション設定のUIキーは `[parameters."session.option_name"]` のように書きます。
**実行エンジンにないパラメーターをTOMLで追加しても、その機能が実装されるわけではありません。** 不明な値を黙って捨てず、エラーを確認してください。

## 新しい実行環境が必要な場合

導入済みaudio.cppにない `family`、未知のニューラルネット構造、独自の前処理・デコードには、対応するruntime/providerが必要です。
信頼できる同一プロトコルの実行ファイルを用意し、設定 → 詳細でネイティブ実行ファイルを指定するか、CLIの明示的なruntime登録を使います。操作例は[実行環境ガイド](provider-models.md)にあります。
TOMLはデータ専用で、任意のシェル・Pythonコード・pipインストールを実行できません。実行ファイルを信頼する操作とは分けています。
WindowsのGPUはVulkan、MacはMetalです。CUDA指定を追加してこの方針を回避することはできません。

変更後は再読込、またはGOで検証します。重複ID・未対応のモデル構造・不正な設定は修正してから再実行してください。
まず短い音声で本文・時刻・出力を確認し、長尺へ進めてください。
