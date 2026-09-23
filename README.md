# ASR2RPP 0.1.0-preview

音声・動画を元ファイルへの**非破壊参照のみ**でREAPERプロジェクトへ配置する試作です。
ASRとDiarizationを別々に選択し、外部JSONでモデルを追加できます。

## Windows

ZIPをすべて書き込み可能なフォルダへ展開してください。exeだけを移動しないでください。
`ASR2RPP.exe` はPySide6 GUI、`asr2rpp-cli.exe` はCLIです。モデル・実行先・パラメータはASRと話者推定で独立です。
Python/PyTorchのインストールは不要。CPU版whisper.cppを同梱し、モデル重みは含みません。
最初はWhisper tinyを選択して「このモデルを取得」で動作確認できます。Tinyは日本語の品質基準ではありません。

**MP4・MP3・ステレオや異なるサンプルレートの音声にはFFmpeg/ffprobeを別途用意**し、
JSONのtoolsに実行ファイルのパスを設定してください。16kHzモノラル16-bit PCM WAVはFFmpegなしで処理できます。
FFmpeg取得元: https://ffmpeg.org/download.html

audio.cppとGPU用実行ファイルは同梱していません。対応版を https://github.com/0xShug0/audio.cpp/releases などから取得し、
enginesにパスを設定します。CUDA/Vulkan/Metalのビルドは同一ではありません。
Windows x64プレビュー版。署名・SmartScreen評価・インストーラーは未提供です。

## CLI

```powershell
.\asr2rpp-cli.exe models
.\asr2rpp-cli.exe doctor
.\asr2rpp-cli.exe install whisper-tiny
.\asr2rpp-cli.exe run input.wav --out output-001 --asr whisper-tiny --duration 55
.\asr2rpp-cli.exe run input.mp4 --out output-002 --start 10 --duration 55 --asr anime-whisper --diarization nemotron-diarization --asr-device cpu --diarization-device vulkan
.\asr2rpp-cli.exe compare input.wav --out comparison-001 --asr anime-whisper nemotron-asr vibevoice-asr --duration 15
.\asr2rpp-cli.exe export output-001\transcript.json --out revised.rpp
```

--asr-paramsと--diarization-paramsは個別のJSONオブジェクトです。
--startはデコードした音声ストリームの開始からの秒数です。動画はffprobeの音声/ファイルの時刻原点差を記録・加算します。
無音を詰めず元時刻に配置します。PTS不連続など特殊なメディアの同期は未保証です。出力先は空/新規フォルダにしてください。
VibeVoice等のtype=jointを使う場合は--diarization noneで統合話者を使います。別diarizationの明示選択時はその結果を優先します。
compareは結果と実行時間の比較です。正解なしに認識精度を採点する機能ではありません。

## 外部JSON

models.user.example.jsonをmodels.user.jsonへコピーすると追加/上書き設定になります。
モデルはID単位で追加、同じIDはユーザー定義で全体を置換します。engines/toolsはキー単位でマージします。

```json
{
  "schema_version": 1,
  "models": [{
    "id": "my-whisper",
    "label": "自分のWhisperモデル",
    "type": "asr",
    "engine": "whisper.cpp",
    "family": "whisper",
    "format": "ggml",
    "repository": "https://huggingface.co/OWNER/MODEL",
    "path": "models/my-whisper.bin",
    "sample_rate": 16000,
    "defaults": {"language": "ja", "threads": 4, "beam_size": 5},
    "status": "unverified"
  }]
}
```

ローカルモデルはartifact不要です。自動取得には以下のartifactをモデル項目へ追加します。

```json
"artifact": {"repo_id": "OWNER/CONVERTED-MODEL", "revision": "main", "filename": "compatible-model.bin"}
```

取得時にrevisionを不変commitへ解決し、HubのLFS SHA-256とサイズを検証します。
モデル横の.provenance.jsonに取得commit/hashを保存します。再現実験ではそのcommitをrevisionへ固定してください。
認証モデルの自動取得は未対応。認証済み環境で取得しローカルパスを指定します。
この版の自動取得は単一の自己完結したGGML/GGUFのみです。

- type: asr / diarization / joint
- engine: whisper.cpp / audio.cpp
- family: audio.cppの実際に対応したfamily名
- format: whisper.cpp専用GGML .bin / audio.cpp互換GGUF

URLだけでは形式や対象ファイルは確定しません。同じGGUF拡張子でも他エンジンと無条件互換ではありません。
対応済みfamilyの追加はJSONでできますが、未実装のモデル構造や別CLI/JSONプロトコルにはアダプターが必要です。
JSONから任意のシェルやPythonコードを実行する仕組みはありません。

## モデルの注意

anime-whisperは初期プロンプトを禁止しています。https://huggingface.co/litagin/anime-whisper
取得先はAratakoの第三者GGML変換版で、配布者が認識結果の不具合を報告しています。
**実験的・品質未検証**です。https://huggingface.co/Aratako/anime-whisper-ggml

Nemotron ASRの生トークン時刻と発話の全区間は同じではありません。
Nemotron 3 Diarizationはnemotron_3_diarという新しい8話者モデルで、4話者Sortformerとは別です。
https://github.com/0xShug0/audio.cpp/blob/main/docs/models/nemotron_3_diar.md
VibeVoice-ASRはvibevoice_asrというオフライン統合モデルです。Streaming版へ無条件に置換しないでください。

## 時刻と編集

transcript.jsonを編集しexportで再出力できます。raw_unitsとエンジンの元JSONを保持します。
範囲外時刻のクリップやトークンの発話単位化はreviewに記録します。
複数話者が混ざる長いASR区間は文字数比例で分割せずUNRESOLVEDにします。
**Forced AlignmentとGUI波形編集は未実装**で、タイムスタンプ精度は未保証です。
音源分離は行いません。同じ混合音声を話者数分複製しない方針です。
RPPは元メディアを参照し、推論キャッシュを削除しても元メディアがあれば参照できます。
個別WAV/stream-copy出力は将来、共通編集データを入力とする別Exporterで追加します。

## モジュール

rpp_writer.pyは単独再利用可能で標準ライブラリのみ。registry.pyは定義/取得、engines.pyはCLI/形式変換、
pipeline.pyは前処理と編集データ、common.pyはプロセス管理、main.pyはCLI、gui.pyは薄いUIです。
UIは同じCLIを子プロセスで実行し、ライブラリ環境を分離します。

指定デバイスを引数へ渡しますが、ランタイム内部のCPUフォールバックは完全検知していません。
requested_deviceとeffective_device=not_verifiedを分けて記録します。doctorは存在確認でGPU検証ではありません。

## 開発

```bash
python -m unittest -v
python main.py models
python -m pip install PySide6-Essentials==6.8.3 pyinstaller==6.22.0
python -m PyInstaller --noconfirm app.spec
```

Windows版はWindows上でビルド。CIのモデル試験はwhisper.cpp同梱の公開音声だけを使い、ユーザー動画は送信しません。
モックテストと実モデル試験、GPU試験、精度試験は区別します。
アプリはMIT。Qt/whisper.cpp/モデル等には各資材のライセンスが適用されます。
