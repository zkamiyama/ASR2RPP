# ASR2RPP

音声・動画を文字起こしして、元メディアを**非破壊参照**するREAPERプロジェクトへ変換します。
CLIとPySide6 GUIは同じパイプラインを使います。モデル推論はwhisper.cpp / audio.cppの独立プロセスです。

## Windowsプレビュー

ZIPをすべて展開し、`ASR2RPP/ASR2RPP.exe`を起動します。exeだけを移動しないでください。
署名のない開発プレビューです。whisper.cpp / audio.cpp はCPU版とVulkan版を同梱し、FFmpegも含めるためPythonの手動導入は不要です。
モデル重みは同梱しません。モデルの説明・配布条件を確認してから**選択モデルを取得**を押してください。
まずWhisper Base / cpu、Diarization OFF、Forced Alignment OFFで短い素材を試してください。
Baseは導入確認用で、認識精度を保証するモデル選定ではありません。

1. ファイル追加・フォルダー直下からの追加・ドラッグ＆ドロップでキューへ登録。
2. 右上の**⚙ 設定**でwhisper.cpp / audio.cppの既定実行先（CPU / Vulkan等）を選択。各ASR・Diarization・Forced Alignment欄では「設定に従う」が既定で、必要な処理だけ個別上書きできます。DiarizationとForced Alignmentは独立してON/OFFできます。
3. `Same directory`がONなら各入力の隣へ保存。OFFなら`Output directory`を指定。
4. **キューを実行**。１ファイルずつ処理し、失敗しても次のファイルへ進みます。
5. 停止で現在の処理を終了。未処理は待機のまま保持。失敗・中断は再キューできます。

実行中は設定とキューの変更をロックします。再実行時に設定変更が意図せず混ざることを防ぎます。
実行済みの素材を別設定で再変換する場合は、その行を削除して追加し直します。
キュー順序は追加順です。キューの永続化や行ドラッグによる並べ替えはこの版にはありません。

## 出力

`入力名.rpp` と `入力名.asr2rpp/`（JSON、ログ、取得条件）を出力します。
既存ファイル・フォルダーがある場合は`_2`等を付けて保護します。上書きしません。
RPPは元メディアを参照するため、元メディアを削除・移動しないでください。
同じドライブでは相対パス、別ドライブでは絶対パスを使用します。
発話ごとのWAVは生成せず、推論用PCMキャッシュは処理終了時に削除します。
`.asr2rpp`内の文字起こし・ログは私的情報を含み得るので、公開リポジトリに追加しないでください。

Diarization OFFでは、統合ASRが話者ラベルを返しても単一の`Transcript`トラックへ出力します。
ONでは外部Diarizationを実行し、重複時間が最大の話者へ対応付けます。曖昧な箇所は警告を残します。
粗いASRセグメントの文章を文字数比例で話者に分割することはありません。
この版のセグメント内話者交代処理は保守的です。精度評価・手修正UIは今後の対象です。
音源分離は行いません。

## CLI

```sh
python -m asr2rpp.cli models list
python -m asr2rpp.cli models install whisper-base
python -m asr2rpp.cli run a.wav b.mp4 --asr whisper-base --output-dir output
python -m asr2rpp.cli run a.mp4 --asr nemotron-asr --diar nemotron-diarization --align qwen-forced-aligner --start 10 --duration 55
```

Windowsパッケージでは`python -m asr2rpp.cli`を`asr2rpp-cli.exe`に置き換えます。
`--asr-device` / `--diar-device` / `--align-device`は独立しています。
`--asr-exe` / `--diar-exe` / `--align-exe` / `--ffmpeg`で外部実行ファイルを指定できます。
任意のシェル引数文字列は受け付けません。パラメーターは型を検査して引数配列で渡します。

## モデル定義：１モデル１TOML

GUIの**モデル定義を開く**で有効な定義フォルダーを開けます。
Windows: `%LOCALAPPDATA%/ASR2RPP/models`。`ASR2RPP_HOME`でデータ保存先を変更可能です。
初回に標準テンプレートをコピーします。既存のユーザー定義は更新時にも上書きしません。
変更後に**再読込**してください。不正なファイルだけをエラーにし、他のモデルは使用できます。

```toml
runtime = "whisper_cpp"
task = "asr"
[source]
repo = "https://huggingface.co/ggerganov/whisper.cpp"
files = ["ggml-base.bin"]
[defaults]
language = "ja"
```

ローカルのモデルは`[source] path = 'D:\Models\model.bin'`で指定します。
audio.cppは`runtime="audio_cpp"`、`family`、`task="asr"/"diar"/"align"`を指定します。
`[defaults.request]`と`[defaults.session]`はaudio.cppのパラメーターです。
エンジン・出力方式が未対応の新しい系列にはアダプター側の追加対応が必要です。
任意のモデルがTOMLだけで動くという意味ではありません。

## GPUと時刻の注意

Windows x64プレビューはwhisper.cpp / audio.cppのCPU版とVulkan版を別ディレクトリで同梱します。
Vulkanを使用するにはVulkan対応GPUと正常なベンダードライバーが必要です。GitHub ActionsではVulkan版のビルドと起動可能性（--help）を確認しますが、GPU推論性能・演算の完全なVulkan配置まではGPUなしのHosted Runnerでは検証しません。
Metal / CUDAは外部対応ビルドを設定できる構造です。DirectMLプロバイダーは未実装です。
要求デバイスとエンジンログを記録しますが、実際の全演算の実行先監査は未実装です。
CPUへの暗黙フォールバックを検出できないエンジンがあるため、性能比較時にはログ確認が必要です。

Whisperはセグメント時刻を編集単位として使用し、full JSONも保持します。
Nemotron ASRのトークン時刻は発音の厳密な開始・終了ではありません。
Forced AlignmentはASRテキストと対応音声を再処理します。初期版では55秒以下の対応区間に限定しています。
秒・ミリ秒を正規化し、範囲外時刻の編集用クリップは履歴に記録します。生出力は変更しません。
日本語・複数話者での精度比較は動作確認とは別です。
Anime WhisperのGGML変換版は配布者による認識異常の注意がある実験的選択肢です。
VibeVoiceは大型モデルです。素材を短縮してもモデル重みのメモリは減りません。

## 開発・ビルド

```sh
python -m pip install -e '.[gui]' pytest
python -m pytest tests/test_app.py
python launcher.py
```

`rpp_writer.py`は標準ライブラリのみの独立モジュールです。ASR・GUI・FFmpegへ依存しません。
`asr2rpp/pipeline.py`と`adapters.py`を使わず、別アプリから直接利用できます。

GitHub Actionsは通常のCPUランナーを使います。`[build]`を含むコミットでWindows版とLinux検証キットをビルドします。
それ以外のコード変更では軽量テストとGUI状態テストを実行します。
実際のユーザー動画や認証情報はCIへ送信・公開しません。
配布物・元の依存ライセンスは`THIRD_PARTY.md`と各エンジンの同梱ライセンスを参照してください。
