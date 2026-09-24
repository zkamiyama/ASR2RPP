# ASR2RPP

音声・動画を文字起こしして、元メディアを**非破壊参照**するREAPERプロジェクトへ変換します。
CLIとPySide6 GUIは同じパイプラインを使います。モデル推論はwhisper.cpp / audio.cppの独立プロセスです。

## Windowsプレビュー

ZIPをすべて展開し、`ASR2RPP/ASR2RPP.exe`を起動します。exeだけを移動しないでください。
署名のない開発プレビューです。whisper.cpp / audio.cpp はCPU版とVulkan版を同梱し、FFmpegも含めるためPythonの手動導入は不要です。
モデル重みは同梱しません。モデルの説明・配布条件を確認してから**選択モデルを準備**を押してください。変換が必要なモデルは取得後にユーザースペースで自動変換します。
まずWhisper Base / cpu、Diarization OFF、Forced Alignment OFFで短い素材を試してください。
Baseは導入確認用で、認識精度を保証するモデル選定ではありません。

1. ファイル追加・フォルダー直下からの追加・ドラッグ＆ドロップでキューへ登録。
2. 右上の**⚙ 設定**でwhisper.cpp / audio.cppの既定実行先（CPU / Vulkan等）を選択。各ASR・Diarization・Forced Alignment欄では「設定に従う」が既定で、必要な処理だけ個別上書きできます。DiarizationとForced Alignmentは独立してON/OFFできます。
3. `Same directory`がONなら各入力の隣へ保存。OFFなら`Output directory`を指定。
4. **キューを実行**。複数ファイル時は既定で **SEP → ASR → Forced Alignment → Diarization → RPP** のステージ優先で処理します。同じモデルをまとめて使用し、別モデルのプロセスは次のステージ開始前に終了するため、複数の大型モデルを同時常駐させません。
5. 停止で現在の処理を終了。未処理は待機のまま保持。失敗・中断は再キューできます。

audio.cppは1つのoffline sessionで複数WAVを処理し、whisper.cppも1つのモデルcontextで複数入力を処理します。巨大キューを無制限に1バッチへ入れず、ギア設定の **audio.cpp batch RAM目安**（既定512 MB）で入力WAV群を分割します。これは入力ファイルサイズを基準とした安全側の目安で、厳密なピークRAM/VRAM制限ではありません。1ファイル自体が大きい場合は1バッチを超えることがあります。
互換用にギア設定から「ファイル優先 — 1ファイルずつ完走」へ戻せます。

実行中は設定とキューの変更をロックします。再実行時に設定変更が意図せず混ざることを防ぎます。
実行済みの素材を別設定で再変換する場合は、その行を削除して追加し直します。
キューの永続化や行ドラッグによる並べ替えはこの版にはありません。

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
背景音除去をONにした場合だけ、設定したsource-separationモデルを推論前に実行します。RPP参照元は元メディア／保存した処理済みWAVから選択できます。

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

GUIの**モデル定義を開く**で有効な定義フォルダーを開けます。ギア設定には実際のモデル重み保存先も表示します。

標準データルート：
- Windows: `%LOCALAPPDATA%\ASR2RPP`
- macOS: `~/Library/Application Support/ASR2RPP/`
- Linux: `$XDG_DATA_HOME/asr2rpp/`、未設定なら `~/.local/share/asr2rpp/`
- `ASR2RPP_HOME`環境変数があれば全OSでその場所を優先します。

データルート直下は `models/`（1モデル1TOML）、`weights/`（取得・変換済み重み）、`cache/`（キュー中の一時処理）に分けます。重みは例えば `weights/mel-big-beta7/<source定義hash>/big_beta7-f16.gguf` に保存されます。
変換モデルの元CKPTは、変換成功後は既定で削除します。変換失敗時は再試行用に残します。ギアの高度設定 **変換成功後も元チェックポイントを保持する** をONにした場合だけ成功後も保持します。小さな設定ファイル、最終GGUF、取得・変換履歴は残します。

初回に標準TOMLテンプレートをコピーします。既存のユーザー定義は更新時にも上書きしません。
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
