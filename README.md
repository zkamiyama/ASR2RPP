# ASR2RPP

音声・動画を文字起こしして、音源を非破壊参照するREAPERプロジェクトを作ります。
推論はwhisper.cpp / audio.cppの別プロセスで実行し、Python版PyTorchは使用しません。

## Windows版

ZIPをすべて展開し、`ASR2RPP/ASR2RPP.exe`を起動します。exeだけを移動しないでください。
CPU版・Vulkan版の推論エンジンを同梱しています。Pythonの手動導入は不要です。
現状は署名のないビルドです。モデル重みとFFmpegは同梱していません。

FFmpegは指定パス、PATH、取得済みのユーザー用ランタイムから探します。Windowsで見つからなければ、
BtbNのLGPL sharedビルドを取得元のSHA-256一覧で照合してからユーザーデータ領域へ配置します。
毎回最新版に更新する方式ではありません。ライセンス文書と取得元も保存します。
モデルはGO時、または設定の「選択モデルを準備」で取得します。配布元ごとの利用条件を確認してください。

初回のUI言語は日本語環境なら日本語、それ以外は英語です。手動で選んだ言語は保存します。
ファイルをキューへドロップし、背景音除去・音声認識・強制アライメント・話者ダイアライゼーションを設定してGOを押します。
右下の設定ボタンから実行環境、保存先、処理順を変更できます。各工程のCPU/Vulkanは独立です。
実行ログはウィンドウ内で選択・コピーできます。処理中の設定変更はロックします。

## RPP出力

**最上段は `ORIGINAL` トラックです。切り分け前の参照音源を全長1アイテムで配置します。**
発話前後の無音も含み、認識された最後の発話で切り詰めません。
下段の編集トラックとの二重再生を避けるため、ORIGINALだけは初期状態でミュートです。
比較再生時は編集トラックをミュートするかORIGINALをソロにし、ORIGINALのミュートを解除してください。

参照元が「元メディア」ならORIGINALも発話アイテムも元ファイルを参照します。
「処理済み」ならRPPの隣に保存した`*_vocals.wav`を参照し、そのWAV全体がORIGINALになります。
CLIで範囲を指定した場合も、元メディアのORIGINALは時刻0から全長を保持します。
処理済みWAVが範囲抽出後の音声なら、そのアイテムは元のタイムライン上の開始位置に置き、ファイル内オフセットは0です。

話者分離OFFでは下段は`Transcript`、ONでは話者別トラックになります。
元ファイルは書き換えません。発話ごとの音声ファイルは保存せず、推論用キャッシュだけを作り、終了時に削除します。
`入力名.rpp`と`入力名.asr2rpp/`を作り、既存の出力があれば番号を付けます。上書きしません。
同一ドライブでは相対パス、別ドライブでは絶対パスを使います。参照音源を移動・削除しないでください。
診断フォルダーには文字起こしや私的情報が含まれるため、公開リポジトリへ追加しないでください。

## 処理と性能

工程順は背景音除去 → 音声認識 → 強制アライメント → 話者割当 → RPPです。
別モデルのプロセスは工程の切り替え時に終了し、複数の大型モデルを同時常駐させません。
複数ファイルでは既定でステージ優先です。設定からファイル優先にも変更できます。

Nemotron ASRは長尺音声の巨大なofflineグラフを避けるため、各ファイルをstreamingモードで処理します。
強制アライメントは、デコード済みPCMから対象区間を切り出し、RAM上限を設けたrequest-sequenceでモデルを再利用します。
各区間ごとに長い動画を先頭からデコードしたり、モデルを再ロードしたりしません。
単一ファイルとキューは同じアライメント実装・同じRPP出力実装を利用します。
バッチ容量は入力WAVサイズに対する上限であり、推論中のVRAM使用量そのものの保証ではありません。

ASRトークン時刻は厳密な発音境界とは限りません。強制アライメントは55秒以下の対応する音声・文章区間に限定します。
句読点だけの区間はASR時刻を保持して警告を記録し、架空の単語時刻を生成しません。
話者は時間重複から割り当て、曖昧な区間はUNKNOWNにします。文章を文字数比例で話者へ分割しません。
日本語や複数話者での認識精度は動作確認とは別です。

## CLI

```sh
python -m asr2rpp.cli models list
python -m asr2rpp.cli models install whisper-base
python -m asr2rpp.cli run a.wav b.mp4 --asr whisper-base --output-dir output
python -m asr2rpp.cli run a.mp4 --asr nemotron-asr --diar nemotron-diarization --align qwen-forced-aligner
```

Windows ZIPでは`python -m asr2rpp.cli`を`asr2rpp-cli.exe`に置き換えます。
`--asr-device` / `--diar-device` / `--align-device` / `--preprocess-device`はCPUまたはVulkanです。
`--preprocess mel-big-beta7 --rpp-audio processed`で処理済み音声を参照できます。
`--start`と`--duration`は推論範囲を指定します。`--ffmpeg`や各`--*-exe`で実行ファイルを指定できます。

## 設定とモデル

データルートはWindowsでは`%LOCALAPPDATA%\ASR2RPP`、macOSでは`~/Library/Application Support/ASR2RPP`、
Linuxでは`$XDG_DATA_HOME/asr2rpp`（未設定なら`~/.local/share/asr2rpp`）です。
`ASR2RPP_HOME`があればそれを優先します。
`weights/`に重み、`cache/`に推論用一時ファイル、`runtime/ffmpeg/`に取得したFFmpegを置きます。
同梱TOMLは展開したアプリの `models/` をGUIとCLIが直接読みます。一時フォルダーやユーザーフォルダーへの自動コピーはしません。
独自TOMLはデータルートの `custom-models/` に置き、同梱モデルと異なるファイル名（モデルID）を使ってください。
重みと一時保存先はGUI設定または`ASR2RPP_WEIGHTS_DIR` / `ASR2RPP_CACHE_DIR`で変更できます。

旧データルート `models/` の未編集同梱コピーは読み飛ばし、独自IDのモデルは旧場所から直接読み続けます。
同梱と同名の編集済みTOMLは自動上書きせず、競合と実ファイルの場所を表示します。独自IDに変更して利用してください。
GO直前と「モデル定義を再読込」で原本を読み直し、実行中は開始時の定義スナップショットを使用します。
ログはモデルID、ツールチップと `models list --json` は読込元のパス、診断manifestはTOMLのSHA-256も記録します。配布モデルと異なる独自モデルにはアダプターの追加実装が必要な場合があります。
Mel-Band RoFormerの元CKPTは検証・変換成功後に既定で削除します。失敗時は再試行用に残します。
設定の「変換成功後も元チェックポイントを保持」をONにすると成功後も残します。

## 開発・検証

```sh
python -m pip install -e '.[gui]' pytest
python -m pytest tests
python launcher.py
```

`rpp_writer.py`は標準ライブラリだけに依存する独立したシリアライザーです。
`rpp_export.py`に参照トラックの構築、`media.py`に音声区間操作、`alignment.py`にアライメント、
`transcript.py`に時刻正規化と話者割当をまとめています。GUI実装は`gui_dcc.py`のみです。
旧GUIモジュール名は互換エントリーポイントとして同じGUIを呼び出します。

mainへのpushと`[build]`付きの作業ブランチコミットでWindows ZIPを生成します。
CPUランナー上で単体テスト、元音源参照、実ネイティブASR、frozen版のアライメント・話者分離・日本語パスを検証します。
GPUがないCIのためRTX 4080/Vulkan上の速度や全工程のピークVRAMは測定していません。
モデルやユーザー素材はリリースに含めません。第三者ライセンスは`THIRD_PARTY.md`を参照してください。


## モデルごとの推論制約

Anime WhisperはTOMLの `constraints.inference` により、VAD短区間化・時刻生成OFF・履歴OFFを使用します。
強制アライメントOFFではVADの発話区間時刻でRPPを出力し、ONなら単語時刻を精細化します。VADモデルは初回にチェックサム付きで自動取得します。
設定例と編集済みTOMLの更新方法は [推論制約](docs/inference-policy.md) を参照してください。


## 停止と再実行

STOP後は子プロセスと一時ファイルの終了処理を待ちます。終了処理中は二重起動できません。
GOは待機・中断・失敗の項目を対象に再実行し、完了済みの出力は再処理しません。
準備中の停止、工程中の停止、例外でも項目を実行中のまま残さず、進捗の終端通知は1回だけにします。
ウィンドウを閉じた場合も停止処理を待ってから閉じます。途中からの推論再開ではなく項目単位の再実行です。
