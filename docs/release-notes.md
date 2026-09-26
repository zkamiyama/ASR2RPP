# ASR2RPP Windows preview

未署名のWindows x64プレビュー版です。ZIPを新しいフォルダーへ全展開し、ASR2RPP.exeを起動してください。EXEだけを移動しないでください。

- STOP後のGO再実行、準備失敗やキュー中断、完了済み項目の保護を修正。
- GUIとCLIは展開フォルダーのmodelsを直接参照。一時／ユーザーフォルダーへの同梱TOMLの自動コピーは廃止。
- 独自定義はユーザーデータのcustom-modelsへ。旧models内の独自IDもその場所から読みます。同梱IDとの競合はログで通知し、ユーザーファイルは変更しません。
- VAD区間時刻、WhisperのPCM直接区間入力・モデル再利用、ORIGINAL全長参照を維持。

モデル重み・FFmpeg・PyTorchは同梱していません。既存のモデル重みは再利用できます。
CPU/Vulkanエンジンを同梱。CIの実推論はCPUで、Vulkanの実機GPU性能検証ではありません。
このZIPはWindows/Linuxテスト、ネイティブPCMテスト、frozen GUIライフサイクル試験と実ネイティブ推論ゲートを通過した同一ファイルです。
SHA256SUMS.txtとversion.jsonでファイルとコミットを確認できます。
