## ASR2RPP 0.3.0

ローカルの音声・動画から、RPP・OpenTimelineIO・再利用可能なJSONを作成します。WindowsはVulkan、Apple SiliconはMetal。CPUも利用できます。

### 試し方

OSに合う一式ZIPを新しいフォルダーへ全展開し、ASR2RPP.exe / ASR2RPP.appを起動してください。`examples/sample.wav`をドラッグし、Whisper Base・言語en・追加工程OFF・時刻付与「自動」でGO。初回は選択モデルと未導入のFFmpegを取得するため、ネット接続が必要です。Pythonや開発用SDKの手動導入は不要です。

### 主な変更

- 出力チップでRPP / OTIO / JSONを追加・削除。初期値RPP、未選択なら実行拒否。
- ORIGINAL全長ミュート、話者別トラック、無音区間、重複発話レーンを共通タイムラインから出力。
- JSONには生の結果・補正履歴・詳細区間・設定・モデル来歴・音源参照を収録。CLI `convert`で推論せず再変換可能。
- 20種類の代表モデルTOML、設定画面でのVAD／強制アライメント、CPU版ReazonSpeechを同梱。
- 新しいオリジナルSVG／EXEアイコン。READMEと依存関係・カスタマイズガイドを整理。
- CUDA・CTranslate2・faster-whisperは配布に含めません。モデル重みとFFmpegも初回の別取得です。

Windows: x64 / AVX2対応CPU。Mac: Apple Silicon / macOS 14以降、Intel非対応。Windowsは未署名、Macはad-hoc署名・未公証です。

OTIOのミュート指定を反映するかは受け側の編集ソフトにも依存します。音声は埋め込まず参照します。JSONと診断フォルダーは本文や個人のパスを含むため、共有前に確認してください。

公開はWindows実機の全20ケースと3形式出力、両OSのCI成果物を検証した後に行います。機能検証は認識品質・全長尺音声の保証ではありません。Mac CIの仮想GPUではQwen3-ASRのMetal行列演算機能が不足する場合、そのケースは未検証として明示します。

English: Extract the complete ZIP, launch the app, drop `examples/sample.wav`, select Whisper Base and language en, then GO. First use requires internet for weights and missing FFmpeg. RPP is the default; add OTIO/JSON with the format chips. JSON can be re-exported without inference. Media is referenced, not embedded. See the included English/Japanese README, licenses and validation assets.
