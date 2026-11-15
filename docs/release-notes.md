## ASR2RPP 0.4.0

### 新機能

- モデル一覧に **NVIDIA Nemotron 3.5 Transcribe 30B-A4B（IQ4_NL）** を追加しました。従来のNemotron ASRや独立したNemotron話者分離モデルとは別です。初回のモデル取得は約22 GBです。
- 対応するモデルに用語ヒントを渡せます。GUIの **Settings → Glossary**、またはCLIの `--terms-file terms.txt` から指定します。UTF-8で1行1語。30B-A4Bの日本語では「表記」と「読み」をタブで区切って指定できます。読みは実験的なヒントで、正しい表記を保証しません。少数の関連語に絞ってお試しください。
- 30B-A4B内蔵の話者推定は **初期状態でオフ** です。編集用の細かい時刻にはQwen Forced Aligner、話者ラベルには独立したNemotron Diarizationを推奨します。アライナーなしではVAD区間単位の粗い時刻になります。内蔵の時刻・話者推定を明示的に選ぶと、不正な区間を検出して書き出しを止めることがあります。

### インストールと検証範囲

OSに合うZIPを新しいフォルダーに全展開し、ASR2RPP.exe / ASR2RPP.appを起動します。既定のASRは従来どおりAnime Whisper／jaです。モデルの重みと未導入のFFmpegは初回に別途取得します。音声は書き出しに埋め込まず参照するため、素材ファイルは同じ場所に置いてください。

Windows x64（AVX2・Vulkan）とApple Silicon（macOS 14以降・Metal）向けです。Windows版は未署名、macOS版はad-hoc署名・未公証です。

Windows 11／RTX 4080実機で、配布ZIPの全21モデルとRPP・OTIO・JSON出力を検証しました。30B-A4Bは約1分の自然な日本語会話で、別のアライナーと話者分離モデルを使って時刻・話者ラベルを付与し、書き出せました。発話の重なりなどで話者不明になる区間は残ります。30分の日本語音声での認識精度・処理時間と、macOS実機での30B-A4B動作は未確認です。

用語ヒントは精度改善を保証しません。無関係なヒントを与えた試験では、音声にない語が入り、区間外のアライメント結果の一部が警告付きで書き出しから除外されました。音声に出なかった用語や読みも生JSON・診断の設定に残るため、共有前に確認してください。

同じWindows ZIPの初回検証では、Whisper TinyのVulkan実行が一度クラッシュしました。単体再試験と、その後の全21モデルの再試験は通過しています。原因は未確定です。詳細は [検証記録](https://github.com/zkamiyama/ASR2RPP/blob/main/docs/validation/0.4.0.md)、配布ファイルのハッシュは添付の `SHA256SUMS.txt` を参照してください。

English: Adds Nemotron 3.5 Transcribe 30B-A4B (IQ4_NL), UTF-8 glossary hints, and optional Japanese readings for Nemotron. Built-in speaker labeling is experimental and off by default. Use the separate forced aligner and diarizer for edit timestamps and speaker labels. Extract the complete ZIP before launching; model files are downloaded separately.
