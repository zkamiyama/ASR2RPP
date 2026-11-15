## ASR2RPP 0.4.0

### 新機能

- モデル一覧に **NVIDIA Nemotron 3.5 Transcribe 30B-A4B（IQ4_NL）** を追加しました。従来のNemotron ASRや独立したNemotron話者分離モデルとは別です。初回のモデル取得は約22 GBです。
- 対応するモデルに用語ヒントを渡せます。GUIの **Settings → Glossary**、またはCLIの `--terms-file terms.txt` から指定します。UTF-8で1行1語。30B-A4Bの日本語では「表記」と「読み」をタブで区切って指定できます。読みは実験的なヒントで、正しい表記を保証しません。少数の関連語に絞ってお試しください。
- 30B-A4B内蔵の話者推定は **初期状態でオフ** です。編集用の細かい時刻にはQwen Forced Aligner、話者ラベルには独立したNemotron Diarizationを推奨します。アライナーなしではVAD区間単位の粗い時刻になります。内蔵の時刻・話者推定を明示的に選ぶと、不正な区間を検出して書き出しを止めることがあります。

### インストールと検証範囲

OSに合うZIPを新しいフォルダーに全展開し、ASR2RPP.exe / ASR2RPP.appを起動します。既定のASRは従来どおりAnime Whisper／jaです。モデルの重みと未導入のFFmpegは初回に別途取得します。音声は書き出しに埋め込まず参照するため、素材ファイルは同じ場所に置いてください。

Windows x64（AVX2・Vulkan）とApple Silicon（macOS 14以降・Metal）向けです。Windows版は未署名、macOS版はad-hoc署名・未公証です。

公開には、両OSの候補ZIPと、Windowsの物理RTX 4080実機で全21モデルおよび3形式出力を検証した記録が必要です。現時点の30B-A4B実音声確認はWSL/CUDAのソース実行で、60.2秒の日本語会話に独立した時刻・話者ラベルを付与したものです。配布Windows/Vulkan版の合格を意味しません。用語ヒントは改善が保証されず、今回も用語に寄せすぎた言い換えがありました。30分の日本語音声での認識精度・処理時間、macOS実機での30B-A4B動作は未確認です。公開時のZIPのハッシュとWindows事後検証記録は、リリースに添付する `SHA256SUMS.txt` と `VALIDATION-Windows.json` で確認してください。

English: Adds Nemotron 3.5 Transcribe 30B-A4B (IQ4_NL), UTF-8 glossary hints, and optional Japanese readings for Nemotron. Built-in speaker labeling is experimental and off by default. Use the separate forced aligner and diarizer for edit timestamps and speaker labels. Extract the complete ZIP before launching; model files are downloaded separately.
