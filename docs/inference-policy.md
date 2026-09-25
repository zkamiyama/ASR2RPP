# TOMLによる推論制約とVAD区間時刻

モデルIDや表示名ではなく、モデル定義の制約で処理方法を決めます。
Anime Whisperの同梱テンプレートには次の設定を適用します。
自前変換したモデルのTOMLにも同じブロックを記述できます。

```toml
[constraints]
disabled_parameters = ["initial_prompt", "carry_initial_prompt"]

[constraints.inference]
segmentation = "vad"
timestamps = false
history = false
max_segment_seconds = 25.0
timestamp_source = "vad"
```

`timestamps=false` はWhisperデコーダの時刻生成OFF、`history=false` は前区間の
認識履歴OFFです。単なる推奨値ではなく強制制約です。GUIでは値を固定し、
CLIの `vad=false` / `no_timestamps=false` / `max_context=-1` 等は実行前に拒否します。
保存済みGUIの矛盾する旧値は除去しますが、CLIの矛盾を黙って書き換えません。

## VADの時刻を使う場合

`timestamp_source="vad"`（VADプロファイルの既定）は、強制アライメントを任意にします。
OFFなら、VADで得た開始・終了の区間全体に認識文字列を1アイテムとして配置します。
発話の前後の余裕や短い間も含む近似的な発話領域です。単語時刻を生成したり、
文字数に比例して区間を分割したりはしません。記録は `method="vad_segment"`、
`granularity="segment"`、manifestの `timestamp_source="vad"` で区別します。

無音を取り除いて音声を連結する方式ではありません。元の時間位置を維持し、
ORIGINALは無音を含む参照音源の全長を保持します。範囲指定でも位置を元へ戻します。
話者分離を併用すると、VAD区間全体に時間重複から話者を割り当てます。
一つのVAD区間中の話者交代を細かく分けたい場合は強制アライメントを併用してください。

強制アライメントをONにすれば、同じ音声と文字列をアライナへ渡して時刻を精細化します。
アライメントが失敗した場合、VAD時刻への切り替えで成功扱いにはしません。
`timestamp_source="alignment"` と明示すれば必須にできます。GUIのOFFを禁止し、
CLIでアライメント指定を省くと実行前にエラーにします。
通常モデルは `timestamp_source="native"`（省略時）で従来どおりです。

## 対応する制約

現在は `runtime="whisper_cpp"` / `task="asr"` / 16 kHz のVAD経路に対応します。
VAD指定時は `timestamps=false` と `history=false` の両方が必要です。
通常の時刻付き処理で `history=false` だけを指定することもできます。
未知のキー、文字列の `"false"`、NaN、無効な組み合わせは拒否します。
最大長は2〜28秒で、前後の余裕を含む実入力WAVの上限です。

## VADと入力区間

Silero VAD v5.1.2 GGMLを、アプリと同じ固定revisionでビルドした
`whisper-vad-speech-segments` CPUヘルパーで実行します。モデルは初回に
固定revision・サイズ・SHA-256を検証して取得し、以後は保存先から再利用します。
PyTorch・Transformers・ONNX Runtimeは不要です。
ヘルパー出力のcentisecondを秒に換算し、順序・件数・音声長を検査します。
長すぎる領域は低エネルギー位置付近で分割し、最大長を必ず守ります。
自然なVAD境界を優先しますが、無休止の発話の強制分割では単語が切れる可能性があります。

| 調整キー | 初期値 | 許容範囲 |
|---|---:|---|
| `vad_threshold` | 0.5 | 0.01〜1.0 |
| `vad_min_speech_duration_ms` | 100 | 0〜1000 ms |
| `vad_min_silence_duration_ms` | 250 | 50〜2000 ms |
| `vad_speech_pad_ms` | 200 | 0〜1000 ms |
| `vad_max_speech_duration_s` | 0 | 0はTOML上限、変更時は2秒以上でその上限以下 |
| `vad_samples_overlap` | 0.2 | 0〜1秒、強制アライメントON時のみ使用 |
| `vad_model` | 空欄 | 独自の互換GGMLパス、相対パスはTOMLの場所基準 |

アライメントOFFでは追加の重なり文脈を無効にします。単語時刻がない状態で
重なりに由来する文字だけを安全に除くことはできないためです。VAD前後の余裕は
維持します。ON時は重なりを許可し、単語中央時刻と担当区間で重複を避けます。

各WAVを `-nt -mc 0` で独立認識し、コマンド長制限付きバッチでモデルを再利用します。
ネイティブの `--vad` による連結は使いません。不正UTF-8・不正JSON・置換文字は
エラーとして原始出力を保存します。先頭文字を機械的に削る修正はしません。
空文字・句読点のみの結果は原始記録に残し、発話アイテムを捏造しません。
VAD境界・実入力範囲・時刻の由来は `asr/vad.json` と `asr/raw.json` に残ります。

単一ファイル／ファイル優先／ステージ優先キューで同じ実装を使います。
推論用WAVはキャッシュのみで、診断フォルダーにはコピーしません。
完全に未編集の旧同梱TOMLだけを既知のSHA-256で識別してバックアップ後に更新し、
ユーザー編集済み・独自TOMLは上書きしません。既存GGMLは再ダウンロード不要です。

## 限界

VADもASRも誤り得ます。小声・短い相づち・効果音・固有名詞・強制分割境界は要確認です。
UTF-8正常は全文正解の保証ではありません。アライメントも誤認識を修正しません。
公開CIは合成／公開fixtureのみを使用し、利用者の動画・台詞・パスは含めません。
