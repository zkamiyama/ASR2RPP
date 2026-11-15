# 用語ヒント / Glossary hints

用語はモデルTOMLへ埋め込まず、**実行ごとのUTF-8 TXT**として指定します。GUIの用語欄へ貼り付けるかTXTを読み込み、CLIでは`--terms-file terms.txt`を使用します。CSVやJSONは不要です。

1行に1用語。読み仮名を付ける場合だけ、表記と読みを**タブ1文字**で区切ります。

```text
ASR2RPP
神﨑	かんざき
音声認識	おんせいにんしき
```

上の2列の間はタブです。読みを省略するときは末尾タブも不要です。空行は無視します。先頭のUTF-8 BOM、LF/CRLF/CR改行を受け付けます。読みはひらがな・カタカナ・長音符等に限定し、半角カナも受け付けます。表記も読みも保存時・出力用ヒント生成時に正規化・書換えしません。文字の正規化は重複検査と読みの妥当性検査にのみ使います。

先頭/末尾の余分な空白、空の読み、3列以上、同一表記の重複（読みが異なる場合も含む）、制御文字、置換文字、モデル制御トークンに使われる`<`/`>`はエラーです。重複の片方を捨てたり、長い一覧を自動で切り捨てたりはしません。入力TXTは最大64 KiBですが、実際にはモデルごとの、より小さい件数・UTF-8 byte予算も適用されます。

## 認識への効き方

これは**ヒント**であり、出力の辞書置換や強制的な表記変換ではありません。音声にない用語、似た音の別の名前、長い一覧は、かえって誤認・幻覚を増やす場合があります。発話に関係する少数の用語に絞り、ヒントなしの認識とも比較してください。読み仮名を転記用文字として追加したり、文字時刻を修正したりはしません。

- Nemotron 3.5 Transcribeの読み仮名は**実験的な自然言語指示**です。音素辞書や専用の読み入力APIではなく、改善を保証しません。実行の警告とprovenanceにこの制限を保存します。現在のアダプターは日本語（`ja`）用です。
- 未対応モデルに用語を指定した場合や、表記のみ対応するモデルに読みを指定した場合は、取得・推論前にエラーにします。読みだけ黙って捨てることはありません。
- 既存の`instruct`/`system_prompt`/`text_input`/`prompt`/`initial_prompt`との併用は競合です。独自指示へ一覧を勝手に追記しません。翻訳等の別タスクとも併用できません。
- 空の用語欄は従来の動作を変えません。保存済みの旧設定も用語なしとして読めます。

GUIの用語はこの実行用の値で、グローバルな既定値として保存しません。実行の設定・報告には元の表記/読み、コードアダプターのversion、入力hash、実際に生成したヒント、警告を残します。**報告ファイルには用語が含まれる**ため、共有時には人名や社内用語を確認してください。TXTのパスを後から読み直して実行中の内容を変える方式ではありません。

## モデル能力の宣言

TOMLは対応能力・上限だけを宣言し、テンプレートやPythonコードは記述しません。`schema_version = 2`モデルに以下の表を追加します。ここにユーザーの用語を保存しないでください。

```toml
[capabilities.glossary]
version = 1
adapter = "nemotron_instruct_v1"
readings = "experimental"
max_terms = 64
max_utf8_bytes = 4096
max_prompt_utf8_bytes = 8192
```

上記は`audio_cpp / asr / nemotron_transcribe`専用です。アダプターはaudio.cpp v0.9.5のverbatim転記指示と日本語指定を維持し、引用した表記/読みの一覧を追加します。明示的にネイティブ話者を選択した場合は、話者タグ指示も維持します。TOMLに任意のnativeオプション名やテンプレートを指定して実行することはできません。

通常のWhisper向けコードアダプターも表記のみを受け付けます。

```toml
[capabilities.glossary]
version = 1
adapter = "whisper_initial_prompt_v1"
readings = "none"
max_terms = 32
max_utf8_bytes = 200
max_prompt_utf8_bytes = 223
```

これは`whisper_cpp / asr`用で、`initial_prompt`と`carry_initial_prompt=true`を使います。224-tokenの半context枠には制御用markerが1個必要なため、コードはpromptを最大223 UTF-8 bytesに制限します。byte単位の上限はtoken数の保守的な上限であり、表記を途中で切りません。明示的な`carry_initial_prompt=false`や小さい`max_context`は競合です。初期`temperature >= 0.5`ではWhisperがpromptを使わないため拒否します。temperature fallbackが有効な場合、途中の高温fallbackでヒントを使わなくなる可能性を警告しますが、decoder設定を勝手に変更しません。**`max_context=0`を強制するWhisperのVAD text-only経路では用語が効かないため、その組合せは拒否します。**

Anime Whisperの既存TOMLは`initial_prompt`を禁止しているため、この能力を有効にしません。既存モデルの禁止事項を用語機能で迂回することはありません。未宣言のモデルを名前だけで対応扱いにすることもありません。

各tableは全フィールド必須で未知フィールド/version/adapterを拒否します。TOML上限はコードの最大値以下にのみ設定できます。`max_utf8_bytes`は元TXT、`max_prompt_utf8_bytes`は指示を含む生成後ヒントの予算です。どちらもモデル取得前に検査します。

## 根拠

- [audio.cpp v0.9.5のNemotron指示とlanguage置換](https://github.com/0xShug0/audio.cpp/blob/v0.9.5/src/community_models/nemotron_transcribe/session.cpp)
- [audio.cpp v0.9.5 Nemotron option contract](https://github.com/0xShug0/audio.cpp/blob/v0.9.5/docs/models/nemotron_transcribe.md)
- [固定Whisper context/prompt処理](https://github.com/ggml-org/whisper.cpp/blob/a664346ea5c6dddff3e61a2b7b32dd4514613f50/src/whisper.cpp#L7093)

能力宣言や構文テストは、実モデルで用語精度が改善したという証拠ではありません。実機検証の音声・モデル・設定・ヒントなし対照・結果を別途確認してください。

## 実音声での確認と限界

RTX 4080 / WSL / CUDAで、自然な日本語の二人会話60.2秒をヒントなし・表記のみ・読み付きで比較しました。3条件とも独立Qwenアライメントの正の区間時刻を保ってJSON/RPP/OTIOを出力しましたが、対象5語はヒントなしでも認識できており、用語救済の効果は示せていません。両ヒント条件で用語へ寄せすぎた言い換えが1か所あり、読み付きが表記のみより優れる結果でもありません。無関係な2語は出ませんでしたが、幻覚しない保証にはなりません。

[比較値と入力・runtimeの出典](validation/nemotron-glossary-source-20261115.json)。これはソース実行の短い検証であり、30分音声や配布Windows/Vulkan版の合格証跡ではありません。

通常モデルの回帰確認では、同じ音声でAnimeWhisper（用語なし）とWhisper base（表記ヒント）も実行し、各83/87区間の正のアライメント時刻と3形式の一致を確認しました。Whisperの実行引数に105-byteのヒントとcarry指定が保存済みprovenanceどおり渡ったことを確認していますが、すべてのwindowでのヒント使用や精度向上は保証しません。[回帰確認値](validation/glossary-regressions-source-20261115.json)。
