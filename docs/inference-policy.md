# Timing policy

Timing is selected by the user under **Settings → Timing**. Use the current [English guide](../README.md), [日本語ガイド](../README.ja.md), [model catalog](models.md) and [customization guide](provider-models.md).

Automatic chooses native model intervals when available, or VAD speech regions for text-only ASR. Selecting an alignment stage refines the transcript against audio. Explicit native/VAD/alignment choices take precedence over compatible legacy defaults.
VAD intervals are utterance windows, not word boundaries. Native emission-frame estimates and forced-alignment times retain distinct provenance. No characters-per-second timing is fabricated.

New TOMLs declare `capabilities.timestamps` and native input/output contracts. Do not encode a mandatory VAD/alignment workflow just because a model has no native timestamps. Schema-1 `[constraints.inference]` remains supported for older user definitions; it is not the recommended interface for new ones.

Aligners have input-length limits. Use the VAD-before-alignment setting when an ASR model returns long, coarse segments. Audio/text correspondence must remain intact; arbitrarily splitting audio and distributing words is not valid alignment.
