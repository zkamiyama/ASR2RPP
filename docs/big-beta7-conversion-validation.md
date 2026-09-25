# big_beta7 automatic torch-free conversion validation

Validated on 2026-09-24 with GitHub Actions run 35946925839.

## Source

- Repository: `pcunwa/Mel-Band-Roformer-big`
- Pinned revision: `1508d1ed7c54cb0017b2cbfaabdaf3ca87d2cf74`
- Checkpoint: `big_beta7.ckpt`
- Checkpoint SHA-256: `9d68b9a8689a3500c45d3418a7811934e557760db07285f483088a0965b0eb88`
- PyTorch installed during validation: **No**

## Conversion

The ASR2RPP installer used its restricted PyTorch-ZIP reader, NumPy, and SafeTensors.
It reconstructed 732 source tensors, split 16 fused QKV tensors, and produced 748
SafeTensors tensors. The intermediate SafeTensors file was 944,446,592 bytes.

The bundled audio.cpp `audiocpp_gguf` converter then produced an F16 standalone GGUF:

- Output size: 472,298,656 bytes
- Output SHA-256: `2db0efd7daf0039a0e80472367cbe5119cdb8cf03ec6148dfe5fa3e23bdd0a25`
- Embedded model spec family: `mel_band_roformer`
- Tensor namespace: `weights`
- Embedded sidecars: yes

Both `audiocpp_gguf --inspect` and
`audiocpp_cli --inspect --family mel_band_roformer --model <gguf>` accepted the
converted model. The intermediate SafeTensors directory was removed after success.

## User-facing behavior

Selecting **Mel-Band RoFormer big beta7** and pressing **選択モデルを準備** performs
source download, SHA verification, local conversion, GGUF inspection, and runtime
inspection in the user's ASR2RPP data directory. No manual conversion step and no
PyTorch installation are required. The original verified checkpoint is retained so a
future conversion can be retried without re-downloading it.
