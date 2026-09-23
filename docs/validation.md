# ASR2RPP validation strategy

## Product boundaries

- Initial RPP export uses only non-destructive references. Source separation and per-utterance media creation are out of scope.
- Keep the RPP writer independent of ML libraries, GUI libraries, and FFmpeg.
- Keep source offsets, project positions, and inference-crop offsets as separate values. A validation crop must not silently move events to the beginning of the original media timeline.
- Future PCM WAV and stream-copy exporters consume the same edit plan; they do not live inside the RPP serializer.

## CPU CI

Use standard `ubuntu-24.04` GitHub-hosted runners for reproducible CPU smoke tests. Record the actual runner image version, environment, runtime version, model revision, and content hashes. The OS label alone is not a complete environment lock.

`tools/native_cpu_smoke.sh` tests NVIDIA NeMo-Speech.cpp 0.1.0 with the Nemotron 3.5 ASR model on CPU without installing PyTorch. It downloads the official CPU archive, verifies a pinned SHA-256, records the CLI model index, synthesizes an original English test sentence locally, limits audio to 60 seconds, and requests JSON transcription output.

This test is **not** a benchmark of Japanese accuracy, human speech boundary accuracy, RPP export, or the new Nemotron 3 Diarization model. It does not use the runtime's older four-speaker Sortformer as a substitute for the requested diarization model.

The workflow has a 30-minute job timeout, bounded download/inference commands, read-only repository permissions, and three-day retention for small reports. It does not use paid/GPU runners or persist model weights. The initial test uses a public synthetic fixture and does not need secrets.

For local Ubuntu execution, install FFmpeg, espeak-ng, libsndfile1, libgomp1, curl, tar, Python 3 and GNU time, then run:

```sh
bash tools/native_cpu_smoke.sh
```

Do not export bundled native-library search paths globally: doing so can make unrelated FFmpeg or Python processes load incompatible runtime libraries. The script scopes its native-library path to the native executable.

## Reference and deployment are different concerns

1. Reference: verify the requested Nemotron 3 Diarization checkpoint using an explicitly pinned NeMo/PyTorch or supported Transformers environment on CPU. Installation success and model inference success must be reported separately.
2. Native candidate: NeMo-Speech.cpp / audio.cpp provide model-specific implementations with CPU/Metal/Vulkan/CUDA paths. A supported accelerator does not imply support for every checkpoint.
3. ONNX candidate: sherpa-onnx has a concrete Nemotron 3.5 streaming ASR export path (encoder, decoder, joiner, tokenizer, caches and language prompts). Provider parity is a separate test from ONNX export success.
4. Hardware tests: actual Metal, DirectML, Vulkan or CUDA execution requires the relevant GPU and driver. CPU CI or a successful binary build must not be reported as a GPU validation.

The deployment application does not require PyTorch globally. Each backend owns its dependencies and can run in a subprocess. Use serializable audio descriptors and results, not framework tensors, across the application/backend boundary.

For Apple devices, investigate PyTorch MPS for the reference implementation, or native ggml Metal/CoreML/MLX implementations when the exact model is supported. For Windows, investigate ONNX Runtime/Windows ML/DirectML or native Vulkan. `torch-directml` still depends on PyTorch and must not be treated as a universal compatibility shim for current NeMo code.

## Precision tests

Record raw ASR emission/token times separately from forced-alignment times and final edit bounds. First compare unquantized baselines, then quantify changes introduced by int8/Q8 or other precision profiles. Merely producing timestamp fields does not establish accurate speech boundaries.

Use speech-bearing crops of at most 60 seconds for initial functional tests. Exclude artificially cut start/end boundaries when estimating recognition/alignment quality. Long-form speaker identity and memory behavior remain separate later tests.

## Private media

Do not commit private samples, signed URLs, tokens, raw private transcripts, or model weights. Public Actions logs and artifacts must be treated as externally visible. Supplying a URL through a secret does not make generated transcripts private. Use local processing or an explicitly authorized private validation environment for nonpublic samples.

## Primary references

- https://docs.github.com/en/actions/reference/runners/github-hosted-runners
- https://github.com/NVIDIA/NeMo-Speech.cpp/tree/v0.1.0
- https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/nemotron-3.5-asr-streaming-0.6b
- https://docs.pytorch.org/docs/stable/notes/mps.html
- https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
- https://learn.microsoft.com/en-us/windows/ai/directml/
