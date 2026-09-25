#!/usr/bin/env bash
# A runtime/download smoke test, not a Japanese ASR accuracy benchmark.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ASR2RPP_REPORT_DIR:-$ROOT/reports/native-cpu}"
WORK="$(mktemp -d)"
mkdir -p "$OUT"
trap 'rm -rf "$WORK"' EXIT
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export NEMO_SPEECH_MODEL_DIR="$WORK/models"
export CUDA_VISIBLE_DEVICES=""
for cmd in curl tar sha256sum ffmpeg espeak-ng python3 timeout; do
  command -v "$cmd" >/dev/null || { echo "Missing dependency: $cmd" >&2; exit 2; }
done
{
  uname -a
  python3 --version
  free -h
  df -h "$ROOT"
  ffmpeg -version | head -n 1
} > "$OUT/environment.txt"
ASSET='nemo-speech-0.1.0-linux-x86_64-cpu.tar.gz'
URL="https://github.com/NVIDIA/NeMo-Speech.cpp/releases/download/v0.1.0/$ASSET"
EXPECTED='0f74131d631ad2c694cf0ec53490866bb6461147959589a69fb6fc231944065b'
curl --fail --location --proto '=https' --proto-redir '=https' \
  --retry 2 --connect-timeout 20 --max-time 180 "$URL" -o "$WORK/$ASSET"
printf '%s  %s\n' "$EXPECTED" "$WORK/$ASSET" | sha256sum --check > "$OUT/runtime-sha256.txt"
tar -xzf "$WORK/$ASSET" -C "$WORK"
BIN="$(find "$WORK" -type f -name nemo-speech -print -quit)"
test -n "$BIN" && test -x "$BIN"
# Do NOT export a bundled C++ runtime into ffmpeg/espeak/Python's environment.
NATIVE_LIB="$(dirname "$BIN")/../lib"
NATIVE=(env "LD_LIBRARY_PATH=$NATIVE_LIB" "$BIN")
"${NATIVE[@]}" --help > "$OUT/cli-help.txt" 2>&1
"${NATIVE[@]}" doctor > "$OUT/doctor.txt" 2>&1
"${NATIVE[@]}" --json model list > "$OUT/model-index.json" 2> "$OUT/model-index.stderr.txt"
# Original synthetic text. No Drive data, signed URLs, or private media.
espeak-ng -v en-us -s 145 -w "$WORK/speech.wav" \
  'This is a local speech recognition test. The sample is shorter than one minute.'
ffmpeg -hide_banner -loglevel error -nostdin -y -i "$WORK/speech.wav" \
  -t 60 -vn -ac 1 -ar 16000 -c:a pcm_s16le "$WORK/input.wav"
python3 - "$WORK/input.wav" "$OUT/input.json" <<'PY'
import hashlib, json, pathlib, sys, wave
p = pathlib.Path(sys.argv[1])
with wave.open(str(p)) as w:
    duration = w.getnframes() / w.getframerate()
    assert 0 < duration <= 60, duration
    data = dict(synthetic=True, language='en-US', duration_seconds=duration,
                sample_rate=w.getframerate(), samples=w.getnframes(),
                sha256=hashlib.sha256(p.read_bytes()).hexdigest())
pathlib.Path(sys.argv[2]).write_text(json.dumps(data, indent=2), encoding='utf-8')
PY
# Separate download and inference. The pinned CLI verifies model size/hash.
/usr/bin/time -v -o "$OUT/download-resources.txt" \
  timeout --signal=TERM --kill-after=30s 600 \
  "${NATIVE[@]}" pull nemotron-3.5 > "$OUT/download.stdout.txt" 2> "$OUT/download.stderr.txt"
/usr/bin/time -v -o "$OUT/inference-resources.txt" \
  timeout --signal=TERM --kill-after=30s 900 \
  "${NATIVE[@]}" transcribe "$WORK/input.wav" --model nemotron-3.5 \
  --device cpu --language en-US --json --output "$OUT/transcript.json" \
  > "$OUT/inference.stdout.txt" 2> "$OUT/inference.stderr.txt"
python3 - "$OUT/transcript.json" "$OUT/smoke-status.json" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
data = json.loads(p.read_text(encoding='utf-8'))
texts, timing_keys = [], set()
def visit(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ('text', 'transcript', 'transcription') and isinstance(v, str) and v.strip():
                texts.append(v)
            if any(t in k.lower() for t in ('start', 'end', 'timestamp', 'time')):
                timing_keys.add(k)
            visit(v)
    elif isinstance(x, list):
        for v in x:
            visit(v)
visit(data)
if not texts:
    raise SystemExit('No nonempty transcript found; inspect transcript.json')
result = dict(runtime='NeMo-Speech.cpp v0.1.0', backend='cpu',
              model='nemotron-3.5', pytorch_required=False,
              transcription_smoke='passed', timing_keys=sorted(timing_keys),
              timestamp_accuracy='not_evaluated', japanese_accuracy='not_evaluated',
              nemotron_3_diarization='not_run')
pathlib.Path(sys.argv[2]).write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
PY
