"""Public-fixture Windows checks. Never uploads the user's media or model weights."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.request import urlopen
import zipfile

root = Path(__file__).resolve().parents[1]
archive = root / 'download/ASR2RPP-Windows-x64.zip'
with zipfile.ZipFile(archive) as z:
    z.extractall(root / 'stage-app')
cli = root / 'stage-app/ASR2RPP/asr2rpp-cli.exe'
reports = root / 'reports/stages'
reports.mkdir(parents=True, exist_ok=True)
os.environ['ASR2RPP_HOME'] = str(root / 'stage-home')
os.environ['PYTHONUTF8'] = '1'
fixture = root / 'public-speech.wav'
url = 'https://raw.githubusercontent.com/ggml-org/whisper.cpp/a664346ea5c6dddff3e61a2b7b32dd4514613f50/samples/jfk.wav'
with urlopen(url, timeout=30) as response, fixture.open('wb') as handle:
    shutil.copyfileobj(response, handle)
results = {}


def run(name, args, timeout=600):
    started = time.monotonic()
    try:
        with (reports / (name + '.log')).open('w', encoding='utf-8') as log:
            result = subprocess.run([str(cli), *map(str, args)], stdout=log, stderr=subprocess.STDOUT,
                                    timeout=timeout)
        results[name] = {'returncode': result.returncode, 'elapsed_seconds': time.monotonic() - started}
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        results[name] = {'error': 'timeout', 'timeout_seconds': timeout}
        # Native workers are separate executables, explicitly terminate only test-owned engine names.
        for binary in ('audiocpp_cli.exe', 'whisper-cli.exe', 'asr2rpp-cli.exe'):
            subprocess.run(['taskkill', '/F', '/T', '/IM', binary], capture_output=True)
        return False
    finally:
        (reports / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if not run('download', ['models', 'install', 'whisper-base', 'nemotron-asr', 'nemotron-diarization', 'qwen-forced-aligner'], 600):
    raise SystemExit('Model acquisition failed; see reports')
for name, extra in [
    ('asr-only', []),
    ('diar-only', ['--diar', 'nemotron-diarization']),
    ('align-only', ['--align', 'qwen-forced-aligner', '--align-language', 'English']),
    ('both', ['--diar', 'nemotron-diarization', '--align', 'qwen-forced-aligner', '--align-language', 'English']),
]:
    run(name, ['run', fixture, '--asr', 'nemotron-asr', '--asr-language', 'en-US', '--duration', '8',
               '--output-dir', reports / name, *extra], 600)
# UTF-8 input and output paths with the frozen launcher and native executable.
unicode_input = root / '日本語入力' / '試験音声.wav'
unicode_input.parent.mkdir(exist_ok=True)
shutil.copy2(fixture, unicode_input)
run('unicode-whisper', ['run', unicode_input, '--asr', 'whisper-base', '--asr-language', 'en',
                        '--duration', '8', '--output-dir', reports / '日本語出力'], 120)
# Check that ON/OFF reached the pipeline independently; detailed raw output is retained.
for name in ('asr-only', 'diar-only', 'align-only', 'both'):
    manifests = list((reports / name).glob('*.asr2rpp/manifest.json'))
    if manifests:
        data = json.loads(manifests[0].read_text(encoding='utf-8'))
        results[name]['stages'] = {'diar': data['settings']['diar'] is not None,
                                  'align': data['settings']['align'] is not None}
        results[name]['rpp_count'] = len(list((reports / name).glob('*.rpp')))
        results[name]['source_unchanged'] = data.get('source_unchanged')
(reports / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
raise SystemExit(0 if all(v.get('returncode') == 0 for v in results.values()) else 1)
