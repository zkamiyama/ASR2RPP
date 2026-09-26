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
archive = root / 'dist/ASR2RPP-Windows-x64.zip'
if not archive.is_file():
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
    run(name, ['run', fixture, '--asr', 'nemotron-asr', '--asr-device', 'cpu',
               '--asr-language', 'en-US', '--duration', '8',
               '--diar-device', 'cpu', '--align-device', 'cpu',
               '--output-dir', reports / name, *extra], 600)
# UTF-8 input and output paths with the frozen launcher and native executable.
unicode_input = root / '日本語入力' / '試験音声.wav'
unicode_input.parent.mkdir(exist_ok=True)
shutil.copy2(fixture, unicode_input)
run('unicode-whisper', ['run', unicode_input, '--asr', 'whisper-base', '--asr-device', 'cpu',
                        '--asr-language', 'en', '--duration', '8',
                        '--output-dir', reports / '日本語出力'], 120)
# The new policy is data-driven even for Whisper Base; no Anime model-ID branch.
source_bin = next((Path(os.environ['ASR2RPP_HOME'])/'weights'/'whisper-base').rglob('ggml-base.bin'))
profile = Path(os.environ['ASR2RPP_HOME'])/'models'/'policy-base.toml'
profile.write_text('runtime="whisper_cpp"\ntask="asr"\n[source]\npath=' +
                   json.dumps(str(source_bin.resolve())) +
                   '\n[constraints.inference]\nsegmentation="vad"\ntimestamps=false\nhistory=false\nmax_segment_seconds=25.0\ntimestamp_source="alignment"\n', encoding='utf-8')
run('policy-requires-align', ['run', fixture, '--asr', 'policy-base', '--asr-device', 'cpu',
                             '--asr-language', 'en', '--duration', '8',
                             '--output-dir', reports/'policy-rejected'], 30)
results['policy-requires-align']['expected_returncode'] = 2
assert results['policy-requires-align']['returncode'] == 2
assert not list((reports/'policy-rejected').glob('*.rpp'))
profile.write_text(profile.read_text(encoding='utf-8').replace('timestamp_source="alignment"', 'timestamp_source="vad"'), encoding='utf-8')
run('vad-region-times', ['run', unicode_input, '--asr', 'policy-base', '--asr-device', 'cpu',
                         '--asr-language', 'en', '--duration', '8', '--output-dir', reports/'vad-region-times'], 180)
if results['vad-region-times'].get('returncode') == 0:
    report = next((reports/'vad-region-times').glob('*.asr2rpp'))
    manifest = json.loads((report/'manifest.json').read_text(encoding='utf-8'))
    transcript = json.loads((report/'transcript.json').read_text(encoding='utf-8'))
    vad = json.loads((report/'asr'/'vad.json').read_text(encoding='utf-8'))
    assert manifest['timestamp_source'] == 'vad' and manifest['source_unchanged']
    assert transcript['units'] and all(u['method'] == 'vad_segment' for u in transcript['units'])
    assert not (report/'align').exists() and not list(report.rglob('*.wav'))
    assert not vad['context_overlap_enabled'] and vad['options']['overlap'] == 0
    results['vad-region-times']['vad_segments'] = len(transcript['units'])
    results['vad-region-times']['timestamp_source'] = manifest['timestamp_source']
    raw = json.loads((report/'asr'/'raw.json').read_text(encoding='utf-8'))
    assert raw['performance']['input_mode'] == 'pcm_plan_v1'
    assert raw['performance']['model_processes'] == 1 and raw['performance']['segment_wav_files'] == 0
run('vad-policy', ['run', unicode_input, '--asr', 'policy-base', '--asr-device', 'cpu',
                   '--asr-language', 'en', '--align', 'qwen-forced-aligner', '--align-device', 'cpu',
                   '--align-language', 'English', '--duration', '8', '--output-dir', reports/'vad-policy'], 600)
if results['vad-policy'].get('returncode') == 0:
    report = next((reports/'vad-policy').glob('*.asr2rpp'))
    manifest = json.loads((report/'manifest.json').read_text(encoding='utf-8'))
    vad = json.loads((report/'asr'/'vad.json').read_text(encoding='utf-8'))
    transcript = json.loads((report/'transcript.json').read_text(encoding='utf-8'))
    assert manifest['source_unchanged'] and manifest['inference_policy']['timestamps'] is False
    assert vad['windows'] and max(w['end']-w['start'] for w in vad['windows']) <= 25.0001
    assert transcript['units'] and all(u['method'] != 'vad_window' for u in transcript['units'])
    assert not list(report.rglob('*.wav'))
    for log in (report/'asr').glob('asr-batch-*.log'):
        command = json.loads(log.read_text(encoding='utf-8').splitlines()[0])['command']
        assert '-nt' in command and command[command.index('-mc')+1] == '0' and '--vad' not in command
    results['vad-policy']['windows'] = len(vad['windows'])
    results['vad-policy']['aligned_units'] = len(transcript['units'])

# Compare the packaged native PCM plan against the same binary's WAV path.
import wave
sys.path.insert(0, str(root))
from asr2rpp.whisper_io import write_plan
from asr2rpp.vad import SpeechWindow
from asr2rpp.media import slice_pcm
import threading
native = cli.parent/'engines'/'whisper_cpp-cpu'/'whisper-cli.exe'
compare = reports/'pcm-equivalence'
compare.mkdir(exist_ok=True)
windows = [SpeechWindow(i,i+2,i,i+2) for i in (0,2,4,6)]
legacy = [compare/f'legacy-{i}' for i in range(4)]
planned = [compare/f'plan-{i}' for i in range(4)]
base = [str(native), '-m', str(source_bin), '-l', 'en', '-nt', '-mc', '0', '-ng', '-oj', '-np']
command = list(base)
for i,w in enumerate(windows):
    audio = compare/f'input-{i}.wav'
    slice_pcm(fixture,audio,w.start,w.end-w.start,threading.Event())
    command += ['-f',str(audio),'-of',str(legacy[i])]
with (compare/'legacy.log').open('w',encoding='utf-8') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
plan = compare/'input.plan'
write_plan(fixture,windows,planned,plan,threading.Event())
with (compare/'plan.log').open('w',encoding='utf-8') as log:
    subprocess.run(base+['--asr2rpp-pcm-plan',str(plan)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
for left,right in zip(legacy,planned):
    a = json.loads(left.with_suffix('.json').read_text(encoding='utf-8'))['transcription']
    b = json.loads(right.with_suffix('.json').read_text(encoding='utf-8'))['transcription']
    assert a == b, (left.name,right.name)
results['pcm-equivalence'] = {'returncode':0,'independent_regions':4,'exact_native_transcription_match':True}

# Check that ON/OFF reached the pipeline independently; detailed raw output is retained.
for name in ('asr-only', 'diar-only', 'align-only', 'both'):
    manifests = list((reports / name).glob('*.asr2rpp/manifest.json'))
    if manifests:
        data = json.loads(manifests[0].read_text(encoding='utf-8'))
        results[name]['stages'] = {'diar': data['settings']['diar'] is not None,
                                  'align': data['settings']['align'] is not None}
        results[name]['rpp_count'] = len(list((reports / name).glob('*.rpp')))
        results[name]['source_unchanged'] = data.get('source_unchanged')
        outputs = list((reports / name).glob('*.rpp'))
        if results[name].get('returncode') == 0:
            assert len(outputs) == 1
            text = outputs[0].read_text(encoding='utf-8')
            first = text.split('  <TRACK\n')[1]
            assert first.startswith('    NAME "ORIGINAL"\n')
            assert first.count('<ITEM') == 1 and 'MUTESOLO 1 0 0' in first
            assert data['original_track']['duration_seconds'] > 8
            results[name]['original_track'] = 'full reference, muted, first'

(reports / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(results, indent=2))
raise SystemExit(0 if all(v.get('returncode') == v.get('expected_returncode', 0) for v in results.values()) else 1)
