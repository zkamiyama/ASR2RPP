"""Exercise shipped model TOMLs through the application, using public/synthetic WAVs.

A passing case means successful acquisition, nonempty bounded transcript, the
expected stage/timing output and an unchanged source. It is not an accuracy or
speed benchmark. GPU failure is never silently replaced by a CPU success.
The summary omits transcripts and user paths; raw diagnostics stay in --report.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tomllib
import wave


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def case_arguments(model_id, definition, device, fixtures, output):
    task = definition['task']
    language = definition.get('defaults', {}).get('language', 'ja')
    english = language in ('en', 'en-US') or model_id == 'voxtral-mini-realtime-q4'
    fixture = fixtures['en' if english else 'ja']
    selected_device = 'cpu' if definition.get('provider', definition.get('runtime')) == 'sherpa_onnx' else device
    base = ['run', str(fixture), '--threads', '4', '--output-dir', str(output)]
    if task == 'asr':
        base += ['--asr', model_id, '--asr-device', selected_device, '--timing', 'auto']
    else:
        base += ['--asr', 'whisper-base', '--asr-device', 'cpu']
        if task == 'align':
            base += ['--align', model_id, '--align-device', selected_device,
                     '--timing', 'alignment', '--vad-before-alignment']
        elif task == 'diar':
            base += ['--diar', model_id, '--diar-device', selected_device,
                     '--speaker-source', 'diarizer']
        elif task == 'sep':
            base += ['--preprocess', model_id, '--preprocess-device', selected_device,
                     '--rpp-audio', 'processed']
        else:
            raise ValueError('Unsupported catalog task: ' + task)
    return fixture, selected_device, base


def validate_output(output, task, text_only, source_hash, duration):
    manifests = list(output.glob('*.asr2rpp/manifest.json'))
    projects = list(output.glob('*.rpp'))
    if len(manifests) != 1 or len(projects) != 1:
        raise ValueError('Expected exactly one completed report and RPP')
    manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
    if (manifest.get('status') != 'completed' or manifest.get('source_unchanged') is not True
            or manifest.get('source_sha256') != source_hash):
        raise ValueError('Incomplete report or changed source')
    transcript = json.loads((manifests[0].parent / 'transcript.json').read_text(encoding='utf-8'))
    units = transcript['units']
    if not units or not any(u['text'].strip() for u in units):
        raise ValueError('Empty transcript')
    for unit in units:
        a, b = unit['start'], unit['end']
        if not all(isinstance(t, (int, float)) and math.isfinite(t) for t in (a, b)):
            raise ValueError('Non-finite interval')
        if not 0 <= a < b <= duration + 0.001:
            raise ValueError('Transcript interval escapes the source')
    if text_only and any(u['method'] != 'vad_segment' for u in units):
        raise ValueError('Text-only ASR did not use honest VAD-region timing')
    if task == 'align' and not any(u['method'] == 'forced_alignment' for u in units):
        raise ValueError('Aligner did not produce aligned units')
    if task == 'diar' and not any(u.get('speaker') for u in units):
        raise ValueError('Diarizer did not produce speaker labels')
    if task == 'sep' and not list(output.glob('*_vocals.wav')):
        raise ValueError('Separator did not preserve the processed reference')
    original = manifest.get('original_track', {})
    if original.get('name') != 'ORIGINAL' or original.get('muted') is not True:
        raise ValueError('Missing muted ORIGINAL reference track')
    return dict(unit_count=len(units), text_characters=sum(len(u['text']) for u in units),
                text_sha256=hashlib.sha256(''.join(u['text'] for u in units).encode()).hexdigest(),
                timestamp_source=manifest.get('timestamp_source'),
                methods=sorted({u['method'] for u in units}),
                speaker_count=len({u['speaker'] for u in units if u.get('speaker')}),
                source_unchanged=True, rpp_created=True)


def invoke(prefix, argv, cwd, env, logfile, timeout):
    started = time.monotonic()
    with logfile.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(prefix + list(map(str, argv)), cwd=cwd, env=env,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=os.name != 'nt')
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                               stdout=log, stderr=subprocess.STDOUT, check=False)
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=30)
            code = -1
    return dict(returncode=code, elapsed_seconds=round(time.monotonic() - started, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--package', type=Path, help='Expanded Windows directory or macOS .app')
    group.add_argument('--source-root', type=Path, help='Developer check, not a frozen-package test')
    parser.add_argument('--fixture-ja', type=Path, required=True)
    parser.add_argument('--fixture-en', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--device', choices=('cpu', 'vulkan', 'metal'), default='cpu')
    parser.add_argument('--models', nargs='*', help='Default: every shipped TOML, including optional stages')
    parser.add_argument('--install', action='store_true', help='Explicitly allow downloading the selected weights')
    parser.add_argument('--timeout', type=int, default=900)
    args = parser.parse_args()
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=False)
    if args.package:
        package = args.package.resolve()
        assets = package / 'Contents/Resources' if package.suffix == '.app' else package
        exe = package / 'Contents/MacOS/asr2rpp-cli' if package.suffix == '.app' else package / 'asr2rpp-cli.exe'
        prefix, cwd, mode = [str(exe)], report, 'frozen-package'
        from portable_runtime import audit_lightweight
        audit_lightweight(assets)
    else:
        assets = args.source_root.resolve()
        prefix, cwd, mode = [sys.executable, '-m', 'asr2rpp.cli'], assets, 'source-with-native-binaries'
    definitions = {p.stem: tomllib.loads(p.read_text(encoding='utf-8')) for p in (assets / 'models').glob('*.toml')}
    selected = args.models if args.models is not None else sorted(definitions)
    if not selected or set(selected) - definitions.keys():
        raise ValueError('No models or unknown model IDs')
    fixtures = {'ja': args.fixture_ja.resolve(), 'en': args.fixture_en.resolve()}
    hashes, durations = {}, {}
    for key, fixture in fixtures.items():
        hashes[key] = digest(fixture)
        with wave.open(str(fixture), 'rb') as wav:
            durations[key] = wav.getnframes() / wav.getframerate()
    env = os.environ.copy()
    env.update(ASR2RPP_HOME=str(report / 'home'), ASR2RPP_CACHE_DIR=str(report / 'cache'),
               ASR2RPP_WEIGHTS_DIR=str(args.weights_dir.resolve()), PYTHONUTF8='1',
               ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP='1')
    for key in list(env):
        if key.startswith(('CUDA_PATH', 'CUDNN', 'PYTHONPATH', 'VULKAN_SDK', 'DYLD_')):
            env.pop(key, None)
    if os.name == 'nt':
        env['PATH'] = str(Path(env.get('SystemRoot', 'C:/Windows')) / 'System32')
    summary = dict(schema_version=1, validation_mode=mode, device=args.device,
                   fixtures={k: dict(sha256=hashes[k], seconds=durations[k]) for k in fixtures},
                   definition_hashes={m: digest(assets / 'models' / (m + '.toml')) for m in selected},
                   cases={})
    version = assets / 'version.json'
    if version.is_file():
        summary['package_commit'] = json.loads(version.read_text())['commit']
    summary_file = report / 'summary.json'
    def save():
        summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for model_id in selected:
        d = definitions[model_id]
        output = report / model_id
        fixture, device, argv = case_arguments(model_id, d, args.device, fixtures, output)
        key = 'ja' if fixture == fixtures['ja'] else 'en'
        record = dict(task=d['task'], requested_device=device, fixture=key, passed=False)
        try:
            if args.install:
                install = [model_id] + (['whisper-base'] if d['task'] != 'asr' else [])
                record['acquisition'] = invoke(prefix, ['models', 'install', *install], cwd, env,
                                               report / (model_id + '-install.log'), 3600)
                if record['acquisition']['returncode']:
                    raise ValueError('Model acquisition failed')
            argv += ['--ffmpeg', str(Path(args.ffmpeg).resolve())]
            record.update(invoke(prefix, argv, cwd, env, report / (model_id + '.log'), args.timeout))
            if record['returncode']:
                raise ValueError('Inference failed; raw diagnostics retained')
            record.update(validate_output(output, d['task'],
                          d['task'] == 'asr' and d.get('capabilities', {}).get('timestamps') == 'none',
                          hashes[key], durations[key]))
            gpu_lines = []
            for log in output.rglob('*.log'):
                for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
                    if ('ggml_vulkan:' in line or 'using Vulkan' in line or 'using Metal' in line):
                        gpu_lines.append(line[:250])
            record['gpu_log_evidence'] = list(dict.fromkeys(gpu_lines))[:6]
            if device == 'vulkan' and not record['gpu_log_evidence']:
                raise ValueError('No Vulkan runtime evidence in the engine log')
            if digest(fixture) != hashes[key]:
                raise ValueError('Source bytes changed')
            record['passed'] = True
        except Exception as error:
            record['error'] = str(error)
        summary['cases'][model_id] = record
        save()
        print(model_id, json.dumps(record, ensure_ascii=False), flush=True)
    summary['source_unchanged'] = all(digest(fixtures[k]) == hashes[k] for k in fixtures)
    summary['passed'] = summary['source_unchanged'] and all(r['passed'] for r in summary['cases'].values())
    save()
    return 0 if summary['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
