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
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def independent_alignment_case(definition):
    """Exercise the shipped default-off 30B pipeline, not experimental native CTC."""
    return (definition.get('task') == 'asr'
            and definition.get('provider', definition.get('runtime')) == 'audio_cpp'
            and definition.get('family') == 'nemotron_transcribe'
            and definition.get('defaults', {}).get('request', {}).get('diarization') is False)


def case_arguments(model_id, definition, device, fixtures, output):
    task = definition['task']
    language = definition.get('defaults', {}).get('language', 'ja')
    english = language in ('en', 'en-US') or model_id == 'voxtral-mini-realtime-q4'
    fixture = fixtures['en' if english else 'ja']
    selected_device = 'cpu' if definition.get('provider', definition.get('runtime')) == 'sherpa_onnx' else device
    base = ['run', str(fixture), '--threads', '4', '--format', 'rpp,otio,json', '--output-dir', str(output)]
    if task == 'asr':
        base += ['--asr', model_id, '--asr-device', selected_device]
        if independent_alignment_case(definition):
            base += ['--align', 'qwen-forced-aligner', '--align-device', selected_device,
                     '--timing', 'alignment', '--vad-before-alignment', '--speaker-source', 'none']
        else:
            base += ['--timing', 'auto']
        if not independent_alignment_case(definition) and definition.get('capabilities', {}).get('speakers'):
            base += ['--speaker-source', 'native']
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


def case_dependencies(model_id, definition):
    models = [model_id] + (['whisper-base'] if definition['task'] != 'asr' else [])
    if independent_alignment_case(definition):
        models.append('qwen-forced-aligner')
    return models


def validate_output(output, task, text_only, source_hash, duration, speakers_required=False,
                    alignment_required=False):
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
    if text_only and not alignment_required and any(u['method'] != 'vad_segment' for u in units):
        raise ValueError('Text-only ASR did not use honest VAD-region timing')
    if alignment_required and any(u['method'] != 'forced_alignment' for u in units):
        raise ValueError('Independent aligner did not supply every final interval')
    if (task == 'align' or alignment_required) and not any(u['method'] == 'forced_alignment' for u in units):
        raise ValueError('Aligner did not produce aligned units')
    if (task == 'diar' or speakers_required) and not any(u.get('speaker') for u in units):
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


def validation_metadata(assets, archive=None, *, frozen=False, physical_hardware=False, hardware=''):
    """Hardware is an explicit operator attestation, never inferred from a GPU name."""
    platform = {'win32': 'windows', 'darwin': 'macos', 'linux': 'linux'}.get(sys.platform, 'unknown')
    if physical_hardware:
        if not frozen or archive is None or platform not in ('windows', 'macos') or not hardware.strip():
            raise ValueError('Physical-hardware attestation requires a desktop package, exact archive and hardware description')
        if any(os.environ.get(key, '').lower() not in ('', '0', 'false')
               for key in ('CI', 'GITHUB_ACTIONS', 'TF_BUILD')):
            raise ValueError('CI cannot attest physical desktop hardware')
    metadata = dict(platform=platform, physical_hardware=physical_hardware, hardware=hardware.strip())
    version_path = assets / 'version.json'
    if frozen and not version_path.is_file():
        raise ValueError('Frozen package is missing version.json')
    if version_path.is_file():
        raw = version_path.read_bytes()
        version = json.loads(raw)
        metadata.update(package_commit=version['commit'], package_version=version['version'])
        if frozen and archive is not None:
            member = ('ASR2RPP.app/Contents/Resources/version.json' if assets.name == 'Resources'
                      else 'ASR2RPP/version.json')
            with zipfile.ZipFile(archive) as zipped:
                if zipped.namelist().count(member) != 1 or zipped.read(member) != raw:
                    raise ValueError('Extracted package metadata does not match the exact archive')
    if archive is not None:
        metadata['package_sha256'] = digest(archive)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--package', type=Path, help='Expanded Windows directory or macOS .app')
    group.add_argument('--source-root', type=Path, help='Developer check, not a frozen-package test')
    parser.add_argument('--archive', type=Path, help='Exact ZIP that was extracted for this test')
    parser.add_argument('--fixture-ja', type=Path, required=True)
    parser.add_argument('--fixture-en', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--weights-dir', type=Path, required=True)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--device', choices=('cpu', 'vulkan', 'metal'), default='cpu')
    parser.add_argument('--models', nargs='*', help='Default: every shipped TOML, including optional stages')
    parser.add_argument('--resume', action='store_true', help='Resume the identical package/fixtures in an existing report')
    parser.add_argument('--install', action='store_true', help='Explicitly allow downloading the selected weights')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--physical-hardware', action='store_true',
                        help='Operator attests this is a physical desktop, not a VM/CI runner; requires --package, --archive and --hardware')
    parser.add_argument('--hardware', default='', help='Observed physical computer/GPU description, not an inferred claim')
    args = parser.parse_args()
    if args.resume and not (args.package and args.archive):
        raise ValueError('Resume requires a frozen package and its exact --archive')
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=args.resume)
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
    machine_metadata = validation_metadata(assets, args.archive, frozen=bool(args.package),
        physical_hardware=args.physical_hardware, hardware=args.hardware)
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
    summary.update(machine_metadata)
    summary_file = report / 'summary.json'
    if args.resume and summary_file.exists():
        previous = json.loads(summary_file.read_text(encoding='utf-8'))
        for key in ('validation_mode', 'package_commit', 'package_version', 'package_sha256', 'device',
                    'fixtures', 'definition_hashes', 'platform', 'physical_hardware', 'hardware'):
            if previous.get(key) != summary.get(key):
                raise ValueError('Resume identity mismatch: ' + key)
        summary['cases'] = previous.get('cases', {})
    def save():
        summary_file.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    for model_id in selected:
        if summary['cases'].get(model_id, {}).get('passed'):
            print(model_id, 'already verified for this exact package', flush=True)
            continue
        d = definitions[model_id]
        output = report / model_id
        if output.exists():
            raise ValueError('Incomplete previous case exists; keep it and select a new report directory: ' + str(output))
        fixture, device, argv = case_arguments(model_id, d, args.device, fixtures, output)
        key = 'ja' if fixture == fixtures['ja'] else 'en'
        independent_alignment = independent_alignment_case(d)
        speaker_source = argv[argv.index('--speaker-source') + 1] if '--speaker-source' in argv else 'auto'
        record = dict(task=d['task'], requested_device=device, fixture=key, passed=False,
                      selected_pipeline=dict(asr=argv[argv.index('--asr') + 1],
                          align=argv[argv.index('--align') + 1] if '--align' in argv else None,
                          vad_before_alignment='--vad-before-alignment' in argv,
                          asr_input='vad_regions' if independent_alignment else 'pipeline_default',
                          speaker_source=speaker_source))
        try:
            if args.install:
                install = case_dependencies(model_id, d)
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
                          hashes[key], durations[key], speaker_source in ('native', 'diarizer'),
                          alignment_required=independent_alignment))
            from smoke_outputs import check_exports
            record['exports'] = check_exports(next(output.glob('*.json')))
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
