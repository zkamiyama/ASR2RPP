"""Relocate and test the exact signed Apple Silicon .app ZIP.

Metal hardware availability is measured, not inferred from successful compilation.
Set ASR2RPP_REQUIRE_METAL_GPU=1 on a hardware runner to require real GPU inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen

from portable_runtime import audit_lightweight, verify_manifests
from package_macos import audit_machos
from metal_smoke_policy import PROBE_SOURCE, select_metal_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    report.mkdir(parents=True, exist_ok=False)
    extracted = report/'relocated'
    subprocess.run(['ditto', '-x', '-k', str(args.archive.resolve()), str(extracted)], check=True)
    app = extracted/'ASR2RPP.app'
    cli = app/'Contents/MacOS/asr2rpp-cli'
    resources = app/'Contents/Resources'
    subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True)
    audit_machos(app)
    verify_manifests(resources)
    audit = audit_lightweight(resources)
    env = os.environ.copy()
    env.update(ASR2RPP_HOME=str(report/'home'), ASR2RPP_CACHE_DIR=str(report/'cache'),
               ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP='1', QT_QPA_PLATFORM='offscreen')
    # Isolate model caches/settings and prevent library paths from masking missing dependencies.
    env.pop('ASR2RPP_WEIGHTS_DIR', None)
    env.pop('PYTHONPATH', None)
    for key in list(env):
        if key.startswith(('DYLD_', 'CUDA_PATH', 'VULKAN_SDK')):
            env.pop(key, None)
    env['PATH'] = '/usr/bin:/bin:/usr/sbin:/sbin'
    results = {'platform': 'macos-arm64', 'signature_verified': True, 'distribution': audit, 'cases': {}, 'skipped': {}}
    def save():
        (report/'summary.json').write_text(json.dumps(results, indent=2))
    def run(name, argv, timeout=600):
        start = time.monotonic()
        with (report/(name+'.log')).open('w') as log:
            completed = subprocess.run(list(map(str, argv)), cwd=report, env=env,
                                       stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        results['cases'][name] = dict(returncode=completed.returncode, seconds=time.monotonic()-start)
        save()
        if completed.returncode:
            raise RuntimeError(f'{name} failed; see {name}.log')
    run('doctor', [cli, 'doctor'])
    run('lifecycle', [app/'Contents/MacOS/ASR2RPP', '--self-test-lifecycle', report/'lifecycle'])
    fixture = report/'jfk.wav'
    url = 'https://raw.githubusercontent.com/ggml-org/whisper.cpp/a664346ea5c6dddff3e61a2b7b32dd4514613f50/samples/jfk.wav'
    with urlopen(url, timeout=30) as response:
        fixture.write_bytes(response.read())
    original = hashlib.sha256(fixture.read_bytes()).hexdigest()
    run('models', [cli, 'models', 'install', 'whisper-base', 'reazonspeech-k2', 'qwen3-asr-06b'])
    for name, model, timing, language in [('whisper-cpu-native','whisper-base','native','en'),
                                         ('whisper-cpu-vad','whisper-base','vad','en'),
                                         ('reazon-cpu-vad','reazonspeech-k2','vad','ja'),
                                         ('qwen-cpu-vad','qwen3-asr-06b','vad','English')]:
        run(name, [cli, 'run', fixture, '--asr', model, '--asr-device', 'cpu', '--timing', timing,
                   '--asr-language', language, '--output-dir', report/name])
        if not list((report/name).glob('*.rpp')):
            raise RuntimeError('Missing RPP: ' + name)
    # Measure the capabilities ggml actually needs, not just device presence.
    source = report/'metal-probe.mm'
    source.write_text(PROBE_SOURCE)
    probe = report/'metal-probe'
    subprocess.run(['xcrun', 'clang++', str(source), '-framework', 'Foundation', '-framework', 'Metal', '-o', str(probe)], check=True)
    device = subprocess.run([str(probe)], capture_output=True, text=True, check=True)
    capabilities = json.loads(device.stdout)
    runnable, skipped = select_metal_cases(capabilities)
    results['metal'] = dict(capabilities, inference_verified_models=[], coverage_complete=False)
    results['skipped'] = {model+'-metal-vad': reason for model, reason in skipped.items()}
    save()
    for model in runnable:
        name = model+'-metal-vad'
        run(name, [cli, 'run', fixture, '--asr', model, '--asr-device', 'metal', '--timing', 'vad',
                   '--asr-language', 'English' if model.startswith('qwen') else 'en',
                   '--output-dir', report/name])
        logs = '\n'.join(p.read_text(errors='replace') for p in (report/name).rglob('*.log'))
        if not any(token in logs.lower() for token in ('using metal', 'ggml_metal', 'ggml_backend_metal')):
            raise RuntimeError('Metal execution evidence missing')
        if not list((report/name).glob('*.rpp')):
            raise RuntimeError('Missing Metal RPP: ' + name)
        results['metal']['inference_verified_models'].append(model)
        save()
    results['metal']['coverage_complete'] = not skipped
    save()
    if skipped and os.getenv('ASR2RPP_REQUIRE_METAL_GPU') == '1':
        raise RuntimeError('This hardware job requires both Metal inference cases: ' + str(skipped))
    results['source_unchanged'] = hashlib.sha256(fixture.read_bytes()).hexdigest() == original
    results['passed'] = results['source_unchanged'] and all(v['returncode'] == 0 for v in results['cases'].values())
    save()
    print(json.dumps(results, indent=2))
    return 0 if results['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
