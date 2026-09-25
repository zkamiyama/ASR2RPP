"""Bounded CI trial: original -> GGML F16/F32 -> pinned whisper.cpp CPU.

Uses only public upstream files; model weights and input audio are not uploaded.
JFK is an English fixture: this is a loading/inference smoke, NOT a Japanese CER test.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
import wave
import convert_anime_whisper as conv
from inspect_ggml import inspect, compare


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--whisper-cli', type=Path, required=True)
    p.add_argument('--work', type=Path, default=Path('build/anime-trial'))
    p.add_argument('--reports', type=Path, default=Path('reports/anime-trial'))
    a = p.parse_args()
    work, reports = a.work, a.reports
    reports.mkdir(parents=True, exist_ok=True)
    assert importlib.util.find_spec('torch') is None, 'Trial environment must not have torch'
    assert importlib.util.find_spec('transformers') is None, 'Trial environment must not have transformers'
    source = work / 'source'
    acquisition = conv.acquire(source)
    report = {'torch_installed': False, 'transformers_installed': False,
              'acquisition': acquisition, 'conversion': {}, 'smoke': {},
              'limitations': ['No reproduction audio from user', 'Public English fixture, not Japanese accuracy benchmark',
                              'CPU only; no Vulkan/RTX 4080 quality or performance claim']}
    for precision in ('f16', 'f32'):
        started = time.monotonic()
        output = work / f'ggml-anime-whisper-{precision}.bin'
        detail = conv.convert(source, output, precision)
        decoded = inspect(output)
        assert decoded['tensors'] == detail['tensors'], 'GGML read-back failed'
        (reports / f'{precision}-conversion.json').write_text(json.dumps(detail, indent=2), encoding='utf-8')
        report['conversion'][precision] = {k: v for k, v in detail.items() if k not in ('tensors', 'acquisition', 'config')}
        report['conversion'][precision]['seconds'] = time.monotonic() - started
    repo = 'Aratako/anime-whisper-ggml'
    with urlopen(f'https://huggingface.co/api/models/{repo}?blobs=true', timeout=60) as r:
        info = json.load(r)
    asset = next(f for f in info['siblings'] if f['rfilename'] == 'ggml-anime-whisper.bin')
    existing = work / 'existing.ggml.bin'
    conv.download(f"https://huggingface.co/{repo}/resolve/{info['sha']}/ggml-anime-whisper.bin",
                  existing, asset['lfs']['sha256'])
    report['existing'] = {'repository': repo, 'revision': info['sha'], 'sha256': conv.digest(existing)}
    report['comparison'] = compare(existing, work / 'ggml-anime-whisper-f16.bin')
    # Check whether changed F32 auxiliary tensors equal the upstream converter's
    # F32 -> F16 -> F32 rounding. Do not assume differences imply better output.
    import hashlib
    import numpy as np
    from safetensors import safe_open
    original_index = inspect(existing)
    cfg = json.loads((source/'config.json').read_text())
    matches, mismatches = [], []
    with safe_open(source/'model.safetensors', framework='numpy') as sf:
        for key, (name, shape) in conv.specification(cfg).items():
            if name not in report['comparison']['different_tensors']:
                continue
            arr = sf.get_tensor(key).astype(np.float16).astype(np.float32)
            expected = original_index['tensors'][name]
            legacy = conv.prepare_tensor(arr, name, 'f16')
            good = hashlib.sha256(memoryview(legacy).cast('B')).hexdigest() == expected['sha256']
            (matches if good else mismatches).append(name)
    report['comparison']['differences_explained_by_legacy_rounding'] = matches
    report['comparison']['other_tensor_differences'] = mismatches
    # Public upstream fixture, clipped to four seconds to bound CPU work.
    jfk = work/'jfk.wav'
    commit = 'a664346ea5c6dddff3e61a2b7b32dd4514613f50'
    conv.download(f'https://raw.githubusercontent.com/ggml-org/whisper.cpp/{commit}/samples/jfk.wav', jfk)
    fixture = work/'fixture.wav'
    with wave.open(str(jfk), 'rb') as inp, wave.open(str(fixture), 'wb') as out:
        out.setparams(inp.getparams())
        out.writeframes(inp.readframes(inp.getframerate()*4))
    report['fixture'] = {'source': f'ggml-org/whisper.cpp@{commit}:samples/jfk.wav',
                         'input_seconds': 4, 'input_language': 'English', 'requested_language': 'ja',
                         'sha256': conv.digest(fixture), 'purpose': 'model loading and inference smoke only'}
    (reports/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    for label, model in [('existing', existing), ('f16', work/'ggml-anime-whisper-f16.bin'), ('f32', work/'ggml-anime-whisper-f32.bin')]:
        for mode in ('timed', 'text'):
            prefix = reports / f'{label}-{mode}'
            argv = [str(a.whisper_cli.resolve()), '-m', str(model.resolve()), '-f', str(fixture.resolve()),
                    '-l', 'ja', '-t', '2', '-bs', '1', '-bo', '1', '-nf', '-ng', '-otxt',
                    '-of', str(prefix.resolve())]
            argv += ['-ojf'] if mode == 'timed' else ['-oj', '-nt']
            started = time.monotonic()
            with prefix.with_suffix('.log').open('w', encoding='utf-8') as log:
                completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=300)
            result = {'command': argv, 'returncode': completed.returncode, 'seconds': time.monotonic()-started}
            text = prefix.with_suffix('.txt')
            if text.is_file():
                result['text'] = text.read_text(encoding='utf-8').strip()
            report['smoke'][f'{label}-{mode}'] = result
            (reports/'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            if completed.returncode or not text.is_file():
                raise RuntimeError(f'Native smoke failed: {label}-{mode}')
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    print(json.dumps({k: v for k, v in report.items() if k != 'acquisition'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
