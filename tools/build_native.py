"""Build native engines in CI. No model weights are bundled in the Windows app."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / 'build' / 'native'
ENGINES = ROOT / 'engines'
SOURCES = {
    'whisper_cpp': ('ggml-org/whisper.cpp', 'a664346ea5c6dddff3e61a2b7b32dd4514613f50', 'whisper-cli'),
    'audio_cpp': ('0xShug0/audio.cpp', '9bdd1d908bbd128e9eb405f5a8e38d0defb84c72', 'audiocpp_cli'),
}


def run(*args, cwd=None):
    print('+', *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def build(name, backend='cpu'):
    repo, revision, target = SOURCES[name]
    destination = ENGINES / (name + '-' + backend)
    required = [target + ('.exe' if os.name == 'nt' else '')]
    if name == 'whisper_cpp' and backend == 'cpu':
        required.append('whisper-vad-speech-segments' + ('.exe' if os.name == 'nt' else ''))
    if name == 'audio_cpp' and backend == 'cpu':
        required.append('audiocpp_gguf' + ('.exe' if os.name == 'nt' else ''))
    # OPT-IN for CI restored caches only. Normal builds are never skipped.
    manifest = destination / 'build-manifest.json'
    if os.getenv('ASR2RPP_REUSE_VERIFIED_NATIVE') == '1' and manifest.is_file():
        try:
            prior = json.loads(manifest.read_text(encoding='utf-8'))
            reusable = (prior['commit'] == revision and prior['backend'] == backend
                        and prior['repository'] == repo and prior['files']
                        and all(filename in prior['files'] for filename in required))
            for filename, expected in prior['files'].items():
                relative = Path(filename)
                if relative.is_absolute() or '..' in relative.parts:
                    reusable = False
                    break
                file = destination / relative
                if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                    reusable = False
                    break
            if reusable:
                print('Using manifest-verified cached runtime:', destination, flush=True)
                return
        except (KeyError, ValueError, OSError):
            pass
    source = WORK / name
    source.mkdir(parents=True, exist_ok=True)
    run('git', 'init', source)
    if subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=source, capture_output=True).returncode:
        run('git', 'remote', 'add', 'origin', 'https://github.com/' + repo, cwd=source)
    run('git', 'fetch', '--depth', '1', 'origin', revision, cwd=source)
    run('git', 'checkout', '--detach', 'FETCH_HEAD', cwd=source)
    # At the pinned audio.cpp revision, ggml is vendored. Its only submodule is
    # the optional server frontend (SSH URL), which a standalone CLI does not use.
    if name != 'audio_cpp':
        run('git', 'submodule', 'update', '--init', '--recursive', '--depth', '1', cwd=source)
    if backend not in {'cpu', 'vulkan'}:
        raise ValueError(f'Unsupported packaged backend: {backend}')
    output = source / ('build-' + backend)
    vulkan = 'ON' if backend == 'vulkan' else 'OFF'
    flags = ['-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF',
             '-DGGML_CUDA=OFF', '-DGGML_METAL=OFF', f'-DGGML_VULKAN={vulkan}']
    if name == 'audio_cpp':
        flags += ['-DAUDIOCPP_DEPLOYMENT_BUILD=ON',
                  '-DAUDIOCPP_MODEL_SET=custom',
                  '-DAUDIOCPP_MODELS=nemotron_asr,nemotron_3_diar,vibevoice_asr,qwen3_forced_aligner,roformer',
                  '-DENGINE_ENABLE_NATIVE_CPU=OFF',
                  '-DENGINE_ENABLE_CUDA=OFF', f'-DENGINE_ENABLE_VULKAN={vulkan}',
                  '-DENGINE_BUILD_TESTS=OFF', '-DENGINE_BUILD_EXAMPLES=OFF',
                  '-DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF']
    else:
        flags += ['-DWHISPER_BUILD_TESTS=OFF', '-DWHISPER_BUILD_SERVER=OFF', '-DWHISPER_CURL=OFF']
    if os.name == 'nt':
        flags += ['-A', 'x64']
    run('cmake', '-S', source, '-B', output, *flags)
    run('cmake', '--build', output, '--config', 'Release', '--target', target, '--parallel', '4')
    if name == 'whisper_cpp' and backend == 'cpu':
        run('cmake', '--build', output, '--config', 'Release', '--target', 'whisper-vad-speech-segments', '--parallel', '4')
    if name == 'audio_cpp' and backend == 'cpu':
        run('cmake', '--build', output, '--config', 'Release', '--target', 'audiocpp_gguf', '--parallel', '4')
    destination = ENGINES / (name + '-' + backend)
    destination.mkdir(parents=True, exist_ok=True)
    binary_name = target + ('.exe' if os.name == 'nt' else '')
    binaries = list(output.rglob(binary_name))
    if not binaries:
        raise RuntimeError('Missing built executable: ' + binary_name)
    binary = binaries[0]
    shutil.copy2(binary, destination / binary.name)
    if name == 'whisper_cpp' and backend == 'cpu':
        vad_name = 'whisper-vad-speech-segments' + ('.exe' if os.name == 'nt' else '')
        vad_binaries = list(output.rglob(vad_name))
        if not vad_binaries:
            raise RuntimeError('Missing built VAD helper: ' + vad_name)
        shutil.copy2(vad_binaries[0], destination / vad_name)
    if name == 'audio_cpp' and backend == 'cpu':
        converter_name = 'audiocpp_gguf' + ('.exe' if os.name == 'nt' else '')
        converters = list(output.rglob(converter_name))
        if not converters:
            raise RuntimeError('Missing built executable: ' + converter_name)
        shutil.copy2(converters[0], destination / converter_name)
    for suffix in ('*.dll', '*.so', '*.so.*', '*.dylib'):
        for library in output.rglob(suffix):
            shutil.copy2(library, destination / library.name)
    licenses = destination / 'licenses'
    licenses.mkdir(exist_ok=True)
    for file in source.rglob('*'):
        if file.is_file() and file.name.lower().startswith(('license', 'copying', 'notice')) and 'build' not in file.relative_to(source).parts:
            if file.suffix.lower() in {'', '.txt', '.md', '.rst'}:
                target_license = licenses / file.relative_to(source)
                target_license.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target_license)
    for filename in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'):
        if (source / filename).is_file():
            shutil.copy2(source / filename, destination / filename)
    if name == 'audio_cpp' and (source / 'model_specs').exists():
        shutil.copytree(source / 'model_specs', destination / 'model_specs', dirs_exist_ok=True)
    metadata = {'repository': repo, 'commit': revision, 'backend': backend, 'files': {}}
    for file in destination.rglob('*'):
        if file.is_file():
            metadata['files'][str(file.relative_to(destination))] = hashlib.sha256(file.read_bytes()).hexdigest()
    (destination / 'build-manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    run(destination / binary.name, '--help')


if __name__ == '__main__':
    for spec in sys.argv[1:] or ['whisper_cpp:cpu', 'audio_cpp:cpu']:
        name, separator, backend = spec.partition(':')
        build(name, backend or 'cpu')
