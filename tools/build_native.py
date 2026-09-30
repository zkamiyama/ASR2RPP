"""Build native engines in CI. No model weights are bundled in the Windows app."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ['ASR2RPP_NATIVE_WORK']) if os.getenv('ASR2RPP_NATIVE_WORK') else ROOT / 'build' / 'native'
ENGINES = ROOT / 'engines'
LOCK = json.loads((ROOT / 'native/versions.json').read_text(encoding='utf-8'))
SOURCES = {name: (entry['repository'], entry['commit'], entry['target'])
           for name, entry in LOCK.items() if isinstance(entry, dict)}


def recipe_digest():
    paths = [Path(__file__), ROOT/'native/versions.json', ROOT/'native/pcm_plan.h',
             ROOT/'native/whisper_regions.cpp', ROOT/'native/whisper/CMakeLists.txt']
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
    for key in ('ASR2RPP_CUDA_ARCHS', 'CMAKE_GENERATOR', 'ASR2RPP_MSVC_TOOLSET'):
        h.update((key + '=' + os.getenv(key, '')).encode())
    return h.hexdigest()


def run(*args, cwd=None):
    print('+', *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def preserve_restored_samples(source: Path):
    """Move CI-only cached fixtures aside before an initial source checkout.

    Cache archives contain samples but not .git. An initial checkout would
    otherwise fail on untracked samples/jfk.mp3. Never force checkout or remove
    a developer's working files; this path requires both explicit CI switches.
    """
    if (os.getenv('GITHUB_ACTIONS') != 'true'
            or os.getenv('ASR2RPP_REUSE_VERIFIED_NATIVE') != '1'
            or (source / '.git').exists()):
        return None
    samples = source / 'samples'
    if not samples.is_dir():
        return None
    backup = Path(tempfile.mkdtemp(prefix=source.name + '-restored-', dir=source.parent))
    shutil.move(str(samples), str(backup / 'samples'))
    print('Preserved restored CI fixtures:', backup, flush=True)
    return backup


def build(name, backend='cpu'):
    repo, revision, target = SOURCES[name]
    extension = recipe_digest()
    destination = ENGINES / (name + '-' + backend)
    required = [target + ('.exe' if os.name == 'nt' else '')]
    if name == 'whisper_cpp' and backend == 'cpu':
        required.append('whisper-vad-speech-segments' + ('.exe' if os.name == 'nt' else ''))
    if name == 'audio_cpp' and backend == 'cpu':
        required.append('audiocpp_gguf' + ('.exe' if os.name == 'nt' else ''))
    if name == 'whisper_cpp':
        required.append('asr2rpp-whisper-regions' + ('.exe' if os.name == 'nt' else ''))
    # OPT-IN for CI restored caches only. Normal builds are never skipped.
    manifest = destination / 'build-manifest.json'
    if os.getenv('ASR2RPP_REUSE_VERIFIED_NATIVE') == '1' and manifest.is_file():
        try:
            prior = json.loads(manifest.read_text(encoding='utf-8'))
            reusable = (prior['commit'] == revision and prior['backend'] == backend
                        and prior['repository'] == repo and prior['files']
                        and prior.get('build_recipe_sha256') == extension
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
    preserve_restored_samples(source)
    run('git', 'init', source)
    if subprocess.run(['git', 'remote', 'get-url', 'origin'], cwd=source, capture_output=True).returncode:
        run('git', 'remote', 'add', 'origin', 'https://github.com/' + repo, cwd=source)
    run('git', 'fetch', '--depth', '1', 'origin', revision, cwd=source)
    run('git', 'checkout', '--detach', 'FETCH_HEAD', cwd=source)
    # At the pinned audio.cpp revision, ggml is vendored. Its only submodule is
    # the optional server frontend (SSH URL), which a standalone CLI does not use.
    if name != 'audio_cpp':
        run('git', 'submodule', 'update', '--init', '--recursive', '--depth', '1', cwd=source)
    dirty = subprocess.check_output(['git', 'diff', '--name-only'], cwd=source, text=True).strip()
    if dirty:
        raise RuntimeError('Upstream source must be unmodified; use a clean build directory: ' + dirty)
    if backend not in {'cpu', 'vulkan', 'cuda'}:
        raise ValueError(f'Unsupported packaged backend: {backend}')
    output = source / ('build-' + backend + ('-' + os.environ['ASR2RPP_BUILD_SUFFIX'] if os.getenv('ASR2RPP_BUILD_SUFFIX') else ''))
    vulkan = 'ON' if backend == 'vulkan' else 'OFF'
    cuda = 'ON' if backend == 'cuda' else 'OFF'
    flags = ['-DCMAKE_BUILD_TYPE=Release', '-DGGML_NATIVE=OFF',
             f'-DGGML_CUDA={cuda}', '-DGGML_METAL=OFF', f'-DGGML_VULKAN={vulkan}']
    if backend == 'cuda':
        flags.append('-DCMAKE_CUDA_ARCHITECTURES=' + os.getenv('ASR2RPP_CUDA_ARCHS', '86;89'))
        if os.name == 'nt' and os.getenv('ASR2RPP_CUDA_HOST_COMPAT') == '1':
            flags.append('-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler')
    if name == 'audio_cpp':
        flags += ['-DAUDIOCPP_DEPLOYMENT_BUILD=ON',
                  '-DAUDIOCPP_MODEL_SET=custom',
                  '-DAUDIOCPP_MODELS=' + ','.join(LOCK['audio_cpp']['models']),
                  '-DENGINE_ENABLE_NATIVE_CPU=OFF',
                  f'-DENGINE_ENABLE_CUDA={cuda}', f'-DENGINE_ENABLE_VULKAN={vulkan}',
                  '-DENGINE_BUILD_TESTS=OFF', '-DENGINE_BUILD_EXAMPLES=OFF',
                  '-DAUDIOCPP_BUILD_SERVER_FRONTENDS=OFF']
    else:
        flags += ['-DWHISPER_BUILD_EXAMPLES=ON', '-DWHISPER_BUILD_TESTS=OFF', '-DWHISPER_BUILD_SERVER=OFF', '-DWHISPER_CURL=OFF']
    if os.name == 'nt' and os.getenv('CMAKE_GENERATOR', '').lower() != 'ninja':
        flags += ['-A', 'x64']
        if os.getenv('ASR2RPP_MSVC_TOOLSET'):
            flags += ['-T', os.environ['ASR2RPP_MSVC_TOOLSET']]
    project = source
    if name == 'whisper_cpp':
        project = ROOT / 'native/whisper'
        flags.append('-DASR2RPP_WHISPER_SOURCE=' + str(source))
    run('cmake', '-S', project, '-B', output, *flags)
    run('cmake', '--build', output, '--config', 'Release', '--target', target, '--parallel', os.getenv('ASR2RPP_BUILD_JOBS', '4'))
    if name == 'whisper_cpp':
        run('cmake', '--build', output, '--config', 'Release', '--target', 'asr2rpp-whisper-regions', '--parallel', os.getenv('ASR2RPP_BUILD_JOBS', '4'))
    if name == 'whisper_cpp' and backend == 'cpu':
        run('cmake', '--build', output, '--config', 'Release', '--target', 'whisper-vad-speech-segments', '--parallel', os.getenv('ASR2RPP_BUILD_JOBS', '4'))
    if name == 'audio_cpp' and backend == 'cpu':
        run('cmake', '--build', output, '--config', 'Release', '--target', 'audiocpp_gguf', '--parallel', os.getenv('ASR2RPP_BUILD_JOBS', '4'))
    destination = ENGINES / (name + '-' + backend)
    destination.mkdir(parents=True, exist_ok=True)
    binary_name = target + ('.exe' if os.name == 'nt' else '')
    binaries = list(output.rglob(binary_name))
    if not binaries:
        raise RuntimeError('Missing built executable: ' + binary_name)
    binary = binaries[0]
    shutil.copy2(binary, destination / binary.name)
    if name == 'whisper_cpp':
        helper = 'asr2rpp-whisper-regions' + ('.exe' if os.name == 'nt' else '')
        built = list(output.rglob(helper))
        if not built: raise RuntimeError('Missing public API helper')
        shutil.copy2(built[0], destination / helper)
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
    if os.name == 'nt' and backend == 'cuda':
        cuda_root = Path(os.environ['CUDA_PATH'])
        for pattern in ('cublas64_*.dll', 'cublasLt64_*.dll', 'cudart64_*.dll'):
            matches = list((cuda_root/'bin').glob(pattern))
            if not matches: raise RuntimeError('Missing CUDA redistributable: ' + pattern)
            for dll in matches: shutil.copy2(dll, destination/dll.name)
        for file in (cuda_root/'EULA.txt', cuda_root/'doc/EULA.txt'):
            if file.is_file(): shutil.copy2(file, destination/'NVIDIA-CUDA-EULA.txt')
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
    metadata = {'repository': repo, 'commit': revision, 'backend': backend, 'files': {},
                'build_recipe_sha256': extension, 'upstream_modified': False,
                'pcm_plan_version': 1 if name == 'whisper_cpp' else None,
                'cuda_architectures': os.getenv('ASR2RPP_CUDA_ARCHS', '86;89') if backend == 'cuda' else None,
                'cmake_flags': flags}
    for file in destination.rglob('*'):
        if file.is_file() and file != destination / 'build-manifest.json':
            metadata['files'][str(file.relative_to(destination))] = hashlib.sha256(file.read_bytes()).hexdigest()
    (destination / 'build-manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    run(destination / binary.name, '--help')
    if name == 'audio_cpp':
        data = subprocess.check_output([str(destination / binary.name), '--list-loaders', '--json'], text=True, encoding='utf-8')
        (destination/'capabilities.json').write_text(data, encoding='utf-8')


if __name__ == '__main__':
    for spec in sys.argv[1:] or ['whisper_cpp:cpu', 'audio_cpp:cpu']:
        name, separator, backend = spec.partition(':')
        build(name, backend or 'cpu')
