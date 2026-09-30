"""Strict lightweight distribution assembly; stale optional engines never leak in."""
from pathlib import Path
import hashlib
import json
import shutil


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def verify_manifests(package):
    """Reject partial/copied-while-changing packs before modifying branding."""
    engines = Path(package)/'engines'
    count = 0
    for manifest in engines.glob('*/build-manifest.json'):
        metadata = json.loads(manifest.read_text(encoding='utf-8'))
        for name, expected in metadata.get('files',{}).items():
            path = (manifest.parent/Path(name.replace('\\','/'))).resolve()
            if not path.is_relative_to(manifest.parent.resolve()) or not path.is_file() or sha(path) != expected:
                raise ValueError('Runtime integrity check failed: ' + str(path))
            count += 1
    return count


def runtime_packs(platform):
    if platform not in ('win32', 'darwin'):
        raise ValueError('No standard distribution for this platform')
    gpu = 'metal' if platform == 'darwin' else 'vulkan'
    return tuple(f'{runtime}-{device}' for runtime in ('whisper_cpp', 'audio_cpp')
                 for device in ('cpu', gpu)) + ('python_worker',)


def copy_runtime_packs(source, package, platform):
    """Copy only allowed packs into a fresh destination; never delete source caches."""
    destination = Path(package)/'engines'
    if destination.exists():
        raise ValueError('Runtime staging directory must be fresh')
    destination.mkdir(parents=True)
    for name in runtime_packs(platform):
        src = Path(source)/name
        if not (src/'build-manifest.json').is_file():
            raise ValueError('Missing runtime manifest: ' + name)
        shutil.copytree(src, destination/name, symlinks=True)
    verify_manifests(package)
    audit_lightweight(package)


def audit_lightweight(package):
    """Fail packaging if CUDA/CT2/faster-whisper artifacts or fonts appear."""
    forbidden = ('cublas', 'cudnn', 'cudart', 'ggml-cuda', 'ctranslate2', 'faster_whisper', 'faster-whisper', 'nvidia')
    count = 0
    for path in Path(package).rglob('*'):
        if not path.is_file():
            continue
        name = path.name.casefold()
        parts = [part.casefold() for part in path.relative_to(package).parts]
        heavy_package = any(part in {'ctranslate2', 'faster_whisper', 'nvidia'} or part.startswith(('ctranslate2-', 'faster_whisper-')) and part.endswith('.dist-info') for part in parts)
        if (heavy_package or name.endswith(('.ttf', '.otf', '.ttc', '.woff', '.woff2')) or
                ('engines' in path.parts and (any(name.startswith(x) for x in forbidden)
                 or any('cuda' in part.lower() for part in path.relative_to(package).parts)))):
            raise ValueError('Forbidden standard-distribution artifact: ' + str(path))
        count += 1
    worker = Path(package)/'engines/python_worker/build-manifest.json'
    if worker.is_file():
        packages = json.loads(worker.read_text(encoding='utf-8')).get('packages', {})
        if set(packages) != {'sherpa-onnx', 'sherpa-onnx-core'}:
            raise ValueError('Standard worker must contain sherpa-onnx only')
    return {'files_checked': count, 'cuda_bundled': False, 'ctranslate2_bundled': False,
            'faster_whisper_bundled': False}
