"""Deduplicate CUDA DLLs without changing code and record shared dependencies."""
from pathlib import Path
import hashlib
import json
import shutil


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def collect_cuda(package):
    engines = Path(package)/'engines'
    shared = engines/'cuda_runtime'
    moved = {}
    for path in sorted(engines.rglob('*.dll')):
        if shared in path.parents:
            continue
        name = path.name.lower()
        if not name.startswith(('cublas64_', 'cublaslt64_', 'cudart64_', 'cudnn')):
            continue
        shared.mkdir(exist_ok=True)
        destination = shared/path.name
        checksum = sha(path)
        if destination.exists():
            if sha(destination) != checksum:
                raise ValueError('Conflicting CUDA runtime DLL: ' + path.name)
            path.unlink()
        else:
            shutil.move(str(path),str(destination))
        moved[path.relative_to(engines).as_posix()] = dict(file=path.name,sha256=checksum)
    for manifest in engines.rglob('build-manifest.json'):
        meta = json.loads(manifest.read_text(encoding='utf-8'))
        dependencies = {}
        for name in list(meta.get('files',{})):
            key = (manifest.parent/Path(name.replace('\\','/'))).relative_to(engines).as_posix()
            if key in moved:
                dependencies[name] = moved[key]
                del meta['files'][name]
        if dependencies:
            meta['shared_cuda_runtime'] = dependencies
            manifest.write_text(json.dumps(meta,indent=2),encoding='utf-8')
    if moved:
        (shared/'manifest.json').write_text(json.dumps({'files':{p.name:sha(p) for p in shared.glob('*.dll')}},indent=2),encoding='utf-8')
    return moved


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
        for info in metadata.get('shared_cuda_runtime',{}).values():
            path = (engines/'cuda_runtime'/info['file']).resolve()
            if not path.is_relative_to((engines/'cuda_runtime').resolve()) or not path.is_file() or sha(path) != info['sha256']:
                raise ValueError('Shared CUDA dependency integrity check failed')
    return count
