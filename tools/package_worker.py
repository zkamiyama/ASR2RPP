"""Freeze optional ASR dependencies in a separate, replaceable worker."""
from pathlib import Path
from importlib import metadata
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ('sherpa-onnx', 'sherpa-onnx-core')
EXCLUDED = ('faster_whisper', 'ctranslate2', 'nvidia', 'onnxruntime', 'av', 'tokenizers', 'huggingface_hub')

def build_worker(icon=None):
    args = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir',
            '--console', '--name', 'asr2rpp-worker',
            '--exclude-module', 'PySide6', '--exclude-module', 'torch',
            '--exclude-module', 'tensorflow', '--exclude-module', 'transformers']
    if icon is not None:
        args += ['--icon', str(icon)]
    if sys.platform == 'darwin':
        args += ['--target-arch', 'arm64']
    for name in EXCLUDED:
        args += ['--exclude-module', name]
    for name in ('sherpa_onnx',):
        args += ['--collect-all', name]
    for name in PACKAGES:
        args += ['--copy-metadata', name]
    args.append('worker_launcher.py')
    subprocess.run(args, cwd=ROOT, check=True)
    source = ROOT/'dist/asr2rpp-worker'
    destination = ROOT/'engines/python_worker'
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source,destination,symlinks=True)
    licenses = destination/'licenses'
    licenses.mkdir()
    for name in PACKAGES:
        dist = metadata.distribution(name)
        for item in dist.files or []:
            if any(x in str(item).lower() for x in ('license','notice','copying')):
                path = Path(dist.locate_file(item))
                if path.is_file() and path.stat().st_size < 4*1024**2:
                    target = licenses/name/str(item).replace('..','_')
                    target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(path,target)
    manifest = {'schema_version':1,'protocol':1,'packages':{n:metadata.version(n) for n in PACKAGES},'files':{}}
    for path in destination.rglob('*'):
        if path.is_file():
            with path.open('rb') as handle:
                manifest['files'][path.relative_to(destination).as_posix()] = hashlib.file_digest(handle,'sha256').hexdigest()
    (destination/'build-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    subprocess.run([str(destination/('asr2rpp-worker.exe' if os.name == 'nt' else 'asr2rpp-worker')),'--capabilities'],check=True)
    return destination
