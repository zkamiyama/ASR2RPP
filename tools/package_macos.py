"""Build a self-contained Apple Silicon .app plus CLI, CPU/Metal and sherpa.

No Developer ID is assumed: this preview is ad-hoc signed and not notarized.
The final ZIP is verified after relocation, including signatures and dependencies.
"""
from pathlib import Path
from importlib.metadata import distribution
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile

from package_worker import build_worker
from portable_runtime import copy_runtime_packs, audit_lightweight, verify_manifests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from asr2rpp import __version__

ROOT = Path(__file__).resolve().parents[1]
MACHO_MAGICS = {b'\xcf\xfa\xed\xfe', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf',
                b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca', b'\xca\xfe\xba\xbf'}


def run(*args, **kwargs):
    print('+', *map(str, args), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def machos(root):
    for path in sorted(root.rglob('*')):
        if path.is_file() and not path.is_symlink():
            with path.open('rb') as stream:
                if stream.read(4) in MACHO_MAGICS:
                    yield path


def audit_machos(root):
    records = []
    for path in machos(root):
        arches = subprocess.check_output(['lipo', '-archs', str(path)], text=True).strip().split()
        if 'arm64' not in arches:
            raise ValueError('Missing arm64 slice: ' + str(path))
        deps = subprocess.check_output(['otool', '-L', str(path)], text=True).splitlines()[1:]
        names = [line.strip().split(' (', 1)[0] for line in deps]
        for name in names:
            if not name.startswith(('/System/Library/', '/usr/lib/', '@rpath/', '@loader_path/', '@executable_path/')):
                raise ValueError('Nonportable Mach-O dependency: ' + str(path) + ': ' + name)
        records.append(dict(file=str(path.relative_to(root)), architectures=arches, dependencies=names))
    if not records:
        raise ValueError('No Mach-O files found')
    return records


def make_icon():
    from PySide6.QtCore import Qt, QRectF
    from PySide6.QtGui import QImage, QPainter
    from PySide6.QtSvg import QSvgRenderer
    directory = ROOT/'build/mac-icons/app.iconset'
    directory.mkdir(parents=True, exist_ok=True)
    renderer = QSvgRenderer(str(ROOT/'assets/branding/app.svg'))
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            pixels = size * scale
            image = QImage(pixels, pixels, QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.transparent)
            painter = QPainter(image)
            renderer.render(painter, QRectF(0, 0, pixels, pixels))
            painter.end()
            suffix = '@2x' if scale == 2 else ''
            if not image.save(str(directory/f'icon_{size}x{size}{suffix}.png')):
                raise OSError('Could not render macOS icon')
    target = directory.with_suffix('.icns')
    run('iconutil', '-c', 'icns', directory, '-o', target)
    return target


def copy_documents(resources):
    for name in ('README.md', 'README.ja.md', 'THIRD_PARTY.md', 'LICENSE'):
        shutil.copy2(ROOT/name, resources/name)
    for name in ('models', 'docs'):
        shutil.copytree(ROOT/name, resources/name)
    # These relative notice links should also work inside Resources.
    for name in ('branding', 'icons'):
        target = resources/'assets'/name
        target.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT/'assets'/name, target, dirs_exist_ok=True)
    for name in ('PySide6', 'PySide6-Essentials', 'PySide6-Addons', 'shiboken6', 'numpy', 'safetensors', 'pyinstaller', 'opentimelineio'):
        dist = distribution(name)
        for entry in dist.files or []:
            if any(x in str(entry).lower() for x in ('license', 'copying', 'notice')):
                path = Path(dist.locate_file(entry))
                if path.is_file() and path.stat().st_size < 4*1024**2:
                    target = resources/'licenses'/name/str(entry).replace('..', '_')
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)


def main():
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Build on an Apple Silicon macOS runner, not by cross compilation')
    os.chdir(ROOT)
    os.environ['PYTHONUTF8'] = '1'
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    icon = make_icon()
    build_worker()
    base = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--target-arch', 'arm64',
            '--exclude-module', 'faster_whisper', '--exclude-module', 'ctranslate2', '--exclude-module', 'nvidia',
            '--collect-all', 'opentimelineio', '--hidden-import', 'numpy', '--hidden-import', 'safetensors.numpy', '--hidden-import', 'asr2rpp.ffmpeg_runtime']
    run(*base, '--onedir', '--windowed', '--name', 'ASR2RPP', '--icon', icon,
        '--osx-bundle-identifier', 'io.github.zkamiyama.asr2rpp',
        '--hidden-import', 'PySide6.QtSvg', '--add-data', 'assets:assets', 'launcher.py')
    run(*base, '--onefile', '--console', '--name', 'asr2rpp-cli', '--exclude-module', 'PySide6', 'cli_launcher.py')
    # The .app is self-contained; all backend searches derive from Contents/Resources.
    app = ROOT/'dist/ASR2RPP.app'
    resources = app/'Contents/Resources'
    cli = app/'Contents/MacOS/asr2rpp-cli'
    shutil.copy2(ROOT/'dist/asr2rpp-cli', cli)
    copy_runtime_packs(ROOT/'engines', resources, 'darwin')
    copy_documents(resources)
    for path in app.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.ttf', '.otf', '.ttc', '.woff', '.woff2'}:
            path.unlink()
    records = audit_machos(app)
    (reports/'macos-machos.json').write_text(json.dumps(records, indent=2))
    # Sign additional executable code; PyInstaller has already signed its frameworks.
    for path in machos(resources/'engines'):
        run('codesign', '--force', '--sign', '-', '--timestamp=none', path)
    run('codesign', '--force', '--sign', '-', '--timestamp=none', cli)
    # Signing changes bytes. Refresh only checksums, keeping upstream build metadata.
    for manifest in (resources/'engines').glob('*/build-manifest.json'):
        data = json.loads(manifest.read_text())
        data['files'] = {p.relative_to(manifest.parent).as_posix(): digest(p)
                         for p in manifest.parent.rglob('*') if p.is_file() and p != manifest}
        data['code_signing'] = 'ad-hoc'
        manifest.write_text(json.dumps(data, indent=2))
    verify_manifests(resources)
    audit = audit_lightweight(resources)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    meta = dict(version=__version__, commit=commit, platform='macos-arm64', minimum_macos='14.0',
                output_formats=['rpp','otio','json'], export_schema_version=1,
                native_backends=['cpu', 'metal'], cuda_bundled=False, ctranslate2_bundled=False,
                faster_whisper_bundled=False, model_weights_included=False, ffmpeg_bundled=False,
                model_definitions='app/Contents/Resources/models', model_schema_versions=[1, 2],
                worker_protocol=1, code_signing='ad-hoc; not notarized', upstream_cli_modified=False)
    (resources/'version.json').write_text(json.dumps(meta, indent=2))
    (reports/'distribution-audit.json').write_text(json.dumps(audit, indent=2))
    run('codesign', '--force', '--sign', '-', '--timestamp=none', app)
    run('codesign', '--verify', '--deep', '--strict', app)
    run(sys.executable, ROOT/'tools/smoke_outputs.py', '--cli', cli, '--report', reports/'exports')
    # ditto preserves app symlinks and executable permissions; ordinary zip extraction may not.
    archive = ROOT/'dist/ASR2RPP-macOS-arm64.zip'
    run('ditto', '-c', '-k', '--keepParent', app, archive)
    from package_sample import make_sample
    sample = make_sample(ROOT/'build/sample')
    with zipfile.ZipFile(archive, 'a', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(sample, 'examples/sample.wav')
        bundle.write(sample.with_name('README.txt'), 'examples/README.txt')
    (ROOT/'dist/SHA256SUMS-macOS.txt').write_text(digest(archive) + '  ' + archive.name + '\n')
    run(sys.executable, ROOT/'tools/smoke_macos.py', '--archive', archive,
        '--report', reports/'macos-portable')


if __name__ == '__main__':
    main()
