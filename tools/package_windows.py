"""Build, brand and smoke-test the complete Windows ZIP before publication."""
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from build_icons import build_icons
from brand_windows import apply_icons

ROOT = Path(__file__).resolve().parent.parent


def run(*args, cwd=None):
    subprocess.run(list(map(str, args)), cwd=cwd or ROOT, check=True)


def copy_licenses(package):
    target = package / 'licenses'
    target.mkdir(exist_ok=True)
    for name in ('PySide6', 'PySide6-Essentials', 'PySide6-Addons', 'shiboken6', 'pyinstaller', 'numpy', 'safetensors'):
        dist = distribution(name)
        for item in dist.files or []:
            if any(x in str(item).lower() for x in ('license', 'copying', 'notice')) and str(item).lower().endswith(('.txt', '.md', '.rst', 'license', 'copying')):
                source = Path(dist.locate_file(item))
                if source.is_file():
                    destination = target / name / str(item).replace('..', '_')
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
    material = target/'material-icons'
    material.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'assets/icons/LICENSE.txt', material/'LICENSE.txt')
    shutil.copy2(ROOT/'assets/branding/NOTICE.md', material/'NOTICE.md')


os.environ['PYTHONUTF8'] = '1'
reports = ROOT / 'reports'
reports.mkdir(exist_ok=True)
icons = build_icons(ROOT/'build/icons', reports/'executable-icons.png')
run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed',
    '--name', 'ASR2RPP', '--icon', icons['app'],
    '--hidden-import', 'numpy', '--hidden-import', 'safetensors.numpy', '--hidden-import', 'PySide6.QtSvg',
    '--hidden-import', 'asr2rpp.ffmpeg_runtime',
    '--add-data', 'assets:assets', 'launcher.py')
run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onefile', '--console',
    '--name', 'asr2rpp-cli', '--icon', icons['cli'], '--exclude-module', 'PySide6',
    '--hidden-import', 'numpy', '--hidden-import', 'safetensors.numpy', '--hidden-import', 'asr2rpp.ffmpeg_runtime',
    'cli_launcher.py')
package = ROOT / 'dist/ASR2RPP'
shutil.copy2(ROOT / 'dist/asr2rpp-cli.exe', package / 'asr2rpp-cli.exe')
shutil.copytree(ROOT / 'engines', package / 'engines', dirs_exist_ok=True)
required_native = [
    package / 'engines/whisper_cpp-cpu/whisper-cli.exe',
    package / 'engines/whisper_cpp-cpu/whisper-vad-speech-segments.exe',
    package / 'engines/whisper_cpp-vulkan/whisper-cli.exe',
    package / 'engines/audio_cpp-cpu/audiocpp_cli.exe',
    package / 'engines/audio_cpp-cpu/audiocpp_gguf.exe',
    package / 'engines/audio_cpp-vulkan/audiocpp_cli.exe',
]
missing = [str(path) for path in required_native if not path.is_file()]
if missing:
    raise RuntimeError('Missing packaged native runtime(s): ' + ', '.join(missing))
icon_records = apply_icons(package, ROOT/'build/icons', reports/'executable-icons.json')
shutil.copy2(reports/'executable-icons.json', package/'icon-manifest.json')
copy_licenses(package)
if any(p.name.casefold() == 'ffmpeg.exe' for p in package.rglob('*')):
    raise RuntimeError('FFmpeg must not be bundled in the Windows ZIP')
for name in ('README.md', 'README.ja.md', 'LICENSE', 'THIRD_PARTY.md'):
    shutil.copy2(ROOT/name, package/name)
shutil.copytree(ROOT/'docs', package/'docs', dirs_exist_ok=True)
# Keep the notice's relative source links valid in the expanded ZIP as well.
shutil.copytree(ROOT/'assets/branding', package/'assets/branding', dirs_exist_ok=True)
shutil.copytree(ROOT/'assets/icons', package/'assets/icons', dirs_exist_ok=True)
shutil.copytree(ROOT / 'models', package / 'models', dirs_exist_ok=True)
if (package/'model-templates').exists() or (package/'_internal/models').exists():
    raise RuntimeError('Duplicate bundled model definitions are not permitted')
for file in package.rglob('*'):
    if file.is_file() and file.suffix.lower() in {'.ttf', '.otf', '.ttc', '.woff', '.woff2'}:
        file.unlink()
run(package / 'asr2rpp-cli.exe', 'doctor')
run(package / 'asr2rpp-cli.exe', 'models', 'list')
listed = subprocess.check_output([str(package/'asr2rpp-cli.exe'), 'models', 'list', '--json'], text=True, encoding='utf-8')
assert len(json.loads(listed)) >= 7
for entry in json.loads(listed):
    file = package/'models'/f"{entry['id']}.toml"
    assert Path(entry['definition']).resolve() == file.resolve()
    assert entry['sha256'] == hashlib.sha256(file.read_bytes()).hexdigest()
run(package / 'ASR2RPP.exe', '--self-test-lifecycle', reports/'frozen-lifecycle')
lifecycle = json.loads((reports/'frozen-lifecycle/summary.json').read_text())
assert lifecycle['passed'] and Path(lifecycle['model_directory']).resolve() == (package/'models').resolve()
assert lifecycle['window_icon_valid']
(reports/'catalog-source.json').write_text(listed, encoding='utf-8')
run(package / 'asr2rpp-cli.exe', 'models', 'install', 'whisper-base')
fixture = ROOT / 'build/native/whisper_cpp/samples/jfk.wav'
if not fixture.exists():
    raise RuntimeError('Missing public upstream speech fixture')
run(package / 'asr2rpp-cli.exe', 'run', fixture, '--asr-device', 'cpu', '--asr-language', 'en',
    '--output-dir', reports / 'frozen-asr')
if not list((reports / 'frozen-asr').glob('*.rpp')):
    raise RuntimeError('Frozen CLI did not produce an RPP')
process = subprocess.Popen([str(package / 'ASR2RPP.exe')], cwd=package)
time.sleep(5)
if process.poll() is not None:
    raise RuntimeError('Frozen GUI exited during startup')
process.terminate()
process.wait(timeout=10)
(reports/'frozen-smoke.json').write_text(json.dumps({
    'cli_doctor': 'passed', 'cli_models_list': 'passed', 'cli_model_download': 'passed', 'cli_asr_to_rpp': 'passed',
    'gui_startup_5s': 'passed', 'gui_interaction': 'frozen GO-STOP-GO, errors and completed-item preservation passed',
    'icons_verified_executables': len(icon_records), 'window_icon': 'passed',
    'ffmpeg_bundled': False, 'code_signing': 'unsigned', 'private_media_used': False}), encoding='utf-8')
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
(package/'version.json').write_text(json.dumps({'version': '0.1.0-preview', 'commit': commit,
    'platform': 'windows-x64', 'native_backends': ['cpu', 'vulkan'], 'minimum_cpu': 'AVX2',
    'model_weights_included': False, 'ffmpeg_bundled': False, 'model_definitions': 'exe-adjacent/models',
    'icons_verified_executables': len(icon_records), 'documentation': ['README.md', 'README.ja.md'],
    'ffmpeg_resolution': 'custom-or-PATH-or-verified-user-download'}, indent=2), encoding='utf-8')
archive = Path(shutil.make_archive(str(ROOT/'dist/ASR2RPP-Windows-x64'), 'zip', ROOT/'dist', 'ASR2RPP'))
with archive.open('rb') as handle:
    archive_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
(archive.parent/'SHA256SUMS.txt').write_text(archive_hash + '  ' + archive.name + '\n', encoding='utf-8')
