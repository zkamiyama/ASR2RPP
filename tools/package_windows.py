"""Build and smoke-test the portable Windows preview on a Windows runner."""
import hashlib
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

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


os.environ['PYTHONUTF8'] = '1'
run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed',
    '--name', 'ASR2RPP', '--hidden-import', 'numpy', '--hidden-import', 'safetensors.numpy', '--hidden-import', 'PySide6.QtSvg',
    '--hidden-import', 'asr2rpp.ffmpeg_runtime',
    '--add-data', 'models:models', '--add-data', 'assets:assets', 'launcher.py')
run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onefile', '--console',
    '--name', 'asr2rpp-cli', '--exclude-module', 'PySide6',
    '--hidden-import', 'numpy', '--hidden-import', 'safetensors.numpy', '--hidden-import', 'asr2rpp.ffmpeg_runtime',
    '--add-data', 'models:models', 'cli_launcher.py')
package = ROOT / 'dist/ASR2RPP'
shutil.copy2(ROOT / 'dist/asr2rpp-cli.exe', package / 'asr2rpp-cli.exe')
shutil.copytree(ROOT / 'engines', package / 'engines', dirs_exist_ok=True)
required_native = [
    package / 'engines/whisper_cpp-cpu/whisper-cli.exe',
    package / 'engines/whisper_cpp-vulkan/whisper-cli.exe',
    package / 'engines/audio_cpp-cpu/audiocpp_cli.exe',
    package / 'engines/audio_cpp-cpu/audiocpp_gguf.exe',
    package / 'engines/audio_cpp-vulkan/audiocpp_cli.exe',
]
missing_native = [str(path) for path in required_native if not path.is_file()]
if missing_native:
    raise RuntimeError('Missing packaged native runtime(s): ' + ', '.join(missing_native))
copy_licenses(package)
# FFmpeg is deliberately not redistributed. On Windows the application resolves
# an existing PATH/custom binary or downloads a verified LGPL build into user data.
if list(package.rglob('ffmpeg.exe')):
    raise RuntimeError('FFmpeg must not be bundled in the Windows application ZIP')
for name in ('README.md', 'LICENSE', 'THIRD_PARTY.md'):
    if (ROOT / name).exists():
        shutil.copy2(ROOT / name, package / name)
shutil.copytree(ROOT / 'models', package / 'model-templates', dirs_exist_ok=True)
# Use OS fonts; never redistribute a system or bundled font file.
for file in package.rglob('*'):
    if file.is_file() and file.suffix.lower() in {'.ttf', '.otf', '.ttc', '.woff', '.woff2'}:
        file.unlink()
reports = ROOT / 'reports'
reports.mkdir(exist_ok=True)
run(package / 'asr2rpp-cli.exe', 'doctor')
run(package / 'asr2rpp-cli.exe', 'models', 'list')
run(package / 'asr2rpp-cli.exe', 'models', 'install', 'whisper-base')
# A public upstream fixture, not the user's private media.
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
(reports / 'frozen-smoke.json').write_text(json.dumps({
    'cli_doctor': 'passed', 'cli_models_list': 'passed', 'cli_model_download': 'passed', 'cli_asr_to_rpp': 'passed',
    'gui_startup_5s': 'passed', 'gui_interaction': 'tested from same source, not frozen',
    'ffmpeg_bundled': False, 'ffmpeg_resolution': 'custom-or-PATH-or-verified-user-download',
    'code_signing': 'unsigned', 'private_media_used': False}), encoding='utf-8')
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
(package / 'version.json').write_text(json.dumps({'version': '0.1.0-preview', 'commit': commit,
    'platform': 'windows-x64', 'native_backends': ['cpu', 'vulkan'], 'minimum_cpu': 'AVX2',
    'model_weights_included': False, 'ffmpeg_bundled': False,
    'ffmpeg_resolution': 'custom-or-PATH-or-verified-user-download'}, indent=2), encoding='utf-8')
archive = Path(shutil.make_archive(str(ROOT / 'dist/ASR2RPP-Windows-x64'), 'zip', ROOT / 'dist', 'ASR2RPP'))
(archive.parent / 'SHA256SUMS.txt').write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n', encoding='utf-8')
