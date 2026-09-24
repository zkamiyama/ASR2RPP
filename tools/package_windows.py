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
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent


def run(*args, cwd=None):
    subprocess.run(list(map(str, args)), cwd=cwd or ROOT, check=True)


def copy_licenses(package):
    target = package / 'licenses'
    target.mkdir(exist_ok=True)
    for name in ('PySide6', 'PySide6-Essentials', 'PySide6-Addons', 'shiboken6', 'pyinstaller', 'imageio-ffmpeg'):
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
    '--name', 'ASR2RPP', '--add-data', 'models:models', 'launcher.py')
run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onefile', '--console',
    '--name', 'asr2rpp-cli', '--exclude-module', 'PySide6', '--add-data', 'models:models', 'cli_launcher.py')
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
ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
ffmpeg_dir = package / 'engines/ffmpeg'
ffmpeg_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(ffmpeg, ffmpeg_dir / 'ffmpeg.exe')
copy_licenses(package)
for name in ('README.md', 'LICENSE', 'THIRD_PARTY.md'):
    if (ROOT / name).exists():
        shutil.copy2(ROOT / name, package / name)
shutil.copytree(ROOT / 'models', package / 'model-templates', dirs_exist_ok=True)
# Include license/build information of the precise bundled FFmpeg executable.
for option, name in [('-L', 'license.txt'), ('-buildconf', 'build-configuration.txt'), ('-version', 'version.txt')]:
    result = subprocess.run([str(ffmpeg_dir / 'ffmpeg.exe'), option], capture_output=True)
    (ffmpeg_dir / name).write_bytes(result.stdout + result.stderr)
# Use OS fonts; never redistribute a system or bundled font file.
for file in package.rglob('*'):
    if file.is_file() and file.suffix.lower() in {'.ttf', '.otf', '.ttc', '.woff', '.woff2'}:
        file.unlink()
reports = ROOT / 'reports'
reports.mkdir(exist_ok=True)
run(package / 'asr2rpp-cli.exe', 'models', 'list')
run(package / 'asr2rpp-cli.exe', 'models', 'install', 'whisper-base')
# A public upstream fixture, not the user's private media.
fixture = ROOT / 'build/native/whisper_cpp/samples/jfk.wav'
if not fixture.exists():
    raise RuntimeError('Missing public upstream speech fixture')
run(package / 'asr2rpp-cli.exe', 'run', fixture, '--asr-language', 'en',
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
    'cli_models_list': 'passed', 'cli_model_download': 'passed', 'cli_asr_to_rpp': 'passed',
    'gui_startup_5s': 'passed', 'gui_interaction': 'tested from same source, not frozen',
    'code_signing': 'unsigned', 'private_media_used': False}), encoding='utf-8')
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
(package / 'version.json').write_text(json.dumps({'version': '0.1.0-preview', 'commit': commit,
    'platform': 'windows-x64', 'native_backends': ['cpu', 'vulkan'], 'minimum_cpu': 'AVX2',
    'model_weights_included': False}, indent=2), encoding='utf-8')
archive = Path(shutil.make_archive(str(ROOT / 'dist/ASR2RPP-Windows-x64'), 'zip', ROOT / 'dist', 'ASR2RPP'))
(archive.parent / 'SHA256SUMS.txt').write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n', encoding='utf-8')
