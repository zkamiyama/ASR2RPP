import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onedir', '--windowed',
    '--name', 'ASR2RPP', '--add-data', 'models:models', 'launcher.py')
run(sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--onefile', '--console',
    '--name', 'asr2rpp-cli', '--exclude-module', 'PySide6', '--add-data', 'models:models', 'cli_launcher.py')
package = ROOT / 'dist/ASR2RPP'
shutil.copy2(ROOT / 'dist/asr2rpp-cli.exe', package / 'asr2rpp-cli.exe')
shutil.copytree(ROOT / 'engines', package / 'engines', dirs_exist_ok=True)
ffmpeg = Path(imageio_ffmpeg.get_ffmpeg_exe())
ffmpeg_dir = package / 'engines/ffmpeg'
ffmpeg_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(ffmpeg, ffmpeg_dir / 'ffmpeg.exe')
for name in ('README.md', 'LICENSE', 'THIRD_PARTY.md'):
    if (ROOT / name).exists():
        shutil.copy2(ROOT / name, package / name)
# Keep the model templates readily visible as well as the bundled first-run copies.
shutil.copytree(ROOT / 'models', package / 'model-templates', dirs_exist_ok=True)
run(package / 'asr2rpp-cli.exe', 'models', 'list')
# Offscreen GUI smoke happens in tests. Also launch the frozen GUI and verify it stays alive.
process = subprocess.Popen([str(package / 'ASR2RPP.exe')], cwd=package)
time.sleep(5)
if process.poll() is not None:
    raise RuntimeError('Frozen GUI exited during startup')
process.terminate()
process.wait(timeout=10)
(ROOT / 'reports').mkdir(exist_ok=True)
(ROOT / 'reports/frozen-smoke.json').write_text(json.dumps({'cli_models_list': 'passed', 'gui_startup_5s': 'passed',
    'gui_interaction': 'tested from same source, not frozen', 'code_signing': 'unsigned'}), encoding='utf-8')
shutil.make_archive(str(ROOT / 'dist/ASR2RPP-Windows-x64'), 'zip', ROOT / 'dist', 'ASR2RPP')
