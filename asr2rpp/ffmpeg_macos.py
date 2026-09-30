"""Pinned Apple Silicon FFmpeg bootstrap, downloaded separately from the app.

Only a verified standalone executable and its notices are extracted; no wheel
or Python package is installed. The user may supply another FFmpeg instead.
"""
from pathlib import Path
import os
import platform
import tempfile
import threading
import zipfile

from .atomic import file_lock, json_replace
from .catalog import checkpoint, digest

URL = ('https://files.pythonhosted.org/packages/40/5c/'
       'f3d8a657d362cc93b81aab8feda487317da5b5d31c0e1fdfd5e986e55d17/'
       'imageio_ffmpeg-0.6.0-py3-none-macosx_11_0_arm64.whl')
SHA256 = 'b1ae3173414b5fc5f538a726c4e48ea97edc0d2cdc11f103afee655c463fa742'
MEMBER = 'imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1'
LICENSE_MEMBER = 'imageio_ffmpeg-0.6.0.dist-info/LICENSE'


def ensure_ffmpeg(progress=None, cancel=None):
    from .ffmpeg_runtime import runtime_root, installed_ffmpeg, manifest_path, _download
    if platform.machine().lower() not in {'arm64', 'aarch64'}:
        raise FileNotFoundError('Automatic macOS FFmpeg acquisition requires Apple Silicon')
    cancel = cancel or threading.Event()
    root = runtime_root()
    with file_lock(root/'install.lock', cancel):
        existing = installed_ffmpeg()
        if existing:
            return existing
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.mac-install-', dir=root) as temporary:
            work = Path(temporary)
            archive = work/'ffmpeg.whl'
            if progress:
                progress('FFmpeg — downloading verified Apple Silicon runtime (21 MB)')
            _download(URL, archive, progress, cancel)
            if digest(archive) != SHA256:
                raise ValueError('FFmpeg SHA-256 mismatch')
            stage = work/'runtime'
            stage.mkdir()
            executable = stage/'ffmpeg'
            with zipfile.ZipFile(archive) as package:
                entry = package.getinfo(MEMBER)
                if entry.file_size > 128*1024**2:
                    raise ValueError('FFmpeg executable exceeds extraction limit')
                with package.open(entry) as source, executable.open('xb') as target:
                    while block := source.read(1024**2):
                        checkpoint(cancel)
                        target.write(block)
                (stage/'imageio-LICENSE.txt').write_bytes(package.read(LICENSE_MEMBER))
            executable.chmod(0o755)
            (stage/'NOTICE.txt').write_text(
                'FFmpeg 7.1 arm64 from imageio-ffmpeg 0.6.0.\n'
                'The FFmpeg executable is GPL-enabled; the wrapper license is separate.\n'
                'Run ffmpeg -L for its license. https://ffmpeg.org/legal.html\n'
                'Build provenance: https://github.com/imageio/imageio-ffmpeg\n', encoding='utf-8')
            checksum = digest(executable)
            final = root/(SHA256[:16]+'-arm64')
            checkpoint(cancel)
            if final.exists():
                # A verified earlier extraction can survive an interrupted manifest write.
                if not (final/'ffmpeg').is_file() or digest(final/'ffmpeg') != checksum:
                    raise ValueError('Existing FFmpeg runtime is corrupt; select a different runtime or remove its cache')
            else:
                os.replace(stage, final)
            executable = final/'ffmpeg'
            json_replace(manifest_path(), dict(provider='imageio/imageio-ffmpeg', source=URL,
                asset=MEMBER, sha256=SHA256, executable_sha256=checksum,
                executable=executable.relative_to(root).as_posix()))
            return executable
