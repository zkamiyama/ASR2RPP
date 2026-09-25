"""Install FFmpeg into user data on demand; never bundle it with ASR2RPP releases."""
from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen
import json
import os
import shutil
import sys
import threading
import uuid
import zipfile

from .catalog import Cancelled, cache_root, data_root, digest

PROVIDER = "BtbN/FFmpeg-Builds"
BASE_URL = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest"
WINDOWS_ASSET = "ffmpeg-master-latest-win64-lgpl-shared.zip"
CHECKSUM_ASSET = "checksums.sha256"
USER_AGENT = "ASR2RPP/0.1"
_DOWNLOAD_LOCK = threading.Lock()


def runtime_root() -> Path:
    return data_root() / "runtime" / "ffmpeg"


def manifest_path() -> Path:
    return runtime_root() / "installed.json"


def installed_ffmpeg() -> Path | None:
    manifest = manifest_path()
    if not manifest.is_file():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        relative = PurePosixPath(str(data["executable"]))
        if relative.is_absolute() or ".." in relative.parts:
            return None
        path = runtime_root().joinpath(*relative.parts)
        return path if path.is_file() else None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _check_cancel(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled("Cancelled by user")


def _checksum_for(text: str, filename: str) -> str:
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2 and parts[1].lstrip("*") == filename:
            value = parts[0].lower()
            if len(value) == 64 and all(ch in "0123456789abcdef" for ch in value):
                return value
    raise ValueError(f"Checksum not found for {filename}")


def _download(url: str, destination: Path, progress=None, cancel=None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response, destination.open("wb") as handle:
        total = int(response.headers.get("Content-Length", "0") or 0)
        done = 0
        last_reported = -1
        while True:
            _check_cancel(cancel)
            block = response.read(1024 * 1024)
            if not block:
                break
            handle.write(block)
            done += len(block)
            if progress:
                mib = done // (1024 * 1024)
                if mib != last_reported and (mib % 8 == 0 or (total and done >= total)):
                    last_reported = mib
                    if total:
                        progress(f"FFmpeg download: {done / 1024**2:.0f}/{total / 1024**2:.0f} MB")
                    else:
                        progress(f"FFmpeg download: {done / 1024**2:.0f} MB")


def _extract_windows_bin(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    bin_dir = destination / "bin"
    bin_dir.mkdir()
    with zipfile.ZipFile(archive) as package:
        members = []
        for info in package.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Unsafe path in FFmpeg archive")
            if info.is_dir() or len(path.parts) < 2 or path.parts[-2] != "bin":
                continue
            members.append((info, path.name))
        if not any(name.lower() == "ffmpeg.exe" for _, name in members):
            raise ValueError("Downloaded FFmpeg archive has no bin/ffmpeg.exe")
        for info, name in members:
            target = bin_dir / name
            with package.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    executable = bin_dir / "ffmpeg.exe"
    if not executable.is_file():
        raise ValueError("FFmpeg extraction failed")
    return executable


def ensure_ffmpeg(progress=None, cancel=None) -> Path:
    """Return an installed Windows FFmpeg, downloading the latest LGPL shared build if absent."""
    if sys.platform != "win32":
        raise FileNotFoundError("Automatic FFmpeg download is currently supported on Windows only")

    with _DOWNLOAD_LOCK:
        existing = installed_ffmpeg()
        if existing:
            return existing

        _check_cancel(cancel)
        root = runtime_root()
        root.mkdir(parents=True, exist_ok=True)
        downloads = cache_root() / "downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        token = uuid.uuid4().hex
        checksum_file = downloads / f"ffmpeg-checksums-{token}.txt"
        archive = downloads / f"ffmpeg-{token}.zip"
        staging = root / f".install-{token}"
        try:
            if progress:
                progress("FFmpeg not found; downloading latest LGPL build")
            _download(f"{BASE_URL}/{CHECKSUM_ASSET}", checksum_file, progress, cancel)
            expected = _checksum_for(checksum_file.read_text(encoding="utf-8"), WINDOWS_ASSET)
            _download(f"{BASE_URL}/{WINDOWS_ASSET}", archive, progress, cancel)
            actual = digest(archive)
            if actual != expected:
                raise ValueError(f"FFmpeg SHA-256 mismatch: expected {expected}, got {actual}")

            final_dir = root / expected[:16]
            if not final_dir.exists():
                _extract_windows_bin(archive, staging)
                try:
                    os.replace(staging, final_dir)
                except FileExistsError:
                    shutil.rmtree(staging, ignore_errors=True)

            executable = final_dir / "bin" / "ffmpeg.exe"
            if not executable.is_file():
                raise ValueError("Installed FFmpeg executable is missing")
            manifest_path().write_text(json.dumps({
                "provider": PROVIDER,
                "asset": WINDOWS_ASSET,
                "sha256": expected,
                "source": f"{BASE_URL}/{WINDOWS_ASSET}",
                "executable": executable.relative_to(root).as_posix(),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            if progress:
                progress(f"FFmpeg ready: {executable}")
            return executable
        finally:
            checksum_file.unlink(missing_ok=True)
            archive.unlink(missing_ok=True)
            shutil.rmtree(staging, ignore_errors=True)
