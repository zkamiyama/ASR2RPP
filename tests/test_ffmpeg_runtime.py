import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest


class FakeResponse(io.BytesIO):
    def __init__(self, data: bytes):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data))}

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
        return False


def make_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as package:
        package.writestr("ffmpeg-master-latest-win64-lgpl-shared/bin/ffmpeg.exe", b"fake-ffmpeg")
        package.writestr("ffmpeg-master-latest-win64-lgpl-shared/bin/avcodec.dll", b"fake-dll")
        package.writestr("ffmpeg-master-latest-win64-lgpl-shared/LICENSE.txt", b"LGPL runtime license")
        package.writestr("ffmpeg-master-latest-win64-lgpl-shared/doc/extra.txt", b"not installed")
    return buffer.getvalue()


def test_checksum_parser_accepts_release_manifest():
    from asr2rpp.ffmpeg_runtime import WINDOWS_ASSET, _checksum_for

    digest = "a" * 64
    text = f"{digest}  {WINDOWS_ASSET}\n"
    assert _checksum_for(text, WINDOWS_ASSET) == digest


def test_verified_ffmpeg_download_is_cached_in_user_data(tmp_path, monkeypatch):
    from asr2rpp import ffmpeg_runtime as runtime

    archive = make_archive()
    expected = hashlib.sha256(archive).hexdigest()
    checksums = f"{expected}  {runtime.WINDOWS_ASSET}\n".encode()
    requests = []

    def fake_urlopen(request, timeout=0):
        url = request.full_url
        requests.append(url)
        if url.endswith(runtime.CHECKSUM_ASSET):
            return FakeResponse(checksums)
        if url.endswith(runtime.WINDOWS_ASSET):
            return FakeResponse(archive)
        raise AssertionError(url)

    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime, "urlopen", fake_urlopen)
    monkeypatch.setattr(runtime, "runtime_root", lambda: tmp_path / "runtime")
    monkeypatch.setattr(runtime, "cache_root", lambda: tmp_path / "cache")

    progress = []
    executable = runtime.ensure_ffmpeg(progress.append)
    assert executable.read_bytes() == b"fake-ffmpeg"
    assert (executable.parent / "avcodec.dll").read_bytes() == b"fake-dll"
    assert not (executable.parent.parent / "doc").exists()
    assert (executable.parent.parent / "LICENSE.txt").read_bytes() == b"LGPL runtime license"
    manifest = json.loads((tmp_path / "runtime" / "installed.json").read_text(encoding="utf-8"))
    assert manifest["provider"] == runtime.PROVIDER
    assert manifest["sha256"] == expected
    assert len(requests) == 2
    assert any("FFmpeg ready:" in line for line in progress)

    # A second resolution is fully local and does not hit the network.
    assert runtime.ensure_ffmpeg() == executable
    assert len(requests) == 2


def test_bad_ffmpeg_checksum_is_rejected(tmp_path, monkeypatch):
    from asr2rpp import ffmpeg_runtime as runtime

    archive = make_archive()
    checksums = f"{'0' * 64}  {runtime.WINDOWS_ASSET}\n".encode()

    def fake_urlopen(request, timeout=0):
        return FakeResponse(checksums if request.full_url.endswith(runtime.CHECKSUM_ASSET) else archive)

    monkeypatch.setattr(runtime.sys, "platform", "win32")
    monkeypatch.setattr(runtime, "urlopen", fake_urlopen)
    monkeypatch.setattr(runtime, "runtime_root", lambda: tmp_path / "runtime")
    monkeypatch.setattr(runtime, "cache_root", lambda: tmp_path / "cache")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runtime.ensure_ffmpeg()
    assert runtime.installed_ffmpeg() is None


def test_windows_packager_does_not_bundle_ffmpeg():
    source = Path("tools/package_windows.py").read_text(encoding="utf-8")
    assert "imageio_ffmpeg" not in source
    assert "engines/ffmpeg" not in source
    assert "FFmpeg must not be bundled" in source
