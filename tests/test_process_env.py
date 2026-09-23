from pathlib import Path
from asr2rpp.adapters import process_environment


def test_system_ffmpeg_does_not_inherit_native_libraries(monkeypatch):
    monkeypatch.setenv('LD_LIBRARY_PATH', '/bundled/old/libraries')
    monkeypatch.delenv('LD_LIBRARY_PATH_ORIG', raising=False)
    env = process_environment(Path('/usr/bin/ffmpeg'))
    assert 'LD_LIBRARY_PATH' not in env


def test_original_library_environment_is_restored(monkeypatch):
    monkeypatch.setenv('LD_LIBRARY_PATH_ORIG', '/original/lib')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/pyinstaller/lib')
    assert process_environment(Path('/usr/bin/ffmpeg'))['LD_LIBRARY_PATH'] == '/original/lib'
