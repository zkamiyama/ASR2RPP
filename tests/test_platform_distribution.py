"""Platform GPU policy, stale-runtime exclusion and macOS bundle layout."""
from pathlib import Path
import hashlib
import json
import sys
import pytest
from asr2rpp.platforms import backends, automatic_devices
from tools.portable_runtime import runtime_packs, copy_runtime_packs, audit_lightweight


def test_platform_gpu_policy():
    assert backends('win32') == ('auto','cpu','vulkan')
    assert automatic_devices('win32') == ('vulkan','cpu')
    assert backends('darwin') == ('auto','cpu','metal')
    assert automatic_devices('darwin') == ('metal','cpu')


@pytest.mark.parametrize('platform,device', [('win32','cuda'),('win32','metal'),('darwin','cuda'),('darwin','vulkan')])
def test_wrong_os_backend_cannot_use_custom_executable(tmp_path,monkeypatch,platform,device):
    from asr2rpp import runtime_registry
    binary=tmp_path/'worker';binary.write_bytes(b'executable')
    monkeypatch.setattr(sys,'platform',platform)
    with pytest.raises(ValueError,match='Unsupported'):
        runtime_registry.resolve('whisper_cpp',device,str(binary))


def test_macos_gui_and_cli_share_signed_resource_root(tmp_path,monkeypatch):
    from asr2rpp.catalog import model_directory,package_root
    macos=tmp_path/'ASR2RPP.app/Contents/MacOS';macos.mkdir(parents=True)
    monkeypatch.setattr(sys,'frozen',True,raising=False)
    monkeypatch.setattr(sys,'platform','darwin')
    monkeypatch.setattr(sys,'_MEIPASS',str(tmp_path/'not-the-model-directory'),raising=False)
    for binary in ('ASR2RPP','asr2rpp-cli'):
        monkeypatch.setattr(sys,'executable',str(macos/binary))
        assert package_root()==tmp_path/'ASR2RPP.app/Contents/Resources'
        assert model_directory()==package_root()/'models'


@pytest.mark.parametrize('platform',['win32','darwin'])
def test_packaging_excludes_stale_cuda_and_other_platforms(tmp_path,platform):
    source=tmp_path/'cache';source.mkdir()
    for name in runtime_packs(platform):
        root=source/name;root.mkdir()
        (root/'binary').write_bytes(b'good')
        metadata={'files':{'binary':hashlib.sha256(b'good').hexdigest()}}
        if name=='python_worker':
            metadata['packages']={'sherpa-onnx':'1.13.8','sherpa-onnx-core':'1.13.8'}
        (root/'build-manifest.json').write_text(json.dumps(metadata))
    (source/'cuda_runtime').mkdir();(source/'cuda_runtime/cudnn64_9.dll').write_bytes(b'never copy')
    (source/'whisper_cpp-cuda').mkdir()
    target=tmp_path/'package'
    copy_runtime_packs(source,target,platform)
    assert set(p.name for p in (target/'engines').iterdir())==set(runtime_packs(platform))
    assert (source/'cuda_runtime/cudnn64_9.dll').is_file()  # user's caches untouched


@pytest.mark.parametrize('name',['cudnn64_9.dll','cublas64_12.dll','cudart64_12.dll','ctranslate2.dll','faster_whisper.py','ggml-cuda.dll'])
def test_accidental_heavy_dependency_blocks_release(tmp_path,name):
    engine=tmp_path/'engines/python_worker';engine.mkdir(parents=True)
    (engine/name).write_bytes(b'bad')
    with pytest.raises(ValueError,match='Forbidden'):
        audit_lightweight(tmp_path)


def test_standard_worker_build_excludes_even_installed_optional_packages():
    from tools.package_worker import PACKAGES, EXCLUDED
    assert set(PACKAGES)=={'sherpa-onnx','sherpa-onnx-core'}
    assert {'faster_whisper','ctranslate2','nvidia'}.issubset(EXCLUDED)
    root=Path(__file__).resolve().parents[1]
    assert not (root/'models/faster-whisper-base.toml').exists()
    assert (root/'docs/optional-models/faster-whisper-base.toml').exists()


def test_ci_declares_real_macos_build_and_no_cuda_install():
    root=Path(__file__).resolve().parents[1]
    workflow=(root/'.github/workflows/app.yml').read_text()
    assert 'runs-on: macos-15' in workflow
    assert 'whisper_cpp:metal audio_cpp:cpu audio_cpp:metal' in workflow
    assert 'python tools/package_macos.py' in workflow
    assert 'cuda-toolkit@' not in workflow and 'cuda-worker' not in workflow
    assert 'whisper_cpp:cuda' not in workflow


@pytest.mark.parametrize('platform,previous,expected',[('win32','cuda','vulkan'),('darwin','vulkan','metal'),('darwin','cpu','cpu')])
def test_runtime_preferences_migrate_only_unavailable_devices(tmp_path,monkeypatch,platform,previous,expected):
    pytest.importorskip('PySide6')
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    from asr2rpp import gui_dcc
    app=QApplication.instance() or QApplication([])
    monkeypatch.setenv('ASR2RPP_DISABLE_RUNTIME_BOOTSTRAP','1')
    from types import SimpleNamespace
    from asr2rpp import platforms
    # Simulate backend policy only; do not make Qt/ctypes believe Linux is Windows.
    monkeypatch.setattr(platforms, 'sys', SimpleNamespace(platform=platform))
    prefs=QSettings(str(tmp_path/'prefs.ini'),QSettings.Format.IniFormat)
    prefs.setValue('runtime_default/whisper_cpp',previous)
    prefs.setValue('runtime_default/audio_cpp',previous)
    window=gui_dcc.MainWindow(preferences=prefs)
    try:
        assert set(window.runtime_defaults.values())=={expected}
        assert [window.asr.device.itemData(i) for i in range(window.asr.device.count())]==['default',*backends(platform)]
    finally:
        window.close();app.processEvents()
