"""Release contracts for unmodified upstreams and replaceable model runtimes."""
from pathlib import Path
import json
import threading
import pytest
from asr2rpp.catalog import load_catalog
from asr2rpp.whisper_io import region_executable

ROOT=Path(__file__).resolve().parents[1]


def test_all_added_model_definitions_are_pinned_and_valid():
    catalog,errors=load_catalog(ROOT/'models')
    assert not errors
    for name in ('qwen3-asr-06b','reazonspeech-k2','faster-whisper-base'):
        model=catalog[name]
        assert model.schema_version==2
        assert len(model.source['revision'])==40
        assert all(len(model.source['sha256'][f])==64 for f in model.source['files'])
    assert catalog['reazonspeech-k2'].defaults['device']=='cpu'


def test_region_helper_is_only_selected_for_supported_options(tmp_path,monkeypatch):
    monkeypatch.delenv('ASR2RPP_WHISPER_IO',raising=False)
    cli=tmp_path/'whisper-cli.exe';cli.write_bytes(b'CLI')
    helper=tmp_path/'asr2rpp-whisper-regions.exe';helper.write_bytes(b'API')
    assert region_executable(cli,{'no_timestamps':True,'max_context':0,'vad_threshold':.5})==helper
    assert region_executable(cli,{'grammar':'x'})==cli
    assert region_executable(cli,{'future_option':1})==cli
    monkeypatch.setenv('ASR2RPP_WHISPER_IO','legacy')
    assert region_executable(cli,{})==cli


def test_no_upstream_cli_source_patch_is_called_by_builder():
    builder=(ROOT/'tools/build_native.py').read_text()
    assert 'patch_whisper(' not in builder
    assert 'ASR2RPP_WHISPER_SOURCE' in builder
    worker=(ROOT/'native/whisper_regions.cpp').read_text()
    assert '#include "whisper.h"' in worker
    assert '#include "cli.cpp"' not in worker
    lock=json.loads((ROOT/'native/versions.json').read_text())
    assert 'qwen3_asr' in lock['audio_cpp']['models']
    assert 'sense_asr' in lock['audio_cpp']['models']


def test_cuda_dedup_records_dependencies_and_rejects_conflicts(tmp_path):
    from tools.portable_runtime import collect_cuda
    for folder in ('whisper_cpp-cuda','audio_cpp-cuda'):
        directory=tmp_path/'engines'/folder;directory.mkdir(parents=True)
        (directory/'cublas64_12.dll').write_bytes(b'same')
        (directory/'build-manifest.json').write_text(json.dumps({'files':{'cublas64_12.dll':'old'}}))
    moved=collect_cuda(tmp_path)
    assert len(moved)==2
    assert (tmp_path/'engines/cuda_runtime/cublas64_12.dll').read_bytes()==b'same'
    for folder in ('whisper_cpp-cuda','audio_cpp-cuda'):
        meta=json.loads((tmp_path/'engines'/folder/'build-manifest.json').read_text())
        assert not meta['files'] and 'cublas64_12.dll' in meta['shared_cuda_runtime']
    (tmp_path/'engines/audio_cpp-cuda/cublas64_12.dll').write_bytes(b'conflict')
    with pytest.raises(ValueError,match='Conflicting'):
        collect_cuda(tmp_path)


def test_gui_preflight_failure_stops_before_model_download(tmp_path,monkeypatch):
    pytest.importorskip('PySide6')
    from asr2rpp import gui_dcc,providers
    from asr2rpp.catalog import Model
    from asr2rpp.pipeline import Stage
    from asr2rpp.preprocessing import Settings
    model=Model('not-installed','audio_cpp','asr',{'path':str(tmp_path/'weights')},family='missing')
    calls=[]
    monkeypatch.setattr(gui_dcc,'resolve_model',lambda *a,**kw:calls.append('download'))
    def unsupported(*args,**kwargs):
        raise ValueError('Unsupported compiled family')
    monkeypatch.setattr(providers,'preflight',unsupported)
    worker=gui_dcc.Worker([],Settings(Stage(model.id)),{model.id:model})
    with pytest.raises(ValueError,match='Unsupported compiled family'):
        worker._prepare_models()
    assert not calls
