"""The real-machine matrix cannot mistake empty or fabricated outputs for success."""
from pathlib import Path
import json
import pytest
from tools.validate_catalog import case_arguments, validate_output


def test_text_model_selection_does_not_force_fake_native_timestamps():
    fixtures = {'ja': Path('ja.wav'), 'en': Path('en.wav')}
    fixture, device, argv = case_arguments('qwen', {'task': 'asr', 'provider': 'audio_cpp'},
                                          'vulkan', fixtures, Path('out'))
    assert fixture == fixtures['ja'] and device == 'vulkan'
    assert argv[argv.index('--timing') + 1] == 'auto'


def test_cpu_worker_and_auxiliary_stage_devices_remain_explicit():
    fixtures = {'ja': Path('ja.wav'), 'en': Path('en.wav')}
    _, device, argv = case_arguments('reazon', {'task': 'asr', 'provider': 'sherpa_onnx'},
                                    'vulkan', fixtures, Path('out'))
    assert device == 'cpu' and argv[argv.index('--asr-device') + 1] == 'cpu'
    _, device, argv = case_arguments('align', {'task': 'align'}, 'vulkan', fixtures, Path('out'))
    assert device == 'vulkan' and argv[argv.index('--align-device') + 1] == 'vulkan'
    assert '--vad-before-alignment' in argv


@pytest.mark.parametrize('case', ['empty', 'outside', 'nan', 'fake-timing', 'changed-source'])
def test_invalid_native_matrix_result_fails(tmp_path, case):
    report = tmp_path / 'test.asr2rpp'
    report.mkdir()
    (tmp_path / 'test.rpp').write_text('project')
    manifest = {'status': 'completed', 'source_unchanged': True, 'source_sha256': 'abc',
                'original_track': {'name': 'ORIGINAL', 'muted': True}}
    units = [{'start': 0., 'end': 1., 'text': 'test', 'method': 'vad_segment'}]
    if case == 'empty': units = []
    if case == 'outside': units[0]['end'] = 20.
    if case == 'nan': units[0]['end'] = float('nan')
    if case == 'fake-timing': units[0]['method'] = 'native_interval'
    if case == 'changed-source': manifest['source_unchanged'] = False
    (report / 'manifest.json').write_text(json.dumps(manifest))
    (report / 'transcript.json').write_text(json.dumps({'units': units}))
    with pytest.raises(ValueError):
        validate_output(tmp_path, 'asr', True, 'abc', 5.)


def test_excluded_version_metadata_does_not_leak_from_build_environment(tmp_path):
    from tools.package_worker import remove_excluded_metadata
    worker = tmp_path / 'worker'
    for name in ('faster_whisper-1.2.1', 'ctranslate2-4.6', 'sherpa_onnx-1.13.8', 'numpy-2.5'):
        path = worker / '_internal' / (name + '.dist-info')
        path.mkdir(parents=True)
        (path / 'METADATA').write_text('metadata')
    external = tmp_path / 'venv/faster_whisper-1.2.1.dist-info'
    external.mkdir(parents=True)
    remove_excluded_metadata(worker)
    names = {p.name for p in worker.rglob('*.dist-info')}
    assert names == {'sherpa_onnx-1.13.8.dist-info', 'numpy-2.5.dist-info'}
    assert external.is_dir()


def test_native_fixed_language_contract_omits_unsupported_request_field():
    from asr2rpp.catalog import Model
    from asr2rpp.native_profile import request_fields
    model = Model('fixed', 'audio_cpp', 'asr', {'path': 'weights'},
                  family='fixed', execution={'pass_language': False}, defaults={'language': 'en'})
    model.validate()
    assert request_fields(model, {'language': 'ja'}) == {}
    model.execution['pass_language'] = 'false'
    with pytest.raises(ValueError, match='boolean'): model.validate()


def test_native_speakers_are_actually_requested_and_verified():
    fixture, device, argv = case_arguments('vibe', {'task': 'asr', 'capabilities': {'speakers': True}},
        'vulkan', {'ja': Path('ja.wav'), 'en': Path('en.wav')}, Path('out'))
    assert argv[argv.index('--speaker-source') + 1] == 'native'


def test_stock_whisper_diagnostics_are_not_suppressed():
    source = (Path(__file__).resolve().parents[1] / 'asr2rpp/native_batches.py').read_text(encoding='utf-8')
    assert "'-np'" not in source
