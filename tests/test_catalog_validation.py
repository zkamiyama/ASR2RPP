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


def transcribe_definition():
    return dict(task='asr', provider='audio_cpp', family='nemotron_transcribe',
                capabilities=dict(timestamps='word', speakers=True),
                defaults=dict(request=dict(diarization=False, return_timestamps=False)))


def test_default_off_30b_runs_same_asr_and_independent_alignment_without_native_speakers():
    fixture, device, argv = case_arguments('nemotron-transcribe-iq4-nl', transcribe_definition(),
        'vulkan', {'ja': Path('ja.wav'), 'en': Path('en.wav')}, Path('out'))
    assert fixture == Path('ja.wav') and device == 'vulkan'
    assert argv[argv.index('--asr') + 1] == 'nemotron-transcribe-iq4-nl'
    assert argv[argv.index('--align') + 1] == 'qwen-forced-aligner'
    assert argv[argv.index('--align-device') + 1] == 'vulkan'
    assert argv[argv.index('--timing') + 1] == 'alignment'
    assert argv[argv.index('--speaker-source') + 1] == 'none'
    assert '--vad-before-alignment' in argv
    assert 'whisper-base' not in argv


def test_explicit_native_30b_definition_is_not_silently_rewritten():
    definition = transcribe_definition()
    definition['defaults']['request']['diarization'] = True
    _, _, argv = case_arguments('native30b', definition, 'vulkan',
                                {'ja': Path('ja.wav'), 'en': Path('en.wav')}, Path('out'))
    assert argv[argv.index('--speaker-source') + 1] == 'native'
    assert '--align' not in argv


@pytest.mark.parametrize('method,start,end,passes', [
    ('forced_alignment', 0., 1., True), ('native_interval', 0., 1., False),
    ('vad_segment', 0., 1., False), ('forced_alignment', 1., 1., False),
    ('forced_alignment', 0., 8., False),
])
def test_30b_final_alignment_requires_positive_independent_intervals(tmp_path, method, start, end, passes):
    report = tmp_path / 'test.asr2rpp'
    report.mkdir()
    (tmp_path / 'test.rpp').write_text('project')
    (report / 'manifest.json').write_text(json.dumps(dict(
        status='completed', source_unchanged=True, source_sha256='abc',
        timestamp_source='alignment', original_track=dict(name='ORIGINAL', muted=True))))
    (report / 'transcript.json').write_text(json.dumps({'units': [dict(
        start=start, end=end, text='発話', method=method)]}))
    if passes:
        result = validate_output(tmp_path, 'asr', True, 'abc', 5.,
                                 speakers_required=False, alignment_required=True)
        assert result['speaker_count'] == 0 and result['methods'] == ['forced_alignment']
    else:
        with pytest.raises(ValueError):
            validate_output(tmp_path, 'asr', True, 'abc', 5., alignment_required=True)


def package_metadata_fixture(tmp_path):
    import zipfile
    assets = tmp_path / 'ASR2RPP'
    assets.mkdir()
    raw = json.dumps(dict(version='0.4.0', commit='a' * 40)).encode()
    (assets / 'version.json').write_bytes(raw)
    archive = tmp_path / 'ASR2RPP-Windows-x64.zip'
    with zipfile.ZipFile(archive, 'w') as zipped:
        zipped.writestr('ASR2RPP/version.json', raw)
    return assets, archive


def test_evidence_metadata_records_actual_platform_and_explicit_operator_attestation(tmp_path, monkeypatch):
    from tools import validate_catalog as validator
    assets, archive = package_metadata_fixture(tmp_path)
    monkeypatch.setattr(validator.sys, 'platform', 'win32')
    for key in ('CI', 'GITHUB_ACTIONS', 'TF_BUILD'):
        monkeypatch.delenv(key, raising=False)
    metadata = validator.validation_metadata(assets, archive, frozen=True)
    assert metadata['platform'] == 'windows' and metadata['physical_hardware'] is False
    assert metadata['package_version'] == '0.4.0' and metadata['package_commit'] == 'a' * 40
    assert metadata['package_sha256'] == validator.digest(archive)
    attested = validator.validation_metadata(assets, archive, frozen=True,
                                            physical_hardware=True, hardware='Observed RTX 4080 desktop')
    assert attested['physical_hardware'] is True
    assert attested['hardware'] == 'Observed RTX 4080 desktop'


@pytest.mark.parametrize('problem', ['source', 'no-archive', 'no-hardware', 'ci', 'unknown-os'])
def test_physical_hardware_cannot_be_inferred_or_attested_in_ci(tmp_path, monkeypatch, problem):
    from tools import validate_catalog as validator
    assets, archive = package_metadata_fixture(tmp_path)
    monkeypatch.setattr(validator.sys, 'platform', 'win32')
    for key in ('CI', 'GITHUB_ACTIONS', 'TF_BUILD'):
        monkeypatch.delenv(key, raising=False)
    if problem == 'ci':
        monkeypatch.setenv('GITHUB_ACTIONS', 'true')
    if problem == 'unknown-os':
        monkeypatch.setattr(validator.sys, 'platform', 'unknown')
    with pytest.raises(ValueError):
        validator.validation_metadata(assets, None if problem == 'no-archive' else archive,
            frozen=problem != 'source', physical_hardware=True,
            hardware='' if problem == 'no-hardware' else 'Observed desktop')


def test_extracted_version_cannot_claim_different_archive_provenance(tmp_path):
    from tools.validate_catalog import validation_metadata
    assets, archive = package_metadata_fixture(tmp_path)
    (assets / 'version.json').write_text(json.dumps(dict(version='0.5.0', commit='b' * 40)))
    with pytest.raises(ValueError, match='does not match'):
        validation_metadata(assets, archive, frozen=True)


def test_30b_acquisition_includes_aligner_and_does_not_substitute_an_asr():
    from tools.validate_catalog import case_dependencies
    assert case_dependencies('nemotron-transcribe-iq4-nl', transcribe_definition()) == [
        'nemotron-transcribe-iq4-nl', 'qwen-forced-aligner']
    assert case_dependencies('diarizer', dict(task='diar')) == ['diarizer', 'whisper-base']
    assert case_dependencies('other-asr', dict(task='asr')) == ['other-asr']
