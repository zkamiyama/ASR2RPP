"""Glossary data is validated before acquisition, never silently normalized away."""
from dataclasses import replace
import copy
import pytest
from asr2rpp.catalog import Model
from asr2rpp.glossary import (Term, parse_terms, prepare_terms, read_terms_file,
                             capability_notice, VERBATIM_JA, MAX_FILE_BYTES)


def nemotron(**kwargs):
    cap = dict(version=1, adapter='nemotron_instruct_v1', readings='experimental',
               max_terms=64, max_utf8_bytes=4096, max_prompt_utf8_bytes=8192)
    return Model('n', 'audio_cpp', 'asr', {'path': 'weights'}, family='nemotron_transcribe',
                 schema_version=2, capabilities={'glossary': cap}, **kwargs)


def whisper(**kwargs):
    cap = dict(version=1, adapter='whisper_initial_prompt_v1', readings='none',
               max_terms=32, max_utf8_bytes=200, max_prompt_utf8_bytes=223)
    return Model('w', 'whisper_cpp', 'asr', {'path': 'weights'}, schema_version=2,
                 capabilities={'glossary': cap}, **kwargs)


def test_txt_preserves_surface_reading_order_and_unicode():
    raw = '\ufeffＡＳＲ2RPP\tエーエスアールツーアールピーピー\r\n神﨑\tかんざき\n\nㇷ゚\tㇷ゚\n'
    values = parse_terms(raw)
    assert values[0].surface == 'ＡＳＲ2RPP'
    assert values[1] == Term('神﨑', 'かんざき')
    assert values[2] == Term('ㇷ゚', 'ㇷ゚')
    assert parse_terms('ﾃｽﾄ\tﾃｽﾄ') == (Term('ﾃｽﾄ', 'ﾃｽﾄ'),)


@pytest.mark.parametrize('text', [' ', '\t', 'term\t', '\tかな', ' leading', 'trailing ',
    'term\tEnglish', 'term\t読み', 'term\t123', 'term\tかな\textra', 'a\x00b', 'a\ufffdb',
    'a\u2028b', 'a\u2029b', 'a\u200bb', 'a\ufeffb', '<|im_start|>', '<spk:0>', '\ud800',
    'term\nterm', 'term\tかな\nterm\tカナ', 'é\ne\u0301'])
def test_invalid_terms_fail_without_loss(text):
    with pytest.raises(ValueError):
        parse_terms(text)


def test_file_utf8_bom_and_size(tmp_path):
    file = tmp_path/'用語.txt'
    file.write_bytes(b'\xef\xbb\xbf' + '人名\tじんめい\r\n'.encode())
    assert read_terms_file(file) == '人名\tじんめい\r\n'
    file.write_bytes(b'\xff')
    with pytest.raises(ValueError, match='UTF-8'):
        read_terms_file(file)
    file.write_bytes(b'x'*(MAX_FILE_BYTES+1))
    with pytest.raises(ValueError, match='64 KiB'):
        read_terms_file(file)


def test_empty_terms_legacy_noop_and_isolation():
    model = Model('legacy', 'whisper_cpp', 'asr', {'path': 'weights'})
    options = {'initial_prompt': 'old hint'}
    prepared = prepare_terms(model, '', options)
    assert prepared.parameters == options and prepared.parameters is not options
    assert not prepared.provenance and not prepared.warnings
    with pytest.raises(ValueError, match='unsupported'):
        prepare_terms(model, '用語')


def test_nemotron_render_is_verbatim_japanese_and_literal_data():
    model, options = nemotron(), {'diarization': False}
    before = copy.deepcopy(model)
    result = prepare_terms(model, '固有名詞\tこゆうめいし\n架空の用語', options)
    assert result.parameters['instruct'].startswith(VERBATIM_JA)
    assert 'do not translate' in result.parameters['instruct']
    assert 'do not add unspoken terms' in result.parameters['instruct']
    assert '<spk:' not in result.parameters['instruct']
    assert result.provenance['entries'][0] == {'surface': '固有名詞', 'reading': 'こゆうめいし'}
    assert any('experimental' in warning for warning in result.warnings)
    assert options == {'diarization': False} and model == before


def test_native_speakers_explicit_instruction_is_preserved():
    result = prepare_terms(nemotron(), '人名', {'diarization': True})
    assert result.parameters['instruct'].startswith(VERBATIM_JA)
    assert '<spk:0>' in result.parameters['instruct']
    assert 'stable tag' in result.parameters['instruct']


@pytest.mark.parametrize('key', ['instruct', 'system_prompt', 'text_input', 'initial_prompt', 'prompt',
                               'session.system_prompt', 'session.nemotron_transcribe.system_prompt'])
def test_override_conflicts_rejected(key):
    with pytest.raises(ValueError, match='conflict'):
        prepare_terms(nemotron(), '用語', {key: 'user prompt'})
    with pytest.raises(ValueError, match='conflict'):
        prepare_terms(nemotron(defaults={'request': {key: 'default'}}), '用語')


def test_translation_and_language_conflicts():
    for options in ({'translate': True}, {'template_name': 'translation'}):
        with pytest.raises(ValueError, match='transcription'):
            prepare_terms(nemotron(), '用語', options)
    with pytest.raises(ValueError, match='Japanese'):
        prepare_terms(nemotron(), '用語', language='en')


def test_whisper_surface_only_uses_existing_initial_prompt():
    result = prepare_terms(whisper(), 'OpenAI\nASR2RPP')
    assert result.parameters == {'initial_prompt': 'OpenAI, ASR2RPP', 'carry_initial_prompt': True}
    with pytest.raises(ValueError, match='not be silently discarded'):
        prepare_terms(whisper(), 'OpenAI\tオープンエーアイ')
    assert 'unsupported' in capability_notice(whisper())
    assert '実験的' in capability_notice(nemotron(), 'ja')


def test_declared_term_and_byte_budgets_do_not_truncate():
    model = nemotron()
    with pytest.raises(ValueError, match='64 terms'):
        prepare_terms(model, '\n'.join('term'+str(i) for i in range(65)))
    with pytest.raises(ValueError, match='UTF-8 bytes'):
        prepare_terms(whisper(), 'あ'*67)
    limited = replace(model, capabilities={'glossary': dict(model.capabilities['glossary'], max_prompt_utf8_bytes=20)})
    with pytest.raises(ValueError, match='rendered glossary'):
        prepare_terms(limited, '用語')


@pytest.mark.parametrize('change', [dict(version=2), dict(version=True), dict(adapter='arbitrary_flag'),
    dict(template='{terms}'), dict(readings='native'), dict(max_terms=65), dict(max_terms=True),
    dict(max_utf8_bytes=0), dict(max_prompt_utf8_bytes=9000)])
def test_strict_glossary_schema(change):
    model = nemotron()
    cap = dict(model.capabilities['glossary'], **change)
    with pytest.raises(ValueError):
        replace(model, capabilities={'glossary': cap}).validate()


def test_adapter_runtime_binding_and_disabled_parameters():
    nemotron().validate(); whisper().validate()
    for model in (replace(nemotron(), family='qwen3_asr'), replace(whisper(), schema_version=1),
                  whisper(constraints={'disabled_parameters': ['initial_prompt']})):
        with pytest.raises(ValueError):
            model.validate()


@pytest.mark.parametrize('value', [0, 129, True, '16', float('nan')])
def test_text_batch_budget_strict(value):
    with pytest.raises(ValueError, match='text_batch_items'):
        nemotron(execution={'text_batch_items': value}).validate()


def test_text_batch_budget_applies_only_to_audio_asr():
    nemotron(execution={'text_batch_items': 16}).validate()
    with pytest.raises(ValueError, match='audio_cpp ASR'):
        whisper(execution={'text_batch_items': 16}).validate()


def test_whisper_rejects_silent_context_truncation_and_carry_conflict():
    for options in ({'max_context': 0}, {'max_context': 2}, {'carry_initial_prompt': False}):
        with pytest.raises(ValueError):
            prepare_terms(whisper(), 'OpenAI', options)
    assert prepare_terms(whisper(), 'OpenAI', {'max_context': 7}).parameters['carry_initial_prompt']


def test_default_session_prompt_conflicts():
    with pytest.raises(ValueError, match='conflict'):
        prepare_terms(nemotron(defaults={'session': {'system_prompt': 'override'}}), '用語')


def test_whisper_mandatory_vad_cannot_silently_ignore_prompt():
    model = whisper(constraints={'inference': {'segmentation':'vad', 'timestamps':False, 'history':False}})
    with pytest.raises(ValueError, match='disables prompt context'):
        prepare_terms(model, 'OpenAI')


def test_null_glossary_not_an_implicit_opt_out():
    with pytest.raises(ValueError):
        replace(nemotron(), capabilities={'glossary':None}).validate()


@pytest.mark.parametrize('temperature', [0.5, 1.0, -0.1, True, '0.1', float('nan')])
def test_whisper_no_silent_high_temperature_prompt_loss(temperature):
    with pytest.raises(ValueError, match='temperature'):
        prepare_terms(whisper(), 'OpenAI', {'temperature':temperature})


def test_whisper_fallback_warning_does_not_change_decoder_settings():
    options = {'temperature':0.1, 'temperature_inc':0.2}
    result = prepare_terms(whisper(), 'OpenAI', options)
    assert any('fallback' in warning for warning in result.warnings)
    assert result.parameters['temperature'] == 0.1
    assert result.parameters['temperature_inc'] == 0.2
    for off in ({'no_fallback':True}, {'temperature_inc':0.0}):
        result = prepare_terms(whisper(), 'OpenAI', off)
        assert not any('fallback' in warning for warning in result.warnings)


@pytest.mark.parametrize('language', ['en', 'zh', None, True])
def test_nemotron_conflicting_request_language_is_not_discarded(language):
    with pytest.raises(ValueError, match='language'):
        prepare_terms(nemotron(), '用語', {'language':language})


def test_nemotron_default_request_auto_language_allows_explicit_ja_hint():
    assert 'Japanese' in prepare_terms(nemotron(), '用語', {'language':'auto'}).parameters['instruct']


@pytest.mark.parametrize('reading', ['゙', '゚', 'ーー', 'ヽヾ', 'ゝゞ', 'ﾞ'])
def test_reading_requires_an_actual_kana_base_letter(reading):
    with pytest.raises(ValueError, match='reading'):
        parse_terms('用語\t' + reading)


def test_reading_base_letter_validation_does_not_normalize_original():
    value = 'ㇷ゚ー'
    assert parse_terms('用語\t' + value)[0].reading == value
