"""audio.cpp v0.9.5 Nemotron Transcribe result contract, not generic words JSON.

The custom schema carries acoustic speaker IDs and CTC diagnostics. Generic
word output has neither. Never infer missing text/times from speaker activity.
"""
from collections import Counter
import re
import unicodedata
from .domain import Result, Unit

SCHEMA = 'nemotron_transcribe.result.v1'


def _lexical(text):
    # Match content rather than whitespace/punctuation or decoder control tags.
    # CTC rows may be time-sorted across overlapping speaker streams.
    text = re.sub(r'<spk:\d+>', '', text, flags=re.IGNORECASE)
    for token in ('<|im_start|>', '<|im_end|>', '<|endoftext|>', '<think>', '</think>'):
        text = text.replace(token, '')
    return Counter(c for c in unicodedata.normalize('NFC', text) if unicodedata.category(c)[0] in 'LMN')


def parse_text_result(raw):
    """Text-only requests still require lossless completion metadata, not CTC."""
    custom = raw.get('custom_schema_output') if isinstance(raw, dict) else None
    if not isinstance(custom, dict) or custom.get('schema') != SCHEMA:
        raise ValueError('Nemotron Transcribe requires audio.cpp custom schema ' + SCHEMA)
    data, text = custom.get('data'), raw.get('text')
    if not isinstance(data, dict) or not isinstance(text, str):
        raise ValueError('Nemotron Transcribe result needs custom data and full transcript text')
    termination = data.get('termination')
    if termination == 'near_silent' and not text.strip():
        return Result([], raw, text, ['Native silence gate returned no transcript or timed speech'])
    if termination != 'end_of_turn':
        raise ValueError(f'Nemotron Transcribe generation ended with {termination!r}; '
                         'partial text retained, timed export refused')
    if data.get('template_name', 'default') != 'default':
        raise ValueError('Nemotron result is not a source-language transcription')
    return Result([], raw, text)


def parse_result(raw):
    from .adapters import number
    result = parse_text_result(raw)
    text = result.text
    data = raw['custom_schema_output']['data']
    if data.get('termination') == 'near_silent':
        return result
    if data.get('alignment_status') != 'aligned' or data.get('failed_streams') != []:
        raise ValueError('Nemotron Transcribe CTC alignment is not complete '
                         f'({data.get("alignment_status")}: {data.get("alignment_failure_reason", "")}); '
                         'full transcript and native turns retained, no times fabricated')
    words = data.get('words')
    if not isinstance(words, list):
        raise ValueError('Nemotron Transcribe custom words must be a list')
    units = []
    for row in words:
        if not isinstance(row, dict):
            raise ValueError('Nemotron Transcribe custom word must be an object')
        start, end = number(row.get('start')), number(row.get('end'))
        if start < 0 or end <= start:
            raise ValueError('Nemotron Transcribe returned an invalid CTC interval; raw retained')
        word, display = row.get('word'), row.get('display')
        joins = row.get('joins_previous')
        column, speaker = row.get('speaker_column'), row.get('speaker_id')
        if (not isinstance(word, str) or not isinstance(display, str) or
                type(joins) is not bool or type(column) is not int or column < -1):
            raise ValueError('Invalid Nemotron Transcribe custom word fields')
        expected = 'unknown' if column == -1 else f'speaker_{column}'
        if speaker != expected:
            raise ValueError('Inconsistent Nemotron Transcribe acoustic speaker ID/column')
        # transcript_speaker_tag is a different namespace; never use it as ID.
        units.append(Unit(start, end, display or word,
            None if column == -1 else speaker, 'word', 'ctc_alignment',
            joins_previous=joins))
    if _lexical(text) != _lexical(''.join(u.text for u in units)):
        raise ValueError('Nemotron Transcribe CTC coverage differs from the full transcript; '
                         'sparse alignment retained without exporting a misleading subset')
    warnings = ['Nemotron Transcribe CTC timing and speakers are experimental; '
                'Japanese timing units may be characters, not lexical words.']
    if any(u.speaker is None for u in units):
        warnings.append('Some native CTC speaker IDs are unknown; no transcript-tag or activity-overlap relabeling was applied.')
    return Result(units, raw, text, warnings)


def validate_duration(result, duration):
    """Reject corrupt/sparse tails before generic bounds cleanup can drop them."""
    # Native CTC boundaries are rounded to milliseconds. Permit rounding only,
    # never an interval lying wholly outside the actual request.
    if any(u.start >= duration or u.end > duration + 0.001 for u in result.units):
        raise ValueError('Nemotron Transcribe CTC intervals exceed the request duration; '
                         'raw retained, timed export refused')
    return result


def read_manifest(path, expected_ids):
    """No stdout parsing: text may contain lines resembling CLI diagnostics."""
    import json
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    records = data.get('requests') if isinstance(data, dict) else None
    if not isinstance(records, list):
        raise ValueError('Invalid audio.cpp batch manifest')
    found = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Invalid audio.cpp batch request result')
        key = record.get('id')
        if not isinstance(key, str) or key not in expected_ids or key in found:
            raise ValueError('Unexpected or duplicate audio.cpp batch result ID')
        found[key] = record
    return found
