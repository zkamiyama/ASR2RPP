"""Native subprocess adapters. Model families are data; engine protocols are code."""
import math
import re
from pathlib import Path
from common import read_json, write_json, digest, run_process
from registry import merge


def number(value, name, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if not low <= value <= high or (integer and type(value) is not int):
        raise ValueError(f'{name} must be in [{low}, {high}]')
    return value


def parse_whisper(data):
    if not isinstance(data, dict) or not isinstance(data.get('transcription'), list):
        raise ValueError('Unsupported whisper.cpp JSON: expected transcription[]')
    result = []
    for row in data['transcription']:
        offsets = row.get('offsets', {})
        if 'from' not in offsets or 'to' not in offsets:
            raise ValueError('Whisper returned text without usable offsets; alignment is required')
        result.append(dict(start=offsets['from'] / 1000, end=offsets['to'] / 1000,
                           text=row.get('text', ''), speaker=None, granularity='segment'))
    return result


def parse_audio(data, sample_rate, output):
    if isinstance(data, dict):
        data = data.get({'turns': 'speaker_turns', 'words': 'word_timestamps',
                         'segments': 'speech_segments'}[output], data.get(output))
    if not isinstance(data, list):
        raise ValueError('Unsupported audio.cpp JSON schema; raw output has been preserved')
    result = []
    for row in data:
        if 'start_sample' not in row or 'end_sample' not in row:
            raise ValueError('audio.cpp output needs start_sample/end_sample; will not guess time units')
        result.append(dict(start=row['start_sample'] / sample_rate,
                           end=row['end_sample'] / sample_rate,
                           text=row.get('text', row.get('word', '')),
                           speaker=str(row['speaker_id']) if row.get('speaker_id') is not None else None,
                           granularity='token' if output == 'words' else 'segment'))
    return result


def native_command(registry, model, wav, directory, device, overrides=None):
    path = registry.model_path(model)
    if not path.is_file():
        raise FileNotFoundError(f'Model not installed: {path}. Use install {model["id"]} or a local model.')
    params = merge(model.get('defaults', {}), overrides or {})
    exe = registry.executable(model['engine'], device)
    if model['engine'] == 'whisper.cpp':
        allowed = {'language', 'threads', 'beam_size', 'temperature', 'prompt'}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f'Unsupported whisper.cpp parameters: {sorted(unknown)}')
        language = params.get('language', 'ja')
        if not re.fullmatch('[a-zA-Z-]{2,12}', language):
            raise ValueError('Invalid language')
        threads = number(params.get('threads', 4), 'threads', 1, 256, True)
        beam = number(params.get('beam_size', 5), 'beam_size', 1, 100, True)
        temperature = number(params.get('temperature', 0), 'temperature', 0, 2)
        output = directory / 'whisper.json'
        args = [exe, '-m', path, '-f', wav, '-ojf', '-of', output.with_suffix(''),
                '-l', language, '-t', threads, '-bs', beam, '-tp', temperature]
        if device == 'cpu':
            args += ['-ng']
        if params.get('prompt'):
            if not model.get('allow_prompt', True):
                raise ValueError('This model disables initial prompts (anime-whisper quality warning)')
            if not isinstance(params['prompt'], str):
                raise ValueError('prompt must be text')
            args += ['--prompt', params['prompt']]
    else:
        allowed = {'threads', 'language', 'temperature', 'max_tokens', 'num_beams',
                   'request_options', 'session_options'}
        unknown = set(params) - allowed
        if unknown:
            raise ValueError(f'Unsupported audio.cpp parameters: {sorted(unknown)}')
        output_kind = model.get('output', 'turns' if model['type'] in ('diarization', 'joint') else 'words')
        if output_kind not in ('turns', 'segments', 'words'):
            raise ValueError('audio.cpp output must be turns, segments or words')
        output = directory / (output_kind + '.json')
        args = [exe, '--task', 'diar' if model['type'] == 'diarization' else 'asr',
                '--family', model['family'], '--model', path, '--mode', 'offline',
                '--backend', device, '--audio', wav, '--' + output_kind + '-out', output,
                '--threads', number(params.get('threads', 4), 'threads', 1, 256, True)]
        for key, bounds in {'temperature': (0, 2), 'max_tokens': (1, 32768), 'num_beams': (1, 100)}.items():
            if key in params:
                args += ['--' + key.replace('_', '-'), number(params[key], key, *bounds, key != 'temperature')]
        if 'language' in params:
            if not isinstance(params['language'], str) or not re.fullmatch('[A-Za-z-]{2,12}', params['language']):
                raise ValueError('Invalid language')
            args += ['--language', params['language']]
        for scope in ('request', 'session'):
            values = params.get(scope + '_options', {})
            if not isinstance(values, dict):
                raise ValueError(f'{scope}_options must be an object')
            for key, value in values.items():
                if not re.fullmatch('[A-Za-z0-9_.-]+', key) or not isinstance(value, (str, int, float, bool)):
                    raise ValueError('Options must have simple keys and scalar values')
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError('Options must be finite')
                rendered = str(value).lower() if isinstance(value, bool) else str(value)
                if '\0' in rendered:
                    raise ValueError('NUL is not allowed')
                args += ['--' + scope + '-option', f'{key}={rendered}']
    return [str(a) for a in args], output, params


def transcribe(registry, model, wav, directory, device, params, timeout, cancel_file=None):
    directory.mkdir(parents=True, exist_ok=True)
    command, output, effective = native_command(registry, model, wav, directory, device, params)
    provenance = dict(model_id=model['id'], repository=model.get('repository'),
                      engine=model['engine'], model_sha256=digest(registry.model_path(model)),
                      executable_sha256=digest(command[0]), requested_device=device,
                      effective_device='not_verified', parameters=effective,
                      validation_status=model.get('status', 'unverified'))
    write_json(directory / 'provenance.json', provenance)
    report = run_process(command, directory / 'inference', timeout, cancel_file)
    if not output.exists():
        raise ValueError('Engine did not produce timestamped output. Check raw logs; no times were fabricated.')
    raw = read_json(output)
    if model['engine'] == 'whisper.cpp':
        units = parse_whisper(raw)
    else:
        units = parse_audio(raw, model.get('output_sample_rate', model.get('sample_rate', 16000)), output.stem)
    provenance['process'] = report
    write_json(directory / 'provenance.json', provenance)
    return units, provenance
