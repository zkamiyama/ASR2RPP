"""Versioned, data-only model schema; executable paths live in user settings."""
import math
import re

PROVIDERS = frozenset(('whisper_cpp', 'audio_cpp', 'sherpa_onnx', 'faster_whisper', 'external_json'))


def finite_scalar(value):
    return type(value) in (str, bool, int, float) and (type(value) is not float or math.isfinite(value))


def validate_value(key, value, spec):
    kind = spec['type']
    valid = finite_scalar(value)
    if kind == 'bool':
        valid = type(value) is bool
    elif kind == 'int':
        valid = type(value) is int
    elif kind == 'float':
        valid = type(value) in (int, float) and math.isfinite(value)
    elif kind == 'str':
        valid = isinstance(value, str)
    elif kind == 'enum':
        valid = any(type(value) is type(v) and value == v for v in spec['values'])
    if valid and kind in ('int', 'float'):
        valid = spec.get('min', -math.inf) <= value <= spec.get('max', math.inf)
    if not valid:
        raise ValueError('Invalid parameter value for ' + key)


def validate_extensions(model):
    from .catalog import safe_relative
    if type(model.schema_version) is not int or model.schema_version not in (1, 2):
        raise ValueError('Unsupported model schema_version; supported: 1 and 2')
    if not isinstance(model.artifacts, dict):
        raise ValueError('artifacts must map roles to relative filenames')
    for role, name in model.artifacts.items():
        if not re.fullmatch(r'[a-z][a-z0-9_]*', role) or not isinstance(name, str) or not name:
            raise ValueError('Invalid model artifact role or filename')
        if safe_relative(name) == '.':
            raise ValueError('An artifact must be a file, not a directory')
    if not isinstance(model.execution, dict):
        raise ValueError('execution must be a table')
    if set(model.execution) - {'mode', 'output', 'max_batch_items', 'max_audio_seconds', 'protocol', 'pass_language'}:
        raise ValueError('Unknown execution fields')
    if type(model.execution.get('pass_language', True)) is not bool:
        raise ValueError('execution.pass_language must be boolean')
    if 'pass_language' in model.execution and model.runtime != 'audio_cpp':
        raise ValueError('execution.pass_language applies to audio_cpp only')
    if model.execution.get('mode', 'offline') not in ('offline', 'streaming'):
        raise ValueError('execution.mode must be offline or streaming')
    if model.execution.get('output', 'segments') not in ('text', 'words', 'segments', 'turns'):
        raise ValueError('execution.output must be text, words, segments or turns')
    if type(model.execution.get('protocol', 1)) is not int or model.execution.get('protocol', 1) != 1:
        raise ValueError('Unsupported provider protocol')
    maximum = model.execution.get('max_batch_items', 128)
    if type(maximum) is not int or not 1 <= maximum <= 4096:
        raise ValueError('execution.max_batch_items must be 1..4096')
    duration = model.execution.get('max_audio_seconds', 0)
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError('execution.max_audio_seconds must be finite and nonnegative')
    if not isinstance(model.parameters, dict):
        raise ValueError('parameters must contain metadata tables')
    for key, spec in model.parameters.items():
        if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_.]*', key) or not isinstance(spec, dict):
            raise ValueError('Invalid parameter metadata')
        allowed = {'type', 'default', 'min', 'max', 'step', 'values', 'en', 'ja', 'tip_en', 'tip_ja', 'suffix'}
        if set(spec) - allowed or spec.get('type') not in ('bool', 'int', 'float', 'str', 'enum'):
            raise ValueError('Invalid parameter metadata fields/type: ' + key)
        if 'default' not in spec or not finite_scalar(spec['default']):
            raise ValueError('Parameter needs a finite scalar default: ' + key)
        for field in ('min', 'max', 'step'):
            if field in spec and (type(spec[field]) not in (int, float) or not math.isfinite(spec[field])):
                raise ValueError('Invalid numeric parameter limit: ' + key)
        if spec.get('min', -math.inf) > spec.get('max', math.inf):
            raise ValueError('Reversed parameter range: ' + key)
        if spec['type'] == 'enum' and (not isinstance(spec.get('values'), list) or not spec['values']
                or any(not finite_scalar(v) for v in spec['values'])):
            raise ValueError('Enum parameter requires scalar values: ' + key)
        validate_value(key, spec['default'], spec)
    if not isinstance(model.defaults, dict):
        raise ValueError('defaults must be a table')
    for category in ('request', 'session'):
        values = model.defaults.get(category, {})
        if not isinstance(values, dict) or any(not finite_scalar(v) for v in values.values()):
            raise ValueError('defaults.' + category + ' requires finite scalar values')
        for key, value in values.items():
            name = 'session.' + key if category == 'session' else key
            if name in model.parameters:
                validate_value(name, value, model.parameters[name])


def validate_parameters(model, request, session):
    values = dict(request, **{'session.' + k: v for k, v in session.items()})
    for key, value in values.items():
        if key in model.parameters:
            validate_value(key, value, model.parameters[key])


def artifact_paths(model, weights):
    """Resolve declared files within a model root, including symlink containment."""
    from pathlib import Path
    weights = Path(weights).resolve()
    root = weights if weights.is_dir() else weights.parent
    result = {}
    for role, relative in model.artifacts.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f'Missing or unsafe model artifact: {role} = {relative}')
        result[role] = str(path)
    return result
