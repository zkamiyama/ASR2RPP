"""Protocol-1 provider client; strict request ownership and partial result recovery."""
from dataclasses import fields
from pathlib import Path
import json
import math
from .domain import Unit, Result
from .catalog import checkpoint
from .model_schema import artifact_paths


def normalize(record):
    if record.get('time_origin') != 'request':
        raise ValueError('Worker timestamps need an explicit request-relative origin')
    text = record.get('text')
    if not isinstance(text, str) or '\x00' in text or '\ufffd' in text:
        raise ValueError('Invalid worker transcript')
    text.encode('utf-8', errors='strict')
    raw_units = record.get('units')
    if not isinstance(raw_units, list):
        raise ValueError('Worker units must be a list')
    allowed = {f.name for f in fields(Unit)}
    units = []
    for item in raw_units:
        if not isinstance(item, dict) or set(item) - allowed:
            raise ValueError('Invalid worker interval schema')
        for key in ('start', 'end'):
            value = item.get(key)
            if type(value) not in (float, int) or not math.isfinite(value):
                raise ValueError('Invalid worker timestamp')
        if item['start'] < 0 or item['end'] <= item['start'] or not isinstance(item.get('text'), str):
            raise ValueError('Invalid worker speech interval')
        if item.get('granularity', 'segment') not in ('segment','word','token','speaker_turn'):
            raise ValueError('Unknown worker timestamp granularity')
        speaker = item.get('speaker')
        if speaker is not None and not isinstance(speaker, str):
            raise ValueError('Worker speaker must be a string or null')
        if '\x00' in item['text'] or '\ufffd' in item['text']:
            raise ValueError('Invalid interval transcript')
        item['text'].encode('utf-8', errors='strict')
        if not isinstance(item.get('method', 'native_interval'), str):
            raise ValueError('Invalid timestamp provenance')
        if item.get('owner_start') is not None or item.get('owner_end') is not None:
            raise ValueError('Provider cannot prescribe host window ownership')
        units.append(Unit(**item))
    return Result(units, record, text)


def read_results(path, ids):
    results, errors, seen = {}, {}, set()
    if not path.is_file():
        return {}, {key:ValueError('Worker did not produce a result') for key in ids}
    with path.open('rb') as handle:
        for line in handle:
            if len(line) > 16*1024**2:
                raise ValueError('Oversized worker result')
            try:
                item = json.loads(line)
            except (ValueError, UnicodeError):
                # A killed worker may leave a partial trailing line. Successful
                # earlier requests remain usable; missing IDs fail below.
                continue
            if not isinstance(item, dict) or type(item.get('schema_version')) is not int or item['schema_version'] != 1:
                raise ValueError('Unsupported worker result schema')
            key = item.get('id')
            if key not in ids or key in seen:
                raise ValueError('Unexpected or duplicate worker result ID')
            seen.add(key)
            try:
                if item.get('status') != 'ok':
                    raise ValueError(str(item.get('error', 'Worker request failed')))
                results[key] = normalize(item)
            except (ValueError, TypeError) as exc:
                errors[key] = exc
    for key in ids - seen:
        errors[key] = ValueError('Worker result is missing or incomplete')
    return results, errors


def infer_batch(model, weights, requests, work, options, cancel, progress, *, text_only=False):
    from .adapters import executable, run_process, split_engine_parameters, validate_model_parameter_constraints
    from .runtime_registry import command
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    binary = executable(model.runtime, options.get('device','cpu'), options.get('executable',''))
    params, session = split_engine_parameters(model, options.get('parameters',{}))
    validate_model_parameter_constraints(model, params, session)
    if session:
        raise ValueError('This worker does not support native audio.cpp session options')
    params = {k:v for k,v in params.items() if not k.startswith('vad_') and k != 'vad'}
    payload = dict(schema_version=1, model=dict(provider=model.runtime, family=model.family,
        weights=str(Path(weights).resolve()), artifacts=artifact_paths(model, weights),
        device=options.get('device','cpu'), threads=options.get('threads',4), sample_rate=model.sample_rate,
        language=options.get('language') or model.defaults.get('language','ja'), parameters=params,
        text_only=text_only), requests=[dict(id=r['id'],audio=str(Path(r['audio']).resolve())) for r in requests])
    request_file, output_file = work/'requests.json', work/'results.jsonl'
    request_file.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    args = command(model.runtime,binary) + ['--request',str(request_file),'--output',str(output_file)]
    process_error = None
    try:
        run_process(args,cancel,progress,work/'engine.log')
    except Exception as exc:
        checkpoint(cancel)
        process_error = exc
    results, errors = read_results(output_file, {r['id'] for r in requests})
    if process_error:
        for key in errors:
            errors[key] = ValueError(f'{errors[key]} ({process_error})')
    return results, errors


def infer_requests(model, weights, requests, work, options, cancel, progress):
    results, errors = {}, {}
    maximum = model.execution.get('max_batch_items',128)
    for batch_no, offset in enumerate(range(0,len(requests),maximum)):
        checkpoint(cancel)
        current, failed = infer_batch(model,weights,requests[offset:offset+maximum],
            Path(work)/f'worker-{batch_no}',options,cancel,progress,text_only=options.get('_text_only',False))
        results.update(current)
        errors.update(failed)
    if errors:
        raise ValueError('; '.join(f'{key}: {error}' for key,error in errors.items()))
    return results
