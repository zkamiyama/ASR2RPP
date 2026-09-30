"""Text-only native requests: one model session per bounded sequence."""
from pathlib import Path
import json
from .adapters import (Result, executable, run_process, split_engine_parameters,
                       validate_model_parameter_constraints, audio_session_args)
from .catalog import checkpoint


def checked_text(path):
    value = path.read_text(encoding='utf-8-sig')
    if '\x00' in value or '\ufffd' in value:
        raise ValueError('Invalid text output; raw bytes retained')
    value.encode('utf-8', errors='strict')
    return value.strip()


def infer_text_requests(model, weights, requests, work, options, cancel, progress):
    if not requests:
        return {}
    if model.runtime != 'audio_cpp':
        from .worker_client import infer_requests
        return infer_requests(model, weights, requests, work, dict(options, _text_only=True), cancel, progress)
    binary = executable(model.runtime, options.get('device', 'cpu'), options.get('executable', ''))
    params, session = split_engine_parameters(model, options.get('parameters', {}))
    validate_model_parameter_constraints(model, params, session)
    # VAD controls are host settings, never forwarded as decoder options.
    params = {k:v for k,v in params.items() if not k.startswith('vad_') and k != 'vad'}
    work = Path(work)
    results = {}
    # Bound command metadata and native batch buffering. Audio windows are <=28s.
    from .native_profile import profile
    maximum = profile(model).max_batch_items
    for batch_no, offset in enumerate(range(0, len(requests), maximum)):
        chunk = requests[offset:offset + maximum]
        folder = work / f'text-{batch_no}'
        folder.mkdir(parents=True, exist_ok=True)
        payload = {'requests': [dict(id=r['id'], audio=str(r['audio'].resolve()),
            language=options.get('language') or model.defaults.get('language', 'ja'),
            options=params) for r in chunk]}
        request_file = folder / 'requests.json'
        request_file.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        output = folder / 'text.txt'
        argv = [str(binary), '--task', 'asr', '--family', model.family, '--model', str(weights),
            '--backend', 'best' if options.get('device') == 'auto' else options.get('device', 'cpu'),
            '--mode', 'offline', '--request-sequence', str(request_file),
            '--threads', str(options.get('threads', 4)), '--text-out', str(output)]
        argv += audio_session_args(model, session)
        run_process(argv, cancel, progress, folder / 'engine.log')
        for req in chunk:
            checkpoint(cancel)
            path = folder / f"text_{req['id']}.txt"
            if not path.is_file():
                raise ValueError('ASR did not produce text for request ' + req['id'])
            text = checked_text(path)
            results[req['id']] = Result([], {'text': text}, text)
    return results
