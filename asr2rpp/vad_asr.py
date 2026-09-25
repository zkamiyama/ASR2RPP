"""Bounded VAD -> independent text-only whisper.cpp requests.

VAD intervals may be exported as coarse speech-region timestamps. Optional
forced alignment refines them; never claim VAD boundaries are word timestamps.
"""
from dataclasses import asdict
from pathlib import Path
import json
import shutil
import subprocess

from .catalog import checkpoint
from .adapters import (Result, Unit, executable, run_process, split_engine_parameters,
                       whisper_parameter_args, validate_model_parameter_constraints)
from .inference_policy import policy_for, policy_notice
from .media import slice_pcm
from .vad import detect_windows


def command_batches(base, pairs, max_chars=24000, max_items=64):
    """Account for both input/output paths and quoting on Windows."""
    batch = []
    for pair in pairs:
        candidate = batch + [pair]
        argv = base + [part for inp, out in candidate for part in ('-f', str(inp), '-of', str(out))]
        if len(candidate) > max_items or len(subprocess.list2cmdline(argv)) > max_chars:
            if not batch:
                raise ValueError('ASR command/path exceeds the supported command length')
            yield batch
            batch = [pair]
            argv = base + ['-f', str(pair[0]), '-of', str(pair[1])]
            if len(subprocess.list2cmdline(argv)) > max_chars:
                raise ValueError('ASR command/path exceeds the supported command length')
        else:
            batch = candidate
    if batch:
        yield batch


def read_text_result(path):
    """Fail closed on invalid UTF-8 or schema; never delete leading characters."""
    try:
        raw = json.loads(path.read_text(encoding='utf-8-sig'))
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f'Invalid UTF-8/JSON from ASR: {path.name}; raw output retained') from exc
    chunks = raw.get('transcription') if isinstance(raw, dict) else None
    if not isinstance(chunks, list) or any(not isinstance(x, dict) or not isinstance(x.get('text'), str) for x in chunks):
        raise ValueError(f'Invalid text-only ASR output schema: {path.name}')
    text = ''.join(x['text'] for x in chunks).strip()
    try:
        text.encode('utf-8', errors='strict')
    except UnicodeError as exc:
        raise ValueError(f'ASR output contains an unpaired surrogate: {path.name}') from exc
    if '\ufffd' in text or '\x00' in text:
        raise ValueError(f'ASR output contains a replacement/NUL character: {path.name}; raw output retained')
    return text


def infer_vad_whisper(model, weights, audio, work, options, cancel, progress):
    policy = policy_for(model)
    if policy.segmentation != 'vad':
        raise ValueError('VAD inference requires the model-defined VAD policy')
    checkpoint(cancel)
    alignment_requested = options.get('alignment_requested', False)
    if type(alignment_requested) is not bool:
        raise ValueError('alignment_requested must be boolean')
    if policy.requires_alignment and not alignment_requested:
        raise ValueError('TOML inference policy requires forced alignment')
    parameters, session = split_engine_parameters(model, options.get('parameters') or {})
    validate_model_parameter_constraints(model, parameters, session)
    device, threads = options.get('device', 'cpu'), options.get('threads', 4)
    if device not in {'cpu', 'auto', 'cuda', 'vulkan', 'metal'}:
        raise ValueError('Unsupported Whisper device')
    if type(threads) is not int or not 1 <= threads <= 128:
        raise ValueError('Threads must be an integer in 1..128')
    binary = executable(model.runtime, device, options.get('executable', ''))
    if session:
        raise ValueError('whisper.cpp VAD inference does not accept audio.cpp session options')
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    progress(policy_notice(model))
    windows, record = detect_windows(model, parameters, audio, work, binary, cancel, progress,
                                     policy.max_segment_seconds, context_overlap=alignment_requested)
    inputs, out = work / 'inputs', work / 'out'
    inputs.mkdir(exist_ok=True)
    out.mkdir(exist_ok=True)
    raw = {'inference_policy': asdict(policy), 'vad': record,
           'timing_kind': 'pending_alignment' if alignment_requested else 'vad_segment',
           'alignment_requested': alignment_requested, 'segments': []}
    # VAD is performed separately; --vad would concatenate speech and change the
    # native decoder's input timeline. Never pass it to these independent requests.
    request = {k: v for k, v in parameters.items() if k != 'vad' and not k.startswith('vad_')}
    argv = [str(binary), '-m', str(weights), '-l', options.get('language') or model.defaults.get('language', 'ja'),
            '-t', str(options.get('threads', 4)), '-oj', '-np']
    if options.get('device', 'cpu') == 'cpu':
        argv.append('-ng')
    argv += whisper_parameter_args(request)
    prepared = []
    units = []
    try:
        for index, window in enumerate(windows):
            checkpoint(cancel)
            key = f'v{index:06d}'
            file, prefix = inputs / f'{key}.wav', out / key
            begin = round(window.start * 16000) / 16000
            length = slice_pcm(audio, file, begin, window.end - begin, cancel)
            if length > policy.max_segment_seconds + 1/16000:
                raise ValueError('PCM exceeds the TOML segment-duration limit')
            prepared.append((file, prefix, window, begin, length))
        pairs = [(file, prefix) for file, prefix, _w, _b, _l in prepared]
        for index, batch in enumerate(command_batches(argv, pairs), 1):
            checkpoint(cancel)
            progress(f'ASR: independent VAD batch {index}, {len(batch)} windows (timestamps OFF, history OFF)')
            command = argv + [v for file, prefix in batch for v in ('-f', str(file), '-of', str(prefix))]
            run_process(command, cancel, progress, work / f'asr-batch-{index}.log')
        for file, prefix, window, begin, length in prepared:
            checkpoint(cancel)
            text = read_text_result(prefix.with_suffix('.json'))
            accepted = any(ch.isalnum() for ch in text)
            raw['segments'].append({'id': prefix.name, 'audio_start': begin, 'audio_end': begin + length,
                                    'owner_start': window.owner_start, 'owner_end': window.owner_end,
                                    'text': text, 'accepted': accepted})
            if accepted:
                if not alignment_requested and (abs(window.start-window.owner_start) > 1/16000 or
                                                abs(window.end-window.owner_end) > 1/16000):
                    raise ValueError('Overlapping ASR context requires alignment; cannot fabricate text ownership')
                units.append(Unit(begin, begin + length, text, granularity='segment',
                                  method='vad_window' if alignment_requested else 'vad_segment',
                                  owner_start=window.owner_start if alignment_requested else None,
                                  owner_end=window.owner_end if alignment_requested else None))
        (work / 'raw.json').write_text(json.dumps(raw, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        (work / 'normalized.json').write_text(json.dumps([asdict(u) for u in units], ensure_ascii=False, indent=2), encoding='utf-8')
        return Result(units, raw, ''.join(u.text for u in units))
    finally:
        # Input media snippets are cache only, including on cancellation/failure.
        shutil.rmtree(inputs, ignore_errors=True)
