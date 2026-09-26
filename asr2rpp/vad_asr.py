"""Bounded VAD -> independent text-only whisper.cpp requests.

VAD intervals may be exported as coarse speech-region timestamps. Optional
forced alignment refines them; never claim VAD boundaries are word timestamps.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import shutil
import subprocess
import time
from .whisper_io import capabilities, response_command, write_plan, read_bundle

from .catalog import checkpoint
from .adapters import (Result, Unit, executable, run_process, split_engine_parameters,
                       whisper_parameter_args, validate_model_parameter_constraints)
from .inference_policy import policy_for, policy_notice
from .media import slice_pcm
from .vad import detect_windows


from .performance import timed

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
    return read_text_payload(raw, path.name)


def read_text_payload(raw, label):
    chunks = raw.get('transcription') if isinstance(raw, dict) else None
    if not isinstance(chunks, list) or any(not isinstance(x, dict) or not isinstance(x.get('text'), str) for x in chunks):
        raise ValueError(f'Invalid text-only ASR output schema: {label}')
    text = ''.join(x['text'] for x in chunks).strip()
    try:
        text.encode('utf-8', errors='strict')
    except UnicodeError as exc:
        raise ValueError(f'ASR output contains an unpaired surrogate: {label}') from exc
    if '\ufffd' in text or '\x00' in text:
        raise ValueError(f'ASR output contains a replacement/NUL character: {label}; raw output retained')
    return text


def _prepare_vad_whisper(model, weights, audio, work, options, cancel, progress):
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
    features = capabilities(binary, work, cancel, progress)
    use_plan = 'ASR2RPP_PCM_PLAN_V1' in features
    vad_started = time.monotonic()
    progress(policy_notice(model))
    windows, record = detect_windows(model, parameters, audio, work, binary, cancel, progress,
                                     policy.max_segment_seconds, context_overlap=alignment_requested)
    vad_seconds = time.monotonic() - vad_started
    inputs, out = work / 'inputs', work / 'out'
    if not use_plan:
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
    prepare_started = time.monotonic()
    try:
        if use_plan and windows:
            prefixes = [out / f'v{i:06d}' for i in range(len(windows))]
            plan = work / 'input.pcm-plan'
            intervals = write_plan(audio, windows, prefixes, plan, cancel)
            prepared = [(audio, prefix, w, b, n) for prefix, w, (b,n) in zip(prefixes, windows, intervals)]
        for index, window in enumerate(windows if not use_plan else []):
            checkpoint(cancel)
            key = f'v{index:06d}'
            file, prefix = inputs / f'{key}.wav', out / key
            begin = round(window.start * 16000) / 16000
            length = slice_pcm(audio, file, begin, window.end - begin, cancel)
            if length > policy.max_segment_seconds + 1/16000:
                raise ValueError('PCM exceeds the TOML segment-duration limit')
            prepared.append((file, prefix, window, begin, length))
        return Prepared(work, inputs, prepared, argv, features, use_plan,
                        work/'input.pcm-plan', raw, alignment_requested,
                        vad_seconds, time.monotonic()-prepare_started)
    except BaseException:
        shutil.rmtree(inputs, ignore_errors=True)
        raise


@dataclass
class Prepared:
    work: Path
    inputs: Path
    prepared: list
    argv: list
    features: frozenset
    use_plan: bool
    plan: Path
    raw: dict
    alignment_requested: bool
    vad_seconds: float
    prepare_seconds: float
    native_results: list | None = None


def _execute(context, cancel, progress):
    work, prepared, argv = context.work, context.prepared, context.argv
    features, use_plan, plan = context.features, context.use_plan, context.plan
    raw, vad_seconds = context.raw, context.vad_seconds
    native_started = time.monotonic()
    launches = 0
    if use_plan and prepared:
        progress(f'ASR: {len(prepared)} independent PCM regions, one model load, no segment WAVs')
        command = argv + ['--asr2rpp-pcm-plan', str(plan)]
        bundled = 'ASR2RPP_PCM_RESULT_V1' in features and len(prepared)<=4096
        if bundled:
            command += ['--asr2rpp-pcm-results',str(work/'native-results.json')]
        run_process(command, cancel, progress, work / 'asr-batch-1.log')
        if bundled:
            context.native_results = read_bundle(work/'native-results.json',len(prepared))
        launches = 1
    elif prepared:
        pairs = [(file, prefix) for file, prefix, _w, _b, _l in prepared]
        if 'ASR2RPP_RESPONSE_V1' in features:
            batches = (pairs[i:i+4096] for i in range(0,len(pairs),4096))
        else:
            batches = command_batches(argv, pairs)
        for index, batch in enumerate(batches, 1):
            checkpoint(cancel)
            progress(f'ASR: independent VAD batch {index}, {len(batch)} windows (timestamps OFF, history OFF)')
            command = argv + [v for file, prefix in batch for v in ('-f', str(file), '-of', str(prefix))]
            if 'ASR2RPP_RESPONSE_V1' in features:
                command = response_command(command, work/f'asr-{index}.args')
            run_process(command, cancel, progress, work / f'asr-batch-{index}.log')
            launches += 1
    raw['performance'] = {'vad_seconds':vad_seconds, 'prepare_seconds':context.prepare_seconds,
                          'asr_seconds':time.monotonic()-native_started, 'model_processes':launches,
                          'input_mode':'pcm_plan_v1' if use_plan else 'segment_wav',
                          'segment_wav_files':0 if use_plan else len(prepared),
                          'result_mode':'bundle' if context.native_results is not None else 'per_region_json'}


def _finish(context, cancel):
    work, prepared, raw = context.work, context.prepared, context.raw
    alignment_requested = context.alignment_requested
    units = []
    if context.native_results is not None and not (work/'native-results.json').exists():
        (work/'native-results.json').write_text(json.dumps({'schema':1,'results':context.native_results},ensure_ascii=False),encoding='utf-8')
    for index, (file, prefix, window, begin, length) in enumerate(prepared):
        checkpoint(cancel)
        text = read_text_result(prefix.with_suffix('.json')) if context.native_results is None else read_text_payload(context.native_results[index],prefix.name)
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


@timed('vad_asr')
def infer_vad_whisper(model, weights, audio, work, options, cancel, progress):
    context = _prepare_vad_whisper(model, weights, audio, work, options, cancel, progress)
    try:
        _execute(context, cancel, progress)
        return _finish(context, cancel)
    finally:
        shutil.rmtree(context.inputs, ignore_errors=True)


@timed('vad_asr_queue')
def infer_vad_many(model, weights, items, options, cancel, progress, failed, root):
    """Prepare each file independently, then load one model for all valid PCM plans."""
    contexts, results = {}, {}
    try:
        for key, audio, work in items:
            checkpoint(cancel)
            try:
                contexts[key] = _prepare_vad_whisper(model, weights, audio, work, options, cancel, progress)
            except Exception as exc:
                checkpoint(cancel)
                failed(key, exc)
        active = [(k,c) for k,c in contexts.items() if c.prepared]
        if active and all(c.use_plan for _,c in active):
            command = list(active[0][1].argv)
            for key,c in active:
                if c.argv != active[0][1].argv:
                    raise ValueError('Incompatible settings in a shared Whisper session')
                command += ['--asr2rpp-pcm-plan',str(c.plan)]
            total = sum(len(c.prepared) for _,c in active)
            bundled = total<=4096 and all('ASR2RPP_PCM_RESULT_V1' in c.features for _,c in active)
            if bundled:
                command += ['--asr2rpp-pcm-results',str(Path(root)/'native-session.json')]
            if len(subprocess.list2cmdline(command)) > 24000:
                command = response_command(command, Path(root)/'queue-asr.args')
            started=time.monotonic()
            try:
                progress(f'ASR: one model session for {len(active)} files / {sum(len(c.prepared) for _,c in active)} regions')
                run_process(command, cancel, progress, Path(root)/'asr-session.log')
            except Exception as exc:
                checkpoint(cancel)
                for key,c in active:
                    failed(key,exc)
                return results
            elapsed=time.monotonic()-started
            if bundled:
                try:
                    all_results = read_bundle(Path(root)/'native-session.json',total)
                except Exception as exc:
                    checkpoint(cancel)
                    for key,c in active:
                        failed(key,exc)
                    return results
                offset = 0
                for key,c in active:
                    c.native_results = all_results[offset:offset+len(c.prepared)]
                    offset += len(c.prepared)
            for key,c in active:
                c.raw['performance']={'input_mode':'pcm_plan_v1','segment_wav_files':0,
                    'model_processes':1,'shared_session_files':len(active),
                    'asr_seconds':elapsed,'asr_scope':'shared session (do not sum per file)',
                    'result_mode':'bundle' if bundled else 'per_region_json',
                    'vad_seconds':c.vad_seconds,'prepare_seconds':c.prepare_seconds}
        else:
            for key,c in active:
                try:
                    _execute(c,cancel,progress)
                except Exception as exc:
                    checkpoint(cancel)
                    failed(key,exc)
                    contexts.pop(key)
                    shutil.rmtree(c.inputs,ignore_errors=True)
        for key,c in contexts.items():
            try:
                results[key] = _finish(c,cancel)
            except Exception as exc:
                checkpoint(cancel)
                failed(key,exc)
        return results
    finally:
        for c in contexts.values():
            shutil.rmtree(c.inputs,ignore_errors=True)
