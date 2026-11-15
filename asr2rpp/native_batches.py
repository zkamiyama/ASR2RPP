"""Native subprocess implementations; no GUI, queue statuses or RPP knowledge."""
from pathlib import Path
import json
import shutil
from .catalog import checkpoint
from .providers import Batch, _save
from .domain import Result
from .native_profile import profile, request_fields
from .adapters import (executable,run_process,parse_whisper,parse_audio,split_engine_parameters,
    validate_model_parameter_constraints,whisper_parameter_args,audio_session_args,scalar)


def _read(output, parse, batch, req, process_error):
    try:
        if not output.is_file():
            raise ValueError(f'No complete result for {req.id}: {process_error or "missing output"}')
        batch.results[req.id] = parse(output)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        batch.errors[req.id] = exc


def whisper_many(model, weights, requests, work, options, cancel, progress, max_bytes):
    from .inference_policy import policy_for
    from .whisper_io import capabilities, response_command
    from .vad_asr import infer_vad_many, command_batches
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    batch = Batch()
    if policy_for(model).segmentation == 'vad':
        batch.results = infer_vad_many(model, weights,
            [(r.id,r.audio,work/r.id) for r in requests], options, cancel, progress,
            lambda key,exc:batch.errors.__setitem__(key,exc), work)
        _save(batch,requests,work,work/'asr-session.log')
        return batch
    binary = executable(model.runtime, options.get('device','cpu'),options.get('executable',''))
    params, session = split_engine_parameters(model,options.get('parameters',{}))
    validate_model_parameter_constraints(model,params,session)
    if session:
        raise ValueError('Whisper does not accept audio.cpp session options')
    base = [str(binary),'-m',str(weights),'-l',options.get('language') or model.defaults.get('language','ja'),
            '-t',str(options.get('threads',4)),'-ojf']
    if options.get('device') == 'cpu': base.append('-ng')
    base += whisper_parameter_args(params)
    features = capabilities(binary, work, cancel, progress)
    pairs = [(r.audio,work/r.id/'native') for r in requests]
    for _,prefix in pairs: prefix.parent.mkdir(parents=True,exist_ok=True)
    by_prefix = {str(work/r.id/'native'):r for r in requests}
    maximum = profile(model).max_batch_items
    if 'ASR2RPP_RESPONSE_V1' in features:
        groups = (pairs[i:i+maximum] for i in range(0,len(pairs),maximum))
    else:
        groups = command_batches(base,pairs,max_items=min(96,maximum))
    for number,group in enumerate(groups):
        checkpoint(cancel)
        chunk = [by_prefix[str(prefix)] for _,prefix in group]
        args = base + [v for audio,prefix in group for v in ('-f',str(audio),'-of',str(prefix))]
        if 'ASR2RPP_RESPONSE_V1' in features:
            args = response_command(args,work/f'chunk-{number}.args')
        log = work/f'chunk-{number}.log'
        error = None
        try:
            run_process(args,cancel,progress,log)
        except Exception as exc:
            checkpoint(cancel)
            error = exc
        finally:
            _save(batch,chunk,work,log)
        for req in chunk:
            output = work/req.id/'native.json'
            _read(output,lambda p:parse_whisper(json.loads(p.read_text(encoding='utf-8-sig'))),batch,req,error)
        _save(batch,chunk,work)
    return batch


def audio_many(model, weights, requests, work, options, cancel, progress, max_bytes):
    from .alignment import chunks_by_size
    from .text_requests import checked_text
    if profile(model).output == 'custom':
        return custom_many(model, weights, requests, work, options, cancel, progress, max_bytes)
    work = Path(work)
    work.mkdir(parents=True,exist_ok=True)
    batch = Batch()
    contract = profile(model)
    binary = executable(model.runtime,options.get('device','cpu'),options.get('executable',''))
    params, session = split_engine_parameters(model,options.get('parameters',{}))
    validate_model_parameter_constraints(model,params,session)
    if options.get('_text_only'):
        output_kind = 'text'
    else:
        output_kind = contract.output
    base = [str(binary),'--task',model.task,'--family',model.family,'--model',str(weights),
            '--backend','best' if options.get('device')=='auto' else options.get('device','cpu'),
            '--threads',str(options.get('threads',4))]
    base += audio_session_args(model,session)
    maximum = 1 if contract.mode == 'streaming' and not options.get('_text_only') else contract.max_batch_items
    output_flag = {'text':'--text-out','words':'--words-out','segments':'--segments-out','turns':'--turns-out'}[output_kind]
    number = 0
    for group in chunks_by_size(requests,lambda r:r.audio,max_bytes):
        for offset in range(0,len(group),maximum):
            checkpoint(cancel)
            chunk = group[offset:offset+maximum]
            folder = work/f'chunk-{number}'
            folder.mkdir(parents=True,exist_ok=True)
            number += 1
            output = folder/('native.txt' if output_kind=='text' else 'native.json')
            streaming = contract.mode == 'streaming' and not options.get('_text_only')
            if streaming:
                args = base + ['--mode','streaming','--audio',str(chunk[0].audio),
                    output_flag,str(output)]
                if contract.pass_language:
                    args += ['--language', options.get('language') or model.defaults.get('language', 'ja')]
                for key,value in params.items():
                    args += ['--request-option',f'{key}={scalar(value)}']
            else:
                sequence = folder/'requests.json'
                payload = []
                for request in chunk:
                    item = dict(id=request.id, audio=str(request.audio.resolve()), options=params)
                    item.update(request_fields(model, options, request.text))
                    payload.append(item)
                sequence.write_text(json.dumps({'requests':payload},ensure_ascii=False,allow_nan=False),encoding='utf-8')
                args = base + ['--mode','offline','--request-sequence',str(sequence),output_flag,str(output)]
            log = folder/'engine.log'
            error = None
            try:
                run_process(args,cancel,progress,log)
            except Exception as exc:
                checkpoint(cancel)
                error = exc
            finally:
                _save(batch,chunk,work,log)
            for req in chunk:
                path = output if streaming else output.with_name(output.stem+'_'+req.id+output.suffix)
                dest = work/req.id
                dest.mkdir(parents=True,exist_ok=True)
                if path.is_file(): shutil.copy2(path,dest/('native-output'+path.suffix))
                if output_kind == 'text':
                    parse = lambda p:Result([],{'text':checked_text(p)},checked_text(p))
                else:
                    parse = lambda p:parse_audio(json.loads(p.read_text(encoding='utf-8-sig')),model.task,model.family,model.sample_rate)
                _read(path,parse,batch,req,error)
            _save(batch,chunk,work)
    return batch



def _retain_custom_files(folder, requests, work):
    # Queue cleanup keeps per-request folders, not the temporary chunk folder.
    # Preserve bytes BEFORE parsing so malformed/partial outputs survive too.
    for req in requests:
        dest = work / req.id / 'native'
        dest.mkdir(parents=True, exist_ok=True)
        names = ('manifest.json', 'requests.json', f'text_{req.id}.txt',
                 f'words_{req.id}.json', f'turns_{req.id}.json')
        for name in names:
            path = folder / name
            if path.is_file():
                shutil.copy2(path, dest / name)


def custom_many(model, weights, requests, work, options, cancel, progress, max_bytes):
    """v0.9.5 structured custom results plus separate text/turn evidence.

    One complete recording per request, not VAD chunks. A file's acoustic IDs
    are local to that request even when a runtime session is reused.
    """
    from .alignment import chunks_by_size
    from .text_requests import checked_text
    from .nemotron_transcribe import read_manifest, validate_duration, parse_text_result
    from .media import wave_info
    batch, work = Batch(), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    params, session = split_engine_parameters(model, options.get('parameters', {}))
    validate_model_parameter_constraints(model, params, session)
    text_only = options.get('_text_only', False)
    if text_only and (params.get('return_timestamps') is not False or params.get('diarization') is not False):
        raise ValueError('Nemotron text-only requests require return_timestamps=false and diarization=false')
    if not text_only and params.get('return_timestamps') is False:
        raise ValueError('Nemotron Transcribe timed export requires return_timestamps=true')
    if params.get('template_name', 'default') != 'default':
        raise ValueError('Nemotron Transcribe timed export requires the transcription template')
    if options.get('timing'):
        from .timing import TimingSettings, plan_for
        plan_for(model, TimingSettings(**options['timing']), options.get('alignment_requested', False))
    binary = executable(model.runtime, options.get('device', 'cpu'), options.get('executable', ''))
    base = [str(binary), '--task', model.task, '--family', model.family, '--model', str(weights),
            '--backend', 'best' if options.get('device') == 'auto' else options.get('device', 'cpu'),
            '--threads', str(options.get('threads', 4)), '--mode', 'offline']
    base += audio_session_args(model, session)
    seen = set()
    for req in requests:
        if not req.id.isascii() or not req.id.replace('_', '').isalnum() or req.id in seen:
            raise ValueError('Custom request IDs must be unique ASCII identifiers')
        seen.add(req.id)
    number = 0
    maximum = model.execution.get('text_batch_items', 16) if text_only else profile(model).max_batch_items
    for group in chunks_by_size(requests, lambda r: r.audio, max_bytes):
        for offset in range(0, len(group), maximum):
            chunk = group[offset:offset + maximum]
            checkpoint(cancel)
            folder = work / f'chunk-{number}'
            folder.mkdir(parents=True, exist_ok=True)
            number += 1
            manifest, sequence = folder / 'manifest.json', folder / 'requests.json'
            try:
                limit = 28 if text_only else profile(model).max_audio_seconds
                for req in chunk:
                    if limit and wave_info(req.audio).duration > limit:
                        raise ValueError(f'Nemotron Transcribe request exceeds {limit:g}s; '
                                         'use bounded text windows, never split and relabel native speakers')
                sequence.write_text(json.dumps({'requests': [dict(id=r.id,
                    audio=str(r.audio.resolve()), options=params,
                    **request_fields(model, options, r.text)) for r in chunk]},
                    ensure_ascii=False, allow_nan=False), encoding='utf-8')
                args = base + ['--request-sequence', str(sequence),
                    '--batch-manifest-out', str(manifest), '--text-out', str(folder / 'text.txt')]
                if not text_only:
                    args += ['--words-out', str(folder / 'words.json'), '--turns-out', str(folder / 'turns.json')]
                run_process(args, cancel, progress, folder / 'engine.log')
                _retain_custom_files(folder, chunk, work)
                records = read_manifest(manifest, {r.id for r in chunk})
            except Exception as exc:
                _retain_custom_files(folder, chunk, work)
                _save(batch, chunk, work, folder / 'engine.log')
                checkpoint(cancel)
                batch.errors.update({r.id: exc for r in chunk})
                continue
            for req in chunk:
                dest = work / req.id
                dest.mkdir(parents=True, exist_ok=True)
                try:
                    record = records.get(req.id)
                    if record is None:
                        raise ValueError('No custom result for request ' + req.id)
                    raw = dict(record)
                    raw['text'] = checked_text(folder / f'text_{req.id}.txt')
                    for name in ('words', 'turns'):
                        path = folder / f'{name}_{req.id}.json'
                        # Upstream intentionally emits no file for an empty vector.
                        raw['word_timestamps' if name == 'words' else 'speaker_turns'] = (
                            json.loads(path.read_text(encoding='utf-8-sig')) if path.is_file() else [])
                    (dest / 'raw.json').write_text(json.dumps(raw, ensure_ascii=False,
                        allow_nan=False), encoding='utf-8')
                    # Keep raw even when diagnostics/coverage prohibit timed export.
                    if text_only:
                        batch.results[req.id] = parse_text_result(raw)
                    else:
                        result = parse_audio(raw, model.task, model.family, model.sample_rate)
                        batch.results[req.id] = validate_duration(result, float(wave_info(req.audio).duration))
                except (ValueError, TypeError, OSError, KeyError) as exc:
                    batch.errors[req.id] = exc
            _save(batch, chunk, work, folder / 'engine.log')
    return batch
