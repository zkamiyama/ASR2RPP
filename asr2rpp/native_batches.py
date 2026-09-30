"""Native subprocess implementations; no GUI, queue statuses or RPP knowledge."""
from pathlib import Path
import json
import shutil
from .catalog import checkpoint
from .providers import Batch, _save
from .domain import Result
from .native_profile import profile
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
            '-t',str(options.get('threads',4)),'-ojf','-np']
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
                    '--language',options.get('language') or model.defaults.get('language','ja'),output_flag,str(output)]
                for key,value in params.items():
                    args += ['--request-option',f'{key}={scalar(value)}']
            else:
                sequence = folder/'requests.json'
                payload = []
                for request in chunk:
                    item = dict(id=request.id, audio=str(request.audio.resolve()), options=params)
                    if model.task in {'asr', 'align'}:
                        item.update(text=request.text, language=options.get('language') or model.defaults.get('language','ja'))
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
