"""Prepare source-relative VAD windows across files and share a provider session."""
from dataclasses import asdict
from pathlib import Path
import json
import shutil
from .catalog import checkpoint
from .domain import Result, Unit
from .media import wave_info, slice_pcm
from .providers import Batch, Request, provider_for
from .timing import TimingSettings, plan_for
from .vad import detect_windows


def infer_many(model, weights, items, work, options, cancel, progress, max_bytes):
    from .adapters import ffmpeg_path, run_process
    from .diagnostics import persist_tree
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    timing = TimingSettings(**options.get('timing', {}))
    plan = plan_for(model, timing, options.get('alignment_requested',False))
    if not plan.segmented:
        raise ValueError('Host-region ASR requires segmentation')
    contexts, requests, batch = {}, [], Batch()
    try:
        for item in items:
            checkpoint(cancel)
            folder = work/item.id
            temporary = folder/'inputs'
            temporary.mkdir(parents=True, exist_ok=True)
            contexts[item.id] = dict(work=folder, temporary=temporary, regions=[], record={})
            context = contexts[item.id]
            try:
                original = wave_info(item.audio)
                vad_pcm = item.audio
                if (original.sample_rate,original.channels,original.bits) != (16000,1,16):
                    vad_pcm = temporary/'vad.wav'
                    run_process([ffmpeg_path(options.get('ffmpeg',''),progress,cancel),'-hide_banner','-loglevel','error',
                        '-nostdin','-i',str(item.audio),'-af','aresample=16000:async=1:first_pts=0',
                        '-ac','1','-ar','16000','-c:a','pcm_s16le','-y',str(vad_pcm)],
                        cancel,progress,folder/'vad-decode.log')
                params = timing.parameters()
                if options.get('parameters',{}).get('vad_model'):
                    params['vad_model'] = options['parameters']['vad_model']
                windows,record = detect_windows(model,params,vad_pcm,folder,None,cancel,progress,
                                                timing.max_seconds,context_overlap=plan.align)
                context['record'] = record
                for number,window in enumerate(windows):
                    key = f'{item.id}r{number:06d}'
                    audio = temporary/(key+'.wav')
                    begin = round(window.start*original.sample_rate)/original.sample_rate
                    length = slice_pcm(item.audio,audio,begin,window.end-begin,cancel)
                    context['regions'].append(dict(id=key,begin=begin,length=length,window=window))
                    requests.append(Request(key,audio))
            except Exception as exc:
                checkpoint(cancel)
                batch.errors[item.id] = exc
                bad = {r['id'] for r in context['regions']}
                requests = [r for r in requests if r.id not in bad]
        native_root = work/'shared-provider'
        native = provider_for(model).infer_many(model,weights,requests,native_root,
            dict(options,_text_only=True),cancel,progress,max_bytes) if requests else Batch()
        for key,context in contexts.items():
            if key in batch.errors:
                continue
            units,records = [],[]
            failures = []
            for region in context['regions']:
                request_id = region['id']
                persist_tree(native_root/request_id,context['work']/'native'/request_id)
                if request_id in native.errors or request_id not in native.results:
                    failures.append(str(native.errors.get(request_id,'Missing provider result')))
                    continue
                text = native.results[request_id].text.strip()
                window = region['window']
                records.append(dict(id=request_id,text=text,start=region['begin'],end=region['begin']+region['length'],
                                    owner_start=window.owner_start,owner_end=window.owner_end))
                if any(c.isalnum() for c in text):
                    units.append(Unit(region['begin'],region['begin']+region['length'],text,
                        method='vad_window' if plan.align else 'vad_segment',
                        owner_start=window.owner_start if plan.align else None,
                        owner_end=window.owner_end if plan.align else None))
            raw = dict(vad=context['record'],timing_plan=asdict(plan),segments=records,errors=failures)
            (context['work']/'raw.json').write_text(json.dumps(raw,ensure_ascii=False,allow_nan=False),encoding='utf-8')
            (context['work']/'normalized.json').write_text(json.dumps([asdict(u) for u in units],ensure_ascii=False),encoding='utf-8')
            if failures:
                batch.errors[key] = ValueError('Incomplete speech-region transcription: ' + '; '.join(failures[:3]))
            else:
                batch.results[key] = Result(units,raw,''.join(u.text for u in units))
        return batch
    finally:
        for context in contexts.values():
            shutil.rmtree(context['temporary'],ignore_errors=True)
