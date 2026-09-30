"""Model-independent bounded VAD -> text ASR with honest timestamp provenance."""
from dataclasses import asdict
from pathlib import Path
import json
import shutil
from .adapters import Result, Unit, ffmpeg_path, run_process
from .catalog import checkpoint
from .media import wave_info, slice_pcm
from .timing import TimingSettings, plan_for
from .vad import detect_windows


def infer_regions(model, weights, audio, work, options, cancel, progress):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    timing = TimingSettings(**options.get('timing', {}))
    plan = plan_for(model, timing, options.get('alignment_requested', False))
    if not plan.segmented:
        raise ValueError('Region ASR requires VAD segmentation')
    original = wave_info(audio)
    vad_pcm = audio
    temporary = work / 'inputs'
    temporary.mkdir(exist_ok=True)
    try:
        if (original.sample_rate, original.channels, original.bits) != (16000, 1, 16):
            vad_pcm = temporary / 'vad.wav'
            run_process([ffmpeg_path(options.get('ffmpeg', ''), progress, cancel),
                '-hide_banner', '-loglevel', 'error', '-nostdin', '-i', str(audio),
                '-af', 'aresample=16000:async=1:first_pts=0', '-ac', '1', '-ar', '16000',
                '-c:a', 'pcm_s16le', '-y', str(vad_pcm)], cancel, progress, work / 'vad-decode.log')
        windows, record = detect_windows(model, timing.parameters(), vad_pcm, work, None,
            cancel, progress, timing.max_seconds, context_overlap=plan.align)
        requests = []
        for i, window in enumerate(windows):
            checkpoint(cancel)
            path = temporary / f'r{i:06d}.wav'
            begin = round(window.start * original.sample_rate) / original.sample_rate
            length = slice_pcm(audio, path, begin, window.end - begin, cancel)
            requests.append(dict(id=f'r{i:06d}', audio=path, begin=begin,
                                 length=length, window=window))
        from .text_requests import infer_text_requests
        results = infer_text_requests(model, weights, requests, work, options, cancel, progress)
        units, raw = [], dict(vad=record, timing_plan=asdict(plan), segments=[])
        for req in requests:
            checkpoint(cancel)
            text = results[req['id']].text.strip()
            if not any(c.isalnum() for c in text):
                continue
            w = req['window']
            units.append(Unit(req['begin'], req['begin'] + req['length'], text,
                method='vad_window' if plan.align else 'vad_segment',
                owner_start=w.owner_start if plan.align else None,
                owner_end=w.owner_end if plan.align else None))
            raw['segments'].append(dict(id=req['id'], text=text, start=req['begin'],
                end=req['begin'] + req['length'], owner_start=w.owner_start, owner_end=w.owner_end))
        (work / 'raw.json').write_text(json.dumps(raw, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        (work / 'normalized.json').write_text(json.dumps([asdict(u) for u in units], ensure_ascii=False), encoding='utf-8')
        return Result(units, raw, ''.join(u.text for u in units))
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
