"""Media preparation, normalized edit data and non-destructive export."""
import math
import os
import wave
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from common import VERSION, digest, read_json, write_json, run_process
from engines import transcribe, number
from rpp_writer import Project, Track, Item, Source, write


def prepare(registry, source, out, start, duration, rates, cancel_file=None):
    number(start, 'start', 0, 864000)
    number(duration, 'duration', 0.01, 864000)
    native = None
    try:
        with wave.open(str(source)) as w:
            if w.getcomptype() == 'NONE':
                native = dict(rate=w.getframerate(), channels=w.getnchannels(),
                              width=w.getsampwidth(), samples=w.getnframes(), kind='WAVE', offset=0.0)
    except (wave.Error, EOFError):
        pass
    if native is None:
        ffprobe = registry.tool(registry.data.get('tools', {}).get('ffprobe', 'ffprobe'))
        run_process([ffprobe, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', source],
                    out / 'probe', 60, cancel_file)
        probe = read_json(str(out / 'probe') + '.stdout.log')
        streams = [s for s in probe['streams'] if s.get('codec_type') == 'audio']
        if not streams:
            raise ValueError('No audio stream')
        stream = streams[0]
        origin = float(stream.get('start_time', 0)) - float(probe.get('format', {}).get('start_time', 0))
        video = any(s.get('codec_type') == 'video' for s in probe['streams'])
        codec = stream.get('codec_name', '')
        kind = 'VIDEO' if video else {'mp3': 'MP3', 'flac': 'FLAC', 'vorbis': 'VORBIS'}.get(codec, 'VIDEO')
        native = dict(rate=int(stream['sample_rate']), channels=int(stream['channels']), kind=kind,
                      offset=origin, stream_index=stream['index'])
        write_json(out / 'source_probe.json', probe)
    audio_start = round(start * native['rate']) / native['rate']
    source_start = audio_start + native['offset']
    if source_start < 0:
        raise ValueError('Negative source timeline origin needs manual mapping; refusing an inaccurate RPP')
    result = {}
    for rate in sorted(rates):
        dest = out / f'inference_{rate}.wav'
        if source.suffix.lower() == '.wav' and native.get('width') == 2 and native['channels'] == 1 and native['rate'] == rate:
            with wave.open(str(source)) as src, wave.open(str(dest), 'wb') as dst:
                src.setpos(min(round(audio_start * rate), src.getnframes()))
                dst.setparams(src.getparams())
                dst.writeframes(src.readframes(round(duration * rate)))
        else:
            ffmpeg = registry.tool(registry.data.get('tools', {}).get('ffmpeg', 'ffmpeg'))
            begin = round(audio_start * native['rate'])
            end = begin + round(duration * native['rate'])
            run_process([ffmpeg, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', source,
                         '-map', '0:a:0', '-vn', '-af', f'atrim=start_sample={begin}:end_sample={end},asetpts=PTS-STARTPTS',
                         '-ac', '1', '-ar', rate, '-c:a', 'pcm_s16le', dest], out / f'decode_{rate}', 600, cancel_file)
        with wave.open(str(dest)) as w:
            if not w.getnframes():
                raise ValueError('Selected range contains no audio')
            result[rate] = dict(path=str(dest), duration=w.getnframes()/w.getframerate(), sha256=digest(dest))
    return native, source_start, result


def assign(units, turns, duration):
    """Conservative assignment. Never invent word boundaries or duplicate mixed audio."""
    if turns is not None:
        for turn in turns:
            if not all(isinstance(turn[k], (int, float)) and math.isfinite(turn[k]) for k in ('start','end')) or turn['end'] <= turn['start']:
                raise ValueError('Invalid diarization timestamp')
    result, warnings = [], []
    previous = -1
    for i, original in enumerate(units):
        a, b = original['start'], original['end']
        if not all(isinstance(t, (int, float)) and not isinstance(t, bool) and math.isfinite(t) for t in (a, b)) or b <= a:
            raise ValueError(f'Invalid timestamp in ASR unit {i}')
        text = str(original.get('text', '')).replace('\r', ' ').replace('\n', ' ').strip()
        if not text:
            continue
        review = []
        if a < 0 or b > duration:
            review.append('clipped_to_audio_bounds')
        if a < previous:
            review.append('overlapping_or_out_of_order_ASR_units')
        previous = b
        a, b = max(0.0, a), min(duration, b)
        if b <= a:
            warnings.append(f'unit {i}: outside audio, excluded from RPP; retained in raw output')
            continue
        speaker = original.get('speaker') or 'Speaker 01'
        if turns is not None:
            spans = defaultdict(list)
            for turn in turns:
                lo, hi = max(a, turn['start']), min(b, turn['end'])
                if hi > lo and turn.get('speaker') is not None:
                    spans[turn['speaker']].append((lo, hi))
            totals = {}
            for sid, values in spans.items():
                merged = []
                for lo, hi in sorted(values):
                    if merged and lo <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], hi)
                    else:
                        merged.append([lo, hi])
                totals[sid] = sum(hi-lo for lo, hi in merged)
            ranked = sorted(totals.items(), key=lambda x: -x[1])
            if not ranked or ranked[0][1] < (b-a)*0.5 or (len(ranked) > 1 and ranked[1][1] > (b-a)*0.2):
                speaker = 'UNRESOLVED'
                review.append('speaker_change_or_overlap_requires_review')
            else:
                speaker = ranked[0][0]
        result.append(dict(id=f'utterance-{i:05d}', start=a, end=b, text=text, speaker=speaker,
                           granularity=original.get('granularity', 'segment'), review=review))
    assembled = []
    for seg in result:
        if seg['granularity'] == 'token':
            seg['review'].append('token_emission_timestamps_not_forced_aligned')
        prev = assembled[-1] if assembled else None
        if (prev and prev['granularity'] == 'token_group' and seg['granularity'] == 'token'
                and prev['speaker'] == seg['speaker'] and seg['start'] - prev['end'] <= 0.5
                and seg['end'] - prev['start'] <= 12 and prev['text'][-1:] not in '。！？!?.'):
            left, right = prev['text'], seg['text']
            sep = ' ' if left[-1:].isascii() and left[-1:].isalnum() and right[:1].isascii() and right[:1].isalnum() else ''
            prev['text'] = left + sep + right
            prev['end'] = max(prev['end'], seg['end'])
            prev['review'] = sorted(set(prev['review'] + seg['review']))
            prev['source_ids'].append(seg['id'])
        else:
            if seg['granularity'] == 'token':
                seg['granularity'] = 'token_group'
                seg['source_ids'] = [seg['id']]
            assembled.append(seg)
    return assembled, warnings


def export_project(data, path):
    path = Path(path).resolve()
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported transcript schema')
    media = data['media']
    source_path = Path(media['path']).resolve()
    if source_path == path:
        raise ValueError('Output cannot overwrite source media')
    if not source_path.is_file():
        raise FileNotFoundError('Referenced source media is missing; update media.path before exporting')
    try:
        ref = os.path.relpath(source_path, path.parent).replace('\\', '/')
    except ValueError:
        ref = str(source_path).replace('\\', '/')
    source = Source(ref, media['kind'])
    sr = int(media['sample_rate'])
    tracks = {}
    for seg in sorted(data['segments'], key=lambda x: x['start']):
        if not 0 <= float(seg['start']) < float(seg['end']) <= float(data['clip_duration']):
            raise ValueError('Edited segment is outside the selected audio range')
        start = Fraction(round(float(seg['start']) * sr), sr)
        end = Fraction(round(float(seg['end']) * sr), sr)
        if end <= start:
            raise ValueError('Segment shorter than one source sample')
        offset = Fraction(str(data.get('clip_source_start', 0))) + start
        speaker = str(seg.get('speaker') or 'UNRESOLVED')
        tracks.setdefault(speaker, []).append(Item(str(seg['text']), source, offset, offset, end-start))
    project = Project(tuple(Track(k, tuple(v)) for k, v in tracks.items()), sr)
    write(project, path)
    return path


def run_job(registry, source, out, asr_id, diar_id=None, start=0, duration=55,
            asr_device='cpu', diar_device='cpu', asr_params=None, diar_params=None,
            timeout=1800, cancel_file=None, log=print):
    source, out = Path(source).resolve(), Path(out).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    asr = registry.model(asr_id)
    diar = registry.model(diar_id) if diar_id and diar_id != 'none' else None
    if asr['type'] not in ('asr', 'joint') or (diar and diar['type'] != 'diarization'):
        raise ValueError('ASR and diarization selectors require matching model types')
    from engines import native_command
    for model, device, params in [(asr, asr_device, asr_params)] + ([(diar, diar_device, diar_params)] if diar else []):
        native_command(registry, model, 'preflight.wav', out, device, params)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output directory must be empty; existing results are never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    try:
        log('Preparing inference audio; the source media remains unchanged.')
        rates = {m.get('sample_rate', 16000) for m in (asr, diar) if m}
        native, source_start, inputs = prepare(registry, source, out, start, duration, rates, cancel_file)
        log(f'Running ASR: {asr_id}')
        units, asr_run = transcribe(registry, asr, Path(inputs[asr.get('sample_rate', 16000)]['path']),
                                    out / 'raw_asr', asr_device, asr_params, timeout, cancel_file)
        turns, diar_run = None, None
        if diar:
            log(f'Running diarization: {diar_id}')
            turns, diar_run = transcribe(registry, diar, Path(inputs[diar.get('sample_rate', 16000)]['path']),
                                         out / 'raw_diarization', diar_device, diar_params, timeout, cancel_file)
        duration_actual = inputs[asr.get('sample_rate', 16000)]['duration']
        segments, warnings = assign(units, turns, duration_actual)
        data = dict(schema_version=1, app_version=VERSION,
                    media=dict(path=str(source), sha256=digest(source), kind=native['kind'],
                               sample_rate=native['rate'], audio_stream_time_offset=native['offset']),
                    clip_source_start=source_start, clip_duration=duration_actual,
                    segments=segments, raw_units=units, raw_speaker_turns=turns,
                    warnings=warnings, runs=dict(asr=asr_run, diarization=diar_run), inputs=inputs)
        write_json(out / 'transcript.json', data)
        export_project(data, out / 'project.rpp')
        write_json(out / 'status.json', dict(status='completed', segments=len(segments), warnings=warnings))
        log(f'Completed: {out / "project.rpp"}')
        return data
    except BaseException as exc:
        write_json(out / 'status.json', dict(status='failed', error=str(exc)))
        raise
