"""Stage-major queue scheduler.

The GUI queue is processed by model stage rather than file:
SEP -> ASR -> forced alignment -> diarization -> RPP.

Native model processes are never kept alive across different model families.
whisper.cpp receives multiple files per process (one model context), while
audio.cpp uses one offline batch session per RAM-bounded chunk.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from pathlib import Path
import copy
import json
import os
import shutil
import tempfile
import threading

from . import pipeline as core
from . import preprocessing as pre
from .catalog import Cancelled, checkpoint, cache_root, digest, resolve_model
from .adapters import (
    Result, Unit, executable, ffmpeg_path, parse_audio, parse_whisper,
    run_process, scalar, whisper_parameter_args, split_engine_parameters,
    audio_session_args, validate_model_parameter_constraints,
)


DEFAULT_BATCH_AUDIO_RAM_MB = 512


@dataclass
class QueueJob:
    index: int
    key: str
    source: Path
    output: Path
    report: Path
    source_sha256: str
    manifest: dict
    warnings: list[str] = field(default_factory=list)
    units: list[Unit] = field(default_factory=list)
    duration: float = 0.0
    inference_source: Path | None = None
    preprocess_info: dict | None = None
    failed: bool = False


def _jsonable_result(job: QueueJob, task: str, result: Result) -> None:
    directory = job.report / task
    directory.mkdir(parents=True, exist_ok=True)
    core.json_write(directory / 'raw.json', result.raw)
    core.json_write(directory / 'normalized.json', [asdict(unit) for unit in result.units])


def _write_manifest(job: QueueJob) -> None:
    core.json_write(job.report / 'manifest.json', job.manifest)


def _fail(job: QueueJob, error, item_callback) -> None:
    if job.failed:
        return
    job.failed = True
    job.manifest.update(status='failed', error=str(error))
    _write_manifest(job)
    item_callback(job.index, '失敗', str(error))


def _active(jobs):
    return [job for job in jobs if not job.failed]


def _chunks_by_size(items, path_of, max_bytes: int):
    chunk, used = [], 0
    for item in items:
        size = max(1, path_of(item).stat().st_size)
        if chunk and used + size > max_bytes:
            yield chunk
            chunk, used = [], 0
        chunk.append(item)
        used += size
    if chunk:
        yield chunk


def _chunks_for_command(items, path_of, max_chars: int = 22000, max_items: int = 96):
    chunk, used = [], 0
    for item in items:
        cost = len(str(path_of(item))) + 140
        if chunk and (len(chunk) >= max_items or used + cost > max_chars):
            yield chunk
            chunk, used = [], 0
        chunk.append(item)
        used += cost
    if chunk:
        yield chunk


def _link_inputs(chunk, path_of, directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    mapping = {}
    for item in chunk:
        dst = directory / f'{item.key}.wav'
        src = path_of(item)
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        mapping[item.key] = dst
    return mapping


def _copy_log(log: Path, jobs, filename: str):
    if not log.is_file():
        return
    for job in jobs:
        try:
            shutil.copy2(log, job.report / filename)
        except OSError:
            pass


def _stage_parameters(model, stage):
    request, session = split_engine_parameters(model, stage.parameters or {})
    validate_model_parameter_constraints(model, request, session)
    return request


def _session_args(model, stage):
    request, session = split_engine_parameters(model, stage.parameters or {})
    validate_model_parameter_constraints(model, request, session)
    return audio_session_args(model, session)


def _whisper_batch(model, weights: Path, stage, jobs, audio_paths, root: Path,
                   cancel, progress, item_callback):
    binary = executable(model.runtime, stage.device, stage.executable)
    parameters = _stage_parameters(model, stage)
    results = {}
    for chunk_no, chunk in enumerate(_chunks_for_command(jobs, lambda j: audio_paths[j.key]), 1):
        checkpoint(cancel)
        out = root / f'chunk-{chunk_no}' / 'out'
        out.mkdir(parents=True, exist_ok=True)
        argv = [str(binary), '-m', str(weights), '-l', stage.language or model.defaults.get('language', 'ja'),
                '-t', str(stage.threads), '-ojf', '-np']
        if stage.device == 'cpu':
            argv.append('-ng')
        elif stage.device not in {'auto', 'vulkan', 'cuda', 'metal'}:
            raise ValueError('Unsupported Whisper device')
        argv += whisper_parameter_args(parameters)
        for job in chunk:
            item_callback(job.index, 'ASR', '')
            argv += ['-f', str(audio_paths[job.key]), '-of', str(out / job.key)]
        log = root / f'chunk-{chunk_no}' / 'engine.log'
        try:
            run_process(argv, cancel, progress, log)
            for job in chunk:
                path = out / f'{job.key}.json'
                if not path.is_file():
                    raise ValueError(f'whisper.cpp did not produce JSON for {job.source.name}')
                raw = json.loads(path.read_text(encoding='utf-8-sig'))
                results[job.key] = parse_whisper(raw)
            _copy_log(log, chunk, f'asr-batch-{chunk_no}.log')
        except BaseException as exc:
            if cancel.is_set():
                raise
            for job in chunk:
                _fail(job, exc, item_callback)
    return results


def _audio_batch(model, weights: Path, stage, jobs, audio_paths, root: Path,
                 cancel, progress, item_callback, task: str, max_bytes: int):
    binary = executable(model.runtime, stage.device, stage.executable)
    parameters = _stage_parameters(model, stage)
    results = {}
    chunks = list(_chunks_by_size(jobs, lambda j: audio_paths[j.key], max_bytes))
    for chunk_no, chunk in enumerate(chunks, 1):
        checkpoint(cancel)
        chunk_root = root / f'chunk-{chunk_no}'
        inputs = _link_inputs(chunk, lambda j: audio_paths[j.key], chunk_root / 'inputs')
        output_root = chunk_root / 'outputs'
        output_root.mkdir(parents=True, exist_ok=True)
        argv = [str(binary), '--task', task, '--family', model.family, '--model', str(weights),
                '--backend', 'best' if stage.device == 'auto' else stage.device,
                '--mode', 'offline', '--batch-audio-dir', str(chunk_root / 'inputs'),
                '--threads', str(stage.threads)]
        if task == 'asr':
            argv += ['--language', stage.language or model.defaults.get('language', 'ja')]
            if model.family == 'vibevoice_asr':
                base = output_root / 'segments.json'
                argv += ['--segments-out', str(base)]
            else:
                base = output_root / 'words.json'
                argv += ['--words-out', str(base)]
        elif task == 'diar':
            base = output_root / 'turns.json'
            argv += ['--turns-out', str(base)]
        else:
            raise ValueError(f'Unsupported timed batch task: {task}')
        for key, value in parameters.items():
            if not key.replace('_', '').replace('.', '').isalnum():
                raise ValueError('Invalid audio.cpp request parameter')
            argv += ['--request-option', f'{key}={scalar(value)}']
        argv += _session_args(model, stage)
        log = chunk_root / 'engine.log'
        try:
            for job in chunk:
                item_callback(job.index, 'ASR' if task == 'asr' else 'Diarization', '')
            run_process(argv, cancel, progress, log)
            for job in chunk:
                path = base.parent / f'{base.stem}_{job.key}{base.suffix}'
                if not path.is_file():
                    raise ValueError(f'audio.cpp did not produce {task} timestamps for {job.source.name}')
                raw = json.loads(path.read_text(encoding='utf-8-sig'))
                results[job.key] = parse_audio(raw, task, model.family, model.sample_rate)
            _copy_log(log, chunk, f'{task}-batch-{chunk_no}.log')
        except BaseException as exc:
            if cancel.is_set():
                raise
            for job in chunk:
                _fail(job, exc, item_callback)
    return results


@dataclass
class AlignRequest:
    key: str
    job: QueueJob
    audio: Path
    begin: float
    length: float
    text: str


def _align_batch(model, weights: Path, stage, requests: list[AlignRequest], root: Path,
                 cancel, progress, item_callback, max_bytes: int):
    binary = executable(model.runtime, stage.device, stage.executable)
    results = {}
    # audio.cpp materializes all request audio in a batch before inference.
    chunks = list(_chunks_by_size(requests, lambda r: r.audio, max_bytes))
    parameters = _stage_parameters(model, stage)
    for chunk_no, chunk in enumerate(chunks, 1):
        checkpoint(cancel)
        chunk_root = root / f'chunk-{chunk_no}'
        inputs = _link_inputs(chunk, lambda r: r.audio, chunk_root / 'inputs')
        request_json = chunk_root / 'requests.json'
        request_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {'requests': []}
        for req in chunk:
            payload['requests'].append({
                'id': req.key,
                'audio': str(Path('inputs') / f'{req.key}.wav').replace('\\', '/'),
                'text': req.text,
                'language': stage.language or model.defaults.get('language', 'ja'),
                'options': parameters,
            })
            item_callback(req.job.index, 'Forced Align', '')
        request_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        out = chunk_root / 'out'
        out.mkdir(exist_ok=True)
        base = out / 'words.json'
        argv = [str(binary), '--task', 'align', '--family', model.family, '--model', str(weights),
                '--backend', 'best' if stage.device == 'auto' else stage.device,
                '--mode', 'offline', '--request-sequence', str(request_json),
                '--threads', str(stage.threads), '--words-out', str(base)]
        argv += _session_args(model)
        log = chunk_root / 'engine.log'
        try:
            run_process(argv, cancel, progress, log)
            for req in chunk:
                path = out / f'words_{req.key}.json'
                if not path.is_file():
                    raise ValueError(f'Aligner did not produce words for {req.job.source.name}')
                raw = json.loads(path.read_text(encoding='utf-8-sig'))
                results[req.key] = parse_audio(raw, 'align', model.family, model.sample_rate)
            _copy_log(log, list({req.job.index: req.job for req in chunk}.values()), f'align-batch-{chunk_no}.log')
        except BaseException as exc:
            if cancel.is_set():
                raise
            for req in chunk:
                _fail(req.job, exc, item_callback)
    return results


def _decode_for_model(job: QueueJob, destination: Path, rate: int, settings,
                      cancel, progress):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if getattr(settings, 'preprocess', None) is not None:
        local = replace(settings, clip_start=0.0, clip_duration=0.0)
        return core.decode(job.inference_source, destination, rate, local, cancel, progress)
    return core.decode(job.source, destination, rate, settings, cancel, progress)


def _prepare_jobs(indexed_paths, settings, catalog, item_callback):
    jobs = []
    for index, value in indexed_paths:
        source = Path(value).expanduser().resolve()
        if not source.is_file() or source.suffix.lower() not in core.MEDIA_EXTENSIONS:
            item_callback(index, '失敗', 'Unsupported or missing input')
            continue
        siblings = ('_vocals.wav',) if (getattr(settings, 'preprocess', None) is not None and
                                         getattr(settings, 'reference_audio', 'original') == 'processed') else ()
        output, report = core.reserve_output(source, settings, sibling_suffixes=siblings)
        source_sha = digest(source)
        manifest = {
            'source': str(source),
            'source_sha256': source_sha,
            'settings': asdict(settings),
            'status': 'queued',
            'queue_strategy': 'stage_major',
            'source_unchanged': None,
        }
        job = QueueJob(index, f'q{index:06d}', source, output, report, source_sha, manifest)
        _write_manifest(job)
        jobs.append(job)
    return jobs


def _resolve_models(settings, catalog, cancel, progress):
    selected = {}
    for task, stage in [
        ('preprocess', getattr(settings, 'preprocess', None)),
        ('asr', settings.asr),
        ('align', settings.align),
        ('diar', settings.diar),
    ]:
        if stage is None:
            continue
        model = catalog[stage.model_id]
        executable(model.runtime, stage.device, stage.executable)
        path, provenance = resolve_model(model, cancel, progress)
        selected[task] = (model, path, provenance, stage)
    ffmpeg_path(settings.ffmpeg)
    return selected


def run_queue(indexed_paths, settings, catalog, cancel: threading.Event, progress,
              item_callback, batch_audio_ram_mb: int = DEFAULT_BATCH_AUDIO_RAM_MB):
    """Run a GUI queue stage-by-stage.

    Returns completed output paths keyed by original queue index.
    """
    settings = copy.deepcopy(settings)
    settings.validate(catalog)
    max_bytes = max(128, min(int(batch_audio_ram_mb), 8192)) * 1024 * 1024
    jobs = _prepare_jobs(indexed_paths, settings, catalog, item_callback)
    if not jobs:
        return {}
    try:
        selected = _resolve_models(settings, catalog, cancel, progress)
    except BaseException as exc:
        for job in jobs:
            _fail(job, exc, item_callback)
        return {}

    for job in jobs:
        job.manifest['models'] = {name: provenance for name, (_m, _p, provenance, _s) in selected.items()}
        job.manifest['status'] = 'running'
        _write_manifest(job)

    cache_directory = cache_root()
    cache_directory.mkdir(parents=True, exist_ok=True)
    completed = {}
    try:
        with tempfile.TemporaryDirectory(prefix='queue-', dir=cache_directory) as temporary:
            root = Path(temporary)

            # 1. Source separation. The audio.cpp process exits before ASR starts,
            # so its model/VRAM is released before the next family is loaded.
            if 'preprocess' in selected:
                model, weights, _provenance, stage = selected['preprocess']
                sep_inputs = {}
                before_info = {}
                for job in _active(jobs):
                    checkpoint(cancel)
                    try:
                        item_callback(job.index, '前処理準備', '')
                        destination = root / 'sep-source' / f'{job.key}.wav'
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        filters = f'aresample={model.sample_rate}:async=1:first_pts=0,atrim=start={settings.clip_start}'
                        if settings.clip_duration:
                            filters += f':duration={settings.clip_duration}'
                        filters += ',asetpts=PTS-STARTPTS'
                        run_process([
                            ffmpeg_path(settings.ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin',
                            '-i', str(job.source), '-map', '0:a:0', '-vn', '-af', filters,
                            '-ac', '2', '-ar', str(model.sample_rate), '-c:a', 'pcm_f32le',
                            '-y', str(destination),
                        ], cancel, progress, job.report / 'preprocess-decode.log')
                        info = pre.wave_info(destination)
                        if not info.frames:
                            raise ValueError('No audio in selected interval')
                        sep_inputs[job.key] = destination
                        before_info[job.key] = info
                    except BaseException as exc:
                        if cancel.is_set():
                            raise
                        _fail(job, exc, item_callback)

                for chunk_no, chunk in enumerate(
                    _chunks_by_size(_active(jobs), lambda j: sep_inputs[j.key], max_bytes), 1
                ):
                    chunk_root = root / 'sep-batch' / f'chunk-{chunk_no}'
                    _link_inputs(chunk, lambda j: sep_inputs[j.key], chunk_root / 'inputs')
                    out = chunk_root / 'out'
                    out.mkdir(parents=True, exist_ok=True)
                    argv = [
                        str(executable(model.runtime, stage.device, stage.executable)),
                        '--task', 'sep', '--family', model.family, '--model', str(weights),
                        '--backend', 'best' if stage.device == 'auto' else stage.device,
                        '--mode', 'offline', '--batch-audio-dir', str(chunk_root / 'inputs'),
                        '--threads', str(stage.threads), '--out-dir', str(out),
                    ]
                    request, session = split_engine_parameters(model, stage.parameters or {})
                    for key, value in request.items():
                        argv += ['--request-option', f'{key}={scalar(value)}']
                    argv += audio_session_args(model, session)
                    log = chunk_root / 'engine.log'
                    try:
                        for job in chunk:
                            item_callback(job.index, 'SEP', '')
                        run_process(argv, cancel, progress, log)
                        for job in chunk:
                            vocals = out / job.key / 'vocals.wav'
                            if not vocals.is_file():
                                raise ValueError(f'Separator did not produce vocals.wav for {job.source.name}')
                            after = pre.wave_info(vocals)
                            pre.validate_duration(before_info[job.key], after)
                            job.inference_source = vocals
                            job.preprocess_info = {
                                'input_audio': asdict(before_info[job.key]),
                                'processed_audio': asdict(after),
                                'length_difference_samples': after.frames - before_info[job.key].frames,
                                'time_mapping': 'contiguous; no silence removal or time stretching',
                            }
                            job.manifest['preprocessing'] = job.preprocess_info
                        _copy_log(log, chunk, f'preprocess-batch-{chunk_no}.log')
                    except BaseException as exc:
                        if cancel.is_set():
                            raise
                        for job in chunk:
                            _fail(job, exc, item_callback)
            else:
                for job in jobs:
                    job.inference_source = job.source

            # 2. ASR. All queue items use one selected model, so batch it.
            asr_model, asr_weights, _prov, asr_stage = selected['asr']
            asr_inputs = {}
            for job in _active(jobs):
                checkpoint(cancel)
                try:
                    item_callback(job.index, 'ASR準備', '')
                    path = root / 'asr-input' / f'{job.key}.wav'
                    duration = _decode_for_model(job, path, asr_model.sample_rate, settings, cancel, progress)
                    if duration <= 0:
                        raise ValueError('No audio in selected interval')
                    job.duration = duration
                    asr_inputs[job.key] = path
                except BaseException as exc:
                    if cancel.is_set():
                        raise
                    _fail(job, exc, item_callback)

            if asr_model.runtime == 'whisper_cpp':
                asr_results = _whisper_batch(asr_model, asr_weights, asr_stage, _active(jobs),
                                             asr_inputs, root / 'asr-batch', cancel, progress,
                                             item_callback)
            else:
                asr_results = _audio_batch(asr_model, asr_weights, asr_stage, _active(jobs),
                                           asr_inputs, root / 'asr-batch', cancel, progress,
                                           item_callback, 'asr', max_bytes)
            for job in _active(jobs):
                try:
                    result = asr_results[job.key]
                    _jsonable_result(job, 'asr', result)
                    units = core.clean_bounds(result.units, job.duration, job.warnings)
                    if not units:
                        raise ValueError('No timed speech was returned')
                    if any(unit.method == 'emission_frame' for unit in units):
                        job.warnings.append(
                            'ASR times are emission-frame estimates, not exact spoken-word boundaries; alignment recommended')
                    job.units = [replace(unit, speaker=None) for unit in core.group_units(units)]
                except BaseException as exc:
                    _fail(job, exc, item_callback)
            shutil.rmtree(root / 'asr-input', ignore_errors=True)

            # 3. Forced alignment, if selected. One request per ASR segment,
            # but requests share an audio.cpp model session within each RAM-bound batch.
            if 'align' in selected:
                model, weights, _prov, stage = selected['align']
                requests = []
                by_job = {job.key: [] for job in _active(jobs)}
                for job in list(_active(jobs)):
                    too_long = next((u for u in job.units if u.end - u.start > 55), None)
                    if too_long is not None:
                        _fail(job, ValueError(
                            'Alignment needs <=55-second matched transcript/audio segments; ASR-only remains available.'), item_callback)
                        continue
                    for n, segment in enumerate(job.units):
                        checkpoint(cancel)
                        begin = max(0.0, segment.start - 0.15)
                        length = min(job.duration, segment.end + 0.25) - begin
                        key = f'{job.key}s{n:05d}'
                        path = root / 'align-source' / f'{key}.wav'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            if getattr(settings, 'preprocess', None) is not None:
                                local = replace(settings, clip_start=0.0, clip_duration=0.0)
                                core.decode(job.inference_source, path, model.sample_rate, local,
                                            cancel, progress, begin, length)
                            else:
                                core.decode(job.source, path, model.sample_rate, settings,
                                            cancel, progress, settings.clip_start + begin, length)
                            requests.append(AlignRequest(key, job, path, begin, length, segment.text))
                        except BaseException as exc:
                            if cancel.is_set():
                                raise
                            _fail(job, exc, item_callback)
                            break
                requests = [req for req in requests if not req.job.failed]
                aligned_results = _align_batch(model, weights, stage, requests, root / 'align-batch',
                                               cancel, progress, item_callback, max_bytes)
                for req in requests:
                    if req.job.failed:
                        continue
                    try:
                        result = aligned_results[req.key]
                        # Preserve raw per-segment output without overwriting siblings.
                        directory = req.job.report / 'align' / req.key
                        directory.mkdir(parents=True, exist_ok=True)
                        core.json_write(directory / 'raw.json', result.raw)
                        local = core.clean_bounds(result.units, req.length, req.job.warnings)
                        if not local:
                            raise ValueError('Aligner returned no intervals')
                        by_job[req.job.key].extend(
                            replace(unit, start=unit.start + req.begin, end=unit.end + req.begin)
                            for unit in local
                        )
                    except BaseException as exc:
                        _fail(req.job, exc, item_callback)
                for job in _active(jobs):
                    aligned = core.clean_bounds(by_job.get(job.key, []), job.duration, job.warnings)
                    if not aligned:
                        _fail(job, ValueError('No valid aligned intervals'), item_callback)
                    else:
                        job.units = aligned
                shutil.rmtree(root / 'align-source', ignore_errors=True)

            # 4. Diarization. It is intentionally after alignment so speaker
            # assignment can use the final/finer text intervals.
            if 'diar' in selected:
                model, weights, _prov, stage = selected['diar']
                diar_inputs = {}
                for job in _active(jobs):
                    checkpoint(cancel)
                    try:
                        path = root / 'diar-input' / f'{job.key}.wav'
                        _decode_for_model(job, path, model.sample_rate, settings, cancel, progress)
                        diar_inputs[job.key] = path
                    except BaseException as exc:
                        if cancel.is_set():
                            raise
                        _fail(job, exc, item_callback)
                diar_results = _audio_batch(model, weights, stage, _active(jobs), diar_inputs,
                                            root / 'diar-batch', cancel, progress, item_callback,
                                            'diar', max_bytes)
                for job in _active(jobs):
                    try:
                        result = diar_results[job.key]
                        _jsonable_result(job, 'diar', result)
                        turns = core.clean_bounds(result.units, job.duration, job.warnings)
                        job.units = core.assign_speakers(job.units, turns, job.warnings)
                    except BaseException as exc:
                        _fail(job, exc, item_callback)
                shutil.rmtree(root / 'diar-input', ignore_errors=True)

            # 5. Final RPP. At this point no inference process/model is resident.
            for job in _active(jobs):
                checkpoint(cancel)
                try:
                    item_callback(job.index, 'RPP', '')
                    job.units = core.group_units(job.units)
                    if not job.units:
                        raise ValueError('No valid intervals remain after normalization')
                    if digest(job.source) != job.source_sha256:
                        raise ValueError('Original input changed while processing')
                    diar_enabled = settings.diar is not None
                    if getattr(settings, 'preprocess', None) is not None:
                        mode = getattr(settings, 'reference_audio', 'original')
                        if mode == 'processed':
                            reference = job.output.with_name(job.output.stem + '_vocals.wav')
                            with job.inference_source.open('rb') as incoming, reference.open('xb') as outgoing:
                                shutil.copyfileobj(incoming, outgoing)
                            origin = settings.clip_start
                            sample_rate = pre.wave_info(reference).sample_rate
                            job.manifest['persistent_audio'] = str(reference)
                        else:
                            reference, origin, sample_rate = job.source, 0.0, 48000
                        pre.write_reference(job.output, reference, job.units, settings.clip_start,
                                            origin, diar_enabled, sample_rate)
                        job.manifest.update(reference_file=str(reference),
                                            reference_origin_seconds=origin,
                                            timeline_origin_seconds=settings.clip_start,
                                            preprocessing_for_inference=True)
                    else:
                        core.export_rpp(job.source, job.output, job.units, settings.clip_start, diar_enabled)
                    core.json_write(job.report / 'transcript.json', {
                        'clip_start': settings.clip_start,
                        'duration': job.duration,
                        'units': [asdict(unit) for unit in job.units],
                        'warnings': job.warnings,
                    })
                    job.manifest.update(status='completed', output=str(job.output),
                                        source_unchanged=True, warnings=job.warnings)
                    _write_manifest(job)
                    completed[job.index] = job.output
                    item_callback(job.index, '完了', str(job.output))
                except BaseException as exc:
                    if cancel.is_set():
                        raise
                    _fail(job, exc, item_callback)

    except Cancelled:
        for job in _active(jobs):
            job.manifest['status'] = 'cancelled'
            _write_manifest(job)
            item_callback(job.index, '中断', 'ユーザーが停止しました')
        raise
    return completed
