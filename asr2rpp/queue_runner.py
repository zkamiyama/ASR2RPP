"""Stage-major queue scheduler.

The GUI queue is processed by model stage rather than file:
SEP -> ASR -> forced alignment -> diarization -> RPP.

Native model processes are never kept alive across different model families.
whisper.cpp receives multiple files per process (one model context), while
audio.cpp uses one offline batch session per RAM-bounded chunk, except Nemotron ASR,
which uses its bounded native streaming session for long-form safety.
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
from .alignment import (infer_requests, chunks_by_size as _chunks_by_size,
                        alignment_bounds, aligned_units)
from .inference_policy import policy_for
from .timing import prepare_run, stage_options
from .media import slice_pcm
from .catalog import Cancelled, checkpoint, cache_root, digest, resolve_model, definition_provenance
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


from .performance import profiled, report_directory

from .diagnostics import persist_tree

def _jsonable_result(job: QueueJob, task: str, result: Result) -> None:
    directory = job.report / task
    directory.mkdir(parents=True, exist_ok=True)
    if task == 'asr' and isinstance(result.raw, dict) and 'inference_policy' in result.raw and (directory/'raw.json').exists() and (directory/'normalized.json').exists():
        return
    core.json_write(directory / 'raw.json', result.raw)
    core.json_write(directory / 'normalized.json', [asdict(unit) for unit in result.units])


def _write_manifest(job: QueueJob) -> None:
    core.json_write(job.report / 'manifest.json', job.manifest)


def _fail(job: QueueJob, error, item_callback) -> None:
    if isinstance(error, Cancelled):
        raise error
    if job.failed or job.manifest.get('status') in {'completed', 'cancelled'}:
        return
    job.failed = True
    job.manifest.update(status='failed', error=str(error))
    _write_manifest(job)
    item_callback(job.index, '失敗', str(error))


def _active(jobs):
    return [job for job in jobs if not job.failed and job.manifest.get('status') not in {'completed', 'cancelled', 'failed'}]


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


def _asr_batch(model, weights, stage, jobs, audio_paths, root, cancel, progress,
               item_callback, alignment_requested=False, options=None, max_bytes=512*1024**2):
    from .providers import provider_for, Request
    from .timing import plan_for, TimingSettings
    options = dict(options or stage.options(), alignment_requested=alignment_requested)
    requests = [Request(job.key, audio_paths[job.key]) for job in jobs]
    for job in jobs:
        item_callback(job.index, 'ASR' if model.task == 'asr' else 'Diarization', '')
    provider = provider_for(model)
    try:
        host_regions = (model.task == 'asr' and not provider.native_regions and
            plan_for(model, TimingSettings(**options.get('timing',{})), alignment_requested).segmented)
        if host_regions:
            from .region_batch import infer_many
            batch = infer_many(model,weights,requests,root,options,cancel,progress,max_bytes)
        else:
            batch = provider.infer_many(model,weights,requests,root,options,cancel,progress,max_bytes)
        for job in jobs:
            if job.key in batch.errors:
                _fail(job,batch.errors[job.key],item_callback)
        return batch.results
    finally:
        for job in jobs:
            persist_tree(Path(root)/job.key,job.report/model.task)


# Compatibility for callers of the former private native batch helpers.
# Both now delegate to the same provider boundary used by the main scheduler.
def _whisper_batch(model, weights, stage, jobs, audio_paths, root, cancel, progress,
                   item_callback, alignment_requested=False):
    return _asr_batch(model,weights,stage,jobs,audio_paths,root,cancel,progress,item_callback,
                      alignment_requested=alignment_requested)


def _audio_batch(model, weights, stage, jobs, audio_paths, root, cancel, progress,
                 item_callback, task, max_bytes):
    if task != model.task:
        raise ValueError('Batch task does not match model')
    return _asr_batch(model,weights,stage,jobs,audio_paths,root,cancel,progress,item_callback,max_bytes=max_bytes)


@dataclass
class AlignRequest:
    key: str
    job: QueueJob
    audio: Path
    begin: float
    length: float
    text: str
    owner_start: float | None = None
    owner_end: float | None = None
    speaker: str | None = None


def _align_batch(model, weights: Path, stage, requests: list[AlignRequest], root: Path,
                 cancel, progress, item_callback, max_bytes: int):
    results = {}
    for chunk_no, chunk in enumerate(_chunks_by_size(requests, lambda r: r.audio, max_bytes), 1):
        jobs = list({req.job.index: req.job for req in chunk}.values())
        try:
            for job in jobs:
                item_callback(job.index, 'Forced Align', '')
            results.update(infer_requests(
                model, weights, stage, chunk, root / f'batch-{chunk_no}',
                cancel, progress, max_bytes))
        except Exception as exc:
            if cancel.is_set():
                raise
            for job in jobs:
                _fail(job, exc, item_callback)
        finally:
            for log in (root / f'batch-{chunk_no}').glob('chunk-*/engine.log'):
                _copy_log(log, jobs, f'align-batch-{chunk_no}-{log.parent.name}.log')
    return results


def _decode_for_model(job: QueueJob, destination: Path, rate: int, settings,
                      cancel, progress):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if getattr(settings, 'preprocess', None) is not None:
        local = replace(settings, clip_start=0.0, clip_duration=0.0)
        return core.decode(job.inference_source, destination, rate, local, cancel, progress)
    return core.decode(job.source, destination, rate, settings, cancel, progress)


def _cancel_jobs(jobs, item_callback):
    for job in _active(jobs):
        job.manifest['status'] = 'cancelled'
        try:
            _write_manifest(job)
        finally:
            item_callback(job.index, '中断', 'ユーザーが停止しました')


def _prepare_jobs(indexed_paths, settings, catalog, item_callback, cancel):
    jobs = []
    try:
        for index, value in indexed_paths:
            checkpoint(cancel)
            try:
                source = Path(value).expanduser().resolve()
                if not source.is_file() or source.suffix.lower() not in core.MEDIA_EXTENSIONS:
                    raise ValueError('Unsupported or missing input')
                siblings = ('_vocals.wav',) if (getattr(settings, 'preprocess', None) is not None and
                                             getattr(settings, 'reference_audio', 'original') == 'processed') else ()
                output, report = core.reserve_output(source, settings, sibling_suffixes=siblings)
                manifest = {'source': str(source), 'settings': asdict(settings), 'status': 'queued',
                    'queue_strategy': 'stage_major', 'source_unchanged': None,
                    'inference_policy': asdict(policy_for(catalog[settings.asr.model_id])),
                    'timestamp_source': 'forced_alignment' if settings.align else
                        'vad' if policy_for(catalog[settings.asr.model_id]).uses_vad_timing else 'native_asr'}
                job = QueueJob(index, f'q{index:06d}', source, output, report, '', manifest)
                jobs.append(job)
                job.source_sha256 = digest(source)
                job.manifest['source_sha256'] = job.source_sha256
                _write_manifest(job)
            except Cancelled:
                raise
            except Exception as exc:
                if jobs and jobs[-1].index == index:
                    _fail(jobs[-1], exc, item_callback)
                else:
                    item_callback(index, '失敗', str(exc))
    except Cancelled:
        _cancel_jobs(jobs, item_callback)
        raise
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


@profiled
def _run_queue_window(indexed_paths, settings, catalog, cancel: threading.Event, progress,
              item_callback, batch_audio_ram_mb: int = DEFAULT_BATCH_AUDIO_RAM_MB):
    """Run a GUI queue stage-by-stage.

    Returns completed output paths keyed by original queue index.
    """
    settings.validate(catalog)
    settings, catalog, timing_plan = prepare_run(settings, catalog)
    settings.validate(catalog)
    checkpoint(cancel)
    max_bytes = max(128, min(int(batch_audio_ram_mb), 8192)) * 1024 * 1024
    jobs = _prepare_jobs(indexed_paths, settings, catalog, item_callback, cancel)
    if not _active(jobs):
        return {}
    try:
        selected = _resolve_models(settings, catalog, cancel, progress)
    except Cancelled:
        _cancel_jobs(jobs, item_callback)
        raise
    except Exception as exc:
        for job in _active(jobs):
            _fail(job, exc, item_callback)
        return {}

    completed = {}
    try:
        for job in _active(jobs):
            job.manifest['model_definitions'] = {name: definition_provenance(m) for name, (m, _p, _prov, _s) in selected.items()}
            job.manifest['models'] = {name: provenance for name, (_m, _p, provenance, _s) in selected.items()}
            job.manifest['status'] = 'running'
            job.manifest['timing_plan'] = asdict(timing_plan)
            job.manifest['timestamp_source'] = timing_plan.timestamp_source
            _write_manifest(job)

        cache_directory = cache_root()
        cache_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='queue-', dir=cache_directory) as temporary:
            root = Path(temporary)

            decoded = {}
            def decode_once(job, destination, rate):
                key = (job.key, rate)
                if key not in decoded:
                    duration = _decode_for_model(job, destination, rate, settings, cancel, progress)
                    decoded[key] = (destination, duration)
                return decoded[key]

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
                    path, duration = decode_once(job, path, asr_model.sample_rate)
                    if duration <= 0:
                        raise ValueError('No audio in selected interval')
                    from .native_profile import profile
                    limit = profile(asr_model).max_audio_seconds
                    if limit and duration > limit and not timing_plan.segmented:
                        raise ValueError(f'Model input exceeds {limit:g}s; choose VAD segmentation in Timing settings')
                    job.duration = duration
                    asr_inputs[job.key] = path
                except BaseException as exc:
                    if cancel.is_set():
                        raise
                    _fail(job, exc, item_callback)

            asr_results = _asr_batch(asr_model,asr_weights,asr_stage,_active(jobs),
                asr_inputs,root/'asr-batch',cancel,progress,item_callback,
                alignment_requested=settings.align is not None, options=stage_options(settings),max_bytes=max_bytes)
            for job in _active(jobs):
                try:
                    result = asr_results[job.key]
                    _jsonable_result(job, 'asr', result)
                    units = core.clean_bounds(result.units, job.duration, job.warnings)
                    if not units:
                        raise ValueError('No timed speech was returned')
                    if any(unit.method == 'vad_segment' for unit in units):
                        job.warnings.append('VAD region timestamps used: approximate speech intervals, not word boundaries.')
                    if any(unit.method == 'emission_frame' for unit in units):
                        job.warnings.append(
                            'ASR times are emission-frame estimates, not exact spoken-word boundaries; alignment recommended')
                    job.units = ([replace(unit,speaker=None) for unit in units]
                                 if settings.diar is not None or settings.timing.speaker_source == 'none' else units)
                    if settings.timing.speaker_source == 'native' and not any(u.speaker for u in job.units):
                        raise ValueError('The ASR did not return native speaker labels')
                except BaseException as exc:
                    _fail(job, exc, item_callback)
            # PCM remains until the last enabled stage; same-rate stages share it.

            # 3. Forced alignment, if selected. One request per ASR segment,
            # but requests share an audio.cpp model session within each RAM-bound batch.
            if 'align' in selected:
                model, weights, _prov, stage = selected['align']
                requests = []
                by_job = {job.key: [] for job in _active(jobs)}
                for job in list(_active(jobs)):
                    job.units = core.group_units(job.units)
                    too_long = next((u for u in job.units if core.has_alignable_text(u.text) and u.end - u.start > 55), None)
                    if too_long is not None:
                        _fail(job, ValueError(
                            'Alignment needs <=55-second matched transcript/audio segments; ASR-only remains available.'), item_callback)
                        continue
                    alignment_pcm = root / 'align-pcm' / f'{job.key}.wav'
                    try:
                        # Decode each file once, rather than starting FFmpeg and
                        # reading a long video from the beginning for every phrase.
                        alignment_pcm, _ = decode_once(job, alignment_pcm, model.sample_rate)
                    except Exception as exc:
                        if cancel.is_set():
                            raise
                        _fail(job, exc, item_callback)
                        continue
                    for n, segment in enumerate(job.units):
                        checkpoint(cancel)
                        if not core.has_alignable_text(segment.text):
                            job.warnings.append(
                                f'{segment.start:.3f}: forced alignment skipped punctuation-only text; '
                                'ASR interval retained')
                            by_job[job.key].append(segment)
                            progress(
                                f'Forced alignment — punctuation-only text skipped for '
                                f'{job.source.name}; ASR timing retained')
                            continue
                        begin, length = alignment_bounds(segment, job.duration)
                        key = f'{job.key}s{n:05d}'
                        path = root / 'align-source' / f'{key}.wav'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            length = slice_pcm(alignment_pcm, path, begin, length, cancel)
                            requests.append(AlignRequest(key, job, path, begin, length, segment.text,
                                                         segment.owner_start, segment.owner_end, segment.speaker))
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
                        by_job[req.job.key].extend(aligned_units(result, req, req.job.warnings))
                    except BaseException as exc:
                        _fail(req.job, exc, item_callback)
                for job in _active(jobs):
                    aligned = core.clean_bounds(by_job.get(job.key, []), job.duration, job.warnings)
                    if not aligned:
                        _fail(job, ValueError('No valid aligned intervals'), item_callback)
                    else:
                        job.units = aligned
                shutil.rmtree(root / 'align-source', ignore_errors=True)
                # Shared PCM cleanup belongs to the queue workspace.

            # 4. Diarization. It is intentionally after alignment so speaker
            # assignment can use the final/finer text intervals.
            if 'diar' in selected:
                model, weights, _prov, stage = selected['diar']
                diar_inputs = {}
                for job in _active(jobs):
                    checkpoint(cancel)
                    try:
                        path = root / 'diar-input' / f'{job.key}.wav'
                        path, _ = decode_once(job, path, model.sample_rate)
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
                # Shared PCM cleanup belongs to the queue workspace.

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
                    diar_enabled = settings.diar is not None or any(u.speaker for u in job.units)
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
                        reference_length = core.full_reference_duration(
                            reference, settings, job.duration, root, cancel, progress)
                        pre.write_reference(job.output, reference, job.units, settings.clip_start,
                                            origin, diar_enabled, sample_rate,
                                            reference_duration=reference_length)
                        job.manifest.update(reference_file=str(reference),
                                            reference_origin_seconds=origin,
                                            timeline_origin_seconds=settings.clip_start,
                                            preprocessing_for_inference=True)
                    else:
                        reference_length = core.full_reference_duration(
                            job.source, settings, job.duration, root, cancel, progress)
                        core.export_rpp(job.source, job.output, job.units, settings.clip_start,
                                        diar_enabled, reference_length)
                    job.manifest['original_track'] = {'name': 'ORIGINAL', 'muted': True,
                                                       'duration_seconds': float(reference_length)}
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
        _cancel_jobs(jobs, item_callback)
        raise
    except Exception as exc:
        for job in _active(jobs):
            _fail(job, exc, item_callback)
    return completed


@profiled
def run_queue(indexed_paths, settings, catalog, cancel, progress, item_callback,
              batch_audio_ram_mb=DEFAULT_BATCH_AUDIO_RAM_MB):
    """Shared GUI/CLI scheduler with bounded stage-major prefetch windows."""
    settings.validate(catalog)
    paths = list(indexed_paths)
    if len({i for i,_ in paths}) != len(paths):
        raise ValueError('Queue request indices must be unique')
    completed = {}
    maximum = settings.queue_window_items
    for offset in range(0,len(paths),maximum):
        checkpoint(cancel)
        completed.update(_run_queue_window(paths[offset:offset+maximum],settings,catalog,
            cancel,progress,item_callback,batch_audio_ram_mb))
    return completed
