"""Optional source separation and explicit ownership of referenced audio assets.

This layer never substitutes a different checkpoint for the configured model.
The existing recognition pipeline stays Qt-free and reusable.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import copy
import json
import shutil
import tempfile
import time
from . import pipeline as core
from .catalog import Model, checkpoint, cache_root, digest, resolve_model
from .adapters import (executable, ffmpeg_path, run_process, Unit, split_engine_parameters,
                       audio_session_args, validate_model_parameter_constraints, scalar)
from .rpp_export import write_reference
from .media import wave_info, validate_duration, full_reference_duration


@dataclass
class Settings(core.Settings):
    preprocess: core.Stage | None = None
    reference_audio: str = 'original'

    def validate(self, catalog):
        super().validate(catalog)
        if self.reference_audio not in {'original', 'processed'}:
            raise ValueError('RPP audio must be original or processed')
        if self.reference_audio == 'processed' and self.preprocess is None:
            raise ValueError('Processed RPP audio requires enabled preprocessing')
        if self.preprocess is not None:
            if self.preprocess.device not in {'cpu', 'auto', 'vulkan', 'metal', 'cuda'}:
                raise ValueError('Unsupported preprocessing device')
            model = catalog.get(self.preprocess.model_id)
            if model is None or model.task != 'sep' or model.runtime != 'audio_cpp':
                raise ValueError('Choose an audio.cpp separation model for preprocessing')



from .performance import timed, profiled, report_directory

@timed('separation')

def separate(source: Path, work: Path, settings: Settings, model: Model, weights: Path, cancel, progress, log_dir: Path | None = None):
    if weights.suffix.lower() in {'.ckpt', '.pt', '.pth'}:
        raise ValueError('Model installation did not finish; converted GGUF is unavailable')
    work.mkdir(parents=True, exist_ok=True)
    logs = log_dir or work
    logs.mkdir(parents=True, exist_ok=True)
    mixture = work / 'mixture.wav'
    rate = model.sample_rate
    filters = f'aresample={rate}:async=1:first_pts=0,atrim=start={settings.clip_start}'
    if settings.clip_duration:
        filters += f':duration={settings.clip_duration}'
    filters += ',asetpts=PTS-STARTPTS'
    run_process([ffmpeg_path(settings.ffmpeg, progress, cancel), '-hide_banner', '-loglevel', 'error', '-nostdin',
                 '-i', str(source), '-map', '0:a:0', '-vn', '-af', filters, '-ac', '2',
                 '-ar', str(rate), '-c:a', 'pcm_f32le', '-y', str(mixture)],
                cancel, progress, logs / 'preprocess-decode.log')
    before = wave_info(mixture)
    if not before.frames:
        raise ValueError('No audio in selected interval')
    stage = settings.preprocess
    stems = work / 'stems'
    stems.mkdir()
    args = [executable(model.runtime, stage.device, stage.executable), '--task', 'sep',
            '--family', model.family, '--model', str(weights), '--backend',
            'best' if stage.device == 'auto' else stage.device, '--threads', str(stage.threads),
            '--audio', str(mixture), '--out-dir', str(stems)]
    parameters = dict(stage.parameters or {})
    # Migrate legacy separation controls at the boundary; DCC uses session.*.
    for key in ('num_overlap', 'weight_type'):
        if key in parameters:
            parameters.setdefault('session.' + key, parameters.pop(key))
    request, session = split_engine_parameters(model, parameters)
    validate_model_parameter_constraints(model, request, session)
    for key, value in request.items():
        if not key.replace('_', '').replace('.', '').isalnum():
            raise ValueError('Invalid preprocessing request parameter')
        args += ['--request-option', f'{key}={scalar(value)}']
    args += audio_session_args(model, session)
    run_process(args, cancel, progress, logs / 'preprocess.log')
    # Never silently use instrumental.wav or another stem if the selected one is missing.
    vocals = stems / 'vocals.wav'
    if not vocals.is_file():
        raise ValueError('The separator did not produce vocals.wav')
    after = wave_info(vocals)
    validate_duration(before, after)
    return vocals, {'input_audio': asdict(before), 'processed_audio': asdict(after),
                    'length_difference_samples': after.frames - before.frames,
                    'time_mapping': 'contiguous; no silence removal or time stretching',
                    'waveform_delay_calibrated': False}


@profiled
def run_job(source, settings: Settings, catalog, cancel, progress):
    settings = copy.deepcopy(settings)
    settings.validate(catalog)
    if settings.preprocess is None:
        return core.run_job(source, settings, catalog, cancel, progress)
    source = Path(source).expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in core.MEDIA_EXTENSIONS:
        raise ValueError('Unsupported or missing input')
    model = catalog[settings.preprocess.model_id]
    weights, provenance = resolve_model(model, cancel, progress)
    siblings = ('_vocals.wav',) if settings.reference_audio == 'processed' else ()
    output, report = core.reserve_output(source, settings, sibling_suffixes=siblings)
    manifest = {'source': str(source), 'source_sha256': digest(source), 'settings': asdict(settings),
                'status': 'running', 'model': provenance, 'reference_audio': settings.reference_audio,
                'clip_start': settings.clip_start, 'source_unchanged': None}
    started = time.monotonic()
    core.json_write(report / 'manifest.json', manifest)
    temp_path = None
    try:
        cache_directory = cache_root()
        cache_directory.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='preprocess-', dir=cache_directory) as temporary:
            work = Path(temporary)
            temp_path = work
            progress('Preprocessing — separating vocals from background audio')
            vocals, metadata = separate(source, work / 'separation', settings, model, weights, cancel, progress, log_dir=report)
            manifest['preprocessing'] = metadata
            manifest['processed_sha256'] = digest(vocals)
            downstream = replace(settings, preprocess=None, reference_audio='original',
                                 clip_start=0.0, clip_duration=0.0, same_directory=False,
                                 output_directory=str(work / 'recognition'))
            # Every enabled inference stage sees the same processed time axis.
            core.run_job(vocals, downstream, catalog, cancel, progress,
                         _analysis_report=report / 'analysis')
            transcript = json.loads((report / 'analysis' / 'transcript.json').read_text(encoding='utf-8'))
            units = [Unit(**record) for record in transcript['units']]
            analysis_manifest = json.loads((report/'analysis'/'manifest.json').read_text(encoding='utf-8'))
            manifest['timestamp_source'] = analysis_manifest.get('timestamp_source')
            checkpoint(cancel)
            if digest(source) != manifest['source_sha256']:
                raise ValueError('Original input changed while processing')
            if settings.reference_audio == 'processed':
                reference = output.with_name(output.stem + '_vocals.wav')
                with vocals.open('rb') as incoming, reference.open('xb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                origin = settings.clip_start
                info = wave_info(reference)
                sample_rate = info.sample_rate
                manifest['persistent_audio'] = str(reference)
            else:
                reference, origin, sample_rate = source, 0.0, 48000
            manifest.update(reference_file=str(reference), reference_origin_seconds=origin,
                            timeline_origin_seconds=settings.clip_start,
                            preprocessing_for_inference=True)
            # The processed reference is a complete WAV; the original reference
            # may extend outside an explicitly selected inference clip.
            reference_length = full_reference_duration(
                reference, settings, wave_info(vocals).duration, work, cancel, progress)
            write_reference(output, reference, units, settings.clip_start, origin,
                            settings.diar is not None, sample_rate,
                            reference_duration=reference_length)
            manifest['original_track'] = {'name': 'ORIGINAL', 'muted': True,
                                          'duration_seconds': float(reference_length)}
            transcript.update(clip_start=settings.clip_start, reference_audio=settings.reference_audio)
            core.json_write(report / 'transcript.json', transcript)
            manifest.update(status='completed', output=str(output), source_unchanged=True)
        return output
    except BaseException as error:
        manifest.update(status='cancelled' if cancel.is_set() else 'failed', error=str(error))
        raise
    finally:
        manifest['elapsed_seconds'] = time.monotonic() - started
        manifest['temporary_preprocessed_audio_removed'] = (temp_path is None or not temp_path.exists())
        core.json_write(report / 'manifest.json', manifest)

