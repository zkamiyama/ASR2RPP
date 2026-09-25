"""Qt-free pipeline; native audio is never split or modified for RPP export."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import copy
import hashlib
import json
import math
import shutil
import threading
import time
import tempfile
import wave
from .catalog import Model, checkpoint, cache_root, resolve_model, digest
from .adapters import Unit, infer, run_process, ffmpeg_path, executable
from .transcript import clean_bounds, group_units, has_alignable_text, assign_speakers, safe_label
from .rpp_export import write_reference
from .media import full_reference_duration
from .alignment import align_segments
from .inference_policy import policy_for
from .adapters import split_engine_parameters, validate_model_parameter_constraints

MEDIA_EXTENSIONS = {'.wav', '.wave', '.mp3', '.flac', '.ogg', '.opus', '.aif', '.aiff',
                    '.m4a', '.aac', '.mp4', '.mkv', '.mov', '.webm', '.avi', '.wma'}
MAX_OUTPUT_STEM_CHARS = 120


def compact_output_stem(value: str, max_chars: int = MAX_OUTPUT_STEM_CHARS) -> str:
    """Keep output components short and stable without losing collision resistance."""
    value = value.rstrip(' .') or 'output'
    if len(value) <= max_chars:
        return value
    tag = hashlib.sha256(value.encode('utf-8')).hexdigest()[:8]
    keep = max(1, max_chars - len(tag) - 1)
    return value[:keep].rstrip(' .') + '~' + tag

@dataclass
class Stage:
    model_id: str
    device: str = 'cpu'
    executable: str = ''
    language: str = 'ja'
    threads: int = 4
    parameters: dict | None = None

    def options(self):
        data = asdict(self)
        data['parameters'] = self.parameters or {}
        return data

@dataclass
class Settings:
    asr: Stage
    diar: Stage | None = None
    align: Stage | None = None
    same_directory: bool = True
    output_directory: str = ''
    ffmpeg: str = ''
    clip_start: float = 0.0
    clip_duration: float = 0.0

    def validate(self, catalog: dict[str, Model]):
        if not self.same_directory and not self.output_directory.strip():
            raise ValueError('Same directory is OFF: specify Output directory.')
        if not all(math.isfinite(x) and x >= 0 for x in (self.clip_start, self.clip_duration)):
            raise ValueError('Clip start/duration must be finite and nonnegative')
        if self.asr is None:
            raise ValueError('ASR is required')
        for task, stage in [('asr', self.asr), ('diar', self.diar), ('align', self.align)]:
            if stage is not None:
                if stage.model_id not in catalog:
                    raise ValueError(f'{task}: select an available model')
                if catalog[stage.model_id].task != task:
                    raise ValueError(f'Wrong model task for {task}')
                if stage.device not in {'cpu', 'auto', 'cuda', 'vulkan', 'metal'}:
                    raise ValueError('Unsupported device')
                model = catalog[stage.model_id]
                request, session = split_engine_parameters(model, stage.parameters or {})
                validate_model_parameter_constraints(model, request, session)
                if policy_for(model).segmentation == 'vad':
                    from .vad import VadOptions
                    VadOptions.from_parameters(request, policy_for(model).max_segment_seconds)

        if policy_for(catalog[self.asr.model_id]).requires_alignment and self.align is None:
            raise ValueError('TOML inference policy requires forced alignment: enable --align or choose timestamp_source=vad in the model definition.')


def reserve_output(source: Path, settings: Settings, sibling_suffixes=()) -> tuple[Path, Path]:
    parent = source.parent if settings.same_directory else Path(settings.output_directory).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 100000):
        suffix = '' if index == 1 else f'_{index}'
        base = compact_output_stem(source.stem, MAX_OUTPUT_STEM_CHARS - len(suffix))
        stem = base + suffix
        output, report = parent / f'{stem}.rpp', parent / f'{stem}.asr2rpp'
        siblings = [parent / f'{stem}{extra}' for extra in sibling_suffixes]
        if output.exists() or any(path.exists() for path in siblings):
            continue
        try:
            report.mkdir()
            return output, report
        except FileExistsError:
            continue
    raise OSError('Too many output name collisions')


def json_write(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def decode(source: Path, destination: Path, rate: int, settings: Settings,
           cancel, progress, start: float | None = None, duration: float | None = None) -> float:
    offset = settings.clip_start if start is None else start
    length = settings.clip_duration if duration is None else duration
    filters = f'aresample={rate}:async=1:first_pts=0,atrim=start={offset}'
    if length:
        filters += f':duration={length}'
    filters += ',asetpts=PTS-STARTPTS'
    argv = [ffmpeg_path(settings.ffmpeg, progress, cancel), '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-i', str(source), '-map', '0:a:0', '-vn', '-af', filters,
            '-ac', '1', '-ar', str(rate), '-c:a', 'pcm_s16le', '-y', str(destination)]
    run_process(argv, cancel, progress, destination.with_suffix('.decode.log'))
    with wave.open(str(destination)) as audio:
        return audio.getnframes() / audio.getframerate()


def export_rpp(source: Path, output: Path, units: list[Unit], offset: float,
               diar: bool, reference_duration):
    """Use the same reference/timeline rules in all execution paths."""
    write_reference(output, source, units, offset, 0.0, diar,
                    reference_duration=reference_duration)


def run_job(source: Path, settings: Settings, catalog: dict[str, Model], cancel: threading.Event, progress) -> Path:
    settings = copy.deepcopy(settings)
    settings.validate(catalog)
    source = source.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in MEDIA_EXTENSIONS:
        raise ValueError(f'Unsupported or missing input: {source}')
    checkpoint(cancel)
    weights, provenance = {}, {}
    for task, stage in [('asr', settings.asr), ('diar', settings.diar), ('align', settings.align)]:
        if stage is not None:
            model = catalog[stage.model_id]
            executable(model.runtime, stage.device, stage.executable)
            weights[task], provenance[task] = resolve_model(model, cancel, progress)
    ffmpeg_path(settings.ffmpeg)
    output, report = reserve_output(source, settings)
    started = time.monotonic()
    warnings, pcm_files = [], []
    cache_directory = cache_root()
    cache_directory.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix='job-', dir=cache_directory)
    work = Path(temporary.name)
    manifest = {'source': str(source), 'source_sha256': digest(source), 'settings': asdict(settings),
                'models': provenance, 'status': 'running', 'reference_mode': 'non_destructive',
                'audio_stream': '0:a:0', 'time_origin': 'FFmpeg normalized demuxed-media origin',
                'source_unchanged': None,
                'inference_policy': asdict(policy_for(catalog[settings.asr.model_id]))}
    json_write(report / 'manifest.json', manifest)
    try:
        def infer_persist(model, weights_path, audio_path, engine_dir, report_dir,
                          options, transcript=''):
            try:
                return infer(
                    model, weights_path, audio_path, engine_dir, options,
                    cancel, progress, transcript)
            finally:
                # Native tools may not support Unicode output paths on Windows.
                # Keep their working files in the ASCII-ish cache workspace and
                # copy diagnostics/results into the user-facing report with Python.
                if engine_dir.exists():
                    report_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(engine_dir, report_dir, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns('*.wav'))

        pcm_by_rate = {}
        def pcm(rate):
            if rate not in pcm_by_rate:
                file = work / f'inference_{rate}.wav'
                pcm_files.append(file)
                progress(f'Decode {rate} Hz (inference cache only)')
                duration = decode(source, file, rate, settings, cancel, progress)
                if duration <= 0:
                    raise ValueError('No audio in selected interval')
                pcm_by_rate[rate] = (file, duration)
            return pcm_by_rate[rate]
        asr_model = catalog[settings.asr.model_id]
        audio, duration = pcm(asr_model.sample_rate)
        progress('ASR — transcribing')
        asr = infer_persist(
            asr_model, weights['asr'], audio, work / 'asr', report / 'asr',
            dict(settings.asr.options(), alignment_requested=settings.align is not None))
        units = clean_bounds(asr.units, duration, warnings)
        if not units:
            raise ValueError('No timed speech was returned; raw engine output is retained')
        if any(u.method == 'emission_frame' for u in units):
            warnings.append('ASR times are emission-frame estimates, not exact spoken-word boundaries; alignment recommended')
        units = group_units(units)
        units = [replace(u, speaker=None) for u in units]
        if settings.align is not None:
            align_model = catalog[settings.align.model_id]
            alignment_pcm, _ = pcm(align_model.sample_rate)
            engine_dir = work / 'alignment'
            try:
                units = align_segments(
                    align_model, weights['align'], settings.align, units,
                    alignment_pcm, duration, engine_dir, cancel, progress, warnings)
            finally:
                # Never copy temporary audio slices into the user's report.
                if engine_dir.exists():
                    shutil.copytree(engine_dir, report / 'align', dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns('*.wav'))
        if settings.diar is not None:
            model = catalog[settings.diar.model_id]
            audio, _ = pcm(model.sample_rate)
            progress('Diarization — assigning speakers')
            result = infer_persist(
                model, weights['diar'], audio, work / 'diar', report / 'diar',
                settings.diar.options())
            units = assign_speakers(units, clean_bounds(result.units, duration, warnings), warnings)
        if any(u.method == 'vad_segment' for u in units):
            warnings.append('VAD region timestamps used: approximate speech intervals, not word boundaries.')
        manifest['timestamp_source'] = ('forced_alignment' if settings.align else
                                        'vad' if policy_for(catalog[settings.asr.model_id]).uses_vad_timing else 'native_asr')
        units = group_units(units)
        if not units:
            raise ValueError('No valid intervals remain after normalization')
        checkpoint(cancel)
        progress('RPP — writing non-destructive references')
        json_write(report / 'transcript.json', {'clip_start': settings.clip_start, 'duration': duration,
                   'units': [asdict(u) for u in units], 'warnings': warnings})
        reference_length = full_reference_duration(
            source, settings, duration, work, cancel, progress)
        export_rpp(source, output, units, settings.clip_start, settings.diar is not None,
                   reference_length)
        manifest['original_track'] = {'name': 'ORIGINAL', 'muted': True,
                                      'duration_seconds': float(reference_length)}
        manifest.update(status='completed', elapsed_seconds=time.monotonic() - started,
                        source_unchanged=(digest(source) == manifest['source_sha256']), warnings=warnings,
                        output=str(output))
        return output
    except BaseException as error:
        manifest.update(status='cancelled' if cancel.is_set() else 'failed', error=str(error),
                        elapsed_seconds=time.monotonic() - started)
        raise
    finally:
        for file in pcm_files:
            file.unlink(missing_ok=True)
        temporary.cleanup()
        manifest['temporary_cache_removed'] = not work.exists()
        json_write(report / 'manifest.json', manifest)
