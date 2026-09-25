"""Optional source separation and explicit ownership of referenced audio assets.

This layer never substitutes a different checkpoint for the configured model.
The existing recognition pipeline stays Qt-free and reusable.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from fractions import Fraction
from pathlib import Path
import copy
import json
import os
import shutil
import struct
import tempfile
import time
from . import pipeline as core
from .catalog import Model, checkpoint, cache_root, digest, resolve_model
from .adapters import executable, ffmpeg_path, run_process, Unit
from rpp_writer import Source, Item, Track, Project, dumps


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


@dataclass(frozen=True)
class WaveInfo:
    sample_rate: int
    channels: int
    frames: int
    bits: int
    format_tag: int


def wave_info(path: Path) -> WaveInfo:
    """Read PCM/IEEE-float WAV headers without converting the saved media."""
    size = path.stat().st_size
    with path.open('rb') as handle:
        if handle.read(4) != b'RIFF':
            raise ValueError('Expected RIFF WAV output; RF64 is not supported in this preview')
        handle.read(4)
        if handle.read(4) != b'WAVE':
            raise ValueError('Expected WAVE format')
        fmt = None
        data_size = None
        while handle.tell() + 8 <= size:
            tag = handle.read(4)
            length = struct.unpack('<I', handle.read(4))[0]
            position = handle.tell()
            if position + length > size:
                raise ValueError('Truncated WAV chunk')
            if tag == b'fmt ':
                if length < 16:
                    raise ValueError('Invalid WAV format chunk')
                fmt = struct.unpack('<HHIIHH', handle.read(16))
            elif tag == b'data':
                data_size = length
            handle.seek(position + length + (length & 1))
            if fmt is not None and data_size is not None:
                break
        if fmt is None or data_size is None:
            raise ValueError('WAV is missing format/data chunks')
        encoding, channels, rate, _, block, bits = fmt
        if encoding not in {1, 3, 65534} or min(channels, rate, block, bits) <= 0:
            raise ValueError('Unsupported WAV encoding')
        if data_size % block:
            raise ValueError('WAV data is not frame-aligned')
        return WaveInfo(rate, channels, data_size // block, bits, encoding)


def validate_duration(before: WaveInfo, after: WaveInfo):
    if before.sample_rate != after.sample_rate or before.channels != after.channels:
        raise ValueError('Preprocessing changed sample rate/channels unexpectedly')
    if abs(before.frames - after.frames) > 1:
        raise ValueError('Preprocessing changed audio duration; refusing an unverified time mapping')


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
    session = {**model.defaults.get('session', {}), **(stage.parameters or {})}
    for key, value in session.items():
        if not isinstance(value, (str, int, float, bool)) or not key.replace('_', '').isalnum():
            raise ValueError('Invalid preprocessing session parameter')
        scalar = str(value).lower() if isinstance(value, bool) else str(value)
        args += ['--session-option', f'{model.family}.{key}={scalar}']
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


def write_reference(output: Path, reference: Path, units, timeline_origin: float,
                    reference_origin: float, diar: bool, sample_rate: int = 48000):
    """Project time and file time are deliberately separate, including for clips."""
    try:
        relative = os.path.relpath(reference, output.parent).replace('\\', '/')
    except ValueError:
        relative = str(reference).replace('\\', '/')
    kind = {'.wav': 'WAVE', '.wave': 'WAVE', '.aif': 'WAVE', '.aiff': 'WAVE',
            '.flac': 'FLAC', '.mp3': 'MP3', '.ogg': 'VORBIS'}.get(reference.suffix.lower(), 'VIDEO')
    src, tracks = Source(relative, kind), {}
    for unit in units:
        position = Fraction(str(timeline_origin)) + Fraction(str(unit.start))
        source_offset = position - Fraction(str(reference_origin))
        key = (unit.speaker or 'UNKNOWN') if diar else 'Transcript'
        item = Item(core.safe_label(unit.text), src, position, source_offset,
                    Fraction(str(unit.end)) - Fraction(str(unit.start)))
        tracks.setdefault(key, []).append(item)
    project = Project(tuple(Track(core.safe_label(str(key)), tuple(items)) for key, items in tracks.items()), sample_rate)
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(dumps(project))


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
            try:
                core.run_job(vocals, downstream, catalog, cancel, progress)
            finally:
                for analysis in (work / 'recognition').glob('*.asr2rpp'):
                    shutil.copytree(analysis, report / 'analysis', dirs_exist_ok=True)
            transcript = json.loads((report / 'analysis' / 'transcript.json').read_text(encoding='utf-8'))
            units = [Unit(**record) for record in transcript['units']]
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
            write_reference(output, reference, units, settings.clip_start, origin,
                            settings.diar is not None, sample_rate)
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
