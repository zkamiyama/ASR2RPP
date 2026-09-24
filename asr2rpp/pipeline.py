"""Qt-free pipeline; native audio is never split or modified for RPP export."""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from fractions import Fraction
from pathlib import Path
import copy
import hashlib
import json
import math
import os
import threading
import time
import tempfile
import wave
from .catalog import Model, checkpoint, cache_root, resolve_model, digest
from .adapters import Unit, infer, run_process, ffmpeg_path, executable
from .text_join import join_timed
from rpp_writer import Source, Item, Track, Project, dumps

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
    argv = [ffmpeg_path(settings.ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin',
            '-i', str(source), '-map', '0:a:0', '-vn', '-af', filters,
            '-ac', '1', '-ar', str(rate), '-c:a', 'pcm_s16le', '-y', str(destination)]
    run_process(argv, cancel, progress, destination.with_suffix('.decode.log'))
    with wave.open(str(destination)) as audio:
        return audio.getnframes() / audio.getframerate()


def clean_bounds(units: list[Unit], duration: float, warnings: list[str]) -> list[Unit]:
    result = []
    for index, unit in enumerate(units):
        if unit.end <= unit.start:
            warnings.append(f'Invalid interval {index}; excluded from edit output')
            continue
        start, end = max(0.0, unit.start), min(duration, unit.end)
        if start != unit.start or end != unit.end:
            warnings.append(f'Interval {index} clipped to media duration for editing; raw unchanged')
        if end > start:
            result.append(replace(unit, start=start, end=end))
    return sorted(result, key=lambda u: (u.start, u.end))


def group_units(units: list[Unit], maximum: float = 18.0) -> list[Unit]:
    """Merge finer units, never invent finer timestamps from a coarse segment."""
    grouped = []
    for unit in units:
        if not unit.text.strip():
            continue
        if (grouped and unit.granularity != 'segment' and
                grouped[-1].speaker == unit.speaker and
                unit.start - grouped[-1].end <= 0.65 and
                unit.end - grouped[-1].start <= maximum and
                not grouped[-1].text.rstrip().endswith(('。', '！', '？', '!', '?'))):
            previous = grouped[-1]
            previous.text = join_timed(previous.text, unit.text, unit.granularity)
            previous.end = max(previous.end, unit.end)
        else:
            grouped.append(replace(unit, text=unit.text.replace('\u2581', ' ')))
    return grouped


def assign_speakers(units: list[Unit], turns: list[Unit], warnings: list[str]) -> list[Unit]:
    assigned = []
    for unit in units:
        overlap = {}
        for turn in turns:
            amount = max(0.0, min(unit.end, turn.end) - max(unit.start, turn.start))
            if amount and turn.speaker is not None:
                overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + amount
        speaker = max(overlap, key=overlap.get) if overlap else None
        if not overlap or overlap[speaker] / (unit.end - unit.start) < 0.55:
            speaker = 'UNKNOWN'
            warnings.append(f'{unit.start:.3f}: speaker unresolved; no text was split by guessed timing')
        elif len(overlap) > 1:
            warnings.append(f'{unit.start:.3f}: multiple speaker candidates {list(overlap)}; largest overlap selected')
        assigned.append(replace(unit, speaker=speaker))
    return assigned


def safe_label(text: str) -> str:
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\0', '')
    if all(c in text for c in ('"', "'", '`')):
        text = text.replace('`', 'ˋ')
    return text or '(speech)'


def export_rpp(source: Path, output: Path, units: list[Unit], offset: float, diar: bool):
    try:
        path = os.path.relpath(source, output.parent).replace('\\', '/')
    except ValueError:
        path = str(source).replace('\\', '/')
    ext = source.suffix.lower()
    kind = {'.wav': 'WAVE', '.wave': 'WAVE', '.aif': 'WAVE', '.aiff': 'WAVE',
            '.mp3': 'MP3', '.flac': 'FLAC', '.ogg': 'VORBIS'}.get(ext, 'VIDEO')
    ref = Source(path, kind)
    tracks = {}
    for unit in units:
        key = (unit.speaker or 'UNKNOWN') if diar else 'Transcript'
        start = Fraction(str(unit.start)) + Fraction(str(offset))
        end = Fraction(str(unit.end)) + Fraction(str(offset))
        tracks.setdefault(key, []).append(Item(safe_label(unit.text), ref, start, start, end - start))
    project = Project(tuple(Track(safe_label(str(name)), tuple(items)) for name, items in tracks.items()))
    with output.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(dumps(project))


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
                'source_unchanged': None}
    json_write(report / 'manifest.json', manifest)
    try:
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
        asr = infer(asr_model, weights['asr'], audio, report / 'asr', settings.asr.options(), cancel, progress)
        units = clean_bounds(asr.units, duration, warnings)
        if not units:
            raise ValueError('No timed speech was returned; raw engine output is retained')
        if any(u.method == 'emission_frame' for u in units):
            warnings.append('ASR times are emission-frame estimates, not exact spoken-word boundaries; alignment recommended')
        units = group_units(units)
        units = [replace(u, speaker=None) for u in units]
        if settings.align is not None:
            align_model = catalog[settings.align.model_id]
            aligned = []
            for index, segment in enumerate(units):
                checkpoint(cancel)
                progress(f'Forced alignment — {index + 1}/{len(units)}')
                if segment.end - segment.start > 55:
                    raise ValueError('Alignment needs <=55-second matched transcript/audio segments; this segment is too long. ASR-only remains available.')
                begin = max(0, segment.start - 0.15)
                length = min(duration, segment.end + 0.25) - begin
                file = work / f'align_{index}.wav'
                pcm_files.append(file)
                decode(source, file, align_model.sample_rate, settings, cancel, progress,
                       settings.clip_start + begin, length)
                result = infer(align_model, weights['align'], file, report / f'align_{index}',
                               settings.align.options(), cancel, progress, segment.text)
                if not result.units:
                    raise ValueError('Aligner returned no intervals')
                local = clean_bounds(result.units, length, warnings)
                aligned += [replace(u, start=u.start + begin, end=u.end + begin) for u in local]
            units = clean_bounds(aligned, duration, warnings)
        if settings.diar is not None:
            model = catalog[settings.diar.model_id]
            audio, _ = pcm(model.sample_rate)
            progress('Diarization — assigning speakers')
            result = infer(model, weights['diar'], audio, report / 'diar', settings.diar.options(), cancel, progress)
            units = assign_speakers(units, clean_bounds(result.units, duration, warnings), warnings)
        units = group_units(units)
        if not units:
            raise ValueError('No valid intervals remain after normalization')
        checkpoint(cancel)
        progress('RPP — writing non-destructive references')
        json_write(report / 'transcript.json', {'clip_start': settings.clip_start, 'duration': duration,
                   'units': [asdict(u) for u in units], 'warnings': warnings})
        export_rpp(source, output, units, settings.clip_start, settings.diar is not None)
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
