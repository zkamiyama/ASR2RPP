"""Output selection, stage history and versioned, self-contained result export."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import base64
import copy
import hashlib
import json

from . import __version__
from .catalog import checkpoint
from .export_timeline import build_timeline, validate_timeline
from .performance import timed

def _invalid_constant(value):
    raise ValueError('Non-finite JSON number: ' + value)


FORMATS = ('rpp', 'otio', 'json')
SCHEMA = 'asr2rpp.export'
SCHEMA_VERSION = 1


def validate_formats(formats) -> tuple[str, ...]:
    if not isinstance(formats, (list, tuple)) or not formats:
        raise ValueError('出力形式を1つ以上選択してください。 / Select at least one output format.')
    if any(not isinstance(key, str) or key not in FORMATS for key in formats):
        raise ValueError('Unknown output format; choose rpp, otio or json')
    if len(set(formats)) != len(formats):
        raise ValueError('Duplicate output format')
    return tuple(formats)


def parse_formats(arguments=None) -> tuple[str, ...]:
    if arguments is None:
        return ('rpp',)
    return validate_formats([key.strip().lower() for value in arguments for key in value.split(',')])


def output_paths(primary: Path, formats) -> dict[str, Path]:
    return {key: Path(primary).with_suffix('.'+key) for key in validate_formats(formats)}


def snapshot(report: Path, name: str, units) -> None:
    """Keep each correction stage before later grouping changes its granularity."""
    directory = Path(report)/'history'
    directory.mkdir(parents=True, exist_ok=True)
    data = [asdict(unit) for unit in units]
    (directory/(name+'.json')).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def result_files(report: Path, cancel=None) -> dict:
    """Embed every persisted native/result JSON or text, never audio/model bytes.

    Paths inside native request diagnostics may refer to deleted inference PCM.
    Stable media paths and the exact time mapping live in media/timeline instead.
    Non-JSON diagnostics remain byte-exact base64, not silently repaired text.
    """
    files = {}
    for path in sorted(Path(report).rglob('*')):
        if cancel is not None:
            checkpoint(cancel)
        if path.is_symlink() or not path.is_file() or path.suffix.lower() not in {'.json', '.jsonl', '.txt'}:
            continue
        if path.parent == Path(report) and path.name in {'manifest.json', 'transcript.json'}:
            continue
        data = path.read_bytes()
        record = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        try:
            text = data.decode('utf-8-sig')
            if path.suffix.lower() == '.json':
                record.update(encoding='json', data=json.loads(text,
                    parse_constant=_invalid_constant))
            elif path.suffix.lower() == '.jsonl':
                record.update(encoding='jsonl', data=[json.loads(line, parse_constant=_invalid_constant) for line in text.splitlines() if line.strip()])
            else:
                record.update(encoding='utf-8', data=text)
        except (UnicodeError, ValueError):
            record.update(encoding='base64', data=base64.b64encode(data).decode('ascii'))
        files[path.relative_to(report).as_posix()] = record
    return files


def make_document(source, reference, units, fine_units, settings, report, manifest,
                  duration, reference_origin, reference_duration, sample_rate=48000,
                  warnings=(), cancel=None) -> dict:
    source, reference = Path(source).resolve(), Path(reference).resolve()
    timeline = build_timeline(reference, units, settings.clip_start, reference_origin,
        settings.diar is not None or any(u.speaker for u in units), sample_rate,
        reference_duration=reference_duration)
    info = json.loads(json.dumps(manifest, ensure_ascii=False, allow_nan=False))
    info.update(status='completed', source_unchanged=True)
    return dict(schema=SCHEMA, schema_version=SCHEMA_VERSION,
        generator=dict(name='ASR2RPP', version=__version__),
        media=dict(source=dict(path=str(source), url=source.as_uri(), sha256=manifest.get('source_sha256'),
                              audio_stream='0:a:0'),
                   reference=copy.deepcopy(timeline['reference']),
                   inference=dict(start_seconds=settings.clip_start, duration_seconds=float(duration),
                                  preprocessing=manifest.get('preprocessing'),
                                  temporary_audio_included=False)),
        timebase=dict(unit='seconds', intervals='[start,end)',
            unit_origin='inference selection', timeline_origin='FFmpeg normalized demuxed-media origin',
            unit_to_timeline_offset_seconds=settings.clip_start,
            reference_file_origin_seconds=reference_origin),
        settings=json.loads(json.dumps(asdict(settings), allow_nan=False)), provenance=info,
        transcript=dict(units=[asdict(u) for u in fine_units], edit_units=[asdict(u) for u in units],
                        warnings=list(warnings)),
        timeline=timeline, results=result_files(Path(report), cancel))


def validate_document(value):
    if not isinstance(value, dict) or value.get('schema') != SCHEMA or type(value.get('schema_version')) is not int or value['schema_version'] != SCHEMA_VERSION:
        raise ValueError('Unsupported ASR2RPP export JSON schema/version')
    validate_timeline(value.get('timeline'))


def _render(format, document, destination):
    if format == 'rpp':
        from .rpp_export import render
        return render(document['timeline'], destination)
    if format == 'otio':
        from .otio_export import dumps
        return dumps(document['timeline'], destination.stem)
    return json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


@timed('export')
def write_document(document: dict, primary: Path, formats, cancel=None) -> dict[str, Path]:
    """All selected outputs succeed or only files created by us are rolled back.

    Exclusive creation prevents overwrites even if another app creates a file
    after reservation. No shell commands or adapters are loaded from input JSON.
    """
    validate_document(document)
    paths = output_paths(primary, formats)
    created = []
    try:
        for format, path in paths.items():
            if cancel is not None:
                checkpoint(cancel)
            text = _render(format, document, path)
            # Track ownership only after exclusive open succeeds.
            with path.open('x', encoding='utf-8', newline='\n') as handle:
                created.append(path)
                handle.write(text)
        if cancel is not None:
            checkpoint(cancel)
        return paths
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def export_results(source, reference, units, fine_units, settings, output, report,
                   manifest, duration, reference_origin, reference_duration,
                   sample_rate=48000, warnings=(), cancel=None) -> dict[str, Path]:
    """Shared finalization for single-file, preprocessed and stage-major jobs."""
    paths = output_paths(output, settings.output_formats)
    manifest.update(reference_file=str(reference), reference_origin_seconds=float(reference_origin),
                    timeline_origin_seconds=settings.clip_start,
                    output=str(output), outputs={key: str(path) for key, path in paths.items()},
                    original_track=dict(name='ORIGINAL', muted=True, duration_seconds=float(reference_duration)))
    # RPP/OTIO-only runs need not load/duplicate hundreds of raw result files.
    if 'json' in paths:
        document = make_document(source, reference, units, fine_units, settings, report, manifest,
            duration, reference_origin, reference_duration, sample_rate, warnings, cancel)
    else:
        document = dict(schema=SCHEMA, schema_version=SCHEMA_VERSION,
            timeline=build_timeline(reference, units, settings.clip_start, reference_origin,
                settings.diar is not None or any(u.speaker for u in units), sample_rate,
                reference_duration=reference_duration))
    return write_document(document, output, settings.output_formats, cancel)
