"""RPP adapter for the common non-destructive timeline model."""
from fractions import Fraction
from pathlib import Path, PureWindowsPath
import os
from rpp_writer import Source, Item, Track, Project, dumps
from .transcript import safe_label
from .export_timeline import build_timeline, validate_timeline
from .performance import timed


def _relative(reference: str, destination: Path) -> str:
    # Re-exporting Windows JSON on another OS must not reinterpret C:\\ as a
    # relative POSIX filename. Cross-drive paths remain absolute in REAPER.
    if os.name != 'nt' and PureWindowsPath(reference).is_absolute():
        return reference.replace('\\', '/')
    try:
        return os.path.relpath(reference, destination.parent).replace('\\', '/')
    except ValueError:
        return reference.replace('\\', '/')


def render(timeline: dict, output: Path) -> str:
    validate_timeline(timeline)
    reference = timeline['reference']['path']
    suffix = Path(reference).suffix.lower()
    kind = {'.wav': 'WAVE', '.wave': 'WAVE', '.aif': 'WAVE', '.aiff': 'WAVE',
            '.flac': 'FLAC', '.mp3': 'MP3', '.ogg': 'VORBIS'}.get(suffix, 'VIDEO')
    source = Source(_relative(reference, Path(output)), kind)
    tracks = tuple(Track(safe_label(track['name']), tuple(Item(
        safe_label(clip['text']), source, Fraction(str(clip['start_seconds'])),
        Fraction(str(clip['source_start_seconds'])), Fraction(str(clip['duration_seconds'])))
        for clip in track['clips']), muted=track['muted']) for track in timeline['tracks'])
    return dumps(Project(tracks, timeline['sample_rate']))


@timed('rpp_export')
def write_reference(output: Path, reference: Path, units, timeline_origin: float,
                    reference_origin: float, diar: bool, sample_rate: int = 48000,
                    *, reference_duration):
    timeline = build_timeline(reference, units, timeline_origin, reference_origin, diar,
                              sample_rate, reference_duration=reference_duration)
    with Path(output).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(render(timeline, output))
