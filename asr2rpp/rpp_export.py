"""Build non-destructive projects with an uncut, muted reference track first."""
from fractions import Fraction
from pathlib import Path
import os
from rpp_writer import Source, Item, Track, Project, write
from .transcript import safe_label


def _seconds(value) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(str(value))


def write_reference(output: Path, reference: Path, units, timeline_origin: float,
                    reference_origin: float, diar: bool, sample_rate: int = 48000,
                    *, reference_duration):
    """ORIGINAL covers the entire reference, not just first/last detected speech.

    A processed clip starts at reference_origin on the source timeline; its file
    offset is zero. Edited items keep their existing source offsets. The reference
    track is muted initially to avoid doubling playback with the edited tracks.
    """
    try:
        relative = os.path.relpath(reference, output.parent).replace('\\', '/')
    except ValueError:
        relative = str(reference).replace('\\', '/')
    kind = {'.wav': 'WAVE', '.wave': 'WAVE', '.aif': 'WAVE', '.aiff': 'WAVE',
            '.flac': 'FLAC', '.mp3': 'MP3', '.ogg': 'VORBIS'}.get(reference.suffix.lower(), 'VIDEO')
    src = Source(relative, kind)
    origin = _seconds(reference_origin)
    timeline = _seconds(timeline_origin)
    original = Item(safe_label(reference.name), src, origin, Fraction(0),
                    _seconds(reference_duration))
    tracks = [Track('ORIGINAL', (original,), muted=True)]
    edits = {}
    for unit in units:
        position = timeline + _seconds(unit.start)
        key = (unit.speaker or 'UNKNOWN') if diar else 'Transcript'
        # Reserve the reference name even if a custom diarizer supplies ORIGINAL.
        if key == 'ORIGINAL':
            key = 'Speaker ORIGINAL'
        item = Item(safe_label(unit.text), src, position, position - origin,
                    _seconds(unit.end) - _seconds(unit.start))
        edits.setdefault(key, []).append(item)
    tracks.extend(Track(safe_label(str(key)), tuple(items)) for key, items in edits.items())
    write(Project(tuple(tracks), sample_rate), output)
