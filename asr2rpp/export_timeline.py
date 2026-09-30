"""One non-destructive edit model shared by RPP, OTIO and the public JSON.

Positions use the normalized source timeline; source offsets use the referenced
file's origin. All intervals are half-open seconds, never character-count timing.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import math


def seconds(value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise ValueError('Timeline times must be finite numbers')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError('Timeline times must be finite and nonnegative')
    return value


def _name(value: str, used: set[str]) -> str:
    """Keep distinct speaker identities distinct even if display names collide."""
    candidate, number = value, 1
    while candidate in used:
        number += 1
        candidate = f'{value} ({number})'
    used.add(candidate)
    return candidate


def build_timeline(reference: Path, units, timeline_origin: float,
                   reference_origin: float, diar: bool, sample_rate: int = 48000,
                   *, reference_duration) -> dict:
    reference = Path(reference).resolve()
    origin, timeline = seconds(reference_origin), seconds(timeline_origin)
    duration = seconds(reference_duration)
    if duration <= 0 or type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError('A positive reference duration and sample rate are required')
    tracks = [dict(name='ORIGINAL', speaker=None, muted=True, role='reference', lane=0,
        clips=[dict(text=reference.name, start_seconds=origin, source_start_seconds=0.0,
                    duration_seconds=duration, unit_index=None)])]
    groups = {}
    used = {'ORIGINAL'}
    for index, unit in sorted(enumerate(units), key=lambda pair: (pair[1].start, pair[1].end)):
        if unit.method == 'vad_window':
            raise ValueError('Cannot export VAD window boundaries as speech timestamps; forced alignment is required')
        start, end = seconds(unit.start), seconds(unit.end)
        position = timeline + start
        offset = position - origin
        if end <= start or offset < -1e-9 or offset + end - start > duration + 0.001:
            raise ValueError('Edit interval is outside its reference media')
        # Resampling can round the selected duration up by one input sample.
        # Only the edit boundary is capped; fine units and native data stay intact.
        length = min(end-start, duration-max(0.0, offset))
        if length <= 0:
            continue
        speaker = unit.speaker if diar else None
        key = speaker if diar else '__transcript__'
        title = (str(speaker) if speaker is not None else 'UNKNOWN') if diar else 'Transcript'
        if title == 'ORIGINAL':
            title = 'Speaker ORIGINAL'
        lanes = groups.setdefault(key, [])
        # OTIO Tracks are sequential. Overlaps with the same speaker go into
        # additional lanes, rather than moving or truncating anyone's speech.
        lane = next((item for item in lanes if item['end'] <= position), None)
        if lane is None:
            track = dict(name=_name(title, used), speaker=speaker, muted=False,
                         role='speech', lane=len(lanes), clips=[])
            tracks.append(track)
            lane = dict(track=track, end=0.0)
            lanes.append(lane)
        lane['track']['clips'].append(dict(text=unit.text, start_seconds=position,
            source_start_seconds=max(0.0, offset), duration_seconds=length,
            reference_boundary_clamped=length < end-start,
            unit_index=index, method=unit.method, granularity=unit.granularity))
        lane['end'] = position + length
    value = dict(sample_rate=sample_rate, time_unit='seconds', interval='[start,end)',
        reference=dict(path=str(reference), url=reference.as_uri(),
                       origin_seconds=origin, duration_seconds=duration), tracks=tracks)
    validate_timeline(value)
    return value


def validate_timeline(value: dict) -> None:
    """Validate also when replaying user JSON; no files are opened by this check."""
    if not isinstance(value, dict) or value.get('time_unit') != 'seconds':
        raise ValueError('Unsupported timeline or time unit')
    rate = value.get('sample_rate')
    if type(rate) is not int or not 1 <= rate <= 768000:
        raise ValueError('Invalid timeline sample rate')
    reference = value.get('reference', {})
    path, url = reference.get('path'), reference.get('url')
    if not isinstance(path, str) or not path or '\0' in path:
        raise ValueError('Missing reference media path')
    if not isinstance(url, str) or not url.startswith('file:') or '\0' in url:
        raise ValueError('Reference media must have a file URL')
    duration = seconds(reference.get('duration_seconds'))
    if not duration:
        raise ValueError('Reference media has no duration')
    tracks = value.get('tracks')
    if not isinstance(tracks, list) or not tracks:
        raise ValueError('No timeline tracks')
    origin = seconds(reference.get('origin_seconds'))
    first = tracks[0]
    if first.get('name') != 'ORIGINAL' or first.get('muted') is not True:
        raise ValueError('The first track must be the muted ORIGINAL reference')
    originals = first.get('clips', [])
    if len(originals) != 1:
        raise ValueError('ORIGINAL must contain one full reference clip')
    clip = originals[0]
    if (seconds(clip.get('source_start_seconds')) != 0 or
            seconds(clip.get('start_seconds')) != origin or
            seconds(clip.get('duration_seconds')) != duration):
        raise ValueError('ORIGINAL must preserve the complete reference media')
    names = set()
    for track in tracks:
        name = track.get('name')
        if not isinstance(name, str) or not name or name in names:
            raise ValueError('Timeline track names must be unique nonempty strings')
        names.add(name)
        if type(track.get('muted')) is not bool or not isinstance(track.get('clips'), list):
            raise ValueError('Invalid track mute state or clips')
        previous = 0.0
        for clip in track['clips']:
            start = seconds(clip.get('start_seconds'))
            source_start = seconds(clip.get('source_start_seconds'))
            length = seconds(clip.get('duration_seconds'))
            if not length or start < previous - 1e-9 or source_start + length > duration + 1e-6:
                raise ValueError('Overlapping or out-of-bounds clip')
            if not isinstance(clip.get('text'), str):
                raise ValueError('Clip text must be a string')
            previous = start + length
