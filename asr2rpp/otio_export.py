"""Native OTIO serialization via the official, small OpenTimelineIO core.

No adapter discovery, external plugin code, media transcoding or frame rounding.
"""
from .export_timeline import validate_timeline


def dumps(timeline: dict, name: str = 'ASR2RPP') -> str:
    import opentimelineio as otio

    validate_timeline(timeline)
    rate = timeline['sample_rate']
    reference = timeline['reference']
    def time(seconds):
        return otio.opentime.RationalTime(float(seconds) * rate, rate)
    def span(start, duration):
        return otio.opentime.TimeRange(time(start), time(duration))
    result = otio.schema.Timeline(name=name, global_start_time=time(0))
    result.metadata['asr2rpp'] = dict(schema_version=1, time_unit='seconds',
        original_track_muted=True, reference_origin_seconds=reference['origin_seconds'])
    for record in timeline['tracks']:
        track = otio.schema.Track(name=record['name'], kind=otio.schema.TrackKind.Audio)
        track.enabled = not record['muted']
        track.metadata['asr2rpp'] = {key: record[key] for key in ('speaker', 'muted', 'role', 'lane')}
        cursor = 0.0
        for item in record['clips']:
            position = item['start_seconds']
            if position > cursor:
                track.append(otio.schema.Gap(duration=time(position-cursor)))
            clip = otio.schema.Clip(name=item['text'],
                media_reference=otio.schema.ExternalReference(
                    target_url=reference['url'], available_range=span(0, reference['duration_seconds'])),
                source_range=span(item['source_start_seconds'], item['duration_seconds']))
            clip.enabled = not record['muted']
            clip.metadata['asr2rpp'] = dict(item, speaker=record['speaker'], muted=record['muted'])
            track.append(clip)
            cursor = position + item['duration_seconds']
        result.tracks.append(track)
    # The core API does not consult arbitrary OTIO adapter/plugin search paths.
    return otio.core.serialize_json_to_string(result) + '\n'
