"""Verify real multi-format exports and replay the public JSON with a frozen CLI."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_exports(document_path):
    """Read OTIO through the official core, not by matching hand-written JSON."""
    import opentimelineio as otio
    from asr2rpp.outputs import validate_document
    path = Path(document_path)
    data = json.loads(path.read_text(encoding='utf-8'))
    validate_document(data)
    timeline = otio.core.deserialize_json_from_file(str(path.with_suffix('.otio')))
    expected = data['timeline']
    assert len(timeline.tracks) == len(expected['tracks'])
    clips = 0
    for track, record in zip(timeline.tracks, expected['tracks']):
        assert track.name == record['name']
        assert track.enabled == (not record['muted'])
        assert track.kind == otio.schema.TrackKind.Audio
        actual = [(index, child) for index, child in enumerate(track) if isinstance(child, otio.schema.Clip)]
        assert len(actual) == len(record['clips'])
        for (index, child), item in zip(actual, record['clips']):
            clips += 1
            assert child.name == item['text']
            assert child.enabled == (not record['muted'])
            assert child.media_reference.target_url == expected['reference']['url']
            assert abs(track.range_of_child_at_index(index).start_time.to_seconds() - item['start_seconds']) < 1e-7
            assert abs(child.source_range.start_time.to_seconds() - item['source_start_seconds']) < 1e-7
            assert abs(child.duration().to_seconds() - item['duration_seconds']) < 1e-7
    rpp = path.with_suffix('.rpp').read_text(encoding='utf-8')
    assert rpp.count('  <TRACK\n') == len(expected['tracks'])
    assert rpp.count('    <ITEM\n') == clips
    first = rpp.split('  <TRACK\n')[1]
    assert 'NAME "ORIGINAL"' in first and 'MUTESOLO 1 0 0' in first
    assert all(k in data for k in ('media', 'timebase', 'settings', 'provenance', 'results', 'transcript'))
    return dict(passed=True, tracks=len(timeline.tracks), clips=clips,
                original_muted=True, source_offsets_verified=True,
                raw_results=len(data['results']), formats=['rpp', 'otio', 'json'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--ffmpeg', default='')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    root = args.report.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cli = str(args.cli.resolve())
    results = {}

    def run(name, *argv, expected=0):
        with (root/(name+'.log')).open('w', encoding='utf-8') as log:
            p = subprocess.run([cli, *map(str, argv)], stdout=log, stderr=subprocess.STDOUT,
                               timeout=1200, cwd=root)
        results[name] = dict(returncode=p.returncode, expected_returncode=expected)
        assert p.returncode == expected, (name, p.returncode)

    from asr2rpp.domain import Unit
    from asr2rpp.pipeline import Settings, Stage
    from asr2rpp.outputs import make_document
    source = root/'reference #1.wav'
    with wave.open(str(source), 'wb') as wav:
        wav.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
        wav.writeframes(b'\0\0' * (16000 * 8))
    native = root/'native-results'
    (native/'asr').mkdir(parents=True)
    (native/'asr/raw.json').write_text('{"original_text":"synthetic export fixture"}')
    units = [Unit(.25, 1.25, 'Speaker A: first', speaker='A'),
             Unit(.75, 2.25, 'Speaker B: overlap', speaker='B'),
             Unit(1., 2., 'Speaker A: overlapping lane', speaker='A'),
             Unit(3.5, 4.5, 'Speaker B: after gap', speaker='B')]
    settings = Settings(Stage('whisper-base'), clip_start=1., output_formats=('rpp', 'otio', 'json'))
    data = make_document(source, source, units, units, settings, native,
        dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()), 5., 0., 8.)
    input_json = root/'portable-result.json'
    input_json.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    run('replay-all', 'convert', input_json, '--format', 'rpp,otio,json', '--output-dir', root/'replay')
    results['replay-all'].update(check_exports(next((root/'replay').glob('*.json'))))
    for format in ('rpp', 'otio', 'json'):
        target = root/('only-'+format)
        run('replay-'+format, 'convert', input_json, '--format', format, '--output-dir', target)
        assert [p.suffix for p in target.iterdir()] == ['.'+format]
    run('empty-convert', 'convert', input_json, '--format', '', '--output-dir', root/'empty', expected=2)
    assert not (root/'empty').exists()
    run('empty-run', 'run', source, '--format', '', '--output-dir', root/'empty-run', expected=2)
    assert not (root/'empty-run').exists()
    if args.fixture:
        fixture = args.fixture.resolve()
        before = hashlib.sha256(fixture.read_bytes()).hexdigest()
        extra = ['--ffmpeg', args.ffmpeg] if args.ffmpeg else []
        run('recognize-all', 'run', fixture, '--asr', 'whisper-base', '--asr-language', 'en',
            '--asr-device', args.device, '--duration', '8', '--format', 'rpp,otio,json',
            '--output-dir', root/'real', *extra)
        result = next((root/'real').glob('*.json'))
        results['recognize-all'].update(check_exports(result))
        public = json.loads(result.read_text(encoding='utf-8'))
        assert public['transcript']['units'] and public['results']
        assert public['provenance']['source_unchanged']
        assert hashlib.sha256(fixture.read_bytes()).hexdigest() == before
        # Conversion must also work without consulting the adjacent report folder.
        standalone = root/'standalone.json'
        standalone.write_bytes(result.read_bytes())
        run('real-replay', 'convert', standalone, '--format', 'rpp,otio,json', '--output-dir', root/'real-replay')
        results['real-replay'].update(check_exports(next((root/'real-replay').glob('*.json'))))
    summary = dict(passed=True, cases=results, fixture='generated silence / public speech',
                   otio_reader='opentimelineio.core', formats=['rpp', 'otio', 'json'])
    (root/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
