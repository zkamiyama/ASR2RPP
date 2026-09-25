"""Independent, stdlib-only RPP serialization. No ASR, Qt or media conversion."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path


def quote(text: str) -> str:
    if any(c in text for c in '\n\r\0'):
        raise ValueError('RPP fields must be single-line and contain no NUL')
    for delimiter in ('"', "'", '`'):
        if delimiter not in text:
            return delimiter + text + delimiter
    raise ValueError('All RPP quote delimiters occur in field; use a safe display label')


def seconds(value: Fraction) -> str:
    if not isinstance(value, Fraction):
        raise TypeError('Time must be Fraction, not a rounded float')
    with localcontext() as context:
        context.prec = 40
        return format(Decimal(value.numerator) / Decimal(value.denominator), '.12f').rstrip('0').rstrip('.') or '0'


@dataclass(frozen=True)
class Source:
    path: str
    kind: str = 'WAVE'

    def __post_init__(self):
        if self.kind not in {'WAVE', 'VIDEO', 'MP3', 'FLAC', 'VORBIS'}:
            raise ValueError('Unsupported RPP source type')
        if not self.path:
            raise ValueError('Empty source path')
        quote(self.path)


@dataclass(frozen=True)
class Item:
    name: str
    source: Source
    position: Fraction
    source_offset: Fraction
    length: Fraction

    def __post_init__(self):
        quote(self.name)
        for value in (self.position, self.source_offset, self.length):
            seconds(value)
        if self.position < 0 or self.source_offset < 0 or self.length <= 0:
            raise ValueError('Invalid item bounds')


@dataclass(frozen=True)
class Track:
    name: str
    items: tuple[Item, ...]
    muted: bool = False

    def __post_init__(self):
        quote(self.name)


@dataclass(frozen=True)
class Project:
    tracks: tuple[Track, ...]
    sample_rate: int = 48000

    def __post_init__(self):
        if self.sample_rate <= 0:
            raise ValueError('Invalid project sample rate')


def dumps(project: Project) -> str:
    lines = ['<REAPER_PROJECT 0.1 "7.0" 1',
             f'  SAMPLERATE {project.sample_rate}', '  TEMPO 120']
    for track in project.tracks:
        lines += ['  <TRACK', f'    NAME {quote(track.name)}']
        if track.muted:
            lines.append('    MUTESOLO 1 0 0')
        for item in track.items:
            lines += ['    <ITEM', f'      POSITION {seconds(item.position)}',
                      f'      LENGTH {seconds(item.length)}', '      LOOP 0',
                      f'      SOFFS {seconds(item.source_offset)}',
                      '      PLAYRATE 1 1 0 -1 0 0.0025',
                      f'      NAME {quote(item.name)}',
                      f'      <SOURCE {item.source.kind}',
                      f'        FILE {quote(item.source.path)}', '      >', '    >']
        lines.append('  >')
    return '\n'.join(lines + ['>']) + '\n'


def write(project: Project, path: Path) -> None:
    """Refuse overwrite; output reservation is the caller's responsibility."""
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(dumps(project))
