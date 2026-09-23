"""Reusable non-destructive RPP writer; no inference, Qt or media dependencies."""
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path


def quote(value: str) -> str:
    if any(c in value for c in ('\n', '\r', '\0')):
        raise ValueError('RPP names/paths must be single-line and cannot contain NUL')
    for delimiter in ('"', "'", '`'):
        if delimiter not in value:
            return delimiter + value + delimiter
    raise ValueError('A field contains all three RPP quote delimiters; edit its display name')


def seconds(value: Fraction) -> str:
    if not isinstance(value, Fraction):
        raise TypeError('Times must be Fraction values')
    with localcontext() as ctx:
        ctx.prec = 40
        text = format(Decimal(value.numerator) / Decimal(value.denominator), '.12f')
    return text.rstrip('0').rstrip('.') or '0'


@dataclass(frozen=True)
class Source:
    path: str
    kind: str = 'WAVE'

    def __post_init__(self):
        if self.kind not in ('WAVE', 'VIDEO', 'MP3', 'FLAC', 'VORBIS') or not self.path:
            raise ValueError('Invalid media source')
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
            raise ValueError('Invalid item time bounds')


@dataclass(frozen=True)
class Track:
    name: str
    items: tuple[Item, ...]

    def __post_init__(self):
        quote(self.name)


@dataclass(frozen=True)
class Project:
    tracks: tuple[Track, ...]
    sample_rate: int = 48000

    def __post_init__(self):
        if type(self.sample_rate) is not int or self.sample_rate <= 0:
            raise ValueError('Invalid project sample rate')


def dumps(project: Project) -> str:
    lines = ['<REAPER_PROJECT 0.1 "7.80" 1', f'  SAMPLERATE {project.sample_rate}',
             '  TEMPO 120', '  TIMELOCKMODE 1']
    for track in project.tracks:
        lines += ['  <TRACK', f'    NAME {quote(track.name)}']
        for item in track.items:
            lines += ['    <ITEM', f'      POSITION {seconds(item.position)}',
                      f'      LENGTH {seconds(item.length)}', '      LOOP 0',
                      f'      SOFFS {seconds(item.source_offset)}', '      VOLPAN 1 0 1 -1',
                      '      FADEIN 1 0 0', '      FADEOUT 1 0 0',
                      '      PLAYRATE 1 1 0 -1 0 0.0025', f'      NAME {quote(item.name)}',
                      f'      <SOURCE {item.source.kind}', f'        FILE {quote(item.source.path)}',
                      '      >', '    >']
        lines += ['  >']
    return '\n'.join(lines + ['>']) + '\n'


def write(project: Project, path: Path) -> None:
    text = dumps(project)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(text, encoding='utf-8', newline='\n')
    temporary.replace(path)
