"""Transcript interval operations; never invent speech times or split words."""
from dataclasses import replace
from .adapters import Unit
from .text_join import join_timed


from .performance import timed

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


def has_alignable_text(text: str) -> bool:
    """True when text contains at least one Unicode letter or number.

    Forced aligners cannot produce timestamps for punctuation/whitespace-only
    fragments. Those fragments retain their ASR timing instead of being dropped.
    """
    return any(character.isalnum() for character in str(text or ""))


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


@timed('speaker_assignment')
def assign_speakers(units: list[Unit], turns: list[Unit], warnings: list[str]) -> list[Unit]:
    ordered_turns = sorted(enumerate(turns), key=lambda pair: pair[1].start)
    active, next_turn, assigned = {}, 0, [None] * len(units)
    for index, unit in sorted(enumerate(units), key=lambda pair: pair[1].start):
        # An earlier wide unit may have added future turns; the overlap check
        # below still excludes these when later units end sooner.
        while next_turn < len(ordered_turns) and ordered_turns[next_turn][1].start < unit.end:
            ordinal, turn = ordered_turns[next_turn]
            active[ordinal] = turn
            next_turn += 1
        active = {i: t for i, t in active.items() if t.end > unit.start}
        overlap = {}
        # Preserve original turn order for deterministic equal-overlap ties.
        for ordinal in sorted(active):
            turn = active[ordinal]
            amount = min(unit.end, turn.end) - max(unit.start, turn.start)
            if amount > 0 and turn.speaker is not None:
                overlap[turn.speaker] = overlap.get(turn.speaker, 0.0) + amount
        speaker = max(overlap, key=overlap.get) if overlap else None
        if not overlap or overlap[speaker] / (unit.end - unit.start) < 0.55:
            speaker = 'UNKNOWN'
            warnings.append(f'{unit.start:.3f}: speaker unresolved; no text was split by guessed timing')
        elif len(overlap) > 1:
            warnings.append(f'{unit.start:.3f}: multiple speaker candidates {list(overlap)}; largest overlap selected')
        assigned[index] = replace(unit, speaker=speaker)
    return assigned


def safe_label(text: str) -> str:
    text = text.replace('\r', ' ').replace('\n', ' ').replace('\0', '')
    if all(c in text for c in ('"', "'", '`')):
        text = text.replace('`', 'ˋ')
    return text or '(speech)'


