"""Normalized speech data, independent of inference engines and UI."""
from dataclasses import dataclass

@dataclass
class Unit:
    start: float
    end: float
    text: str = ''
    speaker: str | None = None
    granularity: str = 'segment'
    method: str = 'native_interval'
    owner_start: float | None = None
    owner_end: float | None = None

@dataclass
class Result:
    units: list[Unit]
    raw: object
    text: str = ''
    warnings: list[str] | None = None

