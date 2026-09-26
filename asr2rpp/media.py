"""Media length and sample-exact cached PCM slicing; no source modification."""
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import math
import os
import struct
import wave
from .catalog import checkpoint
from .adapters import ffmpeg_path, run_process


@dataclass(frozen=True)
class WaveInfo:
    sample_rate: int
    channels: int
    frames: int
    bits: int
    format_tag: int

    @property
    def duration(self) -> Fraction:
        return Fraction(self.frames, self.sample_rate)


from .performance import timed

def wave_info(path: Path) -> WaveInfo:
    """Read PCM/IEEE-float WAV headers without converting the saved media."""
    size = path.stat().st_size
    with path.open('rb') as handle:
        if handle.read(4) != b'RIFF':
            raise ValueError('Expected RIFF WAV output; RF64 is not supported in this preview')
        handle.read(4)
        if handle.read(4) != b'WAVE':
            raise ValueError('Expected WAVE format')
        fmt = None
        data_size = None
        while handle.tell() + 8 <= size:
            tag = handle.read(4)
            length = struct.unpack('<I', handle.read(4))[0]
            position = handle.tell()
            if position + length > size:
                raise ValueError('Truncated WAV chunk')
            if tag == b'fmt ':
                if length < 16:
                    raise ValueError('Invalid WAV format chunk')
                fmt = struct.unpack('<HHIIHH', handle.read(16))
            elif tag == b'data':
                data_size = length
            handle.seek(position + length + (length & 1))
            if fmt is not None and data_size is not None:
                break
        if fmt is None or data_size is None:
            raise ValueError('WAV is missing format/data chunks')
        encoding, channels, rate, _, block, bits = fmt
        if encoding not in {1, 3, 65534} or min(channels, rate, block, bits) <= 0:
            raise ValueError('Unsupported WAV encoding')
        if data_size % block:
            raise ValueError('WAV data is not frame-aligned')
        return WaveInfo(rate, channels, data_size // block, bits, encoding)


def validate_duration(before: WaveInfo, after: WaveInfo):
    if before.sample_rate != after.sample_rate or before.channels != after.channels:
        raise ValueError('Preprocessing changed sample rate/channels unexpectedly')
    if abs(before.frames - after.frames) > 1:
        raise ValueError('Preprocessing changed audio duration; refusing an unverified time mapping')



@timed('reference_duration')
def full_reference_duration(source, settings, known_duration, work, cancel, progress):
    """Reuse full decoded duration; only scan when an original was clipped.

    WAV headers give an exact frame count, including silence. For other media
    with a clipped inference range, a full normalized audio scan to the null muxer
    avoids relying on inaccurate container duration or writing another large WAV.
    """
    source = Path(source)
    if source.suffix.lower() in {'.wav', '.wave'}:
        return wave_info(source).duration
    if not settings.clip_start and not settings.clip_duration:
        length = Fraction(str(known_duration))
        if length <= 0:
            raise ValueError('Reference duration must be positive')
        return length
    # Some valid FFmpeg builds report out_time_us=N/A for audio-only null
    # outputs. Raw PCM byte count is exact and does not depend on progress PTS.
    # Write to the OS null device, never another full WAV on the user's disk.
    total_bytes = None
    finished = False
    def collect(line):
        nonlocal total_bytes, finished
        if line.startswith('total_size='):
            try:
                value = int(line.split('=', 1)[1])
                if value >= 0:
                    total_bytes = value
            except ValueError:
                pass
        elif line == 'progress=end':
            finished = True
        elif not line.startswith(('out_time', 'bitrate=', 'speed=', 'progress=', 'dup_frames=', 'drop_frames=')):
            progress(line)
    progress('ORIGINAL — measuring the full reference duration')
    run_process([ffmpeg_path(settings.ffmpeg, progress, cancel), '-hide_banner',
                 '-loglevel', 'error', '-nostdin', '-nostats', '-progress', 'pipe:1',
                 '-i', str(source), '-map', '0:a:0', '-vn',
                 '-af', 'aresample=48000:async=1:first_pts=0', '-ac', '1', '-ar', '48000',
                 '-c:a', 'pcm_s16le', '-f', 's16le', '-y', os.devnull],
                cancel, collect, Path(work) / 'reference-duration.log')
    if not finished or total_bytes is None or total_bytes <= 0 or total_bytes % 2:
        raise ValueError('Could not determine the full reference duration from complete PCM output')
    return Fraction(total_bytes, 2 * 48000)


@timed('slice_pcm')
def slice_pcm(source: Path, destination: Path, start: float, duration: float, cancel) -> float:
    """Copy only a requested interval from decoded PCM, in bounded blocks.

    This replaces an FFmpeg process (and a decode from the beginning of a video)
    for every alignment segment. Cached audio only; the original is never cut.
    """
    if not all(math.isfinite(v) and v >= 0 for v in (start, duration)) or duration <= 0:
        raise ValueError('Invalid PCM slice interval')
    checkpoint(cancel)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(source), 'rb') as incoming:
        rate = incoming.getframerate()
        first = min(incoming.getnframes(), round(start * rate))
        count = min(incoming.getnframes() - first, round(duration * rate))
        if count <= 0:
            raise ValueError('Empty PCM slice')
        incoming.setpos(first)
        with wave.open(str(destination), 'wb') as outgoing:
            outgoing.setparams(incoming.getparams())
            remaining = count
            frame_bytes = incoming.getnchannels() * incoming.getsampwidth()
            while remaining:
                checkpoint(cancel)
                block = incoming.readframes(min(remaining, 65536))
                if not block:
                    raise ValueError('Truncated PCM cache')
                outgoing.writeframesraw(block)
                remaining -= len(block) // frame_bytes
    return count / rate
