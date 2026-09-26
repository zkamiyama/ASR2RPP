"""Native Silero VAD and bounded windows on the original PCM timeline (no torch)."""
from dataclasses import dataclass, replace
from pathlib import Path
from urllib.request import Request, urlopen
import json
import math
import os
import re
import shutil
import sys
import tempfile
import threading
import wave

from .catalog import assets_root, weights_root, checkpoint, digest

VAD_REVISION = 'e5614ed76a5dd4b03fad5068c89efcd2617a9d1e'
VAD_FILE = 'ggml-silero-v5.1.2.bin'
VAD_SHA256 = '29940d98d42b91fbd05ce489f3ecf7c72f0a42f027e4875919a28fb4c04ea2cf'
VAD_SIZE = 885098
VAD_URL = f'https://huggingface.co/ggml-org/whisper-vad/resolve/{VAD_REVISION}/{VAD_FILE}'
_DOWNLOAD_LOCK = threading.Lock()


def vad_model(model, parameters, cancel, progress):
    custom = parameters.get('vad_model', '')
    if custom:
        path = Path(custom).expanduser()
        if not path.is_absolute():
            path = (model.definition.parent if model.definition else Path.cwd()) / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f'VAD model not found: {path}')
        return path, {'path': str(path), 'sha256': digest(path), 'custom': True}
    path = weights_root() / 'silero-vad' / VAD_SHA256[:16] / VAD_FILE
    while not _DOWNLOAD_LOCK.acquire(timeout=0.1):
        checkpoint(cancel)
    try:
        checkpoint(cancel)
        if path.exists():
            if path.stat().st_size != VAD_SIZE or digest(path) != VAD_SHA256:
                raise ValueError(f'VAD model checksum mismatch: {path}')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(prefix='vad-', suffix='.part', dir=path.parent)
            partial = Path(name)
            try:
                progress('VAD: downloading Silero v5.1.2 (verified, <1 MiB)')
                with os.fdopen(fd, 'wb') as output:
                    with urlopen(Request(VAD_URL, headers={'User-Agent': 'ASR2RPP/0.1'}), timeout=30) as response:
                        size = 0
                        while block := response.read(65536):
                            checkpoint(cancel)
                            size += len(block)
                            if size > VAD_SIZE:
                                raise ValueError('Unexpected VAD download size')
                            output.write(block)
                if partial.stat().st_size != VAD_SIZE or digest(partial) != VAD_SHA256:
                    raise ValueError('VAD download checksum mismatch')
                checkpoint(cancel)
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        return path, {'path': str(path), 'url': VAD_URL, 'revision': VAD_REVISION,
                      'sha256': VAD_SHA256, 'custom': False}
    finally:
        _DOWNLOAD_LOCK.release()


def vad_executable(asr_binary=None):
    name = 'whisper-vad-speech-segments' + ('.exe' if sys.platform == 'win32' else '')
    roots = [Path(sys.executable).parent / 'engines' / 'whisper_cpp-cpu',
             assets_root() / 'engines' / 'whisper_cpp-cpu']
    if asr_binary:
        roots.append(Path(asr_binary).parent)
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError('Native VAD helper is missing. Update/build the whisper.cpp runtime including whisper-vad-speech-segments.')


@dataclass(frozen=True)
class VadOptions:
    threshold: float = 0.5
    min_speech_ms: int = 100
    min_silence_ms: int = 250
    pad_ms: int = 200
    overlap: float = 0.2
    maximum: float = 25.0

    @classmethod
    def from_parameters(cls, parameters, limit):
        names = {'threshold': ('vad_threshold', 0.5, 0.01, 1.0),
                 'min_speech_ms': ('vad_min_speech_duration_ms', 100, 0, 1000),
                 'min_silence_ms': ('vad_min_silence_duration_ms', 250, 50, 2000),
                 'pad_ms': ('vad_speech_pad_ms', 200, 0, 1000),
                 'overlap': ('vad_samples_overlap', 0.2, 0, 1.0)}
        values = {}
        for key, (name, default, low, high) in names.items():
            value = parameters.get(name, default)
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not low <= value <= high or (key.endswith('_ms') and type(value) is not int)):
                raise ValueError(f'{name} must be in {low}..{high} with a valid numeric type')
            values[key] = value
        maximum = parameters.get('vad_max_speech_duration_s', 0) or limit
        if type(maximum) not in (int, float) or not math.isfinite(maximum) or not 2 <= maximum <= limit:
            raise ValueError('VAD segment duration must be 2 seconds or more and within the TOML limit')
        if 2 * max(values['overlap'], values['pad_ms'] / 1000) + 0.25 >= maximum:
            raise ValueError('VAD padding/overlap leaves no room within the maximum duration')
        return cls(**values, maximum=float(maximum))


def parse_segments(text, duration):
    """The app-pinned helper prints centiseconds, not seconds. Fail on truncation."""
    counts = re.findall(r'^Detected (\d+) speech segments:\s*$', text, re.M)
    matches = re.findall(r'^Speech segment (\d+): start = ([-+\d.eE]+), end = ([-+\d.eE]+)\s*$', text, re.M)
    if len(counts) != 1 or len(matches) != int(counts[0]):
        raise ValueError('Missing or incomplete native VAD segment output')
    result = []
    for ordinal, (index, left, right) in enumerate(matches):
        start, end = float(left) / 100, float(right) / 100
        if (int(index) != ordinal or not all(math.isfinite(t) for t in (start, end))
                or start < 0 or end <= start or end > duration + 0.1
                or (result and start < result[-1][1] - 0.001)):
            raise ValueError('Invalid or overlapping native VAD intervals')
        end = min(end, duration)
        if end > start:
            result.append((start, end))
    return result


@dataclass(frozen=True)
class SpeechWindow:
    start: float
    end: float
    owner_start: float
    owner_end: float


def bounded_windows(spans, audio, maximum, overlap, cancel):
    """Keep detected silence gaps. Oversized continuous speech gets context overlap.

    Native VAD's post-merge step can undo its own max-duration split. Enforce
    the bound again here, selecting a low-energy point near the requested cut.
    Disjoint ownership intervals let the aligner remove duplicate boundary words
    by time, not by deleting matching strings (which may be legitimate repeats).
    """
    import numpy as np
    windows, forced = [], 0
    with wave.open(str(audio), 'rb') as pcm:
        if (pcm.getframerate(), pcm.getnchannels(), pcm.getsampwidth()) != (16000, 1, 2):
            raise ValueError('VAD requires 16 kHz mono PCM16')
        rate = pcm.getframerate()
        for start, end in spans:
            checkpoint(cancel)
            boundaries = [start]
            cursor = start
            while end - cursor + (overlap if cursor > start else 0) > maximum:
                target = cursor + maximum - 2 * overlap
                low = max(cursor + 0.5, target - min(2.0, maximum / 4))
                pcm.setpos(round(low * rate))
                values = np.frombuffer(pcm.readframes(round((target - low) * rate)), dtype='<i2').astype('float32')
                frame = 320
                count = len(values) // frame
                cut = target
                if count:
                    energy = np.mean(values[:count * frame].reshape(count, frame) ** 2, axis=1)
                    cut = low + (int(np.argmin(energy)) + 0.5) * frame / rate
                cut = min(target, max(cursor + 0.25, cut))
                boundaries.append(cut)
                cursor = cut
                forced += 1
            boundaries.append(end)
            for i, (left, right) in enumerate(zip(boundaries, boundaries[1:])):
                a = max(start, left - (overlap if i else 0))
                b = min(end, right + (overlap if i < len(boundaries)-2 else 0))
                if b-a > maximum + 1/16000:
                    raise ValueError('Internal VAD window exceeds the TOML bound')
                windows.append(SpeechWindow(a, b, left, right))
    return windows, forced


def detect_windows(model, parameters, audio, root, binary, cancel, progress, limit, context_overlap=True):
    from .adapters import run_process
    from .media import wave_info
    options = VadOptions.from_parameters(parameters, limit)
    requested_overlap = options.overlap
    if not context_overlap:
        # Without word alignment there is no honest way to remove repeated
        # words from overlapping contexts. Keep distinct VAD inputs instead.
        options = replace(options, overlap=0.0)
    weights, provenance = vad_model(model, parameters, cancel, progress)
    helper = vad_executable(binary)
    info = wave_info(audio)
    duration = info.frames / info.sample_rate
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    help_log = root / 'vad-help.log'
    run_process([str(helper), '--help'], cancel, lambda _line: None, help_log, timeout=30)
    if 'centiseconds (1/100th of a second)' not in help_log.read_text(encoding='utf-8'):
        raise ValueError('Unrecognized VAD timestamp units; use the app-pinned native helper')
    log = root / 'vad.log'
    progress('VAD: detecting speech on the original audio timeline')
    run_process([str(helper), '-vm', str(weights), '-f', str(audio), '-t', '4',
                 '-vt', str(options.threshold), '-vspd', str(options.min_speech_ms),
                 '-vsd', str(options.min_silence_ms), '-vmsd', str(options.maximum),
                 '-vp', str(options.pad_ms), '-vo', '0'], cancel, progress, log)
    spans = parse_segments(log.read_text(encoding='utf-8'), duration)
    windows, forced = bounded_windows(spans, audio, options.maximum, options.overlap, cancel)
    from dataclasses import asdict
    record = {'model': provenance, 'options': asdict(options), 'units': 'seconds',
              'time_mapping': 'original PCM timeline; silence is not concatenated',
              'detected_spans': spans, 'windows': [asdict(w) for w in windows],
              'forced_splits': forced, 'requested_overlap': requested_overlap,
              'context_overlap_enabled': context_overlap}
    (root / 'vad.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    progress(f'VAD: {len(spans)} speech regions, {len(windows)} bounded windows, {forced} forced splits')
    return windows, record
