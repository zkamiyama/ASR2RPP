"""Native engine subprocess adapters. Unknown output formats fail, never fabricate words."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from .catalog import Model, Cancelled, checkpoint, assets_root


@dataclass
class Unit:
    start: float
    end: float
    text: str = ''
    speaker: str | None = None
    granularity: str = 'segment'
    method: str = 'native_interval'


@dataclass
class Result:
    units: list[Unit]
    raw: object
    text: str = ''
    warnings: list[str] | None = None


def executable(runtime: str, device: str, custom: str = '') -> Path:
    if custom:
        path = Path(custom).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'Executable not found: {path}')
        return path
    name = {'whisper_cpp': 'whisper-cli', 'audio_cpp': 'audiocpp_cli'}[runtime]
    suffix = '.exe' if sys.platform == 'win32' else ''
    # Bundle folders are backend-specific; do not pretend a CPU binary is Vulkan capable.
    for root in [Path(sys.executable).parent / 'engines', assets_root() / 'engines']:
        for directory in [root / f'{runtime}-{device}', root / runtime]:
            if directory.exists():
                matches = sorted(directory.rglob(name + suffix))
                if matches:
                    return matches[0]
    found = shutil.which(name)
    if found:
        return Path(found)
    raise FileNotFoundError(f'{runtime} ({device}) is not installed. Select its executable in Runtime settings.')


def ffmpeg_path(custom: str = '') -> str:
    if custom:
        if not Path(custom).is_file():
            raise FileNotFoundError(custom)
        return custom
    for root in [Path(sys.executable).parent / 'engines', assets_root() / 'engines']:
        if root.exists():
            for pattern in ('ffmpeg.exe', 'ffmpeg', 'ffmpeg-*.exe'):
                found = list(root.rglob(pattern))
                if found:
                    return str(found[0])
    found = shutil.which('ffmpeg')
    if found:
        return found
    raise FileNotFoundError('FFmpeg not found. Select ffmpeg in Runtime settings.')


def run_process(argv: list[str], cancel: threading.Event, progress, log: Path,
                timeout: float = 7200) -> None:
    checkpoint(cancel)
    env = os.environ.copy()
    # PyInstaller's library search path must not leak into external FFmpeg/engines.
    if 'LD_LIBRARY_PATH_ORIG' in env:
        env['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH_ORIG']
    else:
        env.pop('LD_LIBRARY_PATH', None)
    env['OMP_NUM_THREADS'] = env.get('ASR2RPP_THREADS', '4')
    library = Path(argv[0]).parent.parent / 'lib'
    if library.exists() and sys.platform.startswith('linux'):
        env['LD_LIBRARY_PATH'] = str(library)
    log.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {'start_new_session': True}
    started = time.monotonic()
    with log.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps({'command': argv, 'device_verification': 'requested; inspect engine log'}, ensure_ascii=False) + '\n')
        with subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, env=env, **kwargs) as process:
            lines = queue.Queue()
            def reader():
                for line in iter(process.stdout.readline, b''):
                    lines.put(line.decode('utf-8', errors='replace'))
            thread = threading.Thread(target=reader, daemon=True)
            thread.start()
            try:
                while process.poll() is None or thread.is_alive() or not lines.empty():
                    checkpoint(cancel)
                    if time.monotonic() - started > timeout:
                        raise TimeoutError('Engine timeout exceeded')
                    try:
                        line = lines.get(timeout=0.1)
                        handle.write(line)
                        handle.flush()
                        if line.strip():
                            progress(line.strip()[-400:])
                    except queue.Empty:
                        pass
                if process.returncode:
                    raise RuntimeError(f'{Path(argv[0]).name} exited {process.returncode}. See {log}')
            finally:
                if process.poll() is None:
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
                    else:
                        import signal
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        if os.name != 'nt':
                            os.killpg(process.pid, signal.SIGKILL)
                        else:
                            process.kill()
                        process.wait()
                thread.join(timeout=2)


def number(value) -> float:
    if isinstance(value, bool):
        raise ValueError('Boolean timestamp')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Non-finite timestamp')
    return result


def parse_whisper(data: dict) -> Result:
    units = []
    for segment in data.get('transcription', []):
        offsets = segment.get('offsets')
        if not offsets:
            raise ValueError('whisper.cpp JSON is missing millisecond offsets')
        text = str(segment.get('text', '')).strip()
        if text:
            units.append(Unit(number(offsets['from']) / 1000,
                              number(offsets['to']) / 1000, text))
    return Result(units, data, ''.join(u.text for u in units), [])


def parse_audio(data, task: str, family: str) -> Result:
    keys = {'asr': ('segments', 'words', 'speaker_turns', 'turns'),
            'diar': ('speaker_turns', 'turns', 'segments'), 'align': ('words', 'segments')}[task]
    records = data
    detected = ''
    if isinstance(data, dict):
        for key in keys:
            if isinstance(data.get(key), list):
                records, detected = data[key], key
                break
    if not isinstance(records, list):
        raise ValueError('Unrecognized audio.cpp JSON; expected a timed list. Raw output retained.')
    units = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError('Timed entry must be an object')
        def pick(*names):
            for name in names:
                if name in record:
                    return record[name]
            raise ValueError(f'Missing {names[0]} in timed entry: {list(record)}')
        start = number(pick('start', 'start_time', 'start_sec', 'start_seconds'))
        end = number(pick('end', 'end_time', 'end_sec', 'end_seconds'))
        text = str(record.get('text', record.get('word', record.get('token', record.get('content', '')))))
        speaker = record.get('speaker', record.get('speaker_id', record.get('speaker_label')))
        granularity = 'word' if task == 'align' or detected == 'words' else 'segment'
        method = 'forced_alignment' if task == 'align' else 'native_interval'
        if family == 'nemotron_asr':
            granularity, method = 'token', 'emission_frame'
        if task == 'diar':
            granularity, method = 'speaker_turn', 'speaker_activity'
        units.append(Unit(start, end, text, None if speaker is None else str(speaker), granularity, method))
    return Result(units, data, ''.join(u.text for u in units), [])


def scalar(value) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if not isinstance(value, (int, float, str)):
        raise ValueError('Engine parameters must be scalar values')
    return str(value)


def infer(model: Model, weights: Path, audio: Path, work: Path, options: dict,
          cancel: threading.Event, progress, transcript: str = '') -> Result:
    work.mkdir(parents=True, exist_ok=True)
    device = options.get('device', 'cpu')
    binary = executable(model.runtime, device, options.get('executable', ''))
    language = options.get('language', model.defaults.get('language', 'ja'))
    threads = int(options.get('threads', 4))
    if not 1 <= threads <= 128:
        raise ValueError('Threads must be 1..128')
    if model.runtime == 'whisper_cpp':
        prefix = work / 'asr'
        argv = [str(binary), '-m', str(weights), '-f', str(audio), '-l', language,
                '-t', str(threads), '-ojf', '-of', str(prefix), '-np']
        if device == 'cpu':
            argv.append('-ng')
        elif device not in {'auto', 'vulkan', 'cuda', 'metal'}:
            raise ValueError('Unsupported Whisper device')
        allowed = {'beam_size': '-bs', 'temperature': '-tp', 'no_speech_thold': '-nth'}
        for key, value in options.get('parameters', {}).items():
            if key not in allowed:
                raise ValueError(f'Unsupported Whisper parameter: {key}')
            argv += [allowed[key], scalar(value)]
        run_process(argv, cancel, progress, work / 'engine.log')
        result = parse_whisper(json.loads(prefix.with_suffix('.json').read_text(encoding='utf-8-sig')))
    else:
        output = work / 'timed.json'
        argv = [str(binary), '--task', model.task, '--family', model.family,
                '--model', str(weights), '--backend', 'best' if device == 'auto' else device,
                '--mode', 'offline', '--audio', str(audio), '--threads', str(threads)]
        if model.task == 'diar':
            argv += ['--turns-out', str(output)]
        else:
            argv += ['--language', language]
            if model.task == 'align':
                argv += ['--text', transcript, '--words-out', str(output)]
            elif model.family == 'vibevoice_asr':
                argv += ['--segments-out', str(output), '--text-out', str(work / 'text.txt')]
            else:
                argv += ['--words-out', str(output), '--text-out', str(work / 'text.txt')]
        parameters = {**model.defaults.get('request', {}), **options.get('parameters', {})}
        for key, value in parameters.items():
            if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_.]*', key):
                raise ValueError('Invalid parameter name')
            argv += ['--request-option', f'{key}={scalar(value)}']
        for key, value in model.defaults.get('session', {}).items():
            argv += ['--session-option', f'{key}={scalar(value)}']
        run_process(argv, cancel, progress, work / 'engine.log')
        if not output.exists():
            raise ValueError('Engine did not produce timestamps. Use a supported timed model; no times were fabricated.')
        result = parse_audio(json.loads(output.read_text(encoding='utf-8-sig')), model.task, model.family)
    (work / 'normalized.json').write_text(json.dumps([asdict(u) for u in result.units], ensure_ascii=False, indent=2), encoding='utf-8')
    return result
