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
from .catalog import Model, checkpoint, assets_root

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
    if device == 'auto':
        if sys.platform == 'win32':
            devices = ['vulkan', 'cpu']
        elif sys.platform == 'darwin':
            devices = ['metal', 'cpu']
        else:
            devices = ['cuda', 'vulkan', 'cpu']
    else:
        devices = [device]
    for root in [Path(sys.executable).parent / 'engines', assets_root() / 'engines']:
        directories = [root / f'{runtime}-{candidate}' for candidate in devices] + [root / runtime]
        for directory in directories:
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


def process_environment(binary: Path) -> dict:
    env = os.environ.copy()
    if 'LD_LIBRARY_PATH_ORIG' in env:
        env['LD_LIBRARY_PATH'] = env['LD_LIBRARY_PATH_ORIG']
    else:
        env.pop('LD_LIBRARY_PATH', None)
    env['OMP_NUM_THREADS'] = env.get('ASR2RPP_THREADS', '4')
    # Never add generic /usr/lib: it can contain incompatible system libraries.
    # Only explicitly recognize libraries belonging to a native speech executable.
    if sys.platform.startswith('linux') and binary.name in {'whisper-cli', 'audiocpp_cli', 'nemo-speech'}:
        for directory in (binary.resolve().parent, binary.resolve().parent.parent / 'lib'):
            if list(directory.glob('libggml*.so*')):
                env['LD_LIBRARY_PATH'] = str(directory)
                break
    return env


def run_process(argv: list[str], cancel: threading.Event, progress, log: Path,
                timeout: float = 7200) -> None:
    checkpoint(cancel)
    env = process_environment(Path(argv[0]))
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


def parse_audio(data, task: str, family: str, sample_rate: int | None = None) -> Result:
    keys = {'asr': ('segments', 'words', 'speaker_turns', 'turns'),
            'diar': ('speaker_turns', 'turns', 'segments'), 'align': ('words', 'segments')}[task]
    records, detected = data, ''
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
        if 'start_sample' in record or 'end_sample' in record:
            if sample_rate is None or sample_rate <= 0:
                raise ValueError('Sample timestamps require an explicit positive sample rate')
            start = number(pick('start_sample')) / sample_rate
            end = number(pick('end_sample')) / sample_rate
        else:
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
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Engine parameters must be finite')
    return str(value)



WHISPER_VALUE_OPTIONS = {
    'beam_size': '-bs',
    'best_of': '-bo',
    'audio_ctx': '-ac',
    'max_context': '-mc',
    'max_len': '-ml',
    'word_thold': '-wt',
    'entropy_thold': '-et',
    'logprob_thold': '-lpt',
    'no_speech_thold': '-nth',
    'temperature': '-tp',
    'temperature_inc': '-tpi',
    'initial_prompt': '--prompt',
}
WHISPER_FLAG_OPTIONS = {
    'split_on_word': '-sow',
    'no_fallback': '-nf',
    'translate': '-tr',
    'suppress_nst': '-sns',
    'carry_initial_prompt': '--carry-initial-prompt',
}


def whisper_parameter_args(parameters: dict) -> list[str]:
    args = []
    for key, value in parameters.items():
        if key in WHISPER_VALUE_OPTIONS:
            args += [WHISPER_VALUE_OPTIONS[key], scalar(value)]
        elif key in WHISPER_FLAG_OPTIONS:
            if not isinstance(value, bool):
                raise ValueError(f'Whisper flag {key} must be boolean')
            if value:
                args.append(WHISPER_FLAG_OPTIONS[key])
        else:
            raise ValueError(f'Unsupported Whisper parameter: {key}')
    return args


def split_engine_parameters(model: Model, overrides: dict | None) -> tuple[dict, dict]:
    request = dict(model.defaults.get('request', {}))
    session = dict(model.defaults.get('session', {}))
    for key, value in (overrides or {}).items():
        if key.startswith('session.'):
            session[key[len('session.'):]] = value
        else:
            request[key] = value
    return request, session


def audio_session_args(model: Model, session: dict) -> list[str]:
    args = []
    for key, value in session.items():
        if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_.]*', key):
            raise ValueError('Invalid audio.cpp session parameter name')
        option = key if '.' in key else f'{model.family}.{key}'
        args += ['--session-option', f'{option}={scalar(value)}']
    return args


def infer(model: Model, weights: Path, audio: Path, work: Path, options: dict,
          cancel: threading.Event, progress, transcript: str = '') -> Result:
    work.mkdir(parents=True, exist_ok=True)
    device = options.get('device', 'cpu')
    binary = executable(model.runtime, device, options.get('executable', ''))
    language = options.get('language', model.defaults.get('language', 'ja'))
    threads = int(options.get('threads', 4))
    if not 1 <= threads <= 128:
        raise ValueError('Threads must be 1..128')
    parameters, session_parameters = split_engine_parameters(model, options.get('parameters', {}))
    if model.runtime == 'whisper_cpp':
        prefix = work / 'asr'
        argv = [str(binary), '-m', str(weights), '-f', str(audio), '-l', language,
                '-t', str(threads), '-ojf', '-of', str(prefix), '-np']
        if device == 'cpu':
            argv.append('-ng')
        elif device not in {'auto', 'vulkan', 'cuda', 'metal'}:
            raise ValueError('Unsupported Whisper device')
        argv += whisper_parameter_args(parameters)
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
        for key, value in parameters.items():
            if not re.fullmatch(r'[a-zA-Z][a-zA-Z0-9_.]*', key):
                raise ValueError('Invalid parameter name')
            argv += ['--request-option', f'{key}={scalar(value)}']
        argv += audio_session_args(model, session_parameters)
        run_process(argv, cancel, progress, work / 'engine.log')
        if not output.exists():
            raise ValueError('Engine did not produce timestamps. Use a supported timed model; no times were fabricated.')
        result = parse_audio(json.loads(output.read_text(encoding='utf-8-sig')), model.task, model.family, model.sample_rate)
    (work / 'normalized.json').write_text(json.dumps([asdict(u) for u in result.units], ensure_ascii=False, indent=2), encoding='utf-8')
    return result
