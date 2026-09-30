"""User-selected runtimes and capabilities. Model TOMLs cannot register executables."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from .atomic import json_replace
from .model_schema import PROVIDERS

BACKENDS = ('auto', 'cpu', 'cuda', 'vulkan', 'metal')
WORKER_PROVIDERS = frozenset(('sherpa_onnx', 'faster_whisper', 'external_json'))
_PROBES = {}


def registry_path():
    from .catalog import data_root
    return data_root() / 'runtimes' / 'registered.json'


def entries():
    path = registry_path()
    values = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    if not isinstance(values, dict):
        raise ValueError('Invalid runtime registry')
    return values


def register(runtime, device, binary, *, trust=False):
    """Explicit host action, never called from model loading/downloads."""
    if not trust:
        raise ValueError('Registering a runtime permits executable code. Explicit trust is required.')
    if runtime not in PROVIDERS or device not in BACKENDS:
        raise ValueError('Invalid runtime or backend')
    from .catalog import digest
    from .atomic import file_lock
    binary = Path(binary).expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    path = registry_path()
    with file_lock(path.with_suffix('.lock'), threading.Event()):
        values = entries()
        key = runtime + ':' + device
        previous = values.get(key)
        values[key] = dict(path=str(binary), sha256=digest(binary), previous=previous)
        json_replace(path, values)
    _PROBES.clear()
    return values[key]


def rollback(runtime, device):
    from .atomic import file_lock
    path = registry_path()
    with file_lock(path.with_suffix('.lock'), threading.Event()):
        values = entries()
        key = runtime + ':' + device
        record = values.get(key, {})
        previous = record.get('previous')
        if previous is None:
            raise ValueError('No previous runtime is registered')
        values[key] = previous
        json_replace(path, values)
    _PROBES.clear()


def _registered(runtime, device):
    from .catalog import verified_digest
    value = entries().get(runtime + ':' + device)
    if value is None:
        return None
    path = Path(value['path'])
    if not path.is_file() or verified_digest(path, threading.Event()) != value.get('sha256'):
        raise ValueError('Registered runtime changed. Re-register it explicitly after verification.')
    return path


def resolve(runtime, device, custom=''):
    from .catalog import assets_root
    if runtime not in PROVIDERS or device not in BACKENDS:
        raise ValueError('Unsupported runtime or backend')
    if custom:
        path = Path(custom).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError('Executable not found: ' + str(path))
        return path
    devices = ([device] if device != 'auto' else
               ['metal', 'cpu'] if sys.platform == 'darwin' else ['cuda', 'vulkan', 'cpu'])
    if runtime in WORKER_PROVIDERS:
        devices = [device] + [d for d in devices if d != device]
    for candidate in devices:
        registered = _registered(runtime, candidate)
        if registered:
            return registered
    suffix = '.exe' if sys.platform == 'win32' else ''
    name = {'whisper_cpp':'whisper-cli', 'audio_cpp':'audiocpp_cli'}.get(runtime, 'asr2rpp-worker')
    for root in [Path(sys.executable).parent / 'engines', assets_root() / 'engines']:
        directories = [root / f'{runtime}-{candidate}' for candidate in devices]
        if runtime in WORKER_PROVIDERS and runtime != 'external_json':
            directories += [root / 'python_worker']
        directories += [root / runtime]
        for directory in directories:
            if directory.is_dir():
                matches = sorted(directory.rglob(name + suffix))
                if matches:
                    return matches[0]
    if runtime in ('sherpa_onnx', 'faster_whisper') and not getattr(sys, 'frozen', False):
        return Path(sys.executable)
    found = shutil.which(name) if runtime != 'external_json' else None
    if found:
        return Path(found)
    raise FileNotFoundError(f'{runtime} ({device}) is not installed. Select or register a trusted runtime.')


def command(runtime, binary):
    binary = Path(binary)
    if runtime in ('sherpa_onnx', 'faster_whisper') and not getattr(sys, 'frozen', False) and binary.resolve() == Path(sys.executable).resolve():
        return [str(binary), '-m', 'asr2rpp.provider_worker']
    return [str(binary)]


def probe(runtime, device='auto', custom='', cancel=None):
    """Probe the real executable; never infer model support from a filename."""
    from .adapters import process_environment
    from .catalog import verified_digest, checkpoint
    cancel = cancel or threading.Event()
    checkpoint(cancel)
    binary = resolve(runtime, device, custom)
    fingerprint = verified_digest(binary, cancel)
    key = (runtime, str(binary.resolve()), fingerprint)
    if key in _PROBES:
        return dict(_PROBES[key], requested_device=device)
    args = command(runtime, binary)
    args += ['--list-loaders', '--json'] if runtime == 'audio_cpp' else ['--help'] if runtime == 'whisper_cpp' else ['--capabilities']
    kwargs = {'creationflags':subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
    result = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='strict',
                            timeout=30, env=process_environment(binary), **kwargs)
    checkpoint(cancel)
    if result.returncode:
        raise RuntimeError(f'Runtime capability probe failed ({binary.name}): {result.stderr[-400:]}')
    if len(result.stdout) + len(result.stderr) > 2*1024**2:
        raise ValueError('Oversized runtime capabilities')
    if runtime == 'whisper_cpp':
        data = {'schema_version':1, 'loaders':{'whisper':{'tasks':{'asr':['offline']}}},
                'help':result.stdout + result.stderr}
    else:
        data = json.loads(result.stdout)
        if not isinstance(data, dict) or type(data.get('schema_version')) is not int or data.get('schema_version') != 1 or not isinstance(data.get('loaders'), dict):
            raise ValueError('Unsupported runtime capability schema')
    info = dict(runtime=runtime, binary=str(binary.resolve()), sha256=fingerprint, capabilities=data)
    _PROBES[key] = info
    return dict(info, requested_device=device)
