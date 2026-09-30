"""Model definitions are data, never executable commands or Python imports."""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse, quote
from urllib.request import Request, urlopen
import hashlib
import json
import os
import shutil
import sys
import threading
import tomllib
from .atomic import file_lock, json_replace


class Cancelled(Exception):
    pass


from .performance import timed

def checkpoint(cancel: threading.Event) -> None:
    if cancel.is_set():
        raise Cancelled('Cancelled by user')


def data_root() -> Path:
    if os.getenv('ASR2RPP_HOME'):
        return Path(os.environ['ASR2RPP_HOME']).expanduser()
    if sys.platform == 'win32':
        return Path(os.getenv('LOCALAPPDATA', Path.home() / 'AppData/Local')) / 'ASR2RPP'
    if sys.platform == 'darwin':
        return Path.home() / 'Library/Application Support/ASR2RPP'
    return Path(os.getenv('XDG_DATA_HOME', Path.home() / '.local/share')) / 'asr2rpp'


def assets_root() -> Path:
    return Path(getattr(sys, '_MEIPASS', Path(__file__).resolve().parent.parent))


def _configured_root(env_name: str, fallback: Path) -> Path:
    value = os.getenv(env_name, '').strip()
    return Path(value).expanduser() if value else fallback


def weights_root() -> Path:
    return _configured_root('ASR2RPP_WEIGHTS_DIR', data_root() / 'weights')


def cache_root() -> Path:
    return _configured_root('ASR2RPP_CACHE_DIR', data_root() / 'cache')


def model_directory() -> Path:
    """Authoritative shipped definitions, read in place (never from _MEIPASS)."""
    root = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else assets_root()
    return root / 'models'


def custom_model_directory() -> Path:
    """Explicit user models, separate from shipped definitions; no auto-copy."""
    return data_root() / 'custom-models'


def _definition_files(directory):
    if directory is not None:
        directory = Path(directory)
        return [(p, 'explicit') for p in sorted(directory.glob('*.toml'))], []
    bundled = model_directory()
    if not bundled.is_dir():
        return [], [f'Missing model definitions: {bundled}. Extract the complete application ZIP.']
    files = [(p, 'bundled') for p in sorted(bundled.glob('*.toml'))]
    files += [(p, 'custom') for p in sorted(custom_model_directory().glob('*.toml'))]
    # Read old *custom* definitions in place for compatibility. Old shipped copies
    # must never silently shadow this version's authoritative bundled definition.
    files += [(p, 'legacy') for p in sorted((data_root() / 'models').glob('*.toml'))]
    return files, []


def _known_template_copy(file, bundled):
    fingerprint = hashlib.sha256(file.read_text(encoding='utf-8-sig').encode()).hexdigest()
    if bundled.is_file() and fingerprint == hashlib.sha256(bundled.read_text(encoding='utf-8-sig').encode()).hexdigest():
        return True
    migrations = model_directory() / 'template-migrations.json'
    known = json.loads(migrations.read_text(encoding='utf-8')) if migrations.is_file() else {}
    return fingerprint in known.get(file.name, [])


@dataclass(frozen=True)
class Model:
    id: str
    runtime: str
    task: str
    source: dict
    defaults: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    family: str = ''
    name: str = ''
    sample_rate: int = 16000
    description: str = ''
    definition: Path | None = None
    definition_sha256: str = ''
    capabilities: dict = field(default_factory=dict)
    schema_version: int = 1
    artifacts: dict = field(default_factory=dict)
    execution: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.name or self.id

    def validate(self):
        from .model_schema import PROVIDERS, validate_extensions
        validate_extensions(self)
        if not isinstance(self.capabilities, dict):
            raise ValueError('capabilities must be a table')
        if self.capabilities.get('timestamps', 'segment') not in {'none', 'token', 'word', 'segment'}:
            raise ValueError('capabilities.timestamps must be none, token, word or segment')
        if set(self.capabilities) - {'timestamps', 'speakers'}:
            raise ValueError('Unknown model capability')
        if 'speakers' in self.capabilities and type(self.capabilities['speakers']) is not bool:
            raise ValueError('capabilities.speakers must be boolean')
        if self.runtime not in PROVIDERS:
            raise ValueError('Unsupported provider: ' + str(self.runtime))
        if self.runtime in {'sherpa_onnx', 'faster_whisper', 'external_json'} and self.task != 'asr':
            raise ValueError('This provider currently supports ASR only')
        if self.task not in {'asr', 'diar', 'align', 'sep'}:
            raise ValueError('task must be asr, diar, align or sep')
        if self.runtime == 'whisper_cpp' and self.task != 'asr':
            raise ValueError('whisper_cpp only supports ASR in this adapter')
        if self.runtime == 'audio_cpp' and not self.family:
            raise ValueError('audio_cpp requires family for output interpretation')
        if self.sample_rate not in {16000, 24000, 44100, 48000}:
            raise ValueError('unsupported sample_rate')
        if 'path' not in self.source:
            repo_id(self.source.get('repo', ''))
            files = self.source.get('files', [])
            if not isinstance(files, list) or not files or any(not isinstance(f, str) for f in files):
                raise ValueError('source.files must list runtime-ready model assets')
            if len({f.casefold() for f in files}) != len(files):
                raise ValueError('Duplicate source filename')
            for name in files:
                if safe_relative(name) == '.':
                    raise ValueError('source.files must contain filenames, not directories')
            if 'entry' in self.source:
                safe_relative(str(self.source['entry']))
            recipe = self.source.get('convert')
            if recipe is not None:
                if not isinstance(recipe, dict):
                    raise ValueError('source.convert must be a TOML table')
                if recipe.get('kind') != 'mel_band_roformer_ckpt_to_gguf':
                    raise ValueError('unsupported source.convert.kind')
                if self.runtime != 'audio_cpp' or self.family != 'mel_band_roformer':
                    raise ValueError('Mel-Band conversion requires audio_cpp mel_band_roformer')
                for key in ('checkpoint', 'config', 'output'):
                    safe_relative(str(recipe.get(key, '')))
                if recipe.get('checkpoint') not in files or recipe.get('config') not in files:
                    raise ValueError('conversion checkpoint/config must be listed in source.files')
                if recipe.get('precision', 'f16') not in {'f16', 'q8_0'}:
                    raise ValueError('conversion precision must be f16 or q8_0')
        if self.defaults.get('request') and not isinstance(self.defaults['request'], dict):
            raise ValueError('defaults.request must be a TOML table')
        if not isinstance(self.constraints, dict):
            raise ValueError('constraints must be a TOML table')
        unknown_constraints = set(self.constraints) - {'disabled_parameters', 'inference'}
        if unknown_constraints:
            raise ValueError(f'Unknown constraints: {sorted(unknown_constraints)}')
        disabled = self.constraints.get('disabled_parameters', [])
        if not isinstance(disabled, list) or any(not isinstance(x, str) or not x.strip() for x in disabled):
            raise ValueError('constraints.disabled_parameters must be an array of non-empty strings')

        from .inference_policy import policy_for, constrain_parameters
        policy = policy_for(self)
        if self.disabled_parameters & set(policy.forced_parameters):
            raise ValueError('Disabled parameters conflict with required inference policy')
        constrain_parameters(self, self.defaults.get('request', {}))

    @property
    def disabled_parameters(self) -> frozenset[str]:
        return frozenset(self.constraints.get('disabled_parameters', []))


def safe_relative(value: str) -> str:
    if not isinstance(value, str) or not value or any(c in value for c in '\x00\r\n'):
        raise ValueError('Invalid relative filename')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError(f'Unsafe relative filename: {value}')
    return str(path)


def repo_id(value: str) -> str:
    if value.startswith('https://'):
        parsed = urlparse(value)
        if parsed.hostname != 'huggingface.co' or parsed.query or parsed.fragment:
            raise ValueError('source.repo must be a Hugging Face repository URL or owner/name')
        value = parsed.path.strip('/')
    parts = value.split('/')
    if len(parts) != 2 or not all(parts) or any(x in value for x in ('..', '\\', ':')):
        raise ValueError('Expected source.repo = "owner/name"')
    return value


def load_catalog(directory: Path | None = None) -> tuple[dict[str, Model], list[str]]:
    files, errors = _definition_files(directory)
    models, claimed = {}, {}
    for file, origin in files:
        try:
            key = file.stem.casefold()
            if key in claimed:
                previous = claimed[key]
                if origin == 'legacy' and _known_template_copy(file, previous):
                    continue
                raise ValueError(f'Duplicate model ID; {previous} is authoritative. '
                                 'Rename the custom TOML to a distinct model ID; it was not overwritten.')
            # A broken custom override must not cause silent fallback to another file.
            claimed[key] = file
            payload = file.read_bytes()
            data = tomllib.loads(payload.decode('utf-8-sig'))
            if 'provider' in data:
                if 'runtime' in data:
                    raise ValueError('Use provider or runtime, not both')
                data['runtime'] = data.pop('provider')
            if 'id' in data and data.pop('id') != file.stem:
                raise ValueError('Model id must match the TOML filename stem')
            allowed = {'runtime', 'task', 'source', 'defaults', 'constraints', 'family', 'name', 'sample_rate', 'description', 'capabilities', 'schema_version', 'artifacts', 'execution', 'parameters'}
            unknown = set(data) - allowed
            if unknown:
                raise ValueError(f'Unknown fields: {sorted(unknown)}')
            model = Model(id=file.stem, definition=file.resolve(),
                          definition_sha256=hashlib.sha256(payload).hexdigest(), **data)
            model.validate()
            models[model.id] = model
        except (ValueError, TypeError, OSError, AttributeError) as error:
            errors.append(f'{file}: {error}')
    return models, errors


def definition_provenance(model):
    return {'path': str(model.definition) if model.definition else None,
            'sha256': model.definition_sha256 or None}


@timed('sha256')
def digest(path: Path, cancel=None) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            if cancel is not None:
                checkpoint(cancel)
            hasher.update(block)
    return hasher.hexdigest()


def local_model_path(model):
    """Cheap path validation; checksum remains mandatory at actual resolution."""
    path = Path(model.source['path']).expanduser()
    if not path.is_absolute():
        path = (model.definition.parent if model.definition else model_directory()) / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f'Model file not found: {path}')
    return path


# installed.json is provenance, not proof of integrity. Verify once per process
# and invalidate memoization on any change to the actual file's stat identity.
_VERIFIED_FILES = {}


def verified_digest(path, cancel):
    stat = path.stat()
    key = (str(path.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
           stat.st_mtime_ns, stat.st_ctime_ns)
    if key not in _VERIFIED_FILES:
        sha = digest(path, cancel)
        after = path.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('Model changed during verification')
        if len(_VERIFIED_FILES) > 1024:
            _VERIFIED_FILES.clear()
        _VERIFIED_FILES[key] = sha
    return _VERIFIED_FILES[key]


def _installed_valid(model, directory, state, cancel):
    files = state.get('files', {})
    required = set(model.source['files'])
    recipe = model.source.get('convert') or {}
    entry = safe_relative(model.source.get('entry', model.source['files'][0]))
    target = directory / entry
    if (not isinstance(files, dict) or not required.issubset(files) or
            not target.exists() or not target.resolve().is_relative_to(directory.resolve())):
        return False
    for name, info in files.items():
        if info.get('retained', True) is False:
            if name != recipe.get('checkpoint'):
                return False
            continue
        path = directory / safe_relative(name)
        if not path.resolve().is_relative_to(directory.resolve()):
            return False
        if (not path.is_file() or path.stat().st_size != info.get('size') or
                verified_digest(path, cancel) != info.get('sha256')):
            return False
        expected = model.source.get('sha256', {}).get(name)
        if expected and info['sha256'] != expected:
            return False
    return target.is_dir() or entry in files


@timed('model_resolution')
def resolve_model(model: Model, cancel: threading.Event, progress, download: bool = False, keep_source: bool = False) -> tuple[Path, dict]:
    checkpoint(cancel)
    if 'path' in model.source:
        path = local_model_path(model)
        from .model_schema import artifact_paths
        artifact_paths(model, path)
        if path.is_file():
            files = {path.name: verified_digest(path, cancel)}
        else:
            names = sorted(set(model.source.get('files', [])) | set(model.artifacts.values()))
            if not names:
                names = sorted(str(f.relative_to(path)).replace('\\', '/') for f in path.rglob('*') if f.is_file())
            if not names or len(names) > 10000:
                raise ValueError('Model directory is empty or has too many files')
            files = {}
            for name in names:
                checkpoint(cancel)
                file = (path / safe_relative(name)).resolve()
                if not file.is_relative_to(path) or not file.is_file():
                    raise ValueError('Missing or unsafe local model asset')
                files[name] = verified_digest(file, cancel)
        sha = files[path.name] if path.is_file() else hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
        return path, {'local_path': str(path), 'sha256': sha, 'files': files,
                      'verification': 'sha256 (process-local stat-identity memoization)'}
    identity = hashlib.sha256(json.dumps(model.source, sort_keys=True).encode()).hexdigest()[:16]
    lock = weights_root() / model.id / (identity + '.lock')
    with file_lock(lock, cancel):
        return _resolve_remote_model(model, cancel, progress, download, keep_source)


def _resolve_remote_model(model, cancel, progress, download=False, keep_source=False):
    identity = hashlib.sha256(json.dumps(model.source, sort_keys=True).encode()).hexdigest()[:16]
    directory = weights_root() / model.id / identity
    state_file = directory / 'installed.json'
    if state_file.exists():
        try:
            state = json.loads(state_file.read_text(encoding='utf-8'))
            installed_valid = _installed_valid(model, directory, state, cancel)
        except (ValueError, TypeError, KeyError, AttributeError, OSError):
            state, installed_valid = {}, False
        entry = safe_relative(model.source.get('entry', model.source['files'][0]))
        recipe = model.source.get('convert') or {}
        checkpoint_name = str(recipe.get('checkpoint', ''))
        restore_source = bool(
            keep_source and checkpoint_name and
            state.get('files', {}).get(checkpoint_name, {}).get('retained') is False)
        if installed_valid and not restore_source:
            state['verification'] = 'sha256 (process-local stat-identity memoization)'
            return directory / entry, state
    if not download:
        raise FileNotFoundError(f'{model.id}: model not installed. Use Download models / models install first.')
    directory.mkdir(parents=True, exist_ok=True)
    repository = repo_id(model.source['repo'])
    revision = str(model.source.get('revision', 'main'))
    progress(f'Resolving {model.id}')
    api = f'https://huggingface.co/api/models/{repository}/revision/{quote(revision, safe="")}'
    with urlopen(Request(api, headers={'User-Agent': 'ASR2RPP/0.1'}), timeout=30) as response:
        revision = json.load(response)['sha']
    state = {'repo': repository, 'revision': revision, 'files': {}}
    for filename in model.source['files']:
        checkpoint(cancel)
        filename = safe_relative(filename)
        destination = directory / filename
        if not destination.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Model asset escapes its installation directory')
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = model.source.get('sha256', {}).get(filename)
        if destination.is_file() and expected and digest(destination) == expected:
            state['files'][filename] = {'sha256': expected, 'size': destination.stat().st_size, 'retained': True}
            progress(f'{model.id}: verified cached {filename}')
            continue
        partial = destination.with_name(destination.name + '.part')
        url = f'https://huggingface.co/{repository}/resolve/{revision}/{quote(filename, safe="/")}'
        try:
            with urlopen(Request(url, headers={'User-Agent': 'ASR2RPP/0.1'}), timeout=30) as response:
                total = int(response.headers.get('Content-Length', 0))
                if total and total + 256 * 1024 ** 2 > shutil.disk_usage(directory).free:
                    raise OSError('Insufficient disk space for model')
                done, notified = 0, -1
                downloaded_sha = hashlib.sha256()
                with partial.open('wb') as handle:
                    while block := response.read(1024 * 1024):
                        checkpoint(cancel)
                        handle.write(block)
                        downloaded_sha.update(block)
                        done += len(block)
                        bucket = done // (16 * 1024 ** 2)
                        if bucket != notified:
                            progress(f'{model.id}: {done / 1024**2:.0f} / {total / 1024**2:.0f} MiB')
                            notified = bucket
                if total and done != total:
                    raise OSError('Incomplete model download')
            sha = downloaded_sha.hexdigest()
            if expected and sha != expected:
                raise ValueError('Model SHA-256 mismatch')
            if partial.stat().st_size == 0:
                raise ValueError('Downloaded model asset is empty')
            partial.replace(destination)
            state['files'][filename] = {'sha256': sha, 'size': done, 'retained': True}
        finally:
            partial.unlink(missing_ok=True)
    if model.source.get('convert'):
        from .model_conversion import convert_model
        converted, conversion = convert_model(model, directory, assets_root(), cache_root(), cancel, progress)
        relative = str(converted.relative_to(directory)).replace('\\\\', '/')
        state['files'][relative] = {'sha256': digest(converted), 'size': converted.stat().st_size, 'retained': True}
        recipe = model.source['convert']
        checkpoint_name = safe_relative(str(recipe.get('checkpoint', '')))
        source_checkpoint = directory / checkpoint_name
        if checkpoint_name and source_checkpoint.is_file() and not keep_source:
            source_checkpoint.unlink()
            if checkpoint_name in state['files']:
                state['files'][checkpoint_name]['retained'] = False
                state['files'][checkpoint_name]['removed_after_conversion'] = True
            conversion['source_checkpoint_retained'] = False
        else:
            conversion['source_checkpoint_retained'] = source_checkpoint.is_file()
        state['conversion'] = conversion
    state['verification'] = 'sha256 at installation'
    json_replace(state_file, state)
    entry = safe_relative(model.source.get('entry', model.source['files'][0]))
    return directory / entry, state
