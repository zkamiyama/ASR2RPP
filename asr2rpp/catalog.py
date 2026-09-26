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


class Cancelled(Exception):
    pass


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
    directory = data_root() / 'models'
    directory.mkdir(parents=True, exist_ok=True)
    migration_file = assets_root() / 'models' / 'template-migrations.json'
    migrations = json.loads(migration_file.read_text(encoding='utf-8')) if migration_file.exists() else {}
    for template in (assets_root() / 'models').glob('*.toml'):
        dest = directory / template.name
        if not dest.exists():
            with dest.open('x', encoding='utf-8') as handle:
                handle.write(template.read_text(encoding='utf-8'))
        elif template.name in migrations:
            old = dest.read_text(encoding='utf-8-sig')
            fingerprint = hashlib.sha256(old.encode('utf-8')).hexdigest()
            if fingerprint in migrations[template.name]:
                backup = dest.with_name(dest.name + '.' + fingerprint[:12] + '.bak')
                if not backup.exists():
                    with backup.open('x', encoding='utf-8') as handle:
                        handle.write(old)
                import tempfile
                fd, name = tempfile.mkstemp(prefix='template-', suffix='.tmp', dir=directory)
                try:
                    with os.fdopen(fd, 'w', encoding='utf-8') as handle:
                        handle.write(template.read_text(encoding='utf-8'))
                    if dest.read_text(encoding='utf-8-sig') == old:
                        os.replace(name, dest)
                finally:
                    Path(name).unlink(missing_ok=True)
    return directory


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

    @property
    def label(self) -> str:
        return self.name or self.id

    def validate(self):
        if self.runtime not in {'whisper_cpp', 'audio_cpp'}:
            raise ValueError('runtime must be whisper_cpp or audio_cpp')
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
            if not files:
                raise ValueError('source.files must contain the runtime-ready weights')
            for name in files:
                safe_relative(name)
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
    models, errors = {}, []
    for file in sorted((directory or model_directory()).glob('*.toml')):
        try:
            data = tomllib.loads(file.read_text(encoding='utf-8-sig'))
            allowed = {'runtime', 'task', 'source', 'defaults', 'constraints', 'family', 'name', 'sample_rate', 'description'}
            unknown = set(data) - allowed
            if unknown:
                raise ValueError(f'Unknown fields: {sorted(unknown)}')
            model = Model(id=file.stem, definition=file, **data)
            model.validate()
            models[model.id] = model
        except (ValueError, TypeError, OSError) as error:
            errors.append(f'{file.name}: {error}')
    return models, errors


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def resolve_model(model: Model, cancel: threading.Event, progress, download: bool = False, keep_source: bool = False) -> tuple[Path, dict]:
    """Cache identity includes the complete source definition; edits cannot reuse stale weights."""
    checkpoint(cancel)
    if 'path' in model.source:
        path = Path(model.source['path']).expanduser()
        if not path.is_absolute():
            path = (model.definition.parent if model.definition else model_directory()) / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f'Model file not found: {path}')
        return path, {'local_path': str(path), 'sha256': digest(path) if path.is_file() else None}
    identity = hashlib.sha256(json.dumps(model.source, sort_keys=True).encode()).hexdigest()[:16]
    directory = weights_root() / model.id / identity
    state_file = directory / 'installed.json'
    if state_file.exists():
        state = json.loads(state_file.read_text(encoding='utf-8'))
        entry = safe_relative(model.source.get('entry', model.source['files'][0]))
        installed_valid = ((directory / entry).is_file() and
                all((info.get('retained', True) is False) or
                    ((directory / f).is_file() and (directory / f).stat().st_size == info['size'])
                    for f, info in state['files'].items()))
        recipe = model.source.get('convert') or {}
        checkpoint_name = str(recipe.get('checkpoint', ''))
        restore_source = bool(
            keep_source and checkpoint_name and
            state.get('files', {}).get(checkpoint_name, {}).get('retained') is False)
        if installed_valid and not restore_source:
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
                with partial.open('wb') as handle:
                    while block := response.read(1024 * 1024):
                        checkpoint(cancel)
                        handle.write(block)
                        done += len(block)
                        bucket = done // (16 * 1024 ** 2)
                        if bucket != notified:
                            progress(f'{model.id}: {done / 1024**2:.0f} / {total / 1024**2:.0f} MiB')
                            notified = bucket
                if total and done != total:
                    raise OSError('Incomplete model download')
            sha = digest(partial)
            if expected and sha != expected:
                raise ValueError('Model SHA-256 mismatch')
            if partial.stat().st_size < 1024:
                raise ValueError('Downloaded file is unexpectedly small; check repository/file name')
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
    state_file.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    entry = safe_relative(model.source.get('entry', model.source['files'][0]))
    return directory / entry, state
