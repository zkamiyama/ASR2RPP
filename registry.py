"""Data-only model registry and integrity-checked Hugging Face file downloads."""
import json
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from common import app_dir, read_json, write_json, digest

ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,99}$')
ENGINES = {'whisper.cpp', 'audio.cpp'}
DEVICES = {'cpu', 'cuda', 'vulkan', 'metal'}


def merge(a, b):
    result = dict(a)
    for k, v in b.items():
        result[k] = merge(result[k], v) if isinstance(v, dict) and isinstance(result.get(k), dict) else v
    return result


class HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).scheme != 'https':
            raise ValueError('Refusing a non-HTTPS model download redirect')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Registry:
    def __init__(self, path=None):
        self.path = Path(path or app_dir() / 'models.json').resolve()
        self.root = self.path.parent
        self.data = read_json(self.path)
        user = self.path.with_name(self.path.stem + '.user.json')
        if user.exists():
            extension = read_json(user)
            if extension.get('schema_version') != 1:
                raise ValueError('User registry schema_version must be 1')
            by_id = {m['id']: m for m in self.data.get('models', [])}
            for model in extension.get('models', []):
                by_id[model['id']] = model
            self.data = merge(self.data, {k: v for k, v in extension.items() if k != 'models'})
            self.data['models'] = list(by_id.values())
        if self.data.get('schema_version') != 1:
            raise ValueError('Unsupported registry schema_version')
        self.models = {}
        for model in self.data.get('models', []):
            mid = model.get('id', '')
            if not ID.fullmatch(mid) or mid in self.models:
                raise ValueError(f'Invalid or duplicate model id: {mid}')
            if model.get('type') not in {'asr', 'diarization', 'joint'}:
                raise ValueError(f'{mid}: type must be asr, diarization or joint')
            if model.get('engine') not in ENGINES:
                raise ValueError(f'{mid}: unsupported engine; a new engine needs a code adapter')
            if not isinstance(model.get('path'), str) or not model['path']:
                raise ValueError(f'{mid}: a local model path is required')
            if model['engine'] == 'whisper.cpp' and (model['type'] != 'asr' or model.get('format') != 'ggml'):
                raise ValueError(f'{mid}: whisper.cpp requires type=asr and format=ggml')
            if model['engine'] == 'audio.cpp':
                if not ID.fullmatch(model.get('family', '')) or model.get('format') != 'gguf':
                    raise ValueError(f'{mid}: audio.cpp requires family and format=gguf')
            if model.get('sample_rate', 16000) not in (16000, 24000, 48000):
                raise ValueError(f'{mid}: unsupported sample rate')
            self.models[mid] = model

    def model(self, mid):
        if mid not in self.models:
            raise ValueError(f'Unknown model: {mid}')
        return self.models[mid]

    def model_path(self, model):
        p = Path(model['path']).expanduser()
        return p.resolve() if p.is_absolute() else (self.root / p).resolve()

    def executable(self, engine, device):
        if device not in DEVICES:
            raise ValueError(f'Unknown device: {device}')
        setting = self.data.get('engines', {}).get(engine, {}).get(device)
        if not setting:
            raise ValueError(f'{engine}/{device} is not configured; edit models.user.json')
        return self.tool(setting)

    def tool(self, setting):
        p = Path(setting).expanduser()
        candidate = p if p.is_absolute() else self.root / p
        if candidate.is_file():
            return str(candidate.resolve())
        resolved = shutil.which(setting)
        if resolved:
            return resolved
        raise FileNotFoundError(f'Executable not found: {setting}')

    def install(self, mid, log=print):
        model = self.model(mid)
        spec = model.get('artifact')
        if not spec:
            raise ValueError('This entry needs a local converted model; no downloadable artifact is defined')
        repo, filename = spec['repo_id'], spec['filename']
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo):
            raise ValueError('Invalid Hugging Face repo_id')
        if PurePosixPath(filename).is_absolute() or '..' in PurePosixPath(filename).parts or '\\' in filename:
            raise ValueError('Unsafe artifact filename')
        revision = urllib.parse.quote(spec.get('revision', 'main'), safe='')
        url = f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true'
        log(f'Resolving model {mid} ({repo}); large downloads are not bundled in the app.')
        opener = urllib.request.build_opener(HttpsOnlyRedirect())
        with opener.open(url, timeout=30) as response:
            info = json.load(response)
        commit = info['sha']
        if not re.fullmatch(r'[0-9a-f]{40,64}', commit):
            raise ValueError('Invalid resolved model revision')
        entry = next((f for f in info['siblings'] if f['rfilename'] == filename), None)
        if not entry:
            raise ValueError(f'Artifact not present in repository: {filename}')
        expected = spec.get('sha256') or (entry.get('lfs') or {}).get('sha256')
        size = entry.get('size') or (entry.get('lfs') or {}).get('size')
        if not expected or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('No SHA-256 available. Add a verified artifact.sha256 to the registry.')
        destination = self.model_path(model)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) == expected:
                log('Model is already installed and its hash matches.')
                return
            raise FileExistsError('Existing model has a different hash; it was not overwritten')
        if not size or size > 32 * 1024**3:
            raise ValueError('Invalid model size or model exceeds the 32 GiB download limit')
        if shutil.disk_usage(destination.parent).free < size + 128 * 1024**2:
            raise OSError('Insufficient disk space')
        target = f'https://huggingface.co/{repo}/resolve/{commit}/{urllib.parse.quote(filename, safe="/")}'
        temporary = destination.with_name(destination.name + '.part')
        if temporary.exists():
            raise FileExistsError(f'An earlier partial download exists: {temporary}; remove it before retrying')
        received = 0
        try:
            with opener.open(target, timeout=60) as response, temporary.open('xb') as out:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    received += len(block)
                    if received > size:
                        raise ValueError('Downloaded data exceeds the declared size')
                    out.write(block)
                    if received % (64 * 1024**2) == 0:
                        log(f'{received / size:.0%} downloaded')
            if received != size or digest(temporary) != expected:
                raise ValueError('Model size/SHA-256 verification failed')
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        write_json(str(destination) + '.provenance.json',
                   dict(repo_id=repo, revision=commit, filename=filename, sha256=expected, size=size))
        log('Model installed; resolved revision and SHA-256 recorded.')
