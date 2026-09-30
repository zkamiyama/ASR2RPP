"""ASR provider boundary. The scheduler never selects a model-family code path."""
from dataclasses import dataclass, field
from pathlib import Path
import json
import shutil
from .domain import Result
from .catalog import checkpoint
from .native_profile import profile


@dataclass(frozen=True)
class Request:
    id: str
    audio: Path
    text: str = ''


@dataclass
class Batch:
    results: dict[str, Result] = field(default_factory=dict)
    errors: dict[str, Exception] = field(default_factory=dict)


def preflight(model, stage, cancel=None, *, segmented=False):
    from .runtime_registry import probe, WORKER_PROVIDERS
    info = probe(model.runtime, stage.device, stage.executable, cancel)
    loaders = info['capabilities']['loaders']
    family = ('whisper' if model.runtime == 'whisper_cpp' else
              model.runtime + '/' + model.family if model.runtime in WORKER_PROVIDERS and model.runtime != 'external_json'
              else model.family)
    loader = loaders.get(family)
    if not isinstance(loader, dict):
        raise ValueError(f'{model.label}: family {model.family} is not compiled into the selected runtime')
    modes = loader.get('tasks', {}).get(model.task, [])
    mode = 'offline' if segmented or model.runtime == 'whisper_cpp' else profile(model).mode
    if mode not in modes:
        raise ValueError(f'{model.label}: selected runtime does not support {model.task}/{mode}')
    devices = loader.get('devices')
    if devices and stage.device != 'auto' and stage.device not in devices:
        raise ValueError(f'{model.label}: runtime does not support device {stage.device}; available: {devices}')
    timing = loader.get('timestamps')
    if not segmented and model.task == 'asr' and timing == ['none'] and model.capabilities.get('timestamps') != 'none':
        raise ValueError('The provider returns no complete intervals; declare timestamps="none" and use VAD/alignment')
    return info


def _save(batch, requests, work, log=None):
    from dataclasses import asdict
    work = Path(work)
    for req in requests:
        target = work / req.id
        target.mkdir(parents=True, exist_ok=True)
        if log is not None and log.is_file():
            shutil.copy2(log, target / log.name)
        result = batch.results.get(req.id)
        if result is not None:
            (target/'raw.json').write_text(json.dumps(result.raw, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            (target/'normalized.json').write_text(json.dumps([asdict(u) for u in result.units], ensure_ascii=False), encoding='utf-8')


class WhisperProvider:
    native_regions = True
    def infer_many(self, model, weights, requests, work, options, cancel, progress, max_bytes):
        from . import native_batches
        return native_batches.whisper_many(model, weights, requests, work, options, cancel, progress, max_bytes)


class AudioProvider:
    native_regions = False
    def infer_many(self, model, weights, requests, work, options, cancel, progress, max_bytes):
        from . import native_batches
        return native_batches.audio_many(model, weights, requests, work, options, cancel, progress, max_bytes)


class WorkerProvider:
    native_regions = False
    def infer_many(self, model, weights, requests, work, options, cancel, progress, max_bytes):
        from .alignment import chunks_by_size
        from .worker_client import infer_batch
        batch = Batch()
        work = Path(work)
        maximum = profile(model).max_batch_items
        for number, group in enumerate(chunks_by_size(requests, lambda r:r.audio, max_bytes)):
            for offset in range(0, len(group), maximum):
                checkpoint(cancel)
                chunk = group[offset:offset+maximum]
                directory = work / f'chunk-{number}-{offset}'
                try:
                    result, failed = infer_batch(model, weights,
                        [dict(id=r.id, audio=r.audio) for r in chunk], directory,
                        options, cancel, progress, text_only=options.get('_text_only', False))
                    batch.results.update(result)
                    batch.errors.update(failed)
                except Exception as exc:
                    checkpoint(cancel)
                    batch.errors.update({r.id:exc for r in chunk})
                finally:
                    _save(batch, chunk, work, directory/'engine.log')
        return batch


_PROVIDERS = {'whisper_cpp':WhisperProvider(), 'audio_cpp':AudioProvider(),
              'sherpa_onnx':WorkerProvider(), 'faster_whisper':WorkerProvider(),
              'external_json':WorkerProvider()}


def provider_for(model):
    try:
        return _PROVIDERS[model.runtime]
    except KeyError as exc:
        raise ValueError('Unsupported ASR provider: ' + model.runtime) from exc
