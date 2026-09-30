"""Isolated ASR worker, protocol 1. Fixed providers, local assets, no remote code."""
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
import argparse
import json
import math
from numbers import Real
import os
import sys
import time
import wave


_DLL_HANDLES = []


def configure_libraries():
    """Keep Windows dependency search local to this isolated worker."""
    if os.name != 'nt':
        return
    roots = [Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent)),
             Path(sys.executable).parent.parent/'cuda_runtime']
    try:
        dist = metadata.distribution('nvidia-cudnn-cu12')
        roots.append(Path(dist.locate_file('nvidia/cudnn/bin')))
    except metadata.PackageNotFoundError:
        pass
    if os.getenv('CUDA_PATH'):
        roots.append(Path(os.environ['CUDA_PATH'])/'bin')
    for root in list(roots):
        roots.append(root/'nvidia/cudnn/bin')
    directories = list(dict.fromkeys(str(p.resolve()) for p in roots if p.is_dir()))
    for directory in directories:
        _DLL_HANDLES.append(os.add_dll_directory(directory))
    os.environ['PATH'] = os.pathsep.join(directories + [os.environ.get('PATH','')])


def version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def capabilities():
    loaders = {}
    if version('sherpa-onnx'):
        # PyPI's standard wheel is CPU-only. A GPU-specific runtime can implement
        # the same protocol and advertise only devices it actually supports.
        for family in ('transducer', 'sense_voice', 'paraformer', 'ctc'):
            loaders['sherpa_onnx/' + family] = dict(tasks={'asr':['offline']},
                devices=['cpu'], timestamps=['none'], version=version('sherpa-onnx'))
    if version('faster-whisper'):
        import ctranslate2
        devices = ['cpu']
        try:
            if ctranslate2.get_cuda_device_count() > 0:
                devices.append('cuda')
        except RuntimeError:
            pass
        loaders['faster_whisper/whisper'] = dict(tasks={'asr':['offline']}, devices=devices,
            timestamps=['segment','word'], version=version('faster-whisper'))
    return dict(schema_version=1, protocol='asr2rpp-worker-jsonl', loaders=loaders)


def read_audio(path):
    import numpy as np
    with wave.open(str(path), 'rb') as handle:
        if handle.getsampwidth() != 2 or handle.getnchannels() != 1:
            raise ValueError('Worker input must be mono PCM16 WAV')
        rate = handle.getframerate()
        size = handle.getnframes()
        if size > 16000 * 60 * 60 * 8:
            raise ValueError('Audio exceeds worker resource limit')
        data = handle.readframes(size)
        if len(data) != size * 2:
            raise ValueError('Truncated worker audio')
    return np.frombuffer(data, dtype='<i2').astype(np.float32) / 32768.0, rate


def load_sherpa(config):
    import sherpa_onnx
    family = config['family']
    files = config['artifacts']
    options = dict(config['parameters'])
    # Constructor selection is code-owned, never a module/class from a TOML.
    constructors = {
        'transducer': (sherpa_onnx.OfflineRecognizer.from_transducer, ('encoder','decoder','joiner','tokens')),
        'sense_voice': (sherpa_onnx.OfflineRecognizer.from_sense_voice, ('model','tokens')),
        'paraformer': (sherpa_onnx.OfflineRecognizer.from_paraformer, ('paraformer','tokens')),
        'ctc': (sherpa_onnx.OfflineRecognizer.from_nemo_ctc, ('model','tokens')),
    }
    if family not in constructors:
        raise ValueError('Unsupported sherpa-onnx model family')
    constructor, required = constructors[family]
    kwargs = {key:files[key] for key in required}
    # Reject filesystem arguments in request options. Auxiliary files must be
    # declared and resolved through the model's artifact table.
    allowed = {'feature_dim','dither','decoding_method','max_active_paths','hotwords_score',
               'blank_penalty','modeling_unit','model_type','lm_scale','lodr_scale','use_itn'}
    for key, value in options.items():
        if key not in allowed:
            raise ValueError('Unsupported sherpa-onnx parameter: ' + key)
        kwargs[key] = value
    for role in ('bpe_vocab','hotwords_file','rule_fsts','rule_fars','lm','lodr_fst'):
        if role in files:
            kwargs[role] = files[role]
    kwargs.update(num_threads=config['threads'], sample_rate=config['sample_rate'], provider='cpu')
    if family == 'sense_voice':
        language = config.get('language', '')
        kwargs['language'] = '' if language == 'auto' else language
    return constructor(**kwargs), {'provider':'sherpa_onnx','device':'cpu','version':version('sherpa-onnx')}


def load_faster(config):
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    options = dict(config['parameters'])
    compute = options.pop('compute_type', 'float16' if config['device']=='cuda' else 'int8')
    batch = options.pop('batch_size', 1)
    if type(batch) is not int or not 1 <= batch <= 128:
        raise ValueError('batch_size must be 1..128')
    device_index = options.pop('device_index', 0)
    if type(device_index) is not int or device_index < 0:
        raise ValueError('device_index must be a nonnegative integer')
    model = WhisperModel(config['weights'], device=config['device'], compute_type=compute,
                         device_index=device_index, cpu_threads=config['threads'], local_files_only=True)
    runner = BatchedInferencePipeline(model=model) if batch > 1 else model
    return (runner, batch, options), {'provider':'faster_whisper','device':model.model.device,
            'compute_type':model.model.compute_type, 'version':version('faster-whisper'), 'batch_size':batch}



def faster_units(segment):
    """Keep words only when native boundaries are complete; never invent ends."""
    from .domain import Unit
    words = segment.words or []
    valid = lambda start,end: (isinstance(start,Real) and not isinstance(start,bool) and isinstance(end,Real) and not isinstance(end,bool)
                               and math.isfinite(start) and math.isfinite(end)
                               and 0 <= start < end)
    if words and all(valid(w.start,w.end) for w in words):
        return [Unit(float(w.start),float(w.end),w.word,granularity='word') for w in words]
    if valid(segment.start,segment.end):
        return [Unit(float(segment.start),float(segment.end),segment.text)]
    raise ValueError('Native ASR returned no complete interval for a text segment')


def run(config, requests, output):
    from .domain import Unit
    provider = config['provider']
    requested_device = config['device']
    available = capabilities()['loaders'].get(provider + '/' + config['family'])
    if not available:
        raise ValueError('Provider/family is not present in this runtime')
    device = requested_device
    if device == 'auto':
        device = 'cuda' if 'cuda' in available['devices'] else 'cpu'
    if device not in available['devices']:
        raise ValueError('Requested device is not supported by this worker: ' + device)
    config = dict(config, device=device)
    started = time.monotonic()
    if provider == 'sherpa_onnx':
        model, runtime = load_sherpa(config)
    elif provider == 'faster_whisper':
        model, runtime = load_faster(config)
    else:
        raise ValueError('This built-in worker cannot load the requested provider')
    load_seconds = time.monotonic() - started
    print(json.dumps({'event':'loaded', 'runtime':runtime, 'seconds':load_seconds}), file=sys.stderr, flush=True)
    for request in requests:
        started = time.monotonic()
        try:
            samples, rate = read_audio(Path(request['audio']))
            if rate != config['sample_rate']:
                raise ValueError('Worker input sample rate does not match model')
            units = []
            raw = {}
            if provider == 'sherpa_onnx':
                stream = model.create_stream()
                stream.accept_waveform(rate, samples)
                model.decode_stream(stream)
                result = stream.result
                text = result.text
                # Token emission start times are not complete speech intervals.
                # Preserve them as raw evidence, never invent token end times.
                raw = {'tokens':list(result.tokens), 'emission_starts':list(result.timestamps)}
            else:
                runner, batch, options = model
                args = dict(options)
                args.setdefault('language', None if config.get('language') == 'auto' else config.get('language'))
                args.setdefault('word_timestamps', not config.get('text_only', False))
                if config.get('text_only'):
                    args['condition_on_previous_text'] = False
                    args['without_timestamps'] = True
                    args['vad_filter'] = False
                if batch > 1:
                    args['batch_size'] = batch
                segments, info = runner.transcribe(samples, **args)
                texts = []
                for segment in segments:
                    texts.append(segment.text)
                    if not config.get('text_only'):
                        units.extend(faster_units(segment))
                text = ''.join(texts).strip()
                raw = {'language':info.language, 'language_probability':info.language_probability}
                if config.get('text_only'):
                    units = []
            record = dict(schema_version=1, id=request['id'], status='ok', text=text,
                          units=[asdict(u) for u in units], raw=raw, time_origin='request', runtime=runtime,
                          metrics={'model_load_seconds':load_seconds,'request_seconds':time.monotonic()-started})
        except Exception as exc:
            record = dict(schema_version=1, id=request['id'], status='error', error=str(exc), runtime=runtime)
        output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
        output.flush()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capabilities', action='store_true')
    parser.add_argument('--request')
    parser.add_argument('--output')
    args = parser.parse_args(argv)
    configure_libraries()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    if args.capabilities:
        print(json.dumps(capabilities()))
        return 0
    if not args.request or not args.output:
        parser.error('--request and --output are required')
    request_file = Path(args.request)
    if request_file.stat().st_size > 16*1024**2:
        raise ValueError('Worker request exceeds limit')
    payload = json.loads(request_file.read_text(encoding='utf-8'))
    if payload.get('schema_version') != 1:
        raise ValueError('Unsupported worker request schema')
    config = payload['model']
    if type(config.get('threads')) is not int or not 1 <= config['threads'] <= 128:
        raise ValueError('Invalid worker thread count')
    if config.get('sample_rate') not in (16000, 24000, 44100, 48000):
        raise ValueError('Invalid worker sample rate')
    requests = payload['requests']
    if not isinstance(requests, list) or len(requests) > 4096:
        raise ValueError('Invalid worker request list')
    ids = [r['id'] for r in requests]
    if any(not isinstance(i,str) or not i.isascii() or not i.replace('_','').isalnum() for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Worker request IDs must be unique ASCII identifiers')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as output:
        try:
            run(config, requests, output)
        except Exception as exc:
            for request in requests:
                output.write(json.dumps(dict(schema_version=1,id=request['id'],status='error',error=str(exc)))+'\n')
            print(str(exc), file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
