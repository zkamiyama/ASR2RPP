"""Torch-free local model conversion helpers.

Only explicitly supported conversion recipes are accepted.  In particular the
Mel-Band RoFormer converter reads a plain PyTorch state-dict zip without
importing torch, writes SafeTensors, then invokes audio.cpp's native GGUF
converter.  The source checkpoint is never executed as Python code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
import threading
import time
import zipfile


class ConversionError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorRef:
    storage_key: str
    dtype_name: str
    offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


class RestrictedCheckpointUnpickler(pickle.Unpickler):
    """Decode the tiny state-dict pickle surface used by big_beta7.ckpt.

    Any global outside the exact allowlist is rejected.  This is deliberately
    stricter than torch.load and stricter than a generic pickle loader.
    """

    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return dict
        if module == 'torch' and name == 'FloatStorage':
            return type('FloatStorage', (), {'__name__': 'FloatStorage'})
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            def rebuild(storage, offset=0, size=None, stride=None, *_rest):
                if not isinstance(storage, tuple) or len(storage) < 3 or storage[0] != 'storage':
                    raise ConversionError('Unsupported tensor storage reference')
                storage_type, key = storage[1], storage[2]
                dtype_name = getattr(storage_type, '__name__', '')
                if dtype_name != 'FloatStorage':
                    raise ConversionError(f'Unsupported storage type: {dtype_name}')
                return TensorRef(str(key), dtype_name, int(offset),
                                 tuple(int(v) for v in (size or ())),
                                 tuple(int(v) for v in (stride or ())))
            return rebuild
        raise ConversionError(f'Checkpoint requires disallowed pickle global: {module}.{name}')

    def persistent_load(self, pid):
        if not isinstance(pid, tuple) or len(pid) < 5 or pid[0] != 'storage':
            raise ConversionError('Unsupported checkpoint persistent id')
        storage_type, key, location, numel = pid[1], pid[2], pid[3], pid[4]
        if getattr(storage_type, '__name__', '') != 'FloatStorage':
            raise ConversionError('big_beta7 conversion accepts FloatStorage only')
        if str(location) not in {'cpu'} and not str(location).startswith('cuda'):
            raise ConversionError(f'Unsupported checkpoint storage location: {location}')
        if int(numel) < 0:
            raise ConversionError('Negative storage size')
        return ('storage', storage_type, str(key), str(location), int(numel))


def _checkpoint_payload(path: Path):
    if not zipfile.is_zipfile(path):
        raise ConversionError('Checkpoint is not a modern PyTorch ZIP archive')
    archive = zipfile.ZipFile(path)
    pickle_names = [n for n in archive.namelist() if n.endswith('/data.pkl') or n == 'data.pkl']
    if len(pickle_names) != 1:
        archive.close()
        raise ConversionError(f'Expected exactly one data.pkl, found {pickle_names}')
    pickle_name = pickle_names[0]
    root = pickle_name.rsplit('/', 1)[0] if '/' in pickle_name else ''
    try:
        payload = RestrictedCheckpointUnpickler(io.BytesIO(archive.read(pickle_name))).load()
    except BaseException:
        archive.close()
        raise
    if not isinstance(payload, dict) or not payload:
        archive.close()
        raise ConversionError('Checkpoint does not contain a non-empty state dict')
    for name, value in payload.items():
        if not isinstance(name, str) or not isinstance(value, TensorRef):
            archive.close()
            raise ConversionError(f'Unexpected non-tensor state entry: {name!r}')
    return archive, root, payload


def _parse_scalar(value: str):
    value = value.strip()
    if not value:
        return None
    low = value.lower()
    if low in {'true', 'false'}:
        return low == 'true'
    if low in {'null', 'none', '~'}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value, 10)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parse_big_beta7_yaml(path: Path) -> dict:
    """Read only scalar fields needed by audio.cpp; ignore training-only tags/lists."""
    sections: dict[str, dict[str, object]] = {}
    current = None
    for raw in path.read_text(encoding='utf-8-sig').splitlines():
        line = raw.split('#', 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(' '))
        stripped = line.strip()
        if indent == 0 and stripped.endswith(':'):
            current = stripped[:-1]
            sections.setdefault(current, {})
            continue
        if indent > 0 and current and ':' in stripped:
            key, value = stripped.split(':', 1)
            value = value.strip()
            if value and not value.startswith('!!'):
                sections[current][key.strip()] = _parse_scalar(value)
    audio, model, inference = (sections.get(name, {}) for name in ('audio', 'model', 'inference'))
    required = {
        'audio.chunk_size': audio.get('chunk_size'),
        'audio.sample_rate': audio.get('sample_rate'),
        'audio.n_fft': audio.get('n_fft'),
        'audio.hop_length': audio.get('hop_length'),
        'audio.num_channels': audio.get('num_channels'),
        'model.dim': model.get('dim'),
        'model.depth': model.get('depth'),
        'model.stereo': model.get('stereo'),
        'model.num_stems': model.get('num_stems'),
        'model.time_transformer_depth': model.get('time_transformer_depth'),
        'model.freq_transformer_depth': model.get('freq_transformer_depth'),
        'model.num_bands': model.get('num_bands'),
        'model.dim_head': model.get('dim_head'),
        'model.heads': model.get('heads'),
        'model.dim_freqs_in': model.get('dim_freqs_in'),
        'model.sample_rate': model.get('sample_rate'),
        'model.stft_n_fft': model.get('stft_n_fft'),
        'model.stft_hop_length': model.get('stft_hop_length'),
        'model.stft_win_length': model.get('stft_win_length'),
        'model.stft_normalized': model.get('stft_normalized'),
        'model.mask_estimator_depth': model.get('mask_estimator_depth'),
        'inference.batch_size': inference.get('batch_size'),
        'inference.num_overlap': inference.get('num_overlap'),
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ConversionError('big_beta7.yaml is missing required fields: ' + ', '.join(missing))
    if int(audio['sample_rate']) != int(model['sample_rate']):
        raise ConversionError('audio/model sample_rate mismatch')
    if int(audio['n_fft']) != int(model['stft_n_fft']):
        raise ConversionError('audio/model n_fft mismatch')
    if int(audio['hop_length']) != int(model['stft_hop_length']):
        raise ConversionError('audio/model hop_length mismatch')
    if int(model['dim_freqs_in']) != int(model['stft_n_fft']) // 2 + 1:
        raise ConversionError('dim_freqs_in does not match STFT frequency bins')
    if int(audio['num_channels']) != (2 if bool(model['stereo']) else 1):
        raise ConversionError('channel/stereo configuration mismatch')
    if int(model['num_stems']) != 1:
        raise ConversionError('audio.cpp Mel-Band RoFormer path currently requires one stem')
    return {
        'model_type': 'mel_band_roformer',
        'sample_rate': int(model['sample_rate']),
        'stereo': bool(model['stereo']),
        'chunk_size': int(audio['chunk_size']),
        'batch_size': int(inference['batch_size']),
        'num_overlap': int(inference['num_overlap']),
        'normalize': False,
        'dim': int(model['dim']),
        'depth': int(model['depth']),
        'num_bands': int(model['num_bands']),
        'num_stems': int(model['num_stems']),
        'time_transformer_depth': int(model['time_transformer_depth']),
        'freq_transformer_depth': int(model['freq_transformer_depth']),
        'linear_transformer_depth': 0,
        'dim_head': int(model['dim_head']),
        'heads': int(model['heads']),
        'n_fft': int(model['stft_n_fft']),
        'hop_length': int(model['stft_hop_length']),
        'win_length': int(model['stft_win_length']),
        'stft_normalized': bool(model['stft_normalized']),
        'mask_estimator_depth': int(model['mask_estimator_depth']),
        'mlp_expansion_factor': 4,
        'skip_connection': False,
        'has_final_norm': False,
    }


def _materialize(archive: zipfile.ZipFile, root: str, ref: TensorRef):
    import numpy as np
    member = f'{root}/data/{ref.storage_key}' if root else f'data/{ref.storage_key}'
    try:
        raw = archive.read(member)
    except KeyError as exc:
        raise ConversionError(f'Missing tensor storage {member}') from exc
    flat = np.frombuffer(raw, dtype=np.float32)
    if any(v < 0 for v in ref.size) or ref.offset < 0:
        raise ConversionError('Negative tensor shape/offset')
    if len(ref.size) != len(ref.stride):
        raise ConversionError('Tensor rank/stride mismatch')
    if ref.size:
        last = ref.offset + sum((size - 1) * stride for size, stride in zip(ref.size, ref.stride)) + 1
        if last > flat.size:
            raise ConversionError('Tensor points outside storage')
        view = np.lib.stride_tricks.as_strided(
            flat[ref.offset:], shape=ref.size,
            strides=tuple(stride * flat.itemsize for stride in ref.stride), writeable=False)
        return np.ascontiguousarray(view)
    if ref.offset >= flat.size:
        raise ConversionError('Scalar points outside storage')
    return np.ascontiguousarray(flat[ref.offset:ref.offset + 1].reshape(()))


def checkpoint_to_safetensors(checkpoint: Path, config_yaml: Path, output_dir: Path,
                              cancel: threading.Event, progress) -> dict:
    try:
        import numpy as np
        from safetensors.numpy import save_file
    except ImportError as exc:
        raise ConversionError('Model converter components are missing; reinstall ASR2RPP') from exc
    config = parse_big_beta7_yaml(config_yaml)
    progress('背景音除去モデルを検証中…')
    archive, root, state = _checkpoint_payload(checkpoint)
    try:
        if not any(name.startswith('band_split.to_features.') for name in state):
            raise ConversionError('Checkpoint does not look like Mel-Band RoFormer (band_split missing)')
        if not any(name.startswith('mask_estimators.') for name in state):
            raise ConversionError('Checkpoint does not look like Mel-Band RoFormer (mask_estimators missing)')
        qkv = [name for name in state if name.endswith('.to_qkv.weight')]
        if not qkv:
            raise ConversionError('Checkpoint has no expected RoFormer QKV tensors')
        output_dir.mkdir(parents=True, exist_ok=True)
        tensors = {}
        total = len(state)
        for index, (name, ref) in enumerate(state.items(), 1):
            if cancel.is_set():
                raise ConversionError('Conversion cancelled')
            if name.endswith('.rotary_embed.freqs'):
                continue
            array = _materialize(archive, root, ref)
            if array.dtype != np.float32:
                raise ConversionError(f'Unexpected dtype for {name}: {array.dtype}')
            if not np.isfinite(array).all():
                raise ConversionError(f'Non-finite values in {name}')
            if name.endswith('.to_qkv.weight'):
                if array.ndim < 1 or array.shape[0] % 3:
                    raise ConversionError(f'Invalid QKV shape for {name}: {array.shape}')
                prefix = name[:-len('.to_qkv.weight')]
                q, k, v = np.split(array, 3, axis=0)
                tensors[prefix + '.to_q.weight'] = np.ascontiguousarray(q)
                tensors[prefix + '.to_k.weight'] = np.ascontiguousarray(k)
                tensors[prefix + '.to_v.weight'] = np.ascontiguousarray(v)
            else:
                tensors[name] = array
            if index == total or index % 50 == 0:
                progress(f'背景音除去モデルを変換中… {index}/{total}')
        safetensors = output_dir / 'model.safetensors'
        save_file(tensors, str(safetensors), metadata={
            'format': 'pt', 'source': 'pcunwa/Mel-Band-Roformer-big',
            'checkpoint': checkpoint.name, 'converter': 'ASR2RPP torch-free',
        })
        (output_dir / 'config.json').write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')
        return {'tensor_count_source': len(state), 'tensor_count_output': len(tensors),
                'qkv_split_count': len(qkv), 'config': config,
                'safetensors_size': safetensors.stat().st_size}
    finally:
        archive.close()


def _run(argv: list[str], cancel: threading.Event, progress, timeout=7200) -> str:
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               stdin=subprocess.DEVNULL, text=True, encoding='utf-8',
                               errors='replace', creationflags=flags)
    lines = []
    started = time.monotonic()
    try:
        while process.poll() is None:
            if cancel.is_set():
                process.terminate()
                raise ConversionError('Conversion cancelled')
            if time.monotonic() - started > timeout:
                process.kill()
                raise ConversionError('Model conversion timeout')
            line = process.stdout.readline()
            if line:
                lines.append(line)
                progress(line.strip()[-300:])
            else:
                time.sleep(0.05)
        tail = process.stdout.read()
        if tail:
            lines.append(tail)
        if process.returncode:
            raise ConversionError(f'{Path(argv[0]).name} exited {process.returncode}')
        return ''.join(lines)
    finally:
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()


def find_audio_cpp_tool(name: str, assets_root: Path) -> Path:
    suffix = '.exe' if sys.platform == 'win32' else ''
    filename = name + suffix
    roots = [Path(sys.executable).parent / 'engines', assets_root / 'engines']
    for root in roots:
        for backend in ('audio_cpp-cpu', 'audio_cpp-vulkan', 'audio_cpp'):
            candidate = root / backend / filename
            if candidate.is_file():
                return candidate
            if (root / backend).exists():
                found = list((root / backend).rglob(filename))
                if found:
                    return found[0]
    found = shutil.which(name)
    if found:
        return Path(found)
    raise ConversionError(f'{name} is not installed with ASR2RPP')


def convert_model(model, directory: Path, assets_root: Path, cancel: threading.Event, progress) -> tuple[Path, dict]:
    recipe = model.source.get('convert')
    if not recipe:
        entry = model.source.get('entry', model.source['files'][0])
        return directory / entry, {}
    if recipe.get('kind') != 'mel_band_roformer_ckpt_to_gguf':
        raise ConversionError(f'Unsupported conversion recipe: {recipe.get("kind")}')
    checkpoint = directory / recipe.get('checkpoint', 'big_beta7.ckpt')
    config_yaml = directory / recipe.get('config', 'big_beta7.yaml')
    output = directory / recipe.get('output', 'big_beta7-f16.gguf')
    precision = recipe.get('precision', 'f16')
    if precision not in {'f16', 'q8_0'}:
        raise ConversionError('Mel-Band RoFormer conversion supports f16 or q8_0')
    if output.exists():
        return output, {'reused_converted_model': True}
    # Peak disk use includes the original checkpoint, a near-F32-size
    # SafeTensors intermediate, and the final 16/Q8 GGUF. Fail early.
    minimum_free = checkpoint.stat().st_size * 2 + 256 * 1024 ** 2
    free = shutil.disk_usage(directory).free
    if free < minimum_free:
        raise ConversionError(
            f'Insufficient disk space for local conversion: need about {minimum_free / 1024**3:.1f} GiB free')
    work = directory / '.conversion'
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        info = checkpoint_to_safetensors(checkpoint, config_yaml, work, cancel, progress)
        converter = find_audio_cpp_tool('audiocpp_gguf', assets_root)
        temporary = output.with_name(output.name + '.part.gguf')
        temporary.unlink(missing_ok=True)
        progress(f'GGUF {precision.upper()}へ変換中…')
        _run([str(converter), '--input', str(work / 'model.safetensors'),
              '--root', str(work), '--family', model.family,
              '--output', str(temporary), '--type', precision, '--overwrite'], cancel, progress)
        inspect = _run([str(converter), '--inspect', str(temporary)], cancel, lambda _text: None)
        if 'mel_band_roformer' not in inspect:
            raise ConversionError('GGUF inspection did not confirm mel_band_roformer family')
        cli = find_audio_cpp_tool('audiocpp_cli', assets_root)
        cli_inspect = _run([str(cli), '--inspect', '--family', model.family,
                            '--model', str(temporary)], cancel, lambda _text: None)
        temporary.replace(output)
        info.update({'precision': precision, 'output_size': output.stat().st_size,
                     'converter': str(converter), 'gguf_inspect': inspect[-4000:],
                     'runtime_inspect': cli_inspect[-4000:]})
        progress('背景音除去モデルの準備が完了しました。')
        return output, info
    finally:
        if not bool(recipe.get('keep_intermediate', False)):
            shutil.rmtree(work, ignore_errors=True)
