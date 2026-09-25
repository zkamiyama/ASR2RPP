#!/usr/bin/env python3
"""Convert HF Whisper SafeTensors to whisper.cpp GGML, without torch/transformers.

Only NumPy and safetensors are required. No pickle or model Python is executed.
GGML layout follows whisper.cpp/models/convert-h5-to-ggml.py (MIT); see README.
F32 is genuine F32: it never passes through F16. F16 keeps native F32 auxiliaries.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from urllib.request import Request, urlopen

MODEL_REPO = 'litagin/anime-whisper'
MODEL_REVISION = '22e2008a8182b357da3922a6308d095008f72973'
MODEL_SHA256 = '15c672f0bf687b1c67aa14325f9c382c6919f1fce976f990513618b464a3c626'
FILTER_REVISION = '86098128c0b4f24f0e2aa2994de830614b474227'
FILTER_URL = f'https://raw.githubusercontent.com/openai/whisper/{FILTER_REVISION}/whisper/assets/mel_filters.npz'
MAGIC = 0x67676D6C


def digest(path: Path) -> str:
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def download(url: str, path: Path, expected: str | None = None) -> None:
    """Atomic streaming download; reuse verified weights; never disable TLS checks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if expected and digest(path) != expected:
            raise ValueError(f'Cached file checksum mismatch: {path}; move it aside and retry')
        return
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.part', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            with urlopen(Request(url, headers={'User-Agent': 'ASR2RPP-conversion-trial/1'}), timeout=90) as response:
                total = int(response.headers.get('Content-Length', 0))
                if total and shutil.disk_usage(path.parent).free < total + 256 * 1024**2:
                    raise OSError('Insufficient disk space for model download')
                done, bucket = 0, -1
                while block := response.read(4 * 1024**2):
                    out.write(block)
                    done += len(block)
                    if done // (128 * 1024**2) != bucket:
                        bucket = done // (128 * 1024**2)
                        print(f'{path.name}: {done / 1024**2:.0f} / {total / 1024**2:.0f} MiB', flush=True)
                if total and done != total:
                    raise OSError('Incomplete download')
        if expected and digest(Path(name)) != expected:
            raise ValueError(f'SHA-256 mismatch: {path.name}')
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def acquire(directory: Path) -> dict:
    urls = {}
    for name in ('config.json', 'vocab.json', 'added_tokens.json', 'generation_config.json',
                 'preprocessor_config.json', 'model.safetensors'):
        url = f'https://huggingface.co/{MODEL_REPO}/resolve/{MODEL_REVISION}/{name}'
        download(url, directory / name, MODEL_SHA256 if name == 'model.safetensors' else None)
        urls[name] = url
    download(FILTER_URL, directory / 'mel_filters.npz')
    urls['mel_filters.npz'] = FILTER_URL
    record = {'repository': MODEL_REPO, 'revision': MODEL_REVISION,
              'files': {name: {'url': url, 'sha256': digest(directory / name)} for name, url in urls.items()}}
    (directory / 'download.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    return record


def specification(c: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Explicit tensor allowlist and shape checks; do not silently skip weights."""
    if c.get('model_type') != 'whisper' or c.get('tie_word_embeddings', True) is not True:
        raise ValueError('Only standard Whisper with tied output embeddings is supported')
    d = int(c['d_model'])
    enc, dec = int(c['encoder_layers']), int(c['decoder_layers'])
    for name in ('encoder_attention_heads', 'decoder_attention_heads'):
        if int(c[name]) <= 0 or d % int(c[name]):
            raise ValueError(f'Invalid {name}')
    if min(d, enc, dec, int(c['max_source_positions']), int(c['max_target_positions'])) <= 0:
        raise ValueError('Invalid model dimensions')
    if c.get('activation_function', 'gelu') != 'gelu' or c.get('scale_embedding', False):
        raise ValueError('Unsupported Whisper activation/embedding scaling')
    if c.get('encoder_ffn_dim', 4*d) != 4*d or c.get('decoder_ffn_dim', 4*d) != 4*d:
        raise ValueError('GGML Whisper requires FFN width 4*d_model')
    spec = {}
    def add(hf, ggml, shape):
        spec['model.' + hf] = (ggml, shape)
    add('encoder.conv1.weight', 'encoder.conv1.weight', (d, int(c['num_mel_bins']), 3))
    add('encoder.conv1.bias', 'encoder.conv1.bias', (d,))
    add('encoder.conv2.weight', 'encoder.conv2.weight', (d, d, 3))
    add('encoder.conv2.bias', 'encoder.conv2.bias', (d,))
    add('encoder.embed_positions.weight', 'encoder.positional_embedding', (int(c['max_source_positions']), d))
    add('decoder.embed_positions.weight', 'decoder.positional_embedding', (int(c['max_target_positions']), d))
    add('decoder.embed_tokens.weight', 'decoder.token_embedding.weight', (int(c['vocab_size']), d))
    for side, layers in (('encoder', enc), ('decoder', dec)):
        ln = 'encoder.ln_post' if side == 'encoder' else 'decoder.ln'
        for suffix in ('weight', 'bias'):
            add(f'{side}.layer_norm.{suffix}', f'{ln}.{suffix}', (d,))
        for i in range(layers):
            hf, gg = f'{side}.layers.{i}', f'{side}.blocks.{i}'
            attns = [('self_attn', 'attn')]
            if side == 'decoder':
                attns.append(('encoder_attn', 'cross_attn'))
            for src, dst in attns:
                for short, full in (('q', 'query'), ('k', 'key'), ('v', 'value'), ('out', 'out')):
                    add(f'{hf}.{src}.{short}_proj.weight', f'{gg}.{dst}.{full}.weight', (d, d))
                    if short != 'k':
                        add(f'{hf}.{src}.{short}_proj.bias', f'{gg}.{dst}.{full}.bias', (d,))
                for suffix in ('weight', 'bias'):
                    add(f'{hf}.{src}_layer_norm.{suffix}', f'{gg}.{dst}_ln.{suffix}', (d,))
            for src, dst, shape in (('fc1.weight', 'mlp.0.weight', (4*d, d)),
                                     ('fc1.bias', 'mlp.0.bias', (4*d,)),
                                     ('fc2.weight', 'mlp.2.weight', (d, 4*d)),
                                     ('fc2.bias', 'mlp.2.bias', (d,))):
                add(f'{hf}.{src}', f'{gg}.{dst}', shape)
            for suffix in ('weight', 'bias'):
                add(f'{hf}.final_layer_norm.{suffix}', f'{gg}.mlp_ln.{suffix}', (d,))
    return spec


def vocab_bytes(vocab: dict) -> list[bytes]:
    ids = list(vocab.values())
    if any(type(i) is not int for i in ids) or sorted(ids) != list(range(len(ids))):
        raise ValueError('Vocabulary IDs must be unique, contiguous and start at zero')
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs.copy()
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    decoder = {chr(c): b for b, c in zip(bs, cs)}
    return [bytes(decoder[ch] for ch in token) for token, _ in sorted(vocab.items(), key=lambda item: item[1])]


def prepare_tensor(array, name: str, precision: str):
    import numpy as np
    if array.dtype not in (np.dtype('float32'), np.dtype('float16')):
        raise ValueError(f'Unsupported source dtype {array.dtype} for {name}')
    if not np.isfinite(array).all():
        raise ValueError(f'Non-finite source tensor: {name}')
    if name in ('encoder.conv1.bias', 'encoder.conv2.bias'):
        array = array.reshape(-1, 1)
    auxiliary = array.ndim < 2 or name.endswith('positional_embedding') or name in ('encoder.conv1.bias', 'encoder.conv2.bias')
    dtype = '<f4' if precision == 'f32' or auxiliary else '<f2'
    with np.errstate(over='raise', invalid='raise'):
        return np.ascontiguousarray(array, dtype=dtype)


def convert(directory: Path, output: Path, precision: str = 'f16') -> dict:
    import numpy as np
    from safetensors import safe_open
    if precision not in ('f16', 'f32'):
        raise ValueError('Precision must be f16 or f32')
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    spec = specification(config)
    tokens = vocab_bytes(json.loads((directory / 'vocab.json').read_text(encoding='utf-8')))
    if len(tokens) > config['vocab_size']:
        raise ValueError('Vocabulary larger than embedding table')
    added_file = directory / 'added_tokens.json'
    if added_file.is_file():
        added = json.loads(added_file.read_text(encoding='utf-8'))
        # Anime Whisper uses the unmodified large-v3 special-token IDs.
        if config['vocab_size'] == 51866:
            for token, expected in {'<|startoftranscript|>': 50258, '<|ja|>': 50266,
                                    '<|transcribe|>': 50360, '<|notimestamps|>': 50364}.items():
                if added.get(token) != expected:
                    raise ValueError(f'Unexpected special-token ID: {token}')
    with np.load(directory / 'mel_filters.npz', allow_pickle=False) as data:
        filters = np.ascontiguousarray(data[f"mel_{config['num_mel_bins']}"], dtype='<f4')
    if filters.shape != (config['num_mel_bins'], 201) or not np.isfinite(filters).all():
        raise ValueError('Unexpected mel-filter dimensions or values')
    source = directory / 'model.safetensors'
    if output.exists():
        raise FileExistsError(f'Output already exists; choose another output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    expected_size = sum(math.prod(shape) * 4 for _, shape in spec.values())
    if shutil.disk_usage(output.parent).free < expected_size + 256 * 1024**2:
        raise OSError('Allow at least one F32 model worth of free output space')
    fd, temp_name = tempfile.mkstemp(prefix=output.name+'.', suffix='.part', dir=output.parent)
    report = {'format': 'whisper.cpp GGML', 'precision': precision, 'torch_used': False,
              'tensor_count': len(spec), 'config': config, 'tensors': {}}
    try:
        with os.fdopen(fd, 'wb') as out, safe_open(source, framework='numpy') as weights:
            keys = set(weights.keys())
            embed, proj = 'model.decoder.embed_tokens.weight', 'proj_out.weight'
            alias = {name: name for name in spec}
            if embed not in keys and proj in keys:
                alias[embed] = proj
            expected_keys = set(alias.values())
            extras = keys - expected_keys - {proj}
            missing = expected_keys - keys
            if extras or missing:
                raise ValueError(f'Unsupported tensor set: missing={sorted(missing)} extra={sorted(extras)}')
            if proj in keys and embed in keys:
                if not np.array_equal(weights.get_tensor(proj), weights.get_tensor(embed)):
                    raise ValueError('Output projection differs from embeddings; cannot silently drop it')
            header = [MAGIC, config['vocab_size'], config['max_source_positions'], config['d_model'],
                      config['encoder_attention_heads'], config['encoder_layers'], config['max_target_positions'],
                      config['d_model'], config['decoder_attention_heads'], config['decoder_layers'],
                      config['num_mel_bins'], int(precision == 'f16')]
            out.write(struct.pack('<12i', *header))
            out.write(struct.pack('<2i', *filters.shape))
            out.write(filters.tobytes())
            out.write(struct.pack('<i', len(tokens)))
            for token in tokens:
                out.write(struct.pack('<i', len(token)))
                out.write(token)
            for index, (hf, (name, shape)) in enumerate(spec.items(), 1):
                tensor = weights.get_tensor(alias[hf])
                if tensor.shape != shape:
                    raise ValueError(f'{hf}: expected {shape}, got {tensor.shape}')
                tensor = prepare_tensor(tensor, name, precision)
                raw_name = name.encode('utf-8')
                out.write(struct.pack('<3i', tensor.ndim, len(raw_name), int(tensor.dtype.itemsize == 2)))
                out.write(struct.pack('<' + 'i'*tensor.ndim, *reversed(tensor.shape)))
                out.write(raw_name)
                payload = memoryview(tensor).cast('B')
                out.write(payload)
                report['tensors'][name] = {'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                                           'sha256': hashlib.sha256(payload).hexdigest()}
                del payload, tensor
                if index % 50 == 0 or index == len(spec):
                    print(f'Convert {precision}: {index}/{len(spec)}', flush=True)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_name, output)
    finally:
        Path(temp_name).unlink(missing_ok=True)
    report.update(output=str(output.resolve()), output_size=output.stat().st_size,
                  output_sha256=digest(output), source_sha256=digest(source),
                  mel_sha256=digest(directory / 'mel_filters.npz'),
                  config_sha256=digest(directory / 'config.json'),
                  vocab_sha256=digest(directory / 'vocab.json'))
    download_record = directory / 'download.json'
    if download_record.is_file():
        report['acquisition'] = json.loads(download_record.read_text(encoding='utf-8'))
    output.with_suffix('.conversion.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    toml = (f'runtime = "whisper_cpp"\ntask = "asr"\nname = "Anime Whisper · {precision.upper()}"\n'
            f'[source]\npath = {json.dumps(output.resolve().as_posix(), ensure_ascii=False)}\n'
            '[defaults]\nlanguage = "ja"\n[constraints]\ndisabled_parameters = ["initial_prompt", "carry_initial_prompt"]\n')
    output.with_suffix('.toml').write_text(toml, encoding='utf-8')
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Acquire pinned original weights (~3 GB)')
    parser.add_argument('--model-dir', type=Path, default=Path('anime-whisper-source'))
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--precision', choices=('f16', 'f32'), default='f16')
    args = parser.parse_args()
    try:
        if args.download:
            acquire(args.model_dir)
        output = args.output or Path(f'ggml-anime-whisper-local-{args.precision}.bin')
        report = convert(args.model_dir, output, args.precision)
        print(f"Ready: {output}\nSHA256: {report['output_sha256']}")
        return 0
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
