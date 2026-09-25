"""Read/compare unquantized Whisper GGML tensors without loading a model."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def read_exact(f, n):
    value = f.read(n)
    if len(value) != n:
        raise ValueError('Truncated GGML')
    return value


def block_hash(f, n):
    h = hashlib.sha256()
    while n:
        block = read_exact(f, min(n, 4 * 1024**2))
        h.update(block)
        n -= len(block)
    return h.hexdigest()


def inspect(path):
    tensors = {}
    with Path(path).open('rb') as f:
        header = list(struct.unpack('<12i', read_exact(f, 48)))
        if header[0] != 0x67676D6C:
            raise ValueError('Not Whisper GGML')
        mel_shape = struct.unpack('<2i', read_exact(f, 8))
        if not 0 < math.prod(mel_shape) <= 128 * 4096:
            raise ValueError('Invalid mel table')
        mel_hash = block_hash(f, math.prod(mel_shape) * 4)
        n_vocab, = struct.unpack('<i', read_exact(f, 4))
        if not 0 <= n_vocab <= header[1]:
            raise ValueError('Invalid vocabulary size')
        vocab_hash = hashlib.sha256()
        for i in range(n_vocab):
            b = read_exact(f, 4)
            length, = struct.unpack('<i', b)
            if not 0 <= length <= 100000:
                raise ValueError('Invalid token length')
            vocab_hash.update(b + read_exact(f, length))
        while b := f.read(12):
            if len(b) != 12:
                raise ValueError('Truncated tensor header')
            ndim, length, ftype = struct.unpack('<3i', b)
            if not 1 <= ndim <= 3 or not 0 < length < 1000 or ftype not in (0, 1):
                raise ValueError('Only F32/F16 GGML tensors supported')
            shape = tuple(reversed(struct.unpack('<' + 'i'*ndim, read_exact(f, ndim*4))))
            if min(shape) <= 0:
                raise ValueError('Invalid tensor shape')
            name = read_exact(f, length).decode('utf-8')
            if name in tensors:
                raise ValueError('Duplicate tensor')
            tensors[name] = {'shape': list(shape), 'dtype': 'float32' if ftype == 0 else 'float16',
                             'sha256': block_hash(f, math.prod(shape) * (4 if ftype == 0 else 2))}
    return {'header': header, 'mel_shape': list(mel_shape), 'mel_sha256': mel_hash,
            'vocab_size_in_file': n_vocab, 'vocab_sha256': vocab_hash.hexdigest(), 'tensors': tensors}


def compare(left, right):
    a, b = inspect(left), inspect(right)
    common = sorted(set(a['tensors']) & set(b['tensors']))
    changed = [name for name in common if a['tensors'][name] != b['tensors'][name]]
    return {'left': str(left), 'right': str(right), 'header_equal': a['header'] == b['header'],
            'mel_equal': a['mel_sha256'] == b['mel_sha256'],
            'vocab_equal': a['vocab_sha256'] == b['vocab_sha256'],
            'common_tensor_count': len(common), 'equal_tensor_count': len(common) - len(changed),
            'different_tensors': changed,
            'left_only': sorted(set(a['tensors']) - set(b['tensors'])),
            'right_only': sorted(set(b['tensors']) - set(a['tensors']))}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('left', type=Path)
    p.add_argument('right', type=Path, nargs='?')
    a = p.parse_args()
    print(json.dumps(compare(a.left, a.right) if a.right else inspect(a.left), indent=2))
